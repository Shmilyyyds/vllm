# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure TreeSpark on the corpus used by the original reproduction.

No correctness gate is run. Each mode loads once, warms up, then measures
complete corpus passes. Historical cache-free results are a migration
reference, while current DSpark baselines measure the within-vLLM benefit.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

DATASETS = ("gsm8k", "math500", "humaneval", "mbpp", "mt-bench", "alpaca")
MODES = (
    "ar", "dspark", "chain7", "adapt0.02", "adapt0.05",
    "native_ar", "native_dspark",
)
METRICS = {
    "draft_rounds": "vllm:spec_decode_num_drafts",
    "verified_nodes": "vllm:spec_decode_num_draft_tokens",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens",
}


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def prepare_run(args) -> dict:
    prompts, file_hashes = [], {}
    for dataset in DATASETS:
        path = args.data_dir / f"{dataset}.jsonl"
        file_hashes[dataset] = hashlib.sha256(path.read_bytes()).hexdigest()
        selected = []
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index < args.skip:
                    continue
                if index >= args.skip + args.per_dataset:
                    break
                selected.append({
                    "dataset": dataset,
                    "index": index,
                    "prompt": json.loads(line)["turns"][0],
                })
        if len(selected) != args.per_dataset:
            raise ValueError(f"Not enough selected prompts in {path}")
        prompts.extend(selected)

    historical, historical_manifest = {}, {}
    if args.legacy_run:
        folder = args.legacy_run
        if (folder / "results").is_dir():
            folder = folder / "results"
        old_manifest = folder.parent / "manifest_speed_T0.0.json"
        if old_manifest.is_file():
            historical_manifest = read_json(old_manifest)
        for arm in ("ar", "chain7", "adapt0.02", "adapt0.05"):
            path = folder / f"ar_speedup_T0.0_{arm}.json"
            if path.is_file():
                historical[arm] = {"source": str(path), **read_json(path)}
        if not historical:
            print(f"No historical speed JSON in {folder}; new runs will continue.")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    return {
        "status": "running",
        "git_commit": commit,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "prompts": prompts,
        "dataset_sha256": file_hashes,
        "historical": historical,
        "historical_manifest": historical_manifest,
        "historical_scope_matches": (
            args.batch_size == 1
            and historical_manifest.get("temperature") == 0
            and all(historical_manifest.get(key) == getattr(args, key)
                    for key in ("skip", "per_dataset", "max_new_tokens"))
        ),
        "protocol": {
            "temperature": 0.0,
            "enable_thinking": False,
            "dtype": "bfloat16",
            "prefix_caching": False,
            "dispatch": "sequential" if args.batch_size == 1 else "all_at_once",
            "timed": "chat template, tokenization, generation, output collection",
            "excluded": "model loading, warmup, metrics snapshots, file writing",
            "stopping": "tokenizer EOS or exact max_new_tokens cap",
            "historical_note": (
                "Old cache-free runs can overrun the output cap in their last "
                "round, use different framework versions and were timed once "
                "without uniform warmup. Historical ratios describe overall "
                "migration, not an isolated algorithm speedup."
            ),
        },
    }


def read_metrics(llm) -> dict:
    snapshot = llm.get_metrics()
    return {
        key: sum(int(m.value) for m in snapshot
                 if m.name == name and hasattr(m, "value"))
        for key, name in METRICS.items()
    }


def engine_options(args, max_model_len: int) -> dict:
    options = dict(
        model=str(args.target),
        dtype="bfloat16",
        max_model_len=max_model_len,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=max(2048, args.batch_size * 65),
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        generation_config="vllm",
        disable_log_stats=False,
        seed=0,
    )
    if not args.worker.startswith("native_"):
        options.update(
            enforce_eager=True,
            async_scheduling=False,
            attention_config={"backend": "FLEX_ATTENTION"},
        )
    if args.worker not in ("ar", "native_ar"):
        spec = {
            "method": "dspark",
            "model": str(args.draft),
            "num_speculative_tokens": 7,
        }
        if args.worker == "chain7":
            spec["treespark"] = {
                "max_nodes": 7, "max_children": 1, "theta": 0.0,
            }
        elif args.worker.startswith("adapt"):
            spec["treespark"] = {
                "max_nodes": 64,
                "max_children": 8,
                "theta": float(args.worker.removeprefix("adapt")),
                "calibration_a": 0.6684,
                "calibration_b": -0.4587,
            }
        options["speculative_config"] = spec
    return options


def run_worker(args) -> None:
    import torch
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    manifest = read_json(args.output / "manifest.json")
    prompts = manifest["prompts"]
    tokenizer = AutoTokenizer.from_pretrained(str(args.target), local_files_only=True)

    def tokenize(prompt):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return tokenizer(text)["input_ids"]

    prompt_ids = [tokenize(prompt) for prompt in prompts]
    max_length = max(map(len, prompt_ids)) + args.max_new_tokens + 65
    options = engine_options(args, ((max_length + 127) // 128) * 128)
    llm = LLM(**options)
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        ignore_eos=False,
        stop_token_ids=[tokenizer.eos_token_id],
        seed=0,
    )
    report = {
        "mode": args.worker,
        "engine_options": options,
        "input_token_ids_sha256": digest(prompt_ids),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("vllm", "torch", "transformers")
        },
        "gpu": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "passes": [],
    }
    for number in range(args.warmup + args.repeats):
        phase = "warmup" if number < args.warmup else "measurement"
        repeat = number + 1 if phase == "warmup" else number - args.warmup + 1
        print(f"START {args.worker} {phase} {repeat}", flush=True)
        before = read_metrics(llm)
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs, request_wall = [], []
        if args.batch_size == 1:
            for prompt in prompts:
                request_start = time.perf_counter()
                outputs.extend(llm.generate(
                    [{"prompt_token_ids": tokenize(prompt)}], params,
                    use_tqdm=False,
                ))
                request_wall.append(time.perf_counter() - request_start)
        else:
            outputs = llm.generate(
                [{"prompt_token_ids": tokenize(prompt)} for prompt in prompts],
                params, use_tqdm=False,
            )
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        after = read_metrics(llm)
        generated = [list(output.outputs[0].token_ids) for output in outputs]
        total = sum(map(len, generated))
        record = {
            "phase": phase,
            "repeat": repeat,
            "wall_s": wall,
            "total_tokens": total,
            "requests": len(outputs),
            "tok_per_s": total / wall,
            "request_wall_s": request_wall,
            "metrics": {key: after[key] - before[key] for key in METRICS},
        }
        report["passes"].append(record)
        write_json(args.output / f"{args.worker}.json", report)
        if phase == "measurement":
            write_json(args.output / f"{args.worker}-rep{repeat}-outputs.json", [
                {
                    "dataset": prompt["dataset"],
                    "index": prompt["index"],
                    "prompt_tokens": len(output.prompt_token_ids),
                    "token_ids": tokens,
                    "text": output.outputs[0].text,
                    "finish_reason": output.outputs[0].finish_reason,
                }
                for prompt, output, tokens in zip(
                    prompts, outputs, generated, strict=True
                )
            ])
        print(
            f"DONE {args.worker} {phase} {repeat}: "
            f"{total / wall:.2f} tokens/s, {wall:.2f}s, {total} output tokens",
            flush=True,
        )


def summarize(args, manifest: dict, failed: list) -> None:
    summary = {}
    for mode in args.modes:
        path = args.output / f"{mode}.json"
        if mode in failed or not path.is_file():
            continue
        report = read_json(path)
        passes = [p for p in report["passes"] if p["phase"] == "measurement"]
        if len(passes) != args.repeats:
            continue
        rounds = sum(p["metrics"]["draft_rounds"] for p in passes)
        summary[mode] = {
            "tok_per_s_median": statistics.median(p["tok_per_s"] for p in passes),
            "tok_per_s_each": [p["tok_per_s"] for p in passes],
            "wall_s_median": statistics.median(p["wall_s"] for p in passes),
            "output_tokens_each": [p["total_tokens"] for p in passes],
            "accepted_per_round": (
                sum(p["metrics"]["accepted_tokens"] for p in passes) / rounds
                if rounds else None
            ),
            "nodes_per_round": (
                sum(p["metrics"]["verified_nodes"] for p in passes) / rounds
                if rounds else None
            ),
            "input_token_ids_sha256": report["input_token_ids_sha256"],
        }

    lines = [
        "# TreeSpark performance results", "",
        f"Requests: {len(manifest['prompts'])}; max new tokens: {args.max_new_tokens}; "
        f"batch size: {args.batch_size}; measured repeats: {args.repeats}.", "",
        "Rates are median output tokens/s; actual output counts are saved per pass.",
        "ar/dspark/chain7/adapt*: FlexAttention, eager, synchronous. "
        "native_*: vLLM default attention, compilation and scheduling settings.", "",
        "| mode | tokens/s | seconds | vs matched DSpark | vs native DSpark | "
        "vs vLLM chain7 | vs historical same arm | accepted/round | nodes/round |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, row in summary.items():
        rate = row["tok_per_s_median"]
        ratios = {}
        for baseline in ("dspark", "native_dspark", "chain7"):
            base = summary.get(baseline)
            ratios[baseline] = rate / base["tok_per_s_median"] if base else None
        old_arm = "ar" if mode == "native_ar" else mode
        old = manifest["historical"].get(old_arm)
        historical_ratio = (
            rate / old["tok_per_s"]
            if old and manifest["historical_scope_matches"] and old["tok_per_s"] > 0
            else None
        )
        row["ratios"] = {**ratios, "historical_same_arm": historical_ratio}

        def fmt(value):
            return "—" if value is None else f"{value:.3f}"

        lines.append("| " + " | ".join([
            mode, fmt(rate), fmt(row["wall_s_median"]),
            *(fmt(ratios[key]) for key in ("dspark", "native_dspark", "chain7")),
            fmt(historical_ratio), fmt(row["accepted_per_round"]),
            fmt(row["nodes_per_round"]),
        ]) + " |")
    if manifest["historical"]:
        lines += ["", "## Historical reproduction (single recorded pass)", "",
                  "| arm | tokens/s | seconds | output tokens |",
                  "| --- | ---: | ---: | ---: |"]
        for arm, old in manifest["historical"].items():
            lines.append(
                f"| {arm} | {old['tok_per_s']:.3f} | {old['wall_s']:.3f} | "
                f"{old['total_tokens']} |"
            )
    lines += ["", manifest["protocol"]["historical_note"], "",
              "accepted/round excludes the bonus token and is not the historical "
              "prompt-averaged tau. No output-correctness claim is made by this run."]
    if manifest["historical"] and not manifest["historical_scope_matches"]:
        lines += ["", "Historical ratios omitted: batch size or workload differs, "
                  "or the historical manifest is unavailable."]
    if failed:
        lines += ["", f"FAILED modes (excluded): {', '.join(failed)}"]
    rendered = "\n".join(lines) + "\n"
    (args.output / "summary.md").write_text(rendered, encoding="utf-8")
    write_json(args.output / "summary.json", {"modes": summary, "failed": failed})
    print(rendered, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--legacy-run", type=Path)
    parser.add_argument(
        "--output", type=Path, required=True, help="New results directory"
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--skip", type=int, default=8)
    parser.add_argument("--per-dataset", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1, help="Untimed corpus passes")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.skip < 0 or args.warmup < 0 or min(
        args.per_dataset, args.max_new_tokens, args.batch_size, args.repeats
    ) < 1:
        parser.error("Counts must be positive; skip and warmup may be zero")
    for key in ("target", "draft", "data_dir", "legacy_run", "output"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    if args.worker:
        run_worker(args)
        return 0

    manifest = prepare_run(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", manifest)
    print(f"Results: {args.output}", flush=True)
    failed = []
    for mode in args.modes:
        print(f"\nLOAD {mode}", flush=True)
        command = [
            sys.executable, "-P", "-u", str(Path(__file__).resolve()),
            *sys.argv[1:], "--worker", mode,
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            failed.append(mode)
            print(
                f"FAILED {mode}, exit={completed.returncode}; continuing.", flush=True
            )
    summarize(args, manifest, failed)
    manifest["status"] = "failed" if failed else "completed"
    manifest["failed_modes"] = failed
    write_json(args.output / "manifest.json", manifest)
    print(f"Saved: {args.output / 'summary.md'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
