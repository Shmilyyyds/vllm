# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare local Qwen3 target, DSpark and TreeSpark in separate processes.

This checks deterministic token agreement and sampled-token raw logprobs.
The top-k=1 case exercises stochastic verification with a point-mass target;
it does not establish correctness for general stochastic distributions.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

MODES = ("target", "dspark", "treespark")
PROMPTS = [
    "The first five positive even numbers are",
    "The capital city of France is",
    "请用中文解释为什么一年有四个季节：",
    "def fibonacci(n):\n    \"\"\"Return the first n Fibonacci numbers.\"\"\"\n",
]
CASES = {
    "greedy": {"temperature": 0.0},
    "stochastic_topk1": {"temperature": 0.8, "top_k": 1},
}
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


def read_metrics(llm) -> dict[str, int]:
    snapshot = llm.get_metrics()
    return {
        key: sum(
            int(metric.value)
            for metric in snapshot
            if metric.name == name and hasattr(metric, "value")
        )
        for key, name in METRICS.items()
    }


def run_worker(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    options = dict(
        model=str(args.target),
        dtype="bfloat16",
        enforce_eager=True,
        async_scheduling=False,
        attention_config={"backend": "FLEX_ATTENTION"},
        max_model_len=512,
        max_num_seqs=2,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
        disable_log_stats=False,
        logprobs_mode="raw_logprobs",
        seed=17,
    )
    if args.worker != "target":
        speculative_config = {
            "method": "dspark",
            "model": str(args.draft),
            "num_speculative_tokens": 7,
        }
        if args.worker == "treespark":
            speculative_config["treespark"] = {
                "max_nodes": 16,
                "max_children": 3,
                "theta": 0.0,
            }
        options["speculative_config"] = speculative_config

    llm = LLM(**options)
    result = {"mode": args.worker, "engine_options": options, "cases": {}}
    for case, sampling in CASES.items():
        print(f"RUN {args.worker}/{case}", flush=True)
        before = read_metrics(llm)
        outputs = llm.generate(
            PROMPTS,
            SamplingParams(
                **sampling,
                max_tokens=args.max_tokens,
                ignore_eos=True,
                logprobs=5,
                seed=17,
            ),
            use_tqdm=False,
        )
        after = read_metrics(llm)
        records = []
        if len(outputs) != len(PROMPTS):
            raise RuntimeError(f"{args.worker}/{case}: missing request outputs")
        for prompt, output in zip(PROMPTS, outputs):
            completion = output.outputs[0]
            tokens = list(completion.token_ids)
            if len(tokens) != args.max_tokens:
                raise RuntimeError(
                    f"{args.worker}/{case}: expected {args.max_tokens} tokens, "
                    f"got {len(tokens)}; finish_reason={completion.finish_reason}"
                )
            logprobs = completion.logprobs
            if logprobs is None or len(logprobs) != len(tokens):
                raise RuntimeError(f"{args.worker}/{case}: missing logprobs")
            records.append(
                {
                    "prompt": prompt,
                    "token_ids": tokens,
                    "text": completion.text,
                    "logprobs": [
                        {
                            str(token): float(value.logprob)
                            for token, value in row.items()
                        }
                        for row in logprobs
                    ],
                }
            )
        metrics = {key: after[key] - before[key] for key in METRICS}
        result["cases"][case] = {"outputs": records, "metrics": metrics}
        write_json(args.output_dir / f"{args.worker}.json", result)
        print(f"DONE {args.worker}/{case}: {metrics}", flush=True)


def compare(args: argparse.Namespace) -> bool:
    results = {
        mode: json.loads((args.output_dir / f"{mode}.json").read_text(encoding="utf-8"))
        for mode in MODES
    }
    report = {"logprob_atol": args.logprob_atol, "comparisons": [], "passed": True}
    for mode in MODES[1:]:
        for case in CASES:
            reference = results["target"]["cases"][case]["outputs"]
            actual = results[mode]["cases"][case]["outputs"]
            mismatches = []
            max_error = 0.0
            for prompt_index, (expected, observed) in enumerate(
                zip(reference, actual, strict=True)
            ):
                for position, (want, got) in enumerate(
                    zip(expected["token_ids"], observed["token_ids"], strict=True)
                ):
                    context = {"prompt_index": prompt_index, "position": position}
                    if want != got:
                        mismatches.append(
                            {
                                **context,
                                "kind": "token",
                                "expected": want,
                                "actual": got,
                                "target_logprobs": expected["logprobs"][position],
                                "actual_logprobs": observed["logprobs"][position],
                            }
                        )
                        # After the first divergence the contexts differ.
                        break
                    left = expected["logprobs"][position][str(want)]
                    right = observed["logprobs"][position][str(got)]
                    error = abs(left - right)
                    max_error = max(max_error, error)
                    if not math.isclose(
                        left, right, rel_tol=0, abs_tol=args.logprob_atol
                    ):
                        mismatches.append(
                            {
                                **context,
                                "kind": "sampled_token_logprob",
                                "expected": left,
                                "actual": right,
                            }
                        )

            metrics = results[mode]["cases"][case]["metrics"]
            rounds = metrics["draft_rounds"]
            nodes = metrics["verified_nodes"]
            if rounds <= 0 or nodes <= 0 or metrics["accepted_tokens"] <= 0:
                mismatches.append({"kind": "speculation_not_exercised"})
            if mode == "treespark" and nodes <= 7 * rounds:
                mismatches.append({"kind": "tree_width_not_exercised"})
            passed = not mismatches
            report["passed"] &= passed
            report["comparisons"].append(
                {
                    "mode": mode,
                    "case": case,
                    "passed": passed,
                    "max_sampled_token_logprob_error": max_error,
                    "metrics": metrics,
                    "mismatches": mismatches,
                }
            )
            status = "PASS" if passed else "FAIL"
            print(
                f"{status} {mode}/{case}: "
                f"max_logprob_error={max_error:.6f}, mismatches={len(mismatches)}",
                flush=True,
            )
            if mismatches:
                print(f"First mismatch: {mismatches[0]}", flush=True)
    write_json(args.output_dir / "comparison.json", report)
    return report["passed"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, required=True, help="Local Qwen3 weights"
    )
    parser.add_argument(
        "--draft", type=Path, required=True, help="Local DSpark weights"
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--logprob-atol",
        type=float,
        default=0.1,
        help="Absolute tolerance for sampled-token raw logprobs (default: 0.1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"treespark-validation-{datetime.now():%Y%m%d-%H%M%S}"),
        help="New directory for generated outputs and comparison.json",
    )
    parser.add_argument("--worker", choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 8 <= args.max_tokens <= 256:
        parser.error("--max-tokens must be between 8 and 256")
    if not math.isfinite(args.logprob_atol) or args.logprob_atol < 0:
        parser.error("--logprob-atol must be finite and non-negative")
    for field in ("target", "draft"):
        path = getattr(args, field).resolve()
        if not (path / "config.json").is_file():
            parser.error(f"Missing local checkpoint: {path / 'config.json'}")
        setattr(args, field, path)
    args.output_dir = args.output_dir.resolve()
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    if args.worker:
        run_worker(args)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Results: {args.output_dir}", flush=True)
    for mode in MODES:
        print(f"START {mode}", flush=True)
        command = [
            sys.executable,
            "-P",
            "-u",
            str(Path(__file__).resolve()),
            "--target",
            str(args.target),
            "--draft",
            str(args.draft),
            "--max-tokens",
            str(args.max_tokens),
            "--worker",
            mode,
            "--output-dir",
            str(args.output_dir),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            print(f"FAIL {mode}: subprocess exit={completed.returncode}", flush=True)
            return 1
    passed = compare(args)
    print(f"Report: {args.output_dir / 'comparison.json'}", flush=True)
    print("PASS TreeSpark correctness checks" if passed else "FAIL correctness checks")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
