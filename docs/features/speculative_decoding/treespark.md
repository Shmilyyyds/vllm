# TreeSpark for DSpark

TreeSpark extends DSpark with a tree of draft tokens. The target evaluates
the anchor and candidates in one forward pass. Each candidate attends to the
committed context and its own ancestors. Verification keeps the accepted path
and compacts its KV entries into the request's regular paged cache.

## Serve a Qwen3 checkpoint

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve Qwen/Qwen3-4B \
  --dtype bfloat16 \
  --attention-backend FLEX_ATTENTION \
  --enforce-eager --no-async-scheduling \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 512 \
  --speculative-config '{
    "method": "dspark",
    "model": "deepseek-ai/dspark_qwen3_4b_block7",
    "num_speculative_tokens": 7,
    "treespark": {
      "max_nodes": 32,
      "max_children": 8,
      "theta": 0.02,
      "theta_ladder": [[4, 0.0, 0.02], [8, 1.0, 0.05]]
    }
  }'
```

Use the same configuration dictionary as `LLM(speculative_config=...)`
for offline generation. `num_speculative_tokens` controls the depth and
DSpark backbone width. `max_nodes` separately controls target verification
capacity. Set `theta` to zero to disable threshold pruning.

Each `theta_ladder` row is `[max_batch_size, max_temperature, theta]`.
The first matching row wins, with both limits inclusive. The batch count is
the number of requests that emitted tokens in the step that produced the
draft. When no row matches, TreeSpark uses the configured `theta`. Set
`chain_min_batch_size` to limit proposals to a single branch from that
batch size upward. Tune the ladder and the optional `calibration_a` and
`calibration_b` for the checkpoint and workload.

## Current support

The tree path uses Model Runner V2, synchronous continuous batching and
eager target execution on one CUDA device. The target must be Qwen3 with
FlexAttention. The draft uses the native DSpark checkpoint loader. Greedy
requests walk the target's preferred path. Stochastic requests use ordered
without-replacement sibling proposals and recursive rejection. Sampling
parameters and logprobs use the target sampler's logits processing.
Structured output and thinking token budgets currently use ordinary
target decoding for those requests.

CUDA Graph replay for trees, async scheduling, parallel execution, KV
transfer, LoRA, watermarking, draft top-k projection, and native chain
adaptive verification in combination with TreeSpark are outside the current
support matrix. Removing the `treespark` object selects native DSpark.
The current tree proposer and verification walk synchronize with the host.
Measure end-to-end latency and throughput before using the tree path for
performance-sensitive workloads.

## Correctness checks

With a CUDA build of this checkout and its test dependencies, run:

```bash
.venv/bin/python -m pytest -q tests/v1/spec_decode/test_treespark.py
VLLM_USE_V2_MODEL_RUNNER=1 .venv/bin/python -m pytest -q \
  tests/v1/e2e/spec_decode/acceptance_rates/dspark/test_dspark.py \
  -k treespark
```

The end-to-end test loads the Qwen3 target and DSpark draft weights. It
checks target-only greedy output and logprobs and verifies that the engine
scheduled more nodes per round than the draft depth.

For local checkpoints, a standalone comparison uses the installed runtime
without the pytest dependencies:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python -P -u \
  examples/features/speculative_decoding/treespark_validate.py \
  --target /path/to/Qwen3-4B \
  --draft /path/to/dspark_qwen3_4b_block7 \
  --output-dir /path/to/new-validation-results
```

Run from the repository root using the Python interpreter in your vLLM
environment. The output directory must be new. The script runs target-only,
native DSpark and TreeSpark sequentially in separate processes, using local
weights, Runner V2, BF16, eager execution, synchronous scheduling and
FlexAttention. Each mode generates 64 tokens for four prompts with greedy
sampling and then with temperature 0.8 and top-k 1. Prefix caching is disabled.

The script requires exact token agreement and checks sampled-token raw
logprobs with absolute tolerance 0.1 (configurable with `--logprob-atol`). It
also checks that speculation accepts tokens and that TreeSpark verifies more
nodes per round than the draft depth. JSON files preserve outputs, top
logprobs, engine options and per-case metrics; `comparison.json` records the
first token divergence per request and any logprob mismatches before it.
Exit code zero means all of these checks passed. A mismatch needs investigation:
BF16 numerical differences can change near-tied argmax decisions as well as
implementation errors.

Top-k 1 makes the target distribution deterministic while exercising the
stochastic tree verifier. Passing this case does not establish general
stochastic distribution correctness. This script is not a performance
benchmark; benchmark separately after correctness checks pass, with warmup
and repeated measurements.

## Performance against the original reproduction

`benchmarks/benchmark_treespark.py` directly measures generation performance;
it does not run or require the correctness script. It reads the six local
JSONL files (`gsm8k`, `math500`, `humaneval`, `mbpp`, `mt-bench`, `alpaca`),
selects indices 8 through 24 in each, and uses `turns[0]` with the target chat
template and thinking disabled. Defaults match the original 102-prompt,
temperature-zero, BF16, 150-new-token reproduction.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python -P -u benchmarks/benchmark_treespark.py \
  --target /path/to/TreeSpark/checkpoints/Qwen3-4B \
  --draft /path/to/TreeSpark/checkpoints/dspark_qwen3_4b_block7 \
  --data-dir /path/to/TreeSpark/eval_datasets \
  --legacy-run /path/to/TreeSpark/runs/speed_bf16_formal_20261023_170447 \
  --output /path/to/new-performance-results
```

The command runs seven modes sequentially, with one model instance at a
time. Each loads once, runs one untimed corpus pass, then measures three
passes. `ar` and `dspark` use the same FlexAttention, eager, synchronous
configuration required by the current tree implementation. `chain7` uses
TreeSpark with seven nodes and one child; `adapt0.02` and `adapt0.05` use
64 nodes, eight children, and the original calibration constants. This
distinguishes the original single-branch control from native DSpark.
`native_ar` and `native_dspark` retain vLLM's default attention, compilation
and scheduling options, within Runner V2, as deployment baselines. All
modes disable prefix caching and use the same weights and request limits.
Use `--modes` to select modes, `--repeats` to change measured passes, or
`--warmup` to change untimed corpus passes.

The default batch size is one, sending requests sequentially as in the old
reproduction. Timing includes template rendering, tokenization, generation
and output collection, excluding model loading, warmup, file writes and
metric snapshots. It uses actual output token counts with normal EOS
stopping and a strict 150-token cap. The old generator could emit extra
tokens in its final speculative round; its timing also lacked uniform
warmup and used a different software environment. Ratios against historical
results therefore measure the overall migration, including caching and
runtime differences, rather than an isolated TreeSpark algorithm change.
They are emitted only when the historical manifest matches the sample
range and output budget, and batch size is one. Keep historical and new
raw counts when reporting results.

`summary.md` and `summary.json` report median throughput, median elapsed
time, and throughput ratios against matched DSpark, default DSpark, the
vLLM single-branch control, and the historical matching arm. Each mode's
JSON includes all measured and warmup passes, input-token hashes, versions,
engine options, actual output counts and speculation counters. Measured
outputs are also saved. `accepted/round` excludes the bonus token and must
not be equated to the original prompt-averaged tau. Model initialization
failures are marked and excluded; the other modes continue and the final
exit code is nonzero. Token disagreement does not gate this benchmark.

For a separate queued-throughput run, set `--batch-size 16` and choose a
new output directory. All 102 requests are submitted together with at most
16 active; historical single-request ratios are then omitted. These rates
must not be presented as the single-request migration comparison. The
current tree path's host synchronization and eager execution remain part
of the measured cost; this benchmark does not imply support for tree
CUDA graphs or asynchronous scheduling.
