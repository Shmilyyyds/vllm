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
