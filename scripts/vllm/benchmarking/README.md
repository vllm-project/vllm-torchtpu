# Benchmarking vllm-torchtpu

All configs target TPU v7x-8 (TP=8). KV cache is forced to fp8 in `run_benchmarks.sh`.

## Quick Start

```bash
# Guard configs (PR-time gate)
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-30b-fp8-tp8-ep

# Nightly sweeps
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-30b-tp8-ep
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3.5-35b-fp8-tp4-ep
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3.5-35b-fp8-dp4-tp2-ep
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-480b-fp8-tp8-ep

# Manual full 480B sweep, not run in CI
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-480b-fp8-tp8-ep-full

# Pin a deterministic results dir (used by CI)
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-30b-tp8-ep --results-dir /tmp/perf

# Dry run (print commands without executing)
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-480b-fp8-tp8-ep --dry-run
```

## Available Configs

| Config | Model | MoE | EP | Weight quant | CI tier |
|--------|-------|-----|----|----|----|
| `qwen3-coder-30b-fp8-tp8-ep` | Qwen3-Coder-30B-A3B-Instruct-FP8 | yes | on | fp8 | guard (PR) |
| `qwen3-coder-30b-tp8-ep` | Qwen3-Coder-30B-A3B-Instruct | yes | on | none | nightly |
| `qwen3.5-35b-fp8-tp4-ep` | Qwen3.5-35B-A3B-FP8 | yes | on | fp8 | nightly |
| `qwen3.5-35b-fp8-dp4-tp2-ep` | Qwen3.5-35B-A3B-FP8 | yes | on | fp8 | nightly |
| `qwen3-coder-480b-fp8-tp8-ep` | Qwen3-Coder-480B-A35B-Instruct-FP8 | yes | on | fp8 | nightly short sweep |
| `qwen3-coder-480b-fp8-tp8-ep-full` | Qwen3-Coder-480B-A35B-Instruct-FP8 | yes | on | fp8 | manual only |

Configs are shell files in `configs/`. Add your own by copying an existing one.

CI (`.buildkite/pipeline_perf.yml`) runs `check_regression.py` three times against
the same live server:

- `--mode perf` compares benchmark JSON to `baselines/perf/<config>.baseline.json`
  (5% tolerance). Gated metrics: median TPOT, total token throughput, output
  token throughput, a baseline-relative completed-request floor, and median
  TTFT only where the baseline entry carries it — under closed-loop load
  beyond the prefill-admission capacity (DP x max-num-batched-tokens /
  prompt-len), median TTFT is queue-position noise, so nightly baselines are
  calibrated with `--gate-ttft-max-concurrency 0` to omit it.
- `--mode eval` compares `lm_eval` accuracy to
  `baselines/eval/<config>.<task>.baseline.json` (1.5pp tolerance). Only keys
  present in the baseline are gated. `--run-lm-eval` walks `LM_EVAL_TASKS`
  (default `mmlu_llama mmlu_pro`; configs may narrow it). MMLU tasks
  (`mmlu_llama`, `mmlu_pro`) run against the chat-completions endpoint;
  code-generation tasks
  (`humaneval_plus_tpu`, `mbpp_plus_tpu` — the full EvalPlus datasets,
  nightly only via `--run-code-eval`) also run against the chat-completions
  endpoint with evalplus-style instruct prompting + code-block extraction
  (see `lm_eval_tasks/`) and gate pass@1. Each task gates only when its
  baseline file exists.

Two config vars tune the `lm_eval` invocation, for matching an external
reference run: `EXTRA_LM_EVAL_MODEL_ARGS` (`key=value` pairs merged into
`--model_args`, last-wins per key) and `LM_EVAL_GEN_KWARGS` (*replaces* the
per-task `--gen_kwargs`, which is a merging action in lm-eval). A generation
cap goes in the latter as `max_gen_toks`: the `--model_args` value is only a
fallback for tasks whose yaml omits the key, and `mmlu_pro` sets 2048 itself.

## What It Does

1. Loads a config (model, TP, ISL/OSL sweep, concurrency levels)
2. Starts a `vllm serve` instance with the right flags
3. Optionally runs discarded full-traffic warmup passes
4. Runs `benchmark_serving.py` (from your vLLM install) for each ISL/OSL x concurrency combo
5. Saves JSON results to `benchmark_runs/<model>_tp<N>_<timestamp>/`

## Regression Checks

```bash
python3 scripts/vllm/benchmarking/check_regression.py \
  --mode perf \
  --results-dir <DIR> \
  --baseline scripts/vllm/benchmarking/baselines/perf/<config>.baseline.json

python3 scripts/vllm/benchmarking/check_regression.py \
  --mode eval \
  --results-dir <DIR> \
  --baseline scripts/vllm/benchmarking/baselines/eval/<config>.baseline.json

```

## Prerequisites

- vLLM installed (provides `benchmark_serving.py`)
- `vllm_torchtpu` installed (this repo)
- TPU device available
- Model weights downloaded (or use `--load-format dummy` via config)

## Results

Results are saved to `benchmark_runs/` (gitignored). Each run creates a directory containing:
- `config.json` — benchmark parameters
- `isl<N>_osl<N>_c<N>.json` — per-combo results from benchmark_serving.py
- `server.log` — vLLM server output
- `benchmark.log` — run metadata

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `BENCHMARK_WARMUP_RUNS` | `0` | Full benchmark passes to discard before writing each gated result |
| `RANGE_RATIO_STYLE` (config var) | `symmetric` | How `RANDOM_RANGE_RATIO` is interpreted: `symmetric` = vllm bench serve native `[(1-r)L, (1+r)L]`; `min` = benchmark_serving.py-style `[rL, L]`, translated for vllm bench serve |
