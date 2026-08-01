# Benchmarking torchtpu-vllm

All configs target TPU v7x-8 (TP=8). KV cache is forced to fp8 in `run_benchmarks.sh`.

## Quick Start

```bash
# Guard configs (PR-time gate)
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-30b-fp8-tp8-ep
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-30b-tp8-ep

# Nightly short 480B sweep
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
| `qwen3-coder-30b-tp8-ep` | Qwen3-Coder-30B-A3B-Instruct | yes | on | none | guard (PR) |
| `qwen3-coder-480b-fp8-tp8-ep` | Qwen3-Coder-480B-A35B-Instruct-FP8 | yes | on | fp8 | nightly short sweep |
| `qwen3-coder-480b-fp8-tp8-ep-full` | Qwen3-Coder-480B-A35B-Instruct-FP8 | yes | on | fp8 | manual only |

Configs are shell files in `configs/`. Add your own by copying an existing one.

CI (`.github/workflows/perf.yml`) runs `check_regression.py` three times against
the same live server:

- `--mode perf` compares benchmark JSON to `baselines/perf/<config>.baseline.json`
  (5% tolerance). Gated metrics: median TTFT, median TPOT, total token
  throughput, output token throughput, and a baseline-relative completed-request
  floor.
- `--mode eval` runs `lm_eval` over `mmlu_llama,mmlu_pro` against the vLLM
  chat-completions endpoint and compares accuracy to
  `baselines/eval/<config>.baseline.json` (1.5pp tolerance). Only keys present in
  the baseline are gated.
- `--mode evalplus` (nightly only) runs EvalPlus over `humaneval,mbpp` and
  compares to `baselines/evalplus/<config>.baseline.json` (1.5pp tolerance). The
  PR guard skips EvalPlus; the nightly job uploads results either way and gates
  only when a baseline file exists.

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

python3 scripts/vllm/benchmarking/check_regression.py \
  --mode evalplus \
  --results-dir <DIR> \
  --baseline scripts/vllm/benchmarking/baselines/evalplus/<config>.baseline.json
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
- `evalplus/` — EvalPlus generated samples, logs, and `*_eval_results.json`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `BENCHMARK_WARMUP_RUNS` | `0` | Full benchmark passes to discard before writing each gated result |
| `EVALPLUS_DATASETS` | `humaneval mbpp` | Nightly EvalPlus datasets to run against the live server |
| `EVALPLUS_PARALLEL` | `8` | Nightly EvalPlus local correctness-check worker count |
| `VLLM_MOE_ROUTING_SIMULATION_STRATEGY` | `uniform_random` | MoE routing for consistent results |
