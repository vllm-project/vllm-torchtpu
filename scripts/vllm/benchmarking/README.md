# Benchmarking vllm-torchtpu

All configs target TPU v7x-8 (TP=8). KV cache is forced to fp8 in `run_benchmarks.sh`.

## Quick Start

```bash
# Guard configs (scheduled perf builds)
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
| `qwen3-coder-30b-fp8-tp8-ep` | Qwen3-Coder-30B-A3B-Instruct-FP8 | yes | on | fp8 | guard (scheduled) |
| `qwen3-coder-30b-tp8-ep` | Qwen3-Coder-30B-A3B-Instruct | yes | on | none | nightly |
| `qwen3.5-35b-fp8-tp4-ep` | Qwen3.5-35B-A3B-FP8 | yes | on | fp8 | nightly |
| `qwen3.5-35b-fp8-dp4-tp2-ep` | Qwen3.5-35B-A3B-FP8 | yes | on | fp8 | nightly |
| `qwen3-coder-480b-fp8-tp8-ep` | Qwen3-Coder-480B-A35B-Instruct-FP8 | yes | on | fp8 | nightly short sweep |
| `qwen3-coder-480b-fp8-tp8-ep-full` | Qwen3-Coder-480B-A35B-Instruct-FP8 | yes | on | fp8 | manual only |

Configs are shell files in `configs/`. Add your own by copying an existing one.

## What It Does

1. Loads a config (model, TP, ISL/OSL sweep, concurrency levels)
2. Starts a `vllm serve` instance with the right flags
3. Optionally runs discarded full-traffic warmup passes
4. Runs `benchmark_serving.py` (from your vLLM install) for each ISL/OSL x concurrency combo
5. Saves JSON results to `benchmark_runs/<model>_tp<N>_<timestamp>/`

## Regression Checks

CI (`.buildkite/pipeline_perf.yml`) runs `check_regression.py` against the same
live server: once for performance, and once for each accuracy task.

### Performance (`--mode perf`)

Compares the benchmark JSON to `baselines/perf/<config>.baseline.json`, with a
5% tolerance. Gated metrics:

- Median TPOT
- Total token throughput
- Output token throughput
- Completed requests, as a floor relative to the baseline
- Median TTFT, only where the baseline entry carries it

Under closed-loop load beyond the prefill-admission capacity
(DP x max-num-batched-tokens / prompt-len), median TTFT is queue-position noise,
so nightly baselines are calibrated with `--gate-ttft-max-concurrency 0` to omit
it.

```bash
python3 scripts/vllm/benchmarking/check_regression.py \
  --mode perf \
  --results-dir <DIR> \
  --baseline scripts/vllm/benchmarking/baselines/perf/<config>.baseline.json
```

### Accuracy (`--mode eval`)

Compares `lm_eval` accuracy to `baselines/eval/<config>.<task>.baseline.json`.
The default tolerance is 1.5pp; a config can override it with `EVAL_TOLERANCE`.
Only keys present in the baseline are gated.

Where the baseline carries a standard error (`exact_match_stderr`), the allowed
drop becomes a one-sided 95% significance test:

```text
max(tolerance, 1.645 * hypot(stderr_baseline, stderr_run))
```

When only one side records a standard error, the two cases differ:

- **Baseline only**: it sets the scale for both sides, `sqrt(2) * stderr_baseline`.
- **Run only**: it covers only its own side, and the baseline is treated as
  exact, `stderr_run`.

The flat tolerance applies only where neither side records one.

> [!NOTE]
> **Why combine both standard errors?** The gate compares two independent runs,
> so the noisy quantity is their *difference*, whose standard error combines
> both sides in quadrature. Using one raw stderr would still fail ~24% of clean
> reruns on small benchmarks like `humaneval_plus_tpu` (N=164).

```bash
python3 scripts/vllm/benchmarking/check_regression.py \
  --mode eval \
  --results-dir <DIR> \
  --baseline scripts/vllm/benchmarking/baselines/eval/<config>.<task>.baseline.json
```

## Accuracy Evaluation (`lm_eval`)

### Tasks

| Task | Enabled by | Endpoint | Gated metric |
|------|------------|----------|--------------|
| `mmlu_llama`, `mmlu_pro` | `--run-lm-eval` | chat completions | `exact_match` |
| `humaneval_plus_tpu`, `mbpp_plus_tpu` | `--run-code-eval` (nightly only) | chat completions | pass@1 |

- `--run-lm-eval` walks `LM_EVAL_TASKS` (default `mmlu_llama mmlu_pro`; configs
  may narrow it).
- The code tasks use the full EvalPlus datasets, with evalplus-style instruct
  prompting and code-block extraction (see `lm_eval_tasks/`).
- Each task gates only when its baseline file exists.

### Tuning the `lm_eval` invocation

Two config vars tune the `lm_eval` invocation, for matching an external
reference run:

- `EXTRA_LM_EVAL_MODEL_ARGS`: `key=value` pairs merged into `--model_args`,
  last-wins per key.
- `LM_EVAL_GEN_KWARGS`: *replaces* the per-task `--gen_kwargs`, which is a
  merging action in lm-eval.

A generation cap goes in `LM_EVAL_GEN_KWARGS` as `max_gen_toks`: the
`--model_args` value is only a fallback for tasks whose yaml omits the key, and
`mmlu_pro` sets 2048 itself.

`LM_EVAL_GEN_KWARGS` takes either form lm-eval accepts, but not a mix:

- **JSON**, with double quotes: `{"chat_template_kwargs": {"thinking": false}}`.
  Any value containing `{` must be valid JSON, since lm-eval raises rather than
  falling back to its `key=value` parser once it sees a brace.
- **`key=value`**: `until=['<|im_end|>']`.

> [!TIP]
> **Suppressing reasoning is per-template, not universal.** The harness default
> sends `chat_template_kwargs: {"enable_thinking": false}`; a template keyed on
> a different flag ignores it silently and the model reasons anyway — Kimi-K2.6
> wants `thinking`. Check the checkpoint's `chat_template.jinja` before assuming
> the default lands.

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
| `EVAL_SOFT_FAIL` | `0` | `1` downgrades an eval regression to a `::warning` instead of failing the step. Perf regressions still fail. For a config whose eval baseline is new and whose run-to-run spread is not yet known — prefer it to `soft_fail` on the Buildkite step, which would mask the perf gate too. Forwarded into the pod by `.buildkite/kubernetes/common.sh` |
| `RANGE_RATIO_STYLE` (config var) | `symmetric` | How `RANDOM_RANGE_RATIO` is interpreted: `symmetric` = vllm bench serve native `[(1-r)L, (1+r)L]`; `min` = benchmark_serving.py-style `[rL, L]`, translated for vllm bench serve |
