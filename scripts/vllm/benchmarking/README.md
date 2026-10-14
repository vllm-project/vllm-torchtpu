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
  `baselines/eval/<config>.<task>.baseline.json` (1.5pp default tolerance, or
  config `EVAL_TOLERANCE`). Only keys present in the baseline are gated. Where
  the baseline carries standard error (`exact_match_stderr`), the allowed drop
  becomes `max(tolerance, 1.645 * hypot(stderr_baseline, stderr_run))` — a
  one-sided 95% significance test. The gate compares two independent runs, so
  the noisy quantity is their *difference*, whose standard error combines both
  sides in quadrature; using one raw stderr would still fail ~24% of clean
  reruns on small benchmarks like `humaneval_plus_tpu` (N=164). When only one
  side records a stderr the two cases differ: a baseline stderr sets the scale
  for both sides (`sqrt(2) * stderr_baseline`), while a run stderr covers only
  its own side and the baseline is treated as exact (`stderr_run`). The flat
  tolerance governs only where neither side records one.
  `--run-lm-eval` walks `LM_EVAL_TASKS`
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

`LM_EVAL_GEN_KWARGS` takes either form lm-eval accepts, but not a mix: a value
containing `{` must be valid JSON with double quotes (`{"chat_template_kwargs":
{"thinking": false}}`), since lm-eval raises rather than falling back to its
`key=value` parser once it sees a brace. Otherwise use `key=value`
(`until=['<|im_end|>']`).

Suppressing reasoning is per-template, not universal. The harness default sends
`chat_template_kwargs: {"enable_thinking": false}`; a template keyed on a
different flag ignores it silently and the model reasons anyway — Kimi-K2.6
wants `thinking`. Check the checkpoint's `chat_template.jinja` before assuming
the default lands.

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

## Decode Throughput (Sliding Window)

`benchmark_serving.py` averages over a whole run, so with long prompts its
output throughput mixes prefill and decode. `bench_decode_sliding_window.py`
sends `--concurrency` requests at once, each with a distinct prompt of
`--prefill-tokens` tokens and `--decode-tokens` decoded tokens, and reports
throughput and TPOT over 10 s windows where (nearly) all requests decode
together. It also samples the server's `/metrics` to show whether KV capacity
(waiting requests, preemptions) limited a round. It works with any model
served through the OpenAI-compatible API:

```bash
python3 scripts/vllm/benchmarking/bench_decode_sliding_window.py \
  --base-url http://127.0.0.1:8000 --model <served-model-name> \
  --output-dir <DIR> --concurrency 16 \
  --prefill-tokens 100000 --decode-tokens 2048 \
  --max-num-batched-tokens <server's value>
```

`run_decode_benchmark.sh` serves a config with `run_benchmarks.sh --serve-only`
and runs the benchmark for each `ISL_OSL_CONFIGS` x `CONCURRENCY_OPTIONS`
entry of that config, writing `<results-dir>/decode/sweep.md`:

```bash
./scripts/vllm/benchmarking/run_decode_benchmark.sh \
  --config deepseek-v41-flash-tp8-ep-100k-decode --results-dir <DIR>
```

### DP8 cached-extend and steady DSpark decode CI

The `epic/dsv41` lane runs the audited recipes from
[`server_steps`](../../dsv41_bench/server_steps/README.md), using TP1/DP8/EP8,
FP8 KV, CED, Engram CPU offload and DSpark K5:

- Cached extend: C8, 98,304 cached prefix tokens plus 16,384 new input tokens,
  2,048-token chunks, one output token and five formal rounds. The model-default
  serving context limit and automatic KV sizing are retained.
- Steady decode: C384 (48 requests per DP), 98,304 input tokens and 512 output
  tokens, 2,048-token chunks, one warmup and one formal window. The serving
  context limit is 98,816; compile buckets are `[384,2048]`.

These replace the TP8/DP1 C64 DSpark and C1 cold-prefill CI steps. The ordinary
C4/C8/C16 decode and smoke/GSM8K steps remain separate. Related: #1794, #1890
and #1912; #1891 improves priming for the superseded TP8/DP1 C64 recipe.

`run_dsv41_dp8_benchmark.sh` mounts the checkpoint read-only through GCSFuse,
prepares scenario-specific LongBench-v2 inputs with the existing dataset SHA
check, and invokes `server_steps/run.py` with a fresh cold compilation cache.
The measured Run:ai streamer version is installed only in the disposable test
container. Compilation caches use the agent's persistent disk and are removed
when the job exits; model weights are streamed, not copied to that disk.

Buildkite publishes inputs, package versions, startup records, server/client
logs and `benchmark/fixed/round-*-summary.json` under:

- `perf_eval_results/deepseek-v41-flash-dp8-96k-c8-cached-extend/`
- `perf_eval_results/deepseek-v41-flash-dp8-96k-c384-decode/`

The existing harness rejects incomplete requests, incorrect cache hits,
preemptions, missing full-DP windows and compilation in formal windows. Prefill
reports new-input host cadence; decode counts accepted outputs including bonus
tokens. These metrics are not directly comparable with the old CI metrics.
There is no throughput regression threshold yet. CI measurements do not enable
profiling; use the same audited launcher with `--profile` for a separate capture.

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
| `EVAL_SOFT_FAIL` | `0` | `1` downgrades an eval regression to a `::warning` instead of failing the step. Perf regressions still fail. For a config whose eval baseline is new and whose run-to-run spread is not yet known — prefer it to `soft_fail` on the Buildkite step, which would mask the perf gate too. Forwarded into the container by `run_in_docker.sh` |
| `RANGE_RATIO_STYLE` (config var) | `symmetric` | How `RANDOM_RANGE_RATIO` is interpreted: `symmetric` = vllm bench serve native `[(1-r)L, (1+r)L]`; `min` = benchmark_serving.py-style `[rL, L]`, translated for vllm bench serve |
