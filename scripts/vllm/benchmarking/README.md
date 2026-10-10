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

### C64 DSpark and Engram offload CI

The `epic/dsv41` lane includes
`deepseek-v41-flash-tp8-ep-100k-decode-c64-dspark-offload`: C64, 98,000 input
tokens, 5,000 output tokens, TP8/EP8/DP1, FP8 KV, DSpark K=5, and Engram CPU
offload. The maximum model length is 103424. It uses 8192-token prefill chunks
and `[64,384,8192]` compile buckets.
SP and local Q are off; SparseCore MoE is on. These settings are separate from
the C4/C8/C16 baseline. The job warms up, measures one round, and keeps all
benchmark and server artifacts. It has no throughput regression threshold yet.

After a complete valid round on a non-PR `epic/dsv41` build,
`run_dsv41_c64_ci.sh` writes to the
[`decode-100k-ci` tab](https://docs.google.com/spreadsheets/d/1A2nrPEJgRd6wYGSqW5IZwaHDYph6viYH40kuRaBMpzw/edit#gid=1542472756).
PR builds, other branches, and runs without explicit Buildkite branch/PR
metadata keep benchmark artifacts without accessing Sheets.
Columns A:R remain intact on existing rows. Columns S:Z hold C64 peak-window
throughput, stream ITL P50, TTFT P50, full-overlap throughput, job URL, agent,
workload, and effective TPOT. Stream ITL P50 may be zero when DSpark returns
several tokens in one chunk. Effective TPOT is derived as
`1000 * 64 / full_overlap_throughput` in milliseconds; it is not a percentile.
The build number and commit identify the row. Retrying the job updates
that row; the Buildkite concurrency group serializes these writers. Other sheet
writers must preserve S:Z and avoid replacing the whole tab.

The upload rejects incomplete output, invalid concurrency overlap, missing or
nonzero preemptions, and missing metrics. It preserves `sheet-row.json` before
upload and writes `sheet-upload.json` only after a successful API response.
An upload failure emits a warning and retains the benchmark artifacts without
failing the CI step, matching `run_eval_flow.sh`. Benchmark failures still fail
the step and prevent upload.
For a local row preview, run `upload_decode_sheet.py --dry-run` with the summary,
spreadsheet ID, sheet ID, and the run's `BUILDKITE_BUILD_NUMBER`,
`BUILDKITE_BUILD_URL`, `BUILDKITE_JOB_ID`, and `BUILDKITE_COMMIT` in the environment.

The CI image's `benchmarking` extra includes the Google auth client. Upload uses
Application Default Credentials with the Sheets scope. Grant the CI identity
Editor access to the spreadsheet and enable the Sheets API in its project.
When the agent's credentials cannot request the Sheets scope, set
`DSV41_SHEETS_SERVICE_ACCOUNT` in the Buildkite agent environment to a dedicated
sheet writer. The source identity needs Service Account Token Creator on that
account, the IAM Credentials API must be enabled, and that account needs Editor
access to the sheet. See the
[Google auth impersonation guide](https://google-auth.readthedocs.io/en/latest/reference/google.auth.impersonated_credentials.html).
No personal OAuth credentials or service-account keys are stored in the repo.
The PR does not provision those permissions; verify them before enabling upload.

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
