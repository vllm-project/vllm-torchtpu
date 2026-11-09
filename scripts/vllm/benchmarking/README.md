# Benchmarking vllm-torchtpu

All configs target TPU v7x-8 (TP=8). KV cache is forced to fp8 in `run_benchmarks.sh`.

## Quick Start

```bash
# Guard configs (scheduled perf builds)
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-30b-fp8-tp8-ep

# Nightly sweeps
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-30b-tp8-ep
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3.5-35b-fp8-tp4-ep
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3.5-35b-fp8-tp8-ep
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
| `qwen3.5-35b-fp8-tp8-ep` | Qwen3.5-35B-A3B-FP8 | yes | on | fp8 | nightly |
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

## Speculative Decoding (`run_benchmarks.sh --mode spec_bench`)

`run_benchmarks.sh` supports two runner modes via `--mode serving|spec_bench`
(or `BENCHMARK_MODE=serving|spec_bench`; distinct from `check_regression.py
--mode perf|eval` above). In `spec_bench` mode, the runner sweeps batch size
(`SPEC_BS_LIST`) × speculative draft tokens (`SPEC_K_LIST`) on the
[Spec-Bench](https://github.com/hemingkx/Spec-Bench) dataset:

1. Restarts `vllm serve` per `(BS, K)` cell with `--max-num-seqs <BS>` and
   `--speculative-config` (`K=0` starts without `--speculative-config` as the
   no-speculation baseline).
2. Runs `SPEC_WARMUP_PROMPTS` warmup prompts and `SPEC_REPS` measured passes of
   `vllm bench serve --dataset-name spec_bench --max-concurrency <BS>`.
3. Saves per-cell results to `bs<BS>/k<K>_rep<N>.json` and aggregates mean E2E
   time, speedup vs. `K=0`, and acceptance rate via `summarize_spec_bench.py`
   into `spec_benchmark_summary.json` and `spec_benchmark_summary.md`.

```bash
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-0.6b-ngram
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
  --config <model-config> --results-dir <DIR>
```

### DP8 cached-extend and DP16 steady DSpark decode CI

The `epic/dsv41` lane runs the audited recipes from
[`server_steps`](../../dsv41_bench/server_steps/README.md), using TP1, EP,
FP8 KV, CED, Engram CPU offload and DSpark K5:

- Cached extend: DP8, C8, 98,304 cached prefix tokens plus 16,384 new input tokens,
  2,048-token chunks, one output token and five formal rounds. Compile
  buckets are `[128,2048]`. The model-default serving context limit and automatic KV
  sizing are retained.
- Steady decode: DP16 on eight physical chips, C768 (48 requests per DP),
  push MoE with dynamic EPLB, L1 Engram HBM, 98,304 input tokens and 512 output
  tokens, 2,048-token chunks, one warmup and one formal window. The serving
  context limit is 98,816; compile buckets are `[288,2048]`.

Both jobs run on every build of this lane, including PR builds admitted by the
repository's `ready` label check.

These replace the TP8/DP1 C64 DSpark and C1 cold-prefill CI steps. Their old
config scripts, the unused DSpark config helper and prefill perf baseline are
removed. The ordinary
C4/C8/C16 decode and smoke/GSM8K steps remain separate. Related: #1794, #1890
and #1912; #1891 improves priming for the superseded TP8/DP1 C64 recipe.

`run_dsv41_benchmark.sh` copies the checkpoint's non-weight files (config,
tokenizer, custom code) to scratch and prepares scenario-specific LongBench-v2
inputs with the existing dataset SHA check. The target and DSpark models stream
weights directly from the GCS URI; tokenization uses the local copy. The wrapper uses the shared CI image's
dependencies and records their versions. `server_steps/run.py` disables
persistent compilation caches through environment variables and retains the
compiler's default serialization path.

Buildkite publishes inputs, package versions, startup records, server/client
logs and `benchmark/fixed/round-*-summary.json` under:

- `perf_eval_results/deepseek-v41-flash-dp8-96k-c8-cached-extend/`
- `perf_eval_results/deepseek-v41-flash-dp16-96k-c768-decode/`

Each job also publishes a native GitHub Check with global and per-chip
throughput, the median across formal rounds, benchmark duration and commit.
Open the PR's **Checks** tab and select **DeepSeek-V4.1 DP8 Prefill throughput**
or **DeepSeek-V4.1 DP16 Decode throughput**; reading the summary requires no
Buildkite account. Duration runs from server launch through shutdown and
excludes CI queueing, image setup and input preparation. Failed commands or
incomplete results report no valid throughput. `dsv41_publish_check.sh` runs in
the pod after the benchmark and updates `notify.github_check.output.summary`;
this requires Buildkite's GitHub App integration. Publication errors fail the
step, while an existing command failure keeps its original exit code.

The existing harness rejects incomplete requests, incorrect cache hits,
preemptions, missing full-DP windows and compilation in formal windows. Prefill
reports new-input host cadence. Decode's primary `decode_throughput` is global
concurrency times the actual mean accepted output length, including bonus
tokens, divided by the median full-DP step interval. The observed window rate
remains in `categories.decode`. These metrics are not directly comparable with
the old CI metrics.
There is no throughput regression threshold yet. CI measurements do not enable
profiling; use the same audited launcher with `--profile` for a separate capture.

### Spec-Bench Configs

| Config | Model | Method | Draft | K swept | Topology |
|--------|-------|--------|-------|---------|----------|
| `qwen3-0.6b-ngram` | Qwen3-0.6B | `ngram` | - | `0 1 2 4` | TP=1 (smoke) |
| `llama-3-8b-ngram` | Meta-Llama-3-8B-Instruct | `ngram` | - | `0 1 2 4 8` | TP=1 |
| `llama-3.3-70b-eagle3` | Llama-3.3-70B-Instruct | `eagle3` | `Llama-3.3-70B-Instruct-Eagle3` | `0 1 2 3 4` | TP=8 |
| `gemma-4-31b-it-mtp` | gemma-4-31B-it | `mtp` | `gemma-4-31B-it-assistant` | `0 1 2 3 4 5` | TP=8 |
| `qwen3.5-397b-fp8-mtp` | Qwen3.5-397B-A17B-FP8 | `mtp` | built-in MTP head | `0 1 2 3 4 5` | TP=8, EP |
| `qwen3-4b-dflash` | Qwen3-4B | `dflash` | `Qwen3-4B-DFlash-b16` | `0 15` | TP=1 |
| `qwen3-4b-dspark` | Qwen3-4B | `dspark` | `dspark_qwen3_4b_block7` | `0 7` | TP=1 |
| `qwen3.5-35b-fp8-dflash` | Qwen3.5-35B-A3B-FP8 | `dflash` | `Qwen3.5-35B-A3B-DFlash` | `0 15` | DP=2 × TP=4, EP |

## Prerequisites

- vLLM installed (provides `benchmark_serving.py`)
- `vllm_torchtpu` installed (this repo)
- TPU device available
- Model weights downloaded (or use `--load-format dummy` via config)

## Results

Results are saved to `benchmark_runs/` (gitignored). Each run creates a directory containing:

- `config.json` — benchmark parameters
- `isl<N>_osl<N>_c<N>.json` — per-combo results from benchmark_serving.py (`serving` mode)
- `bs<BS>/k<K>_rep<N>.json`, `spec_benchmark_summary.json`, `spec_benchmark_summary.md` — per-cell results and aggregated speedup/acceptance tables (`spec_bench` mode)
- `server.log` — vLLM server output
- `benchmark.log` — run metadata

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `BENCHMARK_MODE` | `serving` | Runner mode: `serving` or `spec_bench` (also `--mode`) |
| `BENCHMARK_WARMUP_RUNS` | `0` | Full benchmark passes to discard before writing each gated result |
| `EVAL_SOFT_FAIL` | `0` | `1` downgrades an eval regression to a `::warning` instead of failing the step. Perf regressions still fail. For a config whose eval baseline is new and whose run-to-run spread is not yet known — prefer it to `soft_fail` on the Buildkite step, which would mask the perf gate too. Forwarded into the pod by `.buildkite/kubernetes/common.sh` |
| `RANGE_RATIO_STYLE` (config var) | `symmetric` | How `RANDOM_RANGE_RATIO` is interpreted: `symmetric` = vllm bench serve native `[(1-r)L, (1+r)L]`; `min` = benchmark_serving.py-style `[rL, L]`, translated for vllm bench serve |
| `SPEC_METHOD` (config var) | (empty) | Speculative decoding method (`ngram`, `eagle3`, `mtp`, `dflash`, `dspark`); required when any `K > 0` |
| `SPEC_DRAFT_MODEL` / `SPEC_DRAFT_TP` (config var) | (empty) / `TENSOR_PARALLELISM` | Draft model and its TP size for methods that use a separate drafter |
| `SPEC_K_LIST` (config var) | `0 1 2 3 4 5` | Space-separated non-negative integer `K` values to sweep (`0` = baseline) |
| `SPEC_BS_LIST` (config var) | `1 4 8` | Space-separated positive integer batch sizes (`--max-num-seqs` and `--max-concurrency`) |
| `SPEC_BENCH_DATASET` / `SPEC_BENCH_CATEGORY` (config var) | auto-download / `summarization` | Path to Spec-Bench `question.jsonl` and category filter |
| `SPEC_NUM_PROMPTS` / `SPEC_WARMUP_PROMPTS` / `SPEC_REPS` (config var) | `100` / `20` / `2` | Prompts per measured pass, warmup prompts per cell, and measured repetitions per cell |
| `SERVED_MODEL_NAME` (config var) | (empty) | Optional `--served-model-name` passed in `spec_bench` mode |
