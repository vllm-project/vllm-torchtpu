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

Compares the benchmark JSON to `baselines/perf/<chip>/<config>.baseline.json`,
with a 5% tolerance. Baselines are per TPU chip. `run_eval_flow.sh` reads the
chip from `TPU_ACCELERATOR_TYPE`, or from the GCE metadata when unset
(`tpu7x-8` is chip `v7x`; `v6e-8` is `v6e`):

- A chip with no baseline for the config reports its numbers without the gate
  and prints a `::warning::` saying so. Check in a calibrated baseline to gate
  it, or pass `--skip-perf-check` to report on purpose.
- A run that cannot read its chip stops before the server starts, unless it
  passes `--skip-perf-check`.

Gated metrics:

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
  --baseline scripts/vllm/benchmarking/baselines/perf/<chip>/<config>.baseline.json
```

### Accuracy (`--mode eval`)

Compares `lm_eval` accuracy to `baselines/eval/<config>.<task>.baseline.json`.
The default tolerance is 1.5pp; a config can override it with `EVAL_TOLERANCE`.
Only keys present in the baseline are gated.

Each task is compared on the accuracy metric of its baseline entry, the first
in the script's `ACC_KEYS` order when the entry has several. A task whose
results lack that metric fails as `(no result)`. No other accuracy metric of
the task stands in for it.

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

`--calibrate` writes a baseline as the median of the runs, one `--results-dir`
per run:

```bash
python3 scripts/vllm/benchmarking/check_regression.py \
  --mode eval --calibrate --task-filter <task> \
  --results-dir <RUN1> --results-dir <RUN2> --results-dir <RUN3>
```

`--task-filter` selects the tasks that go into the baseline. It takes
comma-separated tasks, or `all` for every task in the runs. Its default is
`mmlu_llama`, `mmlu_llama_humanities`, `mmlu_llama_stem`,
`mmlu_llama_social_sciences`, `mmlu_llama_other`, `mmlu_pro`,
`humaneval_plus_tpu`, and `mbpp_plus_tpu`. Any other task, such as `mmmu_pro`,
needs the flag.

`--calibrate` exits `2` instead of writing a baseline when a run has no
selected task or lacks an accuracy metric that another run has. A run may lack
a standard error. The baseline then takes the median of the standard errors
that the other runs reported.

## Accuracy Evaluation (`lm_eval`)

### Tasks

| Task | Enabled by | Endpoint | Gated metric |
|------|------------|----------|--------------|
| `mmlu_llama`, `mmlu_pro` | `--run-lm-eval` or `--eval-suites` | chat completions | `exact_match` |
| `gsm8k` | `--eval-suites gsm8k` (or `all`) | chat completions | `exact_match` |
| `humaneval_plus_tpu`, `mbpp_plus_tpu` | `--run-code-eval` or `--eval-suites` | chat completions | pass@1 |
| `mmmu_pro` | `--run-mm-eval` or `--eval-suites` (via `evalscope`) | chat completions | `acc` |
| `json_schema` | `--eval-suites json_schema` (or `all`) (via `eval_json_schema.py`) | chat completions | `acc`, which must be 1.0 |

- `--run-lm-eval` walks `LM_EVAL_TASKS` (default `mmlu_llama mmlu_pro`; configs
  may narrow it).
- The code tasks use the full EvalPlus datasets, with evalplus-style instruct
  prompting and code-block extraction (see `lm_eval_tasks/`).
- Each task gates only when its baseline file exists, except `json_schema`,
  which always gates.
- `json_schema` sends every prompt of a task config with its case's JSON Schema
  as `response_format` (`json_schema`, `strict`). A response is valid when it
  is not truncated (`finish_reason` other than `length`), parses as JSON, and
  satisfies the schema. An HTTP 4xx answer counts as an invalid response; a
  5xx answer or an unreachable endpoint fails the suite without a result.
  The pass rule is the same for every config and is not calibrated: `acc` must
  be 1.0 (`baselines/eval/json_schema.pass_rule.baseline.json`, tolerance 0).
  The default task config, `task_configs/json_schema_default.yaml`, has 28
  prompts in 4 cases; a config names another file with
  `JSON_SCHEMA_TASK_CONFIG`. `eval_json_schema.py` checks a fixed set of JSON
  Schema keywords and refuses a task config whose schema uses any other, so
  it never passes a constraint it did not check.

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

### `run_eval_flow.sh` flags

- `--base-url URL`: API root URL (not a full route like `/v1/chat/completions`) with optional scheme, port, and path prefix. Bare host defaults to `:8000`; trailing `/` and `/v1` on the path are stripped, and paths beneath `/v1/` are rejected. Cannot be combined with `--host` or `--port`. Connects to the endpoint instead of starting a local server.
- `--served-model-name NAME`: Model name sent in requests to a running server (`--base-url`, `--host`, or `--skip-benchmark` with the server already on `localhost`) when it differs from the config's `MODEL`. Before a run sends a request to a server it did not start (`--base-url`, `--host`, or `--skip-benchmark` with the server on `localhost`), it reads `<server>/v1/models` (sending `OPENAI_API_KEY` as a bearer token when set) and stops unless the name it will send is listed there. That name is `--served-model-name` when given, otherwise the config's `MODEL`. A lookup that fails (connection error, HTTP error, or a response that is not a model list) also stops the run.
- `--eval-suites LIST`: Comma-separated list of accuracy suites (`mmlu_llama`, `mmlu_pro`, `gsm8k`, `humaneval_plus_tpu`, `mbpp_plus_tpu`, `mmmu_pro`, `json_schema`, `all`, or `none`). `all` covers whichever of `mmlu_llama` and `mmlu_pro` the config's `LM_EVAL_TASKS` lists (both by default; a config that narrows the list to `mmlu_pro` gets no `mmlu_llama`), then `gsm8k`, `humaneval_plus_tpu`, and `mbpp_plus_tpu`, then `mmmu_pro` for `*multimodal*` configs, then `json_schema`. Precedence is `--eval-suites` > legacy `--run-*-eval` flags > config `EVAL_SUITES` > none.
- `--perf-suites LIST`: Comma-separated list of performance suites (`bench_serve`, `all`, or `none`; `all` currently runs `bench_serve`). Precedence matches eval suites: `--perf-suites` > legacy `--skip-benchmark` (which removes `bench_serve`) > config `PERF_SUITES` > `bench_serve`. Runs without `bench_serve` skip the perf regression check and the `ci.vllm.ai` dashboard publish. `--skip-benchmark` also decides which server the evals use: unless `--perf-suites` selects `bench_serve`, it never starts a server (even next to `--perf-suites none`) and sends the evals to the one already on `localhost:$PORT`, while `--perf-suites none` without `--skip-benchmark` runs `cleanup_server.sh` and starts a server for the evals. A `spec_bench` config cannot run eval-only on a server the script starts, because its servers exist only inside its `(BS, K)` sweep; the script rejects that before the cleanup, so point it at a running server with `--host`, `--base-url`, or `--skip-benchmark`.

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

`publish_perf_dashboard.py` sends a results directory's
`isl<N>_osl<N>_c<N>.json` files to the Performance tab of
[ci.vllm.ai](https://ci.vllm.ai/perf), next to the GPU runs of the same model.
A perf step opts in with `UPLOAD_TO_VLLM_CI: "1"` in its `env`;
`run_eval_flow.sh` then publishes that step's own results after the database
upload, and skips it on PRs and non-main branches like that upload. `--dry-run`
prints the rows without posting them.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `BENCHMARK_MODE` | `serving` | Runner mode: `serving` or `spec_bench` (also `--mode`) |
| `BENCHMARK_WARMUP_RUNS` | `0` | Full benchmark passes to discard before writing each gated result |
| `TPU_ACCELERATOR_TYPE` | GCE metadata | The TPU's accelerator type (`tpu7x`, `v6e-8`). `run_eval_flow.sh` reads the chip from it to pick the perf baselines. It reads the machine that runs the script, so a `--host` run must set it to the server's type |
| `EVAL_SOFT_FAIL` | `0` | `1` downgrades an eval regression to a `::warning` instead of failing the step. Perf regressions still fail. For a config whose eval baseline is new and whose run-to-run spread is not yet known — prefer it to `soft_fail` on the Buildkite step, which would mask the perf gate too. Forwarded into the pod by `.buildkite/kubernetes/common.sh` |
| `RANGE_RATIO_STYLE` (config var) | `symmetric` | How `RANDOM_RANGE_RATIO` is interpreted: `symmetric` = vllm bench serve native `[(1-r)L, (1+r)L]`; `min` = benchmark_serving.py-style `[rL, L]`, translated for vllm bench serve |
| `SPEC_METHOD` (config var) | (empty) | Speculative decoding method (`ngram`, `eagle3`, `mtp`, `dflash`, `dspark`); required when any `K > 0` |
| `SPEC_DRAFT_MODEL` / `SPEC_DRAFT_TP` (config var) | (empty) / `TENSOR_PARALLELISM` | Draft model and its TP size for methods that use a separate drafter |
| `SPEC_K_LIST` (config var) | `0 1 2 3 4 5` | Space-separated non-negative integer `K` values to sweep (`0` = baseline) |
| `SPEC_BS_LIST` (config var) | `1 4 8` | Space-separated positive integer batch sizes (`--max-num-seqs` and `--max-concurrency`) |
| `SPEC_BENCH_DATASET` / `SPEC_BENCH_CATEGORY` (config var) | auto-download / `summarization` | Path to Spec-Bench `question.jsonl` and category filter |
| `SPEC_NUM_PROMPTS` / `SPEC_WARMUP_PROMPTS` / `SPEC_REPS` (config var) | `100` / `20` / `2` | Prompts per measured pass, warmup prompts per cell, and measured repetitions per cell |
| `SERVED_MODEL_NAME` (config var) | (empty) | Optional `--served-model-name` passed in `spec_bench` mode |
| `EVAL_SUITES` (config var) | (empty) | Default comma-separated accuracy suites when neither `--eval-suites` nor `--run-*-eval` is passed |
| `PERF_SUITES` (config var) | (empty) | Default comma-separated performance suites (`bench_serve`, `all`, or `none`) when `--perf-suites` is omitted; legacy `--skip-benchmark` still removes `bench_serve` |
| `JSON_SCHEMA_TASK_CONFIG` (config var) | `task_configs/json_schema_default.yaml` | Task config of the `json_schema` suite. Read from the config file only, not from the environment |
| `OPENAI_API_KEY` | (empty) | Sent as a bearer token to `/v1/models` and by `eval_json_schema.py`, and as `--api-key` to `evalscope` (in addition to `lm_eval` and `vllm bench serve`). `FORWARD` in `.buildkite/kubernetes/common.sh` does not name it, so a key set on a Buildkite kube step does not reach the pod; a gateway that requires the key then answers the `/v1/models` lookup with 401 and the run stops |
