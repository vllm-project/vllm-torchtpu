# Scenario reference

Generated from the models by `uv run --project tools/scenario scenario dev regen`; do not edit by hand.

## Layers and merging

Values come from layers, later wins: model defaults (tool) <
`scenarios/defaults.yaml` < the `extends` chain, root parent first < the
scenario < its sweep point < `--set`, in order. Task defaults apply only to
the tasks a later layer declares. `name`, `extends` and baselines are never
inherited.

Mappings merge key by key, lists and scalars replace, and null deletes the
key. In `server.roles.<role>.args` and `.env` a null removes the inherited
server-level key for that role. `tasks.eval.lm_eval.gen_kwargs` replaces as
a whole, since lm-eval reads it as one value.

## Fields

### Scenario

One workload: the servers to start and the tasks to run against them.

| Key | Type | Default | Description |
|---|---|---|---|
| `schema` | `1` | required | Schema version; always 1. |
| `name` | name | required | Unique across scenarios/; names the results directory. Not inherited through extends. |
| `extends` | scenario reference or relative path or null | unset | Parent scenario, merged under this file: mappings merge, lists and scalars replace, null deletes. Baselines are not inherited. |
| `server` | [Server](#server) | required | What runs. |
| `tasks` | [Tasks](#tasks) | `{}` | What runs against the servers. |
| `sweep` | [Sweep](#sweep) or null | unset | Server-side sweep: each point is a fresh server with its overrides. |

### Server

What runs: image, model, slice or roles, flags and env.

| Key | Type | Default | Description |
|---|---|---|---|
| `image` | `ci` or an image URI | required; defaults.yaml can set it | ci for this build's CI image, otherwise a full image URI with a tag or digest. |
| `model` | [ModelSpec](#modelspec) | required | Served name and load path. |
| `slice` | slice: accelerator/topology or null | unset | Slice of the single role. Exactly one of slice and roles. |
| `roles` | mapping of role name to [Role](#role) or null | unset | Two or more roles, e.g. prefill and decode, in run order. Exactly one of slice and roles. |
| `multihost` | `mp` or `ray` or null | unset | How a multi-host slice runs one engine: vLLM's own multi-node DP (mp) or Ray. Required exactly when a slice has more than one host. |
| `startup_timeout` | duration: 2h, 90m, 4h10m, 30s or seconds | required; defaults.yaml can set it | Engine startup budget: sets VLLM_ENGINE_READY_TIMEOUT_S, and the readiness wait is this plus a margin. |
| `args` | arg map: flag name to value | `{}` | vllm serve flags for every role: true -> --k, false -> --no-k, a scalar -> --k v, a list or mapping -> --k \<compact JSON>. |
| `env` | env map: name to string | `{}` | Env for every server process of every role. |
| `router` | [Router](#router) or null | unset | Proxy in front of the roles; only with roles. |
| `scratch` | size, e.g. 64Gi or null | unset | Temporary space every server host needs, e.g. 64Gi; the target decides where it comes from. |

### ModelSpec

The model a server loads and the name clients call it by.

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | string | required | What clients call the model; always passed as --served-model-name. |
| `path` | string or null | unset | What the server loads (--model): a Hugging Face id, a local path, or a gs:// or s3:// URI. Defaults to name. |

### Role

One engine of a multi-role scenario, e.g. prefill or decode.

| Key | Type | Default | Description |
|---|---|---|---|
| `slice` | slice: accelerator/topology | required | Slice this role runs on, e.g. tpu7x/2x2x1. |
| `args` | arg map; null removes an inherited flag | `{}` | vllm serve flags over server.args for this role; null removes an inherited flag. |
| `env` | env map; null removes an inherited entry | `{}` | Env over server.env for this role; null removes an inherited entry. |
| `kv` | [Kv](#kv) or null | unset | KV transfer role of this engine. |

### Kv

The KV transfer side of one engine in a disaggregated scenario.

| Key | Type | Default | Description |
|---|---|---|---|
| `connector` | `TPURaidenConnector` | required | KV connector; the tool adds its module path and port wiring. |
| `role` | `producer` or `consumer` or `both` | required | KV role of this engine: kv_producer, kv_consumer or kv_both. |

### Router

The proxy in front of a multi-role scenario's engines.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | `toy-proxy` | required | Selects the wiring flags the tool adds (hosts and ports of the producer and consumer roles). |
| `command` | non-empty list of string | required | Argv the wiring flags are appended to, e.g. [python3, examples/disagg/toy_proxy_server.py]. |
| `args` | arg map: flag name to value | `{}` | Extra router flags, rendered like server.args. |

### Tasks

What runs against the servers. Empty means serve only.

| Key | Type | Default | Description |
|---|---|---|---|
| `driver` | role name or `client` or null | unset | Where tasks and the router run: a role (its first host) or client (a chipless host). Required with server.roles. |
| `perf` | [PerfTask](#perftask) or null | unset | Benchmark cells. |
| `agentx` | [AgentxTask](#agentxtask) or null | unset | AgentX trace replay; must be the only task. |
| `eval` | [EvalTask](#evaltask) or null | unset | Accuracy evals. |
| `custom` | list of [CustomTask](#customtask) | `[]` | Commands, run after perf and before eval. |

### PerfTask

Fixed-length benchmark cells against the server or router.

| Key | Type | Default | Description |
|---|---|---|---|
| `client` | `vllm-bench` or `inferencex` | `vllm-bench` | vllm-bench runs vllm bench serve; inferencex runs InferenceX's benchmark_serving at the version baked into the image. |
| `dataset` | [Dataset](#dataset) | required; defaults.yaml can set it | Prompt source. |
| `isl_osl` | list of [input length, output length] or null | unset | [input length, output length] pairs; one cell per pair and concurrency. |
| `concurrency` | non-empty list of positive integer or null | unset | Client concurrency levels. |
| `num_prompts` | positive integer or [PerConcurrency](#perconcurrency) | required; defaults.yaml can set it | Prompts per measured run: a number or {per_concurrency: N}. |
| `num_warmups` | non-negative integer or [PerConcurrency](#perconcurrency) or null | unset | Warmup requests before each measured run (--num-warmups): a number or {per_concurrency: N}. |
| `cells` | non-empty list of [PerfCellGroup](#perfcellgroup) or null | unset | Explicit cells instead of isl_osl x concurrency. |
| `random_range_ratio` | number in [0, 1] or null | unset | Length jitter of the random dataset; required with it. |
| `range_ratio_style` | `symmetric` or `min` or null | unset | How random_range_ratio reads for vllm-bench: symmetric is vllm bench serve's own form; min is translated to it as run_benchmarks.sh does. Not used by inferencex. |
| `warmup_runs` | non-negative integer | required; defaults.yaml can set it | Whole warmup passes per cell, discarded, before the measured run. |
| `temperature` | number >= 0 or null | unset | Sampling temperature; unset sends none (server default). |
| `extra_body` | mapping of string to any JSON value or null | unset | JSON merged into every request (--extra-body). |
| `tokenizer` | string or null | unset | Client tokenizer. Unset: the server's pulled weights for an object-storage model, else model.name. |
| `percentile_metrics` | non-empty list of `ttft` or `tpot` or `itl` or `e2el` | required; defaults.yaml can set it | --percentile-metrics. |
| `client_args` | arg map: flag name to value | `{}` | Extra client flags, rendered like server.args. |
| `env` | env map: name to string | `{}` | Env for the client process, passed through untouched. |
| `tolerance` | number > 0 and \< 1 | required; defaults.yaml can set it | Relative tolerance of the perf gate. |
| `baseline` | [PerfBaseline](#perfbaseline) or [BaselineRef](#baselineref) or null | unset | Inline baseline keyed by cell, or {from: \<ref>}. |

### Dataset

Where perf prompts come from.

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | `random` or `sharegpt` | required | --dataset-name. Others are added with a lane that uses them. |
| `path` | string or null | unset | --dataset-path, repo-relative or absolute. |

### PerConcurrency

A count that scales with each cell's concurrency.

| Key | Type | Default | Description |
|---|---|---|---|
| `per_concurrency` | positive integer | required | Multiplied by the cell's concurrency: N x concurrency. |

### PerfCellGroup

Explicit perf cells, as an alternative to the isl_osl x concurrency product.

| Key | Type | Default | Description |
|---|---|---|---|
| `label` | string or null | unset | Cell key and --label. May use {isl}, {osl}, {concurrency} and {repeat}; without it the key is isl{I}_osl{O}_c{C}. |
| `isl` | positive integer or non-empty list of positive integer | required | Input length, or several. |
| `osl` | positive integer or non-empty list of positive integer | required | Output length, or several. |
| `concurrency` | positive integer or non-empty list of positive integer | required | Client concurrency, or several. |
| `num_prompts` | positive integer or [PerConcurrency](#perconcurrency) or null | unset | Prompts per cell; defaults to the task's. |
| `percentile_metrics` | non-empty list of `ttft` or `tpot` or `itl` or `e2el` or null | unset | --percentile-metrics; defaults to the task's. |
| `repeat` | positive integer | `1` | Runs of each cell; more than one needs {repeat} in label. |

### PerfBaseline

Perf gate baseline: cell key -> gated values.

| Key | Type | Default | Description |
|---|---|---|---|
| `_source` | mapping of string to any JSON value or null | unset | Where the numbers came from; documentation only, the gate ignores it. |
| other keys | [PerfCellBaseline](#perfcellbaseline) |  | Entries of the mapping. |

### PerfCellBaseline

Gated values of one perf cell, in check_regression.py's terms.

| Key | Type | Default | Description |
|---|---|---|---|
| `completed` | non-negative integer or null | unset | Requests completed; a run must complete at least 95% as many. |
| `num_prompts` | positive integer or null | unset | Prompts sent at calibration; not gated. |
| `median_ttft_ms` | number >= 0 or null | unset | Median time to first token, ms; gated only when present. |
| `median_tpot_ms` | number >= 0 or null | unset | Median time per output token, ms. |
| `total_token_throughput` | number >= 0 or null | unset | Input and output tokens per second. |
| `output_token_throughput` | number >= 0 or null | unset | Output tokens per second. |
| `output_throughput` | number >= 0 or null | unset | Older name of output_token_throughput, still read by the gate. |

### BaselineRef

Gate on another scenario's inline baseline for the same task.

| Key | Type | Default | Description |
|---|---|---|---|
| `from` | scenario reference | required | Reference id of the scenario whose inline baseline for the same task (and, in a sweep, the same point) gates this one. |

### AgentxTask

InferenceX AgentX trace replay at one concurrency.

| Key | Type | Default | Description |
|---|---|---|---|
| `concurrency` | positive integer | required | Concurrent sessions; one value per server. |
| `duration` | duration: 2h, 90m, 4h10m, 30s or seconds | required; defaults.yaml can set it | How long the replay runs. |
| `timeout` | duration: 2h, 90m, 4h10m, 30s or seconds | required; defaults.yaml can set it | The task is killed and fails after this; longer than duration, to cover warmup and drain. |
| `labels` | [AgentxLabels](#agentxlabels) | required; defaults.yaml can set it | Result row labels. |
| `env` | env map: name to string | `{}` | Client settings passed through untouched, e.g. AIPERF_MAX_CONTEXT_LENGTH. |
| `tolerance` | number > 0 and \< 1 | required; defaults.yaml can set it | Relative tolerance of the AgentX gate. |
| `baseline` | [AgentxBaseline](#agentxbaseline) or [BaselineRef](#baselineref) or null | unset | Inline baseline by metric, or {from: \<ref>}. |

### AgentxLabels

What InferenceX files the AgentX result row under.

| Key | Type | Default | Description |
|---|---|---|---|
| `model_prefix` | label | required; defaults.yaml can set it | InferenceX model prefix; also selects the trace corpus. |
| `precision` | label | required; defaults.yaml can set it | Precision label, e.g. fp4. |
| `framework` | label | required; defaults.yaml can set it | Serving framework label; selects InferenceX's metrics parser. |
| `spec_decoding` | label | required; defaults.yaml can set it | Speculative decoding method recorded with the row, or none. |

### AgentxBaseline

AgentX gate baseline: metric name -> value, for the point's concurrency.

| Key | Type | Default | Description |
|---|---|---|---|
| `_source` | mapping of string to any JSON value or null | unset | Where the numbers came from; documentation only, the gate ignores it. |
| other keys | number |  | Entries of the mapping. |

### EvalTask

Accuracy evals, gated against a baseline.

| Key | Type | Default | Description |
|---|---|---|---|
| `lm_eval` | [LmEval](#lmeval) | required | lm-eval runs. |
| `tolerance` | number > 0 and \< 1 | required | Smallest score drop the eval gate allows, in absolute terms; the gate widens it by the measured stderr. |
| `baseline` | [EvalBaseline](#evalbaseline) or [BaselineRef](#baselineref) or null | unset | Inline baseline by run key, or {from: \<ref>}. |

### LmEval

lm-eval against the endpoint's /v1/chat/completions, one run per task.

| Key | Type | Default | Description |
|---|---|---|---|
| `tasks` | non-empty list of lm_eval task name or [LmEvalTask](#lmevaltask) | required | Task names, or mappings with per-task options. |
| `model` | string | required; defaults.yaml can set it | --model, e.g. local-chat-completions. |
| `model_args` | mapping of model_args key to string, number or boolean | `{}` | --model_args entries; the tool adds model and base_url. |
| `gen_kwargs` | string or mapping of string to any JSON value or null | unset | --gen_kwargs: a string verbatim, or a mapping as JSON. |
| `limit` | positive integer or `all` or null | unset | Questions per task (per subtask); unset or all runs the full set. |
| `seed` | string | required; defaults.yaml can set it | --seed. |
| `apply_chat_template` | boolean | required; defaults.yaml can set it | --apply_chat_template. |
| `fewshot_as_multiturn` | boolean | required; defaults.yaml can set it | --fewshot_as_multiturn true. |
| `concurrency` | non-empty list of positive integer or null | unset | Client-side sweep: every task runs once per value, with num_concurrent set to it, against one server. |

### LmEvalTask

One lm_eval task with options of its own.

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | lm_eval task name | required | lm_eval task name. |
| `limit` | positive integer or `all` or null | unset | Questions (per subtask); overrides lm_eval.limit; all runs the full set. |
| `limit_per_concurrency` | positive integer or null | unset | Raises the limit at each lm_eval.concurrency value to ceil(N x concurrency / subtasks), so that concurrency is reached. |
| `subtasks` | positive integer or null | unset | Subtasks lm_eval applies the limit to one by one, e.g. 14 for mmlu_pro; divides limit_per_concurrency. |
| `gen_kwargs` | string or mapping of string to any JSON value or null | unset | Replaces lm_eval.gen_kwargs for this task. |
| `fewshot_as_multiturn` | boolean or null | unset | Overrides lm_eval.fewshot_as_multiturn. |
| `include_path` | string or null | unset | --include_path, for tasks defined in this repo. |
| `confirm_run_unsafe_code` | boolean | `false` | --confirm_run_unsafe_code, for code tasks. |
| `env` | env map: name to string | `{}` | Env of this task's lm_eval run, e.g. HF_ALLOW_CODE_EVAL. |

### EvalBaseline

Eval gate baseline: lm_eval run key -> {"<metric>,<filter>": value}.

| Key | Type | Default | Description |
|---|---|---|---|
| `_source` | mapping of string to any JSON value or null | unset | Where the numbers came from; documentation only, the gate ignores it. |
| other keys | mapping of metric,filter to number |  | Entries of the mapping. |

### CustomTask

A command run against the servers, e.g. a correctness script.

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | task name | required | Unique name; selects the task in --only and names its log. |
| `command` | non-empty list of string | required | Argv, run from the repo root. |
| `env` | env map: name to string | `{}` | Extra env. Values may use ${NAME} for names the tool provides (e.g. ${SCENARIO_RESULTS_DIR}); $$ is a literal $. |
| `timeout` | duration: 2h, 90m, 4h10m, 30s or seconds | required | The task is killed and fails after this. |
| `skip_rest_if_exit_ge` | positive integer or null | unset | Skip the remaining tasks when this one exits with this code or higher. |

### Sweep

Points that each run on a fresh server. Exactly one of points, zip, product.

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | string or null | unset | Point name template for zip and product, e.g. c{tasks.agentx.concurrency}; each {path} is one of the axes. |
| `points` | non-empty list of [SweepPoint](#sweeppoint) or null | unset | Named points, for irregular tables. |
| `zip` | non-empty mapping of key path to non-empty list of any JSON value or null | unset | Dotted path -> values; equal lengths, point i takes value i of every axis. |
| `product` | non-empty mapping of key path to non-empty list of any JSON value or null | unset | Dotted path -> values; one point per combination, the first axis outermost. |
| `baselines` | mapping of point name to [PointBaselines](#pointbaselines) or null | unset | Baselines of zip and product points, by point name and task. Named points carry their own. |

### SweepPoint

A named point: an overlay over the scenario for one server.

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | point name | required | Unique; results and baselines use it. |
| `server` | mapping of string to any JSON value | `{}` | Overlay on server, merged above the scenario for this point. |
| `tasks` | mapping of string to any JSON value | `{}` | Overlay on tasks; also holds this point's baselines. |

### PointBaselines

Baselines of one generated sweep point, by task.

| Key | Type | Default | Description |
|---|---|---|---|
| `perf` | [PerfBaseline](#perfbaseline) or [BaselineRef](#baselineref) or null | unset | The point's tasks.perf baseline. |
| `agentx` | [AgentxBaseline](#agentxbaseline) or [BaselineRef](#baselineref) or null | unset | The point's tasks.agentx baseline. |
| `eval` | [EvalBaseline](#evalbaseline) or [BaselineRef](#baselineref) or null | unset | The point's tasks.eval baseline. |

## defaults.yaml fields

### Defaults

scenarios/defaults.yaml: the choices every scenario starts from.

| Key | Type | Default | Description |
|---|---|---|---|
| `schema` | `1` | required | Schema version; always 1. |
| `server` | [DefaultsServer](#defaultsserver) or null | unset | Applied under every scenario. |
| `tasks` | [DefaultsTasks](#defaultstasks) or null | unset | Applied only to the tasks a scenario declares. |

### DefaultsServer

Server values applied under every scenario.

| Key | Type | Default | Description |
|---|---|---|---|
| `image` | `ci` or an image URI or null | unset | ci for this build's CI image, otherwise a full image URI with a tag or digest. |
| `startup_timeout` | duration: 2h, 90m, 4h10m, 30s or seconds or null | unset | Engine startup budget: sets VLLM_ENGINE_READY_TIMEOUT_S, and the readiness wait is this plus a margin. |
| `args` | arg map: flag name to value or null | unset | vllm serve flags for every role: true -> --k, false -> --no-k, a scalar -> --k v, a list or mapping -> --k \<compact JSON>. |
| `env` | env map: name to string or null | unset | Env for every server process of every role. |

### DefaultsTasks

Task values, each applied only to scenarios that declare that task.

| Key | Type | Default | Description |
|---|---|---|---|
| `perf` | [DefaultsPerf](#defaultsperf) or null | unset | Defaults for tasks.perf. |
| `agentx` | [DefaultsAgentx](#defaultsagentx) or null | unset | Defaults for tasks.agentx. |
| `eval` | [DefaultsEval](#defaultseval) or null | unset | Defaults for tasks.eval. |

### DefaultsPerf

tasks.perf values, applied only to scenarios that declare tasks.perf.

| Key | Type | Default | Description |
|---|---|---|---|
| `dataset` | [Dataset](#dataset) or null | unset | Prompt source. |
| `num_prompts` | positive integer or [PerConcurrency](#perconcurrency) or null | unset | Prompts per measured run: a number or {per_concurrency: N}. |
| `num_warmups` | non-negative integer or [PerConcurrency](#perconcurrency) or null | unset | Warmup requests before each measured run (--num-warmups): a number or {per_concurrency: N}. |
| `range_ratio_style` | `symmetric` or `min` or null | unset | How random_range_ratio reads for vllm-bench: symmetric is vllm bench serve's own form; min is translated to it as run_benchmarks.sh does. Not used by inferencex. |
| `warmup_runs` | non-negative integer or null | unset | Whole warmup passes per cell, discarded, before the measured run. |
| `percentile_metrics` | non-empty list of `ttft` or `tpot` or `itl` or `e2el` or null | unset | --percentile-metrics. |
| `client_args` | arg map: flag name to value or null | unset | Extra client flags, rendered like server.args. |
| `tolerance` | number in (0, 1) or null | unset | Relative tolerance of the perf gate. |

### DefaultsAgentx

tasks.agentx values, applied only to scenarios that declare tasks.agentx.

| Key | Type | Default | Description |
|---|---|---|---|
| `duration` | duration: 2h, 90m, 4h10m, 30s or seconds or null | unset | How long the replay runs. |
| `timeout` | duration: 2h, 90m, 4h10m, 30s or seconds or null | unset | The task is killed and fails after this; longer than duration, to cover warmup and drain. |
| `labels` | [DefaultsAgentxLabels](#defaultsagentxlabels) or null | unset | Result row labels. |
| `tolerance` | number in (0, 1) or null | unset | Relative tolerance of the AgentX gate. |

### DefaultsAgentxLabels

tasks.agentx.labels values.

| Key | Type | Default | Description |
|---|---|---|---|
| `model_prefix` | label or null | unset | InferenceX model prefix; also selects the trace corpus. |
| `precision` | label or null | unset | Precision label, e.g. fp4. |
| `framework` | label or null | unset | Serving framework label; selects InferenceX's metrics parser. |
| `spec_decoding` | label or null | unset | Speculative decoding method recorded with the row, or none. |

### DefaultsEval

tasks.eval values.

| Key | Type | Default | Description |
|---|---|---|---|
| `lm_eval` | [DefaultsLmEval](#defaultslmeval) or null | unset | Defaults for tasks.eval.lm_eval. |

### DefaultsLmEval

tasks.eval.lm_eval values, applied only where tasks.eval.lm_eval is declared.

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | string or null | unset | --model, e.g. local-chat-completions. |
| `model_args` | mapping of model_args key to string, number or boolean or null | unset | --model_args entries; the tool adds model and base_url. |
| `gen_kwargs` | string or mapping of string to any JSON value or null | unset | --gen_kwargs: a string verbatim, or a mapping as JSON. |
| `limit` | positive integer or `all` or null | unset | Questions per task (per subtask); unset or all runs the full set. |
| `seed` | string or null | unset | --seed. |
| `apply_chat_template` | boolean or null | unset | --apply_chat_template. |
| `fewshot_as_multiturn` | boolean or null | unset | --fewshot_as_multiturn true. |

## Arg rendering

| Value | argv |
|---|---|
| true | --k; for a dotted key, --a.b true |
| false | --no-k; for a dotted key, --a.b false |
| integer, number, string | --k \<value> |
| list, mapping | --k \<compact JSON>, key order kept |
| `null` | removes the key when layers merge; never rendered |

## Validation rules

| Rule | Checked at | Summary |
|---|---|---|
| V-01 | field | schema is 1 |
| V-02 | scenario | exactly one of server.slice and server.roles |
| V-03 | scenario | server.roles has two or more roles |
| V-04 | scenario | server.multihost is set exactly when a slice has more than one host |
| V-05 | scenario | server.router only with server.roles |
| V-06 | scenario | router toy-proxy fronts one producer and one consumer role |
| V-07 | field | every kv block uses the same connector |
| V-08 | scenario | tasks.driver names a role or client, and is required with roles |
| V-09 | scenario | an object-storage model path has a loader that reads it |
| V-10 | scenario | distributed-executor-backend agrees with server.multihost |
| V-11 | scenario | no env block sets a name the tool owns |
| V-12 | scenario | no env block sets VLLM_ENGINE_READY_TIMEOUT_S (use startup_timeout) |
| V-13 | scenario | engines on the ray backend get no JAX_PLATFORMS |
| V-14 | scenario | free-form args do not set flags the tool renders |
| V-15 | field | arg keys are flag names without dashes |
| V-16 | scenario | TP x PP x PCP x DP fits the slice's devices |
| V-17 | scenario | mp splits data-parallel-size evenly over hosts |
| V-18 | scenario | tasks.perf sets isl_osl and concurrency, or cells |
| V-19 | scenario | ISL + OSL of every perf cell fits max-model-len of every role |
| V-20 | scenario | perf cell keys are unique and filename-safe |
| V-21 | scenario | inline baseline keys are the task's cell or run keys |
| V-22 | scenario | custom tasks have unique names and use only provided ${NAME}s |
| V-23 | resolve | a baseline's from: names a scenario with matching keys |
| V-24 | scenario | with a sweep, baselines sit with their point |
| V-26 | validate | name is unique across scenarios/ |
| V-27 | scenario | lm_eval model_args do not set what the tool sets |
| V-28 | scenario | the random dataset has random_range_ratio |
| V-29 | resolve | extends: the parent exists, no cycle, depth \<= 5 |
| V-30 | resolve | --set paths start with server. or tasks. and avoid baselines |
| V-31 | field | tasks.agentx.concurrency is one value |
| V-32 | scenario | max-num-seqs x DP covers tasks.agentx.concurrency, twice over |
| V-33 | scenario | zip axes have equal lengths |
| V-34 | scenario | tasks.agentx is the only task |
| V-35 | scenario | perf fields match the perf client |
| V-36 | scenario | lm_eval tasks, concurrency and per-concurrency limits agree |
| V-37 | scenario | tasks.agentx.timeout is longer than its duration |
| V-38 | scenario | a sweep has one form, axes under server.* or tasks.*, unique names |
| V-39 | resolve | sweep axis paths exist in the resolved scenario |
| V-40 | resolve | --set does not touch a sweep axis |

## Derived plumbing

The only values the tool derives. It never adds an engine flag or env var
because of another value; choices belong in `scenarios/defaults.yaml`.

| Item | What | Rule | Env it sets |
|---|---|---|---|
| P-01 | hosts, chips and devices | From the accelerator table: \<accelerator>/\<topology> gives the hosts, the chips per host and the devices per host (a tpu7x chip shows two devices). | none |
| P-02 | backend | single when the role has one host, else server.multihost (mp or ray). | none |
| P-03 | serve argv | vllm serve \<model.path> --served-model-name \<model.name>, the role's args in order, then the wiring flags, and --port 8000 on the role's first host, which serves the API. | none |
| P-04 | mp DP wiring | Every host of an mp role with several hosts: --data-parallel-size-local \<DP / hosts> --data-parallel-address \<head_ip> --data-parallel-rpc-port 29500. | none |
| P-05 | mp headless hosts | Hosts after the first of an mp role add --headless --data-parallel-start-rank \<index x DP local>, and leave off --port and the frontend-only flags (api-server-count), which vLLM rejects there. | none |
| P-06 | backend and host env | An mp role over several hosts: the seven slice names GKE injects set empty, and TPU_MULTIHOST_BACKEND=mp. A ray role over several hosts: RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1, TPU_MULTIHOST_BACKEND=ray and JAX_PLATFORMS empty. A single-host role of a multi-role plan: the slice names set empty, TPU_HOST_BOUNDS=1,1,1 and TPU_TOPOLOGY=\<topology>. Every host of a role with several hosts: TPU_SKIP_MDS_QUERY=1. Every host, and the driver's tasks: PYTHONUNBUFFERED=1. The driver's tasks also get TPU_MULTIHOST_BACKEND of the driver's role. | `MEGASCALE_COORDINATOR_ADDRESS`, `MEGASCALE_NUM_SLICES`, `MEGASCALE_PORT`, `MEGASCALE_SLICE_ID`, `TPU_PROCESS_ADDRESSES`, `TPU_PROCESS_PORT`, `TPU_WORKER_HOSTNAMES`, `TPU_MULTIHOST_BACKEND`, `RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS`, `JAX_PLATFORMS`, `TPU_HOST_BOUNDS`, `TPU_TOPOLOGY`, `TPU_SKIP_MDS_QUERY`, `PYTHONUNBUFFERED` |
| P-07 | ray bring-up | The first host of a ray role runs ray start --head --port=6379 before serve; the others run ray start --address=\<head_ip>:6379 and serve nothing. | none |
| P-08 | advertise address | VLLM_HOST_IP=\<self_ip> on every host of a role with several hosts or with kv. | `VLLM_HOST_IP` |
| P-09 | startup timeout | VLLM_ENGINE_READY_TIMEOUT_S=\<server.startup_timeout> on every host; the readiness wait, GET /health on port 8000, lasts startup_timeout + 600 s from the server's start. | `VLLM_ENGINE_READY_TIMEOUT_S` |
| P-10 | accelerator env | TPU_ACCELERATOR_TYPE=\<accelerator> on every TPU host. | `TPU_ACCELERATOR_TYPE` |
| P-11 | KV wiring | A role with kv: --kv-transfer-config {kv_connector, kv_connector_module_path, kv_role, kv_port: 14579}, TPU_KV_TRANSFER_PORT=9100 and TPU_SIDE_CHANNEL_PORT=9600. | `TPU_KV_TRANSFER_PORT`, `TPU_SIDE_CHANNEL_PORT` |
| P-12 | Raiden identity | A role with kv at position r among the roles: TPU_KV_TRANSFER_NAMESPACE=\<role>, TPU_RAIDEN_ENGINE_ID=\<role>-engine, TPU_RAIDEN_JOB_NAME=\<role>, TPU_RAIDEN_RESHARD_PORT_BASE=27000+1000r and TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27100+1000r; with several hosts also TPU_RAIDEN_ADVERTISE_HOST=\<role_ips:role>, every host's IPv4 in DP-rank order. | `TPU_KV_TRANSFER_NAMESPACE`, `TPU_RAIDEN_ENGINE_ID`, `TPU_RAIDEN_JOB_NAME`, `TPU_RAIDEN_RESHARD_PORT_BASE`, `TPU_RAIDEN_STORE_DISPATCH_PORT_BASE`, `TPU_RAIDEN_ADVERTISE_HOST` |
| P-13 | router | On the driver host: \<router.command> --host 0.0.0.0 --port 8100 --prefiller-hosts \<role_head:producer> --prefiller-ports 8000 --decoder-hosts \<role_head:consumer> --decoder-ports 8000, then router.args; ready on GET /healthcheck. | none |
| P-14 | driver and endpoint | One role: its first host drives. Several roles: tasks.driver, a role's first host or a chipless client host. Tasks connect to the router (127.0.0.1:8100) when there is one, else to the driver role's own API (127.0.0.1:8000), or from a client host to the first role's head. | none |
| P-15 | client tokenizer | --tokenizer \<tasks.perf.tokenizer> when set; else, for an object-storage model.path, --tokenizer \<pulled_model_dir>, which falls back to model.name when the pulled directory holds no config.json; else none. | none |
| P-16 | result paths | Under \<results_dir>: \<cell>.json per perf cell (the label in cells mode), \<cell>.warmup\<i>.json per warmup pass, lm_eval --output_path \<results_dir> (\<results_dir>/\<task>_c\<N> in a concurrency sweep), and agentx_c\<N> for AgentX. | none |
| P-17 | custom-task env | Every custom task gets SCENARIO_SERVED_MODEL, SCENARIO_RESULTS_DIR (\<results_dir>) and SCENARIO_API_PORT=8000; with one producer and one consumer role PREFILL_URL and DECODE_URL (http://\<role_head>:8000); with a router PROXY_URL=http://127.0.0.1:8100 and SCENARIO_ROUTER_PORT=8100; with several roles SCENARIO_HOST_\<ROLE>=\<role_head:role>; on a role's host SCENARIO_SERVER_LOG. Its env values may use them as ${NAME}; $$ is a literal $. | `SCENARIO_SERVED_MODEL`, `SCENARIO_RESULTS_DIR`, `SCENARIO_API_PORT`, `PREFILL_URL`, `DECODE_URL`, `PROXY_URL`, `SCENARIO_ROUTER_PORT`, `SCENARIO_SERVER_LOG` |
| P-18 | chipless driver env | A chipless client driver's tasks and router get VLLM_PLUGINS empty and TORCH_DEVICE_BACKEND_AUTOLOAD=0, so vLLM's CLI from the TPU image does not start a TPU platform on a host without chips. | `VLLM_PLUGINS`, `TORCH_DEVICE_BACKEND_AUTOLOAD` |
| P-19 | remote-code warm | Before serve on mp hosts after the first, when trust-remote-code is set and model.path is object storage: python3 -m scenario.runtime.warm \<model.path>, best effort, at most 600 s, so the hosts do not race to write the model's remote code. | none |
| P-20 | chips | The chips of the server roles, hosts x chips per host (never TensorCores); the AgentX client divides throughput by them (TP, or PREFILL_TP and DECODE_TP per worker for P/D). | none |
| P-21 | InferenceX client env | The InferenceX perf client gets PYTHONPATH=\<baked checkout>. The AgentX client gets its baked runtime (AIPERF_RUNTIME_DIR, PYTHONPATH, PYTHONSAFEPATH), the endpoint (AIPERF_SERVER_URL), SERVED_MODEL_NAME and MODEL=\<model.name>, the labels (MODEL_PREFIX, PRECISION, FRAMEWORK, SPEC_DECODING), CONC and DURATION, RESULT_DIR and AGENTIC_OUTPUT_DIR (\<results_dir>), RESULT_FILENAME=agentx_c\<N>, RUNNER_TYPE=\<accelerator>, the chip counts of P-20 (one role: IS_MULTINODE=false, TP=\<chips>, PP_SIZE, DCP_SIZE and PCP_SIZE 1; P/D: IS_MULTINODE=true, DISAGG=true, PREFILL_NUM_WORKERS, PREFILL_TP, DECODE_NUM_WORKERS, DECODE_TP, power measurement off), EVAL_ONLY=false, KV_OFFLOADING=none, IS_AGENTIC=1, AIPERF_EXPERIMENTAL_FAST=0 and the three Dynamo switches off. | `AIPERF_RUNTIME_DIR`, `PYTHONPATH`, `PYTHONSAFEPATH`, `AIPERF_SERVER_URL`, `SERVED_MODEL_NAME`, `MODEL`, `MODEL_PREFIX`, `PRECISION`, `FRAMEWORK`, `SPEC_DECODING`, `CONC`, `DURATION`, `RESULT_DIR`, `RESULT_FILENAME`, `AGENTIC_OUTPUT_DIR`, `RUNNER_TYPE`, `IS_MULTINODE`, `TP`, `PP_SIZE`, `DCP_SIZE`, `PCP_SIZE`, `DISAGG`, `PREFILL_NUM_WORKERS`, `PREFILL_TP`, `DECODE_NUM_WORKERS`, `DECODE_TP`, `ENABLE_AGENTX_POWER`, `REQUIRE_POWER`, `IS_AGENTIC`, `EVAL_ONLY`, `KV_OFFLOADING`, `AIPERF_EXPERIMENTAL_FAST`, `AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS`, `AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING`, `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID` |

## Reserved names

A scenario may not set these env names, since the tool sets them.

| Env | Reserved in | Set from |
|---|---|---|
| `VLLM_ENGINE_READY_TIMEOUT_S` | every env block | server.startup_timeout |
| `VLLM_HOST_IP` | every env block | the host's own address |
| `TPU_ACCELERATOR_TYPE` | every env block | the slice's accelerator |
| `TPU_MULTIHOST_BACKEND` | every env block | multi-host and backend wiring |
| `MEGASCALE_COORDINATOR_ADDRESS` | every env block | multi-host and backend wiring |
| `MEGASCALE_NUM_SLICES` | every env block | multi-host and backend wiring |
| `MEGASCALE_PORT` | every env block | multi-host and backend wiring |
| `MEGASCALE_SLICE_ID` | every env block | multi-host and backend wiring |
| `TPU_PROCESS_ADDRESSES` | every env block | multi-host and backend wiring |
| `TPU_PROCESS_PORT` | every env block | multi-host and backend wiring |
| `TPU_WORKER_HOSTNAMES` | every env block | multi-host and backend wiring |
| `TPU_HOST_BOUNDS` | every env block | multi-host and backend wiring |
| `TPU_TOPOLOGY` | every env block | multi-host and backend wiring |
| `TPU_SKIP_MDS_QUERY` | every env block | multi-host and backend wiring |
| `RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS` | every env block | multi-host and backend wiring |
| `PYTHONUNBUFFERED` | every env block | the runtime, which streams child output |
| `TPU_KV_TRANSFER_NAMESPACE` | every env block | KV transfer wiring |
| `TPU_RAIDEN_ENGINE_ID` | every env block | KV transfer wiring |
| `TPU_RAIDEN_JOB_NAME` | every env block | KV transfer wiring |
| `TPU_RAIDEN_RESHARD_PORT_BASE` | every env block | KV transfer wiring |
| `TPU_RAIDEN_STORE_DISPATCH_PORT_BASE` | every env block | KV transfer wiring |
| `TPU_RAIDEN_ADVERTISE_HOST` | every env block | KV transfer wiring |
| `TPU_KV_TRANSFER_PORT` | every env block | KV transfer wiring |
| `TPU_SIDE_CHANNEL_PORT` | every env block | KV transfer wiring |
| `PREFILL_URL` | every env block | the custom-task env contract |
| `DECODE_URL` | every env block | the custom-task env contract |
| `PROXY_URL` | every env block | the custom-task env contract |
| `SCENARIO_*` | every env block | reserved prefix |
| `JAX_PLATFORMS` | engines of a ray role | the ray backend |
| `VLLM_PLUGINS` | task env with a client driver | the chipless driver setup |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | task env with a client driver | the chipless driver setup |
| `PYTHONPATH` | task env with a client driver | the chipless driver setup |
| `AIPERF_RUNTIME_DIR` | `tasks.agentx.env` | the InferenceX checkout baked into the image |
| `PYTHONPATH` | `tasks.agentx.env` | the InferenceX checkout baked into the image |
| `PYTHONSAFEPATH` | `tasks.agentx.env` | the InferenceX checkout baked into the image |
| `AIPERF_SERVER_URL` | `tasks.agentx.env` | the tool (endpoint) |
| `SERVED_MODEL_NAME` | `tasks.agentx.env` | server.model.name |
| `MODEL` | `tasks.agentx.env` | server.model.name |
| `MODEL_PREFIX` | `tasks.agentx.env` | tasks.agentx.labels.model_prefix |
| `PRECISION` | `tasks.agentx.env` | tasks.agentx.labels.precision |
| `FRAMEWORK` | `tasks.agentx.env` | tasks.agentx.labels.framework |
| `SPEC_DECODING` | `tasks.agentx.env` | tasks.agentx.labels.spec_decoding |
| `CONC` | `tasks.agentx.env` | tasks.agentx.concurrency |
| `DURATION` | `tasks.agentx.env` | tasks.agentx.duration |
| `RESULT_DIR` | `tasks.agentx.env` | the point and the results layout |
| `RESULT_FILENAME` | `tasks.agentx.env` | the point and the results layout |
| `AGENTIC_OUTPUT_DIR` | `tasks.agentx.env` | the point and the results layout |
| `RUNNER_TYPE` | `tasks.agentx.env` | the slice's accelerator |
| `IS_MULTINODE` | `tasks.agentx.env` | the server roles and their chip counts |
| `TP` | `tasks.agentx.env` | the server roles and their chip counts |
| `PP_SIZE` | `tasks.agentx.env` | the server roles and their chip counts |
| `DCP_SIZE` | `tasks.agentx.env` | the server roles and their chip counts |
| `PCP_SIZE` | `tasks.agentx.env` | the server roles and their chip counts |
| `DISAGG` | `tasks.agentx.env` | the server roles and their chip counts |
| `PREFILL_NUM_WORKERS` | `tasks.agentx.env` | the server roles and their chip counts |
| `PREFILL_TP` | `tasks.agentx.env` | the server roles and their chip counts |
| `DECODE_NUM_WORKERS` | `tasks.agentx.env` | the server roles and their chip counts |
| `DECODE_TP` | `tasks.agentx.env` | the server roles and their chip counts |
| `ENABLE_AGENTX_POWER` | `tasks.agentx.env` | the tool, fixed for a vLLM server |
| `REQUIRE_POWER` | `tasks.agentx.env` | the tool, fixed for a vLLM server |
| `IS_AGENTIC` | `tasks.agentx.env` | the tool, fixed for a vLLM server |
| `EVAL_ONLY` | `tasks.agentx.env` | the tool, fixed for a vLLM server |
| `KV_OFFLOADING` | `tasks.agentx.env` | the tool, fixed for a vLLM server |
| `AIPERF_EXPERIMENTAL_FAST` | `tasks.agentx.env` | the tool, fixed for a vLLM server |
| `AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS` | `tasks.agentx.env` | the tool, fixed for a vLLM server |
| `AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING` | `tasks.agentx.env` | the tool, fixed for a vLLM server |
| `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID` | `tasks.agentx.env` | the tool, fixed for a vLLM server |
| `PYTHONPATH` | `tasks.perf.env` with client inferencex | the InferenceX checkout baked into the image |

Arg keys a scenario may not set:

| Where | Keys |
|---|---|
| `server.args`, `server.roles.<role>.args` | `model`, `served-model-name`, `port`, `host`, `headless`, `data-parallel-size-local`, `data-parallel-address`, `data-parallel-rpc-port`, `data-parallel-start-rank`, `kv-transfer-config` |
| `server.router.args` | `host`, `port`, `prefiller-hosts`, `prefiller-host`, `prefiller-ports`, `prefiller-port`, `decoder-hosts`, `decoder-host`, `decoder-ports`, `decoder-port` |
| `tasks.perf.client_args` | `model`, `tokenizer`, `host`, `port`, `base-url`, `dataset-name`, `dataset-path`, `random-input-len`, `random-output-len`, `sharegpt-output-len`, `random-range-ratio`, `num-prompts`, `num-warmups`, `max-concurrency`, `percentile-metrics`, `temperature`, `extra-body`, `label`, `save-result`, `append-result`, `result-dir`, `result-filename` |
| `tasks.eval.lm_eval.model_args` | `model`, `base_url` |

## Accelerators

Counts are chips, never TensorCores.

| Slice | Hosts | Chips per host | Devices per host |
|---|---|---|---|
| `tpu7x/1x1x1` | 1 | 1 | 2 |
| `tpu7x/2x2x1` | 1 | 4 | 8 |
| `tpu7x/2x2x2` | 2 | 4 | 8 |
| `tpu7x/2x2x4` | 4 | 4 | 8 |
| `tpu7x/2x4x4` | 8 | 4 | 8 |
| `tpu7x/4x4x4` | 16 | 4 | 8 |

## Plan format

This is Plan format 1; a pod refuses a Plan of another format.
A Plan is one sweep point, as canonical JSON (sorted keys, compact, ASCII).
Pods get it in `SCENARIO_PLAN` as base64 of gzip, at most
32768 bytes. Values only a host knows at run time are
Refs:

| Ref | Filled with |
|---|---|
| `self_ip` | this host's advertise address |
| `head_ip` | the role's first host, as IPv4 |
| `host_index` | this host's index in its role |
| `role_head` | the first host of a role by name; 127.0.0.1 on that host |
| `role_ips` | every host of a role as IPv4, joined by commas |
| `results_dir` | the run directory |
| `pulled_model_dir` | an object-storage model's pulled directory, or a fallback |
| `server_log` | the driver host's own server log |

| Port | Use |
|---|---|
| 8000 | every server's API |
| 8100 | the router, on the driver host |
| 8090 | the driver's status, when other hosts wait on it |
| 29500 | mp DP coordination |
| 6379 | the Ray head |
| 14579 | the KV connector |
| 9100 | KV transfer |
| 9600 | the KV side channel |
