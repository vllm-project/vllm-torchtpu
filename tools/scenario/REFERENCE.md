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

What runs: image, model, hardware or roles, flags and env.

| Key | Type | Default | Description |
|---|---|---|---|
| `image` | image URI or null | unset | The image the pods run: a full image URI with a tag or digest. Unset, the target supplies its own, or fails when it has none. |
| `model` | [ModelSpec](#modelspec) | required | Served name and load path. |
| `accelerator` | accelerator label or null | unset | TPU type: the cloud.google.com/gke-tpu-accelerator label, e.g. tpu7x (tpu-v6e-slice for v6e). For the single role; with roles, each role sets its own. |
| `topology` | topology label or null | unset | TPU topology: the cloud.google.com/gke-tpu-topology label, e.g. 2x2x2. Its chips (the product of its dimensions) must equal hosts x chips_per_host. For the single role; with roles, each role sets its own. |
| `hosts` | positive integer or null | unset | Hosts (pods) of the role; more than one needs server.multihost. For the single role; with roles, each role sets its own. |
| `chips_per_host` | positive integer or null | unset | TPU chips per host: the pod's google.com/tpu, e.g. 4. For the single role; with roles, each role sets its own. |
| `roles` | mapping of role name to [Role](#role) or null | unset | Two or more roles, e.g. prefill and decode, in run order, each with its own hardware. Not with the server-level hardware fields. |
| `multihost` | `mp` or `ray` or null | unset | How a role on several hosts runs one engine: vLLM's own multi-node DP (mp) or Ray. Required exactly when a role has more than one host. |
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
| `accelerator` | accelerator label | required | TPU type: the cloud.google.com/gke-tpu-accelerator label, e.g. tpu7x (tpu-v6e-slice for v6e). |
| `topology` | topology label | required | TPU topology: the cloud.google.com/gke-tpu-topology label, e.g. 2x2x2. Its chips (the product of its dimensions) must equal hosts x chips_per_host. |
| `hosts` | positive integer | required | Hosts (pods) of the role; more than one needs server.multihost. |
| `chips_per_host` | positive integer | required | TPU chips per host: the pod's google.com/tpu, e.g. 4. |
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
| `inferencex` | [InferencexIdentity](#inferencexidentity) or null | unset | The deployment's InferenceX identity, shared by the AgentX rows and the InferenceX evals; both need it. |
| `perf` | [PerfTask](#perftask) or null | unset | Benchmark cells. |
| `agentx` | [AgentxTask](#agentxtask) or null | unset | AgentX trace replay; must be the only task. |
| `eval` | [EvalTask](#evaltask) or null | unset | Accuracy evals; with client inferencex, the only task. |
| `custom` | list of [CustomTask](#customtask) | `[]` | Commands, run after perf and before eval. |

### InferencexIdentity

What InferenceX files a deployment's results under.

| Key | Type | Default | Description |
|---|---|---|---|
| `model_prefix` | label | required; defaults.yaml can set it | InferenceX model prefix (MODEL_PREFIX); also selects the AgentX trace corpus and the eval score floors. |
| `precision` | label | required; defaults.yaml can set it | Precision (PRECISION), e.g. fp4. |
| `framework` | label | required; defaults.yaml can set it | Serving framework (FRAMEWORK); selects InferenceX's metrics parser. |
| `spec_decoding` | `mtp` or `draft_model` or `none` | required; defaults.yaml can set it | Speculative decoding method (SPEC_DECODING). |
| `kv_offloading` | `none` or `dram` | required; defaults.yaml can set it | KV cache offloading (KV_OFFLOADING); part of the row's key. |
| `kv_offload_backend` | [KvOffloadBackend](#kvoffloadbackend) or null | unset | KV_OFFLOAD_BACKEND and its metadata; required with kv_offloading dram, not allowed with none. |
| `total_cpu_dram_gb` | positive integer or null | unset | CPU DRAM given to KV offload, in GB (TOTAL_CPU_DRAM_GB); required with kv_offloading dram. |
| `router` | [Router](#router) or null | unset | The router in front of the roles (ROUTER_METADATA). |
| `kv_p2p_transfer` | label or null | unset | KV transfer between P/D roles (KV_P2P_TRANSFER), e.g. nixl. |

### KvOffloadBackend

The KV offload backend InferenceX records with a dram row.

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | label | required | Backend name, e.g. mooncake. |
| `version` | string or null | unset | Backend version. |

### Router

The router InferenceX records with a row.

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | label | required | Router name, e.g. dynamo-router. |
| `version` | string | required | Router version. |

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

InferenceX AgentX trace replay at one concurrency, by infx.bench agentic.

| Key | Type | Default | Description |
|---|---|---|---|
| `concurrency` | positive integer | required | Concurrent sessions; one value per server. |
| `duration` | duration: 2h, 90m, 4h10m, 30s or seconds | required; defaults.yaml can set it | The profile window (DURATION); InferenceX publishes 1h windows, and aiperf marks one under 15m invalid. |
| `fast` | boolean | `false` | InferenceX's bring-up mode (AIPERF_EXPERIMENTAL_FAST): a fixed 20m window and one warmup request per lane; duration must then be 20m. InferenceX never publishes it. |
| `timeout` | duration: 2h, 90m, 4h10m, 30s or seconds | required; defaults.yaml can set it | The task is killed and fails after this; longer than duration, to cover warmup and drain. |
| `env` | env map: name to string | `{}` | Client settings passed through untouched, as a recipe's benchmark.env, e.g. AGENTIC_WARMUP_GRACE_PERIOD or WEKA_LOADER_OVERRIDE. |

### EvalTask

Accuracy evals: lm_eval runs, or InferenceX's eval client.

| Key | Type | Default | Description |
|---|---|---|---|
| `client` | `lm_eval` or `inferencex` | `lm_eval` | lm_eval runs lm_eval once per task, gated against a baseline; inferencex runs InferenceX's eval client (infx.bench eval) at the version baked into the image, alone on a fresh deployment, gated by InferenceX's score floor. |
| `lm_eval` | [LmEval](#lmeval) or null | unset | lm-eval runs; client lm_eval, which needs it. |
| `tolerance` | number in (0, 1) or null | unset | Smallest score drop the eval gate allows, in absolute terms; the gate widens it by the measured stderr. Client lm_eval, which needs it. |
| `baseline` | [EvalBaseline](#evalbaseline) or [BaselineRef](#baselineref) or null | unset | Inline baseline by run key, or {from: \<ref>}; client lm_eval. |
| `framework` | `lm-eval` or `kimi-vendor` or `minimax-vendor` or `bfcl` or null | unset | InferenceX's eval framework (--framework); unset runs lm-eval, which is GSM8K. Client inferencex. |
| `suite` | suite name or null | unset | EVAL_SUITE: required with kimi-vendor and minimax-vendor, which run their smoke suite without it; not allowed with lm-eval. |
| `concurrency` | positive integer or null | unset | Concurrent requests, one value: the highest AgentX concurrency of the deployment, which InferenceX evaluates. Client inferencex, which needs it. |
| `timeout` | duration: 2h, 90m, 4h10m, 30s or seconds or null | unset | The eval is killed and fails after this. Client inferencex, which needs it. |
| `env` | env map: name to string | `{}` | Settings of InferenceX's eval client passed through untouched, e.g. EVAL_LIMIT. Client inferencex. |

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
| `image` | image URI or null | unset | The image the pods run: a full image URI with a tag or digest. Unset, the target supplies its own, or fails when it has none. |
| `startup_timeout` | duration: 2h, 90m, 4h10m, 30s or seconds or null | unset | Engine startup budget: sets VLLM_ENGINE_READY_TIMEOUT_S, and the readiness wait is this plus a margin. |
| `args` | arg map: flag name to value or null | unset | vllm serve flags for every role: true -> --k, false -> --no-k, a scalar -> --k v, a list or mapping -> --k \<compact JSON>. |
| `env` | env map: name to string or null | unset | Env for every server process of every role. |

### DefaultsTasks

Task values, each applied only to scenarios that declare that task.

| Key | Type | Default | Description |
|---|---|---|---|
| `inferencex` | [DefaultsInferencex](#defaultsinferencex) or null | unset | Defaults for tasks.inferencex. |
| `perf` | [DefaultsPerf](#defaultsperf) or null | unset | Defaults for tasks.perf. |
| `agentx` | [DefaultsAgentx](#defaultsagentx) or null | unset | Defaults for tasks.agentx. |
| `eval` | [DefaultsEval](#defaultseval) or null | unset | Defaults for tasks.eval. |

### DefaultsInferencex

tasks.inferencex values, applied only to scenarios that declare it.

| Key | Type | Default | Description |
|---|---|---|---|
| `model_prefix` | label or null | unset | InferenceX model prefix (MODEL_PREFIX); also selects the AgentX trace corpus and the eval score floors. |
| `precision` | label or null | unset | Precision (PRECISION), e.g. fp4. |
| `framework` | label or null | unset | Serving framework (FRAMEWORK); selects InferenceX's metrics parser. |
| `spec_decoding` | `mtp` or `draft_model` or `none` or null | unset | Speculative decoding method (SPEC_DECODING). |
| `kv_offloading` | `none` or `dram` or null | unset | KV cache offloading (KV_OFFLOADING); part of the row's key. |

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
| `duration` | duration: 2h, 90m, 4h10m, 30s or seconds or null | unset | The profile window (DURATION); InferenceX publishes 1h windows, and aiperf marks one under 15m invalid. |
| `fast` | boolean or null | unset | InferenceX's bring-up mode (AIPERF_EXPERIMENTAL_FAST): a fixed 20m window and one warmup request per lane; duration must then be 20m. InferenceX never publishes it. |
| `timeout` | duration: 2h, 90m, 4h10m, 30s or seconds or null | unset | The task is killed and fails after this; longer than duration, to cover warmup and drain. |
| `env` | env map: name to string or null | unset | Client settings passed through untouched, as a recipe's benchmark.env, e.g. AGENTIC_WARMUP_GRACE_PERIOD or WEKA_LOADER_OVERRIDE. |

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
| V-02 | scenario | server sets the hardware fields of one role, or server.roles |
| V-03 | scenario | server.roles has two or more roles |
| V-04 | scenario | server.multihost is set exactly when a role has more than one host |
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
| V-17 | scenario | data-parallel-size is a positive integer, split evenly over mp hosts |
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
| V-31 | field | tasks.agentx.concurrency and tasks.eval.concurrency are one value |
| V-32 | scenario | max-num-seqs x DP covers tasks.agentx.concurrency, twice over |
| V-33 | scenario | zip axes have equal lengths |
| V-34 | scenario | tasks.agentx is the only task |
| V-35 | scenario | perf fields match the perf client |
| V-36 | scenario | lm_eval tasks, concurrency and per-concurrency limits agree |
| V-37 | scenario | tasks.agentx.timeout is longer than its duration |
| V-38 | scenario | a sweep has one form, axes under server.* or tasks.*, unique names |
| V-39 | resolve | sweep axis paths exist in the resolved scenario |
| V-40 | resolve | --set does not touch a sweep axis |
| V-41 | scenario | tasks.agentx.fast runs a 20m window, so duration is 20m |
| V-42 | scenario | a role's topology has hosts x chips_per_host chips |
| V-43 | scenario | tasks.eval with client inferencex is the only task |
| V-44 | scenario | tasks.eval.suite is set for the vendor frameworks, and only for them |
| V-45 | resolve | an InferenceX eval runs at the highest agentx concurrency of its scenario's points; a warning |
| V-46 | scenario | tasks.eval fields match tasks.eval.client |
| V-47 | scenario | tasks.inferencex is set exactly where AgentX or an InferenceX eval runs |
| V-48 | scenario | kv_offloading dram states its backend and DRAM; none states neither |
| V-49 | scenario | tasks.agentx runs InferenceX's 1h window; a warning |
| V-50 | scenario | the AgentX corpus fits max-model-len; a client cap below it is a warning unless fast |

## Derived plumbing

The only values the tool derives. It never adds an engine flag or env var
because of another value; choices belong in `scenarios/defaults.yaml`.

| Item | What | Rule | Env it sets |
|---|---|---|---|
| P-01 | hardware | The role's accelerator, topology, hosts and chips_per_host as written (on server for one role); the tool looks nothing up. | none |
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
| P-12 | Raiden identity | A role with kv at position r among the roles: TPU_KV_TRANSFER_NAMESPACE=\<role>, TPU_RAIDEN_ENGINE_ID=\<role>-engine, TPU_RAIDEN_JOB_NAME=\<role>, TPU_RAIDEN_RESHARD_PORT_BASE=27000+1000r, TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27100+1000r and TPU_RAIDEN_ADVERTISE_HOST=\<role_ips:role>, every host's IPv4 in DP-rank order (one address on a single host; the store-mode connector requires it). | `TPU_KV_TRANSFER_NAMESPACE`, `TPU_RAIDEN_ENGINE_ID`, `TPU_RAIDEN_JOB_NAME`, `TPU_RAIDEN_RESHARD_PORT_BASE`, `TPU_RAIDEN_STORE_DISPATCH_PORT_BASE`, `TPU_RAIDEN_ADVERTISE_HOST` |
| P-13 | router | On the driver host: \<router.command> --host 0.0.0.0 --port 8100 --prefiller-hosts \<role_head:producer> --prefiller-ports 8000 --decoder-hosts \<role_head:consumer> --decoder-ports 8000, then router.args; ready on GET /healthcheck. | none |
| P-14 | driver and endpoint | One role: its first host drives. Several roles: tasks.driver, a role's first host or a chipless client host. Tasks connect to the router (127.0.0.1:8100) when there is one, else to the driver role's own API (127.0.0.1:8000), or from a client host to the first role's head. | none |
| P-15 | client tokenizer | --tokenizer \<tasks.perf.tokenizer> when set; else, for an object-storage model.path, --tokenizer \<pulled_model_dir>, which falls back to model.name when the pulled directory holds no config.json; else none. | none |
| P-16 | result paths | Under \<results_dir>: \<cell>.json per perf cell (the label in cells mode), \<cell>.warmup\<i>.json per warmup pass, lm_eval --output_path \<results_dir> (\<results_dir>/\<task>_c\<N> in a concurrency sweep), agentx_c\<N> for AgentX, and \<results_dir>/inferencex_eval for InferenceX's eval client. | none |
| P-17 | custom-task env | Every custom task gets SCENARIO_SERVED_MODEL, SCENARIO_RESULTS_DIR (\<results_dir>) and SCENARIO_API_PORT=8000; with one producer and one consumer role PREFILL_URL and DECODE_URL (http://\<role_head>:8000); with a router PROXY_URL=http://127.0.0.1:8100 and SCENARIO_ROUTER_PORT=8100; with several roles SCENARIO_HOST_\<ROLE>=\<role_head:role>; on a role's host SCENARIO_SERVER_LOG. Its env values may use them as ${NAME}; $$ is a literal $. | `SCENARIO_SERVED_MODEL`, `SCENARIO_RESULTS_DIR`, `SCENARIO_API_PORT`, `PREFILL_URL`, `DECODE_URL`, `PROXY_URL`, `SCENARIO_ROUTER_PORT`, `SCENARIO_SERVER_LOG` |
| P-18 | chipless driver env | A chipless client driver's tasks and router get VLLM_PLUGINS empty and TORCH_DEVICE_BACKEND_AUTOLOAD=0, so vLLM's CLI from the TPU image does not start a TPU platform on a host without chips. | `VLLM_PLUGINS`, `TORCH_DEVICE_BACKEND_AUTOLOAD` |
| P-19 | remote-code warm | Before serve on mp hosts after the first, when trust-remote-code is set and model.path is object storage: python3 -m scenario.runtime.warm \<model.path>, best effort, at most 600 s, so the hosts do not race to write the model's remote code. | none |
| P-20 | chips | The chips of the server roles, the sum of hosts x chips_per_host (never TensorCores); InferenceX divides AgentX throughput by them (TP, or PREFILL_TP and DECODE_TP per worker for P/D). | none |
| P-21 | InferenceX client env | The InferenceX perf client gets PYTHONPATH=\<baked checkout>. The AgentX client gets its baked runtime (AIPERF_RUNTIME_DIR, PYTHONPATH, PYTHONSAFEPATH), the endpoint (AIPERF_SERVER_URL), SERVED_MODEL_NAME=\<model.name>, CONC and DURATION, RESULT_DIR and AGENTIC_OUTPUT_DIR (\<results_dir>), RESULT_FILENAME=agentx_c\<N>, AIPERF_EXPERIMENTAL_FAST from tasks.agentx.fast (1 or 0), EVAL_ONLY=false, IS_AGENTIC=1 and the three Dynamo switches off. For several roles it also gets every role head's /metrics in AIPERF_SERVER_METRICS_URLS, since a router re-exports no engine metrics, and power measurement off. InferenceX's eval client gets PYTHONPATH and PYTHONSAFEPATH, MODEL_NAME=\<model.name>, EVAL_SUITE from tasks.eval.suite, EVAL_ONLY=true and OPENAI_API_KEY=EMPTY. Both get the identity of P-22. | `AIPERF_RUNTIME_DIR`, `PYTHONPATH`, `PYTHONSAFEPATH`, `AIPERF_SERVER_URL`, `AIPERF_SERVER_METRICS_URLS`, `SERVED_MODEL_NAME`, `CONC`, `DURATION`, `RESULT_DIR`, `RESULT_FILENAME`, `AGENTIC_OUTPUT_DIR`, `ENABLE_AGENTX_POWER`, `REQUIRE_POWER`, `IS_AGENTIC`, `EVAL_ONLY`, `AIPERF_EXPERIMENTAL_FAST`, `AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS`, `AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING`, `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID`, `MODEL_NAME`, `EVAL_SUITE`, `OPENAI_API_KEY` |
| P-22 | InferenceX identity | What InferenceX files an AgentX row and the evals of the same deployment under, set alike on both clients so they join: MODEL=\<model.name>, tasks.inferencex (MODEL_PREFIX, PRECISION, FRAMEWORK, SPEC_DECODING, KV_OFFLOADING, and when set KV_OFFLOAD_BACKEND with its JSON metadata, TOTAL_CPU_DRAM_GB, ROUTER_METADATA as JSON and KV_P2P_TRANSFER), RUNNER_TYPE=\<accelerator> and IMAGE=\<the image the pods run>. Sizes are chips (P-20). DP attention is data-parallel-size above 1, as InferenceX reads vLLM. One role: IS_MULTINODE=false, TP=\<chips>, EP_SIZE (\<chips> with enable-expert-parallel, else 1), PP_SIZE, DCP_SIZE and PCP_SIZE 1, DP_ATTENTION. P/D: IS_MULTINODE=true, DISAGG=true, and for the producer and the consumer roles PREFILL_ and DECODE_ NUM_WORKERS, TP (one worker's chips), EP (likewise) and DP_ATTN. | `MODEL`, `MODEL_PREFIX`, `PRECISION`, `FRAMEWORK`, `SPEC_DECODING`, `KV_OFFLOADING`, `KV_OFFLOAD_BACKEND`, `KV_OFFLOAD_BACKEND_METADATA`, `TOTAL_CPU_DRAM_GB`, `ROUTER_METADATA`, `KV_P2P_TRANSFER`, `RUNNER_TYPE`, `IMAGE`, `IS_MULTINODE`, `TP`, `EP_SIZE`, `PP_SIZE`, `DCP_SIZE`, `PCP_SIZE`, `DISAGG`, `PREFILL_NUM_WORKERS`, `PREFILL_TP`, `PREFILL_EP`, `DECODE_NUM_WORKERS`, `DECODE_TP`, `DECODE_EP`, `DP_ATTENTION`, `PREFILL_DP_ATTN`, `DECODE_DP_ATTN` |

## Reserved names

A scenario may not set these env names where the tool sets them. A
name reserved in one block only, such as the AgentX client's MODEL in
`tasks.agentx.env`, stays free in every other block.

| Env | Reserved in | Set from |
|---|---|---|
| `VLLM_ENGINE_READY_TIMEOUT_S` | every env block | server.startup_timeout |
| `VLLM_HOST_IP` | every env block | the host's own address |
| `TPU_ACCELERATOR_TYPE` | every env block | the role's accelerator |
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
| `AIPERF_SERVER_METRICS_URLS` | `tasks.agentx.env` | the server roles' heads |
| `SERVED_MODEL_NAME` | `tasks.agentx.env` | server.model.name |
| `CONC` | `tasks.agentx.env` | tasks.agentx.concurrency |
| `DURATION` | `tasks.agentx.env` | tasks.agentx.duration |
| `RESULT_DIR` | `tasks.agentx.env` | the point and the results layout |
| `RESULT_FILENAME` | `tasks.agentx.env` | the point and the results layout |
| `AGENTIC_OUTPUT_DIR` | `tasks.agentx.env` | the point and the results layout |
| `MODEL` | `tasks.agentx.env` | server.model.name |
| `MODEL_PREFIX` | `tasks.agentx.env` | tasks.inferencex.model_prefix |
| `PRECISION` | `tasks.agentx.env` | tasks.inferencex.precision |
| `FRAMEWORK` | `tasks.agentx.env` | tasks.inferencex.framework |
| `SPEC_DECODING` | `tasks.agentx.env` | tasks.inferencex.spec_decoding |
| `KV_OFFLOADING` | `tasks.agentx.env` | tasks.inferencex.kv_offloading |
| `KV_OFFLOAD_BACKEND` | `tasks.agentx.env` | tasks.inferencex.kv_offload_backend |
| `KV_OFFLOAD_BACKEND_METADATA` | `tasks.agentx.env` | tasks.inferencex.kv_offload_backend |
| `TOTAL_CPU_DRAM_GB` | `tasks.agentx.env` | tasks.inferencex.total_cpu_dram_gb |
| `ROUTER_METADATA` | `tasks.agentx.env` | tasks.inferencex.router |
| `KV_P2P_TRANSFER` | `tasks.agentx.env` | tasks.inferencex.kv_p2p_transfer |
| `RUNNER_TYPE` | `tasks.agentx.env` | the server's accelerator |
| `IMAGE` | `tasks.agentx.env` | the image the pods run |
| `IS_MULTINODE` | `tasks.agentx.env` | the server roles and their chip counts |
| `TP` | `tasks.agentx.env` | the server roles and their chip counts |
| `EP_SIZE` | `tasks.agentx.env` | the server roles' chip counts and enable-expert-parallel |
| `PP_SIZE` | `tasks.agentx.env` | the server roles and their chip counts |
| `DCP_SIZE` | `tasks.agentx.env` | the server roles and their chip counts |
| `PCP_SIZE` | `tasks.agentx.env` | the server roles and their chip counts |
| `DISAGG` | `tasks.agentx.env` | the server roles and their chip counts |
| `PREFILL_NUM_WORKERS` | `tasks.agentx.env` | the server roles and their chip counts |
| `PREFILL_TP` | `tasks.agentx.env` | the server roles and their chip counts |
| `PREFILL_EP` | `tasks.agentx.env` | the server roles' chip counts and enable-expert-parallel |
| `DECODE_NUM_WORKERS` | `tasks.agentx.env` | the server roles and their chip counts |
| `DECODE_TP` | `tasks.agentx.env` | the server roles and their chip counts |
| `DECODE_EP` | `tasks.agentx.env` | the server roles' chip counts and enable-expert-parallel |
| `DP_ATTENTION` | `tasks.agentx.env` | the server roles' data-parallel-size |
| `PREFILL_DP_ATTN` | `tasks.agentx.env` | the server roles' data-parallel-size |
| `DECODE_DP_ATTN` | `tasks.agentx.env` | the server roles' data-parallel-size |
| `ENABLE_AGENTX_POWER` | `tasks.agentx.env` | the tool, fixed for a vLLM server |
| `REQUIRE_POWER` | `tasks.agentx.env` | the tool, fixed for a vLLM server |
| `IS_AGENTIC` | `tasks.agentx.env` | the tool, fixed for a vLLM server |
| `EVAL_ONLY` | `tasks.agentx.env` | the tool, fixed for a vLLM server |
| `AIPERF_EXPERIMENTAL_FAST` | `tasks.agentx.env` | tasks.agentx.fast |
| `AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS` | `tasks.agentx.env` | the tool, fixed for a vLLM server |
| `AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING` | `tasks.agentx.env` | the tool, fixed for a vLLM server |
| `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID` | `tasks.agentx.env` | the tool, fixed for a vLLM server |
| `PYTHONPATH` | `tasks.eval.env` with client inferencex | the InferenceX checkout baked into the image |
| `PYTHONSAFEPATH` | `tasks.eval.env` with client inferencex | the InferenceX checkout baked into the image |
| `MODEL_NAME` | `tasks.eval.env` with client inferencex | server.model.name |
| `EVAL_FRAMEWORK` | `tasks.eval.env` with client inferencex | tasks.eval.framework, which it would override |
| `EVAL_SUITE` | `tasks.eval.env` with client inferencex | tasks.eval.suite |
| `EVAL_ONLY` | `tasks.eval.env` with client inferencex | the tool, fixed: the eval runs alone on its deployment |
| `OPENAI_API_KEY` | `tasks.eval.env` with client inferencex | the tool, fixed: the servers take no API key |
| `MODEL` | `tasks.eval.env` with client inferencex | server.model.name |
| `MODEL_PREFIX` | `tasks.eval.env` with client inferencex | tasks.inferencex.model_prefix |
| `PRECISION` | `tasks.eval.env` with client inferencex | tasks.inferencex.precision |
| `FRAMEWORK` | `tasks.eval.env` with client inferencex | tasks.inferencex.framework |
| `SPEC_DECODING` | `tasks.eval.env` with client inferencex | tasks.inferencex.spec_decoding |
| `KV_OFFLOADING` | `tasks.eval.env` with client inferencex | tasks.inferencex.kv_offloading |
| `KV_OFFLOAD_BACKEND` | `tasks.eval.env` with client inferencex | tasks.inferencex.kv_offload_backend |
| `KV_OFFLOAD_BACKEND_METADATA` | `tasks.eval.env` with client inferencex | tasks.inferencex.kv_offload_backend |
| `TOTAL_CPU_DRAM_GB` | `tasks.eval.env` with client inferencex | tasks.inferencex.total_cpu_dram_gb |
| `ROUTER_METADATA` | `tasks.eval.env` with client inferencex | tasks.inferencex.router |
| `KV_P2P_TRANSFER` | `tasks.eval.env` with client inferencex | tasks.inferencex.kv_p2p_transfer |
| `RUNNER_TYPE` | `tasks.eval.env` with client inferencex | the server's accelerator |
| `IMAGE` | `tasks.eval.env` with client inferencex | the image the pods run |
| `IS_MULTINODE` | `tasks.eval.env` with client inferencex | the server roles and their chip counts |
| `TP` | `tasks.eval.env` with client inferencex | the server roles and their chip counts |
| `EP_SIZE` | `tasks.eval.env` with client inferencex | the server roles' chip counts and enable-expert-parallel |
| `PP_SIZE` | `tasks.eval.env` with client inferencex | the server roles and their chip counts |
| `DCP_SIZE` | `tasks.eval.env` with client inferencex | the server roles and their chip counts |
| `PCP_SIZE` | `tasks.eval.env` with client inferencex | the server roles and their chip counts |
| `DISAGG` | `tasks.eval.env` with client inferencex | the server roles and their chip counts |
| `PREFILL_NUM_WORKERS` | `tasks.eval.env` with client inferencex | the server roles and their chip counts |
| `PREFILL_TP` | `tasks.eval.env` with client inferencex | the server roles and their chip counts |
| `PREFILL_EP` | `tasks.eval.env` with client inferencex | the server roles' chip counts and enable-expert-parallel |
| `DECODE_NUM_WORKERS` | `tasks.eval.env` with client inferencex | the server roles and their chip counts |
| `DECODE_TP` | `tasks.eval.env` with client inferencex | the server roles and their chip counts |
| `DECODE_EP` | `tasks.eval.env` with client inferencex | the server roles' chip counts and enable-expert-parallel |
| `DP_ATTENTION` | `tasks.eval.env` with client inferencex | the server roles' data-parallel-size |
| `PREFILL_DP_ATTN` | `tasks.eval.env` with client inferencex | the server roles' data-parallel-size |
| `DECODE_DP_ATTN` | `tasks.eval.env` with client inferencex | the server roles' data-parallel-size |
| `PYTHONPATH` | `tasks.perf.env` with client inferencex | the InferenceX checkout baked into the image |

Arg keys a scenario may not set:

| Where | Keys |
|---|---|
| `server.args`, `server.roles.<role>.args` | `model`, `served-model-name`, `port`, `host`, `headless`, `data-parallel-size-local`, `data-parallel-address`, `data-parallel-rpc-port`, `data-parallel-start-rank`, `kv-transfer-config` |
| `server.router.args` | `host`, `port`, `prefiller-hosts`, `prefiller-host`, `prefiller-ports`, `prefiller-port`, `decoder-hosts`, `decoder-host`, `decoder-ports`, `decoder-port` |
| `tasks.perf.client_args` | `model`, `tokenizer`, `host`, `port`, `base-url`, `dataset-name`, `dataset-path`, `random-input-len`, `random-output-len`, `sharegpt-output-len`, `random-range-ratio`, `num-prompts`, `num-warmups`, `max-concurrency`, `percentile-metrics`, `temperature`, `extra-body`, `label`, `save-result`, `append-result`, `result-dir`, `result-filename` |
| `tasks.eval.lm_eval.model_args` | `model`, `base_url` |

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
| `image` | the image the pods run, as the target set it, else the Plan's |

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

## Runtime

`scenario run` exits with a code that names what failed. When several
apply, the first of these wins: a signal, 4 or 5, 3, 7, 1, 6.

| Exit | Meaning |
|---|---|
| 0 | every task passed (soft gate failures allowed) |
| 1 | a task failed: a command exited non-zero, a hard gate failed, a result is missing, or a cell had too many failed requests |
| 2 | usage, Plan or placement error |
| 3 | bring-up failed before the server started: a name did not resolve, a port was taken |
| 4 | a server was not ready: it died while starting, or the readiness wait ran out |
| 5 | a server or router died after it was ready |
| 6 | a fatal after-task failed while every task passed |
| 7 | the driver aborted, or was lost before it finished |
| 130 | interrupted (SIGINT) |
| 143 | terminated (SIGTERM) |

| Constant | Value |
|---|---|
| readiness poll of a local server | 1 s |
| stop: SIGTERM to the group, then SIGKILL after | 20 s |
| a name's resolution, retried every 5 s, gives up after | 1800 s |
| a server at --url is alive while GET /health answers within | 10 s |
| a timed-out command's status | 124 |
