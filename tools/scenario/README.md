# scenario

One YAML file per vLLM serving workload on TPU, and a CLI that validates,
explains and runs it.

A **scenario** says which server(s) to bring up and which tasks to run against
them: perf, eval, AgentX trace replay or a custom script. It can also sweep
over variants. The `scenario` command turns a scenario into exactly what each
host runs, and runs it on a TPU VM or on the Kubernetes CI fleet.

## Why

Today the settings for one CI workload are spread over many places:

- `scripts/vllm/benchmarking/configs/*.sh`;
- the step's `env:` block;
- the `FORWARD` list in `.buildkite/kubernetes/common.sh`;
- a hand-written JobSet;
- an entry script with its own defaults.

Each launch path (single host, Ray, MP, P/D) has its own bash. The same knob is
often set in two places with different values, and some values never reach
the pod. A config that `export`s `VLLM_ENGINE_READY_TIMEOUT_S` silently
overrides the step's value. Child configs are made by `source` plus `sed` on
the parent's serve flags.

The goal is one file per workload, one tool that reads it, and a printout
(`scenario show`) of every value it resolved and where that value came from.

## Status

Under construction. Nothing in CI uses this yet.

| Milestone | Scope |
|---|---|
| M1 | Schema, resolution, the Plan, `validate` / `show` / `run --dry-run`, scenarios for today's nightly workloads, and a golden check that they reproduce today's scripts |
| M2a | Runtime on a host (bring-up, serve, perf/eval, gates), the Kubernetes target, the CI fleet target; run beside today's steps, then switch |
| M2b | Sweeps (one run per point) and the InferenceX/AgentX clients |
| M3 | P/D: several server roles, KV wiring, router, a chipless client |
| M4 | Delete the kube-side bash and hand-written JobSets |

Bare-metal lanes are out of scope; they keep their scripts until they are
retired. pytest lanes are out of scope too.

## A scenario

```yaml
schema: 1
name: glm-5.2-fp8-dp16-ep
server:
  model:
    name: zai-org/GLM-5.2-FP8     # what clients call it (--served-model-name)
    path: gs://tpu-commons-ci/glm/GLM-5.2-FP8   # what the server loads (--model)
  accelerator: tpu7x              # cloud.google.com/gke-tpu-accelerator
  topology: 2x2x2                 # cloud.google.com/gke-tpu-topology: 8 chips
  hosts: 2                        # pods
  chips_per_host: 4               # google.com/tpu of each pod
  multihost: mp                   # mp or ray; required when hosts > 1
  startup_timeout: 2h
  args:                           # vllm serve flags, as written
    load-format: runai_streamer
    tensor-parallel-size: 1
    data-parallel-size: 16        # the tool gives each host 8
    max-model-len: 9216
tasks:
  perf:
    isl_osl: [[8192, 1024]]
    concurrency: [1024]
    num_prompts: 2048
    tolerance: 0.05
    baseline:                     # gates this task
      isl8192_osl1024_c1024:
        median_tpot_ms: 276.99
        output_token_throughput: 2537.45
  eval:
    lm_eval:
      tasks: [mmlu_pro]
    tolerance: 0.025
```

[REFERENCE.md](REFERENCE.md) lists every field with its type, default and
description, and the validation rules and reserved names. It and the JSON
schemas in `schema/` are generated from the models. For completion and errors
in an editor that uses the YAML language server, start a scenario file with
this line, the path relative to the file (`defaults.yaml` takes
`defaults.schema.json`):

```yaml
# yaml-language-server: $schema=<path>/tools/scenario/schema/scenario.schema.json
```

### Server

`image` is a full image URI. Left unset, the target supplies the image: the CI
fleet target uses the build's own image, and the `kubectl` target, which has
none, asks for `--set server.image=<image URI>`.

`server` describes one role, or holds `roles` for P/D, where each role has its
own hardware, args, env and KV role. Hardware is four fields with GKE's names,
on `server` for one role and on each role otherwise:

| Field | What it is |
|---|---|
| `accelerator` | the `cloud.google.com/gke-tpu-accelerator` label: `tpu7x`, or `tpu-v6e-slice` for v6e |
| `topology` | the `cloud.google.com/gke-tpu-topology` label, e.g. `2x2x2` |
| `hosts` | how many hosts (pods) the role has |
| `chips_per_host` | the pod's `google.com/tpu` |

There is no accelerator table. The one check is arithmetic: the chips in the
topology (the product of its dimensions) must equal `hosts` x
`chips_per_host`. `multihost` is required exactly when `hosts` is more than 1.

`args` are `vllm serve` flags exactly as written. Nothing is implied: a
`gs://` path does not turn on runai, and `load-format` is stated like any other
flag. The per-host data-parallel flags are the exception, as in srt-slurm: on
an mp role with several hosts, the tool derives `data-parallel-size-local`
(DP / hosts), `data-parallel-start-rank` and `headless` from
`data-parallel-size` and `hosts`, and setting any of them in `args` is an
error.

### Tasks

| Task | What it runs |
|---|---|
| `perf` | `vllm bench serve` (or InferenceX's `benchmark_serving`) per ISL/OSL x concurrency cell |
| `eval` | `lm_eval` per task, optionally once per concurrency; or InferenceX's eval client, eval-only and alone on a fresh server, gated by InferenceX's score floor |
| `agentx` | InferenceX AgentX replay (`infx.bench agentic`) at one concurrency against a fresh server; a ladder is a sweep |
| `custom` | any command, given the server URLs and a results directory |

Perf and lm_eval baselines sit next to the task they gate, and `scenario
calibrate` rewrites them from runs. AgentX and InferenceX evals have no
baseline: InferenceX checks them itself.

AgentX follows InferenceX's own runs. `tasks.inferencex` is the deployment's
identity (model prefix, precision, framework, speculative decoding, KV
offload, router), which its AgentX rows and InferenceX evals share. The AgentX
task states only what InferenceX's matrix gives a point: concurrency,
duration (InferenceX publishes 1h windows) and fast mode (bring-up only). Every
other client setting is pass-through `env`, as an InferenceX recipe's
`benchmark.env` is.

### Sweeps

A sweep is a list of named `points`, or a `zip` or `product` of axes such as
`tasks.agentx.concurrency: [1, 2, 4, 8]`. Each point is its own run on a fresh
server.

### Layers

Values resolve in this order, with later layers winning:

1. what the tool itself derives;
2. `scenarios/defaults.yaml`;
3. the `extends` chain;
4. the scenario;
5. the sweep point;
6. `--set` on the command line.

`scenario show` prints every resolved value with its layer and file:line. A
run with `--set` on server or task values is an experiment: its gates are
skipped.

The tool never adds an engine flag or env var because of some other value.
It derives only plumbing that has one correct answer: per-host data-parallel
flags, addresses, ports and KV wiring. Choices go in `defaults.yaml`.

## Plans and targets

`plan()` turns one resolved scenario point into a **Plan**: a frozen,
target-free description of what runs. For each host it holds the `vllm serve`
argv, env and readiness check. For the driver host it holds the task commands.

A **target** takes a Plan and runs it somewhere:

| Target | Where it runs |
|---|---|
| `local` | this machine, e.g. a TPU VM |
| `kubectl` | a generic GKE JobSet (M2) |
| the CI fleet target | the CI fleet through its launcher; it lives under `.buildkite/kubernetes/scenario/` (M2) |

A target can add only a few named things: a container name, annotations,
infrastructure env such as cache paths, and after-task uploads. It cannot
change what runs. Tests enforce this:

- a scenario's Plan is identical with and without a target;
- imports between modules follow fixed rules;
- tool-owned and target-owned env names are reserved;
- fleet-specific strings may not appear under `tools/scenario/` or
  `.buildkite/kubernetes/scenarios/`.

## Layout

| Path | Contents |
|---|---|
| `tools/scenario/` | the CLI, its own Python project (`pyproject.toml`, `uv.lock`) |
| `.buildkite/kubernetes/scenarios/` | the CI scenarios, which people edit, and `defaults.yaml` |
| `.buildkite/kubernetes/scenario/tests/` | checks over the CI scenarios |
| `.buildkite/kubernetes/scenario/golden/` | the golden check, which is temporary |

The tool's root is the directory that holds `scenarios/defaults.yaml`; for
the CI scenarios it is `.buildkite/kubernetes`. A reference id such as
`kimi-k3/tp32-ep` is the path under `scenarios/` without `.yaml`. The CI
fleet target will live in `.buildkite/kubernetes/scenario/` as well.

## Running

`scenario run FILE` runs a scenario on this machine. On one host it:

1. writes the run directory: `config.json` in the shape of today's
   `run_benchmarks.sh`, `scenario_run.json` (the resolved scenario, the Plan,
   the target) and `summary.json` (state, exit code, per-task status, kept
   current as the run goes);
2. starts `vllm serve` with the Plan's argv and env, its output to the log
   and `server.log`, and waits for `/health` up to `startup_timeout` + 600 s;
3. runs the tasks in order (perf, AgentX, custom, eval), with today's result
   file names, and gates each against its baseline with
   `scripts/vllm/benchmarking/check_regression.py`;
4. stops the server's whole process group, and exits with a code that names
   what failed.

A failed task does not stop the next one; a dead server does. Core dumps
are off for the run and everything it starts: a TPU server's core is about
the size of its host memory. The run directory is `<results root>/<name>`,
plus `/<point>` for a sweep point; the root is `--results-dir`, else the
target's, else `./scenario-results/<name>-<UTC stamp>`.

| Exit | Meaning |
|---|---|
| 0 | every task passed (a soft gate may have failed) |
| 1 | a task failed: a command, a hard gate, a missing result |
| 2 | usage, scenario, Plan or placement error |
| 3 | bring-up failed (a name, a taken port, the Ray cluster, a driver that never answered) |
| 4 | the server was not ready: it died starting, or the wait ran out |
| 5 | a server or the router died after it was ready |
| 6 | a fatal after-task failed while every task passed |
| 7 | the driver aborted or was lost (hosts that wait on a driver) |
| 130, 143 | SIGINT, SIGTERM |

In a pod, `scenario run --role ROLE` takes the Plan from `SCENARIO_PLAN` and
its placement from `SCENARIO_HOST_INDEX`, `SCENARIO_HEAD_ADDRESS`,
`SCENARIO_ADVERTISE_IP` and `SCENARIO_ROLE_HOSTS`; the flags of the same
names win. `SCENARIO_TARGET_DATA` carries the target's after-tasks (uploads)
and its lm_eval request cache root; a local run has neither.

A scenario over several hosts runs `scenario run` on every host, with the
same Plan:

```
host-a$ scenario run FILE --host-index 0 --head-address host-a --advertise-ip 10.0.0.1
host-b$ scenario run FILE --host-index 1 --head-address host-a --advertise-ip 10.0.0.2
```

The first host of the role drives. It serves its status on port 8090, waits
for its server (on an mp head, that is every host joining; a Ray head
starts its server only once every host is a Ray node with TPUs), runs the
tasks and says how the run ended. Every other host starts its own part at
once: the remote-code warm when the Plan has it, then its `--headless`
server, or `ray start` once the Ray head answers. `ray start` gets the
scenario's env, so every Ray worker inherits it, and `ray stop --force`
ends every Ray host.
It asks for the driver's status every 2 s and leaves with 0 when the
driver is done, 7 when it aborted or has not answered for 20 s, and 4 or 5
when its own server dies (before or after the driver was serving). It
writes its `summary.json` and `server.log` under `<run dir>/<role>-<index>`.

A scenario of several roles runs `scenario run --role ROLE` on every host,
with every role's hosts in `--role-hosts` (or `SCENARIO_ROLE_HOSTS`), the
chipless `client` included when the scenario has one:

```
scenario run FILE --role decode --host-index 0 --advertise-ip 10.0.0.2 \
  --role-hosts 'prefill=p0;decode=d0;client=c0'
```

The driver is the part `tasks.driver` names: a role's first host, or the
client, a host without chips that runs no server. After its own server it
waits for each other role's head (`GET /health` on its API port, for that
role's readiness wait plus 1800 s from the driver's start, since its host
may start later), then starts the router and gives it 120 s to answer its
health check. A router that dies stops the tasks with exit 5. A chipless
driver's router and tasks get `VLLM_PLUGINS` empty,
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` and the tool's libtpu blocker first on
`PYTHONPATH`, so vLLM's CLI from the TPU image finds no TPU on that host.
The other roles' hosts follow the driver as a multi-host role's do.

## Usage

From the repository root (tasks run from the current directory):

```
uv run --project tools/scenario scenario validate --root .buildkite/kubernetes
uv run --project tools/scenario scenario show .buildkite/kubernetes/scenarios/glm-5.2/fp8-dp16-ep.yaml
uv run --project tools/scenario scenario show --plan .buildkite/kubernetes/scenarios/glm-5.2/fp8-dp16-ep.yaml
uv run --project tools/scenario scenario show --why server.args.max-model-len .buildkite/kubernetes/scenarios/glm-5.2/fp8-dp16-ep.yaml
uv run --project tools/scenario scenario run --dry-run --role server --host-index 1 .buildkite/kubernetes/scenarios/glm-5.2/fp8-dp16-ep-mp.yaml
uv run --project tools/scenario scenario run .buildkite/kubernetes/scenarios/qwen3-0.6b/tp1.yaml --results-dir /tmp/runs
uv run --project tools/scenario scenario up .buildkite/kubernetes/scenarios/qwen3-0.6b/tp1.yaml
uv run --project tools/scenario scenario perf .buildkite/kubernetes/scenarios/qwen3-0.6b/tp1.yaml --url http://127.0.0.1:8000
uv run --project tools/scenario scenario calibrate .buildkite/kubernetes/scenarios/glm-5.2/fp8-dp16-ep.yaml run1/glm-5.2-fp8-dp16-ep run2/glm-5.2-fp8-dp16-ep --write
```

What each command does:

- **`validate`:** with no files, checks every scenario under the root's
  `scenarios/`. `--root` names the root; without it, the root is found from
  the current directory.
- **`show`:** prints resolved values. With `--plan`, it prints what every host
  runs.
- **`run`:** runs the scenario on this host (see Running). `--dry-run` prints
  the commands and env of the selected hosts instead; `--only perf,eval`
  runs some tasks; `--set` makes the run an experiment, whose gates are
  skipped.
- **`up`:** serves the scenario until SIGINT or SIGTERM, running no task.
- **`perf` / `eval`:** run one task against a server that is already up at
  `--url`.
- **`calibrate`:** computes new baselines from run directories with
  `check_regression.py`'s calibration (medians over the runs) and prints the
  change. `--write` rewrites only those baseline entries in the file; every
  other line, comments included, stays. A baseline that is `from:` another
  scenario is refused.

## Development

```
uv run --project tools/scenario --frozen pytest -c tools/scenario/pyproject.toml tools/scenario/tests .buildkite/kubernetes/scenario/tests
uv run --project tools/scenario --frozen pytest -c tools/scenario/pyproject.toml .buildkite/kubernetes/scenario/golden
uv run --project tools/scenario --frozen scenario dev regen        # generated schema and REFERENCE.md
uv lock --project tools/scenario                                   # after changing dependencies
```

- **Python:** 3.12 and 3.13.
- **Lock:** `uv.lock` names only pypi.org, whatever a machine's own uv config
  says.
- **Golden check:** under `.buildkite/kubernetes/scenario/golden`, with its
  own workflow, `scenario-golden.yml`. It runs today's kube scripts with stub
  programs and compares what they would run with each scenario's Plan. It
  needs bash 4.4 or newer. It is temporary: M4 deletes it together with
  today's kube scripts.
