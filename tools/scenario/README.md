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
  image: ci                       # this build's CI image, or a full image URI
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

### Server

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
| `eval` | `lm_eval` per task, optionally once per concurrency |
| `agentx` | InferenceX AgentX replay at one concurrency against a fresh server; a ladder is a sweep |
| `custom` | any command, given the server URLs and a results directory |

Baselines sit next to the task they gate, and `scenario calibrate` (M2)
rewrites them from a run.

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
| the CI fleet target | the CI fleet through its launcher; it lives under `.buildkite/scenario/` (M2) |

A target can add only a few named things: a container name, annotations,
infrastructure env such as cache paths, and after-task uploads. It cannot
change what runs. Tests enforce this:

- a scenario's Plan is identical with and without a target;
- imports between modules follow fixed rules;
- tool-owned and target-owned env names are reserved;
- fleet-specific strings may not appear under `tools/scenario/` or
  `scenarios/`.

## Layout

| Path | Contents |
|---|---|
| `tools/scenario/` | the CLI, its own Python project (`pyproject.toml`, `uv.lock`) |
| `scenarios/` | the files people edit, and `scenarios/defaults.yaml` |
| `.buildkite/scenario/` | CI-only pieces: the golden check, and later the fleet target |

## Usage

From the repository root:

```
uv run --project tools/scenario scenario validate
uv run --project tools/scenario scenario show scenarios/glm-5.2/fp8-dp16-ep.yaml
uv run --project tools/scenario scenario show --plan scenarios/glm-5.2/fp8-dp16-ep.yaml
uv run --project tools/scenario scenario show --why server.args.max-model-len scenarios/glm-5.2/fp8-dp16-ep.yaml
uv run --project tools/scenario scenario run --dry-run --role server --host-index 1 scenarios/glm-5.2/fp8-dp16-ep-mp.yaml
```

What each command does:

- **`validate`:** with no files, checks every scenario under `scenarios/`.
- **`show`:** prints resolved values. With `--plan`, it prints what every host
  runs.
- **`run --dry-run`:** prints the commands and env of the selected hosts.

## Development

```
uv run --project tools/scenario --frozen pytest -c tools/scenario/pyproject.toml tools/scenario/tests .buildkite/scenario
uv run --project tools/scenario --frozen scenario dev regen        # generated schema and REFERENCE.md
uv lock --project tools/scenario                                   # after changing dependencies
```

- **Python:** 3.12 and 3.13.
- **Lock:** `uv.lock` names only pypi.org, whatever a machine's own uv config
  says.
- **Golden check:** under `.buildkite/scenario/golden`. It runs today's kube
  scripts with stub programs and compares what they would run with each
  scenario's Plan. It needs bash 4.4 or newer.
