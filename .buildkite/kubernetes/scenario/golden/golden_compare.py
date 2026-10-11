# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Both sides as normalized calls by slot, the differences, and the allowlist.

A call is a program, its positional words and its flags, plus the env it ran
with. Slots: each host's serve and its commands before serve, the router,
the perf client runs, the AgentX client, the lm_eval runs, the custom tasks
and the gates, in run order. A difference is (where, kind, name, today,
plan); each one must match an allowlist entry, and each entry must match a
difference.

Normalization (representation only, never meaning):
- `--a=b` is `--a b`; a bare flag is true; several words after a flag are a
  list; the last occurrence of a flag wins.
- `vllm serve <model>` is `--model <model>`.
- Paths under the scratch repository are repo-relative; the run's results
  directories are <results>; localhost is 127.0.0.1.
- Values that parse as JSON compare as data (key order is not compared), and
  numbers compare as numbers (0.90 = 0.9, 0 = 0.0).
- lm_eval --model_args compare as key=value pairs, the last one winning.
- Aliases from the programs' own parsers: vllm bench serve --result-file is
  --result-filename; the toy proxy's --prefiller-host is --prefiller-hosts.
- `--tokenizer X` with X equal to the model counts as absent.
- The plan's AgentX client (infx.bench) takes its inputs as env; that side is
  written as the agentx_client.sh flags that export the same names, and only
  flags compare, since agentx_client.sh sets the replay's env itself.

Env is compared on effective values: the pod defaults and the stub PATH,
overlaid by each side. A serve, and the commands of a host with no serve,
compare their whole env; a task on a server host compares only what it
adds to that host's env; a chipless driver's tasks compare their whole env.
"""

import fnmatch
import json
import pathlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from golden_capture import (
    GKE_TPU_ENV,
    POD_DEFAULTS_ENV,
    SCENARIO_ROOT,
    Capture,
    capture,
    load_yaml,
)
from golden_cases import Case
from golden_refs import Placement, fill, fill_argv, fill_env
from scenario.plan.build import plan, select_point
from scenario.plan.types import (
    AgentxTaskPlan,
    CustomTaskPlan,
    EvalTaskPlan,
    PerfTaskPlan,
    Plan,
)
from scenario.resolve import Resolver

HERE = pathlib.Path(__file__).resolve().parent
ALLOWLIST = HERE / "allowlist.yaml"
RESULTS = "<results>"
STATE = "<state>"

# Env names left out of the comparison, by reason.
EXCLUDED_ENV: Mapping[str, frozenset[str]] = {
    "placement inputs; the plan gets placement from SCENARIO_* instead": frozenset(
        {
            "HEAD_HOST",
            "NUM_HOSTS",
            "DP_LOCAL",
            "POD_IP",
            "JOB_COMPLETION_INDEX",
            "ROLE_HOST_PREFIX",
            "ROLE_HOST_DOMAIN",
            "DECODE_HOST",
            "WORKLOAD_NAME",
            "MULTIHOST_ARGS_B64",
            "MULTIHOST_TOPOLOGY",
            "MULTIHOST_HOSTS",
            "K3_1P1D_ROLE",
            "REPO_DIR",
        }
    ),
    # M2 replaces this set with tpu_ci.reserved_env().
    "owned by the tpu_ci target (PLAN.md 3.6)": frozenset(
        {
            "FORCE_COLOR",
            "TQDM_MININTERVAL",
            "VLLM_CACHE_ROOT",
            "VLLM_XLA_CACHE_PATH",
            "HF_DATASETS_CACHE",
            "TMPDIR",
            "RAY_TMPDIR",
            "RAY_LOG_TO_STDERR",
            "HF_TOKEN",
            "HF_HOME",
            "CI_RUNNER",
            "CREATED_BY",
            "RUN_TYPE",
            "REPOSITORY",
            "DOCKER_IMAGE",
            "ARTIFACTS_DIR",
            "SKIP_SPANNER_UPLOAD",
        }
    ),
    "CI plumbing common.sh forwards to every pod; no server, client or eval "
    "reads it, and upload switches are after-task data": frozenset(
        {
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE",
            "UV_INDEX_TORCH_TPU_REGISTRY_USERNAME",
            "UV_NO_CACHE",
            "NIGHTLY",
            "BQ_UPLOAD",
        }
    ),
    "client knobs, compared through the argv they produce": frozenset(
        {
            "BENCHMARK_WARMUP_RUNS",
            "RUN_LM_EVAL",
            "RUN_CODE_EVAL",
            "EVAL_SOFT_FAIL",
            "CAPTURE_PROFILE",
            "MODEL_URI",
        }
    ),
    "bash's own": frozenset({"PWD", "OLDPWD", "SHLVL", "_"}),
}
EXCLUDED_PREFIXES: Mapping[str, str] = {
    "BUILDKITE_": "swept into pods by the launcher; reserved by tpu_ci",
    "GOLDEN_": "the harness's own",
    "__CF_": "added by macOS to processes it starts",
}
_EXCLUDED = frozenset().union(*EXCLUDED_ENV.values())

# agentx_client.sh's flags, each with the AgentX env name it exports with the
# same value; --chips and --label are worked out in agentx_flags().
AGENTX_SCRIPT = "agentx_client.sh"
AGENTX_FLAGS: Mapping[str, str] = {
    "server-url": "AIPERF_SERVER_URL",
    "conc-list": "CONC",
    "results-dir": "AGENTIC_OUTPUT_DIR",
    "duration": "DURATION",
    "fast": "AIPERF_EXPERIMENTAL_FAST",
    "model-prefix": "MODEL_PREFIX",
    "precision": "PRECISION",
    "spec-method": "SPEC_DECODING",
}

# Program-specific flag aliases from each program's own parser.
ALIASES: Mapping[str, Mapping[str, str]] = {
    "vllm bench serve": {"result-file": "result-filename"},
    "python3 examples/disagg/toy_proxy_server.py": {
        "prefiller-host": "prefiller-hosts",
        "prefiller-port": "prefiller-ports",
        "decoder-host": "decoder-hosts",
        "decoder-port": "decoder-ports",
    },
}
_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")
_LOCALHOST = re.compile(r"\blocalhost\b")
_MISSING = object()


def show(value: Any) -> str | None:
    """A value as the text diffs and allowlist entries use."""
    if value is None or value is _MISSING:
        return None
    if value is True or value is False:
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


@dataclass(frozen=True)
class Call:
    program: str
    positional: tuple[Any, ...]
    flags: Mapping[str, Any]
    env: Mapping[str, str]


@dataclass(frozen=True)
class Diff:
    where: str
    kind: str
    name: str
    today: str | None
    plan: str | None


class Normalizer:
    def __init__(
        self, capture: Capture, results: Sequence[str], inputs: Iterable[str] = ()
    ) -> None:
        self.excluded = _EXCLUDED | frozenset(inputs)
        self.replacements = [
            (f"{capture.root}/", ""),
            (str(capture.root), "."),
            (str(capture.state), STATE),
        ]
        self.replacements += [(r, RESULTS) for r in sorted(results, key=len)[::-1]]

    def text(self, value: str) -> str:
        for old, new in self.replacements:
            value = value.replace(old, new)
        return _LOCALHOST.sub("127.0.0.1", value)

    def scalar(self, value: str) -> Any:
        if value[:1] in "{[":
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        if _NUMBER.fullmatch(value):
            return float(value)
        return value

    def env(self, env: Mapping[str, str]) -> dict[str, str]:
        return {
            name: self.text(value)
            for name, value in env.items()
            if name not in self.excluded
            and not name.startswith(tuple(EXCLUDED_PREFIXES))
        }

    def call(self, argv: Sequence[str], env: Mapping[str, str]) -> Call:
        words = [self.text(word) for word in argv]
        program, rest = _program(words)
        aliases = ALIASES.get(program, {})
        positional: list[Any] = []
        flags: dict[str, Any] = {}
        i = 0
        while i < len(rest):
            word = rest[i]
            if not word.startswith("--"):
                positional.append(self.scalar(word))
                i += 1
                continue
            name, eq, inline = word[2:].partition("=")
            values: list[str] = [inline] if eq else []
            i += 1
            while not eq and i < len(rest) and not rest[i].startswith("--"):
                values.append(rest[i])
                i += 1
            name = aliases.get(name, name)
            if not values:
                flags[name] = True
            elif program == "lm_eval" and name == "model_args":
                flags[name] = self._pairs(values[0])
            elif len(values) == 1:
                flags[name] = self.scalar(values[0])
            else:
                flags[name] = [self.scalar(v) for v in values]
        if program == "vllm serve" and positional:
            flags.setdefault("model", positional.pop(0))
        if "tokenizer" in flags and flags["tokenizer"] == flags.get("model"):
            del flags["tokenizer"]
        return Call(program, tuple(positional), flags, self.env(env))

    def _pairs(self, text: str) -> dict[str, Any]:
        pairs: dict[str, Any] = {}
        for part in text.split(","):
            key, _, value = part.partition("=")
            pairs[key] = self.scalar(value)
        return pairs


def _program(words: list[str]) -> tuple[str, list[str]]:
    if words[:2] == ["vllm", "serve"]:
        return "vllm serve", words[2:]
    if words[:3] == ["vllm", "bench", "serve"]:
        return "vllm bench serve", words[3:]
    if words[:1] == ["python3"] and words[1:2] == ["-m"]:
        return f"python3 -m {words[2]}", words[3:]
    if words[:1] in (["python3"], ["bash"]) and len(words) > 1:
        return f"{words[0]} {words[1]}", words[2:]
    if words[:1] in (["ray"], ["pip"]) and len(words) > 1:
        return f"{words[0]} {words[1]}", words[2:]
    return words[0], words[1:]


@dataclass
class Side:
    """What one side runs, by slot; host keys are `<plan role>/<index>`."""

    serve: dict[str, Call] = field(default_factory=dict)
    pre: dict[str, list[Call]] = field(default_factory=dict)
    host_env: dict[str, Mapping[str, str]] = field(default_factory=dict)
    router: Call | None = None
    perf: list[tuple[str, Call]] = field(default_factory=list)
    agentx: list[tuple[str, Call]] = field(default_factory=list)
    evals: list[tuple[str, Call]] = field(default_factory=list)
    custom: list[tuple[str, Call]] = field(default_factory=list)
    gates: dict[str, tuple[float, Any]] = field(default_factory=dict)
    other: list[tuple[str, Call]] = field(default_factory=list)
    # The host the tasks run on; None for a chipless driver.
    driver: str | None = None


def _baseline(path: pathlib.Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    return {key: value for key, value in data.items() if not key.startswith("_")}


# Calls of the target's after-tasks and of the agent, which the plan leaves
# to the target (uploads) or which are no part of a workload.
_TARGET_SCRIPTS = ("upload_results.py", "parse_gke_results.py")
_IGNORED_TOOLS = ("buildkite-agent", "gsutil", "launch")


def today_side(
    capture: Capture, roles: Mapping[str, str], driver: str | None, norm: Normalizer
) -> Side:
    side = Side()
    if driver is not None:
        group, index = driver.rsplit("-", 1)
        role = roles[group]
        side.driver = None if role == "client" else f"{role}/{index}"
    for record in capture.records:
        tool, argv, host = record["tool"], record["argv"], record["host"]
        if host == "agent" or tool in _IGNORED_TOOLS:
            continue
        group, index = host.rsplit("-", 1)
        key = f"{roles[group]}/{index}"
        words = ["bash", *argv] if tool == "script" else [tool, *argv]
        call = norm.call(words, record["env"])
        script = pathlib.Path(argv[0]).name if tool == "python3" else ""
        if tool == "vllm" and argv[:1] == ["serve"]:
            side.serve[key] = call
            side.host_env[key] = call.env
        elif tool == "ray":
            if argv[:1] == ["start"]:
                side.pre.setdefault(key, []).append(call)
                side.host_env.setdefault(key, call.env)
        elif script == "warm_remote_code.py":
            side.pre.setdefault(key, []).append(call)
        elif script == "toy_proxy_server.py":
            side.router = call
        elif script == "check_regression.py":
            mode = call.flags["mode"]
            path = pathlib.Path(record["cwd"]) / argv[argv.index("--baseline") + 1]
            name = path.name.removesuffix(".baseline.json")
            gate = "perf" if mode == "perf" else f"eval {name.rsplit('.', 1)[1]}"
            side.gates[gate] = (call.flags["tolerance"], _baseline(path))
        elif script in _TARGET_SCRIPTS:
            continue
        elif tool == "vllm" and argv[:2] == ["bench", "serve"]:
            label = pathlib.Path(str(call.flags.get("result-filename", "?"))).stem
            side.perf.append((label, call))
        elif tool == "lm_eval":
            side.evals.append((str(call.flags.get("tasks")), call))
        elif tool == "script" and pathlib.Path(argv[0]).name == AGENTX_SCRIPT:
            side.agentx.append((AGENTX_SCRIPT, call))
        elif tool == "script":
            side.custom.append((pathlib.Path(argv[0]).stem, call))
        else:
            side.other.append((key, call))
    return side


def agentx_flags(env: Mapping[str, str]) -> list[str]:
    """The plan's AgentX env as the agentx_client.sh flags that set it.

    --chips is the chips the client divides by: TP for one role, and each
    worker's TP summed for P/D. --label is the result file's prefix, before
    _c<N>.
    """
    argv = []
    for flag, name in AGENTX_FLAGS.items():
        if name in env:
            argv += [f"--{flag}", env[name]]
    if env.get("IS_MULTINODE") == "true":
        chips = sum(
            int(env[f"{kind}_NUM_WORKERS"]) * int(env[f"{kind}_TP"])
            for kind in ("PREFILL", "DECODE")
        )
    else:
        chips = int(env["TP"])
    label = env["RESULT_FILENAME"].removesuffix(f"_c{env['CONC']}")
    return [*argv, "--chips", str(chips), "--label", label]


def plan_side(
    plan: Plan, capture: Capture, roles: Mapping[str, str], norm: Normalizer
) -> Side:
    side = Side()
    ips: dict[str, tuple[str, ...]] = {}
    names: dict[str, tuple[str | None, ...]] = {}
    for group, role in roles.items():
        pods = sorted(
            (p for p in capture.pods if p.group == group), key=lambda p: p.index
        )
        ips[role] = tuple(p.ip for p in pods)
        names[role] = tuple(p.dns for p in pods)

    def placement(role: str | None, index: int) -> Placement:
        return Placement(
            role=role,
            index=index,
            ips=ips,
            names=names,
            results_dir=RESULTS,
            pulled_model_dir=str(capture.state / "pulled-model"),
            server_log=f"{RESULTS}/server.log",
        )

    base = {**capture.base_env, **POD_DEFAULTS_ENV}
    raw_env: dict[str, dict[str, str]] = {}
    for role in plan.roles:
        for host in role.host_plans:
            key, at = f"{role.name}/{host.index}", placement(role.name, host.index)
            env = {
                **base,
                **GKE_TPU_ENV,
                **fill_env(host.scenario_env, at),
                **fill_env(host.tool_env, at),
            }
            raw_env[key] = env
            side.host_env[key] = norm.env(env)
            side.pre[key] = [
                norm.call(fill_argv(c, at), env | fill_env(c.env, at)) for c in host.pre
            ]
            if host.serve is not None:
                side.serve[key] = norm.call(fill_argv(host.serve, at), env)

    driver = plan.driver
    if driver.chipless:
        at, task_base = placement(None, 0), dict(base)
    else:
        assert driver.role is not None
        side.driver = f"{driver.role}/0"
        at, task_base = placement(driver.role, 0), dict(raw_env[side.driver])
    task_base |= fill_env(driver.task_env, at)
    if driver.router is not None:
        side.router = norm.call(
            tuple(fill(v, at) for v in driver.router.argv), task_base
        )

    def task_call(command: Any) -> Call:
        return norm.call(fill_argv(command, at), task_base | fill_env(command.env, at))

    for task in driver.tasks:
        if isinstance(task, PerfTaskPlan):
            for cell in task.cells:
                for n, warmup in enumerate(cell.warmups, 1):
                    side.perf.append((f"{cell.key}.warmup{n}", task_call(warmup)))
                side.perf.append((cell.key, task_call(cell.measured)))
            if task.gate is not None:
                side.gates["perf"] = (task.gate.tolerance, task.gate.as_dict())
        elif isinstance(task, AgentxTaskPlan):
            env = task_base | fill_env(task.command.env, at)
            flags = norm.call(["agentx", *agentx_flags(env)], {}).flags
            program = norm.text(" ".join(fill_argv(task.command, at)))
            side.agentx.append((f"c{env['CONC']}", Call(program, (), flags, {})))
        elif isinstance(task, EvalTaskPlan):
            for run in task.runs:
                side.evals.append((run.key, task_call(run.command)))
                if run.gate is not None:
                    side.gates[f"eval {run.task}"] = (
                        run.gate.tolerance,
                        run.gate.as_dict(),
                    )
        elif isinstance(task, CustomTaskPlan):
            side.custom.append((task.name, task_call(task.command)))
    return side


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    return a == b


def _env_diffs(
    where: str, today: Mapping[str, str], plan: Mapping[str, str]
) -> list[Diff]:
    return [
        Diff(where, "env", name, today.get(name), plan.get(name))
        for name in sorted(set(today) | set(plan))
        if today.get(name) != plan.get(name)
    ]


def _added(env: Mapping[str, str], host: Mapping[str, str]) -> dict[str, str]:
    return {k: v for k, v in env.items() if host.get(k) != v}


def _call_diffs(
    where: str,
    today: Call | None,
    plan: Call | None,
    env: str | None,
    sides: tuple[Side, Side],
    host: str | None,
    label: str,
) -> list[Diff]:
    """Differences of one slot; a call on one side only is one `command` diff."""
    if today is None or plan is None:
        present = today or plan
        assert present is not None
        return [
            Diff(
                where,
                "command",
                present.program,
                label if today else None,
                label if plan else None,
            )
        ]
    diffs = []
    if today.program != plan.program:
        diffs.append(Diff(where, "command", "program", today.program, plan.program))
    if today.positional != plan.positional:
        diffs.append(
            Diff(
                where,
                "command",
                "positional",
                show(list(today.positional)),
                show(list(plan.positional)),
            )
        )
    for name in list(dict.fromkeys([*today.flags, *plan.flags])):
        a, b = today.flags.get(name, _MISSING), plan.flags.get(name, _MISSING)
        if not _same(a, b):
            diffs.append(Diff(where, "flag", f"--{name}", show(a), show(b)))
    if env == "full":
        diffs += _env_diffs(where, today.env, plan.env)
    elif env == "added" and host is not None:
        diffs += _env_diffs(
            where,
            _added(today.env, sides[0].host_env.get(host, {})),
            _added(plan.env, sides[1].host_env.get(host, {})),
        )
    return diffs


def _paired(
    where: str,
    today: Sequence[tuple[str, Call]],
    plan: Sequence[tuple[str, Call]],
    env: str | None,
    sides: tuple[Side, Side],
    host: str | None,
) -> list[Diff]:
    diffs = []
    for i in range(max(len(today), len(plan))):
        t = today[i] if i < len(today) else None
        p = plan[i] if i < len(plan) else None
        label = p[0] if p is not None else t[0] if t is not None else "?"
        diffs += _call_diffs(
            f"{where} {label}", t and t[1], p and p[1], env, sides, host, label
        )
    return diffs


def compare(today: Side, plan: Side, compared: Iterable[str]) -> list[Diff]:
    """Every difference between the sides, in slot order."""
    kinds = set(compared)
    sides = (today, plan)
    diffs: list[Diff] = []
    for key in sorted(set(today.serve) | set(plan.serve)):
        diffs += _call_diffs(
            f"serve {key}",
            today.serve.get(key),
            plan.serve.get(key),
            "full",
            sides,
            key,
            key,
        )
    for key in sorted(set(today.pre) | set(plan.pre)):
        env = None if key in plan.serve else "full"
        pre = [(f"#{i}", c) for i, c in enumerate(today.pre.get(key, []))]
        mine = [(f"#{i}", c) for i, c in enumerate(plan.pre.get(key, []))]
        diffs += _paired(f"pre {key}", pre, mine, env, sides, key)
    task_env = "full" if plan.driver is None else "added"
    if kinds and today.driver != plan.driver:
        diffs.append(Diff("driver", "command", "host", today.driver, plan.driver))
    if "router" in kinds and (today.router or plan.router):
        diffs += _call_diffs(
            "router", today.router, plan.router, task_env, sides, plan.driver, "router"
        )
    if "perf" in kinds:
        diffs += _paired("perf", today.perf, plan.perf, task_env, sides, plan.driver)
    if "agentx" in kinds:
        diffs += _paired("agentx", today.agentx, plan.agentx, None, sides, None)
    if "eval" in kinds:
        diffs += _paired("eval", today.evals, plan.evals, task_env, sides, plan.driver)
    if "custom" in kinds:
        mine = {call.program: (name, call) for name, call in plan.custom}
        theirs = {call.program: (name, call) for name, call in today.custom}
        for program in dict.fromkeys([*theirs, *mine]):
            name = (mine.get(program) or theirs[program])[0]
            t, p = theirs.get(program), mine.get(program)
            diffs += _call_diffs(
                f"custom {name}",
                t and t[1],
                p and p[1],
                task_env,
                sides,
                plan.driver,
                name,
            )
    if "gate" in kinds:
        for gate in sorted(set(today.gates) | set(plan.gates)):
            t, p = today.gates.get(gate), plan.gates.get(gate)
            for i, what in enumerate(("tolerance", "baseline")):
                a = t[i] if t else None
                b = p[i] if p else None
                # A recalibrated value on one side is reported by test_baselines
                # (R-MIG-2); only a lost gate or a changed tolerance is drift.
                if what == "baseline" and a is not None and b is not None:
                    continue
                if a != b:
                    diffs.append(Diff(f"gate {gate}", "gate", what, show(a), show(b)))
    if "other" in kinds:
        for key, call in today.other:
            rest = " ".join(
                [*map(str, call.positional), *(f"--{k}" for k in call.flags)]
            )
            diffs.append(Diff(f"other {key}", "command", call.program, rest, None))
    return diffs


# Allowlist ---------------------------------------------------------------------

_ENTRY_KEYS = {"scenario", "point", "where", "kind", "name", "today", "plan", "reason"}
ANY = "*"


@dataclass(frozen=True)
class Entry:
    scenarios: tuple[str, ...]
    point: str | None
    where: str
    kind: str
    name: str
    today: str | None
    plan: str | None
    reason: str

    def matches(self, scenario: str, point: str | None, diff: Diff) -> bool:
        return (
            scenario in self.scenarios
            and (self.point is None or self.point == point)
            and fnmatch.fnmatchcase(diff.where, self.where)
            and self.kind == diff.kind
            and fnmatch.fnmatchcase(diff.name, self.name)
            and _value_matches(self.today, diff.today)
            and _value_matches(self.plan, diff.plan)
        )


def _value_matches(wanted: str | None, found: str | None) -> bool:
    if wanted == ANY:
        return found is not None
    return wanted == found


def load_allowlist(path: pathlib.Path = ALLOWLIST) -> list[Entry]:
    entries = []
    for index, raw in enumerate(load_yaml(path.read_text()) or []):
        unknown = set(raw) - _ENTRY_KEYS
        missing = _ENTRY_KEYS - {"point"} - set(raw)
        if unknown or missing:
            raise ValueError(
                f"{path.name} entry {index}: unknown keys {sorted(unknown)}, "
                f"missing keys {sorted(missing)}"
            )
        if not str(raw["reason"]).strip():
            raise ValueError(f"{path.name} entry {index}: no reason")
        scenarios = raw["scenario"]
        entries.append(
            Entry(
                scenarios=tuple(
                    scenarios if isinstance(scenarios, list) else [scenarios]
                ),
                point=raw.get("point"),
                where=str(raw["where"]),
                kind=str(raw["kind"]),
                name=str(raw["name"]),
                today=show(raw["today"]),
                plan=show(raw["plan"]),
                reason=str(raw["reason"]),
            )
        )
    return entries


@dataclass
class Outcome:
    """One case point's differences, split by whether the allowlist accepts them."""

    scenario: str
    point: str | None
    diffs: list[Diff]
    accepted: dict[Diff, list[Entry]]
    unexpected: list[Diff]


def judge(
    scenario: str, point: str | None, diffs: list[Diff], entries: Sequence[Entry]
) -> Outcome:
    accepted: dict[Diff, list[Entry]] = {}
    unexpected = []
    for diff in diffs:
        found = [entry for entry in entries if entry.matches(scenario, point, diff)]
        if found:
            accepted[diff] = found
        else:
            unexpected.append(diff)
    return Outcome(scenario, point, diffs, accepted, unexpected)


def stale(entries: Sequence[Entry], outcomes: Iterable[Outcome]) -> list[Entry]:
    """Entries that accepted no difference in any outcome."""
    used = {
        id(e)
        for outcome in outcomes
        for found in outcome.accepted.values()
        for e in found
    }
    return [entry for entry in entries if id(entry) not in used]


def run_case(
    case: Case,
    bash: str,
    work: pathlib.Path,
    entries: Sequence[Entry],
    mutate: Callable[[pathlib.Path], None] | None = None,
) -> tuple[list[Outcome], Capture]:
    """Capture today's side of a case, plan each of its points, and judge them."""
    captured = capture(work, case.launch(), case.driver, bash, mutate)
    norm = Normalizer(captured, case.results, case.inputs)
    today = today_side(captured, case.roles, case.driver, norm)
    resolver = Resolver(SCENARIO_ROOT)
    points = resolver.resolve(resolver.file_of(case.scenario))
    outcomes = []
    for point in case.points:
        mine = plan_side(plan(select_point(points, point)), captured, case.roles, norm)
        diffs = compare(today, mine, case.compared)
        outcomes.append(judge(case.scenario, point, diffs, entries))
    return outcomes, captured
