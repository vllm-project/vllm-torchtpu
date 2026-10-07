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
"""Validation rules V-01..V-40.

Field types enforce the rules about one value. The rules here need several
fields of one validated scenario (a scenario without a sweep, or one point
of a sweep), and report every problem with its key path. Rules that need
other files or several points run during resolution or `validate`; they are
declared here too, so the reference lists every rule in one table.
"""

import math
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Literal

from scenario.errors import Issue
from scenario.model.baseline import EvalBaseline, PerfBaseline
from scenario.model.common import format_duration
from scenario.model.scenario import Scenario
from scenario.model.server import RoleView
from scenario.model.sweep import POINT_NAME, Sweep, expand
from scenario.model.tasks import (
    BUILTIN_TASKS,
    LABEL_FIELDS,
    LABEL_PLACEHOLDER,
    LmEvalTask,
    PerfCell,
)
from scenario.paths import KeyPath, parse_path
from scenario.reserved import (
    AGENTX_CLIENT_ENV,
    CHIPLESS_DRIVER_ENV,
    INFERENCEX_PERF_ENV,
    LM_EVAL_MODEL_ARGS,
    PERF_CLIENT_ARGS,
    RAY_ENV,
    READY_TIMEOUT_ENV,
    ROUTER_ARGS,
    SERVE_ARGS,
    custom_task_env,
    tool_env_owner,
)

Stage = Literal["field", "scenario", "resolve", "validate"]
Check = Callable[[Scenario], Iterator[Issue]]


@dataclass(frozen=True)
class Rule:
    id: str
    summary: str
    stage: Stage
    check: Check | None = None


RULES: dict[str, Rule] = {}


def _declare(rule_id: str, summary: str, stage: Stage) -> None:
    RULES[rule_id] = Rule(rule_id, summary, stage)


def _rule(rule_id: str, summary: str) -> Callable[[Check], Check]:
    def register(check: Check) -> Check:
        RULES[rule_id] = Rule(rule_id, summary, "scenario", check)
        return check

    return register


def check_scenario(scenario: Scenario) -> list[Issue]:
    """Every scenario-stage problem of a validated scenario, in rule order."""
    issues = []
    for rule in RULES.values():
        if rule.check is not None:
            issues.extend(
                issue if issue.rule else replace(issue, rule=rule.id)
                for issue in rule.check(scenario)
            )
    return issues


def _int_arg(role: RoleView, key: str) -> int | None:
    value = role.args.get(key, 1)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


# Rules of the field, resolve and validate stages are declared without a
# check; their code is in the field types, resolve.py and `validate`.

_declare("V-01", "schema is 1", "field")


@_rule("V-02", "exactly one of server.slice and server.roles")
def _one_placement(s: Scenario) -> Iterator[Issue]:
    server = s.server
    if server.slice is None and server.roles is None:
        yield Issue(
            ("server",), "set server.slice for one role, or server.roles for several"
        )
    elif server.slice is not None and server.roles is not None:
        yield Issue(
            ("server", "roles"),
            "server.slice is set too; a scenario has one or the other",
            hint='add "slice: null" under server to drop an inherited slice',
        )


@_rule("V-03", "server.roles has two or more roles")
def _several_roles(s: Scenario) -> Iterator[Issue]:
    if s.server.roles is not None and len(s.server.roles) < 2:
        yield Issue(
            ("server", "roles"),
            "has one role; use server.slice for a single role",
        )


@_rule("V-04", "server.multihost is set exactly when a slice has more than one host")
def _multihost(s: Scenario) -> Iterator[Issue]:
    roles = s.server.role_views()
    multi = [role for role in roles if role.shape.hosts > 1]
    if multi and s.server.multihost is None:
        role = multi[0]
        yield Issue(
            (*role.path, "slice"),
            f"{role.shape} has {role.shape.hosts} hosts; set server.multihost: "
            "mp or ray",
        )
    elif roles and not multi and s.server.multihost is not None:
        slices = ", ".join(sorted({str(role.shape) for role in roles}))
        yield Issue(
            ("server", "multihost"),
            f"is set but every slice is single-host ({slices}); remove it",
        )


@_rule("V-05", "server.router only with server.roles")
def _router_needs_roles(s: Scenario) -> Iterator[Issue]:
    if s.server.router is not None and s.server.roles is None:
        yield Issue(("server", "router"), "needs server.roles")


@_rule("V-06", "router toy-proxy fronts one producer and one consumer role")
def _router_roles(s: Scenario) -> Iterator[Issue]:
    if s.server.router is None or s.server.roles is None:
        return
    kv = {name: role.kv.role for name, role in s.server.roles.items() if role.kv}
    producers = [name for name, role in kv.items() if role == "producer"]
    consumers = [name for name, role in kv.items() if role == "consumer"]
    if len(producers) != 1 or len(consumers) != 1:
        yield Issue(
            ("server", "router", "type"),
            "toy-proxy needs one producer and one consumer role; found "
            f"producers [{', '.join(producers)}] and consumers "
            f"[{', '.join(consumers)}]",
        )


# Kv.connector allows one connector, so the type enforces this today.
_declare("V-07", "every kv block uses the same connector", "field")


@_rule("V-08", "tasks.driver names a role or client, and is required with roles")
def _driver(s: Scenario) -> Iterator[Issue]:
    driver = s.tasks.driver
    if s.server.roles is not None:
        choices = [*s.server.roles, "client"]
        if driver is None:
            yield Issue(
                ("tasks",),
                "tasks.driver is required with server.roles (one of: "
                f"{', '.join(choices)})",
            )
        elif driver not in choices:
            yield Issue(
                ("tasks", "driver"),
                f"{driver} is not a role; one of: {', '.join(choices)}",
            )
    elif driver not in (None, "server"):
        yield Issue(
            ("tasks", "driver"),
            "a scenario without roles runs tasks on its first host; remove "
            "tasks.driver",
        )


STREAMING_LOADERS = ("runai_streamer", "runai_streamer_sharded")


@_rule("V-09", "an object-storage model path has a loader that reads it")
def _object_storage(s: Scenario) -> Iterator[Issue]:
    if not s.server.model.in_object_storage:
        return
    for role in s.server.role_views():
        if role.args.get("load-format") not in STREAMING_LOADERS:
            scope = f" for role {role.name}" if s.server.roles else ""
            yield Issue(
                ("server", "model", "path"),
                f"is object storage; set load-format: runai_streamer{scope} (no "
                "other loader reads object storage)",
            )


@_rule("V-10", "distributed-executor-backend agrees with server.multihost")
def _executor_backend(s: Scenario) -> Iterator[Issue]:
    multihost = s.server.multihost
    for role in s.server.role_views():
        backend = role.args.get("distributed-executor-backend")
        if (
            backend is not None
            and multihost is not None
            and role.shape.hosts > 1
            and backend != multihost
        ):
            yield Issue(
                role.arg_path("distributed-executor-backend"),
                f"is {backend} but server.multihost is {multihost}",
            )


def _env_blocks(s: Scenario) -> Iterator[tuple[KeyPath, Mapping[str, str | None], str]]:
    yield ("server", "env"), s.server.env, "server"
    for name, role in (s.server.roles or {}).items():
        yield ("server", "roles", name, "env"), role.env, "server"
    tasks = s.tasks
    if tasks.perf is not None:
        yield ("tasks", "perf", "env"), tasks.perf.env, f"perf:{tasks.perf.client}"
    if tasks.agentx is not None:
        yield ("tasks", "agentx", "env"), tasks.agentx.env, "agentx"
    if tasks.eval is not None:
        for index, item in enumerate(tasks.eval.lm_eval.tasks):
            if isinstance(item, LmEvalTask):
                path = ("tasks", "eval", "lm_eval", "tasks", index, "env")
                yield path, item.env, "task"
    for index, task in enumerate(tasks.custom):
        yield ("tasks", "custom", index, "env"), task.env, "task"


@_rule("V-11", "no env block sets a name the tool owns")
def _tool_env(s: Scenario) -> Iterator[Issue]:
    chipless = s.server.roles is not None and s.tasks.driver == "client"
    for path, env, kind in _env_blocks(s):
        for name, value in env.items():
            where = (*path, name)
            if value is None:
                continue
            if name == READY_TIMEOUT_ENV:
                yield Issue(
                    where,
                    f"{name} is set from server.startup_timeout; remove it",
                    hint="set server.startup_timeout instead (now "
                    f"{format_duration(s.server.startup_timeout)})",
                    rule="V-12",
                )
            elif owner := tool_env_owner(name):
                yield Issue(where, f"{name} is set by the tool ({owner}); remove it")
            elif kind == "perf:inferencex" and name in INFERENCEX_PERF_ENV:
                yield Issue(
                    where,
                    f"{name} is set by the tool for the inferencex client "
                    f"({INFERENCEX_PERF_ENV[name]}); remove it",
                )
            elif kind == "agentx" and name in AGENTX_CLIENT_ENV:
                yield Issue(
                    where,
                    f"{name} is set by the tool from {AGENTX_CLIENT_ENV[name]}; "
                    "remove it",
                )
            elif kind != "server" and chipless and name in CHIPLESS_DRIVER_ENV:
                yield Issue(
                    where,
                    f"{name} is set by the tool on a client driver "
                    f"({CHIPLESS_DRIVER_ENV[name]}); remove it",
                )


_declare(
    "V-12", f"no env block sets {READY_TIMEOUT_ENV} (use startup_timeout)", "scenario"
)


@_rule("V-13", "engines on the ray backend get no JAX_PLATFORMS")
def _ray_env(s: Scenario) -> Iterator[Issue]:
    if s.server.multihost != "ray":
        return
    reported = set()
    for role in s.server.role_views():
        if role.shape.hosts == 1:
            continue
        for name, owner in RAY_ENV.items():
            path = role.env_path(name)
            if name in role.env and path not in reported:
                reported.add(path)
                yield Issue(path, f"{name} is set by {owner}; remove it")


@_rule("V-14", "free-form args do not set flags the tool renders")
def _reserved_args(s: Scenario) -> Iterator[Issue]:
    server = s.server
    blocks: list[tuple[KeyPath, Mapping[str, object], Mapping[str, str]]] = [
        (("server", "args"), server.args, SERVE_ARGS)
    ]
    for name, role in (server.roles or {}).items():
        blocks.append((("server", "roles", name, "args"), role.args, SERVE_ARGS))
    if server.router is not None:
        blocks.append((("server", "router", "args"), server.router.args, ROUTER_ARGS))
    if s.tasks.perf is not None:
        path = ("tasks", "perf", "client_args")
        blocks.append((path, s.tasks.perf.client_args, PERF_CLIENT_ARGS))
    for path, args, reserved in blocks:
        for key in args:
            if key in reserved:
                yield Issue((*path, key), f"is set by {reserved[key]}; remove it")


_declare("V-15", "arg keys are flag names without dashes", "field")

_SIZES = (
    ("tensor-parallel-size", "TP"),
    ("pipeline-parallel-size", "PP"),
    ("prefill-context-parallel-size", "PCP"),
    ("data-parallel-size", "DP"),
)


@_rule("V-16", "TP x PP x PCP x DP fits the slice's devices")
def _devices(s: Scenario) -> Iterator[Issue]:
    for role in s.server.role_views():
        sizes = {}
        for key, label in _SIZES:
            size = _int_arg(role, key)
            if size is None:
                yield Issue(
                    role.arg_path(key),
                    f"must be a positive integer; got {role.args[key]!r}",
                )
            sizes[label] = size
        if None in sizes.values():
            continue
        total = math.prod(size for size in sizes.values() if size is not None)
        if total > role.shape.devices:
            terms = " x ".join(
                f"{label} {size}"
                for label, size in sizes.items()
                if size != 1 or label in ("TP", "DP")
            )
            yield Issue(
                role.args_path,
                f"{terms} = {total} devices, but {role.shape} has "
                f"{role.shape.devices} ({role.shape.hosts} hosts x "
                f"{role.shape.devices_per_host})",
            )


@_rule("V-17", "mp splits data-parallel-size evenly over hosts")
def _mp_split(s: Scenario) -> Iterator[Issue]:
    if s.server.multihost != "mp":
        return
    for role in s.server.role_views():
        hosts = role.shape.hosts
        dp = _int_arg(role, "data-parallel-size")
        if hosts > 1 and dp is not None and dp % hosts:
            yield Issue(
                role.arg_path("data-parallel-size"),
                f"multihost mp needs data-parallel-size divisible by {hosts} "
                f"hosts; got {dp}",
            )


@_rule("V-18", "tasks.perf sets isl_osl and concurrency, or cells")
def _perf_cells(s: Scenario) -> Iterator[Issue]:
    perf = s.tasks.perf
    if perf is None:
        return
    product = perf.isl_osl is not None or perf.concurrency is not None
    if perf.cells is not None and product:
        yield Issue(
            ("tasks", "perf", "cells"),
            "tasks.perf sets cells and isl_osl/concurrency; use one",
        )
    elif perf.cells is None and (perf.isl_osl is None or perf.concurrency is None):
        yield Issue(
            ("tasks", "perf"), "needs isl_osl and concurrency together, or cells"
        )


@_rule("V-19", "ISL + OSL of every perf cell fits max-model-len of every role")
def _cell_lengths(s: Scenario) -> Iterator[Issue]:
    perf = s.tasks.perf
    if perf is None:
        return
    longest: dict[KeyPath, PerfCell] = {}
    for cell in perf.cell_list():
        known = longest.get(cell.path)
        if known is None or cell.isl + cell.osl > known.isl + known.osl:
            longest[cell.path] = cell
    for role in s.server.role_views():
        limit = role.args.get("max-model-len")
        if not isinstance(limit, int) or isinstance(limit, bool):
            continue
        scope = f" for role {role.name}" if s.server.roles else ""
        for cell in longest.values():
            needed = cell.isl + cell.osl
            if needed > limit:
                yield Issue(
                    cell.path,
                    f"cell {cell.key} needs {needed} tokens; max-model-len is "
                    f"{limit}{scope}",
                )


CELL_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@_rule("V-20", "perf cell keys are unique and filename-safe")
def _cell_keys(s: Scenario) -> Iterator[Issue]:
    perf = s.tasks.perf
    if perf is None:
        return
    allowed = ", ".join(f"{{{name}}}" for name in LABEL_FIELDS)
    broken: set[KeyPath] = set()
    for index, group in enumerate(perf.cells or ()):
        path = ("tasks", "perf", "cells", index)
        label = group.label
        unknown = [
            name
            for name in LABEL_PLACEHOLDER.findall(label or "")
            if name not in LABEL_FIELDS
        ]
        if unknown:
            broken.add(path)
            yield Issue(
                (*path, "label"), f"uses {{{unknown[0]}}}; it may use {allowed}"
            )
        elif group.repeat > 1 and (label is None or "{repeat}" not in label):
            broken.add(path)
            yield Issue(
                (*path, "label") if label is not None else path,
                f"cells repeat {group.repeat} times but label has no {{repeat}}",
            )
    seen: set[str] = set()
    for cell in perf.cell_list():
        if cell.path in broken:
            continue
        if not CELL_KEY.fullmatch(cell.key):
            yield Issue(
                cell.path,
                f"cell key {cell.key!r} is not filename-safe; use letters, digits, "
                "'.', '_' and '-'",
            )
        elif cell.key in seen:
            yield Issue(cell.path, f"cell {cell.key} appears more than once")
        seen.add(cell.key)


def baseline_keys(s: Scenario, task: str) -> tuple[tuple[str, ...], bool] | None:
    """The keys a task's baseline may hold, and whether a subset of them is enough.

    An eval sweep runs every task at every concurrency, and the gate checks
    only the runs a baseline names, so its baseline may name some of them.
    """
    if task == "perf" and s.tasks.perf is not None:
        return tuple(cell.key for cell in s.tasks.perf.cell_list()), False
    if task == "eval" and s.tasks.eval is not None:
        lm_eval = s.tasks.eval.lm_eval
        return lm_eval.run_keys(), lm_eval.concurrency is not None
    return None


def baseline_key_problem(
    have: set[str], want: tuple[str, ...], subset: bool
) -> str | None:
    missing = [] if subset else [key for key in want if key not in have]
    extra = sorted(have - set(want))
    parts = []
    if missing:
        parts.append(f"no entry for {', '.join(missing)}")
    if extra:
        parts.append(f"an extra entry {', '.join(extra)}")
    return f"has {' and '.join(parts)}" if parts else None


@_rule("V-21", "inline baseline keys are the task's cell or run keys")
def _baseline_keys(s: Scenario) -> Iterator[Issue]:
    for task, kind in (("perf", "cell"), ("eval", "run")):
        block = getattr(s.tasks, task)
        rule = baseline_keys(s, task)
        if rule is None or not isinstance(block.baseline, PerfBaseline | EvalBaseline):
            continue
        want, subset = rule
        problem = baseline_key_problem(
            set(block.baseline.model_extra or {}), want, subset
        )
        if problem is not None:
            among = "among the" if subset else "the"
            yield Issue(
                ("tasks", task, "baseline"),
                problem,
                hint=f"its keys are {among} {kind} keys: {', '.join(want)}",
            )


def _interpolation_problems(value: str, provided: frozenset[str]) -> Iterator[str]:
    i = 0
    while (i := value.find("$", i)) != -1:
        rest = value[i + 1 :]
        if rest.startswith("$"):
            i += 2
            continue
        match = re.match(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", rest)
        if match is None:
            yield "has a lone $; write $$ for a literal $, or ${NAME}"
            i += 1
            continue
        if match.group(1) not in provided:
            yield f"uses ${{{match.group(1)}}}; known: {', '.join(sorted(provided))}"
        i += 1 + match.end()


@_rule("V-22", "custom tasks have unique names and use only provided ${NAME}s")
def _custom(s: Scenario) -> Iterator[Issue]:
    roles = s.server.role_views()
    provided = custom_task_env(
        roles=[role.name for role in roles],
        kv_roles={role.name: role.kv.role for role in roles if role.kv},
        router=s.server.router is not None,
        driver_on_role=not (s.server.roles and s.tasks.driver == "client"),
    )
    seen: set[str] = set()
    for index, task in enumerate(s.tasks.custom):
        path = ("tasks", "custom", index)
        if task.name in BUILTIN_TASKS:
            yield Issue((*path, "name"), f"{task.name} names a built-in task")
        elif task.name in seen:
            yield Issue((*path, "name"), f"another custom task is named {task.name}")
        seen.add(task.name)
        for name, value in task.env.items():
            for problem in _interpolation_problems(value, provided):
                yield Issue((*path, "env", name), problem)


_declare("V-23", "a baseline's from: names a scenario with matching keys", "resolve")


@_rule("V-24", "with a sweep, baselines sit with their point")
def _sweep_baselines(s: Scenario) -> Iterator[Issue]:
    if s.sweep is None:
        return
    for task in BUILTIN_TASKS:
        block = getattr(s.tasks, task)
        if block is not None and block.baseline is not None:
            yield Issue(("tasks", task, "baseline"), sweep_baseline_message(s.sweep))


def sweep_baseline_message(sweep: Sweep) -> str:
    where = (
        "under the point it gates"
        if sweep.points is not None
        else "under sweep.baselines.<point>.<task>"
    )
    return f"is not allowed with a sweep; put it {where}"


_declare("V-26", "name is unique across scenarios/", "validate")
_declare("V-29", "extends: the parent exists, no cycle, depth <= 5", "resolve")
_declare(
    "V-30", "--set paths start with server. or tasks. and avoid baselines", "resolve"
)


@_rule("V-27", "lm_eval model_args do not set what the tool sets")
def _model_args(s: Scenario) -> Iterator[Issue]:
    if s.tasks.eval is None:
        return
    for key in s.tasks.eval.lm_eval.model_args:
        if key in LM_EVAL_MODEL_ARGS:
            path = ("tasks", "eval", "lm_eval", "model_args", key)
            yield Issue(path, f"is set by {LM_EVAL_MODEL_ARGS[key]}; remove it")


@_rule("V-28", "the random dataset has random_range_ratio")
def _random_dataset(s: Scenario) -> Iterator[Issue]:
    perf = s.tasks.perf
    if perf is None:
        return
    if perf.dataset.name == "random" and perf.random_range_ratio is None:
        yield Issue(
            ("tasks", "perf", "random_range_ratio"), "is required with dataset random"
        )
    if perf.range_ratio_style == "min" and perf.dataset.name != "random":
        yield Issue(
            ("tasks", "perf", "range_ratio_style"),
            "min applies to the random dataset only",
        )


_declare("V-31", "tasks.agentx.concurrency is one value", "field")


@_rule("V-32", "max-num-seqs x DP covers tasks.agentx.concurrency, twice over")
def _agentx_capacity(s: Scenario) -> Iterator[Issue]:
    agentx = s.tasks.agentx
    if agentx is None:
        return
    want = agentx.concurrency
    for role in s.server.role_views():
        if role.kv is not None and role.kv.role == "producer":
            continue
        seqs = role.args.get("max-num-seqs")
        dp = _int_arg(role, "data-parallel-size")
        if not isinstance(seqs, int) or isinstance(seqs, bool) or dp is None:
            continue
        capacity = seqs * dp
        sizes = f"max-num-seqs {seqs} x DP {dp} = {capacity}"
        if capacity < want:
            yield Issue(
                role.arg_path("max-num-seqs"),
                f"{sizes}, under tasks.agentx.concurrency {want}; the server "
                "would cap the measured concurrency",
            )
        elif capacity < 2 * want:
            yield Issue(
                role.arg_path("max-num-seqs"),
                f"{sizes}, under twice tasks.agentx.concurrency ({2 * want}); "
                "AgentX sessions spawn subagents, so the server may cap them",
                severity="warning",
            )


@_rule("V-34", "tasks.agentx is the only task")
def _agentx_alone(s: Scenario) -> Iterator[Issue]:
    tasks = s.tasks
    if tasks.agentx is None:
        return
    others = [name for name in ("perf", "custom", "eval") if getattr(tasks, name)]
    if others:
        yield Issue(
            ("tasks", "agentx"),
            f"must be the only task, but tasks.{others[0]} is set too: AgentX "
            "needs a fresh server, and an earlier task would warm it",
            hint="move the other tasks to their own scenario",
        )


@_rule("V-35", "perf fields match the perf client")
def _perf_client(s: Scenario) -> Iterator[Issue]:
    perf = s.tasks.perf
    if perf is None:
        return
    path = ("tasks", "perf", "range_ratio_style")
    if perf.client == "inferencex" and perf.range_ratio_style is not None:
        yield Issue(
            path,
            "applies to vllm-bench only; inferencex reads random_range_ratio as "
            "written",
        )
    elif (
        perf.client == "vllm-bench"
        and perf.dataset.name == "random"
        and perf.range_ratio_style is None
    ):
        yield Issue(
            path,
            "is required with client vllm-bench and dataset random: symmetric or min",
            hint="scenarios/defaults.yaml normally sets it",
        )


@_rule("V-36", "lm_eval tasks, concurrency and per-concurrency limits agree")
def _lm_eval(s: Scenario) -> Iterator[Issue]:
    if s.tasks.eval is None:
        return
    lm_eval = s.tasks.eval.lm_eval
    base = ("tasks", "eval", "lm_eval")
    seen: set[str] = set()
    for index, item in enumerate(lm_eval.task_items()):
        if item.name in seen:
            yield Issue((*base, "tasks", index), f"{item.name} is listed twice")
        seen.add(item.name)
    if lm_eval.concurrency is not None:
        if len(set(lm_eval.concurrency)) != len(lm_eval.concurrency):
            yield Issue((*base, "concurrency"), "has repeated values")
        if "num_concurrent" in lm_eval.model_args:
            yield Issue(
                (*base, "model_args", "num_concurrent"),
                "is set per run from tasks.eval.lm_eval.concurrency; remove it",
            )
    for index, item in enumerate(lm_eval.tasks):
        if not isinstance(item, LmEvalTask):
            continue
        path = (*base, "tasks", index)
        if item.limit_per_concurrency is not None:
            limit = item.limit if item.limit is not None else lm_eval.limit
            if lm_eval.concurrency is None:
                yield Issue(
                    (*path, "limit_per_concurrency"),
                    "needs tasks.eval.lm_eval.concurrency",
                )
            elif limit is None or limit == "all":
                yield Issue(
                    (*path, "limit_per_concurrency"),
                    "raises a limit, but this task runs the full set; set limit",
                )
        elif item.subtasks is not None:
            yield Issue(
                (*path, "subtasks"),
                "only divides limit_per_concurrency; set that or remove subtasks",
            )


@_rule("V-37", "tasks.agentx.timeout is longer than its duration")
def _agentx_timeout(s: Scenario) -> Iterator[Issue]:
    agentx = s.tasks.agentx
    if agentx is not None and agentx.timeout <= agentx.duration:
        yield Issue(
            ("tasks", "agentx", "timeout"),
            f"must be longer than duration ({format_duration(agentx.duration)}), "
            "to cover warmup and drain",
        )


@_rule("V-38", "a sweep has one form, axes under server.* or tasks.*, unique names")
def _sweep(s: Scenario) -> Iterator[Issue]:
    if s.sweep is not None:
        yield from check_sweep(s.sweep)


_declare("V-33", "zip axes have equal lengths", "scenario")
_declare("V-39", "sweep axis paths exist in the resolved scenario", "resolve")
_declare("V-40", "--set does not touch a sweep axis", "resolve")


def check_sweep(sweep: Sweep, base: KeyPath = ("sweep",)) -> list[Issue]:
    """Problems of a sweep block on its own (V-24, V-33, V-38).

    Each point is its own workload, so a point may change any server or task
    field, the slices included.
    """
    issues: list[Issue] = []
    forms = [form for form in ("points", "zip", "product") if getattr(sweep, form)]
    if len(forms) != 1:
        found = f"; found {', '.join(forms)}" if forms else ""
        issues.append(
            Issue(
                base, f"set exactly one of points, zip and product{found}", rule="V-38"
            )
        )
        return issues
    (form,) = forms
    if sweep.points is not None:
        if sweep.name is not None:
            issues.append(
                Issue(
                    (*base, "name"),
                    "is for zip and product; named points carry their own names",
                    rule="V-38",
                )
            )
        if sweep.baselines is not None:
            issues.append(
                Issue(
                    (*base, "baselines"),
                    "is for zip and product; a named point carries its baselines "
                    "in its own tasks",
                    rule="V-38",
                )
            )
        seen: set[str] = set()
        for index, point in enumerate(sweep.points):
            path = (*base, "points", index)
            if point.name in seen:
                issues.append(
                    Issue(
                        (*path, "name"),
                        f"another point is named {point.name}",
                        rule="V-38",
                    )
                )
            seen.add(point.name)
        return issues

    axes = sweep.axes
    for text in axes:
        path = parse_path(text)
        where = (*base, form, text)
        if len(path) < 2 or path[0] not in ("server", "tasks"):
            issues.append(
                Issue(where, "axes override server.* or tasks.* only", rule="V-38")
            )
        elif "baseline" in path:
            issues.append(
                Issue(where, "point baselines go under sweep.baselines", rule="V-24")
            )
    if sweep.zip is not None:
        lengths = {text: len(values) for text, values in sweep.zip.items()}
        if len(set(lengths.values())) > 1:
            listed = ", ".join(f"{text} has {n}" for text, n in lengths.items())
            issues.append(
                Issue((*base, "zip"), f"axes need equal lengths; {listed}", rule="V-33")
            )
    if sweep.name is None:
        example = next(iter(axes))
        issues.append(
            Issue(
                base,
                f"name is required with {form}, e.g. name: x{{{example}}}",
                rule="V-38",
            )
        )
    if issues:
        return issues

    try:
        names = [point.name for point in expand(sweep)]
    except ValueError as error:
        return [Issue((*base, "name"), str(error), rule="V-38")]
    invalid = [name for name in names if not POINT_NAME.fullmatch(name)]
    if invalid:
        issues.append(
            Issue(
                (*base, "name"),
                f"gives the point name {invalid[0]!r}; point names are lowercase "
                "letters, digits and '-', start with a letter, and are at most "
                "63 characters",
                rule="V-38",
            )
        )
    for name in sorted(set(names)):
        if names.count(name) > 1:
            issues.append(
                Issue(
                    (*base, "name"),
                    f"gives {name} to {names.count(name)} points; use every axis "
                    "that varies",
                    rule="V-38",
                )
            )
    for name in sweep.baselines or {}:
        if name not in names:
            issues.append(
                Issue(
                    (*base, "baselines", name),
                    f"no point is named {name}; points: {', '.join(names)}",
                    rule="V-38",
                )
            )
    return issues


def rule_table() -> list[Rule]:
    """Every rule, in id order."""
    return sorted(RULES.values(), key=lambda rule: int(rule.id[2:]))
