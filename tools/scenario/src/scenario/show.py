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
"""Text forms of a resolved scenario and of a Plan.

`show`, `show --plan`, `run --dry-run` and the invariance test all use
render_plan, so what is shown is what a target hands to pods.
"""

import json
import re
import shlex
from collections.abc import Iterator, Sequence
from typing import Any

from scenario.merge import TOOL, Origin, replaces_whole
from scenario.model.common import format_duration
from scenario.paths import KeyPath, format_path
from scenario.plan.env import env_rule
from scenario.plan.types import (
    AgentxTaskPlan,
    Command,
    Concat,
    CustomTaskPlan,
    EnvItems,
    EvalTaskPlan,
    GatePlan,
    HostPlan,
    PerfTaskPlan,
    Plan,
    Ref,
    RolePlan,
    Value,
)
from scenario.reserved import (
    AGENTX_CLIENT_ENV,
    INFERENCEX_EVAL_ENV,
    INFERENCEX_PERF_ENV,
)
from scenario.resolve import Resolved

WIDTH = 100
_SAFE = re.compile(r"[A-Za-z0-9_@%+=:,./<>|\-]+")


def render_ref(ref: Ref) -> str:
    text = ref.kind
    if ref.role is not None:
        text += f":{ref.role}"
    if ref.sep != ",":
        text += f" sep {ref.sep!r}"
    if ref.fallback is not None:
        text += f"|{ref.fallback}"
    return f"<{text}>"


def render_value(value: Value) -> str:
    """Text with each Ref as <kind[:role][|fallback]>."""
    if isinstance(value, Ref):
        return render_ref(value)
    if isinstance(value, Concat):
        return "".join(render_value(part) for part in value.parts)
    return value


def _quote(text: str) -> str:
    return text if _SAFE.fullmatch(text) else shlex.quote(text)


def _words(argv: Sequence[Value]) -> list[str]:
    """argv as shell words, each flag kept with its value."""
    words: list[str] = []
    for item in argv:
        word = _quote(render_value(item))
        last = words[-1] if words else ""
        if last.startswith("-") and " " not in last and not word.startswith("-"):
            words[-1] += f" {word}"
        else:
            words.append(word)
    return words


def render_argv(argv: Sequence[Value], indent: int) -> str:
    """argv on lines of at most WIDTH, continued with a backslash."""
    lines: list[str] = []
    line = ""
    for word in _words(argv):
        if line and indent + len(line) + 1 + len(word) + 2 > WIDTH:
            lines.append(line + " \\")
            line = "    " + word
        else:
            line = f"{line} {word}" if line else word
    lines.append(line)
    pad = " " * indent
    return lines[0] + "".join(f"\n{pad}{rest}" for rest in lines[1:])


def _env_lines(items: EnvItems, tagged: set[str] | None, indent: str) -> list[str]:
    """One NAME=value line each; names the tool set carry their P-item."""
    lines = []
    for name, value in items:
        text = f"{indent}{name}={_quote(render_value(value)) if value != '' else ''}"
        rule = env_rule(name)
        if rule is not None and (tagged is None or name in tagged):
            text += f"  [{rule}]"
        lines.append(text)
    return lines


def _command(label: str, command: Command, indent: str, tool: set[str]) -> list[str]:
    head = f"{indent}{label}: "
    lines = [head + render_argv(command.argv, len(head))]
    extra = []
    if command.timeout_s is not None:
        extra.append(f"timeout {format_duration(command.timeout_s)}")
    if command.best_effort:
        extra.append("best effort")
    if extra:
        lines[-1] += f"  ({', '.join(extra)})"
    if command.env:
        lines.append(f"{indent}  env:")
        lines += _env_lines(command.env, tool, indent + "    ")
    return lines


def format_bytes(size: int) -> str:
    """The largest binary unit that divides `size`: 68719476736 -> 64Gi."""
    for unit, scale in (("Ti", 2**40), ("Gi", 2**30), ("Mi", 2**20), ("Ki", 2**10)):
        if size % scale == 0:
            return f"{size // scale}{unit}"
    return f"{size} bytes"


def _gate(gate: GatePlan | None) -> str:
    if gate is None:
        return "no gate"
    soft = ", soft" if gate.soft else ""
    return f"gate tolerance {gate.tolerance} ({gate.source}{soft})"


def _host(role: RolePlan, host: HostPlan) -> list[str]:
    title = f"  host {host.index}" + (" (head)" if host.head else "")
    lines = [title]
    for command in host.pre:
        lines += _command("pre", command, "    ", set())
    if host.serve is None:
        lines.append("    serve: none; a Ray worker serves through the head")
    else:
        lines += _command("serve", host.serve, "    ", set())
    if host.dropped_flags:
        lines.append(f"    left off: {' '.join(host.dropped_flags)}  [P-05]")
    lines.append("    scenario env:" + ("" if host.scenario_env else " none"))
    lines += _env_lines(host.scenario_env, set(), "      ")
    lines.append("    tool env:")
    lines += _env_lines(host.tool_env, None, "      ")
    return lines


def _role(role: RolePlan, hosts: Sequence[int] | None) -> list[str]:
    lines = [
        f"role {role.name}: {role.accelerator} {role.topology}, {role.hosts} "
        f"host{'s' if role.hosts > 1 else ''} x {role.chips_per_host} "
        f"chip{'s' if role.chips_per_host > 1 else ''}, {role.backend}"
    ]
    ready = role.readiness
    lines.append(
        f"  ready: GET :{ready.port}{ready.path} within "
        f"{format_duration(ready.timeout_s)} of the server's start  [P-09]"
    )
    if role.kv is not None:
        lines.append(f"  kv: {role.kv.connector} {role.kv.kv_role}  [P-11]")
    if role.scratch_bytes is not None:
        lines.append(f"  scratch: {format_bytes(role.scratch_bytes)} per host")
    for host in role.host_plans:
        if hosts is None or host.index in hosts:
            lines += _host(role, host)
    return lines


def _tasks(plan: Plan) -> Iterator[str]:
    for task in plan.driver.tasks:
        if isinstance(task, PerfTaskPlan):
            count = len(task.cells)
            yield (
                f"  perf ({task.client}, {task.mode}): {count} "
                f"cell{'s' if count != 1 else ''}, {_gate(task.gate)}"
            )
            tool = set(INFERENCEX_PERF_ENV) if task.client == "inferencex" else set()
            for cell in task.cells:
                yield f"    {cell.key}"
                for index, command in enumerate(cell.warmups, 1):
                    yield from _command(f"warmup {index}", command, "      ", tool)
                yield from _command("run", cell.measured, "      ", tool)
        elif isinstance(task, AgentxTaskPlan):
            yield "  agentx: no gate; InferenceX checks the error rate"
            yield from _command("run", task.command, "    ", set(AGENTX_CLIENT_ENV))
        elif isinstance(task, CustomTaskPlan):
            skip = ""
            if task.skip_rest_if_exit_ge is not None:
                skip = f", skips the rest on exit >= {task.skip_rest_if_exit_ge}"
            yield f"  custom {task.name}{skip}"
            provided = {
                name for name, _ in task.command.env if env_rule(name) == "P-17"
            }
            yield from _command("run", task.command, "    ", provided)
        elif isinstance(task, EvalTaskPlan):
            inferencex = task.client == "inferencex"
            yield "  eval (inferencex)" if inferencex else "  eval"
            tool = set(INFERENCEX_EVAL_ENV) if inferencex else set()
            for run in task.runs:
                gate = _gate(run.gate)
                if run.floor is not None:
                    gate = "gate: InferenceX's score floor"
                yield f"    {run.key}: {run.tool}, {gate}"
                yield from _command("run", run.command, "      ", tool)
                if run.floor is not None:
                    yield from _command("floor", run.floor, "      ", tool)


def _driver(plan: Plan) -> list[str]:
    driver = plan.driver
    where = "a chipless client host" if driver.chipless else f"{driver.role} host 0"
    endpoint = render_value(driver.endpoint_host)
    lines = [f"driver: {where}; tasks connect to {endpoint}:{driver.endpoint_port}"]
    lines[-1] += "  [P-14]"
    if driver.router is not None:
        router = driver.router
        head = "  router: "
        lines.append(head + render_argv(router.argv, len(head)))
        lines.append(f"    ready: GET :{router.port}{router.health_path}  [P-13]")
    lines.append("  task env:")
    lines += _env_lines(driver.task_env, None, "    ")
    if not driver.tasks:
        lines.append("  no tasks: the servers stay up")
    lines += _tasks(plan)
    return lines


def render_plan(plan: Plan, role: str | None = None, host: int | None = None) -> str:
    """Every host's commands and env, then the driver's tasks.

    With `role` (and `host`), only that role's hosts, and the driver section
    only when the driver runs there.
    """
    lines = [
        f"plan {plan.scenario_name} ({plan.scenario_ref}), format {plan.format}, "
        f"tool {plan.tool_version}"
    ]
    if plan.point is not None:
        lines.append(f"point: {plan.point}")
    lines.append(f"image: {plan.image or 'the target supplies it'}")
    lines.append(f"model: {plan.model_name}, loaded from {plan.model_path}")
    if plan.experiment:
        lines.append(f"experiment, gates off: {' '.join(plan.experiment)}")
    if plan.only is not None:
        lines.append(f"only: {', '.join(plan.only)}")
    for role_plan in plan.roles:
        if role is None or role == role_plan.name:
            lines.append("")
            lines += _role(role_plan, None if host is None else [host])
    driver = plan.driver
    on_driver = (
        role is None
        or (driver.chipless and role == "client")
        or (role == driver.role and host in (None, 0))
    )
    if on_driver:
        lines.append("")
        lines += _driver(plan)
    return "\n".join(lines) + "\n"


def describe(plan: Plan) -> str:
    """One line: the roles and the tasks."""
    roles = []
    for role in plan.roles:
        text = f"{role.name} {role.accelerator} {role.topology}"
        if role.hosts > 1:
            text += f" ({role.hosts} hosts, {role.backend})"
        roles.append(text)
    parts = [", ".join(roles)]
    if plan.driver.router is not None:
        parts[-1] += ", router"
    if len(plan.roles) > 1:
        parts[-1] += f", driver {plan.driver.role or 'client'}"
    tasks = []
    for task in plan.driver.tasks:
        if isinstance(task, PerfTaskPlan):
            count = len(task.cells)
            tasks.append(f"perf {count} cell{'s' if count != 1 else ''}")
        elif isinstance(task, AgentxTaskPlan):
            conc = dict(task.command.env).get("CONC")
            tasks.append(f"agentx c{conc}")
        elif isinstance(task, CustomTaskPlan):
            tasks.append(f"custom {task.name}")
        elif isinstance(task, EvalTaskPlan) and task.client == "inferencex":
            for run in task.runs:
                argv = run.command.argv
                conc = argv[argv.index("--concurrency") + 1]
                tasks.append(f"eval inferencex {run.key} c{render_value(conc)}")
        elif isinstance(task, EvalTaskPlan):
            tasks.append("eval " + " ".join(run.key for run in task.runs))
    return "; ".join(parts + [", ".join(tasks) or "no tasks"])


# Resolved scenarios ---------------------------------------------------------


def render_data(value: Any) -> str:
    """A resolved value as one line: scalars as written, the rest as JSON."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return value if value else '""'
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(value, separators=(", ", ": "))


def _leaves(data: Any, path: KeyPath = ()) -> Iterator[tuple[KeyPath, Any]]:
    if isinstance(data, dict) and data and not replaces_whole(path):
        for key, value in data.items():
            yield from _leaves(value, (*path, key))
    elif path:
        yield path, data


def _defaults(resolved: Resolved) -> Iterator[tuple[KeyPath, Any]]:
    """Model defaults no layer set, leaving out unset (null or empty) fields."""
    dumped = resolved.model.model_dump(by_alias=True, exclude_none=True, warnings=False)
    for path, value in _leaves(dumped):
        if value not in ({}, []) and not _has(resolved.data, path):
            yield path, value


def _has(data: Any, path: KeyPath) -> bool:
    node = data
    for key in path:
        in_dict = isinstance(node, dict) and key in node
        in_list = isinstance(node, list) and isinstance(key, int) and key < len(node)
        if not (in_dict or in_list):
            return False
        node = node[key]
    return True


def _get(data: Any, path: KeyPath) -> Any:
    node = data
    for key in path:
        node = node[key]
    return node


def _where(origin: Origin) -> str:
    return origin.render()


def render_resolved(resolved: Resolved) -> str:
    """Every resolved value with the layer and line that set it (R-RES-3)."""
    title = f"{resolved.model.name} ({resolved.ref})"
    lines = [title]
    if resolved.point is not None:
        lines.append(f"point: {resolved.point}")
    if resolved.layers:
        lines.append("layers: " + " < ".join(resolved.layers))
    rows: list[tuple[str, str, str]] = []
    prov = resolved.provenance
    entries = [*_leaves(resolved.data), *_defaults(resolved)]
    entries.sort(key=lambda entry: _order(resolved, entry[0]))
    for path, value in entries:
        origin = prov.leaves.get(path, TOOL)
        where = _where(origin)
        older = prov.superseded.get(path)
        if older:
            where += "  (over " + ", ".join(_where(o) for o in reversed(older)) + ")"
        rows.append((format_path(path), render_data(value), where))
    lines.append("")
    lines += _table(rows)
    if prov.deleted:
        lines.append("")
        lines.append("deleted:")
        deleted = [
            (format_path(path), "", _where(origin))
            for path, origin in prov.deleted.items()
        ]
        lines += ["  " + row for row in _table(deleted)]
    if resolved.overrides:
        lines.append("")
        lines.append("experiment, gates off: " + " ".join(resolved.overrides))
    return "\n".join(lines) + "\n"


_TOP = ("schema", "name", "extends", "server", "tasks")


def _order(resolved: Resolved, path: KeyPath) -> tuple[Any, ...]:
    """Data order: the model's field order, then keys as merged."""
    top = _TOP.index(path[0]) if path[0] in _TOP else len(_TOP)
    return (top, _position(resolved.data, path))


def _position(data: Any, path: KeyPath) -> tuple[int, ...]:
    out = []
    node = data
    for key in path:
        if isinstance(node, dict):
            keys = list(node)
            out.append(keys.index(key) if key in node else len(keys))
            node = node.get(key)
        elif isinstance(node, list) and isinstance(key, int):
            out.append(key)
            node = node[key] if key < len(node) else None
        else:
            out.append(10**6)
            node = None
    return tuple(out)


def _table(rows: Sequence[tuple[str, str, str]]) -> list[str]:
    if not rows:
        return []
    key_width = min(max(len(r[0]) for r in rows), 52)
    value_width = min(max(len(r[1]) for r in rows), 40)
    lines = []
    for key, value, where in rows:
        line = f"{key:<{key_width}}  {value:<{value_width}}  {where}".rstrip()
        lines.append(line)
    return lines


def render_why(resolved: Resolved, path: KeyPath) -> str:
    """Every layer that set `path`, newest first, or why it is unset."""
    prov = resolved.provenance
    name = format_path(path)
    lines: list[str] = []
    if path in prov.leaves:
        value = _get(resolved.data, path)
        lines.append(f"{name} = {render_data(value)}")
        origin = prov.leaves[path]
        lines.append(f"  set by   {_where(origin)}  ({origin.layer})")
        for older in reversed(prov.superseded.get(path, [])):
            lines.append(f"  over     {_where(older)}  ({older.layer})")
    elif _has(resolved.data, path):
        lines.append(f"{name}:")
        for leaf, value in _leaves(_get(resolved.data, path), path):
            origin = prov.leaves.get(leaf, TOOL)
            lines.append(
                f"  {format_path(leaf)} = {render_data(value)}  {_where(origin)}"
            )
    elif path in prov.deleted:
        origin = prov.deleted[path]
        lines.append(f"{name}: deleted by {_where(origin)}  ({origin.layer})")
    else:
        dumped = resolved.model.model_dump(
            by_alias=True, exclude_none=True, warnings=False
        )
        if _has(dumped, path):
            value = _get(dumped, path)
            lines.append(f"{name} = {render_data(value)}  (tool default)")
        else:
            lines.append(f"{name}: not set")
    return "\n".join(lines) + "\n"
