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
"""Targets: where a Plan runs, and the only contract between them and the tool.

plan() never sees a target, and a target sees the Plan, never the scenario
model; a target decides pods from Plan.shape() and the image alone. Env has
one owner per name: the tool owns what the Plan sets, a target owns what its
reserved_env() lists, and a scenario may set neither. With two owners,
precedence would decide the value, so a clash is an error instead.
"""

import dataclasses
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
    NamedTuple,
    Protocol,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

from scenario.plan.types import (
    AgentxTaskPlan,
    EvalTaskPlan,
    PerfTaskPlan,
    Plan,
    RoleShape,
    Shape,
    TaskPlan,
    task_commands,
)
from scenario.reserved import (
    AGENTX_CLIENT_ENV,
    CHIPLESS_DRIVER_ENV,
    INFERENCEX_EVAL_ENV,
    INFERENCEX_PERF_ENV,
    RAY_ENV,
    TOOL_ENV,
    TOOL_ENV_PREFIXES,
    tool_env_owner,
)

__all__ = [
    "TOOL_OWNED_ENV",
    "AfterTask",
    "EnvNames",
    "RoleShape",
    "Shape",
    "Target",
    "TargetData",
    "TargetInfo",
    "ownership_problems",
    "scenario_env_names",
]


class TargetInfo(NamedTuple):
    """What results record about the target that ran them."""

    name: str
    # Commit of the code that defines the target.
    commit: str | None
    # Where the runtime writes run directories; None for its own default.
    results_root: str | None


class EnvNames(NamedTuple):
    """A set of env names: exact names and name prefixes."""

    names: frozenset[str]
    prefixes: tuple[str, ...] = ()

    def covers(self, name: str) -> bool:
        return name in self.names or name.startswith(self.prefixes)


# Every name the Plan may set somewhere: on server hosts, in the driver's
# task env, or in one task command's env. A target's env reaches every process
# of its pods, so a target may reserve none of these. A scenario may still use
# the names the tool sets in one context only (the AgentX client's MODEL, say)
# everywhere else.
TOOL_OWNED_ENV = EnvNames(
    frozenset(
        {
            *TOOL_ENV,
            *RAY_ENV,
            *CHIPLESS_DRIVER_ENV,
            *AGENTX_CLIENT_ENV,
            *INFERENCEX_EVAL_ENV,
            *INFERENCEX_PERF_ENV,
        }
    ),
    TOOL_ENV_PREFIXES,
)


@runtime_checkable
class Target(Protocol):
    def info(self) -> TargetInfo:
        """Recorded beside the results."""
        ...

    def reserved_env(self) -> EnvNames:
        """The env names this target sets; a scenario may not set them."""
        ...

    def submit(self, plan: Plan) -> int:
        """Run the Plan; the exit code of its driver."""
        ...


@dataclass(frozen=True)
class AfterTask:
    """Work a target asks the driver host to do after results are written.

    Passed to pods as data, so pods never import target code. `when=result`
    runs after each measured perf cell whose mode matches `results`;
    `when=end` once, after teardown. Argv and env may use the placeholders
    {run_dir}, {file}, {label}, {unpublished}, {image} and {run_stamp}.
    """

    name: str
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    when: Literal["result", "end"] = "end"
    results: Literal["any", "product", "cells"] = "any"
    require: Literal["always", "result_ok", "commands_ok"] = "always"
    fatal: bool = True
    skip_on_experiment: bool = False
    timeout_s: int = 600

    def to_data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "env": [list(pair) for pair in self.env],
            "when": self.when,
            "results": self.results,
            "require": self.require,
            "fatal": self.fatal,
            "skip_on_experiment": self.skip_on_experiment,
            "timeout_s": self.timeout_s,
        }

    @classmethod
    def from_data(cls, data: Any) -> "AfterTask":
        """An AfterTask from to_data()'s form.

        Raises:
            ValueError: a missing or unknown field, or a value of the wrong kind.
        """
        if not isinstance(data, dict):
            raise ValueError(f"an after-task is a mapping, not {data!r}")
        fields = {f.name: f for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - set(fields))
        if unknown:
            raise ValueError(f"after-task field {unknown[0]} is unknown")
        hints = get_type_hints(cls)
        values: dict[str, Any] = {}
        for name, value in data.items():
            hint = hints[name]
            if name == "argv":
                value = tuple(_strings(value, name))
            elif name == "env":
                value = tuple(_env_pair(pair) for pair in _items(value, name))
            elif get_origin(hint) is Literal:
                if value not in get_args(hint):
                    choices = ", ".join(get_args(hint))
                    raise ValueError(
                        f"after-task {name} {value!r} is not one of {choices}"
                    )
            # bool is an int to isinstance; neither may stand for the other.
            elif type(value) is not hint:
                raise ValueError(
                    f"after-task {name} {value!r} is not a {hint.__name__}"
                )
            values[name] = value
        missing = [n for n in ("name", "argv") if n not in values]
        if missing:
            raise ValueError(f"after-task without {missing[0]}")
        return cls(**values)


def _items(value: Any, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"after-task {what} is a list, not {value!r}")
    return value


def _strings(value: Any, what: str) -> list[str]:
    items = _items(value, what)
    if not all(isinstance(item, str) for item in items):
        raise ValueError(f"after-task {what} holds a non-string: {value!r}")
    return items


def _env_pair(pair: Any) -> tuple[str, str]:
    if not (isinstance(pair, list) and len(pair) == 2):
        raise ValueError(f"after-task env entries are [name, value], not {pair!r}")
    name, value = _strings(pair, "env")
    return name, value


@dataclass(frozen=True)
class TargetData:
    """What a target hands the driver host beside the Plan.

    Pods get it as SCENARIO_TARGET_DATA (gzip and base64 of to_data()),
    built from info() and after_tasks(); the runtime reads nothing else of
    the target. `lm_eval_cache_root` is where lm_eval keeps its request
    cache; None runs it without one.
    """

    info: TargetInfo
    after_tasks: tuple[AfterTask, ...] = ()
    lm_eval_cache_root: str | None = None

    def to_data(self) -> dict[str, Any]:
        return {
            "info": {"name": self.info.name, "commit": self.info.commit},
            "results_root": self.info.results_root,
            "after_tasks": [task.to_data() for task in self.after_tasks],
            "lm_eval_cache_root": self.lm_eval_cache_root,
        }

    @classmethod
    def from_data(cls, data: Any) -> "TargetData":
        """TargetData from to_data()'s form.

        Raises:
            ValueError: the data has another shape.
        """
        if not isinstance(data, dict):
            raise ValueError(f"target data is a mapping, not {data!r}")
        unknown = sorted(
            set(data) - {"info", "results_root", "after_tasks", "lm_eval_cache_root"}
        )
        if unknown:
            raise ValueError(f"target data field {unknown[0]} is unknown")
        info = data.get("info")
        if not isinstance(info, dict) or not isinstance(info.get("name"), str):
            raise ValueError(f"target data info needs a name: {info!r}")
        optional = {
            "info.commit": info.get("commit"),
            "results_root": data.get("results_root"),
            "lm_eval_cache_root": data.get("lm_eval_cache_root"),
        }
        for what, value in optional.items():
            if value is not None and not isinstance(value, str):
                raise ValueError(f"target data {what} is a string, not {value!r}")
        return cls(
            info=TargetInfo(info["name"], info.get("commit"), data.get("results_root")),
            after_tasks=tuple(
                AfterTask.from_data(task)
                for task in _items(data.get("after_tasks", []), "list")
            ),
            lm_eval_cache_root=data.get("lm_eval_cache_root"),
        )


def _task_tool_env(task: TaskPlan) -> frozenset[str]:
    """Names the tool sets in this task's commands, beyond the shared ones."""
    if isinstance(task, AgentxTaskPlan):
        return frozenset(AGENTX_CLIENT_ENV)
    if isinstance(task, PerfTaskPlan) and task.client == "inferencex":
        return frozenset(INFERENCEX_PERF_ENV)
    if isinstance(task, EvalTaskPlan) and task.client == "inferencex":
        return frozenset(INFERENCEX_EVAL_ENV)
    return frozenset()


def scenario_env_names(plan: Plan) -> frozenset[str]:
    """The env names the scenario sets in a Plan, on hosts and in tasks."""
    names: set[str] = set()
    for role in plan.roles:
        for host in role.host_plans:
            names.update(name for name, _ in host.scenario_env)
    for task in plan.driver.tasks:
        tool = _task_tool_env(task)
        for _, command in task_commands(task):
            names.update(
                name
                for name, _ in command.env
                if name not in tool and tool_env_owner(name) is None
            )
    return frozenset(names)


def ownership_problems(plan: Plan, reserved: EnvNames) -> list[str]:
    """Env names claimed by a target and by the tool or the scenario."""
    problems = [
        f"the target reserves {name}, which the tool sets"
        for name in sorted(reserved.names)
        if TOOL_OWNED_ENV.covers(name)
    ]
    problems += [
        f"the target reserves the prefix {prefix}, which overlaps the tool's "
        f"{', '.join(TOOL_OWNED_ENV.prefixes)}"
        for prefix in reserved.prefixes
        if any(
            prefix.startswith(own) or own.startswith(prefix)
            for own in TOOL_OWNED_ENV.prefixes
        )
    ]
    problems += [
        f"the scenario sets {name}, which the target sets; remove it from the scenario"
        for name in sorted(scenario_env_names(plan))
        if reserved.covers(name)
    ]
    return problems
