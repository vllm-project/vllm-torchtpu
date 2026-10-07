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

from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol, runtime_checkable

from scenario.plan.types import (
    AgentxTaskPlan,
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


def _task_tool_env(task: TaskPlan) -> frozenset[str]:
    """Names the tool sets in this task's commands, beyond the shared ones."""
    if isinstance(task, AgentxTaskPlan):
        return frozenset(AGENTX_CLIENT_ENV)
    if isinstance(task, PerfTaskPlan) and task.client == "inferencex":
        return frozenset(INFERENCEX_PERF_ENV)
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
