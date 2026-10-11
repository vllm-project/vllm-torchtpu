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
"""Plan types: the contract between scenario code, targets and the runtime.

Every type is a frozen dataclass and mappings are sorted tuples of pairs, so
a Plan is hashable and has exactly one JSON form (plan/codec.py). A Plan
describes one point of a scenario: a sweep point is its own workload.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

# Ports are plumbing: fixed by the tool, never chosen by a scenario.
API_PORT = 8000
ROUTER_PORT = 8100
STATUS_PORT = 8090
DP_RPC_PORT = 29500
RAY_PORT = 6379
KV_PORT = 14579
KV_TRANSFER_PORT = 9100
SIDE_CHANNEL_PORT = 9600

# The readiness wait runs this long past the engine's own startup timeout, so
# the engine reports its own failure before the wait gives up.
READINESS_MARGIN_S = 600
HEALTH_PATH = "/health"

RefKind = Literal[
    "self_ip",
    "head_ip",
    "host_index",
    "role_head",
    "role_ips",
    "results_dir",
    "pulled_model_dir",
    "server_log",
    "image",
    "libtpu_block",
]
Backend = Literal["single", "mp", "ray"]


@dataclass(frozen=True, slots=True)
class Ref:
    """A value only the host knows at run time.

    self_ip: this host's advertise address. head_ip: the role's first host,
    as IPv4 (its own advertise address on that host). host_index: this
    host's index. role_head: the first host of `role` by name, or 127.0.0.1
    on that host itself. role_ips: every host of `role` as IPv4, joined by
    `sep`. results_dir: the run directory. pulled_model_dir: where the
    server pulled an object-storage model, or `fallback` when it holds no
    config.json. server_log: the driver host's own server log.
    image: the image the pods run, as the target set it, else the Plan's;
    empty when neither names one.
    libtpu_block: the tool's directory whose libtpu cannot be imported,
    ahead of the inherited PYTHONPATH.
    """

    kind: RefKind
    role: str | None = None
    sep: str = ","
    fallback: str | None = None


@dataclass(frozen=True, slots=True)
class Concat:
    """Text made of literal parts and Refs."""

    parts: tuple["str | Ref", ...]


Value = str | Ref | Concat
EnvItems = tuple[tuple[str, Value], ...]


def concat(*parts: Value) -> Value:
    """`parts` joined, with adjacent text merged; plain text when nothing is a Ref."""
    flat: list[str | Ref] = []
    for part in parts:
        items = part.parts if isinstance(part, Concat) else (part,)
        for item in items:
            if isinstance(item, str) and flat and isinstance(flat[-1], str):
                flat[-1] += item
            elif item != "":
                flat.append(item)
    if not flat:
        return ""
    if len(flat) == 1:
        return flat[0]
    return Concat(tuple(flat))


def env_items(env: dict[str, Value]) -> EnvItems:
    return tuple(sorted(env.items()))


@dataclass(frozen=True, slots=True)
class Command:
    """One process: argv, the env it adds, and how it may end."""

    argv: tuple[Value, ...]
    env: EnvItems = ()
    timeout_s: int | None = None
    # A failure is logged and does not count (the remote-code warm).
    best_effort: bool = False


@dataclass(frozen=True, slots=True)
class Readiness:
    path: str
    port: int
    # startup_timeout + READINESS_MARGIN_S; show and the runtime use this one value.
    timeout_s: int


@dataclass(frozen=True, slots=True)
class HostPlan:
    index: int
    head: bool
    # Run in order before serve: ray start, the remote-code warm.
    pre: tuple[Command, ...]
    # None on Ray workers, which serve through the head.
    serve: Command | None
    # Frontend-only flags left off a headless host.
    dropped_flags: tuple[str, ...]
    # server.env and the role's env, as written.
    scenario_env: EnvItems
    # Plumbing env; applies to pre and serve.
    tool_env: EnvItems


@dataclass(frozen=True, slots=True)
class KvPlan:
    connector: str
    kv_role: Literal["kv_producer", "kv_consumer", "kv_both"]
    kv_port: int


@dataclass(frozen=True, slots=True)
class RolePlan:
    name: str
    accelerator: str
    topology: str
    hosts: int
    chips_per_host: int
    backend: Backend
    readiness: Readiness
    kv: KvPlan | None
    scratch_bytes: int | None
    cache: Literal["shared", "private"]
    host_plans: tuple[HostPlan, ...]

    @property
    def chips(self) -> int:
        return self.hosts * self.chips_per_host


@dataclass(frozen=True, slots=True)
class RouterPlan:
    """The proxy on the driver host; it runs with the driver's task env."""

    argv: tuple[Value, ...]
    port: int
    health_path: str


Metrics = tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class GatePlan:
    """A gate's tolerance and baseline: key -> metric -> value.

    Keys are cell keys for perf and run keys for eval.
    """

    tolerance: float
    baseline: tuple[tuple[str, Metrics], ...]
    # "inline", or "from:<ref>" for a baseline copied from another scenario.
    source: str
    soft: bool = False

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {key: dict(metrics) for key, metrics in self.baseline}


@dataclass(frozen=True, slots=True)
class PerfCellPlan:
    key: str
    warmups: tuple[Command, ...]
    measured: Command
    result_file: Value


@dataclass(frozen=True, slots=True)
class PerfTaskPlan:
    client: Literal["vllm-bench", "inferencex"]
    mode: Literal["product", "cells"]
    cells: tuple[PerfCellPlan, ...]
    gate: GatePlan | None
    pause_s: int = 0
    max_failed_requests: int | None = None
    kind: Literal["perf"] = "perf"


@dataclass(frozen=True, slots=True)
class AgentxTaskPlan:
    command: Command
    timeout_s: int
    kind: Literal["agentx"] = "agentx"


@dataclass(frozen=True, slots=True)
class EvalRunPlan:
    # The baseline key: the task, with _c<N> in a concurrency sweep. For
    # InferenceX's eval client, its suite, else its framework.
    key: str
    task: str
    tool: Literal["lm_eval", "evalscope", "inferencex"]
    command: Command
    # A gate against a baseline.
    gate: GatePlan | None
    # A gate against an absolute floor: InferenceX's score validation, run
    # after the eval; its exit code is the verdict.
    floor: Command | None = None


@dataclass(frozen=True, slots=True)
class EvalTaskPlan:
    runs: tuple[EvalRunPlan, ...]
    client: Literal["lm_eval", "inferencex"] = "lm_eval"
    kind: Literal["eval"] = "eval"


@dataclass(frozen=True, slots=True)
class CustomTaskPlan:
    name: str
    command: Command
    timeout_s: int
    skip_rest_if_exit_ge: int | None
    kind: Literal["custom"] = "custom"


TaskPlan = PerfTaskPlan | AgentxTaskPlan | CustomTaskPlan | EvalTaskPlan
TASK_KINDS = ("perf", "agentx", "custom", "eval")


def task_commands(task: TaskPlan) -> Iterator[tuple[str, Command]]:
    """A task's commands in run order, each with a label."""
    if isinstance(task, PerfTaskPlan):
        for cell in task.cells:
            for index, command in enumerate(cell.warmups):
                yield f"perf {cell.key} warmup{index + 1}", command
            yield f"perf {cell.key}", cell.measured
    elif isinstance(task, AgentxTaskPlan):
        yield "agentx", task.command
    elif isinstance(task, CustomTaskPlan):
        yield f"custom {task.name}", task.command
    else:
        for run in task.runs:
            yield f"eval {run.key}", run.command
            if run.floor is not None:
                yield f"eval {run.key} floor", run.floor


@dataclass(frozen=True, slots=True)
class DriverPlan:
    """Where tasks and the router run, and what they connect to."""

    placement: Literal["role", "client"]
    # The role whose first host drives; None for a chipless client.
    role: str | None
    chipless: bool
    router: RouterPlan | None
    endpoint_host: Value
    endpoint_port: int
    task_env: EnvItems
    # perf, agentx, custom in list order, eval.
    tasks: tuple[TaskPlan, ...]


@dataclass(frozen=True, slots=True)
class RoleShape:
    """What a target may know about one role's pods."""

    name: str
    accelerator: str | None
    topology: str | None
    hosts: int
    chips_per_host: int
    backend: Backend | None
    scratch_bytes: int | None
    chipless: bool
    driver: bool
    multi_role: bool
    cache: Literal["shared", "private"]
    ports: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Shape:
    """The part of a Plan that decides pods; targets read nothing else of it."""

    # None: the target supplies the image.
    image: str | None
    roles: tuple[RoleShape, ...]


CLIENT_ROLE = "client"


@dataclass(frozen=True, slots=True)
class Plan:
    format: int
    tool_version: str
    scenario_ref: str
    scenario_name: str
    # The sweep point this Plan runs, or None without a sweep.
    point: str | None
    # A full image URI, or None for the target's own image.
    image: str | None
    model_name: str
    model_path: str
    roles: tuple[RolePlan, ...]
    driver: DriverPlan
    # The --set strings; empty when the run is not an experiment.
    experiment: tuple[str, ...]
    only: tuple[str, ...] | None

    def role(self, name: str) -> RolePlan:
        for role in self.roles:
            if role.name == name:
                return role
        raise KeyError(name)

    def has_followers(self) -> bool:
        """Whether any host other than the driver's waits on the driver."""
        return (
            len(self.roles) > 1
            or self.driver.chipless
            or any(role.hosts > 1 for role in self.roles)
        )

    def driver_ports(self) -> tuple[int, ...]:
        ports = []
        if self.driver.router is not None:
            ports.append(self.driver.router.port)
        if self.has_followers():
            ports.append(STATUS_PORT)
        return tuple(ports)

    def shape(self) -> Shape:
        multi_role = len(self.roles) > 1
        roles = []
        for role in self.roles:
            driver = self.driver.placement == "role" and self.driver.role == role.name
            extra = self.driver_ports() if driver else ()
            roles.append(
                RoleShape(
                    name=role.name,
                    accelerator=role.accelerator,
                    topology=role.topology,
                    hosts=role.hosts,
                    chips_per_host=role.chips_per_host,
                    backend=role.backend,
                    scratch_bytes=role.scratch_bytes,
                    chipless=False,
                    driver=driver,
                    multi_role=multi_role,
                    cache=role.cache,
                    ports=(role.readiness.port, *extra),
                )
            )
        if self.driver.chipless:
            roles.append(
                RoleShape(
                    name=CLIENT_ROLE,
                    accelerator=None,
                    topology=None,
                    hosts=1,
                    chips_per_host=0,
                    backend=None,
                    scratch_bytes=None,
                    chipless=True,
                    driver=True,
                    multi_role=multi_role,
                    cache="shared",
                    ports=self.driver_ports(),
                )
            )
        return Shape(image=self.image, roles=tuple(roles))

    def commands(self) -> Iterator[tuple[str, Command]]:
        """Every command with a label: hosts' pre and serve, then the tasks."""
        for role in self.roles:
            for host in role.host_plans:
                where = f"{role.name}/{host.index}"
                for index, command in enumerate(host.pre):
                    yield f"{where} pre[{index}]", command
                if host.serve is not None:
                    yield f"{where} serve", host.serve
        for task in self.driver.tasks:
            yield from task_commands(task)
