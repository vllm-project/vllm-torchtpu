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
"""plan(): one resolved point to a Plan.

plan() reads nothing: a baseline `from:` another scenario was copied in by
resolution. It sees no target, so a target cannot change what hosts run.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from scenario import PLAN_FORMAT, __version__
from scenario.model.baseline import BaselineRef
from scenario.model.scenario import Scenario
from scenario.model.server import RoleView, Server
from scenario.plan.plumbing import (
    accelerator_env,
    advertise_env,
    backend,
    backend_env,
    chipless_env,
    custom_env,
    endpoint,
    headless,
    kv_wiring,
    mp_dp_wiring,
    raiden_identity,
    ray_start,
    remote_code_warm,
    role_hardware,
    router_plan,
    serve_argv,
    startup,
)
from scenario.plan.tasks import agentx_task, custom_task, eval_task, perf_task
from scenario.plan.types import (
    Command,
    DriverPlan,
    GatePlan,
    HostPlan,
    KvPlan,
    Metrics,
    Plan,
    RolePlan,
    TaskPlan,
    Value,
    env_items,
)
from scenario.resolve import RefBaseline, Resolved


class PlanError(ValueError):
    """A Plan that cannot be built from what was asked."""


def select_point(points: Sequence[Resolved], point: str | None = None) -> Resolved:
    """The one point to plan; a sweep needs `point` named.

    Raises:
        PlanError: a sweep without `point`, or a `point` it does not have.
    """
    if point is not None:
        for resolved in points:
            if resolved.point == point:
                return resolved
    elif len(points) == 1:
        return points[0]
    names = ", ".join(str(resolved.point) for resolved in points)
    ref = points[0].ref if points else "the scenario"
    if point is None:
        raise PlanError(
            f"{ref} is a sweep of {len(points)} points, each its own workload; "
            f"pick one with --point: {names}"
        )
    raise PlanError(f"{ref} has no point {point}; points: {names}")


def plan(
    resolved: Resolved,
    *,
    only: Sequence[str] | None = None,
    endpoint: tuple[str, int] | None = None,
    tool_version: str = __version__,
) -> Plan:
    """The Plan of one resolved point.

    `endpoint` (host, port) points the tasks at a server that is already
    running, for `perf --url` and `eval --url`; by default they go to the
    servers the Plan starts.

    Raises:
        PlanError: `only` names a task the point does not have.
    """
    return build_plan(
        resolved.model,
        ref=resolved.ref,
        point=resolved.point,
        overrides=resolved.overrides,
        ref_baselines=resolved.ref_baselines,
        only=only,
        endpoint=endpoint,
        tool_version=tool_version,
    )


def resolved_payload(resolved: Resolved) -> dict[str, Any]:
    """What pods get as SCENARIO_RESOLVED: the point's merged scenario data."""
    return {"point": resolved.point, "scenario": resolved.data}


def build_plan(
    scenario: Scenario,
    *,
    ref: str,
    point: str | None = None,
    overrides: Sequence[str] = (),
    ref_baselines: Mapping[str, RefBaseline] | None = None,
    only: Sequence[str] | None = None,
    endpoint: tuple[str, int] | None = None,
    tool_version: str = __version__,
) -> Plan:
    """The Plan of a validated scenario point; see plan()."""
    server = scenario.server
    views = server.role_views()
    multi_role = len(views) > 1
    roles = tuple(
        _role_plan(server, view, position, multi_role)
        for position, view in enumerate(views)
    )
    gates = _Gates(scenario, bool(overrides), ref_baselines or {})
    driver = _driver(scenario, views, roles, gates, only, endpoint)
    return Plan(
        format=PLAN_FORMAT,
        tool_version=tool_version,
        scenario_ref=ref,
        scenario_name=scenario.name,
        point=point,
        image=server.image,
        model_name=server.model.name,
        model_path=server.model.load_path,
        roles=roles,
        driver=driver,
        experiment=tuple(overrides),
        only=tuple(only) if only is not None else None,
    )


def _role_plan(
    server: Server, view: RoleView, position: int, multi_role: bool
) -> RolePlan:
    hardware = role_hardware(view)
    role_backend = backend(hardware.hosts, server.multihost)
    args = dict(view.args)
    ready_env, readiness = startup(server.startup_timeout)
    env: dict[str, Value] = {
        **backend_env(role_backend, hardware.hosts, hardware.topology, multi_role),
        **advertise_env(hardware.hosts, view.kv is not None),
        **ready_env,
        **accelerator_env(hardware.accelerator),
    }
    kv_plan: KvPlan | None = None
    kv_flags: tuple[Value, ...] = ()
    if view.kv is not None:
        kv_plan, kv_flags, kv_env = kv_wiring(view.kv)
        env |= kv_env | raiden_identity(view.name, position)
    scenario_env = env_items(dict(view.env))
    hosts = []
    for index in range(hardware.hosts):
        pre: list[Command] = []
        host_args: dict[str, Any] = args
        wiring: list[Value] = list(kv_flags)
        dropped: tuple[str, ...] = ()
        if role_backend == "mp":
            dp_local, dp_flags = mp_dp_wiring(
                args.get("data-parallel-size", 1), hardware.hosts
            )
            wiring += dp_flags
            if index > 0:
                host_args, flags, dropped = headless(index, dp_local, args)
                wiring += flags
            pre += remote_code_warm(role_backend, index, args, server.model)
        elif role_backend == "ray":
            pre.append(ray_start(index))
        serve = None
        if role_backend != "ray" or index == 0:
            serve = Command(serve_argv(server.model, host_args, wiring, api=index == 0))
        hosts.append(
            HostPlan(
                index=index,
                head=index == 0,
                pre=tuple(pre),
                serve=serve,
                dropped_flags=dropped,
                scenario_env=scenario_env,
                tool_env=env_items(env),
            )
        )
    return RolePlan(
        name=view.name,
        accelerator=hardware.accelerator,
        topology=hardware.topology,
        hosts=hardware.hosts,
        chips_per_host=hardware.chips_per_host,
        backend=role_backend,
        readiness=readiness,
        kv=kv_plan,
        scratch_bytes=server.scratch,
        cache="shared",
        host_plans=tuple(hosts),
    )


def _freeze(
    baseline: Mapping[str, Mapping[str, float]],
) -> tuple[tuple[str, Metrics], ...]:
    return tuple(
        (key, tuple(sorted(metrics.items())))
        for key, metrics in sorted(baseline.items())
    )


class _Gates:
    """Each task's tolerance and baseline; none for an experiment."""

    def __init__(
        self,
        scenario: Scenario,
        experiment: bool,
        ref_baselines: Mapping[str, RefBaseline],
    ) -> None:
        self.scenario = scenario
        self.experiment = experiment
        self.ref_baselines = ref_baselines

    def baseline(self, task: str) -> tuple[dict[str, Any], str] | None:
        block = getattr(self.scenario.tasks, task)
        if self.experiment or block is None or block.baseline is None:
            return None
        if isinstance(block.baseline, BaselineRef):
            copied = self.ref_baselines[task]
            data, source = dict(copied.baseline), f"from:{copied.source}"
        else:
            data = block.baseline.model_dump(by_alias=True, exclude_none=True)
            source = "inline"
        # `_`-keys document a baseline; the gate ignores them.
        return {k: v for k, v in data.items() if not k.startswith("_")}, source

    def keyed(self, task: str) -> GatePlan | None:
        found = self.baseline(task)
        if found is None:
            return None
        data, source = found
        tolerance = getattr(self.scenario.tasks, task).tolerance
        return GatePlan(tolerance, _freeze(data), source)

    def eval_runs(self) -> dict[str, GatePlan]:
        gate = self.keyed("eval")
        if gate is None:
            return {}
        return {
            key: GatePlan(gate.tolerance, ((key, metrics),), gate.source)
            for key, metrics in gate.baseline
        }


def _driver(
    scenario: Scenario,
    views: Sequence[RoleView],
    roles: Sequence[RolePlan],
    gates: _Gates,
    only: Sequence[str] | None,
    given: tuple[str, int] | None = None,
) -> DriverPlan:
    server, tasks = scenario.server, scenario.tasks
    names = [view.name for view in views]
    if len(views) > 1:
        chipless = tasks.driver == "client"
        driver_role = None if chipless else tasks.driver
    else:
        chipless, driver_role = False, names[0]
    kv_roles = {view.name: view.kv.role for view in views if view.kv is not None}
    router = None
    if server.router is not None:
        producer = next(name for name, kv in kv_roles.items() if kv == "producer")
        consumer = next(name for name, kv in kv_roles.items() if kv == "consumer")
        router = router_plan(server.router, producer, consumer)
    host: Value
    host, port = given or endpoint(names, driver_role, router is not None)

    task_env: dict[str, Value] = {"PYTHONUNBUFFERED": "1"}
    if driver_role is not None:
        role_backend = next(r.backend for r in roles if r.name == driver_role)
        if role_backend != "single":
            task_env["TPU_MULTIHOST_BACKEND"] = role_backend
    if chipless:
        task_env |= chipless_env()

    built: list[tuple[str, TaskPlan]] = []
    if tasks.perf is not None:
        gate = gates.keyed("perf")
        built.append(("perf", perf_task(tasks.perf, server.model, host, port, gate)))
    if tasks.agentx is not None:
        if tasks.inferencex is None:
            raise PlanError("tasks.agentx needs tasks.inferencex")
        try:
            agentx = agentx_task(
                tasks.agentx, tasks.inferencex, server.model, views, host, port
            )
        except ValueError as error:
            raise PlanError(str(error)) from None
        built.append(("agentx", agentx))
    provided = custom_env(
        server.model,
        names,
        kv_roles,
        router=router is not None,
        driver_on_role=driver_role is not None,
    )
    for task in tasks.custom:
        built.append((task.name, custom_task(task, provided)))
    if tasks.eval is not None:
        try:
            runs = eval_task(
                tasks.eval,
                tasks.inferencex,
                server.model,
                views,
                host,
                port,
                gates.eval_runs(),
            )
        except ValueError as error:
            raise PlanError(str(error)) from None
        built.append(("eval", runs))
    if only is not None:
        available = [name for name, _ in built]
        unknown = [name for name in only if name not in available]
        if unknown:
            raise PlanError(
                f"--only names {', '.join(unknown)}, which this scenario does not "
                f"run; tasks: {', '.join(available) or 'none'}"
            )
        built = [(name, task) for name, task in built if name in only]

    return DriverPlan(
        placement="client" if chipless else "role",
        role=driver_role,
        chipless=chipless,
        router=router,
        endpoint_host=host,
        endpoint_port=port,
        task_env=env_items(task_env),
        tasks=tuple(task for _, task in built),
    )
