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
"""The Target protocol, and invariance: a target changes nothing hosts run.

The test target hands the Plan to pods as a Kubernetes target would, encoded
in env beside env of its own; each pod decodes it and renders its own part.
That part must equal what `run --dry-run` prints with no target at all.
"""

import inspect
import pathlib
from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest
from plan_fixtures import VARIANTS, Data, build
from scenario.plan import codec
from scenario.plan.build import plan
from scenario.plan.types import CLIENT_ROLE, Plan
from scenario.resolve import Resolver
from scenario.show import render_plan
from scenario.target import (
    TOOL_OWNED_ENV,
    AfterTask,
    EnvNames,
    Target,
    TargetInfo,
    ownership_problems,
    scenario_env_names,
)

TESTS = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TESTS.parents[2]
ROOTS = [TESTS / "data" / "repo", REPO_ROOT]


@dataclass
class Pod:
    role: str
    index: int
    env: dict[str, str]


@dataclass
class PodTarget:
    """Renders pods from the Shape and passes the Plan as an env value."""

    reserved: EnvNames = EnvNames(
        frozenset({"FORCE_COLOR", "TQDM_MININTERVAL", "VLLM_CACHE_ROOT"}), ("TEST_",)
    )
    pods: list[Pod] = field(default_factory=list)

    def info(self) -> TargetInfo:
        return TargetInfo("pods", None, "results")

    def reserved_env(self) -> EnvNames:
        return self.reserved

    def submit(self, plan: Plan) -> int:
        problems = ownership_problems(plan, self.reserved)
        if problems:
            raise ValueError("; ".join(problems))
        encoded = codec.encode(plan)
        for role in plan.shape().roles:
            for index in range(role.hosts):
                env = {
                    "SCENARIO_PLAN": encoded,
                    "SCENARIO_HOST_INDEX": str(index),
                    "FORCE_COLOR": "1",
                    "TEST_POD": f"{role.name}-{index}",
                }
                if role.multi_role and not role.chipless:
                    env["VLLM_CACHE_ROOT"] = f"/scratch/{role.name}"
                self.pods.append(Pod(role.name, index, env))
        return 0


def pod_view(pod: Pod) -> str:
    """What a pod runs, from nothing but its own env."""
    return render_plan(codec.decode(pod.env["SCENARIO_PLAN"]), pod.role, pod.index)


def scenario_plans() -> Iterator[tuple[str, Plan]]:
    """Every point of every scenario: the fixture repo, then scenarios/."""
    for root in ROOTS:
        if not (root / "scenarios" / "defaults.yaml").is_file():
            continue
        resolver = Resolver(root)
        for file in resolver.scenario_files():
            for resolved in resolver.resolve(file):
                name = f"{resolver.ref_of(file)}:{resolved.point or ''}"
                yield name, plan(resolved)


def all_plans(variants: dict[str, Data]) -> list[tuple[str, Plan]]:
    found = list(scenario_plans())
    found += [(f"variant:{name}", build(data)) for name, data in variants.items()]
    return found


def test_protocol() -> None:
    assert isinstance(PodTarget(), Target)
    assert not isinstance(object(), Target)


def test_plan_takes_no_target() -> None:
    assert list(inspect.signature(plan).parameters) == [
        "resolved",
        "only",
        "tool_version",
    ]


def test_every_pod_runs_what_the_plan_says(variants: dict[str, Data]) -> None:
    plans = all_plans(variants)
    assert len(plans) >= len(VARIANTS) + 7
    for name, planned in plans:
        before = codec.to_json(planned)
        target = PodTarget()
        assert target.submit(planned) == 0
        assert codec.to_json(planned) == before, name
        shape = planned.shape()
        assert [(p.role, p.index) for p in target.pods] == [
            (role.name, index) for role in shape.roles for index in range(role.hosts)
        ], name
        for pod in target.pods:
            decoded = codec.decode(pod.env["SCENARIO_PLAN"])
            assert decoded == planned, name
            assert pod_view(pod) == render_plan(planned, pod.role, pod.index), name
            if pod.role == CLIENT_ROLE:
                continue
            host = planned.role(pod.role).host_plans[pod.index]
            mine = decoded.role(pod.role).host_plans[pod.index]
            assert (mine.serve, mine.scenario_env, mine.tool_env) == (
                host.serve,
                host.scenario_env,
                host.tool_env,
            ), name
        assert (
            codec.decode(target.pods[0].env["SCENARIO_PLAN"]).driver == planned.driver
        )


def test_target_env_never_reaches_the_plan(variants: dict[str, Data]) -> None:
    planned = build(variants["k3_pd"])
    target = PodTarget()
    target.submit(planned)
    for pod in target.pods:
        decoded = codec.decode(pod.env["SCENARIO_PLAN"])
        text = codec.to_json(decoded)
        assert "/scratch/" not in text and "FORCE_COLOR" not in text


def test_scenario_env_names(variants: dict[str, Data]) -> None:
    names = scenario_env_names(build(variants["k3_pd"]))
    assert names == {
        "JAX_PLATFORMS",
        "TPU_KV_SHM_POOL_GB",
        "LOG_DIR",
        "COST",
    }
    # JAX_PLATFORMS is the tool's only on Ray engines; these are not.
    assert TOOL_OWNED_ENV.covers("JAX_PLATFORMS")


def test_agentx_names_are_scenario_env_outside_agentx(
    variants: dict[str, Data],
) -> None:
    """MODEL is the tool's on the AgentX command only, so a custom task may set it."""
    data = variants["pd"]
    data["server"]["env"]["CONC"] = "4"
    data["tasks"]["custom"][0]["env"] |= {"MODEL": "m", "TP": "8"}
    planned = build(data)
    assert {"CONC", "MODEL", "TP"} <= scenario_env_names(planned)
    assert ownership_problems(planned, PodTarget().reserved) == []
    greedy = EnvNames(frozenset({"MODEL"}))
    assert ownership_problems(planned, greedy)[0] == (
        "the target reserves MODEL, which the tool sets"
    )
    agentx = build(variants["agentx"])
    assert "MODEL" not in scenario_env_names(agentx)
    assert "AIPERF_MAX_CONTEXT_LENGTH" in scenario_env_names(agentx)


def test_ownership_problems(variants: dict[str, Data]) -> None:
    planned = build(variants["pd"])
    assert ownership_problems(planned, PodTarget().reserved) == []
    greedy = EnvNames(
        frozenset({"VLLM_HOST_IP", "TPU_KV_SHM_POOL_GB", "LOG_DIR"}), ("SCENARIO_X",)
    )
    assert ownership_problems(planned, greedy) == [
        "the target reserves VLLM_HOST_IP, which the tool sets",
        "the target reserves the prefix SCENARIO_X, which overlaps the tool's "
        "SCENARIO_",
        "the scenario sets LOG_DIR, which the target sets; remove it from the scenario",
        "the scenario sets TPU_KV_SHM_POOL_GB, which the target sets; remove it "
        "from the scenario",
    ]
    with pytest.raises(ValueError, match="VLLM_HOST_IP"):
        PodTarget(reserved=greedy).submit(planned)


def test_tool_owned_env_covers_what_plans_set(variants: dict[str, Data]) -> None:
    for _, planned in all_plans(variants):
        for role in planned.roles:
            for host in role.host_plans:
                assert all(TOOL_OWNED_ENV.covers(name) for name, _ in host.tool_env)
        assert all(TOOL_OWNED_ENV.covers(name) for name, _ in planned.driver.task_env)


def test_after_task_defaults() -> None:
    task = AfterTask("artifacts", ("upload", "{file}"), when="result")
    assert (task.require, task.fatal, task.timeout_s, task.results) == (
        "always",
        True,
        600,
        "any",
    )
