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

The test target (pod_target.py) hands the Plan to pods as a Kubernetes target
would, encoded in env beside env of its own; each pod decodes it and renders
its own part. That part must equal what `run --dry-run` prints with no target
at all.
"""

import inspect
import pathlib
from collections.abc import Iterator

import pytest
from plan_fixtures import VARIANTS, Data, build
from pod_target import PodTarget, check_pods_run_the_plan
from scenario.plan import codec
from scenario.plan.build import plan
from scenario.plan.types import Plan
from scenario.resolve import Resolver
from scenario.target import (
    TOOL_OWNED_ENV,
    AfterTask,
    EnvNames,
    Target,
    ownership_problems,
    scenario_env_names,
)

FIXTURE_REPO = pathlib.Path(__file__).resolve().parent / "data" / "repo"


def scenario_plans() -> Iterator[tuple[str, Plan]]:
    """Every point of every scenario in the fixture repository."""
    resolver = Resolver(FIXTURE_REPO)
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
        check_pods_run_the_plan(name, planned)


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
    """MODEL is the tool's on InferenceX's clients only, so a custom task may set it."""
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
    assert "AGENTIC_WARMUP_GRACE_PERIOD" in scenario_env_names(agentx)
    evals = build(variants["ix_eval"])
    assert scenario_env_names(evals) == {
        "JAX_PLATFORMS",
        "TPU_KV_SHM_POOL_GB",
        "EVAL_LIMIT",
    }


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
