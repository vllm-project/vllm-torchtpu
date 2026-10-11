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
"""A test target that hands the Plan to pods, and the invariance check over it.

The target passes the Plan to pods as a Kubernetes target would, encoded in
env beside env of its own; each pod decodes it and renders its own part.
That part must equal what `run --dry-run` prints with no target at all.
test_target.py runs the check over the fixture repository and the test
variants; like plan_fixtures.py it is imported by name, so checks over a
repository's own scenarios can run it too.
"""

from dataclasses import dataclass, field

from scenario.plan import codec
from scenario.plan.types import CLIENT_ROLE, Plan
from scenario.show import render_plan
from scenario.target import EnvNames, TargetInfo, ownership_problems


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


def check_pods_run_the_plan(name: str, planned: Plan) -> None:
    """Every pod of `planned` runs what the Plan says; `name` labels failures."""
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
    assert codec.decode(target.pods[0].env["SCENARIO_PLAN"]).driver == planned.driver
