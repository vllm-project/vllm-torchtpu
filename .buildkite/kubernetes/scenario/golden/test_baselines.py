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
"""Inline baselines against today's baseline files (R-MIG-2).

The files under scripts/vllm/benchmarking/baselines gate today's flow, and
inline baselines start as copies of them. This check lasts only while bare
metal keeps those files.
"""

import json
import warnings
from collections.abc import Iterator

import pytest
from golden_capture import REPO, SCENARIO_ROOT
from scenario.plan.build import plan
from scenario.plan.types import EvalTaskPlan, GatePlan, PerfTaskPlan, Plan
from scenario.resolve import Resolved, Resolver

BASELINES = REPO / "scripts" / "vllm" / "benchmarking" / "baselines"

# The config whose baseline files gate each scenario today
# (run_eval_flow.sh --baseline-config, else --config), or None for a
# scenario with no such config. Inline baselines start as copies of those
# files (R-MIG-2), and a scenario gates exactly what the files gate.
TODAY_BASELINE_CONFIG = {
    "qwen3-0.6b/tp1": "qwen3-0.6b-tp1",
    "glm-5.2/fp8-dp16-ep": "glm-5.2-fp8-dp16-ep",
    "glm-5.2/fp8-dp16-ep-mp": "glm-5.2-fp8-dp16-ep",
    "qwen3.5-397b/fp8-pcp8-dp8-1p1d": None,
    "qwen3.5-397b/inferencex-sweep": None,
    "kimi-k3/tp32-ep": "kimi-k3-tp32-ep",
    "kimi-k3/tp8-dp4-ep-1p1d": "kimi-k3-tp32-ep",
    "kimi-k3/agentx": "kimi-k3-tp32-ep-agentx",
    "kimi-k3/agentx-1p1d": "kimi-k3-tp8-dp4-ep-agentx",
    "kimi-k3/eval-conc-sweep": "kimi-k3-tp32-ep-concsweep",
}


def every_point() -> Iterator[tuple[str, Resolved]]:
    found = Resolver(SCENARIO_ROOT)
    for file in found.scenario_files():
        for resolved in found.resolve(file):
            yield found.ref_of(file), resolved


POINTS = list(every_point())


def point_id(ref: str, resolved: Resolved) -> str:
    return f"{ref}:{resolved.point}" if resolved.point else ref


def _today(name: str) -> dict[str, object] | None:
    path = BASELINES / name
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    return {key: value for key, value in data.items() if not key.startswith("_")}


def _gates(planned: Plan) -> Iterator[tuple[str, str, GatePlan | None]]:
    """(baseline file name stem, task, gate) for every gate a Plan could have."""
    for task in planned.driver.tasks:
        if isinstance(task, PerfTaskPlan):
            yield "perf/{}.baseline.json", "", task.gate
        elif isinstance(task, EvalTaskPlan):
            for run in task.runs:
                yield "eval/{}." + run.task + ".baseline.json", run.key, run.gate


@pytest.mark.parametrize(
    ("ref", "resolved"), POINTS, ids=[point_id(*point) for point in POINTS]
)
def test_baselines_match_todays_files(ref: str, resolved: Resolved) -> None:
    """Inline baselines start as copies of today's files (R-MIG-2).

    A gate today's flow has must not be lost. A value that differs is reported,
    not failed: either side may be recalibrated on its own fleet while both exist.
    """
    config = TODAY_BASELINE_CONFIG[ref]
    for pattern, key, gate in _gates(plan(resolved)):
        today = _today(pattern.format(config)) if config is not None else None
        if today is None:
            continue
        assert gate is not None, (
            f"{ref}: today's {pattern.format(config)} has no gate here"
        )
        mine = gate.as_dict()
        theirs = {key: today[key]} if key else today
        if mine != theirs:
            warnings.warn(
                f"{ref}: inline baseline differs from {pattern.format(config)}",
                stacklevel=1,
            )
