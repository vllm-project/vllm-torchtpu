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
"""The scenarios under scenarios/: they validate, plan and fit their env values.

The CLI snapshots use the fixture repository under tests/data, so editing a
real scenario never churns them; these tests read scenarios/ itself.
"""

import json
import pathlib
import warnings
from collections.abc import Iterator

import pytest
from click.testing import CliRunner
from scenario.cli import main
from scenario.plan import codec
from scenario.plan.build import plan, resolved_payload
from scenario.plan.types import EvalTaskPlan, GatePlan, PerfTaskPlan, Plan
from scenario.resolve import Resolved, Resolver

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BASELINES = REPO_ROOT / "scripts" / "vllm" / "benchmarking" / "baselines"

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
    "kimi-k3/eval-conc-sweep": "kimi-k3-tp32-ep-concsweep",
}


def resolver() -> Resolver:
    return Resolver(REPO_ROOT)


def every_point() -> Iterator[tuple[str, Resolved]]:
    found = resolver()
    for file in found.scenario_files():
        for resolved in found.resolve(file):
            yield found.ref_of(file), resolved


POINTS = list(every_point())


def point_id(ref: str, resolved: Resolved) -> str:
    return f"{ref}:{resolved.point}" if resolved.point else ref


def test_the_m1_scenarios_exist() -> None:
    refs = {resolver().ref_of(file) for file in resolver().scenario_files()}
    assert refs == set(TODAY_BASELINE_CONFIG)


@pytest.mark.parametrize(
    ("ref", "resolved"), POINTS, ids=[point_id(*point) for point in POINTS]
)
def test_every_point_plans(ref: str, resolved: Resolved) -> None:
    planned = plan(resolved)
    assert planned.scenario_ref == ref
    assert planned.point == resolved.point
    assert codec.decode(codec.encode(planned)) == planned


def test_validate_passes_and_names_are_unique() -> None:
    result = CliRunner().invoke(main, ["validate", "--root", str(REPO_ROOT)])
    assert result.exit_code == 0, result.output
    count = len(TODAY_BASELINE_CONFIG)
    assert result.stdout.splitlines()[-1] == f"{count} scenarios, 0 errors"
    names: dict[str, pathlib.Path] = {}
    for file in resolver().scenario_files():
        declared = resolver().declared_name(file)
        assert declared is not None, file
        assert declared[0] not in names, (file, names.get(declared[0]))
        names[declared[0]] = file


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


def test_encoded_sizes(capsys: pytest.CaptureFixture[str]) -> None:
    rows = []
    for ref, resolved in POINTS:
        planned = codec.encode(plan(resolved))
        payload = codec.encode_data(resolved_payload(resolved))
        rows.append((point_id(ref, resolved), len(planned), len(payload)))
    width = max(len(name) for name, _, _ in rows)
    lines = [f"{'scenario':<{width}}  SCENARIO_PLAN  SCENARIO_RESOLVED"]
    lines += [f"{name:<{width}}  {p:>13}  {r:>17}" for name, p, r in rows]
    with capsys.disabled():
        print(f"\nEncoded sizes in bytes (limit {codec.MAX_ENCODED_BYTES}):")
        print("\n".join(lines))
    for name, size_plan, size_resolved in rows:
        assert size_plan <= codec.MAX_ENCODED_BYTES, name
        assert size_resolved <= codec.MAX_ENCODED_BYTES, name
