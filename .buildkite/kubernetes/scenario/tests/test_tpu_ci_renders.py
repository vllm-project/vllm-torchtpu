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
"""Every CI scenario point rendered by the CI fleet target.

Each render passes the launcher's checks, and each of its pods runs exactly
its part of the Plan (R-TEST-4). The render depends on the Plan's shape and
image only (R-TGT-11): changing what runs changes nothing but the encoded
Plan and resolved scenario. The snapshots hold one render per distinct
manifest of a scenario, with the encoded values elided; set
SCENARIO_UPDATE_SNAPSHOTS=1 to rewrite them after an intended change.
"""

import os
import pathlib
import re
from collections.abc import Iterator
from types import ModuleType

import pytest
from manifest_checks import check_pods_run_the_plan, load
from scenario.plan import codec
from scenario.plan.build import plan, resolved_payload
from scenario.resolve import Resolved, Resolver

ROOT = pathlib.Path(__file__).resolve().parents[2]
SNAPSHOTS = pathlib.Path(__file__).resolve().parent / "snapshots" / "render"
WORKLOAD = "bk-0123"
ENCODED = ("SCENARIO_PLAN", "SCENARIO_RESOLVED", "SCENARIO_TARGET_DATA")
# What runs, as opposed to where: none of it may change the manifest.
NOT_SHAPE = (
    "server.env.SHAPE_PROBE=1",
    "server.model.name=probe/other-model",
    "server.startup_timeout=5h",
)

pytestmark = pytest.mark.usefixtures("fleet_env")


def scenarios() -> Iterator[tuple[str, list[Resolved]]]:
    resolver = Resolver(ROOT)
    for file in resolver.scenario_files():
        yield resolver.ref_of(file), resolver.resolve(file)


SCENARIOS = list(scenarios())


def point_id(resolved: Resolved) -> str:
    return f"{resolved.ref}:{resolved.point}" if resolved.point else resolved.ref


def elide(text: str, names: tuple[str, ...] = ENCODED) -> str:
    """The render with the encoded env values replaced by their names."""
    pattern = rf"(- name: ({'|'.join(names)})\n\s+value: )'[^']*'"
    return re.sub(pattern, r"\1<\2>", text)


def render(tpu_ci: ModuleType, resolved: Resolved) -> str:
    return tpu_ci.TpuCi().render(plan(resolved), resolved_payload(resolved))


def test_every_scenario_is_covered() -> None:
    points = sum(len(points) for _, points in SCENARIOS)
    assert len(SCENARIOS) == 10 and points >= 21


@pytest.mark.parametrize(
    ("ref", "points"), SCENARIOS, ids=[ref for ref, _ in SCENARIOS]
)
def test_pods_run_the_plan(
    tpu_ci: ModuleType, ref: str, points: list[Resolved]
) -> None:
    for resolved in points:
        planned = plan(resolved)
        before = codec.to_json(planned)
        text = render(tpu_ci, resolved)
        assert codec.to_json(planned) == before, point_id(resolved)
        assert tpu_ci.launcher_problems(text) == [], point_id(resolved)
        doc = load(text, IMAGE="registry/ci@sha256:0", WORKLOAD_NAME=WORKLOAD)
        check_pods_run_the_plan(point_id(resolved), planned, doc, WORKLOAD)


@pytest.mark.parametrize(
    ("ref", "points"), SCENARIOS, ids=[ref for ref, _ in SCENARIOS]
)
def test_render_depends_only_on_the_shape(
    tpu_ci: ModuleType, ref: str, points: list[Resolved]
) -> None:
    resolver = Resolver(ROOT)
    changed = resolver.resolve(resolver.file_of(ref), NOT_SHAPE)
    for before, after in zip(points, changed, strict=True):
        assert plan(before) != plan(after)
        assert plan(before).shape() == plan(after).shape()
        names = ("SCENARIO_PLAN", "SCENARIO_RESOLVED")
        assert elide(render(tpu_ci, before), names) == elide(
            render(tpu_ci, after), names
        ), point_id(before)


@pytest.mark.parametrize(
    ("ref", "points"), SCENARIOS, ids=[ref for ref, _ in SCENARIOS]
)
def test_render_snapshot(tpu_ci: ModuleType, ref: str, points: list[Resolved]) -> None:
    renders: dict[str, list[str]] = {}
    for resolved in points:
        renders.setdefault(elide(render(tpu_ci, resolved)), []).append(
            resolved.point or "-"
        )
    text = "---\n".join(
        f"# {ref} points: {', '.join(names)}\n{body}" for body, names in renders.items()
    )
    path = SNAPSHOTS / f"{ref.replace('/', '__')}.yaml"
    if os.environ.get("SCENARIO_UPDATE_SNAPSHOTS") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert path.is_file(), (
        f"no snapshot {path.name}; run with SCENARIO_UPDATE_SNAPSHOTS=1"
    )
    assert text == path.read_text()
