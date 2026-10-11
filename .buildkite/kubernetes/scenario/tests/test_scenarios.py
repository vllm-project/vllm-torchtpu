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
"""The CI scenarios: they validate, plan, fit their env values and pass a target.

The scenarios live in .buildkite/kubernetes/scenarios, so the tool's root is
.buildkite/kubernetes. The tool's own tests use its fixture repository, so
editing a scenario here never churns them. pod_target and text_checks are
helpers of the tool's tests, on the path through its pytest config.
"""

import os
import pathlib
from collections.abc import Iterator

import pytest
from click.testing import CliRunner
from pod_target import check_pods_run_the_plan
from scenario import __version__
from scenario.cli import main
from scenario.plan import codec
from scenario.plan.build import plan, resolved_payload, select_point
from scenario.plan.types import AgentxTaskPlan, EvalTaskPlan
from scenario.reserved import INFERENCEX_IDENTITY_ENV
from scenario.resolve import Resolved, Resolver
from scenario.show import render_plan
from scenario.target import TOOL_OWNED_ENV
from text_checks import flow_mapping_lines

ROOT = pathlib.Path(__file__).resolve().parents[2]
SNAPSHOTS = pathlib.Path(__file__).resolve().parent / "snapshots"
SCENARIO_FILES = sorted((ROOT / "scenarios").rglob("*.yaml"))
SCHEMAS = ROOT.parents[1] / "tools" / "scenario" / "schema"
SCHEMA_LINE = "# yaml-language-server: $schema="

M1_SCENARIOS = {
    "qwen3-0.6b/tp1",
    "glm-5.2/fp8-dp16-ep",
    "glm-5.2/fp8-dp16-ep-mp",
    "qwen3.5-397b/fp8-pcp8-dp8-1p1d",
    "qwen3.5-397b/inferencex-sweep",
    "kimi-k3/tp32-ep",
    "kimi-k3/tp8-dp4-ep-1p1d",
    "kimi-k3/agentx",
    "kimi-k3/agentx-1p1d",
    "kimi-k3/eval-conc-sweep",
}


def resolver() -> Resolver:
    return Resolver(ROOT)


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
    assert refs == M1_SCENARIOS


@pytest.mark.parametrize(
    ("ref", "resolved"), POINTS, ids=[point_id(*point) for point in POINTS]
)
def test_every_point_plans(ref: str, resolved: Resolved) -> None:
    planned = plan(resolved)
    assert planned.scenario_ref == ref
    assert planned.point == resolved.point
    assert codec.decode(codec.encode(planned)) == planned


def test_validate_passes_and_names_are_unique() -> None:
    result = CliRunner().invoke(main, ["validate", "--root", str(ROOT)])
    assert result.exit_code == 0, result.output
    count = len(M1_SCENARIOS)
    assert result.stdout.splitlines()[-1] == f"{count} scenarios, 0 errors"
    names: dict[str, pathlib.Path] = {}
    for file in resolver().scenario_files():
        declared = resolver().declared_name(file)
        assert declared is not None, file
        assert declared[0] not in names, (file, names.get(declared[0]))
        names[declared[0]] = file


def test_every_pod_runs_what_the_plan_says() -> None:
    """Invariance (R-TEST-4): a target changes nothing a host runs."""
    for ref, resolved in POINTS:
        check_pods_run_the_plan(point_id(ref, resolved), plan(resolved))


def test_tool_owned_env_covers_what_plans_set() -> None:
    for ref, resolved in POINTS:
        planned = plan(resolved)
        name = point_id(ref, resolved)
        for role in planned.roles:
            for host in role.host_plans:
                assert all(TOOL_OWNED_ENV.covers(n) for n, _ in host.tool_env), name
        assert all(TOOL_OWNED_ENV.covers(n) for n, _ in planned.driver.task_env), name


def test_agentx_1p1d_gsm8k_plan() -> None:
    """The plan of the GSM8K eval point.

    No kube step runs InferenceX's eval-only GSM8K today, so no comparison
    with today's scripts covers it; this snapshot follows the scenario instead
    (set SCENARIO_UPDATE_SNAPSHOTS=1 to rewrite it after an intended change).
    """
    points = resolver().resolve(resolver().file_of("kimi-k3/agentx-1p1d"))
    text = render_plan(plan(select_point(points, "gsm8k")))
    text = text.replace(f"tool {__version__}", "tool <version>")
    path = SNAPSHOTS / "plan_agentx_1p1d_gsm8k.txt"
    if os.environ.get("SCENARIO_UPDATE_SNAPSHOTS") == "1":
        path.write_text(text)
    assert text == path.read_text()


def test_agentx_1p1d_eval_joins_its_rows() -> None:
    """The GSM8K point files its eval under the AgentX rows' identity (P-22)."""
    points = resolver().resolve(resolver().file_of("kimi-k3/agentx-1p1d"))
    identities = {}
    for resolved in points:
        (task,) = plan(resolved).driver.tasks
        command = task.runs[0].command if isinstance(task, EvalTaskPlan) else None
        if isinstance(task, AgentxTaskPlan):
            command = task.command
        assert command is not None
        env = dict(command.env)
        identities[resolved.point] = {
            n: env[n] for n in INFERENCEX_IDENTITY_ENV if n in env
        }
    assert identities["gsm8k"] == identities["c1"] == identities["c4"]


@pytest.mark.parametrize(
    "path", SCENARIO_FILES, ids=[str(p.relative_to(ROOT)) for p in SCENARIO_FILES]
)
def test_scenario_files_use_block_mappings(path: pathlib.Path) -> None:
    """Scenario files write mappings as blocks; short scalar lists may stay inline."""
    lines = flow_mapping_lines(path.read_text())
    assert not lines, f"flow mappings at lines {lines}; write them as block mappings"


@pytest.mark.parametrize(
    "path", SCENARIO_FILES, ids=[str(p.relative_to(ROOT)) for p in SCENARIO_FILES]
)
def test_scenario_files_point_editors_at_their_schema(path: pathlib.Path) -> None:
    first = path.read_text().splitlines()[0]
    assert first.startswith(SCHEMA_LINE), first
    schema = "defaults" if path.name == "defaults.yaml" else "scenario"
    found = (path.parent / first.removeprefix(SCHEMA_LINE)).resolve()
    assert found == SCHEMAS / f"{schema}.schema.json"


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
