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
"""Drift check: today's kube scripts and the scenarios' plans run the same.

Each case runs today's scripts with stub programs (golden_capture.py), plans
its scenario, and compares the two (golden_compare.py). A difference fails
unless allowlist.yaml accepts it with a reason, and an entry that accepts
nothing fails too. The self-tests seed a flag drift and an env drift into the
scratch copy of today's scripts, and an entry that matches nothing, and check
that each one is caught.

The scripts need bash >= 4.4: the first one found, or $GOLDEN_BASH.
"""

import json
import os
import pathlib

import pytest
from golden_capture import SCENARIO_ROOT, Capture, find_bash
from golden_cases import Case, all_cases, qwen_point, sharding_table
from golden_compare import (
    Diff,
    Entry,
    Normalizer,
    Outcome,
    agentx_flags,
    judge,
    load_allowlist,
    run_case,
    stale,
)
from scenario.plan.build import plan, select_point
from scenario.plan.types import CustomTaskPlan, EvalTaskPlan
from scenario.resolve import Resolver

CASES = all_cases()
ENTRIES = load_allowlist()


def case(case_id: str) -> Case:
    return next(c for c in CASES if c.id == case_id)


@pytest.fixture(scope="session")
def bash() -> str:
    found = find_bash()
    if found is None:
        message = "the golden test needs bash >= 4.4; set GOLDEN_BASH to one"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    return found


class Runs:
    """Each case runs once per session; the stale check reads every outcome."""

    def __init__(self, bash: str, factory: pytest.TempPathFactory) -> None:
        self.bash = bash
        self.factory = factory
        self.done: dict[str, tuple[list[Outcome], Capture]] = {}

    def run(self, which: Case) -> tuple[list[Outcome], Capture]:
        if which.id not in self.done:
            work = self.factory.mktemp(which.id.replace(":", "-"))
            self.done[which.id] = run_case(which, self.bash, work, ENTRIES)
        return self.done[which.id]

    def outcomes(self, which: Case) -> list[Outcome]:
        return self.run(which)[0]


@pytest.fixture(scope="session")
def runs(bash: str, tmp_path_factory: pytest.TempPathFactory) -> Runs:
    return Runs(bash, tmp_path_factory)


def suggestion(outcome: Outcome, diff: Diff) -> str:
    lines = [f"- scenario: {outcome.scenario}"]
    if outcome.point is not None:
        lines.append(f"  point: {outcome.point}")
    for key in ("where", "kind", "name", "today", "plan"):
        lines.append(f"  {key}: {json.dumps(getattr(diff, key))}")
    lines.append("  reason: <why the plan differs on purpose>")
    return "\n".join(lines)


def report(outcomes: list[Outcome], captured: Capture) -> str:
    parts = []
    for outcome in outcomes:
        for diff in outcome.unexpected:
            parts.append(
                f"{diff.where}: {diff.kind} {diff.name}: today {diff.today!r}, plan "
                f"{diff.plan!r}\n{suggestion(outcome, diff)}"
            )
    logs = [
        f"--- {host} exited {code} ---\n{captured.log_tail(host)}"
        for host, code in captured.exits.items()
        if code
    ]
    return (
        "today's scripts and the plan differ; fix the scenario, or add an entry "
        "to .buildkite/kubernetes/scenario/golden/allowlist.yaml:\n\n"
        + "\n\n".join(parts + logs)
    )


@pytest.mark.parametrize("which", CASES, ids=[c.id for c in CASES])
def test_matches_today(which: Case, runs: Runs) -> None:
    outcomes, captured = runs.run(which)
    unexpected = [d for o in outcomes for d in o.unexpected]
    assert not unexpected, report(outcomes, captured)


def test_allowlist_has_no_stale_entries(runs: Runs) -> None:
    outcomes = [outcome for which in CASES for outcome in runs.outcomes(which)]
    found = stale(ENTRIES, outcomes)
    assert not found, "entries that match no difference; remove them:\n" + "\n".join(
        f"{e.scenarios} {e.where} {e.kind} {e.name}" for e in found
    )


def test_allowlisted_scenarios_have_cases() -> None:
    scenarios = {c.scenario for c in CASES}
    assert {s for e in ENTRIES for s in e.scenarios} <= scenarios


def test_a_stale_entry_is_caught(runs: Runs) -> None:
    seeded = Entry(
        scenarios=("qwen3-0.6b/tp1",),
        point=None,
        where="serve server/0",
        kind="flag",
        name="--no-such-flag",
        today=None,
        plan="1",
        reason="seeded",
    )
    outcomes = [
        judge(o.scenario, o.point, o.diffs, [*ENTRIES, seeded])
        for o in runs.outcomes(case("P2"))
    ]
    mine = [e for e in ENTRIES if "qwen3-0.6b/tp1" in e.scenarios]
    assert mine
    assert stale([*mine, seeded], outcomes) == [seeded]


def _append(root: pathlib.Path, relative: str, line: str) -> None:
    path = root / relative
    path.write_text(path.read_text() + line + "\n")


def test_detects_flag_drift(bash: str, tmp_path: pathlib.Path) -> None:
    def drift(root: pathlib.Path) -> None:
        _append(
            root,
            "scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh",
            'EXTRA_SERVE_ARGS="$EXTRA_SERVE_ARGS --foo-bar 1"',
        )

    (outcome,), _ = run_case(case("P11"), bash, tmp_path, ENTRIES, drift)
    assert outcome.unexpected == [
        Diff("serve server/0", "flag", "--foo-bar", "1", None)
    ]


def test_detects_env_drift(bash: str, tmp_path: pathlib.Path) -> None:
    def drift(root: pathlib.Path) -> None:
        _append(
            root,
            "scripts/vllm/benchmarking/configs/kimi-k3-tp8-dp4-ep-1p1d.sh",
            "export TPU_GOLDEN_DRIFT=1",
        )

    (outcome,), _ = run_case(case("P15"), bash, tmp_path, ENTRIES, drift)
    hosts = [f"{role}/{i}" for role in ("decode", "prefill") for i in range(4)]
    assert outcome.unexpected == [
        Diff(f"serve {host}", "env", "TPU_GOLDEN_DRIFT", "1", None) for host in hosts
    ]


def test_p15_runs_correctness_before_the_eval(runs: Runs) -> None:
    """The order of k3_1p1d.sh: the stages, and the eval unless they could not start."""
    resolver = Resolver(SCENARIO_ROOT)
    points = resolver.resolve(resolver.file_of("kimi-k3/tp8-dp4-ep-1p1d"))
    tasks = plan(select_point(points)).driver.tasks
    assert [type(task) for task in tasks] == [CustomTaskPlan, EvalTaskPlan]
    correctness = tasks[0]
    assert isinstance(correctness, CustomTaskPlan)
    assert (correctness.name, correctness.skip_rest_if_exit_ge) == ("correctness", 2)
    _, captured = runs.run(case("P15"))
    order = [
        pathlib.Path(r["argv"][0]).name
        if r["tool"] in ("python3", "script")
        else r["tool"]
        for r in captured.records
        if r["host"] == "prefill-0" and r["tool"] in ("python3", "script", "lm_eval")
    ]
    order = [name for name in order if name != "toy_proxy_server.py"]
    assert order[:3] == [
        "check_completions.py",
        "run_k3_1p1d_disagg_correctness.sh",
        "lm_eval",
    ]


def test_every_point_is_compared_or_skipped() -> None:
    """A scenario's points are each in a case, or skipped by one with a reason."""
    resolver = Resolver(SCENARIO_ROOT)
    covered: dict[str, set[str | None]] = {}
    for which in CASES:
        covered.setdefault(which.scenario, set()).update(which.points, which.skipped)
    for scenario, names in covered.items():
        points = resolver.resolve(resolver.file_of(scenario))
        assert {p.point for p in points} == names, scenario


def test_qwen_points_are_the_sharding_table() -> None:
    resolver = Resolver(SCENARIO_ROOT)
    points = resolver.resolve(resolver.file_of("qwen3.5-397b/inferencex-sweep"))
    names = [qwen_point(conc, sharding) for _, _, conc, sharding in sharding_table()]
    assert [p.point for p in points] == names


def test_normalization_is_representation_only() -> None:
    root, state = pathlib.Path("/scratch/repo"), pathlib.Path("/scratch/state")
    norm = Normalizer(Capture(root, state, [], [], {}), ["perf_eval_results/x"])

    def flags(*argv: str) -> dict[str, object]:
        return dict(norm.call(argv, {}).flags)

    assert flags("vllm", "serve", "m", "--temperature=0") == flags(
        "vllm", "serve", "--model", "m", "--temperature", "0.0"
    )
    assert flags("x", "--c", '{"b":1,"a":[2]}') == flags(
        "x", "--c", '{"a": [2], "b": 1}'
    )
    assert flags("x", "--gpu-memory-utilization", "0.90") == {
        "gpu-memory-utilization": 0.9
    }
    assert flags("x", "--flag")["flag"] is True
    assert flags(
        "lm_eval", "--model_args", "base_url=http://localhost:8000,n=128,n=16"
    ) == {"model_args": {"base_url": "http://127.0.0.1:8000", "n": 16.0}}
    assert flags("vllm", "bench", "serve", "--result-file=/scratch/repo/r.json") == {
        "result-filename": "r.json"
    }
    assert flags("vllm", "serve", "m", "--tokenizer", "m") == {"model": "m"}
    assert norm.text("perf_eval_results/x/a.json") == "<results>/a.json"
    toy = "/scratch/repo/examples/disagg/toy_proxy_server.py"
    assert flags("python3", toy, "--prefiller-host", "h", "--decoder-port", "1") == {
        "prefiller-hosts": "h",
        "decoder-ports": 1.0,
    }


def test_agentx_env_as_client_flags() -> None:
    pair = {
        "AIPERF_SERVER_URL": "http://127.0.0.1:8100",
        "CONC": "4",
        "MODEL_PREFIX": "kimik3",
        "IS_MULTINODE": "true",
        "PREFILL_NUM_WORKERS": "1",
        "PREFILL_TP": "16",
        "DECODE_NUM_WORKERS": "1",
        "DECODE_TP": "16",
        "RESULT_FILENAME": "agentx_c4",
    }
    assert agentx_flags(pair) == [
        "--server-url",
        "http://127.0.0.1:8100",
        "--conc-list",
        "4",
        "--model-prefix",
        "kimik3",
        "--chips",
        "32",
        "--label",
        "agentx",
    ]
    one = {"CONC": "1", "IS_MULTINODE": "false", "TP": "16", "RESULT_FILENAME": "x_c1"}
    assert agentx_flags(one) == ["--conc-list", "1", "--chips", "16", "--label", "x"]
