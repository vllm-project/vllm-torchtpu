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
"""Gates give check_regression.py's verdicts and reports on the same files."""

import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import pytest
from plan_fixtures import build
from runtime_helpers import REPO_ROOT, stub_env, tiny
from scenario.plan.types import Command, EvalRunPlan, GatePlan
from scenario.runtime import exits, gate
from scenario.runtime.driver import RunOptions, run
from scenario.runtime.log import Log
from scenario.runtime.proc import Proc, capture_output

SCRIPT = REPO_ROOT / gate.CHECK_REGRESSION
LOG = Log("server/0", "sha", "v")


def plan_gate(
    baseline: dict[str, dict[str, float]], tolerance: float, soft: bool = False
) -> GatePlan:
    frozen = tuple(
        (key, tuple(sorted(metrics.items())))
        for key, metrics in sorted(baseline.items())
    )
    return GatePlan(tolerance, frozen, "inline", soft)


def cli(
    mode: str, results: pathlib.Path, baseline: dict[str, Any], tolerance: float
) -> tuple[int, str]:
    """check_regression.py's exit code and its output, stdout and stderr in order."""
    path = results.parent / f"{mode}-baseline.json"
    path.write_text(json.dumps(baseline))
    done = subprocess.run(
        [
            sys.executable,
            "-u",
            str(SCRIPT),
            "--mode",
            mode,
            "--results-dir",
            str(results),
            "--baseline",
            str(path),
            "--tolerance",
            str(tolerance),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return done.returncode, done.stdout


def cell(**metrics: float) -> dict[str, float]:
    return {
        "completed": 64,
        "median_ttft_ms": 100.0,
        "median_tpot_ms": 10.0,
        "total_token_throughput": 1000.0,
        "output_throughput": 500.0,
        **metrics,
    }


PERF_BASELINE = {
    "isl128_osl128_c1": {
        "completed": 64,
        "median_tpot_ms": 10.0,
        "output_token_throughput": 500.0,
        "median_ttft_ms": 100.0,
    },
    "isl128_osl128_c4": {"completed": 64, "median_tpot_ms": 10.0},
}

PERF_CASES = {
    "pass": {"isl128_osl128_c1": cell(), "isl128_osl128_c4": cell()},
    "within tolerance": {
        "isl128_osl128_c1": cell(median_tpot_ms=10.4),
        "isl128_osl128_c4": cell(),
    },
    "slower": {
        "isl128_osl128_c1": cell(median_tpot_ms=11.0),
        "isl128_osl128_c4": cell(),
    },
    "fewer completed": {
        "isl128_osl128_c1": cell(completed=50),
        "isl128_osl128_c4": cell(),
    },
    "a cell missing": {"isl128_osl128_c1": cell()},
    "an extra cell": {
        "isl128_osl128_c1": cell(),
        "isl128_osl128_c4": cell(),
        "isl256_osl64_c1": cell(),
    },
    "no results": {},
}


@pytest.mark.parametrize("case", PERF_CASES)
def test_perf_verdicts_and_reports_equal_the_scripts(
    tmp_path: pathlib.Path, case: str
) -> None:
    results = tmp_path / "run"
    results.mkdir()
    for key, data in PERF_CASES[case].items():
        (results / f"{key}.json").write_text(json.dumps(data))
    # Warmup files are never gated.
    (results / "isl128_osl128_c1.warmup1.json").write_text(
        json.dumps(cell(completed=0))
    )
    code, text = cli("perf", results, PERF_BASELINE, 0.05)
    verdict = gate.perf_gate(
        repo_root=REPO_ROOT,
        run_dir=results,
        gate=plan_gate(PERF_BASELINE, 0.05),
        cell_results=None,
        experiment=False,
        log=LOG,
    )
    assert (verdict.status == "pass") == (code == 0), text
    assert (results / gate.PERF_REPORT).read_text() == text


def lm_eval_result(
    path: pathlib.Path, task: str, score: float, stderr: float | None
) -> None:
    metrics = {"exact_match,custom-extract": score}
    if stderr is not None:
        metrics["exact_match_stderr,custom-extract"] = stderr
    path.mkdir(parents=True, exist_ok=True)
    (path / f"results_{time.time_ns()}.json").write_text(
        json.dumps({"results": {task: metrics}})
    )


EVAL_BASELINE = {
    "mmlu_pro": {
        "exact_match,custom-extract": 0.80,
        "exact_match_stderr,custom-extract": 0.01,
    }
}
EVAL_CASES = {
    "pass": (0.80, 0.01),
    "small drop": (0.79, 0.01),
    "big drop": (0.70, 0.01),
    "drop inside the stderr": (0.77, 0.02),
    "no stderr": (0.795, None),
    "no results": None,
}


@pytest.mark.parametrize("case", EVAL_CASES)
def test_eval_verdicts_and_reports_equal_the_scripts(
    tmp_path: pathlib.Path, case: str
) -> None:
    results = tmp_path / "run"
    results.mkdir()
    values = EVAL_CASES[case]
    if values is not None:
        lm_eval_result(results / "test__tiny", "mmlu_pro", *values)
    code, text = cli("eval", results, EVAL_BASELINE, 0.02)
    run_plan = EvalRunPlan(
        "mmlu_pro",
        "mmlu_pro",
        "lm_eval",
        Command(("lm_eval",)),
        plan_gate(EVAL_BASELINE, 0.02),
    )
    verdict = gate.eval_gate(
        repo_root=REPO_ROOT, run_dir=results, run=run_plan, experiment=False, log=LOG
    )
    assert (verdict.status == "pass") == (code == 0), text
    assert (results / "eval_check_mmlu_pro.md").read_text() == text


def test_a_sweep_run_is_gated_under_its_key(tmp_path: pathlib.Path) -> None:
    lm_eval_result(tmp_path / "gsm8k_c4" / "m", "gsm8k", 0.9, 0.01)
    lm_eval_result(tmp_path / "gsm8k_c8" / "m", "gsm8k", 0.5, 0.01)
    baseline = {
        "gsm8k_c4": {"exact_match,custom-extract": 0.9},
        "gsm8k_c8": {"exact_match,custom-extract": 0.9},
    }
    verdicts = {}
    for key in baseline:
        run_plan = EvalRunPlan(
            key,
            "gsm8k",
            "lm_eval",
            Command(("lm_eval",)),
            plan_gate({key: baseline[key]}, 0.02),
        )
        verdicts[key] = gate.eval_gate(
            repo_root=REPO_ROOT,
            run_dir=tmp_path,
            run=run_plan,
            experiment=False,
            log=LOG,
        ).status
    assert verdicts == {"gsm8k_c4": "pass", "gsm8k_c8": "fail"}
    assert "`gsm8k_c8`" in (tmp_path / "eval_check_gsm8k_c8.md").read_text()


def test_soft_skipped_and_missing_gates(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "isl128_osl128_c1.json").write_text(
        json.dumps(cell(median_tpot_ms=20.0))
    )
    common: dict[str, Any] = {"run_dir": tmp_path, "cell_results": None, "log": LOG}
    soft = gate.perf_gate(
        repo_root=REPO_ROOT,
        gate=plan_gate(PERF_BASELINE, 0.05, soft=True),
        experiment=False,
        **common,
    )
    assert soft == gate.Verdict("soft_fail", "regressed") and not soft.failed
    assert (
        "::warning::Perf regression (soft gate, not failing the run)"
        in capsys.readouterr().out
    )
    hard = gate.perf_gate(
        repo_root=REPO_ROOT,
        gate=plan_gate(PERF_BASELINE, 0.05),
        experiment=False,
        **common,
    )
    assert hard.failed
    assert (
        "::error::Perf regression check failed! See logs above"
        in capsys.readouterr().out
    )
    assert gate.perf_gate(
        repo_root=REPO_ROOT, gate=None, experiment=True, **common
    ) == gate.Verdict("skipped", "experiment: gates are off")
    assert gate.perf_gate(
        repo_root=REPO_ROOT, gate=None, experiment=False, **common
    ) == gate.Verdict("skipped", "no baseline")
    missing = gate.perf_gate(
        repo_root=tmp_path,
        gate=plan_gate(PERF_BASELINE, 0.05),
        experiment=False,
        **common,
    )
    assert missing.failed and "run from the repository root" in missing.reason


def test_cells_mode_gates_by_label(tmp_path: pathlib.Path) -> None:
    baseline = {"pd_c8": {"completed": 64, "median_tpot_ms": 10.0}}
    verdict = gate.perf_gate(
        repo_root=REPO_ROOT,
        run_dir=tmp_path,
        gate=plan_gate(baseline, 0.05),
        cell_results={"pd_c8": cell(median_tpot_ms=12.0)},
        experiment=False,
        log=LOG,
    )
    assert verdict.failed
    assert "`pd_c8`" in (tmp_path / gate.PERF_REPORT).read_text()


def test_a_childs_lines_wait_outside_a_capture(
    capsys: pytest.CaptureFixture[str],
) -> None:
    child = Proc(
        ["sh", "-c", "for i in 1 2 3 4 5 6; do echo child$i; sleep 0.03; done"],
        env=dict(os.environ),
    ).start()
    time.sleep(0.05)
    with capture_output() as out:
        print("inside")
        time.sleep(0.2)
    assert out.getvalue() == "inside\n"
    assert child.wait(5) == 0
    assert capsys.readouterr().out.count("child") == 6


@pytest.mark.usefixtures("api_port", "fast_polls")
@pytest.mark.parametrize(
    ("metrics", "code"),
    [("{}", exits.OK), ('{"median_tpot_ms": 20.0}', exits.TASK_FAILED)],
)
def test_a_run_gates_its_perf_cells(
    tmp_path: pathlib.Path, metrics: str, code: int
) -> None:
    data = tiny()
    data["tasks"].pop("eval")
    env = stub_env(tmp_path, STUB_BENCH_METRICS=metrics)
    options = RunOptions(
        env=env,
        results_root=tmp_path / "results",
        repo_root=REPO_ROOT,
        resolved={"point": None, "scenario": data},
    )
    assert run(build(data), options) == code
    summary = json.loads((tmp_path / "results" / "tiny" / "summary.json").read_text())
    (perf,) = summary["tasks"]
    assert perf["gate"]["status"] == ("pass" if code == 0 else "fail")
    # A gate failure is not a command failure: uploads still happen.
    assert summary["commands_ok"]
    assert (tmp_path / "results" / "tiny" / gate.PERF_REPORT).is_file()


@pytest.mark.usefixtures("api_port", "fast_polls")
def test_an_experiment_skips_its_gates(tmp_path: pathlib.Path) -> None:
    from plan_fixtures import scenario
    from scenario.plan.build import build_plan

    data = tiny()
    data["tasks"].pop("eval")
    env = stub_env(tmp_path, STUB_BENCH_METRICS='{"median_tpot_ms": 20.0}')
    plan = build_plan(
        scenario(data), ref="tiny", overrides=["server.args.max-num-seqs=32"]
    )
    options = RunOptions(
        env=env,
        results_root=tmp_path / "results",
        repo_root=REPO_ROOT,
        resolved={"point": None, "scenario": data},
    )
    assert run(plan, options) == exits.OK
    summary = json.loads((tmp_path / "results" / "tiny" / "summary.json").read_text())
    assert summary["tasks"][0]["gate"] == {
        "status": "skipped",
        "reason": "experiment: gates are off",
    }
