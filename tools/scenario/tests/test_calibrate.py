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
"""calibrate: check_regression.py's values, written into baseline blocks only."""

import json
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

import pytest
from click.testing import CliRunner, Result
from ruamel.yaml import YAML
from runtime_helpers import REPO_ROOT
from scenario.cli import main
from scenario.runtime.gate import CHECK_REGRESSION

TESTS = pathlib.Path(__file__).resolve().parent
CELLS = ("isl128_osl128_c1", "isl128_osl128_c4")


@pytest.fixture
def root(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A copy of the runtime fixture root; commands run from the repository root."""
    found = tmp_path / "root"
    shutil.copytree(TESTS / "data" / "runtime", found)
    monkeypatch.chdir(REPO_ROOT)
    return found


def tiny(root: pathlib.Path) -> pathlib.Path:
    return root / "scenarios" / "smoke" / "tiny.yaml"


def runs(tmp_path: pathlib.Path) -> list[pathlib.Path]:
    """Two runs' result files: perf cells and lm_eval results."""
    found = []
    for index, (tpot, score) in enumerate([(10.4, 0.81), (10.8, 0.79)]):
        run_dir = tmp_path / f"run{index}" / "tiny"
        (run_dir / "test__tiny").mkdir(parents=True)
        for cell, completed in zip(CELLS, (2, 8)):
            result = {
                "completed": completed,
                "num_prompts": completed,
                "median_tpot_ms": tpot,
                "median_ttft_ms": 100.0,
                "total_token_throughput": 1000.0 + index,
                "output_throughput": 500.0 - index,
            }
            (run_dir / f"{cell}.json").write_text(json.dumps(result))
        for task in ("mmlu_pro", "gsm8k"):
            metrics = {
                "exact_match,custom-extract": score,
                "exact_match_stderr,custom-extract": 0.01,
            }
            path = run_dir / "test__tiny" / f"results_{task}_{time.time_ns()}.json"
            path.write_text(json.dumps({"results": {task: metrics}}))
        found.append(run_dir)
    return found


def invoke(*args: Any) -> Result:
    return CliRunner().invoke(main, [str(arg) for arg in args])


def script_calibration(mode: str, dirs: list[pathlib.Path], tasks: str = "") -> Any:
    argv = [
        sys.executable,
        str(REPO_ROOT / CHECK_REGRESSION),
        "--mode",
        mode,
        "--calibrate",
    ]
    for run_dir in dirs:
        argv += ["--results-dir", str(run_dir)]
    if tasks:
        argv += ["--task-filter", tasks]
    return json.loads(
        subprocess.run(argv, capture_output=True, text=True, check=True).stdout
    )


def baselines(file: pathlib.Path) -> tuple[Any, Any]:
    tasks = YAML(typ="safe").load(file.read_text())["tasks"]
    return tasks["perf"].get("baseline"), tasks["eval"].get("baseline")


def test_values_are_check_regressions(
    tmp_path: pathlib.Path, root: pathlib.Path
) -> None:
    dirs = runs(tmp_path)
    file = tiny(root)
    before = file.read_text()
    result = invoke("calibrate", file, *dirs)
    assert result.exit_code == 0, result.output
    assert file.read_text() == before
    assert "median_tpot_ms: 10 -> 10.6 (+6.0%)" in result.output
    assert "output_token_throughput: 500 -> 499.5 (-0.1%)" in result.output
    result = invoke("calibrate", file, *dirs, "--write")
    assert result.exit_code == 0, result.output
    assert result.output.endswith(f"wrote 4 baseline entries to {file}\n")
    perf, evals = baselines(file)
    expected = script_calibration("perf", dirs)
    assert perf == {key: expected[key] for key in CELLS}
    assert evals == script_calibration("eval", dirs, "mmlu_pro,gsm8k")


def test_writing_touches_only_baseline_entries(
    tmp_path: pathlib.Path, root: pathlib.Path
) -> None:
    dirs = runs(tmp_path)
    file = tiny(root)
    before = file.read_text().splitlines()
    assert invoke("calibrate", file, *dirs, "--write").exit_code == 0
    after = file.read_text().splitlines()
    # Everything before the perf entries, the comment over them included, stays.
    first = before.index("      isl128_osl128_c1:")
    assert after[:first] == before[:first]
    assert "      # Calibrated on the stub server." in after
    assert (
        after[
            after.index("  eval:") : after.index(
                "    baseline:", after.index("  eval:")
            )
        ]
        == (
            before[
                before.index("  eval:") : before.index(
                    "    baseline:", before.index("  eval:")
                )
            ]
        )
    )
    # A second write changes nothing.
    again = invoke("calibrate", file, *dirs, "--write")
    assert again.exit_code == 0 and again.output.endswith("baselines unchanged\n")
    assert file.read_text().splitlines() == after
    assert invoke("validate", "--root", root).exit_code == 0


def test_a_task_without_a_baseline_gets_one(
    tmp_path: pathlib.Path, root: pathlib.Path
) -> None:
    dirs = runs(tmp_path)
    file = tiny(root)
    text = file.read_text()
    start = text.index("    baseline:\n      # Calibrated")
    end = text.index("  eval:")
    text = text[:start] + text[end:]
    file.write_text(text.split("    baseline:\n      mmlu_pro:")[0])
    result = invoke("calibrate", file, *dirs, "--write")
    assert result.exit_code == 0, result.output
    assert "isl128_osl128_c4: (new)" in result.output
    perf, evals = baselines(file)
    assert set(perf) == set(CELLS)
    assert set(evals) == {"mmlu_pro", "gsm8k"}
    assert invoke("validate", "--root", root).exit_code == 0


def test_a_concurrency_sweep_gains_its_missing_keys(
    tmp_path: pathlib.Path, root: pathlib.Path
) -> None:
    file = tiny(root)
    text = file.read_text()
    text = text.split("    baseline:\n      mmlu_pro:")[0]
    text = text.replace(
        "      tasks: [mmlu_pro, gsm8k]\n",
        "      tasks: [mmlu_pro]\n"
        "      concurrency: [1, 2]\n"
        "      model_args:\n"
        "        num_concurrent: null\n",
    )
    file.write_text(
        text + "    baseline:\n"
        "      mmlu_pro_c1:\n"
        '        "exact_match,custom-extract": 0.5\n'
    )
    dirs = []
    for index, score in enumerate((0.7, 0.9)):
        run_dir = tmp_path / f"sweep{index}" / "tiny"
        for key in ("mmlu_pro_c1", "mmlu_pro_c2"):
            (run_dir / key / "m").mkdir(parents=True)
            metrics = {"exact_match,custom-extract": score}
            results = run_dir / key / "m" / "results_1.json"
            results.write_text(json.dumps({"results": {"mmlu_pro": metrics}}))
        dirs.append(run_dir)
    result = invoke("calibrate", file, *dirs, "--write")
    assert result.exit_code == 0, result.output
    assert "mmlu_pro_c2: (new)" in result.output
    _, evals = baselines(file)
    assert evals == {
        "mmlu_pro_c1": {"exact_match,custom-extract": 0.8},
        "mmlu_pro_c2": {"exact_match,custom-extract": 0.8},
    }


def test_a_named_point_keeps_its_own_baseline(
    tmp_path: pathlib.Path, root: pathlib.Path
) -> None:
    dirs = runs(tmp_path)
    file = root / "scenarios" / "smoke" / "points.yaml"
    file.write_text(
        tiny(root)
        .read_text()
        .replace("name: tiny", "name: tiny-points")
        .split("    baseline:\n      # Calibrated")[0]
        .replace("  eval:", "  eval-unused:")
        .split("  eval-unused:")[0]
        + "sweep:\n"
        "  points:\n"
        "    - name: low\n"
        "      tasks:\n"
        "        perf:\n"
        "          concurrency: [1]\n"
        "          baseline:\n"
        "            isl128_osl128_c1:\n"
        "              completed: 1\n"
    )
    assert invoke("validate", file).exit_code == 0, invoke("validate", file).output
    result = invoke("calibrate", file, *dirs, "--point", "low", "--write")
    assert result.exit_code == 0, result.output
    point = YAML(typ="safe").load(file.read_text())["sweep"]["points"][0]
    assert point["tasks"]["perf"]["baseline"]["isl128_osl128_c1"]["completed"] == 2


def test_a_from_baseline_is_refused(tmp_path: pathlib.Path, root: pathlib.Path) -> None:
    file = root / "scenarios" / "smoke" / "borrowed.yaml"
    file.write_text(
        "schema: 1\n"
        "name: borrowed\n"
        "extends: smoke/tiny\n"
        "tasks:\n"
        "  perf:\n"
        "    baseline:\n"
        "      from: smoke/tiny\n"
        "  eval: null\n"
    )
    before = file.read_text()
    result = invoke("calibrate", file, *runs(tmp_path), "--write")
    assert result.exit_code == 1
    assert (
        "tasks.perf.baseline is from: smoke/tiny; calibrate smoke/tiny instead"
        in result.stderr
    )
    assert file.read_text() == before
