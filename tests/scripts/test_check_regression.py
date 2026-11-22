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
"""Unit tests for scripts/vllm/benchmarking/check_regression.py."""

import importlib.util
import json
import math
import pathlib
import sys

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "vllm"
    / "benchmarking"
    / "check_regression.py"
)
_spec = importlib.util.spec_from_file_location("check_regression", _SCRIPT)
check_regression = importlib.util.module_from_spec(_spec)
# Must be registered before exec_module: the script uses
# `from __future__ import annotations`, so dataclasses resolves its field
# annotations by looking the module up in sys.modules at class-creation time.
sys.modules["check_regression"] = check_regression
_spec.loader.exec_module(check_regression)

_BASELINE_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "vllm"
    / "benchmarking"
    / "baselines"
    / "eval"
)


def test_primary_metric():
    entry = {
        "exact_match,strict_match": 0.885,
        "exact_match_stderr,strict_match": 0.010,
    }
    metric = check_regression.primary_metric(entry)
    assert metric == ("exact_match,strict_match", 0.885)


def test_primary_stderr_matching():
    entry = {
        "exact_match,strict_match": 0.885,
        "exact_match_stderr,strict_match": 0.012,
    }
    stderr = check_regression.primary_stderr(entry)
    assert stderr == 0.012


def test_primary_stderr_custom_extract():
    entry = {
        "exact_match,custom-extract": 0.807,
        "exact_match_stderr,custom-extract": 0.024,
    }
    stderr = check_regression.primary_stderr(entry)
    assert stderr == 0.024


def test_stderr_key():
    assert (
        check_regression.stderr_key("exact_match,strict_match")
        == "exact_match_stderr,strict_match"
    )
    assert check_regression.stderr_key("acc,none") == "acc_stderr,none"
    assert check_regression.stderr_key("exact_match,none") == "exact_match_stderr,none"
    assert check_regression.stderr_key("acc_norm,none") == "acc_norm_stderr,none"
    assert (
        check_regression.stderr_key("pass@1,create_test") == "pass@1_stderr,create_test"
    )
    assert (
        check_regression.stderr_key("pass_at_1,create_test")
        == "pass_at_1_stderr,create_test"
    )
    assert check_regression.stderr_key("pass_at_1,none") == "pass_at_1_stderr,none"
    assert check_regression.stderr_key("acc") == "acc_stderr"
    assert check_regression.stderr_key("acc_stderr,none") == "acc_stderr,none"


def test_primary_stderr_paired_variant():
    entry_em_none = {
        "exact_match,none": 0.85,
        "exact_match_stderr,none": 0.015,
    }
    assert check_regression.primary_stderr(entry_em_none) == 0.015

    entry_acc_norm = {
        "acc_norm,none": 0.72,
        "acc_norm_stderr,none": 0.022,
    }
    assert check_regression.primary_stderr(entry_acc_norm) == 0.022

    # Mismatched metric and stderr should not pair
    entry_mismatch = {
        "exact_match,strict_match": 0.88,
        "acc_stderr,none": 0.03,
    }
    assert check_regression.primary_stderr(entry_mismatch) is None


def test_primary_stderr_absent():
    entry = {
        "exact_match,strict_match": 0.885,
    }
    stderr = check_regression.primary_stderr(entry)
    assert stderr is None


def test_delta_stderr_combines_both_runs():
    metric = "exact_match,strict_match"
    baseline = {metric: 0.88, "exact_match_stderr,strict_match": 0.03}
    results = {metric: 0.86, "exact_match_stderr,strict_match": 0.04}
    # Errors add in quadrature: hypot(0.03, 0.04) == 0.05 exactly.
    assert check_regression.delta_stderr(baseline, results, metric) == 0.05


def test_delta_stderr_assumes_symmetry_when_run_stderr_missing():
    metric = "exact_match,strict_match"
    baseline = {metric: 0.88, "exact_match_stderr,strict_match": 0.03}
    # No stderr on the results side -> assume the run is as noisy as the
    # baseline, recovering sqrt(2) * se.
    got = check_regression.delta_stderr(baseline, {metric: 0.86}, metric)
    assert math.isclose(got, 0.03 * math.sqrt(2))
    # A missing task entry behaves the same way.
    got_none = check_regression.delta_stderr(baseline, None, metric)
    assert math.isclose(got_none, 0.03 * math.sqrt(2))


def test_delta_stderr_uses_run_stderr_when_baseline_missing():
    """A stderr-less baseline still widens the gate, off the run's stderr.

    The baseline is treated as exact, so only the run's side contributes: the
    width is 1x the run's stderr, not the sqrt(2)x a baseline-side stderr
    would give. This is the direction that moves `kimi-k3-tp32-ep.mmlu_pro`,
    whose shipped baseline carries no stderr.
    """
    metric = "exact_match,custom-extract"
    baseline = {metric: 0.88}
    results = {metric: 0.86, "exact_match_stderr,custom-extract": 0.02}
    # hypot(0.0, 0.02) == 0.02: the run's stderr covers its own side only.
    assert math.isclose(check_regression.delta_stderr(baseline, results, metric), 0.02)


def test_delta_stderr_zero_when_neither_side_reports():
    metric = "exact_match,strict_match"
    assert check_regression.delta_stderr({metric: 0.88}, {metric: 0.86}, metric) == 0.0


def test_load_eval_results_ignores_orphan_stderr(tmp_path):
    """A stderr with no matching accuracy value must not create an entry.

    Otherwise the task looks present but metric-less, and a missing-result
    FAIL silently degrades into a SKIP.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    with (results_dir / "results_x.json").open("w") as f:
        json.dump(
            {
                "results": {
                    "orphan": {"exact_match_stderr,strict_match": 0.01},
                    "good": {
                        "exact_match,strict_match": 0.88,
                        "exact_match_stderr,strict_match": 0.01,
                    },
                }
            },
            f,
        )

    loaded = check_regression.load_eval_results(results_dir)
    assert "orphan" not in loaded
    assert loaded["good"] == {
        "exact_match,strict_match": 0.88,
        "exact_match_stderr,strict_match": 0.01,
    }


def test_format_drop_range():
    assert check_regression.format_drop_range([0.015]) == "1.50pp"
    assert check_regression.format_drop_range([0.015, 0.015]) == "1.50pp"
    assert check_regression.format_drop_range([0.015, 0.0526]) == "1.50pp..5.26pp"
    assert (
        check_regression.format_drop_range([0.015, 0.0526], prefix="-")
        == "-1.50pp..-5.26pp"
    )


def test_eval_check_result_truthiness():
    """Truthy only when something failed; `main()` branches on this."""
    passing = check_regression.EvalCheckResult(
        [check_regression.EvalRow("t", "m", "PASS", 0.015)]
    )
    assert not passing
    assert passing.failed_count == 0

    failing = check_regression.EvalCheckResult(
        [check_regression.EvalRow("t", "m", "FAIL", 0.015)]
    )
    assert failing
    assert failing.failed_count == 1

    # An empty check is not a failure.
    assert not check_regression.EvalCheckResult([])


def test_eval_check_result_excludes_ungated_rows_from_range():
    """Rows with no baseline metric must not widen the reported range."""
    result = check_regression.EvalCheckResult(
        [
            check_regression.EvalRow("real", "m", "PASS", 0.05, gated=True),
            check_regression.EvalRow("bogus", "(no acc)", "SKIP", 0.015, gated=False),
        ]
    )
    assert result.allowed_drops == [0.05]


def test_check_eval_pass_exact():
    baseline = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.88,
            "exact_match_stderr,strict_match": 0.01,
        }
    }
    results = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.88,
        }
    }
    failed = check_regression.check_eval(results, baseline, tolerance=0.02)
    assert failed.failed_count == 0


def test_check_eval_scales_with_stderr():
    # baseline stderr 0.035, run reports none -> delta stderr is
    # sqrt(2) * 0.035 = 0.0495, allowed drop is 1.645 * 0.0495 = 0.0814.
    baseline = {
        "mmlu_llama_other": {
            "exact_match,strict_match": 0.88,
            "exact_match_stderr,strict_match": 0.035,
        }
    }
    # drop of 0.070 -> passes, still inside one-sided 95% noise
    results_pass = {
        "mmlu_llama_other": {
            "exact_match,strict_match": 0.81,
        }
    }
    assert (
        check_regression.check_eval(results_pass, baseline, tolerance=0.02).failed_count
        == 0
    )

    # drop of 0.090 -> fails, beyond 0.0814
    results_fail = {
        "mmlu_llama_other": {
            "exact_match,strict_match": 0.79,
        }
    }
    assert (
        check_regression.check_eval(results_fail, baseline, tolerance=0.02).failed_count
        == 1
    )


def test_check_eval_uses_noisier_run_stderr():
    # The run is much noisier than the baseline (e.g. fewer samples via
    # --limit). The gate must widen to match, not stay pinned to the baseline.
    baseline = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.88,
            "exact_match_stderr,strict_match": 0.005,
        }
    }
    noisy_run = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.83,
            "exact_match_stderr,strict_match": 0.050,
        }
    }
    # hypot(0.005, 0.050) = 0.05025, * 1.645 = 0.0827 -> a 0.05 drop passes.
    assert (
        check_regression.check_eval(noisy_run, baseline, tolerance=0.02).failed_count
        == 0
    )

    # Same drop, but a run as precise as the baseline -> hypot is 0.00707,
    # below the floor, so the 0.02 tolerance applies and 0.05 fails.
    quiet_run = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.83,
            "exact_match_stderr,strict_match": 0.005,
        }
    }
    assert (
        check_regression.check_eval(quiet_run, baseline, tolerance=0.02).failed_count
        == 1
    )


def test_check_eval_uses_tolerance_floor_when_stderr_is_smaller():
    # stderr 0.004 on both sides -> 1.645 * hypot = 0.0093, below the 0.02
    # floor, so the flat tolerance governs.
    baseline = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.88,
            "exact_match_stderr,strict_match": 0.004,
        }
    }
    results_pass = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.865,
            "exact_match_stderr,strict_match": 0.004,
        }
    }
    # drop of 0.015 -> passes (within 0.02 floor)
    assert (
        check_regression.check_eval(results_pass, baseline, tolerance=0.02).failed_count
        == 0
    )

    results_fail = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.855,
            "exact_match_stderr,strict_match": 0.004,
        }
    }
    # drop of 0.025 -> fails (> 0.02 floor)
    assert (
        check_regression.check_eval(results_fail, baseline, tolerance=0.02).failed_count
        == 1
    )


def test_check_eval_without_stderr():
    # When no stderr is provided, tolerance floor is used strictly
    baseline = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.88,
        }
    }
    results = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.865,
        }
    }
    # delta is -0.015, tolerance is 0.02 -> PASS
    assert (
        check_regression.check_eval(results, baseline, tolerance=0.02).failed_count == 0
    )

    # delta is -0.025, tolerance is 0.02 -> FAIL
    results["mmlu_llama"]["exact_match,strict_match"] = 0.855
    assert (
        check_regression.check_eval(results, baseline, tolerance=0.02).failed_count == 1
    )


def test_check_eval_real_humaneval_shape():
    # The pass_at_1 code benchmarks are the rows the stderr path actually
    # affects: N=164 makes stderr ~2.3pp, far above the 1.5pp default floor.
    baseline = {
        "humaneval_plus_tpu": {
            "pass_at_1,create_test": 0.9085365853658537,
            "pass_at_1_stderr,create_test": 0.022578813351856353,
        }
    }
    # hypot(0.0226, 0.0226) * 1.645 = 0.0525, so a 2.2pp drop is noise.
    results_pass = {
        "humaneval_plus_tpu": {
            "pass_at_1,create_test": 0.8865,
            "pass_at_1_stderr,create_test": 0.022578813351856353,
        }
    }
    assert (
        check_regression.check_eval(
            results_pass, baseline, tolerance=0.015
        ).failed_count
        == 0
    )

    # A 6pp drop is a real regression even at this sample size.
    results_fail = {
        "humaneval_plus_tpu": {
            "pass_at_1,create_test": 0.8485,
            "pass_at_1_stderr,create_test": 0.022578813351856353,
        }
    }
    assert (
        check_regression.check_eval(
            results_fail, baseline, tolerance=0.015
        ).failed_count
        == 1
    )


def test_check_eval_missing_task_fails():
    baseline = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.88,
            "exact_match_stderr,strict_match": 0.01,
        }
    }
    failed = check_regression.check_eval({}, baseline, tolerance=0.02)
    assert failed.failed_count == 1
    assert failed.failed_rows[0].task == "mmlu_llama"


# Baselines that predate stderr recording. The two mmmu_pro entries report no
# stderr on either side (evalscope task limitation), so they stay on the flat
# tolerance floor; the kimi-k3 entry was hand-written rather than produced by
# --calibrate, so it gets only 1x the run's stderr (hypot(0, se_run)) instead
# of the full sqrt(2) quadrature. Shrink this set, never grow it.
_BASELINES_MISSING_STDERR = {
    "gemma-4-26b-fp8-dp4-tp2-multimodal.mmmu_pro.baseline.json:mmmu_pro",
    "gemma-4-31b-fp8-dp4-tp2-multimodal.mmmu_pro.baseline.json:mmmu_pro",
    "kimi-k3-tp32-ep.mmlu_pro.baseline.json:mmlu_pro",
}


def test_shipped_baselines_are_well_formed():
    """Every checked-in eval baseline must carry an acc key and its stderr.

    Guards against hand-written baselines, which miss the baseline half of the
    quadrature (or all of it, for tasks whose runner emits no stderr either).
    """
    files = sorted(_BASELINE_DIR.glob("*.baseline.json"))
    assert files, f"no baselines found under {_BASELINE_DIR}"
    missing = set()
    for path in files:
        with path.open() as fh:
            baseline = json.load(fh)
        for task, metrics in baseline.items():
            if task.startswith("_"):
                continue  # metadata block, filtered by main()
            metric = check_regression.primary_metric(metrics)
            assert metric is not None, f"{path.name}:{task} has no known acc key"
            if check_regression.primary_stderr(metrics, metric.name) is None:
                missing.add(f"{path.name}:{task}")

    new_gaps = missing - _BASELINES_MISSING_STDERR
    assert not new_gaps, (
        "new baseline(s) missing stderr, regenerate with --calibrate: "
        + ", ".join(sorted(new_gaps))
    )

    fixed = _BASELINES_MISSING_STDERR - missing
    assert not fixed, (
        "these baselines now have stderr; drop them from "
        "_BASELINES_MISSING_STDERR: " + ", ".join(sorted(fixed))
    )


def test_check_regression_cli(tmp_path, monkeypatch):
    baseline_file = tmp_path / "test.baseline.json"
    baseline_data = {
        "_source": {
            "description": "metadata block, must not be gated as a task",
            "url": "https://example.invalid/run",
        },
        "mmlu_llama": {
            "exact_match,strict_match": 0.88,
            "exact_match_stderr,strict_match": 0.01,
        },
        "mmlu_llama_other": {
            "exact_match,strict_match": 0.88,
            "exact_match_stderr,strict_match": 0.035,
        },
    }
    with baseline_file.open("w") as f:
        json.dump(baseline_data, f)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    results_file = results_dir / "results_mmlu_llama.json"
    # mmlu_llama: drop 0.01, allowed 1.645 * hypot(0.01, 0.01) = 0.0233
    # mmlu_llama_other: drop 0.03, allowed 1.645 * hypot(0.035, 0.035) = 0.0814
    results_data = {
        "results": {
            "mmlu_llama": {
                "exact_match,strict_match": 0.87,
                "exact_match_stderr,strict_match": 0.01,
            },
            "mmlu_llama_other": {
                "exact_match,strict_match": 0.85,
                "exact_match_stderr,strict_match": 0.035,
            },
        }
    }
    with results_file.open("w") as f:
        json.dump(results_data, f)

    # CLI test passing with default tolerance (0.015)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_regression.py",
            "--mode",
            "eval",
            "--results-dir",
            str(results_dir),
            "--baseline",
            str(baseline_file),
        ],
    )
    ret = check_regression.main()
    assert ret == 0

    # CLI test failing when other drops beyond its combined stderr (0.0814)
    results_data["results"]["mmlu_llama_other"]["exact_match,strict_match"] = 0.79
    with results_file.open("w") as f:
        json.dump(results_data, f)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_regression.py",
            "--mode",
            "eval",
            "--results-dir",
            str(results_dir),
            "--baseline",
            str(baseline_file),
        ],
    )
    ret = check_regression.main()
    assert ret == 1


def test_check_regression_cli_output(tmp_path, monkeypatch, capsys):
    baseline_file = tmp_path / "test.baseline.json"
    baseline_data = {
        "mmlu_llama_other": {
            "exact_match,strict_match": 0.88,
            "exact_match_stderr,strict_match": 0.035,
        }
    }
    with baseline_file.open("w") as f:
        json.dump(baseline_data, f)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    results_file = results_dir / "results_mmlu_llama.json"
    results_data = {
        "results": {
            "mmlu_llama_other": {
                "exact_match,strict_match": 0.79,
                "exact_match_stderr,strict_match": 0.035,
            }
        }
    }
    with results_file.open("w") as f:
        json.dump(results_data, f)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_regression.py",
            "--mode",
            "eval",
            "--results-dir",
            str(results_dir),
            "--baseline",
            str(baseline_file),
        ],
    )
    ret = check_regression.main()
    assert ret == 1
    captured = capsys.readouterr()
    assert (
        "| task | metric | actual | baseline | stderr(diff) |"
        " allowed drop | delta | status |" in captured.out
    )
    assert "FAILED: 1 task(s) regressed beyond 8.14pp" in captured.err


# --- Eval: gate on the mean of several runs ---


def test_aggregate_eval_runs_single_run_keeps_reported_stderr():
    runs = [{"mmlu_pro": {"acc,none": 0.8, "acc_stderr,none": 0.02}}]
    assert check_regression.aggregate_eval_runs(runs).entries == {
        "mmlu_pro": {"acc,none": 0.8, "acc_stderr,none": 0.02}
    }


def test_aggregate_eval_runs_single_run_without_stderr():
    runs = [{"mmlu_pro": {"acc,none": 0.8}}]
    assert check_regression.aggregate_eval_runs(runs).entries == {
        "mmlu_pro": {"acc,none": 0.8}
    }


@pytest.mark.parametrize(
    "runs_data",
    [
        ((0.80, 0.03), (0.80, 0.04), (0.86, 0.08)),
        # Any number of runs: five keep the reported stderr like three do.
        ((0.80, 0.03), (0.80, 0.03), (0.80, 0.04), (0.80, 0.07), (0.90, 0.08)),
    ],
)
def test_aggregate_eval_runs_averages_value_and_reported_stderr(runs_data):
    # Both the value and the reported stderr use the mean. Asymmetric inputs
    # separate mean from median: the values have mean 0.82 and median 0.80,
    # the stderrs mean 0.05 and median 0.04.
    runs = [
        {"mmlu_pro": {"acc,none": value, "acc_stderr,none": stderr}}
        for value, stderr in runs_data
    ]
    assert check_regression.aggregate_eval_runs(runs).entries == {
        "mmlu_pro": {
            "acc,none": pytest.approx(0.82, abs=1e-12),
            "acc_stderr,none": pytest.approx(0.05, abs=1e-12),
        }
    }


def test_aggregate_eval_runs_needs_stderr_from_every_run():
    runs = [
        {"mmlu_pro": {"acc,none": 0.80, "acc_stderr,none": 0.05}},
        {"mmlu_pro": {"acc,none": 0.90}},
    ]
    assert check_regression.aggregate_eval_runs(runs).entries == {
        "mmlu_pro": {"acc,none": pytest.approx(0.85, abs=1e-12)}
    }


def test_aggregate_eval_runs_returns_metric_missing_from_a_run(capsys):
    runs = [
        {"mmlu_pro": {"acc,none": 0.8}, "mbpp": {"pass_at_1,none": 0.6}},
        {"mmlu_pro": {"acc,none": 0.9}},
    ]
    entries, dropped = check_regression.aggregate_eval_runs(runs)
    assert entries == {"mmlu_pro": {"acc,none": pytest.approx(0.85, abs=1e-12)}}
    assert dropped == [("mbpp", "pass_at_1,none")]
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_aggregate_eval_runs_no_runs():
    assert check_regression.aggregate_eval_runs([]) == ({}, [])


def _write_eval_run(run_dir, task, metrics):
    run_dir.mkdir()
    (run_dir / f"results_{task}.json").write_text(
        json.dumps({"results": {task: metrics}})
    )
    return run_dir


def _run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["check_regression.py", *argv])
    return check_regression.main()


def test_calibrate_eval_uses_mean_for_any_listed_task(tmp_path):
    # Asymmetric inputs separate mean (0.50) from the previous median (0.40).
    dirs = [
        _write_eval_run(tmp_path / f"run{i}", "gpqa_diamond", {"acc,none": value})
        for i, value in enumerate((0.40, 0.40, 0.70))
    ]
    assert check_regression.calibrate_eval(dirs, None) == {
        "gpqa_diamond": {"acc,none": pytest.approx(0.50, abs=1e-12)}
    }
    assert check_regression.calibrate_eval(dirs, {"gpqa_diamond", "mmlu_pro"}) == {
        "gpqa_diamond": {"acc,none": pytest.approx(0.50, abs=1e-12)}
    }


def test_calibrate_eval_rejects_bad_runs(tmp_path, monkeypatch, capsys):
    good = _write_eval_run(tmp_path / "good", "mmlu_pro", {"acc,none": 0.5})
    other = _write_eval_run(tmp_path / "other", "mbpp", {"acc,none": 0.5})
    both = tmp_path / "both"
    both.mkdir()
    (both / "results_x.json").write_text(
        json.dumps(
            {"results": {"mmlu_pro": {"acc,none": 0.5}, "mbpp": {"acc,none": 0.5}}}
        )
    )
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError, match="no selected eval results under .*empty"):
        check_regression.calibrate_eval([good, empty], None)
    # The task filter removes the only task of the second run.
    with pytest.raises(ValueError, match="no selected eval results under .*other"):
        check_regression.calibrate_eval([good, other], {"mmlu_pro"})
    with pytest.raises(ValueError, match="not in every run: mbpp acc,none"):
        check_regression.calibrate_eval([good, both], None)

    argv = ["--mode", "eval", "--calibrate", "--task-filter", "all"]
    ret = _run_cli(
        monkeypatch, *argv, "--results-dir", str(good), "--results-dir", str(empty)
    )
    assert ret == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"ERROR: no selected eval results under {empty}" in captured.err

    ret = _run_cli(
        monkeypatch, *argv, "--results-dir", str(good), "--results-dir", str(both)
    )
    assert ret == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "ERROR: not in every run: mbpp acc,none\n"


def test_cli_calibrate_eval_writes_mean_of_selected_tasks(
    tmp_path, monkeypatch, capsys
):
    # `extra` is in the first run only; the task filter removes it before the
    # runs are compared, so it does not count as a metric lost by a run.
    run0 = tmp_path / "run0"
    run0.mkdir()
    (run0 / "results_x.json").write_text(
        json.dumps(
            {"results": {"mmlu_pro": {"acc,none": 0.5}, "extra": {"acc,none": 0.1}}}
        )
    )
    run1 = _write_eval_run(tmp_path / "run1", "mmlu_pro", {"acc,none": 0.75})
    ret = _run_cli(
        monkeypatch,
        "--mode", "eval", "--calibrate", "--task-filter", "mmlu_pro",
        "--results-dir", str(run0), "--results-dir", str(run1),
    )  # fmt: skip
    assert ret == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"mmlu_pro": {"acc,none": 0.625}}
    assert captured.err == ""


@pytest.mark.parametrize(
    ("run_entry", "num_runs", "expected_allowed_drop", "expected_status"),
    [
        # One run: 1.645 * hypot(0.02, 0.02) = 0.0465, drop 0.04 passes.
        ({"acc,none": 0.86, "acc_stderr,none": 0.02}, 1, 0.0465276262, "PASS"),
        # Several runs gate exactly as wide as one, with or without a run
        # stderr, also when the runs agree exactly.
        ({"acc,none": 0.86, "acc_stderr,none": 0.02}, 2, 0.0465276262, "PASS"),
        ({"acc,none": 0.86}, 2, 0.0465276262, "PASS"),
        ({"acc,none": 0.86, "acc_stderr,none": 0.02}, 5, 0.0465276262, "PASS"),
    ],
)
def test_check_eval_gate_width_by_number_of_runs(
    run_entry, num_runs, expected_allowed_drop, expected_status
):
    baseline = {"mmlu_pro": {"acc,none": 0.90, "acc_stderr,none": 0.02}}
    results = check_regression.aggregate_eval_runs(
        [{"mmlu_pro": run_entry}] * num_runs
    ).entries
    row = check_regression.check_eval(results, baseline, 0.015).rows[0]
    assert row.allowed_drop == pytest.approx(expected_allowed_drop, abs=1e-9)
    assert row.status == expected_status


def test_check_eval_one_collapsed_run_of_five_fails():
    # The mean is 0.64. The spread of the runs does not widen the gate: the
    # allowed drop is 1.645 * hypot(0.011, 0.011) = 0.0256, as for one run.
    baseline = {"mmlu_pro": {"acc,none": 0.80, "acc_stderr,none": 0.011}}
    results = check_regression.aggregate_eval_runs(
        [
            {"mmlu_pro": {"acc,none": value, "acc_stderr,none": 0.011}}
            for value in (0.80, 0.80, 0.80, 0.80, 0.00)
        ]
    ).entries
    row = check_regression.check_eval(results, baseline, 0.015).rows[0]
    assert row.actual == pytest.approx(0.64, abs=1e-12)
    assert row.allowed_drop == pytest.approx(0.0255901935, abs=1e-9)
    assert row.status == "FAIL"


def test_check_eval_missing_primary_metric_fails_as_no_result():
    # One run lost the baseline's primary metric, so the mean keeps only
    # acc,none. That must not stand in for exact_match,strict_match.
    baseline = {"mmlu_llama": {"exact_match,strict_match": 0.90, "acc,none": 0.90}}
    results = check_regression.aggregate_eval_runs(
        [
            {"mmlu_llama": {"exact_match,strict_match": 0.90, "acc,none": 0.95}},
            {"mmlu_llama": {"acc,none": 0.95}},
        ]
    ).entries
    assert results == {"mmlu_llama": {"acc,none": 0.95}}
    row = check_regression.check_eval(results, baseline, 0.015).rows[0]
    assert (row.metric, row.status) == ("(no result)", "FAIL")


@pytest.mark.parametrize(
    ("run_values", "expected_exit"),
    [
        # mean 0.89, drop 0.01 within the flat 0.015 tolerance
        ((0.88, 0.90), 0),
        # mean 0.88, drop 0.02 beyond the flat 0.015 tolerance
        ((0.87, 0.89), 1),
        ((0.89,), 0),
        # Five runs gate like two: mean 0.88, and their spread does not widen
        # the flat 0.015 tolerance.
        ((0.84, 0.86, 0.88, 0.90, 0.92), 1),
    ],
)
def test_cli_eval_gates_on_mean_of_runs(
    tmp_path, monkeypatch, run_values, expected_exit
):
    baseline_file = tmp_path / "t.baseline.json"
    baseline_file.write_text(json.dumps({"mmlu_pro": {"acc,none": 0.90}}))
    argv = ["--mode", "eval", "--baseline", str(baseline_file)]
    for i, value in enumerate(run_values):
        run_dir = _write_eval_run(tmp_path / f"run{i}", "mmlu_pro", {"acc,none": value})
        argv += ["--results-dir", str(run_dir)]
    assert _run_cli(monkeypatch, *argv) == expected_exit


def test_cli_eval_fails_when_one_run_lacks_the_task(tmp_path, monkeypatch, capsys):
    baseline_file = tmp_path / "t.baseline.json"
    baseline_file.write_text(json.dumps({"mmlu_pro": {"acc,none": 0.90}}))
    run0 = _write_eval_run(tmp_path / "run0", "mmlu_pro", {"acc,none": 0.90})
    run1 = _write_eval_run(tmp_path / "run1", "other", {"acc,none": 0.90})
    ret = _run_cli(
        monkeypatch,
        "--mode", "eval", "--baseline", str(baseline_file),
        "--results-dir", str(run0), "--results-dir", str(run1),
    )  # fmt: skip
    assert ret == 1
    captured = capsys.readouterr()
    assert "| `mmlu_pro` | (no result) |" in captured.out
    assert captured.err.splitlines()[:2] == [
        "Warning: mmlu_pro acc,none is not in every run; dropped.",
        "Warning: other acc,none is not in every run; dropped.",
    ]


def test_cli_eval_errors_when_a_run_dir_is_empty(tmp_path, monkeypatch, capsys):
    baseline_file = tmp_path / "t.baseline.json"
    baseline_file.write_text(json.dumps({"mmlu_pro": {"acc,none": 0.90}}))
    run0 = _write_eval_run(tmp_path / "run0", "mmlu_pro", {"acc,none": 0.90})
    empty = tmp_path / "empty"
    empty.mkdir()
    ret = _run_cli(
        monkeypatch,
        "--mode", "eval", "--baseline", str(baseline_file),
        "--results-dir", str(run0), "--results-dir", str(empty),
    )  # fmt: skip
    assert ret == 2
    assert f"ERROR: no results_*.json found under {empty}" in capsys.readouterr().err


def test_cli_perf_rejects_multiple_results_dirs(tmp_path, monkeypatch, capsys):
    baseline_file = tmp_path / "perf.baseline.json"
    baseline_file.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        _run_cli(
            monkeypatch,
            "--mode", "perf", "--baseline", str(baseline_file),
            "--results-dir", str(tmp_path / "r0"),
            "--results-dir", str(tmp_path / "r1"),
        )  # fmt: skip
    assert exc.value.code == 2
    assert (
        "--results-dir may only be given once in perf check mode"
        in capsys.readouterr().err
    )
