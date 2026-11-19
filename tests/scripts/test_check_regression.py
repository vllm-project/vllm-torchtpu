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


# --- Rule 1: accuracy compares the mean of several runs ---


def test_aggregate_eval_runs_single_run_keeps_reported_stderr():
    runs = [{"mmlu_pro": {"acc,none": 0.8, "acc_stderr,none": 0.02}}]
    assert check_regression.aggregate_eval_runs(runs) == {
        "mmlu_pro": {"acc,none": 0.8, "acc_stderr,none": 0.02, "num_runs": 1}
    }


def test_aggregate_eval_runs_single_run_without_stderr():
    runs = [{"mmlu_pro": {"acc,none": 0.8}}]
    assert check_regression.aggregate_eval_runs(runs) == {
        "mmlu_pro": {"acc,none": 0.8, "num_runs": 1}
    }


def test_aggregate_eval_runs_below_min_runs_keeps_reported_stderr():
    # Three runs are fewer than MIN_RUNS_FOR_EMPIRICAL_STDERR (5): the value is
    # averaged, the stderr stays the per-run reported one (not the 0.0115
    # spread of the three values).
    runs = [
        {"mmlu_pro": {"acc,none": value, "acc_stderr,none": 0.05}}
        for value in (0.80, 0.82, 0.84)
    ]
    assert check_regression.MIN_RUNS_FOR_EMPIRICAL_STDERR == 5
    assert check_regression.aggregate_eval_runs(runs) == {
        "mmlu_pro": {
            "acc,none": pytest.approx(0.82, abs=1e-12),
            "acc_stderr,none": pytest.approx(0.05, abs=1e-12),
            "num_runs": 3,
        }
    }


def test_aggregate_eval_runs_below_min_runs_needs_stderr_from_every_run():
    runs = [
        {"mmlu_pro": {"acc,none": 0.80, "acc_stderr,none": 0.05}},
        {"mmlu_pro": {"acc,none": 0.90}},
    ]
    assert check_regression.aggregate_eval_runs(runs) == {
        "mmlu_pro": {"acc,none": pytest.approx(0.85, abs=1e-12), "num_runs": 2}
    }


def test_aggregate_eval_runs_empirical_stderr_from_five_runs():
    # stdev(0.80 .. 0.84) = sqrt(0.001 / 4) = 0.0158114, / sqrt(5) = 0.0070711.
    # It replaces the per-run reported 0.05.
    runs = [
        {"mmlu_pro": {"acc,none": value, "acc_stderr,none": 0.05}}
        for value in (0.80, 0.81, 0.82, 0.83, 0.84)
    ]
    assert check_regression.aggregate_eval_runs(runs) == {
        "mmlu_pro": {
            "acc,none": pytest.approx(0.82, abs=1e-12),
            "acc_stderr,none": pytest.approx(0.007071067811865475, abs=1e-12),
            "num_runs": 5,
        }
    }


def test_aggregate_eval_runs_five_identical_runs_have_zero_stderr():
    runs = [{"mmlu_pro": {"acc,none": 0.5, "acc_stderr,none": 0.03}}] * 5
    assert check_regression.aggregate_eval_runs(runs) == {
        "mmlu_pro": {"acc,none": 0.5, "acc_stderr,none": 0.0, "num_runs": 5}
    }


def test_aggregate_eval_runs_drops_metric_missing_from_a_run(capsys):
    runs = [
        {"mmlu_pro": {"acc,none": 0.8}, "mbpp": {"pass_at_1,none": 0.6}},
        {"mmlu_pro": {"acc,none": 0.9}},
    ]
    assert check_regression.aggregate_eval_runs(runs) == {
        "mmlu_pro": {"acc,none": pytest.approx(0.85, abs=1e-12), "num_runs": 2}
    }
    assert "mbpp pass_at_1,none is in 1 of 2 runs" in capsys.readouterr().err


def test_aggregate_eval_runs_no_runs():
    assert check_regression.aggregate_eval_runs([]) == {}


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
    dirs = [
        _write_eval_run(tmp_path / f"run{i}", "gpqa_diamond", {"acc,none": value})
        for i, value in enumerate((0.40, 0.50))
    ]
    assert check_regression.calibrate_eval(dirs, None) == {
        "gpqa_diamond": {"acc,none": pytest.approx(0.45, abs=1e-12), "num_runs": 2}
    }
    assert check_regression.calibrate_eval(dirs, {"gpqa_diamond", "mmlu_pro"}) == {
        "gpqa_diamond": {"acc,none": pytest.approx(0.45, abs=1e-12), "num_runs": 2}
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


@pytest.mark.parametrize(
    ("run_entry", "num_runs", "expected_allowed_drop", "expected_status"),
    [
        # One run: 1.645 * hypot(0.02, 0.02) = 0.0465, drop 0.04 passes.
        ({"acc,none": 0.86, "acc_stderr,none": 0.02}, 1, 0.0465276262, "PASS"),
        # Two runs gate exactly as wide as one, with or without a run stderr.
        ({"acc,none": 0.86, "acc_stderr,none": 0.02}, 2, 0.0465276262, "PASS"),
        ({"acc,none": 0.86}, 2, 0.0465276262, "PASS"),
        # Five identical runs have stderr 0.0, so only the baseline's 0.02
        # counts: 1.645 * hypot(0.02, 0.0) = 0.0329, and the same drop fails.
        ({"acc,none": 0.86, "acc_stderr,none": 0.02}, 5, 0.0329, "FAIL"),
    ],
)
def test_check_eval_gate_width_by_number_of_runs(
    run_entry, num_runs, expected_allowed_drop, expected_status
):
    baseline = {"mmlu_pro": {"acc,none": 0.90, "acc_stderr,none": 0.02}}
    results = check_regression.aggregate_eval_runs([{"mmlu_pro": run_entry}] * num_runs)
    row = check_regression.check_eval(results, baseline, 0.015).rows[0]
    assert row.allowed_drop == pytest.approx(expected_allowed_drop, abs=1e-9)
    assert row.status == expected_status


def test_check_eval_baseline_from_five_identical_runs():
    # A baseline with stderr 0.0 is exact: the run's own 0.02 sets the width,
    # 1.645 * 0.02 = 0.0329, and a run without a stderr gets the flat 0.015.
    baseline = check_regression.aggregate_eval_runs(
        [{"mmlu_pro": {"acc,none": 0.9}}] * 5
    )
    with_stderr = {"mmlu_pro": {"acc,none": 0.88, "acc_stderr,none": 0.02}}
    row = check_regression.check_eval(with_stderr, baseline, 0.015).rows[0]
    assert (row.allowed_drop, row.status) == (pytest.approx(0.0329, abs=1e-12), "PASS")
    row = check_regression.check_eval({"mmlu_pro": {"acc,none": 0.88}}, baseline, 0.015)
    assert (row.rows[0].allowed_drop, row.rows[0].status) == (0.015, "FAIL")


@pytest.mark.parametrize(
    ("run_values", "expected_exit"),
    [
        # mean 0.89, drop 0.01 within the flat 0.015 tolerance
        ((0.88, 0.90), 0),
        # mean 0.88, drop 0.02 beyond the flat 0.015 tolerance
        ((0.87, 0.89), 1),
        ((0.89,), 0),
        # Five runs, mean 0.88, stdev 0.0316, stderr 0.01414: allowed drop is
        # 1.645 * 0.01414 = 0.02326, so the same 0.02 drop passes.
        ((0.84, 0.86, 0.88, 0.90, 0.92), 0),
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
    assert "| `mmlu_pro` | (no result) |" in capsys.readouterr().out


def test_cli_eval_errors_when_a_run_dir_is_empty(tmp_path, monkeypatch):
    baseline_file = tmp_path / "t.baseline.json"
    baseline_file.write_text(json.dumps({"mmlu_pro": {"acc,none": 0.90}}))
    run0 = _write_eval_run(tmp_path / "run0", "mmlu_pro", {"acc,none": 0.90})
    (tmp_path / "empty").mkdir()
    ret = _run_cli(
        monkeypatch,
        "--mode", "eval", "--baseline", str(baseline_file),
        "--results-dir", str(run0), "--results-dir", str(tmp_path / "empty"),
    )  # fmt: skip
    assert ret == 2


# --- Rule 2: a frozen baseline changes only with owner approval ---

_TASK = {"mmlu_pro": {"acc,none": 0.90}}
_RAISED = {"mmlu_pro": {"acc,none": 0.93}}
_FROZEN = {"state": "frozen", "owner": "alice"}
_APPROVAL_1 = {"approver": "alice", "reference": "https://example.invalid/pull/1"}
_APPROVAL_2 = {"approver": "alice", "reference": "https://example.invalid/pull/2"}
_NO_APPROVAL = (
    "frozen baseline changed without a new _policy.approval record from owner 'alice'"
)
_WRONG_APPROVER = (
    "frozen baseline changed with approval from 'bob', but its owner is 'alice'"
)


def test_parse_policy_defaults_to_development():
    assert check_regression.parse_policy(_TASK) == check_regression.BaselinePolicy(
        state="development", owner=None, approval=None
    )


def test_parse_policy_frozen_with_approval():
    policy = check_regression.parse_policy(
        {"_policy": {**_FROZEN, "approval": _APPROVAL_1}}
    )
    assert policy == check_regression.BaselinePolicy(
        state="frozen",
        owner="alice",
        approval=("alice", "https://example.invalid/pull/1"),
    )


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ("frozen", "_policy must be an object"),
        ({"state": "golden"}, "_policy.state must be 'development' or 'frozen'"),
        ({"state": "frozen"}, "_policy.owner must be a non-empty string"),
        ({"state": "frozen", "owner": " "}, "_policy.owner must be a non-empty"),
        ({"approval": "alice"}, "_policy.approval must be an object"),
        (
            {"approval": {"approver": "alice"}},
            "_policy.approval.reference must be a non-empty string",
        ),
    ],
)
def test_parse_policy_rejects_malformed_block(policy, message):
    with pytest.raises(ValueError, match=message):
        check_regression.parse_policy({"_policy": policy})


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        # New file, and development baselines, change freely.
        (None, _RAISED, None),
        (_TASK, _RAISED, None),
        ({**_TASK, "_policy": {"state": "development"}}, _RAISED, None),
        # Freezing a development baseline needs no approval.
        (_TASK, {**_RAISED, "_policy": _FROZEN}, None),
        # A frozen baseline may change metadata, but not gated values.
        (
            {**_TASK, "_policy": _FROZEN},
            {
                **_TASK,
                "_policy": _FROZEN,
                "_source": {"url": "https://example.invalid"},
            },
            None,
        ),
        ({**_TASK, "_policy": _FROZEN}, {**_RAISED, "_policy": _FROZEN}, _NO_APPROVAL),
        # Dropping the policy block or the task does not unfreeze it.
        ({**_TASK, "_policy": _FROZEN}, _RAISED, _NO_APPROVAL),
        ({**_TASK, "_policy": _FROZEN}, {"_policy": _FROZEN}, _NO_APPROVAL),
        (
            {**_TASK, "_policy": _FROZEN},
            {**_TASK, "_policy": {"state": "development"}},
            _NO_APPROVAL,
        ),
        # Handing the file to another owner is a change too.
        (
            {**_TASK, "_policy": _FROZEN},
            {**_TASK, "_policy": {"state": "frozen", "owner": "bob"}},
            _NO_APPROVAL,
        ),
        # An approval that was already in the old file does not count again.
        (
            {**_TASK, "_policy": {**_FROZEN, "approval": _APPROVAL_1}},
            {**_RAISED, "_policy": {**_FROZEN, "approval": _APPROVAL_1}},
            _NO_APPROVAL,
        ),
        (
            {**_TASK, "_policy": {**_FROZEN, "approval": _APPROVAL_1}},
            {**_RAISED, "_policy": {**_FROZEN, "approval": _APPROVAL_2}},
            None,
        ),
        (
            {**_TASK, "_policy": _FROZEN},
            {
                **_RAISED,
                "_policy": {
                    "state": "frozen",
                    "owner": "bob",
                    "approval": {
                        "approver": "bob",
                        "reference": "https://example.invalid/3",
                    },
                },
            },
            _WRONG_APPROVER,
        ),
    ],
)
def test_check_baseline_update(old, new, expected):
    assert check_regression.check_baseline_update(old, new) == expected


def test_cli_update_mode(tmp_path, monkeypatch, capsys):
    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    old_file.write_text(json.dumps({**_TASK, "_policy": _FROZEN}))
    new_file.write_text(json.dumps({**_RAISED, "_policy": _FROZEN}))
    argv = ["--mode", "update", "--baseline", str(new_file)]

    assert _run_cli(monkeypatch, *argv, "--previous-baseline", str(old_file)) == 1
    assert f"FAILED: {new_file}: {_NO_APPROVAL}" in capsys.readouterr().err

    # Without a previous file the same content is a new baseline.
    assert _run_cli(monkeypatch, *argv) == 0

    new_file.write_text(
        json.dumps({**_RAISED, "_policy": {**_FROZEN, "approval": _APPROVAL_1}})
    )
    assert _run_cli(monkeypatch, *argv, "--previous-baseline", str(old_file)) == 0

    new_file.write_text(json.dumps({**_RAISED, "_policy": {"state": "frozen"}}))
    assert _run_cli(monkeypatch, *argv, "--previous-baseline", str(old_file)) == 2
    assert "ERROR: _policy.owner must be a non-empty string" in capsys.readouterr().err


def test_cli_update_mode_rejects_perf_flags(tmp_path, monkeypatch, capsys):
    """Speed gate and chip flags are rejected in update mode."""
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(json.dumps(_TASK))
    with pytest.raises(SystemExit) as exc_info:
        _run_cli(
            monkeypatch,
            "--mode",
            "update",
            "--baseline",
            str(baseline_file),
            "--chips",
            "4",
        )
    assert exc_info.value.code == 2
    assert "speed gate and chip flags are only valid in perf check mode" in (
        capsys.readouterr().err
    )


@pytest.mark.parametrize("content", [None, "[]", "{not json"])
def test_cli_update_mode_unreadable_file_exits_2(
    tmp_path, monkeypatch, capsys, content
):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_TASK))
    bad = tmp_path / "bad.json"
    if content is not None:
        bad.write_text(content)
    for baseline, previous in ((bad, good), (good, bad)):
        ret = _run_cli(
            monkeypatch,
            "--mode", "update", "--baseline", str(baseline),
            "--previous-baseline", str(previous),
        )  # fmt: skip
        assert ret == 2
        assert "ERROR: " in capsys.readouterr().err


def test_shipped_baselines_have_valid_policy():
    """Every checked-in perf and eval baseline parses under the policy schema."""
    files = sorted(_BASELINE_DIR.parent.rglob("*.baseline.json"))
    assert len(files) > 40
    for path in files:
        check_regression.parse_policy(json.loads(path.read_text()))


# --- R6: advisory roofline reporting ---

_ROOFLINE_TARGETS_PATH = _BASELINE_DIR.parent / "perf" / "roofline_targets.json"


def _write_perf_cell(results_dir, cell_key, **metrics):
    results_dir.mkdir(exist_ok=True)
    payload = {"completed": 64, "num_prompts": 64, **metrics}
    (results_dir / f"{cell_key}.json").write_text(json.dumps(payload))


def test_shipped_roofline_targets_are_well_formed():
    targets = check_regression.load_roofline_targets(_ROOFLINE_TARGETS_PATH)
    assert "qwen3.5-397b-fp8-tp1-dp8-ep" in targets.targets
    assert "kimi-k3-tp32-ep" in targets.targets
    assert "qwen3.5-397b-fp8-mtp" in targets.not_modeled
    kimi = targets.targets["kimi-k3-tp32-ep"]
    assert kimi.num_chips == 16
    assert kimi.cells["isl8192_osl1024_c1"].status == "unverified"
    assert kimi.cells["isl8192_osl1024_c1"].output_token_throughput_per_chip == 12.54


@pytest.mark.parametrize(
    ("chips", "prefill", "decode", "expected"),
    [
        (None, None, None, None),
        (8, None, None, 8),
        (None, 4, 8, 12),
    ],
)
def test_resolve_physical_chips(chips, prefill, decode, expected):
    assert check_regression.resolve_physical_chips(chips, prefill, decode) == expected


@pytest.mark.parametrize(
    ("chips", "prefill", "decode", "message"),
    [
        (0, None, None, "chips must be a positive integer"),
        (True, None, None, "chips must be a positive integer"),
        (None, 4, None, "--prefill-chips and --decode-chips must be provided together"),
        (None, None, 4, "--prefill-chips and --decode-chips must be provided together"),
        (8, 4, 4, "--chips cannot be combined with --prefill-chips/--decode-chips"),
    ],
)
def test_resolve_physical_chips_rejects_invalid(chips, prefill, decode, message):
    with pytest.raises(ValueError, match=message):
        check_regression.resolve_physical_chips(chips, prefill, decode)


def test_cli_perf_roofline_advisory_ok_and_unverified(tmp_path, monkeypatch, capsys):
    # qwen3.5-397b-fp8-tp1-dp8-ep isl1024_osl8192_c64 roofline is 1282.53 tok/s/chip.
    # Total output_throughput 4104.096 across 4 physical chips = 1026.024 tok/s/chip
    # = 80.00% of roofline; even at 50% (2565.06 total = 641.265/chip) advisory exits 0.
    run_dir = tmp_path / "perf_ok"
    _write_perf_cell(run_dir, "isl1024_osl8192_c64", output_throughput=2565.06)
    _write_perf_cell(run_dir, "isl512_osl512_c1", output_throughput=100.0)
    ret = _run_cli(
        monkeypatch,
        "--mode",
        "perf",
        "--results-dir",
        str(run_dir),
        "--roofline-targets",
        str(_ROOFLINE_TARGETS_PATH),
        "--config-name",
        "qwen3.5-397b-fp8-tp1-dp8-ep",
        "--chips",
        "4",
    )
    assert ret == 0
    out = capsys.readouterr().out
    assert (
        "| `isl1024_osl8192_c64` | output_token_throughput_per_chip |"
        " 641.26 | 1282.53 | 50.00% | ADVISORY |" in out
    )
    assert "| `isl512_osl512_c1` | (no roofline cell) |" in out

    # kimi-k3-tp32-ep isl8192_osl1024_c1 is status=="unverified" (12.54 tok/s/chip).
    # Disaggregated 8P + 8D = 16 physical chips, total 150.48 -> 9.405/chip = 75.00%.
    kimi_dir = tmp_path / "perf_kimi"
    _write_perf_cell(kimi_dir, "isl8192_osl1024_c1", output_throughput=150.48)
    ret = _run_cli(
        monkeypatch,
        "--mode",
        "perf",
        "--results-dir",
        str(kimi_dir),
        "--roofline-targets",
        str(_ROOFLINE_TARGETS_PATH),
        "--config-name",
        "kimi-k3-tp32-ep",
        "--prefill-chips",
        "8",
        "--decode-chips",
        "8",
    )
    assert ret == 0
    out = capsys.readouterr().out
    assert (
        "| `isl8192_osl1024_c1` | output_token_throughput_per_chip |"
        " 9.40 | 12.54 | 75.00% | ADVISORY (unverified) |" in out
    )


def test_cli_perf_roofline_not_modeled_and_unknown_config(
    tmp_path, monkeypatch, capsys
):
    empty_dir = tmp_path / "disagg"
    empty_dir.mkdir()
    ret = _run_cli(
        monkeypatch,
        "--mode",
        "perf",
        "--results-dir",
        str(empty_dir),
        "--roofline-targets",
        str(_ROOFLINE_TARGETS_PATH),
        "--config-name",
        "qwen3.5-397b-fp8-1p1d",
    )
    assert ret == 0
    out = capsys.readouterr().out
    assert "not modeled" in out
    assert "disaggregated_pd_kv_transfer_not_modeled" in out

    ret = _run_cli(
        monkeypatch,
        "--mode",
        "perf",
        "--results-dir",
        str(empty_dir),
        "--roofline-targets",
        str(_ROOFLINE_TARGETS_PATH),
        "--config-name",
        "nonexistent-config",
    )
    assert ret == 2
    assert "not found in roofline targets" in capsys.readouterr().err


def test_cli_perf_roofline_uses_modeled_chips_and_combines_with_baseline(
    tmp_path, monkeypatch, capsys
):
    # Without --chips, check_roofline uses roofline_targets.json num_chips (8).
    # 9009.792 / 8 = 1126.224 tok/s/chip = 80.00% of 1407.78.
    run_dir = tmp_path / "glm"
    _write_perf_cell(run_dir, "isl8192_osl1024_c1024", output_throughput=9009.792)
    baseline_file = tmp_path / "glm.baseline.json"
    baseline_file.write_text(
        json.dumps(
            {
                "isl8192_osl1024_c1024": {
                    "completed": 64,
                    "output_token_throughput": 9000.0,
                }
            }
        )
    )
    ret = _run_cli(
        monkeypatch,
        "--mode",
        "perf",
        "--results-dir",
        str(run_dir),
        "--baseline",
        str(baseline_file),
        "--roofline-targets",
        str(_ROOFLINE_TARGETS_PATH),
        "--config-name",
        "glm-5.2-fp8-dp16-ep",
    )
    assert ret == 0
    out = capsys.readouterr().out
    assert "## Perf regression check" in out
    assert (
        "## Roofline speed check (`glm-5.2-fp8-dp16-ep`, 8 physical chips, advisory)"
        in out
    )
    assert (
        "| `isl8192_osl1024_c1024` | output_token_throughput_per_chip |"
        " 1126.22 | 1407.78 | 80.00% | ADVISORY |" in out
    )

    # A failing relative baseline still exits 1 when advisory roofline runs beside it.
    baseline_file.write_text(
        json.dumps(
            {
                "isl8192_osl1024_c1024": {
                    "completed": 64,
                    "output_token_throughput": 12000.0,
                }
            }
        )
    )
    assert (
        _run_cli(
            monkeypatch,
            "--mode",
            "perf",
            "--results-dir",
            str(run_dir),
            "--baseline",
            str(baseline_file),
            "--roofline-targets",
            str(_ROOFLINE_TARGETS_PATH),
            "--config-name",
            "glm-5.2-fp8-dp16-ep",
        )
        == 1
    )


def test_load_roofline_targets_rejects_malformed(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    with pytest.raises(ValueError, match="must hold an object"):
        check_regression.load_roofline_targets(bad)
    bad.write_text(json.dumps({"targets": {"c": {"cells": {"x": {"status": "bad"}}}}}))
    with pytest.raises(ValueError, match="status must be one of"):
        check_regression.load_roofline_targets(bad)
