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


def test_check_eval_missing_primary_metric_fails_as_no_result():
    # The results carry acc,none only. It must not stand in for the baseline's
    # exact_match,strict_match, which it would pass against (0.95 vs. 0.90).
    baseline = {"mmlu_llama": {"exact_match,strict_match": 0.90, "acc,none": 0.90}}
    results = {"mmlu_llama": {"acc,none": 0.95}}
    row = check_regression.check_eval(results, baseline, 0.015).rows[0]
    assert (row.metric, row.status) == ("(no result)", "FAIL")


def test_check_eval_compares_the_baseline_metric_by_name():
    # exact_match,strict_match comes first in ACC_KEYS, but the baseline gates
    # acc,none: the drop is 0.90 - 0.80, not 0.90 - 0.95.
    baseline = {"mmlu_llama": {"acc,none": 0.90}}
    results = {"mmlu_llama": {"exact_match,strict_match": 0.95, "acc,none": 0.80}}
    row = check_regression.check_eval(results, baseline, 0.015).rows[0]
    assert (row.metric, row.actual, row.status) == ("acc,none", 0.80, "FAIL")


def _write_eval_run(run_dir, results):
    run_dir.mkdir()
    (run_dir / "results_x.json").write_text(json.dumps({"results": results}))
    return run_dir


def _run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["check_regression.py", *argv])
    return check_regression.main()


# How the error for a run with no selected task ends.
_TASK_FILTER_HINT = (
    "--task-filter takes comma-separated tasks or 'all' and defaults to "
    "mmlu_llama, mmlu_llama_humanities, mmlu_llama_stem, "
    "mmlu_llama_social_sciences, mmlu_llama_other, mmlu_pro, "
    "humaneval_plus_tpu, mbpp_plus_tpu"
)


def _assert_usage_error(monkeypatch, capsys, argv, error):
    """The command line exits 2 with `error` as its only message."""
    with pytest.raises(SystemExit) as exc_info:
        _run_cli(monkeypatch, *argv)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.endswith(f"check_regression.py: error: {error}\n")


def test_calibrate_eval_rejects_bad_runs(tmp_path, monkeypatch, capsys):
    good = _write_eval_run(tmp_path / "good", {"mmlu_pro": {"acc,none": 0.5}})
    other = _write_eval_run(tmp_path / "other", {"mbpp": {"acc,none": 0.5}})
    both = _write_eval_run(
        tmp_path / "both", {"mmlu_pro": {"acc,none": 0.5}, "mbpp": {"acc,none": 0.5}}
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
    for bad, error in (
        (
            empty,
            f"ERROR: no selected eval results under {empty} (tasks found: none); "
            f"{_TASK_FILTER_HINT}\n",
        ),
        (both, "ERROR: not in every run: mbpp acc,none\n"),
    ):
        ret = _run_cli(
            monkeypatch, *argv, "--results-dir", str(good), "--results-dir", str(bad)
        )
        assert ret == 2
        captured = capsys.readouterr()
        assert (captured.out, captured.err) == ("", error)


def test_cli_calibrate_eval_names_task_filter_when_no_task_is_selected(
    tmp_path, monkeypatch, capsys
):
    # Without --task-filter the default list selects mmlu_pro in the first run
    # and leaves out mmmu_pro, the only task of the second.
    good = _write_eval_run(tmp_path / "good", {"mmlu_pro": {"acc,none": 0.5}})
    mmmu = _write_eval_run(tmp_path / "mmmu", {"mmmu_pro": {"acc,none": 0.5}})
    argv = ["--mode", "eval", "--calibrate"]
    ret = _run_cli(
        monkeypatch, *argv, "--results-dir", str(good), "--results-dir", str(mmmu)
    )
    assert ret == 2
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == (
        "",
        f"ERROR: no selected eval results under {mmmu} (tasks found: mmmu_pro); "
        f"{_TASK_FILTER_HINT}\n",
    )


@pytest.mark.parametrize(
    ("run_results", "expected_baseline"),
    [
        # `extra` is in the first run only; the task filter removes it before
        # the runs are compared, so it does not count as a metric lost by a
        # run. The asymmetric values separate the median (0.5) from the mean
        # (0.6).
        pytest.param(
            (
                {"mmlu_pro": {"acc,none": 0.5}, "extra": {"acc,none": 0.1}},
                {"mmlu_pro": {"acc,none": 0.5}},
                {"mmlu_pro": {"acc,none": 0.8}},
            ),
            {"mmlu_pro": {"acc,none": 0.5}},
            id="filtered_task_in_one_run",
        ),
        # The second run lacks only the stderr. It still calibrates, and the
        # baseline takes the median of the stderrs the other two reported.
        pytest.param(
            (
                {"mmlu_pro": {"acc,none": 0.5, "acc_stderr,none": 0.02}},
                {"mmlu_pro": {"acc,none": 0.6}},
                {"mmlu_pro": {"acc,none": 0.9, "acc_stderr,none": 0.04}},
            ),
            {"mmlu_pro": {"acc,none": 0.6, "acc_stderr,none": 0.03}},
            id="stderr_missing_from_one_run",
        ),
    ],
)
def test_cli_calibrate_eval_writes_median_of_selected_tasks(
    tmp_path, monkeypatch, capsys, run_results, expected_baseline
):
    runs = [
        _write_eval_run(tmp_path / f"run{i}", results)
        for i, results in enumerate(run_results)
    ]
    argv = ["--mode", "eval", "--calibrate", "--task-filter", "mmlu_pro"]
    for run in runs:
        argv += ["--results-dir", str(run)]
    assert _run_cli(monkeypatch, *argv) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == expected_baseline
    assert captured.err == ""


# --- Freeze policy: a frozen baseline changes only with owner approval ---

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
        {
            "_policy": {
                "state": "frozen",
                "owner": " alice ",
                "approval": {
                    "approver": " alice ",
                    "reference": " https://example.invalid/pull/1 ",
                },
            }
        }
    )
    assert policy == check_regression.BaselinePolicy(
        state="frozen",
        owner="alice",
        approval=check_regression.BaselineApproval(
            "alice", "https://example.invalid/pull/1"
        ),
    )


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (None, "_policy must be an object"),
        ("frozen", "_policy must be an object"),
        (
            {"status": "frozen", "owner": "alice"},
            r"unknown _policy field\(s\): 'status'",
        ),
        ({"state": "golden"}, "_policy.state must be 'development' or 'frozen'"),
        ({"state": "frozen"}, "_policy.owner must be a non-empty string"),
        ({"state": "frozen", "owner": " "}, "_policy.owner must be a non-empty"),
        # A development block need not name an owner, but may not name a bad one.
        ({"owner": ""}, "_policy.owner must be a non-empty string"),
        ({"approval": "alice"}, "_policy.approval must be an object"),
        (
            {"approval": {"approver": "alice"}},
            "_policy.approval.reference must be a non-empty string",
        ),
        (
            {
                "approval": {
                    "approver": "alice",
                    "reference": "https://example.invalid/pull/1",
                    "extra": "nope",
                }
            },
            r"unknown _policy\.approval field\(s\): 'extra'",
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
        ({**_TASK, "_policy": _FROZEN}, _TASK, _NO_APPROVAL),
        ({**_TASK, "_policy": _FROZEN}, {"_policy": _FROZEN}, _NO_APPROVAL),
        # Unfreezing is a change, also when the owner stays.
        (
            {**_TASK, "_policy": _FROZEN},
            {**_TASK, "_policy": {"state": "development", "owner": "alice"}},
            _NO_APPROVAL,
        ),
        # Handing the file to another owner is a change too.
        (
            {**_TASK, "_policy": _FROZEN},
            {**_TASK, "_policy": {"state": "frozen", "owner": "bob"}},
            _NO_APPROVAL,
        ),
        # An approval that was already in the old file does not count again,
        # even with extra whitespace around its fields.
        (
            {**_TASK, "_policy": {**_FROZEN, "approval": _APPROVAL_1}},
            {**_RAISED, "_policy": {**_FROZEN, "approval": _APPROVAL_1}},
            _NO_APPROVAL,
        ),
        (
            {**_TASK, "_policy": {**_FROZEN, "approval": _APPROVAL_1}},
            {
                **_RAISED,
                "_policy": {
                    **_FROZEN,
                    "approval": {
                        "approver": "alice ",
                        "reference": "https://example.invalid/pull/1 ",
                    },
                },
            },
            _NO_APPROVAL,
        ),
        # Stripping or mutating an existing approval without a new owner
        # approval is rejected (blocks two-step strip-then-replay).
        (
            {**_TASK, "_policy": {**_FROZEN, "approval": _APPROVAL_1}},
            {**_TASK, "_policy": _FROZEN},
            _NO_APPROVAL,
        ),
        (
            {**_TASK, "_policy": {**_FROZEN, "approval": _APPROVAL_1}},
            {
                **_TASK,
                "_policy": {
                    **_FROZEN,
                    "approval": {
                        "approver": "bob",
                        "reference": "https://example.invalid/pull/3",
                    },
                },
            },
            _WRONG_APPROVER,
        ),
        (
            {**_TASK, "_policy": {**_FROZEN, "approval": _APPROVAL_1}},
            {**_TASK, "_policy": {**_FROZEN, "approval": _APPROVAL_2}},
            None,
        ),
        # Type-aware gated value equality detects int vs bool substitutions.
        (
            {"task": {"acc,none": 1}, "_policy": _FROZEN},
            {"task": {"acc,none": True}, "_policy": _FROZEN},
            _NO_APPROVAL,
        ),
        (
            {"task": {"acc,none": 0}, "_policy": _FROZEN},
            {"task": {"acc,none": False}, "_policy": _FROZEN},
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


def test_format_update_result():
    path = pathlib.Path("baseline.json")
    assert (
        check_regression.format_update_result(path, None)
        == "PASS: baseline update follows its freeze policy."
    )
    assert (
        check_regression.format_update_result(path, _NO_APPROVAL)
        == f"FAILED: {path}: {_NO_APPROVAL}"
    )


def test_cli_update_mode(tmp_path, monkeypatch, capsys):
    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    old_file.write_text(json.dumps({**_TASK, "_policy": _FROZEN}))
    new_only = ["--mode", "update", "--baseline", str(new_file)]
    with_previous = [*new_only, "--previous-baseline", str(old_file)]
    passed = (0, "PASS: baseline update follows its freeze policy.\n", "")
    no_owner = (2, "", "ERROR: _policy.owner must be a non-empty string\n")

    def run(new_policy, argv):
        new_file.write_text(json.dumps({**_RAISED, "_policy": new_policy}))
        ret = _run_cli(monkeypatch, *argv)
        captured = capsys.readouterr()
        return ret, captured.out, captured.err

    violation = (1, "", f"FAILED: {new_file}: {_NO_APPROVAL}\n")
    assert run(_FROZEN, with_previous) == violation
    # Without a previous file the same content is a new baseline.
    assert run(_FROZEN, new_only) == passed
    assert run({**_FROZEN, "approval": _APPROVAL_1}, with_previous) == passed
    # A malformed block exits 2, in a changed file and in a new one.
    assert run({"state": "frozen"}, with_previous) == no_owner
    assert run({"state": "frozen"}, new_only) == no_owner


# What each mode needs after the flags that select it; BASELINE and RESULTS
# stand for real paths.
_UPDATE = "--mode update"
_MODE_ARGS = {
    _UPDATE: ["--baseline", "BASELINE"],
    "--mode perf": ["--results-dir", "RESULTS", "--baseline", "BASELINE"],
    "--mode eval": ["--results-dir", "RESULTS", "--baseline", "BASELINE"],
    "--mode perf --calibrate": ["--results-dir", "RESULTS"],
    "--mode eval --calibrate": ["--results-dir", "RESULTS"],
}
_RESULT_MODES = [mode for mode in _MODE_ARGS if mode != _UPDATE]
_NOT_PERF_CHECK = [mode for mode in _MODE_ARGS if mode != "--mode perf"]
_ROOFLINE_FLAGS = [
    ["--roofline-targets", "targets.json"],
    ["--config-name", "some-config"],
    ["--chips", "4"],
    ["--prefill-chips", "2"],
    ["--decode-chips", "2"],
]


@pytest.mark.parametrize(
    ("argv", "error"),
    [
        (["--mode", "update"], "--baseline is required in update mode"),
        (
            ["--mode", "perf", "--baseline", "b.json"],
            "--results-dir is required in perf and eval modes",
        ),
        # Without the check this run would write an empty baseline and exit 0.
        (
            ["--mode", "eval", "--calibrate"],
            "--results-dir is required in perf and eval modes",
        ),
        (
            ["--mode", "eval", "--results-dir", "r"],
            "--baseline is required unless --calibrate is set",
        ),
        (
            ["--mode", "perf", "--results-dir", "r"],
            "--baseline or --roofline-targets is required unless --calibrate is set",
        ),
        (
            ["--mode", "perf", "--results-dir", "r", "--roofline-targets", "t.json"],
            "--config-name is required with --roofline-targets",
        ),
        *[
            (
                ["--mode", "perf", "--results-dir", "r", "--baseline", "b.json", *flag],
                f"{flag[0]} requires --roofline-targets",
            )
            for flag in _ROOFLINE_FLAGS[1:]
        ],
    ],
)
def test_cli_requires_the_flags_a_mode_reads(monkeypatch, capsys, argv, error):
    _assert_usage_error(monkeypatch, capsys, argv, error)


# (mode, flags it does not take, the flags the error names)
_NOT_VALID_CASES = [
    (_UPDATE, ["--calibrate"], "--calibrate"),
    (_UPDATE, ["--tolerance", "0.1"], "--tolerance"),
    (_UPDATE, ["--results-dir", "RESULTS"], "--results-dir"),
    (_UPDATE, ["--task-filter", "mmlu_pro"], "--task-filter"),
    # 0 is a real value of this flag, not "flag not given".
    (_UPDATE, ["--gate-ttft-max-concurrency", "0"], "--gate-ttft-max-concurrency"),
    (_UPDATE, ["--tolerance", "0.1", "--calibrate"], "--calibrate, --tolerance"),
    *[
        (mode, ["--previous-baseline", "BASELINE"], "--previous-baseline")
        for mode in _RESULT_MODES
    ],
    *[
        (mode, flag, flag[0])
        for mode in ("--mode perf", "--mode eval")
        for flag in (
            ["--task-filter", "mmlu_pro"],
            ["--gate-ttft-max-concurrency", "0"],
        )
    ],
    *[
        ("--mode perf --calibrate", flag, flag[0])
        for flag in (
            ["--baseline", "BASELINE"],
            ["--tolerance", "0.1"],
            ["--task-filter", "mmlu_pro"],
        )
    ],
    *[
        ("--mode eval --calibrate", flag, flag[0])
        for flag in (
            ["--baseline", "BASELINE"],
            ["--tolerance", "0.1"],
            ["--gate-ttft-max-concurrency", "0"],
        )
    ],
    *[(mode, flag, flag[0]) for flag in _ROOFLINE_FLAGS for mode in _NOT_PERF_CHECK],
]


@pytest.mark.parametrize(
    ("mode", "flags", "named"),
    _NOT_VALID_CASES,
    ids=[f"{mode} {' '.join(flags)}" for mode, flags, _ in _NOT_VALID_CASES],
)
def test_cli_rejects_flags_the_mode_does_not_take(
    tmp_path, monkeypatch, capsys, mode, flags, named
):
    """A flag given to a mode that does not take it is a usage error."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_TASK))
    results = _write_eval_run(tmp_path / "results", _TASK)
    paths = {"BASELINE": str(baseline), "RESULTS": str(results)}
    argv = [paths.get(arg, arg) for arg in mode.split() + _MODE_ARGS[mode] + flags]
    _assert_usage_error(monkeypatch, capsys, argv, f"{named}: not valid with {mode}")


def test_cli_help_renders(monkeypatch, capsys):
    """argparse %-formats each help string, so a bare % in one breaks --help."""
    with pytest.raises(SystemExit) as exc_info:
        _run_cli(monkeypatch, "--help")
    assert exc_info.value.code == 0
    assert "--roofline-targets ROOFLINE_TARGETS" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("content", "error"),
    [
        # A missing file, invalid JSON, and NaN/Infinity: reason follows file name.
        (None, "ERROR: cannot read {bad}: "),
        ("{not json", "ERROR: cannot read {bad}: "),
        ('{"mmlu_pro": {"acc,none": NaN}}', "ERROR: cannot read {bad}: "),
        ('{"mmlu_pro": {"acc,none": Infinity}}', "ERROR: cannot read {bad}: "),
        ("[]", "ERROR: {bad} must hold a JSON object\n"),
    ],
)
def test_cli_update_mode_unreadable_file_exits_2(
    tmp_path, monkeypatch, capsys, content, error
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
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith(error.format(bad=bad))
        assert captured.err.count("\n") == 1


@pytest.mark.parametrize("mode", ["perf", "eval"])
def test_cli_check_modes_reject_nan_baseline_with_exit_2(
    tmp_path, monkeypatch, capsys, mode
):
    bad = tmp_path / "nan.baseline.json"
    bad.write_text('{"mmlu_pro": {"acc,none": NaN}}')
    results = _write_eval_run(tmp_path / "results", _TASK)
    ret = _run_cli(
        monkeypatch,
        "--mode",
        mode,
        "--results-dir",
        str(results),
        "--baseline",
        str(bad),
    )
    assert ret == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "non-standard JSON constant 'NaN'" in captured.err


def test_shipped_baselines_have_valid_policy():
    """Every checked-in perf and eval baseline parses under the policy schema."""
    files = sorted(_BASELINE_DIR.parent.rglob("*.baseline.json"))
    assert files, f"no baselines found under {_BASELINE_DIR.parent}"
    for path in files:
        check_regression.parse_policy(check_regression.load_baseline_file(path))


# --- Advisory roofline reporting ---

_ROOFLINE_TARGETS_PATH = _BASELINE_DIR.parent / "perf" / "v7x" / "roofline_targets.json"
_ROOFLINE_METRIC = "output_token_throughput_per_chip"
_ROOFLINE_TABLE = (
    "| key | metric | actual/chip | roofline/chip | % of roofline | status |\n"
    "|---|---|---:|---:|---:|---|\n"
)
_ROOFLINE_PASS = "PASS: advisory roofline check complete.\n"
# One config with a cell of each status, one without a chip count, and one
# that is not modeled.
_TARGETS = {
    "targets": {
        "cfg": {
            "num_chips": 4,
            "cells": {
                "isl1024_osl1024_c1": {_ROOFLINE_METRIC: 1000.0, "status": "ok"},
                "isl1024_osl1024_c2": {_ROOFLINE_METRIC: 200.0, "status": "unverified"},
                "isl1024_osl1024_c4": {"status": "oom"},
                "isl1024_osl1024_c8": {_ROOFLINE_METRIC: 500.0},
            },
        },
        "cfg-no-chips": {"cells": {"isl1024_osl1024_c1": {_ROOFLINE_METRIC: 1000.0}}},
    },
    "not_modeled": {"cfg-1p1d": {"reason": "disaggregated", "detail": "No model."}},
}


def _write_perf_cell(results_dir, cell_key, **metrics):
    results_dir.mkdir(exist_ok=True)
    payload = {"completed": 64, "num_prompts": 64, **metrics}
    (results_dir / f"{cell_key}.json").write_text(json.dumps(payload))


@pytest.fixture
def roofline_run(tmp_path, monkeypatch, capsys):
    """Run perf check mode on `_TARGETS`; return (exit code, stdout, stderr)."""
    targets_file = tmp_path / "targets.json"
    targets_file.write_text(json.dumps(_TARGETS))
    results_dir = tmp_path / "results"
    _write_perf_cell(results_dir, "isl1024_osl1024_c1", output_throughput=3200.0)

    def run(config, *extra, results=results_dir):
        ret = _run_cli(
            monkeypatch,
            "--mode", "perf", "--results-dir", str(results),
            "--roofline-targets", str(targets_file), "--config-name", config, *extra,
        )  # fmt: skip
        captured = capsys.readouterr()
        return ret, captured.out, captured.err

    return run


def test_shipped_roofline_targets_are_well_formed():
    targets = check_regression.load_roofline_targets(_ROOFLINE_TARGETS_PATH)
    assert "qwen3.5-397b-fp8-tp1-dp8-ep" in targets.targets
    assert "kimi-k3-tp32-ep" in targets.targets
    assert "qwen3.5-397b-fp8-mtp" in targets.not_modeled
    kimi = targets.targets["kimi-k3-tp32-ep"]
    assert kimi.num_chips == 16
    assert kimi.cells["isl8192_osl1024_c1"].status == "unverified"
    assert kimi.cells["isl8192_osl1024_c1"].output_token_throughput_per_chip == 12.63
    # --config-name is a config stem, so every target and not_modeled entry
    # must match a checked-in config file.
    configs = _BASELINE_DIR.parents[1] / "configs"
    assert [n for n in targets.targets if not (configs / f"{n}.sh").is_file()] == []
    assert [n for n in targets.not_modeled if not (configs / f"{n}.sh").is_file()] == []


def test_cli_perf_roofline_reports_each_measured_cell(roofline_run, tmp_path):
    # 4 chips from the targets file: 3200 / 4 = 800.00 is 80.00% of 1000.00,
    # and 400 / 4 = 100.00 is 50.00% of 200.00.
    results_dir = tmp_path / "results"
    _write_perf_cell(results_dir, "isl1024_osl1024_c2", output_throughput=400.0)
    _write_perf_cell(results_dir, "isl1024_osl1024_c4", output_throughput=400.0)
    _write_perf_cell(results_dir, "isl1024_osl1024_c8")
    _write_perf_cell(results_dir, "isl512_osl512_c1", output_throughput=400.0)
    assert roofline_run("cfg") == (
        0,
        "\n## Roofline speed check (`cfg`, 4 physical chips, advisory)\n\n"
        + _ROOFLINE_TABLE
        + f"| `isl1024_osl1024_c1` | {_ROOFLINE_METRIC} | 800.00 | 1000.00 | 80.00% |"
        " ADVISORY |\n"
        f"| `isl1024_osl1024_c2` | {_ROOFLINE_METRIC} | 100.00 | 200.00 | 50.00% |"
        " ADVISORY (unverified) |\n"
        f"| `isl1024_osl1024_c4` | {_ROOFLINE_METRIC} | 0.00 | 0.00 | - |"
        " SKIP (oom) |\n"
        f"| `isl1024_osl1024_c8` | {_ROOFLINE_METRIC} | 0.00 | 500.00 | - | SKIP |\n"
        "| `isl512_osl512_c1` | (no roofline cell) | 0.00 | 0.00 | - | SKIP |\n"
        "\n" + _ROOFLINE_PASS,
        "",
    )


@pytest.mark.parametrize(
    ("config", "chip_flags", "chips", "per_chip", "percent", "expected_err"),
    [
        ("cfg", [], 4, "800.00", "80.00%", ""),
        ("cfg", ["--chips", "4"], 4, "800.00", "80.00%", ""),
        # Explicit chips win over the targets file and emit a note when different.
        (
            "cfg",
            ["--chips", "8"],
            8,
            "400.00",
            "40.00%",
            "NOTE: using 8 chips (modeled on 4 chips)\n",
        ),
        (
            "cfg",
            ["--prefill-chips", "2", "--decode-chips", "8"],
            10,
            "320.00",
            "32.00%",
            "NOTE: using 10 chips (modeled on 4 chips)\n",
        ),
        ("cfg-no-chips", ["--chips", "2"], 2, "1600.00", "160.00%", ""),
    ],
)
def test_cli_perf_roofline_chip_count(
    roofline_run, config, chip_flags, chips, per_chip, percent, expected_err
):
    assert roofline_run(config, *chip_flags) == (
        0,
        f"\n## Roofline speed check (`{config}`, {chips} physical chips, advisory)\n\n"
        + _ROOFLINE_TABLE
        + f"| `isl1024_osl1024_c1` | {_ROOFLINE_METRIC} | {per_chip} | 1000.00 |"
        f" {percent} | ADVISORY |\n\n" + _ROOFLINE_PASS,
        expected_err,
    )


def test_cli_perf_roofline_not_modeled_needs_no_results(roofline_run, tmp_path):
    note = (
        "\n## Roofline speed check (`cfg-1p1d`: not modeled)\n\n"
        "NOTE: `cfg-1p1d` is not modeled in roofline targets"
        " (disaggregated: No model.)\n\n"
    )
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert roofline_run("cfg-1p1d", results=empty_dir) == (0, note + _ROOFLINE_PASS, "")
    # A modeled config has a table to fill, so it needs result files.
    assert roofline_run("cfg", results=empty_dir) == (
        2,
        "",
        f"ERROR: no isl*_osl*_c*.json files found in {empty_dir}\n",
    )


@pytest.mark.parametrize(
    ("config", "flags", "error"),
    [
        ("nope", [], "config 'nope' not found in roofline targets"),
        (
            "cfg-no-chips",
            [],
            "chip count required for roofline check of 'cfg-no-chips'",
        ),
        ("cfg", ["--chips", "0"], "--chips must be a positive integer, got 0"),
        (
            "cfg",
            ["--prefill-chips", "-1", "--decode-chips", "2"],
            "--prefill-chips must be a positive integer, got -1",
        ),
        (
            "cfg",
            ["--prefill-chips", "2", "--decode-chips", "0"],
            "--decode-chips must be a positive integer, got 0",
        ),
        (
            "cfg",
            ["--prefill-chips", "2"],
            "--prefill-chips and --decode-chips must be provided together",
        ),
        (
            "cfg",
            ["--decode-chips", "2"],
            "--prefill-chips and --decode-chips must be provided together",
        ),
        (
            "cfg",
            ["--chips", "4", "--prefill-chips", "2", "--decode-chips", "2"],
            "--chips cannot be combined with --prefill-chips/--decode-chips",
        ),
    ],
)
def test_cli_perf_roofline_setup_errors_exit_2_before_any_table(
    roofline_run, tmp_path, config, flags, error
):
    # The baseline would pass; the error still wins and no table is printed.
    baseline_file = tmp_path / "perf.baseline.json"
    baseline_file.write_text(
        json.dumps({"isl1024_osl1024_c1": {"output_token_throughput": 3200.0}})
    )
    assert roofline_run(config, *flags, "--baseline", str(baseline_file)) == (
        2,
        "",
        f"ERROR: {error}\n",
    )


def test_cli_perf_roofline_runs_beside_the_baseline_check(roofline_run, tmp_path):
    baseline_file = tmp_path / "perf.baseline.json"
    roofline_table = (
        "\n## Roofline speed check (`cfg`, 4 physical chips, advisory)\n\n"
        + _ROOFLINE_TABLE
        + f"| `isl1024_osl1024_c1` | {_ROOFLINE_METRIC} | 800.00 | 1000.00 | 80.00% |"
        " ADVISORY |\n\n"
    )

    def run(baseline_throughput):
        baseline_file.write_text(
            json.dumps(
                {"isl1024_osl1024_c1": {"output_token_throughput": baseline_throughput}}
            )
        )
        return roofline_run("cfg", "--baseline", str(baseline_file))

    ret, out, err = run(3200.0)
    assert (ret, err) == (0, "")
    assert out.startswith("\n## Perf regression check (tolerance: 5.00%)\n")
    assert out.endswith(roofline_table + "PASS: all gated metrics within tolerance.\n")

    # A failing baseline check exits 1; the roofline table is still printed.
    ret, out, err = run(4000.0)
    assert (ret, err) == (1, "FAILED: 1 metric(s) regressed beyond 5.00%\n")
    assert out.endswith(roofline_table)


def test_cli_perf_roofline_reads_the_shipped_targets(tmp_path, monkeypatch, capsys):
    # glm-5.2-fp8-dp16-ep is modeled on 8 chips in the shipped file.
    _write_perf_cell(tmp_path, "isl8192_osl1024_c1024", output_throughput=8000.0)
    ret = _run_cli(
        monkeypatch,
        "--mode", "perf", "--results-dir", str(tmp_path),
        "--roofline-targets", str(_ROOFLINE_TARGETS_PATH),
        "--config-name", "glm-5.2-fp8-dp16-ep",
    )  # fmt: skip
    assert ret == 0
    out = capsys.readouterr().out
    assert out.startswith(
        "\n## Roofline speed check (`glm-5.2-fp8-dp16-ep`, 8 physical chips,"
        " advisory)\n"
    )
    assert f"| `isl8192_osl1024_c1024` | {_ROOFLINE_METRIC} | 1000.00 |" in out


def test_load_roofline_targets_defaults(tmp_path):
    path = tmp_path / "targets.json"
    cells = {
        "x": {_ROOFLINE_METRIC: 5},
        # oom cells always normalize throughput to 0.0 (never negative).
        "y": {_ROOFLINE_METRIC: -10.0, "status": "oom"},
    }
    path.write_text(json.dumps({"targets": {"c": {"cells": cells}}}))
    # No num_chips, status ok, and no not_modeled map.
    assert check_regression.load_roofline_targets(path) == (
        {"c": (None, {"x": (5.0, "ok"), "y": (0.0, "oom")})},
        {},
    )
    path.write_text(
        json.dumps(
            {
                "targets": {},
                "not_modeled": {
                    "c": {"reason": "r"},
                    "d": {"reason": "r2", "detail": None},
                },
            }
        )
    )
    assert check_regression.load_roofline_targets(path) == (
        {},
        {"c": ("r", ""), "d": ("r2", "")},
    )


@pytest.mark.parametrize(
    ("content", "error"),
    [
        (None, "cannot read roofline targets {path}: "),
        ("{not json", "cannot read roofline targets {path}: "),
        ([], "roofline targets {path} must hold a JSON object"),
        ({"targets": []}, "{path} must hold an object with a 'targets' map"),
        ({"targets": {"c": []}}, "roofline config 'c' must be an object"),
        (
            {"targets": {"c": {"num_chips": 0, "cells": {}}}},
            "c.num_chips must be a positive integer, got 0",
        ),
        (
            {"targets": {"c": {"num_chips": True, "cells": {}}}},
            "c.num_chips must be a positive integer, got True",
        ),
        ({"targets": {"c": {"cells": {}}}}, "c.cells must be a non-empty object"),
        ({"targets": {"c": {"cells": {"x": 1}}}}, "c.cells.x must be an object"),
        (
            {"targets": {"c": {"cells": {"x": {"status": "bad"}}}}},
            "c.cells.x.status must be one of ('ok', 'unverified', 'oom'), got 'bad'",
        ),
        (
            {"targets": {"c": {"cells": {"x": {_ROOFLINE_METRIC: 0}}}}},
            "c.cells.x.output_token_throughput_per_chip must be positive",
        ),
        (
            {"targets": {"c": {"cells": {"x": {"status": "unverified"}}}}},
            "c.cells.x.output_token_throughput_per_chip must be positive",
        ),
        # Non-standard JSON NaN is rejected at load time.
        (
            {"targets": {"c": {"cells": {"x": {_ROOFLINE_METRIC: float("nan")}}}}},
            "cannot read roofline targets {path}: non-standard JSON constant 'NaN'",
        ),
        (
            {"targets": {"c": {"cells": {"x": {_ROOFLINE_METRIC: True}}}}},
            "c.cells.x.output_token_throughput_per_chip must be positive",
        ),
        ({"targets": {}, "not_modeled": []}, "{path}: 'not_modeled' must be an object"),
        (
            {"targets": {}, "not_modeled": {"c": "why"}},
            "{path}: not_modeled.c must be an object",
        ),
        (
            {"targets": {}, "not_modeled": {"c": {"reason": " "}}},
            "{path}: not_modeled.c.reason must be non-empty",
        ),
        (
            {"targets": {}, "not_modeled": {"c": {"reason": "r", "detail": 123}}},
            "{path}: not_modeled.c.detail must be a string",
        ),
        (
            {
                "targets": {"dup": {"cells": {"x": {_ROOFLINE_METRIC: 10}}}},
                "not_modeled": {"dup": {"reason": "r"}},
            },
            "{path}: config(s) listed in both 'targets' and 'not_modeled': 'dup'",
        ),
    ],
)
def test_load_roofline_targets_rejects_malformed(tmp_path, content, error):
    path = tmp_path / "targets.json"
    if content is not None:
        path.write_text(content if isinstance(content, str) else json.dumps(content))
    with pytest.raises(ValueError) as exc_info:
        check_regression.load_roofline_targets(path)
    assert str(exc_info.value).startswith(error.format(path=path))
