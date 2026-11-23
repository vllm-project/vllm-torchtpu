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
"""run_eval_flow.sh's lm-eval gate, driven end to end against a fake lm_eval.

Covers the switches a gated caller passes: --lm-eval-limit (the samples per
task) and --require-baselines (a missing baseline fails instead of being
skipped), and that a run passing neither behaves as before.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BENCH = pathlib.Path("scripts/vllm/benchmarking")
_TASK = "toy_task"
_CONFIG = "toy-eval-config"
_METRIC = "exact_match,strict-match"

# Stands in for lm_eval: records its arguments and writes one results file
# scored FAKE_SCORE, the way lm-eval names it under --output_path.
_FAKE_LM_EVAL = """#!{python}
import json, os, pathlib, sys
args = sys.argv[1:]
pathlib.Path(os.environ["FAKE_LM_EVAL_ARGS"]).write_text(json.dumps(args))
out = pathlib.Path(args[args.index("--output_path") + 1])
task = args[args.index("--tasks") + 1]
score = float(os.environ["FAKE_SCORE"])
(out / f"results_{{task}}.json").write_text(json.dumps({{"results": {{task: {{
    "{metric}": score, "exact_match_stderr,strict-match": 0.004}}}}}}))
"""

# run_lm_eval pipes lm_eval through `stdbuf -oL tr`; this keeps the test off
# coreutils' stdbuf, which not every image has.
_FAKE_STDBUF = """#!/bin/bash
shift
exec "$@"
"""


def _executable(path: pathlib.Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def sandbox(tmp_path: pathlib.Path) -> pathlib.Path:
    """A copy of the eval flow's scripts, with fakes for lm_eval and vLLM."""
    root = tmp_path / "repo"
    bench = root / _BENCH
    (bench / "configs").mkdir(parents=True)
    (bench / "baselines" / "eval").mkdir(parents=True)
    (bench / "baselines" / "perf").mkdir(parents=True)
    for name in (
        "run_eval_flow.sh",
        "run_benchmarks.sh",
        "check_regression.py",
        "cleanup_server.sh",
    ):
        shutil.copy2(_REPO_ROOT / _BENCH / name, bench / name)
    shutil.copytree(_REPO_ROOT / _BENCH / "lm_eval_tasks", bench / "lm_eval_tasks")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(
        bin_dir / "lm_eval",
        _FAKE_LM_EVAL.format(python=sys.executable, metric=_METRIC),
    )
    _executable(bin_dir / "stdbuf", _FAKE_STDBUF)
    # run_benchmarks.sh only checks that a vllm package can be found.
    stub = tmp_path / "stub"
    (stub / "vllm").mkdir(parents=True)
    (stub / "vllm" / "__init__.py").write_text("")
    return root


def _write_config(root: pathlib.Path) -> None:
    (root / _BENCH / "configs" / f"{_CONFIG}.sh").write_text(
        f"""MODEL="toy/model"
RANDOM_RANGE_RATIO="0.0"
EVAL_TOLERANCE="0.01"
LM_EVAL_TASKS="{_TASK}"
"""
    )


def _write_baseline(root: pathlib.Path, body: dict) -> None:
    path = root / _BENCH / "baselines" / "eval" / f"{_CONFIG}.{_TASK}.baseline.json"
    path.write_text(json.dumps(body))


def _run(root: pathlib.Path, *flags: str, score: float = 0.9, **env: str):
    tmp = root.parent
    run_env = dict(os.environ)
    run_env.update(
        {
            "PATH": f"{tmp / 'bin'}:{os.environ['PATH']}",
            "PYTHONPATH": str(tmp / "stub"),
            "FAKE_LM_EVAL_ARGS": str(tmp / "lm_eval_args.json"),
            "FAKE_SCORE": str(score),
            "LM_EVAL_REQUEST_CACHE_ROOT": str(tmp / "request_cache"),
            "HOME": str(tmp),
        }
    )
    for name in ("BUILDKITE_BRANCH", "ARTIFACTS_DIR", "EVAL_SOFT_FAIL"):
        run_env.pop(name, None)
    run_env.update(env)
    results = tmp / "results"
    shutil.rmtree(results, ignore_errors=True)
    proc = subprocess.run(
        [
            "bash",
            str(_BENCH / "run_eval_flow.sh"),
            "--config",
            _CONFIG,
            "--results-dir",
            str(results),
            "--run-lm-eval",
            "--skip-benchmark",
            "--skip-db-upload",
            "--host",
            "127.0.0.1",
            "--port",
            "1",
            *flags,
        ],
        cwd=root,
        env=run_env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    args_file = tmp / "lm_eval_args.json"
    lm_eval_args = json.loads(args_file.read_text()) if args_file.exists() else None
    if args_file.exists():
        args_file.unlink()
    return proc, lm_eval_args


def test_default_config_runs_100_samples_and_skips_a_missing_baseline(sandbox):
    _write_config(sandbox)
    proc, args = _run(sandbox)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert args is not None
    assert args[args.index("--limit") + 1] == "100"
    assert "WARNING: Baseline not found for toy_task" in proc.stdout


def test_lm_eval_limit_none_runs_the_whole_split(sandbox):
    _write_config(sandbox)
    proc, args = _run(sandbox, "--lm-eval-limit", "none")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert args is not None
    assert "--limit" not in args


def test_lm_eval_limit_takes_a_count_or_none(sandbox):
    _write_config(sandbox)
    proc, args = _run(sandbox, "--lm-eval-limit", "250")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert args is not None
    assert args[args.index("--limit") + 1] == "250"

    proc, args = _run(sandbox, "--lm-eval-limit", "all")
    assert proc.returncode != 0
    assert "--lm-eval-limit must be a positive integer or none" in proc.stdout
    assert args is None


def test_required_baseline_missing_fails_unless_eval_soft_fail(sandbox):
    _write_config(sandbox)
    proc, _ = _run(sandbox, "--require-baselines")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "::error::No baseline for toy_task" in proc.stdout

    proc, _ = _run(sandbox, "--require-baselines", EVAL_SOFT_FAIL="1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "::warning::Eval regression for toy_task" in proc.stdout


def test_pending_baseline_fails_the_gate(sandbox):
    _write_config(sandbox)
    _write_baseline(sandbox, {"_pending": "measured by the first nightly"})
    proc, _ = _run(sandbox, "--require-baselines")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "baseline is pending" in proc.stdout + proc.stderr


def test_measured_baseline_gates_both_ways(sandbox):
    _write_config(sandbox)
    _write_baseline(
        sandbox, {_TASK: {_METRIC: 0.90, "exact_match_stderr,strict-match": 0.004}}
    )
    gate = ("--require-baselines", "--lm-eval-limit", "none")
    proc, _ = _run(sandbox, *gate, score=0.90)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # Allowed drop max(0.01, 1.645 * hypot(0.004, 0.004)) = 0.01; 0.05 is over.
    proc, _ = _run(sandbox, *gate, score=0.85)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "::error::Eval regression check failed for toy_task" in proc.stdout
