# SPDX-License-Identifier: Apache-2.0
"""CPU checks for GitHub-visible throughput and failure reporting."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "dsv41_check_summary", _REPO / ".buildkite/scripts/dsv41_check_summary.py"
)
assert _SPEC is not None and _SPEC.loader is not None
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)
pytestmark = pytest.mark.cpu_test


def write_result(root, scenario):
    """Use skewed rounds to distinguish median from mean aggregation."""
    (root / "fixed").mkdir(parents=True)
    timing = dict(
        completed=True,
        formal_status="passed",
        profile_status="disabled",
        launch_wall_ns=0,
        finished_wall_ns=120_000_000_000,
    )
    (root / "startup.json").write_text(json.dumps(timing))
    (root / "manifest.json").write_text(
        json.dumps({"identity": {"torchtpu": {"commit": "abc"}}})
    )
    spec = dict(name=scenario, dp_size=8, concurrency=384, physical_chips=4)
    (root / "fixed/manifest.json").write_text(
        json.dumps({"rounds": 3, "scenario": spec})
    )
    for i, rate in enumerate([400, 800, 8000], 1):
        metric = dict(global_tokens_s=rate, per_physical_chip_tokens_s=rate / 4)
        report = dict(
            warmup=False,
            scenario=spec,
            decode_throughput={
                "basis": "fixed_accepted_tokens_median_step_estimate",
                "accepted_tokens_per_request_step": 3.47,
                **metric,
            },
            categories={
                "encoder_middle": metric,
                "decode": dict(
                    global_tokens_s=200,
                    per_physical_chip_tokens_s=50,
                    median_step_ms=384 * 3.47 / rate * 1000,
                ),
            },
        )
        (root / f"fixed/round-{i}-summary.json").write_text(json.dumps(report))


@pytest.mark.parametrize("scenario", ["decode", "encoder_extend"])
def test_formal_median_and_duration_are_visible(tmp_path, scenario):
    write_result(tmp_path, scenario)
    text = summary.render_summary(tmp_path, "abc")
    assert "| 800.00 | 200.00 |" in text
    assert "2.00 min" in text and "3 formal round(s)" in text
    if scenario == "decode":
        assert "Decode (median step, fixed 3.47)" in text
        assert "fixed 3.47 accepted tokens/request/step" in text
        assert "Median step: 1665.600 ms" in text
        assert "not measured output throughput" in text
        assert "Decode (observed window) | 200.00 | 50.00" in text
    else:
        assert "Prefill (middle chunks)" in text


@pytest.mark.parametrize(
    "invalid", ["missing_round", "wrong_commit", "failed", "warmup"]
)
def test_unverified_results_are_rejected(tmp_path, invalid):
    write_result(tmp_path, "decode")
    commit = "wrong" if invalid == "wrong_commit" else "abc"
    if invalid == "missing_round":
        (tmp_path / "fixed/round-2-summary.json").unlink()
    elif invalid in ("failed", "warmup"):
        path = tmp_path / (
            "startup.json" if invalid == "failed" else "fixed/round-1-summary.json"
        )
        data = json.loads(path.read_text())
        data["formal_status" if invalid == "failed" else "warmup"] = (
            "failed" if invalid == "failed" else True
        )
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError, OSError)):
        summary.render_summary(tmp_path, commit)


@pytest.mark.parametrize("invalid", ["legacy_basis", "wrong_fixed_tokens"])
def test_decode_check_rejects_other_accounting_policies(tmp_path, invalid):
    write_result(tmp_path, "decode")
    path = tmp_path / "fixed/round-1-summary.json"
    report = json.loads(path.read_text())
    if invalid == "legacy_basis":
        report["decode_throughput"]["basis"] = "median_step_estimate"
    else:
        report["decode_throughput"]["accepted_tokens_per_request_step"] = 4.0
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="fixed 3.47"):
        summary.render_summary(tmp_path, "abc")


@pytest.mark.parametrize(
    "command_status,agent_status", [(0, 0), (37, 0), (37, 9), (0, 9)]
)
def test_host_hook_publishes_and_preserves_command_failure(
    tmp_path, command_status, agent_status
):
    (tmp_path / ".buildkite").symlink_to(_REPO / ".buildkite", target_is_directory=True)
    root = (
        tmp_path / "perf_eval_results/deepseek-v41-flash-dp8-96k-c384-decode/benchmark"
    )
    write_result(root, "decode")
    agent = tmp_path / "buildkite-agent"
    agent.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > agent-args\ncat > published.md\n'
        f"exit {agent_status}\n"
    )
    agent.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "BUILDKITE_STEP_KEY": "dsv41_flash_dp8_c384_decode",
        "BUILDKITE_COMMAND_EXIT_STATUS": str(command_status),
        "BUILDKITE_COMMIT": "abc",
        "BUILDKITE_BUILD_URL": "https://buildkite.com/test/1",
        "BUILDKITE_JOB_ID": "job",
    }
    result = subprocess.run(
        ["bash", str(_REPO / ".buildkite/hooks/post-command")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == (command_status or (1 if agent_status else 0))
    published = (tmp_path / "published.md").read_text()
    assert "notify.github_check.output.summary" in (tmp_path / "agent-args").read_text()
    assert "Commit: `abc`" in published
    if command_status:
        assert "exit 37" in published and "tokens/s" not in published
    else:
        assert "800.00" in published


def test_host_hook_ignores_unrelated_steps(tmp_path):
    result = subprocess.run(
        ["bash", str(_REPO / ".buildkite/hooks/post-command")],
        cwd=tmp_path,
        env={**os.environ, "BUILDKITE_STEP_KEY": "build_docker"},
        capture_output=True,
    )
    assert result.returncode == 0
    assert not list(tmp_path.iterdir())
