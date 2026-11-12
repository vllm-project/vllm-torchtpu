"""The configuration checker must reject failed launcher dry runs."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

BASH = os.environ.get("BASH_EXECUTABLE") or shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash is required")
CHECKER = Path(
    os.environ.get(
        "CHECK_POINTS_SCRIPT",
        str(
            Path(__file__).resolve().parents[2] /
            "scripts/vllm/benchmark/inferencex/qwen3.5/check_points.sh")))


def run_check(tmp_path, sweep, server, *points):
    for name, text in (("check_points.sh", CHECKER.read_text()),
                       ("sweep.sh", sweep), ("server.sh", server)):
        (tmp_path / name).write_text(text, encoding="utf-8", newline="\n")
    return subprocess.run([BASH, "check_points.sh", *(points or ("32", ))],
                          cwd=tmp_path,
                          capture_output=True,
                          text=True,
                          timeout=20)


SETTINGS = "echo SHARDING=TP8_EP\n"


def test_matching_successful_launchers_pass(tmp_path):
    result = run_check(tmp_path, SETTINGS, SETTINGS)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("sweep,server", [
    ("echo failed >&2; exit 7\n", "echo failed >&2; exit 7\n"),
    (SETTINGS + "exit 7\n", SETTINGS),
    (SETTINGS, SETTINGS + "exit 7\n"),
])
def test_launcher_failure_is_not_a_match(tmp_path, sweep, server):
    result = run_check(tmp_path, sweep, server)
    assert result.returncode != 0
    assert "dry run failed" in result.stderr


def test_missing_settings_fail(tmp_path):
    result = run_check(tmp_path, "exit 0\n", "exit 0\n")
    assert result.returncode != 0


def test_different_settings_fail(tmp_path):
    result = run_check(tmp_path, SETTINGS, "echo SHARDING=DP8_EP\n")
    assert result.returncode != 0


def test_failure_persists_after_a_later_success(tmp_path):
    sweep = 'if [ "$CONCS" = 8 ]; then echo failed >&2; exit 7; fi\n' + SETTINGS
    result = run_check(tmp_path, sweep, SETTINGS, "4", "8", "16")
    assert result.returncode != 0
    assert "16 users:" in result.stdout
