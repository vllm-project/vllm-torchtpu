# SPDX-License-Identifier: Apache-2.0
"""Freeze spy: the out-of-process monitor dumps native stacks when the
watched process stops writing heartbeats while a C call holds the GIL."""

import os
import shutil
import subprocess
import sys
import textwrap
import time

import pytest

from vllm_torchtpu import freeze_spy

SRC = os.path.dirname(os.path.dirname(os.path.abspath(freeze_spy.__file__)))

# The "API server": allows ptrace, starts the monitor, writes heartbeats, then
# blocks for several seconds in one C call that holds the GIL.
CHILD = textwrap.dedent(
    """
    import sys, time
    sys.set_int_max_str_digits(0)
    sys.path.insert(0, {src!r})
    import importlib.util
    spec = importlib.util.spec_from_file_location("freeze_spy", {path!r})
    fs = importlib.util.module_from_spec(spec); spec.loader.exec_module(fs)
    fs.allow_ptrace()
    fs.launch(2.0)
    for _ in range(6):
        fs.heartbeat("api", 0, 0.0)
        fs.heartbeat("rank0", 3, time.time())
        time.sleep(0.5)
    def held_gil():
        return len(str(7 ** 4_000_000))  # int -> str holds the GIL
    t0 = time.time()
    while time.time() - t0 < 12:
        held_gil()
    print("CHILD DONE", flush=True)
    """
)


def run_child(tmp_path, env_extra):
    env = {**os.environ, "CDK_OUTPUT_DIR": str(tmp_path), **env_extra}
    env.pop("VLLM_TPU_FREEZE_DIR", None)
    code = CHILD.format(src=SRC, path=freeze_spy.__file__)
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300
    )


def test_stale_heartbeat_triggers_dump(tmp_path):
    res = run_child(tmp_path, {})
    err = res.stderr
    assert "CHILD DONE" in res.stdout, err[-2000:]
    assert "Freeze spy: monitor watches pid" in err
    assert "FREEZE SPY dump1" in err, err[-2000:]
    assert "trigger: api: heartbeat not written" in err
    # /proc report: the GIL-holding main thread is busy.
    assert "%" in err and "states=" in err
    files = os.listdir(tmp_path / "freeze_spy")
    assert any(f.endswith("_dump1.txt") for f in files)
    if shutil.which("py-spy"):
        assert "py-spy --native pid" in err
        # The dump shows the Python frame stuck in the C call.
        assert "held_gil" in err, err[-4000:]


def test_progress_check_and_heartbeat_files(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_TPU_FREEZE_DIR", str(tmp_path))
    now = time.time()
    freeze_spy.heartbeat("rank0", 4, now - 50)
    freeze_spy.heartbeat("api", 0, 0.0)
    beats = freeze_spy._heartbeats(str(tmp_path))
    assert set(beats) == {"rank0", "api"}
    reasons = freeze_spy._stalled(beats, 30)
    assert reasons == ["rank0: no progress for 50 s with 4 live"]
    freeze_spy.heartbeat("rank0", 0, now - 50)  # idle engine: not a stall
    assert freeze_spy._stalled(freeze_spy._heartbeats(str(tmp_path)), 30) == []


@pytest.mark.parametrize("value,expected", [("0", 0.0), ("", 0.0), ("45", 45.0), ("x", 0.0)])
def test_seconds(monkeypatch, value, expected):
    monkeypatch.setenv("VLLM_TPU_FREEZE_PYSPY_S", value)
    assert freeze_spy.seconds() == expected
