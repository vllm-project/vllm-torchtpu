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
"""Tests for .buildkite/kubernetes/tpu_preflight.py and its hook in run.sh."""

import importlib.util
import os
import pathlib
import socket
import stat
import subprocess
import sys
import time

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_KUBE_DIR = _REPO_ROOT / ".buildkite" / "kubernetes"
_PREFLIGHT = _KUBE_DIR / "tpu_preflight.py"
_RUN_SH = _KUBE_DIR / "run.sh"


def _load_preflight():
    spec = importlib.util.spec_from_file_location("tpu_preflight", _PREFLIGHT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tpu_preflight = _load_preflight()


def _run_preflight(*args: str, **env: str) -> subprocess.CompletedProcess:
    run_env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    run_env.update(env)
    return subprocess.run(
        [sys.executable, str(_PREFLIGHT), *args],
        capture_output=True,
        text=True,
        env=run_env,
        timeout=60,
    )


def test_healthy_probe_passes_and_reports_devices():
    probe = "print('TPU_PREFLIGHT_DEVICES 4 x TPU7x', flush=True)"
    result = _run_preflight("--probe", probe)
    assert result.returncode == 0, result.stderr
    assert "TPU preflight passed" in result.stdout
    assert "4 x TPU7x" in result.stdout


def test_failed_probe_fails_with_diagnostics():
    probe = (
        "import sys\n"
        "print('Failed to connect to unix:///var/run/tpu-plugin/vbar.sock',"
        " file=sys.stderr)\n"
        "sys.exit(134)\n"
    )
    result = _run_preflight("--probe", probe, HOSTNAME="pod-on-a-dead-node")
    assert result.returncode == 1
    assert "TPU PREFLIGHT FAILED" in result.stderr
    assert "exit 134" in result.stderr
    assert "HOSTNAME=pod-on-a-dead-node" in result.stderr
    assert tpu_preflight.VBAR_SOCKET in result.stderr
    # The child's own error is carried into the verdict.
    assert "Failed to connect to unix:///var/run/tpu-plugin/vbar.sock" in (
        result.stderr
    )


def test_hung_probe_is_killed_at_the_deadline():
    start = time.monotonic()
    result = _run_preflight("--timeout", "1", "--probe", "import time; time.sleep(600)")
    assert result.returncode == 1
    assert "did not finish within 1s" in result.stderr
    assert time.monotonic() - start < 30


def test_hung_probe_grandchildren_are_killed(tmp_path):
    # The probe's whole session goes, so nothing outlives it to hold the chip.
    pid_file = tmp_path / "grandchild.pid"
    probe = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; "
        "time.sleep(600)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(600)\n"
    )
    ok, reason, _ = tpu_preflight.run_probe(probe, timeout_s=2)
    assert not ok
    assert "did not finish" in reason
    grandchild = int(pid_file.read_text())
    for _ in range(50):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(grandchild, 9)
        pytest.fail("the probe's grandchild survived the timeout")


def test_skipped_when_jax_is_told_not_to_use_tpu():
    result = _run_preflight(JAX_PLATFORMS="cpu")
    assert result.returncode == 0
    assert "TPU preflight skipped" in result.stdout


# Stands in for jax so the default probe runs as in a pod, minus the chip.
_FAKE_JAX = """
import os


class _Device:
    device_kind = "TPU7x"


def devices(backend=None):
    assert backend == "tpu", backend
    if os.environ.get("FAKE_JAX_BROKEN"):
        raise RuntimeError(
            "Unable to initialize backend 'tpu': Failed to connect to "
            "unix:///var/run/tpu-plugin/vbar.sock"
        )
    return [_Device() for _ in range(4)]
"""


@pytest.mark.parametrize("broken", [False, True])
def test_default_probe_against_fake_jax(tmp_path, broken):
    (tmp_path / "jax").mkdir()
    (tmp_path / "jax" / "__init__.py").write_text(_FAKE_JAX)
    env = {"PYTHONPATH": str(tmp_path)}
    if broken:
        env["FAKE_JAX_BROKEN"] = "1"
    result = _run_preflight(**env)
    if broken:
        assert result.returncode == 1
        assert "TPU PREFLIGHT FAILED" in result.stderr
        assert "Unable to initialize backend 'tpu'" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert "TPU preflight passed" in result.stdout
        assert "4 x TPU7x" in result.stdout


def test_vbar_socket_state(tmp_path):
    path = str(tmp_path / "vbar.sock")
    assert tpu_preflight._vbar_socket_state(path) == "absent"

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(path)
        server.listen(1)
        assert tpu_preflight._vbar_socket_state(path) == "accepting connections"
    # Bound and closed: the file stays behind with nobody listening, which is
    # how a dead device plugin leaves it.
    assert tpu_preflight._vbar_socket_state(path).startswith("connect failed")

    regular = tmp_path / "not-a-socket"
    regular.write_text("")
    assert tpu_preflight._vbar_socket_state(str(regular)) == (
        "present but not a socket"
    )


# A stand-in launcher: drops its own flags and runs the pod program locally.
_FAKE_LAUNCHER = """#!/usr/bin/env bash
while [[ $# -gt 0 && "$1" != "--" ]]; do shift; done
shift
exec "$@"
"""

# A stand-in for the pod's python3, so the hook can be tested without a TPU:
# logs the call and exits with whatever the test asks of it.
_FAKE_PYTHON = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_LOG/python3.log"
exit "${FAKE_PREFLIGHT_RC:-0}"
"""


def _run_run_sh(tmp_path: pathlib.Path, shape: str, preflight_rc: int):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, script in (("launch", _FAKE_LAUNCHER), ("python3", _FAKE_PYTHON)):
        path = bin_dir / name
        path.write_text(script)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    marker = tmp_path / "command-ran"
    env = {
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(tmp_path),
        "SHAPE": shape,
        "WORKLOAD_IMAGE": "example.invalid/image:tag",
        "K3_ENGINE_IMAGE": "example.invalid/image:tag",
        "KUBE_LAUNCHER": str(bin_dir / "launch"),
        "FAKE_LOG": str(tmp_path),
        "FAKE_PREFLIGHT_RC": str(preflight_rc),
    }
    result = subprocess.run(
        ["bash", str(_RUN_SH), "touch", str(marker)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    log = tmp_path / "python3.log"
    calls = log.read_text().splitlines() if log.exists() else []
    return result, marker.exists(), calls


def test_run_sh_tpu_shape_runs_preflight_then_command(tmp_path):
    result, ran, calls = _run_run_sh(tmp_path, "tpu7x-standard-1t/1x1x1", 0)
    assert result.returncode == 0, result.stderr
    assert calls == [".buildkite/kubernetes/tpu_preflight.py"]
    assert ran


def test_run_sh_failed_preflight_skips_command(tmp_path):
    result, ran, calls = _run_run_sh(tmp_path, "tpu7x-standard-1t/1x1x1", 1)
    assert result.returncode == 1
    assert calls == [".buildkite/kubernetes/tpu_preflight.py"]
    assert not ran


def test_run_sh_cpu_shape_has_no_preflight(tmp_path):
    result, ran, calls = _run_run_sh(tmp_path, "cpu", 1)
    assert result.returncode == 0, result.stderr
    assert calls == []
    assert ran
