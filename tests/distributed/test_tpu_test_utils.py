# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from .tpu_test_utils import (
    TPU_LAUNCH_ENV_KEYS,
    clear_tpu_launch_env,
    find_vfio_holder_pids,
    has_tpu_vfio_devices,
    is_transient_tpu_session_error,
    kill_process_group,
    prepare_tpu_environment,
    run_tpu_worker_session,
    wait_for_vfio_idle,
)


def test_has_tpu_vfio_devices_positive_and_cpu_lookalike_controls(
    tmp_path: Path,
) -> None:
    # Negative control 1: Linux CPU host with /dev/vfio/vfio only (no numeric groups).
    cpu_vfio = tmp_path / "cpu_vfio"
    cpu_vfio.mkdir()
    (cpu_vfio / "vfio").write_text("", encoding="utf-8")
    assert not has_tpu_vfio_devices(vfio_dir=str(cpu_vfio), environ={})

    # Positive control: TPU pod with numeric IOMMU group /dev/vfio/0 + /dev/vfio/vfio.
    tpu_vfio = tmp_path / "tpu_vfio"
    tpu_vfio.mkdir()
    (tpu_vfio / "vfio").write_text("", encoding="utf-8")
    (tpu_vfio / "0").write_text("", encoding="utf-8")
    assert has_tpu_vfio_devices(vfio_dir=str(tpu_vfio), environ={})

    # Negative control 2: TPU host explicitly running in JAX_PLATFORMS=cpu or SHAPE=cpu.
    assert not has_tpu_vfio_devices(
        vfio_dir=str(tpu_vfio), environ={"JAX_PLATFORMS": "cpu"}
    )
    assert not has_tpu_vfio_devices(vfio_dir=str(tpu_vfio), environ={"SHAPE": "cpu"})

    # Negative control 3: Missing /dev/vfio directory.
    assert not has_tpu_vfio_devices(vfio_dir=str(tmp_path / "absent"), environ={})


def test_prepare_tpu_environment_prefers_public_api(monkeypatch):
    """When torch.tpu.distributed.set_tpu_launch_env is available (torch_tpu >= 0.1.2),
    prepare_tpu_environment should dispatch to it.
    """
    mock_set_tpu_launch_env = MagicMock()
    mock_dist = SimpleNamespace(set_tpu_launch_env=mock_set_tpu_launch_env)
    mock_tpu = SimpleNamespace(distributed=mock_dist)

    monkeypatch.setattr(torch, "tpu", mock_tpu, raising=False)

    prepare_tpu_environment(world_size=8)
    mock_set_tpu_launch_env.assert_called_once_with(nproc_per_node=8)


def test_prepare_tpu_environment_falls_back_to_legacy_singlehost_wrapper(monkeypatch):
    """When running on torch_tpu <= 0.1.1, dispatch to legacy singlehost_wrapper."""
    monkeypatch.setattr(torch, "tpu", SimpleNamespace(distributed=None), raising=False)

    mock_legacy_prepare = MagicMock()
    legacy_mod_name = "torch_tpu._internal.distributed.launchers.singlehost_wrapper"
    mock_legacy_module = ModuleType(legacy_mod_name)
    mock_legacy_module.prepare_tpu_environment = mock_legacy_prepare

    with patch.dict("sys.modules", {legacy_mod_name: mock_legacy_module}):
        prepare_tpu_environment(world_size=8)
        mock_legacy_prepare.assert_called_once_with(world_size=8)


def test_clear_tpu_launch_env_removes_only_tpu_launch_keys() -> None:
    env = {key: f"val_{idx}" for idx, key in enumerate(TPU_LAUNCH_ENV_KEYS)}
    env["UNRELATED_VAR"] = "keep_me"
    clear_tpu_launch_env(env)
    for key in TPU_LAUNCH_ENV_KEYS:
        assert key not in env
    assert env["UNRELATED_VAR"] == "keep_me"


def test_is_transient_tpu_session_error_positive_and_negative_controls() -> None:
    # Positive controls: real Buildkite multichip failure signatures.
    assert is_transient_tpu_session_error(
        "RuntimeError: failed to retrieve the PjRt client: "
        "TPU initialization failed: GRPC_ERROR error message: START_SESSION failed"
    )
    assert is_transient_tpu_session_error(
        "E0000 00:00:1234 slice failure: SLICE_FAILURE_SW_INJECT_ERROR"
    )
    assert is_transient_tpu_session_error(
        "TPU_WORKER_BOOTSTRAP_TIMEOUT: 3/8 workers initialized TPU runtime within 150s"
    )

    # Negative controls: comments, empty logs, and non-session assertion failures.
    assert not is_transient_tpu_session_error("")
    assert not is_transient_tpu_session_error(
        "# START_SESSION failed in comment should not trigger retry\n"
        "   # SLICE_FAILURE_SW_INJECT_ERROR\n"
        "   # TPU_WORKER_BOOTSTRAP_TIMEOUT: comment lookalike\n"
    )
    assert not is_transient_tpu_session_error(
        "AssertionError: tensor mismatch at index 4\n"
        "RuntimeError: normal shape validation error"
    )


def test_find_vfio_holder_pids_positive_and_lookalike_symlinks(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    vfio_prefix = "/dev/vfio"

    # PID 101 holds /dev/vfio/0 -> positive match.
    fd_101 = proc_root / "101" / "fd"
    fd_101.mkdir(parents=True)
    os.symlink("/dev/vfio/0", fd_101 / "3")

    # PID 202 holds /dev/vfio_lookalike/0 and /dev/null -> negative control.
    fd_202 = proc_root / "202" / "fd"
    fd_202.mkdir(parents=True)
    os.symlink("/dev/vfio_lookalike/0", fd_202 / "3")
    os.symlink("/dev/null", fd_202 / "4")

    # Non-numeric /proc entries ("self", "net") -> ignored.
    (proc_root / "self").mkdir()

    # Ignored self PID (303) holding /dev/vfio/vfio -> excluded.
    fd_303 = proc_root / "303" / "fd"
    fd_303.mkdir(parents=True)
    os.symlink("/dev/vfio/vfio", fd_303 / "5")

    holders = find_vfio_holder_pids(
        proc_root=str(proc_root),
        vfio_prefix=vfio_prefix,
        ignore_pid=303,
    )
    assert holders == [101]


def test_wait_for_vfio_idle_returns_when_holder_releases(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    fd_dir = proc_root / "444" / "fd"
    fd_dir.mkdir(parents=True)
    link = fd_dir / "7"
    os.symlink("/dev/vfio/1", link)
    os.unlink(link)

    remaining = wait_for_vfio_idle(
        proc_root=str(proc_root),
        vfio_prefix=str(tmp_path / "no_vfio_dir"),
        timeout_s=0.2,
        settle_s=0.0,
        poll_s=0.01,
    )
    assert remaining == []


def test_run_tpu_worker_session_retries_transient_error_only() -> None:
    calls: list[int] = []

    def transient_then_pass(attempt: int) -> tuple[int, str, str]:
        calls.append(attempt)
        if attempt == 0:
            return 1, "GRPC_ERROR: START_SESSION failed", f"dir_{attempt}"
        if attempt == 1:
            return (
                124,
                "TPU_WORKER_BOOTSTRAP_TIMEOUT: 0/8 workers initialized "
                "TPU runtime within 150s",
                f"dir_{attempt}",
            )
        return 0, "ok", f"dir_{attempt}"

    rc, log, payload = run_tpu_worker_session(
        transient_then_pass, max_attempts=3, backoff_s=0.0, settle_s=0.0
    )
    assert (rc, log, payload) == (0, "ok", "dir_2")
    assert calls == [0, 1, 2]

    # Deterministic test failure and post-bootstrap transfer deadline (rc=124 without
    # bootstrap marker) must NOT waste retries.
    non_transient_calls: list[int] = []

    def deterministic_failure(attempt: int) -> tuple[int, str, str]:
        non_transient_calls.append(attempt)
        return 1, "AssertionError: numerical mismatch", f"dir_{attempt}"

    rc2, _, payload2 = run_tpu_worker_session(
        deterministic_failure, max_attempts=3, backoff_s=0.0, settle_s=0.0
    )
    assert rc2 == 1
    assert payload2 == "dir_0"
    assert non_transient_calls == [0]

    post_bootstrap_timeout_calls: list[int] = []

    def post_bootstrap_timeout(attempt: int) -> tuple[int, str, str]:
        post_bootstrap_timeout_calls.append(attempt)
        return 124, "Workers bootstrapped; hung in transfer loop", f"dir_{attempt}"

    rc3, _, payload3 = run_tpu_worker_session(
        post_bootstrap_timeout, max_attempts=3, backoff_s=0.0, settle_s=0.0
    )
    assert rc3 == 124
    assert payload3 == "dir_0"
    assert post_bootstrap_timeout_calls == [0]

    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        run_tpu_worker_session(deterministic_failure, max_attempts=0)


def test_kill_process_group_reaps_grandchild_in_session(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"open({str(pid_file)!r}, 'w').write(str(p.pid)); "
        "time.sleep(60)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        start_new_session=True,
    )
    try:
        while not pid_file.exists() or not pid_file.read_text().strip():
            assert child.poll() is None
        grandchild_pid = int(pid_file.read_text().strip())
    finally:
        kill_process_group(child, sigterm_timeout_s=1.0)

    assert child.poll() is not None
    with contextlib.suppress(ProcessLookupError):
        os.kill(grandchild_pid, 0)


def test_call_with_tpu_session_retry_handles_engine_init_and_negative_controls() -> (
    None
):
    from tests.distributed.tpu_test_utils import call_with_tpu_session_retry

    attempts: list[int] = []
    cleanups: list[int] = []

    def engine_init_then_ok(attempt: int) -> str:
        attempts.append(attempt)
        if attempt == 0:
            raise RuntimeError(
                "Engine core initialization failed. See root cause above."
            )
        return f"engine_ok_{attempt}"

    result = call_with_tpu_session_retry(
        engine_init_then_ok,
        cleanup_fn=lambda: cleanups.append(len(attempts)),
        max_attempts=2,
        backoff_s=0.0,
        settle_s=0.0,
    )
    assert result == "engine_ok_1"
    assert attempts == [0, 1]
    assert cleanups == [1]

    # Negative control: non-transient RuntimeError / ValueError must NOT retry.
    bad_calls: list[int] = []

    def deterministic_runtime_error(attempt: int) -> str:
        bad_calls.append(attempt)
        raise RuntimeError("Unsupported model architecture")

    with pytest.raises(RuntimeError, match="Unsupported model architecture"):
        call_with_tpu_session_retry(
            deterministic_runtime_error,
            max_attempts=3,
            backoff_s=0.0,
            settle_s=0.0,
        )
    assert bad_calls == [0]

    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        call_with_tpu_session_retry(lambda _a: "ok", max_attempts=0)
