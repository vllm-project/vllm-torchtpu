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
"""Tests for executors/worker_spawn.py (see its module docstring)."""

import fcntl
import multiprocessing
import multiprocessing.popen_spawn_posix as popen_spawn_posix
import os
import threading
import time
from types import SimpleNamespace

import pytest
from vllm.v1.executor import multiproc_executor

import vllm_torchtpu
from vllm_torchtpu.executors import worker_spawn
from vllm_torchtpu.executors.worker_spawn import (
    _LargePipeOs,
    exclusive_spawn,
    spawn_pipe_size,
)

# Several times the ~75 KB Qwen3.8-Flash-FP8 spawn payload, for headroom.
_PAYLOAD = 500_000


def _resizable_size() -> int:
    """The spawn pipe size this host grants, or skip the test."""
    size = spawn_pipe_size(8)
    if size <= 512 * 1024:
        pytest.skip(f"spawn pipes would be {size} bytes on this host")
    r, w = os.pipe()
    try:
        fcntl.fcntl(w, fcntl.F_SETPIPE_SZ, size)
    except OSError as e:
        pytest.skip(f"the kernel refuses F_SETPIPE_SZ here: {e}")
    finally:
        os.close(r)
        os.close(w)
    return size


def test_large_pipe_holds_a_vllm_config_sized_payload():
    size = _resizable_size()
    r, w = _LargePipeOs(size).pipe()
    try:
        assert fcntl.fcntl(w, fcntl.F_GETPIPE_SZ) >= size
        # The write a spawn makes returns without a reader.
        os.set_blocking(w, False)
        assert os.write(w, b"x" * _PAYLOAD) == _PAYLOAD
    finally:
        os.close(r)
        os.close(w)


def test_one_worker_keeps_the_default_pipes():
    """One local worker has no start to overlap with, so its pipes stay
    default-sized instead of keeping 1 MiB charged to the user for its
    lifetime."""
    assert spawn_pipe_size(1) == 0
    with exclusive_spawn(1):
        assert popen_spawn_posix.os is os


def test_a_small_pipe_max_size_does_not_shrink_the_pipe():
    r, w = os.pipe()
    default = fcntl.fcntl(w, fcntl.F_GETPIPE_SZ)
    os.close(r)
    os.close(w)
    r, w = _LargePipeOs(4096).pipe()
    try:
        assert fcntl.fcntl(w, fcntl.F_GETPIPE_SZ) == default
    finally:
        os.close(r)
        os.close(w)


def test_multi_worker_pipes_are_1_mib_capped_at_the_system_maximum(monkeypatch):
    monkeypatch.setattr(worker_spawn, "_pipe_max_size", lambda: 64 << 20)
    assert spawn_pipe_size(8) == 1 << 20
    monkeypatch.setattr(worker_spawn, "_pipe_max_size", lambda: 512 << 10)
    assert spawn_pipe_size(8) == 512 << 10


@pytest.mark.parametrize(
    "error",
    [OSError("ENOENT"), ValueError("not a number")],
    ids=["oserror", "valueerror"],
)
def test_an_unreadable_pipe_max_size_requests_1_mib_and_warns(monkeypatch, error):
    calls = []
    monkeypatch.setattr(
        worker_spawn.logger,
        "warning_once",
        lambda msg, *args, **kwargs: calls.append((msg, args)),
    )

    def unreadable():
        raise error

    monkeypatch.setattr(worker_spawn, "_pipe_max_size", unreadable)
    assert spawn_pipe_size(8) == 1 << 20

    assert len(calls) == 1
    msg, args = calls[0]
    assert msg % args == (
        f"Could not read /proc/sys/fs/pipe-max-size ({error}); requesting "
        f"{1 << 20}-byte worker spawn pipes."
    )
    # warning_once caches on its args: an exception object would warn again
    # on every call and keep the traceback alive.
    assert [type(a) for a in args] == [str, int]


def test_a_failed_resize_falls_back_and_warns_with_plain_args(monkeypatch):
    calls = []
    monkeypatch.setattr(
        worker_spawn.logger,
        "warning_once",
        lambda msg, *args, **kwargs: calls.append((msg, args)),
    )

    def fail(*args):
        raise OSError("EPERM")

    monkeypatch.setattr(fcntl, "fcntl", fail)
    r, w = _LargePipeOs(1 << 20).pipe()
    os.close(r)
    os.close(w)
    assert len(calls) == 1
    msg, args = calls[0]
    assert "EPERM" in msg % args
    # warning_once caches on its args: an exception object would make every
    # pipe warn again and keep the traceback alive.
    assert all(isinstance(a, (int, str)) for a in args)


def test_large_pipe_os_is_os_otherwise():
    assert _LargePipeOs(1 << 20).getpid() == os.getpid()
    assert _LargePipeOs(1 << 20).close is os.close


def _noop():
    pass


def test_a_real_spawn_gets_the_large_pipes():
    """Goes through CPython's Popen._launch: if it stops reaching os.pipe
    through popen_spawn_posix's module global, this fails instead of the patch
    silently doing nothing."""
    size = _resizable_size()
    process = multiprocessing.get_context("spawn").Process(target=_noop)
    with exclusive_spawn(8):
        process.start()
    try:
        # The sentinel is one of the two pipes _launch creates.
        assert fcntl.fcntl(process.sentinel, fcntl.F_GETPIPE_SZ) >= size
        process.join(120)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
    assert popen_spawn_posix.os is os


@pytest.fixture
def patched_spawn(monkeypatch):
    """Install the worker-spawn wrapper over a fake make_worker_process."""
    calls = []

    def install(make_worker_process):
        monkeypatch.setattr(
            multiproc_executor.WorkerProc,
            "make_worker_process",
            staticmethod(make_worker_process),
        )
        monkeypatch.setattr(
            multiproc_executor.WorkerProc,
            "_tpu_global_rank_env_patch",
            False,
            raising=False,
        )

        def binding(pc, rank, local_rank, env, dp_collectives=False):
            return SimpleNamespace(
                as_env=lambda: {"RANK": str(rank)},
                rank=rank,
                local_rank=local_rank,
                world_size=8,
                local_world_size=pc.local_world_size,
                dp_rank=0,
                dp_size=1,
                local_rank_offset=0,
                init_local_rank=local_rank,
            )

        monkeypatch.setattr(
            "vllm_torchtpu.worker.tpu_rank_binding.get_tpu_worker_binding",
            binding,
        )
        monkeypatch.setenv("VLLM_TPU_DEVICE_ORDER", "")
        monkeypatch.setattr(
            "vllm_torchtpu.models.parallel_requirements.requires_dp_collectives",
            lambda vllm_config: False,
        )
        vllm_torchtpu._patch_multiproc_worker_global_rank_env()
        return multiproc_executor.WorkerProc.make_worker_process

    return install, calls


def test_the_spawn_wrapper_keeps_the_real_os_for_one_worker(patched_spawn):
    """One local worker: the wrapper still serializes the binding env writes,
    but popen_spawn_posix keeps the real os module."""
    install, seen = patched_spawn

    def make_worker_process(vllm_config, local_rank, rank, *args, **kwargs):
        seen.append(popen_spawn_posix.os)
        return "handle"

    spawn = install(make_worker_process)
    config = SimpleNamespace(parallel_config=SimpleNamespace(local_world_size=1))
    assert spawn(config, 0, 0) == "handle"

    assert seen == [os]


@pytest.mark.parametrize("raises", [False, True])
def test_the_spawn_patch_swaps_os_only_during_the_spawn(patched_spawn, raises):
    install, seen = patched_spawn

    def make_worker_process(vllm_config, local_rank, rank, *args, **kwargs):
        seen.append(popen_spawn_posix.os)
        if raises:
            raise RuntimeError("spawn failed")
        return "handle"

    spawn = install(make_worker_process)
    config = SimpleNamespace(parallel_config=SimpleNamespace(local_world_size=8))
    if raises:
        with pytest.raises(RuntimeError):
            spawn(config, 0, 0)
    else:
        assert spawn(config, 0, 0) == "handle"

    assert isinstance(seen[0], _LargePipeOs)
    assert popen_spawn_posix.os is os


def test_spawns_from_several_threads_take_turns(patched_spawn):
    """MultiprocExecutor may start workers from a thread pool (e.g. on
    --headless hosts): each spawn's env writes and pipe swap must not
    interleave with another's, and `os` must come back afterwards."""
    install, _ = patched_spawn
    active, overlaps, swapped, ranks = [0], [], [], []
    guard = threading.Lock()

    def make_worker_process(vllm_config, local_rank, rank, *args, **kwargs):
        with guard:
            active[0] += 1
            overlaps.append(active[0])
        swapped.append(isinstance(popen_spawn_posix.os, _LargePipeOs))
        time.sleep(0.01)
        # Still this spawn's binding: no other spawn wrote os.environ meanwhile.
        ranks.append((rank, os.environ["RANK"]))
        with guard:
            active[0] -= 1
        return rank

    spawn = install(make_worker_process)
    config = SimpleNamespace(parallel_config=SimpleNamespace(local_world_size=8))
    results = [None] * 8
    threads = [
        threading.Thread(target=lambda i=i: results.__setitem__(i, spawn(config, i, i)))
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == list(range(8))
    assert max(overlaps) == 1
    assert all(swapped)
    assert sorted(ranks) == [(i, str(i)) for i in range(8)]
    assert popen_spawn_posix.os is os


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("before", [None, "7"], ids=["unset", "set"])
def test_the_spawn_env_is_restored_after_the_spawn(
    patched_spawn, monkeypatch, raises, before
):
    """The child sees its binding while it starts; afterwards the parent's
    environment is what it was, so no spawn's binding reaches the next one."""
    if before is None:
        monkeypatch.delenv("RANK", raising=False)
    else:
        monkeypatch.setenv("RANK", before)
    install, _ = patched_spawn
    seen = []

    def make_worker_process(vllm_config, local_rank, rank, *args, **kwargs):
        seen.append(os.environ["RANK"])
        if raises:
            raise RuntimeError("spawn failed")
        return "handle"

    spawn = install(make_worker_process)
    config = SimpleNamespace(parallel_config=SimpleNamespace(local_world_size=8))
    if raises:
        with pytest.raises(RuntimeError):
            spawn(config, 3, 3)
    else:
        assert spawn(config, 3, 3) == "handle"

    assert seen == ["3"]
    assert os.environ.get("RANK") == before
