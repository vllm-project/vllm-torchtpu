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
"""Worker spawns must not wait on each other.

Spawning writes the pickled process object (vllm_config included) into a
pipe the child reads only once it has started Python. With the default
64 KiB pipe that write blocks the parent until then, so vLLM's spawn loop
started the 8 workers one after another; the spawn patch enlarges the pipes
of each spawn so the write returns at once.
"""

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
    large_spawn_pipes,
    spawn_pipe_size,
)

# Several times the ~75 KB Qwen3.8-Flash-FP8 spawn payload, for headroom.
_PAYLOAD = 500_000


def _resizable_size() -> int:
    """The spawn pipe size this host grants, or skip the test."""
    size = spawn_pipe_size()
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


def test_pipe_size_defaults_to_1_mib_capped_at_the_system_maximum(monkeypatch):
    monkeypatch.delenv("TPU_WORKER_SPAWN_PIPE_BYTES", raising=False)
    monkeypatch.setattr(worker_spawn, "_pipe_max_size", lambda: 64 << 20)
    assert spawn_pipe_size() == 1 << 20
    monkeypatch.setattr(worker_spawn, "_pipe_max_size", lambda: 512 << 10)
    assert spawn_pipe_size() == 512 << 10


def test_pipe_size_follows_the_env_var(monkeypatch):
    monkeypatch.setattr(worker_spawn, "_pipe_max_size", lambda: 64 << 20)
    monkeypatch.setenv("TPU_WORKER_SPAWN_PIPE_BYTES", str(4 << 20))
    assert spawn_pipe_size() == 4 << 20
    monkeypatch.setenv("TPU_WORKER_SPAWN_PIPE_BYTES", "-1")
    with pytest.raises(ValueError):
        spawn_pipe_size()


def test_zero_keeps_the_default_pipes(monkeypatch):
    monkeypatch.setenv("TPU_WORKER_SPAWN_PIPE_BYTES", "0")
    with large_spawn_pipes():
        assert popen_spawn_posix.os is os


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
    with large_spawn_pipes():
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
        binding = SimpleNamespace(
            as_env=dict,
            rank=0,
            local_rank=0,
            world_size=1,
            local_world_size=1,
            dp_rank=0,
            dp_size=1,
            local_rank_offset=0,
            init_local_rank=0,
        )
        monkeypatch.setattr(
            "vllm_torchtpu.worker.tpu_rank_binding.get_tpu_worker_binding",
            lambda *args, **kwargs: binding,
        )
        monkeypatch.setenv("VLLM_TPU_DEVICE_ORDER", "")
        monkeypatch.setenv("TPU_WORKER_SPAWN_PIPE_BYTES", str(1 << 20))
        vllm_torchtpu._patch_multiproc_worker_global_rank_env()
        return multiproc_executor.WorkerProc.make_worker_process

    return install, calls


@pytest.mark.parametrize("raises", [False, True])
def test_the_spawn_patch_swaps_os_only_during_the_spawn(patched_spawn, raises):
    install, seen = patched_spawn

    def make_worker_process(vllm_config, local_rank, rank, *args, **kwargs):
        seen.append(popen_spawn_posix.os)
        if raises:
            raise RuntimeError("spawn failed")
        return "handle"

    spawn = install(make_worker_process)
    config = SimpleNamespace(parallel_config=SimpleNamespace())
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
    active, overlaps, swapped = [0], [], []
    guard = threading.Lock()

    def make_worker_process(vllm_config, local_rank, rank, *args, **kwargs):
        with guard:
            active[0] += 1
            overlaps.append(active[0])
        swapped.append(isinstance(popen_spawn_posix.os, _LargePipeOs))
        time.sleep(0.01)
        with guard:
            active[0] -= 1
        return rank

    spawn = install(make_worker_process)
    config = SimpleNamespace(parallel_config=SimpleNamespace())
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
    assert popen_spawn_posix.os is os
