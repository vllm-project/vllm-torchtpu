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
started the 8 workers one after another; the spawn patch enlarges the pipe
for each spawn so the write returns at once.
"""

import fcntl
import multiprocessing.popen_spawn_posix as popen_spawn_posix
import os
from types import SimpleNamespace

import pytest
from vllm.v1.executor import multiproc_executor

import vllm_torchtpu
from vllm_torchtpu import _LargePipeOs

_F_GETPIPE_SZ = getattr(fcntl, "F_GETPIPE_SZ", 1032)


def test_large_pipe_holds_a_vllm_config_sized_payload():
    r, w = _LargePipeOs.pipe()
    try:
        assert fcntl.fcntl(w, _F_GETPIPE_SZ) > 512 * 1024
        # The write a spawn makes returns without a reader.
        os.set_blocking(w, False)
        assert os.write(w, b"x" * 500_000) == 500_000
    finally:
        os.close(r)
        os.close(w)


def test_large_pipe_os_is_os_otherwise():
    assert _LargePipeOs().getpid() == os.getpid()
    assert _LargePipeOs().close is os.close


@pytest.mark.parametrize("raises", [False, True])
def test_the_spawn_patch_swaps_os_only_during_the_spawn(monkeypatch, raises):
    seen = []

    def make_worker_process(vllm_config, local_rank, rank, *args, **kwargs):
        seen.append(popen_spawn_posix.os)
        if raises:
            raise RuntimeError("spawn failed")
        return "handle"

    monkeypatch.setattr(
        multiproc_executor.WorkerProc,
        "make_worker_process",
        staticmethod(make_worker_process),
    )
    monkeypatch.delattr(
        multiproc_executor.WorkerProc, "_tpu_global_rank_env_patch", raising=False
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
    vllm_torchtpu._patch_multiproc_worker_global_rank_env()

    config = SimpleNamespace(parallel_config=SimpleNamespace())
    if raises:
        with pytest.raises(RuntimeError):
            multiproc_executor.WorkerProc.make_worker_process(config, 0, 0)
    else:
        assert multiproc_executor.WorkerProc.make_worker_process(config, 0, 0) == (
            "handle"
        )

    assert isinstance(seen[0], _LargePipeOs)
    assert popen_spawn_posix.os is os
