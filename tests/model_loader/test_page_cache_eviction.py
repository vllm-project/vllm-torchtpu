# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Unit tests for the deferred weight page cache eviction.

Eviction must wait until every model is loaded: an MTP draft re-reads the
target checkpoint, and evicting between the two loads makes a rank re-read it
from cold disk while every other rank waits. It must also stop the EP range
prefetch first, or the prefetch reads the evicted pages back in. No TPU, no
network.
"""

import os
import threading

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu import model_loader_patches as mlp


@pytest.fixture
def evicted(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(envs, "TPU_EVICT_WEIGHTS_PAGE_CACHE", True, raising=False)
    monkeypatch.setattr(mlp, "_evict_checkpoint_page_cache", calls.append)
    monkeypatch.setattr(mlp, "_loaded_weight_files", {})
    monkeypatch.setattr(mlp, "_prefetches", [])
    return calls


def test_files_from_every_load_evicted_once_in_load_order(evicted):
    # Target, then an MTP draft over the same shards.
    mlp._loaded_weight_files.update(dict.fromkeys(["a", "b"]))
    mlp._loaded_weight_files.update(dict.fromkeys(["b", "a", "c"]))
    mlp.evict_loaded_weights_page_cache()
    assert evicted == [["a", "b", "c"]]
    mlp.evict_loaded_weights_page_cache()
    assert evicted == [["a", "b", "c"]]


def test_files_evicted_concurrently(tmp_path, monkeypatch):
    """Multiple checkpoint files are evicted in parallel rather than serially."""
    paths = []
    for i in range(4):
        p = tmp_path / f"shard-{i}.safetensors"
        p.write_bytes(b"x" * 16)
        paths.append(str(p))

    barrier = threading.Barrier(len(paths), timeout=5.0)
    fadvise_fds: list[int] = []
    lock = threading.Lock()

    def fake_fadvise(fd: int, offset: int, length: int, advice: int) -> None:
        # All 4 files must enter posix_fadvise concurrently for the barrier to pass.
        barrier.wait()
        with lock:
            fadvise_fds.append(fd)

    monkeypatch.setattr(os, "posix_fadvise", fake_fadvise)
    mlp._evict_checkpoint_page_cache(paths)
    assert len(fadvise_fds) == 4


@pytest.mark.parametrize("local_rank", ["0", "3"])
def test_every_rank_evicts(evicted, monkeypatch, local_rank):
    # Each rank evicts after stopping its own prefetch, so the last eviction
    # on a host follows every rank's stop.
    monkeypatch.setenv("LOCAL_RANK", local_rank)
    mlp._loaded_weight_files.update(dict.fromkeys(["a"]))
    mlp.evict_loaded_weights_page_cache()
    assert evicted == [["a"]]
    assert not mlp._loaded_weight_files


def test_eviction_stops_prefetch_before_evicting(tmp_path, monkeypatch):
    """A prefetch still reading when eviction starts reads nothing after it.

    The draft model's prefetch outlives the draft load: it reads every local
    range of the checkpoint while the load needs a few tensors.
    """
    path = tmp_path / "shard-0.safetensors"
    path.write_bytes(b"\0" * (64 * 4))
    monkeypatch.setattr(envs, "TPU_EVICT_WEIGHTS_PAGE_CACHE", True, raising=False)
    monkeypatch.setattr(mlp, "_loaded_weight_files", {str(path): None})
    monkeypatch.setattr(mlp, "_prefetches", [])
    monkeypatch.setattr(mlp, "_local_tensor_ranges", lambda p, ids: [(0, 64 * 4)])

    events: list[str] = []
    lock = threading.Lock()
    first_read = threading.Event()
    real_pread = os.pread

    def pread_until_stopped(fd: int, size: int, off: int) -> bytes:
        # The first read holds until the eviction sets the prefetch's stop
        # event, so the reads still queued must be the ones it skips.
        with lock:
            events.append("read")
            first = not first_read.is_set()
        if first:
            # Registered before the thread starts, and only cleared by the stop.
            _, stop = mlp._prefetches[0]
            first_read.set()
            assert stop.wait(timeout=5.0)
        data = real_pread(fd, size, off)
        with lock:
            events.append("read-done")
        return data

    def fake_evict(files: list[str]) -> None:
        with lock:
            events.append("evict")

    monkeypatch.setattr(os, "pread", pread_until_stopped)
    monkeypatch.setattr(mlp, "_evict_checkpoint_page_cache", fake_evict)

    # One reader thread and a 64-byte block: four reads, one at a time.
    mlp._start_ep_range_prefetch([str(path)], set(), num_threads=1, block_size=64)
    assert first_read.wait(timeout=5.0)
    mlp.evict_loaded_weights_page_cache()

    # The eviction waits for the read in flight (the join) and skips the rest.
    assert events == ["read", "read-done", "evict"]
    assert not mlp._prefetches


def test_stop_without_prefetch_is_a_no_op(monkeypatch):
    monkeypatch.setattr(mlp, "_prefetches", [])
    mlp._stop_ep_range_prefetch()
    assert not mlp._prefetches


def test_disabled_leaves_files_untouched(evicted, monkeypatch):
    monkeypatch.setattr(envs, "TPU_EVICT_WEIGHTS_PAGE_CACHE", False, raising=False)
    mlp._loaded_weight_files.update(dict.fromkeys(["a"]))
    mlp.evict_loaded_weights_page_cache()
    assert evicted == []
