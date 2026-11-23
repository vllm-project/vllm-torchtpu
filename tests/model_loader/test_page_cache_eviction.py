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


def test_every_rank_evicts(evicted, monkeypatch):
    # Each rank stops its own prefetch, waits for every rank's loads, then
    # evicts. A non-zero local rank must evict too (it used to skip it).
    monkeypatch.setenv("LOCAL_RANK", "3")
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
    monkeypatch.setattr(
        mlp, "_local_tensor_ranges", lambda p, ids, unused=None: [(0, 64 * 4)]
    )

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
    mlp._start_ep_range_prefetch([str(path)], set(), None, num_threads=1, block_size=64)
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


def test_disabled_still_stops_prefetch(evicted, monkeypatch):
    # Once every model is loaded the prefetch serves no read: it stops even
    # without eviction.
    monkeypatch.setattr(envs, "TPU_EVICT_WEIGHTS_PAGE_CACHE", False, raising=False)
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait)
    thread.start()
    mlp._prefetches.append((thread, stop))
    mlp.evict_loaded_weights_page_cache()
    assert stop.is_set()
    assert not thread.is_alive()
    assert not mlp._prefetches
    assert evicted == []


def test_eviction_waits_for_every_loader(evicted, monkeypatch):
    # Stop, then wait for every rank's loads, then evict. A rank with no
    # files still waits, so every rank enters the wait.
    order: list[str] = []
    monkeypatch.setattr(mlp, "_stop_ep_range_prefetch", lambda: order.append("stop"))
    monkeypatch.setattr(mlp, "_wait_for_every_loader", lambda: order.append("wait"))
    monkeypatch.setattr(
        mlp, "_evict_checkpoint_page_cache", lambda files: order.append("evict")
    )
    mlp.evict_loaded_weights_page_cache()
    assert order == ["stop", "wait"]
    order.clear()
    mlp._loaded_weight_files.update(dict.fromkeys(["a"]))
    mlp.evict_loaded_weights_page_cache()
    assert order == ["stop", "wait", "evict"]


def test_wait_for_every_loader_without_distributed_state():
    # A single process has no world group: nothing to wait for.
    mlp._wait_for_every_loader()


class _Group:
    world_size = 2
    cpu_group = object()


def test_wait_for_every_loader_is_bounded(monkeypatch):
    from vllm.distributed import parallel_state

    calls = []

    def barrier(group, timeout):
        calls.append((group, timeout.total_seconds()))

    monkeypatch.setattr(parallel_state, "get_world_group", lambda: _Group)
    monkeypatch.setattr(mlp.torch.distributed, "monitored_barrier", barrier)
    mlp._wait_for_every_loader(timeout_s=12.0)
    assert calls == [(_Group.cpu_group, 12.0)]


def test_load_fails_fast_when_a_rank_never_arrives(evicted, monkeypatch):
    # A rank that failed or stalled before the end of loading fails the load
    # (the barrier's error, naming a rank that did not arrive, is chained),
    # not a hang.
    from vllm.distributed import parallel_state

    def barrier(group, timeout):
        raise RuntimeError("Rank 1 failed to pass monitoredBarrier")

    monkeypatch.setattr(parallel_state, "get_world_group", lambda: _Group)
    monkeypatch.setattr(mlp.torch.distributed, "monitored_barrier", barrier)
    mlp._loaded_weight_files.update(dict.fromkeys(["a"]))
    with pytest.raises(RuntimeError, match="not every rank finished loading"):
        mlp.evict_loaded_weights_page_cache()
    assert evicted == []


def test_patched_loader_records_loaded_files(evicted, monkeypatch):
    # The eviction evicts what the patched DefaultModelLoader recorded.
    from vllm.model_executor.model_loader import default_loader as dl

    class Loader:
        def _prepare_weights(self, *args, **kwargs):
            return ("folder", ["shard-1", "shard-2"], False)

        def load_weights(self, model, model_config):
            pass

    monkeypatch.setattr(dl, "DefaultModelLoader", Loader)
    mlp.patch_default_model_loader_page_cache()
    loader = Loader()
    loader._prepare_weights("model", None, True)
    loader.load_weights(None, None)
    assert list(mlp._loaded_weight_files) == ["shard-1", "shard-2"]
    mlp.evict_loaded_weights_page_cache()
    assert evicted == [["shard-1", "shard-2"]]


def _write_safetensors(path, names_and_sizes):
    import json
    import struct

    header, offset = {}, 0
    for name, size in names_and_sizes:
        header[name] = {
            "dtype": "U8",
            "shape": [size],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)
    return 8 + len(raw)


def test_ranges_skip_the_loading_models_unused_tensors(tmp_path):
    # The prefetch reads what the lazy iterator reads: not a draft's unused
    # tensors.
    path = tmp_path / "shard.safetensors"
    base = _write_safetensors(
        path,
        [
            ("model.a.weight", 16),
            ("model.unused.weight", 32),
            ("model.b.weight", 8),
        ],
    )
    keep_all = mlp._local_tensor_ranges(str(path), set())
    assert keep_all == [(base, base + 56)]
    used = mlp._local_tensor_ranges(str(path), set(), lambda name: "unused" in name)
    assert used == [(base, base + 16), (base + 48, base + 56)]


def test_stop_during_header_scan_skips_reads(tmp_path, monkeypatch):
    # A stop that lands while the headers are scanned ends the scan at the
    # next file and starts no read.
    paths = []
    for i in range(3):
        p = tmp_path / f"shard-{i}.safetensors"
        p.write_bytes(b"\0" * 64)
        paths.append(str(p))
    monkeypatch.setattr(mlp, "_prefetches", [])
    scanned: list[str] = []
    in_scan = threading.Event()

    def ranges(path, ids, unused=None):
        scanned.append(path)
        _, stop = mlp._prefetches[0]
        in_scan.set()
        assert stop.wait(timeout=5.0)
        return [(0, 64)]

    def no_pread(*args):
        raise AssertionError("a stopped prefetch read")

    monkeypatch.setattr(mlp, "_local_tensor_ranges", ranges)
    monkeypatch.setattr(os, "pread", no_pread)
    mlp._start_ep_range_prefetch(paths, set(), None, num_threads=1, block_size=64)
    assert in_scan.wait(timeout=5.0)
    mlp._stop_ep_range_prefetch()
    assert scanned == paths[:1]


def test_ended_prefetch_threads_are_dropped(tmp_path, monkeypatch):
    path = tmp_path / "shard.safetensors"
    path.write_bytes(b"\0" * 64)
    monkeypatch.setattr(mlp, "_prefetches", [])
    monkeypatch.setattr(
        mlp, "_local_tensor_ranges", lambda p, ids, unused=None: [(0, 64)]
    )
    mlp._start_ep_range_prefetch([str(path)], set(), None, num_threads=1, block_size=64)
    first, _ = mlp._prefetches[0]
    first.join(timeout=5.0)
    assert not first.is_alive()
    mlp._start_ep_range_prefetch([str(path)], set(), None, num_threads=1, block_size=64)
    assert [t for t, _ in mlp._prefetches] != [first]
    assert len(mlp._prefetches) == 1
    mlp._stop_ep_range_prefetch()
