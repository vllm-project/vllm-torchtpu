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

import datetime
import json
import logging
import multiprocessing
import os
import threading
import time
from types import SimpleNamespace

import pytest
import torch

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


def _group(cpu_timeout):
    return SimpleNamespace(
        world_size=2, cpu_group=object(), cpu_timeout=cpu_timeout, local_rank=0
    )


@pytest.mark.parametrize(
    "cpu_timeout, expected_s",
    [(datetime.timedelta(seconds=12), 12.0), (None, 1800.0)],
    ids=["cpu-distributed-timeout", "torch-default"],
)
def test_wait_for_every_loader_uses_the_cpu_group_timeout(
    monkeypatch, cpu_timeout, expected_s
):
    # The CPU group's timeout (None: torch's 30-minute default) bounds the wait.
    from vllm.distributed import parallel_state

    calls = []

    def barrier(group, timeout):
        calls.append((group, timeout.total_seconds()))

    group = _group(cpu_timeout)
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: group)
    monkeypatch.setattr(mlp.torch.distributed, "monitored_barrier", barrier)
    mlp._wait_for_every_loader()
    assert calls == [(group.cpu_group, expected_s)]


def test_load_fails_fast_when_a_rank_never_arrives(evicted, monkeypatch):
    # A rank that failed or stalled before the end of loading fails the load
    # (the barrier's error, naming a rank that did not arrive, is chained),
    # not a hang.
    from vllm.distributed import parallel_state

    def barrier(group, timeout):
        raise RuntimeError("Rank 1 failed to pass monitoredBarrier")

    group = _group(datetime.timedelta(seconds=12))
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: group)
    monkeypatch.setattr(mlp.torch.distributed, "monitored_barrier", barrier)
    mlp._loaded_weight_files.update(dict.fromkeys(["a"]))
    with pytest.raises(
        RuntimeError, match="not every rank finished loading within 12 s"
    ):
        mlp.evict_loaded_weights_page_cache()
    assert evicted == []


# The CPU group's timeout in the two-rank test below.
_GLOO_TIMEOUT_S = 3
# Per case: when each rank arrives at the loader wait, in seconds after the
# case starts (None: never), and the outcome expected on each rank.
_TWO_RANK_CASES = {
    "rank1-late": ((0, 1), ("ok", "ok")),
    "rank1-absent": ((0, None), ("RuntimeError", "absent")),
    "rank0-absent": ((None, 0), ("absent", "RuntimeError")),
}


def _loader_wait_rank(rank, init_file, out_file):
    """One rank of a two-rank gloo world; runs the cases in turn.

    Each case has its own gloo group, made as vLLM makes the world group's
    CPU group: with the CPU timeout. A rank absent from a case waits for the
    next case on the default group instead, so it stays up until the other
    rank's wait has ended. Writes the outcomes to ``out_file``.
    """
    from vllm.distributed import parallel_state

    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        # The default group only lines the cases up.
        timeout=datetime.timedelta(seconds=60),
    )
    timeout = datetime.timedelta(seconds=_GLOO_TIMEOUT_S)
    # Made up front: making a group waits, at most its timeout, for both ranks.
    cpu_groups = [
        torch.distributed.new_group(backend="gloo", timeout=timeout)
        for _ in _TWO_RANK_CASES
    ]
    outcomes = {}
    for cpu_group, (case, (arrivals, _)) in zip(cpu_groups, _TWO_RANK_CASES.items()):
        group = SimpleNamespace(
            world_size=2, cpu_group=cpu_group, cpu_timeout=timeout, local_rank=rank
        )
        parallel_state.get_world_group = lambda group=group: group
        torch.distributed.barrier()
        if arrivals[rank] is None:
            outcomes[case] = {"outcome": "absent"}
            continue
        time.sleep(arrivals[rank])
        start = time.monotonic()
        try:
            mlp._wait_for_every_loader()
            outcome = {"outcome": "ok"}
        except Exception as e:
            outcome = {
                "outcome": type(e).__name__,
                "message": str(e),
                "cause": repr(e.__cause__),
            }
        outcome["elapsed_s"] = time.monotonic() - start
        outcomes[case] = outcome
    torch.distributed.barrier()
    with open(out_file, "w") as f:
        json.dump(outcomes, f)
    # Exit without the interpreter teardown, which a broken group could stall.
    os._exit(0)


@pytest.fixture(scope="module")
def two_rank_outcomes(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gloo")
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(
            target=_loader_wait_rank,
            args=(rank, str(tmp / "init"), str(tmp / f"rank{rank}.json")),
        )
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    deadline = time.monotonic() + 120
    try:
        for proc in procs:
            proc.join(timeout=max(0.0, deadline - time.monotonic()))
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.kill()
                proc.join()
    assert [proc.exitcode for proc in procs] == [0, 0]
    outcomes = [json.loads((tmp / f"rank{r}.json").read_text()) for r in range(2)]
    return {case: [o[case] for o in outcomes] for case in _TWO_RANK_CASES}


@pytest.mark.skipif(
    not (torch.distributed.is_available() and torch.distributed.is_gloo_available()),
    reason="needs torch.distributed with gloo",
)
@pytest.mark.parametrize("case", list(_TWO_RANK_CASES))
def test_two_rank_loader_wait(two_rank_outcomes, case):
    """A real gloo barrier: a late rank passes; an absent one fails every rank.

    ``monitored_barrier`` enforces its timeout on rank 0 only, so the
    rank0-absent case shows the gloo group's own timeout bounds rank 1.
    """
    outcomes = two_rank_outcomes[case]
    _, expected = _TWO_RANK_CASES[case]
    assert [o["outcome"] for o in outcomes] == list(expected), outcomes
    for rank, outcome in enumerate(outcomes):
        if outcome["outcome"] == "RuntimeError":
            assert f"loading within {_GLOO_TIMEOUT_S} s" in outcome["message"]
            assert outcome["elapsed_s"] < _GLOO_TIMEOUT_S + 5, outcomes
            if rank == 0:
                # monitored_barrier's error, chained, names the absent rank.
                assert "Rank 1" in outcome["cause"], outcomes
        elif outcome["outcome"] == "ok":
            assert outcome["elapsed_s"] < _GLOO_TIMEOUT_S, outcomes


class _RecordingLogger:
    """Stands in for the module's logger: records (level, message) pairs."""

    def __init__(self):
        self.records: list[tuple[int, str]] = []

    def log(self, level, msg, *args, **kwargs):
        self.records.append((level, msg % args))

    def info(self, msg, *args, **kwargs):
        self.log(logging.INFO, msg, *args)

    def warning(self, msg, *args, **kwargs):
        self.log(logging.WARNING, msg, *args)


@pytest.mark.parametrize(
    "local_rank, level", [(0, logging.INFO), (1, logging.DEBUG)], ids=["first", "other"]
)
def test_eviction_logs_at_info_once_per_host(tmp_path, monkeypatch, local_rank, level):
    # Every rank evicts; only the host's first rank logs the files and the
    # total at INFO.
    from vllm.distributed import parallel_state

    path = tmp_path / "shard.safetensors"
    path.write_bytes(b"\0" * 16)
    advice = []
    monkeypatch.setattr(os, "posix_fadvise", lambda fd, off, n, a: advice.append(a))
    monkeypatch.setattr(
        parallel_state,
        "get_world_group",
        lambda: SimpleNamespace(local_rank=local_rank),
    )
    logger = _RecordingLogger()
    monkeypatch.setattr(mlp, "logger", logger)
    mlp._evict_checkpoint_page_cache([str(path)])
    assert advice == [os.POSIX_FADV_DONTNEED]
    assert [lvl for lvl, msg in logger.records if "Evicted" in msg] == [level, level]


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

    reads = []

    def recording_pread(fd, size, off):
        reads.append((fd, size, off))
        return b""

    monkeypatch.setattr(mlp, "_local_tensor_ranges", ranges)
    monkeypatch.setattr(os, "pread", recording_pread)
    mlp._start_ep_range_prefetch(paths, set(), None, num_threads=1, block_size=64)
    assert in_scan.wait(timeout=5.0)
    mlp._stop_ep_range_prefetch()
    assert scanned == paths[:1]
    assert reads == []


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


# ---------------------------------------------------------------------------
# The installed patch: recorded files, stoppable whole-file prefetch
# ---------------------------------------------------------------------------


@pytest.fixture
def page_cache_patch(monkeypatch):
    """Applies the page cache patch (flag on); restores what it replaces."""
    from vllm.model_executor.model_loader import default_loader as dl
    from vllm.model_executor.model_loader import weight_utils

    monkeypatch.setattr(envs, "TPU_EVICT_WEIGHTS_PAGE_CACHE", True, raising=False)
    loader = dl.DefaultModelLoader
    monkeypatch.setattr(loader, "_prepare_weights", loader._prepare_weights)
    monkeypatch.setattr(loader, "load_weights", loader.load_weights)
    # setattr, not delattr(raising=False), so teardown removes the stamp.
    monkeypatch.setattr(loader, "_tpu_cache_evict_patch", False, raising=False)
    monkeypatch.setattr(
        weight_utils,
        "_prefetch_all_checkpoints",
        weight_utils._prefetch_all_checkpoints,
    )
    monkeypatch.setattr(mlp, "_upstream_prefetch_all_checkpoints", None)
    monkeypatch.setattr(mlp, "_prefetches", [])
    mlp.patch_default_model_loader_page_cache()


def _save_shards(tmp_path, count):
    from safetensors.torch import save_file

    paths = []
    for i in range(count):
        path = tmp_path / f"model-{i:05d}.safetensors"
        tensor = torch.full((64,), i, dtype=torch.uint8)
        save_file({f"layers.{i}.weight": tensor}, str(path))
        paths.append(str(path))
    return paths


def test_patched_loader_records_loaded_files(
    evicted, page_cache_patch, tmp_path, monkeypatch
):
    # The real DefaultModelLoader records the shards its _prepare_weights
    # returned, and the eviction evicts them.
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader import default_loader as dl

    paths = _save_shards(tmp_path, 2)
    prepared = []
    patched_prepare_weights = dl.DefaultModelLoader._prepare_weights

    def recording_prepare_weights(self, *args, **kwargs):
        res = patched_prepare_weights(self, *args, **kwargs)
        prepared.append(res)
        return res

    monkeypatch.setattr(
        dl.DefaultModelLoader, "_prepare_weights", recording_prepare_weights
    )
    # It reads the engine's vLLM config, which a CPU test cannot build; with
    # the EP weight filter off it sets nothing.
    monkeypatch.setattr(
        dl.DefaultModelLoader, "_init_ep_weight_filter", lambda self, config: None
    )
    loaded = []
    model = SimpleNamespace(
        load_weights=lambda weights: loaded.extend(name for name, _ in weights)
    )
    model_config = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        quantization=None,
        multimodal_config=None,
    )
    loader = dl.DefaultModelLoader(LoadConfig(load_format="safetensors"))
    loader.load_weights(model, model_config)

    assert sorted(loaded) == ["layers.0.weight", "layers.1.weight"]
    ((_, shards, _, _),) = prepared
    assert sorted(shards) == paths
    assert list(mlp._loaded_weight_files) == shards
    mlp.evict_loaded_weights_page_cache()
    assert evicted == [shards]


@pytest.mark.parametrize("rank, world_size", [(0, 1), (1, 2)])
def test_whole_file_prefetch_reads_this_ranks_files(
    tmp_path, monkeypatch, page_cache_patch, rank, world_size
):
    # The "prefetch" strategy without the EP filter: this rank's share of the
    # files, whole, as upstream's prefetch reads them.
    from vllm.model_executor.model_loader import weight_utils

    paths = _save_shards(tmp_path, 3)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: world_size > 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    read: dict[str, set[int]] = {}
    real_pread = os.pread

    def recording_pread(fd, size, off):
        path = os.readlink(f"/proc/self/fd/{fd}")
        read.setdefault(path, set()).update(range(off, off + size))
        return real_pread(fd, size, off)

    monkeypatch.setattr(os, "pread", recording_pread)
    weights = weight_utils.safetensors_weights_iterator(
        paths, False, "prefetch", safetensors_prefetch_block_size=64
    )
    assert len(list(weights)) == len(paths)
    ((thread, _),) = mlp._prefetches
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert read == {
        os.path.realpath(p): set(range(os.path.getsize(p)))
        for p in paths[rank::world_size]
    }


def test_prefetch_skips_a_file_it_cannot_read(tmp_path, monkeypatch):
    # As upstream's prefetch: a file that fails is skipped, the rest is read.
    paths = _save_shards(tmp_path, 2)
    missing = str(tmp_path / "missing.safetensors")
    read: set[str] = set()
    real_pread = os.pread

    def recording_pread(fd, size, off):
        read.add(os.readlink(f"/proc/self/fd/{fd}"))
        return real_pread(fd, size, off)

    monkeypatch.setattr(os, "pread", recording_pread)
    monkeypatch.setattr(mlp, "_prefetches", [])
    mlp._prefetch_all_checkpoints_stoppable([paths[0], missing, paths[1]])
    ((thread, _),) = mlp._prefetches
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert read == {os.path.realpath(p) for p in paths}


def test_whole_file_prefetch_stops_before_eviction(
    tmp_path, monkeypatch, page_cache_patch
):
    from vllm.model_executor.model_loader import weight_utils

    paths = _save_shards(tmp_path, 2)
    reads: list[int] = []
    first_read = threading.Event()
    real_pread = os.pread

    def pread_until_stopped(fd, size, off):
        # The first read holds until the stop, so the reads still queued must
        # be the ones the stop skips.
        reads.append(off)
        if len(reads) == 1:
            _, stop = mlp._prefetches[0]
            first_read.set()
            stop.wait(timeout=5.0)
        return real_pread(fd, size, off)

    monkeypatch.setattr(os, "pread", pread_until_stopped)
    # One reader thread and 64-byte blocks: several reads per file, in turn.
    list(
        weight_utils.safetensors_weights_iterator(
            paths,
            False,
            "prefetch",
            safetensors_prefetch_num_threads=1,
            safetensors_prefetch_block_size=64,
        )
    )
    ((thread, _),) = mlp._prefetches
    assert first_read.wait(timeout=5.0)
    mlp._stop_ep_range_prefetch()
    assert not thread.is_alive()
    assert not mlp._prefetches
    assert reads == [0]


def test_upstream_whole_file_prefetch_is_replaceable():
    """Fails when upstream renames ``_prefetch_all_checkpoints``, changes its
    parameters, or stops calling it through the ``weight_utils`` global."""
    import inspect

    from vllm.model_executor.model_loader import weight_utils

    upstream = (
        mlp._upstream_prefetch_all_checkpoints or weight_utils._prefetch_all_checkpoints
    )
    params = inspect.signature(upstream).parameters
    assert list(params) == ["sorted_files", "num_prefetch_threads", "block_size"]
    ours = inspect.signature(mlp._prefetch_all_checkpoints_stoppable).parameters
    assert [(p.name, p.default) for p in ours.values()] == [
        (p.name, p.default) for p in params.values()
    ]
    iterator = inspect.unwrap(weight_utils.safetensors_weights_iterator)
    assert "_prefetch_all_checkpoints" in iterator.__code__.co_names


def test_flag_off_keeps_upstream_prefetch(monkeypatch):
    from vllm.model_executor.model_loader import weight_utils

    upstream = weight_utils._prefetch_all_checkpoints
    monkeypatch.setattr(envs, "TPU_EVICT_WEIGHTS_PAGE_CACHE", False, raising=False)
    mlp.patch_default_model_loader_page_cache()
    assert weight_utils._prefetch_all_checkpoints is upstream


@pytest.mark.parametrize(
    "changed", [None, lambda files, threads=8: None], ids=["missing", "renamed"]
)
def test_changed_upstream_prefetch_is_left_in_place(monkeypatch, changed):
    # Fails open: the eviction still runs, upstream's thread keeps its reads.
    from vllm.model_executor.model_loader import weight_utils

    if changed is None:
        monkeypatch.delattr(weight_utils, "_prefetch_all_checkpoints")
    else:
        monkeypatch.setattr(weight_utils, "_prefetch_all_checkpoints", changed)
    monkeypatch.setattr(mlp, "_upstream_prefetch_all_checkpoints", None)
    logger = _RecordingLogger()
    monkeypatch.setattr(mlp, "logger", logger)
    mlp._install_stoppable_checkpoint_prefetch()
    assert getattr(weight_utils, "_prefetch_all_checkpoints", None) is changed
    assert mlp._upstream_prefetch_all_checkpoints is None
    ((level, msg),) = logger.records
    assert level == logging.WARNING
    assert "--safetensors-load-strategy=lazy" in msg
