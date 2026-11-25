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
    mlp._stop_prefetch()
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


@pytest.mark.parametrize("agreed", [True, False], ids=["every-rank-on", "a-rank-off"])
def test_eviction_waits_for_every_loader(evicted, monkeypatch, agreed):
    # Stop, then agree on the flag, which waits for every rank's loads, then
    # evict only if every rank has the flag on. A rank with no files still
    # joins the agreement, so every rank enters it.
    order: list[str] = []

    def agree():
        order.append("agree")
        return agreed

    monkeypatch.setattr(mlp, "_stop_prefetch", lambda: order.append("stop"))
    monkeypatch.setattr(mlp, "_evict_on_every_rank", agree)
    monkeypatch.setattr(
        mlp, "_evict_checkpoint_page_cache", lambda files: order.append("evict")
    )
    mlp.evict_loaded_weights_page_cache()
    assert order == ["stop", "agree"]
    order.clear()
    mlp._loaded_weight_files.update(dict.fromkeys(["a"]))
    mlp.evict_loaded_weights_page_cache()
    assert order == (["stop", "agree", "evict"] if agreed else ["stop", "agree"])


@pytest.mark.parametrize("flag", [True, False])
def test_agreement_without_distributed_state(monkeypatch, flag):
    # A single process has no world group: its own flag decides.
    monkeypatch.setattr(envs, "TPU_EVICT_WEIGHTS_PAGE_CACHE", flag, raising=False)
    assert mlp._evict_on_every_rank() is flag


@pytest.mark.parametrize(
    "cpu_timeout, expected_s",
    [(datetime.timedelta(seconds=12), 12), (None, 1800)],
    ids=["cpu-distributed-timeout", "torch-default"],
)
def test_load_fails_fast_when_a_rank_never_arrives(
    evicted, monkeypatch, cpu_timeout, expected_s
):
    # A rank that failed or stalled before the end of loading fails the load
    # with the CPU group's timeout (None: torch's 30-minute default) in the
    # message and the collective's error chained, not a hang.
    from vllm.distributed import parallel_state

    from vllm_torchtpu import utils

    def all_reduce(value, cpu_group):
        raise RuntimeError("Timed out waiting for recv operation to complete")

    group = SimpleNamespace(world_size=2, cpu_group=object(), cpu_timeout=cpu_timeout)
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: group)
    monkeypatch.setattr(utils, "min_over_group", all_reduce)
    mlp._loaded_weight_files.update(dict.fromkeys(["a"]))
    with pytest.raises(
        RuntimeError, match=f"finished loading within {expected_s} s"
    ) as info:
        mlp.evict_loaded_weights_page_cache()
    assert "Timed out" in str(info.value.__cause__)
    assert evicted == []


# The CPU group's timeout in the two-rank test below.
_GLOO_TIMEOUT_S = 3
# Per case: when each rank arrives at the agreement, in seconds after the case
# starts (None: never), each rank's TPU_EVICT_WEIGHTS_PAGE_CACHE, and the
# outcome expected on each rank ("on" and "off": the agreed flag).
_TWO_RANK_CASES = {
    "rank1-late": ((0, 1), (True, True), ("on", "on")),
    "flag-off-on-rank1": ((0, 0), (True, False), ("off", "off")),
    "flag-off-everywhere": ((0, 0), (False, False), ("off", "off")),
    "rank1-absent": ((0, None), (True, True), ("RuntimeError", "absent")),
    "rank0-absent": ((None, 0), (True, True), ("absent", "RuntimeError")),
    "rank1-absent-flag-off": ((0, None), (False, False), ("RuntimeError", "absent")),
}


def _loader_wait_rank(rank, init_file, out_file):
    """One rank of a two-rank gloo world; runs the cases in turn.

    Each case has its own gloo group, made as vLLM makes the world group's
    CPU group: with the CPU timeout. A rank absent from a case waits for the
    next case on the default group instead, so it stays up until the other
    rank's agreement has ended. Writes the outcomes, with the warnings each
    rank logged, to ``out_file``.
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
    cases = _TWO_RANK_CASES.items()
    for cpu_group, (case, (arrivals, flags, _)) in zip(cpu_groups, cases):
        group = SimpleNamespace(world_size=2, cpu_group=cpu_group, cpu_timeout=timeout)
        parallel_state.get_world_group = lambda group=group: group
        envs.TPU_EVICT_WEIGHTS_PAGE_CACHE = flags[rank]
        logger = _RecordingLogger()
        mlp.logger = logger
        torch.distributed.barrier()
        if arrivals[rank] is None:
            outcomes[case] = {"outcome": "absent"}
            continue
        time.sleep(arrivals[rank])
        start = time.monotonic()
        try:
            outcome = {"outcome": "on" if mlp._evict_on_every_rank() else "off"}
        except Exception as e:
            outcome = {
                "outcome": type(e).__name__,
                "message": str(e),
                "cause": repr(e.__cause__),
            }
        outcome["elapsed_s"] = time.monotonic() - start
        outcome["warnings"] = [
            msg for level, msg in logger.records if level == logging.WARNING
        ]
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
    """A real gloo all-reduce agrees on the flag and waits for every rank.

    A late rank passes; a rank with the flag off turns eviction off on every
    rank, and the rank that had it on warns; a rank that never arrives fails
    every other rank within the CPU group's timeout, with the flag on or off.
    """
    outcomes = two_rank_outcomes[case]
    _, flags, expected = _TWO_RANK_CASES[case]
    assert [o["outcome"] for o in outcomes] == list(expected), outcomes
    for flag, outcome in zip(flags, outcomes):
        if outcome["outcome"] == "absent":
            continue
        if outcome["outcome"] == "RuntimeError":
            assert f"loading within {_GLOO_TIMEOUT_S} s" in outcome["message"]
            assert outcome["elapsed_s"] < _GLOO_TIMEOUT_S + 5, outcomes
        else:
            assert outcome["elapsed_s"] < _GLOO_TIMEOUT_S, outcomes
        warned = flag and outcome["outcome"] == "off"
        assert len(outcome["warnings"]) == warned, outcomes


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
        parallel_state, "_WORLD", SimpleNamespace(local_rank=local_rank)
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
    mlp._stop_prefetch()
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
    mlp._stop_prefetch()


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
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
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
    loaded = []
    model = SimpleNamespace(
        load_weights=lambda weights: loaded.extend(name for name, _ in weights)
    )
    model_config = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        quantization=None,
        multimodal_config=None,
        is_moe=False,
    )
    loader = dl.DefaultModelLoader(LoadConfig(load_format="safetensors"))
    # A CPU device: the CPU unit-test agents have no TPU device to resolve.
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
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
    mlp._stop_prefetch()
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
