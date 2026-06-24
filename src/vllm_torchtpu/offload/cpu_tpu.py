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
"""TPU ↔ CPU KV cache offloading handler.

This module plugs into vllm's `OffloadingConnector` framework to give TPU
inference a large CPU-resident host pool for KV-cache blocks evicted from
HBM. Cache hits on the host pool are streamed back into HBM via host-side
gather + DMA copy, then scattered into the live `kv_cache` via torch
in-place indexing.

Architecture
------------
- `TPUCPUOffloadingSpec` subclasses `vllm.v1.kv_offload.cpu.spec.
  CPUOffloadingSpec`; it overrides `get_handlers()` and exposes
  `estimate_hbm_reserve_bytes(vllm_config) -> int` so the worker can
  reserve HBM for the H2D staging buffer before profile_run.
- `CpuTpuOffloadingHandlers` holds one D2H and one H2D
  `SingleDirectionOffloadingHandler` instance; the spec yields them per
  the OffloadingConnector contract.
- Each handler owns a single background DMA worker thread plus (H2D only)
  a depth-1 device-side staging buffer; transfers serialize through them.
- Oversized H2D loads (more than `KV_H2D_POOL_MAX_BLOCKS` blocks) are
  split into max-buffer-sized chunks; each chunk consumes one engine
  step's worth of latency, the depth-1 invariant is preserved.
- Scatter dispatch is **scheduler-gated**: per-rank `get_finished` only
  defers the scatter (adds the chunk to `_pending_scatters` and reports
  `finished_recving` based on `dma_done`). The actual scatter HLO is
  enqueued from a monkey-patched `OffloadingConnectorWorker.
  start_kv_transfers`, filtered by `scatter_now_req_ids` set on the
  metadata by a monkey-patched `OffloadingConnectorScheduler.
  build_connector_meta` from `scheduler_output.num_scheduled_tokens`.
  A request only enters that set after `KVOutputAggregator` released
  `finished_recving` from every rank, so the scatter HLOs dispatch in
  lockstep across ranks at the same broadcast-sync entry point.

Usage
-----
Pass `--kv-transfer-config` to `vllm serve`:

    vllm serve <model> \\
        --tensor-parallel-size=8 \\
        --kv-offloading-size 400 \\
        --kv-transfer-config '{
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "TPUCPUOffloadingSpec",
                "spec_module_path": "vllm_torchtpu.offload.cpu_tpu"
            }
        }'

`--kv-offloading-size N` reserves N GiB of host RAM for the pool. Sizing
guidance: pool large enough to fit the working set of repeated prefixes,
but small enough that GPU_MEM_UTIL still leaves usable HBM after the H2D
staging buffer reserve (see `estimate_hbm_reserve_bytes`).

A working launcher script lives at
`examples/kv-offload/run_kv_offload_single_host.sh`.

Environment knobs (all optional)
--------------------------------
- `KV_H2D_POOL_MAX_BLOCKS` (default 2048)
    Caps the H2D staging buffer's `(max_padded, …)` first dim. Rounded
    to the next power of 2. Larger = fewer chunked loads on long-prefix
    workloads but more HBM reserved up front.
- `TPU_PREMAPPED_BUFFER_SIZE` (libtpu env, **per device**)
    Sized so libtpu's premap pool can absorb one H2D transfer worth of
    pinned-host staging. 16 GiB per chip works for the standard 480B
    overflow workload (see recipe). Bigger is fine on host RAM budget;
    too small forces lazy `copy_` to fall back to slower per-call pinning.
"""
from __future__ import annotations

import os
import queue as _queue
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch_tpu._internal.sync import synchronize as _tpu_sync

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (BlockIDsLoadStoreSpec, CanonicalKVCaches,
                                     GPULoadStoreSpec, LoadStoreSpec)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.worker.worker import (OffloadingHandler,
                                              TransferResult, TransferSpec)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level patches: route scatter dispatch through a scheduler-driven
# "ready set" so all ranks scatter the same transfers at the same step
# boundary.
#
# Design
# ------
# Two hooks, both idempotent:
#
# 1. SCHEDULER side — `OffloadingConnectorScheduler.build_connector_meta`
#    is wrapped to annotate req_ids in THIS step's active batch.
#
# 2. WORKER side — `OffloadingConnectorWorker.start_kv_transfers` is
#    reimplemented to:
#      a. Pass `scatter_now_req_ids` into each handler's
#         `flush_pending_scatters`, which filters the deferred scatter
#         list and only dispatches HLOs for transfers whose `req_id` is
#         in the set.
#      b. Stash `req_id` on the handler keyed by `job_id` *before*
#         calling `worker.transfer_async`, so the handler can tag the
#         resulting `Transfer` for later lookup.
#    All ranks reach this from the same `execute_model` broadcast
#    (~µs jitter), so the scatter HLOs go onto libtpu in lockstep.
#
# ---------------------------------------------------------------------------
def _install_scheduler_scatter_now_hook() -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import \
        OffloadingConnectorScheduler  # noqa: E501

    orig = OffloadingConnectorScheduler.build_connector_meta
    if getattr(orig, "_tpu_scatter_now_patched", False):
        return

    def build_connector_meta_with_scatter_now(
            self, scheduler_output):  # type: ignore[no-untyped-def]
        meta = orig(self, scheduler_output)
        # Active-batch req_ids: a req enters the batch only after the
        # aggregator released finished_recving from ALL ranks (or it
        # didn't need a KV load at all). Either way, a req_id in this
        # set is safe to scatter on every rank in lockstep.
        meta.scatter_now_req_ids = set(
            scheduler_output.num_scheduled_tokens.keys())
        return meta

    build_connector_meta_with_scatter_now._tpu_scatter_now_patched = True  # type: ignore[attr-defined]
    OffloadingConnectorScheduler.build_connector_meta = (
        build_connector_meta_with_scatter_now)


def _install_flush_pending_scatters_hook() -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import \
        OffloadingConnectorWorker

    orig = OffloadingConnectorWorker.start_kv_transfers
    if getattr(orig, "_tpu_flush_patched", False):
        return

    def start_kv_transfers_with_flush(
            self, metadata):  # type: ignore[no-untyped-def]
        # Step 1: drain deferred scatter, filtered by scheduler signal.
        # All ranks reach this point from the same `execute_model`
        # broadcast, so the scatter HLOs land on libtpu in lockstep.
        # `scatter_now_req_ids` is unconditionally annotated by the
        # scheduler-side hook above — AttributeError here means the
        # scheduler hook failed to install (loud failure, by design).
        scatter_now = metadata.scatter_now_req_ids
        for handler in self.worker.handlers:
            if hasattr(handler, "flush_pending_scatters"):
                handler.flush_pending_scatters(scatter_now)

        # Step 2: dispatch pending stores (deferred from prior step's
        # `prepare_store_kv`). Stores don't carry the req_id stash —
        # they're not subject to the scatter-now gate.
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        # Step 3: dispatch new loads. Stash req_id on the appropriate
        # H2D handler BEFORE calling `worker.transfer_async` so the
        # handler can copy it onto the resulting Transfer.
        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            # Find the handler that will receive this transfer.
            src, dst = entry.transfer_spec
            ttype = (src.medium(), dst.medium())
            handler = self.worker.transfer_type_to_handler.get(ttype)
            if handler is not None and hasattr(handler,
                                               "_stash_req_id_for_job"):
                handler._stash_req_id_for_job(job_id, entry.req_id)
            success = self.worker.transfer_async(job_id, entry.transfer_spec)
            assert success

    start_kv_transfers_with_flush._tpu_flush_patched = True  # type: ignore[attr-defined]
    OffloadingConnectorWorker.start_kv_transfers = (
        start_kv_transfers_with_flush)


# These hooks reshape the OffloadingConnector's store/load dispatch into the
# torch handler's deferred-scatter + lockstep flow (flush_pending_scatters,
# _unsubmitted_store_jobs). The raiden handler does direct D2h/H2d + poll-based
# get_finished and uses the STOCK connector dispatch/completion path, so
# installing these would replace start_kv_transfers with a flow it doesn't
# participate in -- leaving the scheduler's store-completion/block-free gate
# unsatisfied and the engine busy-spinning. Only install for the torch path.
if os.environ.get("VLLM_TPU_OFFLOAD_RAIDEN", "0") != "1":
    _install_scheduler_scatter_now_hook()
    _install_flush_pending_scatters_hook()
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_to_power_of_2(ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (padded_ids, original_len) padded to the next power of two.

    Repeated final element fills the padding so scattered writes are
    idempotent.  The caller must use `original_len` to trim results when
    needed.
    """
    n = len(ids)
    if n == 0:
        return ids, 0
    p = 1
    while p < n:
        p <<= 1
    if p == n:
        return ids, n
    padded = np.empty(p, dtype=ids.dtype)
    padded[:n] = ids
    padded[n:] = ids[n - 1]
    return padded, n


def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
):
    """
    Convert a list of block IDs to a list of matching block ids,
    assuming each block is composed of actual block_size_factor blocks.
    Outputs to output tensor.
    The first skip_count blocks will be skipped.
    Note that skip_count must be less than block_size_factor.

    For example, if block_ids = [0, 1, 3] and block_size_factor =  4,
    then it yields [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    since 0 maps to [0, 1, 2, 3]
    1 maps to [4, 5, 6, 7]
    and 3 maps to [12, 13, 14, 15]
    """
    assert skip_count < block_size_factor

    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(0, block_size_factor)

    output_idx = 0
    for i, block_id in enumerate(block_ids):
        base_block_id = block_id * block_size_factor
        indices = first_range if i == 0 else full_range
        output_end_idx = output_idx + len(indices)
        output[output_idx:output_end_idx] = base_block_id + indices
        output_idx = output_end_idx


# ---------------------------------------------------------------------------
# Background DMA worker
# ---------------------------------------------------------------------------


class _DmaWorker:
    """Single background thread draining a FIFO work queue.

    One instance per direction (D2H / H2D). Each `submit(fn, *args)` enqueues
    work and returns a `threading.Event` that is set when `fn` returns. The
    worker blocks on `queue.get()` when idle, and the FIFO queue guarantees
    serial execution — combined with `event.synchronize()` inside the task,
    this guarantees only one DMA per direction is in flight at any moment.
    """

    def __init__(self, name: str) -> None:
        self._queue: _queue.SimpleQueue = _queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run,
                                        name=name,
                                        daemon=True)
        self._thread.start()

    def submit(self, fn, *args) -> threading.Event:
        done = threading.Event()
        self._queue.put((fn, args, done))
        return done

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            fn, args, done = item
            try:
                fn(*args)
            except Exception:
                logger.exception("[kv-offload] DMA worker task failed")
            finally:
                done.set()

    def shutdown(self) -> None:
        self._queue.put(None)


# ---------------------------------------------------------------------------
# Transfer record
# ---------------------------------------------------------------------------


@dataclass
class Transfer:
    job_id: int
    num_bytes: int
    n: int
    # threading.Event set by _DmaWorker when the DMA task returns. Because
    # the task ends with `_tpu_sync(wait=True)` (D2H) or a blocking
    # `d.copy_(h)` (H2D), dma_done.is_set() implies the DMA is physically
    # complete on-device. For H2D, dma_done starts as None until
    # `_start_h2d_dma` kicks the worker — deferred submission keeps only one
    # H2D DMA in flight at any time, so the single _h2d_device_buffer is
    # never overwritten while the previous transfer's scatter is still
    # reading it.
    dma_done: threading.Event | None
    # Pinned CPU staging used as the DMA endpoint in BOTH directions.
    # D2H: filled by the worker's blocking copy_; main-thread scatter into
    #      cpu_pool happens after dma_done.is_set() flips True.
    # H2D: filled by main-thread index_select(cpu_pool, padded_src_ids);
    #      consumed by the worker's H2D copy_ into device_buffer.
    host_buffer: list | None = None
    # H2D: device staging (slice of the single _h2d_device_buffer) that
    # holds the freshly DMA'd blocks until the Pallas scatter copies them
    # into kv_cache. D2H: holds the Pallas gather output until the worker's
    # D2H copy_ drains it; kept alive so XLA's gather IR isn't dropped early.
    # For H2D, set lazily by `_start_h2d_dma` (None at register time).
    device_buffer: list | None = None
    # D2H: numpy int64 cpu_pool scatter indices, length n.
    # H2D: numpy int64 KV-cache scatter indices (full, unpadded). Sliced
    # per-chunk by `_start_h2d_dma`.
    dst_ids: np.ndarray | None = None
    # H2D: numpy int64 host pool gather indices (full, unpadded). Sliced
    # per-chunk by `_start_h2d_dma`.
    src_ids: np.ndarray | None = None
    # H2D: total number of chunks the transfer is split into. = 1 for
    # transfers that fit the single device buffer; > 1 for oversized loads
    # that are streamed through the buffer one chunk per get_finished()
    # round-trip. Set at transfer_async time, never mutated thereafter.
    chunks_total: int = 0
    # H2D: number of chunks already dispatched-and-scattered. Incremented
    # in get_finished() after each chunk's Pallas scatter syncs.
    chunks_done: int = 0
    # H2D: current chunk's padded dst_ids on device, used by get_finished()
    # for the Pallas scatter. Rebuilt for each chunk in `_start_h2d_dma`.
    dst_ids_i32: object | None = None
    # Set by the DMA worker task if it raises. get_finished() reads this to
    # report success=False.
    error: BaseException | None = None
    # Request ID owning this transfer. Stashed at transfer_async time from
    # the connector_worker's _jobs mapping. Used by flush_pending_scatters
    # to filter against scheduler-side `scatter_now_req_ids` so scatter
    # dispatch happens in lockstep across ranks (gated by the aggregator,
    # never by per-rank coordination).
    req_id: str | None = None


@dataclass
class _PendingScatter:
    """A chunk whose H2D completed; scatter HLO awaits the next
    broadcast-synchronized `start_kv_transfers` for dispatch.

    Snapshotting device_buffer + dst_ids_i32 here lets the underlying
    Transfer be re-submitted for its next chunk (which overwrites
    Transfer.device_buffer) before this snapshot is consumed.
    """
    transfer: Transfer
    device_buffer: list
    dst_ids_i32: object  # torch.Tensor (lazy import)
    is_last_chunk: bool


# ---------------------------------------------------------------------------
# Single-direction handler
# ---------------------------------------------------------------------------


class SingleDirectionOffloadingHandler(OffloadingHandler):
    """Handles KV-cache transfers for one direction (D2H or H2D) on TPU."""

    def __init__(
        self,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        src_block_size_factor: int,
        dst_block_size_factor: int,
    ):
        assert len(src_tensors) == len(dst_tensors)

        self.src_tensors = src_tensors
        self.dst_tensors = dst_tensors

        min_factor = min(src_block_size_factor, dst_block_size_factor)
        assert src_block_size_factor % min_factor == 0
        assert dst_block_size_factor % min_factor == 0
        self.src_block_size_factor = src_block_size_factor // min_factor
        self.dst_block_size_factor = dst_block_size_factor // min_factor

        self.block_size_in_bytes = [
            t.element_size() * t.stride(0) * min_factor for t in src_tensors
        ]
        self.total_block_size_in_bytes = sum(self.block_size_in_bytes)

        self.tpu_to_cpu: bool = src_tensors[0].device.type != "cpu"
        self.transfer_type = ("GPU", "CPU") if self.tpu_to_cpu else ("CPU",
                                                                     "GPU")
        self._device: torch.device = (src_tensors[0].device if self.tpu_to_cpu
                                      else dst_tensors[0].device)

        self._transfer_map: dict[int, Transfer] = {}
        self._transfers: deque[Transfer] = deque()
        # Bridge from connector_worker (which knows req_id at start_kv
        # _transfers time) to handler.transfer_async (which builds the
        # Transfer). Used by H2D loads only — D2H stores don't get tagged
        # because they're not subject to the scatter-now gate — but
        # initialized here so the unconditional `Transfer.req_id` lookup
        # in `transfer_async` works on both directions' instances.
        self._req_id_by_job_id: dict[int, str] = {}

        # Lightweight diagnostics — printed sparingly to spot stalls.

        # Single background DMA worker for this direction. The FIFO queue
        # plus the worker task's terminating `_tpu_sync(..., wait=True)`
        # together guarantee that only one DMA per direction is in flight
        # at any instant. Thread isolation replaces what a dedicated TPU
        # stream + event.synchronize() would do on CUDA — no explicit
        # stream context is needed.
        worker_name = ("kv-offload-d2h-worker"
                       if self.tpu_to_cpu else "kv-offload-h2d-worker")
        self._dma_worker = _DmaWorker(name=worker_name)

        # H2D device-staging buffer. Single pre-allocated buffer at the max
        # padded shape; each H2D transfer takes a view [:n_padded].
        self._h2d_device_buffer: list[torch.Tensor] | None = None
        self._h2d_max_padded: int = 0
        if not self.tpu_to_cpu:
            ref = self.dst_tensors
            max_pool_blocks = int(
                os.environ.get("KV_H2D_POOL_MAX_BLOCKS", "2048"))
            max_n = min(ref[0].shape[0], max_pool_blocks)
            max_padded = 1
            while max_padded < max_n:
                max_padded <<= 1
            self._h2d_max_padded = max_padded
            self._h2d_device_buffer = [
                torch.zeros(
                    (max_padded, ) + t.shape[1:],
                    dtype=t.dtype,
                    device=self._device,
                ) for t in ref
            ]
            # Force materialization so subsequent copy_ calls don't go
            # through a lazy-allocation path that could stall PJRT.
            _tpu_sync(self._h2d_device_buffer, wait=True)
            # Buffer-reuse gate. SET = device buffer is logically free for
            # the worker to write the next H2D into. CLEARED = a transfer
            # is mid-flight (worker is writing) OR pending scatter (main
            # thread hasn't yet enqueued the TC HLO that reads it).
            #
            # Main thread `set()`s after enqueuing scatter on libtpu's
            # command stream (NOT after the scatter completes on device —
            # we rely on XLA's HBM-range dependency tracking to serialize
            # the next H2D DMA after the scatter on the device). This
            # keeps main thread non-blocking; the worker is the only one
            # that ever waits.
            self._h2d_buffer_free = threading.Event()
            self._h2d_buffer_free.set()
            # H2D: chunks whose DMA completed; scatter HLO deferred to the
            # next broadcast-synchronized `start_kv_transfers`.
            self._pending_scatters: list[_PendingScatter] = []
            logger.info(
                "[kv-offload] H2D device buffer: %d blocks (max_padded)",
                max_padded,
            )

    # -- H2D deferred-submission management ----------------------------------

    def _stash_req_id_for_job(self, job_id: int, req_id: str) -> None:
        """Called from monkey-patched `OffloadingConnectorWorker.
        start_kv_transfers` immediately before `worker.transfer_async`,
        so the downstream `self.transfer_async` can tag the Transfer
        with its owning req_id without a framework signature change.
        """
        self._req_id_by_job_id[job_id] = req_id

    def _start_h2d_dma(self, t: Transfer) -> None:
        """Submit the worker closure that will host-gather, gate on the
        device buffer being free, and DMA the chunk into device_buffer.

        Submission is eager (called from transfer_async and from
        get_finished for the next chunk of an oversized transfer). The
        worker queue serializes execution; the `_h2d_buffer_free` Event
        gates the actual DMA enqueue so libtpu sees:
            scatter(prev) → H2D(this)
        in that command-stream order. XLA's HBM-range dependency on
        `device_buffer` then serializes them on-device — no host-side
        `_tpu_sync` is needed and the main thread never blocks.

        For oversized transfers (chunks_total > 1), dispatches just the
        chunk indexed by `t.chunks_done`. get_finished() re-submits so
        the worker picks up the next chunk.
        """
        chunk_size = self._h2d_max_padded
        start = t.chunks_done * chunk_size
        end = min(start + chunk_size, len(t.src_ids))
        src_chunk = t.src_ids[start:end]
        dst_chunk = t.dst_ids[start:end]

        # Per-chunk padding: non-last chunks already match a power-of-2
        # (chunk_size is rounded to one); only the last chunk of an
        # oversized transfer may need padding.
        src_chunk_padded, _ = _pad_to_power_of_2(src_chunk)
        dst_chunk_padded, _ = _pad_to_power_of_2(dst_chunk)
        n_chunk_padded = len(src_chunk_padded)

        t.dst_ids_i32 = (torch.from_numpy(dst_chunk_padded.astype(
            np.int32)).to(torch.int32).to(self._device))

        device_buffer = [
            full[:n_chunk_padded] for full in self._h2d_device_buffer
        ]
        src_tensors = self.src_tensors  # cpu_pool, captured for the closure
        buffer_free = self._h2d_buffer_free  # captured for the closure

        def _h2d_task(devs: list, src_tensors: list, src_ids_padded):
            try:
                src_ids_torch = torch.from_numpy(src_ids_padded).to(
                    torch.int64)
                n_padded = len(src_ids_padded)
                # Host-side gather first. Safe at any time — it touches only
                # cpu_pool (read-only here) and a freshly allocated host
                # buffer; no shared TPU state.
                hosts = []
                for cpu_pool in src_tensors:
                    h = torch.empty(
                        (n_padded, ) + cpu_pool.shape[1:],
                        dtype=cpu_pool.dtype,
                    )
                    torch.index_select(cpu_pool, 0, src_ids_torch, out=h)
                    hosts.append(h)
                # Gate on the device buffer being free. Blocks the WORKER
                # thread, never the main thread. Main `set()`s after
                # enqueuing the scatter HLO for the prior transfer, which
                # guarantees: prev scatter is on libtpu's command stream
                # before this H2D's copy_ enqueue lands on it.
                buffer_free.wait()
                buffer_free.clear()
                for d, h in zip(devs, hosts):
                    d.copy_(h)
            except BaseException as e:
                t.error = e
                raise

        t.device_buffer = device_buffer
        t.dma_done = self._dma_worker.submit(_h2d_task, device_buffer,
                                             src_tensors, src_chunk_padded)

    @property
    def prewarm_shapes(self) -> list[int]:
        return []

    def prewarm_shape(self, p: int) -> None:  # noqa: ARG002
        return None

    # -- OffloadingHandler interface -----------------------------------------

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        src_sub_count = src_blocks.size * self.src_block_size_factor
        dst_sub_count = dst_blocks.size * self.dst_block_size_factor
        src_skip = -dst_blocks.size % self.src_block_size_factor
        assert dst_sub_count == src_sub_count - src_skip

        src_expanded = np.empty(src_sub_count, dtype=np.int64)
        dst_expanded = np.empty(dst_sub_count, dtype=np.int64)
        expand_block_ids(
            src_blocks,
            self.src_block_size_factor,
            src_expanded,
            skip_count=src_skip,
        )
        expand_block_ids(dst_blocks, self.dst_block_size_factor, dst_expanded)
        src_ids = src_expanded[src_skip:]
        dst_ids = dst_expanded
        n = len(dst_ids)

        if self.tpu_to_cpu:
            # ── D2H: Pallas gather (main) → lazy copy_ + sync (worker thread)
            #         → main-thread scatter into cpu_pool in get_finished()
            #         once dma_done.is_set(). ──
            padded_src_ids, _ = _pad_to_power_of_2(src_ids)
            src_ids_i32 = (torch.from_numpy(padded_src_ids.astype(
                np.int32)).to(torch.int32).to(self._device))

            # (1) Compute stream (main thread): Pallas gather + flush IR.
            # Stays on the main thread because moving it would force XLA
            # to coordinate kv_l (live KV cache) reads across threads.
            src_ids_i64 = src_ids_i32.to(torch.int64)
            device_buffer = [kv_l[src_ids_i64] for kv_l in self.src_tensors]
            _tpu_sync(device_buffer, wait=False)

            # (2) Unpinned host staging matching the gathered device shape.
            # The lazy D2H copy_ below stages through libtpu's premap pool
            # (TPU_PREMAPPED_BUFFER_SIZE), so the dst doesn't need to be
            # pinned ourselves — saves the per-call pin-allocation cost.
            host_buffer = [
                torch.empty(db.shape, dtype=db.dtype) for db in device_buffer
            ]

            # (3) Submit the actual DMA work to the per-direction worker.
            # The task does device -> host copy AND the cpu_pool host
            # scatter. dma_done is signaled only after both are complete.
            cpu_pool_tensors = self.dst_tensors
            dst_ids_for_scatter = dst_ids[:n]
            n_for_scatter = n

            # Build the Transfer first so the worker task closure can stash
            # any exception on it; success=False is then reported from
            # get_finished() instead of silently masking a failed DMA.
            t = Transfer(
                job_id=job_id,
                num_bytes=n * self.total_block_size_in_bytes,
                n=n,
                dma_done=None,
                host_buffer=host_buffer,
                device_buffer=device_buffer,
                dst_ids=dst_ids[:n],
            )

            def _d2h_task(devs: list, hosts: list):
                try:
                    for d, h in zip(devs, hosts):
                        h.copy_(d)
                    for h, dst in zip(hosts, cpu_pool_tensors):
                        dst[dst_ids_for_scatter] = h[:n_for_scatter]
                except BaseException as e:
                    t.error = e
                    raise

            t.dma_done = self._dma_worker.submit(_d2h_task, device_buffer,
                                                 host_buffer)

        else:
            # H2D: host index_select (main) → lazy copy_ + sync (worker)
            #      → Pallas scatter (main) in get_finished().
            # Split into max_padded-sized chunks so the depth=1 device
            # buffer is never overflowed. Oversized transfers stream one
            # chunk per get_finished() round-trip; chunk splitting +
            # padding happens lazily in `_start_h2d_dma`. Each chunk's
            # Pallas scatter completes before the next chunk's DMA starts,
            # preserving the depth=1 buffer-reuse invariant.
            chunk_size = self._h2d_max_padded
            chunks_total = max(1,
                               (len(src_ids) + chunk_size - 1) // chunk_size)
            if chunks_total > 1:
                logger.info_once(
                    "[kv-offload] H2D transfer of %d blocks exceeds device "
                    "buffer (%d); will be loaded as %d chunks across that "
                    "many engine steps.", len(src_ids), chunk_size,
                    chunks_total)

            t = Transfer(
                job_id=job_id,
                num_bytes=n * self.total_block_size_in_bytes,
                n=n,
                dma_done=None,
                host_buffer=None,  # populated by _h2d_task
                device_buffer=None,
                src_ids=src_ids,
                dst_ids=dst_ids,
                chunks_total=chunks_total,
                chunks_done=0,
            )

        # Tag the Transfer with its owning req_id so flush_pending_scatters
        # can later filter against scheduler-provided `scatter_now_req_ids`.
        # The req_id was stashed by the monkey-patched start_kv_transfers
        # immediately before this call.
        t.req_id = self._req_id_by_job_id.pop(job_id, None)
        self._transfer_map[job_id] = t
        self._transfers.append(t)
        if not self.tpu_to_cpu:
            # Eager submit. Worker queue serializes; the buffer_free Event
            # gates the actual DMA enqueue against the prior scatter.
            self._start_h2d_dma(t)
        return True

    def flush_pending_scatters(self, scatter_now_req_ids: set[str]) -> None:
        """Dispatch H2D scatter HLOs for deferred chunks whose owning
        request has been promoted to the active batch on ALL ranks.

        `scatter_now_req_ids` is the set of req_ids in the current step's
        scheduled batch, populated by the scheduler-side hook from
        `scheduler_output.num_scheduled_tokens`. A request only enters
        that set after `KVOutputAggregator` released its `finished_recving`
        from every rank — set membership is the "all ranks done" signal
        that lets each rank dispatch its scatter in lockstep with the
        others. Transfers whose req_id isn't in the set stay in
        `_pending_scatters` for a later step's flush.

        For chunked transfers, re-submits the next chunk to the worker
        after this chunk's scatter is enqueued — preserving the libtpu
        command-stream order `scatter(prev) → H2D(next)` that XLA needs
        to serialize same-buffer reads/writes correctly.
        """
        # Check tpu_to_cpu FIRST — `_pending_scatters` is only initialized
        # on the H2D handler (see __init__'s `if not self.tpu_to_cpu` block).
        # The D2H handler shares this class but has no deferred-scatter
        # concept; bail before touching the missing attribute.
        if self.tpu_to_cpu or not self._pending_scatters:
            return

        # Intermediate (non-last) chunks of a chunked transfer flush
        # eagerly — they write to KV-cache slots already reserved for the
        # request and the request hasn't been promoted yet (its
        # `finished_recving` only fires on the LAST chunk). Gating
        # intermediate chunks on `scatter_now_req_ids` would deadlock:
        # the next chunk can't dispatch until the prior chunk's scatter is
        # enqueued (buffer-reuse invariant), but the req_id won't enter
        # `scatter_now_req_ids` until `finished_recving` is reported,
        # which only happens on the last chunk's `get_finished`.
        to_flush = [
            ps for ps in self._pending_scatters if
            (not ps.is_last_chunk or ps.transfer.req_id in scatter_now_req_ids)
        ]
        to_keep = [
            ps for ps in self._pending_scatters
            if (ps.is_last_chunk
                and ps.transfer.req_id not in scatter_now_req_ids)
        ]

        if not to_flush:
            # Nothing to dispatch on this rank this step. Return early
            # so we don't accidentally freed _h2d_buffer_free below.
            return

        try:
            for ps in to_flush:
                if ps.transfer.error is not None:
                    continue
                dst_ids_i64 = ps.dst_ids_i32.to(torch.int64)
                for kv_cache_l, dev_buf in zip(self.dst_tensors,
                                               ps.device_buffer):
                    kv_cache_l[dst_ids_i64] = dev_buf
        finally:
            # Scatter HLOs are now enqueued on libtpu's command stream;
            # the worker may proceed with the next H2D (XLA serializes
            # via device_buffer's HBM-range event chain).
            self._h2d_buffer_free.set()
        # Re-submit the next chunk of any flushed transfer whose chunk
        # was not its last. The newly-submitted DMA will block on the
        # worker thread's `buffer_free.wait()` if the buffer is still
        # held by another deferred chunk (depth=1 invariant).
        for ps in to_flush:
            if ps.transfer.error is None and not ps.is_last_chunk:
                self._start_h2d_dma(ps.transfer)
        self._pending_scatters = to_keep

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers:
            t = self._transfers[0]

            # dma_done is set by _DmaWorker when the task returns.
            #  - D2H: the task's blocking h.copy_(d) waits for the DMA to
            #    drain into host, then runs the cpu_pool host scatter, so
            #    dma_done.is_set() ⇒ both copy AND scatter are done.
            #  - H2D: the task's d.copy_(h) returns once the op is enqueued
            #    on libtpu's command stream (the DMA itself may still be
            #    in flight on the DMA engine).
            #
            # `dma_done is None` is the chunked re-entry marker: we set it
            # to None after queueing a chunk for deferred scatter, and the
            # next chunk's re-submission (from flush_pending_scatters) will
            # set it to a fresh Event.
            if t.dma_done is None or not t.dma_done.is_set():
                break

            if self.tpu_to_cpu:
                # D2H: the worker task ran both the device→host copy AND
                # the cpu_pool host scatter. The main thread only has to
                # drop buffer refs.
                self._transfers.popleft()
                t.host_buffer = None
                t.device_buffer = None
            else:
                # H2D: defer the scatter HLO dispatch to
                # `flush_pending_scatters`, which runs at the next
                # broadcast-synchronized `start_kv_transfers` on all ranks
                # at once. Reporting `finished_recving` here (without
                # scattering yet) is safe because the scheduler's
                # KVOutputAggregator waits for ALL ranks to report before
                # promoting the request; the request's forward at step k+1
                # is preceded by start_kv_transfers's flush on the same
                # step, so XLA sees `scatter → forward` on the kv_cache
                # buffer and serializes correctly.
                t.chunks_done += 1
                is_last = (t.error is not None
                           or t.chunks_done >= t.chunks_total)
                if t.error is None:
                    self._pending_scatters.append(
                        _PendingScatter(
                            transfer=t,
                            device_buffer=t.device_buffer,
                            dst_ids_i32=t.dst_ids_i32,
                            is_last_chunk=is_last,
                        ))
                else:
                    # Error path: no scatter needed; release the buffer so
                    # the worker isn't stuck on `buffer_free.wait()`.
                    self._h2d_buffer_free.set()
                # Detach so the next chunk's `_start_h2d_dma` (called from
                # flush_pending_scatters) can replace these fields without
                # clobbering this chunk's snapshot in `_pending_scatters`.
                t.device_buffer = None
                t.host_buffer = None
                t.dma_done = None  # re-entry marker for chunked transfers
                if not is_last:
                    # Non-last chunk: keep Transfer at head; do NOT report
                    # finished_recving yet. flush_pending_scatters will
                    # re-submit the next chunk after enqueuing this chunk's
                    # scatter.
                    break
                self._transfers.popleft()

            results.append(
                TransferResult(
                    job_id=t.job_id,
                    success=(t.error is None),
                    transfer_size=t.num_bytes,
                    # TODO: elapsed_time() not yet implemented on TpuEvent.
                    transfer_time=1e-9,
                    transfer_type=self.transfer_type,
                ))
            del self._transfer_map[t.job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        """Block until the named transfers have completed.

        Mirrors the CUDA reference impl in
        vllm/v1/kv_offload/cpu/gpu_worker.py:wait — purely a flush
        primitive. Internal-state cleanup (popping from `_transfers`,
        deleting from `_transfer_map`, building TransferResults) is left
        to the next `get_finished()` call. Removing transfers here would
        prevent `get_finished()` from emitting their TransferResults,
        which the scheduler relies on to mark jobs complete.
        """
        for jid in job_ids:
            t = self._transfer_map.get(jid)
            if t is None:
                continue
            if self.tpu_to_cpu:
                # Worker task ends with the cpu_pool host scatter, so
                # dma_done.is_set() implies the D2H is fully landed.
                t.dma_done.wait()
            else:
                # H2D skips non-ready transfers rather than waiting; the
                # OffloadingConnector only ever calls wait() for D2H flush.
                raise AssertionError(
                    f"wait() called on H2D handler for job={jid}; "
                    "OffloadingConnector should never invoke this path.")

    def shutdown(self) -> None:
        """Tear down the worker thread cleanly.

        For H2D, the worker may be blocked inside `_h2d_task` on
        `buffer_free.wait()` — `_dma_worker.shutdown()` alone would
        deadlock because the worker can't pull the shutdown sentinel
        until the current task returns. Set the buffer-free Event first
        to unblock the wait; the task then runs its (possibly futile)
        copy_ to completion and the worker thread exits on the sentinel.
        """
        if not self.tpu_to_cpu:
            self._h2d_buffer_free.set()
        self._dma_worker.shutdown()


# ---------------------------------------------------------------------------
# Raiden-backed offloading handler
# ---------------------------------------------------------------------------

# When set, KV offload uses raiden's KVCacheManager.D2h/H2d — direct PJRT
# raw-buffer DMA between the device KV cache and raiden's DmaMapped host pool —
# instead of the torch Pallas-gather + copy_ + scatter path. No device staging
# buffer, no cpu_pool tensor, no scatter-now gate (the DMA writes kv_cache
# directly; there is no scatter HLO to dispatch in lockstep).
_USE_RAIDEN_OFFLOAD = os.environ.get("VLLM_TPU_OFFLOAD_RAIDEN", "0") == "1"
# Upper bound on the raiden offload flush wait (seconds). Dedicated to the
# offload path -- intentionally NOT the disagg p2p-pull timeout -- so it can be
# tuned independently.
_RAIDEN_OFFLOAD_WAIT_TIMEOUT_S = float(
    os.environ.get("VLLM_TPU_OFFLOAD_WAIT_TIMEOUT_S", "30"))


class _RaidenOffloadingHandler(OffloadingHandler):
    """One-direction (D2H or H2D) KV offload via raiden's direct DMA.

    `transfer_async` issues `mgr.D2h`/`mgr.H2d` (kernel-block granular) and
    tracks the returned `RaidenFuture`; `get_finished` polls it. The host pool
    lives inside raiden (DmaMapped), so there is no `cpu_pool` tensor here.
    """

    def __init__(self, mgr, tpu_to_cpu: bool, src_block_size_factor: int,
                 dst_block_size_factor: int, bytes_per_kernel_block: int):
        self._mgr = mgr
        self.tpu_to_cpu = tpu_to_cpu
        self.src_block_size_factor = src_block_size_factor
        self.dst_block_size_factor = dst_block_size_factor
        self._bytes_per_kernel_block = bytes_per_kernel_block
        self.transfer_type = ("GPU", "CPU") if tpu_to_cpu else ("CPU", "GPU")
        self._pending: dict[int, tuple] = {}

    def _expand_kernel_ids(
            self,
            transfer_spec: TransferSpec) -> tuple[np.ndarray, np.ndarray]:
        """Expand (src, dst) scheduler block IDs to kernel-block IDs.

        Mirrors SingleDirectionOffloadingHandler.transfer_async so D2H/H2D
        granularity matches the torch path exactly.
        """
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)
        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1 and dst_blocks.ndim == 1

        src_sub_count = src_blocks.size * self.src_block_size_factor
        dst_sub_count = dst_blocks.size * self.dst_block_size_factor
        src_skip = -dst_blocks.size % self.src_block_size_factor
        assert dst_sub_count == src_sub_count - src_skip

        src_expanded = np.empty(src_sub_count, dtype=np.int64)
        dst_expanded = np.empty(dst_sub_count, dtype=np.int64)
        expand_block_ids(src_blocks,
                         self.src_block_size_factor,
                         src_expanded,
                         skip_count=src_skip)
        expand_block_ids(dst_blocks, self.dst_block_size_factor, dst_expanded)
        return src_expanded[src_skip:], dst_expanded

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        # src/dst are kernel-block IDs. D2H: src=device blocks, dst=host slots;
        # H2D: src=host slots, dst=device blocks — matching D2h/H2d's
        # (src_offsets_major_dim, dst_offsets_major_dim) contract.
        src_ids, dst_ids = self._expand_kernel_ids(transfer_spec)
        n = len(dst_ids)
        sizes = [1] * n  # one major-dim slice (= one kernel block) per segment
        if self.tpu_to_cpu:
            fut = self._mgr.D2h(src_ids.tolist(), dst_ids.tolist(), sizes)
        else:
            fut = self._mgr.H2d(src_ids.tolist(), dst_ids.tolist(), sizes)
        self._pending[job_id] = (fut, n * self._bytes_per_kernel_block)
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        for job_id in list(self._pending.keys()):
            fut, num_bytes = self._pending[job_id]
            if not fut.is_ready():
                continue
            # Poll-only completion: is_ready() True means the DMA finished. Do
            # NOT call fut.wait()/Await here -- PJRT_Event_Await (and the
            # xla::Future BlockUntilReady it also drains) can deadlock against
            # the concurrently-executing model. The disagg connector likewise
            # reports completion by polling (complete_read) and never blocks.
            #
            # ok() is a non-blocking error probe (no Await), valid once
            # is_ready() is True; it tells a successful transfer from one that
            # completed with an error. hasattr-guarded so this still works
            # against a raiden build that predates the accessor (falls back to
            # the previous always-success behaviour).
            ok = fut.ok() if hasattr(fut, "ok") else True
            if not ok:
                err = (fut.error_message()
                       if hasattr(fut, "error_message") else "")
                logger.error("[kv-offload] Raiden transfer job=%s failed: %s",
                             job_id, err)
            results.append(
                TransferResult(
                    job_id=job_id,
                    success=ok,
                    transfer_size=num_bytes,
                    transfer_time=1e-9,
                    transfer_type=self.transfer_type,
                ))
            # Completed: drop the future (its keep-alives release the buffers).
            del self._pending[job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        """Block until the named transfers complete (flush primitive).

        Mirrors SingleDirectionOffloadingHandler.wait: cleanup (popping
        `_pending`, emitting TransferResults) is left to the next
        `get_finished()`, which the scheduler relies on to mark jobs done.
        """
        # Flush by POLLING is_ready (never the blocking Await, which can
        # deadlock against the live model -- see get_finished). Bound the wait
        # with a dedicated offload timeout so a stalled raiden DMA can't
        # busy-poll forever; on timeout the jobs stay in _pending and are
        # reported by a later get_finished().
        remaining = {jid for jid in job_ids if jid in self._pending}
        deadline = time.perf_counter() + _RAIDEN_OFFLOAD_WAIT_TIMEOUT_S
        while remaining:
            for jid in list(remaining):
                entry = self._pending.get(jid)
                if entry is None or entry[0].is_ready():
                    remaining.discard(jid)
            if not remaining:
                break
            if time.perf_counter() >= deadline:
                logger.warning(
                    "[kv-offload] Raiden wait timed out after %.1fs for "
                    "jobs=%s", _RAIDEN_OFFLOAD_WAIT_TIMEOUT_S,
                    sorted(remaining))
                return
            time.sleep(0.0005)


# ---------------------------------------------------------------------------
# Factory: allocates CPU pinned tensors, creates bidirectional handlers
# ---------------------------------------------------------------------------


class CpuTpuOffloadingHandlers:
    """Mirrors CpuGpuOffloadingHandlers for TPU.

    TPU KV-cache tensors always have block dimension at index 0:
        shape = (num_kernel_blocks, block_size, num_kv_heads, head_size)
    There is no split_k_and_v variant as in the GPU paged-attention backend.
    """

    def __init__(
        self,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        kv_caches: CanonicalKVCaches,
        kernel_block_size: int,
        kv_dtype: torch.dtype,
        per_block_shape: tuple[int, ...],
    ):
        """
        Args:
          gpu_block_size: scheduler block size (vllm config block_size).
          cpu_block_size: offloaded block size (gpu_block_size * factor).
          num_cpu_blocks: number of CPU-side scheduler blocks.
          kv_caches: vllm 0.19 CanonicalKVCaches — each .tensor is a 2D int8
              view (num_blocks, page_size_bytes) sharing storage with the
              model's kv_cache.
          kernel_block_size: physical block_size in the original 5D Pallas
              layout (typically 16). gpu_block_size must be a multiple of it.
          kv_dtype: original dtype of the kv_cache (bf16, fp8, ...).
          per_block_shape: per-block trailing dims in the 5D Pallas layout
              (e.g. (block_size, num_kv_heads_x2 // kv_packing, kv_packing,
              padded_head_size)) — everything after the leading num_blocks
              dim.
        """
        assert kv_caches.tensors
        assert cpu_block_size % gpu_block_size == 0
        assert gpu_block_size % kernel_block_size == 0

        cpu_block_size_factor = cpu_block_size // kernel_block_size
        gpu_block_size_factor = gpu_block_size // kernel_block_size
        num_cpu_kernel_blocks = num_cpu_blocks * cpu_block_size_factor

        tpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []

        logger.debug(
            "[kv-offload] Allocating %d unpinned CPU pool tensors "
            "(num_cpu_blocks=%d, gpu_factor=%d, cpu_factor=%d)",
            len(kv_caches.tensors),
            num_cpu_blocks,
            gpu_block_size_factor,
            cpu_block_size_factor,
        )
        total_bytes = 0
        for kv_cache_tensor in kv_caches.tensors:
            int8_view = kv_cache_tensor.tensor
            num_blocks = int8_view.shape[0]
            # Rebuild the original 5D Pallas view from the same storage.
            # vllm canonicalized to (num_blocks, page_size_bytes) int8 via
            # .set_(storage).view(num_blocks, page_size_bytes); we round-trip
            # to the kv_dtype-typed 5D shape so Pallas gather/scatter can
            # operate on it directly.
            tpu_tensor = (torch.empty(
                0, dtype=kv_dtype, device=int8_view.device).set_(
                    int8_view.untyped_storage()).view((num_blocks, ) +
                                                      tuple(per_block_shape)))
            cpu_shape = (num_cpu_kernel_blocks, ) + tuple(per_block_shape)
            # cpu_pool is intentionally NOT pinned. The DMA-capable host
            # staging buffer is allocated per-transfer (pinned) in
            # register_store/register_load, and the host
            # gather/scatter step copies between cpu_pool and that pinned
            # staging buffer. Pinning the full cpu_pool would lock hundreds
            # of GB and starve the libtpu premap region.
            tpu_tensors.append(tpu_tensor)
            if not _USE_RAIDEN_OFFLOAD:
                cpu_tensor = torch.zeros(
                    cpu_shape,
                    dtype=kv_dtype,
                    device="cpu",
                )
                total_bytes += cpu_tensor.element_size() * cpu_tensor.numel()
                cpu_tensors.append(cpu_tensor)

        if _USE_RAIDEN_OFFLOAD:
            # raiden owns a DmaMapped host pool (no cpu_pool tensor). Build one
            # KVCacheManager over the device KV-cache views; two direct-DMA
            # handlers share it. host_blocks_to_allocate matches the scheduler's
            # CPU pool sized in kernel blocks, so CPU block IDs map 1:1 to
            # raiden host slots after expand_block_ids.
            from tpu_raiden.api.torch import kv_cache_manager as _kcm
            device_tensors = [[t] for t in tpu_tensors]
            # raiden holds raw pointers / PJRT aliases to these device buffers
            # for its lifetime. Keep the tensors (and thus the underlying
            # buffers) alive on the factory, or they get GC'd after __init__
            # and raiden's D2h/H2d dereferences freed buffers (segfault).
            self._raiden_device_tensors = device_tensors
            self._raiden_mgr = _kcm._impl.KVCacheManager(
                device_tensors,
                host_blocks_to_allocate=num_cpu_kernel_blocks,
                unsafe_skip_buffer_lock=True,
            )
            bytes_per_kernel_block = (
                int(np.prod(per_block_shape)) *
                torch.empty(0, dtype=kv_dtype).element_size())
            self.gpu_to_cpu_handler = _RaidenOffloadingHandler(
                self._raiden_mgr,
                tpu_to_cpu=True,
                src_block_size_factor=gpu_block_size_factor,
                dst_block_size_factor=cpu_block_size_factor,
                bytes_per_kernel_block=bytes_per_kernel_block,
            )
            self.cpu_to_gpu_handler = _RaidenOffloadingHandler(
                self._raiden_mgr,
                tpu_to_cpu=False,
                src_block_size_factor=cpu_block_size_factor,
                dst_block_size_factor=gpu_block_size_factor,
                bytes_per_kernel_block=bytes_per_kernel_block,
            )
            logger.info(
                "[kv-offload] RAIDEN direct-DMA offload: %d host kernel blocks",
                num_cpu_kernel_blocks,
            )
            return

        logger.debug(
            "[kv-offload] Allocated %.2f GiB unpinned CPU pool",
            total_bytes / (1 << 30),
        )

        self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(
            src_tensors=tpu_tensors,
            dst_tensors=cpu_tensors,
            src_block_size_factor=gpu_block_size_factor,
            dst_block_size_factor=cpu_block_size_factor,
        )
        self.cpu_to_gpu_handler = SingleDirectionOffloadingHandler(
            src_tensors=cpu_tensors,
            dst_tensors=tpu_tensors,
            src_block_size_factor=cpu_block_size_factor,
            dst_block_size_factor=gpu_block_size_factor,
        )


# ---------------------------------------------------------------------------
# Spec: subclasses CPUOffloadingSpec, overrides get_handlers() for TPU
# ---------------------------------------------------------------------------


class TPUCPUOffloadingSpec(CPUOffloadingSpec):
    """CPU offloading spec for TPU.

    Reuses CPUOffloadingSpec.__init__ (block-size calculation, eviction-policy
    config) and get_manager() (LRU/ARC setup).  Only get_handlers() is
    overridden to create TPU-specific transfer handlers.
    """

    def __init__(self, vllm_config: VllmConfig,
                 kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)
        self._tpu_handlers: CpuTpuOffloadingHandlers | None = None

    @property
    def prewarm_shapes(self) -> list[int]:
        """H2D Pallas-scatter shapes the runner should pre-compile.

        Delegates to the H2D handler, which is the actual owner of the
        Pallas kernels. Returns [] if get_handlers() hasn't run yet —
        TpuMultiprocExecutor calls this AFTER register_kv_caches, so the
        handler is set by then; the guard is defensive.
        """
        if self._tpu_handlers is None:
            return []
        return self._tpu_handlers.cpu_to_gpu_handler.prewarm_shapes

    def prewarm_shape(self, p: int) -> None:
        """Pre-compile H2D Pallas scatter for block-count `p`. Delegates
        to the H2D handler; no-op before get_handlers() has run."""
        if self._tpu_handlers is None:
            return
        self._tpu_handlers.cpu_to_gpu_handler.prewarm_shape(p)

    @classmethod
    def estimate_hbm_reserve_bytes(cls, vllm_config: VllmConfig) -> int:
        """Upper bound on HBM the H2D staging buffer will consume.

        Called by tpu_worker before profile_run to shrink the KV-cache
        budget so vllm leaves room for the SingleDirectionOffloadingHandler's
        _h2d_device_buffer, which is allocated lazily after
        determine_available_memory() returns.

        Only TPUCPUOffloadingSpec needs this reserve; other connectors
        (e.g., TPUConnector for P/D disagg) don't allocate an HBM staging
        buffer, so the worker must NOT shrink their budgets. Living on the
        spec class makes the gating explicit at the call site.
        """
        max_pool_blocks = int(os.environ.get("KV_H2D_POOL_MAX_BLOCKS", "2048"))
        max_padded = 1
        while max_padded < max_pool_blocks:
            max_padded <<= 1

        from vllm_torchtpu.layers.vllm.attention import (
            TPU_STR_DTYPE_TO_TORCH_DTYPE, PallasAttentionBackend)

        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        # _resolve_kv_cache_dtype is strict on "auto", so resolve it here
        # the same way tpu_worker / tpu_runner already do.
        if cache_config.cache_dtype == "auto":
            model_dtype = model_config.dtype
            kv_dtype = (TPU_STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
                        if isinstance(model_dtype, str) else model_dtype)
        else:
            kv_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        per_block_bytes = PallasAttentionBackend.get_kv_cache_page_size_bytes(
            cache_config.block_size,
            model_config.get_num_kv_heads(parallel_config),
            model_config.get_head_size(),
            kv_dtype,
        )
        # Upper bound: vLLM may group multiple layers into one canonical
        # tensor, in which case the actual H2D buffer is smaller. Safe
        # direction (overestimate leaves a few GiB unused).
        num_layers = model_config.get_num_layers(parallel_config)
        return num_layers * max_padded * per_block_bytes

    def get_handlers(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec],
                        OffloadingHandler]]:
        if self._tpu_handlers is None:
            from vllm.v1.kv_cache_interface import AttentionSpec
            from vllm_torchtpu.layers.vllm.attention import \
                PallasAttentionBackend

            assert len(self.gpu_block_size) == 1
            gpu_block_size = self.gpu_block_size[0]
            offloaded_block_size = gpu_block_size * self.block_size_factor

            # All kv_cache_groups must share the same attention spec so the
            # rebuilt 5D view is consistent across canonical tensors.
            kv_specs = [
                g.kv_cache_spec for g in self.kv_cache_config.kv_cache_groups
                if isinstance(g.kv_cache_spec, AttentionSpec)
            ]
            assert kv_specs, "TPU offloading requires at least one AttentionSpec group"
            spec0 = kv_specs[0]
            for s in kv_specs[1:]:
                assert (
                    s.block_size == spec0.block_size
                    and s.num_kv_heads == spec0.num_kv_heads
                    and s.head_size == spec0.head_size
                    and s.dtype == spec0.dtype
                ), ("TPU offloading assumes a single attention spec across "
                    "all KV cache groups")

            full_5d_shape = PallasAttentionBackend.get_kv_cache_shape(
                num_blocks=1,
                block_size=spec0.block_size,
                num_kv_heads=spec0.num_kv_heads,
                head_size=spec0.head_size,
                cache_dtype_str=spec0.dtype,
            )
            per_block_shape = tuple(full_5d_shape[1:])

            self._tpu_handlers = CpuTpuOffloadingHandlers(
                gpu_block_size=gpu_block_size,
                cpu_block_size=offloaded_block_size,
                num_cpu_blocks=self.num_blocks,
                kv_caches=kv_caches,
                kernel_block_size=spec0.block_size,
                kv_dtype=spec0.dtype,
                per_block_shape=per_block_shape,
            )

        yield (
            GPULoadStoreSpec,
            CPULoadStoreSpec,
            self._tpu_handlers.gpu_to_cpu_handler,
        )
        yield (
            CPULoadStoreSpec,
            GPULoadStoreSpec,
            self._tpu_handlers.cpu_to_gpu_handler,
        )
