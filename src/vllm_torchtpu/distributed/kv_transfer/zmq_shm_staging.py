# SPDX-License-Identifier: Apache-2.0
"""Stages device shards in shared memory and scatters received shards."""

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (LoadMeta,
                                                                      ReqId)
from vllm_torchtpu.distributed.kv_transfer.zmq_shm_coordination import \
    _IPC_COPY_DONE
from vllm_torchtpu.distributed.kv_transfer.zmq_shm_telemetry import (
    _LAT_D2H, _LAT_H2D, _LAT_INSERT, _LAT_SCATTER, _LAT_STAGE)
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm_torchtpu.distributed.kv_transfer.zmq_shm_base import \
        ZmqShmKvConnectorBase

logger = init_logger("vllm_torchtpu.distributed.kv_transfer.zmq_shm_base")


@dataclass
class _StageInFlight:
    """A pending async D2H batch waiting on its PjRt future.

    Tensors are retained here so that the source device buffer and the
    destination shm view stay alive until the DMA completes. A pool worker
    blocks on `future.wait()` and then signals STAGE_DONE for this rank;
    the dispatcher loop may also signal STAGE_DONE early on a deadline
    timeout, in which case the worker's eventual call is deduplicated by
    `_signal_lock` / `_signaled`."""
    uuid: int
    slot_idx: int
    num_blocks: int
    future: Any  # TransferFuture
    tpu_tensors: list
    cpu_tensors: list
    enqueue_t0: float
    issue_t1: float = 0.0
    total_bytes: int = 0
    _signal_lock: threading.Lock = field(default_factory=threading.Lock)
    _signaled: bool = False


def _warmup_kv_ops(worker: "ZmqShmKvConnectorBase") -> None:
    """Pre-compile the XLA executables for the KV gather/scatter hot
    path so the first real request doesn't block on a long compile and
    time out the PULL round-trip on the other side.

    Producer role pre-compiles the D2H path (`index_select` + copy into
    a host staging tensor). Consumer role pre-compiles the H2D +
    `index_put_` path. Both are done per-layer, matching runtime."""
    sizes = worker._warmup_block_sizes()
    logger.info(
        "TPUConnectorWorker(%s) rank%d --> warmup starting | role=%s | "
        "num_layers=%d | block_sizes=%s", worker.node_id, worker.tp_rank,
        "producer" if worker.is_producer else "consumer", worker.num_layers,
        sizes)
    start = time.perf_counter()
    for n in sizes:
        if n <= 0:
            continue
        t0 = time.perf_counter()
        worker._warmup_coord_once(n)
        logger.info(
            "TPUConnectorWorker(%s) rank%d --> warmup num_blocks=%s "
            "took %.2fs", worker.node_id, worker.tp_rank, n,
            time.perf_counter() - t0)
    logger.info("TPUConnectorWorker(%s) rank%d --> warmup done in %.2fs",
                worker.node_id, worker.tp_rank,
                time.perf_counter() - start)


def _warmup_coord_once(worker: "ZmqShmKvConnectorBase",
                       num_blocks: int) -> None:
    """TP>1 coord path: warmup using slot 0 of the shm pool. No traffic
    exists yet at register_runner time, so touching slot 0 is safe.

    Mirrors _coord_stage_shard / _coord_scatter_shard exactly so the
    compiled executable cache hits on first real request. Inlined
    rather than calling the hot path so warmup samples don't leak into
    the latency tracker or emit a stray scatter log line."""
    kv_caches = worker.runner.kv_caches
    indices = torch.arange(num_blocks, dtype=torch.int64, device=worker.device)
    slot_idx = 0
    if worker.is_producer:
        worker._stage_d2h_sync(slot_idx, num_blocks, list(range(num_blocks)))
    else:
        # Mirror _coord_scatter_shard exactly. An earlier
        # `.to(device)` warmup compiled a different HLO than the hot
        # path (batch_transfer_h2d into a pre-allocated torch.empty),
        # so the first real request paid a 3-8s recompile.
        use_kv_scatter = worker._kv_scatter_enabled
        src_views = [
            worker._coord_pool.layer_view(slot_idx, worker.local_tp_rank,
                                          layer_idx, num_blocks)
            for layer_idx in range(len(kv_caches))
        ]
        h2d_future, device_shards = worker._h2d_into_device_async(src_views)
        worker._wait_stage(h2d_future)
        if use_kv_scatter:
            new_caches = worker._try_fast_scatter(device_shards,
                                                  list(kv_caches),
                                                  list(range(num_blocks)))
            for c in new_caches:
                worker._synchronize_device(c)
            for i, c in enumerate(new_caches):
                worker._replace_runner_kv_cache(i, c)
        else:
            for layer_idx, cache in enumerate(kv_caches):
                cache.index_put_((indices, ), device_shards[layer_idx])
                worker._synchronize_device(cache)


def _coord_worker_stage_inline(worker: "ZmqShmKvConnectorBase",
                               uuid: int,
                               req_id: ReqId,
                               timeout: float = 10.0) -> None:
    """Non-zero rank, main-thread path: wait for STAGE_NOTIFY to arrive
    in _worker_pending_stage (populated by the IPC listener thread),
    then run stage_shard synchronously. Must happen before the main
    thread enters model_forward, otherwise .to("cpu") will block behind
    the first-time decode compile."""
    logger.info(
        "TPUConnectorWorker(%d) rank%d --> stage_inline waiting for "
        "STAGE_NOTIFY req_id=%s uuid=%s", worker.node_id, worker.tp_rank,
        req_id, uuid)
    deadline = time.perf_counter() + timeout
    info: Optional[tuple[int, int, list[int]]] = None
    with worker._worker_pending_stage_cv:
        while uuid not in worker._worker_pending_stage:
            rem = deadline - time.perf_counter()
            if rem <= 0:
                logger.warning(
                    "TPUConnectorWorker(%d) rank%d --> stage_inline timeout "
                    "uuid=%s (STAGE_NOTIFY never arrived)", worker.node_id,
                    worker.tp_rank, uuid)
                # Without STAGE_DONE this rank's shard never completes, so
                # rank0 keeps the slot pending and the consumer's PULL
                # blocks for its full timeout, leaking a consumer slot per
                # occurrence. Signal failure like the exception path below
                # so PULL serving answers _MSG_ERR immediately.
                worker._signal_stage_done(uuid, failed=True)
                return
            worker._worker_pending_stage_cv.wait(timeout=rem)
        info = worker._worker_pending_stage.pop(uuid)
    slot_idx, num_blocks, block_ids = info
    try:
        worker._coord_stage_shard(uuid, slot_idx, num_blocks, block_ids)
    except Exception as e:
        logger.exception(
            "TPUConnectorWorker(%d) rank%d --> stage_inline uuid=%s failed: %s",
            worker.node_id, worker.tp_rank, uuid, e)
        # Enqueue path didn't run, so the waiter won't fire STAGE_DONE
        # for us. Signal it directly with failed=True so rank 0 marks
        # the entry stage_failed and PULL serving will _MSG_ERR rather
        # than ship a partially-zero shm slice.
        worker._signal_stage_done(uuid, failed=True)


def _coord_stage_shard(worker: "ZmqShmKvConnectorBase", uuid: int,
                       slot_idx: int, num_blocks: int,
                       block_ids: list[int]) -> None:
    """Issue an async D2H batch for this rank's shard and enqueue the
    future for the waiter thread.

    Returns immediately after the enqueue: the DMA proceeds in
    background, and the per-rank STAGE_DONE is emitted by the waiter
    once `future.wait()` returns. This frees the worker's main RPC
    thread to advance vLLM, and lets the D2H overlap with the
    STAGE_NOTIFY -> scheduler -> PULL-setup round-trip and any
    subsequent model_forward.

    Source device tensors and destination shm views are retained on
    the in-flight entry until the future completes, so the source
    device buffer and destination shm backing stay alive across the
    DMA."""
    logger.info(
        "TPUConnectorWorker(%s) rank%d --> stage_shard enqueue "
        "uuid=%s slot=%d blocks=%s layers=%d", worker.node_id, worker.tp_rank,
        uuid, slot_idx, num_blocks, len(worker.runner.kv_caches))
    enqueue_t0 = time.perf_counter()
    future, src_refs, dst_refs, d2h_total_bytes = worker._stage_d2h(
        slot_idx, num_blocks, block_ids)
    issue_t1 = time.perf_counter()
    enqueue_ms = (time.perf_counter() - enqueue_t0) * 1000.0

    worker._stage_pending_q.put(
        _StageInFlight(uuid=uuid,
                       slot_idx=slot_idx,
                       num_blocks=num_blocks,
                       future=future,
                       tpu_tensors=src_refs,
                       cpu_tensors=dst_refs,
                       enqueue_t0=enqueue_t0,
                       issue_t1=issue_t1,
                       total_bytes=d2h_total_bytes))
    logger.info(
        "TPUConnectorWorker(%s) rank%d --> stage_shard enqueued "
        "uuid=%s slot=%d enqueue=%.2fms", worker.node_id, worker.tp_rank, uuid,
        slot_idx, enqueue_ms)


def _coord_stage_waiter_loop(worker: "ZmqShmKvConnectorBase") -> None:
    """Dispatch each pending stage entry to the waiter pool and enforce
    a per-entry deadline. Pool workers do the blocking ``_wait_stage``
    and signal STAGE_DONE on success; the dispatcher signals a failed
    STAGE_DONE if an entry exceeds `_coord_stage_wait_timeout_s`, so a
    wedged DMA cannot starve later uuids on the same rank.

    ``_wait_stage`` is unbounded under most device backends (e.g.
    xla::Future::Await), so leaking the worker thread is the only way
    to break out of a hung wait from Python. The leaked thread is
    harmless: the device runtime owns the underlying buffers via its
    own holds, and `_complete_stage_entry` is idempotent so the
    eventual return is a no-op.
    """
    timeout_s = worker._coord_stage_wait_timeout_s
    # (deadline, entry, py_future); list ordering is not load-bearing.
    in_flight: list[tuple[float, _StageInFlight, Any]] = []
    while not worker._stop_event.is_set():
        got_entry = True
        try:
            entry = worker._stage_pending_q.get(timeout=0.05)
        except queue.Empty:
            got_entry = False
            entry = None
        if got_entry:
            if entry is None:
                # Shutdown sentinel from _coord_teardown.
                return
            py_fut = worker._coord_stage_waiter_pool.submit(
                worker._coord_stage_wait_worker, entry)
            in_flight.append((time.monotonic() + timeout_s, entry, py_fut))
        if in_flight:
            now = time.monotonic()
            kept: list[tuple[float, _StageInFlight, Any]] = []
            for deadline, ent, fut in in_flight:
                if fut.done():
                    # Worker already signaled (success path or caught
                    # exception path); nothing more to do here.
                    continue
                if now >= deadline:
                    logger.warning(
                        "TPUConnectorWorker(%s) rank%d --> stage_waiter "
                        "uuid=%s slot=%d future.wait() exceeded "
                        "%.1fs; signaling failed STAGE_DONE (worker "
                        "thread leaks until wait returns)", worker.node_id,
                        worker.tp_rank, ent.uuid, ent.slot_idx, timeout_s)
                    worker._complete_stage_entry(ent, failed=True)
                    # Stop tracking; the eventual worker callback
                    # dedups via the entry's signal lock.
                    continue
                kept.append((deadline, ent, fut))
            in_flight = kept


def _coord_stage_wait_worker(worker: "ZmqShmKvConnectorBase",
                             entry: "_StageInFlight") -> None:
    """Pool worker: block on the entry's stage future and complete.

    Runs in `_coord_stage_waiter_pool`. Several can run in parallel
    across distinct entries; a single hung wait only consumes one slot
    until the dispatcher's deadline marks the entry failed."""
    failed = False
    try:
        wait_t0 = time.perf_counter()
        worker._wait_stage(entry.future)
        wait_ms = (time.perf_counter() - wait_t0) * 1000.0
        total_ms = (time.perf_counter() - entry.enqueue_t0) * 1000.0
        issue_ms = ((entry.issue_t1 - entry.enqueue_t0) *
                    1000.0 if entry.issue_t1 else 0.0)
        mb = entry.total_bytes / (1024 * 1024)
        mbps = mb / max(1e-3, total_ms / 1000.0)
        worker._lat.record(_LAT_D2H, total_ms)
        worker._lat.record(_LAT_STAGE, total_ms)
        worker.transfer_stats.record_d2h_transfer(total_ms)
        logger.info(
            "TPUConnectorWorker(%s) rank%d --> stage_done "
            "uuid=%s slot=%d issue=%.2fms wait=%.2fms total=%.2fms "
            "throughput=%.2fMiB/s", worker.node_id, worker.tp_rank, entry.uuid,
            entry.slot_idx, issue_ms, wait_ms, total_ms, mbps)
    except Exception as e:
        failed = True
        logger.exception(
            "TPUConnectorWorker(%s) rank%d --> stage_waiter uuid=%s "
            "future.wait() raised: %s", worker.node_id, worker.tp_rank,
            entry.uuid, e)
    worker._complete_stage_entry(entry, failed=failed)


def _complete_stage_entry(worker: "ZmqShmKvConnectorBase",
                          entry: "_StageInFlight", failed: bool) -> None:
    """Run completion side-effects exactly once per entry.

    Both the worker and the dispatcher's deadline path may call this;
    whichever wins clears the tensor refs and emits STAGE_DONE. The
    loser short-circuits at the lock guard."""
    with entry._signal_lock:
        if entry._signaled:
            return
        entry._signaled = True
    entry.tpu_tensors.clear()
    entry.cpu_tensors.clear()
    worker._signal_stage_done(entry.uuid, failed=failed)


def _coord_scatter_and_ack(worker: "ZmqShmKvConnectorBase", req_id: ReqId,
                           uuid: int, slot_idx: int, num_blocks: int,
                           local_blocks: list[int]) -> None:
    logger.info("PERF D drain_start rank=%d req_id=%s uuid=%s ts=%.6f",
                worker.tp_rank, req_id, uuid, time.time())
    try:
        worker._coord_scatter_shard(slot_idx, num_blocks, local_blocks)
    except Exception as e:
        logger.exception(
            "TPUConnectorWorker(%s) rank%d --> scatter req_id=%s failed: %s",
            worker.node_id, worker.tp_rank, req_id, e)
    finally:
        if worker._is_host_coordinator:
            worker._coord_rank0_register_copy(uuid, worker.local_tp_rank)
        else:
            worker._coord_send_ipc(_IPC_COPY_DONE,
                                   (uuid, worker.local_tp_rank))
        logger.info("PERF D drain_done rank=%d req_id=%s uuid=%s ts=%.6f",
                    worker.tp_rank, req_id, uuid, time.time())


def _coord_drain_scatter(worker: "ZmqShmKvConnectorBase", req_id: ReqId,
                         req_meta: LoadMeta) -> None:
    uuid = req_meta.uuid
    if worker._is_host_coordinator:
        # Rank 0's state is in _coord_recv; grab slot info from there.
        with worker._coord_lock:
            entry = worker._coord_recv.get(uuid)
        if entry is None:
            return
        if not entry.pull_ok:
            # Unknown or failed; scheduler will treat this as a cache hit
            # and we notify upstream so P can free its buffer.
            worker._coord_rank0_send_notify(req_meta, uuid)
            return
        slot_idx = entry.slot_idx
        num_blocks = entry.num_blocks
        local_blocks = entry.local_blocks
    else:
        info = worker._coord_worker_wait_load(uuid)
        if info is None:
            # DROP arrived or timeout. Nothing to do.
            return
        slot_idx, num_blocks, local_blocks = info

    worker._coord_scatter_and_ack(req_id, uuid, slot_idx, num_blocks,
                                  local_blocks)


def _coord_scatter_shard(worker: "ZmqShmKvConnectorBase", slot_idx: int,
                         num_blocks: int, local_blocks: list[int]) -> None:
    """H2D + scatter this rank's shard from shm into the kv cache.

    Two paths:
      * Multi-layer fused scatter (default when ``_kv_scatter_enabled``):
        sync H2D into freshly allocated device buffers, then one fused
        ``_try_fast_scatter`` call writes all layers at once with each
        destination donated+aliased to its output.
      * Naive fallback (smoke test failed): async H2D + explicit wait
        so issue/wait timings are observable, then per-layer
        ``index_put_`` followed by ``_synchronize_device``."""
    if not local_blocks:
        return
    kv_caches = worker.runner.kv_caches
    use_fast_scatter = worker._kv_scatter_enabled
    h2d_ms_total = 0.0
    h2d_issue_ms = 0.0
    h2d_wait_ms = 0.0
    insert_ms_total = 0.0
    alloc_ms_total = 0.0
    if use_fast_scatter:
        alloc_t0 = time.perf_counter()
        src_views = []
        h2d_total_bytes = 0
        for layer_idx in range(len(kv_caches)):
            src_view = worker._coord_pool.layer_view(slot_idx,
                                                     worker.local_tp_rank,
                                                     layer_idx, num_blocks)
            src_views.append(src_view)
            h2d_total_bytes += src_view.numel() * src_view.element_size()
        alloc_t1 = time.perf_counter()

        h2d_t0 = time.perf_counter()
        device_shards = worker._h2d_into_device(src_views)
        h2d_t1 = time.perf_counter()
        new_caches = worker._try_fast_scatter(device_shards, list(kv_caches),
                                              local_blocks)
        # Sync every returned cache -- they're independent outputs of the
        # fused kernel and the device may schedule them in parallel, so
        # syncing only the last would race.
        for c in new_caches:
            worker._synchronize_device(c)
        k_wait_t = time.perf_counter()
        for i, c in enumerate(new_caches):
            worker._replace_runner_kv_cache(i, c)
        h2d_ms_total = (h2d_t1 - h2d_t0) * 1000.0
        # The sync H2D helper blocks until completion, so there is no
        # separately observable issue/wait split.
        h2d_issue_ms = h2d_ms_total
        insert_ms_total = (k_wait_t - h2d_t1) * 1000.0
        alloc_ms_total = (alloc_t1 - alloc_t0) * 1000.0
        path = "kv_scatter"
    else:
        indices = torch.tensor(local_blocks,
                               dtype=torch.int64,
                               device=worker.device)
        alloc_t0 = time.perf_counter()
        src_views = []
        h2d_total_bytes = 0
        for layer_idx in range(len(kv_caches)):
            src_view = worker._coord_pool.layer_view(slot_idx,
                                                     worker.local_tp_rank,
                                                     layer_idx, num_blocks)
            src_views.append(src_view)
            h2d_total_bytes += src_view.numel() * src_view.element_size()
        alloc_t1 = time.perf_counter()

        h_t0 = time.perf_counter()
        h2d_future, device_shards = worker._h2d_into_device_async(src_views)
        h2d_t_issued = time.perf_counter()
        worker._wait_stage(h2d_future)
        h_t1 = time.perf_counter()

        h2d_ms_total = (h_t1 - h_t0) * 1000.0
        h2d_issue_ms = (h2d_t_issued - h_t0) * 1000.0
        h2d_wait_ms = (h_t1 - h2d_t_issued) * 1000.0
        insert_ms_total = 0.0
        alloc_ms_total = (alloc_t1 - alloc_t0) * 1000.0

        for layer_idx, cache in enumerate(kv_caches):
            ins_t0 = time.perf_counter()
            cache.index_put_((indices, ), device_shards[layer_idx])
            worker._synchronize_device(cache)
            insert_ms_total += (time.perf_counter() - ins_t0) * 1000.0
        path = "naive"
    worker._lat.record(_LAT_H2D, h2d_ms_total)
    worker._lat.record(_LAT_INSERT, insert_ms_total)
    worker._lat.record(_LAT_SCATTER, h2d_ms_total + insert_ms_total)
    worker.transfer_stats.record_h2d_transfer(h2d_ms_total)
    h2d_mb = h2d_total_bytes / (1024 * 1024)
    h2d_mbps = h2d_mb / max(1e-3, h2d_ms_total / 1000.0)
    logger.info(
        "TPUConnectorWorker(%s) rank%d --> scatter slot=%d blocks=%s "
        "layers=%d alloc=%.2fms h2d=%.2fms h2d_issue=%.2fms "
        "h2d_wait=%.2fms insert=%.2fms h2d_throughput=%.2fMiB/s path=%s",
        worker.node_id, worker.tp_rank, slot_idx, num_blocks, len(kv_caches),
        alloc_ms_total, h2d_ms_total, h2d_issue_ms, h2d_wait_ms,
        insert_ms_total, h2d_mbps, path)
