# SPDX-License-Identifier: Apache-2.0
"""Coordinates ZMQ sockets, authenticated messages, and request completion."""

import hashlib
import hmac
import os
import pickle
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import zmq
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (LoadMeta,
                                                                      ReqId,
                                                                      SendMeta)
from vllm_torchtpu.distributed.kv_transfer.zmq_shm_telemetry import (
    _LAT_UNPACK, _LAT_WIRE)
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm_torchtpu.distributed.kv_transfer.zmq_shm_base import \
        ZmqShmKvConnectorBase

logger = init_logger("vllm_torchtpu.distributed.kv_transfer.zmq_shm_base")


def _get_default_ipc_key() -> bytes:
    key_str = os.environ.get("VLLM_TORCHTPU_IPC_KEY", "")
    if key_str:
        return key_str.encode("utf-8")
    machine_id = ""
    try:
        if os.path.exists("/etc/machine-id"):
            with open("/etc/machine-id", "r") as f:
                machine_id = f.read().strip()
    except Exception:
        pass
    user_id = str(os.getuid()) if hasattr(os, "getuid") else "0"
    base_secret = f"vllm-torchtpu-ipc-auth-{machine_id}-{user_id}".encode(
        "utf-8")
    return hashlib.sha256(base_secret).digest()


_IPC_AUTH_KEY: bytes = _get_default_ipc_key()


def _secure_dumps(obj: Any) -> bytes:
    """Serialize payload with an HMAC-SHA256 signature to prevent untrusted deserialization."""
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sig = hmac.new(_IPC_AUTH_KEY, raw, hashlib.sha256).digest()
    return sig + raw


def _secure_loads(data: bytes) -> Any:
    """Verify HMAC-SHA256 signature before deserializing payload."""
    sig_len = 32
    if len(data) < sig_len:
        raise ValueError("IPC payload too short for HMAC authentication")
    sig, raw = data[:sig_len], data[sig_len:]
    expected_sig = hmac.new(_IPC_AUTH_KEY, raw, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected_sig):
        raise ValueError(
            "IPC payload HMAC authentication failed: potential tampering detected"
        )
    return pickle.loads(raw)


# Wire-format tags for the ZMQ data channel.
_MSG_PULL = b"PULL"
_MSG_OK = b"OK"
_MSG_ERR = b"ERR"

# Intra-host IPC op tags (rank-0 coordinator <-> other ranks).
_IPC_HELLO = b"HELLO"  # worker -> coord: (rank,)
_IPC_STAGE_NOTIFY = b"STAGE_N"  # coord -> worker: (uuid, slot_idx, num_blocks)
_IPC_STAGE_DONE = b"STAGE_D"  # worker -> coord: (uuid, rank, failed)
_IPC_LOAD_NOTIFY = b"LOAD_N"  # coord -> worker: (uuid, slot_idx, num_blocks, local_blocks)
_IPC_LOAD_SKIP = b"LOAD_S"  # coord -> worker: (uuid,) -- drain w/ no scatter (cache hit)
_IPC_COPY_DONE = b"COPY_D"  # worker -> coord: (uuid, rank)
_IPC_DROP = b"DROP"  # coord -> worker: (uuid,)  -- failure/timeout


@dataclass
class _CoordSendEntry:
    req_id: ReqId
    slot_idx: int
    num_blocks: int
    expiration_time: float
    # tp_size at construction; sizes the per-rank readiness vector below.
    # Default is for backward-compat with any test that instantiates the
    # dataclass without it; production callers always pass it.
    tp_size: int = 1
    # Ranks that have sent STAGE_DONE. Once |staged| == tp_size the slot is
    # ready to serve external PULL requests for any subset of ranks.
    staged: set[int] = field(default_factory=set)
    # Per-rank staged event. PULL serving (`_coord_rank0_build_pull_response`)
    # waits only on the ranks that ride its channel, instead of blocking on
    # global stage_complete -- a slow rank no longer holds back channels
    # whose ranks already finished staging. Sized in __post_init__.
    staged_events: list[threading.Event] = field(default_factory=list)
    # Set when ALL tp_size ranks have staged. Kept around for the existing
    # PERF P stage_complete log line; no longer gates PULL serving.
    stage_complete: threading.Event = field(default_factory=threading.Event)
    # Flipped True when the remote side has acked the pull via NOTIFY, so
    # get_finished() can return this req as done_sending.
    pull_acked: bool = False
    # Set when any rank's stage waiter aborted on timeout (or its
    # `future.wait()` raised). PULL serving must surface this so a doomed
    # request errors out cleanly instead of shipping incomplete shm data.
    stage_failed: bool = False

    def __post_init__(self):
        if not self.staged_events:
            self.staged_events = [
                threading.Event() for _ in range(self.tp_size)
            ]


@dataclass
class _CoordRecvEntry:
    req_id: ReqId
    uuid: int
    slot_idx: int
    num_blocks: int
    local_blocks: list[int]
    remote_blocks: list[int]
    remote_host: str | list[str]
    remote_port: int | list[int]
    remote_side_channel_port: Optional[int] = None
    # Set by the async pull thread once data has been written to shm.
    load_complete: threading.Event = field(default_factory=threading.Event)
    # Ranks that have sent COPY_DONE. Once |copied| == tp_size we can notify
    # the remote producer and release the slot.
    copied: set[int] = field(default_factory=set)
    # Did the pull actually succeed? If False, we skip done_recving.
    pull_ok: bool = False
    # Have we already surfaced this req in get_finished()? Prevents duplicate
    # reporting across scheduler ticks while we wait for per-rank copies.
    reported_done: bool = False


def _coord_send_ipc(worker: "ZmqShmKvConnectorBase", tag: bytes,
                    payload) -> None:
    """Worker (non-zero rank) -> coordinator. Enqueues a DEALER frame."""
    data = _secure_dumps(payload)
    # DEALER frames are [tag, data]; ROUTER side reassembles identity
    # from the socket itself.
    worker._coord_ipc_out.put((None, tag, data))


def _coord_broadcast(worker: "ZmqShmKvConnectorBase", tag: bytes,
                     payload) -> None:
    """Coordinator -> every known non-zero rank. Enqueues one frame per
    rank, addressed to the rank's DEALER identity."""
    data = _secure_dumps(payload)
    with worker._coord_lock:
        identities = list(worker._coord_rank_to_identity.values())
    for ident in identities:
        worker._coord_ipc_out.put((ident, tag, data))


def _coord_ipc_drain_outbound(worker: "ZmqShmKvConnectorBase") -> None:
    """Pop any pending outbound messages and send them. Called from
    inside the owning IPC loop; must not be called from other threads."""
    sock = worker._coord_ipc_sock
    try:
        while True:
            ident, tag, data = worker._coord_ipc_out.get_nowait()
            try:
                if ident is None:
                    # DEALER: [tag, data]
                    sock.send_multipart([tag, data])
                else:
                    # ROUTER: [identity, tag, data]
                    sock.send_multipart([ident, tag, data])
            except zmq.ZMQError as e:
                logger.warning(
                    "TPUConnectorWorker(%s) --> IPC send tag=%s failed: %s",
                    worker.node_id, tag, e)
    except queue.Empty:
        return


def _coord_rank0_handle_new_send(worker: "ZmqShmKvConnectorBase",
                                 req_id: ReqId, req_meta: SendMeta) -> None:
    """Allocate a slot, broadcast the assignment to non-zero ranks, and
    stage rank 0's own shard inline on the main thread.

    The inline stage is deliberate: see the note in
    _coord_process_send_load. Broadcasting first (via the IPC-out queue,
    drained by the IPC thread every ~50ms) lets non-zero ranks begin
    their own inline stage in parallel with us."""
    assert worker.is_producer
    num_blocks = worker._blocks_token(req_meta.local_block_ids)
    logger.info(
        "TPUConnectorWorker(%s) rank0 --> handle_new_send req_id=%s uuid=%s "
        "blocks=%s", worker.node_id, req_id, req_meta.uuid, num_blocks)
    logger.info("PERF P handle_new_send req_id=%s uuid=%s ts=%.6f", req_id,
                req_meta.uuid, time.time())
    try:
        slot_idx = worker._coord_pool.acquire_slot(timeout=0.0)
    except Exception:
        # Pool full. Drop request; scheduler will surface a failure.
        with worker._coord_lock:
            send_holders = [
                f"uuid={u} req={e.req_id} "
                f"acked={e.pull_acked} "
                f"exp_in={max(0.0, e.expiration_time - time.perf_counter()):.1f}s"
                for u, e in worker._coord_send.items()
            ]
            recv_holders = [
                f"uuid={u} req={e.req_id}"
                for u, e in worker._coord_recv.items()
            ]
            worker._coord_done_sending.add(req_id)
        logger.error(
            "TPUConnectorWorker(%s) rank0 --> shm pool exhausted for "
            "req_id=%s (producer). send_holders=%s recv_holders=%s",
            worker.node_id, req_id, send_holders, recv_holders)
        return

    entry = _CoordSendEntry(
        req_id=req_id,
        slot_idx=slot_idx,
        num_blocks=num_blocks,
        expiration_time=req_meta.expiration_time,
        tp_size=worker.tp_size,
    )
    with worker._coord_lock:
        worker._coord_send[req_meta.uuid] = entry

    # Enqueue broadcast first so other ranks can start their own stage
    # while we run ours.
    worker._coord_broadcast(
        _IPC_STAGE_NOTIFY,
        (req_meta.uuid, slot_idx, num_blocks, req_meta.local_block_ids))
    logger.info(
        "TPUConnectorWorker(%d) rank0 --> handle_new_send broadcast uuid=%s "
        "slot=%d to %d non-zero ranks", worker.node_id, req_meta.uuid,
        slot_idx, len(worker._coord_rank_to_identity))

    # Rank 0 enqueues its own async D2H on the main thread; the waiter
    # thread emits STAGE_DONE once the future completes. Returning here
    # before the DMA finishes lets vLLM advance and the
    # STAGE_NOTIFY/scheduler/PULL-setup round-trip overlap with the
    # transfer.
    try:
        worker._coord_stage_shard(req_meta.uuid, slot_idx, num_blocks,
                                  req_meta.local_block_ids)
    except Exception as ex:
        logger.exception(
            "TPUConnectorWorker(%d) rank0 --> stage_self uuid=%s failed: %s",
            worker.node_id, req_meta.uuid, ex)


def _signal_stage_done(worker: "ZmqShmKvConnectorBase",
                       uuid: int,
                       failed: bool = False) -> None:
    """Mark this rank's STAGE_DONE on the rank-0 send entry. On rank 0
    this is a direct local update; on other ranks it goes through IPC
    to rank 0's listener, which performs the same update. If `failed`
    is True the entry is tagged stage_failed so PULL serving aborts
    cleanly with _MSG_ERR rather than shipping stale shm data."""
    if worker._is_host_coordinator:
        fired_complete = False
        staged_count = 0
        per_rank_event: Optional[threading.Event] = None
        with worker._coord_lock:
            entry = worker._coord_send.get(uuid)
            if entry is None:
                return
            if failed:
                entry.stage_failed = True
            entry.staged.add(0)  ## local_tp_rank?
            staged_count = len(entry.staged)
            if staged_count == worker.ranks_per_host:
                entry.stage_complete.set()
                fired_complete = True
            if 0 < len(entry.staged_events):  ## worker.local_tp_rank?
                per_rank_event = entry.staged_events[
                    0]  ## worker.local_tp_rank?
        # Set the per-rank event outside the lock; the wait side is
        # _coord_rank0_build_pull_response, which only takes the lock
        # briefly to look up the entry, then waits on the event.
        if per_rank_event is not None:
            per_rank_event.set()
        if fired_complete:
            logger.info(
                "TPUConnectorWorker(%d) rank0 --> stage_complete uuid=%s "
                "(all %d ranks staged, self-fire for node %d)", worker.node_id,
                uuid, worker.ranks_per_host, worker.node_id)
            logger.info("PERF P stage_complete(%s) uuid=%s ts=%.6f",
                        worker.node_id, uuid, time.time())
    else:
        worker._coord_send_ipc(_IPC_STAGE_DONE,
                               (uuid, worker.local_tp_rank, failed))


def _coord_rank0_handle_new_load(worker: "ZmqShmKvConnectorBase",
                                 req_id: ReqId, req_meta: LoadMeta) -> None:
    assert not worker.is_producer
    logger.info("PERF D handle_new_load req_id=%s uuid=%s ts=%.6f drain=%s",
                req_id, req_meta.uuid, time.time(), req_meta.remote_block_ids
                is None)
    if req_meta.remote_block_ids is None:
        # Either a full cache hit or a drain-for-completed-pull. Only
        # the former requires telling other ranks to skip; the drain
        # case means _worker_pending_load already has the real entry
        # from an earlier LOAD_NOTIFY broadcast.
        with worker._coord_lock:
            is_cache_hit = req_meta.uuid not in worker._coord_recv
        if is_cache_hit:
            worker._coord_broadcast(_IPC_LOAD_SKIP, (req_meta.uuid, ))
        return

    num_blocks = worker._blocks_token(req_meta.local_block_ids)
    # Preemption dedup: vLLM's scheduler re-emits update_state_after_alloc
    # for requests that were preempted and rescheduled. The LoadMeta
    # carries the same uuid (assigned by the producer) but new
    # local_block_ids. If we treat it as a fresh load we acquire a
    # second shm slot, submit a duplicate _coord_rank0_pull thread,
    # and orphan the first slot when we overwrite _coord_recv[uuid]
    # -- eventually exhausting the pool and hanging the scheduler.
    # Detect the re-emit by uuid and just update local_blocks in
    # place so the in-flight pull targets the new block destinations.
    with worker._coord_lock:
        existing = worker._coord_recv.get(req_meta.uuid)
        if existing is not None:
            pull_in_flight = not existing.load_complete.is_set()
            existing.local_blocks = req_meta.local_block_ids
            existing.num_blocks = num_blocks
            existing.req_id = req_id
            existing.remote_blocks = req_meta.remote_block_ids
    if existing is not None:
        if pull_in_flight:
            # LOAD_NOTIFY hasn't been broadcast yet; the pull thread
            # will read the updated local_blocks when it publishes.
            logger.warning(
                "TPUConnectorWorker(%s) rank0 --> dedup preempted load "
                "req_id=%s uuid=%s (pull in flight, new blocks=%s)",
                worker.node_id, req_id, req_meta.uuid, num_blocks)
        else:
            # Pull already completed and LOAD_NOTIFY fired with the
            # old blocks. Re-broadcast so workers' cached
            # _worker_pending_load reflects the new destinations
            # before they drain. There's a small race if a worker
            # already popped the old entry, but scatter under the
            # correct blocks is still the right intent.
            logger.warning(
                "TPUConnectorWorker(%s) rank0 --> dedup preempted load "
                "req_id=%s uuid=%s (pull already complete, "
                "rebroadcasting LOAD_NOTIFY with new blocks=%s)",
                worker.node_id, req_id, req_meta.uuid, num_blocks)
            worker._coord_broadcast(
                _IPC_LOAD_NOTIFY, (existing.uuid, existing.slot_idx,
                                   existing.num_blocks, existing.local_blocks))
        return

    try:
        slot_idx = worker._coord_pool.acquire_slot(timeout=0.0)
    except Exception:
        with worker._coord_lock:
            recv_holders = [
                f"uuid={u} req={e.req_id} "
                f"pull_ok={e.pull_ok} "
                f"load={e.load_complete.is_set()} "
                f"reported={e.reported_done} "
                f"copied={sorted(e.copied)}"
                for u, e in worker._coord_recv.items()
            ]
            send_holders = [
                f"uuid={u} req={e.req_id} "
                f"acked={e.pull_acked}" for u, e in worker._coord_send.items()
            ]
            worker._coord_done_recving.add(req_id)
        logger.error(
            "TPUConnectorWorker(%d) rank0 --> shm pool exhausted for "
            "req_id=%s; surfacing done_recving so scheduler advances "
            "(expect wrong output for this req). recv_holders=%s "
            "send_holders=%s", worker.node_id, req_id, recv_holders,
            send_holders)
        return

    remote_host, remote_port, remote_side_channel_port = worker._resolve_remote_host_port(
        req_meta)
    entry = _CoordRecvEntry(
        req_id=req_id,
        uuid=req_meta.uuid,
        slot_idx=slot_idx,
        num_blocks=num_blocks,
        local_blocks=req_meta.local_block_ids,
        remote_blocks=req_meta.remote_block_ids,
        remote_host=remote_host,
        remote_port=remote_port,
        remote_side_channel_port=remote_side_channel_port,
    )
    with worker._coord_lock:
        worker._coord_recv[req_meta.uuid] = entry

    worker._coord_executor.submit(worker._coord_rank0_pull, entry)


def _coord_rank0_register_copy(worker: "ZmqShmKvConnectorBase",
                               uuid: int,
                               rank_idx: int = 0) -> None:
    entry_for_notify = None
    with worker._coord_lock:
        entry = worker._coord_recv.get(uuid)
        if entry is None:
            return
        entry.copied.add(rank_idx)
        num_hosts = int(os.getenv("TPU_NUM_HOSTS", "1"))
        ranks_per_host = worker.tp_size // num_hosts
        if len(entry.copied) == ranks_per_host:
            entry_for_notify = worker._coord_recv.pop(uuid, None)
    if entry_for_notify is not None:
        # Notify remote P and release slot.
        dummy = LoadMeta(
            uuid=entry_for_notify.uuid,
            local_block_ids=entry_for_notify.local_blocks,
            remote_block_ids=entry_for_notify.remote_blocks,
            remote_host=entry_for_notify.remote_host,
            remote_port=entry_for_notify.remote_port,
            remote_side_channel_port=entry_for_notify.remote_side_channel_port,
        )
        try:
            worker._coord_rank0_send_notify(dummy, uuid)
        finally:
            worker._coord_pool.release_slot(entry_for_notify.slot_idx)
            logger.info("PERF D scatter_complete req_id=%s uuid=%s ts=%.6f",
                        entry_for_notify.req_id, uuid, time.time())


def _coord_worker_wait_load(
        worker: "ZmqShmKvConnectorBase",
        uuid: int,
        timeout: float = 1.0) -> Optional[tuple[int, int, list[int]]]:
    """Block until LOAD_NOTIFY or LOAD_SKIP for uuid arrives, or timeout
    expires. Returns (slot_idx, num_blocks, local_blocks) for a scatter,
    or None for skip/timeout. The dict stores None as a sentinel set
    by LOAD_SKIP, so we can tell "skip" apart from "not there yet" by
    presence."""
    deadline = time.perf_counter() + timeout
    with worker._worker_pending_load_cv:
        while uuid not in worker._worker_pending_load:
            rem = deadline - time.perf_counter()
            if rem <= 0:
                return None
            worker._worker_pending_load_cv.wait(timeout=rem)
        info = worker._worker_pending_load.pop(uuid)
    return info


def _coord_rank0_ipc_loop(worker: "ZmqShmKvConnectorBase") -> None:
    sock = worker._coord_ipc_sock
    while not worker._stop_event.is_set():
        # Drain outbound sends from the queue first so notifications
        # don't starve behind inbound polling.
        worker._coord_ipc_drain_outbound()
        try:
            # Short poll so outbound drains stay responsive. ZMQ will
            # buffer in-flight messages while we're away.
            if sock.poll(timeout=50) == 0:
                continue
            frames = sock.recv_multipart()
        except zmq.ContextTerminated:
            return
        except zmq.ZMQError as e:
            logger.warning(
                "TPUConnectorWorker(%d) rank0 --> IPC recv error: %s",
                worker.node_id, e)
            continue
        # ROUTER gives us [identity, tag, payload].
        if len(frames) != 3:
            logger.warning(
                "TPUConnectorWorker(%d) rank0 --> IPC malformed frames n=%d",
                worker.node_id, len(frames))
            continue
        identity, tag, payload = frames
        try:
            obj = _secure_loads(payload)
        except Exception as e:
            logger.warning(
                "TPUConnectorWorker(%d) rank0 --> bad IPC "
                "payload tag=%s: %s", worker.node_id, tag, e)
            continue
        if tag == _IPC_HELLO:
            local_rank, global_rank = obj
            with worker._coord_lock:
                worker._coord_rank_to_identity[local_rank] = identity
                worker._coord_local_to_global_rank[local_rank] = global_rank
                # All non-zero ranks registered? Unblock new-req
                # processing on rank 0.
                if len(worker._coord_rank_to_identity
                       ) == worker.ranks_per_host - 1:
                    worker._coord_workers_ready.set()
            logger.info(
                "TPUConnectorWorker(%d) rank0 --> HELLO from local_rank=%d global_rank=%d "
                "(%d/%d registered)", worker.node_id, local_rank, global_rank,
                len(worker._coord_rank_to_identity), worker.ranks_per_host - 1)
        elif tag == _IPC_STAGE_DONE:
            # Payload is (uuid, rank, failed). All workers run from
            # the same code so we don't need a length-tolerant unpack.
            uuid, rank, failed = obj
            fired_complete = False
            staged_count = 0
            per_rank_event: Optional[threading.Event] = None
            with worker._coord_lock:
                entry = worker._coord_send.get(uuid)
                if entry is None:
                    continue
                if failed:
                    entry.stage_failed = True
                entry.staged.add(rank)
                staged_count = len(entry.staged)
                if staged_count == worker.ranks_per_host:
                    entry.stage_complete.set()
                    fired_complete = True
                if 0 <= rank < len(entry.staged_events):
                    per_rank_event = entry.staged_events[rank]
            # Fire the per-rank readiness so any PULL channel waiting
            # only on this rank's shard can proceed without waiting on
            # the slowest rank in the request.
            if per_rank_event is not None:
                per_rank_event.set()
            if fired_complete:
                logger.info(
                    "TPUConnectorWorker(%d) rank0 --> stage_complete uuid=%s "
                    "(all %d ranks staged)", worker.node_id, uuid,
                    worker.ranks_per_host)
                logger.info("PERF P stage_complete uuid=%s ts=%.6f", uuid,
                            time.time())
            else:
                logger.info(
                    "TPUConnectorWorker(%d) rank0 --> STAGE_DONE uuid=%s "
                    "rank=%d (%d/%d)", worker.node_id, uuid, rank,
                    staged_count, worker.ranks_per_host)
        elif tag == _IPC_COPY_DONE:
            uuid, rank = obj
            entry_for_notify = None
            with worker._coord_lock:
                entry = worker._coord_recv.get(uuid)
                if entry is None:
                    continue
                entry.copied.add(rank)
                if len(entry.copied) == worker.ranks_per_host:
                    entry_for_notify = worker._coord_recv.pop(uuid, None)
            if entry_for_notify is not None:
                # Synthesize a LoadMeta-like object for the notify path.
                dummy = LoadMeta(
                    uuid=entry_for_notify.uuid,
                    local_block_ids=entry_for_notify.local_blocks,
                    remote_block_ids=entry_for_notify.remote_blocks,
                    remote_host=entry_for_notify.remote_host,
                    remote_port=entry_for_notify.remote_port,
                    remote_side_channel_port=entry_for_notify.
                    remote_side_channel_port,
                )
                try:
                    worker._coord_rank0_send_notify(dummy,
                                                    entry_for_notify.uuid)
                finally:
                    worker._coord_pool.release_slot(entry_for_notify.slot_idx)
                    logger.info(
                        "PERF D scatter_complete req_id=%s uuid=%s ts=%.6f",
                        entry_for_notify.req_id, uuid, time.time())
        else:
            logger.warning(
                "TPUConnectorWorker(%d) rank0 --> unknown IPC tag=%s",
                worker.node_id, tag)


def _coord_worker_ipc_loop(worker: "ZmqShmKvConnectorBase") -> None:
    sock = worker._coord_ipc_sock
    while not worker._stop_event.is_set():
        worker._coord_ipc_drain_outbound()
        try:
            if sock.poll(timeout=50) == 0:
                continue
            frames = sock.recv_multipart()
        except zmq.ContextTerminated:
            return
        except zmq.ZMQError as e:
            logger.warning(
                "TPUConnectorWorker(%d) rank%d --> IPC recv error: %s",
                worker.node_id, worker.tp_rank, e)
            continue
        # DEALER gives us [tag, payload].
        if len(frames) != 2:
            logger.warning(
                "TPUConnectorWorker(%d) rank%d --> IPC malformed frames n=%d",
                worker.node_id, worker.tp_rank, len(frames))
            continue
        tag, payload = frames
        try:
            obj = _secure_loads(payload)
        except Exception as e:
            logger.warning(
                "TPUConnectorWorker(%d) rank%d --> bad IPC payload tag=%s: %s",
                worker.node_id, worker.tp_rank, tag, e)
            continue
        if tag == _IPC_STAGE_NOTIFY:
            uuid, slot_idx, num_blocks, block_ids = obj
            logger.info(
                "TPUConnectorWorker(%s) rank%d --> STAGE_NOTIFY received "
                "uuid=%s slot=%d blocks=%s", worker.node_id, worker.tp_rank,
                uuid, slot_idx, num_blocks)
            # Hand off to the main thread (waiting in
            # _coord_worker_stage_inline); don't stage in an executor
            # here, otherwise .to("cpu") races with model_forward.
            with worker._worker_pending_stage_cv:
                worker._worker_pending_stage[uuid] = (slot_idx, num_blocks,
                                                      block_ids)
                worker._worker_pending_stage_cv.notify_all()
        elif tag == _IPC_LOAD_NOTIFY:
            if len(obj) == 5:
                uuid, _req_id, slot_idx, num_blocks, local_blocks = obj
            else:
                # Backward-compatible decode for any in-flight old control
                # messages during rolling restarts.
                uuid, slot_idx, num_blocks, local_blocks = obj
            with worker._worker_pending_load_cv:
                worker._worker_pending_load[uuid] = (slot_idx, num_blocks,
                                                     local_blocks)
                worker._worker_pending_load_cv.notify_all()
        elif tag == _IPC_LOAD_SKIP:
            (uuid, ) = obj
            # None marks "coord said no scatter needed for this uuid"
            # (cache hit). _coord_worker_wait_load surfaces None
            # immediately without the 200ms-ish timeout wait.
            with worker._worker_pending_load_cv:
                worker._worker_pending_load[uuid] = None
                worker._worker_pending_load_cv.notify_all()
        elif tag == _IPC_DROP:
            (uuid, ) = obj
            with worker._worker_pending_load_cv:
                worker._worker_pending_load[uuid] = None
                worker._worker_pending_load_cv.notify_all()
            with worker._worker_pending_stage_cv:
                worker._worker_pending_stage.pop(uuid, None)
                worker._worker_pending_stage_cv.notify_all()
        else:
            logger.warning(
                "TPUConnectorWorker(%d) rank%d --> unknown IPC tag=%s",
                worker.node_id, worker.tp_rank, tag)


def _coord_rank0_expire_loop(worker: "ZmqShmKvConnectorBase") -> None:
    while not worker._stop_event.is_set():
        if worker._stop_event.wait(timeout=1.0):
            return
        now = time.perf_counter()
        expired: list[int] = []
        with worker._coord_lock:
            for uuid, entry in list(worker._coord_send.items()):
                # Acked entries are handled in get_finished() so they
                # surface to the scheduler in the correct tick.
                if not entry.pull_acked and now > entry.expiration_time:
                    expired.append(uuid)
        for uuid in expired:
            logger.warning(
                "TPUConnectorWorker(%d) rank0 --> expire send uuid=%s",
                worker.node_id, uuid)
            with worker._coord_lock:
                entry = worker._coord_send.pop(uuid, None)
                if entry is None:
                    continue
                worker._coord_done_sending.add(entry.req_id)
            worker._coord_pool.release_slot(entry.slot_idx)


def _coord_get_finished(
        worker: "ZmqShmKvConnectorBase") -> tuple[set[str], set[str]]:
    # Only rank 0 returns non-empty; scheduler unions across workers.
    if not worker._is_host_coordinator:
        return set(), set()
    done_sending: set[str] = set()
    done_recving: set[str] = set()
    released_slots: list[int] = []
    with worker._coord_lock:
        # Producer: pull_acked → done_sending; scheduler unioning means
        # we only need to report once.
        for uuid in list(worker._coord_send.keys()):
            entry = worker._coord_send[uuid]
            if entry.pull_acked:
                done_sending.add(entry.req_id)
                del worker._coord_send[uuid]
                released_slots.append(entry.slot_idx)
        # Merge in any expiration-sweeper results (already removed from
        # _coord_send by that thread, slot already released).
        if worker._coord_done_sending:
            done_sending |= worker._coord_done_sending
            worker._coord_done_sending.clear()
        # Consumer: pull_ok + load_complete -> done_recving. Mark
        # `reported_done` so repeated get_finished() calls don't emit
        # the same req id twice; the entry stays in _coord_recv until
        # all ranks have acknowledged their scatter via COPY_DONE.
        newly_done_recving: list[tuple[str, int]] = []
        for uuid, entry in list(worker._coord_recv.items()):
            if (entry.load_complete.is_set() and entry.pull_ok
                    and not entry.reported_done):
                done_recving.add(entry.req_id)
                entry.reported_done = True
                newly_done_recving.append((entry.req_id, uuid))
        # Merge in failure-surfacing reqs (pull failed, shm pool
        # exhausted on admit). The scheduler advances these as if
        # the KV were ready; the output will be wrong for that req,
        # but the alternative is an engine-wide hang.
        if worker._coord_done_recving:
            done_recving |= worker._coord_done_recving
            worker._coord_done_recving.clear()
    # Free slots outside the lock to avoid nesting lock waits.
    for sidx in released_slots:
        worker._coord_pool.release_slot(sidx)
    for req_id, uuid in newly_done_recving:
        logger.info("PERF D done_recving req_id=%s uuid=%s ts=%.6f", req_id,
                    uuid, time.time())
    if done_sending:
        logger.info("TPUConnectorWorker(%d) rank0 --> done_sending=%s",
                    worker.node_id, done_sending)
    if done_recving:
        logger.info("TPUConnectorWorker(%d) rank0 --> done_recving=%s",
                    worker.node_id, done_recving)
    return done_sending, done_recving


def _try_remove_ipc_endpoint(ipc_path: str) -> None:
    """ZMQ's ipc:// backs onto a filesystem path; a stale file from a
    previous crashed rank-0 makes bind() fail. Remove it best-effort."""
    import os as _os
    if not ipc_path.startswith("ipc://"):
        return
    path = ipc_path[len("ipc://"):]
    try:
        _os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("Could not clean up stale IPC endpoint %s: %s", path, e)


def _coord_setup_rank0(worker: "ZmqShmKvConnectorBase", ipc_path: str) -> None:
    # IPC ROUTER: workers DEAL in.
    ipc = worker.zmq_cxt.socket(zmq.ROUTER)
    ipc.setsockopt(zmq.LINGER, 0)
    # Remove any stale endpoint from a prior crashed run.
    _try_remove_ipc_endpoint(ipc_path)
    ipc.bind(ipc_path)
    worker._coord_ipc_sock = ipc
    logger.info("TPUConnectorWorker(%s) rank0 --> IPC listening on %s",
                worker.node_id, ipc_path)

    # IPC listener thread.
    t_ipc = threading.Thread(target=worker._coord_rank0_ipc_loop,
                             name="tpu_conn_ipc",
                             daemon=True)
    t_ipc.start()
    worker._coord_threads.append(t_ipc)

    # Stage waiter: drains the async-D2H queue and signals STAGE_DONE
    # once each future completes.
    t_stage_wait = threading.Thread(target=worker._coord_stage_waiter_loop,
                                    name="tpu_conn_stage_wait",
                                    daemon=True)
    t_stage_wait.start()
    worker._coord_threads.append(t_stage_wait)

    # Expiration sweeper for pending sends.
    t_exp = threading.Thread(target=worker._coord_rank0_expire_loop,
                             name="tpu_conn_expire",
                             daemon=True)
    t_exp.start()
    worker._coord_threads.append(t_exp)

    # External endpoints are producer-only.
    if worker.is_producer:
        for ch in range(worker._n_channels):
            t_data = threading.Thread(
                target=worker._coord_rank0_external_data_loop,
                args=(ch, ),
                name=f"tpu_conn_data_ch{ch}",
                daemon=True)
            t_data.start()
            worker._coord_threads.append(t_data)

        t_notif = threading.Thread(
            target=worker._coord_rank0_external_notif_loop,
            name="tpu_conn_notif",
            daemon=True)
        t_notif.start()
        worker._coord_threads.append(t_notif)
    else:
        # Consumer-side: outbound notify-done DEALERs are shared across
        # requests; guard the dict (and each send) behind a lock since
        # it's touched by the main-thread drain path, the IPC listener
        # thread, and async pull threads. Pull sockets are opened per-
        # request, so no caching needed for them.
        worker._coord_notif_sockets: dict[str, zmq.Socket] = {}
        worker._coord_sockets_lock = threading.Lock()


def _coord_setup_worker(worker: "ZmqShmKvConnectorBase",
                        ipc_path: str) -> None:
    sock = worker.zmq_cxt.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    # Identity encodes rank so the coordinator can address us explicitly.
    ident = f"tpu_rank_{worker.local_tp_rank}".encode("utf-8")
    sock.setsockopt(zmq.IDENTITY, ident)
    sock.connect(ipc_path)
    worker._coord_ipc_sock = sock
    # Announce ourselves; messages buffer on DEALER until coord ROUTER
    # reads them, so this is safe even if coord isn't up yet.
    worker._coord_send_ipc(_IPC_HELLO, (worker.local_tp_rank, worker.tp_rank))
    logger.info(
        "TPUConnectorWorker(%s) rank%d, local_rank%d --> IPC connected to %s",
        worker.node_id, worker.tp_rank, worker.local_tp_rank, ipc_path)

    t = threading.Thread(target=worker._coord_worker_ipc_loop,
                         name=f"tpu_conn_worker_ipc_{worker.tp_rank}",
                         daemon=True)
    t.start()
    worker._coord_threads.append(t)

    # Stage waiter: same role as on rank 0; sends STAGE_DONE via IPC
    # once the per-rank D2H future completes.
    t_stage_wait = threading.Thread(
        target=worker._coord_stage_waiter_loop,
        name=f"tpu_conn_stage_wait_{worker.tp_rank}",
        daemon=True)
    t_stage_wait.start()
    worker._coord_threads.append(t_stage_wait)


def _coord_rank0_pull(worker: "ZmqShmKvConnectorBase",
                      entry: _CoordRecvEntry) -> None:
    """Pull the full KV blob (all ranks' shards) from remote coord for all hosts
    and unpack it into shm. Then broadcast LOAD_NOTIFY and set the
    load_complete event so get_finished() returns done_recving.

    Uses ``worker._n_channels`` parallel DEALER sockets (one per thread)
    to N producer ROUTER ports ``base_port + ch``. Each channel carries
    its subset of rank payloads; we reassemble them into a single
    rank->payload map then unpack all shards into shm."""
    hosts = entry.remote_host if isinstance(entry.remote_host,
                                            list) else [entry.remote_host]
    ports = entry.remote_port if isinstance(entry.remote_port,
                                            list) else [entry.remote_port]
    num_hosts = len(hosts)
    n_channels = worker._n_channels
    uuid_bytes = str(entry.uuid).encode("utf-8")
    timeout_s = dist_utils.get_p2p_wait_pull_timeout()
    req_blocks_pickle = _secure_dumps(entry.remote_blocks)
    logger.info(
        "TPUConnectorWorker(%s) rank0 --> PULL req_id=%s uuid=%s "
        "from=%s:%s+0..%s blocks=%s n_channels=%s", worker.node_id,
        entry.req_id, entry.uuid, entry.remote_host, ports, n_channels - 1,
        entry.num_blocks, n_channels)

    num_layers = worker.num_layers

    def _pull_channel(h: int, ch: int):
        """Pull one channel's subset of ranks AND unpack their layer
        frames into shm inline. Returns per-channel (wire_ms,
        unpack_ms, ranks_handled) so the orchestrator can log the
        breakdown but does not need to reassemble anything."""
        port = ports[h] + ch
        host = hosts[h]
        sock_path = make_zmq_path("tcp", host, port)
        sock = make_zmq_socket(ctx=worker.zmq_cxt,
                               path=sock_path,
                               socket_type=zmq.DEALER,
                               bind=False,
                               linger=0)
        timeout_ms = max(1, int(timeout_s * 1000))
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        try:
            t0 = time.perf_counter()
            logger.info("PERF D pull_send host_idx=%d ch=%d uuid=%s ts=%.6f",
                        h, ch, entry.uuid, time.time())
            sock.send_multipart([_MSG_PULL, uuid_bytes, req_blocks_pickle])
            deadline = t0 + timeout_s
            frames = None
            while time.perf_counter() < deadline:
                remaining_ms = max(
                    1, int((deadline - time.perf_counter()) * 1000))
                if sock.poll(timeout=min(remaining_ms, 1000)) != 0:
                    # copy=False returns zmq.Frame objects whose
                    # .buffer is a readonly memoryview into the ZMQ
                    # recv buffer -- one less 256MB copy per layer.
                    frames = sock.recv_multipart(copy=False)
                    break
            if frames is None:
                raise TimeoutError(
                    f"PULL host_idx={h} ch={ch} timed out after {timeout_s}s "
                    f"for req_id={entry.req_id}")
            if bytes(frames[0].buffer) != _MSG_OK:
                tag = bytes(frames[0].buffer)
                payload_hint = (bytes(frames[1].buffer)
                                if len(frames) > 1 else b"")
                raise RuntimeError(
                    f"PULL host_idx={h} ch={ch} failed for req_id={entry.req_id}: "
                    f"[{tag!r}, {payload_hint!r}]")
            t_wire_done = time.perf_counter()
            logger.info("PERF D wire_done host_idx=%d ch=%d uuid=%s ts=%.6f",
                        h, ch, entry.uuid, time.time())

            # Wire format after OK:
            #   frames[1]=uuid echo, frames[2]=n_ranks ascii
            #   ch 0: frames[3]=header, then per-rank blocks
            #   ch >0: per-rank blocks start at frames[3]
            # Per-rank block = (rank_idx, layer0, layer1, ..., layer_{L-1})
            n_ranks = int(bytes(frames[2].buffer).decode("utf-8"))
            idx = 3
            header_frame: Optional[zmq.Frame] = None
            if ch == 0:
                header_frame = frames[idx]
                idx += 1

            ranks_handled: list[int] = []
            # Construct reverse mapping global_rank -> local_rank
            with worker._coord_lock:
                global_to_local = {
                    global_rank: local_rank
                    for local_rank, global_rank in
                    worker._coord_local_to_global_rank.items()
                }

            for _ in range(n_ranks):
                global_rank = int(bytes(frames[idx].buffer).decode("utf-8"))
                idx += 1
                layer_buffers = [
                    frames[idx + li].buffer for li in range(num_layers)
                ]
                idx += num_layers

                if global_rank in global_to_local:
                    local_rank = global_to_local[global_rank]
                    worker._coord_pool.unpack_rank_layers(
                        entry.slot_idx, local_rank, entry.num_blocks,
                        layer_buffers)
                    ranks_handled.append(global_rank)
            t_unpack_done = time.perf_counter()
            wire_ms = (t_wire_done - t0) * 1000.0
            unpack_ms = (t_unpack_done - t_wire_done) * 1000.0
            return h, ch, wire_ms, unpack_ms, ranks_handled, header_frame
        finally:
            sock.close(linger=0)

    t_total0 = time.perf_counter()
    try:
        futures = [
            worker._coord_channel_executor.submit(_pull_channel, h, ch)
            for h in range(num_hosts) for ch in range(n_channels)
        ]
        # Annotation was stale: _pull_channel returns 6 values
        # (h, ch, wire_ms, unpack_ms, ranks_handled, header_frame), not 5.
        results: list[tuple[int, int, float, float, list[int],
                            Optional[zmq.Frame]]] = []
        first_error: Optional[BaseException] = None
        for fut in futures:
            try:
                results.append(fut.result())
            except BaseException as e:
                if first_error is None:
                    first_error = e
        # All channel futures have returned or timed out before we mark
        # the request failed and release its shm slot. Otherwise a late
        # successful channel from a failed pull can unpack into a slot
        # already reused by a later request.
        if first_error is not None:
            raise first_error
        t_done = time.perf_counter()

        # Sanity-check all local ranks were delivered exactly once.
        seen: set[int] = set()
        header_frame: Optional[zmq.Frame] = None
        for _h, _ch, _w, _u, ranks, hdr in results:
            if hdr is not None:
                header_frame = hdr
            for r in ranks:
                if r in seen:
                    raise RuntimeError(f"PULL req_id={entry.req_id}: rank {r} "
                                       f"delivered by multiple channels")
                seen.add(r)
        if header_frame is None:
            raise RuntimeError(
                f"PULL req_id={entry.req_id}: channel 0 omitted header")
        if len(seen) != worker.ranks_per_host:
            raise RuntimeError(
                f"PULL req_id={entry.req_id}: got {len(seen)} ranks, "
                f"expected {worker.ranks_per_host}")

        total_ms = (t_done - t_total0) * 1000.0
        # With inline unpack per channel, wire+unpack are already
        # overlapped; the outer total is max over channels of
        # (wire+unpack). Report them separately for continuity
        # with the existing latency tracker.
        max_wire = max(w for _h, _c, w, _u, _r, _hdr in results)
        max_unpack = max(u for _h, _c, _w, u, _r, _hdr in results)
        worker._lat.record(_LAT_WIRE, max_wire)
        worker._lat.record(_LAT_UNPACK, max_unpack)
        worker.transfer_stats.record_kv_pull(total_ms)
        per_ch_ms = ",".join(f"{w + u:.1f}"
                             for _h, _c, w, u, _r, _hdr in results)
        logger.info(
            "TPUConnectorWorker(%d) rank0 --> done PULL req_id=%s uuid=%s "
            "total=%.2fms max_wire=%.2fms max_unpack=%.2fms "
            "per_ch_ms=[%s]", worker.node_id, entry.req_id, entry.uuid,
            total_ms, max_wire, max_unpack, per_ch_ms)
        entry.pull_ok = True
    except Exception as e:
        logger.exception(
            "TPUConnectorWorker(%d) rank0 --> PULL req_id=%s failed: %s",
            worker.node_id, entry.req_id, e)
        entry.pull_ok = False
        worker.transfer_stats.record_failed_transfer()
    finally:
        # Tell workers the slot is ready (or drop it on failure).
        if entry.pull_ok:
            worker._coord_broadcast(_IPC_LOAD_NOTIFY,
                                    (entry.uuid, entry.slot_idx,
                                     entry.num_blocks, entry.local_blocks))
        else:
            worker._coord_broadcast(_IPC_DROP, (entry.uuid, ))
            # Clean up: failed pulls will never reach COPY_DONE, so
            # free the slot and drop bookkeeping now. Also surface
            # the req_id as done_recving -- otherwise the vLLM
            # scheduler waits forever for completion of a req whose
            # entry we just silently removed.
            with worker._coord_lock:
                worker._coord_recv.pop(entry.uuid, None)
                worker._coord_done_recving.add(entry.req_id)
            worker._coord_pool.release_slot(entry.slot_idx)
        entry.load_complete.set()
        logger.info(
            "PERF D pull_complete req_id=%s uuid=%s ts=%.6f pull_ok=%s",
            entry.req_id, entry.uuid, time.time(), entry.pull_ok)


def _coord_rank0_send_notify(worker: "ZmqShmKvConnectorBase",
                             req_meta: LoadMeta, uuid: int) -> None:
    remote_hosts = req_meta.remote_host
    if not isinstance(remote_hosts, list):
        remote_hosts = [remote_hosts]

    # Determine the base port for the target DP group (without host/node offset)
    if req_meta.remote_side_channel_port:
        target_side_port_base = req_meta.remote_side_channel_port
    else:
        # Fallback to current DP group's base port
        target_side_port_base = int(
            dist_utils.get_side_channel_port()) + worker.dp_rank

    for h_idx, r_host in enumerate(remote_hosts):
        side_port = target_side_port_base + h_idx
        sock_path = make_zmq_path("tcp", r_host, side_port)
        # DEALER is not thread-safe; serialize create+send under the lock.
        with worker._coord_sockets_lock:
            sock = worker._coord_notif_sockets.get(sock_path)
            new_sock = sock is None
            if new_sock:
                sock = make_zmq_socket(ctx=worker.zmq_cxt,
                                       path=sock_path,
                                       socket_type=zmq.DEALER,
                                       bind=False)
                worker._coord_notif_sockets[sock_path] = sock
            sock.send_string(str(uuid))
        if new_sock:
            logger.info(
                "TPUConnectorWorker(%d) rank0 --> notify channel to %s",
                worker.node_id, sock_path)
        logger.info(
            "TPUConnectorWorker(%d) rank0 --> notify pull-done uuid=%s to %s",
            worker.node_id, uuid, r_host)


def _coord_rank0_external_data_loop(worker: "ZmqShmKvConnectorBase",
                                    channel_idx: int) -> None:
    """One data server per TCP channel. channel_idx in [0, n_channels).

    A rank's payload rides channel ``rank % n_channels``. Channel 0 also
    prepends the pickled header so the consumer can derive tp_size /
    shapes before reassembling. Each channel binds its own port
    ``base + channel_idx`` and serves its subset of ranks only, giving
    N parallel TCP streams end-to-end."""
    port = worker._kv_transfer_ports[channel_idx]
    sock_path = make_zmq_path("tcp", "*", port)
    sock = make_zmq_socket(ctx=worker.zmq_cxt,
                           path=sock_path,
                           socket_type=zmq.ROUTER,
                           bind=True)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.ROUTER_MANDATORY, 0)
    sock.setsockopt(zmq.SNDTIMEO,
                    max(1,
                        dist_utils.get_p2p_wait_pull_timeout() * 1000))
    ranks_on_channel = [
        r for r in range(worker.ranks_per_host)
        if r % worker._n_channels == channel_idx
    ]
    logger.info(
        "TPUConnectorWorker(%d) rank0 --> data server ch=%d listening on "
        "tcp://%s:%s serving ranks=%s", worker.node_id, channel_idx,
        worker.host_ip, port, ranks_on_channel)

    ready_responses: queue.Queue = queue.Queue()

    def _prepare_response(client_id: bytes, uuid: int,
                          uuid_bytes: bytes) -> None:
        try:
            response = worker._coord_rank0_build_pull_response(
                uuid, uuid_bytes, channel_idx, ranks_on_channel)
        except Exception:
            logger.exception(
                "TPUConnectorWorker(%d) rank0 --> build_pull_response "
                "ch=%d uuid=%s failed", worker.node_id, channel_idx, uuid)
            response = None
        ready_responses.put((client_id, uuid, uuid_bytes, response))

    def _send_ready_responses() -> None:
        while True:
            try:
                client_id, uuid, uuid_bytes, response = (
                    ready_responses.get_nowait())
            except queue.Empty:
                return
            if response is None:
                try:
                    sock.send_multipart([client_id, _MSG_ERR, uuid_bytes])
                except zmq.ZMQError as e:
                    logger.warning(
                        "TPUConnectorWorker(%d) rank0 --> data send ERR "
                        "ch=%d uuid=%s failed: %s", worker.node_id,
                        channel_idx, uuid, e)
                continue
            # copy=False lets ZMQ send the big per-layer memoryviews
            # directly from shm via scatter-gather sendmsg, no userland
            # copy. Small framing frames pay a tiny per-frame overhead;
            # net win is ~256MB of memcpy avoided per rank.
            _t_send_start = time.time()
            logger.info("PERF P send_start ch=%d uuid=%s ts=%.6f", channel_idx,
                        uuid, _t_send_start)
            try:
                sock.send_multipart([client_id, *response], copy=False)
            except zmq.ZMQError as e:
                logger.warning(
                    "TPUConnectorWorker(%d) rank0 --> data send ch=%d "
                    "uuid=%s failed: %s", worker.node_id, channel_idx, uuid, e)
            logger.info("PERF P send_done ch=%d uuid=%s ts=%.6f dur_ms=%.2f",
                        channel_idx, uuid, time.time(),
                        (time.time() - _t_send_start) * 1000.0)

    while not worker._stop_event.is_set():
        _send_ready_responses()
        try:
            if sock.poll(timeout=50) == 0:
                continue
            frames = sock.recv_multipart()
        except zmq.ContextTerminated:
            return
        except zmq.ZMQError as e:
            logger.warning(
                "TPUConnectorWorker(%d) rank0 --> data recv "
                "ch=%d error: %s", worker.node_id, channel_idx, e)
            continue
        # Expected: [client_id, PULL, uuid_bytes, remote_blocks_pickle]
        if len(frames) < 3 or frames[1] != _MSG_PULL:
            logger.warning(
                "TPUConnectorWorker(%d) rank0 --> data malformed "
                "request ch=%d n=%d", worker.node_id, channel_idx, len(frames))
            continue
        client_id = frames[0]
        uuid_bytes = frames[2]
        try:
            uuid = int(uuid_bytes.decode("utf-8"))
        except ValueError:
            try:
                sock.send_multipart([client_id, _MSG_ERR, b"bad-uuid"])
            except zmq.ZMQError as e:
                logger.warning(
                    "TPUConnectorWorker(%d) rank0 --> data send bad-uuid "
                    "ch=%d failed: %s", worker.node_id, channel_idx, e)
            continue
        logger.info(
            "TPUConnectorWorker(%d) rank0 --> PULL received ch=%d uuid=%s",
            worker.node_id, channel_idx, uuid)
        logger.info("PERF P pull_recv ch=%d uuid=%s ts=%.6f", channel_idx,
                    uuid, time.time())
        worker._coord_channel_executor.submit(_prepare_response, client_id,
                                              uuid, uuid_bytes)


def _coord_rank0_build_pull_response(
        worker: "ZmqShmKvConnectorBase", uuid: int, uuid_bytes: bytes,
        channel_idx: int,
        ranks_on_channel: list[int]) -> Optional[list[bytes]]:
    # A PULL can arrive before the scheduler has routed reqs_to_send
    # through process_send_load (the D-side orchestrator may dispatch
    # the pull immediately after handing D the uuid, racing P's own
    # metadata tick). Wait up to the p2p timeout for the entry to
    # appear, then wait up to the remaining budget for staging. With
    # n_channels parallel PULLs the race window is n_channels-fold more
    # likely to fire, so this backoff is required for correctness.
    timeout = dist_utils.get_p2p_wait_pull_timeout()
    deadline = time.perf_counter() + timeout
    entry = None
    while time.perf_counter() < deadline:
        with worker._coord_lock:
            entry = worker._coord_send.get(uuid)
            if entry is not None:
                # A live PULL means the slot is actively being served.
                # Do not let the expiration sweeper release it while this
                # channel is waiting for staging or queued to send.
                entry.expiration_time = max(entry.expiration_time,
                                            time.perf_counter() + timeout)
        if entry is not None:
            break
        time.sleep(0.05)
    if entry is None:
        logger.warning(
            "TPUConnectorWorker(%d) rank0 --> PULL ch=%d unknown uuid=%s "
            "(timed out waiting for register-send)", worker.node_id,
            channel_idx, uuid)
        return None
    # Wait only for the ranks this channel actually serves. Each
    # channel ch carries shards for ranks where r % n_channels == ch,
    # so a slow rank only delays its own channel rather than blocking
    # all `n_channels` PULLs on the global stage_complete event. With
    # n_channels=tp_size (the default) this means each channel waits
    # for exactly one rank.
    for r in ranks_on_channel:
        rem = max(0.0, deadline - time.perf_counter())
        if r >= len(entry.staged_events):
            # Defensive: should never happen because tp_size sizes the
            # vector at construction. Fall back to global wait.
            if not entry.stage_complete.wait(timeout=rem):
                logger.warning(
                    "TPUConnectorWorker(%d) rank0 --> PULL ch=%d uuid=%s "
                    "timed out waiting for staging (have %d of %d)",
                    worker.node_id, channel_idx, uuid, len(entry.staged),
                    worker.ranks_per_host)
                return None
            break
        if not entry.staged_events[r].wait(timeout=rem):
            logger.warning(
                "TPUConnectorWorker(%d) rank0 --> PULL ch=%d uuid=%s timed "
                "out waiting for rank %d to stage (have %d of %d)",
                worker.node_id, channel_idx, uuid, r, len(entry.staged),
                worker.ranks_per_host)
            return None
    with worker._coord_lock:
        entry.expiration_time = max(entry.expiration_time,
                                    time.perf_counter() + timeout)
        stage_failed = entry.stage_failed
    if stage_failed:
        # Some rank's stage waiter aborted (timeout or wait() raised).
        # Returning None routes _send_ready_responses to _MSG_ERR so
        # the consumer drops the request instead of receiving stale
        # shm bytes.
        logger.warning(
            "TPUConnectorWorker(%d) rank0 --> PULL ch=%d uuid=%s "
            "stage_failed; sending ERR", worker.node_id, channel_idx, uuid)
        return None
    # Wire format (per channel, after the ROUTER envelope):
    #   ch 0:  [OK, uuid, n_ranks, header,
    #            (rank_idx, layer_mv_0, ..., layer_mv_{L-1})*]
    #   ch >0: [OK, uuid, n_ranks,
    #            (rank_idx, layer_mv_0, ..., layer_mv_{L-1})*]
    # Each layer_mv is a memoryview straight into the shm slot for
    # that (rank, layer); there is no bytearray/bytes staging on
    # the producer. Consumer knows num_layers from worker.num_layers
    # so it doesn't need to be inline; the header on ch 0 carries
    # the sanity-check metadata.
    frames: list = [
        _MSG_OK, uuid_bytes,
        str(len(ranks_on_channel)).encode("utf-8")
    ]
    if channel_idx == 0:
        header = worker._pull_response_header(entry)
        frames.append(_secure_dumps(header))
    total_bytes = 0
    for r in ranks_on_channel:
        global_rank = worker._coord_local_to_global_rank[r]
        frames.append(str(global_rank).encode("utf-8"))
        layer_views = worker._coord_pool.rank_layer_views(
            entry.slot_idx, r, entry.num_blocks)
        frames.extend(layer_views)
        total_bytes += sum(len(v) for v in layer_views)
    worker.transfer_stats.record_mb_transferred(total_bytes / (1024 * 1024))
    logger.info(
        "TPUConnectorWorker(%d) rank0 --> serving PULL ch=%d uuid=%s "
        "ranks=%s size=%.2fMB", worker.node_id, channel_idx, uuid,
        ranks_on_channel, total_bytes / (1024 * 1024))
    return frames


def _coord_rank0_external_notif_loop(worker: "ZmqShmKvConnectorBase") -> None:
    sock_path = make_zmq_path("tcp", "*", worker.side_channel_port)
    sock = make_zmq_socket(ctx=worker.zmq_cxt,
                           path=sock_path,
                           socket_type=zmq.ROUTER,
                           bind=True)
    logger.info("TPUConnectorWorker(%d) rank0 --> side channel on %s",
                worker.node_id, sock_path)
    while not worker._stop_event.is_set():
        try:
            if sock.poll(timeout=500) == 0:
                continue
            _client_id, uuid_bytes = sock.recv_multipart()
        except zmq.ContextTerminated:
            return
        except zmq.ZMQError as e:
            logger.warning(
                "TPUConnectorWorker(%d) rank0 --> notif recv error: %s",
                worker.node_id, e)
            continue
        try:
            uuid = int(uuid_bytes.decode("utf-8"))
        except ValueError:
            continue
        # Mark the send entry as acked; get_finished() sweeps it.
        with worker._coord_lock:
            entry = worker._coord_send.get(uuid)
            stray = entry is None
            if not stray:
                entry.pull_acked = True
        if stray:
            logger.warning(
                "TPUConnectorWorker(%d) rank0 --> stray pull-done uuid=%s",
                worker.node_id, uuid)
            continue
        logger.info("TPUConnectorWorker(%d) rank0 --> pull-done uuid=%s",
                    worker.node_id, uuid)
