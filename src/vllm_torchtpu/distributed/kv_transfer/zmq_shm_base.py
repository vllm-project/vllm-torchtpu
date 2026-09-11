# SPDX-License-Identifier: Apache-2.0
"""
ZmqShmKvConnectorBase: P/D disaggregated serving over a per-host SharedMemory
pool plus ZMQ ROUTER/DEALER coordination.

  P workflow:
    P receives the request (max_output_tokens=1). P prefills; when the
    request is FINISHED_LENGTH_CAPPED, the scheduler's request_finished()
    stores a SendMeta in reqs_to_send and returns kv_transfer_params =
    {uuid, remote_block_ids, remote_host, remote_port} to the proxy server.
    On the next scheduler step rank 0 acquires a shm slot, broadcasts
    STAGE_NOTIFY, and every rank D2Hs its shard into the slot. Rank 0's
    ROUTER serves PULL <uuid> requests by scatter-gather sending the
    per-rank memoryviews straight from shm. D notifies P over a side
    channel when the pull is done; P frees the slot. Unacked entries
    expire after TPU_P2P_WAIT_PULL_TIMEOUT.

  D workflow:
    D receives the request with kv_transfer_params. Its
    get_num_new_matched_tokens() reports the remote-hit count and
    update_state_after_alloc() stores a LoadMeta. On the next scheduler
    step rank 0 acquires a shm slot, opens N parallel DEALERs to P, pulls
    and unpacks directly into shm, then broadcasts LOAD_NOTIFY. On the
    following scheduler step (remote_block_ids=None) every rank scatters
    its shard from shm into HBM and acks COPY_DONE; rank 0 notifies P and
    releases the slot once all ranks have copied.
"""

import hashlib
import hmac
import os
import pickle
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import zmq
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu import envs
# The scheduler->worker metadata contract lives in connector_metadata (shared
# with the Raiden connector); re-imported here so existing zmq-stack call
# sites and pickles keep working.
from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
    LoadMeta, ReqId, SendMeta, TPUConnectorMetadata)
from vllm_torchtpu.distributed.kv_transfer.host_kv_shm import (HostKVShmPool,
                                                               PoolSpec)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import \
    TpuKVConnectorStats
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _get_default_ipc_key() -> bytes:
    key_str = envs.VLLM_TORCHTPU_IPC_KEY
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

# Phases tracked by _LatencyTracker. Kept as module-level strings so call
# sites don't typo-drift.
_LAT_D2H = "d2h"  # producer: device -> host (index_select + copy_)
_LAT_WIRE = "wire"  # consumer: network PULL round-trip
_LAT_UNPACK = "unpack"  # consumer: rank0 scatter payload -> shm
_LAT_H2D = "h2d"  # consumer: host -> device (to(device))
_LAT_INSERT = "insert"  # consumer: index_put_ + synchronize on device
_LAT_STAGE = "stage"  # producer: total staging (D2H across all layers+sync)
_LAT_SCATTER = "scatter"  # consumer: total scatter (H2D+insert across layers)

# Intra-host IPC op tags (rank-0 coordinator <-> other ranks).
_IPC_HELLO = b"HELLO"  # worker -> coord: (rank,)
_IPC_STAGE_NOTIFY = b"STAGE_N"  # coord -> worker: (uuid, slot_idx, num_blocks)
_IPC_STAGE_DONE = b"STAGE_D"  # worker -> coord: (uuid, rank, failed)
_IPC_LOAD_NOTIFY = b"LOAD_N"  # coord -> worker: (uuid, slot_idx, num_blocks, local_blocks)
_IPC_LOAD_SKIP = b"LOAD_S"  # coord -> worker: (uuid,) -- drain w/ no scatter (cache hit)
_IPC_COPY_DONE = b"COPY_D"  # worker -> coord: (uuid, rank)
_IPC_DROP = b"DROP"  # coord -> worker: (uuid,)  -- failure/timeout


# ---- Rank-0 bookkeeping (TP>1 coordinator-mode only) -------------------
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


class _LatencyTracker:
    """Running per-phase latency stats + periodic summary log.

    Phases are arbitrary string keys (see _LAT_* module constants). Every
    record() updates n/total/min/max under a lock; the first record() after
    log_interval seconds triggers a summary line. Per-sample lines are up
    to the caller (they often carry req-specific context we'd lose here)."""

    def __init__(self, who: str, log_interval_s: float = 30.0):
        self._who = who
        self._log_interval_s = max(0.0, log_interval_s)
        self._lock = threading.Lock()
        # phase -> [n, total_ms, min_ms, max_ms]
        self._stats: dict[str, list[float]] = {}
        self._last_log = time.perf_counter()

    def record(self, phase: str, duration_ms: float) -> None:
        with self._lock:
            row = self._stats.get(phase)
            if row is None:
                self._stats[phase] = [
                    1.0, duration_ms, duration_ms, duration_ms
                ]
            else:
                row[0] += 1
                row[1] += duration_ms
                if duration_ms < row[2]:
                    row[2] = duration_ms
                if duration_ms > row[3]:
                    row[3] = duration_ms
            if self._log_interval_s <= 0:
                return
            now = time.perf_counter()
            if now - self._last_log < self._log_interval_s:
                return
            parts = []
            for name, (n, total, mn, mx) in self._stats.items():
                avg = total / n if n else 0.0
                parts.append(f"{name}: n={int(n)} avg={avg:.1f}ms "
                             f"min={mn:.1f}ms max={mx:.1f}ms")
            self._last_log = now
        logger.info("%s latency summary | %s", self._who, " | ".join(parts))


class ZmqShmKvConnectorBase:
    """Generic shm-staged, ZMQ-coordinated KV connector worker.

    Subclasses implement the transport hooks at the bottom of this
    class.
    """

    def __init__(self, vllm_config: VllmConfig):
        self._coord_threads: list[threading.Thread] = []
        self._coord_pool: Optional[HostKVShmPool] = None
        self.vllm_config = vllm_config
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer

        self.runner: Optional[Any] = None
        self.device: Optional[torch.device] = None
        self.multi_host = envs.TPU_MULTIHOST_BACKEND == "ray"
        self.node_id: int = dist_utils.get_node_id()
        self.dp_rank: int = vllm_config.parallel_config.data_parallel_rank if vllm_config.parallel_config else 0

        self.tp_rank: int = get_tensor_model_parallel_rank()
        self.tp_size: int = get_tensor_model_parallel_world_size()
        self.ranks_per_host = self.tp_size // int(
            os.getenv("TPU_NUM_HOSTS", "1"))
        self.local_tp_rank = self.tp_rank % self.ranks_per_host
        self._is_host_coordinator = (self.local_tp_rank == 0)

        self.host_ip = dist_utils.get_host_ip()
        self.kv_transfer_port = int(
            dist_utils.get_kv_transfer_port()) + (self.dp_rank * self.tp_size)
        self.side_channel_port = int(
            dist_utils.get_side_channel_port()) + self.node_id + self.dp_rank

        # One ZMQ I/O thread per parallel data socket. Default io_threads=1
        # funnels all tp_size data sockets through a single kernel-I/O
        # thread, which caps aggregate wire throughput at what one thread
        # can push (~10 Gbps) regardless of socket count.
        io_threads = max(1, self.tp_size)
        self.zmq_cxt = zmq.Context(io_threads=io_threads)
        self._stop_event = threading.Event()

        self._lat = _LatencyTracker(
            who=f"TPUConnectorWorker({self.node_id}) rank{self.tp_rank}",
            log_interval_s=dist_utils.get_kv_latency_log_interval())

        self.transfer_stats = TpuKVConnectorStats()
        self._init_coord_state()

        logger.info(
            "TPUConnectorWorker(%s) --> init | ip=%s | kv_transfer_port=%s | "
            "side_channel_port=%s | is_producer=%s | node_id=%d | "
            "tp_rank=%d | tp_size=%d | dp_rank=%d", self.node_id, self.host_ip,
            self.kv_transfer_port, self.side_channel_port, self.is_producer,
            self.node_id, self.tp_rank, self.tp_size, self.dp_rank)

    def _init_coord_state(self) -> None:
        """Minimum state needed on both rank-0 and the other ranks. The
        heavy stuff (SHM pool, ZMQ listeners, IPC endpoints) is allocated
        in register_runner once we know the KV cache shapes."""
        # Parallel TCP stream count. Auto -> tp_size (one channel per rank
        # shard). Clamp to [1, tp_size]. Ports used are
        # base..base+n_channels-1, and a rank's payload rides channel
        # `rank % n_channels`.
        n_cfg = dist_utils.get_transfer_channel_number()
        n_channels = self.tp_size if n_cfg <= 0 else min(n_cfg, self.tp_size)
        self._n_channels: int = max(1, n_channels)
        self._kv_transfer_ports: list[int] = [
            self.kv_transfer_port + self.node_id * self._n_channels + i
            for i in range(self._n_channels)
        ]

        # Multi-layer batched HBM->HBM scatter kernel (kv_scatter): one
        # dispatch across all N layers instead of N per-layer
        # ``cache.index_put_`` calls under lazy torch_tpu. On by default
        # when the smoke test passes; falls back to ``index_put_`` on
        # smoke-test or compile failure.
        self._kv_scatter_enabled: bool = False

        self._coord_ipc_sock: Optional[zmq.Socket] = None
        # Outbound IPC queue: the IPC loop owns the socket and drains this
        # queue each iteration, so send_multipart() is only ever called from
        # one thread (ZMQ sockets are not thread-safe).
        self._coord_ipc_out: queue.Queue = queue.Queue()

        # Local rank to global rank mapping
        self._coord_local_to_global_rank: dict[int, int] = {}
        self._coord_local_to_global_rank[self.local_tp_rank] = self.tp_rank

        # Request-level pulls run asynchronously so the worker main thread can
        # keep servicing vLLM RPCs. Each request then fans out to channel pulls
        # on a separate bounded executor. Keeping these pools close to the
        # actual channel count avoids dozens of Python unpack tasks contending
        # for the GIL while the network is already saturated.
        num_hosts = int(os.getenv("TPU_NUM_HOSTS", "1"))
        channel_workers = dist_utils.get_kv_channel_executor_max_workers()
        if channel_workers == 0:
            channel_workers = max(1, self._n_channels * num_hosts * 2)
        coord_workers = dist_utils.get_kv_coord_executor_max_workers()
        if coord_workers == 0:
            coord_workers = max(1, min(2, channel_workers))
        self._coord_executor = ThreadPoolExecutor(
            max_workers=coord_workers, thread_name_prefix="tpu-kv-coord")
        self._coord_channel_executor = ThreadPoolExecutor(
            max_workers=channel_workers, thread_name_prefix="tpu-kv-channel")
        logger.info(
            "TPUConnectorWorker(%s) --> coord executors | request_workers=%d | "
            "channel_workers=%d | n_channels=%d", self.node_id, coord_workers,
            channel_workers, self._n_channels)

        # Rank-0 authoritative state. Non-zero ranks leave these empty and
        # return empty sets from get_finished; the scheduler unions across
        # workers, so rank 0's report suffices.
        #
        # Producer side: uuid -> _CoordSendEntry.
        self._coord_send: dict[int, _CoordSendEntry] = {}
        # done_sending set produced by the expiration sweeper, consumed by
        # the next get_finished() call.
        self._coord_done_sending: set[ReqId] = set()
        # done_recving set for reqs whose pull failed or couldn't be
        # admitted (shm pool exhausted, scatter exception on rank 0,
        # pull timeout). Without surfacing these, the vLLM scheduler
        # keeps waiting for a done_recving that never arrives -- every
        # failure leaks a request and eventually hangs the engine.
        # We always report them so the scheduler can advance; the KV
        # contents for the affected req are zero/stale, but that's
        # strictly better than a hang.
        self._coord_done_recving: set[ReqId] = set()
        # Consumer side: uuid -> _CoordRecvEntry.
        self._coord_recv: dict[int, _CoordRecvEntry] = {}

        # Identity of each rank's DEALER, keyed by rank, populated by HELLO.
        self._coord_rank_to_identity: dict[int, bytes] = {}
        # Signaled on rank 0 once every non-zero rank has registered via
        # HELLO. We gate the first STAGE_NOTIFY/LOAD_NOTIFY on this to avoid
        # a startup race where a broadcast would miss a slow-joining rank.
        # At tp_size=1 there are no other ranks to wait for; set the event
        # eagerly so the first request doesn't eat a 5s false-positive
        # timeout in _coord_process_send_load.
        self._coord_workers_ready: threading.Event = threading.Event()
        if self.ranks_per_host == 1:
            self._coord_workers_ready.set()

        # Ranks != 0: state for work the coordinator has told us to do.
        # uuid -> (slot_idx, num_blocks, block_ids) -- producer: coord has
        # broadcast a slot assignment; the next process_send_load on this
        # rank will stage inline.
        self._worker_pending_stage: dict[int, tuple[int, int, list[int]]] = {}
        self._worker_pending_stage_cv = threading.Condition()
        # uuid -> (slot_idx, num_blocks, local_blocks) -- consumer: after
        # coord has filled the slot, ranks scatter from shm on next step.
        self._worker_pending_load: dict[int, Optional[tuple[int, int,
                                                            list[int]]]] = {}
        self._worker_pending_load_cv = threading.Condition()

        # Lock around the rank-0 bookkeeping dicts above.
        self._coord_lock = threading.Lock()

        # Async-D2H staging: producer enqueues _StageInFlight entries here;
        # a dispatcher thread per rank submits each entry to a small pool
        # whose workers block on `future.wait()` and then signal STAGE_DONE.
        # Pooling (vs. a single serial waiter) means one hung DMA cannot
        # back up every subsequently-enqueued uuid on the same rank; the
        # dispatcher also enforces a per-entry deadline so even pool-wide
        # hangs eventually fail forward.
        self._stage_pending_q: queue.Queue = queue.Queue()
        _stage_pool_size = dist_utils.get_kv_stage_waiter_pool_size() or 4
        self._coord_stage_waiter_pool = ThreadPoolExecutor(
            max_workers=_stage_pool_size,
            thread_name_prefix=f"tpu_conn_stage_wait_r{self.tp_rank}")
        self._coord_stage_wait_timeout_s = (
            dist_utils.get_kv_stage_wait_timeout_secs())

    def __del__(self):
        self._stop_event.set()
        self._coord_teardown()
        if hasattr(self, "zmq_cxt"):
            self.zmq_cxt.destroy(linger=0)

    # ---- Bring-up -------------------------------------------------------
    def register_runner(self, runner: Any):
        self.runner = runner
        self.device = runner.device

        kv_caches = runner.kv_caches
        assert len(kv_caches) > 0, (
            "register_runner called before kv_caches were allocated")
        self._extract_kv_layout()
        logger.info(
            "TPUConnectorWorker(%d) --> register_runner | node_id=%d | "
            "num_layers=%d | layer_shape=%s | dtype=%s | device=%s",
            self.node_id, self.node_id, self.num_layers, self.shape,
            self.dtype, self.device)
        self._coord_setup()
        self._maybe_enable_kv_scatter()

        if dist_utils.get_kv_warmup_enabled():
            try:
                self._warmup_kv_ops()
            except Exception as e:
                logger.exception(
                    "TPUConnectorWorker(%s) rank%d --> warmup failed "
                    "(continuing; first real request will pay the compile "
                    "cost): %s", self.node_id, self.tp_rank, e)

    def _extract_kv_layout(self) -> None:
        """Default (uniform) layout: every kv cache layer has the same shape
        and dtype. """
        kv_caches = self.runner.kv_caches
        kv_layer = kv_caches[0]
        self.num_layers = len(kv_caches)
        self.shape = list(kv_layer.shape)
        self.dtype = kv_layer.dtype

    def _blocks_token(self, local_block_ids: Any) -> Any:
        return len(local_block_ids)

    def _pull_response_header(self, entry: "_CoordSendEntry") -> dict:
        return {
            "tp_size": self.tp_size,
            "num_layers": self.num_layers,
            "num_blocks": entry.num_blocks,
            "dtype": str(self._coord_pool_spec.dtype),
            "layer_shard_shape":
            list(self._coord_pool_spec.layer_shard_shape[1:]),
        }

    def _replace_runner_kv_cache(self, layer_idx: int,
                                 new_cache: torch.Tensor) -> None:
        """Replace a runner KV cache and keep attention layers bound to it.

        vLLM's bind_kv_cache() stores each cache tensor in both
        runner.kv_caches and the static forward context. The fast scatter path
        returns donated destination tensors, so replacing only
        runner.kv_caches leaves Attention.forward() reading a stale tensor.
        """
        assert self.runner is not None
        kv_caches = self.runner.kv_caches
        old_cache = kv_caches[layer_idx]
        for idx, cache in enumerate(kv_caches):
            if cache is old_cache:
                kv_caches[idx] = new_cache

        compilation_config = self.vllm_config.compilation_config
        static_forward_context = compilation_config.static_forward_context
        if not isinstance(static_forward_context, dict):
            return
        for layer in static_forward_context.values():
            if getattr(layer, "kv_cache", None) is old_cache:
                layer.kv_cache = new_cache

    def _warmup_block_sizes(self) -> list[int]:
        """Block count to pre-compile: the worst-case max_blocks only."""
        block_size = self.vllm_config.cache_config.block_size
        max_model_len = self.vllm_config.model_config.max_model_len
        max_blocks = (max_model_len + block_size - 1) // block_size
        return [max_blocks]

    def _warmup_kv_ops(self) -> None:
        """Pre-compile the XLA executables for the KV gather/scatter hot
        path so the first real request doesn't block on a long compile and
        time out the PULL round-trip on the other side.

        Producer role pre-compiles the D2H path (`index_select` + copy into
        a host staging tensor). Consumer role pre-compiles the H2D +
        `index_put_` path. Both are done per-layer, matching runtime."""
        sizes = self._warmup_block_sizes()
        logger.info(
            "TPUConnectorWorker(%s) rank%d --> warmup starting | role=%s | "
            "num_layers=%d | block_sizes=%s", self.node_id, self.tp_rank,
            "producer" if self.is_producer else "consumer", self.num_layers,
            sizes)
        start = time.perf_counter()
        for n in sizes:
            if n <= 0:
                continue
            t0 = time.perf_counter()
            self._warmup_coord_once(n)
            logger.info(
                "TPUConnectorWorker(%s) rank%d --> warmup num_blocks=%s "
                "took %.2fs", self.node_id, self.tp_rank, n,
                time.perf_counter() - t0)
        logger.info("TPUConnectorWorker(%s) rank%d --> warmup done in %.2fs",
                    self.node_id, self.tp_rank,
                    time.perf_counter() - start)

    def _warmup_coord_once(self, num_blocks: int) -> None:
        """TP>1 coord path: warmup using slot 0 of the shm pool. No traffic
        exists yet at register_runner time, so touching slot 0 is safe.

        Mirrors _coord_stage_shard / _coord_scatter_shard exactly so the
        compiled executable cache hits on first real request. Inlined
        rather than calling the hot path so warmup samples don't leak into
        the latency tracker or emit a stray scatter log line."""
        kv_caches = self.runner.kv_caches
        indices = torch.arange(num_blocks,
                               dtype=torch.int64,
                               device=self.device)
        slot_idx = 0
        if self.is_producer:
            self._stage_d2h_sync(slot_idx, num_blocks, list(range(num_blocks)))
        else:
            # Mirror _coord_scatter_shard exactly. An earlier
            # `.to(device)` warmup compiled a different HLO than the hot
            # path (batch_transfer_h2d into a pre-allocated torch.empty),
            # so the first real request paid a 3-8s recompile.
            use_kv_scatter = self._kv_scatter_enabled
            src_views = [
                self._coord_pool.layer_view(slot_idx, self.local_tp_rank,
                                            layer_idx, num_blocks)
                for layer_idx in range(len(kv_caches))
            ]
            h2d_future, device_shards = self._h2d_into_device_async(src_views)
            self._wait_stage(h2d_future)
            if use_kv_scatter:
                new_caches = self._try_fast_scatter(device_shards,
                                                    list(kv_caches),
                                                    list(range(num_blocks)))
                for c in new_caches:
                    self._synchronize_device(c)
                for i, c in enumerate(new_caches):
                    self._replace_runner_kv_cache(i, c)
            else:
                for layer_idx, cache in enumerate(kv_caches):
                    cache.index_put_((indices, ), device_shards[layer_idx])
                    self._synchronize_device(cache)

    # ---- Main dispatch driven by start_load_kv --------------------------
    def process_send_load(self, metadata: TPUConnectorMetadata):
        return self._coord_process_send_load(metadata)

    def _resolve_remote_host_port(
            self, req_meta: LoadMeta) -> tuple[str, int, Optional[int]]:
        if isinstance(req_meta.remote_host, list):
            assert len(req_meta.remote_host) == len(req_meta.remote_port)
            host = req_meta.remote_host[self.node_id]
            port = req_meta.remote_port[self.node_id]
        else:
            host = req_meta.remote_host
            port = req_meta.remote_port
        side_channel_port = None
        if req_meta.remote_side_channel_port:
            side_channel_port = int(req_meta.remote_side_channel_port)
        return host, int(port), side_channel_port

    # ---- Polled by the runner each step --------------------------------
    def get_finished(
            self,
            finished_req_ids: set[str] | None = None
    ) -> tuple[set[str], set[str]]:
        # finished_req_ids is accepted for signature parity with the worker
        # interface; the coordinator path tracks completion internally.
        return self._coord_get_finished()

    # ====================================================================
    #  Coordinator-mode internals
    # ====================================================================
    # Rank 0 owns all external ZMQ endpoints (data + side-channel) and a
    # SharedMemory slot pool; other ranks stage/scatter into/from their
    # slice of a slot. Control messages between rank 0 and the rest flow
    # over ZMQ ROUTER/DEALER on ipc:// .

    def _build_pool_spec(self) -> Any:
        """Build the per-host shm pool spec. Subclasses
        override this and return their own spec."""
        kv_layer = self.runner.kv_caches[0]
        block_size = self.vllm_config.cache_config.block_size
        max_model_len = self.vllm_config.model_config.max_model_len
        max_blocks = (max_model_len + block_size - 1) // block_size

        # Shard shape is what one rank holds for one layer at max block count.
        # kv_layer is already this rank's shard, so kv_layer.shape[0] is
        # total-blocks-in-cache (>> max_blocks per request); we only need
        # the per-layer spatial dims.
        layer_shard_shape = (max_blocks, ) + tuple(kv_layer.shape[1:])

        # Derive slot count from the user's GB budget. Each slot holds one
        # request's worth of KV across all ranks, so the budget scales
        # with max_model_len * num_layers * kv_heads * head_dim * dtype;
        # 128 GB default comfortably covers hundreds of slots for a typical
        # 70B-class model at TP=8 / max_model_len=8192.
        dtype_bytes = torch.tensor([], dtype=kv_layer.dtype).element_size()
        per_layer_bytes = dtype_bytes
        for d in layer_shard_shape:
            per_layer_bytes *= d
        per_slot_bytes = self.ranks_per_host * self.num_layers * per_layer_bytes
        budget_bytes = int(dist_utils.get_kv_shm_pool_gb() * (1024**3))
        num_slots = max(1, budget_bytes // per_slot_bytes)
        logger.info(
            "TPUConnectorWorker(%s) rank=%d (local_rank=%d)--> shm pool budget=%.2fGB "
            "per_slot=%.2fMB -> num_slots=%d", self.node_id, self.tp_rank,
            self.local_tp_rank, budget_bytes / (1024**3),
            per_slot_bytes / (1024**2), num_slots)

        return PoolSpec(
            num_slots=num_slots,
            tp_size=self.ranks_per_host,
            num_layers=self.num_layers,
            max_blocks=max_blocks,
            layer_shard_shape=layer_shard_shape,
            dtype=kv_layer.dtype,
        )

    def _pool_create(self, spec: Any, shm_name: str) -> Any:
        """Rank-0: create the per-host shm pool."""
        return HostKVShmPool.create(spec, shm_name)

    def _pool_attach(self, spec: Any, shm_name: str) -> Any:
        """Non-rank-0: attach to the per-host shm pool."""
        return HostKVShmPool.attach(spec, shm_name)

    def _coord_setup(self) -> None:
        """Allocate the shm pool, bind ZMQ endpoints, spawn threads. Called
        from register_runner() once kv_caches are known."""
        self._coord_pool_spec = self._build_pool_spec()

        # Stagger shm and ipc with dp_rank
        shm_name = dist_utils.get_shm_name(self.node_id, self.dp_rank)
        ipc_path = dist_utils.get_ipc_socket_path(self.node_id, self.dp_rank)

        if self._is_host_coordinator:
            self._coord_pool = self._pool_create(self._coord_pool_spec,
                                                 shm_name)
            self._coord_setup_rank0(ipc_path)
        else:
            self._coord_pool = self._coord_attach_shm_with_retry(
                self._coord_pool_spec, shm_name)
            self._coord_setup_worker(ipc_path)

        # Pin the shm pool into RAM if requested. Each rank's process must
        # mlock independently (mlock is per-VMA, not per physical page).
        # Run on a daemon thread so init isn't blocked: a 128 GB lock at
        # ~1 us/page takes 30+s and trips vLLM's engine-core heartbeat.
        # Lock only this rank's slice -- each rank only reads its own
        # shard for transfer_h2d_batch.
        if dist_utils.get_kv_pin_shm():
            self._coord_pool.mlock_async(my_rank=self.local_tp_rank)

    def _coord_attach_shm_with_retry(self, spec: Any, name: str) -> Any:
        """Non-rank-0 ranks may race ahead of rank 0. Poll the shm name a
        few seconds before giving up."""
        deadline = time.perf_counter() + 30.0
        last_err: Optional[Exception] = None
        while time.perf_counter() < deadline:
            try:
                return self._pool_attach(spec, name)
            except FileNotFoundError as e:
                last_err = e
                time.sleep(0.1)
        raise RuntimeError(
            f"rank {self.tp_rank}: could not attach to shm {name}: {last_err}")

    # ---- Rank-0 bring-up -----------------------------------------------
    def _coord_setup_rank0(self, ipc_path: str) -> None:
        # IPC ROUTER: workers DEAL in.
        ipc = self.zmq_cxt.socket(zmq.ROUTER)
        ipc.setsockopt(zmq.LINGER, 0)
        # Remove any stale endpoint from a prior crashed run.
        _try_remove_ipc_endpoint(ipc_path)
        ipc.bind(ipc_path)
        self._coord_ipc_sock = ipc
        logger.info("TPUConnectorWorker(%s) rank0 --> IPC listening on %s",
                    self.node_id, ipc_path)

        # IPC listener thread.
        t_ipc = threading.Thread(target=self._coord_rank0_ipc_loop,
                                 name="tpu_conn_ipc",
                                 daemon=True)
        t_ipc.start()
        self._coord_threads.append(t_ipc)

        # Stage waiter: drains the async-D2H queue and signals STAGE_DONE
        # once each future completes.
        t_stage_wait = threading.Thread(target=self._coord_stage_waiter_loop,
                                        name="tpu_conn_stage_wait",
                                        daemon=True)
        t_stage_wait.start()
        self._coord_threads.append(t_stage_wait)

        # Expiration sweeper for pending sends.
        t_exp = threading.Thread(target=self._coord_rank0_expire_loop,
                                 name="tpu_conn_expire",
                                 daemon=True)
        t_exp.start()
        self._coord_threads.append(t_exp)

        # External endpoints are producer-only.
        if self.is_producer:
            for ch in range(self._n_channels):
                t_data = threading.Thread(
                    target=self._coord_rank0_external_data_loop,
                    args=(ch, ),
                    name=f"tpu_conn_data_ch{ch}",
                    daemon=True)
                t_data.start()
                self._coord_threads.append(t_data)

            t_notif = threading.Thread(
                target=self._coord_rank0_external_notif_loop,
                name="tpu_conn_notif",
                daemon=True)
            t_notif.start()
            self._coord_threads.append(t_notif)
        else:
            # Consumer-side: outbound notify-done DEALERs are shared across
            # requests; guard the dict (and each send) behind a lock since
            # it's touched by the main-thread drain path, the IPC listener
            # thread, and async pull threads. Pull sockets are opened per-
            # request, so no caching needed for them.
            self._coord_notif_sockets: dict[str, zmq.Socket] = {}
            self._coord_sockets_lock = threading.Lock()

    # ---- Non-zero rank bring-up ----------------------------------------
    def _coord_setup_worker(self, ipc_path: str) -> None:
        sock = self.zmq_cxt.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        # Identity encodes rank so the coordinator can address us explicitly.
        ident = f"tpu_rank_{self.local_tp_rank}".encode("utf-8")
        sock.setsockopt(zmq.IDENTITY, ident)
        sock.connect(ipc_path)
        self._coord_ipc_sock = sock
        # Announce ourselves; messages buffer on DEALER until coord ROUTER
        # reads them, so this is safe even if coord isn't up yet.
        self._coord_send_ipc(_IPC_HELLO, (self.local_tp_rank, self.tp_rank))
        logger.info(
            "TPUConnectorWorker(%s) rank%d, local_rank%d --> IPC connected to %s",
            self.node_id, self.tp_rank, self.local_tp_rank, ipc_path)

        t = threading.Thread(target=self._coord_worker_ipc_loop,
                             name=f"tpu_conn_worker_ipc_{self.tp_rank}",
                             daemon=True)
        t.start()
        self._coord_threads.append(t)

        # Stage waiter: same role as on rank 0; sends STAGE_DONE via IPC
        # once the per-rank D2H future completes.
        t_stage_wait = threading.Thread(
            target=self._coord_stage_waiter_loop,
            name=f"tpu_conn_stage_wait_{self.tp_rank}",
            daemon=True)
        t_stage_wait.start()
        self._coord_threads.append(t_stage_wait)

    # ---- IPC envelope helpers ------------------------------------------
    # All IPC send paths enqueue; the IPC loop thread is the only one that
    # actually calls send_multipart. This keeps the single ZMQ socket off
    # of multiple threads.

    def _coord_send_ipc(self, tag: bytes, payload) -> None:
        """Worker (non-zero rank) -> coordinator. Enqueues a DEALER frame."""
        data = _secure_dumps(payload)
        # DEALER frames are [tag, data]; ROUTER side reassembles identity
        # from the socket itself.
        self._coord_ipc_out.put((None, tag, data))

    def _coord_broadcast(self, tag: bytes, payload) -> None:
        """Coordinator -> every known non-zero rank. Enqueues one frame per
        rank, addressed to the rank's DEALER identity."""
        data = _secure_dumps(payload)
        with self._coord_lock:
            identities = list(self._coord_rank_to_identity.values())
        for ident in identities:
            self._coord_ipc_out.put((ident, tag, data))

    def _coord_ipc_drain_outbound(self) -> None:
        """Pop any pending outbound messages and send them. Called from
        inside the owning IPC loop; must not be called from other threads."""
        sock = self._coord_ipc_sock
        try:
            while True:
                ident, tag, data = self._coord_ipc_out.get_nowait()
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
                        self.node_id, tag, e)
        except queue.Empty:
            return

    # ---- Coord teardown -------------------------------------------------
    def _coord_teardown(self) -> None:
        self._stop_event.set()
        # Wake the stage waiter so it can observe _stop_event and exit.
        if hasattr(self, "_stage_pending_q"):
            try:
                self._stage_pending_q.put_nowait(None)
            except Exception:
                pass
        if hasattr(self, "_coord_executor"):
            self._coord_executor.shutdown(wait=False)
        if hasattr(self, "_coord_channel_executor"):
            self._coord_channel_executor.shutdown(wait=False)
        if hasattr(self, "_coord_stage_waiter_pool"):
            self._coord_stage_waiter_pool.shutdown(wait=False)
        for t in self._coord_threads:
            t.join(timeout=2)
        if self._coord_pool is not None:
            self._coord_pool.close()

    # =========================================================
    # Process send/load dispatcher (coord mode)
    # =========================================================
    def _coord_process_send_load(self, metadata: TPUConnectorMetadata) -> None:
        # Rank 0 is the only one that knows about remote endpoints and owns
        # the free list; it kicks off both staging (producer) and pulling
        # (consumer) here. Every rank, including rank 0, is also responsible
        # for its own H2D scatter when a pulled req is in the "drain" phase
        # (remote_block_ids is None).
        #
        # Staging runs inline on the main thread on every rank (rank 0 while
        # broadcasting, others after waiting on STAGE_NOTIFY). Running stage
        # on a background executor is unsafe under TP>1: on non-zero ranks,
        # by the time the executor got to `.to("cpu")`, the main thread had
        # already queued the next model_forward (first-time decode compile
        # under multi-rank collectives), so the D2H sync blocked behind it
        # and the PULL timed out. Inline stage drains each rank's XLA queue
        # before model_forward adds anything to it.

        if self._is_host_coordinator:
            if metadata.reqs_to_send or metadata.reqs_to_load:
                # Only gate on worker registration when we actually need to
                # broadcast. If all HELLO's are still in flight, wait briefly.
                if not self._coord_workers_ready.wait(timeout=5.0):
                    logger.warning(
                        "TPUConnectorWorker(%d) rank0 --> still waiting on "
                        "HELLOs: %d/%d registered", self.node_id,
                        len(self._coord_rank_to_identity),
                        self.ranks_per_host - 1)
                for req_id, req_meta in metadata.reqs_to_send.items():
                    self._coord_rank0_handle_new_send(req_id, req_meta)
                for req_id, req_meta in metadata.reqs_to_load.items():
                    self._coord_rank0_handle_new_load(req_id, req_meta)
        else:
            # Non-zero ranks: stage inline for each reqs_to_send in this
            # metadata. Rank 0 broadcasts STAGE_NOTIFY (with slot_idx); we
            # block here until it arrives (via IPC thread), run stage in
            # the main thread, and return before model_forward executes.
            for req_id, req_meta in metadata.reqs_to_send.items():
                self._coord_worker_stage_inline(req_meta.uuid, req_id)

        # Drain step (all ranks): scheduler is asking us to scatter reqs for
        # which the pull has already finished.
        for req_id, req_meta in metadata.reqs_to_load.items():
            if req_meta.remote_block_ids is None:
                self._coord_drain_scatter(req_id, req_meta)

    # =========================================================
    # Rank 0 producer path: stage + serve PULLs
    # =========================================================
    def _coord_rank0_handle_new_send(self, req_id: ReqId,
                                     req_meta: SendMeta) -> None:
        """Allocate a slot, broadcast the assignment to non-zero ranks, and
        stage rank 0's own shard inline on the main thread.

        The inline stage is deliberate: see the note in
        _coord_process_send_load. Broadcasting first (via the IPC-out queue,
        drained by the IPC thread every ~50ms) lets non-zero ranks begin
        their own inline stage in parallel with us."""
        assert self.is_producer
        num_blocks = self._blocks_token(req_meta.local_block_ids)
        logger.info(
            "TPUConnectorWorker(%s) rank0 --> handle_new_send req_id=%s uuid=%s "
            "blocks=%s", self.node_id, req_id, req_meta.uuid, num_blocks)
        logger.info("PERF P handle_new_send req_id=%s uuid=%s ts=%.6f", req_id,
                    req_meta.uuid, time.time())
        try:
            slot_idx = self._coord_pool.acquire_slot(timeout=0.0)
        except Exception:
            # Pool full. Drop request; scheduler will surface a failure.
            with self._coord_lock:
                send_holders = [
                    f"uuid={u} req={e.req_id} "
                    f"acked={e.pull_acked} "
                    f"exp_in={max(0.0, e.expiration_time - time.perf_counter()):.1f}s"
                    for u, e in self._coord_send.items()
                ]
                recv_holders = [
                    f"uuid={u} req={e.req_id}"
                    for u, e in self._coord_recv.items()
                ]
                self._coord_done_sending.add(req_id)
            logger.error(
                "TPUConnectorWorker(%s) rank0 --> shm pool exhausted for "
                "req_id=%s (producer). send_holders=%s recv_holders=%s",
                self.node_id, req_id, send_holders, recv_holders)
            return

        entry = _CoordSendEntry(
            req_id=req_id,
            slot_idx=slot_idx,
            num_blocks=num_blocks,
            expiration_time=req_meta.expiration_time,
            tp_size=self.tp_size,
        )
        with self._coord_lock:
            self._coord_send[req_meta.uuid] = entry

        # Enqueue broadcast first so other ranks can start their own stage
        # while we run ours.
        self._coord_broadcast(
            _IPC_STAGE_NOTIFY,
            (req_meta.uuid, slot_idx, num_blocks, req_meta.local_block_ids))
        logger.info(
            "TPUConnectorWorker(%d) rank0 --> handle_new_send broadcast uuid=%s "
            "slot=%d to %d non-zero ranks", self.node_id, req_meta.uuid,
            slot_idx, len(self._coord_rank_to_identity))

        # Rank 0 enqueues its own async D2H on the main thread; the waiter
        # thread emits STAGE_DONE once the future completes. Returning here
        # before the DMA finishes lets vLLM advance and the
        # STAGE_NOTIFY/scheduler/PULL-setup round-trip overlap with the
        # transfer.
        try:
            self._coord_stage_shard(req_meta.uuid, slot_idx, num_blocks,
                                    req_meta.local_block_ids)
        except Exception as ex:
            logger.exception(
                "TPUConnectorWorker(%d) rank0 --> stage_self uuid=%s failed: %s",
                self.node_id, req_meta.uuid, ex)

    def _coord_worker_stage_inline(self,
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
            "STAGE_NOTIFY req_id=%s uuid=%s", self.node_id, self.tp_rank,
            req_id, uuid)
        deadline = time.perf_counter() + timeout
        info: Optional[tuple[int, int, list[int]]] = None
        with self._worker_pending_stage_cv:
            while uuid not in self._worker_pending_stage:
                rem = deadline - time.perf_counter()
                if rem <= 0:
                    logger.warning(
                        "TPUConnectorWorker(%d) rank%d --> stage_inline timeout "
                        "uuid=%s (STAGE_NOTIFY never arrived)", self.node_id,
                        self.tp_rank, uuid)
                    # Without STAGE_DONE this rank's shard never completes, so
                    # rank0 keeps the slot pending and the consumer's PULL
                    # blocks for its full timeout, leaking a consumer slot per
                    # occurrence. Signal failure like the exception path below
                    # so PULL serving answers _MSG_ERR immediately.
                    self._signal_stage_done(uuid, failed=True)
                    return
                self._worker_pending_stage_cv.wait(timeout=rem)
            info = self._worker_pending_stage.pop(uuid)
        slot_idx, num_blocks, block_ids = info
        try:
            self._coord_stage_shard(uuid, slot_idx, num_blocks, block_ids)
        except Exception as e:
            logger.exception(
                "TPUConnectorWorker(%d) rank%d --> stage_inline uuid=%s failed: %s",
                self.node_id, self.tp_rank, uuid, e)
            # Enqueue path didn't run, so the waiter won't fire STAGE_DONE
            # for us. Signal it directly with failed=True so rank 0 marks
            # the entry stage_failed and PULL serving will _MSG_ERR rather
            # than ship a partially-zero shm slice.
            self._signal_stage_done(uuid, failed=True)

    def _maybe_enable_kv_scatter(self) -> None:
        """Subclass hook: optionally enable the multi-layer fused scatter
        on startup. Default is a no-op (leaves ``_kv_scatter_enabled`` as
        False, falling back to per-layer ``index_put_`` in the drain pass).
        Subclasses override to run a device-specific smoke test."""
        return

    def _coord_stage_shard(self, uuid: int, slot_idx: int, num_blocks: int,
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
            "uuid=%s slot=%d blocks=%s layers=%d", self.node_id, self.tp_rank,
            uuid, slot_idx, num_blocks, len(self.runner.kv_caches))
        enqueue_t0 = time.perf_counter()
        future, src_refs, dst_refs, d2h_total_bytes = self._stage_d2h(
            slot_idx, num_blocks, block_ids)
        issue_t1 = time.perf_counter()
        enqueue_ms = (time.perf_counter() - enqueue_t0) * 1000.0

        self._stage_pending_q.put(
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
            "uuid=%s slot=%d enqueue=%.2fms", self.node_id, self.tp_rank, uuid,
            slot_idx, enqueue_ms)

    def _coord_stage_waiter_loop(self) -> None:
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
        timeout_s = self._coord_stage_wait_timeout_s
        # (deadline, entry, py_future); list ordering is not load-bearing.
        in_flight: list[tuple[float, _StageInFlight, Any]] = []
        while not self._stop_event.is_set():
            got_entry = True
            try:
                entry = self._stage_pending_q.get(timeout=0.05)
            except queue.Empty:
                got_entry = False
                entry = None
            if got_entry:
                if entry is None:
                    # Shutdown sentinel from _coord_teardown.
                    return
                py_fut = self._coord_stage_waiter_pool.submit(
                    self._coord_stage_wait_worker, entry)
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
                            "thread leaks until wait returns)", self.node_id,
                            self.tp_rank, ent.uuid, ent.slot_idx, timeout_s)
                        self._complete_stage_entry(ent, failed=True)
                        # Stop tracking; the eventual worker callback
                        # dedups via the entry's signal lock.
                        continue
                    kept.append((deadline, ent, fut))
                in_flight = kept

    def _coord_stage_wait_worker(self, entry: "_StageInFlight") -> None:
        """Pool worker: block on the entry's stage future and complete.

        Runs in `_coord_stage_waiter_pool`. Several can run in parallel
        across distinct entries; a single hung wait only consumes one slot
        until the dispatcher's deadline marks the entry failed."""
        failed = False
        try:
            wait_t0 = time.perf_counter()
            self._wait_stage(entry.future)
            wait_ms = (time.perf_counter() - wait_t0) * 1000.0
            total_ms = (time.perf_counter() - entry.enqueue_t0) * 1000.0
            issue_ms = ((entry.issue_t1 - entry.enqueue_t0) *
                        1000.0 if entry.issue_t1 else 0.0)
            mb = entry.total_bytes / (1024 * 1024)
            mbps = mb / max(1e-3, total_ms / 1000.0)
            self._lat.record(_LAT_D2H, total_ms)
            self._lat.record(_LAT_STAGE, total_ms)
            self.transfer_stats.record_d2h_transfer(total_ms)
            logger.info(
                "TPUConnectorWorker(%s) rank%d --> stage_done "
                "uuid=%s slot=%d issue=%.2fms wait=%.2fms total=%.2fms "
                "throughput=%.2fMiB/s", self.node_id, self.tp_rank, entry.uuid,
                entry.slot_idx, issue_ms, wait_ms, total_ms, mbps)
        except Exception as e:
            failed = True
            logger.exception(
                "TPUConnectorWorker(%s) rank%d --> stage_waiter uuid=%s "
                "future.wait() raised: %s", self.node_id, self.tp_rank,
                entry.uuid, e)
        self._complete_stage_entry(entry, failed=failed)

    def _complete_stage_entry(self, entry: "_StageInFlight",
                              failed: bool) -> None:
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
        self._signal_stage_done(entry.uuid, failed=failed)

    def _signal_stage_done(self, uuid: int, failed: bool = False) -> None:
        """Mark this rank's STAGE_DONE on the rank-0 send entry. On rank 0
        this is a direct local update; on other ranks it goes through IPC
        to rank 0's listener, which performs the same update. If `failed`
        is True the entry is tagged stage_failed so PULL serving aborts
        cleanly with _MSG_ERR rather than shipping stale shm data."""
        if self._is_host_coordinator:
            fired_complete = False
            staged_count = 0
            per_rank_event: Optional[threading.Event] = None
            with self._coord_lock:
                entry = self._coord_send.get(uuid)
                if entry is None:
                    return
                if failed:
                    entry.stage_failed = True
                entry.staged.add(0)  ## local_tp_rank?
                staged_count = len(entry.staged)
                if staged_count == self.ranks_per_host:
                    entry.stage_complete.set()
                    fired_complete = True
                if 0 < len(entry.staged_events):  ## self.local_tp_rank?
                    per_rank_event = entry.staged_events[
                        0]  ## self.local_tp_rank?
            # Set the per-rank event outside the lock; the wait side is
            # _coord_rank0_build_pull_response, which only takes the lock
            # briefly to look up the entry, then waits on the event.
            if per_rank_event is not None:
                per_rank_event.set()
            if fired_complete:
                logger.info(
                    "TPUConnectorWorker(%d) rank0 --> stage_complete uuid=%s "
                    "(all %d ranks staged, self-fire for node %d)",
                    self.node_id, uuid, self.ranks_per_host, self.node_id)
                logger.info("PERF P stage_complete(%s) uuid=%s ts=%.6f",
                            self.node_id, uuid, time.time())
        else:
            self._coord_send_ipc(_IPC_STAGE_DONE,
                                 (uuid, self.local_tp_rank, failed))

    # =========================================================
    # Rank 0 consumer path: pull, write shm, broadcast LOAD_NOTIFY
    # =========================================================
    def _coord_rank0_handle_new_load(self, req_id: ReqId,
                                     req_meta: LoadMeta) -> None:
        assert not self.is_producer
        logger.info(
            "PERF D handle_new_load req_id=%s uuid=%s ts=%.6f drain=%s",
            req_id, req_meta.uuid, time.time(), req_meta.remote_block_ids
            is None)
        if req_meta.remote_block_ids is None:
            # Either a full cache hit or a drain-for-completed-pull. Only
            # the former requires telling other ranks to skip; the drain
            # case means _worker_pending_load already has the real entry
            # from an earlier LOAD_NOTIFY broadcast.
            with self._coord_lock:
                is_cache_hit = req_meta.uuid not in self._coord_recv
            if is_cache_hit:
                self._coord_broadcast(_IPC_LOAD_SKIP, (req_meta.uuid, ))
            return

        num_blocks = self._blocks_token(req_meta.local_block_ids)
        # Preemption dedup: vLLM's scheduler re-emits update_state_after_alloc
        # for requests that were preempted and rescheduled. The LoadMeta
        # carries the same uuid (assigned by the producer) but new
        # local_block_ids. If we treat it as a fresh load we acquire a
        # second shm slot, submit a duplicate _coord_rank0_pull thread,
        # and orphan the first slot when we overwrite _coord_recv[uuid]
        # -- eventually exhausting the pool and hanging the scheduler.
        # Detect the re-emit by uuid and just update local_blocks in
        # place so the in-flight pull targets the new block destinations.
        with self._coord_lock:
            existing = self._coord_recv.get(req_meta.uuid)
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
                    self.node_id, req_id, req_meta.uuid, num_blocks)
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
                    self.node_id, req_id, req_meta.uuid, num_blocks)
                self._coord_broadcast(
                    _IPC_LOAD_NOTIFY,
                    (existing.uuid, existing.slot_idx, existing.num_blocks,
                     existing.local_blocks))
            return

        try:
            slot_idx = self._coord_pool.acquire_slot(timeout=0.0)
        except Exception:
            with self._coord_lock:
                recv_holders = [
                    f"uuid={u} req={e.req_id} "
                    f"pull_ok={e.pull_ok} "
                    f"load={e.load_complete.is_set()} "
                    f"reported={e.reported_done} "
                    f"copied={sorted(e.copied)}"
                    for u, e in self._coord_recv.items()
                ]
                send_holders = [
                    f"uuid={u} req={e.req_id} "
                    f"acked={e.pull_acked}"
                    for u, e in self._coord_send.items()
                ]
                self._coord_done_recving.add(req_id)
            logger.error(
                "TPUConnectorWorker(%d) rank0 --> shm pool exhausted for "
                "req_id=%s; surfacing done_recving so scheduler advances "
                "(expect wrong output for this req). recv_holders=%s "
                "send_holders=%s", self.node_id, req_id, recv_holders,
                send_holders)
            return

        remote_host, remote_port, remote_side_channel_port = self._resolve_remote_host_port(
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
        with self._coord_lock:
            self._coord_recv[req_meta.uuid] = entry

        self._coord_executor.submit(self._coord_rank0_pull, entry)

    def _coord_rank0_pull(self, entry: _CoordRecvEntry) -> None:
        """Pull the full KV blob (all ranks' shards) from remote coord for all hosts
        and unpack it into shm. Then broadcast LOAD_NOTIFY and set the
        load_complete event so get_finished() returns done_recving.

        Uses ``self._n_channels`` parallel DEALER sockets (one per thread)
        to N producer ROUTER ports ``base_port + ch``. Each channel carries
        its subset of rank payloads; we reassemble them into a single
        rank->payload map then unpack all shards into shm."""
        hosts = entry.remote_host if isinstance(entry.remote_host,
                                                list) else [entry.remote_host]
        ports = entry.remote_port if isinstance(entry.remote_port,
                                                list) else [entry.remote_port]
        num_hosts = len(hosts)
        n_channels = self._n_channels
        uuid_bytes = str(entry.uuid).encode("utf-8")
        timeout_s = dist_utils.get_p2p_wait_pull_timeout()
        req_blocks_pickle = _secure_dumps(entry.remote_blocks)
        logger.info(
            "TPUConnectorWorker(%s) rank0 --> PULL req_id=%s uuid=%s "
            "from=%s:%s+0..%s blocks=%s n_channels=%s", self.node_id,
            entry.req_id, entry.uuid, entry.remote_host, ports, n_channels - 1,
            entry.num_blocks, n_channels)

        num_layers = self.num_layers

        def _pull_channel(h: int, ch: int):
            """Pull one channel's subset of ranks AND unpack their layer
            frames into shm inline. Returns per-channel (wire_ms,
            unpack_ms, ranks_handled) so the orchestrator can log the
            breakdown but does not need to reassemble anything."""
            port = ports[h] + ch
            host = hosts[h]
            sock_path = make_zmq_path("tcp", host, port)
            sock = make_zmq_socket(ctx=self.zmq_cxt,
                                   path=sock_path,
                                   socket_type=zmq.DEALER,
                                   bind=False,
                                   linger=0)
            timeout_ms = max(1, int(timeout_s * 1000))
            sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
            try:
                t0 = time.perf_counter()
                logger.info(
                    "PERF D pull_send host_idx=%d ch=%d uuid=%s ts=%.6f", h,
                    ch, entry.uuid, time.time())
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
                logger.info(
                    "PERF D wire_done host_idx=%d ch=%d uuid=%s ts=%.6f", h,
                    ch, entry.uuid, time.time())

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
                with self._coord_lock:
                    global_to_local = {
                        global_rank: local_rank
                        for local_rank, global_rank in
                        self._coord_local_to_global_rank.items()
                    }

                for _ in range(n_ranks):
                    global_rank = int(
                        bytes(frames[idx].buffer).decode("utf-8"))
                    idx += 1
                    layer_buffers = [
                        frames[idx + li].buffer for li in range(num_layers)
                    ]
                    idx += num_layers

                    if global_rank in global_to_local:
                        local_rank = global_to_local[global_rank]
                        self._coord_pool.unpack_rank_layers(
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
                self._coord_channel_executor.submit(_pull_channel, h, ch)
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
                        raise RuntimeError(
                            f"PULL req_id={entry.req_id}: rank {r} "
                            f"delivered by multiple channels")
                    seen.add(r)
            if header_frame is None:
                raise RuntimeError(
                    f"PULL req_id={entry.req_id}: channel 0 omitted header")
            if len(seen) != self.ranks_per_host:
                raise RuntimeError(
                    f"PULL req_id={entry.req_id}: got {len(seen)} ranks, "
                    f"expected {self.ranks_per_host}")

            total_ms = (t_done - t_total0) * 1000.0
            # With inline unpack per channel, wire+unpack are already
            # overlapped; the outer total is max over channels of
            # (wire+unpack). Report them separately for continuity
            # with the existing latency tracker.
            max_wire = max(w for _h, _c, w, _u, _r, _hdr in results)
            max_unpack = max(u for _h, _c, _w, u, _r, _hdr in results)
            self._lat.record(_LAT_WIRE, max_wire)
            self._lat.record(_LAT_UNPACK, max_unpack)
            self.transfer_stats.record_kv_pull(total_ms)
            per_ch_ms = ",".join(f"{w + u:.1f}"
                                 for _h, _c, w, u, _r, _hdr in results)
            logger.info(
                "TPUConnectorWorker(%d) rank0 --> done PULL req_id=%s uuid=%s "
                "total=%.2fms max_wire=%.2fms max_unpack=%.2fms "
                "per_ch_ms=[%s]", self.node_id, entry.req_id, entry.uuid,
                total_ms, max_wire, max_unpack, per_ch_ms)
            entry.pull_ok = True
        except Exception as e:
            logger.exception(
                "TPUConnectorWorker(%d) rank0 --> PULL req_id=%s failed: %s",
                self.node_id, entry.req_id, e)
            entry.pull_ok = False
            self.transfer_stats.record_failed_transfer()
        finally:
            # Tell workers the slot is ready (or drop it on failure).
            if entry.pull_ok:
                self._coord_broadcast(_IPC_LOAD_NOTIFY,
                                      (entry.uuid, entry.slot_idx,
                                       entry.num_blocks, entry.local_blocks))
            else:
                self._coord_broadcast(_IPC_DROP, (entry.uuid, ))
                # Clean up: failed pulls will never reach COPY_DONE, so
                # free the slot and drop bookkeeping now. Also surface
                # the req_id as done_recving -- otherwise the vLLM
                # scheduler waits forever for completion of a req whose
                # entry we just silently removed.
                with self._coord_lock:
                    self._coord_recv.pop(entry.uuid, None)
                    self._coord_done_recving.add(entry.req_id)
                self._coord_pool.release_slot(entry.slot_idx)
            entry.load_complete.set()
            logger.info(
                "PERF D pull_complete req_id=%s uuid=%s ts=%.6f pull_ok=%s",
                entry.req_id, entry.uuid, time.time(), entry.pull_ok)

    # =========================================================
    # Drain pass: every rank (including 0) scatters its shard
    # =========================================================
    def _coord_scatter_and_ack(self, req_id: ReqId, uuid: int, slot_idx: int,
                               num_blocks: int,
                               local_blocks: list[int]) -> None:
        logger.info("PERF D drain_start rank=%d req_id=%s uuid=%s ts=%.6f",
                    self.tp_rank, req_id, uuid, time.time())
        try:
            self._coord_scatter_shard(slot_idx, num_blocks, local_blocks)
        except Exception as e:
            logger.exception(
                "TPUConnectorWorker(%s) rank%d --> scatter req_id=%s failed: %s",
                self.node_id, self.tp_rank, req_id, e)
        finally:
            if self._is_host_coordinator:
                self._coord_rank0_register_copy(uuid, self.local_tp_rank)
            else:
                self._coord_send_ipc(_IPC_COPY_DONE,
                                     (uuid, self.local_tp_rank))
            logger.info("PERF D drain_done rank=%d req_id=%s uuid=%s ts=%.6f",
                        self.tp_rank, req_id, uuid, time.time())

    def _coord_drain_scatter(self, req_id: ReqId, req_meta: LoadMeta) -> None:
        uuid = req_meta.uuid
        if self._is_host_coordinator:
            # Rank 0's state is in _coord_recv; grab slot info from there.
            with self._coord_lock:
                entry = self._coord_recv.get(uuid)
            if entry is None:
                return
            if not entry.pull_ok:
                # Unknown or failed; scheduler will treat this as a cache hit
                # and we notify upstream so P can free its buffer.
                self._coord_rank0_send_notify(req_meta, uuid)
                return
            slot_idx = entry.slot_idx
            num_blocks = entry.num_blocks
            local_blocks = entry.local_blocks
        else:
            info = self._coord_worker_wait_load(uuid)
            if info is None:
                # DROP arrived or timeout. Nothing to do.
                return
            slot_idx, num_blocks, local_blocks = info

        self._coord_scatter_and_ack(req_id, uuid, slot_idx, num_blocks,
                                    local_blocks)

    def _coord_scatter_shard(self, slot_idx: int, num_blocks: int,
                             local_blocks: list[int]) -> None:
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
        kv_caches = self.runner.kv_caches
        use_fast_scatter = self._kv_scatter_enabled
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
                src_view = self._coord_pool.layer_view(slot_idx,
                                                       self.local_tp_rank,
                                                       layer_idx, num_blocks)
                src_views.append(src_view)
                h2d_total_bytes += src_view.numel() * src_view.element_size()
            alloc_t1 = time.perf_counter()

            h2d_t0 = time.perf_counter()
            device_shards = self._h2d_into_device(src_views)
            h2d_t1 = time.perf_counter()
            new_caches = self._try_fast_scatter(device_shards, list(kv_caches),
                                                local_blocks)
            # Sync every returned cache -- they're independent outputs of the
            # fused kernel and the device may schedule them in parallel, so
            # syncing only the last would race.
            for c in new_caches:
                self._synchronize_device(c)
            k_wait_t = time.perf_counter()
            for i, c in enumerate(new_caches):
                self._replace_runner_kv_cache(i, c)
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
                                   device=self.device)
            alloc_t0 = time.perf_counter()
            src_views = []
            h2d_total_bytes = 0
            for layer_idx in range(len(kv_caches)):
                src_view = self._coord_pool.layer_view(slot_idx,
                                                       self.local_tp_rank,
                                                       layer_idx, num_blocks)
                src_views.append(src_view)
                h2d_total_bytes += src_view.numel() * src_view.element_size()
            alloc_t1 = time.perf_counter()

            h_t0 = time.perf_counter()
            h2d_future, device_shards = self._h2d_into_device_async(src_views)
            h2d_t_issued = time.perf_counter()
            self._wait_stage(h2d_future)
            h_t1 = time.perf_counter()

            h2d_ms_total = (h_t1 - h_t0) * 1000.0
            h2d_issue_ms = (h2d_t_issued - h_t0) * 1000.0
            h2d_wait_ms = (h_t1 - h2d_t_issued) * 1000.0
            insert_ms_total = 0.0
            alloc_ms_total = (alloc_t1 - alloc_t0) * 1000.0

            for layer_idx, cache in enumerate(kv_caches):
                ins_t0 = time.perf_counter()
                cache.index_put_((indices, ), device_shards[layer_idx])
                self._synchronize_device(cache)
                insert_ms_total += (time.perf_counter() - ins_t0) * 1000.0
            path = "naive"
        self._lat.record(_LAT_H2D, h2d_ms_total)
        self._lat.record(_LAT_INSERT, insert_ms_total)
        self._lat.record(_LAT_SCATTER, h2d_ms_total + insert_ms_total)
        self.transfer_stats.record_h2d_transfer(h2d_ms_total)
        h2d_mb = h2d_total_bytes / (1024 * 1024)
        h2d_mbps = h2d_mb / max(1e-3, h2d_ms_total / 1000.0)
        logger.info(
            "TPUConnectorWorker(%s) rank%d --> scatter slot=%d blocks=%s "
            "layers=%d alloc=%.2fms h2d=%.2fms h2d_issue=%.2fms "
            "h2d_wait=%.2fms insert=%.2fms h2d_throughput=%.2fMiB/s path=%s",
            self.node_id, self.tp_rank, slot_idx, num_blocks, len(kv_caches),
            alloc_ms_total, h2d_ms_total, h2d_issue_ms, h2d_wait_ms,
            insert_ms_total, h2d_mbps, path)

    def _coord_rank0_register_copy(self, uuid: int, rank_idx: int = 0) -> None:
        entry_for_notify = None
        with self._coord_lock:
            entry = self._coord_recv.get(uuid)
            if entry is None:
                return
            entry.copied.add(rank_idx)
            num_hosts = int(os.getenv("TPU_NUM_HOSTS", "1"))
            ranks_per_host = self.tp_size // num_hosts
            if len(entry.copied) == ranks_per_host:
                entry_for_notify = self._coord_recv.pop(uuid, None)
        if entry_for_notify is not None:
            # Notify remote P and release slot.
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
                self._coord_rank0_send_notify(dummy, uuid)
            finally:
                self._coord_pool.release_slot(entry_for_notify.slot_idx)
                logger.info(
                    "PERF D scatter_complete req_id=%s uuid=%s ts=%.6f",
                    entry_for_notify.req_id, uuid, time.time())

    def _coord_rank0_send_notify(self, req_meta: LoadMeta, uuid: int) -> None:
        remote_hosts = req_meta.remote_host
        if not isinstance(remote_hosts, list):
            remote_hosts = [remote_hosts]

        # Determine the base port for the target DP group (without host/node offset)
        if req_meta.remote_side_channel_port:
            target_side_port_base = req_meta.remote_side_channel_port
        else:
            # Fallback to current DP group's base port
            target_side_port_base = int(
                dist_utils.get_side_channel_port()) + self.dp_rank

        for h_idx, r_host in enumerate(remote_hosts):
            side_port = target_side_port_base + h_idx
            sock_path = make_zmq_path("tcp", r_host, side_port)
            # DEALER is not thread-safe; serialize create+send under the lock.
            with self._coord_sockets_lock:
                sock = self._coord_notif_sockets.get(sock_path)
                new_sock = sock is None
                if new_sock:
                    sock = make_zmq_socket(ctx=self.zmq_cxt,
                                           path=sock_path,
                                           socket_type=zmq.DEALER,
                                           bind=False)
                    self._coord_notif_sockets[sock_path] = sock
                sock.send_string(str(uuid))
            if new_sock:
                logger.info(
                    "TPUConnectorWorker(%d) rank0 --> notify channel to %s",
                    self.node_id, sock_path)
            logger.info(
                "TPUConnectorWorker(%d) rank0 --> notify pull-done uuid=%s to %s",
                self.node_id, uuid, r_host)

    # =========================================================
    # Non-zero rank: wait for coord messages then act
    # =========================================================
    def _coord_worker_wait_load(
            self,
            uuid: int,
            timeout: float = 1.0) -> Optional[tuple[int, int, list[int]]]:
        """Block until LOAD_NOTIFY or LOAD_SKIP for uuid arrives, or timeout
        expires. Returns (slot_idx, num_blocks, local_blocks) for a scatter,
        or None for skip/timeout. The dict stores None as a sentinel set
        by LOAD_SKIP, so we can tell "skip" apart from "not there yet" by
        presence."""
        deadline = time.perf_counter() + timeout
        with self._worker_pending_load_cv:
            while uuid not in self._worker_pending_load:
                rem = deadline - time.perf_counter()
                if rem <= 0:
                    return None
                self._worker_pending_load_cv.wait(timeout=rem)
            info = self._worker_pending_load.pop(uuid)
        return info

    # =========================================================
    # IPC listeners
    # =========================================================
    def _coord_rank0_ipc_loop(self) -> None:
        sock = self._coord_ipc_sock
        while not self._stop_event.is_set():
            # Drain outbound sends from the queue first so notifications
            # don't starve behind inbound polling.
            self._coord_ipc_drain_outbound()
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
                    self.node_id, e)
                continue
            # ROUTER gives us [identity, tag, payload].
            if len(frames) != 3:
                logger.warning(
                    "TPUConnectorWorker(%d) rank0 --> IPC malformed frames n=%d",
                    self.node_id, len(frames))
                continue
            identity, tag, payload = frames
            try:
                obj = _secure_loads(payload)
            except Exception as e:
                logger.warning(
                    "TPUConnectorWorker(%d) rank0 --> bad IPC "
                    "payload tag=%s: %s", self.node_id, tag, e)
                continue
            if tag == _IPC_HELLO:
                local_rank, global_rank = obj
                with self._coord_lock:
                    self._coord_rank_to_identity[local_rank] = identity
                    self._coord_local_to_global_rank[local_rank] = global_rank
                    # All non-zero ranks registered? Unblock new-req
                    # processing on rank 0.
                    if len(self._coord_rank_to_identity
                           ) == self.ranks_per_host - 1:
                        self._coord_workers_ready.set()
                logger.info(
                    "TPUConnectorWorker(%d) rank0 --> HELLO from local_rank=%d global_rank=%d "
                    "(%d/%d registered)",
                    self.node_id, local_rank, global_rank,
                    len(self._coord_rank_to_identity), self.ranks_per_host - 1)
            elif tag == _IPC_STAGE_DONE:
                # Payload is (uuid, rank, failed). All workers run from
                # the same code so we don't need a length-tolerant unpack.
                uuid, rank, failed = obj
                fired_complete = False
                staged_count = 0
                per_rank_event: Optional[threading.Event] = None
                with self._coord_lock:
                    entry = self._coord_send.get(uuid)
                    if entry is None:
                        continue
                    if failed:
                        entry.stage_failed = True
                    entry.staged.add(rank)
                    staged_count = len(entry.staged)
                    if staged_count == self.ranks_per_host:
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
                        "(all %d ranks staged)", self.node_id, uuid,
                        self.ranks_per_host)
                    logger.info("PERF P stage_complete uuid=%s ts=%.6f", uuid,
                                time.time())
                else:
                    logger.info(
                        "TPUConnectorWorker(%d) rank0 --> STAGE_DONE uuid=%s "
                        "rank=%d (%d/%d)", self.node_id, uuid, rank,
                        staged_count, self.ranks_per_host)
            elif tag == _IPC_COPY_DONE:
                uuid, rank = obj
                entry_for_notify = None
                with self._coord_lock:
                    entry = self._coord_recv.get(uuid)
                    if entry is None:
                        continue
                    entry.copied.add(rank)
                    if len(entry.copied) == self.ranks_per_host:
                        entry_for_notify = self._coord_recv.pop(uuid, None)
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
                        self._coord_rank0_send_notify(dummy,
                                                      entry_for_notify.uuid)
                    finally:
                        self._coord_pool.release_slot(
                            entry_for_notify.slot_idx)
                        logger.info(
                            "PERF D scatter_complete req_id=%s uuid=%s ts=%.6f",
                            entry_for_notify.req_id, uuid, time.time())
            else:
                logger.warning(
                    "TPUConnectorWorker(%d) rank0 --> unknown IPC tag=%s",
                    self.node_id, tag)

    def _coord_worker_ipc_loop(self) -> None:
        sock = self._coord_ipc_sock
        while not self._stop_event.is_set():
            self._coord_ipc_drain_outbound()
            try:
                if sock.poll(timeout=50) == 0:
                    continue
                frames = sock.recv_multipart()
            except zmq.ContextTerminated:
                return
            except zmq.ZMQError as e:
                logger.warning(
                    "TPUConnectorWorker(%d) rank%d --> IPC recv error: %s",
                    self.node_id, self.tp_rank, e)
                continue
            # DEALER gives us [tag, payload].
            if len(frames) != 2:
                logger.warning(
                    "TPUConnectorWorker(%d) rank%d --> IPC malformed frames n=%d",
                    self.node_id, self.tp_rank, len(frames))
                continue
            tag, payload = frames
            try:
                obj = _secure_loads(payload)
            except Exception as e:
                logger.warning(
                    "TPUConnectorWorker(%d) rank%d --> bad IPC payload tag=%s: %s",
                    self.node_id, self.tp_rank, tag, e)
                continue
            if tag == _IPC_STAGE_NOTIFY:
                uuid, slot_idx, num_blocks, block_ids = obj
                logger.info(
                    "TPUConnectorWorker(%s) rank%d --> STAGE_NOTIFY received "
                    "uuid=%s slot=%d blocks=%s", self.node_id, self.tp_rank,
                    uuid, slot_idx, num_blocks)
                # Hand off to the main thread (waiting in
                # _coord_worker_stage_inline); don't stage in an executor
                # here, otherwise .to("cpu") races with model_forward.
                with self._worker_pending_stage_cv:
                    self._worker_pending_stage[uuid] = (slot_idx, num_blocks,
                                                        block_ids)
                    self._worker_pending_stage_cv.notify_all()
            elif tag == _IPC_LOAD_NOTIFY:
                if len(obj) == 5:
                    uuid, _req_id, slot_idx, num_blocks, local_blocks = obj
                else:
                    # Backward-compatible decode for any in-flight old control
                    # messages during rolling restarts.
                    uuid, slot_idx, num_blocks, local_blocks = obj
                with self._worker_pending_load_cv:
                    self._worker_pending_load[uuid] = (slot_idx, num_blocks,
                                                       local_blocks)
                    self._worker_pending_load_cv.notify_all()
            elif tag == _IPC_LOAD_SKIP:
                (uuid, ) = obj
                # None marks "coord said no scatter needed for this uuid"
                # (cache hit). _coord_worker_wait_load surfaces None
                # immediately without the 200ms-ish timeout wait.
                with self._worker_pending_load_cv:
                    self._worker_pending_load[uuid] = None
                    self._worker_pending_load_cv.notify_all()
            elif tag == _IPC_DROP:
                (uuid, ) = obj
                with self._worker_pending_load_cv:
                    self._worker_pending_load[uuid] = None
                    self._worker_pending_load_cv.notify_all()
                with self._worker_pending_stage_cv:
                    self._worker_pending_stage.pop(uuid, None)
                    self._worker_pending_stage_cv.notify_all()
            else:
                logger.warning(
                    "TPUConnectorWorker(%d) rank%d --> unknown IPC tag=%s",
                    self.node_id, self.tp_rank, tag)

    # =========================================================
    # Rank 0: external data-server (producer serves PULL)
    # =========================================================
    def _coord_rank0_external_data_loop(self, channel_idx: int) -> None:
        """One data server per TCP channel. channel_idx in [0, n_channels).

        A rank's payload rides channel ``rank % n_channels``. Channel 0 also
        prepends the pickled header so the consumer can derive tp_size /
        shapes before reassembling. Each channel binds its own port
        ``base + channel_idx`` and serves its subset of ranks only, giving
        N parallel TCP streams end-to-end."""
        port = self._kv_transfer_ports[channel_idx]
        sock_path = make_zmq_path("tcp", "*", port)
        sock = make_zmq_socket(ctx=self.zmq_cxt,
                               path=sock_path,
                               socket_type=zmq.ROUTER,
                               bind=True)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.ROUTER_MANDATORY, 0)
        sock.setsockopt(zmq.SNDTIMEO,
                        max(1,
                            dist_utils.get_p2p_wait_pull_timeout() * 1000))
        ranks_on_channel = [
            r for r in range(self.ranks_per_host)
            if r % self._n_channels == channel_idx
        ]
        logger.info(
            "TPUConnectorWorker(%d) rank0 --> data server ch=%d listening on "
            "tcp://%s:%s serving ranks=%s", self.node_id, channel_idx,
            self.host_ip, port, ranks_on_channel)

        ready_responses: queue.Queue = queue.Queue()

        def _prepare_response(client_id: bytes, uuid: int,
                              uuid_bytes: bytes) -> None:
            try:
                response = self._coord_rank0_build_pull_response(
                    uuid, uuid_bytes, channel_idx, ranks_on_channel)
            except Exception:
                logger.exception(
                    "TPUConnectorWorker(%d) rank0 --> build_pull_response "
                    "ch=%d uuid=%s failed", self.node_id, channel_idx, uuid)
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
                            "ch=%d uuid=%s failed: %s", self.node_id,
                            channel_idx, uuid, e)
                    continue
                # copy=False lets ZMQ send the big per-layer memoryviews
                # directly from shm via scatter-gather sendmsg, no userland
                # copy. Small framing frames pay a tiny per-frame overhead;
                # net win is ~256MB of memcpy avoided per rank.
                _t_send_start = time.time()
                logger.info("PERF P send_start ch=%d uuid=%s ts=%.6f",
                            channel_idx, uuid, _t_send_start)
                try:
                    sock.send_multipart([client_id, *response], copy=False)
                except zmq.ZMQError as e:
                    logger.warning(
                        "TPUConnectorWorker(%d) rank0 --> data send ch=%d "
                        "uuid=%s failed: %s", self.node_id, channel_idx, uuid,
                        e)
                logger.info(
                    "PERF P send_done ch=%d uuid=%s ts=%.6f dur_ms=%.2f",
                    channel_idx, uuid, time.time(),
                    (time.time() - _t_send_start) * 1000.0)

        while not self._stop_event.is_set():
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
                    "ch=%d error: %s", self.node_id, channel_idx, e)
                continue
            # Expected: [client_id, PULL, uuid_bytes, remote_blocks_pickle]
            if len(frames) < 3 or frames[1] != _MSG_PULL:
                logger.warning(
                    "TPUConnectorWorker(%d) rank0 --> data malformed "
                    "request ch=%d n=%d", self.node_id, channel_idx,
                    len(frames))
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
                        "ch=%d failed: %s", self.node_id, channel_idx, e)
                continue
            logger.info(
                "TPUConnectorWorker(%d) rank0 --> PULL received ch=%d uuid=%s",
                self.node_id, channel_idx, uuid)
            logger.info("PERF P pull_recv ch=%d uuid=%s ts=%.6f", channel_idx,
                        uuid, time.time())
            self._coord_channel_executor.submit(_prepare_response, client_id,
                                                uuid, uuid_bytes)

    def _coord_rank0_build_pull_response(
            self, uuid: int, uuid_bytes: bytes, channel_idx: int,
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
            with self._coord_lock:
                entry = self._coord_send.get(uuid)
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
                "(timed out waiting for register-send)", self.node_id,
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
                        self.node_id, channel_idx, uuid, len(entry.staged),
                        self.ranks_per_host)
                    return None
                break
            if not entry.staged_events[r].wait(timeout=rem):
                logger.warning(
                    "TPUConnectorWorker(%d) rank0 --> PULL ch=%d uuid=%s timed "
                    "out waiting for rank %d to stage (have %d of %d)",
                    self.node_id, channel_idx, uuid, r, len(entry.staged),
                    self.ranks_per_host)
                return None
        with self._coord_lock:
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
                "stage_failed; sending ERR", self.node_id, channel_idx, uuid)
            return None
        # Wire format (per channel, after the ROUTER envelope):
        #   ch 0:  [OK, uuid, n_ranks, header,
        #            (rank_idx, layer_mv_0, ..., layer_mv_{L-1})*]
        #   ch >0: [OK, uuid, n_ranks,
        #            (rank_idx, layer_mv_0, ..., layer_mv_{L-1})*]
        # Each layer_mv is a memoryview straight into the shm slot for
        # that (rank, layer); there is no bytearray/bytes staging on
        # the producer. Consumer knows num_layers from self.num_layers
        # so it doesn't need to be inline; the header on ch 0 carries
        # the sanity-check metadata.
        frames: list = [
            _MSG_OK, uuid_bytes,
            str(len(ranks_on_channel)).encode("utf-8")
        ]
        if channel_idx == 0:
            header = self._pull_response_header(entry)
            frames.append(_secure_dumps(header))
        total_bytes = 0
        for r in ranks_on_channel:
            global_rank = self._coord_local_to_global_rank[r]
            frames.append(str(global_rank).encode("utf-8"))
            layer_views = self._coord_pool.rank_layer_views(
                entry.slot_idx, r, entry.num_blocks)
            frames.extend(layer_views)
            total_bytes += sum(len(v) for v in layer_views)
        self.transfer_stats.record_mb_transferred(total_bytes / (1024 * 1024))
        logger.info(
            "TPUConnectorWorker(%d) rank0 --> serving PULL ch=%d uuid=%s "
            "ranks=%s size=%.2fMB", self.node_id, channel_idx, uuid,
            ranks_on_channel, total_bytes / (1024 * 1024))
        return frames

    # =========================================================
    # Rank 0: external side-channel (producer receives NOTIFY)
    # =========================================================
    def _coord_rank0_external_notif_loop(self) -> None:
        sock_path = make_zmq_path("tcp", "*", self.side_channel_port)
        sock = make_zmq_socket(ctx=self.zmq_cxt,
                               path=sock_path,
                               socket_type=zmq.ROUTER,
                               bind=True)
        logger.info("TPUConnectorWorker(%d) rank0 --> side channel on %s",
                    self.node_id, sock_path)
        while not self._stop_event.is_set():
            try:
                if sock.poll(timeout=500) == 0:
                    continue
                _client_id, uuid_bytes = sock.recv_multipart()
            except zmq.ContextTerminated:
                return
            except zmq.ZMQError as e:
                logger.warning(
                    "TPUConnectorWorker(%d) rank0 --> notif recv error: %s",
                    self.node_id, e)
                continue
            try:
                uuid = int(uuid_bytes.decode("utf-8"))
            except ValueError:
                continue
            # Mark the send entry as acked; get_finished() sweeps it.
            with self._coord_lock:
                entry = self._coord_send.get(uuid)
                stray = entry is None
                if not stray:
                    entry.pull_acked = True
            if stray:
                logger.warning(
                    "TPUConnectorWorker(%d) rank0 --> stray pull-done uuid=%s",
                    self.node_id, uuid)
                continue
            logger.info("TPUConnectorWorker(%d) rank0 --> pull-done uuid=%s",
                        self.node_id, uuid)

    # =========================================================
    # Rank 0: periodic expiration sweep
    # =========================================================
    def _coord_rank0_expire_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=1.0):
                return
            now = time.perf_counter()
            expired: list[int] = []
            with self._coord_lock:
                for uuid, entry in list(self._coord_send.items()):
                    # Acked entries are handled in get_finished() so they
                    # surface to the scheduler in the correct tick.
                    if not entry.pull_acked and now > entry.expiration_time:
                        expired.append(uuid)
            for uuid in expired:
                logger.warning(
                    "TPUConnectorWorker(%d) rank0 --> expire send uuid=%s",
                    self.node_id, uuid)
                with self._coord_lock:
                    entry = self._coord_send.pop(uuid, None)
                    if entry is None:
                        continue
                    self._coord_done_sending.add(entry.req_id)
                self._coord_pool.release_slot(entry.slot_idx)

    # =========================================================
    # get_finished (coord mode)
    # =========================================================
    def _coord_get_finished(self) -> tuple[set[str], set[str]]:
        # Only rank 0 returns non-empty; scheduler unions across workers.
        if not self._is_host_coordinator:
            return set(), set()
        done_sending: set[str] = set()
        done_recving: set[str] = set()
        released_slots: list[int] = []
        with self._coord_lock:
            # Producer: pull_acked → done_sending; scheduler unioning means
            # we only need to report once.
            for uuid in list(self._coord_send.keys()):
                entry = self._coord_send[uuid]
                if entry.pull_acked:
                    done_sending.add(entry.req_id)
                    del self._coord_send[uuid]
                    released_slots.append(entry.slot_idx)
            # Merge in any expiration-sweeper results (already removed from
            # _coord_send by that thread, slot already released).
            if self._coord_done_sending:
                done_sending |= self._coord_done_sending
                self._coord_done_sending.clear()
            # Consumer: pull_ok + load_complete -> done_recving. Mark
            # `reported_done` so repeated get_finished() calls don't emit
            # the same req id twice; the entry stays in _coord_recv until
            # all ranks have acknowledged their scatter via COPY_DONE.
            newly_done_recving: list[tuple[str, int]] = []
            for uuid, entry in list(self._coord_recv.items()):
                if (entry.load_complete.is_set() and entry.pull_ok
                        and not entry.reported_done):
                    done_recving.add(entry.req_id)
                    entry.reported_done = True
                    newly_done_recving.append((entry.req_id, uuid))
            # Merge in failure-surfacing reqs (pull failed, shm pool
            # exhausted on admit). The scheduler advances these as if
            # the KV were ready; the output will be wrong for that req,
            # but the alternative is an engine-wide hang.
            if self._coord_done_recving:
                done_recving |= self._coord_done_recving
                self._coord_done_recving.clear()
        # Free slots outside the lock to avoid nesting lock waits.
        for sidx in released_slots:
            self._coord_pool.release_slot(sidx)
        for req_id, uuid in newly_done_recving:
            logger.info("PERF D done_recving req_id=%s uuid=%s ts=%.6f",
                        req_id, uuid, time.time())
        if done_sending:
            logger.info("TPUConnectorWorker(%d) rank0 --> done_sending=%s",
                        self.node_id, done_sending)
        if done_recving:
            logger.info("TPUConnectorWorker(%d) rank0 --> done_recving=%s",
                        self.node_id, done_recving)
        return done_sending, done_recving

    # ====================================================================
    # Abstract transport hooks (subclass implements)
    # ====================================================================
    def _stage_d2h(self, slot_idx: int, num_blocks: int,
                   block_ids: list[int]) -> tuple[Any, list, list, int]:
        """Issue an async device->host transfer of this rank's shard. """
        raise NotImplementedError

    def _stage_d2h_sync(self, slot_idx: int, num_blocks: int,
                        block_ids: list[int]) -> None:
        """Synchronous variant of ``_stage_d2h``."""
        raise NotImplementedError

    def _wait_stage(self, future: Any) -> None:
        """Block on a future returned by ``_stage_d2h``."""
        raise NotImplementedError

    def _h2d_into_device(self, src_views: list) -> list[torch.Tensor]:
        """Copy the shm data into HBM synchronously."""
        raise NotImplementedError

    def _h2d_into_device_async(
            self, src_views: list) -> tuple[Any, list[torch.Tensor]]:
        """Async _h2d_into_device. """
        raise NotImplementedError

    def _synchronize_device(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError

    def _try_fast_scatter(
            self, device_shards: list[torch.Tensor],
            kv_caches: list[torch.Tensor],
            local_blocks: list[int]) -> Optional[list[torch.Tensor]]:
        raise NotImplementedError

    def get_kv_connector_stats(self) -> TpuKVConnectorStats | None:
        """
            Get the KV transfer stats for the worker.
            """
        if self._is_host_coordinator:
            with self._coord_lock:
                if self.is_producer:
                    self.transfer_stats.record_prefill_queue_length(
                        len(self._coord_send))
                else:
                    self.transfer_stats.record_decode_queue_length(
                        len(self._coord_recv))

        # Clear stats for next iteration
        if not self.transfer_stats.is_empty():
            return self.transfer_stats.clone_and_reset()
        return None


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
