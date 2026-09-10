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

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import torch
import zmq
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu import envs
from vllm_torchtpu.distributed.kv_transfer import (zmq_shm_coordination,
                                                   zmq_shm_staging)
# The scheduler->worker metadata contract lives in connector_metadata (shared
# with the Raiden connector); re-imported here so existing zmq-stack call
# sites and pickles keep working.
from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
    LoadMeta, ReqId, SendMeta, TPUConnectorMetadata)
from vllm_torchtpu.distributed.kv_transfer.host_kv_shm import (HostKVShmPool,
                                                               PoolSpec)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import \
    TpuKVConnectorStats
from vllm_torchtpu.distributed.kv_transfer.zmq_shm_coordination import (
    _CoordRecvEntry, _CoordSendEntry)
from vllm_torchtpu.distributed.kv_transfer.zmq_shm_staging import \
    _StageInFlight
from vllm_torchtpu.distributed.kv_transfer.zmq_shm_telemetry import \
    _LatencyTracker
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


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
        return zmq_shm_staging._warmup_kv_ops(self)

    def _warmup_coord_once(self, num_blocks: int) -> None:
        return zmq_shm_staging._warmup_coord_once(self, num_blocks)

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
        return zmq_shm_coordination._coord_setup_rank0(self, ipc_path)

    # ---- Non-zero rank bring-up ----------------------------------------
    def _coord_setup_worker(self, ipc_path: str) -> None:
        return zmq_shm_coordination._coord_setup_worker(self, ipc_path)

    # ---- IPC envelope helpers ------------------------------------------
    # All IPC send paths enqueue; the IPC loop thread is the only one that
    # actually calls send_multipart. This keeps the single ZMQ socket off
    # of multiple threads.

    def _coord_send_ipc(self, tag: bytes, payload) -> None:
        return zmq_shm_coordination._coord_send_ipc(self, tag, payload)

    def _coord_broadcast(self, tag: bytes, payload) -> None:
        return zmq_shm_coordination._coord_broadcast(self, tag, payload)

    def _coord_ipc_drain_outbound(self) -> None:
        return zmq_shm_coordination._coord_ipc_drain_outbound(self)

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
        return zmq_shm_coordination._coord_rank0_handle_new_send(
            self, req_id, req_meta)

    def _coord_worker_stage_inline(self,
                                   uuid: int,
                                   req_id: ReqId,
                                   timeout: float = 10.0) -> None:
        return zmq_shm_staging._coord_worker_stage_inline(
            self, uuid, req_id, timeout)

    def _maybe_enable_kv_scatter(self) -> None:
        """Subclass hook: optionally enable the multi-layer fused scatter
        on startup. Default is a no-op (leaves ``_kv_scatter_enabled`` as
        False, falling back to per-layer ``index_put_`` in the drain pass).
        Subclasses override to run a device-specific smoke test."""
        return

    def _coord_stage_shard(self, uuid: int, slot_idx: int, num_blocks: int,
                           block_ids: list[int]) -> None:
        return zmq_shm_staging._coord_stage_shard(self, uuid, slot_idx,
                                                  num_blocks, block_ids)

    def _coord_stage_waiter_loop(self) -> None:
        return zmq_shm_staging._coord_stage_waiter_loop(self)

    def _coord_stage_wait_worker(self, entry: "_StageInFlight") -> None:
        return zmq_shm_staging._coord_stage_wait_worker(self, entry)

    def _complete_stage_entry(self, entry: "_StageInFlight",
                              failed: bool) -> None:
        return zmq_shm_staging._complete_stage_entry(self, entry, failed)

    def _signal_stage_done(self, uuid: int, failed: bool = False) -> None:
        return zmq_shm_coordination._signal_stage_done(self, uuid, failed)

    # =========================================================
    # Rank 0 consumer path: pull, write shm, broadcast LOAD_NOTIFY
    # =========================================================
    def _coord_rank0_handle_new_load(self, req_id: ReqId,
                                     req_meta: LoadMeta) -> None:
        return zmq_shm_coordination._coord_rank0_handle_new_load(
            self, req_id, req_meta)

    def _coord_rank0_pull(self, entry: _CoordRecvEntry) -> None:
        return zmq_shm_coordination._coord_rank0_pull(self, entry)

    # =========================================================
    # Drain pass: every rank (including 0) scatters its shard
    # =========================================================
    def _coord_scatter_and_ack(self, req_id: ReqId, uuid: int, slot_idx: int,
                               num_blocks: int,
                               local_blocks: list[int]) -> None:
        return zmq_shm_staging._coord_scatter_and_ack(self, req_id, uuid,
                                                      slot_idx, num_blocks,
                                                      local_blocks)

    def _coord_drain_scatter(self, req_id: ReqId, req_meta: LoadMeta) -> None:
        return zmq_shm_staging._coord_drain_scatter(self, req_id, req_meta)

    def _coord_scatter_shard(self, slot_idx: int, num_blocks: int,
                             local_blocks: list[int]) -> None:
        return zmq_shm_staging._coord_scatter_shard(self, slot_idx, num_blocks,
                                                    local_blocks)

    def _coord_rank0_register_copy(self, uuid: int, rank_idx: int = 0) -> None:
        return zmq_shm_coordination._coord_rank0_register_copy(
            self, uuid, rank_idx)

    def _coord_rank0_send_notify(self, req_meta: LoadMeta, uuid: int) -> None:
        return zmq_shm_coordination._coord_rank0_send_notify(
            self, req_meta, uuid)

    # =========================================================
    # Non-zero rank: wait for coord messages then act
    # =========================================================
    def _coord_worker_wait_load(
            self,
            uuid: int,
            timeout: float = 1.0) -> Optional[tuple[int, int, list[int]]]:
        return zmq_shm_coordination._coord_worker_wait_load(
            self, uuid, timeout)

    # =========================================================
    # IPC listeners
    # =========================================================
    def _coord_rank0_ipc_loop(self) -> None:
        return zmq_shm_coordination._coord_rank0_ipc_loop(self)

    def _coord_worker_ipc_loop(self) -> None:
        return zmq_shm_coordination._coord_worker_ipc_loop(self)

    # =========================================================
    # Rank 0: external data-server (producer serves PULL)
    # =========================================================
    def _coord_rank0_external_data_loop(self, channel_idx: int) -> None:
        return zmq_shm_coordination._coord_rank0_external_data_loop(
            self, channel_idx)

    def _coord_rank0_build_pull_response(
            self, uuid: int, uuid_bytes: bytes, channel_idx: int,
            ranks_on_channel: list[int]) -> Optional[list[bytes]]:
        return zmq_shm_coordination._coord_rank0_build_pull_response(
            self, uuid, uuid_bytes, channel_idx, ranks_on_channel)

    # =========================================================
    # Rank 0: external side-channel (producer receives NOTIFY)
    # =========================================================
    def _coord_rank0_external_notif_loop(self) -> None:
        return zmq_shm_coordination._coord_rank0_external_notif_loop(self)

    # =========================================================
    # Rank 0: periodic expiration sweep
    # =========================================================
    def _coord_rank0_expire_loop(self) -> None:
        return zmq_shm_coordination._coord_rank0_expire_loop(self)

    # =========================================================
    # get_finished (coord mode)
    # =========================================================
    def _coord_get_finished(self) -> tuple[set[str], set[str]]:
        return zmq_shm_coordination._coord_get_finished(self)

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
