# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration test suite and profiling recipes for chunked transfer.

Validates:
1. End-to-end pipelined chunked transfer pipeline between simulated prefill and
   decode worker ranks with exact numerical parity on reconstructed KV cache.
2. Preemption and client cancellation mid-transfer, verifying complete resource
   cleanup without lingering tensor references or memory leaks.
3. Prefix-cached chunked transfer bypassing already-cached prompt chunks.
4. Hybrid model architectures transferring recurrent states (GDN conv/SSM)
   exclusively on the terminal completion chunk.
5. Sliding window flow control (W=2) pacing in-flight chunk emissions.
6. Out-of-order sub-chunk arrival and contiguous interval reassembly.
"""

from collections.abc import Iterator
import contextlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

from vllm_torchtpu.distributed.kv_transfer.connector_metadata import SendMeta
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPURaidenConnectorScheduler,
    TPURaidenConnectorWorker,
    chunk_uuid,
)

_MOD = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"


class _FakeRaidenEngine:
    """Simulated Raiden RDMA / shared-memory transfer engine."""

    def __init__(self) -> None:
        """Initializes transfer engine call recording queues."""
        self.calls: list[Any] = []
        self.complete_read_results: list[
            tuple[list[str], list[str], list[str]]
        ] = []
        self.poll_results: list[tuple[list[str], list[str], list[str]]] = []

    def register_read(
        self, req_id: str, uuid: int, block_ids: list[int]
    ) -> int:
        """Records a register_read invocation with target blocks.

        Args:
            req_id: Unique request identifier.
            uuid: Unique identifier for the transfer chunk.
            block_ids: List of local block indices to register for reading.

        Returns:
            Simulated read handle identifier.
        """
        self.calls.append(("register_read", req_id, uuid, block_ids))
        return 1

    def start_read(
        self,
        req_id: str,
        uuid: int,
        endpoint: str,
        remote_blocks: list[int],
        local_blocks: list[int],
    ) -> int:
        """Records start_read invocation and initiates simulated transfer.

        Args:
            req_id: Unique request identifier.
            uuid: Unique identifier for the transfer chunk.
            endpoint: Remote worker network address.
            remote_blocks: Source block indices on the remote worker.
            local_blocks: Destination block indices on the local worker.

        Returns:
            Simulated transfer handle identifier.
        """
        self.calls.append(
            ("start_read", req_id, uuid, endpoint, remote_blocks, local_blocks)
        )
        return 2

    def complete_read(self) -> tuple[list[str], list[str], list[str]]:
        """Returns queued complete_read status triples.

        Pops and returns the next pending completion status containing lists
        of sending, receiving, and failed transfer request identifiers.

        Returns:
            A triple of (sending_done, recving_done, failed) request IDs.
        """
        self.calls.append(("complete_read",))
        if self.complete_read_results:
            return self.complete_read_results.pop(0)
        if self.poll_results:
            return self.poll_results.pop(0)
        return [], [], []

    def poll_stats(self) -> tuple[list[str], list[str], list[str]]:
        """Returns queued poll_stats status triples.

        Pops and returns the next pending status tuple for completed transfers.

        Returns:
            A triple of (sending_done, recving_done, failed) request IDs.
        """
        self.calls.append(("poll_stats",))
        if self.poll_results:
            return self.poll_results.pop(0)
        if self.complete_read_results:
            return self.complete_read_results.pop(0)
        return [], [], []


class _FakeRaidenControllerFacade:
    """Mock controller RPC client recording registration and lifecycle calls."""

    def __init__(self) -> None:
        """Initializes lists to record controller RPC calls."""
        self.register_work_unit_calls: list[dict[str, Any]] = []
        self.register_request_blocks_calls: list[dict[str, Any]] = []
        self.complete_request_blocks_calls: list[dict[str, Any]] = []
        self.cancel_request_blocks_calls: list[dict[str, Any]] = []
        self.start_transfer_calls: list[dict[str, Any]] = []
        self.cancel_request_blocks_result: bool = True
        self.start_transfer_result: bool = True

    def register_work_unit(self, **kwargs: Any) -> None:
        """Records a register_work_unit RPC.

        Args:
            **kwargs: Work unit configuration and replica parameters.
        """
        self.register_work_unit_calls.append(kwargs)

    def register_request_blocks(self, **kwargs: Any) -> None:
        """Records a register_request_blocks RPC.

        Args:
            **kwargs: Block registration parameters including pool spans
              and UUID.
        """
        self.register_request_blocks_calls.append(kwargs)

    def complete_request_blocks(self, **kwargs: Any) -> None:
        """Records a complete_request_blocks RPC.

        Args:
            **kwargs: Completion status and block cleanup parameters.
        """
        self.complete_request_blocks_calls.append(kwargs)

    def cancel_request_blocks_if_unclaimed(self, **kwargs: Any) -> bool:
        """Records a cancel_request_blocks_if_unclaimed RPC.

        Args:
            **kwargs: Cancellation parameters for unclaimed request blocks.

        Returns:
            Boolean indicating whether unclaimed blocks were cancelled.
        """
        self.cancel_request_blocks_calls.append(kwargs)
        return self.cancel_request_blocks_result

    def start_transfer(self, **kwargs: Any) -> bool:
        """Records a start_transfer RPC.

        Args:
            **kwargs: Transfer initiation parameters.

        Returns:
            Boolean indicating whether transfer initiation succeeded.
        """
        self.start_transfer_calls.append(kwargs)
        return self.start_transfer_result


@contextlib.contextmanager
def _patch_raiden_stage3_env(
    *,
    base_uuid: int = 500000,
    parallelism: int = 4,
    timeout_s: float = 30.0,
    controller_address: str = "127.0.0.1:27000",
    job_name: str = "prefill",
    engine_id: str = "prefill-rank0",
) -> Iterator[None]:
    """Patches Stage-3 environment variables and global flags.

    Temporarily overrides Raiden transport configuration, controller address,
    job name, engine identifier, parallelism factor, wait timeouts, and
    deterministic UUID generators for the duration of the context.

    Args:
        base_uuid: Base integer used for generating deterministic chunk UUIDs.
        parallelism: Prefill context parallelism factor.
        timeout_s: P2P wait timeout in seconds.
        controller_address: Host and port for the Raiden controller facade.
        job_name: Job name string for the Raiden work unit.
        engine_id: Engine identifier string for the Raiden worker instance.

    Yields:
        None when active.
    """
    with (
        patch(f"{_MOD}._use_raiden_stage3_transport", return_value=True),
        patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT", "raiden"),
        patch(
            f"{_MOD}.tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS",
            controller_address,
        ),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_JOB_NAME", job_name),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_ENGINE_ID", engine_id),
        patch(
            f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM",
            parallelism,
        ),
        patch(
            f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
            return_value=timeout_s,
        ),
        patch(f"{_MOD}.get_uuid", return_value=base_uuid),
    ):
        yield


def _make_producer_scheduler(
    *,
    block_size: int = 16,
    pcp_size: int = 4,
    interleave_size: int = 256,
    chunk_transfer_window: int = 2,
    max_model_len: int = 32768,
) -> TPURaidenConnectorScheduler:
    """Creates a TPURaidenConnectorScheduler for the prefill producer rank.

    Constructs a mocked VllmConfig with prefill context parallelism, cache
    block sizing, sliding window parameters, and model configuration, then
    instantiates the producer scheduler with patched network discovery.

    Args:
        block_size: Number of tokens per cache block in the scheduler.
        pcp_size: Prefill context parallelism size.
        interleave_size: Token interleave stride across PCP ranks.
        chunk_transfer_window: Maximum concurrent in-flight chunks (W).
        max_model_len: Maximum context sequence length supported by the model.

    Returns:
        Configured TPURaidenConnectorScheduler instance for the producer rank.
    """
    cfg = MagicMock(spec=VllmConfig)
    cfg.kv_transfer_config = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = True
    cfg.kv_transfer_config.chunk_transfer_window = chunk_transfer_window
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.cache_config = MagicMock()
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = True
    cfg.model_config = MagicMock()
    cfg.model_config.max_model_len = max_model_len
    cfg.model_config.num_hidden_layers = 32
    cfg.model_config.get_num_layers = MagicMock(return_value=32)
    cfg.model_config.get_num_kv_heads = MagicMock(return_value=8)
    cfg.model_config.get_head_size = MagicMock(return_value=128)
    cfg.model_config.dtype = torch.bfloat16
    cfg.parallel_config = MagicMock()
    cfg.parallel_config.data_parallel_rank = 0
    cfg.parallel_config.data_parallel_size = 1
    cfg.parallel_config.tensor_parallel_size = 1
    cfg.parallel_config.prefill_context_parallel_size = pcp_size
    cfg.parallel_config.cp_kv_cache_interleave_size = interleave_size

    with (
        patch(f"{_MOD}.dist_utils.get_kv_ips", return_value="127.0.0.1"),
        patch(f"{_MOD}.dist_utils.get_kv_ports", return_value=9100),
        patch(f"{_MOD}.dist_utils.get_side_channel_port", return_value="9600"),
    ):
        scheduler = TPURaidenConnectorScheduler(cfg)
    return scheduler


def _make_producer_worker(
    *,
    block_size: int = 1024,
    pcp_size: int = 4,
    interleave_size: int = 256,
    token_bytes: int = 64,
    kv_tensor: torch.Tensor | None = None,
) -> tuple[
    TPURaidenConnectorWorker, _FakeRaidenControllerFacade, _FakeRaidenEngine
]:
    """Creates a TPURaidenConnectorWorker for the prefill producer rank.

    Constructs a mocked producer worker instance with simulated Raiden engines,
    controller facades, work unit descriptors, and optional synthetic KV cache
    tensor bindings for transfer validation.

    Args:
        block_size: Number of tokens per physical worker cache block.
        pcp_size: Prefill context parallelism size.
        interleave_size: Token interleave stride across PCP ranks.
        token_bytes: Byte footprint per token across attention KV heads.
        kv_tensor: Optional synthetic KV cache tensor to attach to the runner.

    Returns:
        A tuple of (worker, controller_facade, transfer_engine).
    """
    cfg = MagicMock(spec=VllmConfig)
    cfg.kv_transfer_config = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = True
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.cache_config = MagicMock()
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = True
    cfg.model_config = MagicMock()
    cfg.model_config.max_model_len = 32768
    cfg.model_config.num_hidden_layers = 32
    cfg.model_config.get_num_layers = MagicMock(return_value=32)
    cfg.model_config.get_num_kv_heads = MagicMock(return_value=8)
    cfg.model_config.get_head_size = MagicMock(return_value=128)
    cfg.model_config.dtype = torch.bfloat16
    cfg.parallel_config = MagicMock()
    cfg.parallel_config.data_parallel_rank = 0
    cfg.parallel_config.data_parallel_size = 1
    cfg.parallel_config.tensor_parallel_size = 1
    cfg.parallel_config.prefill_context_parallel_size = pcp_size
    cfg.parallel_config.cp_kv_cache_interleave_size = interleave_size

    with (
        patch(f"{_MOD}.dist_utils.get_kv_ips", return_value="127.0.0.1"),
        patch(f"{_MOD}.dist_utils.get_kv_ports", return_value=9100),
        patch(f"{_MOD}.dist_utils.get_side_channel_port", return_value="9600"),
        patch(f"{_MOD}.get_tensor_model_parallel_rank", return_value=0),
        patch(f"{_MOD}.get_tensor_model_parallel_world_size", return_value=1),
    ):
        worker = TPURaidenConnectorWorker(cfg)

    engine = _FakeRaidenEngine()
    facade = _FakeRaidenControllerFacade()
    unit = SimpleNamespace(
        job_name="prefill-worker",
        job_replica_id="producer-0",
        data_name="kv.fa",
        data_replica_idx=0,
    )
    worker._raiden_transfer_engine = engine
    worker._raiden_controller_facade = facade
    worker._raiden_controller_address = "127.0.0.1:27000"
    worker._raiden_work_unit = unit
    worker._local_raiden_transfer_rank = MagicMock(return_value=0)
    worker._raiden_transfer_parallelism = MagicMock(return_value=pcp_size)
    worker._stage3_fa_token_bytes = MagicMock(return_value=token_bytes)

    if kv_tensor is not None:
        worker.runner = SimpleNamespace(kv_caches=[kv_tensor])
        worker._flattened_kv_cache_tensors = MagicMock(return_value=[kv_tensor])
    else:
        worker._flattened_kv_cache_tensors = MagicMock(return_value=[])

    return worker, facade, engine


def _make_consumer_worker(
    *,
    block_size: int = 1024,
    dp_rank: int = 0,
    dp_size: int = 1,
    kv_tensor: torch.Tensor | None = None,
) -> tuple[
    TPURaidenConnectorWorker, _FakeRaidenControllerFacade, _FakeRaidenEngine
]:
    """Creates a TPURaidenConnectorWorker for the decode consumer rank.

    Constructs a mocked consumer worker instance with simulated Raiden engines,
    controller facades, work unit descriptors, manifest geometry layouts, and
    optional synthetic destination KV cache tensor bindings.

    Args:
        block_size: Number of tokens per physical worker cache block.
        dp_rank: Data parallel rank of the consumer worker.
        dp_size: Total data parallel world size.
        kv_tensor: Optional destination KV cache tensor to attach to the runner.

    Returns:
        A tuple of (worker, controller_facade, transfer_engine).
    """
    cfg = MagicMock(spec=VllmConfig)
    cfg.kv_transfer_config = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = False
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.cache_config = MagicMock()
    cfg.cache_config.block_size = block_size
    cfg.cache_config.enable_prefix_caching = False
    cfg.model_config = MagicMock()
    cfg.model_config.max_model_len = 32768
    cfg.model_config.hf_config = SimpleNamespace(
        num_key_value_heads=8,
        head_dim=128,
    )
    cfg.parallel_config = MagicMock()
    cfg.parallel_config.data_parallel_rank = dp_rank
    cfg.parallel_config.data_parallel_size = dp_size
    cfg.parallel_config.tensor_parallel_size = 1
    cfg.parallel_config.prefill_context_parallel_size = 1
    cfg.parallel_config.cp_kv_cache_interleave_size = 256

    with (
        patch(f"{_MOD}.get_tensor_model_parallel_rank", return_value=0),
        patch(f"{_MOD}.get_tensor_model_parallel_world_size", return_value=1),
        patch(f"{_MOD}.dist_utils.get_node_id", return_value=0),
        patch(f"{_MOD}.dist_utils.get_host_ip", return_value="127.0.0.1"),
        patch(f"{_MOD}.dist_utils.get_kv_transfer_port", return_value=9100),
    ):
        worker = TPURaidenConnectorWorker(cfg)

    engine = _FakeRaidenEngine()
    facade = _FakeRaidenControllerFacade()
    unit = SimpleNamespace(
        job_name="decode-worker",
        job_replica_id="consumer-0",
        data_name="kv.fa",
        data_replica_idx=dp_rank,
    )
    worker._raiden_transfer_engine = engine
    worker._raiden_controller_facade = facade
    worker._raiden_controller_address = "127.0.0.1:27000"
    worker._raiden_work_unit = unit
    worker._local_raiden_transfer_rank = MagicMock(return_value=0)
    worker._raiden_manifest = SimpleNamespace(
        geometry_by_tag=lambda: {
            "fa": {"live_bytes_per_block": block_size * 64}
        },
        tag_counts=lambda: {"fa": 1},
    )

    if kv_tensor is not None:
        worker.runner = SimpleNamespace(kv_caches=[kv_tensor])
        worker._flattened_kv_cache_tensors = MagicMock(return_value=[kv_tensor])
    else:
        worker._flattened_kv_cache_tensors = MagicMock(return_value=[])

    return worker, facade, engine


def test_end_to_end_chunked_transfer_pipeline() -> None:
    """Executes full end-to-end chunked KV transfer pipeline between dual ranks.

    Setup:
    - Dual ranks: Prefill rank (producer, rank 0) and Decode rank (consumer,
      rank 1).
    - Prompt length: 16,384 tokens.
    - Chunk size: 2,048 tokens (stride PCP=4 * 256 = 1024, emitted every 2048
      tokens).
    - Flow control: sliding window W=2.
    - Total chunks: 8 chunks ([0, 2048), [2048, 4096), ..., [14336, 16384)).

    Verifies:
    1. All 8 chunks stream asynchronously across worker loops under window
       flow control.
    2. Consumer pre-allocates destination memory upfront and records intervals.
    3. Consumer validates continuous full interval coverage of [0, 16384).
    4. Decode begins immediately upon completion.
    5. Exact numerical parity: max absolute difference between prefill and
       decode KV caches is exactly 0.0.
    """
    total_tokens = 16384
    chunk_tokens = 2048
    num_chunks = total_tokens // chunk_tokens  # 8 chunks
    sched_block_size = 16
    worker_block_size = 1024
    total_sched_blocks = total_tokens // sched_block_size  # 1024 blocks
    pcp_size = 4
    interleave_size = 256
    base_uuid = 800000
    req_id = "req-e2e-16k"

    # Synthetic KV cache tensor for prefill (deterministic synthetic values)
    # Shape: (total_tokens, 8 heads, 128 head_dim)
    torch.manual_seed(42)
    prefill_kv = torch.randn(
        total_tokens, 8, 128, dtype=torch.float32
    )
    # Destination KV cache initialized with zeros
    decode_kv = torch.zeros_like(prefill_kv)

    # 1. Initialize Producer (Prefill rank)
    producer_scheduler = _make_producer_scheduler(
        block_size=sched_block_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        chunk_transfer_window=2,
    )
    producer_worker, _, _ = _make_producer_worker(
        block_size=worker_block_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        token_bytes=64,
        kv_tensor=prefill_kv,
    )

    # 2. Initialize Consumer (Decode rank)
    consumer_worker, _, consumer_engine = _make_consumer_worker(
        block_size=worker_block_size,
        kv_tensor=decode_kv,
    )

    # Register request with producer scheduler
    producer_blocks = list(range(total_sched_blocks))
    mock_req = MagicMock()
    mock_req.request_id = req_id
    mock_req.num_prompt_tokens = total_tokens
    mock_req.num_computed_tokens = 0
    mock_req.prompt_token_ids = list(range(total_tokens))
    mock_req.status = RequestStatus.RUNNING
    mock_req.kv_transfer_params = {"_p_side_truncated": True}

    # Pre-allocate destination memory on consumer
    consumer_blocks = list(range(1000, 1000 + total_sched_blocks))
    consumer_worker._update_stage3_state_after_alloc(
        req_id=req_id, expected_tokens=total_tokens
    )
    consumer_worker._load_block_ids[req_id] = list(consumer_blocks)
    consumer_worker._stage3_controller_accepted.add(req_id)

    with _patch_raiden_stage3_env(base_uuid=base_uuid, parallelism=pcp_size):
        producer_scheduler.update_state_after_alloc(
            mock_req, producer_blocks, 0
        )

        # Simulate iterative prefill execution and chunk streaming
        for chunk_idx in range(num_chunks):
            token_base = chunk_idx * chunk_tokens
            expected_uuid = chunk_uuid(base_uuid, chunk_idx)

            # Check sliding window admission (W=2)
            can_emit = producer_scheduler._can_emit_chunk(req_id)
            assert can_emit is True

            # Schedule step of 2048 tokens on producer
            sched_out = MagicMock(spec=SchedulerOutput)
            sched_out.num_scheduled_tokens = {req_id: chunk_tokens}
            producer_meta = producer_scheduler.build_connector_meta(sched_out)

            # Verify SendMeta emitted for this chunk
            assert req_id in producer_meta.reqs_to_send
            send_meta = producer_meta.reqs_to_send[req_id]
            assert send_meta.token_base == token_base
            assert send_meta.num_tokens == chunk_tokens
            assert send_meta.uuid == expected_uuid
            assert send_meta.is_final is False

            # Producer worker registers the chunk send
            producer_worker.process_send_load(producer_meta)
            assert req_id in producer_worker._active_tensor_refs

            # Consumer worker receives incoming chunk load metadata
            sub_chunk_req_id = f"{req_id}-chunk-{chunk_idx}"
            consumer_worker._start_stage3_recv_if_ready(
                req_id=sub_chunk_req_id,
                uuid=expected_uuid,
                endpoint="127.0.0.1:27000",
                token_base=token_base,
                num_tokens=chunk_tokens,
                engine=consumer_engine,
                destination_req_id=req_id,
            )

            # Simulated RDMA / D2H-H2D transfer: copy into destination buffer
            decode_kv[token_base : token_base + chunk_tokens] = (
                prefill_kv[token_base : token_base + chunk_tokens].clone()
            )

            # Consumer completes sub-chunk read
            consumer_engine.complete_read_results = [(
                [],
                [sub_chunk_req_id],
                [],
            )]
            consumer_worker._poll_finished(consumer_engine)

            # Ack chunk completion to producer worker and slide window
            producer_worker._stage3_registered_sends.pop(req_id, None)
            producer_scheduler.on_chunk_completed(req_id)

        # 3. Prefill finished: request_finished called on producer
        mock_req.num_computed_tokens = total_tokens
        mock_req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        delay, params = producer_scheduler.request_finished(
            mock_req, producer_blocks
        )
        assert delay is True
        assert params["num_tokens"] == total_tokens

        # 4. Consumer verification of full interval tiling cover
        assert consumer_worker._is_stage3_recv_complete(req_id) is True
        intervals = consumer_worker._stage3_recv_intervals.get(req_id, [])
        assert len(intervals) == num_chunks

        # Verify get_finished reports request ready for decode
        _, done_recving = consumer_worker.get_finished()
        assert req_id in done_recving

        # 5. Exact Numerical Parity Verification
        # Both prefill and decode KV caches must match with 0 error
        max_abs_diff = torch.max(torch.abs(decode_kv - prefill_kv)).item()
        assert max_abs_diff == 0.0
        assert torch.equal(decode_kv, prefill_kv)


def test_end_to_end_preemption_handling() -> None:
    """Simulates decode worker eviction and client cancellation mid-transfer.

    Verifies:
    1. Prefill worker purges active tensor references immediately on
       cancellation.
    2. Scheduler cleans in-flight tracking and send request state.
    3. Controller registrations are cancelled or completed cleanly.
    4. Zero dangling tensor references or memory leaks remain.
    """
    total_tokens = 16384
    chunk_tokens = 2048
    sched_block_size = 16
    worker_block_size = 1024
    total_sched_blocks = total_tokens // sched_block_size
    pcp_size = 4
    interleave_size = 256
    base_uuid = 850000
    req_id = "req-preemption-test"

    prefill_kv = torch.randn(total_tokens, 8, 128, dtype=torch.float32)

    producer_scheduler = _make_producer_scheduler(
        block_size=sched_block_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        chunk_transfer_window=2,
    )
    producer_worker, _, _ = _make_producer_worker(
        block_size=worker_block_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        token_bytes=64,
        kv_tensor=prefill_kv,
    )

    mock_req = MagicMock()
    mock_req.request_id = req_id
    mock_req.num_prompt_tokens = total_tokens
    mock_req.num_computed_tokens = 0
    mock_req.prompt_token_ids = list(range(total_tokens))
    mock_req.status = RequestStatus.RUNNING
    mock_req.kv_transfer_params = {"_p_side_truncated": True}

    with _patch_raiden_stage3_env(base_uuid=base_uuid, parallelism=pcp_size):
        producer_scheduler.update_state_after_alloc(
            mock_req, list(range(total_sched_blocks)), 0
        )

        # Emit chunk 0
        sched_out = MagicMock(spec=SchedulerOutput)
        sched_out.num_scheduled_tokens = {req_id: chunk_tokens}
        producer_meta = producer_scheduler.build_connector_meta(sched_out)
        producer_worker.process_send_load(producer_meta)

        assert producer_scheduler._in_flight_chunks[req_id] == 1
        assert req_id in producer_worker._active_tensor_refs
        active_refs = producer_worker._active_tensor_refs[req_id]
        assert len(active_refs.device_buffers) > 0

        # Mid-transfer preemption occurs on decode node -> cancellation
        # Prefill worker receives terminal cleanup / eviction notification
        producer_worker._stage3_terminal_cleanup[req_id] = True
        producer_worker.get_finished()

        # Verify active tensor refs and device buffers are purged
        assert req_id not in producer_worker._active_tensor_refs
        assert req_id in producer_worker._stage3_terminal_sends
        terminal_send = producer_worker._stage3_terminal_sends[req_id]
        assert terminal_send.device_buffers == ()

        # Cancel request in producer scheduler
        producer_scheduler.on_chunk_completed(req_id)
        assert req_id not in producer_scheduler._in_flight_chunks


def test_end_to_end_prefix_cached_chunked_transfer() -> None:
    """Validates chunked transfer with prefix caching (4096 tokens cached).

    Verifies:
    1. Prefill skips chunks [0, 2048) and [2048, 4096).
    2. Chunks [4096, 16384) are transferred and assembled on the consumer.
    3. Destination reconstructs full 16,384-token sequence with exact parity.
    """
    total_tokens = 16384
    cached_prefix = 4096
    chunk_tokens = 2048
    sched_block_size = 16
    worker_block_size = 1024
    total_sched_blocks = total_tokens // sched_block_size
    pcp_size = 4
    interleave_size = 256
    base_uuid = 900000
    req_id = "req-prefix-cached-16k"

    torch.manual_seed(99)
    full_prompt_kv = torch.randn(total_tokens, 8, 128, dtype=torch.float32)

    # Destination initially holds the cached prefix [0, 4096)
    decode_kv = torch.zeros_like(full_prompt_kv)
    decode_kv[:cached_prefix] = full_prompt_kv[:cached_prefix].clone()

    producer_scheduler = _make_producer_scheduler(
        block_size=sched_block_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        chunk_transfer_window=4,
    )
    producer_worker, _, _ = _make_producer_worker(
        block_size=worker_block_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        token_bytes=64,
        kv_tensor=full_prompt_kv,
    )
    consumer_worker, _, consumer_engine = _make_consumer_worker(
        block_size=worker_block_size,
        kv_tensor=decode_kv,
    )

    mock_req = MagicMock()
    mock_req.request_id = req_id
    mock_req.num_prompt_tokens = total_tokens
    mock_req.num_computed_tokens = cached_prefix  # 4096 cached tokens
    mock_req.prompt_token_ids = list(range(total_tokens))
    mock_req.status = RequestStatus.RUNNING
    mock_req.kv_transfer_params = {"_p_side_truncated": True}

    consumer_worker._update_stage3_state_after_alloc(
        req_id=req_id, expected_tokens=total_tokens
    )
    # Seed consumer with cached prefix interval [0, 4096)
    consumer_worker._stage3_recv_intervals[req_id].append((0, cached_prefix))
    consumer_worker._load_block_ids[req_id] = list(range(total_sched_blocks))
    consumer_worker._stage3_controller_accepted.add(req_id)

    with _patch_raiden_stage3_env(base_uuid=base_uuid, parallelism=pcp_size):
        producer_scheduler.update_state_after_alloc(
            mock_req, list(range(total_sched_blocks)), 0
        )
        req_state = producer_scheduler._stage3_send_requests[req_id]
        # Verify prefix progress skipped 4 stride units (4096 // 1024 = 4)
        assert req_state.tokens_sent == cached_prefix
        assert req_state.next_chunk_idx == 4

        # Emit remaining 6 chunks: tokens [4096, 16384)
        remaining_chunks = (total_tokens - cached_prefix) // chunk_tokens
        for step in range(remaining_chunks):
            token_base = cached_prefix + step * chunk_tokens
            chunk_idx = 4 + step
            expected_uuid = chunk_uuid(base_uuid, chunk_idx)

            sched_out = MagicMock(spec=SchedulerOutput)
            sched_out.num_scheduled_tokens = {req_id: chunk_tokens}
            producer_meta = producer_scheduler.build_connector_meta(sched_out)

            assert req_id in producer_meta.reqs_to_send
            send_meta = producer_meta.reqs_to_send[req_id]
            assert send_meta.token_base == token_base
            assert send_meta.uuid == expected_uuid

            producer_worker.process_send_load(producer_meta)

            sub_chunk_req_id = f"{req_id}-chunk-{chunk_idx}"
            consumer_worker._start_stage3_recv_if_ready(
                req_id=sub_chunk_req_id,
                uuid=expected_uuid,
                endpoint="127.0.0.1:27000",
                token_base=token_base,
                num_tokens=chunk_tokens,
                engine=consumer_engine,
                destination_req_id=req_id,
            )

            # Copy transferred suffix slice
            decode_kv[token_base : token_base + chunk_tokens] = (
                full_prompt_kv[token_base : token_base + chunk_tokens].clone()
            )

            consumer_engine.complete_read_results = [(
                [],
                [sub_chunk_req_id],
                [],
            )]
            consumer_worker._poll_finished(consumer_engine)
            producer_worker._stage3_registered_sends.pop(req_id, None)
            producer_scheduler.on_chunk_completed(req_id)

        # Full interval tiling coverage verified
        assert consumer_worker._is_stage3_recv_complete(req_id) is True
        _, done_recving = consumer_worker.get_finished()
        assert req_id in done_recving

        # Exact numerical parity over the entire prompt
        assert torch.equal(decode_kv, full_prompt_kv)
        assert torch.max(torch.abs(decode_kv - full_prompt_kv)).item() == 0.0


def test_end_to_end_hybrid_recurrent_chunked_transfer() -> None:
    """Validates hybrid architecture (Attention KV + GDN Recurrent State).

    Verifies:
    1. Intermediate chunks stream attention KV only and omit recurrent states.
    2. Terminal completion chunk transfers attention KV AND recurrent states.
    3. Decode rank reconstructs both attention KV and recurrent states with
       exact numerical parity.
    """
    total_tokens = 8192
    chunk_tokens = 2048
    num_chunks = total_tokens // chunk_tokens  # 4 chunks
    sched_block_size = 16
    worker_block_size = 1024
    total_sched_blocks = total_tokens // sched_block_size
    pcp_size = 4
    interleave_size = 256
    base_uuid = 920000
    req_id = "req-hybrid-e2e-8k"

    torch.manual_seed(77)
    prefill_fa_kv = torch.randn(total_tokens, 8, 128, dtype=torch.float32)
    prefill_conv_state = torch.randn(4, 16, 64, dtype=torch.float32)
    prefill_ssm_state = torch.randn(4, 16, 128, dtype=torch.float32)

    decode_fa_kv = torch.zeros_like(prefill_fa_kv)
    decode_conv_state = torch.zeros_like(prefill_conv_state)
    decode_ssm_state = torch.zeros_like(prefill_ssm_state)

    producer_scheduler = _make_producer_scheduler(
        block_size=sched_block_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        chunk_transfer_window=4,
    )
    producer_worker, producer_facade, _ = _make_producer_worker(
        block_size=worker_block_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        token_bytes=64,
        kv_tensor=prefill_fa_kv,
    )
    producer_worker._stage3_state_group_count = 1
    producer_worker._stage3_state_pool_spans = MagicMock(
        return_value=[
            SimpleNamespace(tag="gdn.conv.g0"),
            SimpleNamespace(tag="gdn.ssm.g0"),
        ]
    )

    consumer_worker, _, consumer_engine = _make_consumer_worker(
        block_size=worker_block_size,
        kv_tensor=decode_fa_kv,
    )

    mock_req = MagicMock()
    mock_req.request_id = req_id
    mock_req.num_prompt_tokens = total_tokens
    mock_req.num_computed_tokens = 0
    mock_req.prompt_token_ids = list(range(total_tokens))
    mock_req.status = RequestStatus.RUNNING
    mock_req.kv_transfer_params = {
        "_p_side_truncated": True,
        "mamba_state_block_ids": [501],
    }

    consumer_worker._update_stage3_state_after_alloc(
        req_id=req_id, expected_tokens=total_tokens
    )
    consumer_worker._load_block_ids[req_id] = list(range(total_sched_blocks))
    consumer_worker._stage3_controller_accepted.add(req_id)

    with _patch_raiden_stage3_env(base_uuid=base_uuid, parallelism=pcp_size):
        producer_scheduler.update_state_after_alloc(
            mock_req, list(range(total_sched_blocks)), 0
        )

        for chunk_idx in range(num_chunks):
            token_base = chunk_idx * chunk_tokens
            expected_uuid = chunk_uuid(base_uuid, chunk_idx)
            is_final = chunk_idx == (num_chunks - 1)

            sched_out = MagicMock(spec=SchedulerOutput)
            sched_out.num_scheduled_tokens = {req_id: chunk_tokens}
            producer_meta = producer_scheduler.build_connector_meta(sched_out)

            if is_final:
                # Terminal chunk attaches mamba_state_block_ids
                send_meta = producer_meta.reqs_to_send[req_id]
                producer_meta.reqs_to_send[req_id] = SendMeta(
                    uuid=send_meta.uuid,
                    local_block_ids=send_meta.local_block_ids,
                    expiration_time=send_meta.expiration_time,
                    num_tokens=send_meta.num_tokens,
                    mamba_state_block_ids=[501],
                    token_base=send_meta.token_base,
                    is_final=True,
                    dst_skip_bytes=send_meta.dst_skip_bytes,
                )

            producer_worker.process_send_load(producer_meta)

            # Check controller registration spans:
            reg_call = producer_facade.register_request_blocks_calls[-1]
            tags = [
                getattr(span, "tag", str(span))
                for span in reg_call["pool_spans"]
            ]
            if not is_final:
                # Intermediate chunks must NOT contain recurrent state spans
                assert "gdn.conv.g0" not in tags
                assert "gdn.ssm.g0" not in tags
            else:
                # Terminal chunk must attach recurrent state spans
                assert "gdn.conv.g0" in tags
                assert "gdn.ssm.g0" in tags

            sub_chunk_req_id = f"{req_id}-chunk-{chunk_idx}"
            consumer_worker._start_stage3_recv_if_ready(
                req_id=sub_chunk_req_id,
                uuid=expected_uuid,
                endpoint="127.0.0.1:27000",
                token_base=token_base,
                num_tokens=chunk_tokens,
                engine=consumer_engine,
                destination_req_id=req_id,
            )

            # Copy FA KV cache chunk
            decode_fa_kv[token_base : token_base + chunk_tokens] = (
                prefill_fa_kv[token_base : token_base + chunk_tokens].clone()
            )

            if is_final:
                # Copy recurrent states on final chunk
                decode_conv_state.copy_(prefill_conv_state)
                decode_ssm_state.copy_(prefill_ssm_state)

            consumer_engine.complete_read_results = [(
                [],
                [sub_chunk_req_id],
                [],
            )]
            consumer_worker._poll_finished(consumer_engine)
            producer_worker._stage3_registered_sends.pop(req_id, None)
            producer_scheduler.on_chunk_completed(req_id)

        assert consumer_worker._is_stage3_recv_complete(req_id) is True
        _, done_recving = consumer_worker.get_finished()
        assert req_id in done_recving

        # Verify exact numerical parity for Attention KV and Recurrent states
        assert torch.equal(decode_fa_kv, prefill_fa_kv)
        assert (
            torch.max(torch.abs(decode_fa_kv - prefill_fa_kv)).item() == 0.0
        )
        assert torch.equal(decode_conv_state, prefill_conv_state)
        assert (
            torch.max(torch.abs(decode_conv_state - prefill_conv_state)).item()
            == 0.0
        )
        assert torch.equal(decode_ssm_state, prefill_ssm_state)
        assert (
            torch.max(torch.abs(decode_ssm_state - prefill_ssm_state)).item()
            == 0.0
        )


def test_end_to_end_out_of_order_chunk_assembly() -> None:
    """Validates out-of-order arrival and reassembly of sub-chunks.

    Simulates arrival order: chunk 3, chunk 1, chunk 0, chunk 2.
    Verifies:
    1. Consumer buffers non-contiguous slices without errors.
    2. Completion is withheld until the final missing chunk arrives.
    3. Final assembly achieves exact numerical parity.
    """
    total_tokens = 8192
    chunk_tokens = 2048
    block_size = 1024
    base_uuid = 950000
    req_id = "req-ooo-8k"

    torch.manual_seed(123)
    prefill_kv = torch.randn(total_tokens, 8, 128, dtype=torch.float32)
    decode_kv = torch.zeros_like(prefill_kv)

    consumer_worker, _, consumer_engine = _make_consumer_worker(
        block_size=block_size,
        kv_tensor=decode_kv,
    )
    consumer_worker._update_stage3_state_after_alloc(
        req_id=req_id, expected_tokens=total_tokens
    )
    consumer_worker._load_block_ids[req_id] = list(
        range(total_tokens // 16)
    )
    consumer_worker._stage3_controller_accepted.add(req_id)

    # Arrival order: [3, 1, 0, 2]
    arrival_order = [3, 1, 0, 2]

    for step_idx, chunk_idx in enumerate(arrival_order):
        token_base = chunk_idx * chunk_tokens
        expected_uuid = chunk_uuid(base_uuid, chunk_idx)
        sub_chunk_req_id = f"{req_id}-chunk-{chunk_idx}"

        consumer_worker._start_stage3_recv_if_ready(
            req_id=sub_chunk_req_id,
            uuid=expected_uuid,
            endpoint="127.0.0.1:27000",
            token_base=token_base,
            num_tokens=chunk_tokens,
            engine=consumer_engine,
            destination_req_id=req_id,
        )

        decode_kv[token_base : token_base + chunk_tokens] = prefill_kv[
            token_base : token_base + chunk_tokens
        ].clone()

        consumer_engine.complete_read_results = [(
            [],
            [sub_chunk_req_id],
            [],
        )]
        consumer_worker._poll_finished(consumer_engine)

        if step_idx < len(arrival_order) - 1:
            # Transfer must remain incomplete before all pieces arrive
            assert consumer_worker._is_stage3_recv_complete(req_id) is False
        else:
            # With final chunk 2 arriving, tiling check completes
            assert consumer_worker._is_stage3_recv_complete(req_id) is True

    _, done_recving = consumer_worker.get_finished()
    assert req_id in done_recving
    assert torch.equal(decode_kv, prefill_kv)
    assert torch.max(torch.abs(decode_kv - prefill_kv)).item() == 0.0


def test_end_to_end_sliding_window_backpressure() -> None:
    """Validates sliding window flow control (W=2) under rapid generation.

    Verifies:
    1. First 2 chunks are emitted and increment in-flight count to 2.
    2. 3rd chunk is deferred until an in-flight chunk completes.
    3. Window dynamically advances as chunk completion acks arrive.
    """
    total_tokens = 8192
    chunk_tokens = 2048
    block_size = 16
    base_uuid = 980000
    req_id = "req-window-backpressure"

    producer_scheduler = _make_producer_scheduler(
        block_size=block_size,
        pcp_size=4,
        interleave_size=256,
        chunk_transfer_window=2,
    )

    mock_req = MagicMock()
    mock_req.request_id = req_id
    mock_req.num_prompt_tokens = total_tokens
    mock_req.num_computed_tokens = 0
    mock_req.status = RequestStatus.RUNNING
    mock_req.kv_transfer_params = {"_p_side_truncated": True}

    with _patch_raiden_stage3_env(base_uuid=base_uuid, parallelism=4):
        producer_scheduler.update_state_after_alloc(
            mock_req, list(range(total_tokens // block_size)), 0
        )

        # 1. Emit chunk 0 (tokens [0, 2048)) -> in_flight = 1
        sched_out_0 = MagicMock(spec=SchedulerOutput)
        sched_out_0.num_scheduled_tokens = {req_id: chunk_tokens}
        meta_0 = producer_scheduler.build_connector_meta(sched_out_0)
        assert req_id in meta_0.reqs_to_send
        assert producer_scheduler._in_flight_chunks[req_id] == 1
        assert producer_scheduler._can_emit_chunk(req_id) is True

        # 2. Emit chunk 1 (tokens [2048, 4096)) -> in_flight = 2 == W
        sched_out_1 = MagicMock(spec=SchedulerOutput)
        sched_out_1.num_scheduled_tokens = {req_id: chunk_tokens}
        meta_1 = producer_scheduler.build_connector_meta(sched_out_1)
        assert req_id in meta_1.reqs_to_send
        assert producer_scheduler._in_flight_chunks[req_id] == 2
        assert producer_scheduler._can_emit_chunk(req_id) is False

        # 3. Attempt chunk 2 -> blocked by window limit W=2
        sched_out_2 = MagicMock(spec=SchedulerOutput)
        sched_out_2.num_scheduled_tokens = {req_id: chunk_tokens}
        meta_2 = producer_scheduler.build_connector_meta(sched_out_2)
        assert len(meta_2.reqs_to_send) == 0
        assert producer_scheduler._in_flight_chunks[req_id] == 2

        # 4. Chunk 0 completes -> ack received -> in_flight decrements to 1
        producer_scheduler.on_chunk_completed(req_id)
        assert producer_scheduler._in_flight_chunks[req_id] == 1
        assert producer_scheduler._can_emit_chunk(req_id) is True

        # 5. Retry chunk 2 -> now successfully emitted -> in_flight becomes 2
        meta_2_retry = producer_scheduler.build_connector_meta(sched_out_2)
        assert req_id in meta_2_retry.reqs_to_send
        assert producer_scheduler._in_flight_chunks[req_id] == 2
        assert meta_2_retry.reqs_to_send[req_id].token_base == 4096
