# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.pcp import \
    all_gather_equal_tokens as _pcp_all_gather_equal_tokens
from vllm_torchtpu.distributed.pcp import all_reduce_sum as _pcp_all_reduce_sum
from vllm_torchtpu.distributed.pcp import get_pcp_rank as _get_native_pcp_rank
from vllm_torchtpu.distributed.pcp import \
    get_pcp_world_size as _get_native_pcp_world_size
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import \
    apply_pcp_rank_major_token_order as _apply_pcp_rank_major_token_order
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import \
    build_pcp_logits_indices as _build_pcp_logits_indices
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import \
    pcp_local_token_counts
from vllm_torchtpu.layers.core.sequence_layout import (
    PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL, AllSequenceLayoutPlanner,
    SequenceLayoutDescriptor, SequenceLayoutKind, SequenceLayoutPlan,
    _first_ge)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR = SequenceLayoutDescriptor(
    kind=SequenceLayoutKind.PARTIAL,
    protocol=PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
    version=1,
)


def _get_pcp_streaming_q_block_size() -> int:
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import \
        PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE

    return PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE


class PcpSequenceLayoutMode(Enum):
    DISABLED = "DISABLED"
    STREAMING = "STREAMING"


@dataclass(frozen=True)
class PcpRequestSpan:
    computed_tokens: int
    scheduled_tokens: int
    prompt_tokens: int
    absolute_query_start: int

    @property
    def crosses_prompt_boundary(self) -> bool:
        return (self.scheduled_tokens > 0
                and self.computed_tokens < self.prompt_tokens
                and self.computed_tokens + self.scheduled_tokens
                > self.prompt_tokens)


@dataclass(frozen=True)
class PcpSequenceLayoutDecision:
    mode: PcpSequenceLayoutMode
    spans: tuple[PcpRequestSpan, ...]
    pcp_size: int = 1
    interleave_size: int = 1
    pcp_alignment: int | None = None
    absolute_query_start_offsets_per_req: np.ndarray | None = None
    token_owner_start_offsets_per_req: np.ndarray | None = None
    local_token_counts: np.ndarray | None = None
    local_required_tokens: int | None = None
    local_padded_tokens: int | None = None
    global_padded_tokens: int | None = None

    @property
    def enabled(self) -> bool:
        return self.mode is not PcpSequenceLayoutMode.DISABLED


class PcpSequenceLayoutEligibility:

    _BOUNDARY_MESSAGE = (
        "PCP partial sequence layout does not support prompt/decode "
        "boundary-crossing schedules yet.")

    def __init__(
        self,
        *,
        pcp_size: int = 1,
        interleave_size: int = 1,
        dcp_size: int = 1,
        pipeline_parallel_size: int = 1,
        async_scheduling: bool = False,
        is_kv_producer: bool | None = None,
    ):
        self.pcp_size = pcp_size
        self.interleave_size = interleave_size
        self.dcp_size = dcp_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.async_scheduling = bool(async_scheduling)
        self.is_kv_producer = is_kv_producer

    @classmethod
    def from_vllm_config(cls,
                         vllm_config: Any) -> "PcpSequenceLayoutEligibility":
        if vllm_config is None:
            return cls()

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        kv_transfer_config = vllm_config.kv_transfer_config

        is_kv_producer = None
        if kv_transfer_config is not None:
            is_kv_producer = kv_transfer_config.is_kv_producer

        return cls(
            pcp_size=parallel_config.prefill_context_parallel_size,
            interleave_size=parallel_config.cp_kv_cache_interleave_size,
            dcp_size=parallel_config.decode_context_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            async_scheduling=bool(scheduler_config.async_scheduling),
            is_kv_producer=is_kv_producer,
        )

    @property
    def enabled(self) -> bool:
        return self.pcp_size > 1

    def evaluate_runner_chunk(
        self,
        *,
        input_batch: Any,
        scheduler_output: Any,
        start_index: int,
        num_reqs: int,
        num_scheduled_tokens_per_req: np.ndarray,
        absolute_query_start_offsets_per_req: np.ndarray | None = None,
        num_tokens_paddings: Sequence[int] | None = None,
        max_num_tokens: int | None = None,
        dp_target_bucket: int | None = None,
    ) -> PcpSequenceLayoutDecision:
        spans = self._build_request_spans(
            input_batch=input_batch,
            scheduler_output=scheduler_output,
            start_index=start_index,
            num_reqs=num_reqs,
            num_scheduled_tokens_per_req=num_scheduled_tokens_per_req,
            absolute_query_start_offsets_per_req=(
                absolute_query_start_offsets_per_req),
        )

        if not self.enabled or not any(span.scheduled_tokens > 0
                                       for span in spans):
            return PcpSequenceLayoutDecision(
                mode=PcpSequenceLayoutMode.DISABLED,
                spans=spans,
                pcp_size=self.pcp_size,
                interleave_size=self.interleave_size,
            )

        if any(span.crosses_prompt_boundary for span in spans):
            raise NotImplementedError(self._BOUNDARY_MESSAGE)

        self._validate_execution_support()
        return self._evaluate_streaming_query(
            spans,
            num_tokens_paddings=num_tokens_paddings,
            max_num_tokens=max_num_tokens,
            dp_target_bucket=dp_target_bucket,
        )

    @staticmethod
    def require_mode(
        decision: PcpSequenceLayoutDecision,
        expected: PcpSequenceLayoutMode,
    ) -> None:
        if decision.mode is not expected:
            raise NotImplementedError(
                f"PCP sequence layout mode {decision.mode.value} cannot be "
                f"used where {expected.value} is required.")

    def _build_request_spans(
        self,
        *,
        input_batch: Any,
        scheduler_output: Any,
        start_index: int,
        num_reqs: int,
        num_scheduled_tokens_per_req: np.ndarray,
        absolute_query_start_offsets_per_req: np.ndarray | None,
    ) -> tuple[PcpRequestSpan, ...]:
        scheduled = np.asarray(num_scheduled_tokens_per_req, dtype=np.int64)
        if scheduled.ndim != 1:
            raise ValueError(
                "num_scheduled_tokens_per_req must be a 1D array.")
        if scheduled.size < num_reqs:
            raise ValueError(
                "num_scheduled_tokens_per_req must cover every request in "
                "the chunk.")

        absolute_query_starts = None
        if absolute_query_start_offsets_per_req is not None:
            absolute_query_starts = np.asarray(
                absolute_query_start_offsets_per_req, dtype=np.int64)
            if (absolute_query_starts.ndim != 1
                    or absolute_query_starts.size < num_reqs):
                raise ValueError(
                    "absolute_query_start_offsets_per_req must cover every "
                    "request in the chunk.")

        spans: list[PcpRequestSpan] = []
        req_ids = input_batch.req_ids[start_index:start_index + num_reqs]
        for chunk_offset, req_id in enumerate(req_ids):
            if req_id is None:
                continue
            req_index = input_batch.req_id_to_index[req_id]
            scheduled_tokens = int(scheduled[chunk_offset])
            scheduler_tokens = scheduler_output.num_scheduled_tokens.get(
                req_id)
            if scheduler_tokens is not None:
                scheduled_tokens = int(scheduler_tokens)
            computed_tokens = int(
                input_batch.num_computed_tokens_cpu[req_index])
            absolute_query_start = (int(absolute_query_starts[chunk_offset])
                                    if absolute_query_starts is not None else
                                    computed_tokens)
            spans.append(
                PcpRequestSpan(
                    computed_tokens=computed_tokens,
                    scheduled_tokens=scheduled_tokens,
                    prompt_tokens=int(
                        input_batch.num_prompt_tokens[req_index]),
                    absolute_query_start=absolute_query_start,
                ))
        return tuple(spans)

    def _validate_execution_support(self) -> None:
        if self.is_kv_producer is False:
            raise NotImplementedError(
                "PCP partial sequence layout does not support KV "
                "consumer/decode workers yet. Disable PCP on decode workers.")
        if self.dcp_size > 1:
            raise NotImplementedError(
                "PCP partial sequence layout does not support DCP yet.")
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "PCP partial sequence layout does not support pipeline "
                "parallelism yet.")
        if self.interleave_size <= 0:
            raise ValueError("PCP partial sequence layout requires "
                             "cp_kv_cache_interleave_size > 0.")

    def _evaluate_streaming_query(
        self,
        spans: tuple[PcpRequestSpan, ...],
        *,
        num_tokens_paddings: Sequence[int] | None,
        max_num_tokens: int | None,
        dp_target_bucket: int | None,
    ) -> PcpSequenceLayoutDecision:
        if any(span.scheduled_tokens <= 0 for span in spans):
            raise NotImplementedError(
                "PCP streaming requires scheduled query tokens for every "
                "request in the chunk.")

        q_block_size = _get_pcp_streaming_q_block_size()
        if q_block_size % self.interleave_size != 0:
            raise NotImplementedError(
                "PCP streaming requires q_block_size to be a "
                "multiple of cp_kv_cache_interleave_size, got "
                f"{q_block_size=} "
                f"cp_kv_cache_interleave_size={self.interleave_size}.")

        q_lens = np.asarray([span.scheduled_tokens for span in spans],
                            dtype=np.int32)
        absolute_query_starts = np.asarray(
            [span.absolute_query_start for span in spans], dtype=np.int64)
        # Assign the current batch's request-major query rows as one flat token
        # stream. This is exactly query_start_loc[:-1]: ownership is local to
        # this runner chunk, while attention positions and KV-cache slots keep
        # using the independent request-absolute coordinates above.
        token_owner_starts = np.zeros(q_lens.shape, dtype=np.int64)
        if q_lens.size > 1:
            np.cumsum(q_lens[:-1], dtype=np.int64, out=token_owner_starts[1:])
        local_counts = pcp_local_token_counts(
            q_lens,
            self.pcp_size,
            self.interleave_size,
            token_owner_start_offsets_per_req=token_owner_starts,
        )
        local_required_tokens = int(local_counts.max(initial=0))

        local_padded_tokens = None
        global_padded_tokens = None
        if num_tokens_paddings is not None:
            paddings = sorted(int(padding) for padding in num_tokens_paddings)
            local_padded_tokens = _first_ge(paddings, local_required_tokens)
            if (dp_target_bucket is not None
                    and dp_target_bucket > local_padded_tokens):
                local_padded_tokens = int(dp_target_bucket)
            if (max_num_tokens is not None
                    and local_padded_tokens > int(max_num_tokens)):
                raise ValueError(
                    "PCP local padded token length exceeds runner max token "
                    f"buffer: {local_padded_tokens=} "
                    f"max_num_tokens={int(max_num_tokens)}.")
            global_padded_tokens = local_padded_tokens * self.pcp_size

        return PcpSequenceLayoutDecision(
            mode=PcpSequenceLayoutMode.STREAMING,
            spans=spans,
            pcp_size=self.pcp_size,
            interleave_size=self.interleave_size,
            pcp_alignment=None,
            absolute_query_start_offsets_per_req=absolute_query_starts,
            token_owner_start_offsets_per_req=token_owner_starts,
            local_token_counts=local_counts,
            local_required_tokens=local_required_tokens,
            local_padded_tokens=local_padded_tokens,
            global_padded_tokens=global_padded_tokens,
        )


@dataclass(frozen=True)
class PcpPreparedBatch:
    local_token_slice: slice
    padded_total_num_scheduled_tokens: int
    local_total_num_scheduled_tokens: int
    local_padded_total_num_scheduled_tokens: int
    logits_indices_cpu: torch.Tensor
    logits_local_indices_cpu: torch.Tensor
    logits_owner_mask_cpu: torch.Tensor
    packed_to_request_major_token_indices: np.ndarray
    request_major_to_packed_token_indices: np.ndarray


class PcpSequenceLayoutPlanner:

    def __init__(self, eligibility: PcpSequenceLayoutEligibility):
        self.eligibility = eligibility
        self._all_planner = AllSequenceLayoutPlanner()

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "PcpSequenceLayoutPlanner":
        return cls(PcpSequenceLayoutEligibility.from_vllm_config(vllm_config))

    @property
    def enabled(self) -> bool:
        return self.eligibility.enabled

    @property
    def requires_backend_preinit(self) -> bool:
        return self.enabled

    @property
    def backend_preinit_world_size(self) -> int:
        return self.eligibility.pcp_size

    @property
    def uses_selected_logits_hidden_states(self) -> bool:
        return self.enabled

    def reserve_host_token_capacity(self, runner: Any,
                                    required_num_tokens: int) -> None:
        if self.enabled:
            _ensure_host_token_buffer_capacity(runner, required_num_tokens)

    def prepare_real(
        self,
        *,
        runner: Any,
        scheduler_output: Any,
        start_index: int,
        num_reqs: int,
        num_scheduled_tokens_per_req: np.ndarray,
        total_num_scheduled_tokens: int,
        use_max_model_len: bool,
        target_num_reqs: int,
        padded_num_reqs: int,
    ) -> SequenceLayoutPlan:
        decision = self.eligibility.evaluate_runner_chunk(
            input_batch=runner.input_batch,
            scheduler_output=scheduler_output,
            start_index=start_index,
            num_reqs=num_reqs,
            num_scheduled_tokens_per_req=num_scheduled_tokens_per_req,
            num_tokens_paddings=runner.num_tokens_paddings,
            max_num_tokens=runner.max_num_tokens,
            dp_target_bucket=runner._dp_target_bucket,
        )
        if decision.mode is PcpSequenceLayoutMode.DISABLED:
            return self._all_planner.prepare_real(
                runner=runner,
                scheduler_output=scheduler_output,
                start_index=start_index,
                num_reqs=num_reqs,
                num_scheduled_tokens_per_req=num_scheduled_tokens_per_req,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
                use_max_model_len=use_max_model_len,
                target_num_reqs=target_num_reqs,
                padded_num_reqs=padded_num_reqs,
            )
        prepared = prepare_pcp_sequence_layout(
            runner,
            scheduler_output,
            start_index,
            num_reqs,
            num_scheduled_tokens_per_req,
            int(total_num_scheduled_tokens),
            use_max_model_len,
            target_num_reqs,
            padded_num_reqs,
            decision=decision,
        )
        return SequenceLayoutPlan(
            descriptor=PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
            token_slice=prepared.local_token_slice,
            global_num_tokens=int(total_num_scheduled_tokens),
            global_padded_num_tokens=(
                prepared.padded_total_num_scheduled_tokens),
            local_num_tokens=prepared.local_total_num_scheduled_tokens,
            local_padded_num_tokens=(
                prepared.local_padded_total_num_scheduled_tokens),
            logits_indices_cpu=prepared.logits_indices_cpu,
            logits_local_indices_cpu=prepared.logits_local_indices_cpu,
            logits_owner_mask_cpu=prepared.logits_owner_mask_cpu,
            requires_hidden_state_gather=True,
            _packed_to_request_major_token_indices=(
                prepared.packed_to_request_major_token_indices),
            _request_major_to_packed_token_indices=(
                prepared.request_major_to_packed_token_indices),
        )

    def prepare_dummy(
        self,
        *,
        num_tokens: int,
        num_reqs: int,
        kv_cache_initialized: bool,
    ) -> SequenceLayoutPlan:
        del kv_cache_initialized
        if not self.enabled:
            return self._all_planner.prepare_dummy(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                kv_cache_initialized=False,
            )
        num_tokens = int(num_tokens)
        return SequenceLayoutPlan(
            descriptor=PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
            token_slice=slice(0, num_tokens),
            global_num_tokens=num_tokens * self.eligibility.pcp_size,
            global_padded_num_tokens=num_tokens * self.eligibility.pcp_size,
            local_num_tokens=num_tokens,
            local_padded_num_tokens=num_tokens,
            requires_hidden_state_gather=True,
        )

    def finalize_hidden_states(
        self,
        hidden_states: torch.Tensor,
        plan: SequenceLayoutPlan | None,
    ) -> torch.Tensor:
        if plan is None or not plan.requires_hidden_state_gather:
            return hidden_states
        return _pcp_all_gather_equal_tokens(hidden_states, dim=0)

    def maybe_select_logits_hidden_states(
        self,
        hidden_states: torch.Tensor,
        plan: SequenceLayoutPlan | None,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        del logits_indices
        if plan is None:
            return None
        if (plan.logits_local_indices_cpu is None
                or plan.logits_owner_mask_cpu is None):
            return None

        local_indices = plan.logits_local_indices_cpu.to(hidden_states.device,
                                                         non_blocking=True)
        owner_mask = plan.logits_owner_mask_cpu.to(hidden_states.device,
                                                   non_blocking=True)
        selected = torch.index_select(hidden_states, 0, local_indices)
        selected = torch.where(owner_mask.unsqueeze(1), selected,
                               torch.zeros_like(selected))
        return _pcp_all_reduce_sum(selected)


def _ensure_host_token_buffer_capacity(runner: Any,
                                       required_num_tokens: int) -> None:
    """Grow CPU-only token staging buffers for partial sequence layouts."""
    required_num_tokens = int(required_num_tokens)
    if required_num_tokens <= 0:
        return

    if len(runner.arange_np) < required_num_tokens:
        runner.arange_np = np.arange(required_num_tokens, dtype=np.int64)

    current_num_tokens = int(runner.input_ids_cpu.shape[0])
    if current_num_tokens >= required_num_tokens:
        return

    old_input_ids_cpu = runner.input_ids_cpu
    runner.input_ids_cpu = torch.zeros(required_num_tokens,
                                       dtype=old_input_ids_cpu.dtype,
                                       device=old_input_ids_cpu.device)
    runner.input_ids_cpu[:current_num_tokens] = old_input_ids_cpu

    old_positions_cpu = runner.positions_cpu
    runner.positions_cpu = torch.zeros(required_num_tokens,
                                       dtype=old_positions_cpu.dtype,
                                       device=old_positions_cpu.device)
    runner.positions_cpu[:current_num_tokens] = old_positions_cpu
    runner.positions_np = runner.positions_cpu.numpy()

    if runner.supports_mm_inputs and hasattr(runner, "is_mm_embed_cpu"):
        old_is_mm_embed_cpu = runner.is_mm_embed_cpu
        runner.is_mm_embed_cpu = torch.zeros(required_num_tokens,
                                             dtype=old_is_mm_embed_cpu.dtype,
                                             device=old_is_mm_embed_cpu.device)
        runner.is_mm_embed_cpu[:current_num_tokens] = old_is_mm_embed_cpu

    if runner.uses_mrope:
        old_mrope_positions = runner.mrope_positions
        new_mrope_positions = runner._make_buffer(3,
                                                  required_num_tokens + 1,
                                                  dtype=torch.int32)
        copy_len = min(current_num_tokens + 1,
                       int(old_mrope_positions.cpu.shape[1]),
                       required_num_tokens + 1)
        new_mrope_positions.cpu[:, :
                                copy_len] = old_mrope_positions.cpu[:, :
                                                                    copy_len]
        runner.mrope_positions = new_mrope_positions


def _debug_pcp_layout_window(
    runner: Any,
    *,
    pcp_rank: int,
    local_token_slice: slice,
    local_padded_tokens: int,
    logits_indices: np.ndarray,
) -> dict[str, Any]:
    local_ids = runner.input_ids_cpu[local_token_slice]
    local_positions = runner.positions_cpu[local_token_slice]
    selected_windows = []
    rank_start = pcp_rank * local_padded_tokens
    rank_end = rank_start + local_padded_tokens
    for index in logits_indices.tolist():
        index = int(index)
        if rank_start <= index < rank_end:
            local_offset = index - rank_start
            start = max(0, local_offset - 8)
            end = min(local_padded_tokens, local_offset + 9)
            selected_windows.append({
                "global_index":
                index,
                "local_offset":
                local_offset,
                "ids":
                local_ids[start:end].tolist(),
                "positions":
                local_positions[start:end].tolist(),
            })
    head = min(40, local_padded_tokens)
    return {
        "head_ids": local_ids[:head].tolist(),
        "head_positions": local_positions[:head].tolist(),
        "selected_windows": selected_windows,
    }


def prepare_pcp_sequence_layout(
    runner: Any,
    scheduler_output: Any,
    start_index: int,
    num_reqs: int,
    num_scheduled_tokens_per_req: np.ndarray,
    total_num_scheduled_tokens: int,
    use_max_model_len: bool,
    target_num_reqs: int,
    padded_num_reqs: int,
    get_native_pcp_rank: Any | None = None,
    get_native_pcp_world_size: Any | None = None,
    decision: PcpSequenceLayoutDecision | None = None,
) -> PcpPreparedBatch:
    """Prepare native PCP streaming host state for a real request."""
    del total_num_scheduled_tokens, use_max_model_len, target_num_reqs
    if decision is None:
        decision = PcpSequenceLayoutEligibility.from_vllm_config(
            runner.vllm_config).evaluate_runner_chunk(
                input_batch=runner.input_batch,
                scheduler_output=scheduler_output,
                start_index=start_index,
                num_reqs=num_reqs,
                num_scheduled_tokens_per_req=num_scheduled_tokens_per_req,
                num_tokens_paddings=runner.num_tokens_paddings,
                max_num_tokens=runner.max_num_tokens,
                dp_target_bucket=runner._dp_target_bucket,
            )
    PcpSequenceLayoutEligibility.require_mode(decision,
                                              PcpSequenceLayoutMode.STREAMING)

    pcp_size = decision.pcp_size
    interleave_size = decision.interleave_size
    if pcp_size <= 1:
        raise ValueError(
            "PCP streaming requires prefill_context_parallel_size > 1.")

    if get_native_pcp_rank is None:
        get_native_pcp_rank = _get_native_pcp_rank
    if get_native_pcp_world_size is None:
        get_native_pcp_world_size = _get_native_pcp_world_size

    native_pcp_world_size = get_native_pcp_world_size()
    if native_pcp_world_size != pcp_size:
        raise RuntimeError("Native PCP group size does not match "
                           "prefill_context_parallel_size: "
                           f"{native_pcp_world_size=} {pcp_size=}.")
    pcp_rank = get_native_pcp_rank()
    if not 0 <= pcp_rank < pcp_size:
        raise RuntimeError("Native PCP rank is out of range: "
                           f"{pcp_rank=} {pcp_size=}.")

    absolute_query_start_offsets_per_req = (
        decision.absolute_query_start_offsets_per_req)
    token_owner_start_offsets_per_req = (
        decision.token_owner_start_offsets_per_req)
    local_counts = decision.local_token_counts
    local_padded_tokens = decision.local_padded_tokens
    padded_total_num_scheduled_tokens = decision.global_padded_tokens
    assert absolute_query_start_offsets_per_req is not None
    assert token_owner_start_offsets_per_req is not None
    assert local_counts is not None
    assert local_padded_tokens is not None
    assert padded_total_num_scheduled_tokens is not None

    _ensure_host_token_buffer_capacity(runner,
                                       padded_total_num_scheduled_tokens)

    mrope_slice = None
    if runner.uses_mrope:
        mrope_slice = runner.mrope_positions.cpu[:, :(
            padded_total_num_scheduled_tokens)].numpy()
    request_major_to_packed_token_indices = _apply_pcp_rank_major_token_order(
        runner.input_ids_cpu[:padded_total_num_scheduled_tokens].numpy(),
        runner.positions_np[:padded_total_num_scheduled_tokens],
        num_scheduled_tokens_per_req,
        pcp_size,
        interleave_size,
        padded_total_num_scheduled_tokens,
        mrope_slice,
        token_owner_start_offsets_per_req=token_owner_start_offsets_per_req,
    )
    packed_to_request_major_token_indices = np.full(
        padded_total_num_scheduled_tokens, -1, dtype=np.int64)
    packed_to_request_major_token_indices[
        request_major_to_packed_token_indices] = np.arange(
            request_major_to_packed_token_indices.size, dtype=np.int64)

    local_padded_total_num_scheduled_tokens = (
        padded_total_num_scheduled_tokens // pcp_size)
    local_total_num_scheduled_tokens = int(local_counts[pcp_rank])
    local_start = pcp_rank * local_padded_total_num_scheduled_tokens
    local_end = local_start + local_padded_total_num_scheduled_tokens
    local_token_slice = slice(local_start, local_end)

    pcp_logits_indices = _build_pcp_logits_indices(
        num_scheduled_tokens_per_req,
        pcp_size,
        interleave_size,
        padded_total_num_scheduled_tokens,
        token_owner_start_offsets_per_req=token_owner_start_offsets_per_req,
    ).astype(np.int32)
    logits_indices_cpu = torch.full((padded_num_reqs, ),
                                    -1,
                                    dtype=torch.int32,
                                    device="cpu")
    logits_indices_cpu[:num_reqs] = torch.from_numpy(pcp_logits_indices)
    logits_owner_mask_np = ((pcp_logits_indices >= local_start) &
                            (pcp_logits_indices < local_end))
    logits_local_indices_np = np.zeros(num_reqs, dtype=np.int32)
    logits_local_indices_np[logits_owner_mask_np] = (
        pcp_logits_indices[logits_owner_mask_np] - local_start)
    logits_local_indices_cpu = torch.zeros((padded_num_reqs, ),
                                           dtype=torch.int32,
                                           device="cpu")
    logits_local_indices_cpu[:num_reqs] = torch.from_numpy(
        logits_local_indices_np)
    logits_owner_mask_cpu = torch.zeros((padded_num_reqs, ),
                                        dtype=torch.bool,
                                        device="cpu")
    logits_owner_mask_cpu[:num_reqs] = torch.from_numpy(logits_owner_mask_np)

    if envs.VLLM_TPU_DEBUG_PCP_LAYOUT:
        layout_debug = _debug_pcp_layout_window(
            runner,
            pcp_rank=pcp_rank,
            local_token_slice=local_token_slice,
            local_padded_tokens=local_padded_total_num_scheduled_tokens,
            logits_indices=pcp_logits_indices,
        )
        logger.warning(
            "PCP_LAYOUT_DEBUG rank=%s scheduled=%s absolute_q_starts=%s "
            "token_owner_starts=%s "
            "local_counts=%s local_padded=%s global_padded=%s logits=%s "
            "query_start=%s seq_lens=%s layout=%s",
            pcp_rank,
            num_scheduled_tokens_per_req.tolist(),
            absolute_query_start_offsets_per_req.tolist(),
            token_owner_start_offsets_per_req.tolist(),
            local_counts.tolist(),
            local_padded_total_num_scheduled_tokens,
            padded_total_num_scheduled_tokens,
            pcp_logits_indices.tolist(),
            np.cumsum(
                np.concatenate([
                    np.array([0], dtype=np.int32), num_scheduled_tokens_per_req
                ])).tolist(),
            runner.seq_lens_np[:num_reqs].copy().tolist(),
            layout_debug,
        )

    return PcpPreparedBatch(
        local_token_slice=local_token_slice,
        padded_total_num_scheduled_tokens=padded_total_num_scheduled_tokens,
        local_total_num_scheduled_tokens=local_total_num_scheduled_tokens,
        local_padded_total_num_scheduled_tokens=(
            local_padded_total_num_scheduled_tokens),
        logits_indices_cpu=logits_indices_cpu,
        logits_local_indices_cpu=logits_local_indices_cpu,
        logits_owner_mask_cpu=logits_owner_mask_cpu,
        packed_to_request_major_token_indices=(
            packed_to_request_major_token_indices),
        request_major_to_packed_token_indices=(
            request_major_to_packed_token_indices),
    )
