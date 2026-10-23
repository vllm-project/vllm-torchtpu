# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.pcp import (
    all_gather_equal_tokens as _pcp_all_gather_equal_tokens,
)
from vllm_torchtpu.distributed.pcp import all_reduce_sum as _pcp_all_reduce_sum
from vllm_torchtpu.distributed.pcp import get_pcp_rank as _get_native_pcp_rank
from vllm_torchtpu.distributed.pcp import (
    get_pcp_world_size as _get_native_pcp_world_size,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    apply_pcp_rank_major_token_order as _apply_pcp_rank_major_token_order,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_logits_indices as _build_pcp_logits_indices,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    pcp_local_token_counts,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
    PCP_ZIGZAG_SEQUENCE_LAYOUT_DESCRIPTOR,
    AllSequenceLayoutPlanner,
    SequenceLayoutDescriptor,
    SequenceLayoutKind,
    SequenceLayoutPlan,
    _first_ge,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR = SequenceLayoutDescriptor(
    kind=SequenceLayoutKind.PARTIAL,
    protocol=PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
    version=1,
)


def _get_pcp_streaming_q_block_size() -> int:
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
        PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
    )

    return PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE


class PcpSequenceLayoutMode(Enum):
    DISABLED = "DISABLED"
    STREAMING = "STREAMING"
    ZIGZAG = "ZIGZAG"


@dataclass(frozen=True)
class PcpRequestSpan:
    computed_tokens: int
    scheduled_tokens: int
    prompt_tokens: int
    absolute_query_start: int

    @property
    def crosses_prompt_boundary(self) -> bool:
        return (
            self.scheduled_tokens > 0
            and self.computed_tokens < self.prompt_tokens
            and self.computed_tokens + self.scheduled_tokens > self.prompt_tokens
        )


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


def _resolve_is_hnd_kv_layout(cache_config: Any = None) -> bool:
    """Resolve whether the KV cache uses HND (LBHNC) layout."""
    if cache_config is not None:
        get_resolved = getattr(cache_config, "get_resolved_kv_cache_layout", None)
        if callable(get_resolved):
            resolved = get_resolved()
            name = getattr(resolved, "name", str(resolved)).upper()
            return name in ("LBHNC", "HND")
        raw_layout = getattr(cache_config, "kv_cache_layout", None)
        if raw_layout:
            return str(raw_layout).upper() in ("LBHNC", "HND")
    try:
        from vllm import envs as vllm_envs

        return (vllm_envs.VLLM_KV_CACHE_LAYOUT or "").upper() in ("LBHNC", "HND")
    except ImportError:
        return False


def resolve_pcp_zigzag_page_size(
    block_size: int,
    kernel_block_size: int | None = None,
    *,
    is_hnd_kv_layout: bool | None = None,
    cache_config: Any = None,
) -> int:
    """Return the attention kernel page size (128 for HND, up to 256 for NHD)."""
    if kernel_block_size is not None and kernel_block_size > 0:
        return int(kernel_block_size)
    block_size = int(block_size)
    is_hnd = (
        bool(is_hnd_kv_layout)
        if is_hnd_kv_layout is not None
        else _resolve_is_hnd_kv_layout(cache_config)
    )
    if is_hnd and block_size > 128 and block_size % 128 == 0:
        return 128
    if block_size > 256:
        if block_size % 256 == 0:
            return 256
        if block_size % 128 == 0:
            return 128
    return block_size


class PcpSequenceLayoutEligibility:
    _BOUNDARY_MESSAGE = (
        "PCP partial sequence layout does not support prompt/decode "
        "boundary-crossing schedules yet."
    )

    def __init__(
        self,
        *,
        pcp_size: int = 1,
        interleave_size: int = 1,
        dcp_size: int = 1,
        pipeline_parallel_size: int = 1,
        async_scheduling: bool = False,
        is_zigzag: bool = False,
        block_size: int = 16,
        max_num_tokens: int = 4096,
        is_hnd_kv_layout: bool | None = None,
        cache_config: Any = None,
    ):
        self.pcp_size = pcp_size
        self.interleave_size = interleave_size
        self.dcp_size = dcp_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.async_scheduling = bool(async_scheduling)
        self.is_zigzag = bool(is_zigzag)
        self.is_hnd_kv_layout = (
            bool(is_hnd_kv_layout)
            if is_hnd_kv_layout is not None
            else (
                _resolve_is_hnd_kv_layout(cache_config)
                if cache_config is not None
                else None
            )
        )
        self.block_size = (
            resolve_pcp_zigzag_page_size(
                block_size,
                is_hnd_kv_layout=self.is_hnd_kv_layout,
                cache_config=cache_config,
            )
            if self.is_zigzag
            else int(block_size)
        )
        self.max_num_tokens = int(max_num_tokens)

    def sync_runner_config(
        self,
        *,
        block_size: int | None = None,
        kernel_block_size: int | None = None,
        max_num_tokens: int | None = None,
        cache_config: Any = None,
    ) -> None:
        if cache_config is not None:
            self.is_hnd_kv_layout = _resolve_is_hnd_kv_layout(cache_config)
        if self.is_zigzag:
            base_block_size = (
                int(block_size) if block_size is not None else self.block_size
            )
            self.block_size = resolve_pcp_zigzag_page_size(
                base_block_size,
                kernel_block_size,
                is_hnd_kv_layout=self.is_hnd_kv_layout,
                cache_config=cache_config,
            )
        elif block_size is not None:
            self.block_size = int(block_size)
        if max_num_tokens is not None:
            self.max_num_tokens = int(max_num_tokens)

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "PcpSequenceLayoutEligibility":
        if vllm_config is None:
            return cls()

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        cache_config = getattr(vllm_config, "cache_config", None)
        block_size = getattr(cache_config, "block_size", 16)
        max_num_tokens = int(
            getattr(scheduler_config, "max_num_batched_tokens", 4096) or 4096
        )

        env_layout = envs.VLLM_TPU_PCP_LAYOUT.lower()
        is_zigzag = env_layout == "zigzag"

        return cls(
            pcp_size=parallel_config.prefill_context_parallel_size,
            interleave_size=parallel_config.cp_kv_cache_interleave_size,
            dcp_size=parallel_config.decode_context_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            async_scheduling=bool(scheduler_config.async_scheduling),
            is_zigzag=is_zigzag,
            block_size=block_size,
            max_num_tokens=max_num_tokens,
            cache_config=cache_config,
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
        block_size: int | None = None,
        kernel_block_size: int | None = None,
        cache_config: Any = None,
    ) -> PcpSequenceLayoutDecision:
        spans = self._build_request_spans(
            input_batch=input_batch,
            scheduler_output=scheduler_output,
            start_index=start_index,
            num_reqs=num_reqs,
            num_scheduled_tokens_per_req=num_scheduled_tokens_per_req,
            absolute_query_start_offsets_per_req=(absolute_query_start_offsets_per_req),
        )

        if not self.enabled or not any(span.scheduled_tokens > 0 for span in spans):
            return PcpSequenceLayoutDecision(
                mode=PcpSequenceLayoutMode.DISABLED,
                spans=spans,
                pcp_size=self.pcp_size,
                interleave_size=self.interleave_size,
            )

        if any(span.crosses_prompt_boundary for span in spans):
            raise NotImplementedError(self._BOUNDARY_MESSAGE)

        self._validate_execution_support()
        if self.is_zigzag:
            return self._evaluate_zigzag_query(
                spans,
                num_tokens_paddings=num_tokens_paddings,
                max_num_tokens=max_num_tokens,
                dp_target_bucket=dp_target_bucket,
                block_size=block_size,
                kernel_block_size=kernel_block_size,
                cache_config=cache_config,
            )
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
                f"used where {expected.value} is required."
            )

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
            raise ValueError("num_scheduled_tokens_per_req must be a 1D array.")
        if scheduled.size < num_reqs:
            raise ValueError(
                "num_scheduled_tokens_per_req must cover every request in the chunk."
            )

        absolute_query_starts = None
        if absolute_query_start_offsets_per_req is not None:
            absolute_query_starts = np.asarray(
                absolute_query_start_offsets_per_req, dtype=np.int64
            )
            if absolute_query_starts.ndim != 1 or absolute_query_starts.size < num_reqs:
                raise ValueError(
                    "absolute_query_start_offsets_per_req must cover every "
                    "request in the chunk."
                )

        spans: list[PcpRequestSpan] = []
        req_ids = input_batch.req_ids[start_index : start_index + num_reqs]
        for chunk_offset, req_id in enumerate(req_ids):
            if req_id is None:
                continue
            req_index = input_batch.req_id_to_index[req_id]
            scheduled_tokens = int(scheduled[chunk_offset])
            scheduler_tokens = scheduler_output.num_scheduled_tokens.get(req_id)
            if scheduler_tokens is not None:
                scheduled_tokens = int(scheduler_tokens)
            computed_tokens = int(input_batch.num_computed_tokens_cpu[req_index])
            absolute_query_start = (
                int(absolute_query_starts[chunk_offset])
                if absolute_query_starts is not None
                else computed_tokens
            )
            spans.append(
                PcpRequestSpan(
                    computed_tokens=computed_tokens,
                    scheduled_tokens=scheduled_tokens,
                    prompt_tokens=int(input_batch.num_prompt_tokens[req_index]),
                    absolute_query_start=absolute_query_start,
                )
            )
        return tuple(spans)

    def _validate_execution_support(self) -> None:
        if self.dcp_size > 1:
            raise NotImplementedError(
                "PCP partial sequence layout does not support DCP yet."
            )
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "PCP partial sequence layout does not support pipeline parallelism yet."
            )
        if not self.is_zigzag and self.interleave_size <= 0:
            raise ValueError(
                "PCP partial sequence layout requires cp_kv_cache_interleave_size > 0."
            )

    def _evaluate_zigzag_query(
        self,
        spans: tuple[PcpRequestSpan, ...],
        *,
        num_tokens_paddings: Sequence[int] | None,
        max_num_tokens: int | None,
        dp_target_bucket: int | None,
        block_size: int | None = None,
        kernel_block_size: int | None = None,
        cache_config: Any = None,
    ) -> PcpSequenceLayoutDecision:
        has_decode = any(
            span.scheduled_tokens == 1 and span.computed_tokens > 0 for span in spans
        )
        has_prefill = any(
            span.scheduled_tokens > 1 or span.computed_tokens == 0 for span in spans
        )
        if has_decode and has_prefill:
            raise NotImplementedError(
                "PCP batched RPA zigzag layout does not support mixed decode and prefill batches."
            )

        effective_block_size = resolve_pcp_zigzag_page_size(
            block_size if block_size is not None else self.block_size,
            kernel_block_size,
            is_hnd_kv_layout=self.is_hnd_kv_layout,
            cache_config=cache_config,
        )
        local_quantum = 2 * effective_block_size
        if has_decode:
            s_live = len(spans)
        else:
            q_lens = [span.scheduled_tokens for span in spans]
            chunk, off, s_live = pcp_zigzag_token_layout(
                q_lens, self.pcp_size, effective_block_size
            )

        local_padded_tokens = None
        global_padded_tokens = None
        if num_tokens_paddings is not None:
            paddings = sorted(int(padding) for padding in num_tokens_paddings)
            local_padded_tokens = _first_ge(paddings, s_live)
            if dp_target_bucket is not None and dp_target_bucket > local_padded_tokens:
                local_padded_tokens = int(dp_target_bucket)
            if not has_decode:
                local_padded_tokens = (
                    (local_padded_tokens + local_quantum - 1) // local_quantum
                ) * local_quantum
            if (local_padded_tokens < local_quantum) != has_decode:
                raise ValueError(
                    f"PCP zigzag padded token bucket {local_padded_tokens} "
                    f"disagrees with batch mode ({has_decode=}, {local_quantum=})."
                )
            if max_num_tokens is not None and local_padded_tokens > int(max_num_tokens):
                raise ValueError(
                    f"PCP local padded token length exceeds runner max token buffer: "
                    f"{local_padded_tokens=} max_num_tokens={int(max_num_tokens)}."
                )
            global_padded_tokens = local_padded_tokens * self.pcp_size

        local_counts = np.full(self.pcp_size, s_live, dtype=np.int32)
        return PcpSequenceLayoutDecision(
            mode=PcpSequenceLayoutMode.ZIGZAG,
            spans=spans,
            pcp_size=self.pcp_size,
            interleave_size=self.interleave_size,
            pcp_alignment=None,
            absolute_query_start_offsets_per_req=np.asarray(
                [span.absolute_query_start for span in spans], dtype=np.int64
            ),
            token_owner_start_offsets_per_req=np.zeros(len(spans), dtype=np.int64),
            local_token_counts=local_counts,
            local_required_tokens=s_live,
            local_padded_tokens=local_padded_tokens,
            global_padded_tokens=global_padded_tokens,
        )

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
                "request in the chunk."
            )

        q_block_size = _get_pcp_streaming_q_block_size()
        if q_block_size % self.interleave_size != 0:
            raise NotImplementedError(
                "PCP streaming requires q_block_size to be a "
                "multiple of cp_kv_cache_interleave_size, got "
                f"{q_block_size=} "
                f"cp_kv_cache_interleave_size={self.interleave_size}."
            )

        q_lens = np.asarray([span.scheduled_tokens for span in spans], dtype=np.int32)
        absolute_query_starts = np.asarray(
            [span.absolute_query_start for span in spans], dtype=np.int64
        )
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
            if dp_target_bucket is not None and dp_target_bucket > local_padded_tokens:
                local_padded_tokens = int(dp_target_bucket)
            if max_num_tokens is not None and local_padded_tokens > int(max_num_tokens):
                raise ValueError(
                    "PCP local padded token length exceeds runner max token "
                    f"buffer: {local_padded_tokens=} "
                    f"max_num_tokens={int(max_num_tokens)}."
                )
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
    packed_to_request_major_token_indices: np.ndarray
    request_major_to_packed_token_indices: np.ndarray
    logits_local_indices_cpu: torch.Tensor | None = None
    logits_owner_mask_cpu: torch.Tensor | None = None
    pcp_metadata: Any | None = None


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

    def reserve_host_token_capacity(
        self, runner: Any, required_num_tokens: int
    ) -> None:
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
            block_size=getattr(runner, "block_size", None),
            kernel_block_size=getattr(runner, "_attention_kernel_block_size", None),
            cache_config=getattr(runner, "cache_config", None),
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
        if decision.mode is PcpSequenceLayoutMode.ZIGZAG:
            prepared = prepare_pcp_zigzag_sequence_layout(
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
                descriptor=PCP_ZIGZAG_SEQUENCE_LAYOUT_DESCRIPTOR,
                token_slice=prepared.local_token_slice,
                global_num_tokens=int(total_num_scheduled_tokens),
                global_padded_num_tokens=(prepared.padded_total_num_scheduled_tokens),
                local_num_tokens=prepared.local_total_num_scheduled_tokens,
                local_padded_num_tokens=(
                    prepared.local_padded_total_num_scheduled_tokens
                ),
                logits_indices_cpu=prepared.logits_indices_cpu,
                logits_local_indices_cpu=prepared.logits_local_indices_cpu,
                logits_owner_mask_cpu=prepared.logits_owner_mask_cpu,
                requires_hidden_state_gather=True,
                pcp_metadata=prepared.pcp_metadata,
                _packed_to_request_major_token_indices=(
                    prepared.packed_to_request_major_token_indices
                ),
                _request_major_to_packed_token_indices=(
                    prepared.request_major_to_packed_token_indices
                ),
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
            global_padded_num_tokens=(prepared.padded_total_num_scheduled_tokens),
            local_num_tokens=prepared.local_total_num_scheduled_tokens,
            local_padded_num_tokens=(prepared.local_padded_total_num_scheduled_tokens),
            logits_indices_cpu=prepared.logits_indices_cpu,
            logits_local_indices_cpu=prepared.logits_local_indices_cpu,
            logits_owner_mask_cpu=prepared.logits_owner_mask_cpu,
            requires_hidden_state_gather=True,
            _packed_to_request_major_token_indices=(
                prepared.packed_to_request_major_token_indices
            ),
            _request_major_to_packed_token_indices=(
                prepared.request_major_to_packed_token_indices
            ),
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
        if self.eligibility.is_zigzag:
            from vllm_torchtpu.layers.core.attention_metadata import PCPMetadata

            two_p = 2 * self.eligibility.pcp_size
            n_off = max(num_reqs, 2)
            block_size = getattr(self.eligibility, "block_size", 16)
            max_tokens_cap = max(
                int(
                    getattr(self.eligibility, "max_num_tokens", num_tokens)
                    or num_tokens
                ),
                num_tokens,
            )
            c_i_max = (max_tokens_cap * self.eligibility.pcp_size) // two_p
            req_pages = max(1, c_i_max // block_size)
            dev = "tpu" if hasattr(torch, "tpu") and torch.tpu.is_available() else "cpu"
            dummy_pcp = PCPMetadata(
                query_start_loc=torch.zeros(
                    (self.eligibility.pcp_size, n_off + 1),
                    dtype=torch.int32,
                    device=dev,
                ),
                kv_cache_lens=torch.zeros((n_off,), dtype=torch.int32, device=dev),
                q_pos_offsets=torch.zeros(
                    (self.eligibility.pcp_size, n_off), dtype=torch.int32, device=dev
                ),
                new_kv_page_indices=torch.zeros(
                    (n_off * two_p * req_pages,), dtype=torch.int32, device=dev
                ),
                update_kv_cache=torch.zeros(
                    (self.eligibility.pcp_size, n_off), dtype=torch.bool, device=dev
                ),
                gdn_rm_to_packed=torch.zeros(
                    (max_tokens_cap * self.eligibility.pcp_size,),
                    dtype=torch.int32,
                    device=dev,
                ),
                gdn_packed_to_local=torch.zeros(
                    (max_tokens_cap,), dtype=torch.int32, device=dev
                ),
                gdn_query_start_loc=torch.zeros(
                    (n_off + 1,), dtype=torch.int32, device=dev
                ),
                cache_pages=0,
                is_decode=False,
                page_size=int(block_size),
            )
            return SequenceLayoutPlan(
                descriptor=PCP_ZIGZAG_SEQUENCE_LAYOUT_DESCRIPTOR,
                token_slice=slice(0, num_tokens),
                global_num_tokens=num_tokens * self.eligibility.pcp_size,
                global_padded_num_tokens=num_tokens * self.eligibility.pcp_size,
                local_num_tokens=num_tokens,
                local_padded_num_tokens=num_tokens,
                requires_hidden_state_gather=True,
                pcp_metadata=dummy_pcp,
            )
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
        if plan.logits_local_indices_cpu is None or plan.logits_owner_mask_cpu is None:
            return None

        local_indices = plan.logits_local_indices_cpu.to(
            hidden_states.device, non_blocking=True
        )
        owner_mask = plan.logits_owner_mask_cpu.to(
            hidden_states.device, non_blocking=True
        )
        selected = torch.index_select(hidden_states, 0, local_indices)
        selected = torch.where(
            owner_mask.unsqueeze(1), selected, torch.zeros_like(selected)
        )
        selected = torch.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0)
        return _pcp_all_reduce_sum(selected)


def _ensure_host_token_buffer_capacity(runner: Any, required_num_tokens: int) -> None:
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
    runner.input_ids_cpu = torch.zeros(
        required_num_tokens,
        dtype=old_input_ids_cpu.dtype,
        device=old_input_ids_cpu.device,
    )
    runner.input_ids_cpu[:current_num_tokens] = old_input_ids_cpu

    old_positions_cpu = runner.positions_cpu
    runner.positions_cpu = torch.zeros(
        required_num_tokens,
        dtype=old_positions_cpu.dtype,
        device=old_positions_cpu.device,
    )
    runner.positions_cpu[:current_num_tokens] = old_positions_cpu
    runner.positions_np = runner.positions_cpu.numpy()

    if runner.supports_mm_inputs and hasattr(runner, "is_mm_embed_cpu"):
        old_is_mm_embed_cpu = runner.is_mm_embed_cpu
        runner.is_mm_embed_cpu = torch.zeros(
            required_num_tokens,
            dtype=old_is_mm_embed_cpu.dtype,
            device=old_is_mm_embed_cpu.device,
        )
        runner.is_mm_embed_cpu[:current_num_tokens] = old_is_mm_embed_cpu

    if runner.uses_mrope:
        old_mrope_positions = runner.mrope_positions
        new_mrope_positions = runner._make_buffer(
            3, required_num_tokens + 1, dtype=torch.int32
        )
        copy_len = min(
            current_num_tokens + 1,
            int(old_mrope_positions.cpu.shape[1]),
            required_num_tokens + 1,
        )
        new_mrope_positions.cpu[:, :copy_len] = old_mrope_positions.cpu[:, :copy_len]
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
            selected_windows.append(
                {
                    "global_index": index,
                    "local_offset": local_offset,
                    "ids": local_ids[start:end].tolist(),
                    "positions": local_positions[start:end].tolist(),
                }
            )
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
            runner.vllm_config
        ).evaluate_runner_chunk(
            input_batch=runner.input_batch,
            scheduler_output=scheduler_output,
            start_index=start_index,
            num_reqs=num_reqs,
            num_scheduled_tokens_per_req=num_scheduled_tokens_per_req,
            num_tokens_paddings=runner.num_tokens_paddings,
            max_num_tokens=runner.max_num_tokens,
            dp_target_bucket=runner._dp_target_bucket,
        )
    PcpSequenceLayoutEligibility.require_mode(decision, PcpSequenceLayoutMode.STREAMING)

    pcp_size = decision.pcp_size
    interleave_size = decision.interleave_size
    if pcp_size <= 1:
        raise ValueError("PCP streaming requires prefill_context_parallel_size > 1.")

    if get_native_pcp_rank is None:
        get_native_pcp_rank = _get_native_pcp_rank
    if get_native_pcp_world_size is None:
        get_native_pcp_world_size = _get_native_pcp_world_size

    native_pcp_world_size = get_native_pcp_world_size()
    if native_pcp_world_size != pcp_size:
        raise RuntimeError(
            "Native PCP group size does not match "
            "prefill_context_parallel_size: "
            f"{native_pcp_world_size=} {pcp_size=}."
        )
    pcp_rank = get_native_pcp_rank()
    if not 0 <= pcp_rank < pcp_size:
        raise RuntimeError(f"Native PCP rank is out of range: {pcp_rank=} {pcp_size=}.")

    absolute_query_start_offsets_per_req = decision.absolute_query_start_offsets_per_req
    token_owner_start_offsets_per_req = decision.token_owner_start_offsets_per_req
    local_counts = decision.local_token_counts
    local_padded_tokens = decision.local_padded_tokens
    padded_total_num_scheduled_tokens = decision.global_padded_tokens
    assert absolute_query_start_offsets_per_req is not None
    assert token_owner_start_offsets_per_req is not None
    assert local_counts is not None
    assert local_padded_tokens is not None
    assert padded_total_num_scheduled_tokens is not None

    _ensure_host_token_buffer_capacity(runner, padded_total_num_scheduled_tokens)

    mrope_slice = None
    if runner.uses_mrope:
        mrope_slice = runner.mrope_positions.cpu[
            :, :(padded_total_num_scheduled_tokens)
        ].numpy()
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
        padded_total_num_scheduled_tokens, -1, dtype=np.int64
    )
    packed_to_request_major_token_indices[request_major_to_packed_token_indices] = (
        np.arange(request_major_to_packed_token_indices.size, dtype=np.int64)
    )

    local_padded_total_num_scheduled_tokens = (
        padded_total_num_scheduled_tokens // pcp_size
    )
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
    logits_indices_cpu = torch.full(
        (padded_num_reqs,), -1, dtype=torch.int32, device="cpu"
    )
    logits_indices_cpu[:num_reqs] = torch.from_numpy(pcp_logits_indices)
    logits_owner_mask_np = (pcp_logits_indices >= local_start) & (
        pcp_logits_indices < local_end
    )
    logits_local_indices_np = np.zeros(num_reqs, dtype=np.int32)
    logits_local_indices_np[logits_owner_mask_np] = (
        pcp_logits_indices[logits_owner_mask_np] - local_start
    )
    logits_local_indices_cpu = torch.zeros(
        (padded_num_reqs,), dtype=torch.int32, device="cpu"
    )
    logits_local_indices_cpu[:num_reqs] = torch.from_numpy(logits_local_indices_np)
    logits_owner_mask_cpu = torch.zeros(
        (padded_num_reqs,), dtype=torch.bool, device="cpu"
    )
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
                np.concatenate(
                    [np.array([0], dtype=np.int32), num_scheduled_tokens_per_req]
                )
            ).tolist(),
            runner.seq_lens_np[:num_reqs].copy().tolist(),
            layout_debug,
        )

    return PcpPreparedBatch(
        local_token_slice=local_token_slice,
        padded_total_num_scheduled_tokens=padded_total_num_scheduled_tokens,
        local_total_num_scheduled_tokens=local_total_num_scheduled_tokens,
        local_padded_total_num_scheduled_tokens=(
            local_padded_total_num_scheduled_tokens
        ),
        logits_indices_cpu=logits_indices_cpu,
        logits_local_indices_cpu=logits_local_indices_cpu,
        logits_owner_mask_cpu=logits_owner_mask_cpu,
        packed_to_request_major_token_indices=(packed_to_request_major_token_indices),
        request_major_to_packed_token_indices=(request_major_to_packed_token_indices),
    )


def pcp_zigzag_token_layout(
    num_scheduled_tokens: list[int] | np.ndarray,
    pcp_size: int,
    block_size: int = 16,
) -> tuple[list[int], list[int], int]:
    """Per-request zigzag chunking for multi-request PCP.

    Each request is split into 2 * pcp_size chunks.
    Rank r owns chunk r (head) and chunk 2P - 1 - r (tail).
    Returns (chunk, off, s_live):
      chunk[i]: chunk size of request i, multiple of block_size
      off[i]: start offset of request i's slot within one rank's region
      s_live: live tokens per rank; the global buffer needs pcp_size * s_live rows
    """
    two_p = 2 * pcp_size
    quantum = two_p * block_size
    off, acc, chunk = [], 0, []
    for n in num_scheduled_tokens:
        c = max(block_size, int(np.ceil(int(n) / quantum)) * block_size)
        chunk.append(c)
        off.append(acc)
        acc += 2 * c
    return chunk, off, acc


def pcp_zigzag_seq_arrays(
    chunk: list[int],
    off: list[int],
    pcp_size: int,
    n_off: int,
    counts: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-seq attention metadata for a zigzag chunking result.

    Returns (cu_q_lens, q_pos_offsets, pcp_write_flags):
      cu_q_lens: shape (pcp_size, n_off + 1)
      q_pos_offsets: shape (pcp_size, n_off)
      pcp_write_flags: shape (pcp_size, n_off)
    """
    two_p = 2 * pcp_size
    n_reqs = len(chunk)
    cu_q_lens = np.zeros((pcp_size, n_off + 1), dtype=np.int32)
    q_pos_offsets = np.zeros((pcp_size, n_off), dtype=np.int32)
    pcp_write_flags = np.zeros((pcp_size, n_off), dtype=bool)

    for r in range(pcp_size):
        for i in range(n_reqs):
            c_i = chunk[i]
            o_i = off[i]
            tail_off = (two_p - 1 - r) * c_i
            tail_real = int(np.clip(counts[i] - tail_off, 0, c_i))

            # Keep full c_i on non-last tail slots so cu_q_lens stays contiguous up
            # to off[i + 1]; causal masking and kv_lens mask padded query positions.
            cu_q_lens[r, 2 * i] = o_i
            cu_q_lens[r, 2 * i + 1] = o_i + c_i
            cu_q_lens[r, 2 * i + 2] = (
                o_i + c_i + (tail_real if i == n_reqs - 1 else c_i)
            )

            q_pos_offsets[r, 2 * i] = r * c_i
            q_pos_offsets[r, 2 * i + 1] = tail_off

            pcp_write_flags[r, 2 * i + (1 if tail_real > 0 else 0)] = True

        if 2 * n_reqs < n_off:
            cu_q_lens[r, 2 * n_reqs + 1 :] = cu_q_lens[r, 2 * n_reqs]

    return cu_q_lens, q_pos_offsets, pcp_write_flags


def pcp_zigzag_page_table(
    chunk: list[int],
    pcp_size: int,
    block_size: int,
    n_off: int,
    off: list[int] | None = None,
    s_pad: int | None = None,
    counts: list[int] | None = None,
    max_s_pad: int | None = None,
) -> np.ndarray:
    """Page table for the all-gathered current KV under zigzag sharding."""
    del counts
    two_p = 2 * pcp_size
    n_reqs = len(chunk)
    cap_s_pad = max_s_pad if max_s_pad is not None else s_pad
    if cap_s_pad is not None:
        max_req_pages = max(1, (cap_s_pad // 2) // block_size)
    else:
        max_req_pages = max(c // block_size for c in chunk) if chunk else 1
        max_req_pages = max(1, max_req_pages)
    if s_pad is not None:
        s_pad_val = s_pad
    else:
        s_pad_val = sum(2 * c for c in chunk)
    pcp_table_np = np.zeros((n_off, two_p * max_req_pages), dtype=np.int32)
    for i in range(n_reqs):
        c_i = chunk[i]
        req_pages = max(1, c_i // block_size)
        o_i = off[i] if off is not None else sum(2 * c for c in chunk[:i])
        for c in range(two_p):
            cols = slice(c * req_pages, (c + 1) * req_pages)
            rank = c if c < pcp_size else two_p - 1 - c
            base = (
                rank * s_pad_val + o_i + (c_i if c >= pcp_size else 0)
            ) // block_size
            pcp_table_np[2 * i, cols] = base + np.arange(req_pages)
            pcp_table_np[2 * i + 1, cols] = base + np.arange(req_pages)
    return pcp_table_np


def apply_pcp_zigzag_token_order(
    input_ids: np.ndarray,
    positions: np.ndarray,
    counts: list[int],
    chunk: list[int],
    off: list[int],
    pcp_size: int,
    padded_total_num_tokens: int,
    mrope_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute token-major buffers into zigzag rank-major buffers."""
    two_p = 2 * pcp_size
    s_pad = padded_total_num_tokens // pcp_size
    num_reqs = len(counts)
    ranks = np.arange(pcp_size)
    src_off = np.cumsum([0] + counts)[:-1]
    perm = np.full(padded_total_num_tokens, -1, dtype=np.int64)

    for i in range(num_reqs):
        c_i = chunk[i]
        j = np.arange(c_i)
        for h in (0, 1):
            chunk_idx = ranks if h == 0 else two_p - 1 - ranks
            dst = ranks[:, None] * s_pad + off[i] + h * c_i + j[None, :]
            tok = chunk_idx[:, None] * c_i + j[None, :]
            real = tok < counts[i]
            perm[dst[real]] = src_off[i] + tok[real]

    valid = perm >= 0
    orig_ids = input_ids.copy()
    input_ids.fill(0)
    input_ids[valid] = orig_ids[perm[valid]]

    orig_pos = positions.copy()
    positions.fill(0)
    positions[valid] = orig_pos[perm[valid]]

    if mrope_positions is not None:
        orig_mrope = mrope_positions.copy()
        mrope_positions.fill(0)
        mrope_positions[:, valid] = orig_mrope[:, perm[valid]]

    packed_to_rm = perm
    rm_to_packed = np.full(padded_total_num_tokens, -1, dtype=np.int64)
    rm_to_packed[perm[valid]] = np.where(valid)[0]
    return rm_to_packed, packed_to_rm


def prepare_pcp_zigzag_sequence_layout(
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
    """Prepare native PCP zigzag host state for real requests."""
    del total_num_scheduled_tokens, use_max_model_len
    if decision is None:
        decision = PcpSequenceLayoutEligibility.from_vllm_config(
            runner.vllm_config
        ).evaluate_runner_chunk(
            input_batch=runner.input_batch,
            scheduler_output=scheduler_output,
            start_index=start_index,
            num_reqs=num_reqs,
            num_scheduled_tokens_per_req=num_scheduled_tokens_per_req,
            num_tokens_paddings=runner.num_tokens_paddings,
            max_num_tokens=runner.max_num_tokens,
            dp_target_bucket=runner._dp_target_bucket,
            block_size=getattr(runner, "block_size", None),
            kernel_block_size=getattr(runner, "_attention_kernel_block_size", None),
            cache_config=getattr(runner, "cache_config", None),
        )
    PcpSequenceLayoutEligibility.require_mode(decision, PcpSequenceLayoutMode.ZIGZAG)

    pcp_size = decision.pcp_size
    if pcp_size <= 1:
        raise ValueError("PCP zigzag requires prefill_context_parallel_size > 1.")

    if get_native_pcp_rank is None:
        get_native_pcp_rank = _get_native_pcp_rank
    if get_native_pcp_world_size is None:
        get_native_pcp_world_size = _get_native_pcp_world_size

    native_pcp_world_size = get_native_pcp_world_size()
    if native_pcp_world_size != pcp_size:
        raise RuntimeError(
            "Native PCP group size does not match prefill_context_parallel_size: "
            f"{native_pcp_world_size=} {pcp_size=}."
        )
    pcp_rank = get_native_pcp_rank()
    if not 0 <= pcp_rank < pcp_size:
        raise RuntimeError(f"Native PCP rank is out of range: {pcp_rank=} {pcp_size=}.")

    counts = [int(c) for c in num_scheduled_tokens_per_req[:num_reqs]]
    two_p = 2 * pcp_size
    padded_total_num_tokens = decision.global_padded_tokens
    assert padded_total_num_tokens is not None
    s_pad = padded_total_num_tokens // pcp_size

    block_size = resolve_pcp_zigzag_page_size(
        getattr(runner, "block_size", 16),
        getattr(runner, "_attention_kernel_block_size", None),
        cache_config=getattr(runner, "cache_config", None),
    )
    max_tokens_cap = max(
        int(getattr(runner, "max_num_tokens", s_pad) or s_pad),
        s_pad,
    )

    is_decode = any(
        span.scheduled_tokens == 1 and span.computed_tokens > 0
        for span in decision.spans
    )

    _ensure_host_token_buffer_capacity(runner, padded_total_num_tokens)

    from vllm_torchtpu.layers.core.attention_metadata import PCPMetadata

    n_off = max(target_num_reqs, 2 * num_reqs)
    dev = runner.device

    computed = [
        int(runner.input_batch.num_computed_tokens_cpu[start_index + i])
        for i in range(num_reqs)
    ]
    kv_cache_lens_np = np.zeros(n_off, dtype=np.int32)
    for i in range(num_reqs):
        kv_cache_lens_np[2 * i] = computed[i]
        kv_cache_lens_np[2 * i + 1] = computed[i]

    logits_indices_cpu = torch.zeros(
        (padded_num_reqs,), dtype=torch.int32, device="cpu"
    )
    logits_local_indices_cpu = torch.zeros(
        (padded_num_reqs,), dtype=torch.int32, device="cpu"
    )
    logits_owner_mask_cpu = torch.zeros(
        (padded_num_reqs,), dtype=torch.bool, device="cpu"
    )

    if is_decode:
        local_start = 0
        local_end = s_pad
        local_token_slice = slice(local_start, local_end)
        local_total_tokens = num_reqs

        request_major_to_packed = np.arange(padded_total_num_tokens, dtype=np.int64)
        packed_to_request_major = np.arange(padded_total_num_tokens, dtype=np.int64)

        for i in range(num_reqs):
            logits_indices_cpu[i] = i
            logits_local_indices_cpu[i] = i
            logits_owner_mask_cpu[i] = pcp_rank == 0
        if num_reqs < padded_num_reqs:
            logits_indices_cpu[num_reqs:] = (
                logits_indices_cpu[num_reqs - 1] if num_reqs > 0 else 0
            )
            logits_local_indices_cpu[num_reqs:] = 0
            logits_owner_mask_cpu[num_reqs:] = False

        cu_q_lens_np = np.zeros((pcp_size, n_off + 1), dtype=np.int32)
        q_pos_offsets_np = np.zeros((pcp_size, n_off), dtype=np.int32)
        pcp_write_np = np.zeros((pcp_size, n_off), dtype=bool)

        for r in range(pcp_size):
            for i in range(num_reqs):
                cu_q_lens_np[r, 2 * i] = i
                cu_q_lens_np[r, 2 * i + 1] = i + 1
                cu_q_lens_np[r, 2 * i + 2] = i + 1
                q_pos_offsets_np[r, 2 * i] = 0
                q_pos_offsets_np[r, 2 * i + 1] = 0
                pcp_write_np[r, 2 * i] = True
                pcp_write_np[r, 2 * i + 1] = False
            if 2 * num_reqs < n_off:
                cu_q_lens_np[r, 2 * num_reqs + 1 :] = num_reqs

        max_req_pages = max(1, (max_tokens_cap // 2) // block_size)
        pcp_table_np = np.zeros((n_off, two_p * max_req_pages), dtype=np.int32)
        max_computed = max(computed) if computed else 0
        cache_pages = max(1, (max_computed + block_size - 1) // block_size)

        gdn_rm_to_packed_np = np.zeros(max_tokens_cap * pcp_size, dtype=np.int32)
        gdn_rm_to_packed_np[:padded_total_num_tokens] = np.arange(
            padded_total_num_tokens, dtype=np.int32
        )
        gdn_packed_to_local_np = np.zeros(max_tokens_cap, dtype=np.int32)
        gdn_packed_to_local_np[:s_pad] = np.arange(s_pad, dtype=np.int32)
        gdn_qsl_np = np.zeros(n_off + 1, dtype=np.int32)
        for i in range(num_reqs):
            gdn_qsl_np[2 * i + 1] = i + 1
            gdn_qsl_np[2 * i + 2] = i + 1
        if 2 * num_reqs < n_off:
            gdn_qsl_np[2 * num_reqs + 1 :] = num_reqs

        runner._request_distribution_cpu[0] = 2 * num_reqs
        runner._request_distribution_cpu[1] = 2 * num_reqs
        runner._request_distribution_cpu[2] = 2 * num_reqs
    else:
        chunk, off, _ = pcp_zigzag_token_layout(counts, pcp_size, block_size)

        mrope_np = None
        if runner.uses_mrope:
            mrope_np = runner.mrope_positions.cpu[:, :padded_total_num_tokens].numpy()

        input_ids_np = runner.input_ids_cpu[:padded_total_num_tokens].numpy()
        positions_np = runner.positions_np[:padded_total_num_tokens]

        request_major_to_packed, packed_to_request_major = apply_pcp_zigzag_token_order(
            input_ids_np,
            positions_np,
            counts,
            chunk,
            off,
            pcp_size,
            padded_total_num_tokens,
            mrope_positions=mrope_np,
        )

        local_start = pcp_rank * s_pad
        local_end = local_start + s_pad
        local_token_slice = slice(local_start, local_end)
        local_total_tokens = (
            int(decision.local_token_counts[pcp_rank])
            if decision.local_token_counts is not None
            else s_pad
        )

        for i in range(num_reqs):
            last = counts[i] - 1
            c_idx = last // chunk[i]
            owner_rank = c_idx if c_idx < pcp_size else two_p - 1 - c_idx
            h = 0 if c_idx < pcp_size else 1
            local_offset = off[i] + h * chunk[i] + (last % chunk[i])
            g_slot = owner_rank * s_pad + local_offset
            logits_indices_cpu[i] = int(g_slot)
            if owner_rank == pcp_rank:
                logits_local_indices_cpu[i] = int(local_offset)
                logits_owner_mask_cpu[i] = True
            else:
                logits_local_indices_cpu[i] = 0
                logits_owner_mask_cpu[i] = False
        if num_reqs < padded_num_reqs:
            logits_indices_cpu[num_reqs:] = (
                logits_indices_cpu[num_reqs - 1] if num_reqs > 0 else 0
            )
            logits_local_indices_cpu[num_reqs:] = 0
            logits_owner_mask_cpu[num_reqs:] = False

        cu_q_lens_np, q_pos_offsets_np, pcp_write_np = pcp_zigzag_seq_arrays(
            chunk, off, pcp_size, n_off, counts
        )
        pcp_table_np = pcp_zigzag_page_table(
            chunk,
            pcp_size,
            block_size,
            n_off,
            off=off,
            s_pad=s_pad,
            counts=counts,
            max_s_pad=max_tokens_cap,
        )

        max_computed = max(computed) if computed else 0
        cache_pages = (
            max(1, (max_computed + block_size - 1) // block_size)
            if max_computed > 0
            else 0
        )

        gdn_rm_to_packed_raw = np.where(
            request_major_to_packed >= 0, request_major_to_packed, 0
        ).astype(np.int32)
        gdn_rm_to_packed_np = np.zeros(max_tokens_cap * pcp_size, dtype=np.int32)
        gdn_rm_to_packed_np[: len(gdn_rm_to_packed_raw)] = gdn_rm_to_packed_raw

        local_packed_to_rm = packed_to_request_major[local_start:local_end]
        gdn_packed_to_local_raw = np.where(
            local_packed_to_rm >= 0, local_packed_to_rm, 0
        ).astype(np.int32)
        gdn_packed_to_local_np = np.zeros(max_tokens_cap, dtype=np.int32)
        gdn_packed_to_local_np[: len(gdn_packed_to_local_raw)] = gdn_packed_to_local_raw
        gdn_qsl_np = np.zeros(n_off + 1, dtype=np.int32)
        acc_q = 0
        for i in range(num_reqs):
            acc_q += counts[i]
            gdn_qsl_np[2 * i + 1] = acc_q
            gdn_qsl_np[2 * i + 2] = acc_q
        if 2 * num_reqs < n_off:
            gdn_qsl_np[2 * num_reqs + 1 :] = acc_q

        runner._request_distribution_cpu[0] = 0
        runner._request_distribution_cpu[1] = 0
        runner._request_distribution_cpu[2] = 2 * num_reqs

    pcp_metadata = PCPMetadata(
        query_start_loc=torch.from_numpy(cu_q_lens_np).to(dev, non_blocking=True),
        kv_cache_lens=torch.from_numpy(kv_cache_lens_np).to(dev, non_blocking=True),
        q_pos_offsets=torch.from_numpy(q_pos_offsets_np).to(dev, non_blocking=True),
        new_kv_page_indices=torch.from_numpy(pcp_table_np.reshape(-1)).to(
            dev, non_blocking=True
        ),
        update_kv_cache=torch.from_numpy(pcp_write_np).to(dev, non_blocking=True),
        gdn_rm_to_packed=torch.from_numpy(gdn_rm_to_packed_np).to(
            dev, non_blocking=True
        ),
        gdn_packed_to_local=torch.from_numpy(gdn_packed_to_local_np).to(
            dev, non_blocking=True
        ),
        gdn_query_start_loc=torch.from_numpy(gdn_qsl_np).to(dev, non_blocking=True),
        cache_pages=cache_pages,
        is_decode=is_decode,
        page_size=int(block_size),
    )

    for i in range(num_reqs):
        total_len = computed[i] + counts[i]
        runner.seq_lens_cpu[2 * i] = total_len
        runner.seq_lens_cpu[2 * i + 1] = total_len
    runner.seq_lens_cpu[2 * num_reqs :] = 0

    runner.query_start_loc_cpu[: n_off + 1] = torch.from_numpy(cu_q_lens_np[pcp_rank])

    return PcpPreparedBatch(
        local_token_slice=local_token_slice,
        padded_total_num_scheduled_tokens=padded_total_num_tokens,
        local_total_num_scheduled_tokens=local_total_tokens,
        local_padded_total_num_scheduled_tokens=s_pad,
        logits_indices_cpu=logits_indices_cpu,
        logits_local_indices_cpu=logits_local_indices_cpu,
        logits_owner_mask_cpu=logits_owner_mask_cpu,
        packed_to_request_major_token_indices=packed_to_request_major,
        request_major_to_packed_token_indices=request_major_to_packed,
        pcp_metadata=pcp_metadata,
    )
