# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence layout planner for the all-gather PCP path.

Sibling of :mod:`pcp_sequence_layout` (the streaming planner). Both shard the
current-token axis across the pcp ring and hand the runner a
:class:`SequenceLayoutPlan`; they differ in *how* tokens are assigned to ranks.

Streaming walks a fixed interleave cycle, so every rank's token count is a
function of the cycle. All-gather does a ONE-SHOT head-tail split: the single
request's tokens are cut into ``2*pcp`` chunks of size ``C`` and rank ``r``
receives chunk ``r`` (head) plus chunk ``2*pcp-1-r`` (tail). Pairing an early
chunk with a late one balances the causal triangle, and it is why every rank
holds exactly ``2*C`` tokens -- there is no ragged per-rank count here.

The layout math itself lives in (and is unit-tested in)
:mod:`vllm_torchtpu.layers.common.pcp_allgather_layout`; this module only sizes
the buckets, drives the repack of the runner's host buffers, and builds the
logits bookkeeping.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm_torchtpu.distributed.pcp import get_pcp_rank as _get_native_pcp_rank
from vllm_torchtpu.distributed.pcp import \
    get_pcp_world_size as _get_native_pcp_world_size
from vllm_torchtpu.layers.common.pcp_allgather_layout import (
    build_headtail_pcp_meta, headtail_slot_for_global_token, to_rank_order)
from vllm_torchtpu.layers.common.pcp_sequence_layout import (
    PcpRequestSpan, _as_bool, _as_int, _ensure_host_token_buffer_capacity,
    _pcp_all_gather_equal_tokens, _pcp_all_reduce_sum)
from vllm_torchtpu.layers.common.sequence_layout import (
    PCP_ALLGATHER_SEQUENCE_LAYOUT_PROTOCOL, AllSequenceLayoutPlanner,
    SequenceLayoutDescriptor, SequenceLayoutKind, SequenceLayoutPlan,
    _first_ge)

PCP_ALLGATHER_SEQUENCE_LAYOUT_DESCRIPTOR = SequenceLayoutDescriptor(
    kind=SequenceLayoutKind.PARTIAL,
    protocol=PCP_ALLGATHER_SEQUENCE_LAYOUT_PROTOCOL,
    version=1,
)


@dataclass(frozen=True)
class PcpAllgatherSizing:
    """Token bucket sizing for one all-gather PCP step.

    ``local_padded_tokens`` must be even because the local buffer is literally
    ``[head_chunk | tail_chunk]`` -- two equal halves of size ``C``. An odd
    bucket has no valid ``C``, so the sizing is rejected rather than rounded.
    """
    pcp_size: int
    num_current_tokens: int
    chunk: int
    local_required_tokens: int
    local_padded_tokens: int
    global_padded_tokens: int


@dataclass(frozen=True)
class PcpAllgatherTensors:
    """Per-rank current-phase metadata for the all-gather attention kernel.

    Both tensors are pcp-sharded (leading dim ``pcp_size``) rather than
    replicated: rank ``r``'s two chunks sit at global offsets ``r*C`` and
    ``(2*pcp-1-r)*C``, and when the request does not fill the padded buffer the
    amount of *real* tail differs per rank. A single shared copy could not
    express either fact.
    """
    # int32 [pcp_size, max_num_seqs + 1]
    pcp_cu_q_lens_cpu: torch.Tensor
    # int32 [pcp_size, max_num_seqs]
    pcp_q_pos_offsets_cpu: torch.Tensor


@dataclass(frozen=True)
class PcpAllgatherPreparedBatch:
    local_token_slice: slice
    padded_total_num_scheduled_tokens: int
    local_total_num_scheduled_tokens: int
    local_padded_total_num_scheduled_tokens: int
    logits_indices_cpu: torch.Tensor
    logits_local_indices_cpu: torch.Tensor
    logits_owner_mask_cpu: torch.Tensor
    request_major_to_packed_token_indices: np.ndarray
    pcp_tensors: PcpAllgatherTensors


class PcpAllgatherSequenceLayoutEligibility:
    """Decides whether a scheduler chunk can use the all-gather PCP layout."""

    _BOUNDARY_MESSAGE = (
        "PCP all-gather sequence layout does not support prompt/decode "
        "boundary-crossing schedules yet.")

    def __init__(
        self,
        *,
        pcp_size: int = 1,
        dcp_size: int = 1,
        pipeline_parallel_size: int = 1,
        speculative_enabled: bool = False,
        is_kv_producer: bool | None = None,
    ):
        self.pcp_size = _as_int(pcp_size, 1)
        self.dcp_size = _as_int(dcp_size, 1)
        self.pipeline_parallel_size = _as_int(pipeline_parallel_size, 1)
        self.speculative_enabled = _as_bool(speculative_enabled)
        self.is_kv_producer = (is_kv_producer if isinstance(
            is_kv_producer, bool) else None)

    @classmethod
    def from_vllm_config(
            cls, vllm_config: Any) -> "PcpAllgatherSequenceLayoutEligibility":
        if vllm_config is None:
            return cls()

        parallel_config = getattr(vllm_config, "parallel_config", None)
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)

        is_kv_producer = None
        if kv_transfer_config is not None:
            raw_is_kv_producer = getattr(kv_transfer_config, "is_kv_producer",
                                         None)
            if isinstance(raw_is_kv_producer, bool):
                is_kv_producer = raw_is_kv_producer

        return cls(
            pcp_size=_as_int(
                getattr(parallel_config, "prefill_context_parallel_size", 1),
                1,
            ),
            dcp_size=_as_int(
                getattr(parallel_config, "decode_context_parallel_size", 1),
                1,
            ),
            pipeline_parallel_size=_as_int(
                getattr(parallel_config, "pipeline_parallel_size", 1),
                1,
            ),
            speculative_enabled=getattr(vllm_config, "speculative_config",
                                        None) is not None,
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
        num_tokens_paddings: Any | None = None,
        max_num_tokens: int | None = None,
        dp_target_bucket: int | None = None,
    ) -> tuple[PcpRequestSpan, ...] | None:
        """Return the chunk's spans, or None when PCP does not apply.

        Returning None means "fall back to the plain all-tokens layout" (the
        chunk has no scheduled tokens, or PCP is off). Anything the all-gather
        kernel cannot express raises instead -- see the decode note below.
        """
        del num_tokens_paddings, max_num_tokens, dp_target_bucket

        spans = self._build_request_spans(
            input_batch=input_batch,
            scheduler_output=scheduler_output,
            start_index=start_index,
            num_reqs=num_reqs,
            num_scheduled_tokens_per_req=num_scheduled_tokens_per_req,
        )

        if not self.enabled or not any(span.scheduled_tokens > 0
                                       for span in spans):
            return None

        if any(span.crosses_prompt_boundary for span in spans):
            raise NotImplementedError(self._BOUNDARY_MESSAGE)

        self._validate_execution_support()
        self._validate_single_request(spans)
        return spans

    def _build_request_spans(
        self,
        *,
        input_batch: Any,
        scheduler_output: Any,
        start_index: int,
        num_reqs: int,
        num_scheduled_tokens_per_req: np.ndarray,
    ) -> tuple[PcpRequestSpan, ...]:
        scheduled = np.asarray(num_scheduled_tokens_per_req, dtype=np.int64)
        if scheduled.ndim != 1:
            raise ValueError(
                "num_scheduled_tokens_per_req must be a 1D array.")
        if scheduled.size < num_reqs:
            raise ValueError(
                "num_scheduled_tokens_per_req must cover every request in "
                "the chunk.")

        spans: list[PcpRequestSpan] = []
        req_ids = input_batch.req_ids[start_index:start_index + num_reqs]
        for chunk_offset, req_id in enumerate(req_ids):
            if req_id is None:
                continue
            req_index = input_batch.req_id_to_index[req_id]
            scheduled_tokens = int(scheduled[chunk_offset])
            scheduler_tokens = getattr(scheduler_output,
                                       "num_scheduled_tokens", {}).get(req_id)
            if scheduler_tokens is not None:
                scheduled_tokens = int(scheduler_tokens)
            computed_tokens = int(
                input_batch.num_computed_tokens_cpu[req_index])
            spans.append(
                PcpRequestSpan(
                    computed_tokens=computed_tokens,
                    scheduled_tokens=scheduled_tokens,
                    prompt_tokens=int(
                        input_batch.num_prompt_tokens[req_index]),
                    q_start=computed_tokens,
                ))
        return tuple(spans)

    def _validate_execution_support(self) -> None:
        if self.is_kv_producer is False:
            raise NotImplementedError(
                "PCP all-gather sequence layout does not support KV "
                "consumer/decode workers yet. Disable PCP on decode workers.")
        if self.dcp_size > 1:
            raise NotImplementedError(
                "PCP all-gather sequence layout does not support DCP yet.")
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "PCP all-gather sequence layout does not support pipeline "
                "parallelism yet.")
        if self.speculative_enabled:
            raise NotImplementedError(
                "PCP all-gather sequence layout does not support speculative "
                "decoding yet.")

    @staticmethod
    def _validate_single_request(spans: tuple[PcpRequestSpan, ...]) -> None:
        # The kernel fuses the head and tail chunks as two seqs of the SAME
        # request (distribution = [0, 0, 2]); there is no room in that shape
        # for a second request.
        if len(spans) > 1:
            raise NotImplementedError(
                "PCP all-gather sequence layout is single-request by "
                f"construction, got num_reqs={len(spans)}. Run the server "
                "with --max-num-seqs 1, or use PCP_IMPL=streaming for "
                "multi-request batches.")

    # Decode needs no special case on this path.
    #
    # A decode step is just a 1-token prefill: the cache phase attends
    # everything already resident, and the current phase appends the single
    # new token. The head-tail machinery still applies -- the token lands in
    # chunk 0 and every other chunk is padding -- so the same
    # `distribution=[0, 0, 2]` current phase is correct as-is. This mirrors
    # tpu-inference's pcp_ragged_paged_attention, which likewise has no decode
    # branch.
    #
    # The cost is real but is a performance property, not a correctness one:
    # each decode step pads out to a full `2*pcp*C` attention launch, so most
    # of the launch is wasted work. Left as-is to match the reference
    # implementation; a dedicated decode distribution would be the optimization.


def _compute_sizing(
    *,
    pcp_size: int,
    num_current_tokens: int,
    num_tokens_paddings: Any,
    max_num_tokens: int | None,
    dp_target_bucket: int | None,
) -> PcpAllgatherSizing:
    """Size the head-tail buckets for `num_current_tokens` on `pcp_size` ranks.

    Mirrors the streaming planner: pick a compile bucket for the LOCAL token
    count and derive the global padded count as `local * pcp_size`, so the
    packed buffer is exactly `2 * pcp_size * C` -- divisible by `2*pcp` as the
    kernel requires.
    """
    if pcp_size <= 1:
        raise ValueError(
            "PCP all-gather requires prefill_context_parallel_size > 1.")
    if num_current_tokens <= 0:
        raise ValueError(
            "PCP all-gather requires at least one scheduled current token.")

    # Each rank holds two chunks, so round the per-rank requirement up to an
    # even number of tokens before bucketing.
    local_required_tokens = 2 * (
        (num_current_tokens + 2 * pcp_size - 1) // (2 * pcp_size))

    paddings = sorted(int(padding) for padding in num_tokens_paddings)
    local_padded_tokens = _first_ge(paddings, local_required_tokens)
    if dp_target_bucket is not None and dp_target_bucket > local_padded_tokens:
        local_padded_tokens = int(dp_target_bucket)
    if (max_num_tokens is not None
            and local_padded_tokens > int(max_num_tokens)):
        raise ValueError(
            "PCP all-gather local padded token length exceeds runner max "
            f"token buffer: {local_padded_tokens=} "
            f"max_num_tokens={int(max_num_tokens)}.")
    if local_padded_tokens % 2 != 0:
        raise ValueError(
            "PCP all-gather requires an even local token bucket so the local "
            "buffer can hold an equal head and tail chunk, but the selected "
            f"bucket is odd: {local_padded_tokens=} "
            f"{local_required_tokens=}. Remove odd entries from "
            "compilation_config.compile_sizes.")

    chunk = local_padded_tokens // 2
    return PcpAllgatherSizing(
        pcp_size=pcp_size,
        num_current_tokens=num_current_tokens,
        chunk=chunk,
        local_required_tokens=local_required_tokens,
        local_padded_tokens=local_padded_tokens,
        global_padded_tokens=local_padded_tokens * pcp_size,
    )


def _apply_headtail_rank_major_token_order(
    input_ids: np.ndarray,
    positions: np.ndarray,
    mrope_positions: np.ndarray | None,
    pcp_size: int,
    chunk: int,
) -> np.ndarray:
    """Repack the runner's host token buffers into rank-major head-tail order.

    Returns the request-major -> packed index map. The runner stages tokens in
    request-major order; every rank then reads a contiguous slice of the packed
    buffer, so the permutation has to be applied in place to the host buffers
    before the device copy.
    """
    num_tokens = input_ids.shape[0]
    input_ids[:] = to_rank_order(input_ids.copy(), pcp_size, chunk)
    positions[:] = to_rank_order(positions.copy(), pcp_size, chunk)
    if mrope_positions is not None:
        # mrope is [3, num_tokens]; to_rank_order permutes axis 0.
        mrope_positions[:] = to_rank_order(mrope_positions.copy().T, pcp_size,
                                           chunk).T

    request_major = np.arange(num_tokens, dtype=np.int64)
    packed_to_request_major = to_rank_order(request_major, pcp_size, chunk)
    request_major_to_packed = np.empty(num_tokens, dtype=np.int64)
    request_major_to_packed[packed_to_request_major] = request_major
    return request_major_to_packed


def prepare_pcp_allgather_sequence_layout(
    runner: Any,
    num_reqs: int,
    padded_num_reqs: int,
    spans: tuple[PcpRequestSpan, ...],
    *,
    get_native_pcp_rank: Any | None = None,
    get_native_pcp_world_size: Any | None = None,
) -> PcpAllgatherPreparedBatch:
    """Prepare native PCP all-gather host state for a real request."""
    if get_native_pcp_rank is None:
        get_native_pcp_rank = _get_native_pcp_rank
    if get_native_pcp_world_size is None:
        get_native_pcp_world_size = _get_native_pcp_world_size

    pcp_size = get_native_pcp_world_size()
    pcp_rank = get_native_pcp_rank()
    if not 0 <= pcp_rank < pcp_size:
        raise RuntimeError("Native PCP rank is out of range: "
                           f"{pcp_rank=} {pcp_size=}.")

    num_current_tokens = int(spans[0].scheduled_tokens)
    sizing = _compute_sizing(
        pcp_size=pcp_size,
        num_current_tokens=num_current_tokens,
        num_tokens_paddings=runner.num_tokens_paddings,
        max_num_tokens=runner.max_num_tokens,
        dp_target_bucket=getattr(runner, "_dp_target_bucket", None),
    )
    chunk = sizing.chunk
    global_padded = sizing.global_padded_tokens
    local_padded = sizing.local_padded_tokens

    # The packed buffer is pcp_size times the local bucket, which is wider than
    # anything the runner sized its host staging tensors for.
    _ensure_host_token_buffer_capacity(runner, global_padded)

    mrope_slice = None
    if runner.uses_mrope:
        mrope_slice = runner.mrope_positions.cpu[:, :global_padded].numpy()
    request_major_to_packed_token_indices = (
        _apply_headtail_rank_major_token_order(
            runner.input_ids_cpu[:global_padded].numpy(),
            runner.positions_np[:global_padded],
            mrope_slice,
            pcp_size,
            chunk,
        ))

    local_start = pcp_rank * local_padded
    local_end = local_start + local_padded
    local_token_slice = slice(local_start, local_end)

    # The request's last real token is global position S-1; the kernel wrote it
    # wherever the head-tail permutation put it.
    last_token_slot = headtail_slot_for_global_token(num_current_tokens - 1,
                                                     chunk, pcp_size)
    packed_logits_indices = np.full(num_reqs, last_token_slot, dtype=np.int32)

    logits_indices_cpu = torch.full((padded_num_reqs, ),
                                    -1,
                                    dtype=torch.int32,
                                    device="cpu")
    logits_indices_cpu[:num_reqs] = torch.from_numpy(packed_logits_indices)
    logits_owner_mask_np = ((packed_logits_indices >= local_start) &
                            (packed_logits_indices < local_end))
    logits_local_indices_np = np.zeros(num_reqs, dtype=np.int32)
    logits_local_indices_np[logits_owner_mask_np] = (
        packed_logits_indices[logits_owner_mask_np] - local_start)
    logits_local_indices_cpu = torch.zeros((padded_num_reqs, ),
                                           dtype=torch.int32,
                                           device="cpu")
    logits_local_indices_cpu[:num_reqs] = torch.from_numpy(
        logits_local_indices_np)
    logits_owner_mask_cpu = torch.zeros((padded_num_reqs, ),
                                        dtype=torch.bool,
                                        device="cpu")
    logits_owner_mask_cpu[:num_reqs] = torch.from_numpy(logits_owner_mask_np)

    cu_q_lens_np, q_pos_offsets_np = build_headtail_pcp_meta(
        pcp_size,
        chunk,
        num_current_tokens,
        int(runner.max_num_reqs),
    )
    pcp_tensors = PcpAllgatherTensors(
        pcp_cu_q_lens_cpu=torch.from_numpy(cu_q_lens_np),
        pcp_q_pos_offsets_cpu=torch.from_numpy(q_pos_offsets_np),
    )

    return PcpAllgatherPreparedBatch(
        local_token_slice=local_token_slice,
        padded_total_num_scheduled_tokens=global_padded,
        # Every rank owns exactly two chunks, so the "real" local count is the
        # part of those two chunks that is not padding.
        local_total_num_scheduled_tokens=int(cu_q_lens_np[pcp_rank, 2]),
        local_padded_total_num_scheduled_tokens=local_padded,
        logits_indices_cpu=logits_indices_cpu,
        logits_local_indices_cpu=logits_local_indices_cpu,
        logits_owner_mask_cpu=logits_owner_mask_cpu,
        request_major_to_packed_token_indices=(
            request_major_to_packed_token_indices),
        pcp_tensors=pcp_tensors,
    )


class PcpAllgatherSequenceLayoutPlanner:
    """Sequence layout planner for the head-tail all-gather PCP kernel.

    Implements the same protocol as :class:`PcpSequenceLayoutPlanner`. In
    addition to the returned :class:`SequenceLayoutPlan`, each successful
    :meth:`prepare_real` publishes the two all-gather-specific metadata tensors
    on ``self.last_pcp_tensors``; ``SequenceLayoutPlan`` is a frozen dataclass
    shared with the other layouts and must not grow kernel-specific fields.

    Runner contract: immediately after ``prepare_real`` returns a plan whose
    descriptor protocol is ``pcp_allgather``, read ``planner.last_pcp_tensors``
    and forward ``pcp_cu_q_lens_cpu`` / ``pcp_q_pos_offsets_cpu`` into the
    attention metadata for that step. The attribute is overwritten on every
    ``prepare_real`` and set to None whenever the step falls back to the plain
    all-tokens layout, so it must be consumed before the next call.
    """

    def __init__(self, eligibility: PcpAllgatherSequenceLayoutEligibility):
        self.eligibility = eligibility
        self._all_planner = AllSequenceLayoutPlanner()
        self.last_pcp_tensors: PcpAllgatherTensors | None = None

    @classmethod
    def from_vllm_config(
            cls, vllm_config: Any) -> "PcpAllgatherSequenceLayoutPlanner":
        return cls(
            PcpAllgatherSequenceLayoutEligibility.from_vllm_config(vllm_config))

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
        spans = self.eligibility.evaluate_runner_chunk(
            input_batch=runner.input_batch,
            scheduler_output=scheduler_output,
            start_index=start_index,
            num_reqs=num_reqs,
            num_scheduled_tokens_per_req=num_scheduled_tokens_per_req,
        )
        if spans is None:
            self.last_pcp_tensors = None
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

        prepared = prepare_pcp_allgather_sequence_layout(
            runner,
            num_reqs,
            padded_num_reqs,
            spans,
        )
        self.last_pcp_tensors = prepared.pcp_tensors
        return SequenceLayoutPlan(
            descriptor=PCP_ALLGATHER_SEQUENCE_LAYOUT_DESCRIPTOR,
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
        if not (self.enabled and kv_cache_initialized):
            return self._all_planner.prepare_dummy(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                kv_cache_initialized=kv_cache_initialized,
            )
        num_tokens = int(num_tokens)
        return SequenceLayoutPlan(
            descriptor=PCP_ALLGATHER_SEQUENCE_LAYOUT_DESCRIPTOR,
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
