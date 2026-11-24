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
"""Interleaved PCP sharding plan for non-MLA attention and GDN.

Deals each step's request-major tokens round-robin across PCP ranks in chunks
of `cp_kv_cache_interleave_size` tokens for streamed PCP attention and GDN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import AttentionSpec, UniformTypeKVCacheSpecs

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.pcp import (
    all_gather_equal_tokens,
    get_pcp_group,
    get_pcp_world_size,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.kernel import (
    PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
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
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.pcp.sharding_plan import (
    PcpDraftLayout,
    PcpShardingPlan,
    PcpShardingPlanName,
    ensure_host_token_buffer_capacity,
    round_up_to_bucket,
)

if TYPE_CHECKING:
    from vllm_torchtpu.layers.core.pcp_metadata import PcpMetadata

logger = init_logger(__name__)


def validate_interleaved_pcp_params(
    pcp_size: int,
    dcp_size: int,
    interleave_size: int,
) -> None:
    """Validate static topology and interleave parameters for interleaved PCP."""
    if pcp_size <= 1:
        raise NotImplementedError(
            "Interleaved PCP requires prefill_context_parallel_size > 1, "
            f"got {pcp_size}."
        )
    if dcp_size != 1:
        raise NotImplementedError(
            "Interleaved PCP does not support DCP "
            f"(decode_context_parallel_size > 1), got {dcp_size}."
        )
    if interleave_size <= 0:
        raise ValueError(
            f"Interleaved PCP requires cp_kv_cache_interleave_size > 0, "
            f"got {interleave_size}."
        )
    if PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE % interleave_size != 0:
        raise ValueError(
            f"cp_kv_cache_interleave_size ({interleave_size}) must divide "
            f"{PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE}."
        )


def interleaved_max_local_tokens(
    max_num_batched_tokens: int,
    pcp_size: int,
    interleave_size: int,
) -> int:
    """Upper bound on a rank's local token count, rounded up to 512."""
    budget = max(1, int(max_num_batched_tokens))
    cycle = int(pcp_size) * int(interleave_size)
    rank0_max = (budget // cycle) * int(interleave_size) + min(
        budget % cycle, int(interleave_size)
    )
    multiple = PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE
    return cdiv(max(1, rank0_max), multiple) * multiple


def attention_kv_page_size(
    kv_cache_config: Any,
    kernel_block_size: int | None = None,
) -> int | None:
    """Resolve the attention kernel page size for interleaved PCP validation."""
    if kernel_block_size is not None:
        return int(kernel_block_size)
    sizes: set[int] = set()
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = next(iter(spec.kv_cache_specs.values()))
        if isinstance(spec, AttentionSpec):
            sizes.add(int(spec.block_size))
    if not sizes:
        return None
    if len(sizes) > 1:
        raise ValueError(
            "Interleaved PCP requires one attention KV cache block size, "
            f"got {sorted(sizes)}."
        )
    return next(iter(sizes))


@dataclass(frozen=True)
class _ChunkTokenCounts:
    q_lens: np.ndarray
    absolute_query_starts: np.ndarray
    token_owner_starts: np.ndarray
    local_counts: np.ndarray
    local_padded_tokens: int


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


class InterleavedPcpShardingPlan(PcpShardingPlan):
    """Interleaved round-robin PCP token sharding plan (`dcp_size == 1`)."""

    plan_name = PcpShardingPlanName.INTERLEAVED
    requires_backend_preinit = True
    local_compile_token_multiple = PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE
    uses_pcp_streaming = True

    def __init__(
        self,
        *,
        pcp_size: int,
        pcp_rank: int,
        interleave_size: int,
        max_num_batched_tokens: int,
        page_size: int | None = None,
    ) -> None:
        validate_interleaved_pcp_params(pcp_size, 1, interleave_size)
        if not 0 <= pcp_rank < pcp_size:
            raise ValueError(f"pcp_rank must be in [0, {pcp_size}), got {pcp_rank}.")
        super().__init__(pcp_size=pcp_size, pcp_rank=pcp_rank, dcp_size=1)
        self.interleave_size = int(interleave_size)
        self._max_local_num_tokens = interleaved_max_local_tokens(
            max_num_batched_tokens, self.pcp_size, self.interleave_size
        )
        if page_size is not None:
            self.set_page_size(page_size)

    def _reset_step_state(self) -> None:
        super()._reset_step_state()
        self._request_major_to_packed: np.ndarray | None = None

    @property
    def max_local_num_tokens(self) -> int:
        return self._max_local_num_tokens

    def set_page_size(self, page_size: int) -> None:
        page_size = int(page_size)
        if page_size <= 0 or page_size % self.interleave_size != 0:
            raise ValueError(
                f"Attention KV page_size ({page_size}) must be a positive "
                f"multiple of cp_kv_cache_interleave_size ({self.interleave_size})."
            )
        super().set_page_size(page_size)

    def configure_kv_cache(
        self, kv_cache_config: Any, kernel_block_size: int | None = None
    ) -> None:
        page_size = attention_kv_page_size(kv_cache_config, kernel_block_size)
        if page_size is not None:
            self.set_page_size(page_size)

    def _evaluate_chunk(
        self,
        *,
        runner: Any,
        scheduler_output: Any,
        start_index: int,
        num_reqs: int,
        num_scheduled_tokens_per_req: Any,
    ) -> _ChunkTokenCounts | None:
        scheduled = np.asarray(num_scheduled_tokens_per_req, dtype=np.int64)
        if scheduled.ndim != 1:
            raise ValueError("num_scheduled_tokens_per_req must be a 1D array.")
        if scheduled.size < num_reqs:
            raise ValueError(
                "num_scheduled_tokens_per_req must cover every request in the chunk."
            )

        input_batch = runner.input_batch
        req_ids = input_batch.req_ids[start_index : start_index + num_reqs]
        q_lens_list: list[int] = []
        absolute_query_starts_list: list[int] = []
        crosses_boundary = False
        for chunk_offset, req_id in enumerate(req_ids):
            if req_id is None:
                continue
            req_index = input_batch.req_id_to_index[req_id]
            scheduled_tokens = int(scheduled[chunk_offset])
            scheduler_tokens = scheduler_output.num_scheduled_tokens.get(req_id)
            if scheduler_tokens is not None:
                scheduled_tokens = int(scheduler_tokens)
            computed_tokens = int(input_batch.num_computed_tokens_cpu[req_index])
            q_lens_list.append(scheduled_tokens)
            absolute_query_starts_list.append(computed_tokens)
            if scheduled_tokens > 0:
                prompt_tokens = int(input_batch.num_prompt_tokens[req_index])
                if (
                    computed_tokens < prompt_tokens
                    and computed_tokens + scheduled_tokens > prompt_tokens
                ):
                    crosses_boundary = True

        if not q_lens_list or not any(n > 0 for n in q_lens_list):
            return None
        if crosses_boundary:
            raise NotImplementedError(
                "Interleaved PCP does not support prompt/decode "
                "boundary-crossing schedules yet."
            )

        if any(n <= 0 for n in q_lens_list):
            raise NotImplementedError(
                "PCP streaming requires scheduled query tokens for every "
                "request in the chunk."
            )

        q_lens = np.asarray(q_lens_list, dtype=np.int32)
        absolute_query_starts = np.asarray(absolute_query_starts_list, dtype=np.int64)
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
        paddings = sorted(int(padding) for padding in runner.num_tokens_paddings)
        local_padded_tokens = round_up_to_bucket(paddings, local_required_tokens)
        dp_target_bucket = runner._dp_target_bucket
        if dp_target_bucket is not None and dp_target_bucket > local_padded_tokens:
            local_padded_tokens = int(dp_target_bucket)
        if runner.max_num_tokens is not None and local_padded_tokens > int(
            runner.max_num_tokens
        ):
            raise ValueError(
                "PCP local padded token length exceeds runner max token "
                f"buffer: {local_padded_tokens=} "
                f"max_num_tokens={int(runner.max_num_tokens)}."
            )
        return _ChunkTokenCounts(
            q_lens=q_lens,
            absolute_query_starts=absolute_query_starts,
            token_owner_starts=token_owner_starts,
            local_counts=local_counts,
            local_padded_tokens=local_padded_tokens,
        )

    def padded_num_tokens(
        self,
        *,
        runner: Any,
        scheduler_output: Any,
        start_index: int,
        num_scheduled_tokens_per_req: Any,
    ) -> int:
        scheduled = np.asarray(num_scheduled_tokens_per_req, dtype=np.int32)
        chunk = self._evaluate_chunk(
            runner=runner,
            scheduler_output=scheduler_output,
            start_index=start_index,
            num_reqs=int(scheduled.shape[0]),
            num_scheduled_tokens_per_req=scheduled,
        )
        if chunk is None:
            paddings = sorted(int(padding) for padding in runner.num_tokens_paddings)
            local_padded = round_up_to_bucket(paddings, int(scheduled.sum()))
            dp_target_bucket = runner._dp_target_bucket
            if dp_target_bucket is not None and dp_target_bucket > local_padded:
                local_padded = int(dp_target_bucket)
            return local_padded
        return chunk.local_padded_tokens

    def prepare_real(
        self,
        *,
        runner: Any,
        scheduler_output: Any,
        start_index: int,
        num_reqs: int,
        num_scheduled_tokens_per_req: Any,
        total_num_scheduled_tokens: int,
        use_max_model_len: bool,
        target_num_reqs: int,
        padded_num_reqs: int,
    ) -> None:
        del use_max_model_len, target_num_reqs
        self._reset_step_state()
        if num_reqs <= 0 or total_num_scheduled_tokens <= 0:
            return
        q_lens = np.asarray(num_scheduled_tokens_per_req[:num_reqs], dtype=np.int32)
        if int(q_lens.sum()) != int(total_num_scheduled_tokens):
            raise ValueError(
                "total_num_scheduled_tokens must match "
                "sum(num_scheduled_tokens_per_req): "
                f"got {total_num_scheduled_tokens} vs {int(q_lens.sum())}."
            )
        encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if encoder_inputs:
            req_ids = runner.input_batch.req_ids[start_index : start_index + num_reqs]
            if any(
                req_id is not None and bool(encoder_inputs.get(req_id))
                for req_id in req_ids
            ):
                raise NotImplementedError(
                    "Interleaved PCP manager does not support multimodal "
                    "encoder inputs."
                )

        chunk = self._evaluate_chunk(
            runner=runner,
            scheduler_output=scheduler_output,
            start_index=start_index,
            num_reqs=num_reqs,
            num_scheduled_tokens_per_req=q_lens,
        )
        if chunk is None:
            return

        if get_pcp_group() is not None:
            native_pcp_world_size = get_pcp_world_size()
            if native_pcp_world_size != self.pcp_size:
                raise RuntimeError(
                    "Native PCP group size does not match "
                    "prefill_context_parallel_size: "
                    f"{native_pcp_world_size=} pcp_size={self.pcp_size}."
                )
        if not 0 <= self.pcp_rank < self.pcp_size:
            raise RuntimeError(
                f"Native PCP rank is out of range: pcp_rank={self.pcp_rank} "
                f"pcp_size={self.pcp_size}."
            )

        local_padded_tokens = chunk.local_padded_tokens
        padded_total_num_scheduled_tokens = local_padded_tokens * self.pcp_size
        ensure_host_token_buffer_capacity(runner, padded_total_num_scheduled_tokens)

        mrope_slice = None
        if runner.uses_mrope:
            mrope_slice = runner.mrope_positions.cpu[
                :, :padded_total_num_scheduled_tokens
            ].numpy()
        request_major_to_packed = _apply_pcp_rank_major_token_order(
            runner.input_ids_cpu[:padded_total_num_scheduled_tokens].numpy(),
            runner.positions_np[:padded_total_num_scheduled_tokens],
            chunk.q_lens,
            self.pcp_size,
            self.interleave_size,
            padded_total_num_scheduled_tokens,
            mrope_slice,
            token_owner_start_offsets_per_req=chunk.token_owner_starts,
        )
        packed_to_request_major = np.full(
            padded_total_num_scheduled_tokens, -1, dtype=np.int64
        )
        packed_to_request_major[request_major_to_packed] = np.arange(
            request_major_to_packed.size, dtype=np.int64
        )

        local_total_tokens = int(chunk.local_counts[self.pcp_rank])
        local_start = self.pcp_rank * local_padded_tokens
        local_end = local_start + local_padded_tokens
        local_token_slice = slice(local_start, local_end)

        local_to_request_major = packed_to_request_major[local_start:local_end].copy()
        owned_tokens = (request_major_to_packed >= local_start) & (
            request_major_to_packed < local_end
        )
        request_major_to_local = np.where(
            owned_tokens, request_major_to_packed - local_start, -1
        ).astype(np.int32)

        pcp_logits_indices = _build_pcp_logits_indices(
            chunk.q_lens,
            self.pcp_size,
            self.interleave_size,
            padded_total_num_scheduled_tokens,
            token_owner_start_offsets_per_req=chunk.token_owner_starts,
        ).astype(np.int32)
        num_chunk_reqs = int(chunk.q_lens.size)
        logits_indices_cpu = torch.full(
            (padded_num_reqs,), -1, dtype=torch.int32, device="cpu"
        )
        logits_indices_cpu[:num_chunk_reqs] = torch.from_numpy(pcp_logits_indices)
        logits_owner_mask_np = (pcp_logits_indices >= local_start) & (
            pcp_logits_indices < local_end
        )
        logits_local_indices_np = np.zeros(num_chunk_reqs, dtype=np.int32)
        logits_local_indices_np[logits_owner_mask_np] = (
            pcp_logits_indices[logits_owner_mask_np] - local_start
        )
        logits_local_indices_cpu = torch.zeros(
            (padded_num_reqs,), dtype=torch.int32, device="cpu"
        )
        logits_local_indices_cpu[:num_chunk_reqs] = torch.from_numpy(
            logits_local_indices_np
        )
        logits_owner_mask_cpu = torch.zeros(
            (padded_num_reqs,), dtype=torch.bool, device="cpu"
        )
        logits_owner_mask_cpu[:num_chunk_reqs] = torch.from_numpy(logits_owner_mask_np)

        if envs.VLLM_TPU_DEBUG_PCP_LAYOUT:
            layout_debug = _debug_pcp_layout_window(
                runner,
                pcp_rank=self.pcp_rank,
                local_token_slice=local_token_slice,
                local_padded_tokens=local_padded_tokens,
                logits_indices=pcp_logits_indices,
            )
            logger.warning(
                "PCP_LAYOUT_DEBUG rank=%s scheduled=%s absolute_q_starts=%s "
                "token_owner_starts=%s "
                "local_counts=%s local_padded=%s global_padded=%s logits=%s "
                "query_start=%s seq_lens=%s layout=%s",
                self.pcp_rank,
                chunk.q_lens.tolist(),
                chunk.absolute_query_starts.tolist(),
                chunk.token_owner_starts.tolist(),
                chunk.local_counts.tolist(),
                local_padded_tokens,
                padded_total_num_scheduled_tokens,
                pcp_logits_indices.tolist(),
                np.cumsum(
                    np.concatenate([np.array([0], dtype=np.int32), chunk.q_lens])
                ).tolist(),
                runner.seq_lens_np[:num_reqs].copy().tolist(),
                layout_debug,
            )

        self.token_slice = local_token_slice
        self.global_num_tokens = int(total_num_scheduled_tokens)
        self.global_padded_num_tokens = padded_total_num_scheduled_tokens
        self.local_num_tokens = local_total_tokens
        self.local_padded_num_tokens = local_padded_tokens
        self.local_num_reqs = int(num_reqs)
        self.local_to_request_major_token_indices = local_to_request_major
        self.request_major_to_local = request_major_to_local
        self._request_major_to_packed = request_major_to_packed
        self.logits_indices_cpu = logits_indices_cpu
        hosts = (logits_local_indices_cpu, logits_owner_mask_cpu)
        self._keepalive.extend(hosts)
        self.logits_local_indices, self.logits_owner_mask = (
            t.to(runner.device, non_blocking=True) for t in hosts
        )

    def draft_layout(self) -> PcpDraftLayout | None:
        if (
            self.local_to_request_major_token_indices is None
            or self._request_major_to_packed is None
        ):
            return None
        num_tokens = self._request_major_to_packed.size
        assert num_tokens <= self.global_padded_num_tokens, (
            "Request-major mapping exceeds the PCP global token capacity: "
            f"{num_tokens=} {self.global_padded_num_tokens=}"
        )
        padded_request_to_packed = np.full(
            self.global_padded_num_tokens,
            -1,
            dtype=self._request_major_to_packed.dtype,
        )
        padded_request_to_packed[:num_tokens] = self._request_major_to_packed
        return PcpDraftLayout(
            local_to_request_major=self.local_to_request_major_token_indices.copy(),
            padded_request_to_packed=padded_request_to_packed,
            local_start=int(self.token_slice.start or 0),
            local_end=int(self.token_slice.stop),
        )

    def prepare_dummy(
        self, *, num_tokens: int, num_reqs: int, max_num_tokens: int
    ) -> None:
        del max_num_tokens
        self._reset_step_state()
        num_tokens = int(num_tokens)
        num_reqs = int(num_reqs)
        num_packed = num_tokens * self.pcp_size
        self.token_slice = slice(0, num_tokens)
        self.global_num_tokens = num_packed
        self.global_padded_num_tokens = num_packed
        self.local_num_tokens = num_tokens
        self.local_padded_num_tokens = num_tokens
        self.local_num_reqs = num_reqs

    def build_metadata(
        self,
        *,
        runner: Any,
        block_table_obj: Any,
        target_num_reqs: int,
        target_num_blocks: int,
        split_block_tables: torch.Tensor | None = None,
    ) -> PcpMetadata | None:
        del runner, block_table_obj, target_num_reqs
        del target_num_blocks, split_block_tables
        return None

    def finalize_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return all_gather_equal_tokens(hidden_states, dim=0)
