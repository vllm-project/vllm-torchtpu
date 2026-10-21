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
"""PCP manager (`USE_TPU_PCP_MANAGER=1`), after upstream vLLM's MRV2
`PCPManager`.

Each PCP rank runs its DualChunkSwap slice of the step's prefills as rank-local
requests; decodes are replicated. Attention reads the split prefills' context
from a DCP-gathered KV workspace (`pcp_kv_gather.py`). Token rows are packed
rank-major (`[pcp_size * num_local_tokens_padded]`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.pcp import (
    all_reduce_sum,
    get_pcp_group,
)
from vllm_torchtpu.layers.adapter.pcp_kv_gather import pcp_chunk_pages
from vllm_torchtpu.layers.core.sequence_layout import (
    SequenceLayoutDescriptor,
    SequenceLayoutKind,
    SequenceLayoutPlan,
    _first_ge,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.pcp_partition import (
    build_pcp_batch_plan,
    build_workspace_plan,
    pcp_max_local_tokens,
    pcp_rank_token_counts,
)

logger = init_logger(__name__)

PCP_KV_GATHER_SEQUENCE_LAYOUT_DESCRIPTOR = SequenceLayoutDescriptor(
    kind=SequenceLayoutKind.PARTIAL,
    protocol="pcp_kv_gather",
    version=1,
)


@dataclass(frozen=True)
class PcpKvGatherPlan(SequenceLayoutPlan):
    # Rank-local requests: [0, local_num_split_reqs) are split prefill
    # segments, the rest replicated decodes starting at row `rep_row_start`.
    local_num_reqs: int = 0
    local_num_split_reqs: int = 0
    rep_row_start: int = 0
    rep_num_reqs: int = 0
    rep_seq_lens: np.ndarray | None = None
    rep_block_rows: np.ndarray | None = None
    # Persistent-batch row of each local request.
    local_block_rows: np.ndarray | None = None
    # The split prefills alone, request-major (what the split ops write).
    split_query_start_loc_cpu: torch.Tensor | None = None
    split_seq_lens_cpu: torch.Tensor | None = None
    split_block_rows: np.ndarray | None = None
    num_split_reqs: int = 0
    # No split request: every rank holds every sampling row.
    logits_all_local: bool = False
    # Split token g is PCP-gathered row `hidden_restore_idx[g]` (padded to one
    # length for every bucket).
    hidden_restore_idx: torch.Tensor | None = None
    dcp_size: int = 1
    # Workspace (see `build_workspace_plan`), and the first workspace page of
    # each local split segment.
    ws_req: np.ndarray | None = None
    ws_vpage: np.ndarray | None = None
    local_ws_page_offset: np.ndarray | None = None
    ws_used_pages: int = 0
    # This rank's local row of each request-major token, -1 if another rank
    # holds it. Decodes are on every rank, after the split prefills.
    request_major_to_local: np.ndarray | None = None

    def local_index_for_request_major_token(
        self, request_major_index: int
    ) -> int | None:
        if self.request_major_to_local is None:
            return super().local_index_for_request_major_token(request_major_index)
        local = int(self.request_major_to_local[int(request_major_index)])
        return None if local < 0 else local


def rep_window_size(max_reqs: int, dcp_size: int) -> int:
    """Decode window rows: `max_reqs` rounded up to a multiple of `dcp_size`."""
    dcp_size = max(int(dcp_size), 1)
    return -(-max(int(max_reqs), 1) // dcp_size) * dcp_size


@dataclass
class PcpKvGatherFields:
    """Device tensors for `PcpMetadata` (same shapes for dummy runs, so warmup
    traces the serving program)."""

    split_query_start_loc: torch.Tensor
    split_seq_lens: torch.Tensor
    hidden_restore_idx: torch.Tensor
    # The decodes as a fixed `rep_window`-row request window at `rep_row_idx`.
    rep_window: int
    rep_row_idx: torch.Tensor
    rep_query_start_loc: torch.Tensor
    rep_seq_lens: torch.Tensor
    rep_request_distribution: torch.Tensor
    ws_req: torch.Tensor
    ws_vpage: torch.Tensor
    ws_page_offset: torch.Tensor
    num_split_reqs: torch.Tensor  # [1]
    ws_used_pages: torch.Tensor  # [1]
    # Persistent-batch rows; None for dummy runs.
    split_block_rows: np.ndarray | None = None
    rep_block_rows: np.ndarray | None = None
    # Host tensors that must outlive their non-blocking uploads.
    keepalive: list = field(default_factory=list)


def pcp_kv_gather_fields(
    plan: PcpKvGatherPlan, device: torch.device
) -> PcpKvGatherFields:
    assert plan.split_query_start_loc_cpu is not None
    assert plan.split_seq_lens_cpu is not None
    assert plan.hidden_restore_idx is not None
    assert plan.ws_req is not None and plan.ws_vpage is not None
    assert plan.local_ws_page_offset is not None
    max_reqs = int(plan.local_ws_page_offset.shape[0])
    window = rep_window_size(max_reqs, plan.dcp_size)
    n = plan.rep_num_reqs
    rep_row_idx = np.minimum(
        np.arange(window, dtype=np.int32) + plan.rep_row_start,
        plan.local_padded_num_tokens - 1,
    )
    rep_seq_lens = np.zeros(window, dtype=np.int32)
    rep_seq_lens[:n] = plan.rep_seq_lens
    # One upload per field: views of a shared device buffer are rejected by
    # the compiled graph (RuntimeProgramInputMismatch).
    parts = {
        "split_query_start_loc": plan.split_query_start_loc_cpu.numpy(),
        "split_seq_lens": plan.split_seq_lens_cpu.numpy(),
        "rep_row_idx": rep_row_idx,
        "rep_query_start_loc": np.minimum(np.arange(window + 1, dtype=np.int32), n),
        "rep_seq_lens": rep_seq_lens,
        "rep_request_distribution": np.array([n, n, n], dtype=np.int32),
        "ws_req": plan.ws_req,
        "ws_vpage": plan.ws_vpage,
        "ws_page_offset": plan.local_ws_page_offset,
        "num_split_reqs": np.array([plan.num_split_reqs], dtype=np.int32),
        "ws_used_pages": np.array([plan.ws_used_pages], dtype=np.int32),
        "hidden_restore_idx": plan.hidden_restore_idx.numpy(),
    }
    hosts = {
        name: torch.from_numpy(np.ascontiguousarray(v, dtype=np.int32).reshape(-1))
        for name, v in parts.items()
    }
    return PcpKvGatherFields(
        rep_window=window,
        rep_block_rows=plan.rep_block_rows,
        split_block_rows=plan.split_block_rows,
        keepalive=list(hosts.values()),
        **{name: t.to(device, non_blocking=True) for name, t in hosts.items()},
    )


def dummy_pcp_kv_gather_fields(
    num_tokens: int,
    num_reqs: int,
    pcp_size: int,
    dcp_size: int,
    num_ws_pages: int,
    restore_capacity: int,
    device: torch.device,
) -> PcpKvGatherFields:
    """Same shapes as `pcp_kv_gather_fields`, for dummy/warmup runs."""
    window = rep_window_size(num_reqs, dcp_size)
    return PcpKvGatherFields(
        split_query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32).to(device),
        split_seq_lens=torch.ones((num_reqs,), dtype=torch.int32).to(device),
        hidden_restore_idx=torch.remainder(
            torch.arange(max(restore_capacity, num_tokens * pcp_size)),
            num_tokens * pcp_size,
        )
        .to(torch.int32)
        .to(device),
        rep_window=window,
        rep_row_idx=torch.clamp(
            torch.arange(window, dtype=torch.int32), max=num_tokens - 1
        ).to(device),
        rep_query_start_loc=torch.clamp(
            torch.arange(window + 1, dtype=torch.int32), max=num_reqs
        ).to(device),
        rep_seq_lens=(torch.arange(window) < num_reqs).to(torch.int32).to(device),
        rep_request_distribution=torch.tensor([num_reqs] * 3, dtype=torch.int32).to(
            device
        ),
        ws_req=torch.zeros((num_ws_pages,), dtype=torch.int32).to(device),
        ws_vpage=torch.zeros((num_ws_pages,), dtype=torch.int32).to(device),
        ws_page_offset=torch.zeros((num_reqs,), dtype=torch.int32).to(device),
        # Warmup runs the split-prefill branch, so serving finds it built.
        num_split_reqs=torch.ones((1,), dtype=torch.int32).to(device),
        ws_used_pages=torch.ones((1,), dtype=torch.int32).to(device),
    )


def validate_pcp_manager_config(vllm_config: Any) -> None:
    """Reject configurations the PCP manager does not handle yet."""
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    pcp_size = int(parallel_config.prefill_context_parallel_size)
    dcp_size = int(parallel_config.decode_context_parallel_size)
    tp_size = int(parallel_config.tensor_parallel_size)
    if pcp_size <= 1 or dcp_size <= 1:
        raise NotImplementedError(
            "USE_TPU_PCP_MANAGER needs prefill and decode context parallel "
            f"sizes > 1, got pcp={pcp_size}, dcp={dcp_size}."
        )
    hf_config = getattr(model_config, "hf_text_config", None)
    if not getattr(model_config, "use_mla", False) or not hasattr(
        hf_config, "index_topk"
    ):
        raise NotImplementedError("The PCP manager supports sparse MLA models only.")
    if parallel_config.is_moe_model and not parallel_config.enable_expert_parallel:
        raise NotImplementedError(
            "PCP on an MoE model requires --enable-expert-parallel."
        )
    if dcp_size != tp_size * pcp_size:
        raise NotImplementedError(
            "PCP x DCP requires decode_context_parallel_size == "
            "tensor_parallel_size * prefill_context_parallel_size, got "
            f"dcp={dcp_size}, tp={tp_size}, pcp={pcp_size}."
        )
    if int(parallel_config.pipeline_parallel_size) != 1:
        raise NotImplementedError("PCP x DCP does not support pipeline parallelism.")
    if int(parallel_config.data_parallel_size) != 1:
        raise NotImplementedError("PCP x DCP does not support data parallelism.")
    if vllm_config.speculative_config is not None:
        raise NotImplementedError(
            "PCP x DCP does not support speculative decoding yet."
        )
    if vllm_config.lora_config is not None:
        raise NotImplementedError("PCP x DCP does not support LoRA.")
    if getattr(model_config, "is_multimodal_model", False):
        raise NotImplementedError("PCP x DCP does not support multimodal inputs.")


def _grow_host_token_buffers(runner: Any, required: int) -> None:
    """Make the runner's host token staging buffers hold `required` rows."""
    if len(runner.arange_np) < required:
        runner.arange_np = np.arange(required, dtype=np.int64)
    current = int(runner.input_ids_cpu.shape[0])
    if current >= required:
        return
    for name in ("input_ids_cpu", "positions_cpu"):
        old = getattr(runner, name)
        new = torch.zeros(required, dtype=old.dtype, device=old.device)
        new[:current] = old
        setattr(runner, name, new)
    runner.positions_np = runner.positions_cpu.numpy()


def mla_kv_page_size(kv_cache_config: Any) -> int:
    """The single block size of the MLA KV cache groups (the sparse-MLA
    cache's page size), looking inside `UniformTypeKVCacheSpecs`."""
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, UniformTypeKVCacheSpecs

    sizes: set[int] = set()
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        specs = (
            spec.kv_cache_specs.values()
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else (spec,)
        )
        sizes.update(
            int(s.block_size) for s in specs if isinstance(s, MLAAttentionSpec)
        )
    if len(sizes) != 1:
        raise ValueError(
            f"PCP manager expects one MLA KV cache block size, got {sorted(sizes)}"
        )
    return sizes.pop()


class PcpManager:
    """PCP x DCP manager with KV-gather attention."""

    uses_selected_logits_hidden_states: bool = True

    def __init__(
        self,
        pcp_size: int,
        pcp_rank: int,
        dcp_size: int,
        page_size: int,
        workspace_tokens: int,
        max_num_batched_tokens: int,
        max_num_seqs: int,
    ) -> None:
        self.pcp_size = int(pcp_size)
        self.pcp_rank = int(pcp_rank)
        self.dcp_size = int(dcp_size)
        self._workspace_tokens = int(workspace_tokens)
        self.set_page_size(page_size)
        self._max_local_num_tokens = pcp_max_local_tokens(
            max_num_batched_tokens, max_num_seqs, self.pcp_size
        )

    def set_page_size(self, page_size: int) -> None:
        """Size the KV-gather workspace for `page_size`-token KV pages.

        Must match the sparse-MLA cache's page size, which can differ from
        `cache_config.block_size` once vLLM unifies the KV cache groups'
        page sizes; the runner calls this again with the real size after
        `initialize_kv_cache` (see `mla_kv_page_size`).
        """
        self.page_size = int(page_size)
        # A DCP virtual page is one physical page on each DCP rank.
        self.virtual_page_size = self.page_size * self.dcp_size
        # A multiple of `chunk_pages`, so the chunk loop never slices past it.
        chunk_pages = pcp_chunk_pages(self.page_size, self.dcp_size)
        num_pages = max(1, self._workspace_tokens // self.virtual_page_size)
        self.num_ws_pages = -(-num_pages // chunk_pages) * chunk_pages

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> PcpManager:
        validate_pcp_manager_config(vllm_config)
        parallel_config = vllm_config.parallel_config
        group = get_pcp_group()
        pcp_rank = 0 if group is None else int(group.rank_in_group)
        return cls(
            pcp_size=parallel_config.prefill_context_parallel_size,
            pcp_rank=pcp_rank,
            dcp_size=parallel_config.decode_context_parallel_size,
            page_size=vllm_config.cache_config.block_size,
            workspace_tokens=envs.TPU_PCP_KV_WORKSPACE_TOKENS,
            max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
            max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
        )

    @property
    def max_local_num_tokens(self) -> int:
        return self._max_local_num_tokens

    @property
    def local_reqs_per_req(self) -> int:
        # A split prefill is two DualChunkSwap chunks on every rank, each its
        # own local request; a decode is one.
        return 2

    def num_reqs_in_chunk(self, is_prefilling: Any, capacity: int) -> int:
        """A split prefill counts as 2 local requests, a decode as 1."""
        # TODO(weiyulin): Remove once the head and tail chunks share one
        # page table row.
        num_local = 0
        for i, prefilling in enumerate(is_prefilling):
            num_local += self.local_reqs_per_req if prefilling else 1
            if num_local > capacity:
                return max(i, 1)
        return len(is_prefilling)

    def reserve_host_token_capacity(
        self, runner: Any, required_num_tokens: int
    ) -> None:
        # Replication and per-rank padding can grow the packed layout.
        _grow_host_token_buffers(runner, int(required_num_tokens) * self.pcp_size)

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
    ) -> PcpKvGatherPlan:
        del scheduler_output, use_max_model_len
        num_tokens = int(total_num_scheduled_tokens)
        q_lens = [int(n) for n in num_scheduled_tokens_per_req[:num_reqs]]
        rows = slice(start_index, start_index + num_reqs)
        num_computed = runner.input_batch.num_computed_tokens_cpu[rows].astype(np.int64)
        num_prompt = np.asarray(
            runner.input_batch.num_prompt_tokens[rows], dtype=np.int64
        )
        is_prefilling = [bool(c < p) for c, p in zip(num_computed, num_prompt)]

        # Every rank pads its local batch to the same compile bucket.
        rank_counts = pcp_rank_token_counts(q_lens, is_prefilling, self.pcp_size)
        if max(rank_counts) > self._max_local_num_tokens:
            raise RuntimeError(
                f"PCP rank has {max(rank_counts)} local tokens, above the "
                f"warmed-up cap {self._max_local_num_tokens}."
            )
        local_padded = _first_ge(runner.num_tokens_paddings, max(1, max(rank_counts)))
        dp_target_bucket = getattr(runner, "_dp_target_bucket", None)
        if dp_target_bucket is not None and dp_target_bucket > local_padded:
            local_padded = int(dp_target_bucket)
        plan = build_pcp_batch_plan(
            q_lens, is_prefilling, self.pcp_size, num_local_tokens_padded=local_padded
        )

        # Snapshot the global request-major view before overwriting buffers.
        global_input_ids = runner.input_ids_cpu[:num_tokens].clone()
        global_positions = runner.positions_np[:num_tokens].copy()
        global_qsl = runner.query_start_loc_np[: num_reqs + 1].copy()
        global_seq_lens = runner.seq_lens_np[:num_reqs].copy()

        # Pack every rank's rows rank-major; this rank reads its own slice.
        num_packed = local_padded * self.pcp_size
        _grow_host_token_buffers(runner, num_packed)
        gather_idx = torch.as_tensor(plan.padded_gather_idx, dtype=torch.long)
        valid = torch.as_tensor(plan.gathered_valid, dtype=torch.bool)
        packed_ids = torch.where(valid, global_input_ids.index_select(0, gather_idx), 0)
        runner.input_ids_cpu[:num_packed] = packed_ids
        packed_pos = np.where(
            np.asarray(plan.gathered_valid),
            global_positions[np.asarray(plan.padded_gather_idx, dtype=np.int64)],
            0,
        )
        runner.positions_np[:num_packed] = packed_pos

        # Rank-local request view, written over the runner's request buffers.
        segments = plan.local_segments(self.pcp_rank)
        local_num_reqs = len(segments)
        if local_num_reqs > target_num_reqs:
            raise RuntimeError(
                "PCP local request count exceeds the runner's request bucket: "
                f"{local_num_reqs} > {target_num_reqs}."
            )
        local_qsl = runner.query_start_loc_np
        local_qsl[0] = 0
        local_seq_lens = runner.seq_lens_np
        local_seq_lens[:] = 0
        local_block_rows = np.empty(local_num_reqs, dtype=np.int64)
        num_split = plan.num_split_segments(self.pcp_rank)
        request_major_to_local = np.full(num_tokens, -1, dtype=np.int32)
        for i, seg in enumerate(segments):
            request_major_to_local[seg.global_start : seg.global_stop] = np.arange(
                seg.local_start, seg.local_stop, dtype=np.int32
            )
            local_qsl[i + 1] = seg.local_stop
            start_pos = int(num_computed[seg.req_idx]) + (
                seg.global_start - int(global_qsl[seg.req_idx])
            )
            local_seq_lens[i] = start_pos + seg.num_tokens
            local_block_rows[i] = start_index + seg.req_idx
        last = int(local_qsl[local_num_reqs])
        rep_segments = segments[num_split:]
        assert all(s.num_tokens == 1 for s in rep_segments)
        rep_row_start = int(local_qsl[num_split])
        rep_seq_lens = local_seq_lens[num_split:local_num_reqs].copy()
        rep_block_rows = local_block_rows[num_split:].copy()

        local_qsl[local_num_reqs + 1 :] = last

        # Split view: the split prefills alone, request-major.
        split_reqs = [r for r in range(num_reqs) if not plan.replicated[r]]
        split_pos = {r: k for k, r in enumerate(split_reqs)}
        num_split_tokens = sum(q_lens[r] for r in split_reqs)
        split_qsl = np.full(target_num_reqs + 1, num_split_tokens, dtype=np.int32)
        split_qsl[0] = 0
        split_seq_lens = np.zeros(target_num_reqs, dtype=np.int32)
        # Padded to one length for every bucket.
        restore_cap = max(
            num_packed, self.pcp_size * int(runner.num_tokens_paddings[-1])
        )
        restore = np.zeros(restore_cap, dtype=np.int32)
        full_restore = np.asarray(plan.hidden_restore_idx, dtype=np.int64)
        cursor = 0
        for k, r in enumerate(split_reqs):
            start_tok = int(global_qsl[r])
            restore[cursor : cursor + q_lens[r]] = full_restore[
                start_tok : start_tok + q_lens[r]
            ]
            cursor += q_lens[r]
            split_qsl[k + 1] = cursor
            split_seq_lens[k] = int(global_seq_lens[r])
        split_block_rows = np.asarray(
            [start_index + r for r in split_reqs], dtype=np.int64
        )

        # KV-gather workspace for the split prefills.
        ws = build_workspace_plan(
            list(range(len(split_reqs))),
            [int(x) for x in split_seq_lens],
            self.virtual_page_size,
            self.num_ws_pages,
        )
        local_ws_off = np.zeros(target_num_reqs, dtype=np.int32)
        for i, seg in enumerate(segments[:num_split]):
            local_ws_off[i] = ws.page_offset[split_pos[seg.req_idx]]

        # Sampling rows: the owner of each request's last token contributes
        # it, the others zeros, summed over PCP. With nothing split every rank
        # holds every row and nothing is summed.
        global_qsl_padded = np.full(target_num_reqs + 1, num_tokens, dtype=np.int32)
        global_qsl_padded[: num_reqs + 1] = global_qsl
        logits_indices = global_qsl_padded[1 : padded_num_reqs + 1] - 1
        logits_all_local = not split_reqs
        if logits_all_local:
            logits_local = np.zeros(padded_num_reqs, dtype=np.int32)
            for seg in segments:
                logits_local[seg.req_idx] = seg.local_stop - 1
            owner_mask = np.ones(padded_num_reqs, dtype=bool)
        else:
            last_rows = full_restore[logits_indices.astype(np.int64)]
            owner_mask = last_rows // local_padded == self.pcp_rank
            owner_mask[num_reqs:] = False
            logits_local = (last_rows % local_padded).astype(np.int32)

        local_start = self.pcp_rank * local_padded
        return PcpKvGatherPlan(
            descriptor=PCP_KV_GATHER_SEQUENCE_LAYOUT_DESCRIPTOR,
            token_slice=slice(local_start, local_start + local_padded),
            global_num_tokens=num_tokens,
            global_padded_num_tokens=num_packed,
            local_num_tokens=plan.per_rank_num_tokens[self.pcp_rank],
            local_padded_num_tokens=local_padded,
            logits_indices_cpu=torch.from_numpy(logits_indices.astype(np.int32)),
            logits_local_indices_cpu=torch.from_numpy(logits_local),
            logits_owner_mask_cpu=torch.from_numpy(owner_mask),
            local_num_reqs=local_num_reqs,
            local_num_split_reqs=num_split,
            rep_row_start=rep_row_start,
            rep_num_reqs=len(rep_segments),
            rep_seq_lens=rep_seq_lens,
            rep_block_rows=rep_block_rows,
            local_block_rows=local_block_rows,
            split_query_start_loc_cpu=torch.from_numpy(split_qsl),
            split_seq_lens_cpu=torch.from_numpy(split_seq_lens),
            split_block_rows=split_block_rows,
            num_split_reqs=len(split_reqs),
            logits_all_local=logits_all_local,
            hidden_restore_idx=torch.from_numpy(restore),
            dcp_size=self.dcp_size,
            ws_req=np.asarray(ws.ws_req, dtype=np.int32),
            ws_vpage=np.asarray(ws.ws_vpage, dtype=np.int32),
            local_ws_page_offset=local_ws_off,
            ws_used_pages=ws.num_used_pages,
            request_major_to_local=request_major_to_local,
        )

    def prepare_dummy(
        self,
        *,
        num_tokens: int,
        num_reqs: int,
        kv_cache_initialized: bool,
    ) -> PcpKvGatherPlan:
        del kv_cache_initialized
        num_tokens = int(num_tokens)
        num_packed = num_tokens * self.pcp_size
        local_start = self.pcp_rank * num_tokens
        return PcpKvGatherPlan(
            descriptor=PCP_KV_GATHER_SEQUENCE_LAYOUT_DESCRIPTOR,
            token_slice=slice(local_start, local_start + num_tokens),
            global_num_tokens=num_packed,
            global_padded_num_tokens=num_packed,
            local_num_tokens=num_tokens,
            local_padded_num_tokens=num_tokens,
            local_num_reqs=int(num_reqs),
        )

    def maybe_select_logits_hidden_states(
        self,
        hidden_states: torch.Tensor,
        plan: SequenceLayoutPlan | None,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        del logits_indices
        if not isinstance(plan, PcpKvGatherPlan):
            return None
        if plan.logits_local_indices_cpu is None or plan.logits_owner_mask_cpu is None:
            return None
        device = hidden_states.device
        local = plan.logits_local_indices_cpu.to(device, non_blocking=True)
        selected = hidden_states.index_select(0, local)
        if plan.logits_all_local:
            return selected
        mask = plan.logits_owner_mask_cpu.to(device, non_blocking=True)
        selected = torch.where(mask.unsqueeze(1), selected, torch.zeros_like(selected))
        return all_reduce_sum(selected)

    def finalize_hidden_states(
        self,
        hidden_states: torch.Tensor,
        plan: SequenceLayoutPlan | None,
    ) -> torch.Tensor:
        del plan
        return hidden_states
