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
"""`HeadTailPcpShardingPlan`: DualChunkSwap PCP sharding, after upstream
vLLM's MRV2 `PCPManager`.

  * A prefill is cut into `2 * pcp_size` chunks: rank r takes chunks r and
    `2 * pcp_size - 1 - r`, balancing causal attention cost.
  * A decode is replicated: every PCP rank runs it.
  * Each rank runs its chunks as rank-local requests, split prefill segments
    first and replicated decodes last. Sparse MLA attention reads the split
    prefills' context from a DCP-gathered KV workspace (`pcp_kv_gather.py`).

Every rank builds the same partition (`PcpBatchPlan.from_schedule`) from the same
schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from vllm.utils.math_utils import cdiv

from vllm_torchtpu.layers.adapter.pcp_kv_gather import pcp_chunk_pages
from vllm_torchtpu.layers.core.pcp_metadata import PcpMetadata
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.pcp.sharding_plan import (
    PcpShardingPlan,
    grow_host_token_buffers,
    round_up_to_bucket,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class PcpSegment:
    """One contiguous run of a request's query tokens on one PCP rank."""

    # Index of the request in the step's (chunk's) request list.
    req_idx: int
    # [start, stop) of the segment in the step's request-major token order.
    global_start: int
    global_stop: int
    # [start, stop) of the segment in the rank's local token order.
    local_start: int
    local_stop: int

    @property
    def num_tokens(self) -> int:
        return self.global_stop - self.global_start


@dataclass
class PcpBatchPlan:
    """The PCP partition of one step, identical on every rank."""

    pcp_size: int
    num_global_tokens: int
    # Local token count per rank before padding.
    per_rank_num_tokens: list[int]
    # Every rank pads its local batch to this many tokens.
    num_local_tokens_padded: int
    # Segments of each rank in local token order: split first, then decodes.
    segments_by_rank: list[list[PcpSegment]]
    # Per request: True if every rank runs the whole query (decodes).
    replicated: list[bool]
    # Row of global token g in the rank-major gathered buffer
    # `[pcp_size * num_local_tokens_padded]`; replicated tokens use rank 0's.
    hidden_restore_idx: list[int]
    # Global token held at gathered row e (0 for padding, see gathered_valid).
    padded_gather_idx: list[int]
    gathered_valid: list[bool] = field(default_factory=list)

    def local_segments(self, rank: int) -> list[PcpSegment]:
        return self.segments_by_rank[rank]

    def num_split_segments(self, rank: int) -> int:
        segments = self.segments_by_rank[rank]
        return sum(1 for s in segments if not self.replicated[s.req_idx])

    @classmethod
    def from_schedule(
        cls,
        num_scheduled_tokens: list[int],
        is_prefilling: list[bool],
        pcp_size: int,
        num_local_tokens_padded: int | None = None,
    ) -> PcpBatchPlan:
        """Partition one step's request-major tokens across the PCP ranks."""
        if pcp_size < 1:
            raise ValueError(f"pcp_size must be positive, got {pcp_size}")
        if len(num_scheduled_tokens) != len(is_prefilling):
            raise ValueError("num_scheduled_tokens and is_prefilling must match")
        query_start_loc = [0]
        for n in num_scheduled_tokens:
            query_start_loc.append(query_start_loc[-1] + int(n))
        num_global_tokens = query_start_loc[-1]

        replicated = [not p for p in is_prefilling]
        segments_by_rank = [
            _rank_segments(
                rank, num_scheduled_tokens, query_start_loc, replicated, pcp_size
            )
            for rank in range(pcp_size)
        ]
        per_rank_num_tokens = [
            sum(s.num_tokens for s in segs) for segs in segments_by_rank
        ]
        largest = max(per_rank_num_tokens) if per_rank_num_tokens else 0
        if num_local_tokens_padded is None:
            num_local_tokens_padded = largest
        elif num_local_tokens_padded < largest:
            raise ValueError(
                "PCP padded local token count is smaller than the largest rank-local "
                f"batch: {num_local_tokens_padded} < {largest}."
            )

        # PCP=2 example:
        #   global batch:       [A B C D E F G]
        #   rank 0 / rank 1:    [A B G] / [C D E F]
        #   padded gathered:    [A B G _ | C D E F]
        #   hidden_restore_idx: [0, 1, 4, 5, 6, 7, 2]
        #   padded_gather_idx:  [0, 1, 6, 0, 2, 3, 4, 5]
        # so global = gathered[hidden_restore_idx] and
        # padded_gathered = global[padded_gather_idx].
        num_gathered = num_local_tokens_padded * pcp_size
        hidden_restore_idx = [-1] * num_global_tokens
        padded_gather_idx = [0] * num_gathered
        gathered_valid = [False] * num_gathered
        for rank, segments in enumerate(segments_by_rank):
            base = rank * num_local_tokens_padded
            for seg in segments:
                n = seg.num_tokens
                row0 = base + seg.local_start
                padded_gather_idx[row0 : row0 + n] = range(
                    seg.global_start, seg.global_stop
                )
                gathered_valid[row0 : row0 + n] = [True] * n
                # A replicated token is restored from rank 0's copy only.
                if replicated[seg.req_idx] and rank != 0:
                    continue
                hidden_restore_idx[seg.global_start : seg.global_stop] = range(
                    row0, row0 + n
                )
        if -1 in hidden_restore_idx:
            raise AssertionError("PCP plan left a global token without an owner")

        return cls(
            pcp_size=pcp_size,
            num_global_tokens=num_global_tokens,
            per_rank_num_tokens=per_rank_num_tokens,
            num_local_tokens_padded=num_local_tokens_padded,
            segments_by_rank=segments_by_rank,
            replicated=replicated,
            hidden_restore_idx=hidden_restore_idx,
            padded_gather_idx=padded_gather_idx,
            gathered_valid=gathered_valid,
        )


def _iter_rank_chunks(
    rank: int,
    num_scheduled_tokens: list[int],
    replicated: list[bool],
    pcp_size: int,
):
    """Yield `(req_idx, query_offset, length)` of the chunks one rank owns.

    PCP=4:

        full:  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
        rank 0:  0                           7
        rank 1:      1                   6
        rank 2:          2           5
        rank 3:              3   4
    """
    num_chunks = 2 * pcp_size
    for req_idx, query_len in enumerate(num_scheduled_tokens):
        if query_len <= 0:
            continue
        if replicated[req_idx]:
            yield req_idx, 0, query_len
            continue
        chunk_size = (query_len + num_chunks - 1) // num_chunks
        for chunk_idx in (rank, num_chunks - 1 - rank):
            offset = chunk_idx * chunk_size
            length = min(chunk_size, query_len - offset)
            if length > 0:
                yield req_idx, offset, length


def _rank_segments(
    rank: int,
    num_scheduled_tokens: list[int],
    query_start_loc: list[int],
    replicated: list[bool],
    pcp_size: int,
) -> list[PcpSegment]:
    raw = []
    for req_idx, offset, length in _iter_rank_chunks(
        rank, num_scheduled_tokens, replicated, pcp_size
    ):
        start = query_start_loc[req_idx] + offset
        raw.append((req_idx, start, start + length))
    # Split prefills first, then decodes; request and position order within.
    raw.sort(key=lambda s: (replicated[s[0]], s[0], s[1]))
    segments = []
    local = 0
    for req_idx, start, stop in raw:
        segments.append(PcpSegment(req_idx, start, stop, local, local + stop - start))
        local += stop - start
    return segments


def pcp_rank_token_counts(
    num_scheduled_tokens: list[int],
    is_prefilling: list[bool],
    pcp_size: int,
) -> list[int]:
    """Local token count of each rank, without building the plan."""
    replicated = [not p for p in is_prefilling]
    return [
        sum(
            n
            for _, _, n in _iter_rank_chunks(
                rank, num_scheduled_tokens, replicated, pcp_size
            )
        )
        for rank in range(pcp_size)
    ]


def pcp_max_local_tokens(
    max_num_batched_tokens: int, max_num_seqs: int, pcp_size: int
) -> int:
    """Upper bound on `pcp_rank_token_counts` for any schedulable step.

    A rank holds 1/pcp of the prefill tokens, plus up to 2 per prefill from
    rounding its two chunks up, plus every (replicated) decode.
    """
    return cdiv(int(max_num_batched_tokens), int(pcp_size)) + 3 * int(max_num_seqs)


@dataclass
class PcpWorkspacePlan:
    """Workspace page w holds virtual page `ws_vpage[w]` of split request
    `ws_req[w]`; the split requests' pages are laid out back to back. Unused
    pages repeat request 0's page 0 and are never read."""

    num_pages: int
    ws_req: list[int]
    ws_vpage: list[int]
    # First workspace page of each split request (by `req_idx`).
    page_offset: dict[int, int]
    num_used_pages: int

    @classmethod
    def from_requests(
        cls,
        split_req_indices: list[int],
        seq_lens: list[int],
        virtual_page_size: int,
        num_pages: int,
    ) -> PcpWorkspacePlan:
        """Lay the split requests' virtual pages out in a `num_pages` workspace."""
        if virtual_page_size <= 0 or num_pages <= 0:
            raise ValueError("workspace page size and page count must be positive")
        ws_req = [0] * num_pages
        ws_vpage = [0] * num_pages
        page_offset: dict[int, int] = {}
        cursor = 0
        for req_idx in split_req_indices:
            n = (int(seq_lens[req_idx]) + virtual_page_size - 1) // virtual_page_size
            if cursor + n > num_pages:
                raise ValueError(
                    "PCP KV workspace too small for this step: needs at least "
                    f"{(cursor + n) * virtual_page_size} context tokens, has "
                    f"{num_pages * virtual_page_size} (TPU_PCP_KV_WORKSPACE_TOKENS)."
                )
            page_offset[req_idx] = cursor
            for v in range(n):
                ws_req[cursor + v] = req_idx
                ws_vpage[cursor + v] = v
            cursor += n
        return cls(num_pages, ws_req, ws_vpage, page_offset, cursor)


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


def _block_tables_for_rows(
    device: torch.device,
    block_table_obj: Any,
    rows: np.ndarray | None,
    num_rows: int,
    num_blocks: int,
    keepalive: list[torch.Tensor],
) -> torch.Tensor:
    table = torch.zeros((num_rows, num_blocks), dtype=torch.int32)
    if rows is not None and len(rows):
        source = block_table_obj.get_cpu_tensor()
        table[: len(rows)] = source[:, :num_blocks].index_select(
            0, torch.as_tensor(rows, dtype=torch.long)
        )
    keepalive.append(table)  # outlive the non-blocking upload
    return table.reshape(-1).to(device, non_blocking=True)


def _int32_fields(**fields: Any) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(value, dtype=np.int32).reshape(-1)
        for name, value in fields.items()
    }


class HeadTailPcpShardingPlan(PcpShardingPlan):
    """DualChunkSwap: rank `r` holds chunks `r` and `2P-1-r` of each split
    request (see `PcpBatchPlan.from_schedule`)."""

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
        super().__init__(pcp_size, pcp_rank, dcp_size)
        self._workspace_tokens = int(workspace_tokens)
        self._max_num_seqs = int(max_num_seqs)
        self.set_page_size(page_size)
        self._max_local_num_tokens = pcp_max_local_tokens(
            max_num_batched_tokens, max_num_seqs, self.pcp_size
        )

    def _reset_step_state(self) -> None:
        super()._reset_step_state()
        # KV-gather: rank-local requests [0, local_num_split_reqs) are split
        # prefill segments, the rest `rep_num_reqs` replicated decodes.
        self.local_num_split_reqs = 0
        self.rep_num_reqs = 0
        # Persistent-batch rows of the split prefills and of the decodes (None
        # in dummy steps: zero block tables).
        self.split_block_rows: np.ndarray | None = None
        self.rep_block_rows: np.ndarray | None = None
        # The step's `PcpMetadata` fields except the block tables, as int32 host
        # arrays; None when attention needs no `PcpMetadata`.
        self.host_metadata: dict[str, np.ndarray] | None = None
        # `host_metadata` uploaded once for every KV cache group of the step,
        # and host tensors outliving their non-blocking uploads.
        self._staged: dict[str, torch.Tensor] | None = None
        self._keepalive: list[torch.Tensor] = []

    @property
    def local_reqs_per_req(self) -> int:
        # A split prefill is two DualChunkSwap chunks on every rank, each its
        # own local request; a decode is one.
        return 2

    @property
    def max_local_num_tokens(self) -> int:
        return self._max_local_num_tokens

    def set_page_size(self, page_size: int) -> None:
        """Also size the KV-gather workspace in `page_size`-token pages."""
        super().set_page_size(page_size)
        # A DCP virtual page is one physical page on each DCP rank.
        self.virtual_page_size = self.page_size * self.dcp_size
        # A multiple of `chunk_pages`, so the chunk loop never slices past it.
        chunk_pages = pcp_chunk_pages(self.page_size, self.dcp_size)
        # Rounded up: a request of up to `workspace_tokens` (>= max_model_len,
        # see `validate_pcp_manager_config`) always fits on its own.
        num_pages = max(1, cdiv(self._workspace_tokens, self.virtual_page_size))
        self.num_ws_pages = cdiv(num_pages, chunk_pages) * chunk_pages
        logger.info(
            "PCP manager KV workspace: page_size=%d, %d workspace pages (%d tokens)",
            self.page_size,
            self.num_ws_pages,
            self.num_ws_pages * self.virtual_page_size,
        )

    def _rep_window_size(self, target_num_reqs: int) -> int:
        """Decode window rows: `min(target_num_reqs, max_num_seqs)` rounded up
        to a multiple of `dcp_size` (`streamindex_topk_dcp` requires it).

        `target_num_reqs` is the local request bucket, where a split prefill
        counts as two requests; decodes are not split, so a step holds at most
        `max_num_seqs` of them.
        """
        # TODO(weiyulin): Evaluate if we need to bucket the window by the step's
        # decode count (`num_reqs_paddings`) and compile each size, so steps with
        # few or no decodes don't run `max_num_seqs` padding rows.
        num_reqs = max(min(target_num_reqs, self._max_num_seqs), 1)
        return cdiv(num_reqs, self.dcp_size) * self.dcp_size

    def num_reqs_in_chunk(
        self, is_prefilling: Any, seq_lens: Any, capacity: int
    ) -> int:
        """Also stop before the split prefills' contexts overflow the KV
        workspace, so the runner runs the rest as more forwards."""
        num_reqs = super().num_reqs_in_chunk(is_prefilling, seq_lens, capacity)
        used_pages = 0
        for i in range(num_reqs):
            if not is_prefilling[i]:
                continue
            used_pages += cdiv(int(seq_lens[i]), self.virtual_page_size)
            if used_pages > self.num_ws_pages:
                return max(i, 1)
        return num_reqs

    def request_distribution_counts(
        self, chunk_num_decode: int, num_reqs: int
    ) -> tuple[int, int]:
        del chunk_num_decode, num_reqs
        # The main metadata covers the split prefills; decodes carry their
        # own request window (`pcp_rep_*`).
        return 0, self.local_num_split_reqs

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
        del scheduler_output, use_max_model_len
        self._reset_step_state()
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
        local_padded = round_up_to_bucket(
            runner.num_tokens_paddings, max(1, max(rank_counts))
        )
        dp_target_bucket = getattr(runner, "_dp_target_bucket", None)
        if dp_target_bucket is not None and dp_target_bucket > local_padded:
            local_padded = int(dp_target_bucket)
        plan = PcpBatchPlan.from_schedule(
            q_lens, is_prefilling, self.pcp_size, num_local_tokens_padded=local_padded
        )

        # Snapshot the global request-major view before overwriting buffers.
        global_input_ids = runner.input_ids_cpu[:num_tokens].clone()
        global_positions = runner.positions_np[:num_tokens].copy()
        global_qsl = runner.query_start_loc_np[: num_reqs + 1].copy()
        global_seq_lens = runner.seq_lens_np[:num_reqs].copy()

        # Pack every rank's rows rank-major; this rank reads its own slice.
        num_packed = local_padded * self.pcp_size
        grow_host_token_buffers(runner, num_packed)
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

        segments = plan.local_segments(self.pcp_rank)
        split_reqs = [r for r in range(num_reqs) if not plan.replicated[r]]
        request_major_to_local = np.full(num_tokens, -1, dtype=np.int32)
        for seg in segments:
            request_major_to_local[seg.global_start : seg.global_stop] = np.arange(
                seg.local_start, seg.local_stop, dtype=np.int32
            )
        full_restore = np.asarray(plan.hidden_restore_idx, dtype=np.int64)

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
        self.token_slice = slice(local_start, local_start + local_padded)
        self.global_num_tokens = num_tokens
        self.global_padded_num_tokens = num_packed
        self.local_num_tokens = plan.per_rank_num_tokens[self.pcp_rank]
        self.local_padded_num_tokens = local_padded
        self.logits_indices_cpu = torch.from_numpy(logits_indices.astype(np.int32))
        self.logits_local_indices_cpu = torch.from_numpy(logits_local)
        self.logits_owner_mask_cpu = torch.from_numpy(owner_mask)
        self.logits_all_local = logits_all_local
        self.request_major_to_local = request_major_to_local

        # Rank-local request view, written over the runner's request buffers.
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
        for i, seg in enumerate(segments):
            local_qsl[i + 1] = seg.local_stop
            start_pos = int(num_computed[seg.req_idx]) + (
                seg.global_start - int(global_qsl[seg.req_idx])
            )
            local_seq_lens[i] = start_pos + seg.num_tokens
            local_block_rows[i] = start_index + seg.req_idx
        last = int(local_qsl[local_num_reqs])
        rep_segments = segments[num_split:]
        assert all(s.num_tokens == 1 for s in rep_segments)
        num_rep = len(rep_segments)
        rep_row_start = int(local_qsl[num_split])
        rep_seq_lens = local_seq_lens[num_split:local_num_reqs].copy()
        self.rep_block_rows = local_block_rows[num_split:].copy()

        local_qsl[local_num_reqs + 1 :] = last

        # Split view: the split prefills alone, request-major.
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
        cursor = 0
        for k, r in enumerate(split_reqs):
            start_tok = int(global_qsl[r])
            restore[cursor : cursor + q_lens[r]] = full_restore[
                start_tok : start_tok + q_lens[r]
            ]
            cursor += q_lens[r]
            split_qsl[k + 1] = cursor
            split_seq_lens[k] = int(global_seq_lens[r])

        # KV-gather workspace for the split prefills.
        ws = PcpWorkspacePlan.from_requests(
            list(range(len(split_reqs))),
            [int(x) for x in split_seq_lens],
            self.virtual_page_size,
            self.num_ws_pages,
        )
        local_ws_off = np.zeros(target_num_reqs, dtype=np.int32)
        for i, seg in enumerate(segments[:num_split]):
            local_ws_off[i] = ws.page_offset[split_pos[seg.req_idx]]

        # The decodes as a fixed `window`-row request window at `rep_row_start`.
        window = self._rep_window_size(target_num_reqs)
        rep_seq_lens_padded = np.zeros(window, dtype=np.int32)
        rep_seq_lens_padded[:num_rep] = rep_seq_lens

        self.local_num_reqs = local_num_reqs
        self.local_block_rows = local_block_rows
        self.local_num_split_reqs = num_split
        self.rep_num_reqs = num_rep
        self.split_block_rows = np.asarray(
            [start_index + r for r in split_reqs], dtype=np.int64
        )
        self.host_metadata = _int32_fields(
            split_query_start_loc=split_qsl,
            split_seq_lens=split_seq_lens,
            hidden_restore_idx=restore,
            rep_row_idx=np.minimum(np.arange(window) + rep_row_start, local_padded - 1),
            rep_query_start_loc=np.minimum(np.arange(window + 1), num_rep),
            rep_seq_lens=rep_seq_lens_padded,
            rep_request_distribution=[num_rep] * 3,
            ws_req=ws.ws_req,
            ws_vpage=ws.ws_vpage,
            ws_page_offset=local_ws_off,
            num_split_reqs=[len(split_reqs)],
            ws_used_pages=[ws.num_used_pages],
        )

    def prepare_dummy(
        self, *, num_tokens: int, num_reqs: int, max_num_tokens: int
    ) -> None:
        self._reset_step_state()
        num_tokens = int(num_tokens)
        num_reqs = int(num_reqs)
        num_packed = num_tokens * self.pcp_size
        local_start = self.pcp_rank * num_tokens
        self.token_slice = slice(local_start, local_start + num_tokens)
        self.global_num_tokens = num_packed
        self.global_padded_num_tokens = num_packed
        self.local_num_tokens = num_tokens
        self.local_padded_num_tokens = num_tokens
        self.local_num_reqs = num_reqs
        # The serving shapes with valid indices but no requests. Warmup runs
        # the split-prefill branch (one split request, one workspace page), so
        # serving finds it built.
        window = self._rep_window_size(num_reqs)
        num_rep = min(num_reqs, window)
        self.host_metadata = _int32_fields(
            split_query_start_loc=np.arange(num_reqs + 1),
            split_seq_lens=np.ones(num_reqs),
            hidden_restore_idx=(
                np.arange(max(self.pcp_size * int(max_num_tokens), num_packed))
                % num_packed
            ),
            rep_row_idx=np.minimum(np.arange(window), num_tokens - 1),
            rep_query_start_loc=np.minimum(np.arange(window + 1), num_rep),
            rep_seq_lens=np.arange(window) < num_rep,
            rep_request_distribution=[num_rep] * 3,
            ws_req=np.zeros(self.num_ws_pages),
            ws_vpage=np.zeros(self.num_ws_pages),
            ws_page_offset=np.zeros(num_reqs),
            num_split_reqs=[1],
            ws_used_pages=[1],
        )

    def build_metadata(
        self,
        *,
        runner: Any,
        block_table_obj: Any,
        target_num_reqs: int,
        target_num_blocks: int,
        split_block_tables: torch.Tensor | None = None,
    ) -> PcpMetadata | None:
        if self.host_metadata is None:
            return None
        device = runner.device
        if self._staged is None:
            # One upload per field: views of a shared device buffer are
            # rejected by the compiled graph (RuntimeProgramInputMismatch).
            hosts = {k: torch.from_numpy(v) for k, v in self.host_metadata.items()}
            self._keepalive.extend(hosts.values())
            self._staged = {
                k: t.to(device, non_blocking=True) for k, t in hosts.items()
            }
        if split_block_tables is None:
            split_block_tables = _block_tables_for_rows(
                device,
                block_table_obj,
                self.split_block_rows,
                target_num_reqs,
                target_num_blocks,
                self._keepalive,
            )
        rep_block_tables = _block_tables_for_rows(
            device,
            block_table_obj,
            self.rep_block_rows,
            len(self.host_metadata["rep_row_idx"]),
            target_num_blocks,
            self._keepalive,
        )
        return PcpMetadata(
            split_block_tables=split_block_tables,
            rep_block_tables=rep_block_tables,
            **self._staged,
        )
