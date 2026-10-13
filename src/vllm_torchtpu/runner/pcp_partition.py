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
"""Host-side PCP batch partitioning (pure Python), after upstream vLLM's
MRV2 `PCPManager`.

  * A prefill is cut into `2 * pcp_size` chunks (DualChunkSwap): rank r takes
    chunks r and `2 * pcp_size - 1 - r`, balancing causal attention cost.
  * A decode is replicated: every PCP rank runs it.

Locally, split prefill segments come first and replicated decodes last.
Every rank builds the same plan from the same schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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


def build_pcp_batch_plan(
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
    per_rank_num_tokens = [sum(s.num_tokens for s in segs) for segs in segments_by_rank]
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

    return PcpBatchPlan(
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


def build_workspace_plan(
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
    return PcpWorkspacePlan(num_pages, ws_req, ws_vpage, page_offset, cursor)
