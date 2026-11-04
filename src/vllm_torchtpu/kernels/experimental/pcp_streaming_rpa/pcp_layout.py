# Copyright 2025 Google LLC
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
"""PCP rank-major token layout helpers."""

import numpy as np


def pcp_token_owner_start_offsets(
    num_scheduled_tokens_per_req: np.ndarray,
    token_owner_start_offsets_per_req: list[int] | np.ndarray | None,
) -> np.ndarray:
    """Normalize the coordinate used to assign query tokens to PCP ranks."""
    if token_owner_start_offsets_per_req is None:
        return np.zeros(num_scheduled_tokens_per_req.size, dtype=np.int64)
    starts = np.asarray(token_owner_start_offsets_per_req, dtype=np.int64)
    if starts.size != num_scheduled_tokens_per_req.size:
        raise ValueError(
            "token_owner_start_offsets_per_req must have the same number of "
            "entries as num_scheduled_tokens_per_req."
        )
    if np.any(starts < 0):
        raise ValueError("token_owner_start_offsets_per_req must be non-negative.")
    return starts


def _resolve_token_owner_start_offsets(
    num_scheduled_tokens_per_req: np.ndarray,
    token_owner_start_offsets_per_req: list[int] | np.ndarray | None,
    token_start_offsets_per_req: list[int] | np.ndarray | None,
) -> np.ndarray:
    if (
        token_owner_start_offsets_per_req is not None
        and token_start_offsets_per_req is not None
    ):
        raise ValueError(
            "Specify only token_owner_start_offsets_per_req; "
            "token_start_offsets_per_req is a legacy alias."
        )
    owner_starts = token_owner_start_offsets_per_req
    if owner_starts is None:
        owner_starts = token_start_offsets_per_req
    return pcp_token_owner_start_offsets(num_scheduled_tokens_per_req, owner_starts)


def pcp_token_owner_chunk_ranges(
    q_len: int,
    token_owner_start: int,
    pcp_rank: int,
    pcp_size: int,
    interleave_size: int,
):
    """Yield absolute owner-coordinate ranges assigned to one PCP rank."""
    if q_len <= 0:
        return
    cycle = pcp_size * interleave_size
    q_end = token_owner_start + q_len
    chunk_start = (token_owner_start // cycle) * cycle + pcp_rank * interleave_size
    if chunk_start + interleave_size <= token_owner_start:
        chunk_start += cycle
    while chunk_start < q_end:
        overlap_start = max(chunk_start, token_owner_start)
        overlap_end = min(chunk_start + interleave_size, q_end)
        if overlap_end > overlap_start:
            yield overlap_start, overlap_end
        chunk_start += cycle


def pcp_query_chunk_ranges(
    q_len: int, q_global_base: int, pcp_rank: int, pcp_size: int, interleave_size: int
):
    """Compatibility alias for :func:`pcp_token_owner_chunk_ranges`."""
    yield from pcp_token_owner_chunk_ranges(
        q_len, q_global_base, pcp_rank, pcp_size, interleave_size
    )


def pcp_local_token_counts(
    num_scheduled_tokens_per_req: list[int] | np.ndarray,
    pcp_size: int,
    interleave_size: int,
    token_owner_start_offsets_per_req: list[int] | np.ndarray | None = None,
    *,
    token_start_offsets_per_req: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Return local token counts for each PCP rank under chunk interleave."""
    if pcp_size <= 1:
        return np.array([int(np.sum(num_scheduled_tokens_per_req))], dtype=np.int32)
    if interleave_size <= 0:
        raise ValueError("cp_kv_cache_interleave_size must be positive.")

    tokens = np.asarray(num_scheduled_tokens_per_req, dtype=np.int64)
    if tokens.size == 0:
        return np.zeros(pcp_size, dtype=np.int32)
    owner_starts = _resolve_token_owner_start_offsets(
        tokens,
        token_owner_start_offsets_per_req,
        token_start_offsets_per_req,
    )
    counts = np.zeros(pcp_size, dtype=np.int64)
    for q_len, token_owner_start in zip(tokens, owner_starts):
        for rank in range(pcp_size):
            for chunk_start, chunk_end in pcp_token_owner_chunk_ranges(
                int(q_len), int(token_owner_start), rank, pcp_size, interleave_size
            ):
                counts[rank] += chunk_end - chunk_start
    return counts.astype(np.int32)


def build_pcp_rank_major_token_order(
    num_scheduled_tokens_per_req: list[int] | np.ndarray,
    pcp_size: int,
    interleave_size: int,
    padded_num_tokens: int,
    token_owner_start_offsets_per_req: list[int] | np.ndarray | None = None,
    *,
    token_start_offsets_per_req: list[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build token reorder indices for rank-major PCP chunk packing.

    Returns:
        token_order: [padded_num_tokens] source indices in original per-DP
            request-major token order. Padding entries are -1.
        inverse_order: [total_num_tokens] destination indices in packed order.
    """
    if pcp_size <= 1:
        total = int(np.sum(num_scheduled_tokens_per_req))
        token_order = np.full(padded_num_tokens, -1, dtype=np.int64)
        token_order[:total] = np.arange(total, dtype=np.int64)
        return token_order, np.arange(total, dtype=np.int64)
    if interleave_size <= 0:
        raise ValueError("cp_kv_cache_interleave_size must be positive.")
    if padded_num_tokens % pcp_size != 0:
        raise ValueError(f"{padded_num_tokens=} must be divisible by {pcp_size=}.")

    local_padded_num_tokens = padded_num_tokens // pcp_size
    token_order = np.full(padded_num_tokens, -1, dtype=np.int64)
    total_num_tokens = int(np.sum(num_scheduled_tokens_per_req))
    inverse_order = np.full(total_num_tokens, -1, dtype=np.int64)
    owner_starts = _resolve_token_owner_start_offsets(
        np.asarray(num_scheduled_tokens_per_req, dtype=np.int64),
        token_owner_start_offsets_per_req,
        token_start_offsets_per_req,
    )

    req_start = 0
    rank_offsets = np.zeros(pcp_size, dtype=np.int64)
    for num_tokens, token_owner_start in zip(
        num_scheduled_tokens_per_req, owner_starts
    ):
        num_tokens = int(num_tokens)
        for rank in range(pcp_size):
            for chunk_start, chunk_end in pcp_token_owner_chunk_ranges(
                num_tokens, int(token_owner_start), rank, pcp_size, interleave_size
            ):
                chunk_len = chunk_end - chunk_start
                dst_start = rank * local_padded_num_tokens + rank_offsets[rank]
                dst_end = dst_start + chunk_len
                if dst_end > (rank + 1) * local_padded_num_tokens:
                    raise ValueError(
                        "PCP local token count exceeds padded local capacity."
                    )
                local_chunk_start = chunk_start - token_owner_start
                local_chunk_end = chunk_end - token_owner_start
                src = np.arange(
                    req_start + local_chunk_start,
                    req_start + local_chunk_end,
                    dtype=np.int64,
                )
                token_order[dst_start:dst_end] = src
                inverse_order[src] = np.arange(dst_start, dst_end, dtype=np.int64)
                rank_offsets[rank] += chunk_len
        req_start += num_tokens

    if total_num_tokens and np.any(inverse_order < 0):
        raise ValueError("PCP token packing did not cover all tokens.")
    return token_order, inverse_order


def apply_pcp_rank_major_token_order(
    input_ids_cpu: np.ndarray,
    positions_cpu: np.ndarray,
    num_scheduled_tokens_per_req: list[int] | np.ndarray,
    pcp_size: int,
    interleave_size: int,
    padded_num_tokens: int,
    mrope_positions_cpu: np.ndarray | None = None,
    token_owner_start_offsets_per_req: list[int] | np.ndarray | None = None,
    *,
    token_start_offsets_per_req: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Reorder per-DP token arrays into PCP rank-major chunk order."""
    total_num_tokens = int(np.sum(num_scheduled_tokens_per_req))
    token_order, inverse_order = build_pcp_rank_major_token_order(
        num_scheduled_tokens_per_req,
        pcp_size,
        interleave_size,
        padded_num_tokens,
        token_owner_start_offsets_per_req,
        token_start_offsets_per_req=token_start_offsets_per_req,
    )
    valid = token_order >= 0
    original_input_ids = input_ids_cpu[:total_num_tokens].copy()
    original_positions = positions_cpu[:total_num_tokens].copy()
    input_ids_cpu[:] = 0
    positions_cpu[:] = 0
    input_ids_cpu[valid] = original_input_ids[token_order[valid]]
    positions_cpu[valid] = original_positions[token_order[valid]]
    if mrope_positions_cpu is not None:
        original_mrope = mrope_positions_cpu[:, :total_num_tokens].copy()
        mrope_positions_cpu[:, :] = 0
        mrope_positions_cpu[:, valid] = original_mrope[:, token_order[valid]]
    return inverse_order


def build_pcp_logits_indices(
    num_scheduled_tokens_per_req: list[int] | np.ndarray,
    pcp_size: int,
    interleave_size: int,
    padded_num_tokens: int,
    token_offset: int = 0,
    token_owner_start_offsets_per_req: list[int] | np.ndarray | None = None,
    *,
    token_start_offsets_per_req: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Return global packed indices for each request's last query token."""
    _, inverse_order = build_pcp_rank_major_token_order(
        num_scheduled_tokens_per_req,
        pcp_size,
        interleave_size,
        padded_num_tokens,
        token_owner_start_offsets_per_req,
        token_start_offsets_per_req=token_start_offsets_per_req,
    )
    if len(num_scheduled_tokens_per_req) == 0:
        return np.empty((0,), dtype=np.int64)
    local_request_ends = np.cumsum(num_scheduled_tokens_per_req, dtype=np.int64) - 1
    return inverse_order[local_request_ends] + token_offset
