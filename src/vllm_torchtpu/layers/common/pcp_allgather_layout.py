# SPDX-License-Identifier: Apache-2.0
"""Head-tail token layout for the all-gather PCP path.

The all-gather PCP kernel splits ONE request's `S = 2*pcp*C` current tokens into
`2*pcp` chunks of size `C` and gives rank `r` chunk `r` (head) and chunk
`2*pcp-1-r` (tail), so its local buffer is `[head_chunk | tail_chunk]`. Pairing
an early chunk with a late one balances the causal triangle across ranks.

This differs from the streaming path's layout in two ways that matter:

* Streaming uses a FIXED interleave cycle (`pcp_query_chunk_ranges`); head-tail
  is a ONE-SHOT split whose chunk size `C = padded_S // (2*pcp)` changes with
  the padded token count of each step.
* The all-gather kernel presents the head and tail as two fused seqs of the
  SAME request (`distribution = [0, 0, 2]`), so it is single-request by
  construction -- which is why it requires `--max-num-seqs 1`.

Pure numpy, no device dependencies, so it can be unit-tested on CPU.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "headtail_row_perm",
    "headtail_inv_row",
    "to_rank_order",
    "build_headtail_pcp_meta",
    "headtail_slot_for_global_token",
]


def headtail_row_perm(pcp_size: int) -> list[int]:
    """Chunk ids in rank-major order: rank r contributes (r, 2P-1-r)."""
    return [c for r in range(pcp_size) for c in (r, 2 * pcp_size - 1 - r)]


def headtail_inv_row(pcp_size: int) -> np.ndarray:
    """Natural chunk index -> its slot in the rank-order layout."""
    inv = np.empty(2 * pcp_size, np.int64)
    inv[headtail_row_perm(pcp_size)] = np.arange(2 * pcp_size)
    return inv


def to_rank_order(x: np.ndarray, pcp_size: int, chunk: int) -> np.ndarray:
    """Token order -> rank order (what each rank's local shard must contain).

    `x` is indexed on axis 0 by global token position and must be padded to
    `2 * pcp_size * chunk` rows.
    """
    x = np.asarray(x)
    expected = 2 * pcp_size * chunk
    if x.shape[0] != expected:
        raise ValueError(
            f"to_rank_order expects {expected} rows (2*pcp*C), got {x.shape[0]}")
    return (x.reshape(2 * pcp_size, chunk,
                      *x.shape[1:])[headtail_row_perm(pcp_size)].reshape(
                          x.shape))


def build_headtail_pcp_meta(
    pcp_size: int,
    chunk: int,
    num_current: int,
    max_num_seqs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-rank fused current-phase metadata for the all-gather kernel.

    Returns `(cu_q_lens, q_pos_offsets)` with shapes
    `[pcp_size, max_num_seqs + 1]` and `[pcp_size, max_num_seqs]`.

    For rank r: `cu = [0, C, C + tail_real, ...]` and
    `q_pos_offsets = [r*C, (2P-1-r)*C, ...]`. The head chunk is always fully
    real; the tail chunk is clamped, because with `num_current < 2*pcp*C` the
    later chunks are partly (or wholly) padding -- and how much differs per
    rank, which is exactly why these must be pcp-sharded rather than shared.
    """
    if pcp_size <= 0 or chunk <= 0:
        raise ValueError("pcp_size and chunk must be positive.")
    two_p = 2 * pcp_size
    cu = np.zeros((pcp_size, max_num_seqs + 1), np.int32)
    qpos = np.zeros((pcp_size, max_num_seqs), np.int32)
    for r in range(pcp_size):
        tail_off = (two_p - 1 - r) * chunk
        tail_real = int(np.clip(num_current - tail_off, 0, chunk))
        cu[r, 1] = chunk
        cu[r, 2:] = chunk + tail_real
        qpos[r, 0] = r * chunk
        qpos[r, 1] = tail_off
    return cu, qpos


def headtail_slot_for_global_token(global_token: int, chunk: int,
                                   pcp_size: int) -> int:
    """Slot of a global token position in the rank-major head-tail buffer.

    Global token `g` sits in chunk `g // C`, which the rank-order layout places
    at slot `inv_row[g // C] * C + (g % C)`. Used to remap `logits_indices`
    (the last real token of the request) onto the packed layout.
    """
    inv = headtail_inv_row(pcp_size)
    return int(inv[global_token // chunk]) * chunk + (global_token % chunk)
