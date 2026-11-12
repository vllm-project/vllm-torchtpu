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
"""Plain-JAX addressing for the indexer K cache writer.

Every token of this step is mapped to the flat cache row it must land in,
`page * page_size + slot`, or to `SKIP_ROW` when it must not be written. The
writer (`kv_cache_append.scatter_rows`) only ever sees `(src, dst_rows)`, so it
knows nothing about requests, pages, ranks, or DCP:

    token ─┬─► get_page_and_slot ─────┬─► (page, slot) ─► get_dst_rows ─► writer
           └─► get_dcp_page_and_slot ─┘

The non-DCP and DCP paths differ only in the first box. This mirrors the
upstream vllm-torchtpu sparse-MLA KV cache writer
(`vllm_torchtpu/kernels/mla/kv_cache_utils.py`: `_row_positions`,
`get_page_and_slot`, `update_sparse_mla_kv_cache_dcp`).

Both boxes drop (map to `OOB_PAGE`) any token that:
  - is batch padding (`tok >= query_start_loc[-1]`),
  - has a negative KV position (`pos < 0`),
  - falls past the end of its row of the page table (`pos // page_size >=
    pages_per_seq`), or
  - resolves to a page id outside `[0, num_pages)` (DCP: `[0, num_pages *
    dcp_size)` before the per-rank `mod num_pages`), e.g. `-1` padding.
"""

import jax
import jax.numpy as jnp

# Page id that marks a token as "do not write". Any value >= num_pages works;
# a large constant keeps it recognizable when debugging.
OOB_PAGE = 2**30
# Flat destination row that marks a token as "do not write".
SKIP_ROW = -1

__all__ = [
    "OOB_PAGE",
    "SKIP_ROW",
    "get_dcp_page_and_slot",
    "get_dst_rows",
    "get_page_and_slot",
]


def _row_positions(
    num_tokens: int,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-token `(sequence, global KV position, in_batch)`.

    `pos` counts back from `seq_lens`, which already includes this step's tokens,
    so a chunked-prefill continuation resumes at the right offset. Padding tokens
    belong to no sequence: they get `seq_id = 0` and `in_batch = False`, so every
    caller masks them the same way.

    The sequence of each token comes from a `[T, S]` membership mask rather than
    `jnp.searchsorted`: on TPU the latter lowers to a while loop of gathers that
    cost ~0.7 ms for `T = 1024, S = 64`, while the mask is a few vector ops.

    Args:
      num_tokens: Number of token rows this step (`T`, possibly padded).
      seq_lens: `[num_seqs]` KV lengths including this step's tokens.
      query_start_loc: `[num_seqs + 1]` cumulative new-token counts.

    Returns:
      `(seq_id, pos, in_batch)`, each `[num_tokens]`.
    """
    num_seqs = seq_lens.shape[0]
    tok = jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
    member = (tok >= query_start_loc[None, :-1]) & (tok < query_start_loc[None, 1:])
    seq_id = jnp.sum(
        jnp.where(member, jnp.arange(num_seqs, dtype=jnp.int32)[None, :], 0),
        axis=1,
    )
    # pos = seq_lens[s] - q_len[s] + (tok - query_start_loc[s]).
    pos_offset = seq_lens - query_start_loc[1:]
    pos = tok[:, 0] + jnp.sum(jnp.where(member, pos_offset[None, :], 0), axis=1)
    return seq_id, pos, jnp.any(member, axis=1)


def _lookup_page(
    block_tables: jax.Array,
    num_seqs: int,
    seq_id: jax.Array,
    page_ordinal: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Returns `(raw_page, ordinal_valid)` for `block_tables[seq_id, ordinal]`."""
    block_tables_2d = block_tables.reshape(num_seqs, -1)
    pages_per_seq = block_tables_2d.shape[1]
    ordinal_valid = (page_ordinal >= 0) & (page_ordinal < pages_per_seq)
    safe_ordinal = jnp.clip(page_ordinal, 0, pages_per_seq - 1)
    return block_tables_2d[seq_id, safe_ordinal], ordinal_valid


def get_page_and_slot(
    num_tokens: int,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    *,
    page_size: int,
    num_pages: int,
) -> tuple[jax.Array, jax.Array]:
    """Target `(page, slot)` of every token in an unsharded paged cache.

    Args:
      num_tokens: Number of token rows this step.
      seq_lens: `[num_seqs]` KV lengths including this step's tokens.
      block_tables: `[num_seqs, pages_per_seq]` (or flattened) physical page ids.
      query_start_loc: `[num_seqs + 1]` cumulative new-token counts.
      page_size: Tokens per page.
      num_pages: Physical pages in the cache.

    Returns:
      `(page, slot)`, each `[num_tokens]` int32; `page == OOB_PAGE` for tokens
      that must not be written.
    """
    seq_id, pos, in_batch = _row_positions(num_tokens, seq_lens, query_start_loc)
    safe_pos = jnp.maximum(pos, 0)
    raw_page, ordinal_valid = _lookup_page(
        block_tables, seq_lens.shape[0], seq_id, safe_pos // page_size
    )
    valid = (
        in_batch & (pos >= 0) & ordinal_valid & (raw_page >= 0) & (raw_page < num_pages)
    )
    page = jnp.where(valid, raw_page, jnp.int32(OOB_PAGE))
    slot = jnp.where(valid, safe_pos % page_size, 0)
    return page, slot


def get_dcp_page_and_slot(
    num_tokens: int,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    dcp_rank: jax.Array | int,
    *,
    page_size: int,
    num_pages: int,
    dcp_size: int,
    interleave_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Target `(page, slot)` of every token in this rank's DCP cache shard.

    Ownership is chunk-interleaved: a *virtual* page spans `page_size *
    dcp_size` positions, and rank `r` holds the `interleave_size`-sized chunk at
    offset `r * interleave_size` within each `dcp_size * interleave_size` cycle.
    `block_tables` holds virtual page ordinals, resolved against this rank's
    shard by `virtual % num_pages`, the same resolution the DCP readers use.
    Tokens owned by another rank map to `OOB_PAGE`, so no collective is needed.

    Because `interleave_size` divides `page_size`, the positions a rank owns in
    one virtual page map to consecutive slots `0, 1, ..., page_size - 1` of one
    local page, so a sequence's owned tokens still form one contiguous run of
    rows per page once the skipped tokens are dropped.

    Args:
      num_tokens: Number of token rows this step (replicated on every rank).
      seq_lens: `[num_seqs]` global KV lengths including this step's tokens.
      block_tables: `[num_seqs, pages_per_seq]` (or flattened) virtual page ids.
      query_start_loc: `[num_seqs + 1]` cumulative new-token counts.
      dcp_rank: This shard's index along the DCP axis (int or traced scalar).
      page_size: Tokens per local page.
      num_pages: Physical pages in this rank's shard.
      dcp_size: Number of DCP ranks.
      interleave_size: Consecutive positions per rank per round-robin step.

    Returns:
      `(page, slot)`, each `[num_tokens]` int32; `page == OOB_PAGE` for tokens
      this rank must not write.
    """
    if dcp_size < 1 or interleave_size < 1:
        raise ValueError(
            f"dcp_size={dcp_size} and interleave_size={interleave_size} must be >= 1."
        )
    if page_size % interleave_size:
        raise ValueError(
            f"interleave_size={interleave_size} must divide page_size={page_size};"
            " otherwise an interleave chunk straddles a page boundary and the"
            " local slot a reader derives from the global position disagrees with"
            " the one written here."
        )
    seq_id, pos, in_batch = _row_positions(num_tokens, seq_lens, query_start_loc)
    safe_pos = jnp.maximum(pos, 0)
    virtual_page_size = page_size * dcp_size
    cycle = dcp_size * interleave_size
    offset = safe_pos % virtual_page_size
    owner = (offset % cycle) // interleave_size
    slot = (offset // cycle) * interleave_size + offset % interleave_size
    raw_page, ordinal_valid = _lookup_page(
        block_tables, seq_lens.shape[0], seq_id, safe_pos // virtual_page_size
    )
    valid = (
        in_batch
        & (pos >= 0)
        & (owner == jnp.asarray(dcp_rank, dtype=jnp.int32))
        & ordinal_valid
        & (raw_page >= 0)
        & (raw_page < num_pages * dcp_size)
    )
    page = jnp.where(
        valid, jnp.mod(raw_page, jnp.int32(num_pages)), jnp.int32(OOB_PAGE)
    )
    return page, jnp.where(valid, slot, 0)


def get_dst_rows(page: jax.Array, slot: jax.Array, page_size: int) -> jax.Array:
    """Flattens `(page, slot)` to cache rows; `OOB_PAGE` becomes `SKIP_ROW`."""
    return jnp.where(page < OOB_PAGE, page * page_size + slot, SKIP_ROW).astype(
        jnp.int32
    )
