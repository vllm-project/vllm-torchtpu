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
"""PCP manager KV workspace helpers; run per DCP rank inside a `shard_map`.

DCP rank d = p * tp + t, and a PCP rank's TP lanes hold identical rows, so
an all-gather over the DCP axis keeping every tp-th block is the PCP
all-gather (checked by `pcp_kv_gather.verify_dcp_rank_order`).

The latent workspace stays rank-major and top-k positions are remapped onto
it (`remap_topk_chunked`); the indexer K workspace is reordered to position
order (`position_ordered`) for the causal top-k kernel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def pcp_gather_rows(
    x: jax.Array, restore_idx: jax.Array, *, axis_name: str, pcp_size: int
) -> jax.Array:
    """`[local, ...]` local rows -> the step's split rows, request-major.

    `restore_idx[g]` is the PCP-gathered (rank-major) row of split token `g`;
    the result has `pcp_size * local` rows, filler past the real tokens.
    """
    num_local = x.shape[0]
    as_bytes = x.dtype.itemsize == 1 and x.dtype != jnp.uint8
    rows = lax.bitcast_convert_type(x, jnp.uint8) if as_bytes else x
    # [dcp, local, ...] by DCP rank d = p * tp + t; TP lane 0 of each PCP rank.
    gathered = lax.all_gather(rows, axis_name, tiled=False)
    tp_size = gathered.shape[0] // pcp_size
    gathered = gathered.reshape(pcp_size, tp_size, *rows.shape)[:, 0]
    gathered = gathered.reshape(pcp_size * num_local, *rows.shape[1:])
    out = gathered[restore_idx[: pcp_size * num_local]]
    return lax.bitcast_convert_type(out, x.dtype) if as_bytes else out


def workspace_pages(
    block_tables: jax.Array,
    num_seqs: int,
    ws_req: jax.Array,
    ws_vpage: jax.Array,
    num_cache_pages: int,
) -> jax.Array:
    """This rank's physical page for each workspace page."""
    table = block_tables.reshape(num_seqs, -1)
    return jnp.mod(table[ws_req, ws_vpage], jnp.int32(num_cache_pages))


def gather_workspace(
    cache: jax.Array, pages: jax.Array, *, axis_name: str
) -> jax.Array:
    """`[dcp, num_ws_pages, *page]`: every DCP rank's shard of the pages."""
    return lax.all_gather(cache[pages], axis_name, tiled=False)


def position_ordered(workspace: jax.Array, interleave: int) -> jax.Array:
    """`[dcp, Pw, P/4, 4, W]` rank-major -> `[Pw, dcp * P/4, 4, W]` in
    position order (a transpose at `interleave` granularity), one page per
    virtual page. Virtual-page-sized pages keep the top-k kernel's page table
    and step metadata, both prefetched into SMEM, `dcp` times smaller than
    cache-sized pages would; `index_pages_per_block` keeps its VMEM block in
    bounds."""
    dcp, num_ws_pages, rows, packing, width = workspace.shape
    page_size = rows * packing
    x = workspace.reshape(dcp, num_ws_pages, page_size // interleave, interleave, width)
    x = jnp.transpose(x, (1, 2, 0, 3, 4))
    return x.reshape(num_ws_pages, dcp * rows, packing, width)


# Top-k kernel KV block over the position-ordered workspace, in tokens. 4K-8K
# fits both of its limits at max_model_len 1M, 16 seqs, chunk_tokens 256: a
# smaller block makes the step metadata overflow SMEM at 4096 query tokens; a
# 16K block overflows VMEM.
INDEX_BLOCK_TOKENS = 4096


def index_pages_per_block(virtual_page: int) -> int:
    """Top-k `num_kv_pages_per_block` over virtual pages of `virtual_page`
    tokens: `INDEX_BLOCK_TOKENS` per block, at least one page."""
    return max(1, INDEX_BLOCK_TOKENS // virtual_page)


def segment_of_rows(num_rows: int, query_start_loc: jax.Array) -> jax.Array:
    """Local segment of each local row (clamped to the last segment)."""
    rows = jnp.arange(num_rows, dtype=jnp.int32)
    seg = jnp.sum(rows[:, None] >= query_start_loc[None, 1:], axis=1)
    return jnp.minimum(seg, query_start_loc.shape[0] - 2).astype(jnp.int32)


def workspace_index_page_table(
    ws_page_offset: jax.Array,
    num_ws_pages: int,
    max_pages_per_seq: int | None = None,
) -> jax.Array:
    """Flat `[max_num_seqs * W]` page table into the position-ordered indexer
    workspace (virtual pages), one row per local segment, where
    `W = min(Pw, max_pages_per_seq)`. A request's pages are contiguous from
    its offset, so `max_pages_per_seq` covers any request up to that many
    pages."""
    width = (
        num_ws_pages
        if max_pages_per_seq is None
        else min(num_ws_pages, int(max_pages_per_seq))
    )
    pages = jnp.arange(width, dtype=jnp.int32)
    table = jnp.minimum(ws_page_offset[:, None] + pages[None, :], num_ws_pages - 1)
    return table.reshape(-1)


# Chunked latent workspace: filled `chunk_pages` pages per all-gather, only for
# the chunks in use. Page `(c * dcp + r) * chunk_pages + j` holds rank r's copy
# of workspace page `c * chunk_pages + j`.


def topk_workspace_coords(
    topk: jax.Array,
    query_start_loc: jax.Array,
    ws_page_offset: jax.Array,
    *,
    page_size: int,
    dcp_size: int,
    interleave: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per top-k entry: (valid, owner rank, workspace page, slot in page)."""
    seg = segment_of_rows(topk.shape[0], query_start_loc)
    offset = ws_page_offset[seg]
    valid = topk >= 0
    p = jnp.maximum(topk, 0)
    cycle = dcp_size * interleave
    owner = (p // interleave) % dcp_size
    lpos = (p // cycle) * interleave + p % interleave
    ws_page = offset[:, None] + lpos // page_size
    return valid, owner, ws_page, lpos % page_size


def remap_topk_chunked(
    coords: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    *,
    chunk_pages: int,
    page_size: int,
    dcp_size: int,
) -> jax.Array:
    """Global top-k entries -> flat rows of the chunked workspace. `-1` stays
    `-1`; a row with no entry gets row 0 (the kernel needs `kv_len >= 1`)."""
    valid, owner, ws_page, slot = coords
    chunk = ws_page // chunk_pages
    page = (chunk * dcp_size + owner) * chunk_pages + (ws_page - chunk * chunk_pages)
    idx = jnp.where(valid, page * page_size + slot, -1).astype(jnp.int32)
    empty = ~jnp.any(valid, axis=1)
    col = jnp.arange(idx.shape[1], dtype=jnp.int32)[None, :]
    return jnp.where((col == 0) & empty[:, None], 0, idx)
