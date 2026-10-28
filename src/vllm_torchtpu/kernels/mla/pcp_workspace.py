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
"""PCP manager KV workspace helpers (run per DCP rank inside `shard_map`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

INDEX_BLOCK_TOKENS = 4096


def pcp_gather_rows(
    x: jax.Array, restore_idx: jax.Array, *, dcp_axis_name: str, pcp_size: int
) -> jax.Array:
    """All-gather local rows over PCP and restore request-major split order.

    This function currently assume DCP = PCP x TP, and only do all gather from 
    PCP group. 
    """
    num_local = x.shape[0]
    as_bytes = x.dtype.itemsize == 1
    rows = lax.bitcast_convert_type(x, jnp.uint8) if as_bytes else x
    tp_size = lax.axis_size(dcp_axis_name) // pcp_size
    groups = [[p * tp_size + t for p in range(pcp_size)] for t in range(tp_size)]
    gathered = lax.all_gather(
        rows,
        dcp_axis_name,
        tiled=True,
        axis_index_groups=groups if tp_size > 1 else None,
    )
    out = gathered[restore_idx[: pcp_size * num_local]]
    return lax.bitcast_convert_type(out, x.dtype) if as_bytes else out


def workspace_pages(
    block_tables: jax.Array,
    num_seqs: int,
    ws_req: jax.Array,
    ws_vpage: jax.Array,
    num_cache_pages: int,
) -> jax.Array:
    """Map workspace pages to this rank's physical cache pages."""
    table = block_tables.reshape(num_seqs, -1)
    return jnp.mod(table[ws_req, ws_vpage], jnp.int32(num_cache_pages))


def gather_workspace(
    cache: jax.Array, pages: jax.Array, *, axis_name: str
) -> jax.Array:
    """All-gather selected cache pages across DCP into `[dcp, num_ws_pages, *page]`."""
    return lax.all_gather(cache[pages], axis_name, tiled=False)


def position_ordered(workspace: jax.Array, interleave: int) -> jax.Array:
    """Reorder the rank-major `[dcp, Pw, P/4, 4, W]` workspace to
    `[Pw, dcp * P/4, 4, W]`."""
    dcp, num_ws_pages, rows, packing, width = workspace.shape
    page_size = rows * packing
    x = workspace.reshape(dcp, num_ws_pages, page_size // interleave, interleave, width)
    x = jnp.transpose(x, (1, 2, 0, 3, 4))
    return x.reshape(num_ws_pages, dcp * rows, packing, width)


def index_pages_per_block(virtual_page: int) -> int:
    """Top-k `num_kv_pages_per_block` for virtual pages of `virtual_page` tokens."""
    return max(1, INDEX_BLOCK_TOKENS // virtual_page)


def segment_of_rows(num_rows: int, query_start_loc: jax.Array) -> jax.Array:
    """Local segment index of each local token row."""
    rows = jnp.arange(num_rows, dtype=jnp.int32)
    seg = jnp.sum(rows[:, None] >= query_start_loc[None, 1:], axis=1)
    return jnp.minimum(seg, query_start_loc.shape[0] - 2).astype(jnp.int32)


def workspace_index_page_table(
    ws_page_offset: jax.Array,
    num_ws_pages: int,
    max_pages_per_seq: int | None = None,
) -> jax.Array:
    """Flat `[num_seqs * width]` page table into the position-ordered workspace."""
    width = (
        num_ws_pages
        if max_pages_per_seq is None
        else min(num_ws_pages, int(max_pages_per_seq))
    )
    pages = jnp.arange(width, dtype=jnp.int32)
    table = jnp.minimum(ws_page_offset[:, None] + pages[None, :], num_ws_pages - 1)
    return table.reshape(-1)


def topk_workspace_coords(
    topk: jax.Array,
    query_start_loc: jax.Array,
    ws_page_offset: jax.Array,
    token_end: jax.Array,
    *,
    page_size: int,
    dcp_size: int,
    interleave: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Map global top-k positions to `(valid, owner_rank, ws_page, slot_in_page)`.

    Rows at or after `token_end` are invalid: they hold the decode rows'
    rank-local cache indices or padding, not global positions, and the
    attention kernel still gathers up to a kernel batch of them.
    """
    seg = segment_of_rows(topk.shape[0], query_start_loc)
    offset = ws_page_offset[seg]
    rows = jnp.arange(topk.shape[0], dtype=jnp.int32)
    valid = (topk >= 0) & (rows[:, None] < token_end)
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
    """Map workspace coordinates to flat token indices in the chunked workspace.

    The shape of chunked workspace is `[chunk, dcp, chunk_pages, page_size]`
    """
    valid, owner, ws_page, slot = coords
    chunk = ws_page // chunk_pages
    page = (chunk * dcp_size + owner) * chunk_pages + (ws_page - chunk * chunk_pages)
    idx = jnp.where(valid, page * page_size + slot, -1).astype(jnp.int32)
    empty = ~jnp.any(valid, axis=1)
    col = jnp.arange(idx.shape[1], dtype=jnp.int32)[None, :]
    return jnp.where((col == 0) & empty[:, None], 0, idx)
