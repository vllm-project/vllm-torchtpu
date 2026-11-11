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
"""PCP manager KV workspace: the JAX helpers that gather the split prefills'
KV into a per-step workspace (run per DCP rank inside `shard_map`), and the
split-prefill indexer top-k built on them."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    cp_rank_as_data,
    streamindex_topk,
)
from vllm_torchtpu.kernels.mla.kv_cache_utils import OOB_PAGE, _row_positions

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


def pcp_chunk_pages(chunk_tokens: int, page_size: int, dcp_size: int) -> int:
    """Workspace pages (DCP virtual pages) per `chunk_tokens`-token gather chunk."""
    return max(1, int(chunk_tokens) // (int(page_size) * int(dcp_size)))


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


# PCP manager: indexer K workspace buckets (context tokens); the smallest that
# fits is used, and TPU_PCP_KV_WORKSPACE_TOKENS is always the largest.
PCP_INDEXER_BUCKET_TOKENS = (131072, 262144, 524288)


def scatter_indexer_k_rows(
    cache_kv: jax.Array,
    k_packed: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    rank: jax.Array,
    *,
    cp_size: int,
    interleave_size: int,
) -> jax.Array:
    """Write the indexer K rows this DCP rank owns (inside a DCP `shard_map`)."""
    _, block_size_per_kv_packing, kv_packing, lkv_dim = cache_kv.shape
    page_size = block_size_per_kv_packing * kv_packing
    # The latent cache is guarded on its own page size in
    # `update_sparse_mla_kv_cache_dcp`; this cache is a separate
    # allocation, so it needs the check on its own number.
    if page_size % interleave_size:
        raise ValueError(
            f"interleave_size={interleave_size} must divide the indexer "
            f"K cache page_size={page_size}. Otherwise an interleave "
            "chunk straddles a page boundary and the local index "
            "`streamindex_topk_dcp` derives from a global position stops "
            "agreeing with the (page, slot) written here."
        )
    k_packed = jnp.pad(k_packed, ((0, 0), (0, lkv_dim - k_packed.shape[-1])))

    # Same notion of "where does this row belong" as the latent-cache
    # writers, so the two cannot drift apart.
    _, seq_id, pos, in_batch = _row_positions(k_packed.shape[0], seq_lens, cu_q_lens)

    virtual_page_size = jnp.int32(page_size * cp_size)
    cycle = jnp.int32(cp_size * interleave_size)
    interleave = jnp.int32(interleave_size)
    offset = pos % virtual_page_size
    owner = (offset % cycle) // interleave
    slot = (offset // cycle) * interleave + (offset % interleave)

    # Virtual ordinal -> physical block in this rank's own shard, exactly
    # as `streamindex_topk_dcp` resolves it on the read side.
    local_page_indices = jnp.mod(page_indices, jnp.int32(cache_kv.shape[0])).reshape(
        seq_lens.shape[0], -1
    )
    # A padded token's `pos` is meaningless and can run off the end of the
    # page table; clip so the gather stays in bounds and let `valid` drop
    # the row, the same way the sparse MLA writer does.
    max_pages_per_seq = local_page_indices.shape[1]
    virtual_page = jnp.clip(pos // virtual_page_size, 0, max_pages_per_seq - 1)

    valid = jnp.logical_and(in_batch, owner == rank)
    page = jnp.where(
        valid, local_page_indices[seq_id, virtual_page], jnp.int32(OOB_PAGE)
    )
    return cache_kv.at[page, slot // kv_packing, slot % kv_packing].set(k_packed)


def pcp_prefill_topk_partition_specs():
    """`(in_specs, out_specs)` of `pcp_prefill_topk` over the DCP mesh."""
    sharded = jax.sharding.PartitionSpec(DCP_AXIS_NAME)
    replicated = jax.sharding.PartitionSpec()
    in_specs = (
        sharded,  # cache_kv
        sharded,  # q_bytes: local rows
        sharded,  # weights: local rows
        sharded,  # k_packed: local rows
        replicated,  # restore_idx
        replicated,  # split_seq_lens
        replicated,  # split_page_indices
        replicated,  # split_cu_q_lens
        replicated,  # ws_req
        replicated,  # ws_vpage
        sharded,  # ws_page_offset: per local segment
        sharded,  # seq_lens: local
        sharded,  # cu_q_lens: local
        sharded,  # distribution: local
        replicated,  # num_split_reqs
        replicated,  # ws_used_pages
    )
    return in_specs, (sharded, sharded)


def pcp_prefill_topk(
    cache_kv: jax.Array,
    q_bytes: jax.Array,
    weights: jax.Array,
    k_packed: jax.Array,
    restore_idx: jax.Array,
    split_seq_lens: jax.Array,
    split_page_indices: jax.Array,
    split_cu_q_lens: jax.Array,
    ws_req: jax.Array,
    ws_vpage: jax.Array,
    ws_page_offset: jax.Array,
    seq_lens: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    num_split_reqs: jax.Array,
    ws_used_pages: jax.Array,
    *,
    mesh: Any,
    topk: int,
    dcp_size: int,
    pcp_size: int,
    interleave_size: int,
    num_queries_per_block: tuple[int, int, int],
    enable_early_exit: bool,
    max_model_len: int | None = None,
    chunk_tokens: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """PCP manager split-prefill top-k: write the split rows' K, gather the K
    workspace, top-k the local queries over it (global positions).

    Each segment's page table row spans at most `max_model_len` tokens of the
    workspace, not all of it: the kernel's score buffer is
    `[num_tokens, row width]`, so a full-workspace row costs
    `num_tokens * workspace_tokens * 4` bytes of HBM.

    Returns the updated cache and `[local_tokens, topk]` global positions.
    """

    def _local(
        cache_kv,
        q_bytes,
        weights,
        k_packed,
        restore_idx,
        split_seq_lens,
        split_page_indices,
        split_cu_q_lens,
        ws_req,
        ws_vpage,
        ws_page_offset,
        seq_lens,
        cu_q_lens,
        distribution,
        num_split_reqs,
        ws_used_pages,
    ):
        num_tokens = q_bytes.shape[0]
        num_ws_pages = ws_req.shape[0]
        _, rows_per_page, kv_packing, _ = cache_kv.shape
        page_size = rows_per_page * kv_packing
        virtual_page = page_size * dcp_size
        max_seq_pages = (
            None if max_model_len is None else -(-int(max_model_len) // virtual_page)
        )
        pages_per_block = index_pages_per_block(virtual_page)
        # Workspace page buckets; the smallest that fits is picked at run time.
        buckets = sorted(
            {
                min(num_ws_pages, max(1, t // virtual_page))
                for t in PCP_INDEXER_BUCKET_TOKENS
            }
            | {num_ws_pages}
        )

        def run(cache_kv):
            rank = cp_rank_as_data(DCP_AXIS_NAME, dcp_size)
            k_split = pcp_gather_rows(
                k_packed,
                restore_idx,
                dcp_axis_name=DCP_AXIS_NAME,
                pcp_size=pcp_size,
            )
            cache_kv = scatter_indexer_k_rows(
                cache_kv,
                k_split,
                split_seq_lens,
                split_page_indices,
                split_cu_q_lens,
                rank,
                cp_size=dcp_size,
                interleave_size=interleave_size,
            )
            pages = workspace_pages(
                split_page_indices,
                split_seq_lens.shape[0],
                ws_req,
                ws_vpage,
                cache_kv.shape[0],
            )
            q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)

            def at_size(size):
                def branch(cache_kv):
                    ws_kv = position_ordered(
                        gather_workspace(
                            cache_kv, pages[:size], axis_name=DCP_AXIS_NAME
                        ),
                        interleave_size,
                    )
                    return streamindex_topk(
                        q,
                        weights,
                        ws_kv,
                        seq_lens,
                        workspace_index_page_table(ws_page_offset, size, max_seq_pages),
                        cu_q_lens,
                        distribution,
                        k=topk,
                        compression_ratio=1,
                        num_kv_pages_per_block=(pages_per_block,) * 3,
                        num_queries_per_block=num_queries_per_block,
                        enable_early_exit=enable_early_exit,
                        chunk_tokens=chunk_tokens,
                    ).astype(jnp.int32)

                return branch

            used = ws_used_pages[0]
            which = sum((used > b).astype(jnp.int32) for b in buckets[:-1])
            topk_indices = jax.lax.switch(
                which, [at_size(b) for b in buckets], cache_kv
            )
            return cache_kv, topk_indices.astype(jnp.int32)

        def skip(cache_kv):
            return cache_kv, jnp.full((num_tokens, topk), -1, jnp.int32)

        return jax.lax.cond(num_split_reqs[0] > 0, run, skip, cache_kv)

    in_specs, out_specs = pcp_prefill_topk_partition_specs()
    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(
        cache_kv,
        q_bytes,
        weights,
        k_packed,
        restore_idx,
        split_seq_lens,
        split_page_indices,
        split_cu_q_lens,
        ws_req,
        ws_vpage,
        ws_page_offset,
        seq_lens,
        cu_q_lens,
        distribution,
        num_split_reqs,
        ws_used_pages,
    )
