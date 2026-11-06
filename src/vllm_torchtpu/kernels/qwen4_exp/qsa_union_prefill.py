# SPDX-License-Identifier: Apache-2.0
"""QSA prefill over each query group's union of selected groups (D2 prototype, P-ATTN).

Not used by any served path. The block-sparse kernel computes every KV tile any
of a 256-query block selected, which at served contexts is nearly every causal
tile (about 45x the selected pairs). Here each group of ``group`` consecutive
query rows attends a virtual cache made of only the union of its rows' selected
compressed groups (4 tokens each), gathered contiguously, so the work is the
union (2.3-5.6x a row's own selection at 8-16 rows) instead of the causal range.
The attention itself is the existing block-sparse kernel run on that virtual
cache with ``query_block = group``.

Re-indexing (exact for every row): the union is sorted ascending, so a row's
tail group (the group holding its own position, its highest selection) is
followed only by groups with higher ids, which are its future tokens. Giving
the row the virtual length ``rank(tail) * ratio + position % ratio + 1``
(``rank(last selected) * ratio + ratio`` when its length is a multiple of the
ratio) makes the kernel's causal mask hide exactly those, and keeps
``length % ratio``. Each row keeps the same keys and values; only the order of
the online-softmax accumulation changes (not bitwise; equal precision).
DeepSeek-V4's union-dense CSA (kernels/deepseek_v4/core_attention/
csa_union_dense.py on its branch) is the template for a dedicated kernel.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import block_sparse_gqa_attention

_BIG = np.int32(2**30)


def union_sizes(blocks, lengths, *, group, ratio=4):
    """Host numpy: union size (in groups) of each group of rows, tail groups included."""
    blocks, lengths = np.asarray(blocks), np.asarray(lengths)
    out = []
    for g0 in range(0, len(lengths), group):
        rows = slice(g0, g0 + group)
        live = lengths[rows] > 0
        sel = blocks[rows][live]
        tails = (lengths[rows] // ratio)[live & (lengths[rows] % ratio != 0)]
        out.append(len(np.unique(np.concatenate([sel[sel >= 0], tails]))))
    return np.asarray(out, np.int32)


@functools.partial(jax.jit, static_argnames=("group", "union_cap", "ratio", "shift"))
def union_plan(blocks, lengths, *, group, union_cap, ratio=4, shift=0):
    """Union and re-indexing of each group of ``group`` consecutive rows.

    blocks: int32[T, K] selected group ids (-1 padded); lengths: int32[T]
    (0 for padded rows); T % group == 0. Returns ``union`` int32[T // group,
    union_cap] (group ids ascending, -1 padded), ``vblocks`` int32[T, K] (union
    ranks, -1 padded), ``vlengths`` int32[T] and ``usize`` int32[T // group].
    Groups whose union exceeds ``union_cap`` are truncated; check ``usize``.
    ``shift`` (test-only mutant) offsets every rank, so rows attend wrong keys.
    """
    tokens, k = blocks.shape
    ngroups = tokens // group
    live = lengths > 0
    tail = lengths // ratio
    has_tail = live & (lengths % ratio != 0)
    sel = jnp.where((blocks >= 0) & live[:, None], blocks, _BIG)
    sel = jnp.concatenate([sel, jnp.where(has_tail, tail, _BIG)[:, None]], axis=1)
    flat = jnp.sort(sel.reshape(ngroups, group * (k + 1)), axis=1)
    first = jnp.concatenate(
        [jnp.ones((ngroups, 1), bool), flat[:, 1:] != flat[:, :-1]], axis=1
    ) & (flat < _BIG)
    rank = jnp.cumsum(first, axis=1, dtype=jnp.int32) - 1
    usize = jnp.sum(first, axis=1, dtype=jnp.int32)
    width = flat.shape[1]
    slots = jnp.where(first, rank, union_cap + jnp.arange(width, dtype=jnp.int32)[None, :])
    gidx = jnp.arange(ngroups, dtype=jnp.int32)[:, None]
    union = jnp.full((ngroups, union_cap), -1, jnp.int32).at[gidx, slots].set(
        flat, mode="drop", unique_indices=True
    )
    # Rank of each selection inside its group's union (searchsorted on the sorted union).
    padded = jnp.where(union >= 0, union, _BIG)
    row_union = jnp.repeat(padded, group, axis=0)  # [T, union_cap]
    find = jax.vmap(lambda u, x: jnp.searchsorted(u, x, side="left"))
    vrank = find(row_union, jnp.where(blocks >= 0, blocks, 0)).astype(jnp.int32) + shift
    vblocks = jnp.where((blocks >= 0) & live[:, None], vrank, -1)
    tail_rank = find(row_union, tail[:, None])[:, 0].astype(jnp.int32) + shift
    # Rows whose length is a multiple of the ratio have no partial group: their
    # causal frontier is the end of their last complete group, i.e. their tail
    # rank (the first union group above all their selections) times the ratio.
    vlengths = jnp.where(live, tail_rank * ratio + lengths % ratio, 0).astype(jnp.int32)
    return union, vblocks, vlengths, usize


def gather_union_cache(pool, union, page_table_row, *, ratio=4):
    """Contiguous K/V rows of each group's union: [G, union_cap * ratio, *pool.shape[2:]].

    pool: [pages, page_size, ...]; page_table_row: int32[pages_per_seq] (one request);
    padded union entries gather group 0 (never visible: beyond every row's length).
    """
    page_size = pool.shape[1]
    groups = jnp.maximum(union, 0)
    tokens = groups[..., None] * ratio + jnp.arange(ratio, dtype=jnp.int32)
    physical = page_table_row[tokens // page_size] * page_size + tokens % page_size
    rows = pool.reshape(-1, *pool.shape[2:])
    gathered = jnp.take(rows, physical.reshape(-1), axis=0)
    return gathered.reshape(union.shape[0], union.shape[1] * ratio, *pool.shape[2:])


def union_attention(query, gate, blocks, lengths, pool, page_table_row, *, group, union_cap,
                    kv_tile, ratio=4, k_scale=1.0, v_scale=1.0, shift=0):
    """D2 prototype for one request's prefill rows: plan, gather, block-sparse attend.

    Returns (output [T, heads, D] bf16 gated as served, usize). ``union_cap`` must
    be a multiple of 32 and at least every group's union size.
    """
    tokens = query.shape[0]
    union, vblocks, vlengths, usize = union_plan(
        blocks, lengths, group=group, union_cap=union_cap, ratio=ratio, shift=shift
    )
    vcache = gather_union_cache(pool, union, page_table_row, ratio=ratio)
    vtable = jnp.repeat(jnp.arange(union.shape[0], dtype=jnp.int32), group)[:, None]
    out, _ = block_sparse_gqa_attention(
        query, gate, vblocks, vlengths, vtable, vcache,
        jnp.asarray([0, tokens], jnp.int32),
        kv_heads=2 if query.shape[1] == 24 else 1, ratio=ratio, k_scale=k_scale,
        v_scale=v_scale, query_block=group, kv_tile=kv_tile,
        # Paired KV tiles (256-2048 keys) at groups below 128 rows: the opt-in path.
        small_query_block=bool(kv_tile in (256, 512, 1024, 2048) and group % 128),
    )
    return out, usize
