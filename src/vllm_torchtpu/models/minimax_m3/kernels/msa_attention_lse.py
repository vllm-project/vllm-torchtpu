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
"""`msa_attention` that also returns each row's log-sum-exp.

Decode context parallelism (`msa_dcp`) attends each token over the share of
its selected pages one rank holds and merges the ranks' partial outputs by
their log-sum-exps. The kernel is `msa_attention._kernel_v2` with one more
output; the work list, page unpacking and masking are shared with it.
"""

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention, msa_fp4

# A work-list step computes only its valid pages, rounded up to a multiple of
# PAGE_BUCKET, instead of all `pages_per_step` slots: a group's pages are a
# sorted prefix of its slots and the padding slots past it (the last group of
# every tile) are fully masked, yet cost the int4 unpack and the score /
# softmax work of a real page. 0 computes every slot.
PAGE_BUCKET = int(os.environ.get("VLLM_TORCHTPU_MSA_PAGE_BUCKET", "4"))
# Pages per step of the prefill tiles (`_work_tiles`) of DCP attention.
TILE_PAGES_PER_STEP = int(
    os.environ.get(
        "VLLM_TORCHTPU_MSA_TILE_PAGES_PER_STEP",
        str(msa_attention.DEFAULT_PAGES_PER_STEP),
    )
)


# `msa_sparse_attention_lse_heads` above DECODE_TOKENS runs `_heads_call`
# (one loop of kernel calls over whole head-major q / out / lse) instead of a
# `lax.map` over heads and chunks of sliced and restacked operands.
FUSED_HEADS = os.environ.get("VLLM_TORCHTPU_MSA_FUSED_HEADS", "1") == "1"
# Score elementwise work: fold sm_scale into the K scales and broadcast the
# per-token mask over the query heads (`_kernel`).
ELEMENT_DIET = os.environ.get("VLLM_TORCHTPU_MSA_ELEMENT_DIET", "1") == "1"


def _kernel(
    # Scalar prefetch
    work_page_ref,  # [groups * G] page key of each slot, -1 for none
    work_meta_ref,  # [groups] tile << _TILE_SHIFT | last and first bits
    page_table_ref,  # page key -> physical page (read by the index maps only)
    head_ref,  # [1] KV head of the cache rows (read by the index maps only)
    # In
    q_ref,  # [H, Bq, D]
    selected_ref,  # [Bq, K] page keys, -1 padded
    local_page_ref,  # [Bq, 1]
    local_limit_ref,  # [Bq, 1] visible slots of the local page
    *refs,  # G KV pages (4-bit: scales folded in), out, lse, scratch
    sm_scale: float,
    fp4: bool = False,
    slots_per_tile: int | None = None,
    page_bucket: int = 0,
    aliased: int = 0,
):
    """`msa_attention._kernel_v2` that also writes ``lse = m + log(l)``
    (``-inf`` for a row that sees no slot; its output is 0).

    With `slots_per_tile` the grid is ``(tile, group)`` and the work list is
    each tile's padded page union (`_work_tiles`): ``work_page_ref`` holds
    `slots_per_tile` pages per tile and ``work_meta_ref`` each tile's group
    count; steps past a tile's last group do nothing.

    With `page_bucket` a step computes only its first ``n`` slots, ``n`` its
    valid page count rounded up to a multiple of `page_bucket` (a group's
    valid pages are a prefix of its slots); a step with none does nothing.

    `aliased` inputs (in HBM, never read) follow the KV pages: the outputs'
    previous contents, which the call writes into in place."""
    pages_per_step = len(refs) - 5 - aliased
    kv_refs = refs[:pages_per_step]
    o_ref, lse_ref, m_ref, l_ref, acc_ref = refs[pages_per_step + aliased :]
    active = None
    if slots_per_tile is None:
        w = pl.program_id(0)
        meta = work_meta_ref[w]
        base = w * pages_per_step
    else:
        tile, group = pl.program_id(0), pl.program_id(1)
        groups = work_meta_ref[tile]
        active = group < groups
        meta = (
            (tile << msa_attention._TILE_SHIFT)
            | jnp.where(group == 0, msa_attention._FIRST_BIT, 0)
            | jnp.where(group == groups - 1, msa_attention._LAST_BIT, 0)
        )
        base = tile * slots_per_tile + group * pages_per_step
    num_heads, block_q, head_dim = q_ref.shape
    rows_per_page, packing = kv_refs[0].shape[1:3]
    page_size = (
        msa_fp4.data_rows(rows_per_page) * msa_fp4.TOKENS_PER_WORD
        if fp4
        else rows_per_page * packing
    )

    @pl.when((meta & msa_attention._FIRST_BIT) != 0)
    def _init():
        m_ref[...] = jnp.full_like(m_ref, msa_attention._MASK_VALUE)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)

    def _step(num_pages):
        row_slots = (
            jax.lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)  # int4: slot order
            if fp4
            else msa_attention._bitcast_slots(rows_per_page, packing)
        )  # [S, 1]
        slots = row_slots.T  # [1, S]
        selected = selected_ref[...]
        local_page = local_page_ref[...]
        local_limit = local_limit_ref[...]
        keys, values, masks, k_scales, v_scales = [], [], [], [], []
        for g, kv_ref in enumerate(kv_refs[:num_pages]):
            page = work_page_ref[base + g]
            is_selected = jnp.max(
                (selected == page).astype(jnp.int32), axis=1, keepdims=True
            )
            limit = jnp.where(local_page == page, local_limit, page_size)
            limit = jnp.where(page >= 0, limit * is_selected, 0)  # [Bq, 1]
            seen = jnp.max(limit)
            if fp4:  # the head's block carries its own K / V scales
                rows, sk, sv = msa_attention._fp4_page(kv_ref)
                values.append(rows[:, head_dim:])
                k_scales.append(sk)
                v_scales.append(sv)
            elif packing == 4:
                rows = msa_attention._page_rows(kv_ref, None)
                values.append(jnp.where(row_slots < seen, rows[:, head_dim:], 0))
            else:
                rows = msa_attention._page_rows(kv_ref, seen)
                values.append(rows[:, head_dim:])
            keys.append(rows[:, :head_dim])
            masks.append(slots < limit)  # [Bq, S]
        k = jnp.concatenate(keys, axis=0)
        v = jnp.concatenate(values, axis=0)
        mask = jnp.concatenate(masks, axis=1)  # [Bq, n * S]

        q = q_ref[...].reshape(num_heads * block_q, head_dim)
        s = jax.lax.dot_general(
            q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )
        if ELEMENT_DIET:
            # One multiply per score (K scales with sm_scale folded in), and
            # the tokens' mask broadcast over the heads rather than copied.
            scale = (
                jnp.concatenate(k_scales, axis=1) * sm_scale if fp4 else sm_scale
            )
            s = jnp.where(
                mask[None], (s * scale).reshape(num_heads, block_q, -1), -jnp.inf
            ).reshape(num_heads * block_q, -1)
        else:
            mask = jnp.concatenate([mask] * num_heads, axis=0)
            if fp4:  # x = int4 * scale: K scales the score columns, V scales p
                s = s * jnp.concatenate(k_scales, axis=1)
            s = jnp.where(mask, s * sm_scale, -jnp.inf)

        m_prev = m_ref[...]
        m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
        alpha = jnp.exp(m_prev - m_new)
        p = jnp.exp(s - m_new)
        l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=1, keepdims=True)
        p_v = p * jnp.concatenate(v_scales, axis=1) if fp4 else p
        acc_ref[...] = alpha * acc_ref[...] + jax.lax.dot_general(
            p_v.astype(v.dtype),
            v,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        m_ref[...] = m_new

    if not page_bucket or page_bucket >= pages_per_step:
        if active is None:
            _step(pages_per_step)
        else:
            pl.when(active)(functools.partial(_step, pages_per_step))
    else:
        valid = sum(
            jnp.where(work_page_ref[base + g] >= 0, 1, 0)
            for g in range(pages_per_step)
        )
        for hi in range(page_bucket, pages_per_step + page_bucket, page_bucket):
            hi = min(hi, pages_per_step)
            run = (valid > hi - page_bucket) & (valid <= hi)
            if active is not None:
                run = run & active
            pl.when(run)(functools.partial(_step, hi))

    @pl.when((meta & msa_attention._LAST_BIT) != 0)
    def _finalize():
        denom = l_ref[...]
        seen_any = denom > 0
        safe = jnp.where(seen_any, denom, 1.0)
        out = jnp.where(seen_any, acc_ref[...] / safe, 0.0)
        o_ref[...] = out.reshape(num_heads, block_q, head_dim).astype(o_ref.dtype)
        lse = jnp.where(seen_any, m_ref[...] + jnp.log(safe), -jnp.inf)
        lse_ref[...] = lse.reshape(num_heads, block_q, 1)


def _attention_call(
    q,
    kv_cache,
    selected,
    local_page,
    local_limit,
    page_table,
    head,
    work=None,
    *,
    sm_scale,
    block_q,
    pages_per_step,
    interpret,
    tiles=False,
):
    """One kernel call over a token count that is a multiple of block_q.

    With `tiles` the grid is ``(tile, group)`` over `_work_tiles`.

    ``head`` (``int32[1]``) is a scalar-prefetch input, not a static, so one
    kernel serves every KV head of the cache rows. `work` is this call's
    precomputed `msa_attention._work_list` (`_work_lists`), else built here."""
    num_tokens, num_heads, head_dim = q.shape
    _, rows_per_page, packing, _ = kv_cache.shape
    # One KV head's [K | V] slice of each row.
    row_width = 2 * head_dim
    topk = selected.shape[1]
    if work is None:
        work = (_work_tiles if tiles else msa_attention._work_list)(
            selected, block_q, pages_per_step
        )
    pages, meta, num_groups = work
    slots = pages.shape[0] // (num_tokens // block_q) if tiles else None

    if tiles:
        # meta: groups per tile. A step past its tile's groups reads the
        # tile's last group again, so its blocks need no new DMA.
        def tile_of(t, grp, pages, meta, *_):
            return t

        def slot_index(t, grp, meta, g):
            return t * slots + jnp.minimum(grp, meta[t] - 1) * pages_per_step + g

        grid = (num_tokens // block_q, num_groups)
    else:

        def tile_of(w, pages, meta, *_):
            return meta[w] >> msa_attention._TILE_SHIFT

        def slot_index(w, meta, g):
            return w * pages_per_step + g

        grid = (num_groups,)

    def tile_block(*idx):
        return (tile_of(*idx), 0)

    def head_major_block(*idx):
        return (0, tile_of(*idx), 0)

    def page_index(g, idx):
        *grid_idx, pages, meta, table, head_idx = idx
        page = table[jnp.maximum(pages[slot_index(*grid_idx, meta, g)], 0)]
        return page, head_idx

    def page_block(g):
        def index_map(*idx):
            page, head_idx = page_index(g, idx)
            return (page, 0, 0, head_idx[0])

        return pl.BlockSpec((1, rows_per_page, packing, row_width), index_map)

    fp4 = kv_cache.dtype == jnp.uint8

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4,
        grid=grid,
        in_specs=[
            pl.BlockSpec((num_heads, block_q, head_dim), head_major_block),
            pl.BlockSpec((block_q, topk), tile_block),
            pl.BlockSpec((block_q, 1), tile_block),
            pl.BlockSpec((block_q, 1), tile_block),
            *(page_block(g) for g in range(pages_per_step)),
        ],
        out_specs=[
            pl.BlockSpec((num_heads, block_q, head_dim), head_major_block),
            pl.BlockSpec((num_heads, block_q, 1), head_major_block),
        ],
        scratch_shapes=[
            pltpu.VMEM((num_heads * block_q, 1), jnp.float32),
            pltpu.VMEM((num_heads * block_q, 1), jnp.float32),
            pltpu.VMEM((num_heads * block_q, head_dim), jnp.float32),
        ],
    )
    out, lse = pl.pallas_call(
        functools.partial(
            _kernel,
            sm_scale=sm_scale,
            slots_per_tile=slots,
            page_bucket=PAGE_BUCKET,
            **({"fp4": True} if fp4 else {}),
        ),
        grid_spec=grid_spec,
        out_shape=[
            jax.ShapeDtypeStruct((num_heads, num_tokens, head_dim), q.dtype),
            jax.ShapeDtypeStruct((num_heads, num_tokens, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",) * len(grid),
            vmem_limit_bytes=msa_attention._VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="minimax_m3_msa_sparse_attention_lse",
    )(
        pages,
        meta,
        page_table,
        head,
        q.transpose(1, 0, 2),
        selected,
        local_page[:, None],
        local_limit[:, None],
        *([kv_cache] * pages_per_step),
    )
    return out.transpose(1, 0, 2), lse[:, :, 0].T


def _work_tiles(selected: jax.Array, block_q: int, pages_per_step: int):
    """Each tile's sorted page union padded to whole groups, for the
    ``(tile, group)`` grid: ``(pages [tiles * slots], groups [tiles],
    max_groups)``. Unlike `msa_attention._work_list` there is no flattened
    group list to search and gather: a step reads its pages straight from
    its tile's union."""
    num_tokens, topk = selected.shape
    num_tiles = num_tokens // block_q
    union_size = block_q * topk
    slots = -(-union_size // pages_per_step) * pages_per_step
    per_tile = jnp.where(selected < 0, msa_attention._NO_PAGE, selected).reshape(
        num_tiles, union_size
    )
    union = msa_attention.tile_unions(per_tile)
    union = jnp.where(union == msa_attention._NO_PAGE, -1, union)
    union = jnp.pad(union, ((0, 0), (0, slots - union_size)), constant_values=-1)
    pages_in_tile = (union >= 0).sum(axis=1)
    groups = jnp.maximum(-(-pages_in_tile // pages_per_step), 1).astype(jnp.int32)
    return union.reshape(-1), groups, groups.max()


def _chunk_len(num_tokens: int, block_q: int) -> tuple[int, int]:
    """(tokens per kernel call, number of calls) for `_chunked_call`; a
    single call takes the tokens as they are, unpadded."""
    limit = msa_attention.MAX_TOKENS_PER_CALL
    step = max(block_q, limit - limit % block_q)
    num_chunks = -(-num_tokens // step)
    return (num_tokens if num_chunks == 1 else step), num_chunks


def _work_lists(selected, block_q, pages_per_step, tiles=False):
    """The work lists of every kernel call of ``selected [..., T, K]`` (any
    leading batch, e.g. KV heads), built in one batched computation rather
    than one per call inside the calls' `lax.map`: a work list is a per-tile
    sort and searches that cost more than the kernel itself when built per
    call (0.44 vs 0.31 ms per 2048-token call at DCP8). Returns arrays with
    leading ``[..., num_calls]``."""
    *batch, num_tokens, topk = selected.shape
    step, num_chunks = _chunk_len(num_tokens, block_q)
    pad = num_chunks * step - num_tokens
    selected = jnp.pad(
        selected, [(0, 0)] * len(batch) + [(0, pad), (0, 0)], constant_values=-1
    ).reshape(-1, step, topk)
    builder = _work_tiles if tiles else msa_attention._work_list
    build = jax.vmap(lambda s: builder(s, block_q, pages_per_step))
    return jax.tree.map(
        lambda x: x.reshape(*batch, num_chunks, *x.shape[1:]), build(selected)
    )


# Calls of at most this many tokens (decode, verify windows) run
# `_decode_call`: one grid step per token over its own selected pages, with no
# work list. It beats the tiled kernel with its work list up to 64 tokens,
# G = 1 and 4, fp8 and 4-bit (j-e34cee54: 15.3 -> 13.2 us at 64, 12.2 -> 11.0
# at 16); above that the tiles' page unions pay off.
DECODE_TOKENS = 64


def _decode_kernel(
    # Scalar prefetch
    keys_ref,  # [(T + 1) * K] page keys, -1 for none
    local_page_ref,  # [T] page key of the token's own (last) page
    local_limit_ref,  # [T] visible slots of that page
    page_table_ref,  # page key -> physical page (read by the index maps only)
    head_ref,  # [1] KV head of the cache rows (read by the index maps only)
    # In
    q_ref,  # [1, H, D]
    *refs,  # K KV pages (4-bit: scales folded in), out, lse
    sm_scale: float,
    fp4: bool = False,
):
    """One token against its `K` selected pages in one step: no running
    softmax and no page union. A page key below 0 contributes no slot."""
    topk = len(refs) - 2
    kv_refs = refs[:topk]
    o_ref, lse_ref = refs[topk:]
    t = pl.program_id(0)
    _, num_heads, head_dim = q_ref.shape
    rows_per_page, packing = kv_refs[0].shape[1:3]
    page_size = (
        msa_fp4.data_rows(rows_per_page) * msa_fp4.TOKENS_PER_WORD
        if fp4
        else rows_per_page * packing
    )
    row_slots = (
        jax.lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)  # int4: slot order
        if fp4
        else msa_attention._bitcast_slots(rows_per_page, packing)
    )  # [S, 1]
    slots = row_slots.T  # [1, S]
    local_page = local_page_ref[t]
    keys, values, masks, k_scales, v_scales = [], [], [], [], []
    for g, kv_ref in enumerate(kv_refs):
        page = keys_ref[t * topk + g]
        limit = jnp.where(page == local_page, local_limit_ref[t], page_size)
        limit = jnp.where(page >= 0, limit, 0)
        if fp4:  # the head's block carries its own K / V scales
            rows, sk, sv = msa_attention._fp4_page(kv_ref)
            values.append(rows[:, head_dim:])
            k_scales.append(sk)
            v_scales.append(sv)
        elif packing == 4:
            rows = msa_attention._page_rows(kv_ref, None)
            values.append(jnp.where(row_slots < limit, rows[:, head_dim:], 0))
        else:
            rows = msa_attention._page_rows(kv_ref, limit)
            values.append(rows[:, head_dim:])
        keys.append(rows[:, :head_dim])
        masks.append(slots < limit)  # [1, S]
    k = jnp.concatenate(keys, axis=0)
    v = jnp.concatenate(values, axis=0)
    mask = jnp.concatenate(masks, axis=1)  # [1, K * S]

    q = q_ref[0]  # [H, D]
    s = jax.lax.dot_general(
        q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )
    if fp4:  # x = int4 * scale: K scales the score columns, V scales p
        s = s * jnp.concatenate(k_scales, axis=1)
    s = jnp.where(mask, s * sm_scale, -jnp.inf)
    m = jnp.maximum(jnp.max(s, axis=1, keepdims=True), msa_attention._MASK_VALUE)
    p = jnp.exp(s - m)
    denom = jnp.sum(p, axis=1, keepdims=True)
    p_v = p * jnp.concatenate(v_scales, axis=1) if fp4 else p
    acc = jax.lax.dot_general(
        p_v.astype(v.dtype), v, (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    seen_any = denom > 0
    safe = jnp.where(seen_any, denom, 1.0)
    o_ref[0] = jnp.where(seen_any, acc / safe, 0.0).astype(o_ref.dtype)
    lse_ref[0] = jnp.where(seen_any, m + jnp.log(safe), -jnp.inf)


def _decode_call(
    q,
    kv_cache,
    selected,
    local_page,
    local_limit,
    page_table,
    head,
    *,
    sm_scale,
    interpret,
):
    """`_attention_call` for a few tokens: grid step `t` is token `t`, whose
    `K` page blocks come straight from its selected keys. A missing key maps
    to page 0's block, which the pipeline does not fetch again while it stays
    the block of that slot."""
    num_tokens, num_heads, head_dim = q.shape
    _, rows_per_page, packing, _ = kv_cache.shape
    row_width = 2 * head_dim
    topk = selected.shape[1]
    fp4 = kv_cache.dtype == jnp.uint8

    def token_block(t, *_):
        return (t, 0, 0)

    def page_block(g):
        def index_map(t, keys, local_page, local_limit, table, head_idx):
            page = table[jnp.maximum(keys[t * topk + g], 0)]
            return (page, 0, 0, head_idx[0])

        return pl.BlockSpec((1, rows_per_page, packing, row_width), index_map)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=5,
        grid=(num_tokens,),
        in_specs=[
            pl.BlockSpec((1, num_heads, head_dim), token_block),
            *(page_block(g) for g in range(topk)),
        ],
        out_specs=[
            pl.BlockSpec((1, num_heads, head_dim), token_block),
            pl.BlockSpec((1, num_heads, 1), token_block),
        ],
    )
    out, lse = pl.pallas_call(
        functools.partial(
            _decode_kernel,
            sm_scale=sm_scale,
            **({"fp4": True} if fp4 else {}),
        ),
        grid_spec=grid_spec,
        out_shape=[
            jax.ShapeDtypeStruct((num_tokens, num_heads, head_dim), q.dtype),
            jax.ShapeDtypeStruct((num_tokens, num_heads, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=msa_attention._VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="minimax_m3_msa_decode_attention_lse",
    )(
        # One token of -1 keys past the end: the pipeline evaluates the page
        # index maps of the step after the last one, and a key read past the
        # array is any bits (an out-of-range page DMA halts the core).
        jnp.pad(selected.reshape(-1), (0, topk), constant_values=-1),
        local_page,
        local_limit,
        page_table,
        head,
        q,
        *([kv_cache] * topk),
    )
    return out, lse[:, :, 0]


def _heads_call(
    q,
    kv_cache,
    selected,
    local_page,
    local_limit,
    page_table,
    *,
    sm_scale,
    block_q,
    pages_per_step,
    interpret,
    tiles,
):
    """Every KV head and token chunk of `msa_sparse_attention_lse_heads` as
    kernel calls in one loop that read their queries from, and write their
    outputs into, whole head-major arrays at the head's and chunk's block:
    no per-call slicing, transposing and stacking of q / out / lse.

    ``q [G, Hg, T, D]`` (T a multiple of block_q), ``selected [G, T, K]``.
    Returns ``(out [G, Hg, T, D], lse [G, Hg, T, 1] f32)``."""
    num_heads, q_per_head, num_tokens, head_dim = q.shape
    _, rows_per_page, packing, _ = kv_cache.shape
    row_width = 2 * head_dim
    topk = selected.shape[-1]
    step, num_chunks = _chunk_len(num_tokens, block_q)
    pad = num_chunks * step - num_tokens
    if pad:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, pad), (0, 0)))
        selected = jnp.pad(selected, ((0, 0), (0, pad), (0, 0)), constant_values=-1)
        local_page = jnp.pad(local_page, (0, pad), constant_values=-1)
        local_limit = jnp.pad(local_limit, (0, pad))
    padded = num_chunks * step
    tiles_per_chunk = step // block_q
    works = _work_lists(selected, block_q, pages_per_step, tiles)  # [G, chunks, ...]
    fp4 = kv_cache.dtype == jnp.uint8

    if tiles:

        def grid_tile(t, grp, pages, meta, *_):
            return t

        def slot_index(t, grp, meta, g, slots):
            return t * slots + jnp.minimum(grp, meta[t] - 1) * pages_per_step + g

    else:

        def grid_tile(w, pages, meta, *_):
            return meta[w] >> msa_attention._TILE_SHIFT

        def slot_index(w, meta, g, slots):
            return w * pages_per_step + g

    def tile_of(*idx):
        return idx[-1][1] + grid_tile(*idx)  # chunk's first tile + tile

    def call(h, c, out, lse):
        pages, meta, num_groups = jax.tree.map(lambda x: x[h, c], works)
        slots = pages.shape[0] // tiles_per_chunk if tiles else None
        grid = (tiles_per_chunk, num_groups) if tiles else (num_groups,)

        def page_block(g):
            def index_map(*idx):
                *grid_idx, pages_, meta_, table, hc = idx
                key = pages_[slot_index(*grid_idx, meta_, g, slots)]
                return (table[jnp.maximum(key, 0)], 0, 0, hc[0])

            return pl.BlockSpec((1, rows_per_page, packing, row_width), index_map)

        def head_block(*idx):
            return (idx[-1][0], 0, tile_of(*idx), 0)

        def sel_block(*idx):
            return (idx[-1][0], tile_of(*idx), 0)

        def row_block(*idx):
            return (tile_of(*idx), 0)

        any_spec = pl.BlockSpec(memory_space=pl.ANY)
        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=grid,
            in_specs=[
                pl.BlockSpec((None, q_per_head, block_q, head_dim), head_block),
                pl.BlockSpec((None, block_q, topk), sel_block),
                pl.BlockSpec((block_q, 1), row_block),
                pl.BlockSpec((block_q, 1), row_block),
                *(page_block(g) for g in range(pages_per_step)),
                any_spec,
                any_spec,
            ],
            out_specs=[
                pl.BlockSpec((None, q_per_head, block_q, head_dim), head_block),
                pl.BlockSpec((None, q_per_head, block_q, 1), head_block),
            ],
            scratch_shapes=[
                pltpu.VMEM((q_per_head * block_q, 1), jnp.float32),
                pltpu.VMEM((q_per_head * block_q, 1), jnp.float32),
                pltpu.VMEM((q_per_head * block_q, head_dim), jnp.float32),
            ],
        )
        first = 8 + pages_per_step  # operand index of `out` (prefetch included)
        return pl.pallas_call(
            functools.partial(
                _kernel,
                sm_scale=sm_scale,
                slots_per_tile=slots,
                page_bucket=PAGE_BUCKET,
                aliased=2,
                **({"fp4": True} if fp4 else {}),
            ),
            grid_spec=grid_spec,
            out_shape=[
                jax.ShapeDtypeStruct(out.shape, out.dtype),
                jax.ShapeDtypeStruct(lse.shape, lse.dtype),
            ],
            input_output_aliases={first: 0, first + 1: 1},
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",) * len(grid),
                vmem_limit_bytes=msa_attention._VMEM_LIMIT_BYTES,
            ),
            interpret=interpret,
            name="minimax_m3_msa_sparse_attention_lse",
        )(
            pages,
            meta,
            page_table,
            jnp.stack([h, c * tiles_per_chunk]).astype(jnp.int32),
            q,
            selected,
            local_page[:, None],
            local_limit[:, None],
            *([kv_cache] * pages_per_step),
            out,
            lse,
        )

    def body(i, carry):
        return tuple(call(i // num_chunks, i % num_chunks, *carry))

    out, lse = jax.lax.fori_loop(
        0,
        num_heads * num_chunks,
        body,
        (
            jnp.zeros((num_heads, q_per_head, padded, head_dim), q.dtype),
            jnp.zeros((num_heads, q_per_head, padded, 1), jnp.float32),
        ),
    )
    return out[:, :, :num_tokens], lse[:, :, :num_tokens]


def _chunked_call(
    q,
    kv_cache,
    selected,
    local_page,
    local_limit,
    page_table,
    head,
    works=None,
    *,
    sm_scale,
    block_q,
    pages_per_step,
    interpret,
    tiles=False,
):
    """`_attention_call` over chunks of at most `MAX_TOKENS_PER_CALL` tokens
    (the work list lives in SMEM). Several chunks run as one `lax.map`, so
    the kernel is emitted once per call site rather than once per chunk."""
    num_tokens = q.shape[0]
    if num_tokens <= DECODE_TOKENS:
        return _decode_call(
            q,
            kv_cache,
            selected,
            local_page,
            local_limit,
            page_table,
            head,
            sm_scale=sm_scale,
            interpret=interpret,
        )
    call = functools.partial(
        _attention_call,
        sm_scale=sm_scale,
        block_q=block_q,
        pages_per_step=pages_per_step,
        interpret=interpret,
        tiles=tiles,
    )
    step, num_chunks = _chunk_len(num_tokens, block_q)
    if works is None:
        works = _work_lists(selected, block_q, pages_per_step, tiles)
    if num_chunks == 1:
        return call(
            q,
            kv_cache,
            selected,
            local_page,
            local_limit,
            page_table,
            head,
            jax.tree.map(lambda x: x[0], works),
        )
    pad = num_chunks * step - num_tokens
    q = jnp.pad(q, ((0, pad), (0, 0), (0, 0)))
    selected = jnp.pad(selected, ((0, pad), (0, 0)), constant_values=-1)
    local_page = jnp.pad(local_page, (0, pad), constant_values=-1)
    local_limit = jnp.pad(local_limit, (0, pad))

    def chunk(xs):
        q_c, sel_c, page_c, limit_c, work_c = xs
        return call(q_c, kv_cache, sel_c, page_c, limit_c, page_table, head, work_c)

    out, lse = jax.lax.map(
        chunk,
        (
            q.reshape(num_chunks, step, *q.shape[1:]),
            selected.reshape(num_chunks, step, -1),
            local_page.reshape(num_chunks, step),
            local_limit.reshape(num_chunks, step),
            works,
        ),
    )
    out = out.reshape(num_chunks * step, *out.shape[2:])[:num_tokens]
    lse = lse.reshape(num_chunks * step, -1)[:num_tokens]
    return out, lse


def _pad_tokens(q, selected, local_page, local_limit, block_q):
    pad = -q.shape[0] % block_q
    if pad:
        q = jnp.pad(q, ((0, pad),) + ((0, 0),) * (q.ndim - 1))
        selected = jnp.pad(
            selected, ((0, pad),) + ((0, 0),) * (selected.ndim - 1), constant_values=-1
        )
        local_page = jnp.pad(local_page, (0, pad), constant_values=-1)
        local_limit = jnp.pad(local_limit, (0, pad))
    return q, selected, local_page, local_limit


def _check_rows(kv_cache, head_dim, head=None):
    num_heads = kv_cache.shape[3] // (2 * head_dim)
    if kv_cache.shape[3] % (2 * head_dim) or (
        head is not None and not 0 <= head < num_heads
    ):
        raise ValueError(
            f"Cache rows hold [K | V] of width {2 * head_dim} per KV head; got "
            f"{kv_cache.shape} for head {head}."
        )
    return num_heads


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "block_q",
        "pages_per_step",
        "head",
        "tiles",
        "interpret",
    ),
)
def msa_sparse_attention_lse(
    q: jax.Array,
    kv_cache: jax.Array,
    selected: jax.Array,
    local_page: jax.Array,
    local_limit: jax.Array,
    page_table: jax.Array,
    *,
    sm_scale: float,
    block_q: int = msa_attention.DEFAULT_BLOCK_Q,
    pages_per_step: int = msa_attention.DEFAULT_PAGES_PER_STEP,
    head: int = 0,
    tiles: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`msa_attention.msa_sparse_attention` (kernel v2, page keys translated
    by `page_table`) returning ``(out [T, H, D], lse [T, H] f32)``.

    A token with no selected page (or a padding token) gets a zero output
    and ``lse = -inf``, so it carries no weight in a log-sum-exp merge.
    Cache rows may hold several KV heads, ``[K_0 | V_0 | K_1 | V_1 ...]``;
    `head` picks the one these queries attend (a 2D-wide block of each page;
    in a 4-bit page the block's last word row holds the head's K / V scales).
    """
    num_tokens, _, head_dim = q.shape
    msa_attention.check_cache(kv_cache)
    msa_attention.check_fp4_rows(kv_cache, head_dim)
    _check_rows(kv_cache, head_dim, head)
    if num_tokens > DECODE_TOKENS:  # decode calls take the tokens as they are
        q, selected, local_page, local_limit = _pad_tokens(
            q, selected, local_page, local_limit, block_q
        )
    out, lse = _chunked_call(
        q,
        kv_cache,
        selected,
        local_page,
        local_limit,
        page_table,
        jnp.full((1,), head, jnp.int32),
        sm_scale=sm_scale,
        block_q=block_q,
        pages_per_step=pages_per_step,
        interpret=interpret,
        tiles=tiles,
    )
    return out[:num_tokens], lse[:num_tokens]


@functools.partial(
    jax.jit,
    static_argnames=("sm_scale", "block_q", "pages_per_step", "tiles", "interpret"),
)
def msa_sparse_attention_lse_heads(
    q: jax.Array,
    kv_cache: jax.Array,
    selected: jax.Array,
    local_page: jax.Array,
    local_limit: jax.Array,
    page_table: jax.Array,
    *,
    sm_scale: float,
    block_q: int = msa_attention.DEFAULT_BLOCK_Q,
    pages_per_step: int = msa_attention.DEFAULT_PAGES_PER_STEP,
    tiles: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`msa_sparse_attention_lse` for every KV head of the cache rows at once.

    ``q [T, G * Hg, D]`` (KV head ``h`` owns query heads ``[h * Hg,
    (h + 1) * Hg)``), ``selected [T, G, K]``. The heads run as one `lax.map`
    over a single kernel whose KV head is an input, so the program carries
    one copy of the kernel instead of one per head. Returns ``(out
    [T, G * Hg, D], lse [T, G * Hg] f32)``.

    A 4-bit cache keeps head ``h``'s K / V scales in lanes ``[2hS, (2h+1)S)``
    / ``[(2h+1)S, (2h+2)S)`` of the pages' last word row (`msa_fp4.kv_scales`),
    inside the head's own ``2D``-wide block.
    """
    num_tokens, num_q_heads, head_dim = q.shape
    num_heads = selected.shape[1]
    msa_attention.check_cache(kv_cache)
    msa_attention.check_fp4_rows(kv_cache, head_dim)
    if _check_rows(kv_cache, head_dim) < num_heads or num_q_heads % num_heads:
        raise ValueError(
            f"{num_heads} KV heads of {num_q_heads} query heads do not fit cache "
            f"rows {kv_cache.shape}."
        )
    if num_tokens > DECODE_TOKENS:  # decode calls take the tokens as they are
        q, selected, local_page, local_limit = _pad_tokens(
            q, selected, local_page, local_limit, block_q
        )
    padded = q.shape[0]
    q_per_head = num_q_heads // num_heads
    selected_heads = selected.transpose(1, 0, 2)  # [G, T, K]
    decode = padded <= DECODE_TOKENS
    if not decode and FUSED_HEADS:
        out, lse = _heads_call(
            q.reshape(padded, num_heads, q_per_head, head_dim).transpose(1, 2, 0, 3),
            kv_cache,
            selected_heads,
            local_page,
            local_limit,
            page_table,
            sm_scale=sm_scale,
            block_q=block_q,
            pages_per_step=pages_per_step,
            interpret=interpret,
            tiles=tiles,
        )  # [G, Hg, T, D], [G, Hg, T, 1]
        out = out.transpose(2, 0, 1, 3).reshape(padded, num_q_heads, head_dim)
        lse = lse[..., 0].transpose(2, 0, 1).reshape(padded, num_q_heads)
        return out[:num_tokens], lse[:num_tokens]
    q_heads = q.reshape(padded, num_heads, q_per_head, head_dim).transpose(1, 0, 2, 3)
    works = (
        jnp.zeros((num_heads,), jnp.int32)  # unused: decode calls need no work list
        if decode
        else _work_lists(selected_heads, block_q, pages_per_step, tiles)
    )

    def one_head(xs):
        q_h, sel_h, h, works_h = xs
        if decode:
            works_h = None
        return _chunked_call(
            q_h,
            kv_cache,
            sel_h,
            local_page,
            local_limit,
            page_table,
            h.reshape(1),
            works_h,
            sm_scale=sm_scale,
            block_q=block_q,
            pages_per_step=pages_per_step,
            interpret=interpret,
            tiles=tiles,
        )

    out, lse = jax.lax.map(
        one_head,
        (q_heads, selected_heads, jnp.arange(num_heads, dtype=jnp.int32), works),
    )  # [G, T, Hg, D], [G, T, Hg]
    out = out.transpose(1, 0, 2, 3).reshape(padded, num_q_heads, head_dim)
    lse = lse.transpose(1, 0, 2).reshape(padded, num_q_heads)
    return out[:num_tokens], lse[:num_tokens]
