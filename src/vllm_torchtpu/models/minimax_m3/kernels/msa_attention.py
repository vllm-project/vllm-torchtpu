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
"""Pallas sparse attention over each token's selected KV pages.

Tokens are grouped into query tiles of `block_q`. A tile streams the union
of the pages its tokens selected, `pages_per_step` pages per grid step, and
masks per token: a page counts for a token only if the token selected it,
and its local page only up to the token itself. Consecutive tokens of a
prefill chunk select mostly the same pages, so a page is read about once per
tile instead of once per token. The (tile, page group) work list is
flattened ahead of the kernel and run on a dynamic grid, so empty union
slots cost nothing.

Layout: one KV head per rank; each token's cache row is ``[K | V]`` in the
packed MLA page layout (`packed_page`), bf16 or fp8. `num_heads` query heads.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels.packed_page import (
    check_page_dtype,
    slot_ids,
    unpack_page,
)

DEFAULT_BLOCK_Q = 16
# Batches of at most this many tokens (decode, MTP verify) use 8-token tiles:
# an 8-token verify group then fills a tile alone instead of sharing a
# 16-token tile (and its page union) with the next request. Twin, c64 verify
# (32 req x 8 tokens): 416 -> 350 us per call (j-cae78264 vs j-2c36aed7);
# prefill chunks keep 16 (8 is 7% slower there, j-3eb5e8aa).
SMALL_BATCH_TOKENS = 1024
SMALL_BATCH_BLOCK_Q = 8
DEFAULT_PAGES_PER_STEP = 16
DEFAULT_KERNEL = "v2"
# Tokens per kernel call. The work list lives in SMEM, sized for the worst
# case of tokens * topk pages.
MAX_TOKENS_PER_CALL = 2048
_FIRST_BIT = 1
_LAST_BIT = 2
_TILE_SHIFT = 2
# The score tile is (heads * block_q) x (pages_per_step * page_size) f32.
_VMEM_LIMIT_BYTES = 48 * 1024 * 1024
_MASK_VALUE = -1e30
_NO_PAGE = jnp.iinfo(jnp.int32).max


def _kernel_v1(
    # Scalar prefetch
    work_page_ref,  # [groups * G] page of each group slot, -1 for none
    work_meta_ref,  # [groups] tile << _TILE_SHIFT | last and first bits
    # In
    q_ref,  # [H, Bq, D]
    selected_ref,  # [Bq, K] physical pages, -1 padded
    local_page_ref,  # [Bq, 1]
    local_limit_ref,  # [Bq, 1] visible slots of the local page
    *refs,  # G x [1, S / p, p, 2D] KV pages, then out and scratch
    sm_scale: float,
):
    pages_per_step = len(refs) - 4
    kv_refs = refs[:pages_per_step]
    o_ref, m_ref, l_ref, acc_ref = refs[pages_per_step:]
    w = pl.program_id(0)
    meta = work_meta_ref[w]
    num_heads, block_q, head_dim = q_ref.shape
    packing = kv_refs[0].shape[2]
    page_size = kv_refs[0].shape[1] * packing

    @pl.when((meta & _FIRST_BIT) != 0)
    def _init():
        m_ref[...] = jnp.full_like(m_ref, _MASK_VALUE)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)

    keys, values, limits = [], [], []
    selected = selected_ref[...]
    # Slot of each value row once a page's unpacked arrays are concatenated.
    row = jax.lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)
    per_array = page_size // packing
    value_slot = (row % per_array) * packing + row // per_array
    for g, kv_ref in enumerate(kv_refs):
        page = work_page_ref[w * pages_per_step + g]
        rows = unpack_page(kv_ref)
        is_selected = jnp.max(
            (selected == page).astype(jnp.int32), axis=1, keepdims=True
        )
        limit = jnp.where(local_page_ref[...] == page, local_limit_ref[...], page_size)
        limit = jnp.where(page >= 0, limit * is_selected, 0)
        keys.append(jnp.concatenate([r[:, :head_dim] for r in rows], axis=0))
        # Zero the value rows no token of the tile sees: a masked slot gets
        # p = 0, but 0 * NaN is NaN, and a slot nobody wrote (the null page 0
        # that fills a group past the tile's union, or a local page's slots
        # past the newest token) may hold any bytes.
        visible = value_slot < jnp.max(limit)
        values.append(
            jnp.where(
                visible,
                jnp.concatenate([r[:, head_dim:] for r in rows], axis=0),
                0,
            )
        )
        # Rows are head-major: row h * block_q + t is token t's head h.
        limits.append(jnp.concatenate([limit] * num_heads, axis=0))
    k = jnp.concatenate(keys, axis=0)
    v = jnp.concatenate(values, axis=0)

    q = q_ref[...].reshape(num_heads * block_q, head_dim)
    s = jax.lax.dot_general(
        q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )
    s = s * sm_scale
    slots = slot_ids(s.shape[0], page_size, packing)
    mask = jnp.concatenate([slots < limit for limit in limits], axis=1)
    s = jnp.where(mask, s, _MASK_VALUE)

    m_prev = m_ref[...]
    m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.where(mask, jnp.exp(s - m_new), 0.0)
    l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=1, keepdims=True)
    acc_ref[...] = alpha * acc_ref[...] + jax.lax.dot_general(
        p.astype(v.dtype),
        v,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    m_ref[...] = m_new

    @pl.when((meta & _LAST_BIT) != 0)
    def _finalize():
        denom = l_ref[...]
        out = jnp.where(denom > 0, acc_ref[...] / jnp.where(denom > 0, denom, 1.0), 0.0)
        o_ref[...] = out.reshape(num_heads, block_q, head_dim).astype(o_ref.dtype)


def _page_rows(kv_ref, seen: jax.Array | None) -> jax.Array:
    """``[1, S / p, p, 2D]`` page -> bf16 ``[S, 2D]`` rows by one width-changing
    bitcast of the packed 32-bit words (rows come out in the bitcast's token
    order; `_bitcast_slots` gives the slot of each row).

    With `seen`, slots ``>= seen`` are zeroed in the packed words first: no token of the
    tile sees them, but p @ v multiplies them by p = 0, and 0 * NaN is NaN,
    so an unwritten slot holding any bytes would poison the tile.
    """
    rows_per_page, packing, width = kv_ref.shape[1:]
    bits = 32 // packing
    words = kv_ref.bitcast(jnp.uint32)[0, :, 0, :]
    if seen is not None:
        row = jax.lax.broadcasted_iota(jnp.int32, (rows_per_page, 1), 0)
        keep = jnp.zeros((rows_per_page, 1), jnp.uint32)
        for j in range(packing):
            field = jnp.uint32(((1 << bits) - 1) << (bits * j))
            keep = keep | jnp.where(row * packing + j < seen, field, jnp.uint32(0))
        words = words & keep
    return pltpu.bitcast(words, kv_ref.dtype).astype(jnp.bfloat16)


def _bitcast_slots(rows_per_page: int, packing: int) -> jax.Array:
    """Slot of each row of `_page_rows`, as an ``[S, 1]`` column.

    Bitcasts words holding their own slot ids the same way the page is
    bitcast, so the order is whatever the hardware bitcast produces.
    """
    bits = 32 // packing
    uint = {16: jnp.uint16, 8: jnp.uint8}[bits]
    row = jax.lax.broadcasted_iota(jnp.uint32, (rows_per_page, 128), 0)
    words = jnp.zeros((rows_per_page, 128), jnp.uint32)
    for j in range(packing):
        words = words | ((row * packing + j) << (bits * j))
    slots = pltpu.bitcast(words, uint).astype(jnp.int32)  # [S, 128], equal lanes
    return slots[:, :1]  # [S, 1]


def _fp4_page(kv_ref, scale_ref, k_row, v_row):
    """4-bit page (uint8 ``[1, S / 8, 4, 2D]``, `msa_fp4`) -> unscaled bf16
    rows ``[S, 2D]`` in slot order, and its K / V scales (scale-page rows
    `k_row` / `v_row`, static ints or run-time scalars) as ``[1, S]`` lane
    vectors. An unwritten slot's scale may be any bits: non-finite reads 0
    (int4 data itself has no NaN code)."""
    words = kv_ref.bitcast(jnp.uint32)[0, :, 0, :]
    rows = pltpu.bitcast(words, jnp.int4).astype(jnp.bfloat16)
    if isinstance(k_row, int):  # static rows
        page = scale_ref[0]  # [R, S]
        sk, sv = page[k_row : k_row + 1], page[v_row : v_row + 1]
    else:  # rows picked at run time (one kernel for every KV head)
        sk = scale_ref[0, pl.ds(k_row, 1), :]
        sv = scale_ref[0, pl.ds(v_row, 1), :]
    return (
        rows,
        jnp.where(jnp.isfinite(sk), sk, 0.0),
        jnp.where(jnp.isfinite(sv), sv, 0.0),
    )


def page_tokens(kv_cache) -> int:
    """Tokens per page of a packed cache (4-bit uint8 caches: 8 per word)."""
    rows, word = kv_cache.shape[1:3]
    return rows * (8 if kv_cache.dtype == jnp.uint8 else word)


def check_cache(kv_cache, scale_cache) -> None:
    """A 4-bit (uint8) cache needs its scale cache; others must not have one."""
    if kv_cache.dtype == jnp.uint8:
        if scale_cache is None or kv_cache.shape[2] != 4:
            raise NotImplementedError(
                "4-bit MSA caches are uint8 [pages, S / 8, 4, W] with a scale cache."
            )
        return
    if scale_cache is not None:
        raise ValueError("scale_cache is only for 4-bit (uint8) caches.")
    check_page_dtype(kv_cache.shape[2], kv_cache.dtype)


def _kernel_v2(
    # Scalar prefetch
    work_page_ref,  # [groups * G] page (or page key) of each slot, -1 for none
    work_meta_ref,  # [groups] tile << _TILE_SHIFT | last and first bits
    page_table_ref,  # page key -> physical page (read by the index maps only)
    # In
    q_ref,  # [H, Bq, D]
    selected_ref,  # [Bq, K] physical pages, -1 padded
    local_page_ref,  # [Bq, 1]
    local_limit_ref,  # [Bq, 1] visible slots of the local page
    *refs,  # G KV pages, [G scale pages (4-bit)], then out and scratch
    sm_scale: float,
    fp4_rows: tuple[int, int] | None = None,
):
    """`_kernel_v1` with pages unpacked by one bitcast each, and one mask
    select per score: masked scores are -inf against a finite running max,
    so their exp is already 0."""
    fp4 = fp4_rows is not None
    pages_per_step = (len(refs) - 4) // (2 if fp4 else 1)
    kv_refs = refs[:pages_per_step]
    scale_refs = refs[pages_per_step : 2 * pages_per_step] if fp4 else ()
    refs = refs[len(refs) - 4 :]
    o_ref, m_ref, l_ref, acc_ref = refs
    w = pl.program_id(0)
    meta = work_meta_ref[w]
    num_heads, block_q, head_dim = q_ref.shape
    rows_per_page, packing = kv_refs[0].shape[1:3]
    page_size = rows_per_page * (8 if fp4 else packing)

    @pl.when((meta & _FIRST_BIT) != 0)
    def _init():
        m_ref[...] = jnp.full_like(m_ref, _MASK_VALUE)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)

    row_slots = (
        jax.lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)  # int4: slot order
        if fp4
        else _bitcast_slots(rows_per_page, packing)
    )  # [S, 1]
    slots = row_slots.T  # [1, S]
    selected = selected_ref[...]
    local_page = local_page_ref[...]
    local_limit = local_limit_ref[...]
    keys, values, masks, k_scales, v_scales = [], [], [], [], []
    for g, kv_ref in enumerate(kv_refs):
        page = work_page_ref[w * pages_per_step + g]
        is_selected = jnp.max(
            (selected == page).astype(jnp.int32), axis=1, keepdims=True
        )
        limit = jnp.where(local_page == page, local_limit, page_size)
        limit = jnp.where(page >= 0, limit * is_selected, 0)  # [Bq, 1]
        # Zero what no token of the tile sees: p = 0 there, but 0 * NaN is
        # NaN and an unwritten slot may hold any bytes. Cheapest form per
        # dtype on v7x (j-3eb5e8aa, j-a9135755): a select on the converted
        # fp8 value rows; an AND on the packed bf16 words.
        seen = jnp.max(limit)
        if fp4:
            rows, sk, sv = _fp4_page(kv_ref, scale_refs[g], *fp4_rows)
            values.append(rows[:, head_dim:])
            k_scales.append(sk)
            v_scales.append(sv)
        elif packing == 4:
            rows = _page_rows(kv_ref, None)
            values.append(jnp.where(row_slots < seen, rows[:, head_dim:], 0))
        else:
            rows = _page_rows(kv_ref, seen)
            values.append(rows[:, head_dim:])
        keys.append(rows[:, :head_dim])
        masks.append(slots < limit)  # [Bq, S]
    k = jnp.concatenate(keys, axis=0)
    v = jnp.concatenate(values, axis=0)
    # Rows are head-major (h * block_q + t): every head shares a token's mask.
    mask = jnp.concatenate(masks, axis=1)
    mask = jnp.concatenate([mask] * num_heads, axis=0)

    q = q_ref[...].reshape(num_heads * block_q, head_dim)
    s = jax.lax.dot_general(
        q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )
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

    @pl.when((meta & _LAST_BIT) != 0)
    def _finalize():
        denom = l_ref[...]
        out = jnp.where(denom > 0, acc_ref[...] / jnp.where(denom > 0, denom, 1.0), 0.0)
        o_ref[...] = out.reshape(num_heads, block_q, head_dim).astype(o_ref.dtype)


_KERNELS = {"v1": _kernel_v1, "v2": _kernel_v2}


def _work_list(selected: jax.Array, block_q: int, pages_per_step: int):
    """Each tile's sorted page union, cut into groups of `pages_per_step`.

    Returns the flattened group pages (``-1`` past a tile's union), one meta
    word per group, and the number of groups.
    """
    num_tokens, topk = selected.shape
    num_tiles = num_tokens // block_q
    union_size = block_q * topk
    per_tile = jnp.where(selected < 0, _NO_PAGE, selected).reshape(
        num_tiles, union_size
    )
    union = jax.vmap(lambda x: jnp.unique(x, size=union_size, fill_value=_NO_PAGE))(
        per_tile
    )
    union = jnp.where(union == _NO_PAGE, -1, union)
    # A tile with no page still gets one all-masked group so its output is set.
    pages_in_tile = (union >= 0).sum(axis=1)
    groups_in_tile = jnp.maximum(-(-pages_in_tile // pages_per_step), 1)
    ends = jnp.cumsum(groups_in_tile)
    max_groups_per_tile = -(-union_size // pages_per_step)
    w = jnp.arange(num_tiles * max_groups_per_tile, dtype=jnp.int32)
    tile = jnp.minimum(
        jnp.searchsorted(ends, w, side="right", method="compare_all").astype(jnp.int32),
        num_tiles - 1,
    )
    group = w - (ends[tile] - groups_in_tile[tile])
    grouped = jnp.pad(
        union,
        ((0, 0), (0, max_groups_per_tile * pages_per_step - union_size)),
        constant_values=-1,
    ).reshape(num_tiles, max_groups_per_tile, pages_per_step)
    pages = grouped[tile, jnp.clip(group, 0, max_groups_per_tile - 1)]
    meta = (
        (tile << _TILE_SHIFT)
        | jnp.where(group == 0, _FIRST_BIT, 0)
        | jnp.where(group == groups_in_tile[tile] - 1, _LAST_BIT, 0)
    )
    return pages.reshape(-1), meta.astype(jnp.int32), ends[-1]


def _attention_call(
    q,
    kv_cache,
    selected,
    local_page,
    local_limit,
    *,
    sm_scale,
    block_q,
    pages_per_step,
    kernel,
    page_table,
    interpret,
    scale_cache=None,
    scale_rows=None,
):
    """One kernel call over a token count that is a multiple of block_q."""
    num_tokens, num_heads, head_dim = q.shape
    _, rows_per_page, packing, row_width = kv_cache.shape
    topk = selected.shape[1]
    pages, meta, num_groups = _work_list(selected, block_q, pages_per_step)
    prefetch = [pages, meta]
    if kernel == "v2":
        prefetch.append(
            jnp.zeros((1,), jnp.int32) if page_table is None else page_table
        )
    elif page_table is not None:
        raise NotImplementedError("page_table needs kernel v2")

    def tile_block(w, pages, meta, *_):
        return (meta[w] >> _TILE_SHIFT, 0)

    def head_major_block(w, pages, meta, *_):
        return (0, meta[w] >> _TILE_SHIFT, 0)

    def page_block(g):
        def index_map(w, pages, *rest):
            page = jnp.maximum(pages[w * pages_per_step + g], 0)
            if page_table is not None:
                page = rest[-1][page]
            return (page, 0, 0, 0)

        return pl.BlockSpec((1, rows_per_page, packing, row_width), index_map)

    def scale_block(g):  # the page's scale page, same page index
        def index_map(*args):
            return page_block(g).index_map(*args)[:3]

        return pl.BlockSpec((1,) + scale_cache.shape[1:], index_map)

    fp4 = scale_rows is not None
    if fp4 and kernel != "v2":
        raise NotImplementedError("4-bit caches need kernel v2")

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=len(prefetch),
        grid=(num_groups,),
        in_specs=[
            pl.BlockSpec((num_heads, block_q, head_dim), head_major_block),
            pl.BlockSpec((block_q, topk), tile_block),
            pl.BlockSpec((block_q, 1), tile_block),
            pl.BlockSpec((block_q, 1), tile_block),
            *(page_block(g) for g in range(pages_per_step)),
            *(scale_block(g) for g in range(pages_per_step) if fp4),
        ],
        out_specs=pl.BlockSpec((num_heads, block_q, head_dim), head_major_block),
        scratch_shapes=[
            pltpu.VMEM((num_heads * block_q, 1), jnp.float32),
            pltpu.VMEM((num_heads * block_q, 1), jnp.float32),
            pltpu.VMEM((num_heads * block_q, head_dim), jnp.float32),
        ],
    )
    out = pl.pallas_call(
        functools.partial(
            _KERNELS[kernel],
            sm_scale=sm_scale,
            **({"fp4_rows": scale_rows} if fp4 else {}),
        ),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((num_heads, num_tokens, head_dim), q.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), vmem_limit_bytes=_VMEM_LIMIT_BYTES
        ),
        interpret=interpret,
        name="minimax_m3_msa_sparse_attention",
    )(
        *prefetch,
        q.transpose(1, 0, 2),
        selected,
        local_page[:, None],
        local_limit[:, None],
        *([kv_cache] * pages_per_step),
        *([scale_cache] * (pages_per_step if fp4 else 0)),
    )
    return out.transpose(1, 0, 2)


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "block_q",
        "pages_per_step",
        "kernel",
        "k_scale_row",
        "v_scale_row",
        "interpret",
    ),
)
def msa_sparse_attention(
    q: jax.Array,
    kv_cache: jax.Array,
    selected: jax.Array,
    local_page: jax.Array,
    local_limit: jax.Array,
    *,
    sm_scale: float,
    block_q: int = DEFAULT_BLOCK_Q,
    pages_per_step: int = DEFAULT_PAGES_PER_STEP,
    kernel: str = DEFAULT_KERNEL,
    page_table: jax.Array | None = None,
    scale_cache: jax.Array | None = None,
    k_scale_row: int = 0,
    v_scale_row: int = 1,
    interpret: bool = False,
) -> jax.Array:
    """Attention of each token over its selected pages.

    Args:
      q: ``[T, H, D]`` queries of one KV head's query heads.
      kv_cache: ``[pages, S / p, p, 2D]`` packed ``[K | V]`` rows (bf16 or fp8).
      selected: ``[T, K]`` physical pages each token attends, ``-1`` padded.
      local_page: ``[T]`` the page holding the token itself (``-1`` for
        padding tokens).
      local_limit: ``[T]`` slots of the local page the token sees.
      kernel: ``"v2"`` (bitcast unpack, one mask select) or the original
        ``"v1"``, kept as the reference arm for benchmarks.
      page_table: optional flat table from page keys to physical pages. With
        it, `selected` and `local_page` hold keys (any ids unique per page,
        e.g. ``seq * pages_per_seq + block``) and only the kernel's page
        fetch translates them, so no [T, K] gather runs in XLA (v2 only).
      scale_cache: with a 4-bit uint8 `kv_cache` (`msa_fp4`), its scale cache
        ``[pages, 8, S]`` f32; rows `k_scale_row` / `v_scale_row` hold the
        K / V scale of each slot (x = int4 * scale). `q` is then in the
        rotated basis and so is the output.
    Returns:
      ``[T, H, D]`` outputs; padding tokens get zeros.
    """
    num_tokens, _, head_dim = q.shape
    check_cache(kv_cache, scale_cache)
    if kv_cache.shape[3] != 2 * head_dim:
        raise ValueError(
            f"Cache rows hold [K | V] of width {2 * head_dim}, got {kv_cache.shape}."
        )
    pad = -num_tokens % block_q
    if pad:
        q = jnp.pad(q, ((0, pad), (0, 0), (0, 0)))
        selected = jnp.pad(selected, ((0, pad), (0, 0)), constant_values=-1)
        local_page = jnp.pad(local_page, (0, pad), constant_values=-1)
        local_limit = jnp.pad(local_limit, (0, pad))
    call = functools.partial(
        _attention_call,
        sm_scale=sm_scale,
        block_q=block_q,
        pages_per_step=pages_per_step,
        kernel=kernel,
        page_table=page_table,
        interpret=interpret,
        scale_cache=scale_cache,
        scale_rows=None if scale_cache is None else (k_scale_row, v_scale_row),
    )
    step = max(block_q, MAX_TOKENS_PER_CALL - MAX_TOKENS_PER_CALL % block_q)
    outputs = [
        call(
            q[start : start + step],
            kv_cache,
            selected[start : start + step],
            local_page[start : start + step],
            local_limit[start : start + step],
        )
        for start in range(0, num_tokens + pad, step)
    ]
    return jnp.concatenate(outputs, axis=0)[:num_tokens]
