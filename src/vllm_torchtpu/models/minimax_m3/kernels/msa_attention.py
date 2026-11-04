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

DEFAULT_BLOCK_Q = 64
DEFAULT_PAGES_PER_STEP = 8
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


def _kernel(
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
    for g, kv_ref in enumerate(kv_refs):
        page = work_page_ref[w * pages_per_step + g]
        rows = unpack_page(kv_ref)
        keys.append(jnp.concatenate([r[:, :head_dim] for r in rows], axis=0))
        values.append(jnp.concatenate([r[:, head_dim:] for r in rows], axis=0))
        is_selected = jnp.max(
            (selected == page).astype(jnp.int32), axis=1, keepdims=True
        )
        limit = jnp.where(local_page_ref[...] == page, local_limit_ref[...], page_size)
        limit = jnp.where(page >= 0, limit * is_selected, 0)
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
        jnp.searchsorted(ends, w, side="right").astype(jnp.int32), num_tiles - 1
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
    interpret,
):
    """One kernel call over a token count that is a multiple of block_q."""
    num_tokens, num_heads, head_dim = q.shape
    _, rows_per_page, packing, row_width = kv_cache.shape
    topk = selected.shape[1]
    pages, meta, num_groups = _work_list(selected, block_q, pages_per_step)

    def tile_block(w, pages, meta):
        return (meta[w] >> _TILE_SHIFT, 0)

    def head_major_block(w, pages, meta):
        return (0, meta[w] >> _TILE_SHIFT, 0)

    def page_block(g):
        def index_map(w, pages, meta):
            return (jnp.maximum(pages[w * pages_per_step + g], 0), 0, 0, 0)

        return pl.BlockSpec((1, rows_per_page, packing, row_width), index_map)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(num_groups,),
        in_specs=[
            pl.BlockSpec((num_heads, block_q, head_dim), head_major_block),
            pl.BlockSpec((block_q, topk), tile_block),
            pl.BlockSpec((block_q, 1), tile_block),
            pl.BlockSpec((block_q, 1), tile_block),
            *(page_block(g) for g in range(pages_per_step)),
        ],
        out_specs=pl.BlockSpec((num_heads, block_q, head_dim), head_major_block),
        scratch_shapes=[
            pltpu.VMEM((num_heads * block_q, 1), jnp.float32),
            pltpu.VMEM((num_heads * block_q, 1), jnp.float32),
            pltpu.VMEM((num_heads * block_q, head_dim), jnp.float32),
        ],
    )
    out = pl.pallas_call(
        functools.partial(_kernel, sm_scale=sm_scale),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((num_heads, num_tokens, head_dim), q.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), vmem_limit_bytes=_VMEM_LIMIT_BYTES
        ),
        interpret=interpret,
        name="minimax_m3_msa_sparse_attention",
    )(
        pages,
        meta,
        q.transpose(1, 0, 2),
        selected,
        local_page[:, None],
        local_limit[:, None],
        *([kv_cache] * pages_per_step),
    )
    return out.transpose(1, 0, 2)


@functools.partial(
    jax.jit, static_argnames=("sm_scale", "block_q", "pages_per_step", "interpret")
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
    Returns:
      ``[T, H, D]`` outputs; padding tokens get zeros.
    """
    num_tokens, _, head_dim = q.shape
    check_page_dtype(kv_cache.shape[2], kv_cache.dtype)
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
        interpret=interpret,
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
