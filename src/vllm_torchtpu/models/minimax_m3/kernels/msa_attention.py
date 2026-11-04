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

Tokens are grouped into query tiles of `block_q`. A tile loads the union of
the pages its tokens selected, one page per grid step, and masks per token:
a page counts for a token only if the token selected it, and its local page
only up to the token itself. Consecutive tokens of a prefill chunk select
mostly the same pages, so each page is read about once per tile instead of
once per token. The (tile, page) work list is flattened ahead of the kernel
and run on a dynamic grid, so empty union slots cost nothing.

Layout: one KV head (K and V packed in the bf16 cache's 32-bit words, as the
ragged paged attention v3 cache stores them), `num_heads` query heads.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_BLOCK_Q = 16
_MASK_VALUE = -1e30
_NO_PAGE = jnp.iinfo(jnp.int32).max


def _kernel(
    # Scalar prefetch
    work_tile_ref,
    work_page_ref,
    work_first_ref,
    work_last_ref,
    # In
    q_ref,  # [H, Bq, D]
    selected_ref,  # [Bq, K] physical pages, -1 padded
    local_page_ref,  # [Bq, 1]
    local_limit_ref,  # [Bq, 1] visible slots of the local page
    kv_ref,  # [S, 1, 2, D] bf16 K/V packed per 32-bit word
    # Out
    o_ref,  # [H, Bq, D]
    # Scratch
    m_ref,
    l_ref,
    acc_ref,
    *,
    sm_scale: float,
):
    w = pl.program_id(0)
    page = work_page_ref[w]
    num_heads, block_q, head_dim = q_ref.shape
    page_size = kv_ref.shape[0]

    @pl.when(work_first_ref[w] == 1)
    def _init():
        m_ref[...] = jnp.full_like(m_ref, _MASK_VALUE)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)

    packed = kv_ref.bitcast(jnp.uint32).reshape(page_size, head_dim)[...]
    k = pltpu.bitcast((packed & 0xFFFF).astype(jnp.uint16), jnp.bfloat16)
    v = pltpu.bitcast((packed >> 16).astype(jnp.uint16), jnp.bfloat16)

    q = q_ref[...].reshape(num_heads * block_q, head_dim)
    s = jax.lax.dot_general(
        q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )
    s = s * sm_scale

    is_selected = jnp.max(
        (selected_ref[...] == page).astype(jnp.int32), axis=1, keepdims=True
    )
    limit = jnp.where(local_page_ref[...] == page, local_limit_ref[...], page_size)
    limit = limit * is_selected
    slots = jax.lax.broadcasted_iota(jnp.int32, (block_q, page_size), 1)
    token_mask = slots < limit
    mask = jnp.concatenate([token_mask] * num_heads, axis=0)
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

    @pl.when(work_last_ref[w] == 1)
    def _finalize():
        denom = l_ref[...]
        out = jnp.where(denom > 0, acc_ref[...] / jnp.where(denom > 0, denom, 1.0), 0.0)
        o_ref[...] = out.reshape(num_heads, block_q, head_dim).astype(o_ref.dtype)


def _work_list(selected: jax.Array, block_q: int):
    """Flattened (tile, page) pairs: each tile's sorted page union."""
    num_tokens, topk = selected.shape
    num_tiles = num_tokens // block_q
    per_tile = jnp.where(selected < 0, _NO_PAGE, selected).reshape(
        num_tiles, block_q * topk
    )
    union = jax.vmap(lambda x: jnp.unique(x, size=block_q * topk, fill_value=_NO_PAGE))(
        per_tile
    )
    # A tile with no page still gets one all-masked step so its output is set.
    count = jnp.maximum((union != _NO_PAGE).sum(axis=1), 1)
    union = jnp.where(union == _NO_PAGE, 0, union)
    ends = jnp.cumsum(count)
    total = ends[-1]
    w = jnp.arange(num_tiles * block_q * topk, dtype=jnp.int32)
    tile = jnp.minimum(
        jnp.searchsorted(ends, w, side="right").astype(jnp.int32), num_tiles - 1
    )
    idx = w - (ends[tile] - count[tile])
    page = union[tile, jnp.clip(idx, 0, block_q * topk - 1)]
    first = (idx == 0).astype(jnp.int32)
    last = (idx == count[tile] - 1).astype(jnp.int32)
    return tile, page, first, last, total


@functools.partial(jax.jit, static_argnames=("sm_scale", "block_q", "interpret"))
def msa_sparse_attention(
    q: jax.Array,
    kv_cache: jax.Array,
    selected: jax.Array,
    local_page: jax.Array,
    local_limit: jax.Array,
    *,
    sm_scale: float,
    block_q: int = DEFAULT_BLOCK_Q,
    interpret: bool = False,
) -> jax.Array:
    """Attention of each token over its selected pages.

    Args:
      q: ``[T, H, D]`` queries of one KV head's query heads.
      kv_cache: ``[pages, S, 1, 2, D]`` bf16 cache (K, V packed per word).
      selected: ``[T, K]`` physical pages each token attends, ``-1`` padded.
      local_page: ``[T]`` the page holding the token itself (``-1`` for
        padding tokens).
      local_limit: ``[T]`` slots of the local page the token sees.
    Returns:
      ``[T, H, D]`` outputs; padding tokens get zeros.
    """
    num_tokens, num_heads, head_dim = q.shape
    _, page_size, kv_groups, packing, _ = kv_cache.shape
    if (kv_groups, packing, kv_cache.dtype) != (1, 2, jnp.bfloat16):
        raise NotImplementedError(
            "MSA attention supports one bf16 KV head per rank, got cache "
            f"{kv_cache.shape} {kv_cache.dtype}."
        )
    topk = selected.shape[1]
    pad = -num_tokens % block_q
    if pad:
        q = jnp.pad(q, ((0, pad), (0, 0), (0, 0)))
        selected = jnp.pad(selected, ((0, pad), (0, 0)), constant_values=-1)
        local_page = jnp.pad(local_page, (0, pad), constant_values=-1)
        local_limit = jnp.pad(local_limit, (0, pad))
    padded_tokens = num_tokens + pad
    tile, page, first, last, total = _work_list(selected, block_q)

    def tile_block(w, tile, *_):
        return (tile[w], 0)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4,
        grid=(total,),
        in_specs=[
            pl.BlockSpec(
                (num_heads, block_q, head_dim),
                lambda w, tile, *_: (0, tile[w], 0),
            ),
            pl.BlockSpec((block_q, topk), tile_block),
            pl.BlockSpec((block_q, 1), tile_block),
            pl.BlockSpec((block_q, 1), tile_block),
            pl.BlockSpec(
                (None, page_size, 1, 2, head_dim),
                lambda w, tile, page, *_: (page[w], 0, 0, 0, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (num_heads, block_q, head_dim), lambda w, tile, *_: (0, tile[w], 0)
        ),
        scratch_shapes=[
            pltpu.VMEM((num_heads * block_q, 1), jnp.float32),
            pltpu.VMEM((num_heads * block_q, 1), jnp.float32),
            pltpu.VMEM((num_heads * block_q, head_dim), jnp.float32),
        ],
    )
    out = pl.pallas_call(
        functools.partial(_kernel, sm_scale=sm_scale),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((num_heads, padded_tokens, head_dim), q.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name="minimax_m3_msa_sparse_attention",
    )(
        tile,
        page,
        first,
        last,
        q.transpose(1, 0, 2),
        selected,
        local_page[:, None],
        local_limit[:, None],
        kv_cache,
    )
    return out.transpose(1, 0, 2)[:num_tokens]
