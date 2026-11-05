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

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention
from vllm_torchtpu.models.minimax_m3.kernels.packed_page import check_page_dtype


def _kernel(
    # Scalar prefetch
    work_page_ref,  # [groups * G] page key of each slot, -1 for none
    work_meta_ref,  # [groups] tile << _TILE_SHIFT | last and first bits
    page_table_ref,  # page key -> physical page (read by the index maps only)
    # In
    q_ref,  # [H, Bq, D]
    selected_ref,  # [Bq, K] page keys, -1 padded
    local_page_ref,  # [Bq, 1]
    local_limit_ref,  # [Bq, 1] visible slots of the local page
    *refs,  # G x [1, S / p, p, 2D] KV pages, then out, lse and scratch
    sm_scale: float,
):
    """`msa_attention._kernel_v2` that also writes ``lse = m + log(l)``
    (``-inf`` for a row that sees no slot; its output is 0)."""
    pages_per_step = len(refs) - 5
    kv_refs = refs[:pages_per_step]
    o_ref, lse_ref, m_ref, l_ref, acc_ref = refs[pages_per_step:]
    w = pl.program_id(0)
    meta = work_meta_ref[w]
    num_heads, block_q, head_dim = q_ref.shape
    rows_per_page, packing = kv_refs[0].shape[1:3]
    page_size = rows_per_page * packing

    @pl.when((meta & msa_attention._FIRST_BIT) != 0)
    def _init():
        m_ref[...] = jnp.full_like(m_ref, msa_attention._MASK_VALUE)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)

    row_slots = msa_attention._bitcast_slots(rows_per_page, packing)  # [S, 1]
    slots = row_slots.T  # [1, S]
    selected = selected_ref[...]
    local_page = local_page_ref[...]
    local_limit = local_limit_ref[...]
    keys, values, masks = [], [], []
    for g, kv_ref in enumerate(kv_refs):
        page = work_page_ref[w * pages_per_step + g]
        is_selected = jnp.max(
            (selected == page).astype(jnp.int32), axis=1, keepdims=True
        )
        limit = jnp.where(local_page == page, local_limit, page_size)
        limit = jnp.where(page >= 0, limit * is_selected, 0)  # [Bq, 1]
        seen = jnp.max(limit)
        if packing == 4:
            rows = msa_attention._page_rows(kv_ref, None)
            values.append(jnp.where(row_slots < seen, rows[:, head_dim:], 0))
        else:
            rows = msa_attention._page_rows(kv_ref, seen)
            values.append(rows[:, head_dim:])
        keys.append(rows[:, :head_dim])
        masks.append(slots < limit)  # [Bq, S]
    k = jnp.concatenate(keys, axis=0)
    v = jnp.concatenate(values, axis=0)
    mask = jnp.concatenate(masks, axis=1)
    mask = jnp.concatenate([mask] * num_heads, axis=0)

    q = q_ref[...].reshape(num_heads * block_q, head_dim)
    s = jax.lax.dot_general(
        q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )
    s = jnp.where(mask, s * sm_scale, -jnp.inf)

    m_prev = m_ref[...]
    m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(s - m_new)
    l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=1, keepdims=True)
    acc_ref[...] = alpha * acc_ref[...] + jax.lax.dot_general(
        p.astype(v.dtype),
        v,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    m_ref[...] = m_new

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
    pages, meta, num_groups = msa_attention._work_list(
        selected, block_q, pages_per_step
    )

    def tile_block(w, pages, meta, *_):
        return (meta[w] >> msa_attention._TILE_SHIFT, 0)

    def head_major_block(w, pages, meta, *_):
        return (0, meta[w] >> msa_attention._TILE_SHIFT, 0)

    def page_block(g):
        def index_map(w, pages, meta, table):
            return (table[jnp.maximum(pages[w * pages_per_step + g], 0)], 0, 0, 0)

        return pl.BlockSpec((1, rows_per_page, packing, row_width), index_map)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        grid=(num_groups,),
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
        functools.partial(_kernel, sm_scale=sm_scale),
        grid_spec=grid_spec,
        out_shape=[
            jax.ShapeDtypeStruct((num_heads, num_tokens, head_dim), q.dtype),
            jax.ShapeDtypeStruct((num_heads, num_tokens, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=msa_attention._VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="minimax_m3_msa_sparse_attention_lse",
    )(
        pages,
        meta,
        page_table,
        q.transpose(1, 0, 2),
        selected,
        local_page[:, None],
        local_limit[:, None],
        *([kv_cache] * pages_per_step),
    )
    return out.transpose(1, 0, 2), lse[:, :, 0].T


@functools.partial(
    jax.jit, static_argnames=("sm_scale", "block_q", "pages_per_step", "interpret")
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
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`msa_attention.msa_sparse_attention` (kernel v2, page keys translated
    by `page_table`) returning ``(out [T, H, D], lse [T, H] f32)``.

    A token with no selected page (or a padding token) gets a zero output
    and ``lse = -inf``, so it carries no weight in a log-sum-exp merge.
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
    limit = msa_attention.MAX_TOKENS_PER_CALL
    step = max(block_q, limit - limit % block_q)
    results = [
        call(
            q[start : start + step],
            kv_cache,
            selected[start : start + step],
            local_page[start : start + step],
            local_limit[start : start + step],
            page_table,
        )
        for start in range(0, num_tokens + pad, step)
    ]
    out = jnp.concatenate([o for o, _ in results], axis=0)[:num_tokens]
    lse = jnp.concatenate([s for _, s in results], axis=0)[:num_tokens]
    return out, lse
