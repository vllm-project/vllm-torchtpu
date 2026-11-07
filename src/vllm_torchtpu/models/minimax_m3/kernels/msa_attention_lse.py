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
    *refs,  # G KV pages, [G scale pages (4-bit)], out, lse, scratch
    sm_scale: float,
    fp4_rows: tuple[int, int] | str | None = None,
):
    """`msa_attention._kernel_v2` that also writes ``lse = m + log(l)``
    (``-inf`` for a row that sees no slot; its output is 0)."""
    fp4 = fp4_rows is not None
    pages_per_step = (len(refs) - 5) // (2 if fp4 else 1)
    kv_refs = refs[:pages_per_step]
    scale_refs = refs[pages_per_step : 2 * pages_per_step] if fp4 else ()
    refs = refs[len(refs) - 5 :]
    o_ref, lse_ref, m_ref, l_ref, acc_ref = refs
    w = pl.program_id(0)
    meta = work_meta_ref[w]
    num_heads, block_q, head_dim = q_ref.shape
    rows_per_page, packing = kv_refs[0].shape[1:3]
    page_size = rows_per_page * (8 if fp4 else packing)

    @pl.when((meta & msa_attention._FIRST_BIT) != 0)
    def _init():
        m_ref[...] = jnp.full_like(m_ref, msa_attention._MASK_VALUE)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)

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
    for g, kv_ref in enumerate(kv_refs):
        page = work_page_ref[w * pages_per_step + g]
        is_selected = jnp.max(
            (selected == page).astype(jnp.int32), axis=1, keepdims=True
        )
        limit = jnp.where(local_page == page, local_limit, page_size)
        limit = jnp.where(page >= 0, limit * is_selected, 0)  # [Bq, 1]
        seen = jnp.max(limit)
        if fp4:
            if fp4_rows == "head":  # rows 2h, 2h + 1 of the scale page
                k_row = 2 * head_ref[0]
                rows, sk, sv = msa_attention._fp4_page(
                    kv_ref, scale_refs[g], k_row, k_row + 1
                )
            else:
                rows, sk, sv = msa_attention._fp4_page(kv_ref, scale_refs[g], *fp4_rows)
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
    *,
    sm_scale,
    block_q,
    pages_per_step,
    interpret,
    scale_cache=None,
    scale_rows=None,
):
    """One kernel call over a token count that is a multiple of block_q.

    ``head`` (``int32[1]``) is a scalar-prefetch input, not a static, so one
    kernel serves every KV head of the cache rows."""
    num_tokens, num_heads, head_dim = q.shape
    _, rows_per_page, packing, _ = kv_cache.shape
    # One KV head's [K | V] slice of each row.
    row_width = 2 * head_dim
    topk = selected.shape[1]
    pages, meta, num_groups = msa_attention._work_list(
        selected, block_q, pages_per_step
    )

    def tile_block(w, pages, meta, *_):
        return (meta[w] >> msa_attention._TILE_SHIFT, 0)

    def head_major_block(w, pages, meta, *_):
        return (0, meta[w] >> msa_attention._TILE_SHIFT, 0)

    def page_block(g):
        def index_map(w, pages, meta, table, head_idx):
            page = table[jnp.maximum(pages[w * pages_per_step + g], 0)]
            return (page, 0, 0, head_idx[0])

        return pl.BlockSpec((1, rows_per_page, packing, row_width), index_map)

    def scale_block(g):  # the page's scale page, same page index
        def index_map(w, pages, meta, table, head_idx):
            return (table[jnp.maximum(pages[w * pages_per_step + g], 0)], 0, 0)

        return pl.BlockSpec((1,) + scale_cache.shape[1:], index_map)

    fp4 = scale_rows is not None

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4,
        grid=(num_groups,),
        in_specs=[
            pl.BlockSpec((num_heads, block_q, head_dim), head_major_block),
            pl.BlockSpec((block_q, topk), tile_block),
            pl.BlockSpec((block_q, 1), tile_block),
            pl.BlockSpec((block_q, 1), tile_block),
            *(page_block(g) for g in range(pages_per_step)),
            *(scale_block(g) for g in range(pages_per_step) if fp4),
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
            _kernel, sm_scale=sm_scale, **({"fp4_rows": scale_rows} if fp4 else {})
        ),
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
        head,
        q.transpose(1, 0, 2),
        selected,
        local_page[:, None],
        local_limit[:, None],
        *([kv_cache] * pages_per_step),
        *([scale_cache] * (pages_per_step if fp4 else 0)),
    )
    return out.transpose(1, 0, 2), lse[:, :, 0].T


def _chunked_call(
    q,
    kv_cache,
    selected,
    local_page,
    local_limit,
    page_table,
    head,
    *,
    sm_scale,
    block_q,
    pages_per_step,
    interpret,
    scale_cache=None,
    scale_rows=None,
):
    """`_attention_call` over chunks of at most `MAX_TOKENS_PER_CALL` tokens
    (the work list lives in SMEM). Several chunks run as one `lax.map`, so
    the kernel is emitted once per call site rather than once per chunk."""
    num_tokens = q.shape[0]
    call = functools.partial(
        _attention_call,
        sm_scale=sm_scale,
        block_q=block_q,
        pages_per_step=pages_per_step,
        interpret=interpret,
        scale_cache=scale_cache,
        scale_rows=scale_rows,
    )
    limit = msa_attention.MAX_TOKENS_PER_CALL
    step = max(block_q, limit - limit % block_q)
    num_chunks = -(-num_tokens // step)
    if num_chunks == 1:
        return call(q, kv_cache, selected, local_page, local_limit, page_table, head)
    pad = num_chunks * step - num_tokens
    q = jnp.pad(q, ((0, pad), (0, 0), (0, 0)))
    selected = jnp.pad(selected, ((0, pad), (0, 0)), constant_values=-1)
    local_page = jnp.pad(local_page, (0, pad), constant_values=-1)
    local_limit = jnp.pad(local_limit, (0, pad))

    def chunk(xs):
        q_c, sel_c, page_c, limit_c = xs
        return call(q_c, kv_cache, sel_c, page_c, limit_c, page_table, head)

    out, lse = jax.lax.map(
        chunk,
        (
            q.reshape(num_chunks, step, *q.shape[1:]),
            selected.reshape(num_chunks, step, -1),
            local_page.reshape(num_chunks, step),
            local_limit.reshape(num_chunks, step),
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
        "k_scale_row",
        "v_scale_row",
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
    scale_cache: jax.Array | None = None,
    k_scale_row: int = 0,
    v_scale_row: int = 1,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`msa_attention.msa_sparse_attention` (kernel v2, page keys translated
    by `page_table`) returning ``(out [T, H, D], lse [T, H] f32)``.

    A token with no selected page (or a padding token) gets a zero output
    and ``lse = -inf``, so it carries no weight in a log-sum-exp merge.
    Cache rows may hold several KV heads, ``[K_0 | V_0 | K_1 | V_1 ...]``;
    `head` picks the one these queries attend (a 2D-wide block of each page).
    """
    num_tokens, _, head_dim = q.shape
    msa_attention.check_cache(kv_cache, scale_cache)
    _check_rows(kv_cache, head_dim, head)
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
        scale_cache=scale_cache,
        scale_rows=None if scale_cache is None else (k_scale_row, v_scale_row),
    )
    return out[:num_tokens], lse[:num_tokens]


@functools.partial(
    jax.jit, static_argnames=("sm_scale", "block_q", "pages_per_step", "interpret")
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
    scale_cache: jax.Array | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`msa_sparse_attention_lse` for every KV head of the cache rows at once.

    ``q [T, G * Hg, D]`` (KV head ``h`` owns query heads ``[h * Hg,
    (h + 1) * Hg)``), ``selected [T, G, K]``. The heads run as one `lax.map`
    over a single kernel whose KV head is an input, so the program carries
    one copy of the kernel instead of one per head. Returns ``(out
    [T, G * Hg, D], lse [T, G * Hg] f32)``.

    With a 4-bit cache, `scale_cache` ``[pages, R, S]`` holds head ``h``'s K / V
    scales in rows ``2h`` / ``2h + 1``.
    """
    num_tokens, num_q_heads, head_dim = q.shape
    num_heads = selected.shape[1]
    msa_attention.check_cache(kv_cache, scale_cache)
    if scale_cache is not None and scale_cache.shape[1] < 2 * num_heads:
        raise ValueError(
            f"scale pages {scale_cache.shape} miss K/V rows of {num_heads} heads."
        )
    if _check_rows(kv_cache, head_dim) < num_heads or num_q_heads % num_heads:
        raise ValueError(
            f"{num_heads} KV heads of {num_q_heads} query heads do not fit cache "
            f"rows {kv_cache.shape}."
        )
    q, selected, local_page, local_limit = _pad_tokens(
        q, selected, local_page, local_limit, block_q
    )
    padded = q.shape[0]
    q_per_head = num_q_heads // num_heads
    q_heads = q.reshape(padded, num_heads, q_per_head, head_dim).transpose(1, 0, 2, 3)

    def one_head(xs):
        q_h, sel_h, h = xs
        return _chunked_call(
            q_h,
            kv_cache,
            sel_h,
            local_page,
            local_limit,
            page_table,
            h.reshape(1),
            sm_scale=sm_scale,
            block_q=block_q,
            pages_per_step=pages_per_step,
            interpret=interpret,
            scale_cache=scale_cache,
            scale_rows=None if scale_cache is None else "head",
        )

    out, lse = jax.lax.map(
        one_head,
        (q_heads, selected.transpose(1, 0, 2), jnp.arange(num_heads, dtype=jnp.int32)),
    )  # [G, T, Hg, D], [G, T, Hg]
    out = out.transpose(1, 0, 2, 3).reshape(padded, num_q_heads, head_dim)
    lse = lse.transpose(1, 0, 2).reshape(padded, num_q_heads)
    return out[:num_tokens], lse[:num_tokens]
