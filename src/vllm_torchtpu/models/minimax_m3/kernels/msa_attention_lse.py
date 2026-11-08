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
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention
from vllm_torchtpu.models.minimax_m3.kernels.packed_page import check_page_dtype

# Page counts a kernel step may run at besides the full group (`_kernel`).
_STEP_WIDTHS = (4, 8)
_LOG2_E = math.log2(math.e)
_LN_2 = math.log(2.0)


def _kernel(
    # Scalar prefetch
    work_page_ref,  # tile page key unions (`msa_attention._work_list`)
    work_meta_ref,  # [groups] union row << _ROW_SHIFT | last and first bits
    page_table_ref,  # page key -> physical page (read by the index maps only)
    head_ref,  # [1] KV head of the cache rows (read by the index maps only)
    tile_offset_ref,  # [1] the call's first tile (read by the index maps only)
    # In
    q_ref,  # [H, Bq, D]
    selected_ref,  # [Bq, K] page keys, -1 padded
    local_page_ref,  # [Bq, 1]
    local_limit_ref,  # [Bq, 1] visible slots of the local page
    *refs,  # G x [1, S / p, p, 2D] KV pages, [aliased out, lse], out, lse, scratch
    sm_scale: float,
    aliased: bool = False,
):
    """`msa_attention._kernel_v2` that also writes ``lse = m + log(l)``
    (``-inf`` for a row that sees no slot; its output is 0). The running
    max ``m`` is kept in log2 units."""
    pages_per_step = len(refs) - (7 if aliased else 5)
    kv_refs = refs[:pages_per_step]
    o_ref, lse_ref, m_ref, l_ref, acc_ref = refs[len(refs) - 5 :]
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

    def attend(width):
        """One online-softmax update over the group's first `width` slots."""
        row_slots = msa_attention._bitcast_slots(rows_per_page, packing)  # [S, 1]
        slots = row_slots.T  # [1, S]
        selected = selected_ref[...]
        local_page = local_page_ref[...]
        local_limit = local_limit_ref[...]
        keys, values, masks = [], [], []
        for g, kv_ref in enumerate(kv_refs[:width]):
            page = work_page_ref[msa_attention._slot(meta, pages_per_step, g)]
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
        # Scores in log2 units (m and l too): exp2 needs no multiply per score.
        s = jnp.where(mask, s * (sm_scale * _LOG2_E), -jnp.inf)

        m_prev = m_ref[...]
        m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
        alpha = jnp.exp2(m_prev - m_new)
        p = jnp.exp2(s - m_new)
        l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=1, keepdims=True)
        acc_ref[...] = alpha * acc_ref[...] + jax.lax.dot_general(
            p.astype(v.dtype),
            v,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        m_ref[...] = m_new

    # Run at the narrowest width holding the group's pages: unions are
    # ascending with -1 at the end, so slot `n` is empty iff at most n pages
    # remain. A step costs per score column, filled or not (~2.3 us at 16
    # pages, ~1.1 us at 4, twin standalone j-952e8a64), and a 16-token
    # tile's union is ~8-12 of its rank's pages at DCP 2, ~2-8 at DCP 8.
    def present(g):
        return work_page_ref[msa_attention._slot(meta, pages_per_step, g)] >= 0

    widths = [n for n in _STEP_WIDTHS if n < pages_per_step] + [pages_per_step]
    for i, width in enumerate(widths):
        wider = present(widths[i - 1]) if i else True
        fits = jnp.logical_not(present(width)) if width < pages_per_step else True
        pl.when(jnp.logical_and(wider, fits))(functools.partial(attend, width))

    @pl.when((meta & msa_attention._LAST_BIT) != 0)
    def _finalize():
        denom = l_ref[...]
        seen_any = denom > 0
        safe = jnp.where(seen_any, denom, 1.0)
        out = jnp.where(seen_any, acc_ref[...] / safe, 0.0)
        o_ref[...] = out.reshape(num_heads, block_q, head_dim).astype(o_ref.dtype)
        lse = jnp.where(seen_any, m_ref[...] * _LN_2 + jnp.log(safe), -jnp.inf)
        lse_ref[...] = lse.reshape(num_heads, block_q, 1)


def _attention_call(
    q,
    kv_cache,
    selected,
    local_page,
    local_limit,
    page_table,
    head,
    tile_offset,
    out,
    *,
    sm_scale,
    block_q,
    pages_per_step,
    interpret,
):
    """One kernel call over the tokens of `selected` (a multiple of block_q),
    which start at tile ``tile_offset`` (``int32[1]``) of the head-major
    ``q [H, T, D]``, ``local_page`` and ``local_limit [T, 1]``.

    Writes those tokens' rows of ``(out [H, T, D], lse [H, T, 1])``: into
    `out` in place when given (an earlier call's result), else into new
    arrays whose other rows are unset. ``head`` (``int32[1]``) is a
    scalar-prefetch input, not a static, so one kernel serves every KV head
    of the cache rows."""
    num_heads, num_tokens, head_dim = q.shape
    _, rows_per_page, packing, _ = kv_cache.shape
    # One KV head's [K | V] slice of each row.
    row_width = 2 * head_dim
    topk = selected.shape[1]
    pages, meta, num_groups = msa_attention._work_list(
        selected, block_q, pages_per_step
    )

    groups_per_tile = msa_attention._groups_per_tile(block_q, topk, pages_per_step)

    def call_tile_block(w, pages, meta, *_):
        return (msa_attention._meta_tile(meta[w], groups_per_tile), 0)

    def tile_block(w, pages, meta, table, head_idx, offset):
        tile = offset[0] + msa_attention._meta_tile(meta[w], groups_per_tile)
        return (tile, 0)

    def head_major_block(*args):
        return (0, *tile_block(*args))

    def page_block(g):
        def index_map(w, pages, meta, table, head_idx, offset):
            slot = msa_attention._slot(meta[w], pages_per_step, g)
            page = table[jnp.maximum(pages[slot], 0)]
            return (page, 0, 0, head_idx[0])

        return pl.BlockSpec((1, rows_per_page, packing, row_width), index_map)

    aliased = out is not None
    in_place = (pl.BlockSpec(memory_space=pl.ANY),) * 2 if aliased else ()
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=5,
        grid=(num_groups,),
        in_specs=[
            pl.BlockSpec((num_heads, block_q, head_dim), head_major_block),
            pl.BlockSpec((block_q, topk), call_tile_block),
            pl.BlockSpec((block_q, 1), tile_block),
            pl.BlockSpec((block_q, 1), tile_block),
            *(page_block(g) for g in range(pages_per_step)),
            *in_place,
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
    num_inputs = 5 + 4 + pages_per_step
    return pl.pallas_call(
        functools.partial(_kernel, sm_scale=sm_scale, aliased=aliased),
        grid_spec=grid_spec,
        out_shape=[
            jax.ShapeDtypeStruct((num_heads, num_tokens, head_dim), q.dtype),
            jax.ShapeDtypeStruct((num_heads, num_tokens, 1), jnp.float32),
        ],
        input_output_aliases=(
            {num_inputs: 0, num_inputs + 1: 1} if aliased else {}
        ),
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
        tile_offset,
        q,
        selected,
        local_page,
        local_limit,
        *([kv_cache] * pages_per_step),
        *(out if aliased else ()),
    )


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
):
    """`_attention_call` over chunks of at most `MAX_TOKENS_PER_CALL` tokens
    (the work list lives in SMEM).

    Every call reads the whole head-major query and writes its chunk's rows
    of one output in place, so the chunks need no per-chunk slices,
    transposes or stacking (~47 us per 2,048-token chunk under the former
    `lax.map` in the c4 prefill twin, j-3c651507). The chunks after the
    first run as one `fori_loop`, so the kernel is emitted twice per call
    site (the first call creates the output) rather than once per chunk."""
    num_tokens = q.shape[0]
    call = functools.partial(
        _attention_call,
        sm_scale=sm_scale,
        block_q=block_q,
        pages_per_step=pages_per_step,
        interpret=interpret,
    )
    limit = msa_attention.MAX_TOKENS_PER_CALL
    step = min(num_tokens, max(block_q, limit - limit % block_q))
    num_chunks = -(-num_tokens // step)
    pad = num_chunks * step - num_tokens
    q = jnp.pad(q, ((0, pad), (0, 0), (0, 0))).transpose(1, 0, 2)
    selected = jnp.pad(selected, ((0, pad), (0, 0)), constant_values=-1)
    local_page = jnp.pad(local_page, (0, pad), constant_values=-1)[:, None]
    local_limit = jnp.pad(local_limit, (0, pad))[:, None]

    def chunk(c, out):
        offset = jnp.full((1,), c * (step // block_q), jnp.int32)
        sel_c = jax.lax.dynamic_slice_in_dim(selected, c * step, step)
        return call(
            q, kv_cache, sel_c, local_page, local_limit, page_table, head, offset, out
        )

    out = chunk(0, None)
    if num_chunks > 1:
        out = jax.lax.fori_loop(1, num_chunks, chunk, out)
    out, lse = out
    return out.transpose(1, 0, 2)[:num_tokens], lse[:, :, 0].T[:num_tokens]


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
    static_argnames=("sm_scale", "block_q", "pages_per_step", "head", "interpret"),
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
    check_page_dtype(kv_cache.shape[2], kv_cache.dtype)
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
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`msa_sparse_attention_lse` for every KV head of the cache rows at once.

    ``q [T, G * Hg, D]`` (KV head ``h`` owns query heads ``[h * Hg,
    (h + 1) * Hg)``), ``selected [T, G, K]``. The heads run as one `lax.map`
    over a single kernel whose KV head is an input, so the program carries
    one copy of the kernel instead of one per head. Returns ``(out
    [T, G * Hg, D], lse [T, G * Hg] f32)``.
    """
    num_tokens, num_q_heads, head_dim = q.shape
    num_heads = selected.shape[1]
    check_page_dtype(kv_cache.shape[2], kv_cache.dtype)
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
        )

    out, lse = jax.lax.map(
        one_head,
        (q_heads, selected.transpose(1, 0, 2), jnp.arange(num_heads, dtype=jnp.int32)),
    )  # [G, T, Hg, D], [G, T, Hg]
    out = out.transpose(1, 0, 2, 3).reshape(padded, num_q_heads, head_dim)
    lse = lse.transpose(1, 0, 2).reshape(padded, num_q_heads)
    return out[:num_tokens], lse[:num_tokens]
