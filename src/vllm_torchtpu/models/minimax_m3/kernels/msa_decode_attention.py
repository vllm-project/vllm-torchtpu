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
"""MSA sparse attention with log-sum-exp for decode batches (DCP attend).

`msa_attention_lse.msa_sparse_attention_lse_heads` for small token counts,
laid out for decode verify groups:

- One query tile per request's verify group (8 tokens): a tile's page union
  is one request's pages, not two requests' disjoint ones, so every score
  column belongs to the row's own request.
- Rows on lanes: ``Hg * block_q = 128`` (16 query heads x 8 tokens) score
  rows fill the lanes of the transposed score tile ``s^T [keys, rows]``. The
  softmax max / sum over keys are sublane reductions, ``m`` and ``l`` one
  vreg each, and the output accumulates as ``acc^T [D, rows]``. 4-bit pages
  take their per-key scales as sublane columns.
- Every KV head of the cache rows runs in one kernel call (the head is a
  function of the tile), so the call pays one pipeline fill, not one per head.
- Narrow step widths (1, 2, 4, 8 pages) below the full group: a step costs
  per score column, and a request's union at DCP 8 is a few pages.

The work list (`msa_attention._work_list`), page unpacking and masking rules
are `msa_attention_lse`'s.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention, msa_fp4

DECODE_BLOCK_Q = 8
PAGES_PER_STEP = 16
STEP_WIDTHS = (1, 2, 4, 8)
TILES_PER_STEP = 4
_FP4_PREFETCH = 2
_LOG2_E = math.log2(math.e)
_LN_2 = math.log(2.0)


def _kernel(
    # Scalar prefetch
    work_page_ref,  # tile page key unions (`msa_attention._work_list`)
    work_meta_ref,  # [groups] union row << _ROW_SHIFT | last and first bits
    work_count_ref,  # [groups] pages in the group (its first slots)
    block_start_ref,  # [steps + 1] first group of each grid step, group count
    page_table_ref,  # page key -> physical page
    # In
    q_ref,  # [Hg, tiles_per_step * Bq, D]
    selected_ref,  # [tiles_per_step * Bq, K] page keys, -1 padded
    local_page_ref,  # [tiles_per_step * Bq, 1]
    local_limit_ref,  # [tiles_per_step * Bq, 1]
    kv_hbm_ref,  # [pages, S / p, p, G * 2D] the whole cache, in HBM
    *refs,  # [scale pages (4-bit)], out, lse, m, l, acc, page buffers,
    # semaphores, [scale buffers, semaphores (4-bit)]
    sm_scale: float,
    fp4: bool,
    tiles_per_step: int,
    groups_per_tile: int,
    tiles_per_head: int,
    widths: tuple[int, ...],
    transposed: bool,
    scale_mode: str,
    translated: bool,
):
    tail = refs[len(refs) - (9 if fp4 else 7) :]
    o_ref, lse_ref, m_ref, l_ref, acc_ref, buf_ref, sem_ref = tail[:7]
    scale_hbm_ref = refs[0] if fp4 else None
    scale_buf_ref, scale_sem_ref = tail[7:] if fp4 else (None, None)
    pages_per_step = buf_ref.shape[1]
    ring = buf_ref.shape[0]
    num_groups = block_start_ref[pl.num_programs(0)]
    num_heads, rows_in_step, head_dim = q_ref.shape
    block_q = rows_in_step // tiles_per_step
    num_rows = num_heads * block_q
    rows_per_page, packing, row_width = buf_ref.shape[3:]
    page_size = rows_per_page * (msa_fp4.TOKENS_PER_WORD if fp4 else packing)

    def tile_of(meta):
        return msa_attention._meta_tile(meta, groups_per_tile)

    def page_copies(slot, g, page, head):
        src = kv_hbm_ref.at[pl.ds(page, 1), :, :, pl.ds(head * row_width, row_width)]
        copies = [pltpu.make_async_copy(src, buf_ref.at[slot, g], sem_ref.at[slot, g])]
        if fp4:
            copies.append(
                pltpu.make_async_copy(
                    scale_hbm_ref.at[pl.ds(page, 1)],
                    scale_buf_ref.at[slot, g],
                    scale_sem_ref.at[slot, g],
                )
            )
        return copies

    def start_group(group, slot):
        meta = work_meta_ref[group]
        base = msa_attention._slot(meta, pages_per_step, 0)
        head = tile_of(meta) // tiles_per_head

        def start(g, carry):
            page = work_page_ref[base + g]
            if not translated:
                page = page_table_ref[page]
            for copy in page_copies(slot, g, page, head):
                copy.start()
            return carry

        jax.lax.fori_loop(0, work_count_ref[group], start, 0)

    def group(w):
        meta = work_meta_ref[w]
        count = work_count_ref[w]
        tile = tile_of(meta)
        head = tile // tiles_per_head
        in_step = tile % tiles_per_step
        rows = pl.ds(pl.multiple_of(in_step * block_q, block_q), block_q)
        slot = w % ring
        first = (meta & msa_attention._FIRST_BIT) != 0
        last = (meta & msa_attention._LAST_BIT) != 0

        @pl.when(w == 0)
        def _first():
            for d in range(ring - 1):
                pl.when(d < num_groups)(functools.partial(start_group, d, d))

        @pl.when(w + ring - 1 < num_groups)
        def _next():
            start_group(w + ring - 1, (w + ring - 1) % ring)

        def wait(g, carry):
            for copy in page_copies(slot, g, 0, 0):  # the source only sizes the wait
                copy.wait()
            return carry

        jax.lax.fori_loop(0, count, wait, 0)

        @pl.when(first & jnp.logical_not(last))
        def _init():
            m_ref[...] = jnp.full_like(m_ref, msa_attention._MASK_VALUE)
            l_ref[...] = jnp.zeros_like(l_ref)
            acc_ref[...] = jnp.zeros_like(acc_ref)

        def write(out, lse):
            """``out`` ``[D, rows]`` (transposed) or ``[rows, D]``, ``lse``
            ``[1, rows]`` or ``[rows, 1]``."""
            if transposed:
                out = out.T
            else:
                lse = lse.T
            o_ref[:, rows, :] = out.reshape(num_heads, block_q, head_dim).astype(
                o_ref.dtype
            )
            lse_ref[:, pl.ds(in_step, 1)] = lse.reshape(1, 1, 1, num_rows)

        def finish(m, denom, acc):
            write(acc * (1.0 / jnp.maximum(denom, 1.0)), m * _LN_2 + jnp.log(denom))

        def scale_column(scale_ref, row):
            """Scale-page row `row` (run time) as an ``[S, 1]`` column."""
            if scale_mode == "small":
                page_t = scale_ref[0].T  # [S, R]
                lane = jax.lax.broadcasted_iota(jnp.int32, page_t.shape, 1)
                return jnp.sum(jnp.where(lane == row, page_t, 0.0), axis=1, keepdims=True)
            r = scale_ref[0, pl.ds(row, 1), :]  # [1, S]
            return jnp.broadcast_to(r, (r.shape[1], r.shape[1])).T[:, :1]

        def attend(width, single):
            """One softmax update over the group's first `width` pages. With
            `transposed`, on ``s^T [width * S, rows]`` (state on lanes),
            else on ``s [rows, width * S]`` (state on sublanes)."""
            row_slots = (
                jax.lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)
                if fp4 or packing == 4
                else msa_attention._bitcast_slots(rows_per_page, packing)
            )  # [S, 1]
            sel = selected_ref[rows, :]  # [Bq, K]
            local_page = local_page_ref[rows, :]  # [Bq, 1]
            local_limit = local_limit_ref[rows, :]
            if transposed:
                sel, local_page, local_limit = sel.T, local_page.T, local_limit.T
            red = 0 if transposed else 1  # the axis over keys / tokens' picks
            keys, values, masks, k_scales, v_scales = [], [], [], [], []
            for g in range(width):
                kv_ref = buf_ref.at[slot, g]
                page = work_page_ref[msa_attention._slot(meta, pages_per_step, g)]
                hit = jnp.max((sel == page).astype(jnp.int32), axis=red, keepdims=True)
                limit = jnp.where(local_page == page, local_limit, page_size)
                limit = jnp.where(page >= 0, limit * hit, 0)  # [1, Bq] / [Bq, 1]
                if fp4:
                    page_rows, sk, sv = msa_attention._fp4_page(
                        kv_ref, scale_buf_ref.at[slot, g], 2 * head, 2 * head + 1
                    )
                    if transposed:
                        sk = scale_column(scale_buf_ref.at[slot, g], 2 * head)
                        sv = scale_column(scale_buf_ref.at[slot, g], 2 * head + 1)
                        sk = jnp.where(jnp.isfinite(sk), sk, 0.0)
                        sv = jnp.where(jnp.isfinite(sv), sv, 0.0)
                    k_scales.append(sk)
                    v_scales.append(sv)
                    values.append(page_rows[:, head_dim:])
                elif packing == 4:
                    page_rows = kv_ref[0].reshape(page_size, -1).astype(jnp.bfloat16)
                    values.append(
                        jnp.where(row_slots < jnp.max(limit), page_rows[:, head_dim:], 0)
                    )
                else:
                    page_rows = msa_attention._page_rows(kv_ref, jnp.max(limit))
                    values.append(page_rows[:, head_dim:])
                keys.append(page_rows[:, :head_dim])
                # Rows are head-major (h * Bq + t): every head shares a token's mask.
                if transposed:
                    masks.append(row_slots < jnp.tile(limit, (1, num_heads)))
                else:
                    masks.append(row_slots.T < limit)  # [Bq, S]

            def cat(xs, axis):
                return jnp.concatenate(xs, axis=axis) if len(xs) > 1 else xs[0]

            k = cat(keys, 0)
            v = cat(values, 0)
            q = q_ref[:, rows, :].reshape(num_rows, head_dim)
            scale = sm_scale * _LOG2_E
            if not fp4:
                q = (q.astype(jnp.float32) * scale).astype(q.dtype)
            if transposed:
                mask = cat(masks, 0)
                s = jax.lax.dot_general(
                    k, q, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
                )  # [W * S, rows]
                if fp4:
                    s = s * (cat(k_scales, 0) * scale)
            else:
                mask = cat(masks, 1)
                mask = jnp.concatenate([mask] * num_heads, axis=0)
                s = jax.lax.dot_general(
                    q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
                )  # [rows, W * S]
                if fp4:
                    s = s * (cat(k_scales, 1) * scale)
            s = jnp.where(mask, s, -jnp.inf)
            m_prev = m_ref[...] if not single else msa_attention._MASK_VALUE
            m_new = jnp.maximum(m_prev, jnp.max(s, axis=red, keepdims=True))
            p = jnp.exp2(s - m_new)
            l_cur = jnp.sum(p, axis=red, keepdims=True)
            if fp4:
                p = p * cat(v_scales, 0 if transposed else 1)
            if transposed:
                pv = jax.lax.dot_general(
                    v, p.astype(v.dtype), (((0,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )  # [D, rows]
            else:
                pv = jax.lax.dot_general(
                    p.astype(v.dtype), v, (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )  # [rows, D]
            if single:
                finish(m_new, l_cur, pv)
                return
            alpha = jnp.exp2(m_prev - m_new)
            l_new = alpha * l_ref[...] + l_cur
            acc_new = alpha * acc_ref[...] + pv

            @pl.when(last)
            def _finalize():
                finish(m_new, l_new, acc_new)

            @pl.when(jnp.logical_not(last))
            def _carry():
                m_ref[...] = m_new
                l_ref[...] = l_new
                acc_ref[...] = acc_new

        single = first & last
        all_widths = [n for n in widths if n < pages_per_step] + [pages_per_step]
        for i, width in enumerate(all_widths):
            fits = count > (all_widths[i - 1] if i else 0)
            if width < pages_per_step:
                fits = fits & (count <= width)
            pl.when(fits & single)(functools.partial(attend, width, True))
            pl.when(fits & jnp.logical_not(single))(
                functools.partial(attend, width, False)
            )

        @pl.when(count == 0)
        def _empty():  # a tile of padding tokens: its only group holds no page
            shape = (head_dim, num_rows) if transposed else (num_rows, head_dim)
            write(
                jnp.zeros(shape, jnp.float32),
                jnp.full((1, num_rows) if transposed else (num_rows, 1), -jnp.inf),
            )

    def loop(w, carry):
        group(w)
        return carry

    b = pl.program_id(0)
    jax.lax.fori_loop(block_start_ref[b], block_start_ref[b + 1], loop, 0)


def _tiles_per_step(num_tiles: int, cap: int) -> int:
    return next(n for n in range(cap, 0, -1) if num_tiles % n == 0)


def _work(selected, block_q, pages_per_step, tiles_per_step, dense):
    """`msa_attention._work_list` over every head's tiles ``selected [G * T, K]``
    (tile ``h * T / Bq + i`` is head ``h``'s tile ``i``), packed as one int32
    row ``[unions | meta | counts | step starts]``."""
    pages, meta, end, counts = msa_attention._work_list(
        selected, block_q=block_q, pages_per_step=pages_per_step, dense=dense
    )
    num_tiles = selected.shape[0] // block_q
    groups_per_tile = msa_attention._groups_per_tile(
        block_q, selected.shape[1], pages_per_step
    )
    live = jnp.arange(meta.shape[0], dtype=jnp.int32) < end
    tile = jnp.where(live, msa_attention._meta_tile(meta, groups_per_tile), num_tiles)
    bounds = jnp.arange(0, num_tiles + 1, tiles_per_step, dtype=jnp.int32)
    starts = jnp.sum(tile[None, :] < bounds[:, None], axis=1, dtype=jnp.int32)
    return jnp.concatenate([pages, meta, counts, starts])


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "block_q",
        "pages_per_step",
        "widths",
        "tiles_per_step",
        "transposed",
        "scale_mode",
        "dense",
        "translate",
        "interpret",
    ),
)
def msa_decode_attention_lse(
    q: jax.Array,
    kv_cache: jax.Array,
    selected: jax.Array,
    local_page: jax.Array,
    local_limit: jax.Array,
    page_table: jax.Array,
    *,
    sm_scale: float,
    scale_cache: jax.Array | None = None,
    block_q: int = DECODE_BLOCK_Q,
    pages_per_step: int = PAGES_PER_STEP,
    widths: tuple[int, ...] = STEP_WIDTHS,
    tiles_per_step: int = TILES_PER_STEP,
    transposed: bool = False,
    scale_mode: str = "small",
    dense: bool | None = None,
    translate: bool = True,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`msa_attention_lse.msa_sparse_attention_lse_heads(..., head_major=True)`
    for decode batches: ``q [T, G * Hg, D]``, ``selected [T, G, K]`` page keys
    (``-1`` padded), ``local_page`` / ``local_limit [T]``, ``page_table``
    the flat key -> page table. Returns head-major ``(out [G * Hg, T, D],
    lse [G * Hg, T] f32)``; a token with no visible slot gets 0 and -inf.

    A 4-bit cache takes its `scale_cache` (KV head ``h``'s K / V scales in
    scale-page rows ``2h`` / ``2h + 1``). With `translate`, keys become
    physical pages in XLA before the work list; else the kernel looks them
    up in the scalar-prefetched `page_table`."""
    num_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads, topk = selected.shape[1:]
    num_heads = num_q_heads // num_kv_heads
    fp4 = scale_cache is not None
    msa_attention.check_cache(kv_cache, scale_cache)
    if kv_cache.shape[3] < 2 * head_dim * num_kv_heads:
        raise ValueError(f"cache rows {kv_cache.shape} miss {num_kv_heads} KV heads")
    if translate:
        # Page keys -> physical pages here, one small gather, so the kernel
        # needs no scalar-prefetched page table (max_num_seqs x pages per
        # sequence words copied to SMEM before every call).
        selected = jnp.where(
            selected >= 0, jnp.take(page_table, jnp.maximum(selected, 0)), -1
        )
        local_page = jnp.where(
            local_page >= 0, jnp.take(page_table, jnp.maximum(local_page, 0)), -1
        )
        page_table = jnp.zeros((1,), jnp.int32)
    pad = -num_tokens % block_q
    padded = num_tokens + pad
    q = jnp.pad(q, ((0, pad), (0, 0), (0, 0))).transpose(1, 0, 2)  # [G*Hg, Tp, D]
    sel = jnp.pad(selected, ((0, pad), (0, 0), (0, 0)), constant_values=-1)
    sel = sel.transpose(1, 0, 2)  # [G, Tp, K]
    local_page = jnp.pad(local_page, (0, pad), constant_values=-1)[:, None]
    local_limit = jnp.pad(local_limit, (0, pad))[:, None]
    tiles_per_head = padded // block_q
    tps = _tiles_per_step(tiles_per_head, tiles_per_step)
    steps_per_head = tiles_per_head // tps
    num_tiles = num_kv_heads * tiles_per_head
    num_steps = num_tiles // tps
    groups_per_tile = msa_attention._groups_per_tile(block_q, topk, pages_per_step)
    max_groups = num_tiles * groups_per_tile
    union_len = max_groups * pages_per_step
    packed = _work(
        sel.reshape(num_tiles * block_q, topk), block_q, pages_per_step, tps, dense
    )

    def kernel(packed_ref, table_ref, *refs, **kwargs):
        return _kernel(
            packed_ref.at[pl.ds(0, union_len)],
            packed_ref.at[pl.ds(union_len, max_groups)],
            packed_ref.at[pl.ds(union_len + max_groups, max_groups)],
            packed_ref.at[pl.ds(union_len + 2 * max_groups, num_steps + 1)],
            table_ref,
            *refs,
            **kwargs,
        )

    def head_block(b, *_):
        return (b // steps_per_head, b % steps_per_head, 0)

    def token_block(b, *_):
        return (b % steps_per_head, 0)

    def lse_block(b, *_):
        return (b // steps_per_head, b % steps_per_head, 0, 0)

    _, rows_per_page, packing, _ = kv_cache.shape
    row_width = 2 * head_dim
    ring = 1 + (_FP4_PREFETCH if fp4 else 1)
    scale_inputs = (scale_cache,) if fp4 else ()
    scale_scratch = (
        (
            pltpu.VMEM((ring, pages_per_step, 1) + scale_cache.shape[1:], jnp.float32),
            pltpu.SemaphoreType.DMA((ring, pages_per_step)),
        )
        if fp4
        else ()
    )
    num_rows = num_heads * block_q
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(num_steps,),
        in_specs=[
            pl.BlockSpec((num_heads, tps * block_q, head_dim), head_block),
            pl.BlockSpec((None, tps * block_q, topk), head_block),
            pl.BlockSpec((tps * block_q, 1), token_block),
            pl.BlockSpec((tps * block_q, 1), token_block),
            pl.BlockSpec(memory_space=pl.ANY),
            *(pl.BlockSpec(memory_space=pl.ANY) for _ in scale_inputs),
        ],
        out_specs=[
            pl.BlockSpec((num_heads, tps * block_q, head_dim), head_block),
            pl.BlockSpec((1, tps, 1, num_rows), lse_block),
        ],
        scratch_shapes=[
            # m, l, acc: rows on lanes (transposed) or sublanes.
            *(
                [
                    pltpu.VMEM((1, num_rows), jnp.float32),
                    pltpu.VMEM((1, num_rows), jnp.float32),
                    pltpu.VMEM((head_dim, num_rows), jnp.float32),
                ]
                if transposed
                else [
                    pltpu.VMEM((num_rows, 1), jnp.float32),
                    pltpu.VMEM((num_rows, 1), jnp.float32),
                    pltpu.VMEM((num_rows, head_dim), jnp.float32),
                ]
            ),
            pltpu.VMEM(
                (ring, pages_per_step, 1, rows_per_page, packing, row_width),
                kv_cache.dtype,
            ),
            pltpu.SemaphoreType.DMA((ring, pages_per_step)),
            *scale_scratch,
        ],
    )
    out, lse = pl.pallas_call(
        functools.partial(
            kernel,
            sm_scale=sm_scale,
            fp4=fp4,
            tiles_per_step=tps,
            groups_per_tile=groups_per_tile,
            tiles_per_head=tiles_per_head,
            widths=tuple(widths),
            transposed=transposed,
            scale_mode=scale_mode,
            translated=translate,
        ),
        grid_spec=grid_spec,
        out_shape=[
            jax.ShapeDtypeStruct((num_q_heads, padded, head_dim), q.dtype),
            jax.ShapeDtypeStruct(
                (num_kv_heads, tiles_per_head, 1, num_rows), jnp.float32
            ),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=msa_attention._VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
            disable_semaphore_checks=True,
        ),
        interpret=interpret,
        name="minimax_m3_msa_decode_attention_lse",
    )(
        packed,
        page_table,
        q,
        sel,
        local_page,
        local_limit,
        kv_cache,
        *scale_inputs,
    )
    # lse [G, tiles, 1, Hg * Bq] (h * Bq + t) -> [G * Hg, Tp]
    lse = lse.reshape(num_kv_heads, tiles_per_head, num_heads, block_q)
    lse = lse.transpose(0, 2, 1, 3).reshape(num_q_heads, padded)
    return out[:, :num_tokens], lse[:, :num_tokens]
