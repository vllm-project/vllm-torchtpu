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
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3.kernels import msa_attention, msa_fp4

logger = init_logger(__name__)

# Page counts a kernel step may run at besides the full group (`_kernel`).
_STEP_WIDTHS = (4, 8)
# With TPU_MINIMAX_M3_MSA_PREFILL_WIDTH12=1, prefill-sized calls
# (`msa_dcp.msa_attend_dcp`) also run 9-12-page groups at a 12-page width
# instead of the full 16: a quarter fewer score columns for those groups.
PREFILL_STEP_WIDTHS = (
    (4, 8, 12)
    if os.environ.get("TPU_MINIMAX_M3_MSA_PREFILL_WIDTH12", "0") == "1"
    else _STEP_WIDTHS
)
# With TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE=<slots> (0 = off), prefill-sized
# calls on 4-bit caches copy each distinct page of a call once into a VMEM
# cache of that many slots (~20 KB each) instead of once per tile holding it
# (`_page_cache`). A call with more distinct pages, or a tile union over
# `PAGE_CACHE_TILE_PAGES`, streams per group as before, in the same kernel.
PREFILL_PAGE_CACHE_SLOTS = int(
    os.environ.get("TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE", "0") or 0
)
PAGE_CACHE_TILE_PAGES = 32
# With TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_BF16=1 (and a page cache), the
# cached 4-bit pages are unpacked to bf16 once, after the cache fill, into a
# second VMEM cache (64 KB a slot), so a group reads bf16 rows instead of
# unpacking each page again for every tile that holds it (~17 tiles a page).
PREFILL_PAGE_CACHE_BF16 = (
    os.environ.get("TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_BF16", "0") == "1"
)
# With TPU_MINIMAX_M3_MSA_PREFILL_ROW_SPLIT=<n> (1 = off), prefill-sized calls
# run each group's scores, softmax and p @ v in `n` independent parts of the
# tile's query heads, so one part's softmax can be scheduled beside the next
# part's QK matmul. Rows are independent: the results are bitwise the same.
PREFILL_ROW_SPLIT = int(os.environ.get("TPU_MINIMAX_M3_MSA_PREFILL_ROW_SPLIT", "1") or 1)
if PREFILL_PAGE_CACHE_SLOTS:
    logger.info(
        "TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE=%d: 4-bit MSA prefill attention "
        "page cache of %d slots.",
        PREFILL_PAGE_CACHE_SLOTS,
        PREFILL_PAGE_CACHE_SLOTS,
    )
if PREFILL_PAGE_CACHE_BF16:
    logger.info(
        "TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_BF16=1: MSA prefill attention page "
        "cache unpacked to bf16 once per call."
    )
if PREFILL_ROW_SPLIT != 1:
    logger.info(
        "TPU_MINIMAX_M3_MSA_PREFILL_ROW_SPLIT=%d: MSA prefill attention groups run "
        "in %d parts of the query heads.",
        PREFILL_ROW_SPLIT,
        PREFILL_ROW_SPLIT,
    )
if PREFILL_STEP_WIDTHS != _STEP_WIDTHS:
    logger.info(
        "TPU_MINIMAX_M3_MSA_PREFILL_WIDTH12=1: MSA prefill attention step widths %s.",
        PREFILL_STEP_WIDTHS,
    )
# Tiles per grid step: a step runs the groups of this many tiles in a loop.
# One step per group paid a per-step cost worth ~2-3% of the call (pool48
# selections, 1-2 groups a tile; bench j-caec1354).
TILES_PER_STEP = 4
# Groups whose pages are in flight ahead of the running one: 4-bit groups
# (a page and a scale page each) gain from 2 (bench j-caec1354: c64 shared
# 2206 -> 2026, forced 2150 -> 2075 us per call), 8/16-bit ones do not.
_FP4_PREFETCH = 2
_LOG2_E = math.log2(math.e)
_LN_2 = math.log(2.0)


def _fp4_rows(kv_ref) -> jax.Array:
    """Unscaled bf16 rows ``[S, 2D]`` of a 4-bit page, as `msa_attention._fp4_page`."""
    words = kv_ref.bitcast(jnp.uint32)[0, :, 0, :]
    return pltpu.bitcast(words, jnp.int4).astype(jnp.bfloat16)


def _fp4_scales(scale_ref, k_row, v_row):
    """K / V scales of a page as `msa_attention._fp4_page` reads them."""
    if isinstance(k_row, int):
        page = scale_ref[0]
        sk, sv = page[k_row : k_row + 1], page[v_row : v_row + 1]
    else:
        sk = scale_ref[0, pl.ds(k_row, 1), :]
        sv = scale_ref[0, pl.ds(v_row, 1), :]
    return jnp.where(jnp.isfinite(sk), sk, 0.0), jnp.where(jnp.isfinite(sv), sv, 0.0)


def _page_cache(pages: jax.Array, num_tiles: int, cache_size: int) -> jax.Array:
    """A call's unions (`msa_attention._work_list`: `num_tiles` rows, ``-1``
    padded) -> ``[unions' cache slots | distinct keys (cache_size) | count |
    fits]``: each union entry's slot among the call's distinct page keys
    (``-1`` for padding; the key itself when they do not all fit), the first
    `cache_size` distinct keys ascending, their count, and whether they all
    fit. Only the first `PAGE_CACHE_TILE_PAGES` of each union are sorted (a
    call with a wider union streams). Sorts only, no gather or scatter."""
    rows = pages.reshape(num_tiles, -1)
    width = min(PAGE_CACHE_TILE_PAGES, rows.shape[1])
    head = rows[:, :width].reshape(-1)
    key = jnp.where(head < 0, msa_attention._NO_PAGE, head)
    position = jnp.arange(key.shape[0], dtype=jnp.int32)
    key, position = jax.lax.sort((key, position), num_keys=1)
    new = jnp.concatenate([key[:1] != msa_attention._NO_PAGE, key[1:] != key[:-1]])
    new = new & (key != msa_attention._NO_PAGE)
    slot = jnp.cumsum(new, dtype=jnp.int32) - 1
    num_distinct = slot[-1] + 1
    distinct = jnp.sort(jnp.where(new, key, msa_attention._NO_PAGE))[:cache_size]
    distinct = jnp.pad(
        distinct,
        (0, cache_size - distinct.shape[0]),
        constant_values=msa_attention._NO_PAGE,
    )
    _, slot = jax.lax.sort((position, slot), num_keys=1)
    slot = jnp.where(head < 0, -1, slot).reshape(num_tiles, width)
    slot = jnp.pad(slot, ((0, 0), (0, rows.shape[1] - width)), constant_values=-1)
    fits = (num_distinct <= cache_size) & jnp.all(rows[:, width:] < 0)
    return jnp.concatenate(
        [
            jnp.where(fits, slot.reshape(-1), pages),
            jnp.where(distinct == msa_attention._NO_PAGE, 0, distinct),
            jnp.minimum(num_distinct, cache_size)[None],
            fits.astype(jnp.int32)[None],
        ]
    ).astype(jnp.int32)


def _kernel(
    # Scalar prefetch
    work_page_ref,  # tile page key unions (`msa_attention._work_list`)
    work_meta_ref,  # [groups] union row << _ROW_SHIFT | last and first bits
    work_count_ref,  # [groups] pages in the group (its first slots)
    page_table_ref,  # page key -> physical page
    head_ref,  # [2] KV head of the cache rows, block of query heads
    tile_offset_ref,  # [1] the call's first tile (read by the index maps only)
    # In
    q_ref,  # [H, Bq, D]
    selected_ref,  # [Bq, K] page keys, -1 padded
    local_page_ref,  # [Bq, 1]
    local_limit_ref,  # [Bq, 1] visible slots of the local page
    kv_hbm_ref,  # [pages, S / p, p, G * 2D] the whole cache, in HBM
    *refs,  # [scale pages (4-bit)], [aliased out, lse], out, lse, m, l, acc,
    # page buffers, semaphores, [scale buffers, semaphores (4-bit)]
    sm_scale: float,
    aliased: bool = False,
    fp4_rows: tuple[int, int] | str | None = None,
    block_start_ref=None,
    tiles_per_step: int = 1,
    groups_per_tile: int = 1,
    transposed: bool = False,
    step_widths: tuple[int, ...] = _STEP_WIDTHS,
    cache_ref=None,
    pages_per_step: int = msa_attention.DEFAULT_PAGES_PER_STEP,
    bf16_cache: bool = False,
    row_split: int = 1,
):
    """`msa_attention._kernel_v2` that also writes ``lse = m + log(l)``
    (``-inf`` for a row that sees no slot; its output is 0). Scores, the
    running max ``m`` and ``l`` are in log2 units (scaled by ``sm_scale *
    log2(e)``): exp2 needs no multiply per score. 4-bit caches fold the scale
    into the K scales of the score columns; others into the queries, which
    then round to bf16 once more before the QK dot (an XLA multiply of the
    whole head-major query outside the kernel was not fused, ~40 us per layer
    in the c4 twin, j-18e7ec75).

    Each group copies a later group's pages into a free slot of the page
    ring itself, one loop iteration per page it holds: unrolled
    per-slot copies run predicated, so every step paid all slots' address
    arithmetic and checks (~800 scalar bundles of a ~2,200-bundle 4-page
    step in the LLO).

    A tile of one group (its first and last) runs a plain softmax that
    keeps nothing in the m / l / acc scratch, and a tile with no page only
    writes its zero output.

    A 4-bit cache (`fp4_rows`: static K / V scale-page rows, or ``"head"``
    for rows ``2k, 2k + 1`` of the cache-row KV head ``k``) copies each
    page's scale page beside it.

    Grid step ``b`` runs the groups ``block_start_ref[b] ..
    block_start_ref[b + 1] - 1`` of its `tiles_per_step` tiles in a loop
    (the last entry is the call's group count); the query, selection and
    output blocks hold those tiles."""
    fp4 = fp4_rows is not None
    # `bf16_cache`: the last scratch is the cache's pages unpacked to bf16,
    # [cache_size, S, 2D]; filled after the cache, read when it is in use.
    bfc_ref = refs[-1] if bf16_cache else None
    if bf16_cache:
        refs = refs[:-1]
    tail = refs[len(refs) - (9 if fp4 else 7) :]
    o_ref, lse_ref, m_ref, l_ref, acc_ref, buf_ref, sem_ref = tail[:7]
    scale_hbm_ref = refs[0] if fp4 else None
    scale_buf_ref, scale_sem_ref = tail[7:] if fp4 else (None, None)
    # `cache_ref` (4-bit prefill, `PREFILL_PAGE_CACHE_SLOTS`): ``[distinct page
    # keys (cache_size) | count | fits]``; the unions then hold cache slots.
    cached = cache_ref is not None
    ring = sem_ref.shape[0] - (1 if cached else 0)  # prefetched groups + running
    if cached:
        cache_size = cache_ref.shape[0] - 2
        num_cached = cache_ref[cache_size]
        use_cache = cache_ref[cache_size + 1] != 0
    num_groups = block_start_ref[pl.num_programs(0)]

    def group(
        w,
        q_ref=q_ref,
        o_ref=o_ref,
        lse_ref=lse_ref,
        selected_ref=selected_ref,
        local_page_ref=local_page_ref,
        local_limit_ref=local_limit_ref,
    ):
        if tiles_per_step > 1:
            tile = msa_attention._meta_tile(work_meta_ref[w], groups_per_tile)
            bq = q_ref.shape[1] // tiles_per_step
            rows = pl.ds(pl.multiple_of(tile % tiles_per_step * bq, bq), bq)
            q_ref = q_ref.at[:, rows, :]
            o_ref = o_ref.at[:, rows, :]
            selected_ref = selected_ref.at[rows, :]
            local_page_ref = local_page_ref.at[rows, :]
            local_limit_ref = local_limit_ref.at[rows, :]
            lse_ref = lse_ref.at[:, pl.ds(tile % tiles_per_step, 1)]
        meta = work_meta_ref[w]
        count = work_count_ref[w]
        num_heads, block_q, head_dim = q_ref.shape
        rows_per_page, packing, row_width = buf_ref.shape[-3:]
        page_size = rows_per_page * (msa_fp4.TOKENS_PER_WORD if fp4 else packing)
        half = w % ring
        first = (meta & msa_attention._FIRST_BIT) != 0
        last = (meta & msa_attention._LAST_BIT) != 0

        def stream_at(half, g):
            return (half * pages_per_step + g,) if cached else (half, g)

        def page_copies(at, sem_at, page):
            src = kv_hbm_ref.at[
                pl.ds(page, 1), :, :, pl.ds(head_ref[0] * row_width, row_width)
            ]
            copies = [pltpu.make_async_copy(src, buf_ref.at[at], sem_ref.at[sem_at])]
            if fp4:
                copies.append(
                    pltpu.make_async_copy(
                        scale_hbm_ref.at[pl.ds(page, 1)],
                        scale_buf_ref.at[at],
                        scale_sem_ref.at[sem_at],
                    )
                )
            return copies

        def stream():
            def start_group(group, half):
                base = msa_attention._slot(work_meta_ref[group], pages_per_step, 0)

                def start(g, carry):
                    page = page_table_ref[work_page_ref[base + g]]
                    for copy in page_copies(stream_at(half, g), (half, g), page):
                        copy.start()
                    return carry

                jax.lax.fori_loop(0, work_count_ref[group], start, 0)

            @pl.when(w == 0)
            def _first():
                for d in range(ring - 1):
                    pl.when(d < num_groups)(functools.partial(start_group, d, d))

            @pl.when(w + ring - 1 < num_groups)
            def _next():
                start_group(w + ring - 1, (w + ring - 1) % ring)

            def wait(g, carry):
                # The source only sizes the wait.
                for copy in page_copies(stream_at(half, g), (half, g), 0):
                    copy.wait()
                return carry

            jax.lax.fori_loop(0, count, wait, 0)

        if not cached:
            stream()

            def group_page(g):
                """Key, page ref and scale page ref of the group's slot `g`."""
                page = work_page_ref[msa_attention._slot(meta, pages_per_step, g)]
                scale_ref = scale_buf_ref.at[half, g] if fp4 else None
                return page, buf_ref.at[half, g], scale_ref

        else:
            # Each distinct page of the call is copied once, at its first
            # group, into the cache instead of once per tile that holds it.
            pl.when(jnp.logical_not(use_cache))(stream)
            cache_sem = (ring, 0)  # past the streaming semaphores

            @pl.when(use_cache & (w == 0))
            def _fill():
                def start(i, carry):
                    for copy in page_copies((i,), cache_sem, page_table_ref[cache_ref[i]]):
                        copy.start()
                    return carry

                def wait_one(i, carry):
                    for copy in page_copies((i,), cache_sem, 0):
                        copy.wait()
                    return carry

                jax.lax.fori_loop(0, num_cached, start, 0)
                jax.lax.fori_loop(0, num_cached, wait_one, 0)
                if bf16_cache:

                    def unpack(i, carry):
                        bfc_ref[i] = _fp4_rows(buf_ref.at[i])
                        return carry

                    jax.lax.fori_loop(0, num_cached, unpack, 0)

            def group_page(g):
                """Key, page ref and scale page ref of the group's slot `g`."""
                entry = work_page_ref[msa_attention._slot(meta, pages_per_step, g)]
                slot = jnp.maximum(entry, 0)
                key = jnp.where(entry >= 0, cache_ref[slot], -1)
                page = jnp.where(use_cache, key, entry)
                buffer = jnp.where(use_cache, slot, half * pages_per_step + g)
                scale_ref = scale_buf_ref.at[buffer] if fp4 else None
                return page, buf_ref.at[buffer], scale_ref

        @pl.when(first & jnp.logical_not(last))
        def _init():
            m_ref[...] = jnp.full_like(m_ref, msa_attention._MASK_VALUE)
            l_ref[...] = jnp.zeros_like(l_ref)
            acc_ref[...] = jnp.zeros_like(acc_ref)

        def write(out, lse, part=0, parts=1):
            """Rows of query-head part `part` of `parts` (all heads by default)."""
            if parts == 1:
                o_ref[...] = out.reshape(num_heads, block_q, head_dim).astype(o_ref.dtype)
                lse_ref[...] = lse.T.reshape(lse_ref.shape)  # lane-dense row
                return
            n = num_heads // parts
            o_ref[part * n : (part + 1) * n] = out.reshape(n, block_q, head_dim).astype(
                o_ref.dtype
            )
            lanes = n * block_q
            lse_ref[..., pl.ds(part * lanes, lanes)] = lse.T.reshape(
                lse_ref.shape[:-1] + (lanes,)
            )

        def finish(m, denom, acc, part=0, parts=1):
            """Output and lse from a tile's softmax state (log2 units, ``m``
            finite). A row that saw no slot has ``denom = 0`` and ``acc = 0``:
            its output is 0 and its lse ``m + log(0) = -inf`` with no select (a
            seen row has ``denom >= 1``)."""
            write(
                acc * (1.0 / jnp.maximum(denom, 1.0)),
                m * _LN_2 + jnp.log(denom),
                part,
                parts,
            )

        def attend_t(width, single):
            """`attend` for 8/16-bit caches on transposed scores ``s^T [keys,
            rows]``: the max and sum over keys are sublane reductions, the
            output accumulates as ``acc^T [D, rows]`` across the p @ v
            contraction, and ``m``, ``l`` are lane vectors over rows (fewer
            spills and pops: bench j-9081937a, c4 pool48 -5.4%). 4-bit pages
            keep `attend`, whose per-key scales are lane vectors there, and
            so do 8-token (decode) tiles: c4 b3 decode +0.4 ms per step
            transposed (twin j-fa6121d0 vs glue-only j-2abe5011)."""
            row_slots = (
                jax.lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)
                if packing == 4
                else msa_attention._bitcast_slots(rows_per_page, packing)
            )  # [S, 1]
            sel_t = selected_ref[...].T  # [K, Bq]
            local_page = local_page_ref[...].T  # [1, Bq]
            local_limit = local_limit_ref[...].T
            keys, values, masks = [], [], []
            for g in range(width):
                page, kv_ref, scale_ref = group_page(g)
                hit = jnp.max((sel_t == page).astype(jnp.int32), axis=0, keepdims=True)
                limit = jnp.where(local_page == page, local_limit, page_size)
                limit = jnp.where(page >= 0, limit * hit, 0)  # [1, Bq]
                seen = jnp.max(limit)
                if packing == 4:
                    rows = kv_ref[0].reshape(page_size, -1).astype(jnp.bfloat16)
                    values.append(jnp.where(row_slots < seen, rows[:, head_dim:], 0))
                else:
                    rows = msa_attention._page_rows(kv_ref, seen)
                    values.append(rows[:, head_dim:])
                keys.append(rows[:, :head_dim])
                # Rows are head-major (h * Bq + t): every head shares a token's mask.
                masks.append(row_slots < jnp.tile(limit, (1, num_heads)))  # [S, rows]
            k = jnp.concatenate(keys, axis=0)
            v = jnp.concatenate(values, axis=0)
            mask = jnp.concatenate(masks, axis=0)
            q = q_ref[...].reshape(num_heads * block_q, head_dim)
            q = (q.astype(jnp.float32) * (sm_scale * _LOG2_E)).astype(q.dtype)
            s = jax.lax.dot_general(
                k, q, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )  # [W * S, rows]
            s = jnp.where(mask, s, -jnp.inf)
            m_prev = m_ref[...] if not single else msa_attention._MASK_VALUE
            m_new = jnp.maximum(m_prev, jnp.max(s, axis=0, keepdims=True))
            p = jnp.exp2(s - m_new)
            l_cur = jnp.sum(p, axis=0, keepdims=True)
            pv = jax.lax.dot_general(
                v,
                p.astype(v.dtype),
                (((0,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )  # [D, rows]
            if single:
                finish_t(m_new, l_cur, pv)
                return
            alpha = jnp.exp2(m_prev - m_new)
            l_new = alpha * l_ref[...] + l_cur
            acc_new = alpha * acc_ref[...] + pv

            @pl.when(last)
            def _finalize():
                finish_t(m_new, l_new, acc_new)

            @pl.when(jnp.logical_not(last))
            def _carry():
                m_ref[...] = m_new
                l_ref[...] = l_new
                acc_ref[...] = acc_new

        def finish_t(m, denom, acc_t):
            """`finish` for `attend_t`'s lane-vector state."""
            out = (acc_t * (1.0 / jnp.maximum(denom, 1.0))).T
            o_ref[...] = out.reshape(num_heads, block_q, head_dim).astype(o_ref.dtype)
            lse_ref[...] = (m * _LN_2 + jnp.log(denom)).reshape(lse_ref.shape)

        def attend(width, single, from_bf16=False):
            """One softmax update over the group's first `width` slots; a
            `single` group (the tile's only one) needs no carried state.
            `from_bf16`: 4-bit pages read from the unpacked bf16 cache.
            The tile's query heads run in `row_split` independent parts."""
            # int4 and fp8 pages unpack in slot order; bf16 in the bitcast's order.
            row_slots = (
                jax.lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)
                if fp4 or packing == 4
                else msa_attention._bitcast_slots(rows_per_page, packing)
            )  # [S, 1]
            slots = row_slots.T  # [1, S]
            selected = selected_ref[...]
            local_page = local_page_ref[...]
            local_limit = local_limit_ref[...]
            keys, values, masks, k_scales, v_scales = [], [], [], [], []
            for g in range(width):
                if from_bf16:
                    entry = work_page_ref[msa_attention._slot(meta, pages_per_step, g)]
                    slot = jnp.maximum(entry, 0)
                    page = jnp.where(entry >= 0, cache_ref[slot], -1)
                    kv_ref, scale_ref = None, scale_buf_ref.at[slot]
                else:
                    page, kv_ref, scale_ref = group_page(g)
                is_selected = jnp.max(
                    (selected == page).astype(jnp.int32), axis=1, keepdims=True
                )
                limit = jnp.where(local_page == page, local_limit, page_size)
                limit = jnp.where(page >= 0, limit * is_selected, 0)  # [Bq, 1]
                seen = jnp.max(limit)
                if fp4:
                    # Stale or unwritten slots hold finite int4 values and
                    # scales that `_fp4_page` reads as 0 when non-finite, so
                    # their p = 0 contributes nothing.
                    if fp4_rows == "head":
                        k_row = 2 * head_ref[0]
                        v_row = k_row + 1
                    else:
                        k_row, v_row = fp4_rows
                    if from_bf16:
                        rows = bfc_ref[slot]
                        sk, sv = _fp4_scales(scale_ref, k_row, v_row)
                    else:
                        rows, sk, sv = msa_attention._fp4_page(
                            kv_ref, scale_ref, k_row, v_row
                        )
                    values.append(rows[:, head_dim:])
                    k_scales.append(sk)
                    v_scales.append(sv)
                elif packing == 4:
                    # The packed [S / 4, 4, 2D] page read as [S, 2D] rows: slot
                    # r * 4 + j at row r * 4 + j, and a cheaper relayout than the
                    # 32-bit words' bitcast (16 vs 30 bundles a page, AOT).
                    rows = kv_ref[0].reshape(page_size, -1).astype(jnp.bfloat16)
                    values.append(jnp.where(row_slots < seen, rows[:, head_dim:], 0))
                else:
                    rows = msa_attention._page_rows(kv_ref, seen)
                    values.append(rows[:, head_dim:])
                keys.append(rows[:, :head_dim])
                masks.append(slots < limit)  # [Bq, S]
            k = jnp.concatenate(keys, axis=0)
            v = jnp.concatenate(values, axis=0)
            token_mask = jnp.concatenate(masks, axis=1)
            scale = sm_scale * _LOG2_E
            if fp4:
                k_scale = jnp.concatenate(k_scales, axis=1) * scale
                v_scale = jnp.concatenate(v_scales, axis=1)
            part_heads = num_heads // row_split
            part_rows = part_heads * block_q

            for part in range(row_split):
                heads = slice(part * part_heads, (part + 1) * part_heads)
                rows_at = pl.ds(part * part_rows, part_rows)
                mask = jnp.concatenate([token_mask] * part_heads, axis=0)
                q = q_ref[heads].reshape(part_rows, head_dim)
                if not fp4:
                    q = (q.astype(jnp.float32) * scale).astype(q.dtype)
                s = jax.lax.dot_general(
                    q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
                )
                if fp4:  # x = int4 * scale: K scales the score columns, V scales p
                    s = s * k_scale
                s = jnp.where(mask, s, -jnp.inf)

                # The running max starts at a finite floor (`_MASK_VALUE`), so a
                # row with no visible slot gets p = exp2(-inf) = 0, not exp2(nan).
                m_prev = (
                    m_ref[rows_at] if not single else msa_attention._MASK_VALUE
                )
                m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
                p = jnp.exp2(s - m_new)
                l_cur = jnp.sum(p, axis=1, keepdims=True)
                p_v = p * v_scale if fp4 else p
                pv = jax.lax.dot_general(
                    p_v.astype(v.dtype),
                    v,
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                if single:
                    finish(m_new, l_cur, pv, part, row_split)
                    continue
                alpha = jnp.exp2(m_prev - m_new)
                l_new = alpha * l_ref[rows_at] + l_cur
                acc_new = alpha * acc_ref[rows_at] + pv

                pl.when(last)(
                    functools.partial(
                        finish, m_new, l_new, acc_new, part, row_split
                    )
                )

                def _carry(rows_at=rows_at, m_new=m_new, l_new=l_new, acc_new=acc_new):
                    m_ref[rows_at] = m_new
                    l_ref[rows_at] = l_new
                    acc_ref[rows_at] = acc_new

                pl.when(jnp.logical_not(last))(_carry)

        # Slots past the group's pages hold stale bytes: their limits are 0, so
        # their scores are masked and their value rows zeroed.
        # Run at the narrowest width holding the group's pages. A step costs per
        # score column, filled or not (~2.3 us at 16 pages, ~1.1 us at 4, twin
        # standalone j-952e8a64), and a 16-token tile's union is ~8-12 of its
        # rank's pages at DCP 2, ~2-8 at DCP 8.
        single = first & last
        widths = [n for n in step_widths if n < pages_per_step] + [pages_per_step]
        for i, width in enumerate(widths):
            fits = count > (widths[i - 1] if i else 0)
            if width < pages_per_step:
                fits = fits & (count <= width)
            body = attend_t if transposed else attend
            if bf16_cache:  # the cache in use reads bf16 pages, a streaming call unpacks
                for from_bf16, when in ((True, use_cache), (False, jnp.logical_not(use_cache))):
                    pl.when(fits & single & when)(
                        functools.partial(attend, width, True, from_bf16)
                    )
                    pl.when(fits & jnp.logical_not(single) & when)(
                        functools.partial(attend, width, False, from_bf16)
                    )
                continue
            pl.when(fits & single)(functools.partial(body, width, True))
            pl.when(fits & jnp.logical_not(single))(
                functools.partial(body, width, False)
            )

        @pl.when(count == 0)
        def _empty():  # a tile of padding tokens: its only group holds no page
            write(
                jnp.zeros((num_heads * block_q, head_dim), jnp.float32),
                jnp.full((num_heads * block_q, 1), -jnp.inf, jnp.float32),
            )

    def loop(w, carry):
        group(w)
        return carry

    b = pl.program_id(0)
    jax.lax.fori_loop(block_start_ref[b], block_start_ref[b + 1], loop, 0)


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
    work,
    *,
    q_per_head,
    sm_scale,
    block_q,
    pages_per_step,
    interpret,
    scale_cache=None,
    scale_rows=None,
    step_widths=_STEP_WIDTHS,
    cache_size=0,
    cache_bf16=False,
    row_split=1,
):
    """One kernel call over the tiles ``tile_offset + [0, tiles)`` of the
    head-major ``q [G * Hg, T, D]``, ``selected [G, T, K]``, ``local_page``
    and ``local_limit [T, 1]``, for query heads ``[b * Hg, (b + 1) * Hg)``
    attending cache-row KV head ``k``, ``head = (k, b)`` (``int32[2]``) with
    ``tile_offset`` (``int32[1]``): scalar-prefetch inputs, not statics, so
    one kernel serves every head and chunk, and no call slices its inputs.

    ``work = (packed, num_groups)``: the call's `msa_attention._work_list`
    as `_packed_work` packs it, and its group count.

    Writes those heads' and tokens' rows of ``(out [G * Hg, T, D], lse
    [G, T / Bq, 1, Hg * Bq])`` (a tile's lse a lane-dense row, ``h * Bq +
    t``): into `out` in place when given (an earlier call's
    result), else into new arrays whose other rows are unset. A 4-bit
    `kv_cache` takes its `scale_cache` and `scale_rows` (`_kernel`'s
    `fp4_rows`)."""
    num_q_heads, num_tokens, head_dim = q.shape
    num_heads = q_per_head
    _, rows_per_page, packing, _ = kv_cache.shape
    # One KV head's [K | V] slice of each row.
    row_width = 2 * head_dim
    topk = selected.shape[2]
    packed, tiles, tiles_per_step = work
    groups_per_tile = msa_attention._groups_per_tile(block_q, topk, pages_per_step)
    num_steps = tiles // tiles_per_step
    max_groups = tiles * groups_per_tile
    union_len = max_groups * pages_per_step

    def kernel(packed_ref, *refs, **kwargs):
        return _kernel(
            packed_ref.at[pl.ds(0, union_len)],
            packed_ref.at[pl.ds(union_len, max_groups)],
            packed_ref.at[pl.ds(union_len + max_groups, max_groups)],
            *refs,
            block_start_ref=packed_ref.at[
                pl.ds(union_len + 2 * max_groups, num_steps + 1)
            ],
            cache_ref=(
                packed_ref.at[
                    pl.ds(union_len + 2 * max_groups + num_steps + 1, cache_size + 2)
                ]
                if cache_size
                else None
            ),
            pages_per_step=pages_per_step,
            **kwargs,
        )

    def tile_block(b, packed, table, heads, offset):
        return (offset[0] // tiles_per_step + b, 0)

    def head_major_block(w, *prefetch):
        return (prefetch[2][1], *tile_block(w, *prefetch))

    def lse_block(w, *prefetch):
        return (*head_major_block(w, *prefetch), 0)

    aliased = out is not None
    in_place = (pl.BlockSpec(memory_space=pl.ANY),) * 2 if aliased else ()
    fp4 = scale_rows is not None
    transposed = not fp4 and block_q >= msa_attention.DEFAULT_BLOCK_Q
    scale_inputs = (scale_cache,) if fp4 else ()
    ring = 1 + (_FP4_PREFETCH if fp4 else 1)
    # Page buffers: a ring of groups, or (page cache) one flat array of
    # max(cache_size, ring groups) pages; the cache's semaphore row follows.
    if cache_size:
        buffers = (max(cache_size, ring * pages_per_step),)
        semaphores = (ring + 1, pages_per_step)
    else:
        buffers = (ring, pages_per_step)
        semaphores = (ring, pages_per_step)
    scale_scratch = (
        (
            pltpu.VMEM(buffers + (1,) + scale_cache.shape[1:], jnp.float32),
            pltpu.SemaphoreType.DMA(semaphores),
        )
        if fp4
        else ()
    )
    bf16_cache = bool(cache_size and cache_bf16 and fp4)
    bf16_scratch = (
        (pltpu.VMEM((cache_size, rows_per_page * 8, row_width), jnp.bfloat16),)
        if bf16_cache
        else ()
    )
    if row_split < 1 or num_heads % row_split or (num_heads // row_split * block_q) % 128:
        row_split = 1
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4,
        grid=(num_steps,),
        in_specs=[
            pl.BlockSpec(
                (num_heads, tiles_per_step * block_q, head_dim), head_major_block
            ),
            pl.BlockSpec((None, tiles_per_step * block_q, topk), head_major_block),
            pl.BlockSpec((tiles_per_step * block_q, 1), tile_block),
            pl.BlockSpec((tiles_per_step * block_q, 1), tile_block),
            pl.BlockSpec(memory_space=pl.ANY),
            *(pl.BlockSpec(memory_space=pl.ANY) for _ in scale_inputs),
            *in_place,
        ],
        out_specs=[
            pl.BlockSpec(
                (num_heads, tiles_per_step * block_q, head_dim), head_major_block
            ),
            pl.BlockSpec((1, tiles_per_step, 1, num_heads * block_q), lse_block),
        ],
        scratch_shapes=[
            # m, l, acc: rows on sublanes for `attend`, lanes for `attend_t`.
            *(
                [
                    pltpu.VMEM((num_heads * block_q, 1), jnp.float32),
                    pltpu.VMEM((num_heads * block_q, 1), jnp.float32),
                    pltpu.VMEM((num_heads * block_q, head_dim), jnp.float32),
                ]
                if not transposed
                else [
                    pltpu.VMEM((1, num_heads * block_q), jnp.float32),
                    pltpu.VMEM((1, num_heads * block_q), jnp.float32),
                    pltpu.VMEM((head_dim, num_heads * block_q), jnp.float32),
                ]
            ),
            pltpu.VMEM(
                buffers + (1, rows_per_page, packing, row_width), kv_cache.dtype
            ),
            pltpu.SemaphoreType.DMA(semaphores),
            *scale_scratch,
            *bf16_scratch,
        ],
    )
    num_inputs = 4 + 5 + len(scale_inputs)
    return pl.pallas_call(
        functools.partial(
            kernel,
            sm_scale=sm_scale,
            aliased=aliased,
            tiles_per_step=tiles_per_step,
            groups_per_tile=groups_per_tile,
            transposed=transposed,
            step_widths=step_widths,
            bf16_cache=bf16_cache,
            row_split=row_split,
            **({"fp4_rows": scale_rows} if fp4 else {}),
        ),
        grid_spec=grid_spec,
        out_shape=[
            jax.ShapeDtypeStruct((num_q_heads, num_tokens, head_dim), q.dtype),
            jax.ShapeDtypeStruct(
                (
                    num_q_heads // num_heads,
                    num_tokens // block_q,
                    1,
                    num_heads * block_q,
                ),
                jnp.float32,
            ),
        ],
        input_output_aliases=({num_inputs: 0, num_inputs + 1: 1} if aliased else {}),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            # The bf16 page cache (64 KB a slot) sits beside the 4-bit one.
            vmem_limit_bytes=(
                min(
                    63 * 1024 * 1024,
                    msa_attention._VMEM_LIMIT_BYTES + cache_size * rows_per_page * 8 * row_width * 2,
                )
                if bf16_cache
                else msa_attention._VMEM_LIMIT_BYTES
            ),
            # Every copy's page comes from the page table, so the checks only
            # add scalar work to each step (as the ragged paged attention).
            disable_bounds_checks=True,
            disable_semaphore_checks=True,
        ),
        interpret=interpret,
        name="minimax_m3_msa_sparse_attention_lse",
    )(
        packed,
        page_table,
        head,
        tile_offset,
        q,
        selected,
        local_page,
        local_limit,
        kv_cache,
        *scale_inputs,
        *(out if aliased else ()),
    )


def _tiles_per_step(num_tiles: int) -> int:
    """The most tiles (up to `TILES_PER_STEP`) a grid step can take evenly."""
    return next(n for n in range(TILES_PER_STEP, 0, -1) if num_tiles % n == 0)


def _packed_work(selected, block_q, pages_per_step):
    """`msa_attention._work_list` of each call's ``selected [C, T, K]`` as
    one int32 row per call, ``[unions | meta words | counts | step
    starts]``, and the tiles each grid step takes: a call then takes one
    slice of its work list, not one per array (each an XLA op of ~1.3 us,
    x 4 calls x G per layer). Step ``b``'s first group is its
    ``tiles_per_step * b``-th tile's; the last entry is the group count."""
    pages, meta, ends, counts = jax.vmap(
        functools.partial(
            msa_attention._work_list, block_q=block_q, pages_per_step=pages_per_step
        )
    )(selected)
    num_tiles = selected.shape[1] // block_q
    tiles_per_step = _tiles_per_step(num_tiles)
    groups_per_tile = msa_attention._groups_per_tile(
        block_q, selected.shape[2], pages_per_step
    )
    live = jnp.arange(meta.shape[1], dtype=jnp.int32)[None] < ends[:, None]
    tile = jnp.where(live, msa_attention._meta_tile(meta, groups_per_tile), num_tiles)
    bounds = jnp.arange(0, num_tiles + 1, tiles_per_step, dtype=jnp.int32)
    starts = jnp.sum(tile[:, None, :] < bounds[None, :, None], axis=2, dtype=jnp.int32)
    return jnp.concatenate([pages, meta, counts, starts], axis=1), tiles_per_step


def _chunked_call(
    q,
    kv_cache,
    selected,
    local_page,
    local_limit,
    page_table,
    first_head,
    *,
    sm_scale,
    block_q,
    pages_per_step,
    interpret,
    scale_cache=None,
    scale_rows=None,
    head_major=False,
    step_widths=_STEP_WIDTHS,
    page_cache_slots=0,
    page_cache_bf16=False,
    row_split=1,
):
    """`_attention_call` per KV head and chunk of at most
    `MAX_TOKENS_PER_CALL` tokens (the work list lives in SMEM).

    With `head_major`, ``(out [G * Hg, T, D], lse [G * Hg, T])`` come back
    head-major, as the kernel writes them, with no transpose.

    ``q [T, G * Hg, D]``: cache-row KV head ``first_head + h`` owns query
    heads ``[h * Hg, (h + 1) * Hg)``; ``selected [T, G, K]``. Every call
    reads the whole head-major query and writes its head's and chunk's rows
    of one output in place, so no call needs a slice, transpose or stacking
    of its own (~47 us per 2,048-token chunk under the former `lax.map` in
    the c4 prefill twin, j-3c651507, and ~43 us more per head at DCP 8).
    The calls after the first run as one `fori_loop`, so the kernel is
    emitted twice per call site (the first call creates the output) rather
    than once per head and chunk."""
    num_tokens, num_q_heads, _ = q.shape
    num_heads = selected.shape[1]
    if num_tokens == 0:
        if head_major:
            return q.transpose(1, 0, 2), jnp.zeros((num_q_heads, 0), jnp.float32)
        return q, jnp.zeros(q.shape[:2], jnp.float32)
    call = functools.partial(
        _attention_call,
        q_per_head=num_q_heads // num_heads,
        sm_scale=sm_scale,
        block_q=block_q,
        pages_per_step=pages_per_step,
        interpret=interpret,
        scale_cache=scale_cache,
        scale_rows=scale_rows,
        step_widths=step_widths,
        row_split=row_split,
    )
    cache_size = page_cache_slots if scale_cache is not None else 0
    if cache_size:
        call = functools.partial(call, cache_size=cache_size, cache_bf16=page_cache_bf16)
    limit = msa_attention.MAX_TOKENS_PER_CALL
    step = min(num_tokens, max(block_q, limit - limit % block_q))
    num_chunks = -(-num_tokens // step)
    pad = num_chunks * step - num_tokens
    q = jnp.pad(q, ((0, pad), (0, 0), (0, 0))).transpose(1, 0, 2)
    selected = jnp.pad(
        selected, ((0, pad), (0, 0), (0, 0)), constant_values=-1
    ).transpose(1, 0, 2)
    local_page = jnp.pad(local_page, (0, pad), constant_values=-1)[:, None]
    local_limit = jnp.pad(local_limit, (0, pad))[:, None]

    # Every call's work list in one batched pass ahead of the loop: built
    # per call inside it, each call paid its own two union sorts and glue
    # (~9 us per call, x G x 4 calls per layer).
    packed, tiles_per_step = _packed_work(
        selected.reshape(num_heads * num_chunks, step, selected.shape[2]),
        block_q,
        pages_per_step,
    )
    if cache_size:
        union_len = (step // block_q) * msa_attention._groups_per_tile(
            block_q, selected.shape[2], pages_per_step
        ) * pages_per_step
        table = jax.vmap(
            functools.partial(
                _page_cache, num_tiles=step // block_q, cache_size=cache_size
            )
        )(packed[:, :union_len])
        packed = jnp.concatenate(
            [table[:, :union_len], packed[:, union_len:], table[:, union_len:]], axis=1
        )

    def head_chunk(i, out):
        h, c = i // num_chunks, i % num_chunks
        offset = jnp.full((1,), c * (step // block_q), jnp.int32)
        heads = jnp.stack([first_head + h, h]).astype(jnp.int32)
        return call(
            q,
            kv_cache,
            selected,
            local_page,
            local_limit,
            page_table,
            heads,
            offset,
            out,
            (packed[i], step // block_q, tiles_per_step),
        )

    out = head_chunk(0, None)
    if num_heads * num_chunks > 1:
        out = jax.lax.fori_loop(1, num_heads * num_chunks, head_chunk, out)
    out, lse = out
    # lse [head blocks, tiles, 1, Hg * Bq] -> [T, G * Hg] (or head-major)
    num_blocks, num_tiles = lse.shape[:2]
    if head_major:
        lse = lse.reshape(num_blocks, num_tiles, -1, block_q).transpose(0, 2, 1, 3)
        lse = lse.reshape(num_q_heads, num_tiles * block_q)
        return out[:, :num_tokens], lse[:, :num_tokens]
    lse = lse.reshape(num_blocks, num_tiles, -1, block_q).transpose(1, 3, 0, 2)
    lse = lse.reshape(num_tiles * block_q, num_q_heads)
    return out.transpose(1, 0, 2)[:num_tokens], lse[:num_tokens]


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
        "head_major",
        "step_widths",
        "page_cache_slots",
        "page_cache_bf16",
        "row_split",
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
    head_major: bool = False,
    step_widths: tuple[int, ...] = _STEP_WIDTHS,
    page_cache_slots: int = 0,
    page_cache_bf16: bool = False,
    row_split: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """`msa_attention.msa_sparse_attention` (kernel v2, page keys translated
    by `page_table`) returning ``(out [T, H, D], lse [T, H] f32)``.

    A token with no selected page (or a padding token) gets a zero output
    and ``lse = -inf``, so it carries no weight in a log-sum-exp merge.
    Cache rows may hold several KV heads, ``[K_0 | V_0 | K_1 | V_1 ...]``;
    `head` picks the one these queries attend (a 2D-wide block of each page).
    A 4-bit cache takes its `scale_cache` (`msa_attention.msa_sparse_attention`).
    With `head_major`, the results come back head-major: ``out [H, T, D]``,
    ``lse [H, T]``.
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
        selected[:, None],
        local_page,
        local_limit,
        page_table,
        head,
        sm_scale=sm_scale,
        block_q=block_q,
        pages_per_step=pages_per_step,
        interpret=interpret,
        scale_cache=scale_cache,
        scale_rows=None if scale_cache is None else (k_scale_row, v_scale_row),
        head_major=head_major,
        step_widths=step_widths,
        page_cache_slots=page_cache_slots,
        page_cache_bf16=page_cache_bf16,
        row_split=row_split,
    )
    if head_major:
        return out[:, :num_tokens], lse[:, :num_tokens]
    return out[:num_tokens], lse[:num_tokens]


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "block_q",
        "pages_per_step",
        "interpret",
        "head_major",
        "step_widths",
        "page_cache_slots",
        "page_cache_bf16",
        "row_split",
    ),
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
    head_major: bool = False,
    step_widths: tuple[int, ...] = _STEP_WIDTHS,
    page_cache_slots: int = 0,
    page_cache_bf16: bool = False,
    row_split: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """`msa_sparse_attention_lse` for every KV head of the cache rows at once.

    ``q [T, G * Hg, D]`` (KV head ``h`` owns query heads ``[h * Hg,
    (h + 1) * Hg)``), ``selected [T, G, K]``. The heads run in one loop over
    a single kernel whose KV head is an input, so the program carries one
    copy of the kernel instead of one per head. Returns ``(out
    [T, G * Hg, D], lse [T, G * Hg] f32)``.

    With a 4-bit cache, `scale_cache` ``[pages, R, S]`` holds head ``h``'s K / V
    scales in rows ``2h`` / ``2h + 1``. `head_major` as `msa_sparse_attention_lse`.
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
    out, lse = _chunked_call(
        q,
        kv_cache,
        selected,
        local_page,
        local_limit,
        page_table,
        0,
        sm_scale=sm_scale,
        block_q=block_q,
        pages_per_step=pages_per_step,
        interpret=interpret,
        scale_cache=scale_cache,
        scale_rows=None if scale_cache is None else "head",
        head_major=head_major,
        step_widths=step_widths,
        page_cache_slots=page_cache_slots,
        page_cache_bf16=page_cache_bf16,
        row_split=row_split,
    )
    if head_major:
        return out[:, :num_tokens], lse[:, :num_tokens]
    return out[:num_tokens], lse[:num_tokens]
