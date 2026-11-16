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

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention, msa_fp4

# Page counts a kernel step may run at besides the full group (`_kernel`).
# 24 serves 32-page groups (32, 48: 64-page ones): a step costs per score
# column, filled or not, and a 17-24-page remainder is common there
# (standalone 4-bit 8-token tiles at served-c48-like dispersion: 2.00 -> 1.83
# ms, j-492a1bb9).
_STEP_WIDTHS = (4, 8, 16, 24, 32, 48)
# Tiles per grid step: a step runs the groups of this many tiles in a loop.
# One step per group paid a per-step cost worth ~2-3% of the call (pool48
# selections, 1-2 groups a tile; bench j-caec1354).
TILES_PER_STEP = 4
# Groups whose pages are in flight ahead of the running one: 4-bit groups
# (a page and a scale page each) gain from 2 (bench j-caec1354: c64 shared
# 2206 -> 2026, forced 2150 -> 2075 us per call), 8/16-bit ones do not.
_FP4_PREFETCH = 2
# Page copies a group's start loop issues per iteration (`_kernel`).
_START_UNROLL = 4
# Rows of ones `attend_t` appends to v^T for the softmax sums: one sublane tile.
_SUM_ROWS = 8
# Groups of more pages than this issue the next group's page copies from
# inside their body (`_kernel`'s issue_next), in calls of one KV head: in
# the every-head kernel (c64 DCP 8 decode rows) the extra body code cost
# +1.5 % (twin j-d5d8af02 vs j-098b7ff3), while c48's 4-bit calls gained
# 3-5 % in decode and mixed steps (j-a0b43b89 vs j-b068ae08).
_INLINE_ISSUE_PAGES = 16
_LOG2_E = math.log2(math.e)
_LN_2 = math.log(2.0)


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
    steps_per_call: int | None = None,
    inline_issue: bool = False,
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
    output blocks hold those tiles. With `steps_per_call`, the grid runs
    several calls' steps back to back and ``b`` counts within the call."""
    fp4 = fp4_rows is not None
    tail = refs[len(refs) - (9 if fp4 else 7) :]
    o_ref, lse_ref, m_ref, l_ref, acc_ref, buf_ref, sem_ref = tail[:7]
    scale_hbm_ref = refs[0] if fp4 else None
    scale_buf_ref, scale_sem_ref = tail[7:] if fp4 else (None, None)
    pages_per_step = buf_ref.shape[1]
    ring = buf_ref.shape[0]  # prefetched groups + the running one
    steps = pl.num_programs(0) if steps_per_call is None else steps_per_call
    num_groups = block_start_ref[steps]

    def group(
        w,
        q_ref=q_ref,
        o_ref=o_ref,
        lse_ref=lse_ref,
        selected_ref=selected_ref,
        local_page_ref=local_page_ref,
        local_limit_ref=local_limit_ref,
    ):
        store_out = None
        if tiles_per_step > 1:
            tile = msa_attention._meta_tile(work_meta_ref[w], groups_per_tile)
            bq = q_ref.shape[1] // tiles_per_step
            start = tile % tiles_per_step * bq
            rows = pl.ds(pl.multiple_of(start, bq), bq)
            q_ref = q_ref.at[:, rows, :]
            packed = _merged_store_rows(bq, o_ref.shape[0], o_ref.dtype)
            if packed > 1:
                store_out = functools.partial(
                    _store_unaligned_rows, o_ref, start, bq, packed
                )
            o_ref = o_ref.at[:, rows, :]
            selected_ref = selected_ref.at[rows, :]
            local_page_ref = local_page_ref.at[rows, :]
            local_limit_ref = local_limit_ref.at[rows, :]
            lse_ref = lse_ref.at[:, pl.ds(tile % tiles_per_step, 1)]
        meta = work_meta_ref[w]
        count = work_count_ref[w]
        num_heads, block_q, head_dim = q_ref.shape
        rows_per_page, packing, row_width = buf_ref.shape[3:]
        page_size = rows_per_page * (msa_fp4.TOKENS_PER_WORD if fp4 else packing)
        # 8-bit pages of `attend` reach the MXU without an unpack.
        direct = not fp4 and not transposed and packing == 4
        half = w % ring
        first = (meta & msa_attention._FIRST_BIT) != 0
        last = (meta & msa_attention._LAST_BIT) != 0

        def page_copies(half, g, page, n=1):
            """Copies of ``n`` pages from `page` on into slots ``g ..``: every
            copy of a ring slot signals its one semaphore, so a wait sized to
            ``n`` pages waits for any ``n`` of them."""
            src = kv_hbm_ref.at[
                pl.ds(page, n), :, :, pl.ds(head_ref[0] * row_width, row_width)
            ]
            copies = [
                pltpu.make_async_copy(
                    src, buf_ref.at[half, pl.ds(g, n), 0], sem_ref.at[half, 0]
                )
            ]
            if fp4:
                copies.append(
                    pltpu.make_async_copy(
                        scale_hbm_ref.at[pl.ds(page, n)],
                        scale_buf_ref.at[half, pl.ds(g, n), 0],
                        scale_sem_ref.at[half, 0],
                    )
                )
            return copies

        def start_group(group, half):
            base = msa_attention._slot(work_meta_ref[group], pages_per_step, 0)

            def start(g, queue=0):
                page = page_table_ref[work_page_ref[base + g]]
                for copy in page_copies(half, g, page):
                    copy.start(priority=queue)

            # Four page copies per iteration: their address chains (two
            # dependent SMEM loads each) overlap.
            count = work_count_ref[group]

            def start4(i, carry):
                for j in range(_START_UNROLL):  # alternate the two DMA queues
                    start(_START_UNROLL * i + j, j % 2)
                return carry

            jax.lax.fori_loop(0, count // _START_UNROLL, start4, 0)
            done = count - count % _START_UNROLL

            def start1(g, carry):
                start(g)
                return carry

            jax.lax.fori_loop(done, count, start1, 0)

        nxt = w + ring - 1

        def issue_next():
            """`start_group` of the group `ring - 1` ahead as straight-line,
            predicated copies of every slot: inside a wide body the scalar
            address work and descriptor issue share bundles with the vector
            work instead of running as a loop of their own ahead of it."""
            valid = nxt < num_groups
            group = jnp.where(valid, nxt, 0)
            base = msa_attention._slot(work_meta_ref[group], pages_per_step, 0)
            count = jnp.where(valid, work_count_ref[group], 0)
            for g in range(pages_per_step):
                page = page_table_ref[jnp.maximum(work_page_ref[base + g], 0)]

                @pl.when(g < count)
                def _start(g=g, page=page):
                    for copy in page_copies(nxt % ring, g, page):
                        copy.start(priority=g % 2)

        # Groups this wide issue the next group's copies inside their body.
        inline = inline_issue and count > _INLINE_ISSUE_PAGES

        @pl.when(w == 0)
        def _first():
            for d in range(ring - 1):
                pl.when(d < num_groups)(functools.partial(start_group, d, d))

        @pl.when((nxt < num_groups) & jnp.logical_not(inline))
        def _next():
            start_group(nxt, nxt % ring)

        # One wait per set bit of `count` (the source only sizes the wait).
        n = 1
        while n <= pages_per_step:
            for copy in page_copies(half, 0, 0, n):
                pl.when((count & n) != 0)(copy.wait)
            n *= 2

        kv_refs = [buf_ref.at[half, g] for g in range(pages_per_step)]
        widths = [n for n in _STEP_WIDTHS if n < pages_per_step] + [pages_per_step]
        if direct:
            # The step reads the slots up to its width; those past the group's
            # pages are stale (any bytes at first use): their values are zeroed,
            # p @ v multiplying them by p = 0.
            span = jnp.int32(widths[-1])
            for width in reversed(widths[:-1]):
                span = jnp.where(count <= width, width, span)

            def zero(g, carry):
                values = (
                    buf_ref.at[half, g]
                    .bitcast(jnp.uint32)
                    .at[0, :, 0, pl.ds(row_width // 2, row_width // 2)]
                )
                values[...] = jnp.zeros(values.shape, jnp.uint32)
                return carry

            jax.lax.fori_loop(count, span, zero, 0)

        @pl.when(first & jnp.logical_not(last))
        def _init():
            m_ref[...] = jnp.full_like(m_ref, msa_attention._MASK_VALUE)
            l_ref[...] = jnp.zeros_like(l_ref)
            acc_ref[...] = jnp.zeros_like(acc_ref)

        def put(out):  # [H * Bq, D] or [H, Bq, D]
            out = out.reshape(num_heads, block_q, head_dim)
            if store_out is None:
                o_ref[...] = out.astype(o_ref.dtype)
            else:
                store_out(out)

        def write(out, lse):
            put(out)
            lse_ref[...] = lse.T.reshape(lse_ref.shape)  # lane-dense row

        def finish(m, denom, acc):
            """Output and lse from a tile's softmax state (log2 units, ``m``
            finite). A row that saw no slot has ``denom = 0`` and ``acc = 0``:
            its output is 0 and its lse ``m + log(0) = -inf`` with no select (a
            seen row has ``denom >= 1``)."""
            write(acc * (1.0 / jnp.maximum(denom, 1.0)), m * _LN_2 + jnp.log(denom))

        def attend_t(width, single):
            if inline_issue and _INLINE_ISSUE_PAGES < width:
                issue_next()
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
            for g, kv_ref in enumerate(kv_refs[:width]):
                page = work_page_ref[msa_attention._slot(meta, pages_per_step, g)]
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
            # l = 1 . p comes out of the p @ v matmul as its row past the D
            # value rows (rows of ones streamed with v^T), not from a VPU add
            # per score vreg: kernel bench j-3e361d8c, ms per 8192-token call,
            # fp8 32-token tiles, 92.5k prefix: 0.855 -> 0.821 / 1.501 ->
            # 1.321 / 2.562 -> 2.148 at low / served-like / high dispersion.
            # The sum is over the bf16 p the matmul weighs v with.
            ones = jnp.ones((v.shape[0], _SUM_ROWS), v.dtype)
            pv = jax.lax.dot_general(
                jnp.concatenate([v, ones], axis=1),
                p.astype(v.dtype),
                (((0,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )  # [D + _SUM_ROWS, rows]
            l_cur, pv = pv[head_dim : head_dim + 1], pv[:head_dim]
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
            put((acc_t * (1.0 / jnp.maximum(denom, 1.0))).T)
            lse_ref[...] = (m * _LN_2 + jnp.log(denom)).reshape(lse_ref.shape)

        def attend(width, single):
            if inline_issue and _INLINE_ISSUE_PAGES < width:
                issue_next()
            """One softmax update over the group's first `width` slots; a
            `single` group (the tile's only one) needs no carried state."""
            # int4 and fp8 pages unpack in slot order; bf16 in the bitcast's order.
            row_slots = (
                jax.lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)
                if fp4
                else msa_attention._bitcast_slots(rows_per_page, packing)
            )  # [S, 1]
            slots = row_slots.T  # [1, S]
            selected = selected_ref[...]
            local_page = local_page_ref[...]
            local_limit = local_limit_ref[...]
            keys, values, masks, k_scales, v_scales = [], [], [], [], []
            for g, kv_ref in enumerate(kv_refs[:width]):
                page = work_page_ref[msa_attention._slot(meta, pages_per_step, g)]
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
                    rows, sk, sv = msa_attention._fp4_page(
                        kv_ref, scale_buf_ref.at[half, g], k_row, v_row
                    )
                    values.append(rows[:, head_dim:])
                    k_scales.append(sk)
                    v_scales.append(sv)
                elif direct:
                    # 8-bit keys and values enter the MXU as stored: the loaded
                    # words' bitcast is free (rows in its slot order,
                    # `row_slots`). No value slot needs zeroing: the KV write
                    # zeroes a page's slots past its newest token
                    # (`msa_write`), and stale ring slots were zeroed above.
                    words = kv_ref.bitcast(jnp.uint32)[0, :, 0, :]
                    rows = pltpu.bitcast(words, kv_ref.dtype)
                    values.append(rows[:, head_dim:])
                else:
                    rows = msa_attention._page_rows(kv_ref, seen)
                    values.append(rows[:, head_dim:])
                keys.append(rows[:, :head_dim])
                masks.append(slots < limit)  # [Bq, S]
            k = jnp.concatenate(keys, axis=0)
            v = jnp.concatenate(values, axis=0)
            mask = jnp.concatenate(masks, axis=1)
            mask = jnp.concatenate([mask] * num_heads, axis=0)

            scale = sm_scale * _LOG2_E
            q = q_ref[...].reshape(num_heads * block_q, head_dim)
            if not fp4:
                q = (q.astype(jnp.float32) * scale).astype(q.dtype)
            s = jax.lax.dot_general(
                q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )
            if fp4:  # x = int4 * scale: K scales the score columns, V scales p
                s = s * (jnp.concatenate(k_scales, axis=1) * scale)
            s = jnp.where(mask, s, -jnp.inf)

            # The running max starts at a finite floor (`_MASK_VALUE`), so a row
            # with no visible slot gets p = exp2(-inf) = 0, not exp2(nan).
            m_prev = m_ref[...] if not single else msa_attention._MASK_VALUE
            m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
            p = jnp.exp2(s - m_new)
            l_cur = jnp.sum(p, axis=1, keepdims=True)
            p_v = p * jnp.concatenate(v_scales, axis=1) if fp4 else p
            pv = jax.lax.dot_general(
                p_v.astype(jnp.bfloat16),
                v,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
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

        # Slots past the group's pages hold stale bytes: their limits are 0, so
        # their scores are masked and their value rows zeroed.
        # Run at the narrowest width holding the group's pages (`_STEP_WIDTHS`;
        # groups are 16, 32 or 64 pages by call). A step costs per score column,
        # filled or not (~2.3 us at 16 pages, ~1.1 us at 4, twin standalone
        # j-952e8a64), plus ~0.7 us whatever its width (j-aec3278d).
        single = first & last
        for i, width in enumerate(widths):
            fits = count > (widths[i - 1] if i else 0)
            if width < pages_per_step:
                fits = fits & (count <= width)
            body = attend_t if transposed else attend
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
    if steps_per_call is not None:
        b = b % steps_per_call
    jax.lax.fori_loop(block_start_ref[b], block_start_ref[b + 1], loop, 0)


def _merged_store_rows(block_q: int, num_heads: int, dtype) -> int:
    """The packed output window (`_store_unaligned_rows`) a tile's output is
    merged into, or 1 for a plain store: a window is the output dtype's
    sublane tile (16 rows for bf16). Blocks of 16 heads lower their 8-row
    stores (decode tiles); 8 heads did not (c64 head-parallel, j-d318070b)."""
    rows = 8 * 4 // jnp.dtype(dtype).itemsize
    return rows if block_q % rows and num_heads % rows else 1


def _store_unaligned_rows(o_ref, start, rows, packed, out):
    """Writes ``out [H, rows, D]`` (f32) to rows ``start ..`` of ``o_ref``
    when `rows` is not a multiple of the output's packed sublane tile
    (`packed`): a store at a run-time offset inside a packed tile does not
    lower (Mosaic E2003), so the tile holding the rows is read, merged and
    written whole. Tiles of a grid step run in order, so the rest of the
    packed tile holds an earlier tile's rows or rows a later one writes."""
    base = pl.multiple_of(start // packed * packed, packed)
    window = o_ref.at[:, pl.ds(base, packed), :]
    copies = jnp.concatenate([out.astype(jnp.float32)] * (packed // rows), axis=1)
    row = jax.lax.broadcasted_iota(jnp.int32, copies.shape, 1) + base
    mine = (row >= start) & (row < start + rows)
    window[...] = jnp.where(mine, copies, window[...].astype(jnp.float32)).astype(
        o_ref.dtype
    )


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
    calls=1,
    inline_issue=False,
):
    """One kernel call over the tiles ``tile_offset + [0, tiles)`` of the
    head-major ``q [G * Hg, T, D]``, ``selected [G, T, K]``, ``local_page``
    and ``local_limit [T, 1]``, for query heads ``[b * Hg, (b + 1) * Hg)``
    attending cache-row KV head ``k``, ``head = (k, b)`` (``int32[2]``) with
    ``tile_offset`` (``int32[1]``): scalar-prefetch inputs, not statics, so
    one kernel serves every head and chunk, and no call slices its inputs.

    ``work = (packed, tiles, tiles_per_step)``: the call's
    `msa_attention._work_list` as `_packed_work` packs it.

    Writes those heads' and tokens' rows of ``(out [G * Hg, T, D], lse
    [G, T / Bq, 1, Hg * Bq])`` (a tile's lse a lane-dense row, ``h * Bq +
    t``): into `out` in place when given (an earlier call's
    result), else into new arrays whose other rows are unset. A 4-bit
    `kv_cache` takes its `scale_cache` and `scale_rows` (`_kernel`'s
    `fp4_rows`).

    ``calls > 1`` runs that many such calls in one grid, back to back:
    ``head`` then holds each call's pair (``int32[2 * calls]``) and
    ``packed`` each call's work list, end to end; `out` must be None."""
    num_q_heads, num_tokens, head_dim = q.shape
    num_heads = q_per_head
    topk = selected.shape[2]
    packed, tiles, tiles_per_step = work
    groups_per_tile = msa_attention._groups_per_tile(block_q, topk, pages_per_step)
    num_steps = tiles // tiles_per_step
    sections, length = _work_layout(
        tiles, groups_per_tile, pages_per_step, num_steps, aligned=False
    )
    if calls > 1 and out is not None:
        raise ValueError("calls run in one grid write new outputs")

    def kernel(packed_ref, page_table_ref, head_ref, *refs, **kwargs):
        if calls > 1:
            call = pl.program_id(0) // num_steps
            base, head_ref = call * length, head_ref.at[pl.ds(2 * call, 2)]
        else:  # static offsets: no address arithmetic on every work-list read
            base = 0
        unions, metas, counts, starts = (
            packed_ref.at[pl.ds(base + at, n)] for at, n in sections
        )
        return _kernel(
            unions,
            metas,
            counts,
            page_table_ref,
            head_ref,
            *refs,
            block_start_ref=starts,
            steps_per_call=num_steps if calls > 1 else None,
            **kwargs,
        )

    def tile_block(b, packed, table, heads, offset):
        return (offset[0] // tiles_per_step + (b % num_steps if calls > 1 else b), 0)

    def head_major_block(w, *prefetch):
        call = w // num_steps if calls > 1 else 0
        return (prefetch[2][2 * call + 1], *tile_block(w, *prefetch))

    def lse_block(w, *prefetch):
        return (*head_major_block(w, *prefetch), 0)

    aliased = out is not None
    in_place = (pl.BlockSpec(memory_space=pl.ANY),) * 2 if aliased else ()
    fp4 = scale_rows is not None
    transposed = not fp4 and block_q >= msa_attention.DEFAULT_BLOCK_Q
    scale_inputs = (scale_cache,) if fp4 else ()
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4,
        grid=(calls * num_steps,),
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
        scratch_shapes=_kernel_scratch(
            num_heads * block_q,
            head_dim,
            pages_per_step,
            kv_cache,
            scale_cache if fp4 else None,
            transposed,
        ),
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
            inline_issue=inline_issue,
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
            vmem_limit_bytes=msa_attention._VMEM_LIMIT_BYTES,
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


def _kernel_scratch(rows, head_dim, pages_per_step, kv_cache, scale_cache, transposed):
    """`_kernel`'s scratch: m, l, acc (rows on sublanes for `attend`, lanes
    for `attend_t`), the page ring and its semaphores, and a 4-bit cache's
    scale-page ring."""
    _, rows_per_page, packing, _ = kv_cache.shape
    ring = 1 + (_FP4_PREFETCH if scale_cache is not None else 1)
    if transposed:
        softmax = [(1, rows), (1, rows), (head_dim, rows)]
    else:
        softmax = [(rows, 1), (rows, 1), (rows, head_dim)]
    scratch = [pltpu.VMEM(shape, jnp.float32) for shape in softmax]
    scratch += [
        pltpu.VMEM(
            (ring, pages_per_step, 1, rows_per_page, packing, 2 * head_dim),
            kv_cache.dtype,
        ),
        pltpu.SemaphoreType.DMA((ring, 1)),
    ]
    if scale_cache is not None:
        scratch += [
            pltpu.VMEM((ring, pages_per_step, 1) + scale_cache.shape[1:], jnp.float32),
            pltpu.SemaphoreType.DMA((ring, 1)),
        ]
    return scratch


def planned_attention(
    q,
    kv_cache,
    page_table,
    plan,
    plan_scalars,
    plan_inputs,
    *,
    topk,
    sm_scale,
    block_q,
    pages_per_step,
    interpret=False,
    scale_cache=None,
    scale_rows=None,
):
    """`msa_sparse_attention_lse` (one KV head, head-major q and results) of
    a decode step whose selection, local pages and work list the kernel
    computes itself in its first grid step: ``plan(plan_scalars_ref (SMEM),
    *plan_input_refs, keys_ref [T, K], local_page_ref [T, 1], local_limit_ref
    [T, 1], work_ref [1, length])`` (`msa_dcp._decode_plan_kernel`) writes
    them into scratch, and the work list moves to SMEM there. One kernel in
    place of a plan kernel and the attention kernel: each Pallas call costs
    ~1 us plus ~0.2 us per operand (standalone bench j-97afc7d4). Returns
    ``(out [H, T, D], lse [H, T])``."""
    num_heads, num_tokens, head_dim = q.shape
    num_tiles = num_tokens // block_q
    sections, length, tiles_per_step, groups_per_tile = call_work_spec(
        num_tokens, num_heads, q.dtype, topk, block_q, pages_per_step
    )
    rows = tiles_per_step * block_q
    fp4 = scale_rows is not None
    scale_inputs = (scale_cache,) if fp4 else ()

    def kernel(table_ref, head_ref, offset_ref, scalars_ref, q_ref, *refs, **kwargs):
        num_plan = len(plan_inputs)
        plan_refs, refs = refs[:num_plan], refs[num_plan:]
        refs, (keys, local_page, local_limit, work_v, work_s, sem) = (
            refs[:-6],
            refs[-6:],
        )
        b = pl.program_id(0)

        @pl.when(b == 0)
        def _plan():
            plan(scalars_ref, *plan_refs, keys, local_page, local_limit, work_v)
            copy = pltpu.make_async_copy(work_v.at[0], work_s, sem)
            copy.start()
            copy.wait()

        unions, metas, counts, starts = (work_s.at[pl.ds(at, n)] for at, n in sections)
        step = pl.ds(pl.multiple_of(b * rows, rows), rows)
        return _kernel(
            unions,
            metas,
            counts,
            table_ref,
            head_ref,
            offset_ref,
            q_ref,
            keys.at[step],
            local_page.at[step],
            local_limit.at[step],
            *refs,
            block_start_ref=starts,
            **kwargs,
        )

    def whole(x):
        return pl.BlockSpec(x.shape, lambda b, *_: (0,) * x.ndim)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4,
        grid=(num_tiles // tiles_per_step,),
        in_specs=[
            pl.BlockSpec((num_heads, rows, head_dim), lambda b, *_: (0, b, 0)),
            *(whole(x) for x in plan_inputs),
            pl.BlockSpec(memory_space=pl.ANY),
            *(pl.BlockSpec(memory_space=pl.ANY) for _ in scale_inputs),
        ],
        out_specs=[
            pl.BlockSpec((num_heads, rows, head_dim), lambda b, *_: (0, b, 0)),
            pl.BlockSpec(
                (1, tiles_per_step, 1, num_heads * block_q), lambda b, *_: (0, b, 0, 0)
            ),
        ],
        scratch_shapes=_kernel_scratch(
            num_heads * block_q,
            head_dim,
            pages_per_step,
            kv_cache,
            scale_cache if fp4 else None,
            transposed=False,
        )
        + [
            pltpu.VMEM((num_tokens, topk), jnp.int32),
            pltpu.VMEM((num_tokens, 1), jnp.int32),
            pltpu.VMEM((num_tokens, 1), jnp.int32),
            pltpu.VMEM((1, length), jnp.int32),
            pltpu.SMEM((length,), jnp.int32),
            pltpu.SemaphoreType.DMA(()),
        ],
    )
    out, lse = pl.pallas_call(
        functools.partial(
            kernel,
            sm_scale=sm_scale,
            tiles_per_step=tiles_per_step,
            groups_per_tile=groups_per_tile,
            **({"fp4_rows": scale_rows} if fp4 else {}),
        ),
        grid_spec=grid_spec,
        out_shape=[
            jax.ShapeDtypeStruct((num_heads, num_tokens, head_dim), q.dtype),
            jax.ShapeDtypeStruct((1, num_tiles, 1, num_heads * block_q), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=msa_attention._VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
            disable_semaphore_checks=True,
        ),
        interpret=interpret,
        name="minimax_m3_msa_planned_attention_lse",
    )(
        page_table,
        jnp.zeros((2,), jnp.int32),
        jnp.zeros((1,), jnp.int32),
        plan_scalars,
        q,
        *plan_inputs,
        kv_cache,
        *scale_inputs,
    )
    lse = lse.reshape(num_tiles, num_heads, block_q).transpose(1, 0, 2)
    return out, lse.reshape(num_heads, num_tokens)


def _tiles_per_step(num_tiles: int, rows: int = 1, align: int = 1) -> int:
    """The most tiles (up to `TILES_PER_STEP`) a grid step can take evenly;
    with `align`, only counts whose `rows`-row tiles fill whole `align`-row
    output windows (`_store_unaligned_rows`), or one tile."""
    return next(
        n
        for n in range(TILES_PER_STEP, 0, -1)
        if num_tiles % n == 0 and (n == 1 or n * rows % align == 0)
    )


_LANES = 128


def _work_layout(num_tiles, groups_per_tile, pages_per_step, num_steps, aligned):
    """``((offset, length), ...)`` of a packed work list's sections,
    ``[unions | meta words | counts | step starts]``, end to end or, when
    `aligned`, each starting on a 128-lane boundary (so `store_work_list`
    stores each whole), and the row's length."""
    max_groups = num_tiles * groups_per_tile
    sections, at = [], 0
    for n in (max_groups * pages_per_step, max_groups, max_groups, num_steps + 1):
        sections.append((at, n))
        at += -(-n // _LANES) * _LANES if aligned else n
    return tuple(sections), at


def call_work_spec(num_tokens, num_q_heads, dtype, topk, block_q, pages_per_step):
    """A one-head call's packed work list over ``num_tokens`` rows of
    ``num_q_heads`` query heads, as `store_work_list` lays it out (aligned
    sections): ``(sections, length, tiles_per_step, groups_per_tile)``."""
    num_tiles = num_tokens // block_q
    align = _merged_store_rows(block_q, num_q_heads, dtype)
    tiles_per_step = _tiles_per_step(num_tiles, block_q, align)
    groups_per_tile = msa_attention._groups_per_tile(block_q, topk, pages_per_step)
    sections, length = _work_layout(
        num_tiles,
        groups_per_tile,
        pages_per_step,
        num_tiles // tiles_per_step,
        aligned=True,
    )
    return sections, length, tiles_per_step, groups_per_tile


def fits_work_row(block_q, topk, pages_per_step) -> bool:
    """Whether `store_work_list` can build a call's work list: a tile's keys
    and its union row are one 128-lane row each."""
    groups_per_tile = msa_attention._groups_per_tile(block_q, topk, pages_per_step)
    return block_q * topk == _LANES and groups_per_tile * pages_per_step == _LANES


def _packed_work(selected, block_q, pages_per_step, align=1):
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
    tiles_per_step = _tiles_per_step(num_tiles, block_q, align)
    groups_per_tile = msa_attention._groups_per_tile(
        block_q, selected.shape[2], pages_per_step
    )
    live = jnp.arange(meta.shape[1], dtype=jnp.int32)[None] < ends[:, None]
    tile = jnp.where(live, msa_attention._meta_tile(meta, groups_per_tile), num_tiles)
    bounds = jnp.arange(0, num_tiles + 1, tiles_per_step, dtype=jnp.int32)
    starts = jnp.sum(tile[:, None, :] < bounds[None, :, None], axis=2, dtype=jnp.int32)
    return jnp.concatenate([pages, meta, counts, starts], axis=1), tiles_per_step


def _dense_unions(keys, num_tiles):
    """`msa_attention._dense_unions` of each of the first `num_tiles` rows of
    ``keys [rows, 128]`` (`_NO_PAGE` for no page), in VMEM."""
    no_page = msa_attention._NO_PAGE
    rows = keys.shape[0]
    keys_t = keys.T  # [128, rows]: a tile's keys down the sublanes
    iota = functools.partial(jax.lax.broadcasted_iota, jnp.int32)
    sub, lane = iota((_LANES, _LANES), 0), iota((_LANES, _LANES), 1)
    tile = iota((rows, _LANES), 0)
    unions = jnp.full((rows, _LANES), no_page, jnp.int32)
    for i in range(num_tiles):
        x_r, x_c = keys[i : i + 1, :], keys_t[:, i : i + 1]
        same = x_c == x_r  # [a, b]
        seen_c = jnp.max(jnp.where(same & (lane < sub), 1, 0), axis=1, keepdims=True)
        seen_r = jnp.max(jnp.where(same & (sub < lane), 1, 0), axis=0, keepdims=True)
        first_c = (x_c != no_page) & (seen_c == 0)
        first_r = (x_r != no_page) & (seen_r == 0)
        # Each distinct page's slot in the ascending union.
        place_c = jnp.sum(jnp.where(first_r & (x_r < x_c), 1, 0), axis=1, keepdims=True)
        union = jnp.min(
            jnp.where(first_c & (place_c == lane), x_c, no_page), axis=0, keepdims=True
        )
        unions = jnp.where(tile == i, union, unions)
    return unions


def store_work_list(
    keys,
    out_ref,
    *,
    num_tiles,
    pages_per_step,
    groups_per_tile,
    tiles_per_step,
    sections,
):
    """Inside a kernel: `_packed_work`'s row ``out_ref [1, length]`` for one
    call of `num_tiles` tiles of 128 page keys each (``keys [rows, 128]``, a
    tile per row, -1 for no page): the same values as
    `msa_attention._work_list` (dense unions) and the step starts."""
    no_page = msa_attention._NO_PAGE
    rows = keys.shape[0]
    iota = functools.partial(jax.lax.broadcasted_iota, jnp.int32)
    tile_c, tile_r = iota((rows, 1), 0), iota((1, rows), 1)
    unions = _dense_unions(jnp.where(keys < 0, no_page, keys), num_tiles)
    present = jnp.where(unions != no_page, 1, 0)
    pages_c = jnp.sum(present, axis=1, keepdims=True)
    pages_r = jnp.sum(present.T, axis=0, keepdims=True)
    unions = jnp.where(unions == no_page, -1, unions)
    unions_at, _ = sections[0]
    for i in range(num_tiles):
        at = unions_at + i * _LANES
        out_ref[:, at : at + _LANES] = unions[i : i + 1]

    def groups_of(pages, real):
        return jnp.where(real, jnp.maximum(-(-pages // pages_per_step), 1), 0)

    real_c = tile_c < num_tiles
    groups_c = groups_of(pages_c, real_c)
    groups_r = groups_of(pages_r, tile_r < num_tiles)
    ends_c = jnp.sum(
        jnp.where(iota((rows, rows), 1) <= iota((rows, rows), 0), groups_r, 0),
        axis=1,
        keepdims=True,
    )
    total = jnp.sum(groups_r, axis=1, keepdims=True)  # [1, 1]
    starts_c = ends_c - groups_c
    big = jnp.iinfo(jnp.int32).max
    first_at = jnp.where(real_c, starts_c, big)
    end_at = jnp.where(real_c, ends_c, big)
    (meta_at, max_groups), (counts_at, _), (steps_at, num_bounds) = sections[1:]
    width = -(-max_groups // _LANES) * _LANES
    w = iota((1, width), 1)
    tile = jnp.sum(
        jnp.where((tile_c >= 1) & (first_at <= w), 1, 0), axis=0, keepdims=True
    )
    start = jnp.max(jnp.where(first_at <= w, first_at, 0), axis=0, keepdims=True)
    end = jnp.min(jnp.where(end_at > w, end_at, total), axis=0, keepdims=True)
    tile_pages = jnp.max(
        jnp.where((first_at <= w) & (end_at > w), pages_c, 0), axis=0, keepdims=True
    )
    group = jnp.clip(w - start, 0, groups_per_tile - 1)
    count = jnp.clip(tile_pages - group * pages_per_step, 0, pages_per_step)
    meta = (
        ((tile * groups_per_tile + group) << msa_attention._ROW_SHIFT)
        | jnp.where(w == start, msa_attention._FIRST_BIT, 0)
        | jnp.where(w + 1 == end, msa_attention._LAST_BIT, 0)
    )
    in_list = w < max_groups
    out_ref[:, meta_at : meta_at + width] = jnp.where(in_list, meta, 0)
    out_ref[:, counts_at : counts_at + width] = jnp.where(in_list, count, 0)
    # Step b starts at the groups of the tiles before its first one.
    steps = -(-num_bounds // _LANES) * _LANES
    bound = iota((1, steps), 1) * tiles_per_step
    step_starts = jnp.sum(
        jnp.where(real_c & (tile_c < bound), groups_c, 0), axis=0, keepdims=True
    )
    out_ref[:, steps_at : steps_at + steps] = jnp.where(
        iota((1, steps), 1) < num_bounds, step_starts, 0
    )


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
    q_head_major=False,
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
    than once per head and chunk. With `q_head_major`, q comes ``[G * Hg, T,
    D]``, as the kernel reads it."""
    if not q_head_major:
        q = q.transpose(1, 0, 2)
    num_q_heads, num_tokens, _ = q.shape
    num_heads = selected.shape[1]
    if num_tokens == 0:
        if head_major:
            return q, jnp.zeros((num_q_heads, 0), jnp.float32)
        return q.transpose(1, 0, 2), jnp.zeros((0, num_q_heads), jnp.float32)
    call = functools.partial(
        _attention_call,
        inline_issue=_INLINE_ISSUE_PAGES > 0 and num_heads == 1,
        q_per_head=num_q_heads // num_heads,
        sm_scale=sm_scale,
        block_q=block_q,
        pages_per_step=pages_per_step,
        interpret=interpret,
        scale_cache=scale_cache,
        scale_rows=scale_rows,
    )
    limit = msa_attention.MAX_TOKENS_PER_CALL
    step = min(num_tokens, max(block_q, limit - limit % block_q))
    num_chunks = -(-num_tokens // step)
    pad = num_chunks * step - num_tokens
    q = jnp.pad(q, ((0, 0), (0, pad), (0, 0)))
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
        align=_merged_store_rows(block_q, num_q_heads // num_heads, q.dtype),
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

    if num_chunks == 1 and num_heads > 1:
        # Every KV head's call in one grid (decode steps of several heads, DCP
        # 8): no call's first pages wait unprefetched and no per-call slice.
        heads = jnp.arange(num_heads, dtype=jnp.int32)
        out, lse = call(
            q,
            kv_cache,
            selected,
            local_page,
            local_limit,
            page_table,
            jnp.stack([first_head + heads, heads], axis=1).reshape(-1),
            jnp.zeros((1,), jnp.int32),
            None,
            (packed.reshape(-1), step // block_q, tiles_per_step),
            calls=num_heads,
        )
    else:
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


def _pad_tokens(q, selected, local_page, local_limit, block_q, q_head_major=False):
    pad = -selected.shape[0] % block_q
    if pad:
        widths = [(0, 0)] * q.ndim
        widths[1 if q_head_major else 0] = (0, pad)
        q = jnp.pad(q, widths)
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
        "q_head_major",
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
    q_head_major: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`msa_attention.msa_sparse_attention` (kernel v2, page keys translated
    by `page_table`) returning ``(out [T, H, D], lse [T, H] f32)``.

    A token with no selected page (or a padding token) gets a zero output
    and ``lse = -inf``, so it carries no weight in a log-sum-exp merge.
    Cache rows may hold several KV heads, ``[K_0 | V_0 | K_1 | V_1 ...]``;
    `head` picks the one these queries attend (a 2D-wide block of each page).
    A 4-bit cache takes its `scale_cache` (`msa_attention.msa_sparse_attention`).
    With `head_major`, the results come back head-major: ``out [H, T, D]``,
    ``lse [H, T]``; with `q_head_major`, q comes head-major, ``[H, T, D]``.
    """
    num_tokens, head_dim = selected.shape[0], q.shape[2]
    msa_attention.check_cache(kv_cache, scale_cache)
    _check_rows(kv_cache, head_dim, head)
    q, selected, local_page, local_limit = _pad_tokens(
        q, selected, local_page, local_limit, block_q, q_head_major
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
        q_head_major=q_head_major,
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
        "q_head_major",
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
    q_head_major: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`msa_sparse_attention_lse` for every KV head of the cache rows at once.

    ``q [T, G * Hg, D]`` (KV head ``h`` owns query heads ``[h * Hg,
    (h + 1) * Hg)``), ``selected [T, G, K]``. The heads run in one loop over
    a single kernel whose KV head is an input, so the program carries one
    copy of the kernel instead of one per head. Returns ``(out
    [T, G * Hg, D], lse [T, G * Hg] f32)``.

    With a 4-bit cache, `scale_cache` ``[pages, R, S]`` holds head ``h``'s K / V
    scales in rows ``2h`` / ``2h + 1``. `head_major` and `q_head_major` as
    `msa_sparse_attention_lse`.
    """
    num_tokens, num_heads = selected.shape[:2]
    num_q_heads, head_dim = q.shape[0 if q_head_major else 1], q.shape[2]
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
        q, selected, local_page, local_limit, block_q, q_head_major
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
        q_head_major=q_head_major,
    )
    if head_major:
        return out[:, :num_tokens], lse[:, :num_tokens]
    return out[:num_tokens], lse[:num_tokens]
