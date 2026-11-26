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
import types

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
# 4-bit `attend_t` groups (prefill-size calls) also run at 12 pages, the
# union of a tile whose tokens select much the same blocks, in one body per
# width (`attend_t_any`). Kernel bench, c48 2048-token call of 64-token tiles
# at locality .999: 96.5 -> 85.3 us; tiles of 12 fixed pages 91.5 -> 74.8 us.
# Their bodies are many and large: with the single-group copies as well,
# 12 bodies ran a c48 call 8 % slower and 16 bodies 8x slower.
_STEP_WIDTHS_4BIT_T = (4, 8, 12, 16, 24, 32, 48)
# Units also run at 12 pages, the width of c48-like decode groups (one KV
# head, 22-34 requests, locality .97: -8..-9 % per call). Other calls that
# run groups alone do not take it (a 64-page head-parallel prefill call ran
# ~5x slower with it among two bodies per width), except 4-bit `attend_t`
# groups, which have one body per width (`_STEP_WIDTHS_4BIT_T`).
_UNIT_STEP_WIDTHS = (4, 8, 12, 16, 24, 32, 48)
# Tiles per grid step: a step runs the groups of this many tiles in a loop.
# One step per group paid a per-step cost worth ~2-3% of the call (pool48
# selections, 1-2 groups a tile; bench j-caec1354).
TILES_PER_STEP = 4
# 4-bit prefill-size calls (8-token tiles) take more: kernel bench j-479565f1,
# c48 shape 1.193 -> 1.167 ms (rho .97), c64 head-parallel shape tps 8 -2.3 %.
# Calls of `attend` tiles running units take `_decode_tiles_per_step` instead.
PREFILL_TILES_PER_STEP_4BIT = 16
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
# Groups an `attend` (8-token decode tile) step runs side by side, with
# `MAX_FREE`: a group's QK matmul, softmax, PV matmul and finish are one
# chain of MXU and EUP latencies (~700 of a 4-page group's ~880 cycles, AOT
# pre-delay schedule), which a lone group cannot overlap; the chains of
# independent groups (other tiles, or other pages of one tile) fill each
# other's gaps. A unit's loop issues the page copies of the unit after it
# (`run_unit`); issuing them inside the body as predicated copies
# (`issue_next`) was no faster. Kernel bench, 4 vs 1 group (b4, b5): c64
# decode 117.9 -> 101.0 us per call (rho .85), 100.7 -> 82.4 (rho .97);
# planned fp8 5 requests 13.3 -> 12.3. Which calls run units: `_unit_groups`.
DECODE_UNIT_GROUPS = 4
# 8-bit units hold groups of at most this many pages; wider 8-bit groups run
# one at a time (their unit bodies spill: twin c16, j-7ca00ad7, 3-5 requests
# +0.26-0.42 ms/step as units of 32-page groups). 4-bit units take any width
# (capping them halved the c64 gain, j-504fb617).
UNIT_MAX_PAGES_8BIT = 16
# Units whose pages are in flight ahead of the running one.
UNIT_PREFETCH = 1
# Unit bodies at most this wide emit every group's QK matmul, then every
# softmax, then every p @ v, so the MXU takes another group's matmul while a
# group's softmax runs (kernel bench, c64 b46 decode at locality .97: -8.6 %
# per call; at 16+ pages the staged scores spill: c64 locality .5 +7 %).
_STAGED_UNIT_PAGES = 8
# Units pair only groups of one grid step (its query and output blocks), so
# decode-tile calls put up to this many tiles in a step, their token count
# padded to whole output windows (`_merged_store_rows`).
UNIT_TILES_PER_STEP = 64
# ... and at most this many query rows (tiles x tokens x query heads) a step,
# which sizes its query and output blocks.
_UNIT_STEP_ROWS = 8192
# Scores need no running max: q and k come out of per-head RMS norms (weight
# 1 + w; partial RoPE and the 4-bit Hadamard rotation keep norms), so |q.k| *
# sm_scale <= sqrt(D) * max|1 + w_q| * max|1 + w_k| <= 57.8 over the
# MiniMax-M3.1 checkpoint's 60 layers (85.7 with the worst 4-bit key rounding,
# x1.476), inside exp's f32 range (88.7) both ways: p = exp2(s) never
# overflows, and a row's largest p stays a normal float.
MAX_FREE = True
# `configure_max_free` keeps MAX_FREE only under this score bound: exp(75)
# leaves e^13 of f32 range for a tile's softmax sum and p @ v.
_MAX_FREE_SCORE_BOUND = 75.0
# With `MAX_FREE`, `attend_t` takes each token's mask from the QK matmul: the
# keys carry a column block that is -_MASK_BIAS where a token does not see the
# key, the queries a one-hot of their token there.
MXU_MASK = True
_MASK_BIAS = 1e30
# 4-bit calls of prefill size take their tokens' page limits from a table the
# caller builds (`_page_limits`), each page's broadcast over its slots:
# comparing every page against the tile's selections, with a lane reduction
# each, cost ~12 % of a 4-bit 8-token call (r9 kernel bench j-ecb17102).
LIMIT_TABLE = True
_LIMIT_TABLE_MIN_TOKENS = msa_attention.SMALL_BATCH_TOKENS
# Calls without the table (decode) compute the same [Bq, W] limits in the
# kernel, once per group: one compare per selection column against the
# group's pages instead of a compare and a lane reduction per page (13 % of
# a c64 decode call, kernel bench j-0a524278), pages along the lanes as the
# table's (transposing a [W, Bq] form was the head of a group's critical path).
# The per-page lane reductions were the critical path of 8-bit decode groups
# too (AOT pre-delay schedule, 32 pages: 2,649 -> 1,990 cycles).
KERNEL_LIMITS = True
# 4-bit tiles of 16+ tokens run `attend_t` with the K / V scales folded into
# their bf16 rows, as the 8/16-bit caches do.
FP4_TRANSPOSED = True
# `attend_t` feeds 8-bit pages to the MXU as stored (a bf16 x fp8 dot is
# exact): no per-page widening on the VPU; the token mask rides as fp8
# columns of -_FP8_MASK beside the keys (`MAX_FREE` scores stay below
# _MAX_FREE_SCORE_BOUND * log2(e) ~ 108 in log2 units, so exp2 of a masked one
# is 0). It relies on `msa_write` zeroing the slots past a page's newest token.
DIRECT_T = True
_FP8_MASK = 448.0  # e4m3fn's largest magnitude
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
    limited: bool = False,
    kernel_limits: bool = False,
    packed_stores: bool = False,
    unit_groups: int = 1,
    whole_rows: bool = False,
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
    several calls' steps back to back and ``b`` counts within the call.

    With ``unit_groups > 1`` (`DECODE_UNIT_GROUPS`; `attend` with `MAX_FREE`),
    a step runs its groups that many at a time: their page copies, waits,
    scores and p @ v side by side, then each group's state update in order."""
    fp4 = fp4_rows is not None
    spreads = limited or kernel_limits
    units = unit_groups > 1
    tail = refs[len(refs) - (9 if fp4 else 7) - units :]
    o_ref, lse_ref, m_ref, l_ref, acc_ref, buf_ref, sem_ref = tail[:7]
    # Units: the groups whose page copies are issued, ``[0, issued_ref[0])``.
    issued_ref = tail[9 if fp4 else 7] if units else None
    scale_hbm_ref = refs[0] if fp4 else None
    scale_buf_ref, scale_sem_ref = tail[7:9] if fp4 else (None, None)
    # `limited`: each union slot's limit per token (`_page_limits`, blocked
    # per grid step).
    limit_ref = refs[1] if limited else None
    pages_per_step = buf_ref.shape[1]
    ring = buf_ref.shape[0]  # prefetched groups + the running one
    steps = pl.num_programs(0) if steps_per_call is None else steps_per_call
    num_groups = block_start_ref[steps]
    q_leaf = q_ref[0] if isinstance(q_ref, tuple) else q_ref
    row_width = 2 * q_leaf.shape[-1]
    # With `whole_rows` (`_whole_rows`) the ring holds whole cache rows: one
    # contiguous copy per page, and a group reads its head's lanes.

    def page_copies(half, g, page, n=1):
        """Copies of ``n`` pages from `page` on into slots ``g ..``: every
        copy of a ring slot signals its one semaphore, so a wait sized to
        ``n`` pages waits for any ``n`` of them."""
        if whole_rows:
            src = kv_hbm_ref.at[pl.ds(page, n)]
        else:
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

    def group(
        w,
        q_ref=q_ref,
        o_ref=o_ref,
        lse_ref=lse_ref,
        selected_ref=selected_ref,
        local_page_ref=local_page_ref,
        local_limit_ref=local_limit_ref,
        unit=False,
        waited=False,
    ):
        """Group `w`: its waits and zeroing, then its body, or with `unit`
        (`unit_groups > 1`) a namespace of its body's parts for the unit."""
        store_out = None
        q_refs = q_ref if isinstance(q_ref, tuple) else (q_ref,)
        if tiles_per_step > 1:
            tile = msa_attention._meta_tile(work_meta_ref[w], groups_per_tile)
            bq = q_refs[0].shape[1] // tiles_per_step
            start = tile % tiles_per_step * bq
            rows = pl.ds(pl.multiple_of(start, bq), bq)
            q_refs = tuple(r.at[:, rows, :] for r in q_refs)
            packed = _merged_store_rows(bq, o_ref.shape[0], o_ref.dtype, packed_stores)
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
        _, block_q, head_dim = q_refs[0].shape
        num_heads = sum(r.shape[0] for r in q_refs)

        def load_q():  # [H, Bq, D]: a pair of query refs holds the heads in order
            if len(q_refs) == 1:
                return q_refs[0][...]
            return jnp.concatenate([r[...] for r in q_refs], axis=0)

        rows_per_page, packing = buf_ref.shape[3:5]
        page_size = rows_per_page * (msa_fp4.TOKENS_PER_WORD if fp4 else packing)
        # 8-bit pages of `attend` reach the MXU without an unpack.
        direct = not fp4 and not transposed and packing == 4
        # ... and of `attend_t` (`DIRECT_T`, with the MXU mask).
        direct_t = (
            DIRECT_T
            and not fp4
            and transposed
            and packing == 4
            and MXU_MASK
            and MAX_FREE
            and block_q <= head_dim
        )
        half = w % ring
        first = (meta & msa_attention._FIRST_BIT) != 0
        last = (meta & msa_attention._LAST_BIT) != 0

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

        # Groups this wide issue the next group's copies inside their body; a
        # unit's copies are in flight a unit ahead instead (`run_unit`).
        inline_issue_here = inline_issue and not unit
        inline = inline_issue_here and count > _INLINE_ISSUE_PAGES
        if not unit:

            @pl.when(w == 0)
            def _first():
                for d in range(ring - 1):
                    pl.when(d < num_groups)(functools.partial(start_group, d, d))

            @pl.when((nxt < num_groups) & jnp.logical_not(inline))
            def _next():
                start_group(nxt, nxt % ring)

        # One wait per set bit of `count` (the source only sizes the wait).
        n = 1
        while n <= pages_per_step and not waited:
            for copy in page_copies(half, 0, 0, n):
                pl.when((count & n) != 0)(copy.wait)
            n *= 2

        if whole_rows:
            lanes = pl.ds(pl.multiple_of(head_ref[0] * row_width, row_width), row_width)
            kv_refs = [
                buf_ref.at[half, g, :, :, :, lanes] for g in range(pages_per_step)
            ]
        else:
            kv_refs = [buf_ref.at[half, g] for g in range(pages_per_step)]
        # 4-bit `attend_t` groups run one body per width whether or not their
        # tile has other groups (`attend_t_any`): half the code.
        one_body = fp4 and transposed and MAX_FREE
        step_widths = _STEP_WIDTHS_4BIT_T if one_body else _STEP_WIDTHS
        widths = [n for n in step_widths if n < pages_per_step] + [pages_per_step]

        def zero_to(span):
            """Zeroes the value rows of the ring slots from the group's pages
            to `span`, which the step reads: they are stale (any bytes at first
            use), and p @ v multiplies them by p = 0 (`attend_t` reads the keys
            too: its additive mask needs finite scores)."""
            if not (direct or direct_t):
                return

            def zero(g, carry):
                lanes = row_width if direct_t else row_width // 2
                values = (
                    buf_ref.at[half, g]
                    .bitcast(jnp.uint32)
                    .at[0, :, 0, pl.ds(row_width - lanes, lanes)]
                )
                values[...] = jnp.zeros(values.shape, jnp.uint32)
                return carry

            jax.lax.fori_loop(count, span, zero, 0)

        if not unit:  # a unit's step width is its widest group's
            span = jnp.int32(widths[-1])
            for width in reversed(widths[:-1]):
                span = jnp.where(count <= width, width, span)
            zero_to(span)

        def init_state():
            if not MAX_FREE:
                m_ref[...] = jnp.full_like(m_ref, msa_attention._MASK_VALUE)
            l_ref[...] = jnp.zeros_like(l_ref)
            acc_ref[...] = jnp.zeros_like(acc_ref)

        if not unit and not one_body:
            pl.when(first & jnp.logical_not(last))(init_state)

        def put(out):  # [H * Bq, D] or [H, Bq, D]
            out = out.reshape(num_heads, block_q, head_dim)
            if store_out is None:
                o_ref[...] = out.astype(o_ref.dtype)
            else:
                store_out(out)

        def store_lse(lse):  # [rows, 1], or [rows, rows] equal along the lanes
            rows = lse.shape[0]
            if rows > _LANES:
                lse_ref[...] = lse[:, :1].T.reshape(lse_ref.shape)
                return
            # Lane-dense row: each row's value picked off the diagonal by a
            # sublane sum (a transpose was the finish's critical path).
            eye = jax.lax.broadcasted_iota(
                jnp.int32, (rows, rows), 0
            ) == jax.lax.broadcasted_iota(jnp.int32, (rows, rows), 1)
            row = jnp.sum(jnp.where(eye, lse, 0.0), axis=0, keepdims=True)
            lse_ref[...] = row.reshape(lse_ref.shape)

        def write(out, lse):
            put(out)
            store_lse(lse)

        def finish(m, denom, acc):
            """Output and lse from a tile's softmax state (log2 units, ``m``
            finite: the running max, or 0 with `MAX_FREE`). A row that saw no
            slot has ``denom = 0`` and ``acc = 0``: its output is 0 and its lse
            ``m + log(0) = -inf`` with no select (a seen row's ``denom`` is at
            least `_denom_floor`)."""
            write(
                acc * (1.0 / jnp.maximum(denom, _denom_floor())),
                m * _LN_2 + jnp.log(denom),
            )

        def exp_scores(s, single, axis):
            """``(p, m_prev, m_new)``: ``p = exp2(s - m_new)`` against the running
            max, or against 0 with `MAX_FREE`."""
            if MAX_FREE:
                return jnp.exp2(s), 0.0, 0.0
            # The running max starts at a finite floor (`_MASK_VALUE`), so a row
            # with no visible slot gets p = exp2(-inf) = 0, not exp2(nan).
            m_prev = m_ref[...] if not single else msa_attention._MASK_VALUE
            m_new = jnp.maximum(m_prev, jnp.max(s, axis=axis, keepdims=True))
            return jnp.exp2(s - m_new), m_prev, m_new

        def update(m_prev, m_new, l_cur, pv, single, done):
            """Folds a group's ``(l_cur, pv)`` into the tile's state, or ends the
            tile with ``done(m, denom, acc)``."""
            if single:
                done(m_new, l_cur, pv)
                return
            if MAX_FREE:
                l_new = l_ref[...] + l_cur
                acc_new = acc_ref[...] + pv
            else:
                alpha = jnp.exp2(m_prev - m_new)
                l_new = alpha * l_ref[...] + l_cur
                acc_new = alpha * acc_ref[...] + pv

            @pl.when(last)
            def _finalize():
                done(m_new, l_new, acc_new)

            @pl.when(jnp.logical_not(last))
            def _carry():
                if not MAX_FREE:
                    m_ref[...] = m_new
                l_ref[...] = l_new
                acc_ref[...] = acc_new

        def attend_t_parts(width, single):
            """`attend_t`'s ``(m_prev, m_new, l_cur, pv)`` over the group's first
            `width` slots (`single` only matters without `MAX_FREE`).

            `attend` for 8/16-bit caches on transposed scores ``s^T [keys,
            rows]``: the max and sum over keys are sublane reductions, the
            output accumulates as ``acc^T [D, rows]`` across the p @ v
            contraction, and ``m``, ``l`` are lane vectors over rows (fewer
            spills and pops: bench j-9081937a, c4 pool48 -5.4%). 4-bit pages
            (`FP4_TRANSPOSED`) fold their K scales into the bf16 rows and
            their V scales into v^T: worth it from 16-token tiles on; 8-token
            (decode) tiles keep `attend` (c4 b3 decode +0.4 ms per step
            transposed, twin j-fa6121d0 vs glue-only j-2abe5011; 4-bit 8-token
            body 2,055 -> 3,977 bundles for 32 pages)."""
            if inline_issue_here and width > _INLINE_ISSUE_PAGES:
                issue_next()
            mxu_mask = MXU_MASK and MAX_FREE and block_q <= head_dim
            row_slots = (
                jax.lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)
                if (packing == 4 and not direct_t) or fp4
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
                if fp4:
                    # Each key's scale (with the score scale) folds into its
                    # bf16 row, a scale column per page, not a multiply per
                    # score; each value's into v^T, where the scales are a lane
                    # vector broadcast down the sublanes: no lane broadcasts of
                    # a column (half the XLU work, kernel bench c48 2048-token
                    # call -2..-4 %). int4 rows and the sanitized scales are
                    # finite, so masked keys and values meet p = 0 harmlessly.
                    if fp4_rows == "head":
                        k_row = 2 * head_ref[0]
                        v_row = k_row + 1
                    else:
                        k_row, v_row = fp4_rows
                    rows, sk, sv = msa_attention._fp4_page(
                        kv_ref, scale_buf_ref.at[half, g], k_row, v_row
                    )
                    values.append((rows[:, head_dim:].T * sv).astype(jnp.bfloat16))
                    k_col = (sk * (sm_scale * _LOG2_E)).reshape(page_size, 1)
                    rows = (rows[:, :head_dim] * k_col).astype(jnp.bfloat16)
                elif direct_t:
                    # As stored (words bitcast, rows in `row_slots` order).
                    # Slots past a page's newest token are zero (`msa_write`),
                    # stale ring slots were zeroed above: every key is finite.
                    words = kv_ref.bitcast(jnp.uint32)[0, :, 0, :]
                    rows = pltpu.bitcast(words, kv_ref.dtype)
                    values.append(rows[:, head_dim:])
                elif packing == 4 and mxu_mask:  # an additive mask needs finite keys
                    rows = kv_ref[0].reshape(page_size, -1).astype(jnp.bfloat16)
                    rows = jnp.where(row_slots < seen, rows, 0)
                    values.append(rows[:, head_dim:])
                elif packing == 4:
                    rows = kv_ref[0].reshape(page_size, -1).astype(jnp.bfloat16)
                    values.append(jnp.where(row_slots < seen, rows[:, head_dim:], 0))
                else:
                    rows = msa_attention._page_rows(kv_ref, seen)
                    values.append(rows[:, head_dim:])
                if mxu_mask:
                    # [K | A]: A[key, t] = -_MASK_BIAS where token t does not
                    # see the key; lanes past Bq meet zero query columns.
                    limit = jnp.pad(limit, ((0, 0), (0, head_dim - block_q)))
                    masked = -_FP8_MASK if direct_t else -_MASK_BIAS
                    bias = jnp.where(row_slots < limit, 0.0, masked)
                    keys.append(
                        jnp.concatenate(
                            [rows[:, :head_dim], bias.astype(rows.dtype)], 1
                        )
                    )
                else:
                    keys.append(rows[:, :head_dim])
                    # Rows are head-major (h * Bq + t): heads share a token's mask.
                    masks.append(row_slots < jnp.tile(limit, (1, num_heads)))
            k = jnp.concatenate(keys, axis=0)
            v = jnp.concatenate(values, axis=1 if fp4 else 0)  # 4-bit: v^T [D, W * S]
            q = load_q().reshape(num_heads * block_q, head_dim)
            if not fp4:  # 4-bit keys carry the scale
                q = (q.astype(jnp.float32) * (sm_scale * _LOG2_E)).astype(q.dtype)
            if mxu_mask:  # [q | onehot(token)]
                shape = (num_heads * block_q, head_dim)
                row = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
                col = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
                onehot = jnp.where(col == row % block_q, 1.0, 0.0).astype(q.dtype)
                q = jnp.concatenate([q, onehot], axis=1)
            s = jax.lax.dot_general(
                k, q, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )  # [W * S, rows]
            if not mxu_mask:
                s = jnp.where(jnp.concatenate(masks, axis=0), s, -jnp.inf)
            p, m_prev, m_new = exp_scores(s, single, 0)
            # l = 1 . p comes out of the p @ v matmul as its row past the D
            # value rows (rows of ones streamed with v^T), not from a VPU add
            # per score vreg (the body is VALU-bound; kernel bench j-3e361d8c).
            # The sum is over the bf16 p the matmul weighs v with.
            if fp4:
                ones = jnp.ones((_SUM_ROWS, v.shape[1]), v.dtype)
                pv = jax.lax.dot_general(
                    jnp.concatenate([v, ones], axis=0),
                    p.astype(jnp.bfloat16),
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )  # [D + _SUM_ROWS, rows]
            else:
                ones = jnp.ones((v.shape[0], _SUM_ROWS), v.dtype)
                pv = jax.lax.dot_general(
                    jnp.concatenate([v, ones], axis=1),
                    p.astype(jnp.bfloat16),  # bf16 x fp8 dot for 8-bit `direct_t`
                    (((0,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )  # [D + _SUM_ROWS, rows]
            l_cur, pv = pv[head_dim : head_dim + 1], pv[:head_dim]
            return m_prev, m_new, l_cur, pv

        def attend_t(width, single):
            update(*attend_t_parts(width, single), single, finish_t)

        def attend_t_any(width):
            """`attend_t` (`MAX_FREE`) for a group whether or not it is its
            tile's only one: a first group carries nothing in, a last one
            writes the output."""
            _, _, l_cur, pv = attend_t_parts(width, False)
            l_new = jnp.where(first, l_cur, l_ref[...] + l_cur)
            acc_new = jnp.where(first, pv, acc_ref[...] + pv)

            @pl.when(last)
            def _finalize():
                finish_t(0.0, l_new, acc_new)

            @pl.when(jnp.logical_not(last))
            def _carry():
                l_ref[...] = l_new
                acc_ref[...] = acc_new

        def finish_t(m, denom, acc_t):
            """`finish` for `attend_t`'s lane-vector state."""
            put((acc_t * (1.0 / jnp.maximum(denom, _denom_floor()))).T)
            lse_ref[...] = (m * _LN_2 + jnp.log(denom)).reshape(lse_ref.shape)

        def group_limits(width, selected, local_page, local_limit):
            """Each token's limit on the group's first `width` pages, ``[Bq,
            W]`` (as `_page_limits`): the pages along the lanes, one compare
            per selection column."""
            slot = jax.lax.broadcasted_iota(jnp.int32, (1, width), 1)
            pages = jnp.zeros((1, width), jnp.int32)
            for g in range(width):
                page = work_page_ref[msa_attention._slot(meta, pages_per_step, g)]
                pages = jnp.where(slot == g, page, pages)
            hit = selected[:, 0:1] == pages
            for k in range(1, selected.shape[1]):
                hit = hit | (selected[:, k : k + 1] == pages)  # [Bq, W]
            limit = jnp.where(pages == local_page, local_limit, page_size)
            return jnp.where((pages >= 0) & hit, limit, 0)

        def attend(width, single):
            """One softmax update over the group's first `width` slots; a
            `single` group (the tile's only one) needs no carried state."""
            update(*attend_parts(width, single), single, finish)

        def attend_parts(width, single, staged=False):
            """`attend`'s ``(m_prev, m_new, l_cur, pv)`` over the group's first
            `width` slots; with `staged`, its QK, softmax and p @ v steps."""
            if inline_issue_here and width > _INLINE_ISSUE_PAGES:
                issue_next()
            # int4 and fp8 pages unpack in slot order; bf16 in the bitcast's order.
            row_slots = (
                jax.lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)
                if fp4
                else msa_attention._bitcast_slots(rows_per_page, packing)
            )  # [S, 1]
            slots = (
                jax.lax.broadcasted_iota(jnp.int32, (1, page_size), 1)
                if fp4
                else row_slots.T
            )  # [1, S]
            selected = selected_ref[...]
            local_page = local_page_ref[...]
            local_limit = local_limit_ref[...]
            keys, values, masks, k_scales, v_scales = [], [], [], [], []
            if limited:
                row = meta >> msa_attention._ROW_SHIFT
                limits = limit_ref[
                    row // groups_per_tile % tiles_per_step, row % groups_per_tile
                ]  # [Bq, pages_per_step]
            elif kernel_limits:
                limits = group_limits(width, selected, local_page, local_limit)
            if spreads:  # each page's [Bq, 1] limit broadcast over its S slots
                lim = limits[:, :width]
                for g in range(width):
                    masks.append(slots < lim[:, g : g + 1])
            for g, kv_ref in enumerate(kv_refs[:width]):
                if not spreads:
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
                if not spreads:
                    masks.append(slots < limit)  # [Bq, S]
            k = jnp.concatenate(keys, axis=0)
            v = jnp.concatenate(values, axis=0)
            mask = jnp.concatenate(masks, axis=1)
            mask = jnp.concatenate([mask] * num_heads, axis=0)

            scale = sm_scale * _LOG2_E
            q = load_q().reshape(num_heads * block_q, head_dim)
            if not fp4:
                q = (q.astype(jnp.float32) * scale).astype(q.dtype)

            def qk():
                s = jax.lax.dot_general(
                    q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
                )
                if fp4:  # x = int4 * scale: K scales the score columns, V scales p
                    s = s * (jnp.concatenate(k_scales, axis=1) * scale)
                return jnp.where(mask, s, -jnp.inf)

            def softmax(s):
                p, m_prev, m_new = exp_scores(s, single, 1)
                l_cur = jnp.sum(p, axis=1, keepdims=True)
                p_v = p * jnp.concatenate(v_scales, axis=1) if fp4 else p
                return m_prev, m_new, l_cur, p_v

            def p_at_v(m_prev, m_new, l_cur, p_v):
                pv = jax.lax.dot_general(
                    p_v.astype(jnp.bfloat16),
                    v,
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                return m_prev, m_new, l_cur, pv

            if staged:
                return qk, softmax, p_at_v
            return p_at_v(*softmax(qk()))

        def fold(l_cur, pv):
            """A unit's group's `update` (`MAX_FREE`), after its earlier groups'."""
            # Straight-line: every group carries its tile's sums and writes
            # its output (a later group of the tile overwrites it). The sums
            # stay replicated along the lanes (`l_ref` [rows, D], as the lane
            # reduction leaves them): no per-row broadcast for the division,
            # and the lse row comes off the diagonal of the same layout.
            carry = jnp.logical_not(first)
            wide = l_ref.shape[1] > 1  # fewer rows than lanes (`_kernel_scratch`)
            if wide:
                l_cur = jnp.broadcast_to(l_cur, l_ref.shape)
            l_new = jnp.where(carry, l_ref[...], 0.0) + l_cur
            acc_new = jnp.where(carry, acc_ref[...], 0.0) + pv
            l_ref[...] = l_new
            acc_ref[...] = acc_new
            if not wide:
                finish(0.0, l_new, acc_new)
                return
            put(acc_new * (1.0 / jnp.maximum(l_new, _denom_floor())))
            store_lse(jnp.log(l_new[:, : l_new.shape[0]]))

        def empty():
            put(jnp.zeros((num_heads * block_q, head_dim), jnp.float32))
            lse_ref[...] = jnp.full(lse_ref.shape, -jnp.inf, jnp.float32)

        def alone():
            """The group's body at its own width, as a unit of one."""
            for i, width in enumerate(widths):
                fits = count > (widths[i - 1] if i else 0)
                if width < pages_per_step:
                    fits = fits & (count <= width)

                @pl.when(fits)
                def _alone(width=width):
                    zero_to(width)
                    fold(*attend_parts(width, False)[2:])

            pl.when(count == 0)(empty)

        if unit:
            return types.SimpleNamespace(
                count=count,
                alone=alone,
                zero_to=zero_to,
                parts=lambda width: attend_parts(width, False)[2:],
                stages=lambda width: attend_parts(width, False, staged=True),
                fold=fold,
                empty=empty,
            )

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
            if one_body:
                pl.when(fits)(functools.partial(attend_t_any, width))
                continue
            body = attend_t if transposed else attend
            pl.when(fits & single)(functools.partial(body, width, True))
            pl.when(fits & jnp.logical_not(single))(
                functools.partial(body, width, False)
            )

        # A tile of padding tokens: its only group holds no page.
        pl.when(count == 0)(empty)

    def loop(w, carry):
        group(w)
        return carry

    b = pl.program_id(0)
    if steps_per_call is not None:
        b = b % steps_per_call
    start, end = block_start_ref[b], block_start_ref[b + 1]
    if unit_groups == 1:
        jax.lax.fori_loop(start, end, loop, 0)
        return
    widths = [n for n in _UNIT_STEP_WIDTHS if n < pages_per_step] + [pages_per_step]

    def run_unit(w0, size):
        """Groups ``w0 .. w0 + size - 1`` side by side, after issuing the page
        copies of every group up to ``w0 + ring - 1``: their ring slots held
        groups before `w0`, all done (the call's first unit: its first ring)."""
        lo = jnp.where(w0 == 0, 0, issued_ref[0])
        hi = jnp.minimum(w0 + ring, num_groups)

        def issue(g, carry):
            start_group(g, g % ring)
            return carry

        jax.lax.fori_loop(lo, hi, issue, 0)
        issued_ref[0] = jnp.maximum(lo, hi)
        groups = [group(w0 + i, unit=True) for i in range(size)]
        if size == 1:
            groups[0].alone()
            return
        widest = groups[0].count
        for g in groups[1:]:
            widest = jnp.maximum(widest, g.count)
        cap = pages_per_step if fp4 else UNIT_MAX_PAGES_8BIT
        unit_widths = [n for n in widths if n < cap] + [min(cap, pages_per_step)]
        for i, width in enumerate(unit_widths):
            fits = widest > (unit_widths[i - 1] if i else 0)
            if width < pages_per_step:
                fits = fits & (widest <= width)

            @pl.when(fits)
            def _body(width=width):
                for g in groups:
                    g.zero_to(width)
                if width <= _STAGED_UNIT_PAGES:
                    stages = [g.stages(width) for g in groups]
                    scores = [qk() for qk, _, _ in stages]
                    probs = [sm(s) for (_, sm, _), s in zip(stages, scores)]
                    parts = [p_at_v(*p)[2:] for (_, _, p_at_v), p in zip(stages, probs)]
                else:
                    parts = [g.parts(width) for g in groups]
                for g, (l_cur, pv) in zip(groups, parts):
                    g.fold(l_cur, pv)

        if cap < pages_per_step:

            @pl.when(widest > cap)
            def _alone():  # one body per width for all of them (code size)
                def alone(w, carry):
                    group(w, unit=True, waited=True).alone()
                    return carry

                jax.lax.fori_loop(w0, w0 + size, alone, 0)

        @pl.when(widest == 0)
        def _empty():
            for g in groups:
                g.empty()

    # The step's trailing groups of no page (padding tiles of a bucketed
    # batch) run alone, as `empty`: in a unit each would cost a body.
    real_end = jax.lax.while_loop(
        lambda e: (e > start) & (work_count_ref[jnp.maximum(e - 1, 0)] == 0),
        lambda e: e - 1,
        end,
    )
    # ... except those that complete the last unit (b3 decode: 3 tiles and
    # a padding tile); a shorter unit's body would only add code (twin c64
    # 0.977x with bodies per remainder size, j-4592aecb).
    full = (real_end - start + unit_groups - 1) // unit_groups
    full = jnp.minimum(full, (end - start) // unit_groups)

    def full_unit(u, carry):
        run_unit(start + u * unit_groups, unit_groups)
        return carry

    def one(w, carry):
        run_unit(w, 1)
        return carry

    jax.lax.fori_loop(0, full, full_unit, 0)
    jax.lax.fori_loop(start + full * unit_groups, end, one, 0)


def configure_max_free(score_bound: float) -> bool:
    """Sets `MAX_FREE` for a loaded model whose largest score bound (natural
    log units, every layer, after its keys' cache rounding) is
    `score_bound`, and returns it: off past `_MAX_FREE_SCORE_BOUND`."""
    global MAX_FREE
    MAX_FREE = score_bound <= _MAX_FREE_SCORE_BOUND
    return MAX_FREE


def _denom_floor() -> float:
    """The least softmax sum a tile divides by: a row that saw no slot has 0
    (and a zero output). Against a true max a seen row's sum is >= 1; against
    `MAX_FREE`'s 0 it is only positive."""
    return float(jnp.finfo(jnp.float32).tiny) if MAX_FREE else 1.0


def _merged_store_rows(block_q: int, num_heads: int, dtype, packed=False) -> int:
    """The packed output window (`_store_unaligned_rows`) a tile's output is
    merged into, or 1 for a plain store: a window is the output dtype's
    sublane tile (16 rows for bf16). Blocks of 16 heads lower their 8-row
    stores (decode tiles) when XLA gives the output 8-row tiles; 8 heads did
    not (c64 head-parallel, j-d318070b). `packed` merges whatever the heads
    (an output whose consumers XLA lays out in 16-row tiles)."""
    rows = 8 * 4 // jnp.dtype(dtype).itemsize
    return rows if block_q % rows and (packed or num_heads % rows) else 1


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
    packed_stores=False,
    aligned_work=False,
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

    With `aligned_work`, ``packed``'s sections start on 128-lane boundaries
    (`store_work_list`, a work list built in a kernel).

    ``calls > 1`` runs that many such calls in one grid, back to back:
    ``head`` then holds each call's pair (``int32[2 * calls]``) and
    ``packed`` each call's work list, end to end; `out` must be None."""
    pair = isinstance(q, tuple)  # two head-major arrays holding the heads in order
    qs = q if pair else (q,)
    if pair and (calls > 1 or len({x.shape for x in qs}) > 1):
        raise ValueError("a query pair serves one call of two equal halves")
    num_q_heads = sum(x.shape[0] for x in qs)
    _, num_tokens, head_dim = qs[0].shape
    num_heads = q_per_head
    topk = selected.shape[2]
    packed, tiles, tiles_per_step, *limits = work  # limits: `_page_limits`
    groups_per_tile = msa_attention._groups_per_tile(block_q, topk, pages_per_step)
    num_steps = tiles // tiles_per_step
    sections, length = _work_layout(
        tiles, groups_per_tile, pages_per_step, num_steps, aligned=aligned_work
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
        if pair:  # tile_offset, the query pair, then the rest
            refs = (refs[0], (refs[1], refs[2])) + refs[3:]
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
    transposed = _transposed(block_q, fp4)
    scale_inputs = (scale_cache,) if fp4 else ()
    limited = bool(limits)
    if limited and (calls > 1 or not fp4 or transposed):
        raise ValueError("page-limit tables serve one 4-bit call per grid")
    limit_specs = [
        pl.BlockSpec((tiles_per_step,) + x.shape[1:], lambda b, *_: (b, 0, 0, 0))
        for x in limits
    ]
    kernel_limits = _kernel_limits(kv_cache) and not limited and not transposed
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4,
        grid=(calls * num_steps,),
        in_specs=[
            *(
                pl.BlockSpec(
                    (
                        x.shape[0] * num_heads // num_q_heads,
                        tiles_per_step * block_q,
                        head_dim,
                    ),
                    head_major_block,
                )
                for x in qs
            ),
            pl.BlockSpec((None, tiles_per_step * block_q, topk), head_major_block),
            pl.BlockSpec((tiles_per_step * block_q, 1), tile_block),
            pl.BlockSpec((tiles_per_step * block_q, 1), tile_block),
            pl.BlockSpec(memory_space=pl.ANY),
            *(pl.BlockSpec(memory_space=pl.ANY) for _ in scale_inputs),
            *limit_specs,
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
            _unit_groups(transposed, tiles),
        ),
    )
    num_inputs = 4 + 4 + len(qs) + len(scale_inputs) + len(limits)
    return pl.pallas_call(
        functools.partial(
            kernel,
            sm_scale=sm_scale,
            aliased=aliased,
            tiles_per_step=tiles_per_step,
            groups_per_tile=groups_per_tile,
            transposed=transposed,
            inline_issue=inline_issue,
            limited=limited,
            kernel_limits=kernel_limits,
            packed_stores=packed_stores,
            unit_groups=_unit_groups(transposed, tiles),
            whole_rows=_whole_rows(
                kv_cache,
                scale_cache if fp4 else None,
                head_dim,
                _unit_groups(transposed, tiles),
            ),
            **({"fp4_rows": scale_rows} if fp4 else {}),
        ),
        grid_spec=grid_spec,
        out_shape=[
            jax.ShapeDtypeStruct((num_q_heads, num_tokens, head_dim), qs[0].dtype),
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
        *qs,
        selected,
        local_page,
        local_limit,
        kv_cache,
        *scale_inputs,
        *limits,
        *(out if aliased else ()),
    )


def _kernel_limits(kv_cache) -> bool:
    """`KERNEL_LIMITS` for a cache whose `attend` needs no per-page ``seen``
    (4- and 8-bit; 16-bit pages zero their unseen slots by it)."""
    return KERNEL_LIMITS and jnp.dtype(kv_cache.dtype).itemsize == 1


def _transposed(block_q: int, fp4: bool) -> bool:
    """Whether a call's tiles run `attend_t` (else `attend`)."""
    return (not fp4 or FP4_TRANSPOSED) and block_q >= msa_attention.DEFAULT_BLOCK_Q


def _unit_groups(transposed: bool, num_tiles: int) -> int:
    """`_kernel`'s ``unit_groups`` for a call of `num_tiles` tiles
    (`DECODE_UNIT_GROUPS`). None for one tile (twin c16 1-request decode
    7.6 -> 14.5 us per layer with units, j-6852856d). Calls of one KV head
    (c48) run units too since their groups' folds are straight-line: 4-bit
    8-token tiles, kernel bench at rho .9-.97, b8 -9 %, b10 -5 %, b22 -15 %,
    b34 -13 % (at rho .5: -2..-5 %)."""
    if transposed or not MAX_FREE or num_tiles == 1:
        return 1
    return DECODE_UNIT_GROUPS


def _decode_units(transposed: bool) -> bool:
    """Whether calls of several tiles run units (`_unit_groups`)."""
    return _unit_groups(transposed, 2) > 1


def _decode_tiles_per_step(block_q: int, fp4: bool, q_per_kv: int) -> int | None:
    """The most tiles a grid step takes in calls of `attend` tiles that run
    units (`UNIT_TILES_PER_STEP`, `_UNIT_STEP_ROWS`), else None."""
    if not _decode_units(_transposed(block_q, fp4)):
        return None
    return max(1, min(UNIT_TILES_PER_STEP, _UNIT_STEP_ROWS // (block_q * q_per_kv)))


def token_multiple(
    num_tokens: int, block_q: int, q_per_kv: int, dtype, fp4: bool
) -> int:
    """The multiple a call of `num_tokens` pads its tokens to: whole tiles,
    and for `attend` tiles running units of several tiles whole output
    windows (`_merged_store_rows`), so that one grid step can take them all
    (`_decode_tiles_per_step`). A single tile is a step of its own: padding
    it would only add pad and slice ops (twin c16 1-request decode, +7.5 us
    per layer)."""
    units = _decode_tiles_per_step(block_q, fp4, q_per_kv)
    if num_tokens <= block_q or units is None:
        return block_q
    return math.lcm(block_q, _merged_store_rows(block_q, q_per_kv, dtype))


def _whole_rows(kv_cache, scale_cache, head_dim: int, unit_groups: int) -> bool:
    """Whether `_kernel`'s page ring holds whole cache rows: 4-bit units over
    rows of two KV heads (c64, DCP4), whose head's lanes alone are a strided
    copy of 256-byte runs."""
    return (
        scale_cache is not None
        and unit_groups > 1
        and kv_cache.shape[-1] == 2 * (2 * head_dim)
    )


def _kernel_scratch(
    rows, head_dim, pages_per_step, kv_cache, scale_cache, transposed, unit_groups=1
):
    """`_kernel`'s scratch: m, l, acc (rows on sublanes for `attend`, lanes
    for `attend_t`), the page ring and its semaphores, and a 4-bit cache's
    scale-page ring. Units of groups (`unit_groups`) run with the next unit's
    pages in flight."""
    _, rows_per_page, packing, _ = kv_cache.shape
    if unit_groups > 1:
        ring = (1 + UNIT_PREFETCH) * unit_groups
    else:
        ring = 1 + (_FP4_PREFETCH if scale_cache is not None else 1)
    whole_rows = _whole_rows(kv_cache, scale_cache, head_dim, unit_groups)
    if transposed:
        softmax = [(1, rows), (1, rows), (head_dim, rows)]
    elif unit_groups > 1 and rows <= head_dim:  # `fold`'s lane-replicated sums
        softmax = [(rows, 1), (rows, head_dim), (rows, head_dim)]
    else:
        softmax = [(rows, 1), (rows, 1), (rows, head_dim)]
    scratch = [pltpu.VMEM(shape, jnp.float32) for shape in softmax]
    scratch += [
        pltpu.VMEM(
            (
                ring,
                pages_per_step,
                1,
                rows_per_page,
                packing,
                kv_cache.shape[-1] if whole_rows else 2 * head_dim,
            ),
            kv_cache.dtype,
        ),
        pltpu.SemaphoreType.DMA((ring, 1)),
    ]
    if scale_cache is not None:
        scratch += [
            pltpu.VMEM((ring, pages_per_step, 1) + scale_cache.shape[1:], jnp.float32),
            pltpu.SemaphoreType.DMA((ring, 1)),
        ]
    if unit_groups > 1:  # the groups issued
        scratch.append(pltpu.SMEM((1,), jnp.int32))
    return scratch


def planned_attention(
    q,
    swap,
    kv_cache,
    page_table,
    plan,
    plan_scalars,
    plan_inputs,
    *,
    rows_in_use,
    topk,
    sm_scale,
    block_q,
    pages_per_step,
    interpret=False,
    scale_cache=None,
    scale_rows=None,
):
    """`msa_sparse_attention_lse` (one KV head, head-major) of a DCP pair's
    decode step, over this rank's query heads ``q [Hl, T, D]`` and the
    peer's, which arrive as the first `Hl` rows of the permuted 16-bit
    buffer ``swap [Hl + 1, T, D]`` (bits; `msa_dcp.attend_and_merge`). The
    kernel computes the selection, local pages and work list itself:
    ``plan(*plan_scalar_refs (SMEM), swap_row_ref [T, D] (swap's last row),
    *plan_input_refs, keys_ref [T, K], local_page_ref [T, 1],
    local_limit_ref [T, 1], work_ref [1, length])``
    (`msa_dcp._decode_plan_kernel`) writes them into scratch, and the work
    list moves to SMEM. One kernel in place of a plan kernel and the
    attention kernel: each Pallas call costs ~1 us plus ~0.2 us per operand
    (standalone bench j-97afc7d4). The page table, and a one-tile step's
    queries, load while the plan runs: as a scalar prefetch and a pipelined
    block they landed before it (c16 b1 decode, a 64 KiB table: 5.73 -> 4.71
    us per call; c4 b1 4.89 -> 4.69, anvil). Of ``page_table [rows, n]``
    only the first ``rows_in_use(*plan_scalar_refs)`` rows load, the ones the
    step's keys index: the whole 64 KiB table outlasted the plan.

    Returns ``(out [2 Hl, T, D], lse [1, T / Bq, 1, 2 Hl * Bq], back [Hl +
    1, T, D] uint16)``: a tile's lse is a lane-dense row (``h * Bq + t``),
    and `back` is the permute buffer of the peer's heads: their output's
    bits and, in its last row, the lse rows' low 16 bits (row ``i`` for
    tile ``i``) then their high 16 bits (row ``T / Bq + i``), so no XLA op
    packs it (`msa_dcp.merge_pair_decode` reads it)."""
    num_heads, num_tokens, head_dim = q.shape
    num_q_heads = 2 * num_heads
    num_tiles = num_tokens // block_q
    sections, length, tiles_per_step, groups_per_tile = call_work_spec(
        num_tokens, num_q_heads, q.dtype, topk, block_q, pages_per_step
    )
    if (
        tiles_per_step != num_tiles
        or num_q_heads * block_q != _LANES
        or swap.shape != (num_heads + 1, num_tokens, head_dim)
        or swap.dtype != jnp.uint16
        or q.dtype.itemsize != 2
    ):
        raise ValueError(
            f"a pair's one-step decode of {num_q_heads} x {block_q}-row tiles in "
            f"128 lanes and a uint16 swap of {num_heads} + 1 heads, got q "
            f"{q.dtype} {q.shape}, swap {swap.dtype} {swap.shape}, "
            f"{num_tiles} tiles of {tiles_per_step} per step"
        )
    fp4 = scale_rows is not None
    scale_inputs = (scale_cache,) if fp4 else ()
    row = page_table.shape[1]
    num_scalars = len(plan_scalars)
    # One tile's queries load while the plan runs (as pipelined blocks they
    # landed before it); several tiles take the blocks, whose layout lowers
    # their tiles' 8-row query slices (a VMEM scratch's does not, E2003).
    async_q = num_tiles == 1

    def kernel(head_ref, offset_ref, *refs, **kwargs):
        scalar_refs, refs = refs[:num_scalars], refs[num_scalars:]
        (q_in, peer_in, table_hbm, swap_row), refs = refs[:4], refs[4:]
        num_plan = len(plan_inputs)
        plan_refs, refs = refs[:num_plan], refs[num_plan:]
        num_scratch = 8 + 2 * async_q
        refs, scratch = refs[:-num_scratch], refs[-num_scratch:]
        keys, local_page, local_limit, work_v, work_s = scratch[:5]
        table_ref, work_sem, load_sems = scratch[5:8]
        outs = 1 + len(scale_inputs)  # o_ref, lse_ref, back_ref after the inputs
        o_ref, lse_ref, back_ref = refs[outs : outs + 3]
        refs = refs[: outs + 2] + refs[outs + 3 :]
        if async_q:
            q_ref, peer_ref = scratch[8:]
            loads = [
                pltpu.make_async_copy(q_in, q_ref, load_sems.at[0]),
                pltpu.make_async_copy(
                    peer_in.at[pl.ds(0, num_heads)], peer_ref, load_sems.at[1]
                ),
            ]
        else:
            q_ref, peer_ref, loads = q_in, peer_in, []
        for copy in loads:
            copy.start()

        def table_row(r):
            return pltpu.make_async_copy(
                table_hbm.at[r], table_ref.at[pl.ds(r * row, row)], load_sems.at[2]
            )

        rows = rows_in_use(*scalar_refs)
        pl.loop(0, rows)(lambda r: table_row(r).start())
        plan(*scalar_refs, swap_row, *plan_refs, keys, local_page, local_limit, work_v)
        work = pltpu.make_async_copy(work_v.at[0], work_s, work_sem)
        work.start()
        work.wait()
        for copy in loads:
            copy.wait()
        pl.loop(0, rows)(lambda r: table_row(r).wait())
        unions, metas, counts, starts = (work_s.at[pl.ds(at, n)] for at, n in sections)
        _kernel(
            unions,
            metas,
            counts,
            table_ref,
            head_ref,
            offset_ref,
            (q_ref, peer_ref.bitcast(q.dtype)),
            keys,
            local_page,
            local_limit,
            *refs,
            block_start_ref=starts,
            **kwargs,
        )
        # The peer's heads go back as bits, their lse words as two planes.
        back_ref.bitcast(q.dtype)[:num_heads] = o_ref[num_heads:]
        words = pltpu.bitcast(lse_ref[0, :, 0, :], jnp.uint32)  # [tiles, 128]
        planes = jnp.concatenate([words & 0xFFFF, words >> 16])
        planes = jnp.pad(planes, ((0, num_tokens - 2 * num_tiles), (0, 0)))
        back_ref[num_heads] = planes.astype(jnp.uint16)

    def whole(x):
        return pl.BlockSpec(x.shape, lambda b, *_: (0,) * x.ndim)

    any_space = pl.BlockSpec(memory_space=pl.ANY)
    heads = pl.BlockSpec((num_heads, num_tokens, head_dim), lambda b, *_: (0, 0, 0))
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2 + num_scalars,
        grid=(1,),
        in_specs=[
            any_space if async_q else heads,
            any_space if async_q else heads,
            any_space,
            pl.BlockSpec((None, num_tokens, head_dim), lambda b, *_: (num_heads, 0, 0)),
            *(whole(x) for x in plan_inputs),
            any_space,
            *(any_space for _ in scale_inputs),
        ],
        out_specs=[
            pl.BlockSpec((num_q_heads, num_tokens, head_dim), lambda b, *_: (0, 0, 0)),
            pl.BlockSpec((1, num_tiles, 1, _LANES), lambda b, *_: (0, 0, 0, 0)),
            whole(swap),
        ],
        scratch_shapes=_kernel_scratch(
            num_q_heads * block_q,
            head_dim,
            pages_per_step,
            kv_cache,
            scale_cache if fp4 else None,
            transposed=False,
            unit_groups=_unit_groups(False, num_tiles),
        )
        + [
            pltpu.VMEM((num_tokens, topk), jnp.int32),
            pltpu.VMEM((num_tokens, 1), jnp.int32),
            pltpu.VMEM((num_tokens, 1), jnp.int32),
            pltpu.VMEM((1, length), jnp.int32),
            pltpu.SMEM((length,), jnp.int32),
            pltpu.SMEM((page_table.size,), jnp.int32),
            pltpu.SemaphoreType.DMA(()),
            pltpu.SemaphoreType.DMA((3,)),
        ]
        + (
            [pltpu.VMEM(q.shape, q.dtype), pltpu.VMEM(q.shape, jnp.uint16)]
            if async_q
            else []
        ),
    )
    return pl.pallas_call(
        functools.partial(
            kernel,
            sm_scale=sm_scale,
            tiles_per_step=tiles_per_step,
            groups_per_tile=groups_per_tile,
            unit_groups=_unit_groups(False, num_tiles),
            whole_rows=_whole_rows(
                kv_cache,
                scale_cache if fp4 else None,
                head_dim,
                _unit_groups(False, num_tiles),
            ),
            kernel_limits=_kernel_limits(kv_cache),
            **({"fp4_rows": scale_rows} if fp4 else {}),
        ),
        grid_spec=grid_spec,
        out_shape=[
            jax.ShapeDtypeStruct((num_q_heads, num_tokens, head_dim), q.dtype),
            jax.ShapeDtypeStruct((1, num_tiles, 1, _LANES), jnp.float32),
            jax.ShapeDtypeStruct(swap.shape, swap.dtype),
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
        jnp.zeros((2,), jnp.int32),
        jnp.zeros((1,), jnp.int32),
        *plan_scalars,
        q,
        swap,
        page_table,
        swap,
        *plan_inputs,
        kv_cache,
        *scale_inputs,
    )


def _tiles_per_step(
    num_tiles: int, rows: int = 1, align: int = 1, most: int | None = None
) -> int:
    """The most tiles (up to `TILES_PER_STEP`) a grid step can take evenly;
    with `align`, only counts whose `rows`-row tiles fill whole `align`-row
    output windows (`_store_unaligned_rows`), or one tile."""
    return next(
        n
        for n in range(most or TILES_PER_STEP, 0, -1)
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
    """A one-head planned call's packed work list over ``num_tokens`` rows
    of ``num_q_heads`` query heads, as `store_work_list` lays it out (aligned
    sections): ``(sections, length, tiles_per_step, groups_per_tile)``."""
    num_tiles = num_tokens // block_q
    align = _merged_store_rows(block_q, num_q_heads, dtype)
    tiles_per_step = _tiles_per_step(
        num_tiles,
        block_q,
        align,
        _decode_tiles_per_step(block_q, False, num_q_heads),
    )
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


def _packed_work(selected, block_q, pages_per_step, align=1, most=None):
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
    tiles_per_step = _tiles_per_step(num_tiles, block_q, align, most)
    groups_per_tile = msa_attention._groups_per_tile(
        block_q, selected.shape[2], pages_per_step
    )
    live = jnp.arange(meta.shape[1], dtype=jnp.int32)[None] < ends[:, None]
    tile = jnp.where(live, msa_attention._meta_tile(meta, groups_per_tile), num_tiles)
    bounds = jnp.arange(0, num_tiles + 1, tiles_per_step, dtype=jnp.int32)
    starts = jnp.sum(tile[:, None, :] < bounds[None, :, None], axis=2, dtype=jnp.int32)
    return jnp.concatenate([pages, meta, counts, starts], axis=1), tiles_per_step


def _page_limits(
    unions, selected, local_page, local_limit, block_q, pages_per_step, page_size
):
    """What `_kernel`'s `attend` computes per page as ``limit`` (the local
    page's visible slots, a whole page if selected, else 0), for every call's
    union slots: ``[calls, tiles, groups per tile, Bq, pages per step]`` from
    the unions (``-1`` past a tile's), ``selected [calls, T, K]`` and
    ``local_page`` / ``local_limit [calls, T]``."""
    calls, num_tokens, topk = selected.shape
    tiles = num_tokens // block_q
    pages = unions.reshape(calls, tiles, -1, 1, pages_per_step)
    sel = selected.reshape(calls, tiles, 1, block_q, 1, topk)
    hit = jnp.any(sel == pages[..., None], axis=-1)
    tok = (calls, tiles, 1, block_q, 1)
    limit = jnp.where(
        local_page.reshape(tok) == pages, local_limit.reshape(tok), page_size
    )
    return jnp.where((pages >= 0) & hit, limit, 0).astype(jnp.int32)


# Page keys a work list built in a kernel may hold: `_tile_unions` carries a
# key + 1 in three exact 8-bit bf16 limbs and its differences in f32.
MAX_PLAN_KEYS = 1 << 22


def _tile_unions(tile_keys):
    """Each tile's `msa_attention._dense_unions` in a kernel: ``tile_keys``,
    a ``[1, 128]`` row of page keys per tile (-1 for no page, else below
    `MAX_PLAN_KEYS`) -> per tile ``(union [1, 128]`` ascending, -1 past its
    pages, ``count [1, 128]`` its page count in every lane). The pairwise
    differences, each distinct page's slot and the page at each slot are
    three small bf16 matmuls of exact integers (sums below 2**24), issued a
    stage at a time over the tiles so the MXU pipelines them; compares and
    reductions over [128, 128] keys took lane reductions, which made the plan
    of a 24-tile step ~8 us (AOT critical path)."""
    f32, bf16 = jnp.float32, jnp.bfloat16
    iota = functools.partial(jax.lax.broadcasted_iota, jnp.int32)
    sub, lane = iota((_LANES, _LANES), 0), iota((_LANES, _LANES), 1)
    ones = jnp.ones((1, _LANES), jnp.int32)
    zeros = jnp.zeros((1, _LANES), jnp.int32)

    def limbs(x):
        return [x >> 16, (x >> 8) & 0xFF, x & 0xFF]

    diffs = []
    for keys in tile_keys:
        # diff[a, b] = key[b] - key[a] = sum_m lhs[m, a] * rhs[m, b]; no
        # page is 0, unequal to any page + 1.
        shifted = jnp.where(keys >= 0, keys + 1, 0)
        lhs = jnp.concatenate(
            [-x for x in limbs(shifted)] + [ones << 16, ones << 8, ones, zeros, zeros]
        )
        rhs = jnp.concatenate(
            [ones << 16, ones << 8, ones] + limbs(shifted) + [zeros, zeros]
        )
        diffs.append(
            jnp.dot(lhs.astype(bf16).T, rhs.astype(bf16), preferred_element_type=f32)
        )
    firsts, places = [], []
    all_ones = jnp.ones((_LANES, _LANES), bf16)
    for keys, diff in zip(tile_keys, diffs, strict=True):
        dup = (diff == 0) & (sub < lane)
        seen = jnp.max(jnp.where(dup, 1, 0), axis=0, keepdims=True)
        first = (keys >= 0) & (seen == 0)  # [1, b]: a page's first occurrence
        # Its slot: the first occurrences of smaller pages; and the count.
        smaller = jnp.concatenate([(diff > 0).astype(bf16), all_ones], axis=1)
        first8 = jnp.broadcast_to(first, (8, _LANES)).astype(bf16)
        place = jnp.dot(first8, smaller, preferred_element_type=f32)
        firsts.append(first)
        places.append(place[0:1].astype(jnp.int32))
    out = []
    for keys, first, place in zip(tile_keys, firsts, places, strict=True):
        slot = (first & (place[:, :_LANES] == sub)).astype(bf16)  # [slot, b]
        # The page at each slot, limb by limb, and whether the slot is filled.
        parts = jnp.concatenate(limbs(keys) + [ones, zeros, zeros, zeros, zeros])
        got = jax.lax.dot_general(
            parts.astype(bf16),
            slot,
            (((1,), (1,)), ((), ())),
            preferred_element_type=f32,
        ).astype(jnp.int32)  # [limb, slot]
        union = (got[0:1] << 16) | (got[1:2] << 8) | got[2:3]
        out.append((jnp.where(got[3:4] > 0, union, -1), place[:, _LANES:]))
    return out


def store_work_list(
    tile_keys,
    out_ref,
    *,
    pages_per_step,
    groups_per_tile,
    tiles_per_step,
    sections,
):
    """Inside a kernel: `_packed_work`'s row ``out_ref [1, length]`` for one
    call of tiles of 128 page keys each (``tile_keys``, a ``[1, 128]`` row
    per tile, -1 for no page): the same values as `msa_attention._work_list`
    (dense unions) and the step starts. Per-tile values are rows of one
    value in every lane, so no step reduces across lanes."""
    _store_work(
        _tile_unions(tile_keys),
        out_ref,
        pages_per_step=pages_per_step,
        groups_per_tile=groups_per_tile,
        tiles_per_step=tiles_per_step,
        sections=sections,
    )


def _store_work(
    unions, out_ref, *, pages_per_step, groups_per_tile, tiles_per_step, sections
):
    """`store_work_list` of the tiles' `_tile_unions`."""
    (unions_at, _), (meta_at, max_groups), (counts_at, _), (steps_at, bounds) = sections
    width = -(-max_groups // _LANES) * _LANES
    steps = -(-bounds // _LANES) * _LANES
    w = jax.lax.broadcasted_iota(jnp.int32, (1, width), 1)
    bound = jax.lax.broadcasted_iota(jnp.int32, (1, steps), 1) * tiles_per_step

    def wide(x, n):  # a one-value row as n lanes
        return jnp.broadcast_to(x[:, :1], (1, n))

    tile = jnp.zeros((1, width), jnp.int32)
    start = jnp.zeros((1, width), jnp.int32)
    tile_pages = jnp.zeros((1, width), jnp.int32)
    step_starts = jnp.zeros((1, steps), jnp.int32)
    ends = []
    end_at = jnp.zeros((1, _LANES), jnp.int32)
    for i, (union, pages) in enumerate(unions):
        out_ref[:, unions_at + i * _LANES : unions_at + (i + 1) * _LANES] = union
        groups = jnp.maximum(-(-pages // pages_per_step), 1)
        first_at = end_at
        end_at = end_at + groups
        here = wide(first_at, width) <= w
        if i > 0:
            tile += jnp.where(here, 1, 0)
        start = jnp.where(here, wide(first_at, width), start)
        live = here & (w < wide(end_at, width))
        tile_pages = jnp.where(live, wide(pages, width), tile_pages)
        step_starts += jnp.where(i < bound, wide(groups, steps), 0)
        ends.append(end_at)
    total = wide(end_at, width)
    end = total
    for end_at in reversed(ends):  # the least end past w
        end = jnp.where(wide(end_at, width) > w, wide(end_at, width), end)
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
    out_ref[:, steps_at : steps_at + steps] = jnp.where(
        jax.lax.broadcasted_iota(jnp.int32, (1, steps), 1) < bounds, step_starts, 0
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
    if isinstance(q, tuple):
        if not q_head_major:
            raise ValueError("a query pair comes head-major")
        num_q_heads, num_tokens = sum(x.shape[0] for x in q), q[0].shape[1]
    else:
        if not q_head_major:
            q = q.transpose(1, 0, 2)
        num_q_heads, num_tokens, _ = q.shape
    num_heads = selected.shape[1]
    if num_tokens == 0 and isinstance(q, tuple):
        raise ValueError("a query pair needs tokens")
    if num_tokens == 0:
        if head_major:
            return q, jnp.zeros((num_q_heads, 0), jnp.float32)
        return q.transpose(1, 0, 2), jnp.zeros((0, num_q_heads), jnp.float32)
    call = functools.partial(
        _attention_call,
        inline_issue=num_heads == 1,
        q_per_head=num_q_heads // num_heads,
        sm_scale=sm_scale,
        block_q=block_q,
        pages_per_step=pages_per_step,
        interpret=interpret,
        scale_cache=scale_cache,
        scale_rows=scale_rows,
    )
    dtype = (q[0] if isinstance(q, tuple) else q).dtype
    step, num_chunks, pad, selected, (packed, tiles_per_step) = _chunk_work(
        selected, num_q_heads, dtype, block_q, pages_per_step, scale_rows is not None
    )
    q = jax.tree.map(lambda x: jnp.pad(x, ((0, 0), (0, pad), (0, 0))), q)
    local_page = jnp.pad(local_page, (0, pad), constant_values=-1)[:, None]
    local_limit = jnp.pad(local_limit, (0, pad))[:, None]
    limits = None
    if (
        LIMIT_TABLE
        and scale_rows is not None
        and not (FP4_TRANSPOSED and block_q >= msa_attention.DEFAULT_BLOCK_Q)
        and step >= _LIMIT_TABLE_MIN_TOKENS
        and not (num_chunks == 1 and num_heads > 1)
    ):  # 4-bit prefill-size calls, one per grid
        calls = num_heads * num_chunks
        groups_per_tile = msa_attention._groups_per_tile(
            block_q, selected.shape[2], pages_per_step
        )
        limits = _page_limits(
            packed[:, : step // block_q * groups_per_tile * pages_per_step],
            selected.reshape(calls, step, selected.shape[2]),
            jnp.tile(local_page.reshape(num_chunks, step), (num_heads, 1)),
            jnp.tile(local_limit.reshape(num_chunks, step), (num_heads, 1)),
            block_q,
            pages_per_step,
            msa_attention.page_tokens(kv_cache),
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
            (packed[i], step // block_q, tiles_per_step)
            + (() if limits is None else (limits[i],)),
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


def _chunk_work(
    selected, num_q_heads, dtype, block_q, pages_per_step, fp4=False, packed=False
):
    """`_chunked_call`'s calls over ``selected [T, G, K]``: ``(step,
    chunks, pad, selected [G, chunks * step, K], (packed, tiles_per_step))``
    with every call's work list built in one batched pass ahead of the loop
    (built per call inside it, each call paid its own two union sorts and
    glue: ~9 us per call, x G x 4 calls per layer)."""
    num_tokens, num_heads, topk = selected.shape
    limit = msa_attention.MAX_TOKENS_PER_CALL
    step = min(num_tokens, max(block_q, limit - limit % block_q))
    num_chunks = -(-num_tokens // step)
    pad = num_chunks * step - num_tokens
    selected = jnp.pad(
        selected, ((0, pad), (0, 0), (0, 0)), constant_values=-1
    ).transpose(1, 0, 2)
    packed = _packed_work(
        selected.reshape(num_heads * num_chunks, step, topk),
        block_q,
        pages_per_step,
        align=_merged_store_rows(block_q, num_q_heads // num_heads, dtype, packed),
        most=_decode_tiles_per_step(block_q, fp4, num_q_heads // num_heads)
        or (
            PREFILL_TILES_PER_STEP_4BIT
            if fp4
            and block_q < msa_attention.DEFAULT_BLOCK_Q
            and num_tokens > msa_attention.SMALL_BATCH_TOKENS
            else None
        ),
    )
    return step, num_chunks, pad, selected, packed


def _union_ratio(selected, tile):
    """A step's mean page union per `tile`-token tile over its mean pages per
    token (``selected [T, G, K]`` page keys, -1 none, `T` a multiple of
    `tile`), from each tile's bitmap of its keys modulo `_UNION_BITS`
    (exact while a tile's keys span fewer: a request's blocks on a rank). A
    few vector passes: no work list, no sort."""
    num_tokens, num_heads, topk = selected.shape
    keys = selected.transpose(1, 0, 2).reshape(num_heads, -1, tile * topk)
    valid = keys >= 0
    words = _UNION_BITS // 32
    word = jax.lax.broadcasted_iota(jnp.int32, (1, 1, 1, words), 3)
    bit = jnp.left_shift(jnp.uint32(1), (keys & 31).astype(jnp.uint32))
    onehot = jnp.where(
        valid[..., None] & ((keys[..., None] >> 5) % words == word),
        bit[..., None],
        jnp.uint32(0),
    )
    bitmap = jax.lax.reduce(onehot, jnp.uint32(0), jax.lax.bitwise_or, (2,))
    union = jnp.sum(jax.lax.population_count(bitmap), axis=-1)  # [G, tiles]
    tiles = jnp.maximum(jnp.sum(union > 0), 1)
    per_token = jnp.sum(valid) / jnp.maximum(jnp.sum(jnp.any(selected >= 0, -1)), 1)
    return jnp.sum(union) / tiles / jnp.maximum(per_token, 1)


# Prefill-size calls (past `WIDE_TILE_MIN_TOKENS` tokens) of caches whose
# tiles run `attend_t` may take `WIDE_BLOCK_Q`-token tiles: a tile's page
# union and its per-page work then serve more rows, which pays while the
# step's tokens choose much the same pages (served c48 selections look at
# least that local). A step takes the wide tile when its mean union per
# wide tile is at most `_WIDE_TILE_MAX_RATIO[kind]` times its tokens' mean
# pages, under each kind's crossover in the kernel bench (synthetic
# selections, 8192 tokens, j-9c5505d9): int4 8 -> 64-token tiles 0.98 ->
# 0.50 ms at ratio ~1.7, 1.08 -> 0.81 at ~4, even at ~5.9 (c48 shape) and ~4.4
# (c64 head-parallel shape); fp8 32 -> 64 0.60 -> 0.50 at ~1.6, even at ~3.3.
WIDE_BLOCK_Q = 64
# "auto" (as above), "on" (every such call) or "off" (`TPU_MINIMAX_M3_MSA_WIDE_TILES`).
WIDE_TILES = "off"
WIDE_TILE_MIN_TOKENS = msa_attention.SMALL_BATCH_TOKENS
_WIDE_TILE_MAX_RATIO = {"fp4": 4.0, "8bit": 3.0}
# Bitmap width of `_union_ratio`: a rank's blocks of a 256K-token request
# at DCP 2 (1,024 at 128-token pages; head-parallel calls see 2,048, where
# far blocks alias, lowering the estimate only for very wide tiles).
_UNION_BITS = 1024


def _wide_tile_kind(kv_cache, scale_cache, head_dim):
    """The `_WIDE_TILE_MAX_RATIO` key of a call's cache, or None where wide
    tiles were not measured (16-bit pages) or do not run `attend_t` with the
    MXU mask (which needs ``WIDE_BLOCK_Q <= D``)."""
    if not (MAX_FREE and MXU_MASK) or head_dim < WIDE_BLOCK_Q:
        return None
    if scale_cache is not None:
        return "fp4" if FP4_TRANSPOSED else None
    return "8bit" if jnp.dtype(kv_cache.dtype).itemsize == 1 else None


def _adaptive_call(
    q,
    kv_cache,
    selected,
    local_page,
    local_limit,
    page_table,
    head,
    *,
    block_q,
    pages_per_step,
    scale_cache,
    head_major,
    q_head_major,
    **kwargs,
):
    """`_chunked_call` at `block_q` or, when the step's selections are local
    enough (`_WIDE_TILE_MAX_RATIO`), at `WIDE_BLOCK_Q`; ``q`` etc. padded to
    whole wide tiles. Both tile sizes give the same results up to the order
    of the softmax sums."""
    num_tokens = selected.shape[0]
    kind = _wide_tile_kind(kv_cache, scale_cache, jax.tree.leaves(q)[0].shape[2])
    if isinstance(q, tuple):
        kind = None  # query pairs take the regular tile
    call = functools.partial(
        _chunked_call,
        kv_cache=kv_cache,
        page_table=page_table,
        first_head=head,
        pages_per_step=pages_per_step,
        scale_cache=scale_cache,
        head_major=head_major,
        q_head_major=q_head_major,
        **kwargs,
    )
    if (
        WIDE_TILES == "off"
        or kind is None
        or block_q >= WIDE_BLOCK_Q
        or num_tokens <= WIDE_TILE_MIN_TOKENS
    ):
        return call(
            q,
            selected=selected,
            local_page=local_page,
            local_limit=local_limit,
            block_q=block_q,
        )
    q, selected, local_page, local_limit = _pad_tokens(
        q, selected, local_page, local_limit, WIDE_BLOCK_Q, q_head_major
    )
    operands = (q, selected, local_page, local_limit)

    def wide(operands):
        q, selected, local_page, local_limit = operands
        return call(
            q,
            selected=selected,
            local_page=local_page,
            local_limit=local_limit,
            block_q=WIDE_BLOCK_Q,
        )

    def regular(operands):
        q, selected, local_page, local_limit = operands
        return call(
            q,
            selected=selected,
            local_page=local_page,
            local_limit=local_limit,
            block_q=block_q,
        )

    if WIDE_TILES == "on":
        out, lse = wide(operands)
    else:
        ratio = _union_ratio(selected, WIDE_BLOCK_Q)
        out, lse = jax.lax.cond(
            ratio <= _WIDE_TILE_MAX_RATIO[kind], wide, regular, operands
        )
    if head_major:
        return out[:, :num_tokens], lse[:, :num_tokens]
    return out[:num_tokens], lse[:num_tokens]


def _pad_tokens(q, selected, local_page, local_limit, block_q, q_head_major=False):
    pad = -selected.shape[0] % block_q
    if pad:

        def pad_q(x):
            widths = [(0, 0)] * x.ndim
            widths[1 if q_head_major else 0] = (0, pad)
            return jnp.pad(x, widths)

        q = jax.tree.map(pad_q, q)
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
    num_tokens, head_dim = selected.shape[0], jax.tree.leaves(q)[0].shape[2]
    msa_attention.check_cache(kv_cache, scale_cache)
    _check_rows(kv_cache, head_dim, head)
    leaf = jax.tree.leaves(q)[0]
    q_per_kv = sum(x.shape[0 if q_head_major else 1] for x in jax.tree.leaves(q))
    multiple = token_multiple(
        num_tokens, block_q, q_per_kv, leaf.dtype, scale_cache is not None
    )
    q, selected, local_page, local_limit = _pad_tokens(
        q, selected, local_page, local_limit, multiple, q_head_major
    )
    out, lse = _adaptive_call(
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
    multiple = token_multiple(
        num_tokens,
        block_q,
        num_q_heads // num_heads,
        q.dtype,
        scale_cache is not None,
    )
    q, selected, local_page, local_limit = _pad_tokens(
        q, selected, local_page, local_limit, multiple, q_head_major
    )
    out, lse = _adaptive_call(
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


# Decode steps of up to this many rows build the per-KV-head calls' work
# lists in a kernel (`_head_work_lists`) instead of XLA (two sorts and their
# glue, `_chunk_work`). The kernel keeps every tile's [128, 128] f32 key
# differences live at once: ~6 MiB of VMEM for 2 heads at 384 rows, growing
# with both.
HEAD_PLAN_MAX_TOKENS = 384


def _tile_key_rows(keys, block_q):
    """Each ``block_q``-token tile's keys of ``keys [T, K]`` as one
    ``[1, block_q * K]`` row (token ``t``'s at lanes ``[t * K, (t + 1) *
    K)``): two transposes repeat every token's keys across the lanes, then a
    select per token."""
    num_tokens, topk = keys.shape
    width = block_q * topk
    repeated = jnp.concatenate([keys.T] * block_q).T  # [T, width]
    token = jax.lax.broadcasted_iota(jnp.int32, (1, width), 1) // topk
    rows = []
    for i in range(num_tokens // block_q):
        row = jnp.full((1, width), -1, jnp.int32)
        for t in range(block_q):
            j = i * block_q + t
            row = jnp.where(token == t, repeated[j : j + 1], row)
        rows.append(row)
    return rows


def _head_work_lists(
    selected, *, block_q, pages_per_step, tiles_per_step, interpret=False
):
    """`_packed_work` of each KV head's ``selected [G, T, K]`` (page keys, -1
    for none, below `MAX_PLAN_KEYS`) built in one kernel, in
    `store_work_list`'s aligned layout: ``[G, length]``."""
    num_heads, num_tokens, topk = selected.shape
    num_tiles = num_tokens // block_q
    groups_per_tile = msa_attention._groups_per_tile(block_q, topk, pages_per_step)
    sections, length = _work_layout(
        num_tiles,
        groups_per_tile,
        pages_per_step,
        num_tiles // tiles_per_step,
        aligned=True,
    )

    def kernel(keys_ref, work_ref):
        # Every head's tiles a stage at a time (`_tile_unions`), then each
        # head's work list.
        rows = [_tile_key_rows(keys_ref[h], block_q) for h in range(num_heads)]
        unions = _tile_unions(sum(rows, []))
        for h in range(num_heads):
            _store_work(
                unions[h * num_tiles : (h + 1) * num_tiles],
                work_ref.at[h],
                pages_per_step=pages_per_step,
                groups_per_tile=groups_per_tile,
                tiles_per_step=tiles_per_step,
                sections=sections,
            )

    work = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_heads, 1, length), jnp.int32),
        interpret=interpret,
        name="minimax_m3_msa_head_work_lists",
    )(selected)
    return work.reshape(num_heads, length)


def msa_sparse_attention_lse_head_calls(
    q: jax.Array,
    kv_cache: jax.Array,
    selected: jax.Array,
    local_page: jax.Array,
    local_limit: jax.Array,
    page_table: jax.Array,
    head_order: jax.Array,
    *,
    sm_scale: float,
    block_q: int,
    pages_per_step: int,
    scale_cache: jax.Array,
    interpret: bool = False,
) -> list:
    """`msa_sparse_attention_lse_heads` of a 4-bit cache as one kernel call
    per KV head, in the traced order ``head_order`` (``int32[G]``), so a
    consumer of an earlier head's partial (a collective) can overlap the
    later calls. Returns one thunk per call: ``call(after=None)`` runs it
    after the arrays `after` exist (an optimization barrier) and returns
    ``(out [G * Hg, T, D], lse [G * Hg, T])`` head-major, of which only that
    head's query heads are set. One chunk (decode-sized steps) only."""
    num_tokens, num_heads = selected.shape[:2]
    num_q_heads, head_dim = q.shape[1], q.shape[2]
    msa_attention.check_cache(kv_cache, scale_cache)
    if _check_rows(kv_cache, head_dim) < num_heads or num_q_heads % num_heads:
        raise ValueError(
            f"{num_heads} KV heads of {num_q_heads} query heads do not fit cache "
            f"rows {kv_cache.shape}."
        )
    # Packed stores (below) need whole output windows per grid step: an even
    # tile count.
    store_rows = _merged_store_rows(block_q, 1, q.dtype, True)
    q, selected, local_page, local_limit = _pad_tokens(
        q, selected, local_page, local_limit, max(block_q, store_rows)
    )
    q = q.transpose(1, 0, 2)
    planned = (
        selected.shape[0] <= HEAD_PLAN_MAX_TOKENS
        and fits_work_row(block_q, selected.shape[2], pages_per_step)
        and page_table.size <= MAX_PLAN_KEYS
    )
    if planned:  # the work lists from a kernel, no XLA sorts
        step, num_chunks, pad = selected.shape[0], 1, 0
        # `_chunk_work`'s tiles per step for these calls (several heads,
        # packed stores, decode-sized): keep the two in step.
        tiles_per_step = _tiles_per_step(
            step // block_q,
            block_q,
            store_rows,
            _decode_tiles_per_step(block_q, True, num_q_heads // num_heads),
        )
        selected = selected.transpose(1, 0, 2)
        packed = _head_work_lists(
            selected,
            block_q=block_q,
            pages_per_step=pages_per_step,
            tiles_per_step=tiles_per_step,
            interpret=interpret,
        )
    else:
        step, num_chunks, pad, selected, (packed, tiles_per_step) = _chunk_work(
            selected, num_q_heads, q.dtype, block_q, pages_per_step, True, packed=True
        )
    if num_chunks != 1:
        raise ValueError(f"one chunk per head, got {num_chunks} of {step} tokens")
    q = jnp.pad(q, ((0, 0), (0, pad), (0, 0)))
    local_page = jnp.pad(local_page, (0, pad), constant_values=-1)[:, None]
    local_limit = jnp.pad(local_limit, (0, pad))[:, None]

    def call(h, after=None):
        head = head_order[h]
        q_h = q if after is None else jax.lax.optimization_barrier((q, after))[0]
        out, lse = _attention_call(
            q_h,
            kv_cache,
            selected,
            local_page,
            local_limit,
            page_table,
            jnp.stack([head, head]).astype(jnp.int32),
            jnp.zeros((1,), jnp.int32),
            None,
            (
                jax.lax.dynamic_index_in_dim(packed, head, 0, False),
                step // block_q,
                tiles_per_step,
            ),
            q_per_head=num_q_heads // num_heads,
            sm_scale=sm_scale,
            block_q=block_q,
            pages_per_step=pages_per_step,
            interpret=interpret,
            scale_cache=scale_cache,
            scale_rows="head",
            # The caller's selects lay the output out in 16-row tiles.
            packed_stores=True,
            aligned_work=planned,
        )
        num_blocks, num_tiles = lse.shape[:2]
        lse = lse.reshape(num_blocks, num_tiles, -1, block_q).transpose(0, 2, 1, 3)
        lse = lse.reshape(num_q_heads, num_tiles * block_q)
        return out[:, :num_tokens], lse[:, :num_tokens]

    return [functools.partial(call, h) for h in range(num_heads)]
