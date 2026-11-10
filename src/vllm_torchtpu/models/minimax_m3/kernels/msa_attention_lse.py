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
_STEP_WIDTHS = (4, 8)
# Tiles per grid step: a step runs the groups of this many tiles in a loop.
# One step per group paid a per-step cost worth ~2-3% of the call (pool48
# selections, 1-2 groups a tile; bench j-caec1354).
TILES_PER_STEP = 4
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
):
    """`msa_attention._kernel_v2` that also writes ``lse = m + log(l)``
    (``-inf`` for a row that sees no slot; its output is 0). Scores, the
    running max ``m`` and ``l`` are in log2 units (scaled by ``sm_scale *
    log2(e)``): exp2 needs no multiply per score. 4-bit caches fold the scale
    into the K scales of the score columns; others into the queries, which
    then round to bf16 once more before the QK dot (an XLA multiply of the
    whole head-major query outside the kernel was not fused, ~40 us per layer
    in the c4 twin, j-18e7ec75).

    Each step copies the next group's pages into the other half of a
    double buffer itself, one loop iteration per page it holds: unrolled
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
    tail = refs[len(refs) - (9 if fp4 else 7) :]
    o_ref, lse_ref, m_ref, l_ref, acc_ref, buf_ref, sem_ref = tail[:7]
    scale_hbm_ref = refs[0] if fp4 else None
    scale_buf_ref, scale_sem_ref = tail[7:] if fp4 else (None, None)
    pages_per_step = buf_ref.shape[1]
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
        rows_per_page, packing, row_width = buf_ref.shape[3:]
        page_size = rows_per_page * (msa_fp4.TOKENS_PER_WORD if fp4 else packing)
        half = w % 2
        first = (meta & msa_attention._FIRST_BIT) != 0
        last = (meta & msa_attention._LAST_BIT) != 0

        def page_copies(half, g, page):
            src = kv_hbm_ref.at[
                pl.ds(page, 1), :, :, pl.ds(head_ref[0] * row_width, row_width)
            ]
            copies = [pltpu.make_async_copy(src, buf_ref.at[half, g], sem_ref.at[half, g])]
            if fp4:
                copies.append(
                    pltpu.make_async_copy(
                        scale_hbm_ref.at[pl.ds(page, 1)],
                        scale_buf_ref.at[half, g],
                        scale_sem_ref.at[half, g],
                    )
                )
            return copies

        def start_group(group, half):
            base = msa_attention._slot(work_meta_ref[group], pages_per_step, 0)

            def start(g, carry):
                for copy in page_copies(half, g, page_table_ref[work_page_ref[base + g]]):
                    copy.start()
                return carry

            jax.lax.fori_loop(0, work_count_ref[group], start, 0)

        @pl.when(w == 0)
        def _first():
            start_group(0, 0)

        @pl.when(w + 1 < num_groups)
        def _next():
            start_group(w + 1, 1 - half)

        def wait(g, carry):
            for copy in page_copies(half, g, 0):  # the source only sizes the wait
                copy.wait()
            return carry

        jax.lax.fori_loop(0, count, wait, 0)

        kv_refs = [buf_ref.at[half, g] for g in range(pages_per_step)]

        @pl.when(first & jnp.logical_not(last))
        def _init():
            m_ref[...] = jnp.full_like(m_ref, msa_attention._MASK_VALUE)
            l_ref[...] = jnp.zeros_like(l_ref)
            acc_ref[...] = jnp.zeros_like(acc_ref)

        def write(out, lse):
            o_ref[...] = out.reshape(num_heads, block_q, head_dim).astype(o_ref.dtype)
            lse_ref[...] = lse.T.reshape(lse_ref.shape)  # lane-dense row

        def finish(m, denom, acc):
            """Output and lse from a tile's softmax state (log2 units, ``m``
            finite). A row that saw no slot has ``denom = 0`` and ``acc = 0``:
            its output is 0 and its lse ``m + log(0) = -inf`` with no select (a
            seen row has ``denom >= 1``)."""
            write(acc * (1.0 / jnp.maximum(denom, 1.0)), m * _LN_2 + jnp.log(denom))

        def attend(width, single):
            """One softmax update over the group's first `width` slots; a
            `single` group (the tile's only one) needs no carried state."""
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
                p_v.astype(v.dtype),
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
        # Run at the narrowest width holding the group's pages. A step costs per
        # score column, filled or not (~2.3 us at 16 pages, ~1.1 us at 4, twin
        # standalone j-952e8a64), and a 16-token tile's union is ~8-12 of its
        # rank's pages at DCP 2, ~2-8 at DCP 8.
        single = first & last
        widths = [n for n in _STEP_WIDTHS if n < pages_per_step] + [pages_per_step]
        for i, width in enumerate(widths):
            fits = count > (widths[i - 1] if i else 0)
            if width < pages_per_step:
                fits = fits & (count <= width)
            pl.when(fits & single)(functools.partial(attend, width, True))
            pl.when(fits & jnp.logical_not(single))(functools.partial(attend, width, False))

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
            block_start_ref=packed_ref.at[pl.ds(union_len + 2 * max_groups, num_steps + 1)],
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
    scale_inputs = (scale_cache,) if fp4 else ()
    scale_scratch = (
        (
            pltpu.VMEM((2, pages_per_step, 1) + scale_cache.shape[1:], jnp.float32),
            pltpu.SemaphoreType.DMA((2, pages_per_step)),
        )
        if fp4
        else ()
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4,
        grid=(num_steps,),
        in_specs=[
            pl.BlockSpec((num_heads, tiles_per_step * block_q, head_dim), head_major_block),
            pl.BlockSpec((None, tiles_per_step * block_q, topk), head_major_block),
            pl.BlockSpec((tiles_per_step * block_q, 1), tile_block),
            pl.BlockSpec((tiles_per_step * block_q, 1), tile_block),
            pl.BlockSpec(memory_space=pl.ANY),
            *(pl.BlockSpec(memory_space=pl.ANY) for _ in scale_inputs),
            *in_place,
        ],
        out_specs=[
            pl.BlockSpec((num_heads, tiles_per_step * block_q, head_dim), head_major_block),
            pl.BlockSpec((1, tiles_per_step, 1, num_heads * block_q), lse_block),
        ],
        scratch_shapes=[
            pltpu.VMEM((num_heads * block_q, 1), jnp.float32),
            pltpu.VMEM((num_heads * block_q, 1), jnp.float32),
            pltpu.VMEM((num_heads * block_q, head_dim), jnp.float32),
            pltpu.VMEM(
                (2, pages_per_step, 1, rows_per_page, packing, row_width),
                kv_cache.dtype,
            ),
            pltpu.SemaphoreType.DMA((2, pages_per_step)),
            *scale_scratch,
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
    )
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
    )
    if head_major:
        return out[:, :num_tokens], lse[:, :num_tokens]
    return out[:num_tokens], lse[:num_tokens]
