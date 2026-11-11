# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DeepSeek-V4.1 attention output projection: wo_a, then wo_b.

Computes ``buffered_matmul(wo_a_projection(x, wa, sa, None), wb, sb)`` in one
call. The wo_a result stays in VMEM instead of a write to HBM and a read back,
and the wo_b weight stream starts while the last wo_a groups compute. Both
matmuls are hand-scheduled on the two MXUs, and each wo_a group's quantize
runs inside the previous group's matmul.
Arithmetic and rounding match the two buffered kernels: wo_a quantizes each
group row of 4096 to FP8 with a BF16 scale; wo_b quantizes each full row of
8192 with an FP32 scale, whose amax the wo_a epilogues collect.

One kernel step holds at most MAX_TILE tokens in VMEM. Longer inputs
(prefill) run equal token tiles as grid steps of one kernel, then a shorter
last tile as a second, static call, each writing its rows of one shared
output. Every quantization is per row, so tiling does not change any bits.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import get_tpu_version
from vllm_torchtpu.layers.core.quantization import FP8_E4M3_MAX

HEADS_PER_GROUP = 8
# wo_a groups (x and weight slice) in flight.
GROUP_DEPTH = 3
# wo_b output columns per weight window.
WINDOW = 1024
# wo_b weight windows in flight. Window i + WINDOW_DEPTH refills after window
# i's matmul, so each refill has WINDOW_DEPTH - 1 windows of compute to land.
WINDOW_DEPTH = 3
# Rows per MXU accumulator and per quantize work item. fp8 matmul_acc_lhs
# takes multiples of it, so the rows held in VMEM are padded up to one.
QUANT_ROWS = 32
# MXU RHS tile edge; the hand-scheduled matmuls work in these tiles.
MXU_TILE = 256
# wo_a loop iterations at which the first WINDOW_DEPTH wo_b windows start;
# the last iteration index means just before the final group's matmul. v7
# T288, 100 calls each: (5, 6, 7) 35.9 us, (4, 6, 7) 37.0 us, (4, 5, 6)
# 37.2 us, (3, 4, 5) 40.2 us; earlier starts take bandwidth from wo_a groups.
B_ISSUE = (5, 6, 7)
# Weight streams yield to activation and scale copies.
WEIGHT_PRIORITY = 1
# All of v7's VMEM. A 320-token tile fits; 384 exceeds it by 224 KiB.
VMEM_LIMIT_BYTES = 64 * 1024**2
MAX_TILE = 320
# Just past MAX_TILE, two tiles each read the full weights, so the two-kernel
# chain is faster until the chain's own cost catches up. v7, 50 calls each,
# fused vs chain: T330 62.4 vs 58.0 us, T384 64.3 vs 63.3 us, T416 65.3 vs
# 69.9 us. Token counts in (MAX_TILE, TWO_TILE_MIN) take the chain.
TWO_TILE_MIN = 416
# Row granule of the 2D output DMA: Mosaic rejects bf16 row slices that are
# not whole (16, 128) tiles (v7: T = 1, 100, 300 failed to compile).
STORE_ROWS = 16


def is_supported(
    x: jax.Array, wa: jax.Array, sa: jax.Array, wb: jax.Array, sb: jax.Array
) -> bool:
    """Whether inputs match the v7 output-projection contract at a token count
    where the fused kernel beats the two-kernel chain."""
    return (
        x.ndim == 3
        and (0 < x.shape[0] <= MAX_TILE or x.shape[0] >= TWO_TILE_MIN)
        and x.shape[1:] == (64, 512)
        and wa.shape == (4096, 8192)
        and sa.shape == (8192,)
        and wb.shape == (8192, 5120)
        and sb.shape == (5120,)
        and x.dtype == jnp.bfloat16
        and wa.dtype == jnp.float8_e4m3fn
        and wb.dtype == jnp.float8_e4m3fn
        and sa.dtype == jnp.float32
        and sb.dtype == jnp.float32
        and get_tpu_version() == 7
    )


def _kernel(
    # HBM inputs and output; each grid step reads and writes rows
    # start .. start + tokens.
    x,  # [T, 64, 512] bf16
    wa,  # [4096, 8192] fp8
    sa,  # [1, 8192] f32
    wb,  # [8192, 5120] fp8
    sb,  # [5120] f32
    *refs,
    tokens: int,
    start: int | None,
):
    # start None: grid step i covers rows i * tokens, found at run time, so
    # the full tiles share one body for any token count. A static start > 0
    # is the shorter remainder tile, which takes the full tiles' output as
    # an input aliased to its own; their rows pass through.
    if start is None:
        start = pl.multiple_of(pl.program_id(0) * tokens, QUANT_ROWS)
    elif start:
        refs = refs[1:]
    _tile(x, wa, sa, wb, sb, *refs, start=start, tokens=tokens)


def _tile(
    x,
    wa,
    sa,
    wb,
    sb,
    out,  # [T, 5120] bf16
    # VMEM scratch; P is tokens rounded up to QUANT_ROWS. Padding rows of xb
    # stay zero, so they quantize and project to zeros.
    xb,  # [GROUP_DEPTH, P, 8, 512] bf16
    wab,  # [GROUP_DEPTH, 4096, 1024] fp8
    qa,  # [2, P, 4096] fp8, quantized wo_a input of groups g and g + 1
    ia,  # [2, P, 1] f32, their BF16 inverse scales
    y,  # [P, 8192] bf16, wo_a output
    ya,  # [P, 1] f32, running row amax of y
    qb,  # [P, 8192] fp8, quantized wo_b input
    xs,  # [P, 1] f32, its row scales
    wbb,  # [WINDOW_DEPTH, 8192, WINDOW] fp8
    sav,  # [1, 8192] f32
    sbv,  # [5120] f32
    ob,  # [2, P, WINDOW] bf16
    xsem,
    wsem,
    bsem,
    osem,
    ssem,
    *,
    start: int | jax.Array,
    tokens: int,
):
    rows_p = y.shape[0]
    groups = x.shape[1] // HEADS_PER_GROUP
    width = wa.shape[1] // groups
    windows = wb.shape[1] // WINDOW
    chunks = rows_p // QUANT_ROWS
    group_depth, window_depth = xb.shape[0], wbb.shape[0]

    def succ(slot, depth):
        # Next ring slot. Loops carry the slot instead of computing i % depth,
        # a scalar division.
        if isinstance(slot, int):
            return (slot + 1) % depth
        return jnp.where(slot == depth - 1, 0, slot + 1)

    def x_copy(g, slot):
        return pltpu.make_async_copy(
            x.at[pl.ds(start, tokens), pl.ds(g * HEADS_PER_GROUP, HEADS_PER_GROUP)],
            xb.at[slot, pl.ds(0, tokens)],
            xsem.at[slot],
        )

    def wa_copy(g, slot):
        return pltpu.make_async_copy(
            wa.at[:, pl.ds(g * width, width)],
            wab.at[slot],
            wsem.at[slot],
        )

    def wb_copy(i, slot):
        return pltpu.make_async_copy(
            wb.at[:, pl.ds(i * WINDOW, WINDOW)],
            wbb.at[slot],
            bsem.at[slot],
        )

    # The output DMA moves whole bf16 row tiles; out has room for the zero
    # padding rows past the last token, which the caller slices off.
    stored = -(-tokens // STORE_ROWS) * STORE_ROWS

    def out_copy(i):
        return pltpu.make_async_copy(
            ob.at[i % 2, pl.ds(0, stored)],
            out.at[pl.ds(start, stored), pl.ds(i * WINDOW, WINDOW)],
            osem.at[i % 2],
        )

    sa_copy = pltpu.make_async_copy(sa, sav, ssem.at[0])
    sb_copy = pltpu.make_async_copy(sb, sbv, ssem.at[1])

    # The quantized wo_a input alternates between two slots. Callers pass the
    # slot as a Python int: with a traced index the compiler cannot tell the
    # slots apart and keeps one group's quantize stores ahead of the other
    # group's matmul loads.
    def quantize_a(xslot, slot, rows=None):
        # Same operations as wo_a_projection's quantized _kernel body. Rows
        # are independent, so a row chunk gives the same bits as the whole.
        rows = pl.ds(0, rows_p) if rows is None else rows
        v = xb[xslot, rows].reshape(rows.size, -1)
        amax = jnp.max(jnp.abs(v), axis=1, keepdims=True)
        inv = (FP8_E4M3_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(
            jnp.bfloat16
        )
        qa[slot, rows] = (v * inv).astype(qa.dtype)
        ia[slot, rows] = inv.astype(jnp.float32)

    def aligned(start, size):
        if not isinstance(start, int):
            start = pl.multiple_of(start, size)
        return pl.ds(start, size)

    # Every matmul is hand-scheduled, CSA2-style: a 1024-column output is four
    # 256-column blocks, MXU mx owning blocks 2 * mx + p for pairs p = 0, 1,
    # with one [QUANT_ROWS, 256] accumulator per row chunk. K tiles stream
    # through two RHS staging registers; pair 0 pops while pair 1 streams.
    # Each output column is still one full-K MXU accumulation.
    accs = tuple(
        tuple(
            tuple(
                jax.empty_ref(
                    jax.ShapeDtypeStruct((QUANT_ROWS, MXU_TILE), jnp.float32),
                    memory_space=pltpu.ACC(mx),
                )
                for _ in range(chunks)
            )
            for _ in range(2)
        )
        for mx in range(2)
    )

    def mxu_matmul(lhs, rhs, k_tiles, epilogue, quant=None):
        """lhs(rows, ks) @ rhs(ks, cols) over all row chunks.

        epilogue(p, mx, m, value) consumes each popped [QUANT_ROWS, 256]
        block. quant(c), c in [0, chunks), is one vector work item; the
        items are spread evenly over the K-tile pairs of both block pairs.
        """
        pairs = k_tiles // 2

        def push(p, kb, stage):
            for mx in range(2):
                block = pl.ds((2 * mx + p) * MXU_TILE, MXU_TILE)
                pltpu.matmul_push_rhs(
                    rhs(aligned(kb * MXU_TILE, MXU_TILE), block), stage, mx
                )

        def acc(p, kb, stage):
            for m in range(chunks):
                part = lhs(
                    pl.ds(m * QUANT_ROWS, QUANT_ROWS), aligned(kb * MXU_TILE, MXU_TILE)
                )
                for mx in range(2):
                    pltpu.matmul_acc_lhs(
                        accs[mx][p][m], part, load_staged_rhs=stage if m == 0 else None
                    )

        def two(p, kb, following):
            # Tiles kb (staged in 0) and kb + 1; stage `following` behind.
            push(p, kb + 1, 1)
            acc(p, kb, 0)
            if following is not None:
                fp, fkb = following
                push(fp, fkb, 0)
            acc(p, kb + 1, 1)

        def pop(p):
            for mx in range(2):
                for m in range(chunks):
                    epilogue(p, mx, m, pltpu.matmul_pop(accs[mx][p][m]))

        # Fully unrolled: a loop around the K-tile pairs refills the MXU
        # pipeline every iteration (v7 T288 wo_a group, VMEM-resident: fori
        # 2.60 us vs unrolled 2.15 us with the quantize interleaved).
        seq = [(p, 2 * j) for p in range(2) for j in range(pairs)]
        at = {}
        if quant is not None:
            for c in range(chunks):
                at.setdefault(c * len(seq) // chunks, []).append(c)
        # Block pair 0 pops right after pair 1's first K-tile pair, so its
        # epilogue runs beside pair 1's MXU work (v7 T288 wo_a group,
        # VMEM-resident, quantize interleaved: 2.11 us vs 2.51 us popping
        # both pairs at the end).
        push(0, 0, 0)
        for i, (p, kb) in enumerate(seq):
            two(p, kb, seq[i + 1] if i + 1 < len(seq) else None)
            if quant is not None:
                for c in at.get(i, ()):
                    quant(c)
            if i == pairs:
                pop(0)
        pop(1)

    def project_a(g, wslot, slot, quant=None):
        def epilogue(p, mx, m, r):
            rows = pl.ds(m * QUANT_ROWS, QUANT_ROWS)
            cols = aligned(g * width + (2 * mx + p) * MXU_TILE, MXU_TILE)
            v = ((r * (1.0 / ia[slot, rows])) * sav[:, cols]).astype(y.dtype)
            y[rows, cols] = v
            # wo_b's row amax, exact: max is order independent.
            ya[rows] = jnp.maximum(
                ya[rows],
                jnp.max(jnp.abs(v.astype(jnp.float32)), axis=1, keepdims=True),
            )

        mxu_matmul(
            lambda r, k: qa[slot, r, k],
            lambda k, c: wab[wslot, k, c],
            wab.shape[1] // MXU_TILE,
            epilogue,
            quant,
        )

    def issue_b(g):
        for i, at in enumerate(B_ISSUE):
            if isinstance(g, int):
                if g == at:
                    wb_copy(i, i).start(priority=WEIGHT_PRIORITY)
            else:

                @pl.when(g == at)
                def _(i=i):
                    wb_copy(i, i).start(priority=WEIGHT_PRIORITY)

    with jax.named_scope("op_start"):
        sa_copy.start()
        # Group 0's matmul needs both x0 and wa0; queue wa0 right behind x0
        # instead of behind every activation copy.
        x_copy(0, 0).start()
        wa_copy(0, 0).start()
        for g in range(1, group_depth):
            x_copy(g, g).start()
        sb_copy.start()
        for g in range(1, group_depth):
            wa_copy(g, g).start(priority=WEIGHT_PRIORITY)
        ya[...] = jnp.zeros(ya.shape, ya.dtype)
        if rows_p > tokens:
            xb[:, tokens:] = jnp.zeros(
                (group_depth, rows_p - tokens, *xb.shape[2:]), xb.dtype
            )
        x_copy(0, 0).wait()
        quantize_a(0, 0)
        sa_copy.wait()

    # Step g quantizes group g + 1 and multiplies group g; the two are
    # independent, so the quantize's row chunks run between the matmul's
    # K-tile pairs. slot is g's parity, static in every caller; cur is g's
    # ring slot, which group g + group_depth reuses once g's matmul is done.
    # Returns g + 1's slot.
    def step(g, slot, cur):
        nxt = succ(cur, group_depth)
        with jax.named_scope("op_a_group"):
            # Both waits first: the scheduler does not move vector work
            # across a DMA wait, so a wait between the two would serialize them.
            x_copy(g + 1, nxt).wait()
            wa_copy(g, cur).wait()
            project_a(
                g,
                cur,
                slot,
                lambda j: quantize_a(
                    nxt, 1 - slot, aligned(j * QUANT_ROWS, QUANT_ROWS)
                ),
            )

        if isinstance(g, int):
            if g + group_depth < groups:
                x_copy(g + group_depth, cur).start()
                wa_copy(g + group_depth, cur).start(priority=WEIGHT_PRIORITY)
        else:

            @pl.when(g + group_depth < groups)
            def _():
                x_copy(g + group_depth, cur).start()
                wa_copy(g + group_depth, cur).start(priority=WEIGHT_PRIORITY)

        issue_b(g)
        return nxt

    with jax.named_scope("op_a_first"):
        # Group 0's matmul needs only wa0, which lands before x1: run it
        # first instead of waiting for x1 with the steady-state step.
        wa_copy(0, 0).wait()
        project_a(0, 0, 0)
        x_copy(1, 1).wait()
        quantize_a(1, 1)
    x_copy(group_depth, 0).start()
    wa_copy(group_depth, 0).start(priority=WEIGHT_PRIORITY)
    issue_b(0)

    # Group 1 runs straight-line ahead of the loop, which removes a 0.8 us
    # gap after it on v7 T288. The loop then takes two steps per iteration
    # so every quantized-input slot is static; the rest run straight-line.
    step(1, 1, 1)
    pairs = (groups - 3) // 2

    def pair(p, cur):
        g = 2 * p + 2
        return step(g + 1, 1, step(g, 0, cur))

    jax.lax.fori_loop(0, pairs, pair, jnp.int32(2 % group_depth))
    for g in range(2 + 2 * pairs, groups - 1):
        step(g, g % 2, g % group_depth)

    with jax.named_scope("op_a_last"):
        last = groups - 1
        issue_b(last)
        wa_copy(last, last % group_depth).wait()
        project_a(last, last % group_depth, last % 2)

    def project_b(i, slot):
        def epilogue(p, mx, m, z):
            r = pl.ds(m * QUANT_ROWS, QUANT_ROWS)
            block = (2 * mx + p) * MXU_TILE
            scale = sbv[aligned(i * WINDOW + block, MXU_TILE)][None, :]
            ob[i % 2, r, pl.ds(block, MXU_TILE)] = ((z * xs[r]) * scale).astype(
                out.dtype
            )

        mxu_matmul(
            lambda r, k: qb[r, k],
            lambda k, c: wbb[slot, k, c],
            qb.shape[1] // MXU_TILE,
            epilogue,
        )

    def refill_and_store(i, slot):
        @pl.when(i + window_depth < windows)
        def _():
            wb_copy(i + window_depth, slot).start(priority=WEIGHT_PRIORITY)

        out_copy(i).start()

    # Same operations and order as buffered._quantize_rows. Window 0's weights
    # land during wo_a, so its matmul only waits for the quantize. Running it
    # on all rows at once beats splitting it into two row chunks that each
    # start as soon as they are quantized (v7 T288: 36.0 us vs 36.7 us).
    with jax.named_scope("op_b_quant"):
        s = ya[...] * jnp.float32(1.0 / 448.0)
        xs[...] = jnp.where(s == 0, 1.0, s)
        wb_copy(0, 0).wait()
        sb_copy.wait()
        for r in range(0, rows_p, QUANT_ROWS):
            rows = pl.ds(r, QUANT_ROWS)
            scale = xs[rows]
            for c in range(0, y.shape[1], width):
                cols = pl.ds(c, width)
                a = y[rows, cols].astype(jnp.float32)
                qb[rows, cols] = jnp.clip(a / scale, -448.0, 448.0).astype(qb.dtype)
        project_b(0, 0)
    refill_and_store(0, 0)

    # Same operations as buffered._kernel's window loop. slot is window i's
    # ring slot, which window i + window_depth refills; returns i + 1's slot.
    def window(i, slot):
        with jax.named_scope("op_b_window"):
            wb_copy(i, slot).wait()

            @pl.when(i >= 2)
            def _():
                out_copy(i - 2).wait()

            project_b(i, slot)

        refill_and_store(i, slot)
        return succ(slot, window_depth)

    jax.lax.fori_loop(1, windows, window, jnp.int32(1))

    with jax.named_scope("op_drain"):
        out_copy(windows - 2).wait()
        out_copy(windows - 1).wait()


def _call(x, wa, sa, wb, sb, out, *, tokens: int, count: int, start: int | None):
    """``count`` tiles of ``tokens`` rows as grid steps (start None), or one
    tile at a static ``start``, written into ``out`` (fresh when None)."""
    total, heads, head_dim = x.shape
    reduction, features = wa.shape
    width = features // (heads // HEADS_PER_GROUP)
    k, n = wb.shape
    # The output DMA moves whole bf16 row tiles, so the output has
    # STORE_ROWS-aligned rows; the caller's rows are sliced back out.
    padded = -(-total // STORE_ROWS) * STORE_ROWS
    rows = -(-tokens // QUANT_ROWS) * QUANT_ROWS
    group_depth, window_depth = GROUP_DEPTH, WINDOW_DEPTH
    hbm = pltpu.HBM
    inputs = (x, wa, sa, wb, sb) + (() if out is None else (out,))
    inputs = tuple(pltpu.with_memory_space_constraint(a, hbm) for a in inputs)
    f32, fp8 = jnp.float32, wa.dtype
    return pl.pallas_call(
        functools.partial(_kernel, tokens=tokens, start=start),
        out_shape=jax.ShapeDtypeStruct((padded, n), x.dtype),
        grid=(count,),
        in_specs=(pl.BlockSpec(memory_space=hbm),) * len(inputs),
        out_specs=pl.BlockSpec(memory_space=hbm),
        scratch_shapes=(
            pltpu.VMEM((group_depth, rows, HEADS_PER_GROUP, head_dim), x.dtype),
            pltpu.VMEM((group_depth, reduction, width), fp8),
            pltpu.VMEM((2, rows, reduction), fp8),
            pltpu.VMEM((2, rows, 1), f32),
            pltpu.VMEM((rows, features), x.dtype),
            pltpu.VMEM((rows, 1), f32),
            pltpu.VMEM((rows, k), fp8),
            pltpu.VMEM((rows, 1), f32),
            pltpu.VMEM((window_depth, k, WINDOW), fp8),
            pltpu.VMEM((1, features), f32),
            pltpu.VMEM((n,), f32),
            pltpu.VMEM((2, rows, WINDOW), x.dtype),
            pltpu.SemaphoreType.DMA((group_depth,)),
            pltpu.SemaphoreType.DMA((group_depth,)),
            pltpu.SemaphoreType.DMA((window_depth,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        input_output_aliases={} if out is None else {5: 0},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        name="fused_output_projection",
    )(*inputs)


@jax.jit
def fused_output_projection(
    x: jax.Array,  # [T, 64, 512] bf16
    wa: jax.Array,  # [4096, 8192] fp8
    sa: jax.Array,  # [8192] f32
    wb: jax.Array,  # [8192, 5120] fp8
    sb: jax.Array,  # [5120] f32
) -> jax.Array:
    """``wo_b(wo_a(x))`` for any token count, see ``is_supported``."""
    if not is_supported(x, wa, sa, wb, sb):
        raise ValueError("Fused output projection requires qualified v7 inputs")
    total = x.shape[0]
    groups = x.shape[1] // HEADS_PER_GROUP
    assert len(B_ISSUE) == WINDOW_DEPTH and all(0 <= at < groups for at in B_ISSUE)
    assert groups > GROUP_DEPTH
    # Ordinary JAX primitives cannot consume Pallas memory-space annotations.
    # Reshape before constraining the array at the custom-call boundary.
    sa = sa.reshape(1, wa.shape[1])
    count, tokens, rest = token_tiles(total)
    # At most two kernels for any token count: the full tiles as grid steps
    # of one, a shorter remainder tile as a second, static call.
    out = None
    if count:
        out = _call(x, wa, sa, wb, sb, None, tokens=tokens, count=count, start=None)
    if rest:
        start = count * tokens
        out = _call(x, wa, sa, wb, sb, out, tokens=rest, count=1, start=start)
    assert out is not None
    # A no-op on the serving buckets, which are STORE_ROWS multiples.
    return out[:total] if out.shape[0] != total else out


def token_tiles(total: int) -> tuple[int, int, int]:
    """(count, tokens, rest): ``count`` full tiles of ``tokens`` rows, a
    QUANT_ROWS multiple of at most MAX_TILE, then a shorter last tile of
    ``rest`` rows (0 if none). One tile up to MAX_TILE is the whole input."""
    if total <= MAX_TILE:
        return 0, 0, total
    tiles = -(-total // MAX_TILE)
    tokens = -(-total // tiles // QUANT_ROWS) * QUANT_ROWS
    count = total // tokens
    return count, tokens, total - count * tokens
