# SPDX-License-Identifier: Apache-2.0
"""Engram's wkv projection, gate and residual in one pass over the weight.

The XLA path writes the [T, 5*D] projection, reads it back as FP32 and makes
five FP32 passes over [T, 4, D] for the gate and residual, about 1.1 ms per
layer at 2,048 tokens against a 0.4 ms FP8 matmul. Here each weight window is
read once (the first head's once per token tile) while the vector unit
consumes the previous product: key windows accumulate the gate's three row
sums, value windows apply the residual, and the first head's tiles leave room
to quantize the next tile's rows.

Numerics follow the module: per-token FP8 activation quantization
(``quantize_block``), BF16 projection, FP32 gate and one BF16 rounding of the
output. Row sums run in a different order than XLA's, so outputs differ from
the XLA path by reordering only; the contract is error against an exact
reference no larger than the XLA path's.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8 = jnp.float8_e4m3fn
FP8_MAX = 448.0
LANES = 128
ROWS = 16
# The MXU's output width on v7x: projections split into chunks of it
# reproduce the whole window's dot bit for bit; other widths do not, and run
# slower.
MXU_COLUMNS = 256
ROW_TILE = 8
# v7x's 64 MiB of VMEM less 1 MiB: the 2,048-token fused schedule needs 62.1.
VMEM_LIMIT = 63 * 2**20
# A value group's output stores start as its unit ends; with three slot sets
# the next group's loads wait on stores a whole unit old.
VALUE_SETS = 3


def _rows(index, size, real, run):
    """``run(n)`` for block ``index`` of ``size`` rows over ``real`` rows: n is
    ``size``, or the last block's tail. Each size is a fixed DMA shape, so
    rows past the input are never read or written, only left as scratch."""
    blocks = -(-real // size)
    tail = real - (blocks - 1) * size
    if tail == size:
        run(size)
    elif blocks == 1:
        run(tail)
    else:
        pl.when(index < blocks - 1)(lambda: run(size))
        pl.when(index == blocks - 1)(lambda: run(tail))


def _fused_blocks(tiles, per_head, tile_blocks):
    """Quantization blocks per head-0 unit when head 0 runs token-tile-major
    and quantizes tile t + 1 under tile t's projections; 0 runs it all first."""
    if tiles > 1 and per_head > 1 and tile_blocks % per_head == 0:
        return tile_blocks // per_head
    return 0


def _kernel(
    rows,
    w,
    w_scale,
    hidden,
    q_weight,
    k_weight,
    mask,
    out,
    stage,
    q,
    x_scale,
    wb,
    hb,
    pb,
    dot_sum,
    h_sum,
    k_sum,
    gate,
    x_sem,
    w_sem,
    h_sem,
    o_sem,
    *,
    heads: int,
    dim: int,
    window: int,
    token_tile: int,
    quant_rows: int,
    eps: float,
    clamp: float,
    block_size: int | None,
):
    real = hidden.shape[0]
    tiles = q.shape[0] // token_tile
    per_head = dim // window
    key_windows = heads * per_head
    windows = key_windows + per_head

    def at(offset, size):
        return pl.ds(pl.multiple_of(offset, size), size)

    if block_size is None:
        q_copy, q_compute = _rows_quantizer(
            *rows, *stage, x_sem, q, x_scale, quant_rows=quant_rows, real=real
        )
    else:
        q_copy, q_compute = _packed_quantizer(
            *rows,
            *stage,
            x_sem,
            q,
            x_scale,
            quant_rows=quant_rows,
            block=block_size,
            real=real,
        )
    tile_blocks = token_tile // quant_rows
    real_blocks = -(-real // quant_rows)
    fused = _fused_blocks(tiles, per_head, tile_blocks)
    # Unit n is a weight window and a token tile. Head 0 goes tile by tile
    # when fused (each unit then reads its window again), so tile t + 1 is
    # quantized in the vector slack of tile t's projections; every later
    # window is read once for all tiles.
    head0 = per_head * tiles
    if fused:
        visits = head0 + per_head * heads

        def cell(n):
            early = n < head0
            return (
                jnp.where(early, n % per_head, n // tiles),
                jnp.where(early, n // per_head, n % tiles),
            )

        def visit(n):
            return jnp.where(n < head0, n, head0 + n // tiles - per_head)

        def window_of(v):
            return jnp.where(v < head0, v % per_head, v - head0 + per_head)

        def first(n):
            return (n < head0) | (n % tiles == 0)

        def last(n):
            return (n < head0) | (n % tiles == tiles - 1)

    else:
        visits = windows

        def cell(n):
            return n // tiles, n % tiles

        def visit(n):
            return n // tiles

        def window_of(v):
            return v

        def first(n):
            return n % tiles == 0

        def last(n):
            return n % tiles == tiles - 1

    def w_copy(v):
        """Weight visit v: window window_of(v) into slot v % 2."""
        return pltpu.make_async_copy(
            w.at[:, at(window_of(v) * window, window)], wb.at[v % 2], w_sem.at[v % 2]
        )

    # Head-0 unit m's slot quantizes ``fused`` blocks of tile m // per_head + 1
    # (clamped to the last real block: a repeat writes the same values).
    quant_slots = per_head * (tiles - 1) if fused else 0

    def slot_blocks(m):
        base = (m // per_head + 1) * tile_blocks + m % per_head * fused
        return [jnp.minimum(base + j, real_blocks - 1) for j in range(fused)]

    def slot_copy(m, op):
        for j, b in enumerate(slot_blocks(m)):
            q_copy(b, m % 2 * fused + j, op)

    def slot_compute(m):
        for j, b in enumerate(slot_blocks(m)):
            q_compute(b, m % 2 * fused + j)

    def project(n, cols):
        """BF16 projection of unit n (weight window and token tile cell(n)),
        columns ``cols`` of the window, into pb[n % 2]."""
        i, t = cell(n)
        rows = at(t * token_tile, token_tile)
        width = cols.stop - cols.start
        z = jax.lax.dot_general(
            q[rows, :],
            wb[visit(n) % 2, :, cols],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        scale = w_scale[pl.ds(pl.multiple_of(i * window + cols.start, LANES), width)]
        z = (z * x_scale[rows, :]) * scale[None, :]
        pb[n % 2, :, cols] = z.astype(pb.dtype)

    chunk = MXU_COLUMNS if window % MXU_COLUMNS == 0 else window

    chunks = window // chunk
    blocks = token_tile // ROWS

    def interleaved(vpu, n):
        """Unit n's vector work, row blocks in turn, between the column
        chunks of unit n + 1's projection."""
        for c in range(chunks):
            lo, hi = c * blocks // chunks, (c + 1) * blocks // chunks
            vpu(n, range(lo * ROWS, hi * ROWS, ROWS))
            project(n + 1, slice(c * chunk, (c + 1) * chunk))

    def projected(m):
        """After unit m's projection: its visit's last frees a weight slot."""

        @pl.when(last(m) & (visit(m) + 2 < visits))
        def _():
            w_copy(visit(m) + 2).start()

    # Key units stream hidden tiles through the second value group's first
    # two slots, free until the value phase.
    def k_load(n, op):
        i, t = cell(n)

        def run(size):
            slot = heads + n % 2
            row = pl.multiple_of(t * token_tile, token_tile)
            src = hidden.at[pl.ds(row, size), at(i * window, window)]
            copy = pltpu.make_async_copy(
                src, hb.at[slot, pl.ds(0, size)], h_sem.at[slot]
            )
            getattr(copy, op)()

        _rows(t, token_tile, real, run)

    # Value groups g = j * tiles + t own heads slots each, VALUE_SETS groups
    # in turn: a group's hidden tiles load while the unit before it is
    # consumed, and the output overwrites them in place.
    def v_slot(g, s):
        return (g % VALUE_SETS) * heads + s

    def v_copy(g, s, op, load):
        def run(size):
            slot = v_slot(g, s)
            row = pl.multiple_of(g % tiles * token_tile, token_tile)
            col = at(s * dim + g // tiles * window, window)
            local = hb.at[slot, pl.ds(0, size)]
            if load:
                copy = pltpu.make_async_copy(
                    hidden.at[pl.ds(row, size), col], local, h_sem.at[slot]
                )
            else:
                copy = pltpu.make_async_copy(
                    local, out.at[pl.ds(row, size), col], o_sem.at[slot]
                )
            getattr(copy, op)()

        _rows(g % tiles, token_tile, real, run)

    def v_load(g, s, op):
        v_copy(g, s, op, True)

    def v_store(g, s, op):
        v_copy(g, s, op, False)

    key_units = key_windows * tiles
    units = windows * tiles

    def key_vpu(n, starts, s):
        i, t = cell(n)
        c, row0 = i - s * per_head, t * token_tile
        cols = at(c * window, window)
        qw = q_weight[s : s + 1, cols].astype(jnp.float32)
        kw = k_weight[s : s + 1, cols].astype(jnp.float32)
        # A row block's three lane-partial sums stay in registers across
        # the window's lane groups: the same additions in the same order as
        # accumulating in VMEM group by group, a tenth of the traffic.
        for r in starts:
            rows = pl.ds(pl.multiple_of(row0 + r, ROWS), ROWS)
            d, hs, ks = dot_sum[rows, :], h_sum[rows, :], k_sum[rows, :]
            for lane in range(0, window, LANES):
                part = slice(lane, lane + LANES)
                h = hb[heads + n % 2, r : r + ROWS, part].astype(jnp.float32)
                kp = pb[n % 2, r : r + ROWS, part].astype(jnp.float32)
                d = d + ((h * qw[:, part]) * kw[:, part]) * kp
                hs = hs + h * h
                ks = ks + kp * kp
            dot_sum[rows, :], h_sum[rows, :], k_sum[rows, :] = d, hs, ks

    def value_vpu(n, starts):
        g = n - key_units
        row0 = g % tiles * token_tile
        for r in starts:
            rows = pl.ds(pl.multiple_of(row0 + r, ROWS), ROWS)
            gs = [gate[rows, s : s + 1] for s in range(heads)]
            for lane in range(0, window, LANES):
                part = slice(lane, lane + LANES)
                v = pb[n % 2, r : r + ROWS, part].astype(jnp.float32)
                for s in range(heads):
                    slot = v_slot(g, s)
                    h = hb[slot, r : r + ROWS, part].astype(jnp.float32)
                    hb[slot, r : r + ROWS, part] = (h + gs[s] * v).astype(hb.dtype)

    def prefetch(n):
        """DMA starts and waits before unit n is consumed and n + 1 projected."""

        @pl.when(n + 1 < key_units)
        def _():
            k_load(n + 1, "start")

        @pl.when(n + 1 >= key_units)
        def _():
            g = n + 1 - key_units
            for s in range(heads):

                @pl.when(g >= VALUE_SETS)
                def _(*, s=s):
                    v_store(g - VALUE_SETS, s, "wait")

                v_load(g, s, "start")

        @pl.when(first(n + 1))
        def _():
            w_copy(visit(n + 1)).wait()

    def consume_waits(n, key):
        if key:
            k_load(n, "wait")
        else:
            for s in range(heads):
                v_load(n - key_units, s, "wait")

    # Software pipeline: unit n's vector work and unit n + 1's projection
    # share one basic block, free to overlap; DMA waits and starts sit
    # outside it.
    _quantize(q_copy, q_compute, tile_blocks if fused else real_blocks)
    w_copy(0).start()
    w_copy(1).start()
    k_load(0, "start")
    for m in range(min(quant_slots, 2)):
        slot_copy(m, "start")
    w_copy(0).wait()
    if quant_slots:
        slot_copy(0, "wait")
    project(0, slice(0, window))
    if quant_slots:
        slot_compute(0)
    projected(0)

    def key_unit(n, s, quantize):
        prefetch(n)
        if quantize:

            @pl.when(n + 2 < quant_slots)
            def _():
                slot_copy(n + 2, "start")

            slot_copy(n + 1, "wait")
        consume_waits(n, True)
        interleaved(functools.partial(key_vpu, s=s), n)
        if quantize:
            slot_compute(n + 1)
        projected(n + 1)

    # Key windows, head by head (heads unrolled, so weight rows and gate
    # slots are static): lane-partial sums, reduced once per head.
    for s in range(heads):
        dot_sum[...] = jnp.zeros_like(dot_sum)
        h_sum[...] = jnp.zeros_like(h_sum)
        k_sum[...] = jnp.zeros_like(k_sum)

        start = s * head0
        if s == 0 and quant_slots > 1:
            start = quant_slots - 1
            pl.loop(0, start)(functools.partial(key_unit, s=s, quantize=True))
        pl.loop(start, (s + 1) * head0)(
            functools.partial(key_unit, s=s, quantize=False)
        )

        dot = jnp.sum(dot_sum[...], axis=1, keepdims=True)
        dot = dot * jax.lax.rsqrt(
            jnp.sum(h_sum[...], axis=1, keepdims=True) / dim + eps
        )
        dot = dot * (
            jax.lax.rsqrt(jnp.sum(k_sum[...], axis=1, keepdims=True) / dim + eps)
            * dim**-0.5
        )
        magnitude = jnp.sqrt(jnp.maximum(jnp.abs(dot), clamp))
        signed = jnp.where(dot < 0, -magnitude, magnitude)
        gate[:, s : s + 1] = jax.nn.sigmoid(signed) * mask[...]

    # Value windows: out = BF16(hidden + gate * value).
    def store(n):
        for s in range(heads):
            v_store(n - key_units, s, "start")

    @pl.loop(key_units, units - 1)
    def _(n):
        prefetch(n)
        consume_waits(n, False)
        interleaved(value_vpu, n)
        projected(n + 1)
        store(n)

    consume_waits(units - 1, False)
    value_vpu(units - 1, range(0, token_tile, ROWS))
    store(units - 1)
    groups = per_head * tiles
    for g in range(max(0, groups - VALUE_SETS), groups):
        for s in range(heads):
            v_store(g, s, "wait")


def _quantize(copy, compute, blocks):
    """Quantize blocks [0, blocks) in turn, staging through slots 0 and 1."""
    copy(0, 0, "start")

    @pl.loop(0, blocks)
    def _(block):
        slot = block % 2

        @pl.when(block + 1 < blocks)
        def _():
            copy(block + 1, 1 - slot, "start")

        copy(block, slot, "wait")
        compute(block, slot)


def _rows_quantizer(x, xb, x_sem, q, x_scale, *, quant_rows, real):
    """Per-token FP8 activation quantization, as quantize_block does it: a
    block's staging copy and its quantization into q and x_scale."""

    def copy(block, slot, op):
        def run(size):
            rows = pl.ds(pl.multiple_of(block * quant_rows, quant_rows), size)
            dma = pltpu.make_async_copy(
                x.at[rows], xb.at[slot, pl.ds(0, size)], x_sem.at[slot]
            )
            getattr(dma, op)()

        _rows(block, quant_rows, real, run)

    def compute(block, slot):
        _store_quantized(xb[slot].astype(jnp.float32), block, q, x_scale, quant_rows)

    return copy, compute


def _store_quantized(a, block, q, x_scale, quant_rows):
    scale = jnp.max(jnp.abs(a), axis=1, keepdims=True) / FP8_MAX
    scale = jnp.where(scale == 0, 1.0, scale)
    rows = pl.ds(pl.multiple_of(block * quant_rows, quant_rows), quant_rows)
    q[rows, :] = jnp.clip(a / scale, -FP8_MAX, FP8_MAX).astype(FP8)
    x_scale[rows, :] = scale


def _packed_quantizer(
    raw, scales, rb, sb, xt, x_sem, q, x_scale, *, quant_rows, block, real
):
    """``_rows_quantizer`` of the BF16 rows ``dequant_fp8_rows`` builds from
    packed staging, unpacked here instead of in XLA glue.

    ``raw`` [T, K / 4] int32 holds little-endian FP8 bytes, so word i of a
    row is K values 4i .. 4i + 3; ``scales`` [T, K / block / 4] int32 holds
    one E8M0 byte per ``block`` values. Byte plane j of a 128-word chunk is
    every fourth value; the quantized planes interleave back into natural K
    order before the projection, which reads q unchanged.
    """
    words = raw.shape[1]
    chunks, per_chunk = words // LANES, LANES * 4 // block // 4
    lane = jax.lax.broadcasted_iota(jnp.int32, (1, LANES), 1)
    # Lane l of chunk c is word 128c + l: scale word 128c + l // (block / 4) // 4
    # of its row, byte (l // (block / 4)) % 4 of it.
    word_of_lane = lane // (block // 4) // 4
    shift = (lane // (block // 4)) % 4 * 8

    def copy(index, slot, op):
        def run(size):
            rows = pl.ds(pl.multiple_of(index * quant_rows, quant_rows), size)
            local = pl.ds(0, size)
            for k, (src, dst) in enumerate(((raw, rb), (scales, sb))):
                dma = pltpu.make_async_copy(
                    src.at[rows], dst.at[slot, local], x_sem.at[k, slot]
                )
                getattr(dma, op)()

        _rows(index, quant_rows, real, run)

    def compute(index, slot):
        # dequant_fp8_rows, plane by plane: FP8 value times the power of two
        # built from the scale byte's exponent bits, +0 for byte 255, BF16.
        amax = jnp.zeros((quant_rows, LANES), jnp.float32)
        for c in range(chunks):
            packed = sb[slot, :, pl.ds(c * per_chunk, 1)]
            for k in range(1, per_chunk):
                column = sb[slot, :, pl.ds(c * per_chunk + k, 1)]
                packed = jnp.where(word_of_lane == k, column, packed)
            byte = (packed >> shift) & 0xFF
            power = jax.lax.bitcast_convert_type(byte << 23, jnp.float32)
            invalid = byte == 0xFF
            chunk_words = rb[slot, :, pl.ds(c * LANES, LANES)]
            for plane in range(4):
                value = pltpu.unpack_elementwise(
                    chunk_words,
                    index=plane,
                    packed_dtype=jnp.uint8,
                    unpacked_dtype=jnp.uint32,
                )
                value = jax.lax.bitcast_convert_type(value.astype(jnp.uint8), FP8)
                x = jnp.where(invalid, 0.0, value.astype(jnp.float32) * power)
                x = x.astype(jnp.bfloat16)
                xt[plane, :, pl.ds(c * LANES, LANES)] = x
                amax = jnp.maximum(amax, jnp.abs(x.astype(jnp.float32)))
        # Max is order independent: lane maxima, then one row maximum.
        scale = jnp.max(amax, axis=1, keepdims=True) / FP8_MAX
        scale = jnp.where(scale == 0, 1.0, scale)
        rows = pl.ds(pl.multiple_of(index * quant_rows, quant_rows), quant_rows)
        x_scale[rows, :] = scale
        for c in range(chunks):
            planes = [
                jnp.clip(
                    xt[plane, :, pl.ds(c * LANES, LANES)].astype(jnp.float32) / scale,
                    -FP8_MAX,
                    FP8_MAX,
                ).astype(FP8)
                for plane in range(4)
            ]
            q[rows, pl.ds(c * LANES * 4, LANES * 4)] = _interleave(planes)

    return copy, compute


def _interleave(planes):
    """Byte planes [R, 128] -> [R, 512] with plane j at columns 4l + j."""
    bytes_ = [
        jax.lax.bitcast_convert_type(p, jnp.uint8).astype(jnp.uint32) for p in planes
    ]
    words = pltpu.pack_elementwise(bytes_, packed_dtype=jnp.uint8)
    # A word's four bytes become four adjacent second-minor rows of the
    # bitcast, so the words go second-minor and the result comes back.
    return pltpu.bitcast(words.T, FP8).T


@functools.partial(
    jax.jit,
    static_argnames=(
        "eps",
        "clamp",
        "window",
        "token_tile",
        "quant_rows",
        "max_tokens",
        "block_size",
        "interpret",
    ),
)
def engram_gate(
    x: jax.Array | tuple[jax.Array, jax.Array],
    w: jax.Array,
    w_scale: jax.Array,
    hidden: jax.Array,
    q_weight: jax.Array,
    k_weight: jax.Array,
    mask: jax.Array,
    *,
    eps: float,
    clamp: float,
    window: int = 1280,
    token_tile: int = 512,
    quant_rows: int = 64,
    max_tokens: int = 2048,
    block_size: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Engram's gated residual from rows ``x`` [T, K] and ``hidden`` [T, H, D].

    ``w`` is the canonical [K, (H + 1) * D] FP8 weight with per-column FP32
    ``w_scale``; ``mask`` [T] gates padded tokens shut. Any T: the kernel
    computes whole token tiles but reads and writes only T rows (rounded up
    to ROW_TILE). Returns [T, H, D] BF16.

    With ``block_size``, ``x`` is the packed staging instead: int32 words of
    the FP8 bytes [T, K / 4] and of the E8M0 scale bytes, one per
    ``block_size`` values [T, K / block_size / 4], dequantized in the kernel
    exactly as ``dequant_fp8_rows`` would.
    """
    tokens = jax.tree.leaves(x)[0].shape[0]
    _, heads, dim = hidden.shape
    if block_size is None:
        k_dim = x.shape[1]
    else:
        raw, scales = x
        k_dim = raw.shape[1] * 4
        if (
            raw.shape[1] % LANES
            or LANES % block_size
            or block_size % 4
            or scales.shape != (tokens, k_dim // block_size // 4)
        ):
            raise ValueError("Unsupported packed Engram rows")
    if tokens > max_tokens:
        # The quantized rows stay in VMEM (12 MiB at 2,048 tokens), so longer
        # inputs run in chunks, each re-reading the weight (~50 us).
        return jnp.concatenate(
            [
                engram_gate(
                    jax.tree.map(lambda a, i=i: a[i : i + max_tokens], x),
                    w,
                    w_scale,
                    hidden[i : i + max_tokens],
                    q_weight,
                    k_weight,
                    mask[i : i + max_tokens],
                    eps=eps,
                    clamp=clamp,
                    window=window,
                    token_tile=token_tile,
                    quant_rows=quant_rows,
                    max_tokens=max_tokens,
                    block_size=block_size,
                    interpret=interpret,
                )
                for i in range(0, tokens, max_tokens)
            ]
        )
    if w.shape != (k_dim, (heads + 1) * dim) or dim % window or token_tile % quant_rows:
        raise ValueError("Unsupported Engram gate shapes")
    # The kernel computes whole token tiles but moves only the input's rows,
    # whole HBM row tiles of ROW_TILE; rows past them stay in VMEM.
    token_tile = min(token_tile, -(-tokens // quant_rows) * quant_rows)
    padded = -(-tokens // token_tile) * token_tile
    pad = ((0, -tokens % ROW_TILE), (0, 0))
    # Zero packed bytes dequantize to the zero rows of the BF16 input.
    x = jax.tree.map(lambda a: jnp.pad(a, pad), x)
    mask = jnp.pad(mask.astype(jnp.float32), (0, padded - tokens))
    hidden = jnp.pad(hidden.reshape(tokens, heads * dim), pad)
    hbm, vmem = (pl.BlockSpec(memory_space=s) for s in (pltpu.HBM, pltpu.VMEM))
    kernel = functools.partial(
        _kernel,
        heads=heads,
        dim=dim,
        window=window,
        token_tile=token_tile,
        quant_rows=quant_rows,
        eps=eps,
        clamp=clamp,
        block_size=block_size,
    )
    # Staging slots: two in turn, of `fused` blocks each when head 0 is fused.
    fused = _fused_blocks(padded // token_tile, dim // window, token_tile // quant_rows)
    slots = 2 * max(fused, 1)
    if block_size is None:
        rows_spec = (hbm,)
        stage = (pltpu.VMEM((slots, quant_rows, k_dim), x.dtype),)
        stage_sem = pltpu.SemaphoreType.DMA((slots,))
    else:
        rows_spec = (hbm, hbm)
        stage = (
            pltpu.VMEM((slots, quant_rows, k_dim // 4), raw.dtype),
            pltpu.VMEM((slots, quant_rows, scales.shape[1]), scales.dtype),
            pltpu.VMEM((4, quant_rows, k_dim // 4), jnp.bfloat16),
        )
        stage_sem = pltpu.SemaphoreType.DMA((2, slots))
    flat = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(hidden.shape, hidden.dtype),
        in_specs=(rows_spec, hbm, vmem, hbm, vmem, vmem, vmem),
        out_specs=hbm,
        scratch_shapes=(
            stage,
            pltpu.VMEM((padded, k_dim), FP8),
            pltpu.VMEM((padded, 1), jnp.float32),
            pltpu.VMEM((2, k_dim, window), w.dtype),
            pltpu.VMEM((VALUE_SETS * heads, token_tile, window), hidden.dtype),
            pltpu.VMEM((2, token_tile, window), jnp.bfloat16),
            pltpu.VMEM((padded, LANES), jnp.float32),
            pltpu.VMEM((padded, LANES), jnp.float32),
            pltpu.VMEM((padded, LANES), jnp.float32),
            pltpu.VMEM((padded, heads), jnp.float32),
            stage_sem,
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((VALUE_SETS * heads,)),
            pltpu.SemaphoreType.DMA((VALUE_SETS * heads,)),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="engram_gate",
    )(
        (x,) if block_size is None else x,
        w,
        w_scale,
        hidden,
        q_weight,
        k_weight,
        mask[:, None],
    )
    return flat[:tokens].reshape(tokens, heads, dim)
