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
"""Gather-fused MoE expert kernels for the TensorCore grouped-matmul MoE path.

The grouped-matmul path materializes every routed row twice in HBM: once in
expert order before the first matmul (``hidden[token_indices_sorted]``) and
once in token order after the second (``gmm2_res[argsort_revert_indices]``),
each ``[tokens * topk, hidden]`` BF16, and the top-k sum then reads the second
copy again. These kernels keep only the down-projection rows:

* ``gather_fused_moe`` walks expert tiles of the expert-sorted routing order,
  DMAs each tile's token rows straight from ``hidden`` by token id, runs
  gate_up, the activation and down with the intermediate in VMEM, and writes
  the tile's down-projection rows in expert order.
* ``gather_combine`` DMAs each token's top-k rows by their expert-order
  position and writes the weighted sum.

Rows cross HBM in a packed form so a single row is one contiguous DMA: BF16
``[n, hidden]`` is stored as uint32 ``[n * S, 128]`` with ``S = hidden // 256``,
row ``r`` in sublanes ``[S * r, S * r + S)`` and word ``(S * r + j, l)``
holding column ``256 * j + l`` in its low half and ``256 * j + 128 + l`` in its
high half. (A BF16 array's native tiling interleaves row pairs, so one BF16 row
is not addressable on its own.) The expert kernel's input rows,
``quantize_pack_rows``, carry one more sublane: each token is quantized once,
as ``bf16(x * scale_inv)`` per 512-column block, with the block scales in the
extra sublane, instead of once per routed copy inside the expert kernel.

The arithmetic of each routed row matches ``gmm_v2`` (per-row, per-512-column
FP8 activation quantization with BF16 accumulation, the fused activation, and
the same for the down projection); only where rows live changes.
"""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import apply_gated_act

LANES = 128
# 640 rows hold a typical 128-expert, top-8 group of an 8192-token step in one
# tile (fewer tail tiles than 512; 617 vs 678 us at Gemma4-26B TP2).
DEFAULT_TILE_M = 640
# Share of the core's VMEM gather_fused_moe asks the compiler for by default.
VMEM_LIMIT_FRACTION = 0.85
# BF16 columns one packed uint32 lane block carries (low and high halves).
PACKED_COLUMNS = 2 * LANES
# gmm_v2's activation quantization block (per row, per this many columns).
QUANT_BLOCK = 512
FP8 = jnp.float8_e4m3fn
_FP8_MAX = float(jnp.finfo(FP8).max)
_HI_MASK = 0xFFFF0000


def packed_sublanes(hidden_size: int) -> int:
    if hidden_size % PACKED_COLUMNS:
        raise ValueError(
            f"hidden size {hidden_size} is not a multiple of {PACKED_COLUMNS}"
        )
    return hidden_size // PACKED_COLUMNS


def unpack_rows(packed: jax.Array, n: int) -> jax.Array:
    """Packed uint32 ``[n * S, 128]`` -> BF16 ``[n, S * 256]``."""
    s = packed.shape[0] // n
    p = packed.reshape(n, s, 1, LANES)
    halves = jnp.concatenate([p & 0xFFFF, p >> 16], axis=2).astype(jnp.uint16)
    return lax.bitcast_convert_type(halves, jnp.bfloat16).reshape(n, s * PACKED_COLUMNS)


def _unpack_lane_block(words: jax.Array) -> tuple[jax.Array, jax.Array]:
    """uint32 ``[rows, 128]`` -> (low, high) columns as exact F32 values."""
    lo = pltpu.bitcast(jnp.left_shift(words, 16), jnp.float32)
    hi = pltpu.bitcast(jnp.bitwise_and(words, jnp.uint32(_HI_MASK)), jnp.float32)
    return lo, hi


def _bf16_bits(x: jax.Array) -> jax.Array:
    """The BF16 bit pattern of each element, zero-extended to uint32.

    Bitcasting the BF16 value (rather than widening it to F32) pins the
    stored bits to the rounded BF16 value even where the compiler would keep
    excess precision in a BF16 intermediate.
    """
    return pltpu.bitcast(x.astype(jnp.bfloat16), jnp.uint16).astype(jnp.uint32)


def _pack_lane_block(lo: jax.Array, hi: jax.Array) -> jax.Array:
    """BF16 (low, high) ``[rows, 128]`` -> packed uint32 ``[rows, 128]``."""
    return jnp.bitwise_or(_bf16_bits(lo), jnp.left_shift(_bf16_bits(hi), 16))


def _scale_rows(block: jax.Array) -> tuple[jax.Array, jax.Array]:
    """gmm_v2's activation quantization of one BF16 block, before the FP8 cast.

    Returns ``(block * scale_inv, scale)`` in BF16, exactly as gmm_v2 forms them.
    """
    abs_max = jnp.max(jnp.abs(block), axis=1, keepdims=True)
    scale = abs_max / _FP8_MAX
    scale_inv = jnp.where(scale == 0, 0, 1 / scale)
    return block * scale_inv, scale


def _quantize_rows(block: jax.Array) -> tuple[jax.Array, jax.Array]:
    """gmm_v2's activation quantization of one BF16 block: (fp8 values, BF16 scale)."""
    scaled, scale = _scale_rows(block)
    return scaled.astype(FP8), scale


def _quantize_pack_token_kernel(x_ref, o_ref, *, s, tt):
    x = x_ref[...].astype(jnp.float32)
    scale = jnp.max(jnp.abs(x), axis=1, keepdims=True) / _FP8_MAX
    xs = jnp.clip(x * jnp.where(scale == 0, 0, 1 / scale), -_FP8_MAX, _FP8_MAX)
    for j in range(s):
        lo = xs[:, PACKED_COLUMNS * j : PACKED_COLUMNS * j + LANES]
        hi = xs[:, PACKED_COLUMNS * j + LANES : PACKED_COLUMNS * (j + 1)]
        o_ref[pl.ds(j, tt, stride=s + 1), :] = _pack_lane_block(lo, hi)
    # The row's F32 scale in every lane of its last sublane.
    o_ref[pl.ds(s, tt, stride=s + 1), :] = jnp.broadcast_to(
        pltpu.bitcast(scale, jnp.uint32), (tt, LANES)
    )


def _quantize_pack_kernel(x_ref, o_ref, *, s, tt, hidden):
    scaled, scales = [], []
    for k0 in range(0, hidden, QUANT_BLOCK):
        block_scaled, block_scale = _scale_rows(
            x_ref[:, k0 : min(hidden, k0 + QUANT_BLOCK)]
        )
        scaled.append(block_scaled)
        scales.append(block_scale)
    xs = jnp.concatenate(scaled, axis=1)
    for j in range(s):
        lo = xs[:, PACKED_COLUMNS * j : PACKED_COLUMNS * j + LANES]
        hi = xs[:, PACKED_COLUMNS * j + LANES : PACKED_COLUMNS * (j + 1)]
        o_ref[pl.ds(j, tt, stride=s + 1), :] = _pack_lane_block(lo, hi)
    # Block b's scale, as F32 bits, in lane b of the row's last sublane.
    lane = lax.broadcasted_iota(jnp.int32, (tt, LANES), 1)
    words = jnp.zeros((tt, LANES), jnp.uint32)
    for b, block_scale in enumerate(scales):
        bits = jnp.left_shift(_bf16_bits(block_scale), 16)
        words = jnp.where(lane == b, bits, words)
    o_ref[pl.ds(s, tt, stride=s + 1), :] = words


@functools.partial(jax.jit, static_argnames=("tile_tokens", "per_token", "interpret"))
def quantize_pack_rows(
    x: jax.Array,
    *,
    tile_tokens: int = 256,
    per_token: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """BF16 ``[n, hidden]`` -> the expert kernel's rows, uint32 ``[n * (S + 1), 128]``.

    Sublanes ``[0, S)`` of a row hold ``bf16(x * scale_inv)`` per 512-column
    block in the packed form; lane ``b`` of sublane ``S`` holds block ``b``'s
    BF16 scale as F32 bits. With ``per_token`` the row has one F32 scale
    (its absolute maximum / the FP8 maximum, in every lane of sublane ``S``),
    for ``gather_fused_moe_v2(per_token=True)``.
    """
    n, h = x.shape
    s = packed_sublanes(h)
    if pl.cdiv(h, QUANT_BLOCK) > LANES:
        raise ValueError(f"hidden size {h} has more than {LANES} quantization blocks")
    # Any token bucket: the largest tile that divides both n and tile_tokens.
    tile = math.gcd(n, tile_tokens)
    if tile % 8:
        raise ValueError(f"gcd({n=}, {tile_tokens=}) = {tile} is not a multiple of 8")
    tile_tokens = tile
    if per_token:
        kernel = functools.partial(_quantize_pack_token_kernel, s=s, tt=tile_tokens)
    else:
        kernel = functools.partial(_quantize_pack_kernel, s=s, tt=tile_tokens, hidden=h)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n * (s + 1), LANES), jnp.uint32),
        grid=(n // tile_tokens,),
        in_specs=[pl.BlockSpec((tile_tokens, h), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((tile_tokens * (s + 1), LANES), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name=f"quantize_pack_rows-h_{h}-tt_{tile_tokens}"
        + ("-token" if per_token else ""),
        interpret=pltpu.InterpretParams() if interpret else False,
    )(x)


def gather_fused_moe_vmem_bytes(
    hidden: int, inter: int, tile_m: int = DEFAULT_TILE_M
) -> int:
    """Upper estimate of the VMEM gather_fused_moe compiles into.

    Double-buffered whole-expert weights and scales, double-buffered input and
    output row tiles, and the body's live values (gate_up accumulator, one
    activation block in BF16 and FP8, the intermediate in BF16 and FP8, and an
    F32 down-projection block), which the compiler places in the same memory.
    Calibrated at tile_m 640 against the smallest vmem_limit_bytes the kernel
    compiles under on tpu7x: 0.7 to 1.7 MiB above it at hidden 2816
    (intermediates 384, 768, 1536) and 0.6 to 1.1 MiB at hidden 4096 (512,
    1024). Recheck it before relying on it far from those shapes.
    """
    s = packed_sublanes(hidden)
    row_bytes = LANES * 4
    buffers = (
        2 * tile_m * (2 * s + 1) * row_bytes  # x rows (S + 1 sublanes) + y rows (S)
        + 2 * (hidden * 2 * inter + inter * hidden)  # fp8 w1, w2
        + 2 * 8 * (2 * inter + hidden) * 4  # f32 scales, sublane-padded
    )
    body = tile_m * (
        2 * inter * 2 + QUANT_BLOCK * 3 + inter * 3 + PACKED_COLUMNS * 4 * 2
    )
    return buffers + body


@dataclasses.dataclass(frozen=True)
class FusedMoeConfig:
    num_experts: int
    hidden_size: int
    inter_size: int  # per gate/up half, padded to 128
    tile_m: int
    height_step: int
    max_tiles: int
    act: str
    mxu_cols: int
    issue_group: int
    per_token: bool = False  # gather_fused_moe_v2's per-row activation scales

    @property
    def sublanes(self) -> int:
        return packed_sublanes(self.hidden_size)

    @property
    def x_sublanes(self) -> int:
        return self.sublanes + 1


def _gate_up_block_scales(lane_block, weights, cfg: FusedMoeConfig, height: int):
    """gate_up of ``height`` rows as gmm_v2 computes it: per 512-column block,
    the FP8 cast of the stored bf16(x * scale_inv), then BF16 accumulation of
    the dequantized block products. ``lane_block(j)`` reads packed sublane j
    of the rows (sublane S holds the block scales)."""
    w1, w1s, _, _ = weights
    scale_words = lane_block(cfg.sublanes)
    n1 = 2 * cfg.inter_size
    acc = [
        jnp.zeros((height, cfg.mxu_cols), jnp.bfloat16)
        for _ in range(n1 // cfg.mxu_cols)
    ]
    for k0 in range(0, cfg.hidden_size, QUANT_BLOCK):
        k1 = min(cfg.hidden_size, k0 + QUANT_BLOCK)
        cols = []
        for j in range(k0 // PACKED_COLUMNS, k1 // PACKED_COLUMNS):
            cols.extend(_unpack_lane_block(lane_block(j)))
        block_q = jnp.concatenate(cols, axis=1).astype(jnp.bfloat16).astype(FP8)
        b = k0 // QUANT_BLOCK
        block_scale = pltpu.bitcast(scale_words[:, b : b + 1], jnp.float32)
        block_scale = block_scale.astype(jnp.bfloat16)
        for n, n0 in enumerate(range(0, n1, cfg.mxu_cols)):
            prod = jnp.dot(
                block_q,
                w1[k0:k1, n0 : n0 + cfg.mxu_cols],
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
            prod *= block_scale
            prod *= w1s[:, n0 : n0 + cfg.mxu_cols].astype(jnp.bfloat16)
            acc[n] += prod
    return jnp.concatenate(acc, axis=1)


def _down_block_scales(gate_up, weights, cfg: FusedMoeConfig, height: int):
    """The activation and down projection as gmm_v2 computes them (per
    512-column block of the intermediate): yields ``(j, words)``, the packed
    output sublane j of the rows."""
    _, _, w2, w2s = weights
    mid = apply_gated_act(
        gate_up[:, : cfg.inter_size], gate_up[:, cfg.inter_size :], cfg.act
    )
    mid_blocks = []
    for k0 in range(0, cfg.inter_size, QUANT_BLOCK):
        k1 = min(cfg.inter_size, k0 + QUANT_BLOCK)
        mid_q, mid_scale = _quantize_rows(mid[:, k0:k1])
        mid_blocks.append((k0, k1, mid_q, mid_scale.astype(jnp.bfloat16)))
    for j in range(cfg.sublanes):
        n0 = j * PACKED_COLUMNS
        halves = []
        for h0 in range(n0, n0 + PACKED_COLUMNS, cfg.mxu_cols):
            down_acc = jnp.zeros((height, cfg.mxu_cols), jnp.bfloat16)
            for k0, k1, mid_q, mid_scale in mid_blocks:
                prod = jnp.dot(
                    mid_q,
                    w2[k0:k1, h0 : h0 + cfg.mxu_cols],
                    preferred_element_type=jnp.float32,
                ).astype(jnp.bfloat16)
                prod *= mid_scale
                prod *= w2s[:, h0 : h0 + cfg.mxu_cols].astype(jnp.bfloat16)
                down_acc += prod
            halves.append(down_acc)
        down = jnp.concatenate(halves, axis=1)
        yield j, _pack_lane_block(down[:, :LANES], down[:, LANES:])


def _gate_up_token_scales(lane_block, weights, cfg: FusedMoeConfig, height: int):
    """gate_up with one activation scale per row, as quantize_pack_rows'
    ``per_token`` rows carry it: the MXU accumulates each output column over
    the whole hidden size in F32, dequantized once by the row's and the
    column's scale."""
    del height
    w1, w1s, _, _ = weights
    row_scale = pltpu.bitcast(lane_block(cfg.sublanes)[:, :1], jnp.float32)
    cols = []
    for j in range(cfg.sublanes):
        cols.extend(_unpack_lane_block(lane_block(j)))
    x_q = jnp.concatenate(cols, axis=1).astype(jnp.bfloat16).astype(FP8)
    out = []
    for n0 in range(0, 2 * cfg.inter_size, cfg.mxu_cols):
        acc = jnp.dot(
            x_q, w1[:, n0 : n0 + cfg.mxu_cols], preferred_element_type=jnp.float32
        )
        scaled = acc * row_scale * w1s[:, n0 : n0 + cfg.mxu_cols]
        out.append(scaled.astype(jnp.bfloat16))
    return jnp.concatenate(out, axis=1)


def _down_token_scales(gate_up, weights, cfg: FusedMoeConfig, height: int):
    """The activation, quantized with one scale per row, and the down
    projection with F32 accumulation over the whole intermediate: yields
    ``(j, words)``, the packed output sublane j of the rows."""
    del height
    _, _, w2, w2s = weights
    mid = apply_gated_act(
        gate_up[:, : cfg.inter_size], gate_up[:, cfg.inter_size :], cfg.act
    )
    amax = jnp.max(jnp.abs(mid), axis=1, keepdims=True).astype(jnp.float32)
    scale = amax / _FP8_MAX
    scale_inv = jnp.where(scale == 0, 0, 1 / scale)
    mid_q = jnp.clip(mid * scale_inv.astype(jnp.bfloat16), -_FP8_MAX, _FP8_MAX)
    mid_q = mid_q.astype(FP8)
    for j in range(cfg.sublanes):
        n0 = j * PACKED_COLUMNS
        acc = jnp.dot(
            mid_q, w2[:, n0 : n0 + PACKED_COLUMNS], preferred_element_type=jnp.float32
        )
        down = acc * scale * w2s[:, n0 : n0 + PACKED_COLUMNS]
        yield j, _pack_lane_block(down[:, :LANES], down[:, LANES:])


def _fill_tiles(gs_ref, m_expert, m_start, m_count, m_visit, v_expert, *, cfg):
    """Expert tiles of the expert-sorted row order; returns (tiles, visits)."""

    def expert_body(e, carry):
        n_tiles, n_visits, row = carry
        size = gs_ref[e]
        tiles_e = pl.cdiv(size, cfg.tile_m)

        def tile_body(j, _):
            t = n_tiles + j
            m_expert[t] = e
            m_start[t] = row + j * cfg.tile_m
            m_count[t] = jnp.minimum(cfg.tile_m, size - j * cfg.tile_m)
            m_visit[t] = n_visits
            return _

        lax.fori_loop(0, tiles_e, tile_body, 0)

        @pl.when(size > 0)
        def _():
            v_expert[n_visits] = e

        return n_tiles + tiles_e, n_visits + (size > 0).astype(jnp.int32), row + size

    zero = jnp.int32(0)
    n_tiles, n_visits, _ = lax.fori_loop(
        0, cfg.num_experts, expert_body, (zero, zero, zero)
    )
    return n_tiles, n_visits


def _fused_moe_kernel(
    gs_ref,  # SMEM int32 [E] (scalar prefetch)
    tok_hbm,  # int32 [rows + tile_m + 128]: token id of each expert-sorted row
    x_hbm,  # uint32 [tokens * (S + 1), 128] quantize_pack_rows rows
    w1_hbm,  # fp8 [E, H, 2I] gate | up
    w1s_hbm,  # f32 [E, 1, 2I]
    w2_hbm,  # fp8 [E, I, H]
    w2s_hbm,  # f32 [E, 1, H]
    y_hbm,  # out: uint32 [rows * S, 128] packed down rows, expert order
    m_expert,
    m_start,
    m_count,
    m_visit,
    v_expert,
    tok_sm0,  # SMEM int32 [tile_m + 128]: token ids of even tiles
    tok_sm1,  # SMEM int32 [tile_m + 128]: token ids of odd tiles
    x_vm,  # VMEM uint32 [2, tile_m * (S + 1), 128]
    w1_vm,
    w1s_vm,
    w2_vm,
    w2s_vm,
    y_vm,  # VMEM uint32 [2, tile_m * S, 128]
    sem_tok,
    sem_x,
    sem_w,
    sem_y,
    *,
    cfg: FusedMoeConfig,
):
    s = cfg.sublanes
    sx = cfg.x_sublanes
    group = cfg.issue_group
    n_tiles, n_visits = _fill_tiles(
        gs_ref, m_expert, m_start, m_count, m_visit, v_expert, cfg=cfg
    )

    # Separate 1-D windows per slot keep each row's SMEM address one add.
    tok_windows = (tok_sm0, tok_sm1)

    def per_slot(slot, fn):
        """Run fn(static_slot) for the dynamic slot, so its addressing is static."""
        for static_slot in (0, 1):
            pl.when(slot == static_slot)(functools.partial(fn, static_slot))

    def tok_copy(i, static_slot):
        # A 1-D HBM window DMA must start on a lane-tile boundary.
        base = pl.multiple_of(m_start[i] - m_start[i] % LANES, LANES)
        return pltpu.make_async_copy(
            tok_hbm.at[pl.ds(base, cfg.tile_m + LANES)],
            tok_windows[static_slot],
            sem_tok.at[static_slot],
        )

    def issued_rows(i):
        # Rows are issued in unrolled groups; the tail of the last group
        # fetches the next tile's (or padding) tokens into rows nothing reads.
        return pl.cdiv(m_count[i], group) * group

    def issue_rows(i, static_slot):
        window = tok_windows[static_slot]
        offset = m_start[i] % LANES

        def body(g, carry):
            for u in range(group):
                r = g * group + u
                pltpu.make_async_copy(
                    x_hbm.at[pl.ds(window[offset + r] * sx, sx)],
                    x_vm.at[static_slot, pl.ds(r * sx, sx)],
                    sem_x.at[static_slot],
                ).start()
            return carry

        lax.fori_loop(0, issued_rows(i) // group, body, 0)

    def rows_copy(i, slot):
        # Waits count bytes, so one descriptor the size of the issued rows
        # retires all of their DMAs whatever order they landed in.
        rows = pl.ds(0, issued_rows(i) * sx)
        return pltpu.make_async_copy(
            x_vm.at[slot, rows], x_vm.at[slot, rows], sem_x.at[slot]
        )

    def weight_copies(e, wslot):
        pairs = ((w1_hbm, w1_vm), (w1s_hbm, w1s_vm), (w2_hbm, w2_vm), (w2s_hbm, w2s_vm))
        return [
            pltpu.make_async_copy(src.at[e], dst.at[wslot], sem_w.at[wslot])
            for src, dst in pairs
        ]

    def out_copy(i, slot):
        rows = m_count[i] * s
        return pltpu.make_async_copy(
            y_vm.at[slot, pl.ds(0, rows)],
            y_hbm.at[pl.ds(m_start[i] * s, rows)],
            sem_y.at[slot],
        )

    @pl.when(n_tiles > 0)
    def _():
        first = tok_copy(0, 0)
        first.start()
        for c in weight_copies(v_expert[0], 0):
            c.start()
        first.wait()
        issue_rows(0, 0)

        @pl.when(n_tiles > 1)
        def _():
            tok_copy(1, 1).start()

    def compute(slot, wslot, height):
        def lane_block(j):
            return x_vm[slot, pl.ds(j, height, stride=sx), :]

        weights = (w1_vm.at[wslot], w1s_vm.at[wslot], w2_vm.at[wslot], w2s_vm.at[wslot])
        gate_up = _gate_up_block_scales(lane_block, weights, cfg, height)
        for j, words in _down_block_scales(gate_up, weights, cfg, height):
            y_vm[slot, pl.ds(j, height, stride=s), :] = words

    def tile_loop(i, carry):
        slot = i % 2

        @pl.when(i + 1 < n_tiles)
        def _():
            def next_tile(next_slot):
                tok_copy(i + 1, next_slot).wait()
                issue_rows(i + 1, next_slot)

            per_slot(1 - slot, next_tile)

            @pl.when(i + 2 < n_tiles)
            def _():
                per_slot(slot, lambda static_slot: tok_copy(i + 2, static_slot).start())

        visit = m_visit[i]
        wslot = visit % 2
        first_of_visit = jnp.logical_or(i == 0, m_visit[jnp.maximum(i - 1, 0)] != visit)

        @pl.when(first_of_visit)
        def _():
            for c in weight_copies(m_expert[i], wslot):
                c.wait()

            @pl.when(visit + 1 < n_visits)
            def _():
                for c in weight_copies(v_expert[visit + 1], 1 - wslot):
                    c.start()

        rows_copy(i, slot).wait()

        @pl.when(i >= 2)
        def _():
            out_copy(i - 2, slot).wait()

        rung = (m_count[i] - 1) // cfg.height_step
        lax.switch(
            rung,
            [
                functools.partial(compute, slot, wslot, cfg.height_step * (b + 1))
                for b in range(cfg.tile_m // cfg.height_step)
            ],
        )
        out_copy(i, slot).start()
        return carry

    lax.fori_loop(0, n_tiles, tile_loop, 0)

    @pl.when(n_tiles >= 2)
    def _():
        out_copy(n_tiles - 2, n_tiles % 2).wait()

    @pl.when(n_tiles >= 1)
    def _():
        out_copy(n_tiles - 1, (n_tiles - 1) % 2).wait()


@functools.partial(
    jax.jit,
    static_argnames=(
        "act",
        "tile_m",
        "height_step",
        "issue_group",
        "vmem_limit_bytes",
        "interpret",
    ),
)
def gather_fused_moe(
    x_packed: jax.Array,  # uint32 [tokens * (S + 1), 128] (quantize_pack_rows)
    token_indices_sorted: jax.Array,  # int32 [rows]
    group_sizes: jax.Array,  # int32 [E]
    w1: jax.Array,  # fp8 [E, H, 2I]: gate columns [0, I), up columns [I, 2I)
    w1_scale: jax.Array,  # f32 per-channel [E, 1, 1, 2I] or [E, 1, 2I]
    w2: jax.Array,  # fp8 [E, I, H]
    w2_scale: jax.Array,  # f32 per-channel [E, 1, 1, H] or [E, 1, H]
    *,
    act: str,
    tile_m: int = DEFAULT_TILE_M,
    height_step: int = 128,
    issue_group: int = 16,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Expert-order packed down-projection rows ``[rows * S, 128]``.

    Row ``r`` is the down projection of token ``token_indices_sorted[r]``
    through the expert that owns sorted position ``r``: ``group_sizes`` gives
    each expert's contiguous run of sorted positions, from position 0. Only
    that prefix (``sum(group_sizes)`` rows) is written; rows past it (routes to
    other ranks' experts, skipped padded tokens) are left uninitialized and
    must not be read (``gather_combine(masked=True)``). ``w1`` holds gate and
    up as contiguous halves, the layout gmm_v2's fused-activation path reads.
    """
    num_experts, hidden, n1 = w1.shape
    inter = n1 // 2
    rows = token_indices_sorted.shape[0]
    s = packed_sublanes(hidden)
    if w2.shape != (num_experts, inter, hidden):
        raise ValueError(f"w2 {w2.shape} does not match w1 {w1.shape}")
    if inter % LANES:
        raise ValueError(f"intermediate {inter} must be a multiple of {LANES}")
    if tile_m % height_step or tile_m % issue_group:
        raise ValueError(
            f"{tile_m=} is not a multiple of {height_step=} and {issue_group=}"
        )
    if x_packed.shape[0] % (s + 1):
        raise ValueError(f"x_packed {x_packed.shape} is not quantize_pack_rows output")
    w1_scale = w1_scale.reshape(num_experts, 1, n1).astype(jnp.float32)
    w2_scale = w2_scale.reshape(num_experts, 1, hidden).astype(jnp.float32)
    # The token-id windows read up to tile_m + 128 entries past a tile's start.
    tok = jnp.pad(token_indices_sorted.astype(jnp.int32), (0, tile_m + LANES))
    max_tiles = pl.cdiv(rows, tile_m) + num_experts
    cfg = FusedMoeConfig(
        num_experts=num_experts,
        hidden_size=hidden,
        inter_size=inter,
        tile_m=tile_m,
        height_step=height_step,
        max_tiles=max_tiles,
        act=act,
        mxu_cols=256,
        issue_group=issue_group,
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    smem_tiles = [pltpu.SMEM((max_tiles,), jnp.int32) for _ in range(4)]
    return pl.pallas_call(
        functools.partial(_fused_moe_kernel, cfg=cfg),
        out_shape=jax.ShapeDtypeStruct((rows * s, LANES), jnp.uint32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[hbm] * 6,
            out_specs=hbm,
            scratch_shapes=[
                *smem_tiles,
                pltpu.SMEM((num_experts,), jnp.int32),
                pltpu.SMEM((tile_m + LANES,), jnp.int32),
                pltpu.SMEM((tile_m + LANES,), jnp.int32),
                pltpu.VMEM((2, tile_m * (s + 1), LANES), jnp.uint32),
                pltpu.VMEM((2, hidden, n1), w1.dtype),
                pltpu.VMEM((2, 1, n1), jnp.float32),
                pltpu.VMEM((2, inter, hidden), w2.dtype),
                pltpu.VMEM((2, 1, hidden), jnp.float32),
                pltpu.VMEM((2, tile_m * s, LANES), jnp.uint32),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes
            or int(pltpu.get_tpu_info().vmem_capacity_bytes * VMEM_LIMIT_FRACTION),
            disable_bounds_checks=True,
        ),
        name=f"gather_fused_moe-e_{num_experts}-h_{hidden}-i_{inter}-tm_{tile_m}",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(group_sizes.astype(jnp.int32), tok, x_packed, w1, w1_scale, w2, w2_scale)


# ------------------------------------------------------------ expert kernel v2

# gather_fused_moe_v2's x row slots: the tile in compute, the next one
# landing, and the one after it, whose row DMAs the compute starts.
_V2_ROW_SLOTS = 3
# A tile of at least twice this many rows is computed as two independent row
# chains, so one chain's activation and epilogues overlap the other's matmuls.
_V2_CHAIN_ROWS = 192


def gather_fused_moe_v2_vmem_bytes(
    hidden: int, inter: int, tile_m: int = DEFAULT_TILE_M
) -> int:
    """Upper estimate of the VMEM gather_fused_moe_v2 compiles into: that of
    gather_fused_moe plus its third x row slot."""
    row_slot = tile_m * (packed_sublanes(hidden) + 1) * LANES * 4
    return gather_fused_moe_vmem_bytes(hidden, inter, tile_m) + row_slot


def _fused_moe_v2_kernel(
    gs_ref,  # SMEM int32 [E] (scalar prefetch)
    tok_hbm,  # int32 [rows + tile_m + 128]: token id of each expert-sorted row
    x_hbm,  # uint32 [tokens * (S + 1), 128] quantize_pack_rows rows
    w1_hbm,  # fp8 [E, H, 2I] gate | up
    w1s_hbm,  # f32 [E, 1, 2I]
    w2_hbm,  # fp8 [E, I, H]
    w2s_hbm,  # f32 [E, 1, H]
    y_hbm,  # out: uint32 [rows * S, 128] packed down rows, expert order
    m_expert,
    m_start,
    m_count,
    m_visit,
    v_expert,
    tok_sm,  # SMEM int32 [2 * (tile_m + 128)]: token-id windows of even, odd tiles
    rows_vm,  # VMEM uint32: 3 x slots of tile_m * (S + 1), then 2 y slots of tile_m * S
    w1_vm,
    w1s_vm,
    w2_vm,
    w2s_vm,
    sem_tok,
    sem_x,
    sem_w,
    sem_y,
    *,
    cfg: FusedMoeConfig,
):
    s = cfg.sublanes
    sx = cfg.x_sublanes
    group = cfg.issue_group
    window = cfg.tile_m + LANES
    n_tiles, n_visits = _fill_tiles(
        gs_ref, m_expert, m_start, m_count, m_visit, v_expert, cfg=cfg
    )
    gate_up_rows, down_rows = (
        (_gate_up_token_scales, _down_token_scales)
        if cfg.per_token
        else (_gate_up_block_scales, _down_block_scales)
    )

    # The x and y row slots share one allocation: the row DMA starts (writes
    # of an x slot) that the compute emits between its y stores then stay
    # where they are emitted instead of sinking to the end of the body.
    def x_start(slot):
        return slot * (cfg.tile_m * sx)

    def y_start(slot):
        return (_V2_ROW_SLOTS * sx + slot * s) * cfg.tile_m

    def tok_copy(i):
        w = i % 2
        # A 1-D HBM window DMA must start on a lane-tile boundary.
        base = pl.multiple_of(m_start[i] - m_start[i] % LANES, LANES)
        return pltpu.make_async_copy(
            tok_hbm.at[pl.ds(base, window)],
            tok_sm.at[pl.ds(w * window, window)],
            sem_tok.at[w],
        )

    def issued_rows(i):
        # Rows are issued in whole groups; the tail of the last group fetches
        # the next tile's (or padding) tokens into rows nothing reads.
        return pl.cdiv(m_count[i], group) * group

    def row_copy(token, slot, r):
        return pltpu.make_async_copy(
            x_hbm.at[pl.ds(token * sx, sx)],
            rows_vm.at[pl.ds(x_start(slot) + r * sx, sx)],
            sem_x.at[slot],
        )

    def tile_tokens(i):
        """Tile i's first entry in tok_sm and its x row slot."""
        return (i % 2) * window + m_start[i] % LANES, i % _V2_ROW_SLOTS

    def issue_rows_loop(i):
        first, slot = tile_tokens(i)

        def body(g, carry):
            for u in range(group):
                r = g * group + u
                row_copy(tok_sm[first + r], slot, r).start()
            return carry

        lax.fori_loop(0, issued_rows(i) // group, body, 0)

    def rows_issue(i, valid):
        """emit(r0, r1): straight-line starts of tile ``i``'s rows [r0, r1),
        each predicated on the tile having that row (none unless ``valid``)."""
        first, slot = tile_tokens(i)
        count = jnp.where(valid, issued_rows(i), 0)

        def emit(r0, r1):
            for r in range(r0, r1):
                token = tok_sm[first + r]
                start = functools.partial(
                    lambda token, r: row_copy(token, slot, r).start(), token, r
                )
                pl.when(r < count)(start)

        return emit

    def rows_wait(i):
        slot = i % _V2_ROW_SLOTS
        # Waits count bytes, so one descriptor the size of the issued rows
        # retires all of their DMAs whatever order they landed in.
        rows = rows_vm.at[pl.ds(x_start(slot), issued_rows(i) * sx)]
        pltpu.make_async_copy(rows, rows, sem_x.at[slot]).wait()

    def weight_copies(e, wslot):
        pairs = ((w1_hbm, w1_vm), (w1s_hbm, w1s_vm), (w2_hbm, w2_vm), (w2s_hbm, w2s_vm))
        return [
            pltpu.make_async_copy(src.at[e], dst.at[wslot], sem_w.at[wslot])
            for src, dst in pairs
        ]

    def out_copy(i):
        slot = i % 2
        rows = m_count[i] * s
        return pltpu.make_async_copy(
            rows_vm.at[pl.ds(y_start(slot), rows)],
            y_hbm.at[pl.ds(m_start[i] * s, rows)],
            sem_y.at[slot],
        )

    # Tiles 0 and 1 issue their rows before the loop; tile i + 2's rows are
    # started by tile i's compute.
    @pl.when(n_tiles > 0)
    def _():
        tok_copy(0).start()
        for c in weight_copies(v_expert[0], 0):
            c.start()
        tok_copy(0).wait()
        issue_rows_loop(0)

        @pl.when(n_tiles > 1)
        def _():
            tok_copy(1).start()
            tok_copy(1).wait()
            issue_rows_loop(1)

            @pl.when(n_tiles > 2)
            def _():
                tok_copy(2).start()

    def compute(x_slot, y_slot, wslot, emit, height):
        chains = 2 if height >= 2 * _V2_CHAIN_ROWS else 1
        rows = height // chains
        weights = (w1_vm.at[wslot], w1s_vm.at[wslot], w2_vm.at[wslot], w2s_vm.at[wslot])

        def lane_blocks(r0):
            return lambda j: rows_vm[
                pl.ds(x_start(x_slot) + r0 * sx + j, rows, stride=sx), :
            ]

        gate_ups = [
            gate_up_rows(lane_blocks(c * rows), weights, cfg, rows)
            for c in range(chains)
        ]
        per_store = pl.cdiv(cfg.tile_m, chains * s)
        for c in range(chains):
            for j, words in down_rows(gate_ups[c], weights, cfg, rows):
                k = c * s + j
                emit(
                    min(cfg.tile_m, k * per_store), min(cfg.tile_m, (k + 1) * per_store)
                )
                out_rows = pl.ds(y_start(y_slot) + c * rows * s + j, rows, stride=s)
                rows_vm[out_rows, :] = words

    def tile_loop(i, carry):
        ahead = i + 2
        has_ahead = ahead < n_tiles

        @pl.when(has_ahead)
        def _():
            tok_copy(ahead).wait()

        # Window (i + 3) % 2 last held tile i + 1's tokens, which tile i - 1
        # issued: refill it a whole tile before tile i + 1 issues from it.
        @pl.when(ahead + 1 < n_tiles)
        def _():
            tok_copy(ahead + 1).start()

        visit = m_visit[i]
        wslot = visit % 2
        first_of_visit = jnp.logical_or(i == 0, m_visit[jnp.maximum(i - 1, 0)] != visit)

        @pl.when(first_of_visit)
        def _():
            for c in weight_copies(m_expert[i], wslot):
                c.wait()

            @pl.when(visit + 1 < n_visits)
            def _():
                for c in weight_copies(v_expert[visit + 1], 1 - wslot):
                    c.start()

        rows_wait(i)

        @pl.when(i >= 2)
        def _():
            out_copy(i - 2).wait()

        emit = rows_issue(jnp.minimum(ahead, n_tiles - 1), has_ahead)
        rung = (m_count[i] - 1) // cfg.height_step
        lax.switch(
            rung,
            [
                functools.partial(
                    compute,
                    i % _V2_ROW_SLOTS,
                    i % 2,
                    wslot,
                    emit,
                    cfg.height_step * (b + 1),
                )
                for b in range(cfg.tile_m // cfg.height_step)
            ],
        )
        out_copy(i).start()
        return carry

    lax.fori_loop(0, n_tiles, tile_loop, 0)

    @pl.when(n_tiles >= 2)
    def _():
        out_copy(n_tiles - 2).wait()

    @pl.when(n_tiles >= 1)
    def _():
        out_copy(n_tiles - 1).wait()


@functools.partial(
    jax.jit,
    static_argnames=(
        "act",
        "per_token",
        "tile_m",
        "height_step",
        "issue_group",
        "vmem_limit_bytes",
        "interpret",
    ),
)
def gather_fused_moe_v2(
    x_packed: jax.Array,  # uint32 [tokens * (S + 1), 128] (quantize_pack_rows)
    token_indices_sorted: jax.Array,  # int32 [rows]
    group_sizes: jax.Array,  # int32 [E]
    w1: jax.Array,  # fp8 [E, H, 2I]: gate columns [0, I), up columns [I, 2I)
    w1_scale: jax.Array,  # f32 per-channel [E, 1, 1, 2I] or [E, 1, 2I]
    w2: jax.Array,  # fp8 [E, I, H]
    w2_scale: jax.Array,  # f32 per-channel [E, 1, 1, H] or [E, 1, H]
    *,
    act: str,
    per_token: bool = False,
    tile_m: int = DEFAULT_TILE_M,
    height_step: int = 128,
    issue_group: int = 16,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """`gather_fused_moe` with its row DMAs started by the compute.

    Same contract and, without ``per_token``, the same arithmetic and output
    bits. Each tile's rows are issued two tiles ahead as straight-line
    predicated DMA starts placed between the compute's output stores, where
    they co-issue with the vector work, instead of in a scalar loop that runs
    before each tile's compute. Tiles of at least 384 rows run as two row
    chains. With ``per_token`` the rows come from
    ``quantize_pack_rows(per_token=True)``: one activation scale per row for
    gate_up and one per row of the activation for down, each dequantizing an
    F32 product accumulated over the whole contraction.
    """
    num_experts, hidden, n1 = w1.shape
    inter = n1 // 2
    rows = token_indices_sorted.shape[0]
    s = packed_sublanes(hidden)
    if w2.shape != (num_experts, inter, hidden):
        raise ValueError(f"w2 {w2.shape} does not match w1 {w1.shape}")
    if inter % LANES:
        raise ValueError(f"intermediate {inter} must be a multiple of {LANES}")
    if tile_m % height_step or tile_m % issue_group:
        raise ValueError(
            f"{tile_m=} is not a multiple of {height_step=} and {issue_group=}"
        )
    if x_packed.shape[0] % (s + 1):
        raise ValueError(f"x_packed {x_packed.shape} is not quantize_pack_rows output")
    w1_scale = w1_scale.reshape(num_experts, 1, n1).astype(jnp.float32)
    w2_scale = w2_scale.reshape(num_experts, 1, hidden).astype(jnp.float32)
    # The token-id windows read up to tile_m + 128 entries past a tile's start.
    tok = jnp.pad(token_indices_sorted.astype(jnp.int32), (0, tile_m + LANES))
    max_tiles = pl.cdiv(rows, tile_m) + num_experts
    cfg = FusedMoeConfig(
        num_experts=num_experts,
        hidden_size=hidden,
        inter_size=inter,
        tile_m=tile_m,
        height_step=height_step,
        max_tiles=max_tiles,
        act=act,
        mxu_cols=256,
        issue_group=issue_group,
        per_token=per_token,
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    smem_tiles = [pltpu.SMEM((max_tiles,), jnp.int32) for _ in range(4)]
    row_sublanes = tile_m * (_V2_ROW_SLOTS * (s + 1) + 2 * s)
    return pl.pallas_call(
        functools.partial(_fused_moe_v2_kernel, cfg=cfg),
        out_shape=jax.ShapeDtypeStruct((rows * s, LANES), jnp.uint32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[hbm] * 6,
            out_specs=hbm,
            scratch_shapes=[
                *smem_tiles,
                pltpu.SMEM((num_experts,), jnp.int32),
                pltpu.SMEM((2 * (tile_m + LANES),), jnp.int32),
                pltpu.VMEM((row_sublanes, LANES), jnp.uint32),
                pltpu.VMEM((2, hidden, n1), w1.dtype),
                pltpu.VMEM((2, 1, n1), jnp.float32),
                pltpu.VMEM((2, inter, hidden), w2.dtype),
                pltpu.VMEM((2, 1, hidden), jnp.float32),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((_V2_ROW_SLOTS,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes
            or int(pltpu.get_tpu_info().vmem_capacity_bytes * VMEM_LIMIT_FRACTION),
            disable_bounds_checks=True,
        ),
        name=(
            f"gather_fused_moe_v2-e_{num_experts}-h_{hidden}-i_{inter}-tm_{tile_m}"
            f"{'-token' if per_token else ''}"
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
    )(group_sizes.astype(jnp.int32), tok, x_packed, w1, w1_scale, w2, w2_scale)


# ---------------------------------------------------------------- combine


# Masked combine route entries: (row << _SLOT_BITS) | buffer slot, in int32.
_SLOT_BITS = 12
# Expert-order rows a masked combine can address.
MASKED_COMBINE_MAX_ROWS = 1 << (31 - _SLOT_BITS)
_ISSUE_GROUP = 8


def _combine_kernel(route_smem, *rest, tt, topk, s, masked, skip_zero_weights, into):
    if into:  # the aliased output buffer, written through out_ref
        rest = rest[:-4] + rest[-3:]
    if masked:
        count_smem, w_ref, pos_ref, y_hbm, out_ref, buf, sem = rest
    else:
        w_ref, y_hbm, out_ref, buf, sem = rest
    i = pl.program_id(0)
    routes = tt * topk

    def issued_rows(tile):
        return pl.cdiv(count_smem[tile], _ISSUE_GROUP) * _ISSUE_GROUP

    def issue(tile, slot):
        if masked:
            # The tile's valid routes come first in its entry list; the rest
            # point at row 0 and the spare slot, which compute never reads.
            def group(g, carry):
                for u in range(_ISSUE_GROUP):
                    entry = route_smem[tile * routes + g * _ISSUE_GROUP + u]
                    row = jnp.right_shift(entry, _SLOT_BITS)
                    dst = jnp.bitwise_and(entry, (1 << _SLOT_BITS) - 1)
                    pltpu.make_async_copy(
                        y_hbm.at[pl.ds(row * s, s)],
                        buf.at[slot, pl.ds(dst * s, s)],
                        sem.at[slot],
                    ).start()
                return carry

            lax.fori_loop(0, issued_rows(tile) // _ISSUE_GROUP, group, 0)
            return

        def body(t, carry):
            for k in range(topk):
                row = route_smem[(tile * tt + t) * topk + k]
                pltpu.make_async_copy(
                    y_hbm.at[pl.ds(row * s, s)],
                    buf.at[slot, pl.ds((k * tt + t) * s, s)],
                    sem.at[slot],
                ).start()
            return carry

        lax.fori_loop(0, tt, body, 0)

    @pl.when(i == 0)
    def _():
        issue(0, 0)

    slot = i % 2

    @pl.when(i + 1 < pl.num_programs(0))
    def _():
        issue(i + 1, 1 - slot)

    if masked:
        # Waits count bytes: one descriptor the size of the issued rows.
        rows = pl.ds(0, issued_rows(i) * s)

        @pl.when(count_smem[i] > 0)
        def _():
            pltpu.make_async_copy(
                buf.at[slot, rows], buf.at[slot, rows], sem.at[slot]
            ).wait()

    else:
        pltpu.make_async_copy(buf.at[slot], buf.at[slot], sem.at[slot]).wait()
    # BF16 rows times BF16 weights are exact in F32; sum in F32, round once.
    w = w_ref[...].astype(jnp.float32)
    zero_skipped = masked or skip_zero_weights
    if masked:
        valid = pos_ref[...] >= 0
    elif skip_zero_weights:
        # The row read for a zero-weight route need not be finite.
        valid = w != 0
    for j in range(s):
        acc_lo = jnp.zeros((tt, LANES), jnp.float32)
        acc_hi = jnp.zeros((tt, LANES), jnp.float32)
        for k in range(topk):
            words = buf[slot, pl.ds(k * tt * s + j, tt, stride=s), :]
            lo, hi = _unpack_lane_block(words)
            wk = w[:, k : k + 1]
            lo, hi = lo * wk, hi * wk
            if zero_skipped:
                # Skipped routes' slots hold stale contents or a row read
                # for a zero weight.
                lo = jnp.where(valid[:, k : k + 1], lo, 0.0)
                hi = jnp.where(valid[:, k : k + 1], hi, 0.0)
            acc_lo += lo
            acc_hi += hi
        out_ref[:, PACKED_COLUMNS * j : PACKED_COLUMNS * j + LANES] = acc_lo.astype(
            out_ref.dtype
        )
        out_ref[:, PACKED_COLUMNS * j + LANES : PACKED_COLUMNS * (j + 1)] = (
            acc_hi.astype(out_ref.dtype)
        )


def _masked_route_entries(positions: jax.Array, tt: int) -> tuple[jax.Array, jax.Array]:
    """Per token tile, its valid routes first as packed (row, slot) entries.

    Slot ``k * tt + t`` holds route ``(t, k)``'s row; entries past a tile's
    count read row 0 into the spare slot ``tt * topk``.
    """
    tokens, topk = positions.shape
    routes = tt * topk
    by_tile = positions.reshape(tokens // tt, routes)  # entry t * topk + k
    valid = by_tile >= 0
    order = jnp.argsort(~valid, axis=1, stable=True)
    rows = jnp.take_along_axis(by_tile, order, axis=1)
    slots = (order % topk) * tt + order // topk
    entries = jnp.where(
        jnp.take_along_axis(valid, order, axis=1),
        jnp.left_shift(rows, _SLOT_BITS) | slots,
        routes,
    )
    return entries.reshape(-1).astype(jnp.int32), valid.sum(axis=1, dtype=jnp.int32)


@functools.partial(
    jax.jit,
    static_argnames=(
        "hidden_size",
        "tile_tokens",
        "masked",
        "skip_zero_weights",
        "out_tokens",
        "out_start",
        "interpret",
    ),
)
def gather_combine(
    y_packed: jax.Array,  # uint32 [rows * S, 128] (gather_fused_moe*)
    positions: jax.Array,  # int32 [tokens, topk]: expert-order row of each route
    weights: jax.Array,  # BF16 [tokens, topk]
    *,
    hidden_size: int,
    tile_tokens: int = 64,
    masked: bool = False,
    skip_zero_weights: bool = False,
    into: jax.Array | None = None,
    out_tokens: int | None = None,
    out_start: int = 0,
    interpret: bool = False,
) -> jax.Array:
    """BF16 ``out[t] = sum_k weights[t, k] * row(positions[t, k])``.

    ``masked=True`` admits negative positions: those routes contribute zero
    and are never read, and only the valid routes are fetched. With
    ``skip_zero_weights`` every route is fetched and zero-weight routes
    contribute zero whatever their row holds.

    The result is rows ``[out_start, out_start + tokens)`` of an
    ``[out_tokens, hidden]`` array, the other rows unwritten: of ``into``
    (donated), or of a new array.
    """
    tokens, topk = weights.shape
    s = packed_sublanes(hidden_size)
    if into is not None:
        out_tokens = into.shape[0]
    out_tokens = tokens if out_tokens is None else out_tokens
    if tokens % tile_tokens or out_start % tile_tokens:
        raise ValueError(
            f"{tokens=} or {out_start=} is not a multiple of {tile_tokens=}"
        )
    if out_start + tokens > out_tokens:
        raise ValueError(
            f"rows [{out_start}, {out_start + tokens}) exceed {out_tokens}"
        )
    routes = tile_tokens * topk
    if masked and (
        routes >= 1 << _SLOT_BITS or y_packed.shape[0] // s >= MASKED_COMBINE_MAX_ROWS
    ):
        raise ValueError(
            f"{routes=} or {y_packed.shape[0] // s} rows exceed the entry fields"
        )
    positions = positions.astype(jnp.int32)
    route_block = pl.BlockSpec((tile_tokens, topk), lambda i, *_: (i, 0))
    if masked:
        entries, counts = _masked_route_entries(positions, tile_tokens)
        prefetch = [entries, counts]
        in_specs = [route_block, route_block]
        operands = [weights, positions]
        slots = routes + 1  # one spare slot for the padding entries
    else:
        prefetch = [positions.reshape(-1)]
        in_specs = [route_block]
        operands = [weights]
        slots = routes
    return pl.pallas_call(
        functools.partial(
            _combine_kernel,
            tt=tile_tokens,
            topk=topk,
            s=s,
            masked=masked,
            skip_zero_weights=skip_zero_weights,
            into=into is not None,
        ),
        out_shape=jax.ShapeDtypeStruct((out_tokens, s * PACKED_COLUMNS), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(prefetch),
            grid=(tokens // tile_tokens,),
            in_specs=[*in_specs, pl.BlockSpec(memory_space=pltpu.HBM)]
            + ([pl.BlockSpec(memory_space=pltpu.HBM)] if into is not None else []),
            out_specs=pl.BlockSpec(
                (tile_tokens, s * PACKED_COLUMNS),
                lambda i, *_: (i + out_start // tile_tokens, 0),
            ),
            scratch_shapes=[
                pltpu.VMEM((2, slots * s, LANES), jnp.uint32),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), disable_bounds_checks=True
        ),
        name=(
            f"gather_combine-k_{topk}-s_{s}-tt_{tile_tokens}-m{int(masked)}"
            f"{'-z' if skip_zero_weights else ''}"
        ),
        input_output_aliases=(
            {len(prefetch) + len(operands) + 1: 0} if into is not None else {}
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
    )(*prefetch, *operands, y_packed, *([into] if into is not None else []))


def gather_combine_in_chunks(
    y_packed: jax.Array,  # uint32 [rows * S, 128] (gather_fused_moe)
    positions: jax.Array,  # int32 [tokens, topk]: expert-order row of each route
    weights: jax.Array,  # BF16 [tokens, topk]
    *,
    hidden_size: int,
    max_routes: int,
    tile_tokens: int = 64,
    masked: bool = False,
    skip_zero_weights: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """`gather_combine` run on token chunks of at most `max_routes` routes.

    A call scalar-prefetches one int32 per route into SMEM, which bounds the
    routes a single call takes; every chunk reads the same `y_packed` and
    writes its rows of one output buffer, so no concatenation follows.
    """
    tokens, topk = weights.shape
    chunk = max_routes // topk // tile_tokens * tile_tokens
    if not chunk:
        raise ValueError(f"{max_routes=} holds no {tile_tokens}-token tile")
    combine = functools.partial(
        gather_combine,
        y_packed,
        hidden_size=hidden_size,
        tile_tokens=tile_tokens,
        masked=masked,
        skip_zero_weights=skip_zero_weights,
        interpret=interpret,
    )
    if tokens <= chunk:
        return combine(positions, weights)
    out = None
    for start in range(0, tokens, chunk):
        rows = slice(start, start + chunk)
        out = combine(
            positions[rows],
            weights[rows],
            into=out,
            out_tokens=None if out is not None else tokens,
            out_start=start,
        )
    return out
