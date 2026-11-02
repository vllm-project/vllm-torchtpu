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
# tile, with fewer tail tiles than 512.
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


@functools.partial(jax.jit, static_argnames=("tile_tokens", "interpret"))
def quantize_pack_rows(
    x: jax.Array, *, tile_tokens: int = 256, interpret: bool = False
) -> jax.Array:
    """BF16 ``[n, hidden]`` -> the expert kernel's rows, uint32 ``[n * (S + 1), 128]``.

    Sublanes ``[0, S)`` of a row hold ``bf16(x * scale_inv)`` per 512-column
    block in the packed form; lane ``b`` of sublane ``S`` holds block ``b``'s
    BF16 scale as F32 bits.
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
    return pl.pallas_call(
        functools.partial(_quantize_pack_kernel, s=s, tt=tile_tokens, hidden=h),
        out_shape=jax.ShapeDtypeStruct((n * (s + 1), LANES), jnp.uint32),
        grid=(n // tile_tokens,),
        in_specs=[pl.BlockSpec((tile_tokens, h), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((tile_tokens * (s + 1), LANES), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name=f"quantize_pack_rows-h_{h}-tt_{tile_tokens}",
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

    @property
    def sublanes(self) -> int:
        return packed_sublanes(self.hidden_size)

    @property
    def x_sublanes(self) -> int:
        return self.sublanes + 1


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

        scale_words = lane_block(s)
        # gate_up: per 512-column block, the FP8 cast of the stored
        # bf16(x * scale_inv) as gmm_v2 does, then BF16 accumulation of the
        # dequantized block products.
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
                    w1_vm[wslot, k0:k1, n0 : n0 + cfg.mxu_cols],
                    preferred_element_type=jnp.float32,
                ).astype(jnp.bfloat16)
                prod *= block_scale
                prod *= w1s_vm[wslot, :, n0 : n0 + cfg.mxu_cols].astype(jnp.bfloat16)
                acc[n] += prod
        gate_up = jnp.concatenate(acc, axis=1)
        mid = apply_gated_act(
            gate_up[:, : cfg.inter_size], gate_up[:, cfg.inter_size :], cfg.act
        )

        # down: per 512-column block of the intermediate, as gmm_v2 does.
        mid_blocks = []
        for k0 in range(0, cfg.inter_size, QUANT_BLOCK):
            k1 = min(cfg.inter_size, k0 + QUANT_BLOCK)
            mid_q, mid_scale = _quantize_rows(mid[:, k0:k1])
            mid_blocks.append((k0, k1, mid_q, mid_scale.astype(jnp.bfloat16)))
        for j in range(s):
            n0 = j * PACKED_COLUMNS
            halves = []
            for h0 in range(n0, n0 + PACKED_COLUMNS, cfg.mxu_cols):
                down_acc = jnp.zeros((height, cfg.mxu_cols), jnp.bfloat16)
                for k0, k1, mid_q, mid_scale in mid_blocks:
                    prod = jnp.dot(
                        mid_q,
                        w2_vm[wslot, k0:k1, h0 : h0 + cfg.mxu_cols],
                        preferred_element_type=jnp.float32,
                    ).astype(jnp.bfloat16)
                    prod *= mid_scale
                    prod *= w2s_vm[wslot, :, h0 : h0 + cfg.mxu_cols].astype(
                        jnp.bfloat16
                    )
                    down_acc += prod
                halves.append(down_acc)
            down = jnp.concatenate(halves, axis=1)
            y_vm[slot, pl.ds(j, height, stride=s), :] = _pack_lane_block(
                down[:, :LANES], down[:, LANES:]
            )

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


# ---------------------------------------------------------------- combine


# Masked combine route entries: (row << _SLOT_BITS) | buffer slot.
_SLOT_BITS = 12
_ISSUE_GROUP = 8


def _combine_kernel(route_smem, *rest, tt, topk, s, masked):
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
    if masked:
        valid = pos_ref[...] >= 0
    for j in range(s):
        acc_lo = jnp.zeros((tt, LANES), jnp.float32)
        acc_hi = jnp.zeros((tt, LANES), jnp.float32)
        for k in range(topk):
            words = buf[slot, pl.ds(k * tt * s + j, tt, stride=s), :]
            lo, hi = _unpack_lane_block(words)
            wk = w[:, k : k + 1]
            lo, hi = lo * wk, hi * wk
            if masked:
                # Slots of skipped routes hold stale buffer contents.
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
    jax.jit, static_argnames=("hidden_size", "tile_tokens", "masked", "interpret")
)
def gather_combine(
    y_packed: jax.Array,  # uint32 [rows * S, 128] (gather_fused_moe)
    positions: jax.Array,  # int32 [tokens, topk]: expert-order row of each route
    weights: jax.Array,  # BF16 [tokens, topk]
    *,
    hidden_size: int,
    tile_tokens: int = 64,
    masked: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """BF16 ``out[t] = sum_k weights[t, k] * row(positions[t, k])``.

    ``masked=True`` admits negative positions: those routes contribute zero
    and are never read, and only the valid routes are fetched.
    """
    tokens, topk = weights.shape
    s = packed_sublanes(hidden_size)
    if tokens % tile_tokens:
        raise ValueError(f"{tokens=} is not a multiple of {tile_tokens=}")
    routes = tile_tokens * topk
    if masked and (routes >= 1 << _SLOT_BITS or y_packed.shape[0] // s >= 1 << 19):
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
            _combine_kernel, tt=tile_tokens, topk=topk, s=s, masked=masked
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, s * PACKED_COLUMNS), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(prefetch),
            grid=(tokens // tile_tokens,),
            in_specs=[*in_specs, pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(
                (tile_tokens, s * PACKED_COLUMNS), lambda i, *_: (i, 0)
            ),
            scratch_shapes=[
                pltpu.VMEM((2, slots * s, LANES), jnp.uint32),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), disable_bounds_checks=True
        ),
        name=f"gather_combine-k_{topk}-s_{s}-tt_{tile_tokens}-m{int(masked)}",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(*prefetch, *operands, y_packed)
