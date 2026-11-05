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
"""VMEM gather-fused MoE expert and combine kernels for grouped-matmul MoE.

The standard grouped-matmul path materializes every routed row twice in HBM:
once in expert order before the gate-up projection (``hidden[token_indices]``)
and once in token order after the down projection, each ``[tokens * topk,
hidden]`` BF16, before summing across ``topk``. These kernels keep only the
expert-order down-projection rows in HBM:

* ``gather_fused_moe`` walks expert tiles of the expert-sorted routing order,
  DMAs each tile's token rows directly from ``quantize_pack_rows(hidden)`` by
  token id, executes gate-up, gated activation, and down projection with the
  intermediate activation in VMEM, and writes packed down-projection rows in
  expert order.
* ``gather_combine`` (and ``gather_combine_in_chunks``) DMAs each token's
  top-k expert rows by their expert-order positions and accumulates the
  weighted sum in FP32.

Rows cross HBM in a packed uint32 layout so one logical BF16 row is a single
contiguous DMA: BF16 ``[n, hidden]`` is stored as uint32 ``[n * S, 128]`` with
``S = hidden // 256``, row ``r`` occupying sublanes ``[S * r, S * r + S)`` and
word ``(S * r + j, l)`` storing column ``256 * j + l`` in the low 16 bits and
``256 * j + 128 + l`` in the high 16 bits. ``quantize_pack_rows`` appends one
extra sublane per token row holding per-512-column FP8 block scales so each
token is scaled once rather than ``topk`` times.
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
# Maximum number of height_step rungs compiled into the lax.switch dispatch.
_MAX_HEIGHT_RUNGS = 5
# Fraction of per-core VMEM capacity budgeted for gather_fused_moe.
VMEM_LIMIT_FRACTION = 0.85
# BF16 columns carried by one packed uint32 lane block (low and high 16-bit halves).
PACKED_COLUMNS = 2 * LANES
# Per-row activation quantization block size matching gmm_v2.
QUANT_BLOCK = 512
FP8 = jnp.float8_e4m3fn
_FP8_MAX = float(jnp.finfo(FP8).max)
_HI_MASK = 0xFFFF0000


def _get_tpu_info_or_cpu_fallback() -> pltpu.TpuInfo:
    """Return active worker ``TpuInfo``, or a static TPU chip fallback on CPU."""
    try:
        return pltpu.get_tpu_info()
    except ValueError:
        return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def _resolve_interpret(
    interpret: bool | pltpu.InterpretParams,
) -> bool | pltpu.InterpretParams:
    if isinstance(interpret, pltpu.InterpretParams):
        return interpret
    return pltpu.InterpretParams() if interpret else False


def packed_sublanes(hidden_size: int) -> int:
    if hidden_size % PACKED_COLUMNS:
        raise ValueError(
            f"hidden size {hidden_size} is not a multiple of {PACKED_COLUMNS}"
        )
    return hidden_size // PACKED_COLUMNS


def unpack_rows(packed: jax.Array, n: int) -> jax.Array:
    """Unpack uint32 ``[n * S, 128]`` rows into BF16 ``[n, S * 256]``."""
    s = packed.shape[0] // n
    p = packed.reshape(n, s, 1, LANES)
    halves = jnp.concatenate([p & 0xFFFF, p >> 16], axis=2).astype(jnp.uint16)
    return lax.bitcast_convert_type(halves, jnp.bfloat16).reshape(n, s * PACKED_COLUMNS)


def _unpack_lane_block(words: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Unpack uint32 ``[rows, 128]`` into ``(low, high)`` columns in FP32."""
    lo = pltpu.bitcast(jnp.left_shift(words, 16), jnp.float32)
    hi = pltpu.bitcast(jnp.bitwise_and(words, jnp.uint32(_HI_MASK)), jnp.float32)
    return lo, hi


def _bf16_bits(x: jax.Array) -> jax.Array:
    """Return rounded BF16 bit patterns zero-extended to uint32."""
    return pltpu.bitcast(x.astype(jnp.bfloat16), jnp.uint16).astype(jnp.uint32)


def _pack_lane_block(lo: jax.Array, hi: jax.Array) -> jax.Array:
    """Pack BF16 ``(low, high)`` ``[rows, 128]`` halves into uint32 ``[rows, 128]``."""
    return jnp.bitwise_or(_bf16_bits(lo), jnp.left_shift(_bf16_bits(hi), 16))


def _scale_rows(block: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Pre-scale one BF16 activation block as in ``gmm_v2`` before the FP8 cast."""
    abs_max = jnp.max(jnp.abs(block), axis=1, keepdims=True)
    scale = abs_max / _FP8_MAX
    scale_inv = jnp.where(scale == 0, 0, 1 / scale)
    return block * scale_inv, scale


def _quantize_rows(block: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Quantize one BF16 block per row to ``(fp8_values, bf16_scale)``."""
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
    lane = lax.broadcasted_iota(jnp.int32, (tt, LANES), 1)
    words = jnp.zeros((tt, LANES), jnp.uint32)
    for b, block_scale in enumerate(scales):
        bits = jnp.left_shift(_bf16_bits(block_scale), 16)
        words = jnp.where(lane == b, bits, words)
    o_ref[pl.ds(s, tt, stride=s + 1), :] = words


@functools.partial(jax.jit, static_argnames=("tile_tokens", "interpret"))
def quantize_pack_rows(
    x: jax.Array,
    *,
    tile_tokens: int | None = None,
    interpret: bool | pltpu.InterpretParams = False,
) -> jax.Array:
    """Scale and pack BF16 ``[n, hidden]`` into uint32 ``[n * (S + 1), 128]``.

    Sublanes ``[0, S)`` of each row hold ``bf16(x * scale_inv)`` per 512-column
    block; lane ``b`` of sublane ``S`` stores block ``b``'s BF16 scale in the
    high 16 bits as FP32 bits.
    """
    n, h = x.shape
    s = packed_sublanes(h)
    if pl.cdiv(h, QUANT_BLOCK) > LANES:
        raise ValueError(f"hidden size {h} has more than {LANES} quantization blocks")
    if tile_tokens is None:
        tile_tokens = int(_get_tpu_info_or_cpu_fallback().mxu_column_size)
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
        interpret=_resolve_interpret(interpret),
    )(x)


def gather_fused_moe_vmem_bytes(hidden: int, inter: int, tile_m: int) -> int:
    """Upper-bound estimate of the VMEM footprint of ``gather_fused_moe``.

    Accounts for double-buffered whole-expert weights and per-channel scales,
    double-buffered packed input and output row tiles, and live tile-body
    intermediates (gate-up accumulator, per-block BF16/FP8 activations, BF16/FP8
    gated intermediate, and FP32 down-projection tiles).
    """
    s = packed_sublanes(hidden)
    row_bytes = LANES * 4
    buffers = (
        2 * tile_m * (2 * s + 1) * row_bytes
        + 2 * (hidden * 2 * inter + inter * hidden)
        + 2 * 8 * (2 * inter + hidden) * 4
    )
    body = tile_m * (
        2 * inter * 2 + QUANT_BLOCK * 3 + inter * 3 + PACKED_COLUMNS * 4 * 2
    )
    return buffers + body


def select_tile_m(
    hidden: int,
    inter: int,
    *,
    height_step: int = LANES,
    issue_group: int = 16,
    vmem_limit_bytes: int | None = None,
) -> int | None:
    """Select the largest ``tile_m`` rung fitting ``vmem_limit_bytes``, or ``None``."""
    if vmem_limit_bytes is None:
        info = _get_tpu_info_or_cpu_fallback()
        vmem_limit_bytes = int(info.vmem_capacity_bytes * VMEM_LIMIT_FRACTION)
    step = math.lcm(height_step, issue_group)
    max_tile_m = (_MAX_HEIGHT_RUNGS * height_step // step) * step
    for tile_m in range(max_tile_m, step - 1, -step):
        if gather_fused_moe_vmem_bytes(hidden, inter, tile_m) <= vmem_limit_bytes:
            return tile_m
    return None


@dataclasses.dataclass(frozen=True)
class FusedMoeConfig:
    num_experts: int
    hidden_size: int
    inter_size: int
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
    """Populate expert tile metadata in expert-sorted row order."""

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
    gs_ref,
    tok_hbm,
    x_hbm,
    w1_hbm,
    w1s_hbm,
    w2_hbm,
    w2s_hbm,
    y_hbm,
    m_expert,
    m_start,
    m_count,
    m_visit,
    v_expert,
    tok_sm0,
    tok_sm1,
    x_vm,
    w1_vm,
    w1s_vm,
    w2_vm,
    w2s_vm,
    y_vm,
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

    tok_windows = (tok_sm0, tok_sm1)

    def per_slot(slot, fn):
        for static_slot in (0, 1):
            pl.when(slot == static_slot)(functools.partial(fn, static_slot))

    def tok_copy(i, static_slot):
        base = pl.multiple_of(m_start[i] - m_start[i] % LANES, LANES)
        return pltpu.make_async_copy(
            tok_hbm.at[pl.ds(base, cfg.tile_m + LANES)],
            tok_windows[static_slot],
            sem_tok.at[static_slot],
        )

    def issued_rows(i):
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
    x_packed: jax.Array,
    token_indices_sorted: jax.Array,
    group_sizes: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array,
    w2: jax.Array,
    w2_scale: jax.Array,
    *,
    act: str,
    tile_m: int | None = None,
    height_step: int = LANES,
    issue_group: int = 16,
    vmem_limit_bytes: int | None = None,
    interpret: bool | pltpu.InterpretParams = False,
) -> jax.Array:
    """Compute expert-order packed down-projection rows ``[rows * S, 128]``.

    Row ``r`` is the down projection of token ``token_indices_sorted[r]``
    through the expert owning sorted position ``r``. Only the valid prefix
    ``sum(group_sizes)`` is written; trailing positions (remote EP experts or
    skipped padded tokens) are left uninitialized and masked by
    ``gather_combine(..., masked=True)``. ``w1`` stores gate in ``[0, I)`` and
    up in ``[I, 2I)``.
    """
    num_experts, hidden, n1 = w1.shape
    inter = n1 // 2
    rows = token_indices_sorted.shape[0]
    s = packed_sublanes(hidden)
    if w2.shape != (num_experts, inter, hidden):
        raise ValueError(f"w2 {w2.shape} does not match w1 {w1.shape}")
    if inter % LANES:
        raise ValueError(f"intermediate {inter} must be a multiple of {LANES}")
    if x_packed.shape[0] % (s + 1):
        raise ValueError(f"x_packed {x_packed.shape} is not quantize_pack_rows output")

    info = _get_tpu_info_or_cpu_fallback()
    mxu_cols = int(info.mxu_column_size)
    if mxu_cols <= 0 or PACKED_COLUMNS % mxu_cols != 0 or n1 % mxu_cols != 0:
        raise ValueError(
            f"mxu_column_size {mxu_cols} must divide {PACKED_COLUMNS=} and {n1=}"
        )
    resolved_vmem_limit = (
        vmem_limit_bytes
        if vmem_limit_bytes is not None
        else int(info.vmem_capacity_bytes * VMEM_LIMIT_FRACTION)
    )
    if tile_m is None:
        tile_m = select_tile_m(
            hidden,
            inter,
            height_step=height_step,
            issue_group=issue_group,
            vmem_limit_bytes=resolved_vmem_limit,
        )
        if tile_m is None:
            raise ValueError(
                f"Expert weights ({hidden=}, {inter=}) exceed VMEM budget "
                f"{resolved_vmem_limit}"
            )
    if tile_m % height_step or tile_m % issue_group:
        raise ValueError(
            f"{tile_m=} is not a multiple of {height_step=} and {issue_group=}"
        )

    w1_scale = w1_scale.reshape(num_experts, 1, n1).astype(jnp.float32)
    w2_scale = w2_scale.reshape(num_experts, 1, hidden).astype(jnp.float32)
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
        mxu_cols=mxu_cols,
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
            vmem_limit_bytes=resolved_vmem_limit,
            disable_bounds_checks=True,
        ),
        name=f"gather_fused_moe-e_{num_experts}-h_{hidden}-i_{inter}-tm_{tile_m}",
        interpret=_resolve_interpret(interpret),
    )(group_sizes.astype(jnp.int32), tok, x_packed, w1, w1_scale, w2, w2_scale)


# Masked combine route entries pack (row << _SLOT_BITS) | buffer_slot in int32.
_SLOT_BITS = 12
# Maximum expert-order rows addressable by a masked combine entry.
MASKED_COMBINE_MAX_ROWS = 1 << (31 - _SLOT_BITS)
_ISSUE_GROUP = 8
MASKED_COMBINE_MAX_TILE_ROUTES = (1 << _SLOT_BITS) - _ISSUE_GROUP


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
        rows = pl.ds(0, issued_rows(i) * s)

        @pl.when(count_smem[i] > 0)
        def _():
            pltpu.make_async_copy(
                buf.at[slot, rows], buf.at[slot, rows], sem.at[slot]
            ).wait()

    else:
        pltpu.make_async_copy(buf.at[slot], buf.at[slot], sem.at[slot]).wait()

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
    """Pack valid routes first per token tile as ``(row, slot)`` int32 entries.

    Slot ``k * tt + t`` holds route ``(t, k)``'s expert row; trailing invalid
    entries within an ``_ISSUE_GROUP`` target disjoint spare slots
    ``routes .. routes + _ISSUE_GROUP - 1`` so concurrent padding DMAs never
    write to the same VMEM address.
    """
    tokens, topk = positions.shape
    routes = tt * topk
    by_tile = positions.reshape(tokens // tt, routes)
    valid = by_tile >= 0
    order = jnp.argsort(~valid, axis=1, stable=True)
    rows = jnp.take_along_axis(by_tile, order, axis=1)
    slots = (order % topk) * tt + order // topk
    spare_slots = routes + (jnp.arange(routes, dtype=jnp.int32) % _ISSUE_GROUP)[None, :]
    entries = jnp.where(
        jnp.take_along_axis(valid, order, axis=1),
        jnp.left_shift(rows, _SLOT_BITS) | slots,
        spare_slots,
    )
    return entries.reshape(-1).astype(jnp.int32), valid.sum(axis=1, dtype=jnp.int32)


@functools.partial(
    jax.jit,
    static_argnames=("hidden_size", "tile_tokens", "masked", "out_dtype", "interpret"),
)
def gather_combine(
    y_packed: jax.Array,
    positions: jax.Array,
    weights: jax.Array,
    *,
    hidden_size: int,
    tile_tokens: int = 64,
    masked: bool = False,
    out_dtype: jnp.dtype = jnp.bfloat16,
    interpret: bool | pltpu.InterpretParams = False,
) -> jax.Array:
    """Gather expert-order packed rows and sum ``weights[t, k] * row(positions[t, k])``.

    Accepts ``weights`` in ``float32`` or ``bfloat16``, accumulates in
    ``float32``, and returns ``out_dtype`` (default ``bfloat16``). When
    ``masked=True``, negative ``positions`` contribute zero without reading
    uninitialized expert rows.
    """
    tokens, topk = weights.shape
    s = packed_sublanes(hidden_size)
    if tokens % tile_tokens:
        raise ValueError(f"{tokens=} is not a multiple of {tile_tokens=}")
    routes = tile_tokens * topk
    if masked and (
        routes + _ISSUE_GROUP > 1 << _SLOT_BITS
        or y_packed.shape[0] // s >= MASKED_COMBINE_MAX_ROWS
    ):
        raise ValueError(
            f"{routes=} or {y_packed.shape[0] // s} rows exceed masked entry capacity"
        )
    if weights.dtype not in (jnp.bfloat16, jnp.float32):
        weights = weights.astype(jnp.float32)
    positions = positions.astype(jnp.int32)
    route_block = pl.BlockSpec((tile_tokens, topk), lambda i, *_: (i, 0))
    if masked:
        entries, counts = _masked_route_entries(positions, tile_tokens)
        prefetch = [entries, counts]
        in_specs = [route_block, route_block]
        operands = [weights, positions]
        slots = routes + _ISSUE_GROUP
    else:
        prefetch = [positions.reshape(-1)]
        in_specs = [route_block]
        operands = [weights]
        slots = routes
    return pl.pallas_call(
        functools.partial(
            _combine_kernel, tt=tile_tokens, topk=topk, s=s, masked=masked
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, s * PACKED_COLUMNS), out_dtype),
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
        interpret=_resolve_interpret(interpret),
    )(*prefetch, *operands, y_packed)


def gather_combine_in_chunks(
    y_packed: jax.Array,
    positions: jax.Array,
    weights: jax.Array,
    *,
    hidden_size: int,
    max_routes: int,
    tile_tokens: int = 64,
    masked: bool = False,
    out_dtype: jnp.dtype = jnp.bfloat16,
    interpret: bool | pltpu.InterpretParams = False,
) -> jax.Array:
    """Run ``gather_combine`` over token chunks of at most ``max_routes`` routes."""
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
        out_dtype=out_dtype,
        interpret=interpret,
    )
    if tokens <= chunk:
        return combine(positions, weights)
    return jnp.concatenate(
        [
            combine(positions[start : start + chunk], weights[start : start + chunk])
            for start in range(0, tokens, chunk)
        ],
        axis=0,
    )
