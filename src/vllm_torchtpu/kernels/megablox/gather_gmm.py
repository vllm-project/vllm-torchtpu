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
"""Gather-fused MoE expert kernels for TensorCore per-channel FP8 prefill."""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import apply_gated_act
from vllm_torchtpu.kernels.row_blocks import get_vmem_capacity_bytes

LANES = 128
DEFAULT_TILE_M = 640
VMEM_LIMIT_FRACTION = 0.85
PACKED_COLUMNS = 2 * LANES
QUANT_BLOCK = 512
MAX_COMBINE_SMEM_BYTES = 256 * 1024
MAX_COMBINE_ROUTES = MAX_COMBINE_SMEM_BYTES // 4
FP8 = jnp.float8_e4m3fn
_FP8_MAX = float(jnp.finfo(FP8).max)
_HI_MASK = 0xFFFF0000


def packed_sublanes(hidden_size: int) -> int:
    if hidden_size % PACKED_COLUMNS:
        raise ValueError(
            f"hidden size {hidden_size} is not a multiple of {PACKED_COLUMNS}"
        )
    return hidden_size // PACKED_COLUMNS


def _unpack_lane_block(words: jax.Array) -> tuple[jax.Array, jax.Array]:
    lo = pltpu.bitcast(jnp.left_shift(words, 16), jnp.float32)
    hi = pltpu.bitcast(jnp.bitwise_and(words, jnp.uint32(_HI_MASK)), jnp.float32)
    return lo, hi


def _bf16_bits(x: jax.Array) -> jax.Array:
    return pltpu.bitcast(x.astype(jnp.bfloat16), jnp.uint16).astype(jnp.uint32)


def _pack_lane_block(lo: jax.Array, hi: jax.Array) -> jax.Array:
    return jnp.bitwise_or(_bf16_bits(lo), jnp.left_shift(_bf16_bits(hi), 16))


def _quantize_pack_token_kernel(x_ref, o_ref, *, s, tt):
    x = x_ref[...].astype(jnp.float32)
    scale = jnp.max(jnp.abs(x), axis=1, keepdims=True) / _FP8_MAX
    xs = jnp.clip(x * jnp.where(scale == 0, 0, 1 / scale), -_FP8_MAX, _FP8_MAX)
    for j in range(s):
        lo = xs[:, PACKED_COLUMNS * j : PACKED_COLUMNS * j + LANES]
        hi = xs[:, PACKED_COLUMNS * j + LANES : PACKED_COLUMNS * (j + 1)]
        o_ref[pl.ds(j, tt, stride=s + 1), :] = _pack_lane_block(lo, hi)
    o_ref[pl.ds(s, tt, stride=s + 1), :] = jnp.broadcast_to(
        pltpu.bitcast(scale, jnp.uint32), (tt, LANES)
    )


@functools.partial(jax.jit, static_argnames=("tile_tokens", "interpret"))
def quantize_pack_rows(
    x: jax.Array, *, tile_tokens: int = 256, interpret: bool = False
) -> jax.Array:
    n, h = x.shape
    s = packed_sublanes(h)
    if pl.cdiv(h, QUANT_BLOCK) > LANES:
        raise ValueError(f"hidden size {h} has more than {LANES} quantization blocks")
    tile = math.gcd(n, tile_tokens)
    if tile % 8:
        raise ValueError(f"gcd({n=}, {tile_tokens=}) = {tile} is not a multiple of 8")
    tile_tokens = tile
    kernel = functools.partial(_quantize_pack_token_kernel, s=s, tt=tile_tokens)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n * (s + 1), LANES), jnp.uint32),
        grid=(n // tile_tokens,),
        in_specs=[pl.BlockSpec((tile_tokens, h), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((tile_tokens * (s + 1), LANES), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name=f"quantize_pack_rows-h_{h}-tt_{tile_tokens}-token",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(x)


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


def _gate_up_token_scales(lane_block, weights, cfg: FusedMoeConfig, height: int):
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
    del height
    _, _, w2, w2s = weights
    mid = apply_gated_act(
        gate_up[:, : cfg.inter_size], gate_up[:, cfg.inter_size :], cfg.act
    )
    mid_f32 = mid.astype(jnp.float32)
    amax = jnp.max(jnp.abs(mid_f32), axis=1, keepdims=True)
    scale = amax / _FP8_MAX
    scale_inv = jnp.where(scale == 0, 0.0, 1.0 / scale)
    mid_q = jnp.clip(mid_f32 * scale_inv, -_FP8_MAX, _FP8_MAX).astype(FP8)
    for j in range(cfg.sublanes):
        n0 = j * PACKED_COLUMNS
        acc = jnp.dot(
            mid_q, w2[:, n0 : n0 + PACKED_COLUMNS], preferred_element_type=jnp.float32
        )
        down = acc * scale * w2s[:, n0 : n0 + PACKED_COLUMNS]
        yield j, _pack_lane_block(down[:, :LANES], down[:, LANES:])


def _fill_tiles(
    gs_ref, m_expert, m_start, m_count, m_visit, v_expert, *, cfg: FusedMoeConfig
):
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

        return (n_tiles + tiles_e, n_visits + (size > 0).astype(jnp.int32), row + size)

    zero = jnp.int32(0)
    n_tiles, n_visits, _ = lax.fori_loop(
        0, cfg.num_experts, expert_body, (zero, zero, zero)
    )
    return n_tiles, n_visits


_V2_ROW_SLOTS = 3
_V2_CHAIN_ROWS = 192


def gather_fused_moe_v2_vmem_bytes(
    hidden: int, inter: int, tile_m: int = DEFAULT_TILE_M
) -> int:
    s = packed_sublanes(hidden)
    row_bytes = LANES * 4
    buffers = (
        tile_m * (_V2_ROW_SLOTS * (s + 1) + 2 * s) * row_bytes
        + 2 * (hidden * 2 * inter + inter * hidden)
        + 2 * 8 * (2 * inter + hidden) * 4
    )
    body = tile_m * (
        2 * inter * 2 + QUANT_BLOCK * 3 + inter * 3 + PACKED_COLUMNS * 4 * 2
    )
    return buffers + body


def _fused_moe_v2_kernel(
    gs_ref,  # SMEM int32 [E] (scalar prefetch)
    tok_hbm,  # int32 [rows + tile_m + 128]: token id per expert-sorted row
    x_hbm,  # uint32 [tokens * (S + 1), 128] quantize_pack_rows output
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
    tok_sm,  # SMEM int32 [2 * (tile_m + 128)]: token-id windows of even/odd tiles
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

    def x_start(slot):
        return slot * (cfg.tile_m * sx)

    def y_start(slot):
        return (_V2_ROW_SLOTS * sx + slot * s) * cfg.tile_m

    def tok_copy(i):
        w = i % 2
        base = pl.multiple_of(m_start[i] - m_start[i] % LANES, LANES)
        return pltpu.make_async_copy(
            tok_hbm.at[pl.ds(base, window)],
            tok_sm.at[pl.ds(w * window, window)],
            sem_tok.at[w],
        )

    def issued_rows(i):
        return pl.cdiv(m_count[i], group) * group

    def row_copy(token, slot, r):
        return pltpu.make_async_copy(
            x_hbm.at[pl.ds(token * sx, sx)],
            rows_vm.at[pl.ds(x_start(slot) + r * sx, sx)],
            sem_x.at[slot],
        )

    def tile_tokens(i):
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
        """Return `emit(r0, r1)` issuing straight-line predicated DMA starts."""
        first, slot = tile_tokens(i)
        count = jnp.where(valid, issued_rows(i), 0)

        def emit(r0, r1):
            for r in range(r0, r1):
                token = tok_sm[first + r]

                @pl.when(r < count)
                def _start_row(tok=token, row=r):
                    row_copy(tok, slot, row).start()

        return emit

    def rows_wait(i):
        slot = i % _V2_ROW_SLOTS
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
            _gate_up_token_scales(lane_blocks(c * rows), weights, cfg, rows)
            for c in range(chains)
        ]
        per_store = pl.cdiv(cfg.tile_m, chains * s)
        for c in range(chains):
            for j, words in _down_token_scales(gate_ups[c], weights, cfg, rows):
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
    tile_m: int = DEFAULT_TILE_M,
    height_step: int = 128,
    issue_group: int = 16,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> jax.Array:
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
    if x_packed.shape[0] % (s + 1) or x_packed.shape[0] == 0:
        raise ValueError(f"x_packed {x_packed.shape} is not quantize_pack_rows output")
    num_tokens = x_packed.shape[0] // (s + 1)
    w1_scale = w1_scale.reshape(num_experts, 1, n1).astype(jnp.float32)
    w2_scale = w2_scale.reshape(num_experts, 1, hidden).astype(jnp.float32)
    clamped = jnp.clip(token_indices_sorted.astype(jnp.int32), 0, num_tokens - 1)
    tok = jnp.pad(clamped, (0, tile_m + LANES))
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
    row_sublanes = tile_m * (_V2_ROW_SLOTS * (s + 1) + 2 * s)
    resolved_vmem = vmem_limit_bytes or int(
        get_vmem_capacity_bytes() * VMEM_LIMIT_FRACTION
    )
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
            vmem_limit_bytes=resolved_vmem, disable_bounds_checks=True
        ),
        name=(
            f"gather_fused_moe_v2-e_{num_experts}-h_{hidden}"
            f"-i_{inter}-tm_{tile_m}-token"
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
    )(group_sizes.astype(jnp.int32), tok, x_packed, w1, w1_scale, w2, w2_scale)


_SLOT_BITS = 12
MASKED_COMBINE_MAX_ROWS = 1 << (31 - _SLOT_BITS)
_ISSUE_GROUP = 8


def _combine_kernel(route_smem, *rest, tt, topk, s, masked, skip_zero_weights, into):
    if into:
        rest = rest[:-4] + rest[-3:]
    count_smem = None
    pos_ref = None
    if masked:
        count_smem, w_ref, pos_ref, y_hbm, out_ref, buf, sem = rest
    else:
        w_ref, y_hbm, out_ref, buf, sem = rest
    i = pl.program_id(0)
    routes = tt * topk

    def issued_rows(tile):
        assert count_smem is not None
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

    if count_smem is not None:
        rows = pl.ds(0, issued_rows(i) * s)

        @pl.when(count_smem[i] > 0)
        def _():
            pltpu.make_async_copy(
                buf.at[slot, rows], buf.at[slot, rows], sem.at[slot]
            ).wait()

    else:
        pltpu.make_async_copy(buf.at[slot], buf.at[slot], sem.at[slot]).wait()

    w = w_ref[...].astype(jnp.float32)
    valid = None
    if pos_ref is not None:
        valid = pos_ref[...] >= 0
    elif skip_zero_weights:
        valid = w != 0
    for j in range(s):
        acc_lo = jnp.zeros((tt, LANES), jnp.float32)
        acc_hi = jnp.zeros((tt, LANES), jnp.float32)
        for k in range(topk):
            words = buf[slot, pl.ds(k * tt * s + j, tt, stride=s), :]
            lo, hi = _unpack_lane_block(words)
            wk = w[:, k : k + 1]
            lo, hi = lo * wk, hi * wk
            if valid is not None:
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


def _masked_route_entries(
    positions: jax.Array, tt: int, num_y_rows: int
) -> tuple[jax.Array, jax.Array]:
    tokens, topk = positions.shape
    routes = tt * topk
    by_tile = positions.reshape(tokens // tt, routes)
    valid = (by_tile >= 0) & (by_tile < num_y_rows)
    order = jnp.argsort(~valid, axis=1, stable=True)
    rows = jnp.clip(
        jnp.take_along_axis(by_tile, order, axis=1), 0, max(num_y_rows - 1, 0)
    )
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
    y_packed: jax.Array,
    positions: jax.Array,
    weights: jax.Array,
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
    tokens, topk = weights.shape
    if tokens * topk > MAX_COMBINE_ROUTES:
        raise ValueError(
            f"{tokens * topk} routes exceed SMEM cap {MAX_COMBINE_ROUTES}; "
            "use gather_combine_in_chunks"
        )
    s = packed_sublanes(hidden_size)
    if y_packed.shape[0] < s or y_packed.shape[0] % s != 0:
        raise ValueError(
            f"y_packed rows {y_packed.shape[0]} must be a positive multiple of {s}"
        )
    num_y_rows = y_packed.shape[0] // s
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
    if masked and (routes >= 1 << _SLOT_BITS or num_y_rows >= MASKED_COMBINE_MAX_ROWS):
        raise ValueError(f"{routes=} or {num_y_rows} rows exceed entry fields")
    positions = positions.astype(jnp.int32)
    route_block = pl.BlockSpec((tile_tokens, topk), lambda i, *_: (i, 0))
    if masked:
        entries, counts = _masked_route_entries(positions, tile_tokens, num_y_rows)
        prefetch = [entries, counts]
        in_specs = [route_block, route_block]
        operands = [weights, positions]
        slots = routes + 1
    else:
        clamped_pos = jnp.clip(positions, 0, max(num_y_rows - 1, 0))
        prefetch = [clamped_pos.reshape(-1)]
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
    y_packed: jax.Array,  # uint32 [rows * S, 128] (gather_fused_moe_v2)
    positions: jax.Array,  # int32 [tokens, topk]: expert-order row of each route
    weights: jax.Array,  # BF16 or F32 [tokens, topk]
    *,
    hidden_size: int,
    max_routes: int,
    tile_tokens: int = 64,
    masked: bool = False,
    skip_zero_weights: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """Run `gather_combine` in token chunks of at most `max_routes` routes."""
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
    assert out is not None
    return out
