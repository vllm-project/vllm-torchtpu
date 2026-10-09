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

"""Fused attention prologue for any AttentionConfig.

One Pallas kernel takes the decoder hidden states, applies the residual add,
input norm and sub-channel fp8 quant, runs the QKV matmul (fp8 or bf16, optional
bias) and, in its epilogue, applies the q/k/v norms, RoPE, gate split and static
KV cast. Q and new KV are written directly in batched_rpa's input layouts. All
config branching is resolved at trace time.

Each token tile (inner_kernel), one group of rows at a time:
  Step 0: residual add, input norm and (fp8) sub-channel quant into x_q.
  Step 1: the QKV matmul of those rows into qkv, K/V columns first.
Then, over the whole tile:
  Step 2: K/V norm, RoPE and fp8 cast into batched_rpa's KV layout.
  Step 3: Q norm and RoPE, packed into batched_rpa's Q layout.
  Step 4: gates copied out.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs as rpa_configs
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel import configs, memory
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel import tiling as tiling_lib
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel.library import (
    layouts,
    norm,
    quant,
)
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel.library import rope as rope_lib

# Weights the kernel copies HBM -> VMEM itself.
_STREAMED = ("w",)


def _two_heads_per_vreg(config: configs.AttentionConfig) -> bool:
    """Whether D = 64 heads 2i and 2i + 1 are processed together in one vreg.

    Every step must then handle both heads at once: no per-head norms, gates or
    v_from_k, and RoPE must be the lane trick.
    """
    q_per_kv = config.num_q_heads // config.num_kv_heads
    return (
        2 * config.head_dim == pltpu.get_tpu_info().num_lanes
        and config.num_kv_heads % 2 == 0  # K/V heads pair up.
        and q_per_kv % 2 == 0  # Q heads pair up within each KV group.
        and config.q_packing == 2  # Q stores both heads in one u32.
        and not config.gate  # Gate columns sit between Q heads.
        and not config.v_from_k  # V is derived from K one head at a time.
        and config.q_norm == config.k_norm == config.v_norm == configs.Norm.NONE
        and (
            config.rope is None
            or rope_lib.lane_rope_eligible(config.rope, config.head_dim)
        )
    )


def _block_order(config: configs.AttentionConfig) -> list[int]:
    """Step 1's column blocks: K/V first, so Step 2 overlaps the Q/gate dots."""
    col_block = tiling_lib.column_block(config)
    num_blocks, first_kv_block = (
        config.num_cols // col_block,
        config.kv_col // col_block,
    )
    assert config.kv_col % col_block == 0
    return [*range(first_kv_block, num_blocks), *range(first_kv_block)]


def inner_kernel(
    # Inputs.
    x_ref: jax.Ref,  # (tile, K) bf16 hidden states
    residual_ref: jax.Ref | None,  # (tile, K) bf16
    cos_ref: jax.Ref | None,  # (tile, n_pairs) bf16
    sin_ref: jax.Ref | None,  # (tile, n_pairs) bf16
    # Outputs.
    # (Hkv, tile, q_per_kv/q_packing, q_packing, align(D, 128)) bf16
    q_ref: jax.Ref,
    # fp8. SEQ_ALONG_LANE: (2Hkv, D/kv_packing, kv_packing, align(tile, 128))
    # HEAD_ALONG_SUBLANE: (tile, 2Hkv/kv_packing, kv_packing, align(D, 128))
    kv_ref: jax.Ref,
    gate_ref: jax.Ref | None,  # (tile, Hq*D) bf16
    residual_out_ref: jax.Ref | None,  # (tile, K) bf16
    # Scratch.
    weights_ref: configs.QkvWeights,  # Refs; w is the streamed VMEM copy.
    x_q_ref: jax.Ref,  # (tile, K) W dtype
    qkv_ref: jax.Ref,  # (tile, N) bf16
    x_scale_ref: jax.Ref | None,  # fp8: (K / block, tile, lanes) f32
    row_scale_ref: jax.Ref | None,  # fp8: (tile, lanes) f32 input-norm 1/rms
    *,
    config: configs.AttentionConfig,
    kv_layout: rpa_configs.KVLayout,
    tiling: tiling_lib.Tiling,
    w_pending: dict[int, Any],
):
    """One token tile: Steps 0-4 of the module docstring.

    w_pending: W's chunk copies not yet waited on.
    """
    num_lanes = pltpu.get_tpu_info().num_lanes
    hq, hkv, d = config.num_q_heads, config.num_kv_heads, config.head_dim
    q_per_kv = hq // hkv
    tile_size, hidden_size = x_ref.shape
    has_input_norm = config.input_norm != configs.Norm.NONE
    col_block = tiling_lib.column_block(config)
    mm_rows, prep_rows = tiling.matmul_rows, tiling.prep_rows
    cos = sin = None
    if cos_ref is not None:
        cos, sin = cos_ref[...], sin_ref[...]

    if config.fp8:
        qb = config.act_quant_block
        assert hidden_size % qb == 0 and col_block % num_lanes == 0

        def add_norm_quant(rows):
            """Step 0 on `rows`, fp8.

            Math is bf16 (two values per lane) except the f32 sum of squares. 1/rms
            is one scale per row, so it doesn't change the quant and is applied in
            Step 1 instead. Scales are stored broadcast across lanes, so
            Step 1 reads them as plain vregs.
            """
            sum_sq = jnp.zeros((prep_rows, 1), jnp.float32)
            for kb in range(hidden_size // qb):
                ks = pl.ds(kb * qb, qb)
                x = x_ref[rows, ks]
                if residual_ref is not None:
                    x = x + residual_ref[rows, ks]
                    residual_out_ref[rows, ks] = x
                if has_input_norm:
                    sum_sq = sum_sq + norm.sum_sq(x.astype(jnp.float32))
                    # Stride-0 load: one vld per vreg replicates the weight on all rows.
                    x = x * norm.load_weight(
                        weights_ref.input_norm_w, kb, prep_rows, x.dtype
                    )
                x_q, x_scale = quant.quantize_block(x)
                x_q_ref[rows, ks] = x_q
                x_scale_ref[kb, rows] = layouts.broadcast_lanes(x_scale)
            if has_input_norm:
                row_scale_ref[rows] = layouts.broadcast_lanes(
                    norm.inv_rms(sum_sq, hidden_size, config.eps)
                )

        add_norm_rows = add_norm_quant

    else:

        def add_norm(rows):
            """Step 0 on `rows`, bf16.

            Two passes over column blocks, the f32 sum of squares and then the
            normed rows (redoing the add), keep only a few vregs live.
            """
            col_blocks = [
                pl.ds(k, min(tiling.norm_cols, hidden_size - k))
                for k in range(0, hidden_size, tiling.norm_cols)
            ]

            def add(ks):
                x = x_ref[rows, ks].astype(jnp.float32)
                if residual_ref is not None:
                    x = x + residual_ref[rows, ks].astype(jnp.float32)
                return x

            inv_rms = None
            if has_input_norm:
                sum_sq = jnp.zeros((prep_rows, 1), jnp.float32)
                for ks in col_blocks:
                    sum_sq = sum_sq + norm.sum_sq(add(ks))
                inv_rms = norm.inv_rms(sum_sq, hidden_size, config.eps)
            w = weights_ref.input_norm_w
            for ks in col_blocks:
                x = add(ks)
                if residual_ref is not None:
                    residual_out_ref[rows, ks] = x.astype(residual_out_ref.dtype)
                x_q_ref[rows, ks] = norm.scale_norm(
                    config.input_norm,
                    x,
                    inv_rms,
                    None if w is None else w[:, ks],
                    x_q_ref.dtype,
                )

        add_norm_rows = add_norm

    assert tile_size % mm_rows == 0 and mm_rows % prep_rows == 0
    order = _block_order(config)

    for m in range(0, tile_size, mm_rows):
        rows = pl.ds(m, mm_rows)
        # Step 0: add, norm and quant of these rows, prep_rows at a time.
        for start in range(m, m + mm_rows, prep_rows):
            add_norm_rows(pl.ds(start, prep_rows))

        # Step 1: projection of these rows, one MXU-wide column block at a time.
        for b in order:
            cols = pl.ds(b * col_block, col_block)
            memory.wait_chunk(w_pending, b * col_block)
            if config.fp8:
                # Each quant block has its own per-row scale, so the partial dots are
                # scaled before they are summed.
                acc = quant.scaled_dot(x_q_ref, x_scale_ref, weights_ref.w, rows, cols)
                if has_input_norm:  # Deferred 1/rms from Step 0.
                    acc = norm.apply_row_scale(acc, row_scale_ref[rows])
            else:
                acc = jnp.dot(
                    x_q_ref[rows],
                    weights_ref.w[:, cols],
                    preferred_element_type=jnp.float32,
                )
            if weights_ref.w_scale is not None:
                acc = acc * weights_ref.w_scale[:, cols]
            if weights_ref.bias is not None:
                acc = acc + weights_ref.bias[:, cols].astype(jnp.float32)
            qkv_ref[rows, cols] = acc.astype(qkv_ref.dtype)

    two_heads_per_vreg = _two_heads_per_vreg(config)
    # LaneRope rotates whole 128-lane vregs: one head, or two D = 64 heads.
    use_lane_rope = rope_lib.lane_rope_eligible(config.rope, d) and (
        d % num_lanes == 0 or two_heads_per_vreg
    )
    if use_lane_rope:
        assert config.rope is not None
        rope = rope_lib.LaneRope(config.rope, d, cos, sin, qkv_ref.dtype)
    else:
        rope = functools.partial(rope_lib.rope, config.rope, cos=cos, sin=sin)

    seq_along_lane = kv_layout == rpa_configs.KVLayout.SEQ_ALONG_LANE
    # D = 64 heads are read two per vreg, then rotated and transposed together.
    heads_per_read = 2 if two_heads_per_vreg else 1

    def read(col):
        return qkv_ref[:, pl.ds(col, heads_per_read * d)]

    def weight(ref):
        return None if ref is None else ref[...]

    def to_kv(y):
        """f32; SEQ_ALONG_LANE: (tile, D) -> (D, tile), tokens on lanes."""
        y = y.astype(jnp.float32)
        return y.T if seq_along_lane else y

    def split_heads(y):
        """A to_kv value of heads_per_read heads -> a list of single heads."""
        if heads_per_read == 1:
            return [y]
        if seq_along_lane:
            return [y[:d], y[d:]]  # (2D, tile): heads are row blocks.
        # (tile, 2D): heads are lane halves.
        # NOTE: A lane-offset slice here miscompiles in the fp8 pack.
        return [y[:, :d], pltpu.roll(y, d, 1)[:, :d]]

    def kv_heads():
        """Yields fp8 K/V heads in cache order: k_0, v_0, k_1, v_1, ..."""
        for h in range(0, hkv, heads_per_read):
            k = norm.norm(
                config.k_norm,
                read(config.k_col(h)),
                weight(weights_ref.k_norm_w),
                config.eps,
            )
            v = norm.norm(config.v_norm, read(config.v_col(h)), None, config.eps)
            for k_h, v_h in zip(split_heads(to_kv(rope(k))), split_heads(to_kv(v))):
                yield quant.quantize_fp8_per_tensor(k_h, config.k_scale)
                yield quant.quantize_fp8_per_tensor(v_h, config.v_scale)

    # Step 2: K/V heads -> kv_ref.
    kv_packing = config.kv_packing
    if seq_along_lane:
        # Each head has its own slab; kv_packing sublanes pack into one u32:
        # (D/kv_packing, kv_packing, kv_lanes) == (D/kv_packing, kv_lanes) u32. A
        # short lone tile's block runs into batched_rpa's token pad, which must be
        # zero.
        kv_lanes = kv_ref.shape[-1]
        for j, y in enumerate(kv_heads()):
            slab = kv_ref.at[j].bitcast(jnp.uint32)
            slab = slab.reshape(d // kv_packing, kv_lanes)
            if kv_lanes != tile_size:
                slab[...] = jnp.zeros(slab.shape, jnp.uint32)
            slab[:, pl.ds(0, tile_size)] = pltpu.bitcast(y, jnp.uint32)
    else:
        # kv_packing heads share each u32: pack them in zero-padded groups.
        heads = list(kv_heads())
        heads += [jnp.zeros_like(heads[0])] * (-len(heads) % kv_packing)
        for g in range(len(heads) // kv_packing):
            group = heads[g * kv_packing : (g + 1) * kv_packing]
            layouts.strided_store_heads(kv_ref, g, layouts.pack_heads(group))

    def packed_q(h, g):
        """Q heads of group g (of KV head h), packed q_packing per u32."""
        first = h * q_per_kv + g * config.q_packing
        if two_heads_per_vreg:
            # Both heads are in one read. Rolling the second head's bits onto the
            # first's lanes packs them into one u32 per lane (pack_heads order);
            # lanes >= D are the layout's pad.
            q = rope(read(config.q_col(first)))
            bits = jax.lax.bitcast_convert_type(q.astype(jnp.float32), jnp.uint32)
            lane = jax.lax.broadcasted_iota(jnp.int32, bits.shape, 1)
            return jnp.where(lane < d, (bits >> 16) | pltpu.roll(bits, d, 1), 0)
        return layouts.pack_heads(
            [
                rope(
                    norm.norm(
                        config.q_norm,
                        read(config.q_col(first + i)),
                        weight(weights_ref.q_norm_w),
                        config.eps,
                    )
                )
                for i in range(config.q_packing)
            ]
        )

    # Step 3: Q heads -> q_ref.
    for h in range(hkv):
        for g in range(q_per_kv // config.q_packing):
            layouts.strided_store_heads(q_ref.at[h], g, packed_q(h, g))

    # Step 4: Gates.
    if gate_ref is not None:
        for i in range(hq):
            gate_ref[:, pl.ds(i * d, d)] = read(config.gate_col(i))


def prepare_weights(
    config: configs.AttentionConfig, weights: configs.QkvWeights
) -> configs.QkvWeights:
    """Returns `weights` in fused_qkv's layout. Call once, at weight load time.

    With fp8 and an input norm, input_norm_w is pre-chopped into quant blocks,
    with the bf16 weight in both halves of each u32 so one stride-0 load
    broadcasts it over a bf16 vreg: (K / block, 1, block) u32.
    """
    if not config.fp8 or config.input_norm == configs.Norm.NONE:
        return weights
    return weights._replace(
        input_norm_w=norm.packed_bf16_weight(
            config.input_norm, weights.input_norm_w, config.act_quant_block
        )
    )


@functools.partial(
    jax.jit,
    static_argnums=0,
    static_argnames=("tiling", "kv_layout", "page_size"),
)
def fused_qkv(
    config: configs.AttentionConfig,
    x: jax.Array,
    residual: jax.Array | None,
    weights: configs.QkvWeights,
    cos: jax.Array | None,
    sin: jax.Array | None,
    *,
    tiling: tiling_lib.Tiling | None = None,
    kv_layout: rpa_configs.KVLayout = rpa_configs.KVLayout.SEQ_ALONG_LANE,
    page_size: int = 128,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array | None,
    jax.Array | None,
]:
    """Fused input norm + QKV projection + q/k/v norms + RoPE + KV fp8 cast.

    Args:
      x: decoder hidden states, before the input norm.
      residual: required iff config.residual.
      cos: RoPE cos; required iff config.rope.
      sin: RoPE sin; required iff config.rope.
      tiling: defaults to choose_tiling's.
      kv_layout: batched_rpa KV layout to write.
      page_size: batched_rpa KV cache page size; SEQ_ALONG_LANE pads to it.

    Returns:
      (q, kv, gate, residual):
      q: Dp = align(D, 128).
      kv: config.kv_dtype, heads ordered [k_0, v_0, k_1, v_1, ...] (zero-padded
        to a multiple of kv_packing):
        SEQ_ALONG_LANE: [2Hkv, D/kv_packing, kv_packing,
          align(T, max(128, page_size))], tokens past T zero.
        HEAD_ALONG_SUBLANE: [T, align(2Hkv, kv_packing)/kv_packing, kv_packing,
          align(D, 128)].
      gate: None without config.gate.
      residual: x + residual, or None.
    """
    num_tokens, hidden_size = x.shape
    hq, hkv, d = config.num_q_heads, config.num_kv_heads, config.head_dim
    tpu_info = pltpu.get_tpu_info()
    num_lanes, num_sublanes = tpu_info.num_lanes, tpu_info.num_sublanes
    q_per_kv = hq // hkv
    assert config.q_dtype == jnp.bfloat16, "only bf16 Q is implemented"
    assert config.kv_dtype == quant.F8, "only fp8 KV is implemented"
    assert (residual is not None) == config.residual
    assert weights.side_w is None
    assert q_per_kv % config.q_packing == 0
    assert d % (num_sublanes * config.kv_packing) == 0
    assert config.num_cols % max(d, num_lanes) == 0
    assert not config.gate or d % num_lanes == 0
    if tiling is None:
        tiling = choose_tiling(config, num_tokens, weights, kv_layout=kv_layout)
    tile_size = tiling.tile_size
    assert num_tokens % tile_size == 0 and tile_size % tiling.prep_rows == 0
    num_tiles = num_tokens // tile_size
    # A lone tile's blocks are whole arrays, which Mosaic allows at any size.
    assert num_tiles == 1 or tile_size % num_lanes == 0

    def row_spec(width):
        return pl.BlockSpec((tile_size, width), lambda i: (i, 0))

    rope_spec = None if cos is None else row_spec(cos.shape[1])
    residual_spec = None if residual is None else row_spec(hidden_size)
    tiled_in = (x, residual, cos, sin)
    tiled_in_specs = (row_spec(hidden_size), residual_spec, rope_spec, rope_spec)

    q_shape = (
        hkv,
        num_tokens,
        q_per_kv // config.q_packing,
        config.q_packing,
        layouts.align(d, num_lanes),
    )
    kv_pad_starts = ()  # Lane offsets of the KV token pad's zero DMAs.
    match kv_layout:
        case rpa_configs.KVLayout.SEQ_ALONG_LANE:
            # batched_rpa reads new KV in page- and lane-aligned token spans, so the
            # token dim is zero-padded to both. Blocks are lane-aligned (only a lone
            # tile can be short), and the kernel zeroes their lanes past T. Whole
            # lane chunks past the last block get zero DMAs instead.
            block_lanes = layouts.align(tile_size, num_lanes)
            kv_tokens = layouts.align(num_tokens, max(num_lanes, page_size))
            kv_pad_starts = tuple(range(num_tiles * block_lanes, kv_tokens, num_lanes))
            kv_shape = (2 * hkv, d // config.kv_packing, config.kv_packing, kv_tokens)
            kv_block_spec = pl.BlockSpec(
                (*kv_shape[:3], block_lanes), lambda i: (0, 0, 0, i)
            )
        case rpa_configs.KVLayout.HEAD_ALONG_SUBLANE:
            num_kv_groups = (
                layouts.align(2 * hkv, config.kv_packing) // config.kv_packing
            )
            kv_shape = (
                num_tokens,
                num_kv_groups,
                config.kv_packing,
                layouts.align(d, num_lanes),
            )
            kv_block_spec = pl.BlockSpec(
                (tile_size, *kv_shape[1:]), lambda i: (i, 0, 0, 0)
            )
    gate_shape = (num_tokens, hq * d)
    out_shapes = (
        jax.ShapeDtypeStruct(q_shape, config.q_dtype),
        jax.ShapeDtypeStruct(kv_shape, config.kv_dtype),
        jax.ShapeDtypeStruct(gate_shape, jnp.bfloat16) if config.gate else None,
        None if residual is None else jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    out_specs = (
        pl.BlockSpec((hkv, tile_size, *q_shape[2:]), lambda i: (0, i, 0, 0, 0)),
        kv_block_spec,
        row_spec(gate_shape[1]) if config.gate else None,
        residual_spec,
    )
    scratch_shapes = [
        pltpu.VMEM((tile_size, hidden_size), weights.w.dtype),  # x_q
        pltpu.VMEM((tile_size, config.num_cols), jnp.bfloat16),  # qkv
    ]
    if config.fp8:
        num_qb = hidden_size // config.act_quant_block
        scratch_shapes += [
            pltpu.VMEM((num_qb, tile_size, num_lanes), jnp.float32),  # x_scale
            pltpu.VMEM((tile_size, num_lanes), jnp.float32),  # row_scale
        ]
    if kv_pad_starts:
        scratch_shapes += [
            pltpu.VMEM((*kv_shape[:3], num_lanes), config.kv_dtype),  # zeros
            pltpu.SemaphoreType.DMA,
        ]
    # W is streamed in Step 1's block order.
    col_block = tiling_lib.column_block(config)
    w_chunks = memory.stream_chunks(
        [b * col_block for b in _block_order(config)], col_block
    )
    scratch_shapes += memory.stream_scratch(weights.w, w_chunks)

    def start_kv_pad_zeroing(kv_hbm_ref, zeros_ref, sem):
        """Starts zero DMAs over the KV token pad past the pipeline's blocks."""
        for j in range(zeros_ref.shape[0]):
            slab = zeros_ref.at[j].bitcast(jnp.uint32)
            slab = slab.reshape(d // config.kv_packing, num_lanes)
            slab[...] = jnp.zeros(slab.shape, jnp.uint32)
        copies = [
            pltpu.make_async_copy(
                zeros_ref, kv_hbm_ref.at[:, :, :, pl.ds(start, num_lanes)], sem
            )
            for start in kv_pad_starts
        ]
        for copy in copies:
            copy.start()
        return copies

    def outer_kernel(
        x_ref, residual_ref, cos_ref, sin_ref, weights_ref, *outs_and_scratch
    ):
        outs, scratches = outs_and_scratch[:4], outs_and_scratch[4:]
        *scratches, w_vmem, w_sems = scratches
        (w_pending,) = memory.start_streams(
            [(weights_ref.w, w_vmem, w_sems, w_chunks)], num_tiles
        )
        weights_ref = weights_ref._replace(w=w_vmem)
        pad_copies = []
        if kv_pad_starts:
            *scratches, zeros_ref, sem = scratches
            pad_copies = start_kv_pad_zeroing(outs[1], zeros_ref, sem)
        if not config.fp8:
            scratches = (*scratches, None, None)  # No x_scale or row_scale.
        pipeline = pltpu.emit_pipeline(
            functools.partial(
                inner_kernel,
                config=config,
                kv_layout=kv_layout,
                tiling=tiling,
                w_pending=w_pending,
            ),
            grid=(num_tiles,),
            in_specs=tiled_in_specs,
            out_specs=out_specs,
        )
        pipeline(
            x_ref,
            residual_ref,
            cos_ref,
            sin_ref,
            *outs,
            scratches=(weights_ref, *scratches),
        )
        for copy in pad_copies:
            copy.wait()

    # Memory spaces: tiled inputs stream from HBM (VMEM for a lone tile), W is
    # HBM and streamed, the other weights are pinned to VMEM, outputs are HBM.
    weights = memory.pin_weights(weights, _STREAMED)
    return pl.pallas_call(
        outer_kernel,
        name="fused_qkv",
        in_specs=memory.in_specs(tiled_in, weights, num_tiles, _STREAMED),
        out_specs=memory.out_specs(out_shapes),
        out_shape=out_shapes,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=tiling.vmem_limit_bytes
            + memory.stream_vmem_bytes(weights, _STREAMED)
        ),
    )(*tiled_in, weights)


# VMEM budget.


def bytes_per_row(
    config: configs.AttentionConfig,
    kv_layout: rpa_configs.KVLayout,
    hidden_size: int,
) -> dict[str, int]:
    """{term: VMEM bytes per token row}; pipelined terms are double-buffered."""
    num_lanes = pltpu.get_tpu_info().num_lanes
    hq, hkv, d = config.num_q_heads, config.num_kv_heads, config.head_dim
    d_pad = layouts.align(d, num_lanes)
    bf16, kv_bytes = 2, jnp.dtype(config.kv_dtype).itemsize
    rope_bytes = 0
    if config.rope is not None:
        rope_bytes = 2 * layouts.align(config.rope.n_pairs, num_lanes) * bf16
    match kv_layout:
        case rpa_configs.KVLayout.SEQ_ALONG_LANE:
            kv = 2 * hkv * d * kv_bytes
        case rpa_configs.KVLayout.HEAD_ALONG_SUBLANE:
            kv = layouts.align(2 * hkv, config.kv_packing) * d_pad * kv_bytes
    return {
        **tiling_lib.input_bytes_per_row(config, hidden_size),
        "rope": 2 * rope_bytes,
        "q": 2 * hq * d_pad * bf16,
        "kv": 2 * kv,
        "gate": 2 * hq * d * bf16 if config.gate else 0,
        "qkv": config.num_cols * bf16,
    }


def choose_tiling(
    config: configs.AttentionConfig,
    num_tokens: int,
    weights: configs.QkvWeights,
    *,
    kv_layout: rpa_configs.KVLayout = rpa_configs.KVLayout.SEQ_ALONG_LANE,
    tile_size: int | None = None,
) -> tiling_lib.Tiling:
    """fused_qkv's Tiling; tile_size, if given, overrides the token tile."""
    return tiling_lib.choose_tiling(
        config,
        num_tokens,
        weights,
        bytes_per_row(config, kv_layout, weights.w.shape[0]),
        tile_size=tile_size,
    )
