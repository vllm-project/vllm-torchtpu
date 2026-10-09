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

"""Fused GDN (Gated DeltaNet) prologue for Qwen3-Next and Qwen3.5.

One Pallas kernel takes the decoder hidden states and applies the residual
add, GemmaRMSNorm, sub-channel fp8 quant, the fp8 in_proj_qkvz matmul and the
bf16 in_proj_ba matmul. It writes straight into the layouts the GDN kernel
(fused_conv1d_gdn) reads:
  mixed_qkv: [T, 1, qkv_cols] f32, compact (1, 128) tiling.
  z: [T, Hv, D] bf16, head-major.
  b, a: [T, 1, 128] f32, lanes >= Hv zero.

Each token tile (inner_kernel), one group of matmul_rows rows at a time:
  Step 0: residual add; x * norm weight (bf16, or f32 rounded to bf16 once:
    config.norm_dtype) into `normed`; sub-channel fp8 quant of it into
    x_q. 1/rms goes to row_scale and is applied after the matmuls (norm.py,
    FlashNorm).
  Step 1: in_proj_qkvz of those rows, one MXU-wide column block at a time;
    each f32 block goes straight to mixed_qkv or z with strided stores.
Then, over the whole tile:
  Step 2: in_proj_ba of the normed rows, two row groups per dot.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.fused_qkv.kernel import configs, memory
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel import tiling as tiling_lib
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel.library import (
    layouts,
    norm,
    quant,
)

# Weights the kernel copies HBM -> VMEM itself (memory.start_streams), in the
# tile's read order.
_STREAMED = ("w", "ba_w")


class GdnWeights(NamedTuple):
    w: jax.Array  # in_proj_qkvz (K, N) fp8, [q | k | v | z] as loaded.
    w_scale: jax.Array  # (1, N) f32.
    # in_proj_ba (K, 128) bf16: b in lanes 0:Hv and a in lanes 64:64 + Hv,
    # the rest exact zeros, so the ba dot is a single MXU column block.
    ba_w: jax.Array
    # Input norm weight, pre-chopped into quant blocks and packed into u32.
    # (K / block, 1, block) u32, or f32 if norm_dtype is f32.
    norm_w: jax.Array | None


class _Scratch(NamedTuple):
    x_q: Any  # (tile, K) fp8.
    x_scale: Any  # (K / block, tile, lanes) f32.
    row_scale: Any  # (tile, lanes) f32 1/rms.
    normed: Any  # (tile, K) bf16 x * norm weight.


def prepare_weights(
    config: configs.GdnConfig, weights: configs.QkvWeights
) -> GdnWeights:
    """Re-lays out the loaded weights for fused_gdn. Call once, at load time.

    Args:
      weights: w is in_proj_qkvz and side_w is in_proj_ba.
    """
    num_lanes = pltpu.get_tpu_info().num_lanes
    half, nv = num_lanes // 2, config.num_v_heads
    side_w = weights.side_w
    ba_w = jnp.zeros((side_w.shape[0], num_lanes), side_w.dtype)
    ba_w = ba_w.at[:, :nv].set(side_w[:, :nv])
    ba_w = ba_w.at[:, half : half + nv].set(side_w[:, nv:])
    norm_w = None
    if config.input_norm != configs.Norm.NONE:
        layout = {
            jnp.dtype(jnp.bfloat16): norm.packed_bf16_weight,
            jnp.dtype(jnp.float32): norm.f32_weight,
        }
        norm_w = layout[jnp.dtype(config.norm_dtype)](
            config.input_norm, weights.input_norm_w, config.act_quant_block
        )
    return GdnWeights(weights.w, weights.w_scale, ba_w, norm_w)


def inner_kernel(
    # Inputs.
    x_ref: jax.Ref,  # (tile, K) bf16 hidden states
    residual_ref: jax.Ref | None,  # (tile, K) bf16
    # Outputs.
    mixed_ref: jax.Ref,  # (tile, 1, qkv_cols) f32
    z_ref: jax.Ref,  # (tile, Hv, D) bf16
    b_ref: jax.Ref,  # (tile, 1, lanes) f32
    a_ref: jax.Ref,  # (tile, 1, lanes) f32
    residual_out_ref: jax.Ref | None,  # (tile, K) bf16
    # Scratch.
    weights_ref: GdnWeights,  # Refs; w and ba_w are the streamed VMEM copies.
    *scratch: jax.Ref,  # _Scratch
    config: configs.GdnConfig,
    tiling: tiling_lib.Tiling,
    pending: list[dict[int, Any]],
):
    """One token tile: Steps 0-2 of the module docstring.

    pending: per streamed weight (w, ba_w), its chunk copies not yet waited on.
    """
    s = _Scratch(*scratch)
    tile_size, hidden_size = x_ref.shape
    qb = config.act_quant_block
    mm_rows, prep_rows = tiling.matmul_rows, tiling.prep_rows
    col_block = tiling_lib.column_block(config)
    pair_cols = 2 * config.head_dim
    has_input_norm = config.input_norm != configs.Norm.NONE
    # No block straddles mixed_qkv and z, and each z block is one head pair.
    assert config.z_col % col_block == 0 and col_block == pair_cols
    assert tile_size % mm_rows == 0 and mm_rows % prep_rows == 0

    def add_norm_quant(rows):
        """Step 0 on `rows`.

        Math is bf16 (two values per lane) except the sum of squares, which is
        f32, and x * w, which is f32 if norm_dtype is.
        """
        sum_sq = jnp.zeros((prep_rows, 1), jnp.float32)
        for kb in range(hidden_size // qb):
            ks = pl.ds(kb * qb, qb)
            x = x_ref[rows, ks]
            if residual_ref is not None:
                x = x + residual_ref[rows, ks]
                residual_out_ref[rows, ks] = x
            if has_input_norm:
                h = x.astype(jnp.float32)
                sum_sq = sum_sq + norm.sum_sq(h)
                # Stride-0 load: one vld per vreg replicates the weight on all rows.
                if jnp.dtype(config.norm_dtype) == jnp.float32:
                    w_kb = norm.load_weight(
                        weights_ref.norm_w, kb, prep_rows, jnp.float32
                    )
                    x = (h * w_kb).astype(x.dtype)
                else:
                    x = x * norm.load_weight(weights_ref.norm_w, kb, prep_rows, x.dtype)
            s.normed[rows, ks] = x
            x_q, x_scale = quant.quantize_block(x)
            s.x_q[rows, ks] = x_q
            s.x_scale[kb, rows] = layouts.broadcast_lanes(x_scale)
        if has_input_norm:
            s.row_scale[rows] = layouts.broadcast_lanes(
                norm.inv_rms(sum_sq, hidden_size, config.eps)
            )

    for m in range(0, tile_size, mm_rows):
        rows = pl.ds(m, mm_rows)
        # Step 0: add, norm and quant of these rows, prep_rows at a time.
        for start in range(m, m + mm_rows, prep_rows):
            add_norm_quant(pl.ds(start, prep_rows))

        # Step 1: in_proj_qkvz of these rows, one column block at a time.
        for col in range(0, config.num_cols, col_block):
            cols = pl.ds(col, col_block)
            memory.wait_chunk(pending[0], col)
            # Each quant block has its own per-row scale, so the partial dots are
            # scaled before they are summed.
            acc = quant.scaled_dot(s.x_q, s.x_scale, weights_ref.w, rows, cols)
            if has_input_norm:  # Deferred 1/rms from Step 0.
                acc = norm.apply_row_scale(acc, s.row_scale[rows])
            acc = acc * weights_ref.w_scale[:, cols]
            if col < config.z_col:
                # mixed_qkv is compact: each token's lane chunks are consecutive rows.
                layouts.strided_store(mixed_ref, m, 1, acc, col=col)
            else:
                pair = (col - config.z_col) // pair_cols
                layouts.store_head_pair(z_ref, rows, pair, acc)

    # Step 2: in_proj_ba of the bf16 normed rows, after all row groups, two
    # row groups per dot. That pushes ba_w to the MXU half as often and keeps
    # it off each group's matmul tail (8% faster at T=1024).
    half = pltpu.get_tpu_info().num_lanes // 2
    ba_rows = 2 * mm_rows if tile_size % (2 * mm_rows) == 0 else mm_rows
    for m in range(0, tile_size, ba_rows):
        rows = pl.ds(m, ba_rows)
        memory.wait_chunk(pending[1], 0)
        # (ba_rows, lanes)
        ba = jnp.dot(
            s.normed[rows],
            weights_ref.ba_w[...],
            preferred_element_type=jnp.float32,
        )
        if has_input_norm:
            ba = norm.apply_row_scale(ba, s.row_scale[rows])
        # b is lanes 0:64 and a lanes 64:128; each is stored at lane 0 with the
        # other half zeroed.
        low = jax.lax.broadcasted_iota(jnp.int32, ba.shape, 1) < half
        layouts.strided_store(b_ref, m, 1, jnp.where(low, ba, 0.0))
        a = pltpu.roll(ba, shift=half, axis=1)
        layouts.strided_store(a_ref, m, 1, jnp.where(low, a, 0.0))


@functools.partial(jax.jit, static_argnums=0, static_argnames=("tiling",))
def fused_gdn(
    config: configs.GdnConfig,
    x: jax.Array,
    residual: jax.Array | None,
    weights: GdnWeights,
    *,
    tiling: tiling_lib.Tiling | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
]:
    """Fused GDN prologue: input norm + in_proj_qkvz (fp8) + in_proj_ba (bf16).

    Args:
      x: decoder hidden states, before the input norm. T % 128 == 0 (pad decode
        batches).
      residual: required iff config.residual.
      tiling: defaults to choose_tiling's.

    Returns:
      (mixed_qkv, z, b, a, residual): mixed_qkv is [q | k | v] per token; b and
      a have lanes >= Hv zero; residual is x + residual, or None.
    """
    num_tokens, hidden_size = x.shape
    num_lanes = pltpu.get_tpu_info().num_lanes
    assert config.fp8, "fused_gdn is fp8-only"
    assert (residual is not None) == config.residual
    assert num_tokens % num_lanes == 0, "pad T to a multiple of 128"
    assert config.head_dim == num_lanes
    assert config.num_v_heads % 2 == 0 and config.num_v_heads <= num_lanes // 2
    assert config.qkv_cols % num_lanes == 0
    assert hidden_size % config.act_quant_block == 0
    if tiling is None:
        tiling = choose_tiling(config, num_tokens, weights)
    tile_size = tiling.tile_size
    assert num_tokens % tile_size == 0 and tile_size % tiling.prep_rows == 0
    num_tiles = num_tokens // tile_size
    # A lone tile's blocks are whole arrays, which Mosaic allows at any size.
    assert num_tiles == 1 or tile_size % num_lanes == 0

    def row_spec(width):
        return pl.BlockSpec((tile_size, width), lambda i: (i, 0))

    residual_spec = None if residual is None else row_spec(hidden_size)
    out_shapes = (
        jax.ShapeDtypeStruct((num_tokens, 1, config.qkv_cols), jnp.float32),
        jax.ShapeDtypeStruct(
            (num_tokens, config.num_v_heads, config.head_dim), jnp.bfloat16
        ),
        jax.ShapeDtypeStruct((num_tokens, 1, num_lanes), jnp.float32),
        jax.ShapeDtypeStruct((num_tokens, 1, num_lanes), jnp.float32),
        None if residual is None else jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    out_specs = (
        *(
            pl.BlockSpec((tile_size, *s.shape[1:]), lambda i: (i, 0, 0))
            for s in out_shapes[:4]
        ),
        residual_spec,
    )
    num_qb = hidden_size // config.act_quant_block
    scratch_shapes = _Scratch(
        x_q=pltpu.VMEM((tile_size, hidden_size), weights.w.dtype),
        x_scale=pltpu.VMEM((num_qb, tile_size, num_lanes), jnp.float32),
        row_scale=pltpu.VMEM((tile_size, num_lanes), jnp.float32),
        normed=pltpu.VMEM((tile_size, hidden_size), x.dtype),
    )
    col_block = tiling_lib.column_block(config)
    w_chunks = memory.stream_chunks(range(0, config.num_cols, col_block), col_block)
    ba_chunks = ((0, weights.ba_w.shape[1]),)
    stream_scratch_shapes = [
        *memory.stream_scratch(weights.w, w_chunks),
        *memory.stream_scratch(weights.ba_w, ba_chunks),
    ]

    def outer_kernel(x_ref, residual_ref, weights_ref, *refs):
        outs, refs = refs[: len(out_shapes)], refs[len(out_shapes) :]
        *scratch, w_vmem, w_sems, ba_vmem, ba_sems = refs
        pending = memory.start_streams(
            [
                (weights_ref.w, w_vmem, w_sems, w_chunks),
                (weights_ref.ba_w, ba_vmem, ba_sems, ba_chunks),
            ],
            num_tiles,
        )
        weights_ref = weights_ref._replace(w=w_vmem, ba_w=ba_vmem)
        pipeline = pltpu.emit_pipeline(
            functools.partial(
                inner_kernel, config=config, tiling=tiling, pending=pending
            ),
            grid=(num_tiles,),
            in_specs=(row_spec(hidden_size), residual_spec),
            out_specs=out_specs,
        )
        pipeline(x_ref, residual_ref, *outs, scratches=(weights_ref, *scratch))

    weights = memory.pin_weights(weights, _STREAMED)
    return pl.pallas_call(
        outer_kernel,
        name="fused_qkv_gdn",
        in_specs=memory.in_specs((x, residual), weights, num_tiles, _STREAMED),
        out_specs=memory.out_specs(out_shapes),
        out_shape=out_shapes,
        scratch_shapes=[*scratch_shapes, *stream_scratch_shapes],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=tiling.vmem_limit_bytes
            + memory.stream_vmem_bytes(weights, _STREAMED)
        ),
    )(x, residual, weights)


# VMEM budget.


def bytes_per_row(config: configs.GdnConfig, hidden_size: int) -> dict[str, int]:
    """{term: VMEM bytes per token row}; the outputs are double-buffered."""
    num_lanes = pltpu.get_tpu_info().num_lanes
    return {
        **tiling_lib.input_bytes_per_row(config, hidden_size),
        "normed": layouts.align(hidden_size, num_lanes) * 2,  # bf16 scratch.
        "mixed_qkv": 2 * config.qkv_cols * 4,
        "z": 2 * config.value_dim * 2,
        "b": 2 * num_lanes * 4,
        "a": 2 * num_lanes * 4,
    }


def choose_tiling(
    config: configs.GdnConfig,
    num_tokens: int,
    weights: GdnWeights,
    *,
    tile_size: int | None = None,
) -> tiling_lib.Tiling:
    """fused_gdn's Tiling; tile_size, if given, overrides the token tile."""
    return tiling_lib.choose_tiling(
        config,
        num_tokens,
        weights,
        bytes_per_row(config, weights.w.shape[0]),
        tile_size=tile_size,
    )
