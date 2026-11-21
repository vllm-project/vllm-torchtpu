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
"""The per-expert FFN arithmetic of Fused EP MoE v3: the up projection, the
activation, the intermediate's fp8 requantization, the down projection and the
result rows, written so the same lines run inside the kernel body."""

import jax
import jax.numpy as jnp
from jax import lax

from .config import ACTIVATION_COLUMNS_PER_PASS_DEFAULT
from .rowquant import FP8, apply_row_scale, quantize_rows, row_scale

# The dot dimension numbers of every matmul in the package: the rows'
# last axis against the weight's first, no batch axes.
DOT_ROWS_BY_COLUMNS = (((1,), (0,)), ((), ()))
ACTIVATIONS = ("silu",)
NO_GATE_ACTIVATIONS = ("silu",)


def apply_activation(gate, up, activation):
    """The gated activation on the post-scale gate and up halves."""
    if activation == "silu":
        return jax.nn.silu(gate) * up
    raise ValueError(f"activation {activation!r} is not one of {ACTIVATIONS}")


def apply_activation_no_gate(x, activation):
    """The activation of the no-gate form: one projection, no product."""
    if activation == "silu":
        return jax.nn.silu(x)
    raise ValueError(f"activation {activation!r} is not one of {NO_GATE_ACTIVATIONS}")


def blocked_rows_dot(rows, weight, row_scales, block, *, upcast=None):
    """The contraction of fp8 rows against a weight in `block`-wide blocks
    of the contraction axis, each block's product scaled by that block's
    row scale: sum over b of (rows[:, b] @ weight[b]) * row_scales[:, b].
    row_scales is [rows, k // block] f32. `upcast` names a dtype the rows
    are cast to first (bf16 weights)."""
    acc = None
    for b in range(row_scales.shape[-1]):
        lhs = rows[:, b * block : (b + 1) * block]
        if upcast is not None:
            lhs = lhs.astype(upcast)
        part = lax.dot_general(
            lhs,
            weight[b * block : (b + 1) * block],
            DOT_ROWS_BY_COLUMNS,
            preferred_element_type=jnp.float32,
        )
        part = part * row_scales[:, b : b + 1]
        acc = part if acc is None else acc + part
    return acc


def intermediate_rows(
    acc1,
    row_scales,
    inter,
    *,
    w1_scales,
    activation,
    w1_bias,
    no_gate,
    block=0,
    dtype=jnp.float32,
    columns_per_pass=ACTIVATION_COLUMNS_PER_PASS_DEFAULT,
):
    """The first matmul's accumulator through the activation, requantized
    to fp8 as the second matmul's left-hand side.

    Args:
      acc1: [rows, 2 * inter] f32 (or [rows, inter] without a gate).
      row_scales: [rows, 1] f32, the activation row scales not yet
        applied to the accumulator, or None where a blocked contraction
        applied them already.
      inter: the intermediate width.
      w1_scales: [1, columns] f32 per-channel weight scales, or None where
        the block scales are already inside acc1.
      activation: the activation's name.
      w1_bias: [1, columns] f32 or None. Added to the post-scale halves
        before the activation.
      no_gate: whether acc1 is a single projection.
      block: 0 for one scale per row over the whole width; otherwise one
        scale per `block` columns of the intermediate.
      dtype: the element type the fp8 scaling product is formed in
        (Config.elementwise_dtype); the scales, bias and activation stay in
        float32 on the matmul's accumulator either way.

    Returns:
      (fp8 intermediate [rows, inter], f32 scales [rows, 1], or
      [rows, inter // block] with a block).
    """

    def half(lo, hi):
        scaled = acc1[:, lo:hi]
        if row_scales is not None:
            scaled = scaled * row_scales
        if w1_scales is not None:
            scaled = scaled * w1_scales[:, lo:hi]
        if w1_bias is not None:
            scaled = scaled + w1_bias[:, lo:hi]
        return scaled

    def activated(lo, hi):
        if no_gate:
            chunk = apply_activation_no_gate(half(lo, hi), activation)
        else:
            chunk = apply_activation(
                half(lo, hi), half(inter + lo, inter + hi), activation
            )
        return chunk.astype(jnp.bfloat16)

    if block:
        chunks, scales = [], []
        for lo in range(0, inter, block):
            chunk = activated(lo, lo + block)
            amax = jnp.max(jnp.abs(chunk), axis=-1, keepdims=True)
            scale, inverse, scalable = row_scale(amax)
            chunks.append(apply_row_scale(chunk, inverse, scalable, dtype))
            scales.append(scale)
        return (jnp.concatenate(chunks, axis=-1), jnp.concatenate(scales, axis=-1))

    chunks, amax = [], None
    for lo in range(0, inter, columns_per_pass):
        hi = min(lo + columns_per_pass, inter)
        chunk = activated(lo, hi)
        chunks.append(chunk)
        slice_amax = jnp.max(jnp.abs(chunk), axis=-1, keepdims=True)
        amax = slice_amax if amax is None else jnp.maximum(amax, slice_amax)
    scale, inverse, scalable = row_scale(amax)
    quantized = jnp.concatenate(
        [apply_row_scale(chunk, inverse, scalable, dtype) for chunk in chunks], axis=-1
    )
    return quantized, scale


def result_rows(
    acc2, mid_scales, *, w2_scales, w2_bias, dtype=FP8, elementwise=jnp.float32
):
    """The second matmul's accumulator as result rows: fp8 rows and their
    scales, or (dtype bf16) bf16 rows and None. `elementwise` is the
    element type the fp8 scaling product is formed in.

    The intermediate row scale (None where a blocked contraction applied
    it), the per-channel weight scale (None where block scales are already
    inside acc2) and the down bias (None where the model has none) are
    applied, then the row is quantized. Under expert
    parallelism a routed row's whole down projection is computed on one
    shard, so the bias enters it exactly once. The destination weights it
    with the router weight, per selected expert.
    """
    down = acc2 if mid_scales is None else acc2 * mid_scales
    if w2_scales is not None:
        down = down * w2_scales
    if w2_bias is not None:
        down = down + w2_bias
    if jnp.dtype(dtype) == jnp.dtype(jnp.bfloat16):
        return down.astype(jnp.bfloat16), None
    return quantize_rows(down.astype(jnp.bfloat16), dtype=elementwise)


def expert_ffn_fp8(
    rows,
    row_scales,
    w1,
    w2,
    w1_scales,
    *,
    activation,
    w1_bias,
    no_gate,
    block=0,
    elementwise=jnp.float32,
    columns_per_pass=ACTIVATION_COLUMNS_PER_PASS_DEFAULT,
    token_block=None,
):
    """One expert's FFN on fp8 weights with per-channel scales. `block` is
    the intermediate's rounding block (0: one scale per row), `token_block`
    the token rows' (None: the same as `block`). Returns the down matmul's
    accumulator and the intermediate row scales not yet applied to it (None
    with a block: the down contraction applied them block by block)."""
    inter = w1.shape[-1] // (1 if no_gate else 2)
    token_block = block if token_block is None else token_block
    if token_block:
        acc1 = blocked_rows_dot(rows, w1, row_scales, token_block)
        row_scales = None  # applied block by block above
    else:
        acc1 = lax.dot_general(
            rows, w1, DOT_ROWS_BY_COLUMNS, preferred_element_type=jnp.float32
        )
    mid, mid_scales = intermediate_rows(
        acc1,
        row_scales,
        inter,
        w1_scales=w1_scales,
        activation=activation,
        w1_bias=w1_bias,
        no_gate=no_gate,
        block=block,
        dtype=elementwise,
        columns_per_pass=columns_per_pass,
    )
    if block:
        return blocked_rows_dot(mid, w2, mid_scales, block), None
    acc2 = lax.dot_general(
        mid, w2, DOT_ROWS_BY_COLUMNS, preferred_element_type=jnp.float32
    )
    return acc2, mid_scales


def expert_ffn_bf16(
    rows,
    row_scales,
    w1,
    w2,
    *,
    activation,
    w1_bias,
    no_gate,
    block=0,
    elementwise=jnp.float32,
    columns_per_pass=ACTIVATION_COLUMNS_PER_PASS_DEFAULT,
    token_block=None,
):
    """One expert's FFN on unscaled bf16 weights. The fp8 token rows and
    the fp8 intermediate are upcast to bf16 for the matrix unit, which is
    exact, so the row scales are the only scales in the body. `block` and
    `token_block` as in expert_ffn_fp8. Returns the down matmul's
    accumulator and the intermediate row scales not yet applied to it
    (None with a block)."""
    inter = w1.shape[-1] // (1 if no_gate else 2)
    token_block = block if token_block is None else token_block
    if token_block:
        acc1 = blocked_rows_dot(rows, w1, row_scales, token_block, upcast=jnp.bfloat16)
        row_scales = None
    else:
        acc1 = lax.dot_general(
            rows.astype(jnp.bfloat16),
            w1,
            DOT_ROWS_BY_COLUMNS,
            preferred_element_type=jnp.float32,
        )
    mid, mid_scales = intermediate_rows(
        acc1,
        row_scales,
        inter,
        w1_scales=None,
        activation=activation,
        w1_bias=w1_bias,
        no_gate=no_gate,
        block=block,
        dtype=elementwise,
        columns_per_pass=columns_per_pass,
    )
    if block:
        return blocked_rows_dot(mid, w2, mid_scales, block, upcast=jnp.bfloat16), None
    acc2 = lax.dot_general(
        mid.astype(jnp.bfloat16),
        w2,
        DOT_ROWS_BY_COLUMNS,
        preferred_element_type=jnp.float32,
    )
    return acc2, mid_scales
