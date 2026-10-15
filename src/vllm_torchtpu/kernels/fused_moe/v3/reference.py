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
"""The layer's arithmetic in plain jnp on one device: dense over every
expert, no Pallas, no tables, no transport, with the kernel's quantization
steps and the kernel's dots, fp8 rows against fp8 weights accumulated in
f32 on the matrix unit with the scales applied after, so the two agree to
the order of f32 additions. Small shapes only: every token is
contracted against every expert."""

import functools

import jax
import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.fused_moe.v3 import ffn
from vllm_torchtpu.kernels.fused_moe.v3.ffn import DOT_ROWS_BY_COLUMNS as _DOT
from vllm_torchtpu.kernels.fused_moe.v3.formats import (
    FP4,
    WEIGHT_BLOCK_DEFAULT,
    weight_form,
)
from vllm_torchtpu.kernels.fused_moe.v3.rowquant import FP8, quantize_rows


def _fp8_dot(rows, weight):
    """[m, k] fp8 against [k, n] fp8, accumulated in f32 as the kernel's
    matrix-unit dots are. Both operands are upcast to bf16 first, which is
    exact for fp8 values: an fp8 x fp8 dot_general run eagerly on the TPU
    (outside jit) takes a lossy path (4.2e-3 relative L2 against float64,
    studies/fp8_dot_probe.py) while the same op under jit, and the bf16
    dot in either mode, are exact to 1e-8."""
    return lax.dot_general(
        rows.astype(jnp.bfloat16),
        weight.astype(jnp.bfloat16),
        _DOT,
        preferred_element_type=jnp.float32,
    )


def scaled_dot(rows, weight, scales, *, block_scaled, block):
    """rows [m, k] fp8 against one expert's weight with its scales applied
    after the contraction: none (bf16 weights, the rows upcast to bf16),
    per output channel ([n] scales on an fp8 weight), or per contraction
    block ([k // block, n] scales on a four-bit
    weight, each block upcast to fp8 and contracted on its own, as the
    kernel does)."""
    if scales is None:
        return lax.dot_general(
            rows.astype(jnp.bfloat16), weight, _DOT, preferred_element_type=jnp.float32
        )
    if not block_scaled:
        return _fp8_dot(rows, weight) * scales[None, :]
    acc = None
    for b in range(scales.shape[0]):
        part = _fp8_dot(
            rows[:, b * block : (b + 1) * block],
            weight[b * block : (b + 1) * block].astype(FP8),
        )
        part = part * scales[b][None, :]
        acc = part if acc is None else acc + part
    assert acc is not None, "a contraction of zero blocks"
    return acc


def activation_dot(
    rows, row_scales, weight, scales, *, block_scaled, block, activation_block
):
    """scaled_dot with the activation's row scales applied: after the whole
    contraction (one scale per row), or per activation block of the
    contraction (row_scales [m, k // activation_block]), each block's
    product scaled by its own row scale before the sum, as the kernel's
    blocked contraction does."""
    if not activation_block:
        return (
            scaled_dot(rows, weight, scales, block_scaled=block_scaled, block=block)
            * row_scales
        )
    acc = None
    for b in range(row_scales.shape[1]):
        lo, hi = b * activation_block, (b + 1) * activation_block
        if scales is None or not block_scaled:
            block_scales = scales
        else:
            block_scales = scales[lo // block : hi // block]
        part = scaled_dot(
            rows[:, lo:hi],
            weight[lo:hi],
            block_scales,
            block_scaled=block_scaled,
            block=block,
        )
        part = part * row_scales[:, b : b + 1]
        acc = part if acc is None else acc + part
    assert acc is not None, "a contraction of zero blocks"
    return acc


FP4_MAX = 6.0


def fp4_block_rows(rows, block, scale_dtype=jnp.float32):
    """The result rows rounded through float4_e2m1fn with one scale per
    `block` values (the scale = the block's absolute maximum over 6,
    rounded to `scale_dtype`), dequantized back to float32: what a
    four-bit result row would carry to the token's home core."""
    m, width = rows.shape
    if width % block:
        raise ValueError(
            f"a row of {width} values is not a whole number of {block}-value blocks"
        )
    blocks = rows.astype(jnp.float32).reshape(m, width // block, block)
    amax = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True)
    scale = (amax / FP4_MAX).astype(scale_dtype).astype(jnp.float32)
    scalable = jnp.isfinite(scale) & (scale > 0)
    inverse = jnp.where(scalable, 1.0 / jnp.where(scalable, scale, 1.0), 0.0)
    q = jnp.where(scalable, blocks * inverse, 0.0).astype(FP4)
    return (q.astype(jnp.float32) * scale).reshape(m, width)


@functools.partial(
    jax.jit, static_argnames=("block", "exp_bits", "man_bits", "bias", "scale_dtype")
)
def float_grid_rows(rows, block, *, exp_bits, man_bits, bias, scale_dtype=jnp.float32):
    """The result rows rounded through a small float format with no
    infinities or NaNs (the OCP MX element formats: E2M1, E2M3, E3M2), one
    scale per `block` values (the block's absolute maximum over the
    format's largest value, rounded to `scale_dtype`), dequantized back to
    float32. Exponent code 0 is subnormal; the largest normal is
    (2 - 2**-man_bits) * 2**(2**exp_bits - 1 - bias); ties round to even.
    With exp_bits=2, man_bits=1, bias=1 this matches the native
    float4_e2m1fn cast bit for bit (checked on the CPU and the TPU)."""
    m, width = rows.shape
    if width % block:
        raise ValueError(
            f"a row of {width} values is not a whole number of {block}-value blocks"
        )
    min_normal = 2.0 ** (1 - bias)
    # No infinity or NaN codes: the all-ones exponent is a normal.
    largest = (2.0 - 2.0**-man_bits) * 2.0 ** (2**exp_bits - 1 - bias)
    blocks = rows.astype(jnp.float32).reshape(m, width // block, block)
    amax = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True)
    scale = (amax / largest).astype(scale_dtype).astype(jnp.float32)
    scalable = jnp.isfinite(scale) & (scale > 0)
    inverse = jnp.where(scalable, 1.0 / jnp.where(scalable, scale, 1.0), 0.0)
    v = jnp.where(scalable, blocks * inverse, 0.0)
    a = jnp.minimum(jnp.abs(v), largest)
    exponent = jnp.floor(jnp.log2(jnp.maximum(a, min_normal)))
    step = jnp.where(
        a >= min_normal, 2.0 ** (exponent - man_bits), min_normal * 2.0**-man_bits
    )
    q = jnp.minimum(jnp.round(a / step) * step, largest)
    return (jnp.sign(v) * q * scale).reshape(m, width)


FLOAT_GRIDS = {  # result_rows name -> (exponent bits, mantissa bits, bias)
    "fp4_grid": (2, 1, 1),  # e2m1 by the same arithmetic, a check on fp4_block_rows
}


def select(router_logits, *, top_k, renormalize):
    """The router: (weights [T, K] f32, indices [T, K] int32), ties to the
    lower index, a row with no finite logit routing nowhere."""
    logits = router_logits.astype(jnp.float32)
    top, indices = jax.lax.top_k(logits, top_k)
    if renormalize:
        weights = jax.nn.softmax(top, axis=-1)
    else:
        weights = jnp.take_along_axis(jax.nn.softmax(logits, axis=-1), indices, axis=-1)
    routes = jnp.any(jnp.isfinite(logits), axis=-1, keepdims=True)
    return jnp.where(routes, weights, 0.0), indices.astype(jnp.int32)


def fused_ep_moe_reference(
    x,
    w1,
    w2,
    w1_scales,
    w2_scales,
    router_logits,
    w1_bias=None,
    w2_bias=None,
    *,
    top_k,
    renormalize,
    weight_format,
    weight_block=None,
    activation="silu",
    swiglu_limit=ffn.SWIGLU_LIMIT_DEFAULT,
    routing=None,
    no_gate=False,
    activation_block=0,
    result_rows="fp8",
    elementwise_dtype="float32",
    token_block=None,
    intermediate="fp8",
    result_block=32,
    result_scale_dtype=jnp.float32,
    combine_dtype="float32",
    token_rows="fp8",
):
    """fused_ep_moe's output, [tokens, hidden] in x's dtype. token_block:
    the token rows' rounding block (None: activation_block).
    result_rows: "fp8" (one scale per row), "fp8_block" (fp8 with one
    scale per `result_block` values), "bf16", "f32" (no rounding: the
    yardstick for the result rows), "fp4_grid" (e2m1 by float_grid_rows, a check on
    "fp4"), or "fp4" (float4_e2m1fn with one
    scale per `result_block` values, the scale rounded to
    `result_scale_dtype` before use).
    token_rows: "fp8" (the token rows rounded to fp8 with one scale per
    row or per token block before the dispatch) or "bf16" (as they are).
    combine_dtype: "float32" (the stock layer's XLA combine, expert by
    expert in f32) or "bfloat16" (the kernel's bf16 sum: a token's rows
    in slot order, each row on the wire as bf16 times one bf16
    coefficient, the row's scale and the router weight multiplied in f32
    and rounded once, the terms summed in bf16; fp8 and bf16 result rows
    only).
    activation_block, result_rows and elementwise_dtype: the layer's
    Config fields; with bfloat16 the compiler's own rounding choices keep
    the kernel and this reference apart by an fp8 step on a few percent
    of the values."""
    elementwise = jnp.float32 if elementwise_dtype == "float32" else jnp.bfloat16
    form = weight_form(weight_format)
    block = WEIGHT_BLOCK_DEFAULT if weight_block is None else weight_block
    num_experts = w1.shape[0]
    inter = w1.shape[2] // (1 if no_gate else 2)
    token_block = activation_block if token_block is None else token_block
    if token_rows == "bf16":
        # The token rows as they are, no rounding: unit scales.
        rows = x.astype(jnp.bfloat16)
        row_scales = jnp.ones(
            (x.shape[0], x.shape[1] // token_block if token_block else 1), jnp.float32
        )
    else:
        rows, row_scales = quantize_rows(
            x.astype(jnp.bfloat16), block=token_block, dtype=elementwise
        )
    if routing is not None:
        indices, weights = routing
        indices = indices.astype(jnp.int32)
        routes = jnp.logical_and(indices >= 0, indices < num_experts)
        indices = jnp.where(routes, indices, 0)
        weights = jnp.where(routes, weights.astype(jnp.float32), 0.0)
    else:
        weights, indices = select(router_logits, top_k=top_k, renormalize=renormalize)
    out = jnp.zeros((x.shape[0], x.shape[1]), jnp.float32)
    slot_rows, slot_scales = [], None
    if combine_dtype != "float32":
        if combine_dtype != "bfloat16" or result_rows not in ("fp8", "bf16"):
            raise ValueError(
                f"combine_dtype {combine_dtype!r} with "
                f"{result_rows!r} result rows: the bf16 sum "
                "is defined for fp8 and bf16 rows"
            )
        # A token's rows as they cross the wire, slot by slot, and their
        # scales (fp8 rows), gathered over the expert loop.
        slot_rows = [jnp.zeros(out.shape, jnp.bfloat16)] * top_k
        slot_scales = (
            [jnp.zeros(out.shape[0], jnp.float32)] * top_k
            if result_rows == "fp8"
            else None
        )
    for e in range(num_experts):
        acc1 = activation_dot(
            rows,
            row_scales,
            w1[e],
            None if w1_scales is None else w1_scales[e],
            block_scaled=form.block_scaled,
            block=block,
            activation_block=token_block,
        )
        if w1_bias is not None:
            acc1 = acc1 + w1_bias[e].reshape(1, -1)
        if no_gate:
            mid = ffn.apply_activation_no_gate(acc1, activation)
        else:
            mid = ffn.apply_activation(
                acc1[:, :inter], acc1[:, inter:], activation, swiglu_limit=swiglu_limit
            )
        if intermediate == "fp8":
            mid, mid_scales = quantize_rows(
                mid.astype(jnp.bfloat16), block=activation_block, dtype=elementwise
            )
            acc2 = activation_dot(
                mid,
                mid_scales,
                w2[e],
                None if w2_scales is None else w2_scales[e],
                block_scaled=form.block_scaled,
                block=block,
                activation_block=activation_block,
            )
        else:
            # "bf16": the kernel's bf16 intermediate against w2 upcast to
            # bf16; "exact": float32 rows against float32 weights, the
            # yardstick both forms are measured against.
            mid_dtype = jnp.bfloat16 if intermediate == "bf16" else jnp.float32
            acc2 = jnp.dot(
                mid.astype(mid_dtype),
                w2[e].astype(mid_dtype),
                preferred_element_type=jnp.float32,
            )
            if w2_scales is not None:
                acc2 = acc2 * w2_scales[e].astype(jnp.float32)[None, :]
        if w2_bias is not None:
            acc2 = acc2 + w2_bias[e].reshape(1, -1)

        def round_rows(acc2):
            if result_rows == "bf16":
                rounded = acc2.astype(jnp.bfloat16).astype(jnp.float32)
            elif result_rows == "f32":
                rounded = acc2
            elif result_rows == "fp4":
                rounded = fp4_block_rows(
                    acc2.astype(jnp.bfloat16), result_block, result_scale_dtype
                )
            elif result_rows in FLOAT_GRIDS:
                exp_bits, man_bits, bias = FLOAT_GRIDS[result_rows]
                rounded = float_grid_rows(
                    acc2.astype(jnp.bfloat16),
                    result_block,
                    exp_bits=exp_bits,
                    man_bits=man_bits,
                    bias=bias,
                    scale_dtype=result_scale_dtype,
                )
            elif result_rows == "fp8_block":
                result, result_scales = quantize_rows(
                    acc2.astype(jnp.bfloat16), block=result_block, dtype=elementwise
                )
                blocks = result.reshape(-1, x.shape[1] // result_block, result_block)
                rounded = (
                    blocks.astype(jnp.float32) * result_scales[:, :, None]
                ).reshape(result.shape)
            else:
                result, result_scales = quantize_rows(
                    acc2.astype(jnp.bfloat16), dtype=elementwise
                )
                rounded = result.astype(jnp.float32) * result_scales
            return rounded

        weight_e = jnp.sum(jnp.where(indices == e, weights, 0.0), axis=1, keepdims=True)
        if combine_dtype != "float32":
            if result_rows == "fp8":
                wire, wire_scales = quantize_rows(
                    acc2.astype(jnp.bfloat16), dtype=elementwise
                )
                wire, wire_scales = (wire.astype(jnp.bfloat16), wire_scales.reshape(-1))
            else:
                wire, wire_scales = acc2.astype(jnp.bfloat16), None
            for k in range(top_k):
                hit = indices[:, k] == e
                slot_rows[k] = jnp.where(hit[:, None], wire, slot_rows[k])
                if wire_scales is not None:
                    assert slot_scales is not None  # both mean fp8 result rows
                    slot_scales[k] = jnp.where(hit, wire_scales, slot_scales[k])
        else:
            out = out + weight_e * round_rows(acc2)
    if combine_dtype != "float32":
        acc = None
        for k in range(top_k):
            coef = weights[:, k]
            if slot_scales is not None:
                coef = coef * slot_scales[k]
            term = slot_rows[k] * coef.astype(jnp.bfloat16)[:, None]
            acc = term if acc is None else acc + term
        assert acc is not None
        out = acc.astype(jnp.float32)
    return out.astype(x.dtype)
