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
"""GDN output gated RMSNorm and FP8 out-projection quantization in one pass.

The gated delta net kernel writes a bf16 ``[tokens, heads, head_dim]`` output.
vLLM's ``RMSNormGated`` then normalizes each head in float32, multiplies by
the norm weight, gates with ``silu(z)`` and rounds to bf16. The per-channel
FP8 out-projection (``xla_quantized_matmul``) then takes a per-token absolute
maximum over all columns and converts the row to FP8. That is several passes
over the tensor.

This kernel does the same arithmetic in one pass per token tile. It rounds
the normalized output to bf16 before the absolute maximum, as the two-step
path does, so the FP8 rows and scales it returns are the ones the
out-projection would compute itself.

Per head ``h`` of a token (``norm_before_gate=True``, silu gate)::

    y = bf16(x_h * rsqrt(mean(x_h ** 2) + eps) * w * silu(z_h))
    scale = max(|y|) / 448 over the token's row (1 where the row is all zero)
    q = float8_e4m3fn(clip(y / scale, -448, 448))
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8_DTYPE = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8_DTYPE).max)


def _gated_norm_quant_kernel(
    attn_out_ref, gate_ref, weight_ref, quantized_ref, scale_ref, *, eps: float
):
    attn_out = attn_out_ref[...].astype(jnp.float32)  # [tile, heads, head_dim]
    gate = gate_ref[...].astype(jnp.float32)
    weight = weight_ref[...].astype(jnp.float32)  # [head_dim]
    variance = jnp.mean(attn_out * attn_out, axis=-1, keepdims=True)
    normed = attn_out * jax.lax.rsqrt(variance + eps) * weight[None, None, :]
    normed = normed * (gate * jax.nn.sigmoid(gate))
    # The norm's bf16 output, which the unfused out-projection quantizes.
    normed = normed.astype(jnp.bfloat16).astype(jnp.float32)
    abs_max = jnp.max(jnp.abs(normed), axis=(1, 2), keepdims=True)  # [tile, 1, 1]
    scale = abs_max / FP8_MAX
    scale = jnp.where(scale == 0, 1.0, scale)
    quantized_ref[...] = jnp.clip(normed / scale, -FP8_MAX, FP8_MAX).astype(FP8_DTYPE)
    scale_ref[...] = scale.reshape(scale.shape[0], 1)


def legal_row_tile(rows: int, max_tile: int) -> int:
    """The largest legal Pallas row block at or below ``max_tile``.

    A legal block is a multiple of 8 that divides ``rows``, or ``rows`` itself
    when no such divisor exists (a block equal to the full dimension is always
    legal). For example 48 rows run in tiles of 16 with ``max_tile=32``, and
    12 rows run as one tile of 12.
    """
    for tile in range(min(max_tile, rows), 7, -1):
        if rows % tile == 0 and tile % 8 == 0:
            return tile
    return rows


def gated_norm_quant(
    attn_out: jax.Array,
    gate: jax.Array,
    weight: jax.Array,
    eps: float,
    *,
    max_tile: int = 32,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Gated RMSNorm plus per-token FP8 quantization.

    Args:
        attn_out: GDN output, bf16 ``[tokens, heads, head_dim]``.
        gate: The ``z`` gate, same shape as ``attn_out``.
        weight: Norm weight ``[head_dim]``.
        eps: Norm epsilon.
        max_tile: Upper bound on the token rows per grid step.
        interpret: Run the Pallas kernel in interpret mode (CPU tests).

    Returns:
        FP8 ``[tokens, heads, head_dim]`` rows and float32 ``[tokens, 1]``
        per-token scales.
    """
    tokens, heads, head_dim = attn_out.shape
    if gate.shape != attn_out.shape or weight.shape != (head_dim,):
        raise ValueError(
            f"gated_norm_quant shapes do not match: {attn_out.shape=}, "
            f"{gate.shape=}, {weight.shape=}"
        )
    tile = legal_row_tile(tokens, max_tile)
    row_spec = pl.BlockSpec((tile, heads, head_dim), lambda i: (i, 0, 0))
    return pl.pallas_call(
        functools.partial(_gated_norm_quant_kernel, eps=float(eps)),
        grid=(tokens // tile,),
        in_specs=[row_spec, row_spec, pl.BlockSpec((head_dim,), lambda i: (0,))],
        out_specs=[row_spec, pl.BlockSpec((tile, 1), lambda i: (i, 0))],
        out_shape=[
            jax.ShapeDtypeStruct((tokens, heads, head_dim), FP8_DTYPE),
            jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(attn_out, gate, weight)


def fp8_matmul_on_quantized_rows(
    x_q: jax.Array,
    x_scale: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    out_dtype=jnp.bfloat16,
) -> jax.Array:
    """Per-channel FP8 matmul on already-quantized activation rows.

    ``w_q`` is the canonical ``[n_in, n_out]`` layout and ``w_scale`` is
    ``[n_out]``. Same arithmetic as ``xla_quantized_matmul`` after its own
    activation quantization.
    """
    out = jax.lax.dot_general(
        x_q,
        w_q,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    out = out * x_scale
    out = out * jnp.expand_dims(w_scale, 0)
    return out.astype(out_dtype)
