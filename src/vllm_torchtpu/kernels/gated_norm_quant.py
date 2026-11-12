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
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8_DTYPE = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8_DTYPE).max)
FP8_ROW_TILE_ALIGN = 32
TPU_LANE_ALIGN = 128


def _gated_norm_quant_kernel(
    attn_out_ref, gate_ref, weight_ref, quantized_ref, scale_ref, *, eps: float
):
    attn_out = attn_out_ref[...].astype(jnp.float32)  # [tile, heads, head_dim]
    gate = gate_ref[...].astype(jnp.float32)
    weight = weight_ref[...].astype(jnp.float32)  # [head_dim]
    variance = jnp.mean(attn_out * attn_out, axis=-1, keepdims=True)
    normed = attn_out * jax.lax.rsqrt(variance + eps) * weight[None, None, :]
    normed = normed * (gate * jax.nn.sigmoid(gate))
    # Reshape to 2D in float32 before casting to FP8 so single-head TP shards
    # (`heads == 1`) pack directly into 2D FP8 VREGs without an 8-bit relayout.
    tile, heads, head_dim = normed.shape
    normed_2d = (
        normed.astype(jnp.bfloat16).astype(jnp.float32).reshape(tile, heads * head_dim)
    )
    abs_max = jnp.max(jnp.abs(normed_2d), axis=1, keepdims=True)  # [tile, 1]
    scale = jnp.where(abs_max == 0, 1.0, abs_max / FP8_MAX)
    quantized_ref[...] = jnp.clip(normed_2d / scale, -FP8_MAX, FP8_MAX).astype(
        FP8_DTYPE
    )
    scale_ref[...] = scale


def effective_max_tile(max_tile: int) -> int:
    """Normalize ``max_tile`` to at least ``FP8_ROW_TILE_ALIGN`` (32)."""
    if max_tile <= 0:
        raise ValueError(f"max_tile must be positive, got {max_tile=}")
    return max(max_tile, FP8_ROW_TILE_ALIGN)


def legal_row_tile(rows: int, max_tile: int) -> int:
    """The 2D FP8 Pallas row block for ``rows`` within ``effective_max_tile``.

    On TPU, a 2D ``float8_e4m3fn`` array tiles axis 0 to 32 sublanes, so any
    multi-step block (``tile < rows``) must be a multiple of 32; ``max_tile``
    values below 32 are raised to 32. When ``rows <= cap`` a single block of
    ``rows`` covers the full dimension; otherwise returns the largest multiple
    of 32 at or below ``cap`` dividing ``rows``, or ``(cap // 32) * 32`` when
    the last input tile is window-padded to ``tile``.
    """
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows=}")
    cap = effective_max_tile(max_tile)
    if rows <= cap:
        return rows
    for tile in range(cap, FP8_ROW_TILE_ALIGN - 1, -1):
        if tile % FP8_ROW_TILE_ALIGN == 0 and rows % tile == 0:
            return tile
    return (cap // FP8_ROW_TILE_ALIGN) * FP8_ROW_TILE_ALIGN


def gated_norm_quant(
    attn_out: jax.Array,
    gate: jax.Array,
    weight: jax.Array,
    eps: float,
    *,
    max_tile: int = 32,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Gated RMSNorm plus per-token FP8 quantization into 2D activation rows.

    Emitting 2D ``[tokens, heads * head_dim]`` directly gives the FP8 output
    projection canonical ``T(32, 128)(4, 1)`` activation tiling instead of a 3D
    ``[tokens, heads, head_dim]`` layout that XLA bitcasts into a 4D transposed
    contraction.

    Args:
        attn_out: GDN output, bf16 ``[tokens, heads, head_dim]``.
        gate: The ``z`` gate, same shape as ``attn_out``.
        weight: Norm weight ``[head_dim]``.
        eps: Norm epsilon.
        max_tile: Upper bound on the token rows per grid step (at least 32 when
            tiling across multiple steps).
        interpret: Run the Pallas kernel in interpret mode (CPU tests).

    Returns:
        FP8 ``[tokens, heads * head_dim]`` rows and float32 ``[tokens, 1]``
        per-token scales.
    """
    if attn_out.ndim != 3:
        raise ValueError(
            f"attn_out must be 3D [tokens, heads, head_dim], got {attn_out.shape=}"
        )
    tokens, heads, head_dim = attn_out.shape
    if tokens <= 0 or heads <= 0 or head_dim <= 0:
        raise ValueError(
            f"gated_norm_quant dimensions must be positive, got {attn_out.shape=}"
        )
    if not interpret and head_dim % TPU_LANE_ALIGN != 0:
        raise ValueError(
            f"head_dim must be a multiple of {TPU_LANE_ALIGN} on TPU, got {head_dim=}"
        )
    if gate.shape != attn_out.shape or weight.shape != (head_dim,):
        raise ValueError(
            f"gated_norm_quant shapes do not match: {attn_out.shape=}, "
            f"{gate.shape=}, {weight.shape=}"
        )
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError(f"eps must be finite and positive, got {eps=}")

    tile = legal_row_tile(tokens, max_tile)
    pad = (-tokens) % tile
    padded_tokens = tokens + pad
    hidden_size = heads * head_dim
    if pad:
        in_spec = pl.BlockSpec(
            (pl.Element(tile, (0, pad)), pl.Element(heads), pl.Element(head_dim)),
            lambda i: (i * tile, 0, 0),
        )
    else:
        in_spec = pl.BlockSpec((tile, heads, head_dim), lambda i: (i, 0, 0))

    quantized, scale = pl.pallas_call(
        functools.partial(_gated_norm_quant_kernel, eps=float(eps)),
        grid=(padded_tokens // tile,),
        in_specs=[in_spec, in_spec, pl.BlockSpec((head_dim,), lambda i: (0,))],
        out_specs=[
            pl.BlockSpec((tile, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((tile, 1), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, hidden_size), FP8_DTYPE),
            jax.ShapeDtypeStruct((padded_tokens, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(attn_out, gate, weight)
    if pad:
        return quantized[:tokens], scale[:tokens]
    return quantized, scale


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
