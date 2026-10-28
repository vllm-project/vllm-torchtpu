# SPDX-License-Identifier: Apache-2.0
"""Two independent, single-buffer Pallas attention-preprocessing kernels.

Weights use [input, output] layout and FP32 per-output-channel scales.
RoPE uses trailing interleaved channels and FP32 coefficient tables already
gathered for the current token positions. Neither kernel writes the KV cache.
The first returns contiguous rotated SWA values for the caller to publish.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.attention_preprocessing.tuned_block_sizes import (
    HEAD_BLOCK,
    TOKEN_BLOCK,
)

_LANE = 128


def _quantize_rows(x):
    values = x.astype(jnp.float32)
    limit = float(jnp.finfo(jnp.float8_e4m3fn).max)
    scale = jnp.max(jnp.abs(values), axis=-1, keepdims=True) / limit
    scale = jnp.where(scale == 0, 1.0, scale)
    encoded = jnp.clip(values / scale, -limit, limit)
    return encoded.astype(jnp.float8_e4m3fn), scale


def _rms_norm(values, gamma, eps):
    inv_rms = jax.lax.rsqrt(jnp.mean(values * values, axis=-1, keepdims=True) + eps)
    return (values * inv_rms * gamma.astype(jnp.float32)).astype(jnp.bfloat16)


def _lane_gather(operand, indices):
    rank = operand.ndim
    lanes = indices.shape[0]
    coords = jnp.broadcast_to(indices, (*operand.shape[:-1], lanes))[..., None]
    batching = tuple(range(rank - 1))
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(rank - 1,),
        start_index_map=(rank - 1,),
        operand_batching_dims=batching,
        start_indices_batching_dims=batching,
    )
    return jax.lax.gather(
        operand,
        coords,
        dimension_numbers=dimension_numbers,
        slice_sizes=(1,) * rank,
        unique_indices=False,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def _rotate(values, cos, sin):
    rotary_dim = 2 * cos.shape[-1]
    lane = jnp.arange(_LANE)
    channel = lane - (_LANE - rotary_dim)
    is_rot = channel >= 0
    freq = jnp.maximum(channel, 0) >> 1
    cos_lanes = jnp.where(is_rot, _lane_gather(cos, freq), 1.0)
    sin_lanes = jnp.where(is_rot, _lane_gather(sin, freq), 0.0)
    if values.ndim == 3:
        cos_lanes, sin_lanes = cos_lanes[:, None, :], sin_lanes[:, None, :]
    tail = values[..., -_LANE:].astype(jnp.float32)
    sign = jnp.where((lane & 1) == 0, -1.0, 1.0)
    partner = _lane_gather(tail, lane ^ 1) * sign
    rotated = (tail * cos_lanes + partner * sin_lanes).astype(jnp.bfloat16)
    if values.shape[-1] == _LANE:
        return rotated
    return jnp.concatenate((values[..., :-_LANE], rotated), axis=-1)


def _qa_kv_norm_swa_kernel(
    x_ref,
    weight_ref,
    scale_ref,
    gamma_qr_ref,
    gamma_kv_ref,
    cos_ref,
    sin_ref,
    qr_out_ref,
    swa_out_ref,
    projected_ref,
    *,
    eps,
):
    encoded, scale = _quantize_rows(x_ref[...])
    product = jnp.dot(encoded, weight_ref[...], preferred_element_type=jnp.float32)
    projected = product * scale * scale_ref[...][None, :]
    # Preserve the linear layer's BF16 output boundary before normalization.
    projected_ref[...] = projected.astype(jnp.bfloat16).astype(jnp.float32)
    rank = gamma_qr_ref.shape[0]
    qr_out_ref[...] = _rms_norm(
        projected_ref[:, :rank], gamma_qr_ref[...][None, :], eps
    )
    kv = _rms_norm(projected_ref[:, rank:], gamma_kv_ref[...][None, :], eps)
    if swa_out_ref.shape[-1] > _LANE:
        swa_out_ref[:, :-_LANE] = kv[:, :-_LANE]
    swa_out_ref[:, -_LANE:] = _rotate(kv[:, -_LANE:], cos_ref[...], sin_ref[...])


def _qb_rope_kernel(qr_ref, weight_ref, scale_ref, cos_ref, sin_ref, out_ref, q_ref):
    encoded, scale = _quantize_rows(qr_ref[...])
    product = jnp.dot(encoded, weight_ref[...], preferred_element_type=jnp.float32)
    projected = (product * scale * scale_ref[...][None, :]).astype(jnp.bfloat16)
    q_ref[...] = projected.reshape(q_ref.shape)
    if out_ref.shape[-1] > _LANE:
        out_ref[:, :, :-_LANE] = q_ref[:, :, :-_LANE]
    out_ref[:, :, -_LANE:] = _rotate(q_ref[:, :, -_LANE:], cos_ref[...], sin_ref[...])


def _check_projection(x, weight, scale):
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("Activations and weights must both be rank 2")
    if x.dtype != jnp.bfloat16 or weight.dtype != jnp.float8_e4m3fn:
        raise ValueError("Use BF16 activations and FP8 E4M3FN weights")
    if x.shape[0] == 0 or x.shape[0] % TOKEN_BLOCK:
        raise ValueError(f"Pad token rows to a positive multiple of {TOKEN_BLOCK}")
    if x.shape[1] != weight.shape[0]:
        raise ValueError("Weight input dimension must match the activation width")
    if scale.shape not in ((weight.shape[1],), (1, weight.shape[1])):
        raise ValueError("Use one weight scale per output column")
    if scale.dtype != jnp.float32:
        raise ValueError("Weight scales must be FP32")


def _check_rope(x, cos, sin, head_dim):
    if head_dim <= 0 or head_dim % _LANE:
        raise ValueError(f"Head dimension must be a positive multiple of {_LANE}")
    if cos.ndim != 2 or cos.shape != sin.shape or cos.shape[0] != x.shape[0]:
        raise ValueError(
            "Use matching cos/sin arrays with shape [tokens, rotary_dim/2]"
        )
    if cos.dtype != jnp.float32 or sin.dtype != jnp.float32:
        raise ValueError("Use FP32 cos/sin coefficients to preserve RoPE precision")
    if not 0 < 2 * cos.shape[1] < head_dim or 2 * cos.shape[1] > _LANE:
        raise ValueError("Rotary dimension must be positive and smaller than head_dim")


def fused_qa_kv_norm_swa_serial(
    x: jax.Array,
    weight: jax.Array,
    weight_scale: jax.Array,
    gamma_qr: jax.Array,
    gamma_kv: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    eps: float = 1e-6,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Return BF16 (normalized qr, normalized and rotated SWA KV).

    The token dimension must already be padded to TOKEN_BLOCK. Gamma arrays
    are BF16 vectors, defining the qr and KV widths. No token is scattered
    into a cache here; the caller retains slot validity and cache ownership.
    """
    _check_projection(x, weight, weight_scale)
    if gamma_qr.ndim != 1 or gamma_kv.ndim != 1:
        raise ValueError("RMSNorm weights must be vectors")
    if gamma_qr.dtype != jnp.bfloat16 or gamma_kv.dtype != jnp.bfloat16:
        raise ValueError("RMSNorm weights must be BF16")
    rank, head_dim = gamma_qr.size, gamma_kv.size
    if weight.shape[1] != rank + head_dim:
        raise ValueError("Projection width must equal qr width plus KV width")
    _check_rope(x, cos, sin, head_dim)
    if eps <= 0:
        raise ValueError("RMSNorm epsilon must be positive")
    tokens, hidden = x.shape
    # One buffer per operand fixes a serial baseline, including output DMA.
    serial = pl.Buffered(buffer_count=1)
    return pl.pallas_call(
        functools.partial(_qa_kv_norm_swa_kernel, eps=eps),
        grid=(tokens // TOKEN_BLOCK,),
        in_specs=(
            pl.BlockSpec((TOKEN_BLOCK, hidden), lambda t: (t, 0), serial),
            pl.BlockSpec(weight.shape, lambda t: (0, 0), serial),
            pl.BlockSpec((rank + head_dim,), lambda t: (0,), serial),
            pl.BlockSpec(gamma_qr.shape, lambda t: (0,), serial),
            pl.BlockSpec(gamma_kv.shape, lambda t: (0,), serial),
            pl.BlockSpec((TOKEN_BLOCK, cos.shape[1]), lambda t: (t, 0), serial),
            pl.BlockSpec((TOKEN_BLOCK, sin.shape[1]), lambda t: (t, 0), serial),
        ),
        out_specs=(
            pl.BlockSpec((TOKEN_BLOCK, rank), lambda t: (t, 0), serial),
            pl.BlockSpec((TOKEN_BLOCK, head_dim), lambda t: (t, 0), serial),
        ),
        scratch_shapes=(pltpu.VMEM((TOKEN_BLOCK, rank + head_dim), jnp.float32),),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, rank), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens, head_dim), jnp.bfloat16),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name="qa_kv_norm_swa_serial",
    )(x, weight, weight_scale.reshape(-1), gamma_qr, gamma_kv, cos, sin)


def fused_qb_rope_serial(
    qr: jax.Array,
    weight: jax.Array,
    weight_scale: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    head_dim: int = 512,
    interpret: bool = False,
) -> jax.Array:
    """Return BF16 Q[tokens, heads, head_dim] without an intermediate HBM Q.

    Head windows are the outer grid axis; token blocks are the inner axis.
    The window weight address stays fixed through one complete token sweep.
    """
    _check_projection(qr, weight, weight_scale)
    if head_dim <= 0 or weight.shape[1] % head_dim:
        raise ValueError("Q-B output width must be divisible by head_dim")
    _check_rope(qr, cos, sin, head_dim)
    heads = weight.shape[1] // head_dim
    if heads % HEAD_BLOCK:
        raise ValueError(f"Pad head count to a multiple of {HEAD_BLOCK}")
    tokens, rank = qr.shape
    width = HEAD_BLOCK * head_dim
    serial = pl.Buffered(buffer_count=1)
    return pl.pallas_call(
        _qb_rope_kernel,
        grid=(heads // HEAD_BLOCK, tokens // TOKEN_BLOCK),
        in_specs=(
            pl.BlockSpec((TOKEN_BLOCK, rank), lambda h, t: (t, 0), serial),
            pl.BlockSpec((rank, width), lambda h, t: (0, h), serial),
            pl.BlockSpec((width,), lambda h, t: (h,), serial),
            pl.BlockSpec((TOKEN_BLOCK, cos.shape[1]), lambda h, t: (t, 0), serial),
            pl.BlockSpec((TOKEN_BLOCK, sin.shape[1]), lambda h, t: (t, 0), serial),
        ),
        out_specs=pl.BlockSpec(
            (TOKEN_BLOCK, HEAD_BLOCK, head_dim), lambda h, t: (t, h, 0), serial
        ),
        scratch_shapes=(pltpu.VMEM((TOKEN_BLOCK, HEAD_BLOCK, head_dim), jnp.bfloat16),),
        out_shape=jax.ShapeDtypeStruct((tokens, heads, head_dim), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary")
        ),
        interpret=interpret,
        name="qb_rope_serial",
    )(qr, weight, weight_scale.reshape(-1), cos, sin)
