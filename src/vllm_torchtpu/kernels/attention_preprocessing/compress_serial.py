# SPDX-License-Identifier: Apache-2.0
"""Serial V4.1 compressor: projection, pooling, RMSNorm, RoPE and CSA2 encoding.

One kernel replaces ``DeepseekCompressor.project`` (BF16 MXU, FP32 output),
``save_compress_norm`` (ratio-2 gated pairing over the previous row, RMSNorm)
and the main-KV ``_rotate`` + ``encode_csa2_kv`` that ``publish_csa2_kv`` runs
before its scatter. The caller keeps cache ownership: it scatters ``words`` and
``scales`` to the main cache, ``kv_score`` rows to the state ring, and feeds the
unrotated ``latent`` to the Index-K path.

Token tiles run in order. The last projected row of a tile is carried in VMEM
as the previous row of the next tile's first token; ``has_prev`` selects that
in-chunk row, otherwise the caller-gathered ring row is used.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.attention_preprocessing.serial import _LANE, _rotate
from vllm_torchtpu.kernels.attention_preprocessing.tuned_block_sizes import (
    COMPRESS_FRAGMENT,
    COMPRESS_TOKEN_BLOCK,
    COMPRESS_VMEM_LIMIT_BYTES,
)

_DIM = 512
_FP8_MAX = 448.0


def _encode_csa2(values):
    """FP32 [rows,512] post-RoPE -> INT32 words [rows,128], FP32 scales [rows,1]."""
    scale = jnp.maximum(jnp.max(jnp.abs(values), axis=-1, keepdims=True), 1e-4)
    scale = scale / _FP8_MAX
    quantized = (values / scale).astype(jnp.float8_e4m3fn)
    planes = [
        jax.lax.bitcast_convert_type(
            quantized[:, g * _LANE : (g + 1) * _LANE], jnp.uint8
        ).astype(jnp.uint32)
        for g in range(_DIM // _LANE)
    ]
    words = planes[0] | (planes[1] << 8) | (planes[2] << 16) | (planes[3] << 24)
    return jax.lax.bitcast_convert_type(words, jnp.int32), scale


def _compress_kernel(
    x_ref,
    weight_ref,
    gamma_ref,
    has_prev_ref,
    ring_ref,
    valid_ref,
    cos_ref,
    sin_ref,
    score_out_ref,
    latent_out_ref,
    words_out_ref,
    scales_out_ref,
    shifted_ref,
    *,
    eps,
    ratio,
):
    tb = x_ref.shape[0]
    product = jnp.dot(x_ref[...], weight_ref[...], preferred_element_type=jnp.float32)
    score_out_ref[...] = product
    if ratio == 2:

        @pl.when(pl.program_id(0) == 0)
        def _():
            shifted_ref[pl.ds(tb + 7, 1), :] = jnp.zeros((1, 2 * _DIM), jnp.float32)

        # Row 7 holds the previous tile's last row; rows 8.. hold this tile, so
        # rows 7..tb+6 are each token's in-chunk predecessor.
        shifted_ref[pl.ds(7, 1), :] = shifted_ref[pl.ds(tb + 7, 1), :]
        shifted_ref[pl.ds(8, tb), :] = product
        in_chunk = shifted_ref[pl.ds(7, tb), :]
        previous = jnp.where(has_prev_ref[...] != 0, in_chunk, ring_ref[...])
        score_prev, score_cur = previous[:, _DIM:], product[:, _DIM:]
        peak = jnp.maximum(score_prev, score_cur)
        e_prev = jnp.exp(score_prev - peak)
        e_cur = jnp.exp(score_cur - peak)
        total = e_prev + e_cur
        values = previous[:, :_DIM] * (e_prev / total) + product[:, :_DIM] * (
            e_cur / total
        )
    else:
        values = product
    inv_rms = jax.lax.rsqrt(jnp.mean(values * values, axis=-1, keepdims=True) + eps)
    latent = (values * inv_rms * gamma_ref[...].astype(jnp.float32)[None, :]).astype(
        jnp.bfloat16
    )
    latent = jnp.where(valid_ref[...] != 0, latent, jnp.zeros_like(latent))
    latent_out_ref[...] = latent
    tail = _rotate(latent[:, -_LANE:], cos_ref[...], sin_ref[...])
    rotated = jnp.concatenate((latent[:, :-_LANE], tail), axis=-1).astype(jnp.float32)
    words, scales = _encode_csa2(rotated)
    words_out_ref[...] = words
    scales_out_ref[...] = scales


def _compress_fragment_kernel(
    x_ref,
    weight_ref,
    gamma_ref,
    has_prev_ref,
    ring_ref,
    valid_ref,
    cos_ref,
    sin_ref,
    score_out_ref,
    latent_out_ref,
    words_out_ref,
    scales_out_ref,
    shifted_ref,
    *,
    eps,
    ratio,
):
    """Same arithmetic as ``_compress_kernel``; post-matmul work per fragment.

    The projected tile is staged in VMEM once, then pooling, RMSNorm, RoPE and
    encoding run over ``COMPRESS_FRAGMENT`` rows at a time, so only one
    fragment's FP32 rows are live in vector registers.
    """
    tb = x_ref.shape[0]
    score_out_ref[...] = jnp.dot(
        x_ref[...], weight_ref[...], preferred_element_type=jnp.float32
    )
    if ratio == 2:

        @pl.when(pl.program_id(0) == 0)
        def _():
            shifted_ref[pl.ds(tb + 7, 1), :] = jnp.zeros((1, 2 * _DIM), jnp.float32)

        shifted_ref[pl.ds(7, 1), :] = shifted_ref[pl.ds(tb + 7, 1), :]
        shifted_ref[pl.ds(8, tb), :] = score_out_ref[...]
    gamma = gamma_ref[...].astype(jnp.float32)[None, :]

    def fragment(f, _):
        start = pl.multiple_of(f * COMPRESS_FRAGMENT, COMPRESS_FRAGMENT)
        rows = pl.ds(start, COMPRESS_FRAGMENT)
        current = score_out_ref[rows, :]
        if ratio == 2:
            in_chunk = shifted_ref[pl.ds(start + 7, COMPRESS_FRAGMENT), :]
            previous = jnp.where(
                has_prev_ref[rows, :] != 0, in_chunk, ring_ref[rows, :]
            )
            score_prev, score_cur = previous[:, _DIM:], current[:, _DIM:]
            peak = jnp.maximum(score_prev, score_cur)
            e_prev = jnp.exp(score_prev - peak)
            e_cur = jnp.exp(score_cur - peak)
            total = e_prev + e_cur
            values = previous[:, :_DIM] * (e_prev / total) + current[:, :_DIM] * (
                e_cur / total
            )
        else:
            values = current
        inv_rms = jax.lax.rsqrt(jnp.mean(values * values, axis=-1, keepdims=True) + eps)
        latent = (values * inv_rms * gamma).astype(jnp.bfloat16)
        latent = jnp.where(valid_ref[rows, :] != 0, latent, jnp.zeros_like(latent))
        latent_out_ref[rows, :] = latent
        tail = _rotate(latent[:, -_LANE:], cos_ref[rows, :], sin_ref[rows, :])
        rotated = jnp.concatenate((latent[:, :-_LANE], tail), axis=-1).astype(
            jnp.float32
        )
        words, scales = _encode_csa2(rotated)
        words_out_ref[rows, :] = words
        scales_out_ref[rows, :] = scales

    jax.lax.fori_loop(0, tb // COMPRESS_FRAGMENT, fragment, None)


def fused_compress_serial(
    x: jax.Array,
    weight: jax.Array,
    gamma: jax.Array,
    has_prev: jax.Array,
    ring_prev: jax.Array,
    valid: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    eps: float = 1e-6,
    ratio: int = 2,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return (kv_score FP32, latent BF16, CSA2 words INT32, scales FP32 [T,1]).

    ``has_prev`` and ``valid`` are INT32 [T,1]; ``ring_prev`` is FP32
    [T, ratio*512]; ``cos``/``sin`` are FP32 [T,32] at each group start.
    Every operand is single-buffered, so each tile's DMA is exposed.
    """
    return _compress_call(
        x,
        weight,
        gamma,
        has_prev,
        ring_prev,
        valid,
        cos,
        sin,
        eps=eps,
        ratio=ratio,
        interpret=interpret,
        pipelined=False,
    )


def fused_compress_pipelined(
    x: jax.Array,
    weight: jax.Array,
    gamma: jax.Array,
    has_prev: jax.Array,
    ring_prev: jax.Array,
    valid: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    eps: float = 1e-6,
    ratio: int = 2,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Same arithmetic as the serial kernel; token tiles are double-buffered.

    The resident weight and norm gamma keep one buffer and are fetched once.
    """
    return _compress_call(
        x,
        weight,
        gamma,
        has_prev,
        ring_prev,
        valid,
        cos,
        sin,
        eps=eps,
        ratio=ratio,
        interpret=interpret,
        pipelined=True,
    )


def fused_compress_fragmented(
    x: jax.Array,
    weight: jax.Array,
    gamma: jax.Array,
    has_prev: jax.Array,
    ring_prev: jax.Array,
    valid: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    eps: float = 1e-6,
    ratio: int = 2,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pipelined kernel with post-matmul work split into row fragments."""
    return _compress_call(
        x,
        weight,
        gamma,
        has_prev,
        ring_prev,
        valid,
        cos,
        sin,
        eps=eps,
        ratio=ratio,
        interpret=interpret,
        pipelined=True,
        fragmented=True,
    )


def _compress_call(
    x,
    weight,
    gamma,
    has_prev,
    ring_prev,
    valid,
    cos,
    sin,
    *,
    eps,
    ratio,
    interpret,
    pipelined,
    fragmented=False,
):
    tokens, hidden = x.shape
    width = ratio * _DIM
    if ratio not in (1, 2):
        raise ValueError("V4.1 compression ratio must be 1 or 2")
    if x.dtype != jnp.bfloat16 or weight.dtype != jnp.bfloat16:
        raise ValueError("Compressor projection uses BF16 inputs and weights")
    if weight.shape != (hidden, width) or gamma.shape != (_DIM,):
        raise ValueError("Weight must be [hidden, ratio*512]; gamma [512]")
    if tokens % COMPRESS_TOKEN_BLOCK:
        raise ValueError(f"Pad token rows to a multiple of {COMPRESS_TOKEN_BLOCK}")
    tb = COMPRESS_TOKEN_BLOCK
    once = pl.Buffered(buffer_count=1)
    # Token tiles: two buffers when pipelined, otherwise one (exposed DMA).
    tile = pl.Buffered(buffer_count=2 if pipelined else 1)
    row = lambda t: (t, 0)  # noqa: E731
    return pl.pallas_call(
        functools.partial(
            _compress_fragment_kernel if fragmented else _compress_kernel,
            eps=eps,
            ratio=ratio,
        ),
        grid=(tokens // tb,),
        in_specs=(
            pl.BlockSpec((tb, hidden), row, tile),
            pl.BlockSpec(weight.shape, lambda t: (0, 0), once),
            pl.BlockSpec(gamma.shape, lambda t: (0,), once),
            pl.BlockSpec((tb, 1), row, tile),
            pl.BlockSpec((tb, ring_prev.shape[1]), row, tile),
            pl.BlockSpec((tb, 1), row, tile),
            pl.BlockSpec((tb, cos.shape[1]), row, tile),
            pl.BlockSpec((tb, sin.shape[1]), row, tile),
        ),
        out_specs=(
            pl.BlockSpec((tb, width), row, tile),
            pl.BlockSpec((tb, _DIM), row, tile),
            pl.BlockSpec((tb, _LANE), row, tile),
            pl.BlockSpec((tb, 1), row, tile),
        ),
        scratch_shapes=(pltpu.VMEM((tb + 8, 2 * _DIM), jnp.float32),),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, width), jnp.float32),
            jax.ShapeDtypeStruct((tokens, _DIM), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens, _LANE), jnp.int32),
            jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=COMPRESS_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name=f"compress_ratio{ratio}_"
        + ("fragmented" if fragmented else "pipelined" if pipelined else "serial"),
    )(x, weight, gamma, has_prev, ring_prev, valid, cos, sin)
