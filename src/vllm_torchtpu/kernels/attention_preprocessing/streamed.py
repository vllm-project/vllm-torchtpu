# SPDX-License-Identifier: Apache-2.0
"""Q-B projection with reusable FP8 rows and immediate interleaved RoPE."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.attention_preprocessing.serial import (
    _LANE,
    _check_projection,
    _check_rope,
    _quantize_rows,
    _rotate,
)
from vllm_torchtpu.kernels.attention_preprocessing.tuned_block_sizes import (
    QB_HEAD_BLOCK,
    QB_OUTPUT_BUFFERS,
    QB_QUANT_TOKEN_BLOCK,
    QB_TOKEN_BLOCK,
)


def _qb_rope_streamed_kernel(
    qr_ref,
    weight_ref,
    scale_ref,
    cos_ref,
    sin_ref,
    out_ref,
    encoded_ref,
    row_scale_ref,
):
    token_start = pl.program_id(1) * QB_TOKEN_BLOCK
    head_dim = out_ref.shape[-1]

    @pl.when(pl.program_id(0) == 0)
    def quantize_once():
        def quantize_fragment(fragment, _):
            rows = pl.ds(fragment * QB_QUANT_TOKEN_BLOCK, QB_QUANT_TOKEN_BLOCK)
            cached_rows = pl.ds(
                token_start + fragment * QB_QUANT_TOKEN_BLOCK, QB_QUANT_TOKEN_BLOCK
            )
            encoded, scale = _quantize_rows(qr_ref[rows, :])
            encoded_ref[cached_rows, :] = encoded
            row_scale_ref[cached_rows, :] = scale

        jax.lax.fori_loop(
            0, QB_TOKEN_BLOCK // QB_QUANT_TOKEN_BLOCK, quantize_fragment, None
        )

    cached_rows = pl.ds(token_start, QB_TOKEN_BLOCK)
    encoded = encoded_ref[cached_rows, :]
    row_scale = row_scale_ref[cached_rows, :]
    product = jnp.dot(
        encoded, weight_ref[...], preferred_element_type=jnp.float32
    ).reshape(QB_TOKEN_BLOCK, QB_HEAD_BLOCK, head_dim)
    projected = (
        product
        * row_scale[:, None, :]
        * scale_ref[...].reshape(1, QB_HEAD_BLOCK, head_dim)
    ).astype(jnp.bfloat16)
    if head_dim > _LANE:
        out_ref[:, :, :-_LANE] = projected[:, :, :-_LANE]
    out_ref[:, :, -_LANE:] = _rotate(
        projected[:, :, -_LANE:], cos_ref[...], sin_ref[...]
    )


def fused_qb_rope_streamed(
    qr: jax.Array,
    weight: jax.Array,
    weight_scale: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    head_dim: int = 512,
    interpret: bool = False,
) -> jax.Array:
    """Quantize each qr row once and return BF16 Q[tokens, heads, head_dim].

    The arbitrary grid axes run in order. The first head window fills the
    VMEM quantization cache; later windows reuse it. Scratch grows with tokens.
    The output is reshaped before its BF16 rounding and written directly.
    """
    _check_projection(qr, weight, weight_scale)
    if head_dim <= 0 or weight.shape[1] % head_dim:
        raise ValueError("Q-B output width must be divisible by head_dim")
    _check_rope(qr, cos, sin, head_dim)
    heads = weight.shape[1] // head_dim
    if heads % QB_HEAD_BLOCK:
        raise ValueError(f"Pad head count to a multiple of {QB_HEAD_BLOCK}")
    tokens, rank = qr.shape
    width = QB_HEAD_BLOCK * head_dim
    serial = pl.Buffered(buffer_count=1)
    return pl.pallas_call(
        _qb_rope_streamed_kernel,
        grid=(heads // QB_HEAD_BLOCK, tokens // QB_TOKEN_BLOCK),
        in_specs=(
            pl.BlockSpec((QB_TOKEN_BLOCK, rank), lambda h, t: (t, 0), serial),
            pl.BlockSpec((rank, width), lambda h, t: (0, h), serial),
            pl.BlockSpec((width,), lambda h, t: (h,), serial),
            pl.BlockSpec((QB_TOKEN_BLOCK, cos.shape[1]), lambda h, t: (t, 0), serial),
            pl.BlockSpec((QB_TOKEN_BLOCK, sin.shape[1]), lambda h, t: (t, 0), serial),
        ),
        out_specs=pl.BlockSpec(
            (QB_TOKEN_BLOCK, QB_HEAD_BLOCK, head_dim),
            lambda h, t: (t, h, 0),
            pl.Buffered(buffer_count=QB_OUTPUT_BUFFERS),
        ),
        scratch_shapes=(
            pltpu.VMEM((tokens, rank), jnp.float8_e4m3fn),
            pltpu.VMEM((tokens, 1), jnp.float32),
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, heads, head_dim), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary")
        ),
        interpret=interpret,
        name="qb_rope_streamed",
    )(qr, weight, weight_scale.reshape(-1), cos, sin)
