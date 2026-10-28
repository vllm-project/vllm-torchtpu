# SPDX-License-Identifier: Apache-2.0
"""Q-B projection and interleaved RoPE without in-register head transposes.

The MXU yields token-major rows, while Q[tokens, heads, head_dim] tiles put
heads on sublanes. Each projected head is stored with a sublane stride equal to
the head block, so VMEM row ``token * heads + head`` holds that head of that
token. Eight-token groups are then read back contiguously as complete output
tiles. Activation quantization, scale order and BF16 rounding before RoPE match
the serial kernel.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.attention_preprocessing.serial import (
    _LANE,
    _check_projection,
    _check_rope,
    _lane_gather,
    _quantize_rows,
)
from vllm_torchtpu.kernels.attention_preprocessing.tuned_block_sizes import (
    QB2_COLUMN_BLOCK,
    QB2_HEAD_BLOCK,
    QB2_QUANT_TOKEN_BLOCK,
    QB2_TOKEN_BLOCK,
    QB2_VMEM_LIMIT_BYTES,
)

_GROUP = 8  # tokens per FP32 vreg of a projected column tile


def _coefficients(cos, sin):
    """Per-token 128-lane cos and sign-folded sin; NoPE lanes get identity."""
    rotary_dim = 2 * cos.shape[-1]
    lane = jnp.arange(_LANE)
    channel = lane - (_LANE - rotary_dim)
    is_rot = channel >= 0
    freq = jnp.maximum(channel, 0) >> 1
    sign = jnp.where((lane & 1) == 0, -1.0, 1.0)
    cos_lanes = jnp.where(is_rot, _lane_gather(cos, freq), 1.0)
    # Negation is exact, so (partner * sign) * sin == partner * (sign * sin).
    sin_lanes = jnp.where(is_rot, _lane_gather(sin, freq), 0.0) * sign
    return cos_lanes, sin_lanes


def _qb_rope_strided_kernel(
    qr_ref,
    weight_ref,
    scale_ref,
    cos_ref,
    sin_ref,
    out_ref,
    encoded_ref,
    row_scale_ref,
    coeff_ref,
    stage_ref,
    *,
    head_dim,
):
    window = pl.program_id(0)
    token_start = pl.program_id(1) * QB2_TOKEN_BLOCK
    heads = QB2_HEAD_BLOCK

    @pl.when(window == 0)
    def prepare_rows():
        def quantize_fragment(fragment, _):
            rows = pl.ds(fragment * QB2_QUANT_TOKEN_BLOCK, QB2_QUANT_TOKEN_BLOCK)
            cached = pl.ds(
                token_start + fragment * QB2_QUANT_TOKEN_BLOCK, QB2_QUANT_TOKEN_BLOCK
            )
            encoded, scale = _quantize_rows(qr_ref[rows, :])
            encoded_ref[cached, :] = encoded
            row_scale_ref[cached, :] = scale

        jax.lax.fori_loop(
            0, QB2_TOKEN_BLOCK // QB2_QUANT_TOKEN_BLOCK, quantize_fragment, None
        )
        cos_lanes, sin_lanes = _coefficients(cos_ref[...], sin_ref[...])
        coeff_ref[0, pl.ds(token_start, QB2_TOKEN_BLOCK), :] = cos_lanes
        coeff_ref[1, pl.ds(token_start, QB2_TOKEN_BLOCK), :] = sin_lanes

    tokens = pl.ds(token_start, QB2_TOKEN_BLOCK)
    tail_start = head_dim - _LANE
    for half in range(head_dim // QB2_COLUMN_BLOCK):
        first = half * QB2_COLUMN_BLOCK

        def project(head, _, first=first):
            column = pl.multiple_of(head * head_dim + first, QB2_COLUMN_BLOCK)
            columns = pl.ds(column, QB2_COLUMN_BLOCK)
            product = jnp.dot(
                encoded_ref[tokens, :],
                weight_ref[:, columns],
                preferred_element_type=jnp.float32,
            )
            projected = product * row_scale_ref[tokens, :]
            projected = projected * scale_ref[columns][None, :]
            # Strided stores need a 128-lane base memref: one plane per lane block.
            for plane in range(QB2_COLUMN_BLOCK // _LANE):
                stage_ref.at[plane][pl.ds(head, QB2_TOKEN_BLOCK, stride=heads), :] = (
                    projected[:, plane * _LANE : (plane + 1) * _LANE]
                )

        jax.lax.fori_loop(0, heads, project, None)

        def emit(group, _, first=first):
            rows = pl.ds(
                pl.multiple_of(group * _GROUP * heads, _GROUP * heads), _GROUP * heads
            )
            local = pl.ds(pl.multiple_of(group * _GROUP, _GROUP), _GROUP)
            values = jnp.concatenate(
                [
                    stage_ref.at[plane][rows, :].reshape(_GROUP, heads, _LANE)
                    for plane in range(QB2_COLUMN_BLOCK // _LANE)
                ],
                axis=-1,
            )
            plain = min(QB2_COLUMN_BLOCK, max(0, tail_start - first))
            if plain:
                out_ref[local, :, pl.ds(first, plain)] = values[:, :, :plain].astype(
                    jnp.bfloat16
                )
            if plain < QB2_COLUMN_BLOCK:
                # BF16 linear output boundary precedes RoPE, as in the reference.
                tail = values[:, :, plain:].astype(jnp.bfloat16).astype(jnp.float32)
                coeff_rows = pl.ds(token_start + group * _GROUP, _GROUP)
                cos = coeff_ref[0, coeff_rows, :][:, None, :]
                sin = coeff_ref[1, coeff_rows, :][:, None, :]
                partner = _lane_gather(tail, jnp.arange(_LANE) ^ 1)
                out_ref[local, :, pl.ds(tail_start, _LANE)] = (
                    tail * cos + partner * sin
                ).astype(jnp.bfloat16)

        jax.lax.fori_loop(0, QB2_TOKEN_BLOCK // _GROUP, emit, None)


def fused_qb_rope_strided(
    qr: jax.Array,
    weight: jax.Array,
    weight_scale: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    head_dim: int = 512,
    interpret: bool = False,
) -> jax.Array:
    """Return BF16 Q[tokens, heads, head_dim]; numerics match the serial kernel.

    Grid axes run in order: the first head window quantizes every qr row once
    and builds per-token RoPE coefficients; later windows reuse both. Weights
    for the next head window prefetch during the current window's last tile.
    """
    _check_projection(qr, weight, weight_scale)
    if head_dim <= 0 or weight.shape[1] % head_dim:
        raise ValueError("Q-B output width must be divisible by head_dim")
    if head_dim % QB2_COLUMN_BLOCK:
        raise ValueError(f"Head dimension must be a multiple of {QB2_COLUMN_BLOCK}")
    _check_rope(qr, cos, sin, head_dim)
    heads = weight.shape[1] // head_dim
    if heads % QB2_HEAD_BLOCK:
        raise ValueError(f"Pad head count to a multiple of {QB2_HEAD_BLOCK}")
    tokens, rank = qr.shape
    if tokens % QB2_TOKEN_BLOCK:
        raise ValueError(f"Pad token rows to a multiple of {QB2_TOKEN_BLOCK}")
    width = QB2_HEAD_BLOCK * head_dim
    last = tokens // QB2_TOKEN_BLOCK - 1

    # Row inputs are only read in the first window; keep their block index
    # fixed afterwards so the pipeline does not fetch them again.
    def rows(h, t):
        return (jnp.where(h == 0, t, last), 0)

    return pl.pallas_call(
        functools.partial(_qb_rope_strided_kernel, head_dim=head_dim),
        grid=(heads // QB2_HEAD_BLOCK, tokens // QB2_TOKEN_BLOCK),
        in_specs=(
            pl.BlockSpec((QB2_TOKEN_BLOCK, rank), rows),
            pl.BlockSpec((rank, width), lambda h, t: (0, h)),
            pl.BlockSpec((width,), lambda h, t: (h,)),
            pl.BlockSpec((QB2_TOKEN_BLOCK, cos.shape[1]), rows),
            pl.BlockSpec((QB2_TOKEN_BLOCK, sin.shape[1]), rows),
        ),
        out_specs=pl.BlockSpec(
            (QB2_TOKEN_BLOCK, QB2_HEAD_BLOCK, head_dim), lambda h, t: (t, h, 0)
        ),
        scratch_shapes=(
            pltpu.VMEM((tokens, rank), jnp.float8_e4m3fn),
            pltpu.VMEM((tokens, 1), jnp.float32),
            pltpu.VMEM((2, tokens, _LANE), jnp.float32),
            pltpu.VMEM(
                (QB2_COLUMN_BLOCK // _LANE, QB2_TOKEN_BLOCK * QB2_HEAD_BLOCK, _LANE),
                jnp.float32,
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, heads, head_dim), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=QB2_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="qb_rope_strided",
    )(qr, weight, weight_scale.reshape(-1), cos, sin)
