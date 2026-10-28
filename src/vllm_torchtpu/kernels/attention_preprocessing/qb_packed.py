# SPDX-License-Identifier: Apache-2.0
"""Q-B projection and interleaved RoPE written as packed head pairs.

Q[tokens, heads, 512] in BF16 tiles pairs heads into one 32-bit sublane, so
the output viewed as u32 is [tokens, heads/2, 512] and word (t, p, c) holds
heads 2p (low half) and 2p+1 (high half). The MXU yields token-major tiles of
one head; packing two such tiles elementwise produces exactly that word, and a
DMA places the [tokens, 512] word tile at head pair p. No in-register
token/head transposition is needed.

Activation quantization, scale order and BF16 rounding before RoPE match the
serial kernel.
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
    QB3_HEAD_BLOCK,
    QB3_OUTPUT_SLOTS,
    QB3_QUANT_TOKEN_BLOCK,
    QB3_TOKEN_BLOCK,
    QB3_VMEM_LIMIT_BYTES,
)


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


def _qb_rope_packed_kernel(
    qr_ref,
    weight_ref,
    scale_ref,
    cos_ref,
    sin_ref,
    out_hbm,
    encoded_ref,
    row_scale_ref,
    coeff_ref,
    obuf_ref,
    sems,
    *,
    head_dim,
):
    window = pl.program_id(0)
    tile = pl.program_id(1)
    windows, tiles = pl.num_programs(0), pl.num_programs(1)
    tb, pairs = QB3_TOKEN_BLOCK, QB3_HEAD_BLOCK // 2
    token_start = pl.multiple_of(tile * tb, tb)
    out_words = out_hbm.bitcast(jnp.uint32)  # [tokens, heads/2, head_dim]

    @pl.when(window == 0)
    def prepare_rows():
        def quantize_fragment(fragment, _):
            rows = pl.ds(fragment * QB3_QUANT_TOKEN_BLOCK, QB3_QUANT_TOKEN_BLOCK)
            cached = pl.ds(
                token_start + fragment * QB3_QUANT_TOKEN_BLOCK, QB3_QUANT_TOKEN_BLOCK
            )
            encoded, scale = _quantize_rows(qr_ref[rows, :])
            encoded_ref[cached, :] = encoded
            row_scale_ref[cached, :] = scale

        jax.lax.fori_loop(0, tb // QB3_QUANT_TOKEN_BLOCK, quantize_fragment, None)
        cos_lanes, sin_lanes = _coefficients(cos_ref[...], sin_ref[...])
        coeff_ref[0, pl.ds(token_start, tb), :] = cos_lanes
        coeff_ref[1, pl.ds(token_start, tb), :] = sin_lanes

    tokens = pl.ds(token_start, tb)
    step = window * tiles + tile

    def copy(slot, pair):
        return pltpu.make_async_copy(
            obuf_ref.at[slot], out_words.at[tokens, pair, :], sems.at[slot]
        )

    def head(local):
        columns = pl.ds(pl.multiple_of(local * head_dim, head_dim), head_dim)
        product = jnp.dot(
            encoded_ref[tokens, :],
            weight_ref[:, columns],
            preferred_element_type=jnp.float32,
        )
        projected = product * row_scale_ref[tokens, :]
        projected = (projected * scale_ref[columns][None, :]).astype(jnp.bfloat16)
        # RoPE on the trailing lane block, after the linear BF16 boundary.
        tail = projected[:, -_LANE:].astype(jnp.float32)
        partner = _lane_gather(tail, jnp.arange(_LANE) ^ 1)
        rotated = (
            tail * coeff_ref[0, tokens, :] + partner * coeff_ref[1, tokens, :]
        ).astype(jnp.bfloat16)
        return projected[:, :-_LANE].astype(jnp.float32), rotated.astype(jnp.float32)

    def emit(pair, _):
        index = step * pairs + pair
        slot = index % QB3_OUTPUT_SLOTS

        @pl.when(index >= QB3_OUTPUT_SLOTS)
        def recycle():
            copy(slot, 0).wait()

        low_plain, low_tail = head(2 * pair)
        high_plain, high_tail = head(2 * pair + 1)
        tail_start = head_dim - _LANE
        # Values are already BF16-rounded, so the FP32->BF16 pack is exact.
        obuf_ref[slot, :, pl.ds(0, tail_start)] = pltpu.pack_elementwise(
            [low_plain, high_plain], packed_dtype=jnp.bfloat16
        )
        obuf_ref[slot, :, pl.ds(tail_start, _LANE)] = pltpu.pack_elementwise(
            [low_tail, high_tail], packed_dtype=jnp.bfloat16
        )
        copy(slot, window * pairs + pair).start()

    jax.lax.fori_loop(0, pairs, emit, None)

    @pl.when((window == windows - 1) & (tile == tiles - 1))
    def drain():
        total = windows * tiles * pairs
        for back in range(QB3_OUTPUT_SLOTS):
            index = total - 1 - back

            @pl.when(index >= 0)
            def _(index=index):
                copy(index % QB3_OUTPUT_SLOTS, 0).wait()


def fused_qb_rope_packed(
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

    Grid axes run in order. The first head window quantizes every qr row and
    builds per-token RoPE coefficients; later windows reuse both. Each head pair
    is packed into a VMEM word tile whose writeback overlaps later pairs.
    """
    _check_projection(qr, weight, weight_scale)
    if head_dim <= 0 or weight.shape[1] % head_dim or head_dim % _LANE:
        raise ValueError("Q-B output width must be a multiple of a 128-lane head")
    _check_rope(qr, cos, sin, head_dim)
    heads = weight.shape[1] // head_dim
    if heads % QB3_HEAD_BLOCK:
        raise ValueError(f"Pad head count to a multiple of {QB3_HEAD_BLOCK}")
    tokens, rank = qr.shape
    if tokens % QB3_TOKEN_BLOCK:
        raise ValueError(f"Pad token rows to a multiple of {QB3_TOKEN_BLOCK}")
    tb, width = QB3_TOKEN_BLOCK, QB3_HEAD_BLOCK * head_dim
    last = tokens // tb - 1

    # Row inputs are only read in the first window; keep their block index
    # fixed afterwards so the pipeline does not fetch them again.
    def rows(h, t):
        return (jnp.where(h == 0, t, last), 0)

    return pl.pallas_call(
        functools.partial(_qb_rope_packed_kernel, head_dim=head_dim),
        grid=(heads // QB3_HEAD_BLOCK, tokens // tb),
        in_specs=(
            pl.BlockSpec((tb, rank), rows),
            pl.BlockSpec((rank, width), lambda h, t: (0, h)),
            pl.BlockSpec((width,), lambda h, t: (h,)),
            pl.BlockSpec((tb, cos.shape[1]), rows),
            pl.BlockSpec((tb, sin.shape[1]), rows),
        ),
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=(
            pltpu.VMEM((tokens, rank), jnp.float8_e4m3fn),
            pltpu.VMEM((tokens, 1), jnp.float32),
            pltpu.VMEM((2, tokens, _LANE), jnp.float32),
            pltpu.VMEM((QB3_OUTPUT_SLOTS, tb, head_dim), jnp.uint32),
            pltpu.SemaphoreType.DMA((QB3_OUTPUT_SLOTS,)),
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, heads, head_dim), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=QB3_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="qb_rope_packed",
    )(qr, weight, weight_scale.reshape(-1), cos, sin)
