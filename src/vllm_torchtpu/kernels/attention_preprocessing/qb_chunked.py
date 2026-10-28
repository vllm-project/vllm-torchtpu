# SPDX-License-Identifier: Apache-2.0
"""Q-B projection and interleaved RoPE written as packed head pairs, chunked.

Same output contract as ``qb_packed``: Q[tokens, heads, 512] in BF16 tiles pairs
heads 2p/2p+1 into one 32-bit word, so a [tokens, 512] u32 tile placed at head
pair p by DMA is the final layout. Here each head is produced in column chunks
of ``QB4_COLUMN_BLOCK`` lanes, so only one [256, chunk] FP32 product is live.

A BF16-rounded FP32 value carries its BF16 bits in the upper half word, so the
low head is stored as ``bits >> 16`` and the high head is merged in with
``bits & 0xFFFF0000``. This is bit-exact with the serial kernel.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.attention_preprocessing.qb_packed import _coefficients
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

_HIGH = 0xFFFF0000


def _qb_rope_chunked_kernel(
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
    chunk,
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
    tail_start = head_dim - _LANE

    def copy(slot, pair):
        return pltpu.make_async_copy(
            obuf_ref.at[slot], out_words.at[tokens, pair, :], sems.at[slot]
        )

    def chunk_bits(local, first):
        """BF16-rounded projection (RoPE applied on the tail) as FP32 bits."""
        columns = pl.ds(pl.multiple_of(local * head_dim + first, chunk), chunk)
        product = jnp.dot(
            encoded_ref[tokens, :],
            weight_ref[:, columns],
            preferred_element_type=jnp.float32,
        )
        projected = product * row_scale_ref[tokens, :]
        projected = (projected * scale_ref[columns][None, :]).astype(jnp.bfloat16)
        values = projected.astype(jnp.float32)
        if first + chunk > tail_start:
            split = tail_start - first
            tail = values[:, split:]
            partner = _lane_gather(tail, jnp.arange(_LANE) ^ 1)
            rotated = (
                (tail * coeff_ref[0, tokens, :] + partner * coeff_ref[1, tokens, :])
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
            values = (
                rotated
                if split == 0
                else jnp.concatenate((values[:, :split], rotated), axis=-1)
            )
        return jax.lax.bitcast_convert_type(values, jnp.uint32)

    def emit(pair, _):
        index = step * pairs + pair
        slot = index % QB3_OUTPUT_SLOTS

        @pl.when(index >= QB3_OUTPUT_SLOTS)
        def recycle():
            copy(slot, 0).wait()

        for first in range(0, head_dim, chunk):
            lanes = pl.ds(first, chunk)
            obuf_ref[slot, :, lanes] = chunk_bits(2 * pair, first) >> 16
            obuf_ref[slot, :, lanes] = obuf_ref[slot, :, lanes] | (
                chunk_bits(2 * pair + 1, first) & jnp.uint32(_HIGH)
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


def fused_qb_rope_chunked(
    qr: jax.Array,
    weight: jax.Array,
    weight_scale: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    head_dim: int = 512,
    chunk: int = 128,
    interpret: bool = False,
) -> jax.Array:
    """Return BF16 Q[tokens, heads, head_dim]; numerics match the serial kernel."""
    _check_projection(qr, weight, weight_scale)
    if head_dim <= 0 or weight.shape[1] % head_dim or head_dim % chunk:
        raise ValueError("Q-B head width must be a multiple of the column chunk")
    if chunk % _LANE:
        raise ValueError(f"Column chunk must be a multiple of {_LANE}")
    _check_rope(qr, cos, sin, head_dim)
    heads = weight.shape[1] // head_dim
    if heads % QB3_HEAD_BLOCK:
        raise ValueError(f"Pad head count to a multiple of {QB3_HEAD_BLOCK}")
    tokens, rank = qr.shape
    if tokens % QB3_TOKEN_BLOCK:
        raise ValueError(f"Pad token rows to a multiple of {QB3_TOKEN_BLOCK}")
    tb, width = QB3_TOKEN_BLOCK, QB3_HEAD_BLOCK * head_dim
    last = tokens // tb - 1

    def rows(h, t):
        return (jnp.where(h == 0, t, last), 0)

    return pl.pallas_call(
        functools.partial(_qb_rope_chunked_kernel, head_dim=head_dim, chunk=chunk),
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
        name=f"qb_rope_chunked{chunk}",
    )(qr, weight, weight_scale.reshape(-1), cos, sin)
