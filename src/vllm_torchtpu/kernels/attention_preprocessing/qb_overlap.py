# SPDX-License-Identifier: Apache-2.0
"""Q-B projection and interleaved RoPE with post-processing one head pair behind.

Same output layout and arithmetic as ``qb_packed``: two heads are packed into
one u32 word tile and a DMA places it at head pair p of the output's u32 view.
The difference is scheduling. Each head pair's FP32 product goes straight to one
of two VMEM slots, and the loop body that issues pair g's matmul also scales,
rounds, rotates and packs pair g-1 from the other slot. The two halves share no
data, so the bundle scheduler can interleave MXU pushes with VPU work, and only
a 32-row fragment of post-processing is live in registers at a time.

The last pair of a grid step is post-processed by the next step, so output
slots are indexed by the global pair number g modulo ``_SLOTS`` and each slot is
waited exactly once before reuse.
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
    QB5_HEAD_BLOCK,
    QB5_QUANT_TOKEN_BLOCK,
    QB5_TOKEN_BLOCK,
    QB5_VMEM_LIMIT_BYTES,
)

_FRAGMENT = 32  # rows of post-processing held in registers at once


def _qb_rope_overlap_kernel(
    qr_ref,
    weight_ref,
    scale_ref,
    cos_ref,
    sin_ref,
    out_hbm,
    encoded_ref,
    row_scale_ref,
    coeff_ref,
    product_ref,
    obuf_ref,
    sems,
    *,
    head_dim,
):
    window, tile = pl.program_id(0), pl.program_id(1)
    windows, tiles = pl.num_programs(0), pl.num_programs(1)
    tb, pairs, slots = QB5_TOKEN_BLOCK, QB5_HEAD_BLOCK // 2, obuf_ref.shape[0]
    step = window * tiles + tile
    last = windows * tiles * pairs - 1
    tail_start = head_dim - _LANE
    out_words = out_hbm.bitcast(jnp.uint32)  # [tokens, heads/2, head_dim]
    token_start = pl.multiple_of(tile * tb, tb)

    @pl.when(window == 0)
    def prepare_rows():
        def quantize_fragment(fragment, _):
            rows = pl.ds(fragment * QB5_QUANT_TOKEN_BLOCK, QB5_QUANT_TOKEN_BLOCK)
            cached = pl.ds(
                token_start + fragment * QB5_QUANT_TOKEN_BLOCK, QB5_QUANT_TOKEN_BLOCK
            )
            encoded, scale = _quantize_rows(qr_ref[rows, :])
            encoded_ref[cached, :] = encoded
            row_scale_ref[cached, :] = scale

        jax.lax.fori_loop(0, tb // QB5_QUANT_TOKEN_BLOCK, quantize_fragment, None)
        cos_lanes, sin_lanes = _coefficients(cos_ref[...], sin_ref[...])
        coeff_ref[0, pl.ds(token_start, tb), :] = cos_lanes
        coeff_ref[1, pl.ds(token_start, tb), :] = sin_lanes

    def copy(g):
        """Output DMA of global pair g (dst address only matters for start)."""
        t, q = g // pairs, g % pairs
        rows = pl.ds(pl.multiple_of((t % tiles) * tb, tb), tb)
        return pltpu.make_async_copy(
            obuf_ref.at[g % slots],
            out_words.at[rows, (t // tiles) * pairs + q, :],
            sems.at[g % slots],
        )

    def matmul(pair):
        """FP32 products of this step's head pair into slot pair % 2."""
        tokens = pl.ds(token_start, tb)
        for half in range(2):
            columns = pl.ds(
                pl.multiple_of((2 * pair + half) * head_dim, head_dim), head_dim
            )
            product_ref[pair % 2, half] = jnp.dot(
                encoded_ref[tokens, :],
                weight_ref[:, columns],
                preferred_element_type=jnp.float32,
            )

    def post(g):
        """Scale, round, rotate and pack global pair g, then start its DMA."""
        t, q = g // pairs, g % pairs
        rows0 = pl.multiple_of((t % tiles) * tb, tb)
        head0 = ((t // tiles) * pairs + q) * 2
        slot = g % slots
        for f in range(tb // _FRAGMENT):
            local = pl.ds(f * _FRAGMENT, _FRAGMENT)
            tokens = pl.ds(rows0 + f * _FRAGMENT, _FRAGMENT)
            halves = []
            for half in range(2):
                columns = pl.ds(
                    pl.multiple_of((head0 + half) * head_dim, head_dim), head_dim
                )
                projected = (
                    product_ref[q % 2, half, local, :] * row_scale_ref[tokens, :]
                )
                projected = (projected * scale_ref[columns][None, :]).astype(
                    jnp.bfloat16
                )
                tail = projected[:, -_LANE:].astype(jnp.float32)
                partner = _lane_gather(tail, jnp.arange(_LANE) ^ 1)
                rotated = (
                    tail * coeff_ref[0, tokens, :] + partner * coeff_ref[1, tokens, :]
                ).astype(jnp.bfloat16)
                halves.append(
                    (
                        projected[:, :-_LANE].astype(jnp.float32),
                        rotated.astype(jnp.float32),
                    )
                )
            (low_plain, low_tail), (high_plain, high_tail) = halves
            # Values are already BF16-rounded, so the FP32->BF16 pack is exact.
            obuf_ref[slot, local, pl.ds(0, tail_start)] = pltpu.pack_elementwise(
                [low_plain, high_plain], packed_dtype=jnp.bfloat16
            )
            obuf_ref[slot, local, pl.ds(tail_start, _LANE)] = pltpu.pack_elementwise(
                [low_tail, high_tail], packed_dtype=jnp.bfloat16
            )
        copy(g).start()

    first = step * pairs  # global index of this step's pair 0

    # Slots used this step held pairs g-slots for g in [first-1, first+pairs-2];
    # those DMAs started at least one step ago.
    @pl.when(step >= 2)
    def recycle():
        for k in range(pairs):
            g = first - 1 - slots + k

            @pl.when(g >= 0)
            def _(g=g):
                copy(g).wait()

    matmul(0)

    @pl.when(step > 0)
    def previous_last():
        post(first - 1)

    def body(pair, _):
        matmul(pair)
        post(first + pair - 1)

    jax.lax.fori_loop(1, pairs, body, None)

    @pl.when(step == windows * tiles - 1)
    def finish():
        copy(last - slots).wait()
        post(last)
        for back in range(slots):
            copy(last - back).wait()


def fused_qb_rope_overlap(
    qr: jax.Array,
    weight: jax.Array,
    weight_scale: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    head_dim: int = 512,
    interpret: bool = False,
) -> jax.Array:
    """Return BF16 Q[tokens, heads, head_dim]; numerics match the serial kernel."""
    _check_projection(qr, weight, weight_scale)
    if head_dim <= 0 or weight.shape[1] % head_dim or head_dim % _LANE:
        raise ValueError("Q-B output width must be a multiple of a 128-lane head")
    _check_rope(qr, cos, sin, head_dim)
    heads = weight.shape[1] // head_dim
    if heads % QB5_HEAD_BLOCK:
        raise ValueError(f"Pad head count to a multiple of {QB5_HEAD_BLOCK}")
    tokens, rank = qr.shape
    tb, width = QB5_TOKEN_BLOCK, QB5_HEAD_BLOCK * head_dim
    if tokens % tb:
        raise ValueError(f"Pad token rows to a multiple of {tb}")
    steps = heads // QB5_HEAD_BLOCK * (tokens // tb)
    if steps < 2:
        raise ValueError("Overlapped Q-B needs at least two grid steps")
    slots = QB5_HEAD_BLOCK  # two steps of head pairs
    last_tile = tokens // tb - 1

    # Row inputs are only read in the first window; keep their block index
    # fixed afterwards so the pipeline does not fetch them again.
    def rows(h, t):
        return (jnp.where(h == 0, t, last_tile), 0)

    once = pl.Buffered(buffer_count=1)
    return pl.pallas_call(
        functools.partial(_qb_rope_overlap_kernel, head_dim=head_dim),
        grid=(heads // QB5_HEAD_BLOCK, tokens // tb),
        in_specs=(
            pl.BlockSpec((tb, rank), rows),
            pl.BlockSpec((rank, width), lambda h, t: (0, h)),
            # The previous step's last pair may belong to the previous window.
            pl.BlockSpec(weight_scale.reshape(-1).shape, lambda h, t: (0,), once),
            pl.BlockSpec((tb, cos.shape[1]), rows),
            pl.BlockSpec((tb, sin.shape[1]), rows),
        ),
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=(
            pltpu.VMEM((tokens, rank), jnp.float8_e4m3fn),
            pltpu.VMEM((tokens, 1), jnp.float32),
            pltpu.VMEM((2, tokens, _LANE), jnp.float32),
            pltpu.VMEM((2, 2, tb, head_dim), jnp.float32),
            pltpu.VMEM((slots, tb, head_dim), jnp.uint32),
            pltpu.SemaphoreType.DMA((slots,)),
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, heads, head_dim), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=QB5_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="qb_rope_overlap",
    )(qr, weight, weight_scale.reshape(-1), cos, sin)
