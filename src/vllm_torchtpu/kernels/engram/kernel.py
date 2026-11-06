# SPDX-License-Identifier: Apache-2.0
"""Packed Engram with BN2560 WKV, FP32 gate and BF16 residual.

Ported from pallas-samples' packed-dequantize-quantize D1 experiment, with
the packed input and Torch boundary contract used by PR #2668.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.engram.tuned_block_sizes import TUNED_BLOCK_SIZES

TOKENS = 288
RAW_VALUES = 6144
RAW_WORDS = RAW_VALUES // 4
SCALE_WORDS = RAW_VALUES // 32 // 4
K_HEADS = 4
HIDDEN_DIM = 5120
WKV_N = (K_HEADS + 1) * HIDDEN_DIM
TUNING = TUNED_BLOCK_SIZES[(7, TOKENS, RAW_VALUES, WKV_N, K_HEADS, HIDDEN_DIM)]
BLOCK_TOKENS = TUNING.block_tokens
GROUP_TOKENS = TUNING.group_tokens
CHUNK_WORDS = TUNING.chunk_words
CHUNK_VALUES = CHUNK_WORDS * 4
K_CHUNKS = RAW_WORDS // CHUNK_WORDS
DEQUANT_WORDS = TUNING.dequant_words
DEQUANT_CHUNKS = RAW_WORDS // DEQUANT_WORDS
SCALE_WORDS_PER_DEQUANT = DEQUANT_WORDS // 32
WKV_BN = TUNING.wkv_bn
WKV_STEPS = WKV_N // WKV_BN
REDUCE_ROWS = TUNING.reduce_rows
REDUCE_COLS = TUNING.reduce_cols
VALUE_COLS = TUNING.value_cols
GATE_ROWS = TUNING.gate_rows
FP8 = jnp.float8_e4m3fn


def unpack_plane(words: jax.Array, index: int) -> jax.Array:
    byte_u32 = pltpu.unpack_elementwise(
        words, index=index, packed_dtype=jnp.uint8, unpacked_dtype=jnp.uint32
    )
    byte_u8 = byte_u32.astype(jnp.uint8)
    return jax.lax.bitcast_convert_type(byte_u8, FP8).astype(jnp.float32)


def natural_q(qs: list[jax.Array]) -> jax.Array:
    bs = [jax.lax.bitcast_convert_type(q, jnp.uint8).astype(jnp.uint32) for q in qs]
    packed = pltpu.pack_elementwise(bs, packed_dtype=jnp.uint8)
    # Each input word holds four adjacent natural-K values. Move that
    # word axis to second-minor for bitcast, then restore token-major K.
    return pltpu.bitcast(packed.T, FP8).T


@partial(jax.jit, static_argnames=("eps", "clamp_value"))
def fused_engram(
    packed: jax.Array,
    hidden: jax.Array,
    weight: jax.Array,
    weight_scale: jax.Array,
    query_weight: jax.Array,
    key_weight: jax.Array,
    token_mask: jax.Array,
    eps: float,
    clamp_value: float,
) -> jax.Array:
    """Consume PR #2668 inputs and return BF16 hidden plus gated residual.

    token_mask is bool[288], or preconverted int32 0/1[288]. eps and
    clamp_value are Python scalars specialized by JIT; neither needs DMA.
    """
    # The upload contains all raw words followed by all scale words. These
    # word-domain views match PR #2668 and preserve the existing VMEM logic.
    raw_end = TOKENS * RAW_WORDS
    raw = packed[:raw_end].reshape(TOKENS, RAW_WORDS)
    scale = packed[raw_end:].reshape(TOKENS, SCALE_WORDS)
    # This conversion is outside Pallas. Passing preconverted int32 removes
    # the conversion; reshape only exposes the existing scalar-per-token axis.
    token_mask = token_mask.astype(jnp.int32).reshape(TOKENS, 1)

    def kernel(
        raw_hbm,
        scale_hbm,
        hidden_hbm,
        wkv_scale_hbm,
        q_weight_hbm,
        k_weight_hbm,
        wkv_hbm,
        token_mask_hbm,
        output_hbm,
        raw_vmem,
        scale_vmem,
        q_vmem,
        sx_vmem,
        x_temp,
        power_temp,
        b_vmem,
        c_even,
        c_odd,
        lane_partial,
        statistics_vmem,
        hidden_vmem,
        wkv_scale_vmem,
        q_weight_vmem,
        k_weight_vmem,
        gate_vmem,
        hidden_staging,
        q_weight_cache,
        k_weight_cache,
        token_mask_vmem,
        raw_sems,
        scale_sem,
        b_sems,
        hidden_sems,
        wkv_scale_sem,
        q_weight_sem,
        k_weight_sem,
        output_sems,
        token_mask_sem,
    ):
        def raw_copy(block, slot):
            return pltpu.make_async_copy(
                raw_hbm.at[pl.ds(block * BLOCK_TOKENS, BLOCK_TOKENS), :],
                raw_vmem.at[slot],
                raw_sems.at[slot],
            )

        def hidden_copy(block):
            return pltpu.make_async_copy(
                hidden_hbm.at[pl.ds(block * BLOCK_TOKENS, BLOCK_TOKENS), :, :],
                hidden_staging.at[block & 1],
                hidden_sems.at[block & 1],
            )

        def output_copy(block, slot):
            return pltpu.make_async_copy(
                hidden_staging.at[slot],
                output_hbm.at[pl.ds(block * BLOCK_TOKENS, BLOCK_TOKENS), :, :],
                output_sems.at[slot],
            )

        def weight_copies(part):
            return (
                pltpu.make_async_copy(
                    wkv_scale_hbm.at[pl.ds(part * 3200, 3200)],
                    wkv_scale_vmem.at[pl.ds(part * 3200, 3200)],
                    wkv_scale_sem,
                ),
                pltpu.make_async_copy(
                    q_weight_hbm.at[:, pl.ds(part * 640, 640)],
                    q_weight_vmem.at[:, pl.ds(part * 640, 640)],
                    q_weight_sem,
                ),
                pltpu.make_async_copy(
                    k_weight_hbm.at[:, pl.ds(part * 640, 640)],
                    k_weight_vmem.at[:, pl.ds(part * 640, 640)],
                    k_weight_sem,
                ),
            )

        def load_b(index, slot):
            return pltpu.make_async_copy(
                wkv_hbm.at[:, pl.ds(index * WKV_BN, WKV_BN)],
                b_vmem.at[slot],
                b_sems.at[slot],
            )

        token_mask_copy = pltpu.make_async_copy(
            token_mask_hbm, token_mask_vmem, token_mask_sem
        )
        scale_copy = pltpu.make_async_copy(scale_hbm, scale_vmem, scale_sem)
        scale_copy.start()
        raw_copy(0, 0).start()
        hidden_copy(0).start()
        scale_copy.wait()
        word_scale_shift = ((jnp.arange(DEQUANT_WORDS, dtype=jnp.uint32) // 8) % 4) * 8
        word_scale_index = (jnp.arange(DEQUANT_WORDS, dtype=jnp.uint32) // 32).astype(
            jnp.int32
        )

        @pl.loop(0, TOKENS // BLOCK_TOKENS)
        def block_loop(block):
            slot = block & 1

            @pl.when(block + 1 < TOKENS // BLOCK_TOKENS)
            def prefetch():
                raw_copy(block + 1, slot ^ 1).start()
                hidden_copy(block + 1).start()
                for copy in weight_copies(block):
                    copy.start()

            # Start the first WKV weight copy before the last raw DMA wait.
            @pl.when(block == TOKENS // BLOCK_TOKENS - 1)
            def prefetch_wkv():
                load_b(0, 0).start()
                token_mask_copy.start()

            raw_copy(block, slot).wait()
            q_block = q_vmem.at[pl.ds(block * BLOCK_TOKENS, BLOCK_TOKENS), :]
            sx_block = sx_vmem.at[pl.ds(block * BLOCK_TOKENS, BLOCK_TOKENS), :]

            for group in range(BLOCK_TOKENS // GROUP_TOKENS):
                group_scale = scale_vmem[
                    pl.ds(block * BLOCK_TOKENS + group * GROUP_TOKENS, GROUP_TOKENS), :
                ]
                # Python-static chunks select from the full-width scale load.
                # The power cache keeps gather outside the runtime D loop.
                for chunk in range(DEQUANT_CHUNKS):
                    scale_indices = jnp.broadcast_to(
                        chunk * SCALE_WORDS_PER_DEQUANT + word_scale_index[None, :],
                        (GROUP_TOKENS, DEQUANT_WORDS),
                    )
                    packed_scale = jnp.take_along_axis(
                        group_scale, scale_indices, axis=-1, mode="promise_in_bounds"
                    ).astype(jnp.uint32)
                    byte = (packed_scale >> word_scale_shift[None, :]) & 255
                    power_temp[chunk, :, :] = jax.lax.bitcast_convert_type(
                        byte << 23, jnp.float32
                    )

                @pl.loop(
                    0,
                    DEQUANT_CHUNKS,
                    init_carry=jnp.zeros((GROUP_TOKENS, DEQUANT_WORDS), jnp.float32),
                    unroll=False,
                )
                def dequant_loop(chunk, lane_amax, *, group=group):
                    words = raw_vmem[
                        slot,
                        pl.ds(group * GROUP_TOKENS, GROUP_TOKENS),
                        pl.ds(chunk * DEQUANT_WORDS, DEQUANT_WORDS),
                    ]
                    power = power_temp[chunk, :, :]
                    valid = power != jnp.float32(jnp.inf)
                    for plane in range(4):
                        values = unpack_plane(words, plane)
                        x = jnp.where(valid, values * power, jnp.float32(0))
                        x_temp[
                            plane,
                            pl.ds(group * GROUP_TOKENS, GROUP_TOKENS),
                            pl.ds(chunk * DEQUANT_WORDS, DEQUANT_WORDS),
                        ] = x
                        lane_amax = jnp.maximum(lane_amax, jnp.abs(x))
                    return lane_amax

                # Each lane accumulates max across all four chunks; only one
                # cross-lane reduction per eight-token group follows the loop.
                token_amax = jnp.max(dequant_loop, axis=-1, keepdims=True)
                sx = jnp.where(
                    token_amax == 0, jnp.float32(1), token_amax / jnp.float32(448)
                )
                sx_block[group * GROUP_TOKENS : (group + 1) * GROUP_TOKENS, :] = sx

            # All four groups have stored FP32 sx and FP32 dequantized data.
            # Read sx without changing precision, then use the same 1/sx math.
            block_sx = sx_block[...]
            inv_sx = jnp.float32(1) / block_sx

            @pl.loop(0, K_CHUNKS, unroll=TUNING.quant_unroll)
            def quant_loop(chunk):
                qs = []
                for plane in range(4):
                    x = x_temp[plane, :, pl.ds(chunk * CHUNK_WORDS, CHUNK_WORDS)]
                    qs.append((x * inv_sx).astype(FP8))
                q = natural_q(qs)
                q_block[:, pl.ds(chunk * CHUNK_VALUES, CHUNK_VALUES)] = q

            # All hidden blocks and all eight weight parts finish before WKV.
            # Each semaphore is waited before it is reused by a later part.
            hidden_copy(block).wait()

            # Reorder bounded BF16 values; Ref transpose is not used. The
            # destination's token axis is the aligned second-minor dimension.
            for row_half in range(BLOCK_TOKENS // REDUCE_ROWS):
                staging_rows = pl.ds(row_half * REDUCE_ROWS, REDUCE_ROWS)
                hidden_rows = pl.ds(
                    block * BLOCK_TOKENS + row_half * REDUCE_ROWS, REDUCE_ROWS
                )
                for column in range(HIDDEN_DIM // VALUE_COLS):
                    cols = pl.ds(column * VALUE_COLS, VALUE_COLS)
                    token_major = hidden_staging[slot, staging_rows, :, cols]
                    hidden_vmem[:, hidden_rows, cols] = jnp.transpose(
                        token_major, (1, 0, 2)
                    )

            @pl.when(block < TOKENS // BLOCK_TOKENS - 1)
            def wait_current_weights():
                for copy in weight_copies(block):
                    copy.wait()
                # BF16 input loads cover all four heads. Decode each part
                # once; the cached head is a leading dimension of an F32 Ref.
                cols = pl.ds(block * 640, 640)
                qw_all = q_weight_vmem[:, cols].astype(jnp.float32)
                kw_all = k_weight_vmem[:, cols].astype(jnp.float32)
                for head in range(K_HEADS):
                    q_weight_cache[head, :, cols] = qw_all[head, :][None, :]
                    k_weight_cache[head, :, cols] = kw_all[head, :][None, :]

        def consume_key(c_ref, head, half):
            # Each lane accumulates twenty chunks per half, changing the
            # reduction grouping to 128 channels. sx stays outside the sums.
            for row_index in range(TOKENS // REDUCE_ROWS):
                rows = pl.ds(row_index * REDUCE_ROWS, REDUCE_ROWS)
                dot_lanes = jnp.zeros((REDUCE_ROWS, REDUCE_COLS), jnp.float32)
                hidden_lanes = jnp.zeros((REDUCE_ROWS, REDUCE_COLS), jnp.float32)
                key_lanes = jnp.zeros((REDUCE_ROWS, REDUCE_COLS), jnp.float32)
                for column in range(WKV_BN // REDUCE_COLS):
                    cols = pl.ds(column * REDUCE_COLS, REDUCE_COLS)
                    hidden_cols = pl.ds(
                        half * WKV_BN + column * REDUCE_COLS, REDUCE_COLS
                    )
                    scale_cols = pl.ds(
                        (2 * head + half) * WKV_BN + column * REDUCE_COLS, REDUCE_COLS
                    )
                    key = c_ref[rows, cols] * wkv_scale_vmem[scale_cols][None, :]
                    h = hidden_vmem[head, rows, hidden_cols].astype(jnp.float32)
                    qw = q_weight_cache[head, :, hidden_cols]
                    kw = k_weight_cache[head, :, hidden_cols]
                    dot_lanes = dot_lanes + ((h * qw) * kw) * key
                    hidden_lanes = hidden_lanes + h * h
                    key_lanes = key_lanes + key * key

                lanes = (dot_lanes, hidden_lanes, key_lanes)
                if half == 0:
                    for statistic in range(3):
                        lane_partial[statistic, rows, :] = lanes[statistic]
                else:
                    for statistic in range(3):
                        total = lane_partial[statistic, rows, :] + lanes[statistic]
                        statistics_vmem[statistic, head, rows, :] = jnp.sum(
                            total, axis=-1, keepdims=True
                        )

        # B0 was started by the last raw block. Consume it once, and fill
        # the other B slot before the first matrix product.
        load_b(0, 0).wait()
        load_b(1, 1).start()
        c_even[...] = jax.lax.dot(
            q_vmem[...], b_vmem[0], preferred_element_type=jnp.float32
        )

        # One runtime head graph preserves the two-phase dot/previous-C
        # schedule. Head-major Refs select a leading dimension directly.
        @pl.loop(0, K_HEADS, unroll=False)
        def wkv_head_loop(head):
            odd_n = 2 * head + 1
            load_b(odd_n, 1).wait()
            load_b(odd_n + 1, 0).start()
            c_odd[...] = jax.lax.dot(
                q_vmem[...], b_vmem[1], preferred_element_type=jnp.float32
            )

            # Current dot is N=2h+1. Previous C_even is K head h, half 0.
            consume_key(c_even, head, 0)

            even_n = odd_n + 1
            load_b(even_n, 0).wait()
            load_b(even_n + 1, 1).start()
            c_even[...] = jax.lax.dot(
                q_vmem[...], b_vmem[0], preferred_element_type=jnp.float32
            )

            # Current dot is N=2h+2. Previous C_odd is K head h, half 1.
            consume_key(c_odd, head, 1)

        # The previous eight K tiles have finished all three statistics.
        # Work on 32 token scalars at a time; no 5120-wide normalization.
        token_mask_copy.wait()
        for row_index in range(TOKENS // GATE_ROWS):
            rows = pl.ds(row_index * GATE_ROWS, GATE_ROWS)
            sx = sx_vmem[rows, :]
            mask = token_mask_vmem[rows, :].astype(jnp.float32)
            for head in range(K_HEADS):
                dot_sum = statistics_vmem[0, head, rows, :] * sx
                hidden_sq = statistics_vmem[1, head, rows, :]
                key_sq = (statistics_vmem[2, head, rows, :] * sx) * sx
                score = dot_sum * jax.lax.rsqrt(
                    hidden_sq / jnp.float32(HIDDEN_DIM) + jnp.float32(eps)
                )
                score = score * (
                    jax.lax.rsqrt(key_sq / jnp.float32(HIDDEN_DIM) + jnp.float32(eps))
                    * jnp.float32(HIDDEN_DIM**-0.5)
                )
                magnitude = jnp.sqrt(
                    jnp.maximum(jnp.abs(score), jnp.float32(clamp_value))
                )
                signed_root = jnp.where(score < 0, -magnitude, magnitude)
                gate_vmem[head, rows, :] = jax.nn.sigmoid(signed_root) * mask

        def consume_value(c_ref, half):
            # Each head is loaded and stored as an aligned BF16 [16,128]
            # slice. The shared V is computed once for all four heads.
            for row_index in range(TOKENS // REDUCE_ROWS):
                rows = pl.ds(row_index * REDUCE_ROWS, REDUCE_ROWS)
                sx = sx_vmem[rows, :]
                for column in range(WKV_BN // VALUE_COLS):
                    cols = pl.ds(column * VALUE_COLS, VALUE_COLS)
                    hidden_cols = pl.ds(half * WKV_BN + column * VALUE_COLS, VALUE_COLS)
                    scale_cols = pl.ds(
                        (K_HEADS * 2 + half) * WKV_BN + column * VALUE_COLS, VALUE_COLS
                    )
                    value = (c_ref[rows, cols] * sx) * wkv_scale_vmem[scale_cols][
                        None, :
                    ]
                    value = value.astype(jnp.bfloat16).astype(jnp.float32)
                    for head in range(K_HEADS):
                        gate = gate_vmem[head, rows, :]
                        hidden = hidden_vmem[head, rows, hidden_cols].astype(
                            jnp.float32
                        )
                        hidden_vmem[head, rows, hidden_cols] = (
                            hidden + gate * value
                        ).astype(jnp.bfloat16)

        # N8 is the first V half and head3's second K half (N7) statistics
        # have completed. Expose V0 consumption alongside the final N9 dot.
        # No B10 transfer is started; V1 is then drained from its own C Ref.
        load_b(WKV_STEPS - 1, 1).wait()
        c_odd[...] = jax.lax.dot(
            q_vmem[...], b_vmem[1], preferred_element_type=jnp.float32
        )
        consume_value(c_even, 0)
        consume_value(c_odd, 1)

        # All incoming hidden DMA requests finished in the raw loop. Reuse
        # its two staging slots for output; wait before overwriting a slot.
        @pl.loop(0, TOKENS // BLOCK_TOKENS, unroll=False)
        def output_loop(block):
            slot = block & 1

            @pl.when(block >= 2)
            def wait_previous_output():
                output_copy(block - 2, slot).wait()

            for row_half in range(BLOCK_TOKENS // REDUCE_ROWS):
                staging_rows = pl.ds(row_half * REDUCE_ROWS, REDUCE_ROWS)
                hidden_rows = pl.ds(
                    block * BLOCK_TOKENS + row_half * REDUCE_ROWS, REDUCE_ROWS
                )
                for column in range(HIDDEN_DIM // VALUE_COLS):
                    cols = pl.ds(column * VALUE_COLS, VALUE_COLS)
                    head_major = hidden_vmem[:, hidden_rows, cols]
                    hidden_staging[slot, staging_rows, :, cols] = jnp.transpose(
                        head_major, (1, 0, 2)
                    )
            output_copy(block, slot).start()

        output_copy(TOKENS // BLOCK_TOKENS - 2, 1).wait()
        output_copy(TOKENS // BLOCK_TOKENS - 1, 0).wait()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((TOKENS, K_HEADS, HIDDEN_DIM), jnp.bfloat16),
        in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),) * 8,
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=(
            pltpu.VMEM((2, BLOCK_TOKENS, RAW_WORDS), jnp.int32),
            pltpu.VMEM((TOKENS, SCALE_WORDS), jnp.int32),
            pltpu.VMEM((TOKENS, RAW_VALUES), FP8),
            pltpu.VMEM((TOKENS, 1), jnp.float32),
            pltpu.VMEM((4, BLOCK_TOKENS, RAW_WORDS), jnp.float32),
            pltpu.VMEM((DEQUANT_CHUNKS, GROUP_TOKENS, DEQUANT_WORDS), jnp.float32),
            pltpu.VMEM((2, RAW_VALUES, WKV_BN), FP8),
            pltpu.VMEM((TOKENS, WKV_BN), jnp.float32),
            pltpu.VMEM((TOKENS, WKV_BN), jnp.float32),
            pltpu.VMEM((3, TOKENS, REDUCE_COLS), jnp.float32),
            pltpu.VMEM((3, K_HEADS, TOKENS, 1), jnp.float32),
            pltpu.VMEM((K_HEADS, TOKENS, HIDDEN_DIM), jnp.bfloat16),
            pltpu.VMEM((WKV_N,), jnp.float32),
            pltpu.VMEM((K_HEADS, HIDDEN_DIM), jnp.bfloat16),
            pltpu.VMEM((K_HEADS, HIDDEN_DIM), jnp.bfloat16),
            pltpu.VMEM((K_HEADS, TOKENS, 1), jnp.float32),
            pltpu.VMEM((2, BLOCK_TOKENS, K_HEADS, HIDDEN_DIM), jnp.bfloat16),
            pltpu.VMEM((K_HEADS, 1, HIDDEN_DIM), jnp.float32),
            pltpu.VMEM((K_HEADS, 1, HIDDEN_DIM), jnp.float32),
            pltpu.VMEM((TOKENS, 1), jnp.int32),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA,
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=TUNING.vmem_limit_bytes),
        name="engram_offload_dq_wkv_bn2560",
    )(
        *(
            pltpu.with_memory_space_constraint(v, pltpu.HBM)
            for v in (
                raw,
                scale,
                hidden,
                weight_scale,
                query_weight,
                key_weight,
                weight,
                token_mask,
            )
        )
    )
