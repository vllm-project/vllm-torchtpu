# SPDX-License-Identifier: Apache-2.0
"""Packed Engram dequantization, FP8 WKV and gating in one Pallas call.

Adapted from vllm-project/vllm-torchtpu PR #1853, kernel.py at bbefef5720ee.
This TP1 variant reads the existing raw-then-scale int32 staging allocation
without unpacked HBM intermediates, and covers the 288-token bucket in one
weight pass. Host lookup, H2D and mHC remain outside the call.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.engram.tuned_block_sizes import (
    VECTOR_TILE_M,
    get_block_sizes,
)
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor


def unpack_words(words: jax.Array) -> jax.Array:
    """Little-endian int32 words [..., W] to uint8 [..., 4*W]."""
    shifts = jnp.arange(4, dtype=jnp.int32) * 8
    return (
        ((words[..., None] >> shifts) & 255)
        .astype(jnp.uint8)
        .reshape(*words.shape[:-1], words.shape[-1] * 4)
    )


def dequantize_rows(raw: jax.Array, scales: jax.Array) -> jax.Array:
    """Packed FP8 rows and E8M0 scales to BF16; scale 255 produces +0."""
    values = jax.lax.bitcast_convert_type(unpack_words(raw), jnp.float8_e4m3fn)
    scale_bytes = unpack_words(scales)
    block = values.shape[-1] // scale_bytes.shape[-1]
    values = values.reshape(*scale_bytes.shape, block).astype(jnp.float32)
    power = jax.lax.bitcast_convert_type(
        scale_bytes.astype(jnp.int32) << 23, jnp.float32
    )
    values = jnp.where(scale_bytes[..., None] == 255, 0.0, values * power[..., None])
    return values.reshape(*raw.shape[:-1], raw.shape[-1] * 4).astype(jnp.bfloat16)


def gated_residual(
    hidden: jax.Array,
    projection: jax.Array,
    query_weight: jax.Array,
    key_weight: jax.Array,
    token_mask: jax.Array,
    eps: float,
    clamp_value: float,
) -> jax.Array:
    """BF16 [T,S,D] hidden plus gated BF16 [T,(S+1)*D] keys and values."""
    streams, dim = hidden.shape[-2:]
    keys = projection[:, : streams * dim].reshape(-1, streams, dim).astype(jnp.float32)
    values = projection[:, streams * dim :].astype(jnp.float32)
    h = hidden.astype(jnp.float32)
    dot = jnp.sum(h * query_weight * key_weight * keys, axis=-1)
    dot *= jax.lax.rsqrt(jnp.mean(h * h, axis=-1) + eps)
    dot *= jax.lax.rsqrt(jnp.mean(keys * keys, axis=-1) + eps) * dim**-0.5
    magnitude = jnp.sqrt(jnp.maximum(jnp.abs(dot), clamp_value))
    gate = jax.nn.sigmoid(jnp.where(dot < 0, -magnitude, magnitude))
    gate *= token_mask[:, None]
    return (h + gate[..., None] * values[:, None, :]).astype(hidden.dtype)


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
    *,
    embedding_block_size: int = 32,
    interpret: bool = False,
) -> jax.Array:
    """TP1 packed host rows through WKV and gated residual, with no HBM joins.

    Args:
        packed: Flat int32 [T*(K/4 + K/(4*B))], all raw bytes then all scales.
        hidden: BF16 [T,S,D]. Tokens must divide the selected token tile.
        weight: FP8 [K,(S+1)*D], in the processed linear layer's (k,n) layout.
        weight_scale: FP32 [(S+1)*D], per-output-channel scales.
        query_weight: BF16 [S,D].
        key_weight: BF16 [S,D].
        token_mask: Boolean [T], with the original Engram masking semantics.
        eps: RMS epsilon.
        clamp_value: Minimum squared gate magnitude.
        embedding_block_size: Values sharing one embedding E8M0 scale (B).
        interpret: Execute the kernel through the CPU interpreter.

    Returns:
        BF16 [T,S,D], including the original pre/post-projection BF16 roundings.
    """
    tokens, streams, dim = hidden.shape
    in_features = weight.shape[0]
    out_features = (streams + 1) * dim
    assert in_features % (4 * embedding_block_size) == 0
    raw_words = in_features // 4
    scale_words = in_features // embedding_block_size // 4
    assert packed.shape == (tokens * (raw_words + scale_words),), packed.shape
    assert packed.dtype == jnp.int32
    assert weight.shape == (in_features, out_features), weight.shape
    assert weight_scale.shape == (out_features,), weight_scale.shape
    assert weight_scale.dtype == jnp.float32
    assert query_weight.shape == key_weight.shape == (streams, dim)
    assert token_mask.shape == (tokens,), token_mask.shape
    assert hidden.dtype == jnp.bfloat16, hidden.dtype
    assert weight.dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2), weight.dtype
    tile_m, tile_k = get_block_sizes(tokens, in_features, streams, dim)
    vector_m = min(VECTOR_TILE_M, tile_m)
    assert tokens % tile_m == 0 and in_features % tile_k == 0
    assert tile_m % vector_m == 0
    steps = in_features // tile_k
    chunks = tile_m // vector_m
    buffer_count = min(2, chunks)

    def kernel(
        packed_ref,
        h_ref,
        w_ref,
        ws_ref,
        q_ref,
        k_ref,
        mask_ref,
        out_ref,
        x_scratch,
        xs_scratch,
        acc_scratch,
        projection_scratch,
        hidden_buffers,
        output_buffers,
        hidden_sems,
        output_sems,
    ):
        output_block = pl.program_id(1)
        step = pl.program_id(2)
        token_start = pl.program_id(0) * tile_m

        def hidden_copy(chunk):
            rows = pl.ds(token_start + chunk * vector_m, vector_m)
            bank = chunk % buffer_count
            return pltpu.make_async_copy(
                h_ref.at[rows, :, :], hidden_buffers.at[bank], hidden_sems.at[bank]
            )

        def output_copy(chunk):
            rows = pl.ds(token_start + chunk * vector_m, vector_m)
            bank = chunk % buffer_count
            return pltpu.make_async_copy(
                output_buffers.at[bank], out_ref.at[rows, :, :], output_sems.at[bank]
            )

        @pl.when((output_block == 0) & (step == 0))
        def prepare():
            hidden_copy(0).start()

            def prepare_rows(index, unused):
                first = token_start + index * vector_m
                raw = packed_ref[pl.ds(first * raw_words, vector_m * raw_words)]
                scales = packed_ref[
                    pl.ds(
                        tokens * raw_words + first * scale_words, vector_m * scale_words
                    )
                ]
                # Put packed bytes on the sublane axis before widening them;
                # appending a trailing size-4 axis is not a legal TPU layout.
                raw_bytes = pltpu.bitcast(
                    raw.reshape(vector_m, raw_words).T, jnp.uint8
                ).T
                scale_bytes = pltpu.bitcast(
                    scales.reshape(vector_m, scale_words).T, jnp.uint8
                ).T
                values = pltpu.bitcast(raw_bytes, jnp.float8_e4m3fn).astype(jnp.float32)
                block = embedding_block_size
                powers = jax.lax.bitcast_convert_type(
                    scale_bytes.astype(jnp.int32) << 23, jnp.float32
                )
                power = jnp.repeat(powers, block, axis=1)
                invalid = jnp.repeat(scale_bytes == 255, block, axis=1)
                embedded = jnp.where(invalid, 0.0, values * power).astype(jnp.bfloat16)
                quantized, activation_scale = quantize_tensor(embedded, weight.dtype)
                tile = pl.ds(index * vector_m, vector_m)
                x_scratch[tile, :] = quantized
                xs_scratch[tile, :] = activation_scale
                return unused

            jax.lax.fori_loop(0, chunks, prepare_rows, None)

        @pl.when(step == 0)
        def reset_accumulator():
            acc_scratch[...] = jnp.zeros((tile_m, dim), jnp.float32)

        x = x_scratch[:, pl.ds(step * tile_k, tile_k)]
        acc_scratch[...] += jnp.dot(x, w_ref[...], preferred_element_type=jnp.float32)

        @pl.when(step == steps - 1)
        def finish():
            def finish_rows(index, unused):
                tile = pl.ds(index * vector_m, vector_m)
                projection = (
                    acc_scratch[tile, :] * xs_scratch[tile, :] * ws_ref[...]
                ).astype(hidden.dtype)
                group = jnp.where(output_block == 0, streams, output_block - 1)
                projection_scratch[tile, pl.ds(group * dim, dim)] = projection
                return unused

            jax.lax.fori_loop(0, chunks, finish_rows, None)

            @pl.when(output_block == streams)
            def apply_gates():
                def gate_rows(chunk, unused):
                    bank = chunk % buffer_count
                    hidden_copy(chunk).wait()

                    @pl.when(chunk + 1 < chunks)
                    def prefetch():
                        hidden_copy(chunk + 1).start()

                    @pl.when(chunk >= buffer_count)
                    def release_output_bank():
                        output_copy(chunk - buffer_count).wait()

                    rows = pl.ds(chunk * vector_m, vector_m)
                    output_buffers[bank] = gated_residual(
                        hidden_buffers[bank],
                        projection_scratch[rows, :],
                        q_ref[...],
                        k_ref[...],
                        mask_ref[rows, 0],
                        eps,
                        clamp_value,
                    )
                    output_copy(chunk).start()
                    return unused

                jax.lax.fori_loop(0, chunks, gate_rows, None)
                for chunk in range(max(0, chunks - buffer_count), chunks):
                    output_copy(chunk).wait()

    def projection_index(j):
        return jnp.where(j == 0, streams, j - 1)

    return pl.pallas_call(
        kernel,
        grid=(tokens // tile_m, streams + 1, steps),
        in_specs=[
            pl.BlockSpec(packed.shape, lambda i, j, k: (0,)),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec((tile_k, dim), lambda i, j, k: (k, projection_index(j))),
            pl.BlockSpec((dim,), lambda i, j, k: (projection_index(j),)),
            pl.BlockSpec((streams, dim), lambda i, j, k: (0, 0)),
            pl.BlockSpec((streams, dim), lambda i, j, k: (0, 0)),
            pl.BlockSpec((tile_m, 1), lambda i, j, k: (i, 0)),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        out_shape=jax.ShapeDtypeStruct(hidden.shape, hidden.dtype),
        scratch_shapes=[
            pltpu.VMEM((tile_m, in_features), weight.dtype),
            pltpu.VMEM((tile_m, 1), jnp.float32),
            pltpu.VMEM((tile_m, dim), jnp.float32),
            pltpu.VMEM((tile_m, out_features), hidden.dtype),
            pltpu.VMEM((buffer_count, vector_m, streams, dim), hidden.dtype),
            pltpu.VMEM((buffer_count, vector_m, streams, dim), hidden.dtype),
            pltpu.SemaphoreType.DMA((buffer_count,)),
            pltpu.SemaphoreType.DMA((buffer_count,)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            vmem_limit_bytes=(
                None
                if interpret
                else int(0.9 * pltpu.get_tpu_info().vmem_capacity_bytes)
            ),
        ),
        interpret=interpret,
        name="engram_offload_fused",
    )(
        packed,
        hidden,
        weight,
        weight_scale,
        query_weight,
        key_weight,
        token_mask.reshape(tokens, 1),
    )
