# SPDX-License-Identifier: Apache-2.0
"""Packed Engram with K-resident explicit MXU accumulators for the T288 bucket."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.engram.kernel import gated_residual
from vllm_torchtpu.kernels.engram.tuned_block_sizes import (
    ACC_GATE_TILE_M,
    ACC_OUTPUT_TILE,
    VECTOR_TILE_M,
    get_block_sizes,
)
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor


def fused_engram_acc(
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
    gate_tile_m: int = ACC_GATE_TILE_M,
) -> jax.Array:
    """TP1 packed Engram with K-resident MXU accumulator slabs.

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
        gate_tile_m: Token rows per gate/output DMA tile, independent of quantization.

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
    assert tile_k == 256
    assert tile_m <= 512
    chunks = tile_m // vector_m
    assert gate_tile_m > 0 and tile_m % gate_tile_m == 0
    gate_chunks = tile_m // gate_tile_m
    buffer_count = min(2, gate_chunks)
    tile_n = ACC_OUTPUT_TILE
    assert dim % tile_n == 0
    slabs = dim // tile_n

    raw_end = tokens * raw_words
    raw = packed[:raw_end].reshape(tokens, raw_words)
    scales = packed[raw_end:].reshape(tokens, scale_words)

    def kernel(
        raw_ref,
        scale_ref,
        h_ref,
        w_ref,
        ws_ref,
        q_ref,
        k_ref,
        mask_ref,
        out_ref,
        x_scratch,
        xs_scratch,
        projection_scratch,
        hidden_buffers,
        output_buffers,
        hidden_sems,
        output_sems,
    ):
        output_block = pl.program_id(1)
        slab = pl.program_id(2)
        token_start = pl.program_id(0) * tile_m

        def hidden_copy(chunk):
            rows = pl.ds(token_start + chunk * gate_tile_m, gate_tile_m)
            bank = chunk % buffer_count
            return pltpu.make_async_copy(
                h_ref.at[rows, :, :], hidden_buffers.at[bank], hidden_sems.at[bank]
            )

        def output_copy(chunk):
            rows = pl.ds(token_start + chunk * gate_tile_m, gate_tile_m)
            bank = chunk % buffer_count
            return pltpu.make_async_copy(
                output_buffers.at[bank], out_ref.at[rows, :, :], output_sems.at[bank]
            )

        @pl.when((output_block == 0) & (slab == 0))
        def prepare():
            hidden_copy(0).start()

            def prepare_rows(index, unused):
                tile = pl.ds(index * vector_m, vector_m)
                raw_bytes = pltpu.bitcast(raw_ref[tile, :].T, jnp.uint8).T
                scale_bytes = pltpu.bitcast(scale_ref[tile, :].T, jnp.uint8).T
                values = pltpu.bitcast(raw_bytes, jnp.float8_e4m3fn).astype(jnp.float32)
                block = embedding_block_size
                scale_columns = scale_bytes.astype(jnp.int32).T
                expanded_scale = (
                    jnp.broadcast_to(
                        scale_columns[:, None, :],
                        (in_features // block, block, vector_m),
                    )
                    .reshape(in_features, vector_m)
                    .T
                )
                power = jax.lax.bitcast_convert_type(expanded_scale << 23, jnp.float32)
                invalid = expanded_scale == 255
                embedded = jnp.where(invalid, 0.0, values * power).astype(jnp.bfloat16)
                quantized, activation_scale = quantize_tensor(embedded, weight.dtype)
                tile = pl.ds(index * vector_m, vector_m)
                x_scratch[tile, :] = quantized
                xs_scratch[tile, :] = activation_scale
                return unused

            jax.lax.fori_loop(0, chunks, prepare_rows, None)

        def project(*accumulators):
            def push(step, pair):
                for mxu in range(2):
                    rhs = w_ref[
                        pl.ds(step * tile_k, tile_k),
                        pl.ds(pair * 512 + mxu * 256, 256),
                    ]
                    pltpu.matmul_push_rhs(rhs, step % 2, mxu)

            for pair in range(2):
                push(0, pair)
                for step in range(steps):
                    if step + 1 < steps:
                        push(step + 1, pair)
                    lhs = x_scratch[:, pl.ds(step * tile_k, tile_k)]
                    for mxu in range(2):
                        pltpu.matmul_acc_lhs(
                            accumulators[pair * 2 + mxu],
                            lhs,
                            load_staged_rhs=step % 2,
                        )
            group = jnp.where(output_block == 0, streams, output_block - 1)
            for pair in range(2):
                for mxu in range(2):
                    local = pair * 512 + mxu * 256
                    values = pltpu.matmul_pop(accumulators[pair * 2 + mxu])
                    projection = (
                        values * xs_scratch[...] * ws_ref[pl.ds(local, 256)]
                    ).astype(hidden.dtype)
                    columns = pl.ds(group * dim + slab * tile_n + local, 256)
                    projection_scratch[:, columns] = projection

        pl.run_scoped(
            project,
            *(
                pltpu.ACC(mxu)((tile_m, 256), jnp.float32)
                for _ in range(2)
                for mxu in range(2)
            ),
        )

        @pl.when((output_block == streams) & (slab == slabs - 1))
        def apply_gates():
            def gate_rows(chunk, unused):
                bank = chunk % buffer_count
                hidden_copy(chunk).wait()

                @pl.when(chunk + 1 < gate_chunks)
                def prefetch():
                    hidden_copy(chunk + 1).start()

                @pl.when(chunk >= buffer_count)
                def release_output_bank():
                    output_copy(chunk - buffer_count).wait()

                rows = pl.ds(chunk * gate_tile_m, gate_tile_m)
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

            jax.lax.fori_loop(0, gate_chunks, gate_rows, None)
            for chunk in range(max(0, gate_chunks - buffer_count), gate_chunks):
                output_copy(chunk).wait()

    def projection_index(j):
        return jnp.where(j == 0, streams, j - 1)

    return pl.pallas_call(
        kernel,
        grid=(tokens // tile_m, streams + 1, slabs),
        in_specs=[
            pl.BlockSpec((tile_m, raw_words), lambda i, j, k: (i, 0)),
            pl.BlockSpec((tile_m, scale_words), lambda i, j, k: (i, 0)),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(
                (in_features, tile_n),
                lambda i, j, k: (0, projection_index(j) * slabs + k),
            ),
            pl.BlockSpec((tile_n,), lambda i, j, k: (projection_index(j) * slabs + k,)),
            pl.BlockSpec((streams, dim), lambda i, j, k: (0, 0)),
            pl.BlockSpec((streams, dim), lambda i, j, k: (0, 0)),
            pl.BlockSpec((tile_m, 1), lambda i, j, k: (i, 0)),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        out_shape=jax.ShapeDtypeStruct(hidden.shape, hidden.dtype),
        scratch_shapes=[
            pltpu.VMEM((tile_m, in_features), weight.dtype),
            pltpu.VMEM((tile_m, 1), jnp.float32),
            pltpu.VMEM((tile_m, out_features), hidden.dtype),
            pltpu.VMEM((buffer_count, gate_tile_m, streams, dim), hidden.dtype),
            pltpu.VMEM((buffer_count, gate_tile_m, streams, dim), hidden.dtype),
            pltpu.SemaphoreType.DMA((buffer_count,)),
            pltpu.SemaphoreType.DMA((buffer_count,)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            vmem_limit_bytes=int(0.9 * pltpu.get_tpu_info().vmem_capacity_bytes),
        ),
        name="engram_offload_acc_resident",
    )(
        raw,
        scales,
        hidden,
        weight,
        weight_scale,
        query_weight,
        key_weight,
        token_mask.reshape(tokens, 1),
    )
