# SPDX-License-Identifier: Apache-2.0
"""Packed FP8 Engram rows through projection and gating in one Pallas call.

The caller gathers packed rows across TP before this kernel. CPU lookup and
H2D remain outside it. Only this rank's SP tokens are read from each head
shard; dequantized embeddings and projected keys never leave VMEM.
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
    """Unpack int32 words [..., W] into little-endian uint8 bytes [..., 4*W]."""
    shifts = jnp.arange(4, dtype=jnp.int32) * 8
    return (
        ((words[..., None] >> shifts) & 255)
        .astype(jnp.uint8)
        .reshape(*words.shape[:-1], words.shape[-1] * 4)
    )


def dequantize_rows(raw: jax.Array, scales: jax.Array) -> jax.Array:
    """Decode packed FP8 rows, with B values sharing one E8M0 scale.

    Args:
        raw: int32 [..., C/4], packing C FP8 E4M3FN values per row.
        scales: int32 [..., C/(4*B)], packing C/B scale bytes per row.

    Returns:
        BF16 [..., C]; scale byte 255 marks an invalid block and yields zeros.
    """
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
    """Apply S stream gates to a shared value vector of hidden width D.

    Args:
        hidden: BF16 [T, S, D], residual inputs for T local tokens.
        projection: BF16 [T, (S+1)*D], S keys followed by one shared value.
        query_weight: BF16 [S, D], learned query normalization weights.
        key_weight: BF16 [S, D], learned key normalization weights.
        token_mask: [T], boolean or numeric 0/1; zero preserves the residual.
        eps: Scalar RMS normalization epsilon.
        clamp_value: Scalar lower bound before the gate's square root.

    Returns:
        BF16 [T, S, D], gated value plus residual.
    """
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
    raw: jax.Array,
    scales: jax.Array,
    hidden: jax.Array,
    weight: jax.Array,
    weight_scale: jax.Array,
    query_weight: jax.Array,
    key_weight: jax.Array,
    token_mask: jax.Array,
    rank: int,
    eps: float,
    clamp_value: float,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Fuse packed-row dequant, FP8 WKV, normalization, gating and residual.

    Supports DP1 with SP tokens, equally sized head shards, per-channel FP8
    projection scales, and four-byte-aligned scale rows. Inputs are already
    padded by the runner. The BF16 roundings before and after WKV match the
    unfused Engram, including invalid embedding scales (byte 255).

    P is TP size, T is the full padded token count, t=T/P is local SP tokens,
    C is embedding width per head shard, B is the embedding scale block size,
    S is stream count, and D is hidden width. K=P*C is projection input width.

    Args:
        raw: int32 [P, T, C/4], gathered packed FP8 E4M3FN embedding rows.
        scales: int32 [P, T, C/(4*B)], gathered packed E8M0 scale bytes.
        hidden: BF16 [t, S, D], this rank's token shard.
        weight: FP8 [K, (S+1)*D], S key projections and one value projection.
        weight_scale: FP32 [(S+1)*D], per-output-channel projection scales.
        query_weight: BF16 [S, D], learned query normalization weights.
        key_weight: BF16 [S, D], learned key normalization weights.
        token_mask: [t], boolean or numeric 0/1 for this rank's tokens.
        rank: Scalar TP rank selecting rows [rank*t:(rank+1)*t] in each shard.
        eps: Scalar RMS normalization epsilon.
        clamp_value: Scalar lower bound before the gate's square root.
        interpret: Run Pallas on the CPU interpreter for numerical tests.

    Returns:
        BF16 [t, S, D], matching hidden's layout.
    """
    tp, full_tokens, raw_words = raw.shape
    tokens, streams, dim = hidden.shape
    in_features = tp * raw_words * 4
    out_features = (streams + 1) * dim
    assert full_tokens == tokens * tp, (full_tokens, tokens, tp)
    assert 0 <= rank < tp, (rank, tp)
    assert scales.shape[:2] == raw.shape[:2]
    assert raw_words % scales.shape[-1] == 0
    assert weight.shape == (in_features, out_features), weight.shape
    assert weight_scale.shape == (out_features,), weight_scale.shape
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
        acc_scratch,
        projection_scratch,
        hidden_buffers,
        output_buffers,
        hidden_sems,
        output_sems,
    ):
        """Process M=tile_m tokens, one output group and tile_k input columns.

        Ref shapes: raw_ref [P,M,C/4], scale_ref [P,M,C/(4*B)], h_ref and
        out_ref HBM [t,S,D], w_ref [tile_k,D], ws_ref [D], q_ref/k_ref [S,D],
        mask_ref [M,1]. VMEM: x_scratch FP8 [M,K], xs_scratch FP32 [M,1],
        acc_scratch FP32 [M,D], projection_scratch BF16 [M,(S+1)*D], and
        hidden_buffers/output_buffers BF16 [N,vector_m,S,D], N=min(2,M/vector_m).
        DMA semaphores hidden_sems/output_sems each have shape [N].
        """
        output_block = pl.program_id(1)
        step = pl.program_id(2)
        token_start = pl.program_id(0) * tile_m

        def hidden_copy(chunk):
            """Copy one scalar-indexed HBM [vector_m,S,D] chunk into VMEM."""
            rows = pl.ds(token_start + chunk * vector_m, vector_m)
            bank = chunk % buffer_count
            return pltpu.make_async_copy(
                h_ref.at[rows, :, :], hidden_buffers.at[bank], hidden_sems.at[bank]
            )

        def output_copy(chunk):
            """Copy one scalar-indexed VMEM [vector_m,S,D] chunk into HBM."""
            rows = pl.ds(token_start + chunk * vector_m, vector_m)
            bank = chunk % buffer_count
            return pltpu.make_async_copy(
                output_buffers.at[bank], out_ref.at[rows, :, :], output_sems.at[bank]
            )

        @pl.when((output_block == 0) & (step == 0))
        def prepare():
            hidden_copy(0).start()

            def prepare_rows(index, _):
                tile = pl.ds(index * vector_m, vector_m)
                rows = dequantize_rows(raw_ref[:, tile, :], scale_ref[:, tile, :])
                embedded = rows.transpose(1, 0, 2).reshape(vector_m, in_features)
                quantized, activation_scale = quantize_tensor(embedded, weight.dtype)
                x_scratch[tile, :] = quantized
                xs_scratch[tile, :] = activation_scale

            jax.lax.fori_loop(0, tile_m // vector_m, prepare_rows, None)

        @pl.when(step == 0)
        def reset_accumulator():
            acc_scratch[...] = jnp.zeros((tile_m, dim), jnp.float32)

        x = x_scratch[:, pl.ds(step * tile_k, tile_k)]
        acc_scratch[...] += jnp.dot(x, w_ref[...], preferred_element_type=jnp.float32)

        @pl.when(step == steps - 1)
        def finish():
            def finish_rows(index, _):
                tile = pl.ds(index * vector_m, vector_m)
                projection = (
                    acc_scratch[tile, :] * xs_scratch[tile, :] * ws_ref[...]
                ).astype(hidden.dtype)

                group = jnp.where(output_block == 0, streams, output_block - 1)
                projection_scratch[tile, pl.ds(group * dim, dim)] = projection

            jax.lax.fori_loop(0, tile_m // vector_m, finish_rows, None)

            @pl.when(output_block == streams)
            def apply_gates():
                def gate_rows(chunk, _):
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

                jax.lax.fori_loop(0, chunks, gate_rows, None)
                # Drain outstanding stores before reusing scratch for a new tile.
                for chunk in range(max(0, chunks - buffer_count), chunks):
                    output_copy(chunk).wait()

    def token_block(i, j, k):
        return 0, i + rank * (tokens // tile_m), 0

    def projection_index(j):
        # Project the shared value first, followed by each key stream.
        return jnp.where(j == 0, streams, j - 1)

    result = pl.pallas_call(
        kernel,
        grid=(tokens // tile_m, streams + 1, steps),
        in_specs=[
            pl.BlockSpec((tp, tile_m, raw_words), token_block),
            pl.BlockSpec((tp, tile_m, scales.shape[-1]), token_block),
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
        raw,
        scales,
        hidden,
        weight,
        weight_scale,
        query_weight,
        key_weight,
        token_mask.reshape(tokens, 1),
    )
    return result
