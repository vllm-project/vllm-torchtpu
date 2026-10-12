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
    shifts = jnp.arange(4, dtype=jnp.int32) * 8
    return (
        ((words[..., None] >> shifts) & 255)
        .astype(jnp.uint8)
        .reshape(*words.shape[:-1], words.shape[-1] * 4)
    )


def dequantize_rows(raw: jax.Array, scales: jax.Array) -> jax.Array:
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
        values_scratch,
    ):
        output_block = pl.program_id(1)
        step = pl.program_id(2)

        @pl.when((output_block == 0) & (step == 0))
        def prepare():
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

                @pl.when(output_block == 0)
                def save_values():
                    values_scratch[tile, :] = projection

                @pl.when(output_block != 0)
                def apply_gate():
                    kv = jnp.concatenate((projection, values_scratch[tile, :]), axis=-1)
                    out_ref[tile, :] = gated_residual(
                        h_ref[tile, :][:, None, :],
                        kv,
                        q_ref[...],
                        k_ref[...],
                        mask_ref[tile, 0],
                        eps,
                        clamp_value,
                    )[:, 0, :]

            jax.lax.fori_loop(0, tile_m // vector_m, finish_rows, None)

    def token_block(i, j, k):
        return 0, i + rank * (tokens // tile_m), 0

    def stream_index(j):
        return jnp.maximum(j - 1, 0)

    def projection_index(j):
        # Retain the shared value vector, then consume one key stream at a time.
        return jnp.where(j == 0, streams, j - 1)

    def local_block(i, j, k):
        return stream_index(j), i, 0

    result = pl.pallas_call(
        kernel,
        grid=(tokens // tile_m, streams + 1, steps),
        in_specs=[
            pl.BlockSpec((tp, tile_m, raw_words), token_block),
            pl.BlockSpec((tp, tile_m, scales.shape[-1]), token_block),
            pl.BlockSpec((None, tile_m, dim), local_block),
            pl.BlockSpec((tile_k, dim), lambda i, j, k: (k, projection_index(j))),
            pl.BlockSpec((dim,), lambda i, j, k: (projection_index(j),)),
            pl.BlockSpec((None, 1, dim), lambda i, j, k: (stream_index(j), 0, 0)),
            pl.BlockSpec((None, 1, dim), lambda i, j, k: (stream_index(j), 0, 0)),
            pl.BlockSpec((tile_m, 1), lambda i, j, k: (i, 0)),
        ],
        out_specs=pl.BlockSpec((None, tile_m, dim), local_block),
        out_shape=jax.ShapeDtypeStruct((streams, tokens, dim), hidden.dtype),
        scratch_shapes=[
            pltpu.VMEM((tile_m, in_features), weight.dtype),
            pltpu.VMEM((tile_m, 1), jnp.float32),
            pltpu.VMEM((tile_m, dim), jnp.float32),
            pltpu.VMEM((tile_m, dim), hidden.dtype),
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
        hidden.transpose(1, 0, 2),
        weight,
        weight_scale,
        query_weight[:, None, :],
        key_weight[:, None, :],
        token_mask.reshape(tokens, 1),
    )
    return result.transpose(1, 0, 2)
