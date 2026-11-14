# SPDX-License-Identifier: Apache-2.0
# Provenance: copied from src/vllm_torchtpu/layers/adapter/custom_ops/ple_op.py
#   at f261b79dfc1d99c52359e42de606d80dc64c01f9.
#   Owned by models/qwen38_flash from this copy on.
"""Torch boundaries for encoded PLE embedding lookup."""

import functools

import jax
import torch
from torch_tpu._internal import pallas

from .. import settings
from ..kernels.ple import (
    decode_embeddings,
    resident_lookup_raw,
    short_conv,
)
from ..kernels.ple_gate import (
    normalized_gate_dot,
)
from ..kernels.pool import (
    pooled_short_conv,
    pooled_short_conv_speculative,
    write_prefill_checkpoint,
)


@functools.cache
def build_raw_lookup_op(local_rows: int, head_dim: int, chunk_tokens: int = 8):
    """The shard's first row is the int32[1] operand ``vocab_start``, not part
    of the op: a rank-specific op name or constant would make every TP rank
    trace a different graph (TPU_PRECOMPILE_PEER_REPLAY could not share it).
    ``chunk_tokens``: tokens per gather (VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS).
    """

    def fn(table: jax.Array, ids: jax.Array, vocab_start: jax.Array) -> jax.Array:
        return resident_lookup_raw(
            table,
            ids,
            vocab_start,
            local_rows=local_rows,
            head_dim=head_dim,
            chunk_tokens=chunk_tokens,
        )

    suffix = "" if chunk_tokens == 8 else f"_c{chunk_tokens}"
    op = pallas.jax_op(
        f"qwen38_flash::ple_raw_lookup_{local_rows}_{head_dim}{suffix}", fn
    )

    @op.register_fake
    def fake(table, ids, vocab_start):
        dtype = torch.int16 if table.dtype == torch.int16 else torch.int8
        return table.new_empty((ids.shape[0], ids.shape[1] * head_dim), dtype=dtype)

    return op


@functools.cache
def build_decode_op():
    def fn(raw_embeddings: jax.Array, scale: jax.Array) -> jax.Array:
        return decode_embeddings(raw_embeddings, scale)

    op = pallas.jax_op("qwen38_flash::ple_decode", fn)

    @op.register_fake
    def fake(raw_embeddings, scale):
        return torch.empty_like(raw_embeddings, dtype=torch.bfloat16)

    return op


@functools.cache
def build_short_conv_op(dilation: int):
    def fn(
        x: jax.Array,
        weight: jax.Array,
        history: jax.Array,
        starts: jax.Array,
        slots: jax.Array,
        initial: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        output, updated = short_conv(
            x, weight, history, starts, slots, initial, dilation=dilation
        )
        return updated, output

    op = pallas.jax_op(
        f"qwen38_flash::ple_short_conv_{dilation}", fn, donate_argnums=(2,)
    )

    @op.register_fake
    def fake(x, weight, history, starts, slots, initial):
        return torch.empty_like(history), torch.empty_like(x)

    def run(x, weight, history, starts, slots, initial):
        updated, output = op(x, weight, history, starts, slots, initial)
        history.copy_(updated)
        return output

    return run


@functools.cache
def build_pooled_short_conv_op(dilation, state_len):
    fn = functools.partial(pooled_short_conv, dilation=dilation, state_len=state_len)
    op = pallas.jax_op(
        f"qwen38_flash::pooled_short_conv_{dilation}_{state_len}",
        fn,
        donate_argnums=(2,),
    )

    @op.register_fake
    def fake(x, weight, pool, starts, ids, initial):
        return torch.empty_like(pool), torch.empty_like(x)

    def run(x, weight, pool, starts, ids, initial):
        updated, output = op(x, weight, pool, starts, ids, initial)
        pool.copy_(updated)
        return output

    return run


@functools.cache
def build_pooled_speculative_short_conv_op(dilation, state_len):
    """Pooled PLE history with per-verify-position checkpoint blocks."""
    fn = functools.partial(
        pooled_short_conv_speculative, dilation=dilation, state_len=state_len
    )
    op = pallas.jax_op(
        f"qwen38_flash::pooled_spec_short_conv_{dilation}_{state_len}",
        fn,
        donate_argnums=(2,),
    )

    @op.register_fake
    def fake(x, weight, pool, starts, ids, initial, read_ids, ckpt_ids, windowed):
        return torch.empty_like(pool), torch.empty_like(x)

    def run(x, weight, pool, starts, ids, initial, read_ids, ckpt_ids, windowed):
        updated, output = op(
            x, weight, pool, starts, ids, initial, read_ids, ckpt_ids, windowed
        )
        pool.copy_(updated)
        return output

    return run


@functools.cache
def build_pooled_prefill_checkpoint_op(state_len):
    """Pooled PLE history at internal prefill checkpoints (tail fold)."""
    fn = functools.partial(write_prefill_checkpoint, state_len=state_len)
    op = pallas.jax_op(
        f"qwen38_flash::pooled_prefill_ckpt_{state_len}", fn, donate_argnums=(0,)
    )

    @op.register_fake
    def fake(pool, x, starts, offsets, ids):
        return torch.empty_like(pool)

    def run(pool, x, starts, offsets, ids):
        pool.copy_(op(pool, x, starts, offsets, ids))

    return run


@functools.cache
def build_normalized_gate_dot_op(streams: int):
    """Fused per-stream RMS-normalized key . query gate dot, BF16 [T, S]."""

    def fn(
        key: jax.Array,
        query: jax.Array,
        key_inv: jax.Array,
        query_inv: jax.Array,
        key_weight: jax.Array,
        query_weight: jax.Array,
    ) -> jax.Array:
        return normalized_gate_dot(
            key, query, key_inv, query_inv, key_weight, query_weight, streams=streams
        )

    op = pallas.jax_op(f"qwen38_flash::ple_gate_dot_s{streams}", fn)

    @op.register_fake
    def fake(key, query, key_inv, query_inv, key_weight, query_weight):
        return key.new_empty((key.shape[0], streams), dtype=torch.bfloat16)

    @op.register_kernel("cpu")
    def cpu(key, query, key_inv, query_inv, key_weight, query_weight):
        raise NotImplementedError("The fused PLE gate requires TPU tensors")

    return op


def build_normalized_gate_dot_dispatch(streams: int, hidden: int):
    """PLE gate dot ``[T, S, 1]``, chosen when the layer is constructed.

    Default: the Torch path (two grouped RMS norms, BF16 products, FP32 sum).
    VLLM_TPU_PLE_FUSED_GATE: one Pallas kernel applies the norm weights,
    multiplies and reduces per stream without materializing the normalized
    key and query; only the per-stream inverse RMS is computed outside.
    """
    if not settings.VLLM_TPU_PLE_FUSED_GATE:

        def dot(key, query, norm_key, norm_query):
            key = norm_key(key).unflatten(-1, (streams, hidden))
            query = norm_query(query).unflatten(-1, (streams, hidden))
            # As in upstream's gate kernel, the products are rounded to the
            # activation dtype and then summed in fp32.
            return (key * query).float().sum(dim=-1, keepdim=True).to(key.dtype)

        return dot

    native = build_normalized_gate_dot_op(streams)

    def inverse_rms(raw, norm):
        grouped = raw.float().unflatten(-1, (streams, hidden))
        return torch.rsqrt(grouped.square().mean(dim=-1) + norm.variance_epsilon)

    def dispatch(key, query, norm_key, norm_query):
        return native(
            key,
            query,
            inverse_rms(key, norm_key),
            inverse_rms(query, norm_query),
            norm_key.weight,
            norm_query.weight,
        ).unsqueeze(-1)

    return dispatch
