# SPDX-License-Identifier: Apache-2.0
"""Decode presort metadata with the native activation gather and fallback."""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.experimental.decode_preparation import (
    fuse_function,
    prefix_sum,
)
from vllm_torchtpu.kernels.experimental.push_fused_moe import host
from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import get_tpu_version


def _metadata(pairs, counts):
    """Retain count semantics, excluding the rejected activation matmul."""
    topk, tokens = pairs.shape
    experts = counts.size
    bins = jnp.arange(experts, dtype=jnp.int32)
    selection = jnp.zeros((tokens, experts), bool)
    for k in range(topk):
        column = pairs[k]
        selection = selection | (column[:, None] == bins[None, :])
    cumulative = prefix_sum(selection.astype(jnp.int32), axis=0)
    end = prefix_sum(counts)
    start = end - counts
    jj = jnp.arange(tokens * topk, dtype=jnp.int32)
    expert = jnp.minimum(jnp.sum(end[None, :] <= jj[:, None], axis=1), experts - 1)
    onehot = expert[:, None] == bins[None, :]
    offset = jj - jnp.sum(jnp.where(onehot, start[None, :], 0), axis=1)
    # Each one-hot row selects one integer in [0, 384]. Base-256 digits
    # are exact in BF16, so FP32 accumulation and reconstruction are exact.
    hot = onehot.astype(jnp.bfloat16)
    low = (cumulative & 255).T.astype(jnp.bfloat16)
    high = (cumulative >> 8).T.astype(jnp.bfloat16)
    col = (
        jnp.dot(hot, low, preferred_element_type=jnp.float32)
        + 256 * jnp.dot(hot, high, preferred_element_type=jnp.float32)
    ).astype(jnp.int32)
    order = jnp.minimum(jnp.sum(col <= offset[:, None], axis=1), tokens - 1)
    return (order, start)


def supports_local_sort(
    pairs: jax.Array,
    counts: jax.Array,
    q: jax.Array,
    *,
    e_total: int,
    topk: int,
) -> bool:
    """Limit fusion to the measured v7 FP8 decode contract."""
    return (
        (e_total, topk) == (384, 6)
        and q.shape in tuple((t, 40, 128) for t in (96, 192, 256, 288, 384))
        and pairs.shape == (q.shape[0] * 6,)
        and counts.shape == (384,)
        and pairs.dtype == counts.dtype == jnp.int32
        and q.dtype == jnp.float8_e4m3fn
        and jax.default_backend() == "tpu"
        and get_tpu_version() == 7
    )


def local_sort_activations(
    pairs: jax.Array,
    counts: jax.Array,
    q: jax.Array,
    *,
    e_total: int,
    topk: int,
) -> tuple[jax.Array, jax.Array]:
    """Fuse small tables while retaining the original FP8 gather."""
    if not supports_local_sort(pairs, counts, q, e_total=e_total, topk=topk):
        return host.local_sort_activations(pairs, counts, q, e_total=e_total, topk=topk)
    inputs = (pairs.reshape(q.shape[0], topk).T, counts)
    fused = fuse_function(_metadata, inputs, name="dsv41_presort_metadata_v2")
    order, start = fused(*inputs)
    return q[order], start
