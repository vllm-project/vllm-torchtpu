# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Decode presort metadata with the native activation gather and fallback."""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.experimental.decode_preparation import (
    PRESORT_TOKENS,
    admit,
    first_problem,
    fuse_function,
    prefix_sum,
    tpu_version_problem,
)
from vllm_torchtpu.kernels.experimental.push_fused_moe import host

# (experts, topk) the metadata kernel was checked for; its BF16 base-256
# one-hot reconstruction is exact while per-expert counts stay below 2**16.
PRESORT_GEOMETRY = (384, 6)


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


def local_sort_problem(pairs, counts, q, *, e_total, topk) -> str | None:
    """First violated contract of the fused metadata kernel, or None.

    Only ``pairs`` and ``counts`` enter the kernel; ``q`` is gathered by the
    native path, so its trailing shape and dtype are not constrained.
    """
    if q.ndim < 1:
        return "activations are a scalar"
    tokens = q.shape[0]
    return (
        first_problem(
            (
                (e_total, topk) == PRESORT_GEOMETRY,
                f"(experts, topk)={(e_total, topk)}, kernel {PRESORT_GEOMETRY}",
            ),
            (
                pairs.shape == (tokens * topk,),
                f"pairs {pairs.shape} for {tokens} tokens",
            ),
            (
                counts.shape == (e_total,),
                f"counts {counts.shape} for {e_total} experts",
            ),
            (pairs.dtype == counts.dtype == jnp.int32, "pairs/counts are not int32"),
        )
        or tpu_version_problem()
    )


def supports_local_sort(
    pairs: jax.Array,
    counts: jax.Array,
    q: jax.Array,
    *,
    e_total: int,
    topk: int,
) -> bool:
    """Whether the fused presort metadata path is admitted for these inputs."""
    return admit(
        "moe_presort",
        local_sort_problem(pairs, counts, q, e_total=e_total, topk=topk),
        tokens=q.shape[0] if q.ndim else None,
        buckets=PRESORT_TOKENS,
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
