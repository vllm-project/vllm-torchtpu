# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real SparseCore gather + SparseMLA against independently decoded FP32 scales."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.csa_cache_layout import words_to_slabs
from vllm_torchtpu.kernels.deepseek_v41.fp8_cache import encode_fp8_kv
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    sparse_ragged_paged_attention_union_gather,
)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("union", [False, True])
def test_row_scale_sparse_attention_matches_dense_reference(split, union):
    if jax.default_backend() != "tpu":
        pytest.skip("Real TPU attention")
    rng = np.random.default_rng(73)
    tokens, heads, count, topk = 8, 4, 512, 128
    latent = jnp.asarray(
        rng.normal(size=(count, 512)) * rng.uniform(0.1, 2, (count, 1)), jnp.bfloat16
    )
    q = jnp.asarray(rng.normal(size=(tokens, heads, 512)), jnp.bfloat16)
    nope, rope = encode_fp8_kv(latent)
    pool = jnp.concatenate((nope, rope), -1).reshape(8, 128, 128)
    banks = (pool[:4], pool[4:]) if split else pool
    indices = np.stack([rng.choice(count, topk, replace=False) for _ in range(tokens)])
    indices[:, -7:] = -1
    indices[0, 3:] = -1
    sink = jnp.linspace(-0.3, 0.5, heads)
    scale = 512**-0.5
    output = sparse_ragged_paged_attention_union_gather(
        q,
        banks,
        banks,
        jnp.asarray(indices, jnp.int32),
        jnp.tile(jnp.arange(8, dtype=jnp.int32), 2),
        jnp.asarray([0, 4, 8], jnp.int32),
        jnp.asarray([0, 2, 2], jnp.int32),
        sink,
        jnp.zeros(q.shape, jnp.bfloat16),
        jnp.zeros((tokens, heads), jnp.float32),
        jnp.full((tokens, heads), -jnp.inf, jnp.float32),
        sm_scale=scale,
        gather_and_attention_chunk_size=4,
        attention_kernel_batch_size=2,
        union_cap=count,
        union_gather=union,
    )
    # Decode the serialized cache with NumPy, independently of the TPU reader.
    payload = np.asarray(words_to_slabs(nope)).reshape(count, 512)
    row_scale = payload[:, 448:452].copy().view("<f4")
    decoded = payload[:, :448].view(jnp.float8_e4m3fn).astype(np.float32) * row_scale
    decoded = decoded.astype(jnp.bfloat16).astype(np.float32)
    kv = np.concatenate((decoded, np.asarray(latent[:, 448:], np.float32)), -1)
    selected = kv[np.maximum(indices, 0)]
    logits = np.einsum("thd,tkd->thk", np.asarray(q, np.float32), selected) * scale
    logits = np.where(indices[:, None, :] >= 0, logits, -np.inf)
    maximum = np.maximum(logits.max(-1, keepdims=True), np.asarray(sink)[None, :, None])
    weights = np.exp(logits - maximum)
    denominator = weights.sum(-1, keepdims=True) + np.exp(
        np.asarray(sink)[None, :, None] - maximum
    )
    expected = np.einsum("thk,tkd->thd", weights / denominator, selected)
    np.testing.assert_allclose(
        np.asarray(output, np.float32), expected, rtol=0.02, atol=0.015
    )
