# SPDX-License-Identifier: Apache-2.0
"""Chunk boundaries preserve ragged requests, SWA visibility and cached KV."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.csa2 import csa2
from vllm_torchtpu.kernels.deepseek_v41.csa2_cache import encode_csa2_kv
from vllm_torchtpu.kernels.deepseek_v41.csa2_metadata import prepare_swa_metadata
from vllm_torchtpu.kernels.deepseek_v41.csa2_pipeline import csa2_pipeline


@pytest.fixture
def sparsecore_device():
    if jax.default_backend() != "tpu":
        pytest.skip("CSA2 pipeline requires a TPU with SparseCore")
    if pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("CSA2 pipeline requires SparseCore")


@pytest.mark.parametrize("banked", [False, True])
@pytest.mark.parametrize("non_causal", [False, True])
def test_chunked_attention_preserves_requests_and_reusable_kv(
    sparsecore_device, banked, non_causal
):
    # Three requests straddle chunk boundaries, with empty requests interleaved
    # and a final chunk containing only padding. Noncausal windows include KV
    # beyond the end of the current chunk.
    tokens, heads, chunk_size, page_size = 193, 8, 64, 32
    rng = np.random.default_rng(31)

    def random(shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.2, jnp.bfloat16)

    q = random((tokens, heads, 512))
    words, scales = encode_csa2_kv(random((257, 512)))
    starts = jnp.array([0, 0, 43, 43, 103, 155, 155], jnp.int32)
    positions = jnp.array(
        [*range(160, 203), *range(240, 300), *range(330, 382)] + [0] * 38,
        jnp.int32,
    )
    counts = jnp.array(rng.integers(0, 513, tokens), jnp.int32)
    counts = jnp.where(jnp.arange(tokens) < 155, counts, 0)
    records = jnp.array(rng.integers(0, 257, (tokens, 512)), jnp.int32)
    records = jnp.where(jnp.arange(512) < counts[:, None], records, -1)
    selected = jnp.maximum(records, 0)
    expected_words = jnp.where(records[:, :, None] >= 0, words[selected], 0)
    expected_scales = jnp.where(records >= 0, scales[selected], 0)
    table = jnp.arange(1, 97, dtype=jnp.int32).reshape(6, 16)
    metadata, bounds, _ = prepare_swa_metadata(
        table,
        starts,
        positions,
        jnp.array([0, 150, 0, 200, 300, 0], jnp.int32),
        page_size=page_size,
        page_stride=page_size,
        row_offset=0,
        non_causal=non_causal,
    )
    swa = random((97, page_size, 512)).at[0].set(0)
    bits = jax.lax.bitcast_convert_type(swa, jnp.uint16).astype(jnp.uint32)
    bits = bits.reshape(97, page_size, 2, 2, 128)
    packed = bits[:, :, :, 0] | (bits[:, :, :, 1] << 16)
    swa = jax.lax.bitcast_convert_type(packed, jnp.int32).transpose(0, 2, 1, 3)
    word_banks = (words[:113], words[113:]) if banked else words
    scale_banks = (scales[:113], scales[113:]) if banked else scales
    swa_banks = (swa[:48], swa[48:]) if banked else swa
    angles = jnp.asarray(rng.standard_normal((tokens, 64)) * 0.1)
    rope = jnp.stack(
        (
            jnp.concatenate((jnp.ones_like(angles), jnp.cos(angles)), axis=1),
            jnp.concatenate((jnp.zeros_like(angles), jnp.sin(angles)), axis=1),
        ),
        axis=1,
    )
    sink = jnp.linspace(-1, 1, heads, dtype=jnp.float32)
    kwargs = dict(scale=512**-0.5, compress_ratio=2, bq=6)

    def attend(query, selected_words, selected_scales, offset=0):
        end = offset + query.shape[0]
        return csa2(
            query,
            selected_words,
            selected_scales,
            swa_banks,
            jnp.clip(starts - offset, 0, query.shape[0]),
            positions[offset:end],
            sink,
            rope[offset:end],
            swa_metadata=metadata,
            swa_bounds=bounds if non_causal else None,
            main_lengths=counts[offset:end],
            **kwargs,
        )

    expected = attend(q, expected_words, expected_scales)
    output, gathered_words, gathered_scales = csa2_pipeline(
        q,
        word_banks,
        scale_banks,
        records,
        swa_banks,
        starts,
        positions,
        sink,
        rope,
        metadata,
        bounds if non_causal else None,
        counts,
        chunk_size=chunk_size,
        **kwargs,
    )
    np.testing.assert_allclose(output, expected, atol=2e-3, rtol=2e-2)
    np.testing.assert_array_equal(output[155:], 0)
    np.testing.assert_array_equal(gathered_words, expected_words)
    np.testing.assert_array_equal(gathered_scales, expected_scales)

    next_q = q * jnp.bfloat16(0.7)
    reused = attend(next_q, gathered_words, gathered_scales)
    np.testing.assert_allclose(
        reused, attend(next_q, expected_words, expected_scales), atol=2e-3, rtol=2e-2
    )
