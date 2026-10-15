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
"""Joint Q/KV preprocessing and HCA attention against independent RoPE kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout, rope
from vllm_torchtpu.kernels.deepseek_v4.core_attention import (
    mla,
    mla_swa,
    swa_hca_decode,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
    deepseek_v4_attention_op,
)

pytestmark = pytest.mark.tpu_test


@pytest.mark.parametrize("padded_sequences", (1, 256))
@pytest.mark.parametrize("position", (1, 127, 128, 129, 65535, 65536))
@pytest.mark.parametrize("decode", (False, True))
@pytest.mark.parametrize("word_cache", (False, True))
def test_fused_qkv_hca_continuous_cache(position, word_cache, decode, padded_sequences):
    rng = np.random.default_rng(577)
    q = jnp.asarray(rng.normal(size=(1, 64, 512)), jnp.bfloat16)
    new_kv = jnp.asarray(rng.normal(size=(1, 512)), jnp.bfloat16)
    raw = jnp.asarray(rng.normal(size=(65 * 8, 512)), jnp.bfloat16)
    main = mla_swa.prepare_kv_inputs(raw).reshape(65, 16, 4, 128)
    raw_swa = jnp.asarray(rng.normal(size=(3 * 128, 512)), jnp.bfloat16)
    cache = mla_swa.prepare_kv_inputs(raw_swa).reshape(3, 256, 4, 128)
    if word_cache:
        cache = csa_cache_layout.slabs_to_words(cache)
    block_pages = swa_hca_decode.get_block_pages(
        q,
        new_kv,
        cache,
        main,
        sliding_window=128,
        logical_page_size=128,
        swa_only=False,
        two_caches_same_buffer=False,
        non_causal_block=False,
    )
    if block_pages is None:
        pytest.skip("No fused HCA tuning entry for this TPU")
    reference_cache = jnp.array(cache, copy=True)
    pages = jnp.arange((position + 16 + 127) // 128, dtype=jnp.int32) % 3
    main_pages = jnp.asarray(rng.permutation(65), dtype=jnp.int32)
    pages = jnp.tile(pages, padded_sequences)
    main_pages = jnp.tile(main_pages, padded_sequences)
    cu = jnp.minimum(jnp.arange(padded_sequences + 1), 1).astype(jnp.int32)
    distribution = jnp.array([1, 1, 1] if decode else [0, 0, 1], dtype=jnp.int32)
    angle = np.arange(position + 32, dtype=np.float32)[:, None] * np.exp(
        -np.arange(32, dtype=np.float32)[None, :] / 4
    )
    cos_sin = jnp.asarray(np.concatenate([np.cos(angle), np.sin(angle)], axis=1))
    sinks = jnp.asarray(rng.normal(size=64), jnp.float32)
    scale = 512**-0.5
    for step in range(16):
        lengths = jnp.zeros((padded_sequences,), jnp.int32).at[0].set(position + step)
        main_lengths = lengths // 128
        new_kv = jnp.asarray(rng.normal(size=(1, 512)), jnp.bfloat16)
        positions = lengths[:1] - 1
        # Standalone RoPE donates its input; each arm needs its own buffer.
        q_ref = rope.qnorm_rope(jnp.array(q, copy=True), positions, cos_sin, eps=1e-6)
        kv_ref = rope.rope(jnp.array(new_kv, copy=True), positions, cos_sin)
        swa_output, reference_cache, normalizer, row_max = (
            mla_swa.mla_sliding_window_ragged_paged_attention(
                q_ref,
                kv_ref,
                reference_cache,
                lengths,
                pages,
                cu,
                distribution,
                sinks,
                sm_scale=scale,
                sliding_window=128,
                logical_page_size=128,
                num_kv_pages_per_block=(2, 2, 2),
                num_queries_per_block=(1, 32, 32),
                unnormalized_output=True,
            )
        )
        expected = mla.mla_ragged_paged_attention(
            q_ref,
            main,
            main_lengths,
            main_lengths,
            main_pages,
            cu,
            distribution,
            sinks,
            swa_output,
            normalizer,
            row_max,
            sm_scale=scale,
            num_kv_pages_per_block=(32, 32, 32),
            num_queries_per_block=(1, 32, 32),
        )
        actual, cache = deepseek_v4_attention_op._attention_hca(
            q,
            new_kv,
            cache,
            lengths,
            pages,
            cu,
            distribution,
            main,
            main_lengths,
            main_lengths,
            main_pages,
            cu,
            distribution,
            sinks,
            positions,
            cos_sin,
            eps=1e-6,
            sm_scale=scale,
            sliding_window=128,
            logical_page_size=128,
            swa_only=False,
            two_caches_same_buffer=False,
        )
        jax.block_until_ready((actual, cache, expected, reference_cache))
        assert np.isfinite(np.asarray(actual, np.float32)).all()
        np.testing.assert_array_equal(
            np.asarray(actual).view(np.uint8), np.asarray(expected).view(np.uint8)
        )
        np.testing.assert_array_equal(
            np.asarray(cache).view(np.uint8), np.asarray(reference_cache).view(np.uint8)
        )
        q = (q.astype(jnp.float32) + actual.astype(jnp.float32) * 0.01).astype(
            jnp.bfloat16
        )
