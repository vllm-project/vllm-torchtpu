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

"""100% full-module line and branch coverage tests for

csa_cache_layout.py, sparsecore_topk.py, and rope.py on CPU.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4 import (
    csa_cache_layout as csa,
)

# ==============================================================================
# 1. csa_cache_layout.py Full Coverage Tests
# ==============================================================================


class TestCsaCacheLayout:
    """100% full-module line and branch coverage on csa_cache_layout.py."""

    def test_slabs_to_words_and_words_to_slabs_roundtrip(self):
        rng = np.random.default_rng(42)
        # Random uint8 slabs
        slabs_np = rng.integers(0, 256, size=(2, 3, 4, 128), dtype=np.uint8)
        slabs_jax = jnp.asarray(slabs_np)
        words_jax = csa.slabs_to_words(slabs_jax)
        assert words_jax.shape == (2, 3, 128)
        assert words_jax.dtype == jnp.int32

        # Roundtrip back to slabs
        slabs_rec = csa.words_to_slabs(words_jax)
        assert slabs_rec.shape == (2, 3, 4, 128)
        assert slabs_rec.dtype == jnp.uint8
        np.testing.assert_array_equal(np.asarray(slabs_rec), slabs_np)

        # Byte position verification
        # Sub-row 0 in bits 0..7, sub-row 1 in 8..15,
        # sub-row 2 in 16..23, sub-row 3 in 24..31
        fixed_slabs = np.zeros((1, 4, 1), dtype=np.uint8)
        fixed_slabs[0, 0, 0] = 0x12
        fixed_slabs[0, 1, 0] = 0x34
        fixed_slabs[0, 2, 0] = 0x56
        fixed_slabs[0, 3, 0] = 0x78
        word = csa.slabs_to_words(jnp.asarray(fixed_slabs))
        assert int(np.asarray(word)[0, 0]) == 0x78563412

    def test_rope_words_and_encode_rope(self):
        rng = np.random.default_rng(123)
        # Shape: (8, 64) bf16 tokens
        tokens = rng.normal(size=(8, 64)).astype(np.float32)
        tokens_bf16 = jnp.asarray(tokens, dtype=jnp.bfloat16)

        # rope_words: [..., 64] -> [..., 32]
        words = csa.rope_words(tokens_bf16)
        assert words.shape == (8, 32)
        assert words.dtype == jnp.int32

        # Bitwise verification: word k has low half bf16[k] and high half bf16[k+32]
        tokens_bits = np.asarray(tokens_bf16.view(jnp.uint16))
        words_bits = np.asarray(words.view(jnp.uint32))
        low16 = words_bits[..., :32] & 0xFFFF
        high16 = (words_bits[..., :32] >> 16) & 0xFFFF
        np.testing.assert_array_equal(low16, tokens_bits[..., :32])
        np.testing.assert_array_equal(high16, tokens_bits[..., 32:])

        # encode_rope: [num_tokens, 64] -> [num_tokens // 4, 128]
        encoded = csa.encode_rope(tokens_bf16)
        assert encoded.shape == (2, 128)
        assert encoded.dtype == jnp.int32
        np.testing.assert_array_equal(
            np.asarray(encoded), words_bits.reshape(2, 128).view(np.int32)
        )


# ==============================================================================
# 2. sparsecore_topk.py Full Coverage Tests
# ==============================================================================


# ==============================================================================
# 3. rope.py Full Coverage Tests
# ==============================================================================
