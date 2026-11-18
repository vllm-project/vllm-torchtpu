# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
#   tests/megakernel/test_structured_outputs.py
"""Tests for sampling mask_logits unpacking and sharding."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.models.kimi_k3.megakernel.program import TP_SIZE
from vllm_torchtpu.models.kimi_k3.megakernel.sampling import mask_logits

_NEG_INF = -1e30
VOCAB_WORDS = 163840 // 32


def test_mask_logits_unpacking_and_sharding() -> None:
    """Unpack bitmask shards per TP rank and set disallowed logits to -1e30."""
    width = 163840 // TP_SIZE
    rows = 3
    rng = np.random.default_rng(0)
    mask = rng.integers(-(2**31), 2**31, (rows, VOCAB_WORDS), dtype=np.int32)
    logits = rng.normal(size=(rows, width)).astype(np.float32)

    masked = jax.jit(mask_logits)
    for rank in (0, 1, 3, TP_SIZE - 1):
        out = np.asarray(masked(jnp.asarray(logits), jnp.asarray(mask), rank))
        words = mask[:, rank * width // 32 : (rank + 1) * width // 32]
        allowed = ((words[:, :, None] >> np.arange(32, dtype=np.int32)) & 1) != 0
        allowed = allowed.reshape(rows, width)
        assert ((out == _NEG_INF) == ~allowed).all()
        np.testing.assert_array_equal(out[allowed], logits[allowed])


def test_mask_logits_all_ones_is_identity_and_finite_when_rank_empty() -> None:
    """Preserve finite logits unchanged when all-ones, and mask out other ranks."""
    width = 163840 // TP_SIZE
    logits = np.random.default_rng(1).normal(size=(2, width)).astype(np.float32)
    masked = jax.jit(mask_logits)
    ones = np.full((2, VOCAB_WORDS), -1, dtype=np.int32)
    for rank in (0, TP_SIZE - 1):
        np.testing.assert_array_equal(
            np.asarray(masked(jnp.asarray(logits), jnp.asarray(ones), rank)), logits
        )
    # A schema allowing only rank 3's tokens fully masks every other rank:
    # finite sentinel, never NaN or -inf (the per-rank normalizers stay sane).
    narrow = np.zeros((2, VOCAB_WORDS), np.int32)
    narrow[:, 3 * width // 32 : 4 * width // 32] = -1
    out = np.asarray(masked(jnp.asarray(logits), jnp.asarray(narrow), 0))
    assert (out == _NEG_INF).all() and np.isfinite(out).all()
    out = np.asarray(masked(jnp.asarray(logits), jnp.asarray(narrow), 3))
    np.testing.assert_array_equal(out, logits)
