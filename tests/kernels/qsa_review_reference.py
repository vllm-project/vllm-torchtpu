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
"""Selected-token reference with the original BF16 boundaries and 1% oracle."""

import math

import jax.numpy as jnp
import numpy as np


def check_reference(args, output, *, kv_heads, rows, ratio=4, page_size=1664):
    q, gate, blocks, lengths, table, cache, _ = map(np.asarray, args)
    result = np.asarray(output, np.float32)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[lengths == 0], 0)
    for row in rows:
        n = int(lengths[row])
        selected = (
            blocks[row, : min(n // ratio, 512), None] * ratio + np.arange(ratio)
        ).reshape(-1)
        selected = np.concatenate((selected, np.arange(n // ratio * ratio, n)))
        values = (
            cache[table[row, selected // page_size], selected % page_size]
            .reshape(len(selected), -1, q.shape[-1])
            .astype(np.float64)
        )
        wanted = []
        per_head = q.shape[1] // kv_heads
        for h in range(kv_heads):
            score = (
                q[row, h * per_head : (h + 1) * per_head].astype(np.float64)
                @ values[:, 2 * h].T
                / math.sqrt(q.shape[-1])
            )
            mass = np.exp(score - score.max(axis=1, keepdims=True))
            wanted.append(
                (mass / mass.sum(axis=1, keepdims=True)) @ values[:, 2 * h + 1]
            )
        expected = np.concatenate(wanted)
        # Match the existing two BF16 boundaries and original 1% threshold.
        weight = jnp.asarray(
            1 / (1 + np.exp(-gate[row].astype(np.float64))), jnp.bfloat16
        )
        expected = np.asarray(
            jnp.asarray(
                jnp.asarray(expected, jnp.bfloat16).astype(jnp.float32) * weight,
                jnp.bfloat16,
            ),
            np.float32,
        )
        assert np.linalg.norm(result[row] - expected) / np.linalg.norm(expected) < 0.01
