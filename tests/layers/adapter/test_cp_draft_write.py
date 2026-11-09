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
"""The drafter's run-wise KV write against the XLA scatter it replaces."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.adapter.cp_draft_attention import write_runs


@pytest.mark.parametrize(
    "shape, dtype",
    [((40, 128, 2, 4, 128), jnp.float8_e4m3fn), ((30, 128, 1, 4, 128), jnp.bfloat16)],
)
def test_write_runs_matches_scatter(shape, dtype):
    rng = np.random.default_rng(0)
    num_pages = shape[0]
    cache = jnp.asarray(rng.standard_normal(shape), jnp.float32).astype(dtype)
    # A 700-token prefill from position 300, then five 8-token decode rows;
    # pages past 36 and every other block of the decode rows are dropped.
    pages = [7 + (300 + t) // 128 for t in range(700)]
    slots = [(300 + t) % 128 for t in range(700)]
    for s in range(5):
        base = int(rng.integers(0, 3 * 128))
        for t in range(8):
            pos = base + t
            pages.append(1 + 2 * s if (pos // 128) % 2 == 0 else num_pages)
            slots.append(pos % 128)
    pages = jnp.asarray(pages, jnp.int32)
    pages = jnp.where(pages > 36, num_pages, pages)
    slots = jnp.asarray(slots, jnp.int32)
    rows = jnp.asarray(
        rng.standard_normal((slots.shape[0], *shape[2:])), jnp.float32
    ).astype(dtype)
    expected = cache.at[pages, slots].set(rows, mode="drop")
    got = write_runs(
        cache, rows, pages, slots, max_runs=slots.shape[0] // 128 + 16, interpret=True
    )
    np.testing.assert_array_equal(
        np.asarray(got.astype(jnp.float32)), np.asarray(expected.astype(jnp.float32))
    )
