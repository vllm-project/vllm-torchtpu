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
"""The expert-load counter behind a static expert placement
(`distributed/expert_placement.py`) counts exactly the routed ids."""

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.fused_moe.v2.router_ops import expert_load_counts


def test_expert_load_counts_equal_a_histogram_of_the_routed_ids():
    rng = np.random.default_rng(5)
    t, e, k = 300, 64, 8
    logits = jnp.asarray(rng.standard_normal((t, e)), jnp.float32)
    bias = jnp.asarray(rng.standard_normal(e) / 4, jnp.float32)
    valid = rng.random(t) > 0.25
    start = rng.integers(0, 1000, e).astype(np.int32)
    got = jax.jit(expert_load_counts, static_argnames="topk")(
        jnp.asarray(start), logits, bias, jnp.asarray(valid), topk=k
    )
    _, ids = jax.lax.top_k(jax.nn.sigmoid(logits) + bias, k)
    ids = np.asarray(ids)
    assert np.array_equal(
        np.sort(ids, 1),
        np.sort(
            np.argsort(-(1 / (1 + np.exp(-np.asarray(logits))) + np.asarray(bias)), 1)[
                :, :k
            ],
            1,
        ),
    )
    want = start + np.bincount(ids[valid].ravel(), minlength=e)
    assert got.dtype == jnp.int32
    assert np.array_equal(np.asarray(got), want)
