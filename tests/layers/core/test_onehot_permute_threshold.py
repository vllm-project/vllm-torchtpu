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
"""One-hot permute/combine against the SparseCore path at a decode batch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    quantize_to_native_fp4_kmajor,
)


@pytest.mark.parametrize("num_tokens", [96, 192])
def test_onehot_combine_matches_sparse_core_path(num_tokens):
    if jax.devices()[0].platform != "tpu":
        pytest.skip("needs a TPU")
    hidden, inter, topk, num_experts, local = 8192, 2048, 10, 512, 64
    experts_start = 2 * local
    kx, kl, k1, k2 = jax.random.split(jax.random.key(num_tokens), 4)
    x = (jax.random.normal(kx, (num_tokens, hidden)) / 4).astype(jnp.bfloat16)
    probs = jax.nn.softmax(jax.random.normal(kl, (num_tokens, num_experts)), axis=-1)
    weights, ids = jax.lax.top_k(probs, topk)
    w1, s1 = quantize_to_native_fp4_kmajor(
        jax.random.normal(k1, (local, hidden, 2 * inter)) / 8, 256, pack=False
    )
    w2, s2 = quantize_to_native_fp4_kmajor(
        jax.random.normal(k2, (local, inter, hidden)) / 8, 256, pack=False
    )

    def run(threshold):
        return np.asarray(
            jax.jit(
                lambda *a: fused_moe_func(
                    *a,
                    experts_start=jnp.int32(experts_start),
                    topk=topk,
                    activation="silu",
                    use_ep=True,
                    use_sparse_core=True,
                    onehot_moe_permute_threshold=threshold,
                )
            )(x, w1, w2, s1, s2, None, None, weights.astype(jnp.bfloat16), ids),
            np.float32,
        )

    sparse_core = run(0)
    onehot = run(num_tokens * topk)
    assert np.abs(sparse_core).max() > 0
    ulp = np.spacing(np.abs(sparse_core).astype(jnp.bfloat16).astype(np.float32))
    ulp = np.maximum(ulp, np.float32(2.0**-126))
    worst = float((np.abs(onehot - sparse_core) / ulp).max())
    differ = float((onehot != sparse_core).mean())
    print(
        f"tokens={num_tokens} max diff {worst:.2f} bf16 ulp, "
        f"{differ:.4%} elements differ"
    )
    assert worst <= 2.0
