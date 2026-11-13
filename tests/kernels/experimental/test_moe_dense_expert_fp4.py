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
"""TPU parity of the fp4 dense-expert kernel with the GMM path under EP."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental import moe_dense_expert_fp4 as kernel
from vllm_torchtpu.layers.core import fused_moe_gmm as wrapper

pytestmark = pytest.mark.tpu_test


def _inputs(m, h, i, e, topk, seed):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(size=(m, h)), jnp.bfloat16)
    w1 = jnp.asarray(rng.normal(size=(e, h, 2 * i)), jnp.float4_e2m1fn)
    w2 = jnp.asarray(rng.normal(size=(e, i, h)), jnp.float4_e2m1fn)
    s1 = jnp.asarray(rng.uniform(0.01, 0.05, size=(e, h // 128, 1, 2 * i)), jnp.float32)
    s2 = jnp.asarray(rng.uniform(0.01, 0.05, size=(e, i // 128, 1, h)), jnp.float32)
    # Global ids over 2 * e experts: this rank owns [e, 2e), the rest are remote.
    ids = jnp.asarray(
        np.stack([rng.choice(2 * e, topk, replace=False) for _ in range(m)]), jnp.int32
    )
    weights = jnp.asarray(rng.uniform(0.1, 1.0, size=(m, topk)), jnp.bfloat16)
    return x, w1, w2, s1, s2, weights, ids


def _moe(x, w1, w2, s1, s2, weights, ids, e, topk):
    return wrapper.fused_moe_func(
        x,
        w1,
        w2,
        s1,
        s2,
        None,
        None,
        weights,
        ids,
        jnp.asarray(e, jnp.int32),
        topk=topk,
        activation="swigluoai",
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=wrapper.resolve_onehot_permute_threshold(),
        skip_padded_tokens=True,
    )


@pytest.mark.parametrize("m,i", [(8, 1024), (13, 512), (16, 384)])
def test_dense_fp4_matches_gmm_path(monkeypatch, m, i):
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    h, e, topk = 512, 4, 4
    args = _inputs(m, h, i, e, topk, seed=m)
    monkeypatch.setattr(wrapper, "_DENSE_FP4_MAX_TOKENS", 0)
    jax.clear_caches()
    ref = np.asarray(_moe(*args, e, topk), np.float32)
    monkeypatch.setattr(wrapper, "_DENSE_FP4_MAX_TOKENS", 16)
    assert kernel.can_use_dense_expert_fp4(*args[:5], activation="swigluoai")
    jax.clear_caches()
    out = np.asarray(_moe(*args, e, topk), np.float32)
    np.testing.assert_allclose(out, ref, rtol=0, atol=5e-3 * np.abs(ref).max())


def test_dense_fp4_no_local_route_returns_zeros():
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    x, w1, w2, s1, s2, weights, _ = _inputs(8, 512, 512, 4, 2, seed=3)
    ids = jnp.full((8, 2), -1, jnp.int32)
    out = kernel.dense_expert_moe_fp4(x, w1, w2, s1, s2, weights, ids)
    assert not np.asarray(out, np.float32).any()
