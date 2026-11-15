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


def _shared(h, i_s, seed):
    rng = np.random.default_rng(seed)
    return kernel.SharedExpert(
        jnp.asarray(rng.normal(size=(h, 2 * i_s)) * 64, jnp.float8_e4m3fn),
        jnp.exp2(jnp.asarray(rng.integers(-14, -11, size=(1, 2 * i_s)), jnp.float32)),
        jnp.asarray(rng.normal(size=(i_s, h)) * 64, jnp.float8_e4m3fn),
        jnp.exp2(jnp.asarray(rng.integers(-12, -9, size=(1, h)), jnp.float32)),
    )


def _moe(
    x, w1, w2, s1, s2, weights, ids, e, topk, shared=None, use_ep=True, tp_experts=None
):
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
        jnp.asarray(e, jnp.int32) if use_ep else None,
        topk=topk,
        activation="swigluoai",
        use_ep=use_ep,
        use_sparse_core=True,
        onehot_moe_permute_threshold=wrapper.resolve_onehot_permute_threshold(),
        skip_padded_tokens=True,
        shared=shared,
        tp_experts=tp_experts,
    )


# i=2048 streams two 1024-column units per expert; the others one whole expert.
@pytest.mark.parametrize("m,i", [(8, 2048), (8, 1024), (13, 512), (16, 384)])
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


@pytest.mark.parametrize("with_shared", [False, True])
def test_dense_fp4_matches_gmm_path_without_ep(monkeypatch, with_shared):
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    h, i, e, topk = 512, 384, 8, 4
    x, w1, w2, s1, s2, weights, _ = _inputs(8, h, i, e, topk, seed=11)
    shared = _shared(h, 128, seed=11) if with_shared else None
    rng = np.random.default_rng(12)
    # TP-MoE: every expert is local and ids are global.
    ids = jnp.asarray(
        np.stack([rng.choice(e, topk, replace=False) for _ in range(8)]), jnp.int32
    )
    args = (x, w1, w2, s1, s2, weights, ids, e, topk, shared, False)
    monkeypatch.setattr(wrapper, "_DENSE_FP4_MAX_TOKENS", 0)
    jax.clear_caches()
    ref = np.asarray(_moe(*args), np.float32)
    monkeypatch.setattr(wrapper, "_DENSE_FP4_MAX_TOKENS", 16)
    jax.clear_caches()
    out = np.asarray(_moe(*args), np.float32)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-2 * np.abs(ref).max())


def test_dense_fp4_no_local_route_returns_zeros():
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    x, w1, w2, s1, s2, weights, _ = _inputs(8, 512, 512, 4, 2, seed=3)
    ids = jnp.full((8, 2), -1, jnp.int32)
    out = kernel.dense_expert_moe_fp4(x, w1, w2, s1, s2, weights, ids)
    assert not np.asarray(out, np.float32).any()


@pytest.mark.parametrize("m", [8, 13, 24])
def test_dense_fp4_shared_matches_gmm_plus_xla_shared(monkeypatch, m):
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    h, i, e, topk = 512, 512, 4, 4
    args = _inputs(m, h, i, e, topk, seed=m)
    shared = _shared(h, 256, seed=m)
    monkeypatch.setattr(wrapper, "_DENSE_FP4_MAX_TOKENS", 0)
    jax.clear_caches()
    ref = np.asarray(_moe(*args, e, topk, shared), np.float32)
    monkeypatch.setattr(wrapper, "_DENSE_FP4_MAX_TOKENS", 32)
    assert kernel.can_use_dense_expert_fp4(
        *args[:5], activation="swigluoai", shared=shared
    )
    jax.clear_caches()
    out = np.asarray(_moe(*args, e, topk, shared), np.float32)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-2 * np.abs(ref).max())


def test_dense_fp4_shared_without_local_route_is_the_shared_expert():
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    x, w1, w2, s1, s2, weights, _ = _inputs(8, 512, 512, 4, 2, seed=3)
    shared = _shared(512, 128, seed=3)
    ids = jnp.full((8, 2), -1, jnp.int32)
    out = kernel.dense_expert_moe_fp4(x, w1, w2, s1, s2, weights, ids, shared=shared)
    zeros = jnp.zeros_like(x)
    ref = np.asarray(wrapper.shared_expert_mlp(x, shared, zeros), np.float32)
    out = np.asarray(out, np.float32)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-2 * np.abs(ref).max())


@pytest.mark.parametrize("m,with_shared", [(8, False), (13, True), (32, True)])
def test_dual_layout_matches_ep(monkeypatch, m, with_shared):
    """Ranks emulated on one chip: the sum of the ranks' partials over the
    TP-sliced copy equals the EP ranks' sum, up to summation order."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    num_ranks, e, h, i, topk = 4, 2, 512, 512, 4
    x, w1, w2, s1, s2, weights, _ = _inputs(m, h, i, num_ranks * e, topk, seed=m)
    rng = np.random.default_rng(m)
    ids = jnp.asarray(
        np.stack([rng.choice(num_ranks * e, topk, replace=False) for _ in range(m)]),
        jnp.int32,
    )
    ep = [[w[r * e : (r + 1) * e] for w in (w1, w2, s1, s2)] for r in range(num_ranks)]
    sends = [wrapper.tp_slice_sends(*w, num_ranks) for w in ep]
    copies = [
        wrapper.tp_slices_from_received(
            *(
                jnp.concatenate([s[k][q * e : (q + 1) * e] for s in sends])
                for k in range(4)
            )
        )
        for q in range(num_ranks)
    ]
    # Each rank's slice of a TP-sliced shared expert, the same in both layouts.
    shared = [
        _shared(h, 128, seed=m + r) if with_shared else None for r in range(num_ranks)
    ]
    monkeypatch.setattr(wrapper, "_DENSE_FP4_MAX_TOKENS", 32)
    jax.clear_caches()

    def total(tp):
        out = 0
        for r in range(num_ranks):
            out = out + np.asarray(
                _moe(
                    x,
                    *ep[r],
                    weights,
                    ids,
                    r * e,
                    topk,
                    shared[r],
                    tp_experts=copies[r] if tp else None,
                ),
                np.float32,
            )
        return out

    monkeypatch.setattr(wrapper, "_DUAL_LAYOUT_MAX_TOKENS", 0)
    ref = total(tp=True)  # the cap keeps the EP path
    monkeypatch.setattr(wrapper, "_DUAL_LAYOUT_MAX_TOKENS", 32)
    jax.clear_caches()
    assert kernel.can_use_dense_expert_fp4(
        x, *copies[0], activation="swigluoai", shared=shared[0]
    )
    out = total(tp=True)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-2 * np.abs(ref).max())
