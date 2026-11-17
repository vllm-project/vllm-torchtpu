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
"""Dense fp4 decode MoE (keys moe_dense_fp4_max_tokens / moe_dense_fp4_shared)
against an XLA emulation of the GMM path's arithmetic (Pallas interpret mode,
tpu7x abstract mesh)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.experimental import moe_dense_expert_fp4 as D
from vllm_torchtpu.kernels.megablox.gmm_v2 import apply_act_fn, interleave_lane
from vllm_torchtpu.layers.core import fused_moe_gmm as F

TPU7X = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
E, H, I, TOPK, IS = 4, 256, 256, 2, 128


@pytest.fixture
def tpu():
    with use_abstract_mesh(TPU7X), pltpu.force_tpu_interpret_mode():
        yield


def _fp4(rng, shape):
    """fp4 K-major weights [E, K, N] with f32 per-128-row block scales."""
    w = rng.uniform(-1, 1, shape).astype(np.float32)
    e, k, n = shape
    blocks = w.reshape(e, k // 128, 128, n)
    scale = np.abs(blocks).max(axis=2, keepdims=True) / 6.0
    q = jnp.asarray(blocks / scale).astype(jnp.float4_e2m1fn).reshape(shape)
    return q, jnp.asarray(scale.astype(np.float32))  # [E, K/128, 1, N]


def _inputs(m, seed, local_ids=True):
    rng = np.random.default_rng(seed)
    w1, s1 = _fp4(rng, (E, H, 2 * I))
    w2, s2 = _fp4(rng, (E, I, H))
    x = jnp.asarray(rng.standard_normal((m, H)).astype(np.float32)).astype(
        jnp.bfloat16
    )
    ids = np.argsort(rng.random((m, E)), axis=1)[:, :TOPK].astype(np.int32)
    if not local_ids:
        ids[0, 0] = -1  # a route off this rank
    tw = rng.uniform(0.1, 1, (m, TOPK)).astype(np.float32)
    tw[ids < 0] = 0
    return x, w1, w2, s1, s2, jnp.asarray(tw).astype(jnp.bfloat16), jnp.asarray(ids)


def _gmm_emulation(x, w1, w2, s1, s2, tw, ids, activation="swigluoai"):
    """The GMM path per token and expert: f32 block partials scaled per
    128-row block, bf16 activation, bf16 expert output, f32 weighted sum."""

    def proj(v, w, s):
        k = w.shape[0]
        blocks = v.astype(jnp.float32).reshape(v.shape[0], k // 128, 128)
        wf = w.astype(jnp.float32).reshape(k // 128, 128, -1)
        return jnp.einsum("mbk,bkn,bn->mn", blocks, wf, s[:, 0, :])

    out = jnp.zeros((x.shape[0], H), jnp.float32)
    for e in range(E):
        gu = proj(x, w1[e], s1[e])
        act = apply_act_fn(interleave_lane(gu[:, :I], gu[:, I:]), activation)
        y = proj(act.astype(jnp.bfloat16), w2[e], s2[e]).astype(jnp.bfloat16)
        coeff = jnp.sum(jnp.where(ids == e, tw.astype(jnp.float32), 0), axis=1)
        out += coeff[:, None] * y.astype(jnp.float32)
    return out.astype(jnp.bfloat16)


def _close(a, b, rel=2e-2):
    a = np.asarray(a.astype(jnp.float32))
    b = np.asarray(b.astype(jnp.float32))
    # Same roundings, other f32 sum order: a bf16 ulp or two per element.
    np.testing.assert_allclose(a, b, rtol=rel, atol=rel * np.abs(b).max())


@pytest.mark.parametrize("m,seed", [(8, 0), (16, 1), (5, 2)])
def test_dense_fp4_matches_gmm_arithmetic(tpu, m, seed):
    x, w1, w2, s1, s2, tw, ids = _inputs(m, seed, local_ids=seed != 1)
    assert D.can_use_dense_expert_fp4(x, w1, w2, s1, s2, activation="swigluoai")
    got = D.dense_expert_moe_fp4(x, w1, w2, s1, s2, tw, ids, activation="swigluoai")
    _close(got, _gmm_emulation(x, w1, w2, s1, s2, tw, ids))


def _shared(seed):
    rng = np.random.default_rng(100 + seed)
    w1 = jnp.asarray(rng.uniform(-1, 1, (H, 2 * IS)).astype(np.float32)).astype(
        jnp.float8_e4m3fn
    )
    w2 = jnp.asarray(rng.uniform(-1, 1, (IS, H)).astype(np.float32)).astype(
        jnp.float8_e4m3fn
    )
    s1 = jnp.asarray(rng.uniform(0.01, 0.02, (1, 2 * IS)).astype(np.float32))
    s2 = jnp.asarray(rng.uniform(0.01, 0.02, (1, H)).astype(np.float32))
    return D.SharedExpert(w1, s1, w2, s2)


def test_dense_fp4_shared_matches_xla_shared(tpu):
    x, w1, w2, s1, s2, tw, ids = _inputs(8, 3)
    shared = _shared(0)
    assert D.can_use_dense_expert_fp4(
        x, w1, w2, s1, s2, activation="swigluoai", shared=shared
    )
    got = D.dense_expert_moe_fp4(
        x, w1, w2, s1, s2, tw, ids, activation="swigluoai", shared=shared
    )
    routed = D.dense_expert_moe_fp4(x, w1, w2, s1, s2, tw, ids, activation="swigluoai")
    want = F.shared_expert_mlp(x, shared, routed)
    _close(got, want)


def test_fused_moe_func_dispatch(tpu):
    """Key off: fused_moe_func never reaches the dense fp4 kernel; on, steps
    of up to moe_dense_fp4_max_tokens tokens return its result."""
    x, w1, w2, s1, s2, tw, ids = _inputs(8, 4)
    calls = []
    real = F.dense_expert_moe_fp4

    def spy(*a, **k):
        calls.append(a[0].shape[0])
        return real(*a, **k)

    args = (x, w1, w2, s1, s2, None, None, tw, ids)
    kw = dict(topk=TOPK, activation="swigluoai", use_ep=False, use_sparse_core=False)
    try:
        F.dense_expert_moe_fp4 = spy
        jax.clear_caches()
        try:
            F.fused_moe_func(*args, **kw)
        except Exception:
            pass  # the GMM path does not run in this interpreter
        assert calls == []
        F.set_dense_fp4(8, False)
        jax.clear_caches()
        got = F.fused_moe_func(*args, **kw)
        assert calls == [8]
        _close(got, _gmm_emulation(x, w1, w2, s1, s2, tw, ids))
        F.set_dense_fp4(4, False)  # 8 tokens > 4: GMM path
        jax.clear_caches()
        try:
            F.fused_moe_func(*args, **kw)
        except Exception:
            pass
        assert calls == [8]
    finally:
        F.dense_expert_moe_fp4 = real
        F.set_dense_fp4(0, False)
        jax.clear_caches()


def test_set_dense_fp4_rejects_negative():
    with pytest.raises(ValueError, match="moe_dense_fp4_max_tokens"):
        F.set_dense_fp4(-1, False)
