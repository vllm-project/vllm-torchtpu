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
"""The dense fp4 MoE kernel's sparse-rows mode (TPU_MOE_DENSE_FP4_SPARSE_MAX_TOKENS)
in Pallas TPU interpret mode on the CPU, against tpu7x's VMEM and lane
geometry: bit-identical to the all-rows kernel on the same tokens, and close to
a plain-JAX evaluation with the GMM path's roundings."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental import moe_dense_expert_fp4 as k
from vllm_torchtpu.kernels.megablox.gmm_v2 import gate_up_act

_INFO = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
FP4 = jnp.float4_e2m1fn


@pytest.fixture(autouse=True)
def _tpu7x(monkeypatch):
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: _INFO)
    k.dense_expert_moe_fp4.clear_cache()
    yield
    k.dense_expert_moe_fp4.clear_cache()


def _experts(e, h, i, seed=0):
    rng = np.random.default_rng(seed)
    # fp4 e2m1 values: every half-integer in [-3, 3] is exact.
    w1 = jnp.asarray(rng.integers(-6, 7, size=(e, h, 2 * i)) / 2, jnp.float32)
    w2 = jnp.asarray(rng.integers(-6, 7, size=(e, i, h)) / 2, jnp.float32)
    s1 = jnp.asarray(rng.uniform(0.01, 0.02, (e, h // 128, 1, 2 * i)), jnp.float32)
    s2 = jnp.asarray(rng.uniform(0.01, 0.02, (e, i // 128, 1, h)), jnp.float32)
    return w1.astype(FP4), w2.astype(FP4), s1, s2


def _routing(m, e, topk, seed=1, off_rank=True):
    """Distinct ids per token; with ``off_rank`` some are -1 (other EP ranks)."""
    rng = np.random.default_rng(seed)
    extra = 2 if off_rank else 0
    ids = np.stack([rng.permutation(e + extra)[:topk] for _ in range(m)]) - extra
    ids = np.where(ids < 0, -1, ids).astype(np.int32)
    tw = rng.uniform(0.05, 1.0, size=(m, topk))
    return jnp.asarray(ids), jnp.asarray(tw, jnp.bfloat16)


def _tokens(m, h, seed=2):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(m, h)), jnp.bfloat16)


def _run(x, w1, w2, s1, s2, tw, ids, **kw):
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()):
        out = k.dense_expert_moe_fp4(x, w1, w2, s1, s2, tw, ids, **kw)
    return np.asarray(out.astype(jnp.float32))


def _reference(x, w1, w2, s1, s2, tw, ids, activation="swigluoai"):
    """The GMM path's roundings: bf16 activation before the down projection,
    bf16 expert output before its routing weight, f32 sum, bf16 result."""
    f32 = jnp.float32
    e, i, h = w2.shape
    acc = jnp.zeros(x.shape, f32)
    for ex in range(e):
        w1e = w1[ex].astype(f32) * jnp.repeat(s1[ex, :, 0, :], 128, axis=0)
        w2e = w2[ex].astype(f32) * jnp.repeat(s2[ex, :, 0, :], 128, axis=0)
        gu = jnp.dot(x.astype(f32), w1e, precision="highest")
        act = gate_up_act(gu[:, :i], gu[:, i:], activation).astype(jnp.bfloat16)
        out = jnp.dot(act.astype(f32), w2e, precision="highest")
        out = out.astype(jnp.bfloat16).astype(f32)
        coeff = jnp.sum(jnp.where(ids == ex, tw.astype(f32), 0), axis=1, keepdims=True)
        acc = acc + coeff * out
    return np.asarray(acc.astype(jnp.bfloat16).astype(f32))


@pytest.mark.parametrize(
    "m,h,i,e,topk",
    [
        (17, 256, 256, 8, 4),  # just above the dense cap; pads to 24 / 32
        (24, 256, 256, 8, 4),  # whole-expert units (chunk == i)
        (40, 256, 512, 8, 2),
        (100, 384, 1536, 16, 4),  # 768-column units, three slots
        (128, 256, 256, 128, 4),  # the expert lane cap
    ],
)
def test_sparse_rows_match_all_rows_kernel_bitwise(monkeypatch, m, h, i, e, topk):
    w1, w2, s1, s2 = _experts(e, h, i)
    ids, tw = _routing(m, e, topk)
    x = _tokens(m, h)
    assert k.can_use_dense_expert_fp4(
        x, w1, w2, s1, s2, activation="swigluoai", sparse_rows=True
    )
    dense = _run(x, w1, w2, s1, s2, tw, ids)
    sparse = _run(x, w1, w2, s1, s2, tw, ids, sparse_rows=True)
    assert np.isfinite(sparse).all()
    if k._rows_chunk(m, h, i, _INFO.vmem_capacity_bytes) == k._chunk(i):
        np.testing.assert_array_equal(sparse, dense)
    else:
        # Wider units group the f32 down-projection partials differently:
        # at most one bf16 rounding step apart. With the all-rows kernel's
        # unit width the rows mode is bit-identical.
        np.testing.assert_allclose(sparse, dense, rtol=2**-7, atol=2**-7)
        monkeypatch.setattr(k, "_rows_chunk", lambda m, h, i, cap: k._chunk(i))
        k.dense_expert_moe_fp4.clear_cache()
        same = _run(x, w1, w2, s1, s2, tw, ids, sparse_rows=True)
        np.testing.assert_array_equal(same, dense)
    ref = _reference(x, w1, w2, s1, s2, tw, ids)
    rel = np.linalg.norm(sparse - ref) / np.linalg.norm(ref)
    assert rel < 1e-2, rel


def test_sparse_rows_silu_and_empty_experts():
    m, h, i, e = 33, 256, 256, 8
    w1, w2, s1, s2 = _experts(e, h, i, seed=3)
    x = _tokens(m, h, seed=4)
    # Experts 0..3 only, some tokens entirely off this rank.
    ids = np.tile(np.array([[0, 3]], np.int32), (m, 1))
    ids[::5] = -1
    ids = jnp.asarray(ids)
    tw = jnp.full((m, 2), 0.5, jnp.bfloat16)
    dense = _run(x, w1, w2, s1, s2, tw, ids, activation="silu")
    sparse = _run(x, w1, w2, s1, s2, tw, ids, activation="silu", sparse_rows=True)
    np.testing.assert_array_equal(sparse, dense)
    np.testing.assert_array_equal(sparse[::5], 0)


def test_sparse_rows_no_local_route_is_zero():
    m, h, i, e = 20, 256, 256, 4
    w1, w2, s1, s2 = _experts(e, h, i)
    x = _tokens(m, h)
    ids = jnp.full((m, 4), -1, jnp.int32)
    tw = jnp.ones((m, 4), jnp.bfloat16)
    np.testing.assert_array_equal(_run(x, w1, w2, s1, s2, tw, ids, sparse_rows=True), 0)


def test_sparse_rows_nonfinite_token_stays_local():
    m, h, i, e = 24, 256, 256, 4
    w1, w2, s1, s2 = _experts(e, h, i)
    x = _tokens(m, h).at[5, 7].set(jnp.nan)
    ids, tw = _routing(m, e, 2, off_rank=False)
    tw = tw.at[9, 0].set(jnp.inf)
    out = _run(x, w1, w2, s1, s2, tw, ids, sparse_rows=True)
    assert np.isfinite(out).all()
    dense = _run(x, w1, w2, s1, s2, tw, ids)
    np.testing.assert_array_equal(out, dense)


def test_admission():
    h, i = 256, 256
    w1, w2, s1, s2 = _experts(4, h, i)
    x = _tokens(24, h)
    ok = dict(activation="swigluoai", sparse_rows=True)
    assert k.can_use_dense_expert_fp4(x, w1, w2, s1, s2, **ok)
    # More local experts than one vreg's lanes.
    big = (jax.ShapeDtypeStruct((129, h, 2 * i), FP4),)
    w2b = jax.ShapeDtypeStruct((129, i, h), FP4)
    s1b = jax.ShapeDtypeStruct((129, h // 128, 1, 2 * i), jnp.float32)
    s2b = jax.ShapeDtypeStruct((129, i // 128, 1, h), jnp.float32)
    assert not k.can_use_dense_expert_fp4(x, big[0], w2b, s1b, s2b, **ok)
    assert k.can_use_dense_expert_fp4(
        x, big[0], w2b, s1b, s2b, activation="swigluoai"
    )
    # Token buffers past VMEM (MiniMax-M3-sized hidden): no unit fits.
    xh = jax.ShapeDtypeStruct((4096, 3072), jnp.bfloat16)
    w1h = jax.ShapeDtypeStruct((8, 3072, 2 * 1536), FP4)
    w2h = jax.ShapeDtypeStruct((8, 1536, 3072), FP4)
    s1h = jax.ShapeDtypeStruct((8, 24, 1, 2 * 1536), jnp.float32)
    s2h = jax.ShapeDtypeStruct((8, 12, 1, 3072), jnp.float32)
    assert not k.can_use_dense_expert_fp4(xh, w1h, w2h, s1h, s2h, **ok)
    xs = jax.ShapeDtypeStruct((128, 3072), jnp.bfloat16)
    assert k.can_use_dense_expert_fp4(xs, w1h, w2h, s1h, s2h, **ok)
