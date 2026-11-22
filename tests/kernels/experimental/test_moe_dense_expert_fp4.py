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


def _moe(x, w1, w2, s1, s2, weights, ids, e, topk, shared=None):
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
        shared=shared,
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


@pytest.mark.parametrize("m,with_shared", [(40, False), (48, True), (64, True)])
def test_sparse_rows_match_dense(m, with_shared):
    """Each expert on its own routes' rows (16-row tiles, experts above 16
    rows included) is bit-identical to evaluating every row."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    h, i, e, topk = 512, 1024, 4, 4
    x, w1, w2, s1, s2, weights, ids = _inputs(m, h, i, e, topk, seed=m)
    local = np.where(np.asarray(ids) >= e, np.asarray(ids) - e, -1)
    heavy = ~(local == 1).any(axis=1) & (np.arange(m) % 2 == 0)
    local[heavy, 0] = 1  # expert 1 takes more than one 16-row tile
    ids = jnp.asarray(local, jnp.int32)
    shared = _shared(h, 256, seed=m) if with_shared else None
    args = (x, w1, w2, s1, s2, weights, ids)
    assert kernel.can_use_dense_expert_fp4(
        *args[:5], activation="swigluoai", shared=shared, sparse_rows=True
    )
    dense = kernel.dense_expert_moe_fp4(*args, shared=shared)
    sparse = kernel.dense_expert_moe_fp4(*args, shared=shared, sparse_rows=True)
    np.testing.assert_array_equal(
        np.asarray(sparse, np.float32), np.asarray(dense, np.float32)
    )


def test_sparse_rows_match_dense_with_f32_weights_and_repeated_ids():
    """A row naming one local expert twice adds it once with the summed
    weight, and f32 routing weights reach the scatter unrounded."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    m, h, i, e, topk = 48, 512, 1024, 4, 4
    x, w1, w2, s1, s2, _, ids = _inputs(m, h, i, e, topk, seed=7)
    local = np.where(np.asarray(ids) >= e, np.asarray(ids) - e, -1)
    local[5, :2] = 2
    local[9, :] = 3
    weights = jnp.asarray(
        np.random.default_rng(3).uniform(0.1, 1.0, size=(m, topk)), jnp.float32
    )
    args = (x, w1, w2, s1, s2, weights, jnp.asarray(local, jnp.int32))
    dense = kernel.dense_expert_moe_fp4(*args)
    sparse = kernel.dense_expert_moe_fp4(*args, sparse_rows=True)
    np.testing.assert_array_equal(
        np.asarray(sparse, np.float32), np.asarray(dense, np.float32)
    )


def _replica_choice(slots, table, groups, block):
    """Python model of the kernel's copy choice: slot -> (block, local)."""
    touched = sorted({int(s) for s in slots.ravel() if s >= 0})
    listed = {table[d][0] for d in range(len(table))}
    load = [0] * groups
    owner = {}
    for s in touched:
        if s not in listed:
            load[s // block] += 1
            owner[s] = (s // block, s % block)
    for s, home, copy, copy_local in table:
        if s in touched:
            to_copy = load[copy] < load[home]
            load[copy if to_copy else home] += 1
            owner[s] = (copy, copy_local) if to_copy else (home, s % block)
    return owner


@pytest.mark.parametrize("m,sparse_rows", [(8, False), (24, False), (48, True)])
def test_replicas_match_dense_on_the_chosen_copies(m, sparse_rows):
    """Each block computes exactly the routes the copy choice gives it, as
    the dense kernel does on those routes' local ids."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    groups, block, h, i, topk = 4, 8, 512, 512, 4
    rng = np.random.default_rng(m)
    # Two replicas per block: copies of its slots 0 and 3 on the next blocks.
    pairs = [(q * block, (q + 1) % groups) for q in range(groups)]
    pairs += [(q * block + 3, (q + 2) % groups) for q in range(groups)]
    used, table = [0] * groups, []
    for s, b in pairs:
        table.append((s, s // block, b, block + used[b]))
        used[b] += 1
    num_slots = groups * block
    hot = np.r_[np.full(block, 4.0), np.ones(num_slots - block)]
    slots = np.stack(
        [
            rng.choice(num_slots, topk, replace=False, p=hot / hot.sum())
            for _ in range(m)
        ]
    )
    slots[-2:] = -1
    x = jnp.asarray(rng.normal(size=(m, h)), jnp.bfloat16)
    weights = jnp.asarray(rng.uniform(0.1, 1.0, size=(m, topk)), jnp.bfloat16)
    w1 = rng.normal(size=(num_slots, h, 2 * i))
    w2 = rng.normal(size=(num_slots, i, h))
    s1 = rng.uniform(0.01, 0.05, size=(num_slots, h // 128, 1, 2 * i))
    s2 = rng.uniform(0.01, 0.05, size=(num_slots, i // 128, 1, h))
    owner = _replica_choice(slots, table, groups, block)
    served = np.zeros(slots.shape, int)
    for q in range(groups):
        own = list(range(q * block, (q + 1) * block))
        copies = [s for s, _, b, _ in table if b == q]
        idx = own + copies

        def local(a, dtype, idx=idx):
            return jnp.asarray(a[idx], dtype)

        weights_q = (
            local(w1, jnp.float4_e2m1fn),
            local(w2, jnp.float4_e2m1fn),
            local(s1, jnp.float32),
            local(s2, jnp.float32),
        )
        lids = np.full(slots.shape, -1)
        for (t, j), s in np.ndenumerate(slots):
            if s >= 0 and owner[s][0] == q:
                lids[t, j] = owner[s][1]
                served[t, j] += 1
        dense = kernel.dense_expert_moe_fp4(
            x, *weights_q, weights, jnp.asarray(lids, jnp.int32)
        )
        chosen = kernel.dense_expert_moe_fp4(
            x,
            *weights_q,
            weights,
            jnp.asarray(slots, jnp.int32),
            sparse_rows=sparse_rows,
            replicas=jnp.asarray(np.ravel(table), jnp.int32),
            block_index=jnp.asarray(q, jnp.int32),
            replica_groups=groups,
        )
        np.testing.assert_array_equal(
            np.asarray(chosen, np.float32), np.asarray(dense, np.float32)
        )
    assert (served[slots >= 0] == 1).all()


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
