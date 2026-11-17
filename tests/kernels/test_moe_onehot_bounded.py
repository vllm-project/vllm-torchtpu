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
"""Bounded EP one-hot permute / combine (key moe_bounded_onehot) against the
dense one-hot forms of fused_moe_gmm (Pallas interpret mode, tpu7x)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.megablox import moe_onehot_bounded as B
from vllm_torchtpu.layers.core import fused_moe_gmm as F

TPU7X = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
TOPK, H = 8, 1024


@pytest.fixture
def tpu():
    with use_abstract_mesh(TPU7X), pltpu.force_tpu_interpret_mode():
        yield


def _routing(t, local, num_experts, seed):
    """Counting-sort layout of an EP rank owning experts [0, local)."""
    rng = np.random.default_rng(seed)
    ids = np.argsort(rng.random((t, num_experts)), axis=1)[:, :TOPK]
    valid = ids < local
    keys = np.where(valid, ids, local).reshape(-1)
    order = np.argsort(keys, kind="stable")
    revert = np.empty_like(order)
    revert[order] = np.arange(order.size)
    w = rng.uniform(0.05, 1, (t, TOPK)).astype(np.float32)
    return valid, revert.astype(np.int32), int(valid.sum()), w, rng


def test_shapes_admitted(tpu):
    assert B.can_use_bounded_onehot(128, 128 * TOPK, 6144)
    assert not B.can_use_bounded_onehot(96, 96 * TOPK, 6144)


@pytest.mark.parametrize("t,local,seed", [(128, 16, 0), (256, 16, 1), (128, 128, 2)])
def test_permute_matches_dense_onehot(tpu, t, local, seed):
    _, revert, count, _, rng = _routing(t, local, 128, seed)
    hidden = rng.standard_normal((t, H)).astype(np.float32)
    hidden[3, 5] = np.nan
    hidden[7, 9] = np.inf
    hb = jnp.asarray(hidden).astype(jnp.bfloat16)
    got = B.bounded_onehot_permute(hb, jnp.asarray(revert), jnp.array([count]))
    onehot = (
        jax.nn.one_hot(revert.reshape(t, TOPK), t * TOPK, dtype=jnp.bfloat16)
        .sum(axis=1)
        .T
    )
    want = onehot @ F._zero_non_finite(hb)
    np.testing.assert_array_equal(
        np.asarray(got[:count]).view(np.uint16), np.asarray(want[:count]).view(np.uint16)
    )


@pytest.mark.parametrize("t,local,seed", [(128, 16, 0), (256, 16, 3), (128, 128, 4)])
def test_combine_matches_dense_onehot(tpu, t, local, seed):
    valid, revert, count, w, rng = _routing(t, local, 128, seed)
    routed = rng.standard_normal((t * TOPK, H)).astype(np.float32)
    routed[count:] = np.nan  # never written by the GMM
    rb = jnp.asarray(routed).astype(jnp.bfloat16)
    wb = jnp.asarray(w).astype(jnp.bfloat16)
    vm = jnp.asarray(valid)
    got = B.bounded_onehot_combine(
        rb, jnp.asarray(revert), jnp.where(vm, wb, 0), jnp.array([count])
    )
    # fused_moe_gmm.moe_gmm's dense combine.
    computed = jnp.arange(t * TOPK) < count
    rows = jnp.where(computed[:, None], rb, 0)
    onehot = jax.nn.one_hot(revert.reshape(t, TOPK), t * TOPK, dtype=jnp.bfloat16)
    combine = (onehot * wb[..., None] * vm[..., None]).sum(axis=1)
    want = combine @ rows
    assert np.isfinite(np.asarray(got.astype(jnp.float32))).all()
    # Same nonzero products, f32 accumulation: equal up to the sum order.
    np.testing.assert_allclose(
        np.asarray(got.astype(jnp.float32)),
        np.asarray(want.astype(jnp.float32)),
        rtol=1e-2,
        atol=1e-2,
    )


def test_key_gates_dispatch(tpu):
    rows = jnp.zeros((128, H), jnp.bfloat16)
    try:
        assert not F._use_bounded_onehot(128, 128 * TOPK, rows)
        F.set_bounded_onehot(True)
        assert F._use_bounded_onehot(128, 128 * TOPK, rows)
        assert not F._use_bounded_onehot(128, 128 * TOPK, rows.astype(jnp.float32))
    finally:
        F.set_bounded_onehot(False)
