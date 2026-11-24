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
"""TPU_MOE_ONEHOT_COMBINE_FLAT: the dense one-hot MoE combine built from the
flat revert indices (`fused_moe_gmm._flat_onehot_combine`) equals the
reshaped [T, topk] one-hot form bit for bit, alone and through `moe_gmm`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.layers.core import fused_moe_gmm


def _case(num_tokens, topk, seed):
    rng = np.random.default_rng(seed)
    rows = num_tokens * topk
    revert = jnp.asarray(rng.permutation(rows).astype(np.int32))
    w = jnp.asarray(rng.random((num_tokens, topk)), jnp.bfloat16)
    valid = jnp.asarray(rng.random((num_tokens, topk)) < 0.8)
    return revert, w, valid


def _reference(revert, w, valid):
    num_tokens, topk = w.shape
    onehot = jax.nn.one_hot(revert.reshape(num_tokens, topk), revert.size, dtype=w.dtype)
    return (onehot * w[..., None] * valid[..., None]).sum(axis=1)


@pytest.mark.parametrize("num_tokens,topk", [(8, 4), (96, 4), (160, 4), (384, 4), (5, 2)])
def test_flat_combine_matrix_bitwise(num_tokens, topk):
    revert, w, valid = _case(num_tokens, topk, num_tokens)
    ref = jax.jit(_reference)(revert, w, valid)
    new = jax.jit(fused_moe_gmm._flat_onehot_combine)(revert, w, valid)
    assert new.dtype == ref.dtype == jnp.bfloat16 and new.shape == ref.shape
    np.testing.assert_array_equal(
        np.asarray(new.view(jnp.uint16)), np.asarray(ref.view(jnp.uint16))
    )
    assert int(jnp.count_nonzero(new)) == int(jnp.count_nonzero(valid))  # control


@pytest.mark.parametrize("num_tokens", [8, 96])
def test_moe_gmm_flat_combine_bitwise(monkeypatch, num_tokens):
    topk, hidden = 4, 128
    rows = num_tokens * topk
    revert, w, valid = _case(num_tokens, topk, 7 + num_tokens)
    k1, k2 = jax.random.split(jax.random.key(num_tokens))
    gmm1 = jax.random.normal(k1, (rows, hidden), jnp.float32).astype(jnp.bfloat16)
    gmm2 = jax.random.normal(k2, (rows, hidden), jnp.float32).astype(jnp.bfloat16)
    monkeypatch.setattr(fused_moe_gmm, "get_packing_factor", lambda *a, **k: 1)

    def run():
        outputs = iter([gmm1, gmm2])
        monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper", lambda *a, **k: next(outputs))
        return fused_moe_gmm.moe_gmm(
            x=jnp.ones((rows, hidden), jnp.bfloat16),
            w1=jnp.ones((1, hidden, hidden), jnp.bfloat16),
            w1_scale=None,
            w1_bias=None,
            w2=jnp.ones((1, hidden, hidden), jnp.bfloat16),
            w2_scale=None,
            w2_bias=None,
            group_sizes=jnp.array([rows - 3], jnp.int32),
            argsort_revert_indices=revert,
            token_indices_sorted=jnp.zeros((rows,), jnp.int32),
            topk_weights_flat=w.reshape(-1),
            valid_mask_flat=valid.reshape(-1),
            activation="silu",
            num_tokens=num_tokens,
            topk=topk,
            use_ep=True,
            use_sparse_core=True,
            onehot_moe_permute_threshold=rows,
        )

    monkeypatch.setattr(envs, "TPU_MOE_ONEHOT_COMBINE_FLAT", False)
    ref = run()
    monkeypatch.setattr(envs, "TPU_MOE_ONEHOT_COMBINE_FLAT", True)
    called = []
    orig = fused_moe_gmm._flat_onehot_combine
    monkeypatch.setattr(
        fused_moe_gmm, "_flat_onehot_combine", lambda *a: called.append(1) or orig(*a)
    )
    new = run()
    assert called
    np.testing.assert_array_equal(
        np.asarray(new.view(jnp.uint16)), np.asarray(ref.view(jnp.uint16))
    )
