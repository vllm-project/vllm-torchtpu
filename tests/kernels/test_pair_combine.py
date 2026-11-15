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
"""pair_combine against the gather-sum it replaces (TP-sliced experts)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import pair_combine


def _case(num_tokens, hidden, experts, topk, padded, seed=0):
    """Expert-sorted routed rows as the non-EP GMM writes them (padded tokens'
    routes sorted last, their rows zero as gmm_v2 zero_initialize leaves them)."""
    rng = np.random.default_rng(seed)
    ids = np.stack(
        [rng.choice(experts, topk, replace=False) for _ in range(num_tokens)]
    )
    w = rng.uniform(0.1, 1.0, (num_tokens, topk)).astype(np.float32)
    w[num_tokens - padded :] = 0
    valid = (w != 0).ravel()
    keys = np.where(valid, ids.ravel(), experts)
    order = np.argsort(keys, kind="stable")
    revert = np.empty_like(order)
    revert[order] = np.arange(order.size)
    rows = rng.standard_normal((num_tokens * topk, hidden)).astype(np.float32)
    rows[int(valid.sum()) :] = 0
    x = jnp.asarray(rows, jnp.bfloat16)
    wb = jnp.asarray(w.ravel(), jnp.bfloat16)
    ref = (
        np.asarray(x.astype(jnp.float32))[revert]
        * np.asarray(wb.astype(jnp.float32))[:, None]
        * valid[:, None]
    )
    ref = ref.reshape(num_tokens, topk, hidden).sum(axis=1)
    return x, jnp.asarray(revert, jnp.int32), wb, jnp.asarray(valid), ref


def _check(out, ref):
    out = np.asarray(out.astype(jnp.float32))
    # f32 accumulation of exact bf16 products, rounded to bf16 once.
    np.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2 * np.abs(ref).max())


@pytest.mark.parametrize("padded", [0, 5])
def test_matches_gather_sum_interpret(padded):
    x, pos, w, valid, ref = _case(128, 256, 16, 4, padded)
    assert pair_combine.supported(128, x.shape[0], 256, 4, x.dtype, w.dtype)
    out = pair_combine.pair_combine(x, pos, w, valid, topk=4, interpret=True)
    _check(out, ref)


def test_admission():
    bf16 = jnp.bfloat16
    assert pair_combine.supported(8192, 32768, 6144, 4, bf16, bf16)
    assert not pair_combine.supported(8192, 32768, 6144, 4, bf16, jnp.float32)
    assert not pair_combine.supported(8200, 32800, 6144, 4, bf16, bf16)
    assert not pair_combine.supported(32768, 131072, 6144, 4, bf16, bf16)


@pytest.mark.tpu_test
@pytest.mark.parametrize("padded", [0, 37])
def test_matches_gather_sum_tpu(padded):
    if jax.devices()[0].platform != "tpu":
        pytest.skip("needs a TPU")
    x, pos, w, valid, ref = _case(8192, 6144, 128, 4, padded)
    _check(pair_combine.pair_combine(x, pos, w, valid, topk=4), ref)
