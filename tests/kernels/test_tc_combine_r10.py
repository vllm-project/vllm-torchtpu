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
"""r10 TensorCore EP combine (key moe_tc_combine) against an f64 reference
(Pallas interpret mode)."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import tc_combine as K
from vllm_torchtpu.layers.core import fused_moe_gmm as F

TOPK, E = 8, 128


def _routes(t, local, seed):
    rng = np.random.default_rng(seed)
    ids = np.argsort(rng.random((t, E)), axis=1)[:, :TOPK]
    valid = ids < local
    keys = np.where(valid.reshape(-1), ids.reshape(-1), local)
    order = np.argsort(keys, kind="stable")
    pos = np.empty_like(order)
    pos[order] = np.arange(order.size)
    sizes = np.bincount(keys, minlength=local + 1)[:local].astype(np.int32)
    w = rng.uniform(0.05, 1, (t, TOPK)).astype(np.float32)
    return valid, pos.astype(np.int32), sizes, w, rng


@pytest.mark.parametrize("t,local,seed", [(256, 16, 0), (512, 16, 1), (512, 8, 2)])
def test_tc_combine_vs_f64(t, local, seed):
    h = 256
    valid, pos, sizes, w, rng = _routes(t, local, seed)
    count = int(sizes.sum())
    y = rng.standard_normal((t * TOPK, h)).astype(np.float32)
    y[count:] = np.nan  # rows past the computed prefix are never written
    yb = jnp.asarray(y).astype(jnp.bfloat16)
    wb = jnp.asarray(w).astype(jnp.bfloat16)
    assert K.supported(t, h, TOPK, local, jnp.bfloat16)
    out = K.tc_combine(
        yb,
        jnp.asarray(pos),
        wb.reshape(-1),
        jnp.asarray(valid.reshape(-1)),
        jnp.asarray(sizes),
        topk=TOPK,
        interpret=True,
    )
    yf = np.asarray(yb.astype(jnp.float32), np.float64)
    wf = np.asarray(wb.astype(jnp.float32), np.float64)
    ref = np.zeros((t, h))
    for i in range(t):
        for k in range(TOPK):
            if valid[i, k]:
                ref[i] += wf[i, k] * yf[pos[i * TOPK + k]]
    got = np.asarray(out.astype(jnp.float32), np.float64)
    assert np.isfinite(got).all()
    # f32 accumulation of exact bf16 x bf16 products, one bf16 rounding:
    # within one bf16 ulp (2^-8 relative) of the exact sum, plus slack for
    # cancellation near zero.
    np.testing.assert_allclose(got, ref, rtol=2**-8, atol=1e-2)


def test_unsupported_shapes():
    assert not K.supported(300, 256, TOPK, 16, jnp.bfloat16)  # not % 256
    assert not K.supported(256, 256, TOPK, 16, jnp.float32)


def test_key_excludes_the_slab_combine():
    try:
        F.set_row_movement(tc_combine_min_tokens=4096)
        with pytest.raises(ValueError, match="moe_tc_combine"):
            F.set_tc_combine(True)
        F.set_row_movement()
        F.set_tc_combine(True)
        assert F._R10_TC_COMBINE
    finally:
        F.set_tc_combine(False)
        F.set_row_movement()
