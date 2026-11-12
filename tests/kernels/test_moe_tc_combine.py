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
"""TensorCore MoE slab combine vs the v3 combine arithmetic (interpret mode)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import moe_tc_combine as K

TOPK = 4
E = 128


def _routes(t, local, seed):
    rng = np.random.default_rng(seed)
    ids = np.argsort(rng.random((t, E)), axis=1)[:, :TOPK]
    valid = ids < local
    keys = np.where(valid.reshape(-1), ids.reshape(-1), local)
    order = np.argsort(keys, kind="stable")
    pos = np.empty_like(order)
    pos[order] = np.arange(order.size)
    w = jnp.asarray(rng.uniform(0.05, 1, (t, TOPK)).astype(np.float32)).astype(
        jnp.bfloat16
    )
    return ids, valid, order, pos.astype(np.int32), w, rng


def _v3_reference(y, pos, w, valid):
    """f32 products, summed in top-k order from +0.0, one bf16 rounding."""
    yf = np.asarray(y.astype(jnp.float32))
    wf = np.asarray(w.astype(jnp.float32))
    t = valid.shape[0]
    out = np.zeros((t, yf.shape[1]), np.float32)
    for i in range(t):
        acc = np.zeros(yf.shape[1], np.float32)
        for k in range(TOPK):
            if valid[i, k]:
                acc = acc + (yf[pos[i * TOPK + k]] * wf[i, k]).astype(np.float32)
        out[i] = acc
    return np.asarray(jnp.asarray(out).astype(jnp.bfloat16)).view(np.uint16)


@pytest.mark.parametrize(
    "t,local,seed,scap,xcap,expect_overflow",
    [
        (512, 16, 0, 64, 128, False),
        (1024, 16, 1, 64, 128, False),
        (768, 8, 5, 64, 128, False),
        (512, 32, 4, 128, 1024, False),
        (512, 32, 3, 8, 128, True),  # more slabs than staged
        (512, 32, 6, 64, 4, True),  # more extra routes than staged
    ],
)
def test_slab_combine_bitwise(t, local, seed, scap, xcap, expect_overflow):
    h = 256
    _, valid, _, pos, w, rng = _routes(t, local, seed)
    count = int(valid.sum())
    y = (rng.standard_normal((t * TOPK, h)) * 3).astype(np.float32)
    y[count:] = np.nan  # rows past the computed prefix are never written
    y[0, :7] = -0.0
    yb = jnp.asarray(y).astype(jnp.bfloat16)
    out, overflow = K.tc_slab_combine(
        yb,
        jnp.asarray(pos),
        w.reshape(-1),
        jnp.asarray(valid.reshape(-1)),
        topk=TOPK,
        tb=256,
        scap=scap,
        xcap=xcap,
        interpret=True,
    )
    assert bool(overflow) == expect_overflow
    if not expect_overflow:
        ref = _v3_reference(yb, pos, w, valid)
        np.testing.assert_array_equal(np.asarray(out).view(np.uint16), ref)


def test_row_movement_keys_default_off(monkeypatch):
    fm = pytest.importorskip("vllm_torchtpu.layers.core.fused_moe_gmm")
    assert (fm._TC_COMBINE_MIN_TOKENS, fm._V3_PAYLOAD_SORT) == (0, False)
    assert fm._FLAT_ROUTING_MIN_TOKENS == 0
    assert fm._ep_combine() is fm.ragged_gather_reduce
    try:
        monkeypatch.setattr(fm.envs, "RAGGED_GATHER_REDUCE_VERSION", "v3")
        fm.set_row_movement(4096, True, 1024)
        assert (fm._TC_COMBINE_MIN_TOKENS, fm._FLAT_ROUTING_MIN_TOKENS) == (4096, 1024)
        assert fm._ep_combine().keywords == {"payload_sort": True}
        monkeypatch.setattr(fm.envs, "RAGGED_GATHER_REDUCE_VERSION", "v2")
        fm.set_row_movement(0, True, 0)  # the payload sort needs the v3 combine
        assert fm._V3_PAYLOAD_SORT is False
        with pytest.raises(ValueError):
            fm.set_row_movement(-1, False, 0)
    finally:
        fm.set_row_movement()
    assert fm._ep_combine() is fm.ragged_gather_reduce
