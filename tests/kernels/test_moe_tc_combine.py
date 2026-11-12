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
"""TensorCore MoE dispatch / combine (interpret mode, u32 pair-row inputs)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import moe_tc_combine as K

TOPK = 4
E = 32


def _pairs(a):
    u = np.asarray(a).view(np.uint16).astype(np.uint32)
    return jnp.asarray(u[0::2] | (u[1::2] << 16))


def _unpairs(p):
    p = np.asarray(p)
    out = np.empty((p.shape[0] * 2, p.shape[1]), np.uint16)
    out[0::2] = (p & 0xFFFF).astype(np.uint16)
    out[1::2] = (p >> 16).astype(np.uint16)
    return out


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
    "local,seed,xcap,expect_overflow",
    [(8, 0, 32, False), (32, 2, 256, False), (16, 3, 4, True)],
)
def test_combine_bitwise(local, seed, xcap, expect_overflow):
    t, h = 256, 256
    _, valid, _, pos, w, rng = _routes(t, local, seed)
    count = int(valid.sum())
    y = (rng.standard_normal((t * TOPK, h)) * 3).astype(np.float32)
    y[count:] = np.nan  # rows past the computed prefix are never read
    y[0, :7] = -0.0
    yb = jnp.asarray(y).astype(jnp.bfloat16)
    out, overflow = K.tc_combine(
        _pairs(yb),
        jnp.asarray(pos),
        w.reshape(-1),
        jnp.asarray(valid.reshape(-1)),
        topk=TOPK,
        tb=64,
        xcap=xcap,
        interpret=True,
    )
    assert bool(overflow) == expect_overflow
    if not expect_overflow:
        ref = _v3_reference(yb, pos, w, valid)
        np.testing.assert_array_equal(np.asarray(out).view(np.uint16), ref)


@pytest.mark.parametrize("count_override", [None, 0, 64, 65])
def test_row_gather_bitwise(count_override):
    t, h = 256, 256
    _, valid, order, _, _, rng = _routes(t, 8, 4)
    count = int(valid.sum()) if count_override is None else count_override
    x = jnp.asarray(rng.standard_normal((t, h)).astype(np.float32)).astype(jnp.bfloat16)
    rows = (order // TOPK).astype(np.int32)
    g = K.tc_row_gather(
        _pairs(x), jnp.asarray(rows), jnp.asarray([count], jnp.int32), tbd=64, interpret=True
    )
    got = _unpairs(g)[:count]
    want = np.asarray(x).view(np.uint16)[rows[:count]]
    np.testing.assert_array_equal(got, want)


def test_combine_metadata_first_route_and_extras():
    valid = np.array([[0, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 1]], bool)
    valid = np.tile(valid, (32, 1))  # 128 tokens, one block of 128
    pos = np.arange(valid.size, dtype=np.int32)
    w = jnp.ones(valid.size, jnp.bfloat16)
    pos0, _, cnt0, ex_pos, ex_t, _, nex, overflow = K.combine_metadata(
        jnp.asarray(pos), w, jnp.asarray(valid.reshape(-1)), topk=TOPK, tb=128, xcap=256
    )
    pos0 = np.asarray(pos0)
    assert pos0[0] == 1 and pos0[1] == -1 and pos0[2] == 8 and pos0[3] == 12
    assert int(cnt0[0]) == 96
    # extras in (token, k) order: token 0 -> k2; token 2 -> k1,k2,k3; token 3 -> k3
    assert int(nex[0]) == 32 * 5 and not bool(overflow)
    assert list(np.asarray(ex_pos)[:5]) == [2, 9, 10, 11, 15]
    assert list(np.asarray(ex_t)[:5]) == [0, 2, 2, 2, 3]
