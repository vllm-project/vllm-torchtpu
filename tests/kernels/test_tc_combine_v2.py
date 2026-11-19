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
"""TPU_MOE_TC_COMBINE_V2: the Pallas plan and the boundary-tile reuse give
`tc_combine`'s plan and output bit for bit (interpret mode)."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import tc_combine as K

TOPK = 4
E = 128


def _case(t, local, seed, h=128):
    rng = np.random.default_rng(seed)
    ids = np.argsort(rng.random((t, E)), axis=1)[:, :TOPK]
    valid = ids < local
    keys = np.where(valid.reshape(-1), ids.reshape(-1), local)
    order = np.argsort(keys, kind="stable")
    pos = np.empty_like(order)
    pos[order] = np.arange(order.size)
    gs = np.bincount(ids[valid], minlength=local).astype(np.int32)
    w = jnp.asarray(rng.uniform(0.05, 1, (t * TOPK,)).astype(np.float32)).astype(
        jnp.bfloat16
    )
    count = int(valid.sum())
    y = (rng.standard_normal((t * TOPK, h)) * 3).astype(np.float32)
    y[count:] = np.nan  # rows past the computed prefix are never written
    return (
        jnp.asarray(y).astype(jnp.bfloat16),
        jnp.asarray(pos.astype(np.int32)),
        w,
        jnp.asarray(valid.reshape(-1)),
        jnp.asarray(gs),
    )


@pytest.mark.parametrize("t,local,seed", [(512, 16, 0), (768, 8, 2)])
def test_plan_v2_matches_plan(t, local, seed):
    _, pos, _, valid, gs = _case(t, local, seed)
    tiles, counts, rows = K._plan(pos, valid, gs, t, TOPK, K.TOKEN_BLOCK)
    nb = t // K.TOKEN_BLOCK
    maxt = K.max_tiles(K.TOKEN_BLOCK, TOPK, local)
    kt = lambda a: jnp.pad(a.reshape(t, TOPK).T, ((0, K._KPAD - TOPK), (0, 0)))
    ends = jnp.cumsum(gs).reshape(-1, 1)
    t2, src, c2, r2 = K._plan_v2(
        kt(pos), kt(valid.astype(jnp.int32)), ends, t, TOPK, K.TOKEN_BLOCK, True
    )
    c2 = np.asarray(c2)[:, 0, 0]
    np.testing.assert_array_equal(c2, np.asarray(counts))  # lane 0
    t2 = np.asarray(t2)[:, 0, :maxt]
    np.testing.assert_array_equal(t2.reshape(-1), np.asarray(tiles))
    np.testing.assert_array_equal(np.asarray(r2)[:TOPK].T, np.asarray(rows))
    # A reused slot names a slot of the previous block holding the same tile.
    src = np.asarray(src)[:, 0, :maxt]
    tl = np.asarray(tiles).reshape(nb, maxt)
    assert (src[0] < 0).all()
    for b in range(1, nb):
        for i in range(c2[b]):
            if src[b, i] >= 0:
                assert src[b, i] < c2[b - 1] and tl[b - 1, src[b, i]] == tl[b, i]
    assert (src >= 0).sum() > 0


@pytest.mark.parametrize("reuse", ["", "vec", "dma"])
@pytest.mark.parametrize("t,local,seed", [(512, 16, 0), (768, 8, 4)])
def test_v2_bitwise(t, local, seed, reuse):
    x, pos, w, valid, gs = _case(t, local, seed)
    want = K.tc_combine(x, pos, w, valid, gs, topk=TOPK, interpret=True)
    got = K.tc_combine_v2(x, pos, w, valid, gs, topk=TOPK, reuse=reuse, interpret=True)
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16)
    )
