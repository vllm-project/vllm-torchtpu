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
"""rebalance_ep_tp_pair_experts: this shard's remapped pairs are its row of the plan."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v2.layer import rebalance_ep_tp_pair_experts

EP, G_LOCAL, K_SPLIT, TOPK, T_LOCAL, N_BLOCKS = 16, 4, 2, 2, 32, 4
E_TOTAL = EP * G_LOCAL
G_PLAN = G_LOCAL + K_SPLIT + 1


def _skewed_pairs() -> np.ndarray:
    rng = np.random.default_rng(23)
    ids = rng.integers(0, E_TOTAL, size=(EP * T_LOCAL, TOPK), dtype=np.int32)
    ids[:, 0] = 3  # one hot expert on chip 0: Z-clones remote-host tokens
    ids[: EP * T_LOCAL // 2, 1] = 1  # local-host load on the same chip
    ids[5, 1] = E_TOTAL  # masked pairs keep their sentinels
    ids[EP * T_LOCAL - 3, 1] = -1
    return ids.reshape(EP, T_LOCAL * TOPK)


def _rebalance(pairs_g: jnp.ndarray, me: int):
    local_blocks = pairs_g[me].reshape(N_BLOCKS, -1)
    rows_by_dest = jnp.stack(
        [
            jnp.bincount(
                jnp.where((r >= 0) & (r < E_TOTAL), r, E_TOTAL), length=E_TOTAL + 1
            )[:E_TOTAL]
            for r in pairs_g
        ]
    ).astype(jnp.int32)
    block_hist = jnp.stack(
        [
            jnp.bincount(
                jnp.where((b >= 0) & (b < E_TOTAL), b, E_TOTAL), length=E_TOTAL + 1
            )[:E_TOTAL]
            for b in local_blocks
        ]
    ).astype(jnp.int32)
    return rebalance_ep_tp_pair_experts(
        pairs_g,
        local_blocks,
        block_hist,
        rows_by_dest,
        ep=EP,
        ep_tp_size=2,
        g_local=G_LOCAL,
        e_total=E_TOTAL,
        k_split=K_SPLIT,
        capacity=16,
        me=jnp.int32(me),
        t_local=T_LOCAL,
        topk=TOPK,
        z_clone=True,
        z_min_total=128,
        z_min_remote=64,
    )


@pytest.mark.parametrize("me", [0, 9, 15])
def test_should_remap_local_blocks_to_this_shards_row_of_the_plan(me):
    pairs_g = jnp.asarray(_skewed_pairs())

    new_pairs, new_local, *_ = _rebalance(pairs_g, me)

    np.testing.assert_array_equal(
        np.asarray(new_local).reshape(-1), np.asarray(new_pairs)[me]
    )


def test_should_move_host1_local_pairs_onto_the_z_slot():
    pairs_g = jnp.asarray(_skewed_pairs())

    _, new_local, _, _, _, z_push = _rebalance(pairs_g, me=9)

    assert int(np.asarray(z_push)[0]) == 3
    ids = np.asarray(new_local).reshape(-1)
    slots = ids[(ids >= 0) & (ids < EP * G_PLAN)] % G_PLAN
    assert int((slots == G_PLAN - 1).sum()) > 0
