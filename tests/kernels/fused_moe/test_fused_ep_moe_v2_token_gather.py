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
"""shard_token_gather: each live slab row holds the token of the pair routed to it."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v2 import host

EP, T_LOCAL, G_LOCAL = 4, 64, 8


@pytest.mark.parametrize("topk", [8, 6], ids=["pow2_topk", "non_pow2_topk"])
def test_should_store_each_routed_pairs_token_in_its_slab_row(topk):
    e_total = EP * G_LOCAL
    n_tokens = EP * T_LOCAL
    rng = np.random.default_rng(7)
    idx = np.stack([rng.permutation(e_total)[:topk] for _ in range(n_tokens)])
    stride = host.ragged_stride_bound(n_tokens, topk, e_total, 128)
    routing = host.build_routing_tables(
        jnp.asarray(idx.astype(np.int32)),
        e_total=e_total,
        ep=EP,
        t_local=T_LOCAL,
        block=host.routing_block(T_LOCAL, topk),
        tile_m=128,
        shard_stride=stride,
    )
    me = 1
    rows = np.asarray(host.local_slab_rows(routing, me, shard_stride=stride))
    expected = np.zeros(stride, np.int64)
    live = rows < stride
    np.add.at(expected, rows[live], np.flatnonzero(live) // topk)
    lo, hi = host.gather_clamp_bounds(n_tokens)

    got = host.shard_token_gather(routing, me, shard_stride=stride)

    assert live.sum() > 0
    np.testing.assert_array_equal(np.asarray(got), np.clip(expected, lo, hi))
