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
"""Decode MoE dispatch kernel vs the XLA routing prep it replaces, bit for bit.

``interpret=True`` runs the kernel body on the host; it does not gate Mosaic
lowering.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox.moe_decode_dispatch import decode_dispatch
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    computed_rows,
    prepare_routed_gmm_inputs,
    route_token_onehot,
)


def _reference(hidden, ids, weights, num_experts, skip_padded_tokens):
    num_tokens, topk = ids.shape
    x, sizes, revert, flat_weights, valid, _ = prepare_routed_gmm_inputs(
        hidden,
        ids,
        weights,
        local_num_experts=num_experts,
        topk=topk,
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=2048,
        skip_padded_tokens=skip_padded_tokens,
    )
    computed = computed_rows(revert, valid, num_tokens * topk)
    weight = jnp.where(valid, flat_weights, 0)
    combine = jnp.where(
        route_token_onehot(revert, num_tokens, topk), weight[:, None, None], 0
    ).sum(axis=0)
    return x, sizes, combine, computed


@pytest.mark.parametrize("num_tokens", [1, 4, 8, 24, 32])
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.parametrize("skip_padded_tokens", [False, True])
@pytest.mark.parametrize("all_local", [False, True])
def test_decode_dispatch_matches_routed_inputs(
    num_tokens, topk, skip_padded_tokens, all_local
):
    num_experts, ep = 16, 8
    rng = np.random.default_rng(num_tokens * 31 + topk)
    hidden = jnp.asarray(rng.standard_normal((num_tokens, 256)), jnp.bfloat16)
    if all_local:  # every route local, with ties between tokens
        ids = rng.integers(0, num_experts, size=(num_tokens, topk))
    else:
        ids = rng.integers(0, ep * num_experts, size=(num_tokens, topk)) - num_experts
        ids = np.where((ids >= 0) & (ids < num_experts), ids, -1)
    weights = rng.random((num_tokens, topk), dtype=np.float32)
    weights[rng.random((num_tokens, topk)) < 0.2] = 0
    ids = jnp.asarray(ids, jnp.int32)
    weights = jnp.asarray(weights, jnp.bfloat16)

    got = decode_dispatch(
        hidden,
        ids,
        weights,
        num_experts=num_experts,
        skip_padded_tokens=skip_padded_tokens,
        interpret=True,
    )
    want = _reference(hidden, ids, weights, num_experts, skip_padded_tokens)
    for name, g, w in zip(("x", "group_sizes", "combine", "computed"), got, want):
        assert g.dtype == w.dtype and g.shape == w.shape, name
        np.testing.assert_array_equal(
            np.asarray(g).view(np.uint8), np.asarray(w).view(np.uint8), err_msg=name
        )
