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
"""Ragged all-to-all EP exchange against the all-gather + reduce-scatter path."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    quantize_to_native_fp4_kmajor,
    resolve_onehot_permute_threshold,
)
from vllm_torchtpu.layers.core.ragged_a2a_moe import (
    DP_AXIS,
    TP_AXIS,
    exchange_need,
    ragged_a2a_moe,
)

HIDDEN = 2048
INTER = 512
TOPK = 10


def _need_reference(ids, ep_rank, tp_size, local_num_experts):
    dp_size, num_tokens, _ = ids.shape
    need = np.zeros((dp_size, num_tokens, dp_size), dtype=bool)
    for s in range(dp_size):
        for t in range(num_tokens):
            for e in ids[s, t]:
                owner = e // local_num_experts
                if e >= 0 and owner % tp_size == ep_rank % tp_size:
                    need[s, t, owner // tp_size] = True
    return need


@pytest.mark.cpu_test
@pytest.mark.parametrize("dp_size,tp_size", [(2, 4), (4, 2), (8, 1)])
def test_exchange_need_matches_brute_force(dp_size, tp_size):
    local = 4
    rng = np.random.default_rng(dp_size)
    ids = np.stack(
        [
            np.stack(
                [
                    rng.choice(dp_size * tp_size * local, TOPK, replace=False)
                    for _ in range(6)
                ]
            )
            for _ in range(dp_size)
        ]
    ).astype(np.int32)
    for ep_rank in range(dp_size * tp_size):
        got = exchange_need(
            jnp.asarray(ids),
            jnp.int32(ep_rank),
            tp_size=tp_size,
            local_num_experts=local,
        )
        np.testing.assert_array_equal(
            np.asarray(got), _need_reference(ids, ep_rank, tp_size, local)
        )


def _require_eight_tpu_devices():
    devices = jax.devices()
    if devices[0].platform != "tpu" or len(devices) < 8:
        pytest.skip("needs 8 TPU devices")
    return devices[:8]


def _allgather_moe_local(
    x, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, es, **kw
):
    """The default path: DP all-gather, local experts, DP reduce-scatter."""
    gathered = [
        jax.lax.all_gather(a, DP_AXIS, tiled=True) for a in (x, topk_weights, topk_ids)
    ]
    out = fused_moe_func(
        gathered[0],
        w1,
        w2,
        w1_scale,
        w2_scale,
        None,
        None,
        gathered[1],
        gathered[2],
        experts_start=es[0],
        topk=TOPK,
        use_ep=True,
        **kw,
    )
    return jax.lax.psum_scatter(out, DP_AXIS, scatter_dimension=0, tiled=True)


def _inputs(num_ranks, tp_size, num_tokens, num_experts, *, skew):
    key = jax.random.key(num_ranks * 100 + num_tokens + skew)
    kx, kl, k1, k2 = jax.random.split(key, 4)
    engines = num_ranks // tp_size
    # TP ranks of one engine hold the same tokens.
    x = (jax.random.normal(kx, (engines, 1, num_tokens, HIDDEN)) / 4).astype(
        jnp.bfloat16
    )
    x = jnp.broadcast_to(x, (engines, tp_size, num_tokens, HIDDEN)).reshape(-1, HIDDEN)
    logits = jax.random.normal(kl, (engines, 1, num_tokens, num_experts))
    if skew:
        # Route most tokens to the experts of DP position 0 only.
        local = num_experts // num_ranks
        owner_dp = (jnp.arange(num_experts) // local) // tp_size
        logits = logits + jnp.where(owner_dp == 0, 4.0, 0.0)
    logits = jnp.broadcast_to(logits, (engines, tp_size, num_tokens, num_experts))
    logits = logits.reshape(-1, num_experts)
    weights, ids = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), TOPK)
    w1 = jax.random.normal(k1, (num_experts, HIDDEN, 2 * INTER)) / 8
    w2 = jax.random.normal(k2, (num_experts, INTER, HIDDEN)) / 8
    w1, s1 = quantize_to_native_fp4_kmajor(w1, 256, pack=False)
    w2, s2 = quantize_to_native_fp4_kmajor(w2, 256, pack=False)
    es = (jnp.arange(num_ranks, dtype=jnp.int32) * (num_experts // num_ranks)).reshape(
        num_ranks
    )
    return x, w1, w2, s1, s2, weights.astype(jnp.bfloat16), ids.astype(jnp.int32), es


@pytest.mark.multichip
@pytest.mark.parametrize(
    "dp_size,tp_size,num_tokens,skew",
    [
        (2, 4, 12, False),  # proxy layout, decode batch below the one-hot cap
        (2, 4, 48, False),  # proxy layout, 960 routed rows (SparseCore permute)
        (2, 4, 12, True),
        (4, 2, 12, False),
        (8, 1, 12, False),
        (8, 1, 12, True),
    ],
)
def test_ragged_a2a_matches_allgather_path(dp_size, tp_size, num_tokens, skew):
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    num_experts = 64
    args = _inputs(8, tp_size, num_tokens, num_experts, skew=skew)
    kw = dict(
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(),
    )
    spec = P((DP_AXIS, TP_AXIS))
    reference = jax.jit(
        jax.shard_map(
            functools.partial(_allgather_moe_local, **kw),
            mesh=mesh,
            in_specs=(spec,) * 8,
            out_specs=spec,
            check_vma=False,
        )
    )(*args)
    stacked = tuple(a.reshape(dp_size, tp_size, -1, *a.shape[1:]) for a in args)
    ragged = jax.jit(functools.partial(ragged_a2a_moe, mesh=mesh, topk=TOPK, **kw))(
        *stacked
    ).reshape(reference.shape)
    reference, ragged = (
        np.asarray(reference, np.float32),
        np.asarray(ragged, np.float32),
    )
    assert np.isfinite(ragged).all()
    assert np.abs(reference).max() > 0
    if dp_size == 2:
        # Two partials per token: the sum is the same in either order.
        np.testing.assert_array_equal(ragged, reference)
    else:
        np.testing.assert_allclose(ragged, reference, rtol=2e-2, atol=2e-2)
