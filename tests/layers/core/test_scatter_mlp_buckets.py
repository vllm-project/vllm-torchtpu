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
"""The scattering expert kernel's 16-row bucket: bitwise the kernel without it
(each matmul row is computed alone, whatever the bucket height)."""

import functools
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.layers.core import fused_moe_gmm
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    quantize_to_native_fp4_kmajor,
    resolve_onehot_permute_threshold,
)
from vllm_torchtpu.layers.core.ragged_a2a_moe import DP_AXIS, TP_AXIS, ragged_a2a_moe

HIDDEN = 2048
INTER = 512
# An expert kernel with the 16-row bucket in a lowered program.
BUCKET16 = re.compile(r"gmm_v2_mlp_scatter-[A-Za-z0-9_\-]*-bm_16_32_64-")


def _require_eight_tpu_devices():
    devices = jax.devices()
    if devices[0].platform != "tpu" or len(devices) < 8:
        pytest.skip("needs 8 TPU devices")
    return devices[:8]


def _inputs(tp_size, num_tokens, num_experts, *, topk, skew):
    num_ranks = 8
    key = jax.random.key(num_ranks * 100 + num_tokens + 7 * skew)
    kx, kl, k1, k2 = jax.random.split(key, 4)
    engines = num_ranks // tp_size
    x = (jax.random.normal(kx, (engines, 1, num_tokens, HIDDEN)) / 4).astype(
        jnp.bfloat16
    )
    x = jnp.broadcast_to(x, (engines, tp_size, num_tokens, HIDDEN)).reshape(-1, HIDDEN)
    logits = jax.random.normal(kl, (engines, 1, num_tokens, num_experts))
    if skew:
        local = num_experts // num_ranks
        owner_dp = (jnp.arange(num_experts) // local) // tp_size
        logits = logits + jnp.where(owner_dp == 0, 4.0, 0.0)
    logits = jnp.broadcast_to(logits, (engines, tp_size, num_tokens, num_experts))
    weights, ids = jax.lax.top_k(jax.nn.softmax(logits.reshape(-1, num_experts)), topk)
    w1 = jax.random.normal(k1, (num_experts, HIDDEN, 2 * INTER)) / 8
    w2 = jax.random.normal(k2, (num_experts, INTER, HIDDEN)) / 8
    w1, s1 = quantize_to_native_fp4_kmajor(w1, 256, pack=False)
    w2, s2 = quantize_to_native_fp4_kmajor(w2, 256, pack=False)
    es = jnp.arange(num_ranks, dtype=jnp.int32) * (num_experts // num_ranks)
    return x, w1, w2, s1, s2, weights.astype(jnp.bfloat16), ids.astype(jnp.int32), es


# (dp, tp, tokens per engine, topk, rows x, routes x, padded tokens, skewed
# routing, combine). Every layout takes the capacity exchange with the plan
# kernel; with capacity-sized buffers the skewed one overflows into the
# full-size exchange.
LAYOUTS = [
    (2, 4, 92, 10, 1.2, 2.0, "", False, "owner"),
    (2, 4, 48, 10, 1.0, 2.0, "skip", False, "owner"),  # padded tokens skipped
    (4, 2, 48, 10, 1.15, 1.5, "", False, "owner"),
    (4, 2, 24, 2, 2.0, 0.0, "", False, "owner"),  # top-2
    (8, 1, 24, 10, 1.15, 1.5, "", False, "owner"),
    (4, 2, 48, 10, 1.15, 1.5, "", False, "streamed"),
    (8, 1, 24, 10, 1.15, 1.5, "", False, "streamed"),
    (8, 1, 24, 10, 1.15, 1.5, "", True, "owner"),  # overflows: full size
]


@pytest.mark.multichip
@pytest.mark.parametrize("whole_group", [False, True])
@pytest.mark.parametrize("layout", range(len(LAYOUTS)))
def test_bucket16_keeps_every_output_bit(monkeypatch, layout, whole_group):
    """Every rank's result bitwise the exchange with the 32/64-row buckets
    only: TP 1 / 2 / 4, top-10 and top-2, padded tokens skipped, owner and
    streamed combine, capacity and whole-group sizes."""
    dp_size, tp_size, num_tokens, topk, factor, route_factor, padded, skew, combine = (
        LAYOUTS[layout]
    )
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    x, w1, w2, s1, s2, weights, ids, es = _inputs(
        tp_size, num_tokens, 64, topk=topk, skew=skew
    )
    if padded:
        weights = weights.reshape(8, num_tokens, topk).at[:, -5:].set(0)
        weights = weights.reshape(-1, topk)
    stacked = tuple(
        a.reshape(dp_size, tp_size, -1, *a.shape[1:])
        for a in (x, w1, w2, s1, s2, weights, ids, es)
    )
    owner = combine == "owner"
    common = dict(
        mesh=mesh,
        topk=topk,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(
            dp_size * num_tokens * topk
        ),
        skip_padded_tokens=padded == "skip",
        capacity_factor=factor,
        route_factor=route_factor,
        row_gather=True,
        row_scatter=True,
        streamed_combine=not owner,
        owner_combine=owner,
        owner_tail=owner,
        sorted_plan=True,
        routing_dma=True,
        send_index_rows_minor=True,
        plan_kernel=True,
        whole_group=whole_group,
    )
    kwargs = {}
    if owner:
        rng = np.random.default_rng(3)
        kwargs["shared"] = jnp.asarray(
            rng.standard_normal((dp_size, tp_size, num_tokens, HIDDEN)) / 8,
            jnp.bfloat16,
        )
    run = jax.jit(functools.partial(ragged_a2a_moe, **common))
    lowered = run.lower(*stacked, **kwargs).as_text()
    assert BUCKET16.search(lowered)
    got = np.asarray(run(*stacked, **kwargs))
    jax.clear_caches()
    monkeypatch.setattr(fused_moe_gmm, "_SCATTER_MLP_BUCKETS", (32,))
    base = np.asarray(
        jax.jit(functools.partial(ragged_a2a_moe, **common))(*stacked, **kwargs)
    )
    assert np.abs(base.astype(np.float32)).max() > 0
    np.testing.assert_array_equal(got.view(np.uint16), base.view(np.uint16))
