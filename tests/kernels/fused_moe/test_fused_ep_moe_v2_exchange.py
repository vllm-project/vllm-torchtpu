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
"""The routing payload's layout: lane-aligned sections and a lane-aligned width."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.fused_moe.v2 import layer  # noqa: E402

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "t_local,topk,e_total,quantized,width",
    [
        (288, 6, 384, True, 2560),  # DP8/DP16 C384-per-host: unchanged
        (144, 6, 384, True, 1536),  # DP32 at 24 per rank: unchanged
        (288, 6, 392, True, 2688),  # DP8 with one spare slot per rank
        (144, 6, 416, True, 1664),  # DP32 with one spare slot per rank
        (240, 3, 128, True, 1152),  # the drafter at DP8
        (120, 3, 160, True, 768),  # the drafter at DP32 with 5 slots
        (256, 4, 32, False, 1152),  # the kernel tests' bf16 shape
    ],
)
def test_payload_width_is_lane_aligned_and_old_widths_are_unchanged(
    t_local, topk, e_total, quantized, width
):
    idx, count_at, got = layer._exchange_layout(t_local, topk, e_total, quantized)
    assert got == width and got % layer.HIDDEN_LANE_BLOCK == 0
    assert (
        idx % layer.HIDDEN_LANE_BLOCK == 0 and count_at % layer.HIDDEN_LANE_BLOCK == 0
    )
    assert count_at + e_total <= got


@pytest.mark.parametrize("e_total", [384, 392, 416])
def test_pack_and_unpack_round_trip_with_a_padded_tail(e_total):
    t_local, topk, ep = 144, 6, 2
    lay = layer._exchange_layout(t_local, topk, e_total, True)
    rng = np.random.default_rng(e_total)
    pairs = [
        rng.integers(0, e_total, size=t_local * topk).astype(np.int32)
        for _ in range(ep)
    ]
    scales = [rng.integers(0, 2**30, size=t_local).astype(np.int32) for _ in range(ep)]
    counts = [np.bincount(p, minlength=e_total).astype(np.int32) for p in pairs]
    packed = [
        layer._pack_routing_exchange(
            jnp.asarray(p), jnp.asarray(s), jnp.asarray(c), lay
        )
        for p, s, c in zip(pairs, scales, counts)
    ]
    assert all(p.shape == (lay[-1],) for p in packed)
    exchange = jnp.stack(packed)
    got_pairs, got_bits, got_counts = layer._unpack_routing_exchange(
        exchange, lay, t_local, topk, e_total
    )
    np.testing.assert_array_equal(np.asarray(got_pairs), np.stack(pairs))
    np.testing.assert_array_equal(np.asarray(got_bits), np.concatenate(scales))
    np.testing.assert_array_equal(np.asarray(got_counts), np.stack(counts))
