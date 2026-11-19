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
"""The grouped shared expert on a host-device mesh: each rank's rows come back
as the whole expert applied to them, whatever the group size.

Needs at least 4 devices; CPU virtual devices are enough
(XLA_FLAGS=--xla_force_host_platform_device_count=8).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

from vllm_torchtpu.kernels.fused_moe.grouped_shared_expert import (
    blocked_down,
    grouped_shared_expert_sharded,
    silu_and_mul,
)

AX = "d"
FP8 = jnp.float8_e4m3fn


def _member(w13, s13, w2, s2, n, group, m):
    """Member m's channels: gate/up columns and down rows of its blocks."""
    c = w13.shape[1] // 2 // group
    gate, up = w13[:, m * c : (m + 1) * c], w13[:, w13.shape[1] // 2 + m * c :][:, :c]
    sg, su = s13[m * c : (m + 1) * c], s13[s13.shape[0] // 2 + m * c :][:c]
    per = s2.shape[0] // group
    return (
        jnp.concatenate([gate, up], 1),
        jnp.concatenate([sg, su]),
        w2[m * c : (m + 1) * c],
        s2[m * per : (m + 1) * per],
    )


@pytest.mark.parametrize("group", [1, 2, 4])
def test_each_rank_gets_the_whole_expert_on_its_rows(group):
    n = len(jax.devices())
    if n < 4 or n % group:
        pytest.skip("needs 4+ devices divisible by the group")
    mesh = Mesh(np.array(jax.devices()), (AX,))
    k, inter, hid, t, blocks = 32, 64, 32, 8, 8
    ks = jax.random.split(jax.random.key(0), 5)
    w13 = (jax.random.normal(ks[0], (k, 2 * inter)) * 32).astype(FP8)
    s13 = jax.random.uniform(ks[1], (2 * inter,), minval=0.5, maxval=1.5) / 32
    w2 = (jax.random.normal(ks[2], (inter, hid)) * 32).astype(FP8)
    s2 = jax.random.uniform(ks[3], (blocks, hid), minval=0.5, maxval=1.5) / 32
    x = jax.random.normal(ks[4], (n * t, k)).astype(jnp.bfloat16)

    want = blocked_down(silu_and_mul(xla_quantized_matmul(x, w13, s13)), w2, s2)
    members = [_member(w13, s13, w2, s2, n, group, m) for m in range(group)]
    stacked = [
        jnp.concatenate([members[r % group][i] for r in range(n)]) for i in range(4)
    ]
    # As the served op calls it: global arrays, sharded on the mesh axis.
    got = jax.jit(grouped_shared_expert_sharded(mesh, AX, group))(x, *stacked)
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    # Same arithmetic up to fusion (group 1) and the members' bf16 partials
    # being summed in bf16 (group > 1).
    tol = 1e-3 if group == 1 else 1e-2
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < tol


def _block_quantized(key, k, n, blocks):
    """An e4m3 `[k, n]` weight with one f32 scale per (k-block, channel), and
    its dequantized f32 value."""
    w = jax.random.normal(key, (k, n)) * 0.02
    wb = w.reshape(blocks, k // blocks, n)
    s = jnp.max(jnp.abs(wb), axis=1) / 448.0
    q = (wb / s[:, None, :]).astype(FP8)
    deq = (q.astype(jnp.float32) * s[:, None, :]).reshape(k, n)
    return q.reshape(k, n), s, deq


def _per_block_w8a8(act, w2, s2):
    """Each k-block contracted on its own with its activation columns
    quantized per row, the partials summed in f32."""
    blocks = s2.shape[0]
    kb = w2.shape[0] // blocks
    out = jnp.zeros((act.shape[0], w2.shape[1]), jnp.float32)
    for b in range(blocks):
        rows = slice(b * kb, (b + 1) * kb)
        out += xla_quantized_matmul(act[:, rows], w2[rows], s2[b]).astype(jnp.float32)
    return out.astype(act.dtype)


@pytest.mark.parametrize("m,blocks,kb,n", [(37, 3, 32, 96), (64, 8, 128, 256)])
def test_blocked_down_is_one_dot_over_the_dequantized_weight(m, blocks, kb, n):
    ks = jax.random.split(jax.random.key(1), 2)
    q, s, deq = _block_quantized(ks[0], blocks * kb, n, blocks)
    act = jax.random.normal(ks[1], (m, blocks * kb)).astype(jnp.bfloat16)
    want = act.astype(jnp.float32) @ deq
    got = jax.jit(blocked_down)(act, q, s)
    assert got.dtype == jnp.bfloat16 and got.shape == (m, n)

    def err(y):
        y = np.asarray(y, np.float32)
        return np.linalg.norm(y - want) / np.linalg.norm(want)

    # Only the bf16 rounding of the weight and of the result.
    assert err(got) < 5e-3
    # No worse than contracting each block with an fp8-quantized activation.
    assert err(got) < err(_per_block_w8a8(act, q, s))


def test_blocked_down_refuses_a_weight_that_does_not_split_into_blocks():
    act = jnp.zeros((8, 96), jnp.bfloat16)
    with pytest.raises(ValueError):
        blocked_down(act, jnp.zeros((96, 32), FP8), jnp.ones((5, 32), jnp.float32))
