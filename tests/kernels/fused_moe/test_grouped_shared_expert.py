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
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.grouped_shared_expert import (
    blocked_down,
    grouped_shared_expert,
    silu_and_mul,
)
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

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
    groups = tuple(tuple(range(g, g + group)) for g in range(0, n, group))
    got = jax.jit(
        jax.shard_map(
            lambda *a: grouped_shared_expert(*a, axis=AX, groups=groups),
            mesh=mesh,
            in_specs=(P(AX),) * 5,
            out_specs=P(AX),
            check_vma=False,
        )
    )(x, *stacked)
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    # Same arithmetic up to fusion (group 1) and the members' bf16 partials
    # being summed in bf16 (group > 1).
    tol = 1e-3 if group == 1 else 1e-2
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < tol
