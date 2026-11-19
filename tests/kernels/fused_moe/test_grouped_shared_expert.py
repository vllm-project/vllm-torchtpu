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

from vllm_torchtpu.kernels.fused_moe.grouped_shared_expert import (
    blocked_down,
    grouped_shared_expert_sharded,
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
    # As the served op calls it: global arrays, sharded on the mesh axis.
    got = jax.jit(grouped_shared_expert_sharded(mesh, AX, group))(x, *stacked)
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    # Same arithmetic up to fusion (group 1) and the members' bf16 partials
    # being summed in bf16 (group > 1).
    tol = 1e-3 if group == 1 else 1e-2
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < tol


def _reference_blocked_down_f32(
    act: jax.Array, w2: jax.Array, s2: jax.Array
) -> np.ndarray:
    blocks, n = s2.shape
    kb = w2.shape[0] // blocks
    out = jnp.zeros((act.shape[0], n), jnp.float32)
    for b in range(blocks):
        rows = slice(b * kb, (b + 1) * kb)
        out = out + xla_quantized_matmul(act[:, rows], w2[rows], s2[b]).astype(
            jnp.float32
        )
    return np.asarray(out.astype(act.dtype), dtype=np.float32)


@pytest.mark.parametrize(
    "m,blocks,n",
    [
        (64, 8, 128),
        (64, 16, 256),
        (128, 24, 256),
    ],
    ids=["single_step_b8", "single_step_b16", "multi_step_b24"],
)
def test_should_match_reference_blocked_down_when_pallas_kernel_runs(m, blocks, n):
    kb = 128
    k = blocks * kb
    ks = jax.random.split(jax.random.key(blocks + m), 3)
    # Anti-cancellation: non-uniform per-block scale magnitudes and zero rows
    # catch wrong block indexing, dropped scales, or 0/0 NaNs.
    block_gain = jnp.geomspace(0.02, 5.0, blocks, dtype=jnp.float32)
    act = (
        jax.random.normal(ks[0], (m, k), jnp.float32)
        * jnp.repeat(block_gain, kb)[None, :]
    ).astype(jnp.bfloat16)
    act = act.at[0].set(jnp.bfloat16(0.0))
    act = act.at[-1, :kb].set(jnp.bfloat16(0.0))
    w2 = (jax.random.normal(ks[1], (k, n), jnp.float32) * 16.0).astype(FP8)
    s2 = (
        jax.random.uniform(ks[2], (blocks, n), jnp.float32, 0.2, 2.0)
        / block_gain[:, None]
    )

    want = _reference_blocked_down_f32(act, w2, s2)
    got_arr = jax.jit(
        lambda a, w, s: blocked_down(a, w, s, interpret=jax.default_backend() != "tpu")
    )(act, w2, s2)
    got = np.asarray(got_arr, dtype=np.float32)

    assert got_arr.dtype == act.dtype
    assert np.isfinite(got).all()
    assert np.all(got[0] == 0.0)
    rel_l2 = np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-8)
    assert rel_l2 < 5e-3


def test_should_fall_back_cleanly_when_shapes_or_dtype_unaligned():
    ks = jax.random.split(jax.random.key(99), 3)
    act = jax.random.normal(ks[0], (32, 256), jnp.float32).astype(jnp.bfloat16)
    w2_bf16 = jax.random.normal(ks[1], (256, 128), jnp.float32).astype(jnp.bfloat16)
    s2 = jax.random.uniform(ks[2], (4, 128), jnp.float32, 0.5, 1.5)

    want = _reference_blocked_down_f32(act, w2_bf16, s2)
    got = np.asarray(
        blocked_down(act, w2_bf16, s2, interpret=jax.default_backend() != "tpu"),
        dtype=np.float32,
    )
    assert np.allclose(got, want, atol=1e-3, rtol=1e-3)


def test_should_fall_back_to_xla_when_wide_dense_tile_exceeds_vmem_cap(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    moe_jaxpr = jax.make_jaxpr(blocked_down)(
        jax.ShapeDtypeStruct((1024, 2048), jnp.bfloat16),
        jax.ShapeDtypeStruct((2048, 6144), FP8),
        jax.ShapeDtypeStruct((16, 6144), jnp.float32),
    )
    dense_jaxpr = jax.make_jaxpr(blocked_down)(
        jax.ShapeDtypeStruct((1024, 12288), jnp.bfloat16),
        jax.ShapeDtypeStruct((12288, 6144), FP8),
        jax.ShapeDtypeStruct((16, 6144), jnp.float32),
    )
    assert "pallas_call" in str(moe_jaxpr)
    assert "pallas_call" not in str(dense_jaxpr)
