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
"""The few-token MoE kernel against plain-JAX references.

Runs on the host in Pallas interpret mode, so it needs no TPU. The
expert-parallel test needs several devices; on a CPU host run it with
XLA_FLAGS=--xla_force_host_platform_device_count=8.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.moe_lowtok import (
    can_use_moe_lowtok,
    moe_lowtok,
    moe_lowtok_reference,
)
from vllm_torchtpu.kernels.moe_lowtok.expert_parallel import (
    moe_lowtok_ep,
    moe_lowtok_ep_serves,
    route_softmax_topk,
)

FP8 = jnp.float8_e4m3fn


def _interpret():
    """Interpret mode off a TPU (queried lazily: collection must not
    initialize a JAX backend)."""
    return jax.default_backend() != "tpu"


def _weights(rng, experts, hidden, inter):
    w13 = jnp.asarray(
        np.clip(rng.standard_normal((experts, hidden, 2 * inter)) * 40, -448, 448),
        FP8,
    )
    w2 = jnp.asarray(
        np.clip(rng.standard_normal((experts, inter, hidden)) * 40, -448, 448), FP8
    )
    s13 = jnp.asarray(
        rng.random((experts, 1, 1, 2 * inter)).astype(np.float32)
        / (40 * np.sqrt(hidden))
    )
    s2 = jnp.asarray(
        rng.random((experts, 1, 1, hidden)).astype(np.float32) / (40 * np.sqrt(inter))
    )
    return w13, s13, w2, s2


def _operands(seed, tokens, experts, hidden, inter, topk):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(
        rng.standard_normal((tokens, hidden)).astype(np.float32), jnp.bfloat16
    )
    # Global ids over twice the shard's experts: about half the routed pairs
    # belong to this shard.
    ids = np.stack(
        [rng.choice(2 * experts, size=topk, replace=False) for _ in range(tokens)]
    )
    local = jnp.asarray(np.where(ids < experts, ids, -1).astype(np.int32))
    weights = rng.random((tokens, topk)).astype(np.float32)
    weights = jnp.asarray(weights / weights.sum(axis=1, keepdims=True), jnp.bfloat16)
    return (x, local, weights, *_weights(rng, experts, hidden, inter))


@pytest.mark.parametrize("tokens", [1, 5, 32, 33])
@pytest.mark.parametrize("shape", [(8, 512, 256, 4), (4, 256, 512, 2)])
def test_matches_reference(tokens, shape):
    experts, hidden, inter, topk = shape
    ops = _operands(tokens * 7 + experts, tokens, experts, hidden, inter, topk)
    got = np.asarray(moe_lowtok(*ops, interpret=_interpret()), np.float32)
    want = np.asarray(moe_lowtok_reference(*ops), np.float32)
    assert got.shape == (tokens, hidden)
    assert np.abs(want).max() > 0
    scale = np.abs(want).max()
    np.testing.assert_allclose(got / scale, want / scale, atol=2e-2, rtol=0)


def test_no_local_expert_gives_zero():
    x, ids, weights, w13, s13, w2, s2 = _operands(3, 4, 8, 512, 256, 4)
    got = moe_lowtok(
        x, jnp.full_like(ids, -1), weights, w13, s13, w2, s2, interpret=_interpret()
    )
    assert np.all(np.asarray(got, np.float32) == 0)


def test_support_check():
    x, _, _, w13, s13, w2, s2 = _operands(0, 4, 8, 512, 256, 4)
    supported = dict(activation="silu")
    assert can_use_moe_lowtok(x, w13, w2, s13, s2, None, None, **supported)
    assert not can_use_moe_lowtok(x, w13, w2, s13, s2, None, None, activation="gelu")
    assert not can_use_moe_lowtok(
        x.astype(jnp.float32), w13, w2, s13, s2, None, None, **supported
    )
    assert not can_use_moe_lowtok(x, w13, w2, s13, s2, s13, None, **supported)
    # Block-scaled weights: more than one scale per output channel.
    blocked = jnp.concatenate([s13, s13], axis=1)
    assert not can_use_moe_lowtok(x, w13, w2, blocked, s2, None, None, **supported)
    # A gate|up width that does not split into two lane-aligned halves.
    _, _, _, w13_odd, s13_odd, w2_odd, s2_odd = _operands(0, 4, 8, 512, 128, 4)
    assert not can_use_moe_lowtok(
        x, w13_odd, w2_odd, s13_odd, s2_odd, None, None, **supported
    )


def _ep_mesh(ep):
    devices = jax.devices()
    if len(devices) < ep:
        pytest.skip(
            f"needs {ep} devices; on a CPU host set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=8"
        )
    return jax.sharding.Mesh(np.array(devices[:ep]), ("ep",))


def _dense_moe_reference(x, gating, w13, s13, w2, s2, topk, renormalize):
    """Every row through every selected expert, with the reference arithmetic
    and global expert ids: what the expert-parallel layer must return."""
    weights, ids = route_softmax_topk(gating, topk, renormalize)
    return moe_lowtok_reference(x, ids, weights, w13, s13, w2, s2)


@pytest.mark.parametrize("mesh_ep_ranks", [None, (2, 0, 3, 1)])
@pytest.mark.parametrize("rows_per_rank", [1, 3])
def test_expert_parallel_matches_dense_layer(mesh_ep_ranks, rows_per_rank):
    """The all-gather, kernel and reduce-scatter return every rank's own rows
    of the full MoE layer, including a mesh whose order differs from the EP
    rank order."""
    ep, local, hidden, inter, topk = 4, 4, 256, 256, 3
    mesh = _ep_mesh(ep)
    rng = np.random.default_rng(rows_per_rank)
    rows = ep * rows_per_rank
    x = jnp.asarray(
        rng.standard_normal((rows, hidden)).astype(np.float32), jnp.bfloat16
    )
    gating = jnp.asarray(rng.standard_normal((rows, ep * local)).astype(np.float32))
    w13, s13, w2, s2 = _weights(rng, ep * local, hidden, inter)
    want = np.asarray(
        _dense_moe_reference(x, gating, w13, s13, w2, s2, topk, True), np.float32
    )

    # The shard at mesh index i holds EP rank order[i]'s expert block.
    order = list(range(ep)) if mesh_ep_ranks is None else list(mesh_ep_ranks)
    blocks = np.concatenate([np.arange(r * local, (r + 1) * local) for r in order])
    got = moe_lowtok_ep(
        x,
        gating,
        jnp.arange(ep, dtype=jnp.int32).reshape(ep, 1),
        w13[blocks],
        w2[blocks],
        s13.reshape(ep * local, -1)[blocks],
        s2.reshape(ep * local, -1)[blocks],
        mesh=mesh,
        topk=topk,
        renormalize=True,
        mesh_ep_ranks=mesh_ep_ranks,
        interpret=_interpret(),
    )
    got = np.asarray(got, np.float32)
    scale = np.abs(want).max()
    # Each shard's partial is rounded to bf16 before the float32 sum.
    np.testing.assert_allclose(got / scale, want / scale, atol=3e-2, rtol=0)


def test_expert_parallel_selection():
    ep, local, hidden, inter = 4, 4, 512, 256
    rng = np.random.default_rng(0)
    w13, s13, w2, s2 = _weights(rng, ep * local, hidden, inter)
    s13, s2 = s13.reshape(ep * local, -1), s2.reshape(ep * local, -1)

    def serves(rows, max_tokens=16, groups=None, activation="silu"):
        x = jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16)
        return moe_lowtok_ep_serves(
            x,
            w13,
            w2,
            s13,
            s2,
            ep=ep,
            max_tokens=max_tokens,
            activation=activation,
            token_replica_groups=groups,
        )

    assert serves(16)
    assert serves(4, groups=((0,), (1,), (2,), (3,)))
    assert not serves(20)  # above the cap
    assert not serves(16, max_tokens=0)  # off
    assert not serves(8, groups=((0, 1), (2, 3)))  # replicated rows
    assert not serves(8, activation="gelu")
