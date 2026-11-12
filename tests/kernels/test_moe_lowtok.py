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
from vllm_torchtpu.kernels.moe_lowtok.kernel import vmem_required_bytes

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


@pytest.mark.parametrize("whole_expert", [False, True])
@pytest.mark.parametrize("tokens", [1, 5, 32, 33])
@pytest.mark.parametrize("shape", [(8, 512, 256, 4), (4, 256, 512, 2)])
def test_matches_reference(whole_expert, tokens, shape):
    experts, hidden, inter, topk = shape
    ops = _operands(tokens * 7 + experts, tokens, experts, hidden, inter, topk)
    got = np.asarray(
        moe_lowtok(*ops, whole_expert=whole_expert, interpret=_interpret()),
        np.float32,
    )
    want = np.asarray(moe_lowtok_reference(*ops), np.float32)
    assert got.shape == (tokens, hidden)
    assert np.abs(want).max() > 0
    scale = np.abs(want).max()
    tol = 2e-2 if _interpret() else 3e-2
    np.testing.assert_allclose(got / scale, want / scale, atol=tol, rtol=0)


@pytest.mark.tpu_test
@pytest.mark.parametrize("whole_expert", [False, True])
def test_tpu_whole_expert_matches_reference(whole_expert):
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    tokens, experts, hidden, inter, topk = 8, 8, 512, 256, 4
    ops = _operands(41, tokens, experts, hidden, inter, topk)
    got = np.asarray(
        moe_lowtok(*ops, whole_expert=whole_expert, interpret=False),
        np.float32,
    )
    want = np.asarray(moe_lowtok_reference(*ops), np.float32)
    scale = np.abs(want).max()
    np.testing.assert_allclose(got / scale, want / scale, atol=3e-2, rtol=0)


@pytest.mark.parametrize("whole_expert", [False, True])
def test_unrouted_nan_row_gives_zero(whole_expert):
    x, ids, weights, w13, s13, w2, s2 = _operands(9, 4, 8, 512, 256, 4)
    # Row 0 contains NaNs and routes nowhere locally while rows 1..3 hit local
    # experts: 0 * NaN must not poison row 0's output.
    x = x.at[0].set(jnp.nan)
    ids = ids.at[0].set(-1)
    ids = ids.at[1, 0].set(0)
    got = np.asarray(
        moe_lowtok(
            x,
            ids,
            weights,
            w13,
            s13,
            w2,
            s2,
            whole_expert=whole_expert,
            interpret=_interpret(),
        ),
        np.float32,
    )
    assert np.all(got[0] == 0)
    assert np.all(np.isfinite(got[1:]))


@pytest.mark.parametrize("whole_expert", [False, True])
def test_no_local_expert_gives_zero(whole_expert):
    x, ids, weights, w13, s13, w2, s2 = _operands(3, 4, 8, 512, 256, 4)
    got = moe_lowtok(
        x,
        jnp.full_like(ids, -1),
        weights,
        w13,
        s13,
        w2,
        s2,
        whole_expert=whole_expert,
        interpret=_interpret(),
    )
    assert np.all(np.asarray(got, np.float32) == 0)


@pytest.mark.parametrize("hit_experts", [[0], [3, 5], [0, 1, 3, 5, 7]])
def test_whole_expert_matches_split_stream(hit_experts):
    tokens, experts, hidden, inter = 6, 8, 512, 256
    rng = np.random.default_rng(19 + len(hit_experts))
    x = jnp.asarray(
        rng.standard_normal((tokens, hidden)).astype(np.float32), jnp.bfloat16
    )
    topk = len(hit_experts)
    local_ids = jnp.broadcast_to(
        jnp.asarray(hit_experts, dtype=jnp.int32), (tokens, topk)
    )
    weights = rng.random((tokens, topk)).astype(np.float32)
    weights = jnp.asarray(weights / weights.sum(axis=1, keepdims=True), jnp.bfloat16)
    w13, s13, w2, s2 = _weights(rng, experts, hidden, inter)
    split_out = np.asarray(
        moe_lowtok(
            x,
            local_ids,
            weights,
            w13,
            s13,
            w2,
            s2,
            whole_expert=False,
            interpret=_interpret(),
        ),
        np.float32,
    )
    whole_out = np.asarray(
        moe_lowtok(
            x,
            local_ids,
            weights,
            w13,
            s13,
            w2,
            s2,
            whole_expert=True,
            interpret=_interpret(),
        ),
        np.float32,
    )
    scale = max(float(np.abs(split_out).max()), 1.0)
    tol = 1e-6 if _interpret() else 5e-3
    np.testing.assert_allclose(whole_out / scale, split_out / scale, atol=tol, rtol=0)


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
    # Unaligned hidden width.
    x_unaligned, _, _, w13_u, s13_u, w2_u, s2_u = _operands(0, 4, 8, 192, 256, 4)
    assert not can_use_moe_lowtok(
        x_unaligned, w13_u, w2_u, s13_u, s2_u, None, None, **supported
    )
    assert not can_use_moe_lowtok(
        x_unaligned,
        w13_u,
        w2_u,
        s13_u,
        s2_u,
        None,
        None,
        whole_expert=True,
        **supported,
    )
    # A gate|up width that does not split into two lane-aligned halves:
    # rejected by the split stream, accepted by whole_expert.
    _, _, _, w13_odd, s13_odd, w2_odd, s2_odd = _operands(0, 4, 8, 512, 128, 4)
    assert not can_use_moe_lowtok(
        x, w13_odd, w2_odd, s13_odd, s2_odd, None, None, **supported
    )
    assert can_use_moe_lowtok(
        x,
        w13_odd,
        w2_odd,
        s13_odd,
        s2_odd,
        None,
        None,
        whole_expert=True,
        **supported,
    )
    # VMEM fit check: ids entry fits at its exact requirement while scores
    # entry (which carries routing scratch) is refused at that same cap.
    need_ids = vmem_required_bytes(32, 512, 256, 2, 512, whole_expert=True)
    assert can_use_moe_lowtok(
        x,
        w13,
        w2,
        s13,
        s2,
        None,
        None,
        whole_expert=True,
        scores=False,
        vmem_limit_bytes=need_ids,
        **supported,
    )
    assert not can_use_moe_lowtok(
        x,
        w13,
        w2,
        s13,
        s2,
        None,
        None,
        whole_expert=True,
        scores=True,
        vmem_limit_bytes=need_ids,
        **supported,
    )
    assert not can_use_moe_lowtok(
        x, w13, w2, s13, s2, None, None, vmem_limit_bytes=1024, **supported
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


@pytest.mark.multichip
@pytest.mark.parametrize("whole_expert", [False, True])
@pytest.mark.parametrize("mesh_ep_ranks", [None, (2, 0, 3, 1)])
@pytest.mark.parametrize("rows_per_rank", [1, 3])
def test_expert_parallel_matches_dense_layer(
    whole_expert, mesh_ep_ranks, rows_per_rank
):
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
        whole_expert=whole_expert,
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

    def serves(rows, max_tokens=16, groups=None, activation="silu", whole_expert=False):
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
            whole_expert=whole_expert,
        )

    assert serves(16)
    assert serves(4, groups=((0,), (1,), (2,), (3,)))
    assert not serves(20)  # above the cap
    assert not serves(16, max_tokens=0)  # off
    assert not serves(8, groups=((0, 1), (2, 3)))  # replicated rows
    assert not serves(8, activation="gelu")
    # inter=128 (two_inter=256) requires whole_expert=True.
    w13_odd, s13_odd, w2_odd, s2_odd = _weights(rng, ep * local, hidden, 128)
    x8 = jax.ShapeDtypeStruct((8, hidden), jnp.bfloat16)
    assert not moe_lowtok_ep_serves(
        x8,
        w13_odd,
        w2_odd,
        s13_odd.reshape(ep * local, -1),
        s2_odd.reshape(ep * local, -1),
        ep=ep,
        max_tokens=16,
        activation="silu",
        token_replica_groups=None,
        whole_expert=False,
    )
    assert moe_lowtok_ep_serves(
        x8,
        w13_odd,
        w2_odd,
        s13_odd.reshape(ep * local, -1),
        s2_odd.reshape(ep * local, -1),
        ep=ep,
        max_tokens=16,
        activation="silu",
        token_replica_groups=None,
        whole_expert=True,
    )


def test_whole_expert_executes_single_lane_half_intermediate():
    """whole_expert=True executes inter=128 (two_inter=256) against reference."""
    rng = np.random.default_rng(71)
    tokens, hidden, inter, experts, topk = 9, 256, 128, 8, 3
    x = jnp.asarray(
        rng.standard_normal((tokens, hidden)).astype(np.float32), jnp.bfloat16
    )
    w13, s13, w2, s2 = _weights(rng, experts, hidden, inter)
    ids = jnp.asarray(rng.integers(-1, experts, size=(tokens, topk), dtype=np.int32))
    wts = jnp.asarray(rng.uniform(0.1, 1.0, size=(tokens, topk)).astype(np.float32))
    want = moe_lowtok_reference(x, ids, wts, w13, s13, w2, s2)
    got = np.asarray(
        moe_lowtok(
            x,
            ids,
            wts,
            w13,
            s13,
            w2,
            s2,
            whole_expert=True,
            interpret=_interpret(),
        ),
        np.float32,
    )
    scale = max(float(np.abs(want).max()), 1e-6)
    np.testing.assert_allclose(got / scale, want / scale, atol=2e-2, rtol=0)
    got_split = np.asarray(
        moe_lowtok(
            x,
            ids,
            wts,
            w13,
            s13,
            w2,
            s2,
            whole_expert=False,
            interpret=_interpret(),
        ),
        np.float32,
    )
    np.testing.assert_allclose(
        got / scale,
        got_split / scale,
        atol=1e-6 if _interpret() else 5e-3,
        rtol=0,
    )


def test_qwen35_397b_320_rows_fits_tpu7x_vmem_only_with_whole_expert():
    """At 320 global rows on Qwen3.5-397B (hidden=4096, inter=1024), TPU v7x's
    48 MiB budget (75% of 64 MiB) admits whole_expert=True and refuses split."""
    from vllm_torchtpu.kernels.moe_lowtok.kernel import can_use_moe_lowtok

    tpu7x_vmem_75pct = 48 * 1024 * 1024
    x320 = jax.ShapeDtypeStruct((320, 4096), jnp.bfloat16)
    w13 = jax.ShapeDtypeStruct((64, 4096, 2048), jnp.float8_e4m3fn)
    w2 = jax.ShapeDtypeStruct((64, 1024, 4096), jnp.float8_e4m3fn)
    s13 = jax.ShapeDtypeStruct((64, 2048), jnp.float32)
    s2 = jax.ShapeDtypeStruct((64, 4096), jnp.float32)
    assert not can_use_moe_lowtok(
        x320,
        w13,
        w2,
        s13,
        s2,
        None,
        None,
        activation="silu",
        vmem_limit_bytes=tpu7x_vmem_75pct,
        whole_expert=False,
        scores=True,
    )
    assert can_use_moe_lowtok(
        x320,
        w13,
        w2,
        s13,
        s2,
        None,
        None,
        activation="silu",
        vmem_limit_bytes=tpu7x_vmem_75pct,
        whole_expert=True,
        scores=True,
    )


def test_resolve_whole_expert_threshold_and_coverage_fallback():
    """resolve_whole_expert selects split below min_tokens when split is legal,
    and retains whole_expert=True on shapes only whole-expert supports."""
    from vllm_torchtpu.kernels.moe_lowtok.expert_parallel import (
        resolve_whole_expert_ep,
    )
    from vllm_torchtpu.kernels.moe_lowtok.kernel import resolve_whole_expert

    tpu7x_vmem_75pct = 48 * 1024 * 1024
    w13 = jax.ShapeDtypeStruct((64, 4096, 2048), jnp.float8_e4m3fn)
    w2 = jax.ShapeDtypeStruct((64, 1024, 4096), jnp.float8_e4m3fn)
    s13 = jax.ShapeDtypeStruct((64, 2048), jnp.float32)
    s2 = jax.ShapeDtypeStruct((64, 4096), jnp.float32)
    x160 = jax.ShapeDtypeStruct((160, 4096), jnp.bfloat16)
    x192 = jax.ShapeDtypeStruct((192, 4096), jnp.bfloat16)
    x320 = jax.ShapeDtypeStruct((320, 4096), jnp.bfloat16)

    with pytest.raises(ValueError, match="non-negative"):
        resolve_whole_expert(
            x160,
            w13,
            w2,
            s13,
            s2,
            activation="silu",
            whole_expert=True,
            whole_expert_min_tokens=-1,
        )

    assert not resolve_whole_expert(
        x192,
        w13,
        w2,
        s13,
        s2,
        activation="silu",
        whole_expert=False,
        whole_expert_min_tokens=192,
        scores=True,
        vmem_limit_bytes=tpu7x_vmem_75pct,
    )
    assert resolve_whole_expert(
        x160,
        w13,
        w2,
        s13,
        s2,
        activation="silu",
        whole_expert=True,
        whole_expert_min_tokens=0,
        scores=True,
        vmem_limit_bytes=tpu7x_vmem_75pct,
    )
    assert not resolve_whole_expert(
        x160,
        w13,
        w2,
        s13,
        s2,
        activation="silu",
        whole_expert=True,
        whole_expert_min_tokens=192,
        scores=True,
        vmem_limit_bytes=tpu7x_vmem_75pct,
    )
    assert resolve_whole_expert(
        x192,
        w13,
        w2,
        s13,
        s2,
        activation="silu",
        whole_expert=True,
        whole_expert_min_tokens=192,
        scores=True,
        vmem_limit_bytes=tpu7x_vmem_75pct,
    )
    # 320 rows exceeds split's 48 MiB VMEM cap: retain whole_expert=True even
    # when whole_expert_min_tokens exceeds 320.
    assert resolve_whole_expert(
        x320,
        w13,
        w2,
        s13,
        s2,
        activation="silu",
        whole_expert=True,
        whole_expert_min_tokens=512,
        scores=True,
        vmem_limit_bytes=tpu7x_vmem_75pct,
    )

    # inter=128 (two_inter=256) is supported only by whole_expert=True.
    ep = 8
    w13_narrow = jax.ShapeDtypeStruct((ep * 8, 256, 256), jnp.float8_e4m3fn)
    w2_narrow = jax.ShapeDtypeStruct((ep * 8, 128, 256), jnp.float8_e4m3fn)
    s13_narrow = jax.ShapeDtypeStruct((ep * 8, 256), jnp.float32)
    s2_narrow = jax.ShapeDtypeStruct((ep * 8, 256), jnp.float32)
    x_small = jax.ShapeDtypeStruct((16, 256), jnp.bfloat16)
    assert resolve_whole_expert_ep(
        x_small,
        w13_narrow,
        w2_narrow,
        s13_narrow,
        s2_narrow,
        ep=ep,
        activation="silu",
        whole_expert=True,
        whole_expert_min_tokens=192,
    )
    # Layers or buckets unsupported even by whole_expert=True return False
    # rather than falsely claiming whole-expert streaming applies.
    assert not resolve_whole_expert(
        x160,
        w13,
        w2,
        s13,
        s2,
        activation="gelu",
        whole_expert=True,
        whole_expert_min_tokens=192,
        scores=True,
        vmem_limit_bytes=tpu7x_vmem_75pct,
    )
    assert not resolve_whole_expert(
        x320,
        w13,
        w2,
        s13,
        s2,
        activation="silu",
        whole_expert=True,
        whole_expert_min_tokens=192,
        scores=True,
        vmem_limit_bytes=1024,
    )
