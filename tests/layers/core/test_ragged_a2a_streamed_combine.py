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
"""Ragged all-to-all EP exchange whose expert kernel sends the output rows back."""

import functools

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2_mlp, gmm_v2_mlp_scatter
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    quantize_to_native_fp4_kmajor,
    resolve_onehot_permute_threshold,
)
from vllm_torchtpu.layers.core.ragged_a2a_moe import (
    DP_AXIS,
    TP_AXIS,
    _member_index,
    _sender_sum,
    exchange_capacity,
    ragged_a2a_moe,
    route_order,
    sender_sum_kernel,
)

HIDDEN = 2048
INTER = 512
BF16 = ml_dtypes.bfloat16


@pytest.mark.cpu_test
def test_route_order_groups_routes_by_member():
    """A permutation per token: in-group routes first by expert id (members in
    DP order, each member's experts ascending), then the others in k order."""
    rng = np.random.default_rng(0)
    ids = np.stack([rng.choice(64, 10, replace=False) for _ in range(32)])
    ids[3, 4] = -1
    in_group = (ids >= 0) & ((ids // 8) % 2 == 0)
    got = np.asarray(route_order(jnp.asarray(ids, jnp.int32), jnp.asarray(in_group)))
    for t in range(ids.shape[0]):
        assert sorted(got[t]) == list(range(10))
        inside = [k for k in range(10) if in_group[t, k]]
        outside = [k for k in range(10) if not in_group[t, k]]
        want = sorted(inside, key=lambda k: ids[t, k]) + outside
        assert [int(np.argwhere(got[t] == j)[0, 0]) for j in range(10)] == want


def _sender_sum_reference(rows, ids, weights, member, dp_size, skip_padded):
    """Per member: its routes in expert order, products and sums in f32, one
    bf16 rounding; then the members' partials summed in f32 in DP order."""
    num_tokens, topk = ids.shape
    w = weights.astype(BF16).astype(np.float32)
    w = np.where(np.isfinite(w), w, 0)
    out = np.zeros((num_tokens, rows.shape[-1]), np.float32)
    for t in range(num_tokens):
        order = route_order_np(ids[t], member[t] < dp_size)
        for d in range(dp_size):
            acc = np.zeros(rows.shape[-1], np.float32)
            hit = False
            for j in range(topk):
                k = order.index(j)
                if member[t, k] != d:
                    continue
                hit = True
                if skip_padded and w[t, k] == 0:
                    continue
                acc = acc + np.float32(w[t, k]) * rows[t, j].astype(np.float32)
            if hit:
                out[t] = out[t] + acc.astype(BF16).astype(np.float32)
    return out.astype(BF16)


def route_order_np(ids, in_group):
    keys = [(0, int(ids[k]), k) if in_group[k] else (1, k, k) for k in range(len(ids))]
    ranked = sorted(range(len(ids)), key=lambda k: keys[k])
    order = [0] * len(ids)
    for j, k in enumerate(ranked):
        order[k] = j
    return order


@pytest.mark.cpu_test
@pytest.mark.parametrize("skip_padded", [False, True])
def test_sender_sum_matches_reference(skip_padded):
    dp_size, tp_size, local, num_tokens, topk = 4, 2, 4, 6, 10
    rng = np.random.default_rng(1)
    ids = np.stack(
        [
            rng.choice(dp_size * tp_size * local, topk, replace=False)
            for _ in range(num_tokens)
        ]
    ).astype(np.int32)
    weights = rng.uniform(0.01, 1.0, size=ids.shape).astype(np.float32)
    weights[0, :3] = 0
    weights[1, 2] = np.nan
    ep_rank = 2  # DP position 1, TP rank 0
    member_index = functools.partial(
        _member_index,
        ep_rank=jnp.int32(ep_rank),
        dp_size=dp_size,
        tp_size=tp_size,
        local_num_experts=local,
    )
    member = np.asarray(member_index(jnp.asarray(ids)))
    rows = (
        rng.standard_normal((num_tokens, topk, HIDDEN))
        * np.exp2(rng.integers(-6, 6, size=(num_tokens, topk, 1)))
    ).astype(BF16)
    # Slots of routes no member of this group serves are never written.
    for t in range(num_tokens):
        order = route_order_np(ids[t], member[t] < dp_size)
        for k in range(topk):
            if member[t, k] >= dp_size:
                rows[t, order[k]] = np.nan
    got = _sender_sum(
        jnp.asarray(rows.transpose(1, 0, 2)).reshape(
            topk * num_tokens, 16, HIDDEN // 16
        ),
        jnp.asarray(ids),
        jnp.asarray(weights, jnp.bfloat16),
        member_index=member_index,
        dp_size=dp_size,
        skip_padded_tokens=skip_padded,
        dtype=jnp.bfloat16,
    )
    want = _sender_sum_reference(rows, ids, weights, member, dp_size, skip_padded)
    np.testing.assert_array_equal(np.asarray(got).view(np.uint16), want.view(np.uint16))


def _require_eight_tpu_devices():
    devices = jax.devices()
    if devices[0].platform != "tpu" or len(devices) < 8:
        pytest.skip("needs 8 TPU devices")
    return devices[:8]


def _inputs(num_ranks, tp_size, num_tokens, num_experts, *, topk, skew, seed=0):
    key = jax.random.key(seed * 1000 + num_ranks * 100 + num_tokens + skew)
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


@pytest.mark.multichip
@pytest.mark.parametrize("factor", [0.0, 0.5, 2.0])
@pytest.mark.parametrize(
    "dp_size,tp_size,num_tokens,topk,skew,padded",
    [
        (8, 1, 12, 2, False, False),
        (8, 1, 12, 2, True, False),
        (8, 1, 24, 10, False, False),
        (8, 1, 24, 10, False, True),
        (4, 2, 24, 10, False, False),
        (4, 2, 24, 2, True, False),
        (2, 4, 48, 10, False, False),  # proxy layout: full-size exchange
    ],
)
def test_streamed_combine_matches_returned_partials(
    dp_size, tp_size, num_tokens, topk, skew, padded, factor
):
    """Bitwise the returned-partials exchange where no member serves three or
    more routes of one token (top-2); otherwise the members' f32 sums may round
    differently in the last place."""
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    x, w1, w2, s1, s2, weights, ids, es = _inputs(
        8, tp_size, num_tokens, 64, topk=topk, skew=skew
    )
    if padded:
        weights = weights.reshape(8, num_tokens, topk).at[:, -5:].set(0)
        weights = weights.reshape(-1, topk)
    stacked = tuple(
        a.reshape(dp_size, tp_size, -1, *a.shape[1:])
        for a in (x, w1, w2, s1, s2, weights, ids, es)
    )
    common = dict(
        mesh=mesh,
        topk=topk,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(
            dp_size * num_tokens * topk
        ),
        skip_padded_tokens=padded,
        capacity_factor=factor,
    )
    base = np.asarray(jax.jit(functools.partial(ragged_a2a_moe, **common))(*stacked))
    got = np.asarray(
        jax.jit(functools.partial(ragged_a2a_moe, streamed_combine=True, **common))(
            *stacked
        )
    )
    assert np.abs(base.astype(np.float32)).max() > 0
    assert np.isfinite(got.astype(np.float32)).all()
    if topk <= 2:
        np.testing.assert_array_equal(got.view(np.uint16), base.view(np.uint16))
    else:
        b, g = base.astype(np.float32), got.astype(np.float32)
        differ = (got.view(np.uint16) != base.view(np.uint16)).mean()
        assert differ < 1e-3, differ
        np.testing.assert_allclose(g, b, rtol=2**-7, atol=1e-6)


@pytest.mark.multichip
def test_streamed_combine_runs_both_capacity_branches():
    """One step that fits the capacity buffers, one that overflows them, through
    the same jitted program (the overflowing step's capacity kernel sends
    nothing and expects nothing)."""
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(4, 2), (DP_AXIS, TP_AXIS))
    common = dict(
        mesh=mesh,
        topk=2,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(4 * 24 * 2),
        capacity_factor=2.0,
    )
    rows, _ = exchange_capacity(
        dp_size=4,
        tp_size=2,
        num_tokens=24,
        topk=2,
        local_num_experts=8,
        factor=2.0,
        onehot_moe_permute_threshold=common["onehot_moe_permute_threshold"],
        use_sparse_core=True,
    )
    assert rows < 4 * 24
    base_run = jax.jit(functools.partial(ragged_a2a_moe, **common))
    streamed_run = jax.jit(
        functools.partial(ragged_a2a_moe, streamed_combine=True, **common)
    )
    args = _inputs(8, 2, 24, 64, topk=2, skew=False)
    for forced in (False, True):
        if forced:
            ids = jnp.broadcast_to(jnp.asarray([0, 1], jnp.int32), args[6].shape)
            args = args[:6] + (ids, args[7])
        stacked = tuple(a.reshape(4, 2, -1, *a.shape[1:]) for a in args)
        base = np.asarray(base_run(*stacked)).view(np.uint16)
        got = np.asarray(streamed_run(*stacked)).view(np.uint16)
        np.testing.assert_array_equal(got, base)


@pytest.mark.multichip
@pytest.mark.parametrize("rows_per_expert", [3, 40, 70])
def test_mlp_scatter_rows_match_gmm_v2_mlp(rows_per_expert):
    """Every routed row lands bitwise at its destination row, local rows
    included; each member's buffer holds what the others sent it."""
    devices = _require_eight_tpu_devices()
    n = 8
    mesh = Mesh(np.asarray(devices).reshape(n, 1), (DP_AXIS, TP_AXIS))
    groups = 8
    size_m = groups * rows_per_expert + 5  # 5 rows past the routed rows
    k1, k2, kx = jax.random.split(jax.random.key(rows_per_expert), 3)
    w1 = jax.random.normal(k1, (groups, HIDDEN, 2 * INTER)) / 8
    w2 = jax.random.normal(k2, (groups, INTER, HIDDEN)) / 8
    w1, s1 = quantize_to_native_fp4_kmajor(w1, 256, pack=False)
    w2, s2 = quantize_to_native_fp4_kmajor(w2, 256, pack=False)
    lhs = (jax.random.normal(kx, (n, size_m, HIDDEN)) / 4).astype(jnp.bfloat16)
    sizes = np.full((groups,), rows_per_expert, np.int32)
    sizes[3] = 0  # an expert without rows
    sizes[4] = 2 * rows_per_expert
    routed = int(sizes.sum())
    rng = np.random.default_rng(rows_per_expert)
    # Member m sends its routed row j to member dest[m, j] at row slot[m, j];
    # each member's receive rows are a permutation of what it gets.
    dest = rng.integers(0, n, size=(n, size_m)).astype(np.int32)
    dest[:, routed:] = 0
    slot = np.zeros((n, size_m), np.int32)
    expected = np.zeros((n,), np.int32)
    num_recv = n * size_m
    for d in range(n):
        senders = [(m, j) for m in range(n) for j in range(routed) if dest[m, j] == d]
        perm = rng.permutation(num_recv)[: len(senders)]
        for (m, j), r in zip(senders, perm):
            slot[m, j] = r
        expected[d] = sum(1 for m, _ in senders if m != d)

    def body(lhs, dest, slot, me, expected):
        recv = gmm_v2_mlp_scatter(
            lhs[0, 0],
            w1,
            w2,
            jnp.asarray(sizes),
            s1,
            s2,
            jnp.zeros((1,), jnp.int32),
            dest[0, 0],
            slot[0, 0],
            me[0, 0],
            expected[0, 0],
            num_recv_rows=num_recv,
            axis=DP_AXIS,
            num_members=n,
            collective_id=0,
            fuse_act="silu",
            maybe_quantize_lhs=False,
        )
        ref = gmm_v2_mlp(
            lhs[0, 0],
            w1,
            w2,
            jnp.asarray(sizes),
            s1,
            s2,
            fuse_act="silu",
            maybe_quantize_lhs=False,
        )
        return recv.reshape(num_recv, HIDDEN)[None, None], ref[None, None]

    spec = P(DP_AXIS, TP_AXIS)

    def per_member(a):
        return jnp.asarray(a).reshape(n, 1, *a.shape[1:])

    recv, ref = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * 5,
            out_specs=(spec, spec),
            check_vma=False,
        )
    )(
        lhs.reshape(n, 1, size_m, HIDDEN),
        per_member(dest),
        per_member(slot),
        per_member(np.arange(n, dtype=np.int32)[:, None]),
        per_member(expected[:, None]),
    )
    recv = np.asarray(recv).view(np.uint16)[:, 0]
    ref = np.asarray(ref).view(np.uint16)[:, 0]
    for m in range(n):
        for j in range(routed):
            np.testing.assert_array_equal(recv[dest[m, j], slot[m, j]], ref[m, j])


@pytest.mark.multichip
@pytest.mark.parametrize("num_tokens", [92, 48, 12, 7])
@pytest.mark.parametrize("skip_padded", [False, True])
def test_sender_sum_kernel_matches_reference(num_tokens, skip_padded):
    """The sending-side sum kernel is bitwise the jnp formulation."""
    _require_eight_tpu_devices()
    dp_size, tp_size, local, topk = 8, 4, 16, 10
    rng = np.random.default_rng(num_tokens)
    ids = np.stack(
        [
            rng.choice(dp_size * tp_size * local, topk, replace=False)
            for _ in range(num_tokens)
        ]
    ).astype(np.int32)
    weights = rng.uniform(0.01, 1.0, size=ids.shape).astype(np.float32)
    weights[0, :3] = 0
    member_index = functools.partial(
        _member_index,
        ep_rank=jnp.int32(5),
        dp_size=dp_size,
        tp_size=tp_size,
        local_num_experts=local,
    )
    rows = (
        rng.standard_normal((topk * num_tokens, HIDDEN))
        * np.exp2(rng.integers(-8, 8, size=(topk * num_tokens, 1)))
    ).astype(BF16)
    recv = jnp.asarray(rows).reshape(topk * num_tokens, 16, HIDDEN // 16)
    common = dict(
        member_index=member_index,
        dp_size=dp_size,
        skip_padded_tokens=skip_padded,
        dtype=jnp.bfloat16,
    )
    args = (recv, jnp.asarray(ids), jnp.asarray(weights, jnp.bfloat16))
    want = jax.jit(functools.partial(_sender_sum, **common))(*args)
    got = jax.jit(functools.partial(sender_sum_kernel, **common))(*args)
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16)
    )


def _tp_sum_f32(partials):
    """Sum over the TP axis (axis 1) in f32 in rank order, one rounding."""
    acc = partials[:, 0].astype(np.float32)
    for r in range(1, partials.shape[1]):
        acc = acc + partials[:, r].astype(np.float32)
    return acc.astype(BF16)


@pytest.mark.multichip
@pytest.mark.parametrize("with_shared", [False, True])
@pytest.mark.parametrize(
    "dp_size,tp_size,num_tokens,topk,factor",
    [
        (2, 4, 24, 10, 0.0),  # proxy layout: full-size exchange
        (4, 2, 24, 10, 2.0),
        (4, 2, 24, 2, 0.5),  # overflows into the full-size exchange
        (2, 4, 22, 10, 0.0),  # tokens not a multiple of tp: padded owner blocks
        (2, 4, 12, 10, 0.0),  # proxy K=0 decode: three owned tokens per core
        (2, 4, 4, 10, 0.0),  # one owned token per core
        (2, 4, 13, 10, 0.0),
        (2, 4, 92, 10, 0.0),  # proxy K=3 decode at 23 requests per engine
    ],
)
def test_owner_combine_matches_tp_sum(
    dp_size, tp_size, num_tokens, topk, factor, with_shared
):
    """Owner mode returns every TP rank the engine's tokens summed over TP,
    bitwise the returned-partials path plus (the shared partial, rounded per
    rank, and) an f32 rank-order TP sum."""
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    x, w1, w2, s1, s2, weights, ids, es = _inputs(
        8, tp_size, num_tokens, 64, topk=topk, skew=False
    )
    stacked = tuple(
        a.reshape(dp_size, tp_size, -1, *a.shape[1:])
        for a in (x, w1, w2, s1, s2, weights, ids, es)
    )
    common = dict(
        mesh=mesh,
        topk=topk,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(
            dp_size * num_tokens * topk
        ),
        capacity_factor=factor,
    )
    shared = None
    if with_shared:
        rng = np.random.default_rng(3)
        shared = jnp.asarray(
            rng.standard_normal((dp_size, tp_size, num_tokens, HIDDEN)) / 8,
            jnp.bfloat16,
        )
    base = np.asarray(jax.jit(functools.partial(ragged_a2a_moe, **common))(*stacked))
    if shared is not None:
        base = (base.astype(np.float32) + np.asarray(shared).astype(np.float32)).astype(
            BF16
        )
    want = _tp_sum_f32(base)
    got = np.asarray(
        jax.jit(functools.partial(ragged_a2a_moe, owner_combine=True, **common))(
            *stacked, shared=shared
        )
    )
    for r in range(tp_size):
        np.testing.assert_array_equal(got[:, r].view(np.uint16), want.view(np.uint16))


@pytest.mark.multichip
@pytest.mark.parametrize("num_tokens", [24, 22, 96])
def test_owner_combine_without_fused_kernel_adds_shared_and_tp_sum(num_tokens):
    """Paths that do not take the fused kernel (prefill-size steps) return the
    returned-partials result plus the shared partial, summed over TP with the
    owner ranks' arithmetic (f32, rank order, one rounding)."""
    devices = _require_eight_tpu_devices()
    dp_size, tp_size, topk = 2, 4, 10
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    x, w1, w2, s1, s2, weights, ids, es = _inputs(
        8, tp_size, num_tokens, 64, topk=topk, skew=False
    )
    stacked = tuple(
        a.reshape(dp_size, tp_size, -1, *a.shape[1:])
        for a in (x, w1, w2, s1, s2, weights, ids, es)
    )
    # Threshold 0: every step takes the SparseCore path, never the fused kernel.
    common = dict(
        mesh=mesh,
        topk=topk,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=0,
    )
    rng = np.random.default_rng(4)
    shared = jnp.asarray(
        rng.standard_normal((dp_size, tp_size, num_tokens, HIDDEN)) / 8, jnp.bfloat16
    )
    base = np.asarray(jax.jit(functools.partial(ragged_a2a_moe, **common))(*stacked))
    base = (base.astype(np.float32) + np.asarray(shared).astype(np.float32)).astype(
        BF16
    )
    want = _tp_sum_f32(base)
    got = np.asarray(
        jax.jit(functools.partial(ragged_a2a_moe, owner_combine=True, **common))(
            *stacked, shared=shared
        )
    )
    for r in range(tp_size):
        np.testing.assert_array_equal(got[:, r].view(np.uint16), want.view(np.uint16))
