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
    _capacity_overflow,
    _gather_routing,
    _tile_rows,
    exchange_capacity,
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


@pytest.mark.cpu_test
@pytest.mark.parametrize("weight_dtype", [jnp.bfloat16, jnp.float32])
def test_gather_routing_reads_rows_and_empty_slots(weight_dtype):
    """Rows read the gathered routing bitwise; out-of-range rows read id -1
    with weight +0.0."""
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("needs 2 devices")
    mesh = Mesh(np.asarray(devices[:2]), (DP_AXIS,))
    rng = np.random.default_rng(0)
    ids = rng.integers(-1, 512, size=(2, 6, TOPK)).astype(np.int32)
    weights = rng.standard_normal((2, 6, TOPK)).astype(np.float32)
    weights[0, 0, 0] = -0.0
    weights = jnp.asarray(weights).astype(weight_dtype)
    rows = jnp.asarray([[11, 0, 12, 5, 6, 13]] * 2, jnp.int32)

    def body(i, w, r):
        gathered_ids, read, _ = _gather_routing(i[0], w[0])
        got_ids, got_weights = read(r[0])
        return gathered_ids[None], got_ids[None], got_weights[None]

    spec = P(DP_AXIS)
    gathered, got_ids, got_weights = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * 3,
            out_specs=(spec,) * 3,
            check_vma=False,
        )
    )(jnp.asarray(ids), weights, rows)
    # Compare on the host: device-side indexing may drop the sign of -0.0.
    gathered, got_ids = np.asarray(gathered), np.asarray(got_ids)
    got_bits = np.asarray(got_weights).view(np.uint8)
    flat_ids = ids.reshape(12, TOPK)
    flat_bits = np.asarray(weights).reshape(12, TOPK).view(np.uint8)
    for rank in range(2):
        np.testing.assert_array_equal(gathered[rank], ids)
        for j, row in enumerate(np.asarray(rows[rank])):
            if row < 12:
                np.testing.assert_array_equal(got_ids[rank, j], flat_ids[row])
                np.testing.assert_array_equal(got_bits[rank, j], flat_bits[row])
            else:
                assert (got_ids[rank, j] == -1).all()
                assert (got_bits[rank, j] == 0).all()


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


def _previous_gather_routing(topk_ids, topk_weights):
    if topk_weights.dtype != jnp.bfloat16:
        return (
            jax.lax.all_gather(topk_ids, DP_AXIS),
            jax.lax.all_gather(topk_weights, DP_AXIS),
        )
    bits = jax.lax.bitcast_convert_type(topk_weights, jnp.uint16).astype(jnp.int32)
    packed = jax.lax.all_gather(jnp.left_shift(topk_ids, 16) | bits, DP_AXIS)
    weights = jax.lax.bitcast_convert_type(
        jnp.bitwise_and(packed, 0xFFFF).astype(jnp.uint16), jnp.bfloat16
    )
    return jnp.right_shift(packed, 16), weights


def _previous_ragged_a2a_moe_local(
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_weights,
    topk_ids,
    experts_start,
    *,
    tp_size,
    dp_size,
    topk,
    activation,
    use_sparse_core,
    onehot_moe_permute_threshold,
    rhs_quant_dtype=None,
    skip_padded_tokens=False,
):
    """The exchange as first built (one routing gather, per-member slices)."""
    num_tokens, hidden = x.shape
    local_num_experts = w1.shape[0]
    ep_rank = experts_start[0] // local_num_experts
    dp_pos = ep_rank // tp_size
    slots = dp_size * num_tokens

    gathered_ids, gathered_weights = _previous_gather_routing(topk_ids, topk_weights)
    need = exchange_need(
        gathered_ids, ep_rank, tp_size=tp_size, local_num_experts=local_num_experts
    )
    counts = need.sum(axis=1, dtype=jnp.int32)
    send_sizes = jnp.take(counts, dp_pos, axis=0)
    recv_sizes = jnp.take(counts, dp_pos, axis=1)
    member = jnp.arange(dp_size, dtype=jnp.int32)
    token = jnp.arange(num_tokens, dtype=jnp.int32)
    region = member * num_tokens
    landing = jnp.full((dp_size,), dp_pos * num_tokens, dtype=jnp.int32)

    need_out = jnp.take(need, dp_pos, axis=0)
    pos_out = jnp.cumsum(need_out, axis=0, dtype=jnp.int32) - 1
    slot_out = jnp.where(need_out, region[None, :] + pos_out, slots)
    send_index = (
        jnp.zeros((slots,), jnp.int32)
        .at[slot_out.reshape(-1)]
        .set(jnp.broadcast_to(token[:, None], need_out.shape).reshape(-1), mode="drop")
    )

    need_in = jnp.take(need, dp_pos, axis=2)
    pos_in = jnp.cumsum(need_in, axis=1, dtype=jnp.int32) - 1
    slot_in = jnp.where(need_in, region[:, None] + pos_in, slots)
    source_row = (
        jnp.zeros((slots,), jnp.int32)
        .at[slot_in.reshape(-1)]
        .set((region[:, None] + token[None, :]).reshape(-1), mode="drop")
    )
    filled = (jnp.arange(slots, dtype=jnp.int32) % num_tokens) < jnp.repeat(
        recv_sizes, num_tokens, total_repeat_length=slots
    )
    recv_ids = jnp.where(
        filled[:, None], gathered_ids.reshape(slots, topk)[source_row], -1
    )
    recv_weights = jnp.where(
        filled[:, None],
        gathered_weights.reshape(slots, topk)[source_row],
        jnp.zeros((), gathered_weights.dtype),
    )

    sent = _tile_rows(x[send_index])
    received = jax.lax.ragged_all_to_all(
        sent,
        jnp.zeros_like(sent),
        region,
        send_sizes,
        landing,
        recv_sizes,
        axis_name=DP_AXIS,
    ).reshape(slots, hidden)
    partial = fused_moe_func(
        received,
        w1,
        w2,
        w1_scale,
        w2_scale,
        None,
        None,
        recv_weights,
        recv_ids,
        experts_start=experts_start[0],
        topk=topk,
        activation=activation,
        use_ep=True,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=onehot_moe_permute_threshold,
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=skip_padded_tokens,
    )
    returned = _tile_rows(partial)
    back = jax.lax.ragged_all_to_all(
        returned,
        jnp.zeros_like(returned),
        region,
        recv_sizes,
        landing,
        send_sizes,
        axis_name=DP_AXIS,
    ).reshape(slots, hidden)
    back = jnp.concatenate([back, jnp.zeros((1, hidden), back.dtype)], axis=0)
    rows = back[slot_out.T]
    return rows.astype(jnp.float32).sum(axis=0).astype(x.dtype)


def _inputs(num_ranks, tp_size, num_tokens, num_experts, *, skew, topk=TOPK):
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
    weights, ids = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), topk)
    w1 = jax.random.normal(k1, (num_experts, HIDDEN, 2 * INTER)) / 8
    w2 = jax.random.normal(k2, (num_experts, INTER, HIDDEN)) / 8
    w1, s1 = quantize_to_native_fp4_kmajor(w1, 256, pack=False)
    w2, s2 = quantize_to_native_fp4_kmajor(w2, 256, pack=False)
    es = (jnp.arange(num_ranks, dtype=jnp.int32) * (num_experts // num_ranks)).reshape(
        num_ranks
    )
    return x, w1, w2, s1, s2, weights.astype(jnp.bfloat16), ids.astype(jnp.int32), es


def _onehot_threshold(rule, dp_size, num_tokens):
    """``auto``: the SparseCore-block default; ``decode``: raised to this
    step's routed rows, as the adapters resolve it for a decode step."""
    if rule == "auto":
        return resolve_onehot_permute_threshold()
    return resolve_onehot_permute_threshold(dp_size * num_tokens * TOPK)


@pytest.mark.multichip
@pytest.mark.parametrize("threshold", ["auto", "decode"])
@pytest.mark.parametrize(
    "dp_size,tp_size,num_tokens,skew",
    [
        (2, 4, 12, False),  # proxy layout, decode batch below the one-hot cap
        (2, 4, 48, False),  # proxy layout, 960 routed rows
        (2, 4, 12, True),
        (4, 2, 12, False),
        (8, 1, 12, False),
        (8, 1, 12, True),
    ],
)
def test_ragged_a2a_matches_allgather_path(
    dp_size, tp_size, num_tokens, skew, threshold
):
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    num_experts = 64
    args = _inputs(8, tp_size, num_tokens, num_experts, skew=skew)
    kw = dict(
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=_onehot_threshold(threshold, dp_size, num_tokens),
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


@pytest.mark.multichip
@pytest.mark.parametrize(
    "dp_size,tp_size,num_tokens,skew,weight_dtype,padded",
    [
        (2, 4, 48, False, jnp.bfloat16, False),  # proxy decode step, 960 rows
        (2, 4, 12, False, jnp.bfloat16, False),
        (2, 4, 48, True, jnp.bfloat16, False),
        (2, 4, 48, False, jnp.float32, False),  # unpacked routing gather
        (2, 4, 48, False, jnp.bfloat16, True),  # zero-weight padded tokens
        (4, 2, 24, False, jnp.bfloat16, False),
        (8, 1, 12, True, jnp.bfloat16, False),
    ],
)
def test_ragged_a2a_matches_previous_exchange(
    dp_size, tp_size, num_tokens, skew, weight_dtype, padded
):
    """Bitwise the exchange as first built, which permuted with the SparseCore
    kernels above the auto threshold: with that threshold, and with the
    decode-step threshold that moves the local permute to one-hot matmuls."""
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    x, w1, w2, s1, s2, weights, ids, es = _inputs(8, tp_size, num_tokens, 64, skew=skew)
    weights = weights.astype(weight_dtype)
    if padded:
        # Padded tokens keep their expert ids but carry zero routing weights.
        weights = weights.reshape(8, num_tokens, TOPK).at[:, -5:].set(0)
        weights = weights.reshape(-1, TOPK)
    args = (x, w1, w2, s1, s2, weights, ids, es)
    common = dict(
        topk=TOPK, activation="silu", use_sparse_core=True, skip_padded_tokens=padded
    )
    previous_local = functools.partial(
        _previous_ragged_a2a_moe_local,
        tp_size=tp_size,
        dp_size=dp_size,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(),
        **common,
    )
    spec = P((DP_AXIS, TP_AXIS))
    previous = jax.jit(
        jax.shard_map(
            lambda *a: previous_local(*a[:7], a[7].reshape(1)),
            mesh=mesh,
            in_specs=(spec,) * 8,
            out_specs=spec,
            check_vma=False,
        )
    )(*args)
    assert np.isfinite(np.asarray(previous, np.float32)).all()
    assert np.abs(np.asarray(previous, np.float32)).max() > 0
    previous = np.asarray(previous).view(np.uint16)
    stacked = tuple(a.reshape(dp_size, tp_size, -1, *a.shape[1:]) for a in args)
    for rule in ("auto", "decode"):
        got = jax.jit(
            functools.partial(
                ragged_a2a_moe,
                mesh=mesh,
                onehot_moe_permute_threshold=_onehot_threshold(
                    rule, dp_size, num_tokens
                ),
                **common,
            )
        )(*stacked)
        got = np.asarray(got).reshape(previous.shape[0], -1).view(np.uint16)
        np.testing.assert_array_equal(got, previous, err_msg=rule)


@pytest.mark.cpu_test
def test_exchange_capacity_sizes():
    # Proxy dp2 x tp4, top-10 of 512: ~74% of the 96 rows reach a rank, so the
    # rows stay full; the routes shrink to 2 x 120, rounded to 256.
    kw = dict(topk=10, local_num_experts=64, onehot_moe_permute_threshold=960)
    assert ra_capacity(dp_size=2, tp_size=4, num_tokens=48, factor=2.0, **kw) == (
        96,
        256,
    )
    assert ra_capacity(dp_size=2, tp_size=4, num_tokens=48, factor=0.0, **kw) == (
        96,
        960,
    )
    # 4 hosts dp8 x tp4 (16 experts per rank): ~27% of 96 rows, 30 routes.
    kw = dict(topk=10, local_num_experts=16, onehot_moe_permute_threshold=960)
    assert ra_capacity(dp_size=8, tp_size=4, num_tokens=12, factor=2.0, **kw) == (
        64,
        128,
    )
    # 8 hosts dp16 x tp4 with 3 drafts: 768 rows of 7680 routed rows exceed the
    # one-hot cap; capacity brings them under it.
    kw = dict(topk=10, local_num_experts=8, onehot_moe_permute_threshold=3840)
    assert ra_capacity(dp_size=16, tp_size=4, num_tokens=48, factor=2.0, **kw) == (
        240,
        256,
    )
    # Above the one-hot threshold the routes are not bounded.
    kw = dict(topk=10, local_num_experts=64, onehot_moe_permute_threshold=511)
    assert ra_capacity(dp_size=2, tp_size=4, num_tokens=48, factor=2.0, **kw) == (
        96,
        960,
    )


def ra_capacity(**kw):
    return exchange_capacity(use_sparse_core=True, **kw)


@pytest.mark.multichip
@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    "dp_size,tp_size,num_tokens,topk,skew",
    [
        (2, 4, 48, 10, False),  # proxy decode step: route capacity only
        (2, 4, 48, 2, False),  # ~24% occupancy: row capacity too
        (2, 4, 48, 2, True),  # skewed onto DP position 0: overflow
        (4, 2, 24, 2, False),
        (8, 1, 48, 10, False),  # 8 members, 3840 routed rows
        (8, 1, 12, 2, True),
    ],
)
def test_capacity_exchange_matches_full_exchange(
    dp_size, tp_size, num_tokens, topk, skew, factor
):
    """Bitwise the full-size exchange, whether the step fits the capacity
    buffers or overflows into the full-size path (factor 0.5 always
    overflows at top-10)."""
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    x, w1, w2, s1, s2, weights, ids, es = _inputs(
        8, tp_size, num_tokens, 64, skew=skew, topk=topk
    )
    stacked = tuple(
        a.reshape(dp_size, tp_size, -1, *a.shape[1:])
        for a in (x, w1, w2, s1, s2, weights, ids, es)
    )
    threshold = resolve_onehot_permute_threshold(dp_size * num_tokens * topk)
    common = dict(
        mesh=mesh,
        topk=topk,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=threshold,
    )
    full = jax.jit(functools.partial(ragged_a2a_moe, **common))(*stacked)
    full = np.asarray(full)
    assert np.isfinite(full.astype(np.float32)).all()
    assert np.abs(full.astype(np.float32)).max() > 0
    got = jax.jit(functools.partial(ragged_a2a_moe, capacity_factor=factor, **common))(
        *stacked
    )
    np.testing.assert_array_equal(np.asarray(got).view(np.uint16), full.view(np.uint16))


@pytest.mark.multichip
def test_capacity_exchange_runs_both_branches():
    """One step that fits the capacity buffers, one that overflows them (every
    token routed to rank 0's experts), through the same jitted program."""
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(2, 4), (DP_AXIS, TP_AXIS))
    threshold = resolve_onehot_permute_threshold(2 * 48 * 2)
    common = dict(
        mesh=mesh,
        topk=2,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=threshold,
    )
    full_run = jax.jit(functools.partial(ragged_a2a_moe, **common))
    capacity_run = jax.jit(
        functools.partial(ragged_a2a_moe, capacity_factor=2.0, **common)
    )
    rows, routes = exchange_capacity(
        dp_size=2,
        tp_size=4,
        num_tokens=48,
        topk=2,
        local_num_experts=8,
        factor=2.0,
        onehot_moe_permute_threshold=threshold,
        use_sparse_core=True,
    )
    assert rows < 96
    args = _inputs(8, 4, 48, 64, skew=False, topk=2)
    for forced in (False, True):
        if forced:
            ids = jnp.broadcast_to(jnp.asarray([0, 1], jnp.int32), args[6].shape)
            args = args[:6] + (ids, args[7])
        stacked = tuple(a.reshape(2, 4, -1, *a.shape[1:]) for a in args)
        gathered = np.asarray(ids_by_engine(args[6], 2, 4))
        assert _overflow_reference(gathered, 4, 8, rows, routes) == forced
        full = np.asarray(full_run(*stacked)).view(np.uint16)
        got = np.asarray(capacity_run(*stacked)).view(np.uint16)
        np.testing.assert_array_equal(got, full)


def ids_by_engine(ids, dp_size, tp_size):
    return ids.reshape(dp_size, tp_size, -1, ids.shape[-1])[:, 0]


def _overflow_reference(ids, tp_size, local_num_experts, rows, routes):
    dp_size, num_tokens, _ = ids.shape
    num_ranks = dp_size * tp_size
    hit = np.zeros((dp_size, num_tokens, num_ranks), bool)
    rank_routes = np.zeros(num_ranks, int)
    for s in range(dp_size):
        for t in range(num_tokens):
            for e in ids[s, t]:
                hit[s, t, e // local_num_experts] = True
                rank_routes[e // local_num_experts] += 1
    recv = hit.sum(axis=(0, 1))
    send = hit.reshape(dp_size, num_tokens, dp_size, tp_size).sum(axis=(1, 2))
    return bool(rank_routes.max() > routes or recv.max() > rows or send.max() > rows)


@pytest.mark.cpu_test
@pytest.mark.parametrize("dp_size,tp_size", [(2, 4), (4, 2), (8, 1)])
@pytest.mark.parametrize("topk", [2, 10])
def test_capacity_overflow_matches_brute_force(dp_size, tp_size, topk):
    devices = jax.devices()
    if len(devices) < 8:
        pytest.skip("needs 8 devices")
    local = 8
    rng = np.random.default_rng(dp_size * 10 + topk)
    ids = np.stack(
        [
            np.stack(
                [
                    rng.choice(dp_size * tp_size * local, topk, replace=False)
                    for _ in range(12)
                ]
            )
            for _ in range(dp_size)
        ]
    ).astype(np.int32)
    mesh = Mesh(np.asarray(devices[:8]).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    for rows, routes in [
        (4, 1000),
        (8, 1000),
        (12, 1000),
        (24, 10),
        (24, 20),
        (24, 1000),
    ]:
        body = functools.partial(
            _capacity_overflow,
            tp_size=tp_size,
            local_num_experts=local,
            rows=rows,
            routes=routes,
        )
        got = jax.jit(
            jax.shard_map(
                lambda g, body=body: body(g[0, 0])[None, None],
                mesh=mesh,
                in_specs=P(DP_AXIS, TP_AXIS),
                out_specs=P(DP_AXIS, TP_AXIS),
                check_vma=False,
            )
        )(
            jnp.broadcast_to(
                jnp.asarray(ids)[None, None], (dp_size, tp_size) + ids.shape
            )
        )
        want = _overflow_reference(ids, tp_size, local, rows, routes)
        assert (np.asarray(got) == want).all(), (rows, routes)


@pytest.mark.multichip
@pytest.mark.parametrize(
    "num_tokens,max_local_routes,skip_padded",
    [
        (96, 128, False),  # proxy rank: 96 rows, ~120 local routes
        (96, 256, False),
        (96, 256, True),
        (384, 1024, False),  # 8 members: 3840 routed rows
        (64, 640, False),  # bound at the routed rows: no rows skipped
    ],
)
def test_local_routes_matches_onehot_path(num_tokens, max_local_routes, skip_padded):
    """``max_local_routes`` gives the one-hot path's output bitwise when the
    local routes fit."""
    devices = _require_eight_tpu_devices()
    num_experts, local = 64, 8
    x, w1, w2, s1, s2, weights, ids, _ = _inputs(
        8, 1, num_tokens, num_experts, skew=False
    )
    x, weights, ids = x[:num_tokens], weights[:num_tokens], ids[:num_tokens]
    if skip_padded:
        weights = weights.at[-7:].set(0)
    # Rank 0 hosts experts 0..7; keep the local routes within the bound.
    ids_np = np.array(ids)
    local_rows = np.nonzero((ids_np < local).any(axis=1))[0]
    keep = max_local_routes
    drop = local_rows[np.cumsum((ids_np[local_rows] < local).sum(axis=1)) > keep]
    ids_np[drop] = np.where(ids_np[drop] < local, ids_np[drop] + local, ids_np[drop])
    ids = jnp.asarray(ids_np)
    assert (ids_np < local).sum() <= max_local_routes
    threshold = num_tokens * TOPK
    kw = dict(
        experts_start=jnp.int32(0),
        topk=TOPK,
        activation="silu",
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=threshold,
        skip_padded_tokens=skip_padded,
    )
    args = (x, w1[:local], w2[:local], s1[:local], s2[:local], None, None, weights, ids)
    with jax.default_device(devices[0]):
        want = jax.jit(functools.partial(fused_moe_func, **kw))(*args)
        got = jax.jit(
            functools.partial(fused_moe_func, max_local_routes=max_local_routes, **kw)
        )(*args)
    want = np.asarray(want)
    assert np.isfinite(want.astype(np.float32)).all()
    assert np.abs(want.astype(np.float32)).max() > 0
    np.testing.assert_array_equal(np.asarray(got).view(np.uint16), want.view(np.uint16))
