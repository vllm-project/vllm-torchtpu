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
"""Exchange options of the ragged all-to-all EP MoE that must not change any
output bit: each against the same exchange without it."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.layers.core import expert_placement as core
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    quantize_to_native_fp4_kmajor,
    resolve_onehot_permute_threshold,
    scatter_route_plan,
)
from vllm_torchtpu.layers.core.ragged_a2a_moe import (
    _EMPTY_ROUTE,
    DP_AXIS,
    TP_AXIS,
    _member_index,
    _unpack_routes,
    owner_dests,
    owner_order,
    owner_remote_routes,
    pack_served_routes,
    primary_slots,
    ragged_a2a_moe,
    route_order,
    route_plan,
    serving_slots,
    streamed_dests,
    unpack_served_routes,
)

HIDDEN = 2048
INTER = 512

# Each option set is compared against the same call without it.
OPTIONS = {
    "sorted_plan": dict(sorted_plan=True),
}
# (option, combine) pairs: the sorted plan only acts with the streamed or owner
# combine.
CASES = [(option, "owner") for option in OPTIONS] + [
    ("sorted_plan", "streamed"),
]
# Base outputs by (layout, combine): computed once for every option.
_BASE = {}
# The row copy kernels act on the sorted plan's path.
OPTIONS["row_gather"] = dict(sorted_plan=True, row_gather=True)
OPTIONS["row_scatter"] = dict(sorted_plan=True, row_scatter=True)
OPTIONS["row_copy"] = dict(sorted_plan=True, row_gather=True, row_scatter=True)
CASES += [
    (option, combine)
    for option in ("row_gather", "row_scatter", "row_copy")
    for combine in ("owner", "streamed")
]
# The one-shot routing all-gather moves the same words, alone and with the
# served options (the owner tail only with the owner combine).
OPTIONS["routing_dma"] = dict(routing_dma=True)
OPTIONS["routing_dma_served"] = dict(
    sorted_plan=True, row_gather=True, row_scatter=True, routing_dma=True
)
OPTIONS["routing_dma_owner_tail"] = dict(OPTIONS["routing_dma_served"], owner_tail=True)
CASES += [("routing_dma", "owner"), ("routing_dma_owner_tail", "owner")] + [
    ("routing_dma_served", combine) for combine in ("owner", "streamed")
]


def _routing(rng, dp_size, num_tokens, num_experts, topk, skew):
    p = np.ones(num_experts)
    if skew:
        p[: num_experts // 8] = 20.0
    p = p / p.sum()
    ids = np.stack(
        [
            rng.choice(num_experts, topk, replace=False, p=p)
            for _ in range(dp_size * num_tokens)
        ]
    ).reshape(dp_size, num_tokens, topk)
    weights = rng.uniform(0.01, 1.0, size=ids.shape).astype(np.float32)
    return ids.astype(np.int32), weights


@pytest.mark.cpu_test
@pytest.mark.parametrize("combine", ["owner", "streamed"])
@pytest.mark.parametrize(
    "dp_size,tp_size,num_tokens,local,topk,skew,padded,capacity",
    [
        (8, 4, 96, 16, 10, False, False, True),  # 4-host decode step
        (8, 4, 96, 16, 10, True, True, True),
        (8, 4, 22, 16, 10, False, False, False),
        (2, 4, 92, 64, 10, False, False, False),  # proxy decode step
        (4, 2, 24, 8, 2, True, True, True),
        (8, 1, 13, 8, 10, False, True, True),
    ],
)
def test_route_plan_matches_the_received_rows_plan(
    combine, dp_size, tp_size, num_tokens, local, topk, skew, padded, capacity
):
    """`route_plan` from the gathered routing gives the expert kernel the
    routes `scatter_route_plan` derives from the received rows: same rows,
    destinations and group sizes, in the same order, on every rank."""
    rng = np.random.default_rng(dp_size * 1000 + num_tokens + 7 * skew)
    num_experts = dp_size * tp_size * local
    ids, weights = _routing(rng, dp_size, num_tokens, num_experts, topk, skew)
    ids[0, 1, topk - 1] = -1
    if padded:
        weights[:, -3:] = 0
        weights[1, 0, topk - 1] = np.nan
    w_bf16 = jnp.asarray(weights, jnp.bfloat16)
    gathered = jnp.asarray(ids)
    mesh = dict(dp_size=dp_size, tp_size=tp_size, local_num_experts=local)
    for ep_rank in range(dp_size * tp_size):
        dp_pos = ep_rank // tp_size
        member_index = functools.partial(
            _member_index,
            ep_rank=jnp.int32(ep_rank),
            dp_size=dp_size,
            tp_size=tp_size,
            local_num_experts=local,
        )
        es = jnp.asarray([ep_rank * local], jnp.int32)
        need_in = np.asarray((member_index(gathered) == dp_pos).any(axis=2))
        # Received rows: the sources in DP order, each in its token order;
        # capacity buffers are filled without gaps, full-size ones by region.
        sources = [(s, t) for s in range(dp_size) for t in range(num_tokens)]
        in_row = np.zeros((dp_size, num_tokens), np.int32)
        if capacity:
            rows = int(need_in.sum()) + 5
            got_rows = [(s, t) for s, t in sources if need_in[s, t]]
            for r, (s, t) in enumerate(got_rows):
                in_row[s, t] = r
            row_of = dict(enumerate(got_rows))
        else:
            rows = dp_size * num_tokens
            row_of = {}
            for s in range(dp_size):
                pos = np.cumsum(need_in[s]) - 1
                for t in range(num_tokens):
                    in_row[s, t] = s * num_tokens + pos[t]
                    if need_in[s, t]:
                        row_of[int(in_row[s, t])] = (s, t)
        recv_ids = np.full((rows, topk), -1, np.int32)
        recv_w = np.zeros((rows, topk), np.float32)
        sender_member = np.zeros(rows, np.int32)
        sender_token = np.zeros(rows, np.int32)
        for r, (s, t) in row_of.items():
            recv_ids[r] = ids[s, t]
            recv_w[r] = np.asarray(w_bf16[s, t].astype(jnp.float32))
            sender_member[r], sender_token[r] = s, t
        recv_ids_j = jnp.asarray(recv_ids)
        if combine == "owner":
            blk = -(-num_tokens // tp_size)
            row_member = sender_member * tp_size + sender_token // blk
            row_dest = (
                np.asarray(owner_order(recv_ids_j, **mesh)) * blk
                + (sender_token % blk)[:, None]
            )
            dests = functools.partial(owner_dests, **mesh)
        else:
            in_group = member_index(recv_ids_j) < dp_size
            row_dest = (
                np.asarray(route_order(recv_ids_j, in_group)) * num_tokens
                + sender_token[:, None]
            )
            row_member = sender_member
            dests = functools.partial(
                streamed_dests, member_index=member_index, dp_size=dp_size
            )
        routes = rows * topk if not capacity else int(rows * topk * 0.6) // 8 * 8
        num_local = int(
            ((ids >= ep_rank * local) & (ids < (ep_rank + 1) * local)).sum()
        )
        routes = max(routes, num_local)
        want = scatter_route_plan(
            jnp.asarray(recv_w, jnp.bfloat16),
            recv_ids_j,
            es[0],
            jnp.asarray(row_member, jnp.int32),
            jnp.asarray(row_dest, jnp.int32),
            local_num_experts=local,
            topk=topk,
            skip_padded_tokens=padded,
            max_local_routes=routes,
            dtype=jnp.bfloat16,
        )
        got = route_plan(
            gathered,
            jnp.asarray(in_row),
            jnp.bool_(True),
            es,
            dests,
            local_num_experts=local,
            routes=routes,
            rows=rows,
            gathered_weights=w_bf16 if padded else None,
        )
        want = [np.asarray(a) for a in want]
        got = [np.asarray(a) for a in got]
        np.testing.assert_array_equal(got[1], want[1])
        n = int(want[1].sum())
        for g, w in zip(got[:1] + got[2:], want[:1] + want[2:]):
            assert g.shape == w.shape
            np.testing.assert_array_equal(g[:n], w[:n])
        assert (got[0][n:] == rows).all()


@pytest.mark.cpu_test
def test_route_plan_has_no_routes_when_the_step_does_not_fit():
    rng = np.random.default_rng(5)
    ids, _ = _routing(rng, 8, 24, 512, 10, False)
    mesh = dict(dp_size=8, tp_size=4, local_num_experts=16)
    got = route_plan(
        jnp.asarray(ids),
        jnp.zeros((8, 24), jnp.int32),
        jnp.bool_(False),
        jnp.asarray([32], jnp.int32),
        functools.partial(owner_dests, **mesh),
        local_num_experts=16,
        routes=128,
        rows=96,
    )
    assert int(np.asarray(got[1]).sum()) == 0
    assert (np.asarray(got[0]) == 96).all()


@pytest.mark.cpu_test
@pytest.mark.parametrize("recv_rows", [1 << 14, 1 << 20])
def test_route_plan_wide_destinations_match_one_key(recv_rows):
    """Destinations too wide for one sort key ride the sort as a packed payload
    (2**14 receive rows) or as separate operands (2**20), with the same
    result."""
    rng = np.random.default_rng(6)
    ids, _ = _routing(rng, 8, 96, 512, 10, True)
    mesh = dict(dp_size=8, tp_size=4, local_num_experts=16)
    # Distinct receive rows per (source, token), as the exchange assigns them.
    in_row = jnp.asarray(np.arange(8 * 96).reshape(8, 96), jnp.int32)

    def wide(gathered_ids):
        member, row, members, _ = owner_dests(gathered_ids, **mesh)
        return member, row, members, recv_rows

    args = (jnp.asarray(ids), in_row, jnp.bool_(True), jnp.asarray([16], jnp.int32))
    kw = dict(local_num_experts=16, routes=512, rows=8 * 96)
    one_key = route_plan(*args, functools.partial(owner_dests, **mesh), **kw)
    got = route_plan(*args, wide, **kw)
    for a, b in zip(one_key, got):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.cpu_test
def test_served_route_fields_round_trip_through_the_packed_routing():
    """With replicas, the id field of a packed route holds its serving slot and
    owner order; the weight bits and an empty slot (serving -1) survive."""
    rng = np.random.default_rng(0)
    serving = rng.integers(-1, 64 * 9, size=(96, 10)).astype(np.int32)
    order = rng.integers(0, 16, size=(96, 10)).astype(np.int32)
    weights = jnp.asarray(rng.uniform(0, 1, size=(96, 10)), jnp.bfloat16)
    fields = pack_served_routes(jnp.asarray(serving), jnp.asarray(order))
    # The word _gather_routing packs.
    bits = jax.lax.bitcast_convert_type(weights, jnp.uint16).astype(jnp.int32)
    ids, got_weights = _unpack_routes(jnp.left_shift(fields, 16) | bits)
    got_serving, got_order = unpack_served_routes(ids)
    np.testing.assert_array_equal(np.asarray(got_serving), serving)
    np.testing.assert_array_equal(np.asarray(got_order), order)
    np.testing.assert_array_equal(
        np.asarray(got_weights).view(np.uint16), np.asarray(weights).view(np.uint16)
    )
    empty, _ = _unpack_routes(jnp.asarray([_EMPTY_ROUTE], jnp.int32))
    assert int(unpack_served_routes(empty)[0][0]) == -1


def _replicas_layout(dp_size, tp_size, num_tokens, local, topk, seed=0):
    """Skewed routing [D, S, K] (logical ids, one unrouted route) and a
    placement with one redundant slot per rank from the builder: the routing
    in physical primaries, the replica table and each physical slot's
    expert."""
    rng = np.random.default_rng(seed)
    ep_size = dp_size * tp_size
    num_experts = ep_size * local
    ids, weights = _routing(rng, dp_size, num_tokens, num_experts, topk, True)
    ids[0, 1, topk - 1] = -1
    counts = np.bincount(ids[ids >= 0], minlength=num_experts)
    slots = core.balance(counts, ep_size)
    redundant = core.place_replicas(
        counts, slots, ep_size, 1, core.replicate(counts, ep_size)
    )
    table = core.replica_table(slots, redundant, ep_size)
    primary = core.physical_slots(slots, ep_size, 1)
    expert_of = np.full(ep_size * (local + 1), -1)
    expert_of[primary] = np.arange(num_experts)
    expert_of[table[0][redundant >= 0]] = redundant[redundant >= 0]
    phys = np.where(ids >= 0, primary[np.maximum(ids, 0)], -1).astype(np.int32)
    return ids, weights, slots, table, phys, expert_of


def _serving(phys, table):
    topk = phys.shape[-1]
    got = serving_slots(jnp.asarray(phys.reshape(-1, topk)), jnp.asarray(table))
    return np.asarray(got).reshape(phys.shape)


@pytest.mark.cpu_test
def test_serving_slots_hold_the_routed_experts():
    ids, _, _, table, phys, expert_of = _replicas_layout(16, 4, 28, 8, 10)
    serving = _serving(phys, table)
    assert serving.dtype == np.int32
    np.testing.assert_array_equal(serving < 0, ids < 0)
    np.testing.assert_array_equal(expert_of[serving[ids >= 0]], ids[ids >= 0])
    assert (serving != phys).mean() > 0.1  # replicas serve a share of the routes
    # The full-size exchange's primaries come back exactly.
    got = primary_slots(jnp.asarray(serving), jnp.asarray(table))
    np.testing.assert_array_equal(np.asarray(got), phys)


@pytest.mark.cpu_test
def test_serving_slots_follow_each_token_alone_and_split_evenly():
    """The replica a route takes depends on its token's own routing only: a token
    alone, or the batch reordered, gets the same slots. Over many tokens the
    instances of a replicated expert share its routes about evenly."""
    _, _, _, table, phys, _ = _replicas_layout(16, 4, 256, 8, 10, seed=1)
    flat = phys.reshape(-1, phys.shape[-1])
    serving = _serving(flat, table)
    for i in (0, 5, 777, len(flat) - 1):
        np.testing.assert_array_equal(_serving(flat[i : i + 1], table)[0], serving[i])
    perm = np.random.default_rng(2).permutation(len(flat))
    np.testing.assert_array_equal(_serving(flat[perm], table), serving[perm])
    replica_slot, primary, _ = table
    checked = 0
    for p in np.unique(primary[primary >= 0]):
        replicas = np.concatenate([[p], replica_slot[primary == p]])
        n = np.array([(serving[flat == p] == r).sum() for r in replicas])
        if n.sum() >= 100:
            checked += 1
            share = n / n.sum() * len(replicas)
            assert np.abs(share - 1).max() < 0.3, (p, n)
    assert checked >= 16


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "dp_size,tp_size,num_tokens,local", [(16, 4, 28, 8), (2, 4, 92, 8), (4, 2, 26, 8)]
)
def test_route_plan_with_replicas_lands_every_route_at_its_primary_row(
    dp_size, tp_size, num_tokens, local
):
    """Every route is computed exactly once over the mesh, at its serving slot,
    and its output goes to the owner row it has without replicas: the senders'
    owner order from the primaries equals that of the file's slots."""
    topk = 10
    ids, _, slots, table, phys, _ = _replicas_layout(
        dp_size, tp_size, num_tokens, local, topk, seed=dp_size
    )
    mesh9 = dict(dp_size=dp_size, tp_size=tp_size, local_num_experts=local + 1)
    mesh8 = dict(dp_size=dp_size, tp_size=tp_size, local_num_experts=local)
    order = np.stack([np.asarray(owner_order(jnp.asarray(p), **mesh9)) for p in phys])
    file_ids = np.where(ids >= 0, slots[np.maximum(ids, 0)], -1)
    order_off = np.asarray(owner_order(jnp.asarray(file_ids), **mesh8))
    np.testing.assert_array_equal(order, order_off)
    serving = _serving(phys, table)
    gathered = jnp.asarray(serving)
    dests = functools.partial(
        owner_dests, **mesh9, order=jnp.moveaxis(jnp.asarray(order), -1, 0)
    )
    # The destinations without replicas, from the file's slots.
    member_off, row_off, _, _ = owner_dests(
        jnp.moveaxis(jnp.asarray(file_ids), -1, 0), **mesh8
    )
    member_off = np.broadcast_to(np.asarray(member_off), (dp_size, num_tokens))
    row_off = np.moveaxis(np.asarray(row_off), 0, -1)  # [D, S, K]
    blk = -(-num_tokens // tp_size)
    seen = np.zeros(ids.shape, np.int32)
    for ep_rank in range(dp_size * tp_size):
        dp_pos = ep_rank // tp_size
        member = _member_index(
            gathered,
            jnp.int32(ep_rank),
            dp_size=dp_size,
            tp_size=tp_size,
            local_num_experts=local + 1,
        )
        need_in = np.asarray((member == dp_pos).any(axis=2))
        got_rows = [(s, t) for s in range(dp_size) for t in range(num_tokens)]
        got_rows = [(s, t) for s, t in got_rows if need_in[s, t]]
        in_row = np.zeros((dp_size, num_tokens), np.int32)
        for r, (s, t) in enumerate(got_rows):
            in_row[s, t] = r
        rows = len(got_rows) + 5
        token, sizes, dest_member, dest_row = (
            np.asarray(a)
            for a in route_plan(
                gathered,
                jnp.asarray(in_row),
                jnp.bool_(True),
                jnp.asarray([ep_rank * (local + 1)], jnp.int32),
                dests,
                local_num_experts=local + 1,
                routes=rows * topk,
                rows=rows,
            )
        )
        group = np.repeat(np.arange(local + 1), sizes)
        for j, g in enumerate(group):
            s, t = got_rows[token[j]]
            k = int(np.flatnonzero(serving[s, t] == ep_rank * (local + 1) + g)[0])
            seen[s, t, k] += 1
            assert dest_member[j] == member_off[s, t] == s * tp_size + t // blk
            assert dest_row[j] == row_off[s, t, k]
    np.testing.assert_array_equal(seen, (ids >= 0).astype(np.int32))


@pytest.mark.cpu_test
@pytest.mark.parametrize("padded", [False, True])
def test_owner_waits_for_the_rows_other_ranks_serve(padded):
    dp_size, tp_size, num_tokens, local, topk = 4, 2, 26, 8, 10
    ids, weights, _, table, phys, _ = _replicas_layout(
        dp_size, tp_size, num_tokens, local, topk, seed=7
    )
    if padded:
        weights[:, -3:] = 0
    serving = _serving(phys, table)
    mesh = dict(dp_size=dp_size, tp_size=tp_size, local_num_experts=local + 1)
    blk = -(-num_tokens // tp_size)
    for d in range(dp_size):
        for r in range(tp_size):
            me = d * tp_size + r
            own = slice(r * blk, (r + 1) * blk)
            got = owner_remote_routes(
                jnp.asarray(phys[d, own]),
                jnp.asarray(weights[d, own], jnp.bfloat16),
                jnp.asarray(serving[d, own]),
                jnp.int32(me),
                skip_padded_tokens=padded,
                dtype=jnp.bfloat16,
                **mesh,
            )
            sent = (ids[d, own] >= 0) & ((weights[d, own] != 0) | (not padded))
            want = (sent & (serving[d, own] // (local + 1) != me)).sum()
            assert int(got) == want
            without = owner_remote_routes(
                jnp.asarray(phys[d, own]),
                jnp.asarray(weights[d, own], jnp.bfloat16),
                jnp.asarray(phys[d, own]),
                jnp.int32(me),
                skip_padded_tokens=padded,
                dtype=jnp.bfloat16,
                **mesh,
            )
            want = (sent & (phys[d, own] // (local + 1) != me)).sum()
            assert int(without) == want


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
@pytest.mark.parametrize("option,combine", CASES)
@pytest.mark.parametrize(
    "dp_size,tp_size,num_tokens,topk,factor,route_factor,skew,padded",
    [
        (8, 1, 24, 10, 1.15, 1.5, False, False),  # capacity exchange
        (8, 1, 24, 10, 1.15, 1.5, False, True),
        (8, 1, 12, 2, 2.0, 0.0, True, False),  # skewed: overflows
        (4, 2, 24, 10, 1.0, 1.5, False, False),
        (2, 4, 92, 10, 2.5, 3.0, False, False),  # proxy layout: full-size exchange
    ],
)
def test_option_keeps_every_output_bit(
    option,
    combine,
    dp_size,
    tp_size,
    num_tokens,
    topk,
    factor,
    route_factor,
    skew,
    padded,
):
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
        route_factor=route_factor,
        streamed_combine=combine == "streamed",
        owner_combine=combine == "owner",
    )
    kwargs = {}
    if combine == "owner":
        rng = np.random.default_rng(3)
        kwargs["shared"] = jnp.asarray(
            rng.standard_normal((dp_size, tp_size, num_tokens, HIDDEN)) / 8,
            jnp.bfloat16,
        )
    key = (dp_size, tp_size, num_tokens, topk, factor, route_factor, skew, padded)
    key += (combine,)
    if key not in _BASE:
        run = jax.jit(functools.partial(ragged_a2a_moe, **common))
        _BASE[key] = np.asarray(run(*stacked, **kwargs))
    base = _BASE[key]
    got = jax.jit(functools.partial(ragged_a2a_moe, **common, **OPTIONS[option]))(
        *stacked, **kwargs
    )
    got = np.asarray(got)
    assert np.abs(base.astype(np.float32)).max() > 0
    np.testing.assert_array_equal(got.view(np.uint16), base.view(np.uint16))


# Zeros, subnormals, infinities, NaNs and the smallest normals. The largest
# normals would overflow the experts into NaN everywhere; the kernel tests
# carry them.
_SPECIALS = np.array(
    [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x7F80, 0xFF80, 0x7FC0,
     0xFFC1, 0x0080, 0x8080],
    np.uint16,
)  # fmt: skip


@pytest.mark.multichip
@pytest.mark.parametrize("option", ["row_gather", "row_scatter", "row_copy"])
@pytest.mark.parametrize("combine", ["owner", "streamed"])
@pytest.mark.parametrize(
    "dp_size,tp_size,num_tokens,factor,route_factor",
    [
        (8, 1, 24, 1.15, 1.5),  # capacity exchange
        (2, 4, 92, 2.5, 3.0),  # full-size exchange
    ],
)
def test_row_copy_keeps_every_output_bit_with_special_values(
    option, combine, dp_size, tp_size, num_tokens, factor, route_factor
):
    """The row copies carry zeros, subnormals and non-finite token values the
    way the one-hot permute does."""
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    x, w1, w2, s1, s2, weights, ids, es = _inputs(
        8, tp_size, num_tokens, 64, topk=10, skew=False
    )
    rng = np.random.default_rng(11)
    bits = np.array(x).view(np.uint16)
    hit = rng.random(bits.shape) < 0.02
    bits[hit] = rng.choice(_SPECIALS, size=int(hit.sum()))
    # Tokens are replicated across their engine's TP ranks.
    bits = bits.reshape(-1, tp_size, num_tokens, HIDDEN)
    bits[:] = bits[:, :1]
    x = jnp.asarray(bits.reshape(-1, HIDDEN).view(jnp.bfloat16))
    stacked = tuple(
        a.reshape(dp_size, tp_size, -1, *a.shape[1:])
        for a in (x, w1, w2, s1, s2, weights, ids, es)
    )
    common = dict(
        mesh=mesh,
        topk=10,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(
            dp_size * num_tokens * 10
        ),
        capacity_factor=factor,
        route_factor=route_factor,
        streamed_combine=combine == "streamed",
        owner_combine=combine == "owner",
    )
    kwargs = {}
    if combine == "owner":
        kwargs["shared"] = jnp.asarray(
            rng.standard_normal((dp_size, tp_size, num_tokens, HIDDEN)) / 8,
            jnp.bfloat16,
        )
    base = jax.jit(functools.partial(ragged_a2a_moe, **common, sorted_plan=True))(
        *stacked, **kwargs
    )
    got = jax.jit(functools.partial(ragged_a2a_moe, **common, **OPTIONS[option]))(
        *stacked, **kwargs
    )
    base, got = np.asarray(base), np.asarray(got)
    assert np.isfinite(base.astype(np.float32)).all()
    np.testing.assert_array_equal(got.view(np.uint16), base.view(np.uint16))
