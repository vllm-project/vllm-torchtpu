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
"""The plan kernel of the capacity exchange against the XLA ops it replaces:
the fit check, sizes, offsets, send index and receive rows, for every rank."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox.exchange_plan import (
    exchange_plan,
    exchange_plan_supported,
)
from vllm_torchtpu.layers.core import ragged_a2a_moe as ram
from vllm_torchtpu.layers.core.expert_placement import ROUTE_ORDER_BITS

TOPK = 10


def _require_tpu() -> None:
    if jax.default_backend() != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {jax.default_backend()}.")


def reference(words, ep_rank, *, tp_size, slots_per_rank, rows, routes, order_bits):
    """The capacity exchange's XLA ops (`_capacity_overflow` and the planned
    branch of `_capacity_exchange` with the row scatter) on the gathered words
    [D, K, S]."""
    num_members, _, num_tokens = words.shape
    ids = jnp.swapaxes(words, 1, 2) >> 16
    if order_bits:
        ids = ram.unpack_served_routes(ids)[0]
    over = ram._capacity_overflow(
        ids,
        tp_size=tp_size,
        local_num_experts=slots_per_rank,
        rows=rows,
        routes=routes,
    )
    dp_pos = ep_rank // tp_size
    member_index = functools.partial(
        ram._member_index,
        ep_rank=ep_rank,
        dp_size=num_members,
        tp_size=tp_size,
        local_num_experts=slots_per_rank,
    )
    member = jnp.arange(num_members, dtype=jnp.int32)
    token = jnp.arange(num_tokens, dtype=jnp.int32)
    fits = jnp.logical_not(over)
    need_any = (member_index(ids[dp_pos])[:, :, None] == member).any(axis=1)
    need = (member_index(ids)[..., None] == member).any(axis=2) & fits
    need_in = (need & (member == dp_pos)).any(axis=2)
    counts = need.sum(axis=1, dtype=jnp.int32)
    send_sizes = (need_any & fits).sum(axis=0, dtype=jnp.int32)
    recv_sizes = need_in.sum(axis=1, dtype=jnp.int32)
    recv_start = ram._exclusive_cumsum(recv_sizes, 0)
    earlier = (member < dp_pos).astype(jnp.int32)
    out_row = (
        ram._exclusive_cumsum(need_any.sum(axis=0, dtype=jnp.int32), 0)
        + jnp.cumsum(need_any, axis=0, dtype=jnp.int32)
        - 1
    )
    out_row = jnp.where(need_any, out_row, rows)
    row = jnp.arange(rows, dtype=jnp.int32)
    return dict(
        overflow=over,
        send_start=ram._exclusive_cumsum(send_sizes, 0),
        send_sizes=send_sizes,
        send_landing=(counts * earlier[:, None]).sum(axis=0),
        recv_sizes=recv_sizes,
        need_in=need_in,
        in_row=recv_start[:, None] + jnp.cumsum(need_in, axis=1, dtype=jnp.int32) - 1,
        send_index=jnp.where(out_row[:, :, None] == row, token[:, None, None], 0).sum(
            axis=(0, 1)
        ),
        count=jnp.max(jnp.where(out_row < rows, out_row + 1, 0), initial=0),
        src_rows=recv_sizes.sum(),
    )


def gathered_words(rng, num_members, tp_size, num_tokens, slots_per_rank, *, routing,
                   order_bits=0):  # fmt: skip
    """Packed routing [D, K, S] as the routing all-gather delivers it:
    ``uniform``, ``skewed`` (most routes on the first members), ``hot_rank``
    (as many routes as fit on one rank), ``padded`` (uniform with a sixth of
    the tokens unrouted) or ``unrouted``."""
    ranks = num_members * tp_size
    slots = ranks * slots_per_rank
    if routing in ("uniform", "padded", "unrouted"):
        ids = np.stack(
            [rng.permutation(slots)[:TOPK] for _ in range(num_members * num_tokens)]
        )
    elif routing == "skewed":
        weights = np.exp(-np.arange(slots) / (slots / 6))
        ids = np.stack([
            rng.choice(slots, TOPK, replace=False, p=weights / weights.sum())
            for _ in range(num_members * num_tokens)
        ])  # fmt: skip
    elif routing == "hot_rank":
        hot = min(TOPK, slots_per_rank)
        ids = np.stack([
            np.r_[rng.permutation(slots_per_rank)[:hot],
                  rng.permutation(np.arange(slots_per_rank, slots))[:TOPK - hot]]
            for _ in range(num_members * num_tokens)
        ])  # fmt: skip
    else:
        raise ValueError(routing)
    ids = ids.reshape(num_members, num_tokens, TOPK)
    if routing == "padded":
        ids[:, -max(1, num_tokens // 6) :] = -1
    if routing == "unrouted":
        ids[:] = -1
    field = ids
    if order_bits:
        field = np.where(
            ids >= 0, ids << order_bits | rng.integers(0, TOPK, ids.shape), -1
        )
    words = (field.astype(np.int64) << 16) | rng.integers(0, 1 << 16, ids.shape)
    return np.swapaxes(words, 1, 2).astype(np.int32)


def check(plan, want, num_tokens):
    count = int(want["count"])
    for name in ("send_start", "send_sizes", "send_landing", "recv_sizes"):
        np.testing.assert_array_equal(
            np.asarray(getattr(plan, name)), np.asarray(want[name]), name
        )
    assert bool(plan.overflow) == bool(want["overflow"])
    assert int(plan.count[0]) == count
    assert int(plan.src_rows[0]) == int(want["src_rows"])
    np.testing.assert_array_equal(
        np.asarray(plan.send_index)[:count], np.asarray(want["send_index"])[:count]
    )
    need_in = np.asarray(want["need_in"])
    np.testing.assert_array_equal(
        np.where(need_in, np.asarray(plan.in_row), 0),
        np.where(need_in, np.asarray(want["in_row"]), 0),
    )  # fmt: skip


def _capacity(num_members, tp_size, num_tokens, slots_per_rank, factor):
    return ram.exchange_capacity(
        dp_size=num_members,
        tp_size=tp_size,
        num_tokens=num_tokens,
        topk=TOPK,
        local_num_experts=slots_per_rank,
        factor=factor,
        onehot_moe_permute_threshold=1 << 30,
        use_sparse_core=True,
        row_tile=32,
    )


# (members, TP size, slots per rank, order bits): the exchanges of 8 and 16
# members with 16 / 8 / 9 slots per rank (9: one redundant slot, the routing
# field carrying the owner order), and the one-host test layouts.
LAYOUTS = [
    (8, 4, 16, 0),
    (16, 4, 8, 0),
    (16, 4, 9, ROUTE_ORDER_BITS),
    (2, 4, 8, 0),
    (4, 2, 8, 0),
    (8, 1, 8, 0),
]
TOKENS = [4, 8, 26, 104, 128]


def _cases(layout, num_tokens):
    """(routing, factor) pairs: capacity sizes from fitting to overflowing."""
    return [
        ("uniform", 2.0),
        ("padded", 1.15),
        ("skewed", 1.0),
        ("hot_rank", 4.0),
        ("unrouted", 2.0),
        ("uniform", 0.3),
    ]


def _run(layout, num_tokens, ranks, interpret):
    num_members, tp_size, slots_per_rank, order_bits = layout
    rng = np.random.default_rng(num_members * 1000 + num_tokens * 10 + tp_size)
    for routing, factor in _cases(layout, num_tokens):
        rows, routes = _capacity(
            num_members, tp_size, num_tokens, slots_per_rank, factor
        )
        if rows >= num_members * num_tokens:
            # The full-size exchange only: no capacity plan.
            rows = max(32, (num_members * num_tokens) // 2 // 32 * 32)
            routes = max(128, rows)
        words = gathered_words(
            rng, num_members, tp_size, num_tokens, slots_per_rank,
            routing=routing, order_bits=order_bits,
        )  # fmt: skip
        common = dict(
            tp_size=tp_size,
            slots_per_rank=slots_per_rank,
            rows=rows,
            routes=routes,
            order_bits=order_bits,
        )
        for ep in ranks(num_members * tp_size, rng):
            plan = exchange_plan(
                jnp.asarray(words),
                jnp.array([ep], jnp.int32),
                interpret=interpret,
                **common,
            )
            check(plan, reference(jnp.asarray(words), ep, **common), num_tokens)


@pytest.mark.cpu_test
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("num_tokens", TOKENS)
def test_plan_kernel_matches_the_xla_plan(layout, num_tokens):
    """Interpret mode: the kernel's plan equals the XLA ops' for fitting and
    overflowing steps, padded and unrouted tokens, at three ranks."""

    def ranks(n, rng):
        return sorted({0, n - 1, int(rng.integers(n))})

    _run(layout, num_tokens, ranks, interpret=True)


@pytest.mark.cpu_test
def test_plan_kernel_matches_the_xla_plan_on_every_rank():
    """Interpret mode, the 8-member layout at 26 tokens: every EP rank."""
    _run((8, 4, 16, 0), 26, lambda n, rng: range(n), interpret=True)


@pytest.mark.cpu_test
def test_plan_kernel_counts_received_rows_when_routes_exceed_the_rows():
    """Some rank's routes exceed the rows but its rows fit: the exact
    received-row count decides (no fallback); one more route per rank does
    not fit."""
    num_members, tp_size, slots_per_rank = 8, 4, 16
    rng = np.random.default_rng(7)
    # Every token sends two routes to rank 0: rank 0 sees twice its rows.
    words = gathered_words(
        rng, num_members, tp_size, 26, slots_per_rank, routing="uniform"
    )
    ids = (np.swapaxes(words, 1, 2) >> 16).copy()
    ids[:, :, 0], ids[:, :, 1] = 0, 1
    for k in range(2, TOPK):
        ids[:, :, k] = np.where(ids[:, :, k] < 2, ids[:, :, k] + 2, ids[:, :, k])
    words = np.swapaxes((ids.astype(np.int64) << 16) | 7, 1, 2).astype(np.int32)
    for rows in (num_members * 26, num_members * 26 - 32):
        common = dict(tp_size=tp_size, slots_per_rank=slots_per_rank, rows=rows,
                      routes=4096, order_bits=0)  # fmt: skip
        for ep in (0, 5):
            plan = exchange_plan(
                jnp.asarray(words), jnp.array([ep], jnp.int32), interpret=True, **common
            )
            want = reference(jnp.asarray(words), ep, **common)
            check(plan, want, 26)
        assert bool(plan.overflow) == (rows < num_members * 26)


@pytest.mark.cpu_test
def test_the_plan_kernel_takes_the_served_layouts():
    for num_members, tp_size, num_tokens, slots in [(8, 4, 104, 16), (16, 4, 128, 9)]:
        assert exchange_plan_supported(
            num_members=num_members,
            tp_size=tp_size,
            num_tokens=num_tokens,
            topk=TOPK,
            slots_per_rank=slots,
        )
    # Past 128 tokens per member or 64 ranks the exchange keeps XLA's ops.
    assert not exchange_plan_supported(
        num_members=16, tp_size=4, num_tokens=256, topk=TOPK, slots_per_rank=8
    )
    assert not exchange_plan_supported(
        num_members=32, tp_size=4, num_tokens=128, topk=TOPK, slots_per_rank=4
    )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("num_tokens", TOKENS)
def test_plan_kernel_matches_the_xla_plan_on_tpu(layout, num_tokens):
    """Compiled for TPU: the same cases at every rank."""
    _require_tpu()
    _run(layout, num_tokens, lambda n, rng: range(n), interpret=False)


# The balanced replica choice.


def balanced_reference(prim, table, *, slots_per_rank, num_slots):
    """numpy form of the kernel's choice: the serving slot of every route
    (``prim`` [...] primary slots, -1 unrouted, last axis the token's
    routes)."""
    prim = np.where((prim >= 0) & (prim < num_slots), prim, -1).astype(np.int64)
    table = table.astype(np.int64)
    inst_rank, inst_slot, held = table[0:8], table[8:16], table[16:32]
    group_of = table[32:40].reshape(-1)
    routed = prim >= 0
    g = np.where(routed, group_of[np.maximum(prim, 0)], -1)
    total = np.maximum(prim, 0).sum(-1, keepdims=True)
    key = ((total * 1031 + prim * 7) >> 3) & 511
    rank = np.where(routed, prim // slots_per_rank, -1)
    prim_load = np.bincount(rank[routed], minlength=128)[:128]
    n_g = np.bincount(g[g >= 0], minlength=128)[:128]
    valid = inst_rank >= 0
    sub = np.arange(8)[:, None]

    def rank_share(x):
        out = np.zeros(128, np.int64)
        for entry in held:
            col = x[:, entry & 127]
            out += np.where((sub == entry >> 7) & (entry >= 0), col, 0).sum(0)
        return out

    n8 = np.broadcast_to(n_g, (8, 128))
    base = prim_load - rank_share(np.where(sub == 0, n8, 0))
    ninst = np.maximum(valid.sum(0), 1)
    x = np.where(valid, n8 // ninst, 0)
    x[0] += n_g - x.sum(0)
    for _ in range(8):
        load = base + rank_share(x)
        others = np.where(valid, load[np.maximum(inst_rank, 0)] - x, 1 << 20)
        lo = np.broadcast_to(others.min(0), (8, 128)).copy()
        hi = lo + n8
        for _ in range(12):
            mid = (lo + hi + 1) >> 1
            fits = np.maximum(mid - others, 0).sum(0) <= n8
            lo, hi = np.where(fits, mid, lo), np.where(fits, hi, mid - 1)
        target = np.maximum(lo - others, 0)
        left = n8 - target.sum(0)
        level = (others <= lo) & valid
        target = target + (level & (np.cumsum(level, 0) <= left))
        x = (x + target) >> 1
        x[0] += n_g - x.sum(0)
    cum = np.cumsum(x, 0)
    g0 = np.maximum(g, 0)
    kk = (key * n_g[g0]) >> 9
    j = sum((cum[jj][g0] <= kk).astype(np.int64) for jj in range(7))
    slot = np.where(j > 0, inst_slot[np.minimum(j, 7), g0], prim)
    return np.where(g >= 0, slot, prim)


def _replica_layout(num_members, tp_size, slots, redundant_per_rank, seed=0):
    """A balanced placement with replicas of the busiest experts from skewed
    counts: (physical primary of each expert, replica table, plan table)."""
    from vllm_torchtpu.kernels.megablox.exchange_plan import replica_plan_table
    from vllm_torchtpu.layers.core import expert_placement as core

    ranks = num_members * tp_size
    experts = ranks * slots
    rng = np.random.default_rng(seed)
    counts = rng.zipf(1.6, experts).astype(np.float64)
    placed = core.balance(counts, ranks)
    extra = ranks * redundant_per_rank
    redundant = core.place_replicas(
        counts, placed, ranks, redundant_per_rank, core.replicate(counts, extra, 7)
    )
    table = core.replica_table(placed, redundant, ranks)
    plan = replica_plan_table(
        table, slots_per_rank=slots + redundant_per_rank, ep_size=ranks
    )
    primary = core.physical_slots(placed, ranks, redundant_per_rank)
    return counts, primary, table, plan


def _balanced_words(rng, counts, primary, num_members, num_tokens, padded=False):
    """Gathered words [D, K, S] with primaries and owner orders in the id
    fields, experts drawn by their counts, and the primaries [D, S, K]."""
    p = counts / counts.sum()
    ids = np.stack(
        [
            rng.choice(len(counts), TOPK, replace=False, p=p)
            for _ in range(num_members * num_tokens)
        ]
    ).reshape(num_members, num_tokens, TOPK)
    prim = primary[ids]
    if padded:
        prim[:, -max(1, num_tokens // 6) :] = -1
    order = np.argsort(np.argsort(prim, axis=-1), axis=-1)
    field = np.where(prim >= 0, prim << ROUTE_ORDER_BITS | order, -1)
    words = (field.astype(np.int64) << 16) | rng.integers(0, 1 << 16, prim.shape)
    return np.swapaxes(words, 1, 2).astype(np.int32), prim, order


BALANCED_LAYOUTS = [(16, 4, 8, 1), (8, 4, 16, 1), (2, 4, 8, 1), (4, 2, 8, 1)]


@pytest.mark.cpu_test
@pytest.mark.parametrize("layout", BALANCED_LAYOUTS)
@pytest.mark.parametrize("num_tokens", [8, 26, 128])
def test_balanced_choice_matches_the_numpy_form_on_every_rank(layout, num_tokens):
    """Interpret mode: every rank's kernel chooses the same serving slots, the
    numpy form's; each serves the route's expert; the plan is the XLA ops' on
    those slots, fallback decision included; and every rank's sends to a member
    are what that member expects to receive."""
    num_members, tp_size, slots, redundant = layout
    counts, primary, table, plan_table = _replica_layout(
        num_members, tp_size, slots, redundant
    )
    slots_per_rank = slots + redundant
    num_slots = num_members * tp_size * slots_per_rank
    rng = np.random.default_rng(num_tokens)
    rows, routes = _capacity(num_members, tp_size, num_tokens, slots, 2.0)
    if rows >= num_members * num_tokens:
        rows = max(32, num_members * num_tokens // 2 // 32 * 32)
    for padded in (False, True):
        words, prim, order = _balanced_words(
            rng, counts, primary, num_members, num_tokens, padded
        )
        want = balanced_reference(
            prim, plan_table, slots_per_rank=slots_per_rank, num_slots=num_slots
        )
        # Every serving slot holds the route's expert.
        held_by = dict(enumerate(range(num_slots)))
        for q in np.flatnonzero(table[1] >= 0):
            held_by[int(table[0][q])] = int(table[1][q])
        assert all(held_by[s] == p for s, p in zip(want[prim >= 0], prim[prim >= 0]))
        served_words = (
            np.where(want >= 0, want << ROUTE_ORDER_BITS | order, -1).astype(np.int64)
            << 16
        )
        served_words = np.swapaxes(served_words, 1, 2).astype(np.int32)
        common = dict(
            tp_size=tp_size,
            slots_per_rank=slots_per_rank,
            rows=rows,
            routes=routes,
            order_bits=ROUTE_ORDER_BITS,
        )
        plans = {}
        for ep in range(num_members * tp_size):
            got = exchange_plan(
                jnp.asarray(words),
                jnp.array([ep], jnp.int32),
                replica_plan=jnp.asarray(plan_table),
                interpret=True,
                **common,
            )
            np.testing.assert_array_equal(
                np.moveaxis(np.asarray(got.serving), 0, -1), want
            )
            check(got, reference(jnp.asarray(served_words), ep, **common), num_tokens)
            plans[ep] = got
        for ep, got in plans.items():
            col, dp_pos = ep % tp_size, ep // tp_size
            for m in range(num_members):
                peer = plans[m * tp_size + col]
                assert int(got.send_sizes[m]) == int(peer.recv_sizes[dp_pos])


@pytest.mark.cpu_test
def test_balanced_choice_decides_the_fallback_on_its_own_slots():
    """Capacity between the busiest rank under the primaries and under the
    balanced choice: the step fits with the choice, and the kernel says so."""
    num_members, tp_size, slots, redundant = 16, 4, 8, 1
    counts, primary, table, plan_table = _replica_layout(
        num_members, tp_size, slots, redundant, seed=3
    )
    slots_per_rank = slots + redundant
    num_slots = num_members * tp_size * slots_per_rank
    rng = np.random.default_rng(11)
    words, prim, order = _balanced_words(rng, counts, primary, num_members, 128)
    want = balanced_reference(
        prim, plan_table, slots_per_rank=slots_per_rank, num_slots=num_slots
    )
    busiest_primary = np.bincount(prim[prim >= 0] // slots_per_rank).max()
    busiest_served = np.bincount(want[want >= 0] // slots_per_rank).max()
    assert busiest_served < busiest_primary
    routes = int(busiest_served + busiest_primary) // 2
    common = dict(
        tp_size=tp_size,
        slots_per_rank=slots_per_rank,
        rows=num_members * 128,
        routes=routes,
        order_bits=ROUTE_ORDER_BITS,
    )
    got = exchange_plan(
        jnp.asarray(words),
        jnp.array([0], jnp.int32),
        replica_plan=jnp.asarray(plan_table),
        interpret=True,
        **common,
    )
    assert not bool(got.overflow)
    primaries = exchange_plan(
        jnp.asarray(words), jnp.array([0], jnp.int32), interpret=True, **common
    )
    assert bool(primaries.overflow)


@pytest.mark.cpu_test
@pytest.mark.parametrize("layout", [(16, 4, 8, 1), (2, 4, 8, 1)])
def test_owners_expect_the_rows_the_expert_ranks_send(layout):
    """With the kernel's serving slots, the routes each expert rank's route plan
    sends to an owner rank, summed over the other ranks, are the rows that owner
    expects (its owned routes served elsewhere)."""
    num_members, tp_size, slots, redundant = layout
    counts, primary, table, plan_table = _replica_layout(
        num_members, tp_size, slots, redundant, seed=5
    )
    slots_per_rank = slots + redundant
    ranks = num_members * tp_size
    num_tokens = 26
    rng = np.random.default_rng(2)
    words, prim, order = _balanced_words(
        rng, counts, primary, num_members, num_tokens, padded=True
    )
    got = exchange_plan(
        jnp.asarray(words),
        jnp.array([0], jnp.int32),
        tp_size=tp_size,
        slots_per_rank=slots_per_rank,
        rows=num_members * num_tokens,
        routes=num_members * num_tokens * TOPK,
        order_bits=ROUTE_ORDER_BITS,
        replica_plan=jnp.asarray(plan_table),
        interpret=True,
    )
    serving = np.moveaxis(np.asarray(got.serving), 0, -1)  # [D, S, K]
    mesh = dict(dp_size=num_members, tp_size=tp_size, local_num_experts=slots_per_rank)
    blk = -(-num_tokens // tp_size)
    weights = np.where(prim >= 0, 1.0, 0.0).astype(np.float32)
    sent = np.zeros(ranks, np.int64)
    for r in range(ranks):
        dests = functools.partial(
            ram.owner_dests, order=jnp.asarray(np.moveaxis(order, -1, 0)), **mesh
        )
        member, _, _, _ = dests(jnp.asarray(np.moveaxis(serving, -1, 0)))
        local = serving - r * slots_per_rank
        mine = (local >= 0) & (local < slots_per_rank)
        owners = np.broadcast_to(np.asarray(member)[:, :, None], serving.shape)
        np.add.at(sent, owners[mine & (owners != r)], 1)
    for o in range(ranks):
        s, t0 = o // tp_size, (o % tp_size) * blk
        owned = slice(t0, min(t0 + blk, num_tokens))
        expected = ram.owner_remote_routes(
            jnp.asarray(prim[s, owned]),
            jnp.asarray(weights[s, owned]),
            jnp.asarray(serving[s, owned]),
            o,
            skip_padded_tokens=False,
            dtype=jnp.float32,
            **mesh,
        )
        assert int(expected) == int(sent[o]), o


@pytest.mark.cpu_test
def test_replica_plan_table_refuses_more_instances_than_it_takes():
    from vllm_torchtpu.kernels.megablox.exchange_plan import (
        REPLICA_INSTANCES,
        replica_plan_table,
    )

    ranks, slots = 16, 9
    own = np.arange(ranks) * slots + 8
    primary = np.full(ranks, -1)
    primary[:REPLICA_INSTANCES] = 0  # expert slot 0 with 8 replicas: 9 instances
    index = np.where(primary >= 0, np.cumsum(primary >= 0), 0)
    table = np.stack([own, primary, index]).astype(np.int32)
    with pytest.raises(ValueError, match="instances"):
        replica_plan_table(table, slots_per_rank=slots, ep_size=ranks)
    primary[REPLICA_INSTANCES - 1] = -1  # 7 replicas: 8 instances
    index = np.where(primary >= 0, np.cumsum(primary >= 0), 0)
    table = np.stack([own, primary, index]).astype(np.int32)
    plan = replica_plan_table(table, slots_per_rank=slots, ep_size=ranks)
    assert (plan[0:8, 0] >= 0).sum() == REPLICA_INSTANCES
    assert plan[32, 0] == 0 and (plan[32, 1:] == -1).all()


@pytest.mark.parametrize("layout", BALANCED_LAYOUTS)
@pytest.mark.parametrize("num_tokens", [8, 128])
def test_balanced_choice_matches_the_numpy_form_on_tpu(layout, num_tokens):
    """Compiled for TPU: the serving slots and every rank's plan."""
    _require_tpu()
    num_members, tp_size, slots, redundant = layout
    counts, primary, table, plan_table = _replica_layout(
        num_members, tp_size, slots, redundant
    )
    slots_per_rank = slots + redundant
    num_slots = num_members * tp_size * slots_per_rank
    rng = np.random.default_rng(num_tokens)
    rows, routes = _capacity(num_members, tp_size, num_tokens, slots, 2.0)
    if rows >= num_members * num_tokens:
        rows = max(32, num_members * num_tokens // 2 // 32 * 32)
    words, prim, order = _balanced_words(rng, counts, primary, num_members, num_tokens)
    want = balanced_reference(
        prim, plan_table, slots_per_rank=slots_per_rank, num_slots=num_slots
    )
    served_words = (
        np.where(want >= 0, want << ROUTE_ORDER_BITS | order, -1).astype(np.int64) << 16
    )
    served_words = np.swapaxes(served_words, 1, 2).astype(np.int32)
    common = dict(
        tp_size=tp_size,
        slots_per_rank=slots_per_rank,
        rows=rows,
        routes=routes,
        order_bits=ROUTE_ORDER_BITS,
    )
    for ep in range(num_members * tp_size):
        got = exchange_plan(
            jnp.asarray(words),
            jnp.array([ep], jnp.int32),
            replica_plan=jnp.asarray(plan_table),
            **common,
        )
        np.testing.assert_array_equal(np.moveaxis(np.asarray(got.serving), 0, -1), want)
        check(got, reference(jnp.asarray(served_words), ep, **common), num_tokens)
