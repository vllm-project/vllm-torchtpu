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
"""Experimental sparse dispatch (`fused_ep_moe_v2(..., sparse_waves=W)`).

Sparse dispatch changes only how the routed rows reach the expert shards: the
same bytes arrive by a pairwise exchange instead of the all-gather, and the
kernel runs the same matmuls on them in another expert order. So the gate is
bit equality with `sparse_waves=0` on the same inputs.

The routing is forced through `hash_topk_ids`, so each case pins how the
pairs spread over the shards:

  uniform  random experts: every shard is near its uniform share;
  skewed   7 of 8 tokens on shard 0's experts only;
  worst    every token on experts 0..topk-1, i.e. all of it on shard 0 -- the
           largest exchange and expert slab any routing can produce, which is
           what the no-drop buffer bounds are sized for;
  dominant every shard's local expert 0 takes one token in ep, from every
           source shard: the one-big-expert-per-shard shape a placement
           table leaves, which streaming spreads over the waves.

waves=1 waits for the whole exchange before the first expert; waves equal to
the experts per shard waits once per expert, in the fewest-rows-first order.
sparse_stream spreads each shard's busiest expert over the waves by source
group and waits for it tile by tile; it is held to the same bit equality.

The plan tests replay the exchange the dispatch plan describes in numpy, on
CPU: every row a shard needs lands exactly once, no later than the wave of
the first of its experts on that shard (for a streamed expert, the wave of
the token's source group), and the per-wave counts the kernel waits for are
exactly what the senders send.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v2 import (
    AXIS,
    WeightFormat,
    fused_ep_moe_v2,
    sparse_dispatch,
)
from vllm_torchtpu.kernels.fused_moe.v2.host import (
    ROWBLK,
    align_up,
    owner_first_run_starts,
    sparse_experts_per_wave,
)

EP = 8
EXPERTS_PER_SHARD = 4
E_TOTAL = EP * EXPERTS_PER_SHARD
HIDDEN = 1024
INTER = 512
TOPK = 4
CAPACITY = 128
FP8 = jnp.float8_e4m3fn


def _expert_ids(kind, tokens, e_total, topk, experts_per_shard, seed):
    """[tokens, topk] expert ids, distinct within a row as a real table is."""
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        rows = [rng.choice(e_total, topk, replace=False) for _ in range(tokens)]
    elif kind == "skewed":
        # 7 of 8 tokens pick only from shard 0's experts.
        hot = np.arange(max(topk, experts_per_shard))
        rows = [
            rng.choice(hot if t % 8 else e_total, topk, replace=False)
            for t in range(tokens)
        ]
    elif kind == "worst":
        rows = [np.arange(topk)] * tokens
    elif kind == "dominant":
        # Token t hits shard t % ep's local expert 0, plus topk - 1 others.
        ep = e_total // experts_per_shard
        rows = []
        for t in range(tokens):
            hot = (t % ep) * experts_per_shard
            others = np.delete(np.arange(e_total), hot)
            rows.append([hot, *rng.choice(others, topk - 1, replace=False)])
    else:
        raise ValueError(kind)
    return np.asarray(rows, np.int32)


# -- the dispatch plan, on CPU ----------------------------------------------


def test_experts_are_visited_fewest_rows_first_and_empties_last():
    hist = np.array([[5, 0, 2, 2], [0, 0, 0, 0], [1, 9, 3, 0]])
    pos, order = sparse_dispatch.ascending_expert_order(
        jnp.asarray(hist.reshape(-1), jnp.int32), ep=3, g_local=4
    )
    # Ties break by index, as in expert_visit_list.
    want = np.array([[2, 3, 0, 1], [0, 1, 2, 3], [0, 2, 1, 3]])
    np.testing.assert_array_equal(np.asarray(order), want)
    np.testing.assert_array_equal(
        np.take_along_axis(np.asarray(pos), want, axis=1),
        np.broadcast_to(np.arange(4), (3, 4)),
    )


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "stream"])
@pytest.mark.parametrize("routing", ["uniform", "skewed", "worst", "dominant"])
@pytest.mark.parametrize(
    "ep,t_local,g_local,topk,waves",
    [
        pytest.param(4, 16, 4, 3, 1, id="ep4_w1"),
        pytest.param(4, 16, 4, 3, 2, id="ep4_w2"),
        pytest.param(4, 16, 4, 3, 4, id="ep4_w4"),
        pytest.param(8, 16, 4, 3, 4, id="ep8_w4"),
        # More selections than shards: a token reaches at most ep shards.
        pytest.param(2, 24, 8, 4, 4, id="ep2_topk4"),
    ],
)
def test_plan_delivers_every_needed_row_once_by_its_wave(
    routing, ep, t_local, g_local, topk, waves, stream
):
    groups = sparse_dispatch.stream_groups(ep, waves) if stream else ()
    if stream and not groups:
        pytest.skip(f"nothing to stream at ep={ep} with {waves} waves")
    T = ep * t_local
    ids = _expert_ids(routing, T, ep * g_local, topk, g_local, seed=ep + waves)
    hist = jnp.asarray(np.bincount(ids.reshape(-1), minlength=ep * g_local), jnp.int32)
    pos, order = sparse_dispatch.ascending_expert_order(hist, ep=ep, g_local=g_local)
    wave_of = np.asarray(pos) // sparse_experts_per_wave(g_local, waves, stream=stream)
    split = (
        sparse_dispatch.stream_split(hist, order, ep=ep, g_local=g_local)
        if stream
        else None
    )
    send_alloc = align_up(t_local * min(topk, ep), ROWBLK)
    plans = [
        tuple(
            np.asarray(a)
            for a in sparse_dispatch.dispatch_plan(
                jnp.asarray(ids),
                jnp.int32(me),
                ep=ep,
                t_local=t_local,
                g_local=g_local,
                waves=waves,
                send_alloc=send_alloc,
                wave_of=jnp.asarray(wave_of),
                split=split,
                stream=groups,
            )
        )
        for me in range(ep)
    ]

    def owed_wave(t, e):
        """The wave pair (t, e) is owed by: its expert's, or, for a
        streamed expert, the wave of t's source group counted from e's
        shard."""
        d, local = divmod(int(e), g_local)
        if stream and local == int(split[d]):
            rel = (t // t_local - d) % ep
            return int(np.searchsorted(np.cumsum(groups), rel, side="right"))
        return int(wave_of[d, local])

    # Replay the exchange: every (destination, wave) segment of every sender.
    landed = np.full((ep, T), -1)
    landed_wave = np.full((ep, T), -1)
    for src, (perm, tab, _) in enumerate(plans):
        remote = 0
        for d in range(ep):
            for w in range(waves):
                off, n, dst = tab[d, w], tab[ep + d, w], tab[2 * ep + d, w]
                assert off + n <= send_alloc
                assert (landed[d, dst : dst + n] == -1).all(), "rows overlap"
                landed[d, dst : dst + n] = src * t_local + perm[off : off + n]
                landed_wave[d, dst : dst + n] = w
                remote += n if d != src else 0
        assert tab[3 * ep + 1, 0] == remote

    dest = ids // g_local
    for d, (_, tab, posmap) in enumerate(plans):
        # The kernel waits for exactly what the senders send, wave by wave.
        for w in range(waves):
            assert tab[3 * ep, w] == sum(p[1][ep + d, w] for p in plans)
        needed = np.flatnonzero((dest == d).any(axis=1))
        # Every needed token exactly once, packed from row 0.
        np.testing.assert_array_equal(np.sort(landed[d, : needed.size]), needed)
        assert (landed[d, needed.size :] == -1).all()
        for t in needed:
            row = posmap[t]
            assert landed[d, row] == t
            first = min(owed_wave(t, e) for e in ids[t] if e // g_local == d)
            assert landed_wave[d, row] == first


@pytest.mark.parametrize(
    "ep,waves,want",
    [
        (16, 4, (2, 2, 4, 8)),
        (8, 4, (1, 1, 2, 4)),
        (16, 2, (8, 8)),
        (16, 1, ()),
        (4, 4, ()),
    ],
)
def test_stream_groups_double_and_cover_every_source(ep, waves, want):
    assert sparse_dispatch.stream_groups(ep, waves) == want
    if want:
        assert len(want) == waves and sum(want) == ep


def test_the_streamed_expert_is_the_last_visited():
    hist = jnp.asarray([5, 0, 2, 2, 0, 0, 0, 0, 1, 9, 3, 9], jnp.int32)
    _, order = sparse_dispatch.ascending_expert_order(hist, ep=3, g_local=4)
    split = sparse_dispatch.stream_split(hist, order, ep=3, g_local=4)
    # Shard 0 visits 2, 3, 0, so 0 is last; shard 1 has no rows at all; on
    # shard 2 the tie between 1 and 3 breaks by index, so 3 is last.
    np.testing.assert_array_equal(np.asarray(split), [0, 0, 3])


@pytest.mark.parametrize("waves", [2, 4, 8])
@pytest.mark.parametrize("g_local", [1, 2, 4, 8, 24, 32])
def test_streaming_leaves_the_last_wave_to_the_streamed_expert(g_local, waves):
    width = sparse_experts_per_wave(g_local, waves, stream=True)
    # Every visit position but the last lands before the last wave, and the
    # waves before it are as even as the count allows.
    assert max(g_local - 2, 0) // width <= waves - 2
    assert width == 1 or (width - 1) * (waves - 1) < g_local - 1
    if g_local % waves == 0:
        assert sparse_experts_per_wave(g_local, waves) == g_local // waves


def test_owner_first_run_starts_lay_runs_out_from_the_owner():
    ep, g_local = 4, 2
    rng = np.random.default_rng(7)
    runs = rng.integers(0, 5, (ep * g_local, ep)) * ROWBLK
    starts = np.asarray(owner_first_run_starts(jnp.asarray(runs, jnp.int32), g_local))
    for e in range(ep * g_local):
        at = 0
        for r in range(ep):  # the owner's own run first, then mesh order
            s = (e // g_local + r) % ep
            assert starts[e, s] == at
            at += runs[e, s]


@pytest.mark.parametrize("ep,waves", [(8, 4), (16, 4), (4, 2)])
def test_stream_tiles_wait_for_every_wave_their_rows_are_in(ep, waves):
    """kernel.stream_need, replayed: the waves a slab prefix needs."""
    groups = sparse_dispatch.stream_groups(ep, waves)
    g_local, me = 2, ep - 1  # the last shard: its sources wrap around
    rng = np.random.default_rng(ep)
    runs = rng.integers(0, 4, (ep * g_local, ep)) * ROWBLK  # some runs empty
    starts = owner_first_run_starts(jnp.asarray(runs, jnp.int32), g_local)
    e = me * g_local
    run_start = np.asarray(starts)[e] // ROWBLK  # expert-local blocks
    run_len = runs[e] // ROWBLK
    group_of_block = np.full(run_len.sum(), -1)
    for s in range(ep):
        group = np.searchsorted(np.cumsum(groups), (s - me) % ep, side="right")
        group_of_block[run_start[s] : run_start[s] + run_len[s]] = group
    assert (group_of_block >= 0).all()
    for end in range(1, run_len.sum() + 1):
        need = sum(int(run_start[(me + b) % ep] < end) for b in np.cumsum(groups)[:-1])
        assert need == group_of_block[:end].max()


# -- the layer, on the eight-way slice --------------------------------------


@pytest.fixture(scope="module")
def mesh():
    """The eight-way mesh, or a skip.

    A fixture rather than a `skipif`: see `_require_eight_devices` in
    test_fused_ep_moe_v2.py. The plan tests above must not touch it.
    """
    if jax.device_count() < EP:
        pytest.skip(
            "sparse dispatch is an eight-way expert-parallel exchange and this "
            "host reports fewer than 8 JAX devices"
        )
    return Mesh(np.asarray(jax.devices()[:EP]), axis_names=(AXIS,))


@pytest.fixture(scope="module")
def weights(mesh):
    """fp8 (per output channel) and bf16 expert weights, sharded on experts."""
    rng = np.random.default_rng(0)
    w1 = rng.normal(0, HIDDEN**-0.5, (E_TOTAL, HIDDEN, 2 * INTER)).astype(np.float32)
    w2 = rng.normal(0, INTER**-0.5, (E_TOTAL, INTER, HIDDEN)).astype(np.float32)
    shard = NamedSharding(mesh, P(AXIS))

    def fp8(w):
        scale = np.maximum(np.abs(w).max(axis=1), 1e-12) / float(jnp.finfo(FP8).max)
        q = jnp.asarray(w / scale[:, None, :]).astype(FP8)
        return jax.device_put(q, shard), jax.device_put(scale, shard)

    (q1, s1), (q2, s2) = fp8(w1), fp8(w2)
    b1, b2 = (jax.device_put(jnp.asarray(w, jnp.bfloat16), shard) for w in (w1, w2))
    return {WeightFormat.FP8: (q1, q2, s1, s2), WeightFormat.BF16: (b1, b2, None, None)}


def _inputs(mesh, x, gating):
    shard = NamedSharding(mesh, P(AXIS))
    return (
        jax.device_put(jnp.asarray(x, jnp.bfloat16), shard),
        jax.device_put(np.asarray(gating, np.float32), shard),
    )


def _layer(mesh, x, weights, gating, *, weight_format, sparse_waves, **kwargs):
    out = fused_ep_moe_v2(
        x,
        *weights,
        gating,
        topk=TOPK,
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=weight_format,
        sharded_plan=True,
        sparse_waves=sparse_waves,
        **kwargs,
    )
    assert out.shape == x.shape
    return np.asarray(out, np.float32)


def _assert_bit_exact(
    mesh, x, weights, gating, *, weight_format, waves, stream=False, **kwargs
):
    want = _layer(
        mesh, x, weights, gating, weight_format=weight_format, sparse_waves=0, **kwargs
    )
    got = _layer(
        mesh,
        x,
        weights,
        gating,
        weight_format=weight_format,
        sparse_waves=waves,
        **({"sparse_stream": True} if stream else {}),
        **kwargs,
    )
    assert np.isfinite(want).all()
    np.testing.assert_array_equal(got, want)


@pytest.mark.multichip
@pytest.mark.parametrize(
    "waves,stream",
    [(1, False), (EXPERTS_PER_SHARD, False), (2, True), (EXPERTS_PER_SHARD, True)],
    ids=["w1", "w4", "w2_stream", "w4_stream"],
)
@pytest.mark.parametrize("routing", ["uniform", "skewed", "worst", "dominant"])
@pytest.mark.parametrize("rows_per_shard", [64, 256])
def test_bit_exact_with_the_all_gather(
    mesh, weights, rows_per_shard, routing, waves, stream
):
    tokens = rows_per_shard * EP
    rng = np.random.default_rng(rows_per_shard)
    x, gating = _inputs(
        mesh, rng.normal(size=(tokens, HIDDEN)), rng.normal(size=(tokens, E_TOTAL))
    )
    ids = _expert_ids(routing, tokens, E_TOTAL, TOPK, EXPERTS_PER_SHARD, seed=1)
    _assert_bit_exact(
        mesh,
        x,
        weights[WeightFormat.FP8],
        gating,
        weight_format=WeightFormat.FP8,
        waves=waves,
        stream=stream,
        hash_topk_ids=jax.device_put(ids, NamedSharding(mesh, P(AXIS))),
    )


@pytest.mark.multichip
def test_unquantized_rows_are_bit_exact(mesh, weights):
    """bf16 weights put bf16 rows, with no row scale, on the wire."""
    tokens = 64 * EP
    rng = np.random.default_rng(2)
    x, gating = _inputs(
        mesh, rng.normal(size=(tokens, HIDDEN)), rng.normal(size=(tokens, E_TOTAL))
    )
    _assert_bit_exact(
        mesh,
        x,
        weights[WeightFormat.BF16],
        gating,
        weight_format=WeightFormat.BF16,
        waves=EXPERTS_PER_SHARD,
    )


@pytest.mark.multichip
def test_token_replica_groups_are_bit_exact(mesh, weights):
    """Replicated rows: each group member routes and sends a disjoint half."""
    groups = ((0, 1), (2, 3), (4, 5), (6, 7))
    rows = 128
    rng = np.random.default_rng(3)
    x_g = rng.normal(size=(len(groups), rows, HIDDEN))
    gating_g = rng.normal(size=(len(groups), rows, E_TOTAL))
    owner = {d: i for i, group in enumerate(groups) for d in group}
    x, gating = _inputs(
        mesh,
        np.concatenate([x_g[owner[d]] for d in range(EP)]),
        np.concatenate([gating_g[owner[d]] for d in range(EP)]),
    )
    _assert_bit_exact(
        mesh,
        x,
        weights[WeightFormat.FP8],
        gating,
        weight_format=WeightFormat.FP8,
        waves=EXPERTS_PER_SHARD,
        token_replica_groups=groups,
    )


@pytest.mark.multichip
def test_rejects_configurations_it_does_not_serve(mesh, weights):
    tokens = 64 * EP
    rng = np.random.default_rng(4)
    x, gating = _inputs(
        mesh, rng.normal(size=(tokens, HIDDEN)), rng.normal(size=(tokens, E_TOTAL))
    )
    common = dict(
        topk=TOPK,
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP8,
    )
    fp8 = weights[WeightFormat.FP8]
    with pytest.raises(ValueError, match="sharded_plan"):
        fused_ep_moe_v2(
            x,
            *fp8,
            gating,
            sharded_plan=False,
            sparse_waves=EXPERTS_PER_SHARD,
            **common,
        )
    with pytest.raises(ValueError, match="divide"):
        fused_ep_moe_v2(x, *fp8, gating, sharded_plan=True, sparse_waves=3, **common)
    with pytest.raises(ValueError, match="sparse_stream"):
        fused_ep_moe_v2(
            x,
            *fp8,
            gating,
            sharded_plan=True,
            sparse_waves=1,
            sparse_stream=True,
            **common,
        )
