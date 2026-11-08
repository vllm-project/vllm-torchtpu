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
"""Ragged all-to-all EP MoE with the token rows dispatched by the expert kernel
(``expert_arrival``): every output bit of the same exchange without it."""

import functools

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.megablox.row_copy import _canonical_bits
from vllm_torchtpu.layers.core import expert_placement
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    quantize_to_native_fp4_kmajor,
    resolve_onehot_permute_threshold,
)
from vllm_torchtpu.layers.core.ragged_a2a_moe import (
    ARRIVAL_PREFETCH,
    ARRIVAL_WINDOW,
    DP_AXIS,
    TP_AXIS,
    _arrival_sends,
    _member_index,
    _permuted_values,
    _routes_fit,
    exchange_capacity,
    owner_dests,
    ragged_a2a_moe,
    ragged_a2a_moe_local,
    route_plan,
    streamed_dests,
)

HIDDEN = 2048
INTER = 512
BF16 = ml_dtypes.bfloat16


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
    "dp_size,tp_size,num_tokens,local,topk,routes,skew,padded",
    [
        (8, 4, 104, 16, 10, 640, False, ""),  # 4-host decode step
        (8, 4, 104, 16, 10, 640, False, "unrouted"),  # padded tokens: id -1
        (8, 4, 104, 17, 10, 1536, True, "skip"),  # 17 experts per rank
        (8, 1, 24, 16, 10, 128, True, ""),  # overflow: nothing is sent
        (2, 4, 22, 9, 10, 256, False, "skip"),
        (2, 4, 22, 9, 10, 256, False, "unrouted"),
        (8, 1, 24, 16, 10, 1 << 14, True, ""),  # wide rows: two-key sort
    ],
)
def test_sends_fill_the_route_plan_rows(
    combine, dp_size, tp_size, num_tokens, local, topk, routes, skew, padded
):
    """Every member's sends land exactly on the lhs rows `route_plan` reserves
    for them on their member, once each, with that row's local expert; sends
    go in (local expert, member) order with the right per-expert offsets.
    Padded tokens (weight 0 with ``skip``, id -1 when ``unrouted``) send
    nothing."""
    rng = np.random.default_rng(dp_size * 100 + num_tokens + local)
    num_experts = dp_size * tp_size * local
    ids, weights = _routing(rng, dp_size, num_tokens, num_experts, topk, skew)
    ids[0, 1, topk - 1] = -1
    if padded:
        weights[:, -3:] = 0
    if padded == "skip":
        weights[1, 0, topk - 1] = np.nan
    if padded == "unrouted":
        ids[:, -3:] = -1
    gathered = jnp.asarray(ids)
    gathered_weights = jnp.asarray(weights, jnp.bfloat16) if padded == "skip" else None
    fits = _routes_fit(
        gathered,
        gathered_weights,
        tp_size=tp_size,
        local_num_experts=local,
        routes=routes,
    )
    num_groups = dp_size * local
    for col in range(tp_size):
        landed = {}
        for dp_pos in range(dp_size):
            member_index = functools.partial(
                _member_index,
                ep_rank=jnp.int32(dp_pos * tp_size + col),
                dp_size=dp_size,
                tp_size=tp_size,
                local_num_experts=local,
            )
            member = member_index(gathered)
            valid = (member < dp_size) & fits
            if gathered_weights is not None:
                w = jnp.where(jnp.isfinite(gathered_weights), gathered_weights, 0)
                valid = valid & (w != 0)
            group = jnp.where(valid, member * local + gathered % local, num_groups)
            token, dest, row, expert, num_sends, offsets = (
                np.asarray(a)
                for a in _arrival_sends(
                    group,
                    jnp.int32(dp_pos),
                    num_groups=num_groups,
                    local=local,
                    routes=routes,
                )
            )
            n = int(num_sends[0])
            order = list(zip(expert[:n], dest[:n]))
            assert order == sorted(order)
            assert offsets[-1] == n
            np.testing.assert_array_equal(
                np.diff(offsets), np.bincount(expert[:n], minlength=local)
            )
            if padded:
                assert (token[:n] < num_tokens - 3).all()
            for i in range(n):
                key = (int(dest[i]), int(row[i]))
                assert key not in landed
                landed[key] = (dp_pos * num_tokens + int(token[i]), int(expert[i]))
        for d in range(dp_size):
            ep_rank = d * tp_size + col
            member_index = functools.partial(
                _member_index,
                ep_rank=jnp.int32(ep_rank),
                dp_size=dp_size,
                tp_size=tp_size,
                local_num_experts=local,
            )
            if combine == "owner":
                dests = functools.partial(
                    owner_dests,
                    dp_size=dp_size,
                    tp_size=tp_size,
                    local_num_experts=local,
                )
            else:
                dests = functools.partial(
                    streamed_dests, member_index=member_index, dp_size=dp_size
                )
            token_sorted, group_sizes, _, _ = route_plan(
                gathered,
                jnp.arange(dp_size * num_tokens, dtype=jnp.int32).reshape(
                    dp_size, num_tokens
                ),
                fits,
                jnp.asarray([ep_rank * local], jnp.int32),
                dests,
                local_num_experts=local,
                routes=routes,
                rows=dp_size * num_tokens,
                gathered_weights=gathered_weights,
            )
            group_sizes = np.asarray(group_sizes)
            token_sorted = np.asarray(token_sorted)
            rows = int(group_sizes.sum())
            mine = {r: v for (m, r), v in landed.items() if m == d}
            assert len(mine) == rows
            expert = np.repeat(np.arange(local), group_sizes)
            for r in range(rows):
                assert mine[r] == (int(token_sorted[r]), int(expert[r]))


@pytest.mark.cpu_test
def test_dispatched_rows_take_the_permuted_values():
    """Every bf16 bit pattern: kept when its exponent field is 1..254, else +0
    (what the one-hot permute and the row gather of the expert side give)."""
    bits = np.arange(1 << 16, dtype=np.uint16)
    got = np.asarray(_permuted_values(jnp.asarray(bits.view(BF16)))).view(np.uint16)
    field = (bits >> 7) & 0xFF
    want = np.where((field >= 1) & (field <= 254), bits, 0).astype(np.uint16)
    np.testing.assert_array_equal(got, want)
    pairs = bits.astype(np.uint32) * np.uint32(0x10001)
    gathered = np.asarray(_canonical_bits(jnp.asarray(pairs)))
    np.testing.assert_array_equal(gathered & 0xFFFF, want)
    np.testing.assert_array_equal(gathered >> 16, want)


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


def _fits_on_host(ids, weights, *, dp_size, tp_size, local, routes, padded):
    """Whether no rank sees more than `routes` routes (`_routes_fit` in numpy);
    ``ids`` / ``weights`` [dp * tp * S, K] repeat each engine's routing on its
    TP ranks."""
    num_ranks = dp_size * tp_size
    ids = np.asarray(ids).reshape(dp_size, tp_size, -1)[:, 0].reshape(-1)
    w = np.asarray(jnp.asarray(weights).astype(jnp.float32))
    w = w.reshape(dp_size, tp_size, -1)[:, 0].reshape(-1)
    keep = (ids >= 0) & (ids // local < num_ranks)
    if padded == "skip":
        keep &= np.isfinite(w) & (w != 0)
    return np.bincount(ids[keep] // local, minlength=num_ranks).max() <= routes


# (dp, tp, tokens, topk, rows x, routes x, skew, padded, experts, fits): padded
# tokens have weight 0 and are skipped by the exchange ("skip"), or reach it
# unrouted with id -1 ("unrouted"); fits is whether the step stays on the
# in-kernel dispatch (None: the full-size exchange only, where the capacity
# rows cover every token).
LAYOUTS = [
    (8, 1, 24, 10, 1.15, 1.5, False, "", 64, True),
    (8, 1, 24, 10, 1.15, 1.5, False, "skip", 64, True),
    (8, 1, 12, 2, 2.0, 0.0, True, "", 64, False),  # skewed: full-size exchange
    (4, 2, 24, 10, 1.0, 1.5, False, "", 64, True),
    (4, 2, 24, 2, 0.5, 0.0, True, "", 64, False),
    (2, 4, 92, 10, 1.0, 2.0, False, "", 64, True),  # proxy K=3 step
    (2, 4, 22, 10, 0.5, 2.0, False, "skip", 64, True),  # padded owner blocks
    (4, 2, 24, 10, 1.0, 2.0, False, "", 72, True),  # 9 experts per rank
    (2, 4, 92, 10, 2.5, 3.0, False, "", 64, None),  # proxy default sizing
    (8, 1, 24, 10, 1.15, 1.5, False, "unrouted", 64, True),
    (2, 4, 22, 10, 0.5, 2.0, False, "unrouted", 64, True),
]
# (window, prefetch, sorted plan, row tile, row gather and scatter) beside the
# default schedule.
SCHEDULES = {
    "default": (ARRIVAL_WINDOW, ARRIVAL_PREFETCH, True, 16, True),
    "no_row_copies": (ARRIVAL_WINDOW, ARRIVAL_PREFETCH, True, 16, False),
    "all_at_entry": (0, 0, True, 16, True),
    "w1_after": (1, 1, True, 16, True),
    "w8": (8, 2, True, 32, True),
    "unsorted_base": (ARRIVAL_WINDOW, ARRIVAL_PREFETCH, False, 16, False),
}
CASES = (
    [
        (layout, combine, "default")
        for layout in range(len(LAYOUTS))
        for combine in ("owner", "streamed")
    ]
    + [(layout, "owner", "no_row_copies") for layout in (0, 2, 5, 6, 9)]
    + [
        (0, "streamed", "no_row_copies"),
        (0, "owner", "all_at_entry"),
        (5, "owner", "all_at_entry"),
        (0, "owner", "w1_after"),
        (5, "owner", "w1_after"),
        (5, "owner", "w8"),
        (6, "owner", "w8"),
        (3, "owner", "unsorted_base"),
    ]
)
_BASE = {}


def _setup(layout, combine, sorted_plan, row_tile, row_copies):
    dp_size, tp_size, num_tokens, topk, factor, route_factor, skew, padded, ne, _ = (
        LAYOUTS[layout]
    )
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    x, w1, w2, s1, s2, weights, ids, es = _inputs(
        8, tp_size, num_tokens, ne, topk=topk, skew=skew
    )
    if padded:
        weights = weights.reshape(8, num_tokens, topk).at[:, -5:].set(0)
        weights = weights.reshape(-1, topk)
    if padded == "unrouted":
        ids = ids.reshape(8, num_tokens, topk).at[:, -5:].set(-1).reshape(-1, topk)
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
        skip_padded_tokens=padded == "skip",
        capacity_factor=factor,
        route_factor=route_factor,
        capacity_row_tile=row_tile,
        row_gather=row_copies,
        row_scatter=row_copies,
        streamed_combine=combine == "streamed",
        owner_combine=combine == "owner",
        sorted_plan=sorted_plan,
    )
    kwargs = {}
    if combine == "owner":
        rng = np.random.default_rng(3)
        kwargs["shared"] = jnp.asarray(
            rng.standard_normal((dp_size, tp_size, num_tokens, HIDDEN)) / 8,
            jnp.bfloat16,
        )
    return common, stacked, kwargs, weights, ids


@pytest.mark.multichip
@pytest.mark.parametrize("layout,combine,schedule", CASES)
def test_expert_arrival_keeps_every_output_bit(layout, combine, schedule):
    """Bitwise the same exchange without the in-kernel dispatch, in every
    layout (TP 1 / 2 / 4, padded tokens skipped or unrouted, padded owner
    blocks, 9 experts per rank, steps that fall back to the full-size
    exchange), with the shared expert's partial in owner mode, for every
    schedule, with and without the row copies."""
    window, prefetch, sorted_plan, row_tile, row_copies = SCHEDULES[schedule]
    dp_size, tp_size, num_tokens, topk, factor, route_factor, _, padded, ne, fit = (
        LAYOUTS[layout]
    )
    common, stacked, kwargs, weights, ids = _setup(
        layout, combine, sorted_plan, row_tile, row_copies
    )
    _, routes = exchange_capacity(
        dp_size=dp_size,
        tp_size=tp_size,
        num_tokens=num_tokens,
        topk=topk,
        local_num_experts=ne // 8,
        factor=factor,
        onehot_moe_permute_threshold=common["onehot_moe_permute_threshold"],
        use_sparse_core=True,
        route_factor=route_factor or None,
        row_tile=row_tile,
    )
    run = jax.jit(
        functools.partial(
            ragged_a2a_moe,
            **common,
            expert_arrival=True,
            arrival_window=window,
            arrival_prefetch=prefetch,
        )
    )
    text = run.lower(*stacked, **kwargs).as_text()
    assert ("gmm_v2_mlp_exchange" in text) == (fit is not None)
    if fit is not None:
        fits = _fits_on_host(
            ids,
            weights,
            dp_size=dp_size,
            tp_size=tp_size,
            local=ne // 8,
            routes=routes,
            padded=padded,
        )
        assert fits == fit
    key = (layout, combine, sorted_plan, row_tile, row_copies)
    if key not in _BASE:
        base_run = jax.jit(functools.partial(ragged_a2a_moe, **common))
        _BASE[key] = np.asarray(base_run(*stacked, **kwargs))
    base = _BASE[key]
    got = np.asarray(run(*stacked, **kwargs))
    assert np.abs(base.astype(np.float32)).max() > 0
    np.testing.assert_array_equal(got.view(np.uint16), base.view(np.uint16))


@pytest.mark.multichip
@pytest.mark.parametrize("combine", ["owner", "streamed"])
def test_expert_arrival_runs_both_branches(combine):
    """Steps that fit and steps that overflow the routes through one jitted
    program (an overflowing step's kernel sends and expects nothing)."""
    common, stacked, kwargs, _, _ = _setup(3, combine, True, 16, True)
    base_run = jax.jit(functools.partial(ragged_a2a_moe, **common))
    run = jax.jit(
        functools.partial(
            ragged_a2a_moe,
            **common,
            expert_arrival=True,
            arrival_window=ARRIVAL_WINDOW,
            arrival_prefetch=ARRIVAL_PREFETCH,
        )
    )
    assert "gmm_v2_mlp_exchange" in run.lower(*stacked, **kwargs).as_text()
    for forced in (False, True, False):
        args = stacked
        if forced:
            # Every route on one rank: far past the route capacity.
            hot = jnp.broadcast_to(jnp.arange(10, dtype=jnp.int32), stacked[6].shape)
            args = stacked[:6] + (hot, stacked[7])
        base = np.asarray(base_run(*args, **kwargs)).view(np.uint16)
        got = np.asarray(run(*args, **kwargs)).view(np.uint16)
        np.testing.assert_array_equal(got, base)


def _in_slots(args):
    """The inputs with the experts in the slots of a count-balanced placement.
    Returns (args, slots per rank)."""
    x, w1, w2, s1, s2, weights, ids, es = args
    num_experts, ranks = w1.shape[0], es.shape[0]
    counts = np.bincount(np.asarray(ids).ravel(), minlength=num_experts)
    slots = expert_placement.balance(counts, ranks)
    local = num_experts // ranks
    expert = np.argsort(slots)
    placed = slots[np.asarray(ids)]
    w1, w2, s1, s2 = (a[jnp.asarray(expert)] for a in (w1, w2, s1, s2))
    es = jnp.arange(ranks, dtype=jnp.int32) * local
    return (x, w1, w2, s1, s2, weights, jnp.asarray(placed, jnp.int32), es), local


@pytest.mark.multichip
@pytest.mark.parametrize(
    "dp_size,tp_size,factor", [(8, 1, 1.0), (4, 2, 0.5), (2, 4, 0.5)]
)
def test_expert_arrival_in_slot_space(dp_size, tp_size, factor):
    """Skewed top-10 routing placed by a count-balanced table: bitwise the
    owner combine (sorted plan, 32-row tiles, row copies) on the same slots."""
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    num_tokens, topk = 24, 10
    args, local = _in_slots(_inputs(8, tp_size, num_tokens, 64, topk=topk, skew=True))
    stacked = tuple(a.reshape(dp_size, tp_size, -1, *a.shape[1:]) for a in args)
    threshold = resolve_onehot_permute_threshold(dp_size * num_tokens * topk)
    common = dict(
        mesh=mesh,
        topk=topk,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=threshold,
        capacity_factor=factor,
        route_factor=2.0,
        capacity_row_tile=32,
        row_gather=True,
        row_scatter=True,
        owner_combine=True,
        sorted_plan=True,
    )
    rng = np.random.default_rng(6)
    shared = jnp.asarray(
        rng.standard_normal((dp_size, tp_size, num_tokens, HIDDEN)) / 8,
        jnp.bfloat16,
    )
    run = jax.jit(
        functools.partial(
            ragged_a2a_moe,
            **common,
            expert_arrival=True,
            arrival_window=ARRIVAL_WINDOW,
            arrival_prefetch=ARRIVAL_PREFETCH,
        )
    )
    assert "gmm_v2_mlp_exchange" in run.lower(*stacked, shared=shared).as_text()
    _, routes = exchange_capacity(
        dp_size=dp_size,
        tp_size=tp_size,
        num_tokens=num_tokens,
        topk=topk,
        local_num_experts=local,
        factor=factor,
        onehot_moe_permute_threshold=threshold,
        use_sparse_core=True,
        route_factor=2.0,
        row_tile=32,
    )
    assert _fits_on_host(
        args[6],
        args[5],
        dp_size=dp_size,
        tp_size=tp_size,
        local=local,
        routes=routes,
        padded=False,
    )
    base = np.asarray(
        jax.jit(functools.partial(ragged_a2a_moe, **common))(*stacked, shared=shared)
    )
    got = np.asarray(run(*stacked, shared=shared))
    assert np.abs(base.astype(np.float32)).max() > 0
    np.testing.assert_array_equal(got.view(np.uint16), base.view(np.uint16))


@pytest.mark.multichip
@pytest.mark.parametrize("owner", [False, True])
def test_expert_arrival_leaves_unscaled_experts_alone(owner):
    """Unscaled (bf16) experts, as an MTP draft layer's exchange runs them: the
    fused expert kernel does not take them, so the flag changes nothing."""
    devices = _require_eight_tpu_devices()
    dp_size, tp_size, num_tokens, topk = 2, 4, 96, 2
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    x, _, _, _, _, weights, ids, es = _inputs(
        8, tp_size, num_tokens, 64, topk=topk, skew=False
    )
    k1, k2 = jax.random.split(jax.random.key(17))
    w1 = (jax.random.normal(k1, (64, HIDDEN, 2 * INTER)) / 32).astype(jnp.bfloat16)
    w2 = (jax.random.normal(k2, (64, INTER, HIDDEN)) / 16).astype(jnp.bfloat16)
    stacked = tuple(
        a.reshape(dp_size, tp_size, -1, *a.shape[1:])
        for a in (x, w1, w2, weights, ids, es)
    )
    common = dict(
        mesh=mesh,
        topk=topk,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(
            dp_size * num_tokens * topk
        ),
        capacity_factor=2.0,
        route_factor=2.0,
        capacity_row_tile=32,
        row_gather=True,
        row_scatter=True,
        streamed_combine=not owner,
        owner_combine=owner,
        sorted_plan=True,
    )
    kwargs = {}
    if owner:
        rng = np.random.default_rng(8)
        kwargs["shared"] = jnp.asarray(
            rng.standard_normal((dp_size, tp_size, num_tokens, HIDDEN)) / 8,
            jnp.bfloat16,
        )

    def build(**arrival):
        call = functools.partial(ragged_a2a_moe, **common, **arrival)
        return jax.jit(
            lambda x, w1, w2, w, i, e, **kw: call(x, w1, w2, None, None, w, i, e, **kw)
        )

    run = build(
        expert_arrival=True,
        arrival_window=ARRIVAL_WINDOW,
        arrival_prefetch=ARRIVAL_PREFETCH,
    )
    assert "gmm_v2_mlp_exchange" not in run.lower(*stacked, **kwargs).as_text()
    base = np.asarray(build()(*stacked, **kwargs))
    got = np.asarray(run(*stacked, **kwargs))
    assert np.abs(base.astype(np.float32)).max() > 0
    np.testing.assert_array_equal(got.view(np.uint16), base.view(np.uint16))


def _late(es, nanos, after):
    """`es`, returned after this core sleeps `nanos[0]` ns, once `after` exists:
    the MoE layer that takes it starts that much later on this core."""

    def kernel(es_ref, nanos_ref, after_ref, out_ref):
        del after_ref
        pl.delay(nanos_ref[0])
        out_ref[0] = es_ref[0]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
    )(es, nanos, after)


@pytest.mark.multichip
@pytest.mark.parametrize("dp_size,tp_size", [(8, 1), (2, 4)])
def test_expert_arrival_survives_late_members(dp_size, tp_size):
    """1000 layers back to back, each started 50 us late on one member (a
    different one each layer), over routings that fit (one with unrouted
    padded tokens) and one that overflows; every output row of every layer is
    checked: no hang, no wrong row."""
    calls, late_ns = 1000, 50_000
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    num_tokens, topk, ne = (24, 10, 64) if tp_size == 1 else (22, 10, 64)
    x, w1, w2, s1, s2, weights, ids, es = _inputs(
        8, tp_size, num_tokens, ne, topk=topk, skew=False
    )
    # Routings (weights, ids): three that fit, the first again with its last
    # five tokens unrouted, one that overflows (every route on rank 0's and
    # rank 1's experts).
    routings = [(weights, ids)] + [
        _inputs(8, tp_size, num_tokens, ne, topk=topk, skew=False, seed=s)[5:7]
        for s in (1, 2)
    ]
    pad = (slice(None), slice(num_tokens - 5, None))
    routings.append(
        tuple(
            a.reshape(8, num_tokens, topk).at[pad].set(v).reshape(a.shape)
            for a, v in ((weights, 0), (ids, -1))
        )
    )
    routings.append(
        (weights, jnp.broadcast_to(jnp.arange(10, dtype=jnp.int32), ids.shape))
    )
    common = dict(
        topk=topk,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(
            dp_size * num_tokens * topk
        ),
        capacity_factor=1.15 if tp_size == 1 else 0.5,
        route_factor=1.5 if tp_size == 1 else 2.0,
        row_gather=True,
        row_scatter=True,
        owner_combine=True,
        sorted_plan=True,
    )
    rng = np.random.default_rng(5)
    shared = jnp.asarray(
        rng.standard_normal((dp_size, tp_size, num_tokens, HIDDEN)) / 8,
        jnp.bfloat16,
    )

    def stack(a):
        return a.reshape(dp_size, tp_size, -1, *a.shape[1:])

    base_run = jax.jit(functools.partial(ragged_a2a_moe, mesh=mesh, **common))
    want = np.stack(
        [
            np.asarray(
                base_run(
                    *(stack(a) for a in (x, w1, w2, s1, s2, tw_v, tid_v, es)),
                    shared=shared,
                )
            )
            for tw_v, tid_v in routings
        ],
        axis=2,
    )  # [dp, tp, V, S, H]
    tw = jnp.stack([r[0] for r in routings], axis=1)  # [ranks * S, V, K]
    tid = jnp.stack([r[1] for r in routings], axis=1)
    local = functools.partial(
        ragged_a2a_moe_local,
        tp_size=tp_size,
        dp_size=dp_size,
        expert_arrival=True,
        arrival_window=ARRIVAL_WINDOW,
        arrival_prefetch=ARRIVAL_PREFETCH,
        **common,
    )
    num_variants = len(routings)

    def body(x, w1, w2, s1, s2, tw, tid, es, shared, want):
        x, w1, w2, s1, s2, tw, tid, es, shared, want = (
            a[0, 0] for a in (x, w1, w2, s1, s2, tw, tid, es, shared, want)
        )
        me = lax.axis_index(DP_AXIS) * tp_size + lax.axis_index(TP_AXIS)
        want_bits = lax.bitcast_convert_type(want, jnp.uint16)

        def layer(i, wrong):
            v = i % num_variants
            nanos = jnp.where(i % (dp_size * tp_size) == me, late_ns, 0)
            es_i = _late(es, nanos.astype(jnp.int32)[None], x)
            out = local(
                x,
                w1,
                w2,
                s1,
                s2,
                lax.dynamic_index_in_dim(tw, v, 1, keepdims=False),
                lax.dynamic_index_in_dim(tid, v, 1, keepdims=False),
                es_i,
                shared=shared,
            )
            bits = lax.bitcast_convert_type(out, jnp.uint16)
            ref = lax.dynamic_index_in_dim(want_bits, v, 0, keepdims=False)
            return wrong + (bits != ref).any(axis=-1).sum(dtype=jnp.int32)

        wrong = lax.fori_loop(0, calls, layer, jnp.int32(0))
        return wrong[None, None, None]

    spec = P(DP_AXIS, TP_AXIS)
    prog = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 10, out_specs=spec, check_vma=False
        )
    )
    args = (
        stack(x),
        stack(w1),
        stack(w2),
        stack(s1),
        stack(s2),
        stack(tw),
        stack(tid),
        stack(es),
        shared,
        jnp.asarray(want),
    )
    assert "gmm_v2_mlp_exchange" in prog.lower(*args).as_text()
    wrong = prog(*args)
    assert np.asarray(wrong).sum() == 0, np.asarray(wrong).reshape(-1)
