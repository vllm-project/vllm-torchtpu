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
"""Owner combine whose expert kernel sums the owned tokens in its tail
(``owner_tail``): bitwise the owner combine without it."""

import functools
import re

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

from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2_mlp_scatter
from vllm_torchtpu.layers.core import expert_placement
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    quantize_to_native_fp4_kmajor,
    resolve_onehot_permute_threshold,
)
from vllm_torchtpu.layers.core.ragged_a2a_moe import (
    DP_AXIS,
    TP_AXIS,
    ragged_a2a_moe,
    ragged_a2a_moe_local,
)

HIDDEN = 2048
INTER = 512
BF16 = ml_dtypes.bfloat16
# A fused expert kernel with the owner tail in a lowered program.
TAIL_KERNEL = re.compile(r"gmm_v2_mlp_scatter-[A-Za-z0-9_\-]*-t\b")


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


def _shared(dp_size, tp_size, num_tokens, seed=3):
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        rng.standard_normal((dp_size, tp_size, num_tokens, HIDDEN)) / 8, jnp.bfloat16
    )


# (dp, tp, tokens per engine, topk, rows x, routes x, padded tokens, experts,
# shared expert, sorted plan with the row copies). Factor 0: the full-size
# kernel only; the capacity layouts leave the busiest rank's rows and routes
# well inside the buffers (a few standard deviations) unless marked.
LAYOUTS = [
    (2, 4, 24, 10, 0.0, 0.0, "", 64, True, True),  # TP4: chip-pair forward
    (2, 4, 4, 10, 0.0, 0.0, "", 64, True, True),  # one owned token per core
    (2, 4, 12, 10, 0.0, 0.0, "", 64, True, True),
    (2, 4, 13, 10, 0.0, 0.0, "", 64, True, True),  # padded owner block
    (2, 4, 22, 10, 0.0, 0.0, "unrouted", 64, True, True),
    (2, 4, 92, 10, 1.2, 2.0, "", 64, True, True),  # capacity: 6 chunks per core
    (2, 4, 24, 10, 0.0, 0.0, "", 64, False, True),  # no shared expert
    (4, 2, 48, 10, 1.15, 1.5, "", 64, True, True),  # capacity, TP2
    (4, 2, 24, 2, 2.0, 0.0, "", 64, True, True),  # capacity, top-2
    (4, 2, 24, 10, 0.5, 2.0, "skip", 64, True, True),  # overflows: full size
    (4, 2, 24, 10, 0.0, 0.0, "", 72, True, True),  # 9 experts per rank
    (4, 2, 24, 10, 0.0, 0.0, "", 64, True, False),  # no sorted plan, row copies
    (8, 1, 24, 10, 1.15, 1.5, "", 64, True, True),  # capacity, TP1
    (8, 1, 24, 10, 0.0, 0.0, "", 64, False, True),
]


def _setup(layout):
    dp_size, tp_size, num_tokens, topk, factor, route_factor, padded, ne, shared, sp = (
        LAYOUTS[layout]
    )
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    x, w1, w2, s1, s2, weights, ids, es = _inputs(
        8, tp_size, num_tokens, ne, topk=topk, skew=False
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
        row_gather=sp,
        row_scatter=sp,
        owner_combine=True,
        sorted_plan=sp,
    )
    kwargs = {"shared": _shared(dp_size, tp_size, num_tokens)} if shared else {}
    return common, stacked, kwargs


def _check_tail(common, stacked, kwargs):
    """The tail's result vs the owner combine without it, bitwise."""
    run = jax.jit(functools.partial(ragged_a2a_moe, owner_tail=True, **common))
    lowered = run.lower(*stacked, **kwargs).as_text()
    assert TAIL_KERNEL.search(lowered)
    assert "ragged_a2a_owner_sum" not in lowered
    base = np.asarray(
        jax.jit(functools.partial(ragged_a2a_moe, **common))(*stacked, **kwargs)
    )
    got = np.asarray(run(*stacked, **kwargs))
    assert np.abs(base.astype(np.float32)).max() > 0
    np.testing.assert_array_equal(got.view(np.uint16), base.view(np.uint16))


@pytest.mark.multichip
@pytest.mark.parametrize("layout", range(len(LAYOUTS)))
def test_owner_tail_keeps_every_output_bit(layout):
    """Every TP rank's result bitwise the owner combine without the tail: TP 1
    / 2 / 4, 1 to 23 owned tokens per core (padded owner blocks), with and
    without the shared expert, padded tokens skipped or unrouted, capacity
    and full-size kernels, a step that overflows, 9 experts per rank, with
    and without the sorted plan and the row copies."""
    _check_tail(*_setup(layout))


@pytest.mark.multichip
@pytest.mark.parametrize("dp_size,tp_size", [(2, 4), (4, 2)])
def test_owner_tail_with_all_routes_on_one_core(dp_size, tp_size):
    """Every route on rank 0's experts: the other cores run no expert tile and
    every owned token's routes come from one core."""
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    num_tokens, topk = 24, 10
    x, w1, w2, s1, s2, weights, ids, es = _inputs(
        8, tp_size, num_tokens, 128, topk=topk, skew=False
    )
    ids = jnp.broadcast_to(jnp.arange(topk, dtype=jnp.int32), ids.shape)
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
        row_gather=True,
        row_scatter=True,
        owner_combine=True,
        sorted_plan=True,
    )
    _check_tail(common, stacked, {"shared": _shared(dp_size, tp_size, num_tokens)})


@pytest.mark.multichip
def test_owner_tail_runs_both_branches():
    """Steps that fit and one that overflows, through one jitted program: the
    overflowing step's capacity kernel sums and sends nothing in its tail, and
    the full-size kernel's tail runs on its own collective id."""
    common, stacked, kwargs = _setup(8)
    base_run = jax.jit(functools.partial(ragged_a2a_moe, **common))
    run = jax.jit(functools.partial(ragged_a2a_moe, owner_tail=True, **common))
    assert len(set(TAIL_KERNEL.findall(run.lower(*stacked, **kwargs).as_text()))) == 2
    for forced in (False, True, False):
        args = stacked
        if forced:
            # Every route on rank 0's and rank 1's experts: past the capacity.
            hot = jnp.broadcast_to(jnp.asarray([0, 1], jnp.int32), stacked[6].shape)
            args = stacked[:6] + (hot, stacked[7])
        base = np.asarray(base_run(*args, **kwargs)).view(np.uint16)
        got = np.asarray(run(*args, **kwargs)).view(np.uint16)
        np.testing.assert_array_equal(got, base)


def _in_slots(args):
    """The inputs with the experts in the slots of a count-balanced placement."""
    x, w1, w2, s1, s2, weights, ids, es = args
    num_experts, ranks = w1.shape[0], es.shape[0]
    counts = np.bincount(np.asarray(ids).ravel(), minlength=num_experts)
    slots = expert_placement.balance(counts, ranks)
    expert = np.argsort(slots)
    placed = slots[np.asarray(ids)]
    w1, w2, s1, s2 = (a[jnp.asarray(expert)] for a in (w1, w2, s1, s2))
    return x, w1, w2, s1, s2, weights, jnp.asarray(placed, jnp.int32), es


@pytest.mark.multichip
@pytest.mark.parametrize("dp_size,tp_size,factor", [(4, 2, 1.15), (2, 4, 0.0)])
def test_owner_tail_in_slot_space(dp_size, tp_size, factor):
    """Skewed top-10 routing placed by a count-balanced table."""
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    num_tokens, topk = 48, 10
    args = _in_slots(_inputs(8, tp_size, num_tokens, 64, topk=topk, skew=True))
    stacked = tuple(a.reshape(dp_size, tp_size, -1, *a.shape[1:]) for a in args)
    common = dict(
        mesh=mesh,
        topk=topk,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(
            dp_size * num_tokens * topk
        ),
        capacity_factor=factor,
        route_factor=2.0,
        row_gather=True,
        row_scatter=True,
        owner_combine=True,
        sorted_plan=True,
    )
    _check_tail(common, stacked, {"shared": _shared(dp_size, tp_size, num_tokens)})


@pytest.mark.multichip
def test_owner_tail_leaves_unscaled_experts_alone():
    """Unscaled (bf16) experts, as an MTP draft layer's exchange runs them, do
    not take the fused kernel, so the flag changes nothing."""
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
        row_gather=True,
        row_scatter=True,
        owner_combine=True,
        sorted_plan=True,
    )
    shared = _shared(dp_size, tp_size, num_tokens, seed=8)

    def build(**tail):
        call = functools.partial(ragged_a2a_moe, **common, **tail)
        return jax.jit(
            lambda x, w1, w2, w, i, e, **kw: call(x, w1, w2, None, None, w, i, e, **kw)
        )

    run = build(owner_tail=True)
    assert "gmm_v2_mlp_scatter" not in run.lower(*stacked, shared=shared).as_text()
    base = np.asarray(build()(*stacked, shared=shared))
    got = np.asarray(run(*stacked, shared=shared))
    assert np.abs(base.astype(np.float32)).max() > 0
    np.testing.assert_array_equal(got.view(np.uint16), base.view(np.uint16))


def _late(value, nanos, after):
    """`value` [1] int32, returned after this core sleeps `nanos[0]` ns, once
    `after` exists: a kernel that takes it starts that much later here."""

    def kernel(value_ref, nanos_ref, after_ref, out_ref):
        del after_ref
        pl.delay(nanos_ref[0])
        out_ref[0] = value_ref[0]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
    )(value, nanos, after)


@pytest.mark.multichip
@pytest.mark.parametrize(
    "dp_size,tp_size,num_tokens,factor,route_factor",
    [(2, 4, 92, 1.2, 2.0), (4, 2, 48, 1.15, 1.5)],
)
def test_owner_tail_survives_late_members(
    dp_size, tp_size, num_tokens, factor, route_factor
):
    """1000 MoE layers back to back, each started 50 us late on one core (a
    different one each layer), over routings that fit (one with unrouted
    padded tokens) and one that overflows (whose capacity kernel sends
    nothing, then the full-size kernel runs on the other collective id);
    every output row of every layer is checked."""
    calls, late_ns, topk = 1000, 50_000, 10
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    x, w1, w2, s1, s2, weights, ids, es = _inputs(
        8, tp_size, num_tokens, 64, topk=topk, skew=False
    )
    routings = [(weights, ids)] + [
        _inputs(8, tp_size, num_tokens, 64, topk=topk, skew=False, seed=s)[5:7]
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
        (weights, jnp.broadcast_to(jnp.arange(topk, dtype=jnp.int32), ids.shape))
    )
    common = dict(
        topk=topk,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(
            dp_size * num_tokens * topk
        ),
        capacity_factor=factor,
        route_factor=route_factor,
        row_gather=True,
        row_scatter=True,
        owner_combine=True,
        sorted_plan=True,
    )
    shared = _shared(dp_size, tp_size, num_tokens, seed=5)

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
        owner_tail=True,
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
    assert len(set(TAIL_KERNEL.findall(prog.lower(*args).as_text()))) == 2
    wrong = prog(*args)
    assert np.asarray(wrong).sum() == 0, np.asarray(wrong).reshape(-1)


@pytest.mark.multichip
def test_owner_tail_survives_a_late_tp_peer():
    """1000 expert kernels with the owner tail back to back, collective ids
    alternating. In each call one TP rank of every engine runs 16 expert
    tiles to the others' one, all kept on that core, so it reaches its tail
    last: its peers' sums land on it while it computes and they wait for its
    sums (a different rank each call). Every fifth call sends nothing, and
    one core enters each call 50 us late. Every row of every other call is
    checked (shared-expert partials only, a different set each call, so a
    row left from an earlier call shows)."""
    devices = _require_eight_tpu_devices()
    dp_size, tp_size, topk, groups, blk = 2, 4, 10, 16, 26
    members, calls, late_ns = dp_size * tp_size, 1000, 50_000
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    k1, k2, kx = jax.random.split(jax.random.key(23), 3)
    w1 = jax.random.normal(k1, (groups, HIDDEN, 2 * INTER)) / 8
    w2 = jax.random.normal(k2, (groups, INTER, HIDDEN)) / 8
    w1, s1 = quantize_to_native_fp4_kmajor(w1, 256, pack=False)
    w2, s2 = quantize_to_native_fp4_kmajor(w2, 256, pack=False)
    size_m = groups * 64
    lhs = (jax.random.normal(kx, (members, size_m, HIDDEN)) / 4).astype(jnp.bfloat16)
    # Per call variant, the late TP rank (-1: the call sends nothing).
    late_rank = [0, 1, -1, 2, 3]
    num_variants = len(late_rank)
    sizes = np.zeros((num_variants, members, groups), np.int32)
    for v, r in enumerate(late_rank):
        if r >= 0:
            sizes[v, :, 0] = 16
            sizes[v, r::tp_size] = 64
    fits = np.asarray([[int(r >= 0)] for r in late_rank], np.int32)
    # Every expert row stays on its core (rows past the receive buffer wrap:
    # the tail reads no route row here, every route is marked unsent).
    dest = np.repeat(np.arange(members, dtype=np.int32)[:, None], size_m, axis=1)
    dest_row = np.broadcast_to(
        np.arange(size_m, dtype=np.int32) % (topk * blk), (members, size_m)
    )
    rng = np.random.default_rng(29)
    shared = (
        rng.standard_normal((num_variants, members, tp_size * blk, HIDDEN)) / 4
    ).astype(BF16)
    engine = np.arange(members) // tp_size
    extra_member = (engine[:, None] * tp_size + np.arange(tp_size)).astype(np.int32)
    extra_row = np.repeat(
        (np.arange(members) % tp_size * blk).astype(np.int32)[:, None], tp_size, 1
    )
    counts = fits * np.asarray([(tp_size - 1) * blk, tp_size * blk], np.int32)
    # Each core's result: per token, the TP ranks' shared partials (plus a
    # zero routed partial, rounded) summed in f32 in rank order and rounded
    # once: no route contributes.
    want = np.zeros((num_variants, members, tp_size * blk, HIDDEN), np.uint16)
    for v in range(num_variants):
        for m in range(members):
            ranks = shared[v, engine[m] * tp_size : (engine[m] + 1) * tp_size]
            acc = np.zeros(ranks.shape[1:], np.float32)
            for r in range(tp_size):
                part = ranks[r].astype(np.float32) + np.float32(0)
                acc = acc + part.astype(BF16).astype(np.float32)
            want[v, m] = acc.astype(BF16).view(np.uint16)

    def body(lhs, me, dest, dest_row, shared, extra_member, extra_row, want):
        lhs, me, dest, dest_row, shared, extra_member, extra_row, want = (
            a[0, 0]
            for a in (lhs, me, dest, dest_row, shared, extra_member, extra_row, want)
        )
        idx = me[0]
        tail_routes = (
            jnp.full((topk, blk), tp_size, jnp.int32),
            jnp.zeros((topk, blk), jnp.int32),
            jnp.zeros((topk, blk), jnp.int32),
            jnp.zeros((topk, blk), jnp.float32),
        )

        def call(i, collective_id, wrong):
            v = i % num_variants
            nanos = jnp.where(i % members == idx, late_ns, 0).astype(jnp.int32)
            me_i = _late(me, nanos[None], lhs)
            rows = lax.dynamic_index_in_dim(shared, v, 0, keepdims=False)
            fit = jnp.asarray(fits)[v]
            out = gmm_v2_mlp_scatter(
                lhs,
                w1,
                w2,
                jnp.asarray(sizes)[v, idx],
                s1,
                s2,
                jnp.zeros((1,), jnp.int32),
                dest,
                dest_row,
                me_i,
                jnp.zeros((1,), jnp.int32),
                (
                    rows.reshape(tp_size * blk, 16, HIDDEN // 16),
                    extra_member,
                    extra_row,
                    jnp.asarray(counts)[v],
                ),
                (*tail_routes, fit),
                num_recv_rows=topk * blk,
                num_extra_recv_rows=tp_size * blk,
                axis=(DP_AXIS, TP_AXIS),
                num_members=(dp_size, tp_size),
                collective_id=collective_id,
                fuse_act="silu",
                maybe_quantize_lhs=False,
            )
            bits = lax.bitcast_convert_type(
                out.reshape(tp_size * blk, HIDDEN), jnp.uint16
            )
            ref = lax.dynamic_index_in_dim(want, v, 0, keepdims=False)
            bad = (bits != ref).any(axis=-1).sum(dtype=jnp.int32)
            return wrong + jnp.where(fit[0] != 0, bad, 0)

        def pair(k, wrong):
            wrong = call(2 * k, 0, wrong)
            return call(2 * k + 1, 1, wrong)

        wrong = lax.fori_loop(0, calls // 2, pair, jnp.int32(0))
        return wrong[None, None, None]

    def per_member(a):
        a = np.asarray(a)
        return jnp.asarray(a.reshape(dp_size, tp_size, *a.shape[1:]))

    spec = P(DP_AXIS, TP_AXIS)
    prog = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 8, out_specs=spec, check_vma=False
        )
    )
    wrong = prog(
        per_member(lhs),
        per_member(np.arange(members, dtype=np.int32)[:, None]),
        per_member(dest),
        per_member(dest_row),
        per_member(np.moveaxis(shared, 0, 1)),
        per_member(extra_member),
        per_member(extra_row),
        per_member(np.moveaxis(want, 0, 1).view(np.uint16)),
    )
    assert np.asarray(wrong).sum() == 0, np.asarray(wrong).reshape(-1)
