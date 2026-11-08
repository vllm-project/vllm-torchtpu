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
"""Padded rows left unrouted (id -1, weight 0) in the EP MoE paths.

The padded rows of a partly filled step carry identical hidden states, so with
their selected experts kept they all route to the same experts. Routed nowhere,
nothing is sent, counted or computed for them, their routed output is 0, and
the real rows' outputs stay those of the step with the padded rows routed.
"""

import functools

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    quantize_to_native_fp4_kmajor,
    resolve_onehot_permute_threshold,
)
from vllm_torchtpu.layers.core.ragged_a2a_moe import (
    DP_AXIS,
    TP_AXIS,
    _capacity_overflow,
    _member_index,
    exchange_capacity,
    exchange_need,
    owner_dests,
    ragged_a2a_moe,
    route_plan,
    streamed_dests,
)

HIDDEN = 2048
INTER = 512
NUM_EXPERTS = 64
BF16 = ml_dtypes.bfloat16


def _hot_routing(rng, dp_size, num_tokens, num_padded, num_experts, topk, *, drop):
    """[D, S, K] ids and f32 weights: real rows a random top-k, the last
    ``num_padded`` rows of every member one shared top-k (experts 0..k-1, the
    first ranks' experts), or id -1 / weight 0 with ``drop``."""
    logits = rng.standard_normal((dp_size, num_tokens, num_experts))
    order = np.argsort(-logits, axis=-1)[..., :topk]
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    ids = order.astype(np.int32)
    weights = np.take_along_axis(probs, order, axis=-1).astype(np.float32)
    pad = slice(num_tokens - num_padded, num_tokens)
    if drop:
        ids[:, pad] = -1
        weights[:, pad] = 0
    else:
        ids[:, pad] = np.arange(topk, dtype=np.int32)
        weights[:, pad] = np.linspace(0.3, 0.05, topk, dtype=np.float32)
    return ids, weights


def _require_eight_tpu_devices():
    devices = jax.devices()
    if devices[0].platform != "tpu" or len(devices) < 8:
        pytest.skip("needs 8 TPU devices")
    return devices[:8]


def _exchange_inputs(dp_size, tp_size, num_tokens, num_padded, topk, *, drop):
    """Stacked [dp, tp, ...] operands of `ragged_a2a_moe`; padded rows share
    one hidden state."""
    rng = np.random.default_rng(dp_size * 100 + num_tokens + topk)
    x = rng.standard_normal((dp_size, 1, num_tokens, HIDDEN)) / 4
    x[:, :, num_tokens - num_padded :] = x[0, 0, -1].copy()
    x = np.broadcast_to(x, (dp_size, tp_size, num_tokens, HIDDEN))
    ids, weights = _hot_routing(
        np.random.default_rng(7),
        dp_size,
        num_tokens,
        num_padded,
        NUM_EXPERTS,
        topk,
        drop=drop,
    )
    ids = np.broadcast_to(ids[:, None], (dp_size, tp_size, num_tokens, topk))
    weights = np.broadcast_to(weights[:, None], ids.shape)
    key = jax.random.key(11)
    k1, k2 = jax.random.split(key)
    w1 = jax.random.normal(k1, (NUM_EXPERTS, HIDDEN, 2 * INTER)) / 8
    w2 = jax.random.normal(k2, (NUM_EXPERTS, INTER, HIDDEN)) / 8
    w1, s1 = quantize_to_native_fp4_kmajor(w1, 256, pack=False)
    w2, s2 = quantize_to_native_fp4_kmajor(w2, 256, pack=False)
    num_ranks = dp_size * tp_size
    local = NUM_EXPERTS // num_ranks
    es = jnp.arange(num_ranks, dtype=jnp.int32) * local

    def per_rank(a):
        return a.reshape(dp_size, tp_size, local, *a.shape[1:])

    return (
        jnp.asarray(x, jnp.bfloat16),
        per_rank(w1),
        per_rank(w2),
        per_rank(s1),
        per_rank(s2),
        jnp.asarray(weights, jnp.bfloat16),
        jnp.asarray(ids),
        es.reshape(dp_size, tp_size, 1),
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_real", [52, 80, 104])
def test_dropped_padded_rows_keep_the_capacity_exchange(num_real):
    """Served 4-host decode step (DP8 x TP4, 16 experts per rank, 26 requests
    per engine with 3 drafts = 104 rows, capacity x2 / x2, 32-row tiles): rows
    past ``num_real`` routed to one shared top-10 overflow the capacity
    buffers at half load (every padded row of the group lands on the same
    ranks); routed nowhere they leave the step within capacity."""
    dp_size, tp_size, num_tokens, local, topk = 8, 4, 104, 16, 10
    rows, routes = exchange_capacity(
        dp_size=dp_size,
        tp_size=tp_size,
        num_tokens=num_tokens,
        topk=topk,
        local_num_experts=local,
        factor=2.0,
        onehot_moe_permute_threshold=3840,
        use_sparse_core=True,
        route_factor=2.0,
        row_tile=32,
    )
    assert (rows, routes) == (480, 640)
    overflow = {}
    for drop in (False, True):
        ids, _ = _hot_routing(
            np.random.default_rng(3),
            dp_size,
            num_tokens,
            num_tokens - num_real,
            dp_size * tp_size * local,
            topk,
            drop=drop,
        )
        overflow[drop] = bool(
            _capacity_overflow(
                jnp.asarray(ids),
                tp_size=tp_size,
                local_num_experts=local,
                rows=rows,
                routes=routes,
            )
        )
    assert not overflow[True]
    if num_real == 52:
        assert overflow[False]
    if num_real == num_tokens:
        assert not overflow[False]


@pytest.mark.cpu_test
@pytest.mark.parametrize("combine", ["streamed", "owner"])
@pytest.mark.parametrize("dp_size,tp_size,local", [(8, 4, 16), (2, 4, 64), (4, 2, 8)])
def test_dropped_rows_reach_no_member_and_plan_no_route(
    combine, dp_size, tp_size, local
):
    """Id -1 rows need no member of any TP rank's group, and the sorted plan of
    every rank computes none of their routes; the real rows' plan is that of
    the same step without the padded rows."""
    num_tokens, num_padded, topk = 24, 9, 10
    num_experts = dp_size * tp_size * local
    hot, hot_w = _hot_routing(
        np.random.default_rng(5),
        dp_size,
        num_tokens,
        num_padded,
        num_experts,
        topk,
        drop=False,
    )
    ids, weights = _hot_routing(
        np.random.default_rng(5),
        dp_size,
        num_tokens,
        num_padded,
        num_experts,
        topk,
        drop=True,
    )
    assert (
        ids[:, : num_tokens - num_padded] == hot[:, : num_tokens - num_padded]
    ).all()
    gathered = jnp.asarray(ids)
    mesh = dict(dp_size=dp_size, tp_size=tp_size, local_num_experts=local)
    for ep_rank in range(dp_size * tp_size):
        need = np.asarray(
            exchange_need(
                gathered, jnp.int32(ep_rank), tp_size=tp_size, local_num_experts=local
            )
        )
        assert not need[:, num_tokens - num_padded :].any()
        member_index = functools.partial(
            _member_index,
            ep_rank=jnp.int32(ep_rank),
            dp_size=dp_size,
            tp_size=tp_size,
            local_num_experts=local,
        )
        if combine == "owner":
            dests = functools.partial(owner_dests, **mesh)
        else:
            dests = functools.partial(
                streamed_dests, member_index=member_index, dp_size=dp_size
            )
        slots = dp_size * num_tokens
        in_row = jnp.arange(slots, dtype=jnp.int32).reshape(dp_size, num_tokens)
        es = jnp.asarray([ep_rank * local], jnp.int32)
        plan = route_plan(
            gathered,
            in_row,
            jnp.bool_(True),
            es,
            dests,
            local_num_experts=local,
            routes=slots * topk,
            rows=slots,
            gathered_weights=jnp.asarray(weights, jnp.bfloat16),
        )
        token_sorted, group_sizes, _, _ = (np.asarray(a) for a in plan)
        computed = token_sorted[: int(group_sizes.sum())]
        assert ((computed % num_tokens) < num_tokens - num_padded).all()
        # The same routes as the step whose padded rows route but are skipped
        # as zero-weight rows.
        hot_w[:, num_tokens - num_padded :] = 0
        ref = route_plan(
            jnp.asarray(hot),
            in_row,
            jnp.bool_(True),
            es,
            dests,
            local_num_experts=local,
            routes=slots * topk,
            rows=slots,
            gathered_weights=jnp.asarray(hot_w, jnp.bfloat16),
        )
        for got, want in zip(plan, ref, strict=True):
            n = int(group_sizes.sum())
            np.testing.assert_array_equal(np.asarray(got)[:n], np.asarray(want)[:n])


def _tp_sum_f32(a):
    """[dp, tp, S, H] bf16 -> [dp, S, H]: f32 sum in rank order, one rounding."""
    acc = a[:, 0].astype(np.float32)
    for r in range(1, a.shape[1]):
        acc = acc + a[:, r].astype(np.float32)
    return acc.astype(BF16)


@pytest.mark.multichip
@pytest.mark.parametrize(
    "combine,topk,sorted_plan,row_copy",
    [
        ("partials", 2, False, False),
        ("partials", 10, False, False),
        ("streamed", 10, False, False),
        ("streamed", 10, True, False),
        ("streamed", 10, True, True),
        ("owner", 10, False, False),
        ("owner", 10, True, False),
        ("owner", 10, True, True),
    ],
)
@pytest.mark.parametrize(
    "dp_size,tp_size,num_tokens,factor,route_factor",
    [
        (8, 1, 24, 1.15, 1.5),  # capacity exchange; the padded hot spot overflows it
        (4, 2, 24, 1.0, 1.5),
        (8, 1, 24, 0.0, 0.0),  # full-size exchange
        (2, 4, 48, 2.5, 3.0),  # proxy layout: full-size exchange
    ],
)
def test_dropped_padded_rows_keep_real_rows(
    combine,
    topk,
    sorted_plan,
    row_copy,
    dp_size,
    tp_size,
    num_tokens,
    factor,
    route_factor,
):
    """Real rows: bitwise the step with the padded rows routed (the streamed and
    owner sums run in a fixed per-token order; the returned partials' combine
    matmul only where no member serves three or more routes of one token).
    Padded rows: routed output 0 (owner mode: the shared partials' TP sum)."""
    devices = _require_eight_tpu_devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    num_padded = num_tokens // 2
    num_real = num_tokens - num_padded
    hot = _exchange_inputs(dp_size, tp_size, num_tokens, num_padded, topk, drop=False)
    dropped = _exchange_inputs(
        dp_size, tp_size, num_tokens, num_padded, topk, drop=True
    )
    threshold = resolve_onehot_permute_threshold(dp_size * num_tokens * topk)
    rows, routes = exchange_capacity(
        dp_size=dp_size,
        tp_size=tp_size,
        num_tokens=num_tokens,
        topk=topk,
        local_num_experts=NUM_EXPERTS // (dp_size * tp_size),
        factor=factor,
        onehot_moe_permute_threshold=threshold,
        use_sparse_core=True,
        route_factor=route_factor or None,
    )
    if rows < dp_size * num_tokens:
        # The hot step runs the full-size exchange, the dropped one fits.
        def overflow(ids):
            return bool(
                _capacity_overflow(
                    ids[:, 0],
                    tp_size=tp_size,
                    local_num_experts=NUM_EXPERTS // (dp_size * tp_size),
                    rows=rows,
                    routes=routes,
                )
            )

        assert overflow(hot[6]) and not overflow(dropped[6])
    common = dict(
        mesh=mesh,
        topk=topk,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=threshold,
        skip_padded_tokens=True,
        capacity_factor=factor,
        route_factor=route_factor,
        streamed_combine=combine == "streamed",
        owner_combine=combine == "owner",
        sorted_plan=sorted_plan,
        row_gather=row_copy,
        row_scatter=row_copy,
    )
    kwargs = {}
    if combine == "owner":
        shared = np.random.default_rng(3).standard_normal(
            (dp_size, tp_size, num_tokens, HIDDEN)
        )
        kwargs["shared"] = jnp.asarray(shared / 8, jnp.bfloat16)
    run = jax.jit(functools.partial(ragged_a2a_moe, **common))
    base = np.asarray(run(*hot, **kwargs))
    got = np.asarray(run(*dropped, **kwargs))
    assert np.isfinite(got.astype(np.float32)).all()
    real_base, real_got = base[:, :, :num_real], got[:, :, :num_real]
    assert np.abs(real_base.astype(np.float32)).max() > 0
    if combine != "partials" or topk <= 2:
        np.testing.assert_array_equal(
            real_got.view(np.uint16), real_base.view(np.uint16)
        )
    else:
        differ = (real_got.view(np.uint16) != real_base.view(np.uint16)).mean()
        assert differ < 1e-3, differ
        np.testing.assert_allclose(
            real_got.astype(np.float32),
            real_base.astype(np.float32),
            rtol=2**-7,
            atol=1e-6,
        )
    padded = got[:, :, num_real:]
    if combine == "owner":
        want = _tp_sum_f32(np.asarray(kwargs["shared"]))[:, None, num_real:]
        want = np.broadcast_to(want, padded.shape)
        np.testing.assert_array_equal(padded.view(np.uint16), want.view(np.uint16))
    else:
        assert (padded.astype(np.float32) == 0).all()


@pytest.mark.multichip
@pytest.mark.parametrize("topk", [2, 10])
@pytest.mark.parametrize("path", ["onehot", "sparse_core"])
def test_local_moe_drops_padded_routes(path, topk):
    """The all-gather EP path's local MoE (an unquantized draft layer's): a rank
    whose experts the padded rows all select computes none of their routes; the
    real rows keep their outputs (bitwise where the combine sums each token's
    routes in a fixed order: the SparseCore combine, or at most two routes of a
    token on the rank) and the padded rows' output is 0."""
    devices = _require_eight_tpu_devices()
    num_tokens, num_padded, local = 96, 40, 8
    num_real = num_tokens - num_padded
    rng = np.random.default_rng(topk)
    x = rng.standard_normal((num_tokens, HIDDEN)) / 4
    x[num_real:] = x[-1].copy()
    x = jnp.asarray(x, jnp.bfloat16)
    k1, k2 = jax.random.split(jax.random.key(13))
    w1 = (jax.random.normal(k1, (local, HIDDEN, 2 * INTER)) / 8).astype(jnp.bfloat16)
    w2 = (jax.random.normal(k2, (local, INTER, HIDDEN)) / 8).astype(jnp.bfloat16)
    outs = {}
    for drop in (False, True):
        ids, weights = _hot_routing(
            np.random.default_rng(9),
            1,
            num_tokens,
            num_padded,
            NUM_EXPERTS,
            topk,
            drop=drop,
        )
        kw = dict(
            experts_start=jnp.int32(0),
            topk=topk,
            activation="silu",
            use_ep=True,
            use_sparse_core=True,
            onehot_moe_permute_threshold=num_tokens * topk if path == "onehot" else 0,
            skip_padded_tokens=True,
        )
        args = (
            x,
            w1,
            w2,
            None,
            None,
            None,
            None,
            jnp.asarray(weights[0], jnp.bfloat16),
            jnp.asarray(ids[0]),
        )
        with jax.default_device(devices[0]):
            outs[drop] = np.asarray(
                jax.jit(functools.partial(fused_moe_func, **kw))(*args)
            )
    base, got = outs[False], outs[True]
    assert np.isfinite(got.astype(np.float32)).all()
    assert np.abs(base[:num_real].astype(np.float32)).max() > 0
    if path == "sparse_core" or topk <= 2:
        np.testing.assert_array_equal(
            got[:num_real].view(np.uint16), base[:num_real].view(np.uint16)
        )
    else:
        differ = (
            got[:num_real].view(np.uint16) != base[:num_real].view(np.uint16)
        ).mean()
        assert differ < 1e-2, differ
        np.testing.assert_allclose(
            got[:num_real].astype(np.float32),
            base[:num_real].astype(np.float32),
            rtol=2**-7,
            atol=1e-6,
        )
    assert (got[num_real:].astype(np.float32) == 0).all()
