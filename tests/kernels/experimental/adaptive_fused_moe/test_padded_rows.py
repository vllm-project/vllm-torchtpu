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
"""Padded rows through ``adaptive_fused_moe(..., padded=...)``.

A padded row routes nowhere: its pairs take expert id ``e_total``, so no
count, table or slab holds them, and its output row is selected to zero. The
checks, per cell of the runner's mask (``TokenPaddingState.update``):

- real rows track a float64 dense reference, per row;
- real rows are bit-equal to the same call without a mask;
- padded rows are exactly zero and finite;
- the routing plan counts only the real rows' pairs.

Each check is held to a fault it must catch (``test_fault_is_caught``).

The device tests need the eight-chip TPU slice: in Pallas TPU interpret mode
on eight CPU devices the kernel's io_callbacks deadlock. The routing side --
mask split, sentinel, plan, combine, select -- also runs on any eight devices
(XLA_FLAGS=--xla_force_host_platform_device_count=8) with pure-JAX stand-ins
for the two Pallas kernels, against the same faults.
"""

import inspect
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import (
    AXIS,
    WeightFormat,
    host,
)
from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import layer as layer_mod
from vllm_torchtpu.kernels.experimental.adaptive_fused_moe.kernel import (
    combine_step_tokens,
)
from vllm_torchtpu.layers.adapter.token_padding import TokenPaddingState

from .test_kernel import (
    CAPACITY,
    EP,
    FP8,
    FP8_TOKEN_BOUND,
    SMALL,
    _make_inputs,
    _make_weights,
    _mesh,
)

T_LOCAL = SMALL["tokens"] // EP
# Real rows per rank, as `TokenPaddingState.update(valid, t_local)` writes.
# U31: a decode rank beside a prefilling peer. M1: that peer is rank 0.
# U0: warmup, or every rank idle -- the mesh routes no pair at all.
CELLS = {
    "U31": (31,) * EP,
    "M1": (T_LOCAL,) + (31,) * (EP - 1),
    "U0": (0,) * EP,
}
ORDER = (0, 1, 6, 7, 2, 3, 4, 5)
TP2_GROUPS = ((6, 1), (4, 3), (0, 7), (2, 5))
# Replica members hold the same rows, so they hold the same mask too.
TP2_VALID = (T_LOCAL, 30, 31, 31, 31, 31, 30, T_LOCAL)


@pytest.fixture(scope="module")
def tpu():
    """At setup, not collection: see test_kernel._require_eight_devices."""
    if jax.device_count() < 8 or jax.devices()[0].platform != "tpu":
        pytest.skip("needs the eight-chip TPU slice")


def _mask(valid, fault_shift=0, t_local=T_LOCAL):
    """The global [tokens] int32 mask, a rank's rows from the runner's state."""
    rows = []
    for v in valid:
        state = TokenPaddingState.create(t_local, device="cpu")
        state.update(max(v - fault_shift, 0), t_local)
        rows.append(state.get_local_padding_mask(t_local).numpy())
    return np.concatenate(rows).astype(np.int32)


def _weights(mesh, order=None):
    w = _make_weights(
        mesh,
        30,
        hidden=SMALL["hidden"],
        inter=SMALL["inter"],
        e_total=SMALL["e_total"],
        weight_format=WeightFormat.FP8,
    )
    if order is None:
        return w
    g_local = SMALL["e_total"] // EP
    sharding = NamedSharding(mesh, P(AXIS))

    def place(a):
        by_rank = np.asarray(a).reshape((EP, g_local) + a.shape[1:])
        return jax.device_put(by_rank[np.asarray(order)].reshape(a.shape), sharding)

    return tuple(place(a) for a in w)


def _inputs(mesh, groups=None, tokens=SMALL["tokens"]):
    """(x, gating); members of a replica group carry identical rows."""
    x, gating = _make_inputs(
        mesh,
        31,
        tokens=tokens,
        hidden=SMALL["hidden"],
        e_total=SMALL["e_total"],
    )
    if groups is None:
        return x, gating
    blocks = np.arange(EP)
    for group in groups:
        blocks[list(group)] = group[0]
    sharding = NamedSharding(mesh, P(AXIS))

    def replicate(a):
        by_rank = np.asarray(a).reshape((EP, T_LOCAL) + a.shape[1:])
        return jax.device_put(by_rank[blocks].reshape(a.shape), sharding)

    return replicate(x), replicate(gating)


def _run(layer, mesh, weights, x, gating, padded=None, **kwargs):
    """The layer under one jit, so the stand-ins and callbacks trace into it."""
    if padded is not None:
        padded = jax.device_put(jnp.asarray(padded), NamedSharding(mesh, P(AXIS)))

    def run(x, gating, padded, *weights):
        return layer.adaptive_fused_moe(
            x,
            *weights,
            gating,
            topk=SMALL["topk"],
            renormalize=True,
            mesh=mesh,
            capacity=CAPACITY,
            weight_format=WeightFormat.FP8,
            padded=padded,
            **kwargs,
        )

    return np.asarray(jax.jit(run)(x, gating, padded, *weights).astype(jnp.float32))


def _topk(gating):
    """[tokens, topk] expert ids in EP-rank order; the logits carry no ties."""
    g = np.asarray(gating, np.float64)
    return np.argsort(-g, axis=-1, kind="stable")[:, : SMALL["topk"]]


def _oracle(x, gating, weights):
    """float64 dense MoE: softmax, top-k, renormalize, dequantized experts."""
    w1, w2, s1, s2 = (np.asarray(a).astype(np.float64) for a in weights)
    w1 = w1 * s1[:, None, :]
    w2 = w2 * s2[:, None, :]
    inter = w1.shape[2] // 2
    x = np.asarray(x).astype(np.float64)
    g = np.asarray(gating, np.float64)
    scores = np.exp(g - g.max(axis=-1, keepdims=True))
    scores /= scores.sum(axis=-1, keepdims=True)
    idx = _topk(gating)
    weight = np.take_along_axis(scores, idx, axis=-1)
    weight /= weight.sum(axis=-1, keepdims=True)
    out = np.zeros_like(x)
    for e in range(w1.shape[0]):
        tok, slot = np.nonzero(idx == e)
        if tok.size:
            h = x[tok] @ w1[e]
            act = h[:, :inter] / (1 + np.exp(-h[:, :inter])) * h[:, inter:]
            out[tok] += weight[tok, slot, None] * (act @ w2[e])
    return out


def _check_rows(out, want, padded, unmasked=None):
    """(a) real rows vs the oracle per row, (b) bit-equal to the unmasked run,
    (c) padded rows exactly zero and finite."""
    live = padded == 0
    assert np.all(np.isfinite(out)), "non-finite output"
    assert not np.any(out[~live]), "a padded row's output is not zero"
    if live.any():
        err = np.linalg.norm(out[live] - want[live], axis=-1)
        scale = np.linalg.norm(want[live], axis=-1)
        worst = float(np.max(err / scale))
        assert worst < FP8_TOKEN_BOUND, f"worst real row {worst:.4f}"
    if unmasked is not None:
        np.testing.assert_array_equal(out[live], unmasked[live])


def _record_run_rows(layer, monkeypatch):
    """Record `routing.run_rows` [E, ep] per shard, and stash each trace's
    plan for `_poison_unwritten_arrivals`."""
    seen, stash = {}, {}
    real = layer.build_routing_tables_sharded

    def record(me, rows):
        seen[int(me)] = np.asarray(rows)

    def wrapped(topk_idx, me, **kwargs):
        routing = real(topk_idx, me, **kwargs)
        stash["routing"] = routing
        jax.debug.callback(record, me, routing.run_rows)
        return routing

    monkeypatch.setattr(layer, "build_routing_tables_sharded", wrapped)
    monkeypatch.setattr(layer, "_LAYER_SM_CACHE", {})
    return seen, stash


def _expected_run_rows(gating, padded, order=None):
    """Pairs per (mesh-order expert, source shard), over real rows only."""
    idx = _topk(gating)
    g_local = SMALL["e_total"] // EP
    if order is not None:
        mesh_index = {rank: i for i, rank in enumerate(order)}
        idx = np.vectorize(lambda e: mesh_index[e // g_local] * g_local + e % g_local)(
            idx
        )
    rows = np.zeros((SMALL["e_total"], EP), np.int32)
    for t in np.nonzero(padded == 0)[0]:
        for e in idx[t]:
            rows[e, t // T_LOCAL] += 1
    return rows


def _check_counts(seen, gating, padded, order=None):
    want = _expected_run_rows(gating, padded, order)
    assert sorted(seen) == list(range(EP))
    for me, rows in seen.items():
        np.testing.assert_array_equal(rows, want, err_msg=f"shard {me}")


@pytest.fixture(scope="module")
def setup(tpu):
    mesh = _mesh()
    weights = _weights(mesh)
    x, gating = _inputs(mesh)
    unmasked = _run(layer_mod, mesh, weights, x, gating)
    return mesh, weights, x, gating, _oracle(x, gating, weights), unmasked


@pytest.mark.multichip
@pytest.mark.parametrize("cell", CELLS)
def test_padded_rows_do_no_work_and_real_rows_are_unchanged(setup, cell, monkeypatch):
    mesh, weights, x, gating, want, unmasked = setup
    padded = _mask(CELLS[cell])
    seen, _ = _record_run_rows(layer_mod, monkeypatch)
    out = _run(layer_mod, mesh, weights, x, gating, padded)
    _check_rows(out, want, padded, unmasked)
    _check_counts(seen, gating, padded)


@pytest.mark.multichip
def test_padded_rows_after_a_nonidentity_mesh_relabel(setup, monkeypatch):
    mesh, weights, x, gating, want, _ = setup
    padded = _mask(CELLS["U31"])
    identity = _run(layer_mod, mesh, weights, x, gating, padded)
    seen, _ = _record_run_rows(layer_mod, monkeypatch)
    placed = _weights(mesh, ORDER)
    out = _run(layer_mod, mesh, placed, x, gating, padded, mesh_ep_ranks=ORDER)
    np.testing.assert_array_equal(out, identity)
    _check_rows(out, want, padded)
    _check_counts(seen, gating, padded, ORDER)


@pytest.mark.multichip
def test_padded_rows_under_token_replicas(tpu):
    mesh = _mesh()
    weights = _weights(mesh)
    x, gating = _inputs(mesh, TP2_GROUPS)
    padded = _mask(TP2_VALID)
    out = _run(
        layer_mod, mesh, weights, x, gating, padded, token_replica_groups=TP2_GROUPS
    )
    unreplicated = _run(layer_mod, mesh, weights, x, gating, padded)
    _check_rows(out, _oracle(x, gating, weights), padded, unreplicated)


@pytest.mark.multichip
@pytest.mark.parametrize("real", [31, 0], ids=["U31", "U0"])
def test_padded_rows_through_the_combine_kernel(setup, real):
    """256 local rows take the Mosaic combine, which DMAs a dead pair's
    clamped arrival row rather than gathering it."""
    mesh, weights, _, _, _, _ = setup
    t_local = 256
    assert combine_step_tokens(t_local)
    x, gating = _inputs(mesh, tokens=EP * t_local)
    padded = _mask((real,) * EP, t_local=t_local)
    unmasked = _run(layer_mod, mesh, weights, x, gating)
    out = _run(layer_mod, mesh, weights, x, gating, padded)
    _check_rows(out, _oracle(x, gating, weights), padded, unmasked)


def _mutated_layer(*edits):
    """layer.py with (old, new) edits, as a fresh module with its own op cache."""
    source = inspect.getsource(layer_mod)
    for old, new in zip(edits[::2], edits[1::2]):
        assert source.count(old) == 1, f"anchor not found once: {old!r}"
        source = source.replace(old, new)
    module = types.ModuleType(f"{layer_mod.__name__}_fault")
    exec(compile(source, layer_mod.__file__, "exec"), module.__dict__)
    return module


_SENTINEL = "topk_idx = jnp.where(row_routes, topk_idx, jnp.int32(e_total))\n"
_SELECT = "combined = jnp.where(row_routes, combined, jnp.zeros_like(combined))\n"
_SPLIT = "padded_l = token_layout.split(padded_l[:, None], me, fill_value=1)"
# The sentinel relabelled as if it were set first: it lands on expert 0.
_SENTINEL_RELABELLED = (
    "topk_idx = jnp.where(row_routes, topk_idx, "
    "_relabel_expert_ids_to_mesh_order(jnp.full_like(topk_idx, e_total), "
    "g_local=g_local, mesh_ep_ranks=mesh_ep_ranks))\n"
)
# Every replica member takes the group's first slice of the mask.
_NOT_SPLIT = "padded_l = padded_l[:t_local, None]"


def _poison_unwritten_arrivals(layer, stash, monkeypatch):
    """Arrival rows past what this shard received this step become NaN, the
    way uninitialized memory may read."""
    build = layer.build_fused_ep_moe_kernel

    def poisoned_build(**kwargs):
        kfn = build(**kwargs)

        def call(*args, rank, **kw):
            arrivals, scales = kfn(*args, rank=rank, **kw)
            received = stash["routing"].recv_rows[rank[0, 0]]
            row = jnp.arange(arrivals.shape[0])[:, None, None]
            nan = jnp.asarray(jnp.nan, arrivals.dtype)
            return jnp.where(row < received, arrivals, nan), scales

        return call

    monkeypatch.setattr(layer, "build_fused_ep_moe_kernel", poisoned_build)


def _fault_f1_sentinel_before_relabel(setup, monkeypatch):
    mesh, weights, x, gating, want, _ = setup
    layer = _mutated_layer(_SENTINEL, _SENTINEL_RELABELLED)
    padded = _mask(CELLS["U31"])
    seen, _ = _record_run_rows(layer, monkeypatch)
    placed = _weights(mesh, ORDER)
    out = _run(layer, mesh, placed, x, gating, padded, mesh_ep_ranks=ORDER)
    _check_rows(out, want, padded)
    _check_counts(seen, gating, padded, ORDER)


def _fault_f2_select_removed(setup, monkeypatch):
    mesh, weights, x, gating, want, _ = setup
    layer = _mutated_layer(_SELECT, "pass\n")
    padded = _mask(CELLS["U0"])
    _, stash = _record_run_rows(layer, monkeypatch)
    _poison_unwritten_arrivals(layer, stash, monkeypatch)
    _check_rows(_run(layer, mesh, weights, x, gating, padded), want, padded)


def _fault_f3_mask_off_by_one(setup, monkeypatch):
    mesh, weights, x, gating, want, _ = setup
    padded = _mask(CELLS["U31"])
    out = _run(layer_mod, mesh, weights, x, gating, _mask(CELLS["U31"], 1))
    _check_rows(out, want, padded)


def _fault_f4_mask_not_split(setup, monkeypatch):
    mesh, weights, _, _, _, _ = setup
    layer = _mutated_layer(_SPLIT, _NOT_SPLIT)
    x, gating = _inputs(mesh, TP2_GROUPS)
    padded = _mask((31,) * EP)
    out = _run(layer, mesh, weights, x, gating, padded, token_replica_groups=TP2_GROUPS)
    _check_rows(out, _oracle(x, gating, weights), padded)


def _fault_f5_weight_zeroed_only(setup, monkeypatch):
    mesh, weights, x, gating, want, _ = setup
    layer = _mutated_layer(_SENTINEL, "\n")
    padded = _mask(CELLS["U31"])
    seen, _ = _record_run_rows(layer, monkeypatch)
    _check_rows(_run(layer, mesh, weights, x, gating, padded), want, padded)
    _check_counts(seen, gating, padded)


@pytest.mark.multichip
@pytest.mark.parametrize(
    "fault",
    [
        _fault_f1_sentinel_before_relabel,
        _fault_f2_select_removed,
        _fault_f3_mask_off_by_one,
        _fault_f4_mask_not_split,
        _fault_f5_weight_zeroed_only,
    ],
    ids=lambda f: f.__name__.removeprefix("_fault_"),
)
def test_fault_is_caught(setup, monkeypatch, fault):
    """Positive controls: each fault must fail one of the checks above."""
    with pytest.raises(AssertionError):
        fault(setup, monkeypatch)


@pytest.mark.cpu_test
def test_sentinel_pairs_fall_out_of_the_host_tables():
    """No device: the plan built over ids carrying `e_total` counts nothing
    for them and sends every one of them off every shard's slab."""
    ep, t_local, topk, e_total, block, tile_m = 2, 8, 2, 4, 16, 128
    rng = np.random.default_rng(0)
    idx = np.stack([rng.permutation(e_total)[:topk] for _ in range(ep * t_local)])
    dead = rng.random(ep * t_local) < 0.5
    idx[dead] = e_total
    idx = jnp.asarray(idx, jnp.int32)
    stride = host.ragged_stride_bound(ep * t_local, topk, e_total, tile_m)

    def gather(rows):
        return jnp.concatenate(
            [
                host.pair_block_hist(
                    idx[d * t_local : (d + 1) * t_local].reshape(-1, block), e_total
                ).sum(axis=0, keepdims=True)
                for d in range(ep)
            ]
        )

    for me in range(ep):
        routing = host.build_routing_tables_sharded(
            idx,
            me,
            e_total=e_total,
            ep=ep,
            t_local=t_local,
            block=block,
            tile_m=tile_m,
            shard_stride=stride,
            all_gather_rows=gather,
        )
        ids = np.asarray(idx)
        assert int(routing.run_rows.sum()) == int((~dead).sum()) * topk
        slab = np.asarray(routing.slab_row).reshape(ep * t_local, topk)
        off = host.off_shard_slab_row(stride, ep)
        assert np.all(slab[dead] == off)
        mine = ~dead[:, None] & (ids // (e_total // ep) == me)
        assert np.all(slab[mine] != off)
        assert len(set(slab[mine].tolist())) == int(mine.sum())


@pytest.mark.multichip
@pytest.mark.parametrize(
    "options",
    [{}, {"mesh_ep_ranks": ORDER}, {"token_replica_groups": TP2_GROUPS}],
    ids=["identity", "mesh_order", "replicas"],
)
def test_padded_layer_lowers_for_tpu_from_a_host(monkeypatch, options):
    """The program the slice runs, lowered for TPU from any eight devices."""
    if jax.device_count() < 8:
        pytest.skip("needs eight devices; CPU virtual devices are sufficient")
    if jax.devices()[0].platform != "tpu":
        from jax._src.pallas.mosaic import tpu_info

        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
        monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
        monkeypatch.setattr(tpu_info, "get_tpu_info", lambda: info)
    mesh = _mesh()
    weights = _weights(mesh)
    x, gating = _inputs(mesh)
    padded = jnp.asarray(_mask(CELLS["M1"]))

    def run(x, gating, padded, *weights):
        return layer_mod.adaptive_fused_moe(
            x,
            *weights,
            gating,
            topk=SMALL["topk"],
            renormalize=True,
            mesh=mesh,
            capacity=CAPACITY,
            weight_format=WeightFormat.FP8,
            padded=padded,
            **options,
        )

    lowered = jax.jit(run).trace(x, gating, padded, *weights)
    lowered.lower(lowering_platforms=("tpu",))
    with pytest.raises(ValueError, match="padded must have shape"):
        run(x, gating, padded[:-1], *weights)


def _stand_in_kernels(layer, stash, monkeypatch):
    """Pure-JAX stand-ins for the two Pallas kernels: top-k for the selector,
    and for the expert kernel an arrival buffer holding 1 up to the rows this
    shard received and NaN past them. A routed row then combines to 1."""

    def build(**kwargs):
        ep, g_local, tile_m = kwargs["ep"], kwargs["g_local"], kwargs["capacity"]
        lanes = (kwargs["hidden"] // host.HIDDEN_LANE_BLOCK, host.HIDDEN_LANE_BLOCK)

        def call(*args, recv_rows, rank, **kw):
            received = stash["routing"].recv_rows[rank[0, 0]]
            row = jnp.arange(recv_rows)[:, None, None]
            ones = jnp.where(row < received, 1.0, jnp.nan)
            arrivals = jnp.broadcast_to(ones, (recv_rows, *lanes)).astype(FP8)
            mirror = host.arrival_mirror_rows(recv_rows, ep * g_local, tile_m)
            return arrivals, jnp.ones((mirror, tile_m), jnp.float32)

        return call

    monkeypatch.setattr(layer, "build_fused_ep_moe_kernel", build)
    monkeypatch.setattr(
        layer, "pallas_select", lambda scores, topk, block_rows: lax.top_k(scores, topk)
    )


def _run_stand_in(layer, monkeypatch, valid, mask=None, **options):
    """Run on stand-ins and check every row against the true mask `valid`."""
    mesh = _mesh()
    groups = options.get("token_replica_groups")
    weights = _weights(mesh, options.get("mesh_ep_ranks"))
    x, gating = _inputs(mesh, groups)
    padded = _mask(valid)
    seen, stash = _record_run_rows(layer, monkeypatch)
    _stand_in_kernels(layer, stash, monkeypatch)
    out = _run(
        layer, mesh, weights, x, gating, padded if mask is None else mask, **options
    )
    live = padded == 0
    assert np.all(np.isfinite(out)), "non-finite output"
    assert not np.any(out[~live]), "a padded row's output is not zero"
    np.testing.assert_allclose(out[live], 1, atol=1e-2, err_msg="a real row")
    if groups is None:
        _check_counts(seen, gating, padded, options.get("mesh_ep_ranks"))


@pytest.mark.multichip
@pytest.mark.parametrize(
    "valid,options",
    [
        (CELLS["U31"], {}),
        (CELLS["M1"], {}),
        (CELLS["U0"], {}),
        (CELLS["M1"], {"mesh_ep_ranks": ORDER}),
        (TP2_VALID, {"token_replica_groups": TP2_GROUPS}),
    ],
    ids=["U31", "M1", "U0", "mesh_order", "replicas"],
)
def test_routing_side_on_stand_in_kernels(monkeypatch, valid, options):
    if jax.device_count() < 8:
        pytest.skip("needs eight devices; CPU virtual devices are sufficient")
    _run_stand_in(layer_mod, monkeypatch, valid, **options)


@pytest.mark.multichip
@pytest.mark.parametrize(
    "fault",
    ["sentinel_before_relabel", "select_removed", "mask_off_by_one", "not_split"]
    + ["weight_zeroed_only"],
)
def test_stand_in_kernels_catch_fault(monkeypatch, fault):
    if jax.device_count() < 8:
        pytest.skip("needs eight devices; CPU virtual devices are sufficient")
    layer, valid, mask, options = layer_mod, CELLS["U31"], None, {}
    if fault == "sentinel_before_relabel":
        layer, options = (
            _mutated_layer(_SENTINEL, _SENTINEL_RELABELLED),
            {"mesh_ep_ranks": ORDER},
        )
    elif fault == "select_removed":
        layer, valid = _mutated_layer(_SELECT, "pass\n"), CELLS["U0"]
    elif fault == "mask_off_by_one":
        mask = _mask(valid, 1)
    elif fault == "not_split":
        layer = _mutated_layer(_SPLIT, _NOT_SPLIT)
        valid, options = TP2_VALID, {"token_replica_groups": TP2_GROUPS}
    else:
        layer = _mutated_layer(_SENTINEL, "\n")
    with pytest.raises(AssertionError):
        _run_stand_in(layer, monkeypatch, valid, mask, **options)


# (this change, main) pairs: layer.py without the padded-row skip, which is
# main's program. An anchor that drifts fails the test rather than passing it.
_WITHOUT_SKIP = (
    "    padded=None,\n):",
    "):",
    "    if padded is not None and padded.shape != (T,):\n"
    '        raise ValueError("padded must have shape [tokens]")\n',
    "",
    "    has_padded = padded is not None\n",
    "",
    "        if has_padded:\n"
    "            *scales_gating_biases, padded_l = scales_gating_biases\n",
    "",
    "        if has_padded:\n"
    "            padded_l = token_layout.split(padded_l[:, None], me, fill_value=1)\n",
    "",
    "        if has_padded:\n"
    "            # With the skip armed, a row that routes nowhere takes the expert\n"
    "            # id one past the last. Every count, table and slab bins over\n"
    "            # arange(e_total), so its pairs get none of them and no expert\n"
    "            # computes it. After the relabel, which would map this id onto\n"
    "            # expert 0.\n"
    "            row_routes = row_routes & (padded_l == 0)\n"
    "            " + _SENTINEL,
    "",
    "        if has_padded:\n"
    "            # Such a row indexes arrival rows nothing wrote this step, and a\n"
    "            # zero weight times a NaN there is NaN: select, not scale.\n"
    "            " + _SELECT,
    "",
    "    padded_args = (padded,) if has_padded else ()\n"
    "    in_specs = (P(ax),) * (5 + len(scale_args) + len(bias_args)"
    " + len(padded_args))\n"
    "    args = (x, rank, w1, w2) + scale_args + (gating,) + bias_args + padded_args\n",
    "    in_specs = (P(ax),) * (5 + len(scale_args) + len(bias_args))\n"
    "    args = (x, rank, w1, w2) + scale_args + (gating,) + bias_args\n",
    "        has_padded and padded.dtype,\n",
    "",
)


@pytest.mark.multichip
def test_flags_off_program_is_mains(monkeypatch):
    """Without `padded` the layer lowers to the program it did before the
    skip: rows with no finite gating keep weight 0, not the sentinel."""
    if jax.device_count() < 8:
        pytest.skip("needs eight devices; CPU virtual devices are sufficient")
    if jax.devices()[0].platform != "tpu":
        from jax._src.pallas.mosaic import tpu_info

        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
        monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
        monkeypatch.setattr(tpu_info, "get_tpu_info", lambda: info)
    mesh = _mesh()
    weights = _weights(mesh)
    x, gating = _inputs(mesh, TP2_GROUPS)
    main = _mutated_layer(*_WITHOUT_SKIP)
    assert "padded" not in inspect.signature(main.adaptive_fused_moe).parameters

    def lowered(layer):
        def run(x, gating, *weights):
            return layer.adaptive_fused_moe(
                x,
                *weights,
                gating,
                topk=SMALL["topk"],
                renormalize=True,
                mesh=mesh,
                capacity=CAPACITY,
                weight_format=WeightFormat.FP8,
                token_replica_groups=TP2_GROUPS,
            )

        traced = jax.jit(run).trace(x, gating, *weights)
        return traced.lower(lowering_platforms=("tpu",)).as_text()

    assert lowered(layer_mod) == lowered(main)
