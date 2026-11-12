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
"""Opt-in dense-ring schedule options of `TwoRing` (`Tiles.dwait_hoist`,
`ws_pre`, the `hook=` / `split=` forms of `dmatmul`), `TwoRing.claim` and the
deferred lazy-segment flush.

Each option is run in Pallas interpret mode against the default schedule
(bit-identical outputs: the options only move waits and bank arithmetic) and
checked structurally on the kernel jaxpr, so an option that is silently
ignored fails.
"""

from __future__ import annotations

import collections
import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tests.megakernel.core.megakernel_fakes import TINY_TILES, FakeGeometry
from vllm_torchtpu.megakernel.core import planner, stream, vmem
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.core.ops import BF16, F32
from vllm_torchtpu.megakernel.core.spec import LayerType, ModelSpec, Module, WeightDecl

_K, _N, _L = 768, 640, 3


class _WideTileModule(Module):
    """One `n_t = 6`, `N = 640` static weight (`t_k = 128` at a 160 KiB bank)."""

    name = "wide_tile"
    reduction = "allreduce"

    def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
        return [WeightDecl("w", K=_K, N=_N, dtype=BF16, stream_ring="static")]

    def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
        return {"acc": _N, "xb2": _K}


def _tiles(**kw: Any) -> Tiles:
    return dataclasses.replace(TINY_TILES, bank=160 << 10, nbd=6, **kw)


LayerFn = Callable[[stream.TwoRing, dict[str, Any]], Any]


def _plain_layer(ring: stream.TwoRing, scr: dict[str, Any]) -> Any:
    return ring.dmatmul("w", scr["xb"], _N)


def _run(
    tiles: Tiles,
    layer: LayerFn = _plain_layer,
    *,
    lazy: bool = False,
    own_segment: bool = False,
):
    """Run `_L` layers of `res[:, :N] += layer(...)`; returns `(out, ref, call, args)`.

    `layer` runs inside `segment(lazy=lazy)` unless `own_segment`.
    """
    ex = ExecSpec(B=4, T=1, R=1, ctx=128, tiles=tiles, interpret=True)
    g = FakeGeometry(L=_L, H=_K)
    ms = ModelSpec(
        "stream_schedule", config=g, layer_type=LayerType("lt", (_WideTileModule(),))
    )
    cp, scratch = vmem.build_scratch(ms, ex)
    sp = cp.stream
    assert sp.nt["w"] == 6 and sp.tk["w"] == 128
    assert sp.dwait_hoist == tiles.dwait_hoist
    s_names = list(scratch.keys())

    def body(x_ref, w_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        ring = stream.TwoRing(sp, {"w": w_ref}, scr, interpret=True)
        for i in range(sp.NBD):
            ring.fetch(jnp.int32(i))
        scr["xb"][...] = x_ref[...].astype(BF16)

        def layer_fn(lyr, _):
            ring.begin_layer(lyr)
            if own_segment:
                y = layer(ring, scr)
            else:
                with ring.segment(lazy=lazy):
                    y = layer(ring, scr)
            scr["xb"][:, :_N] = (scr["xb"][:, :_N].astype(F32) + y).astype(BF16)
            ring.assert_flushed()
            return 0

        lax.fori_loop(0, _L, layer_fn, 0)
        out_ref[...] = scr["xb"][:, :_N].astype(F32)

    wsh = planner.weight_shapes(ms, ex, cp.plan)
    k0, k1 = jax.random.split(jax.random.PRNGKey(31), 2)
    x = jax.random.normal(k0, (ex.MB, _K), dtype=BF16)
    w_full = jax.random.normal(k1, (_L, _K, _N), dtype=BF16) * 0.05
    args = (x, w_full.reshape(wsh["w"][0]))
    call = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((ex.MB, _N), F32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[scratch[k] for k in s_names],
        interpret=pltpu.InterpretParams(detect_races=True),
    )
    out = np.asarray(jax.block_until_ready(jax.jit(call)(*args)))
    assert ipc.races is not None and not ipc.races.races_found
    ref = x.astype(F32)
    for lyr in range(_L):
        y = ref.astype(BF16).astype(F32) @ w_full[lyr].astype(F32)
        ref = ref.at[:, :_N].set(
            (ref[:, :_N].astype(BF16).astype(F32) + y).astype(BF16).astype(F32)
        )
    return out, np.asarray(ref[:, :_N]), call, args


def _sub_jaxprs(v: Any):
    """Jaxprs inside an equation parameter (a `Jaxpr`, a `ClosedJaxpr`, or a
    tuple / list of them)."""
    if hasattr(v, "eqns"):
        yield v
    elif hasattr(getattr(v, "jaxpr", None), "eqns"):
        yield v.jaxpr
    elif isinstance(v, (tuple, list)):
        for x in v:
            yield from _sub_jaxprs(x)


def prim_counts(fn: Callable[..., Any], *args: Any) -> collections.Counter:
    """Primitive histogram of `fn`'s jaxpr, recursing into every sub-jaxpr
    (including the Pallas kernel body)."""
    cnt: collections.Counter = collections.Counter()

    def walk(jx: Any) -> None:
        for eqn in jx.eqns:
            cnt[eqn.primitive.name] += 1
            for v in eqn.params.values():
                for sub in _sub_jaxprs(v):
                    walk(sub)

    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    return cnt


# (dense_acc_pair, dmm_unroll): U=2 in a traced fori step / fully unrolled / plain.
_FORMS = [(True, 0), (True, 8), (False, 6)]


@pytest.mark.parametrize("dense_acc_pair,dmm_unroll", _FORMS)
@pytest.mark.parametrize("lazy,rel_fetch", [(False, False), (True, True)])
def test_dwait_hoist_bit_identical_and_reaches_kernel(
    dense_acc_pair: bool, dmm_unroll: int, lazy: bool, rel_fetch: bool
):
    """`dwait_hoist` keeps outputs bit-identical, waits the same tiles, and
    computes fewer ring banks (`rem`) in the kernel."""
    kw = dict(dense_acc_pair=dense_acc_pair, dmm_unroll=dmm_unroll, rel_fetch=rel_fetch)
    out0, ref, call0, args = _run(_tiles(**kw), lazy=lazy)
    out1, _, call1, _ = _run(_tiles(dwait_hoist=True, **kw), lazy=lazy)
    np.testing.assert_allclose(out0, ref, atol=1e-2, rtol=1e-2)
    np.testing.assert_array_equal(out1, out0)
    c0, c1 = prim_counts(call0, *args), prim_counts(call1, *args)
    assert c1["dma_wait"] == c0["dma_wait"] > 0
    assert c1["dma_start"] == c0["dma_start"] > 0
    assert c1["dot_general"] == c0["dot_general"] > 0
    assert c1["rem"] < c0["rem"], (c0["rem"], c1["rem"])


def test_dbanks_static_and_traced():
    """`dbanks` wraps at `NBD` for trace-time and traced starting tiles."""
    ex = ExecSpec(B=4, T=1, R=1, ctx=128, tiles=_tiles(), interpret=True)
    ms = ModelSpec(
        "dbanks",
        config=FakeGeometry(L=_L, H=_K),
        layer_type=LayerType("lt", (_WideTileModule(),)),
    )
    sp = vmem.build_scratch(ms, ex)[0].stream
    ring = stream.TwoRing(sp, {}, {})
    assert ring.dbanks(4, 4) == [4, 5, 0, 1]
    got = jax.jit(lambda g: jnp.stack(ring.dbanks(g, 6)))(jnp.int32(10))
    assert [int(v) for v in got] == [(10 + u) % 6 for u in range(6)]


def _ws_pre_layer(ring: stream.TwoRing, scr: dict[str, Any]) -> Any:
    g0 = ring.cursor
    views = ring.dtake_group([("w", g0 + u) for u in range(6)])
    return ring.dmatmul("w", scr["xb"], _N, ws_pre=views)


@pytest.mark.parametrize("dwait_hoist", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
def test_dmatmul_ws_pre_matches_dmatmul(dwait_hoist: bool, lazy: bool):
    """Tiles taken first (`dtake_group`) and passed as `ws_pre` give the same
    result as `dmatmul` taking them itself, and the same waits."""
    kw = dict(dense_acc_pair=True, dmm_unroll=8, dwait_hoist=dwait_hoist)
    out0, ref, call0, args = _run(_tiles(**kw), lazy=lazy)
    out1, _, call1, _ = _run(_tiles(**kw), _ws_pre_layer, lazy=lazy)
    np.testing.assert_allclose(out0, ref, atol=1e-2, rtol=1e-2)
    np.testing.assert_array_equal(out1, out0)
    c0, c1 = prim_counts(call0, *args), prim_counts(call1, *args)
    assert c1["dma_wait"] == c0["dma_wait"]
    assert c1["dma_start"] == c0["dma_start"]


def test_dmatmul_ws_pre_requires_unrolled_form():
    """`ws_pre` with a stepped (`U < n_t`) dense matmul is rejected at trace time."""
    with pytest.raises(AssertionError, match="ws_pre"):
        _run(_tiles(dense_acc_pair=True, dmm_unroll=0), _ws_pre_layer)


# ---------------------------------------------------------------------------
# hook= (two K-lane halves around a hook) and split= (column groups to hooks)
# ---------------------------------------------------------------------------
_HALF = _K // 2  # lanes per half; tiles 0..2 are half 0, 3..5 half 1 (t_k = 128)
_GARBAGE = 7.0


def _hook_layer(first: int) -> LayerFn:
    """Half `1 - first` of the input holds garbage until the hook restores it, and
    the hook overwrites half `first`, so the result is only right if the hook
    runs after the first half's tiles and before the other half's."""
    lo_f, lo_o = first * _HALF, (1 - first) * _HALF

    def layer(ring: stream.TwoRing, scr: dict[str, Any]) -> Any:
        xb, keep = scr["xb"], scr["xb2"]
        keep[...] = xb[...]
        xb[:, lo_o : lo_o + _HALF] = jnp.full((xb.shape[0], _HALF), _GARBAGE, BF16)

        def fn() -> None:
            xb[:, lo_o : lo_o + _HALF] = keep[:, lo_o : lo_o + _HALF]
            xb[:, lo_f : lo_f + _HALF] = jnp.full((xb.shape[0], _HALF), _GARBAGE, BF16)

        hook = stream.HalfSplitHook((0, 1, 2), 3, jnp.int32(first), fn)
        y = ring.dmatmul("w", xb, _N, hook=hook)
        xb[...] = keep[...]
        return y

    return layer


@pytest.mark.parametrize("first", [0, 1])
@pytest.mark.parametrize("lazy", [False, True])
def test_dmatmul_hook_runs_between_lane_halves(first: int, lazy: bool):
    kw = dict(dense_acc_pair=True, dmm_unroll=8)
    out0, ref, call0, args = _run(_tiles(**kw), lazy=lazy)
    out1, _, call1, _ = _run(_tiles(**kw), _hook_layer(first), lazy=lazy)
    np.testing.assert_allclose(out1, ref, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(out1, out0, atol=1e-5, rtol=1e-5)
    c0, c1 = prim_counts(call0, *args), prim_counts(call1, *args)
    assert c1["dma_wait"] == c0["dma_wait"]
    assert c1["dma_start"] == c0["dma_start"]


def test_dmatmul_hook_is_not_vacuous():
    """The garbage really reaches the matmul when the hook does not restore it."""

    def layer(ring: stream.TwoRing, scr: dict[str, Any]) -> Any:
        hook = stream.HalfSplitHook((0, 1, 2), 3, jnp.int32(0), lambda: None)
        scr["xb2"][...] = scr["xb"][...]
        scr["xb"][:, _HALF:] = jnp.full((scr["xb"].shape[0], _HALF), _GARBAGE, BF16)
        y = ring.dmatmul("w", scr["xb"], _N, hook=hook)
        scr["xb"][...] = scr["xb2"][...]
        return y

    out, ref, _, _ = _run(_tiles(dense_acc_pair=True, dmm_unroll=8), layer)
    assert not np.allclose(out, ref, atol=1e-2, rtol=1e-2)


def _split_layer(traced_c0: bool) -> LayerFn:
    """Columns [0, 384) in two chunks to hook a, [384, 640) to hook b (whose
    `c0` may be traced); the hooks and `pre` write `part`."""

    def layer(ring: stream.TwoRing, scr: dict[str, Any]) -> Any:
        part = scr["part"]

        def pre() -> None:
            part[:, _N:] = jnp.ones((part.shape[0], _K - _N), F32)

        def hook_a(vals: list[Any]) -> None:
            part[:, 0:256] = vals[0]
            part[:, 256:384] = vals[1]

        def hook_b(vals: list[Any]) -> None:
            part[:, 384:640] = vals[0]

        c0_b = jnp.int32(384) if traced_c0 else 384
        split = stream.ColumnSplit(
            pre, (((0, 256), (256, 128)), ((c0_b, 256),)), (hook_a, hook_b)
        )
        ring.dmatmul("w", scr["xb"], _N, split=split)
        return part[:, :_N] * part[:, _N : _N + 1]

    return layer


@pytest.mark.parametrize("traced_c0", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
def test_dmatmul_split_hands_column_groups_to_hooks(traced_c0: bool, lazy: bool):
    kw = dict(dense_acc_pair=True, dmm_unroll=8)
    out0, ref, call0, args = _run(_tiles(**kw), lazy=lazy)
    out1, _, call1, _ = _run(_tiles(**kw), _split_layer(traced_c0), lazy=lazy)
    np.testing.assert_allclose(out1, ref, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(out1, out0, atol=1e-5, rtol=1e-5)
    c0, c1 = prim_counts(call0, *args), prim_counts(call1, *args)
    assert c1["dma_wait"] == c0["dma_wait"]
    assert c1["dma_start"] == c0["dma_start"]


@pytest.mark.parametrize("form", ["hook", "split"])
def test_dmatmul_hook_and_split_require_unrolled_form(form: str):
    def layer(ring: stream.TwoRing, scr: dict[str, Any]) -> Any:
        if form == "hook":
            hook = stream.HalfSplitHook((0, 1, 2), 3, jnp.int32(0), lambda: None)
            return ring.dmatmul("w", scr["xb"], _N, hook=hook)
        split = stream.ColumnSplit(lambda: None, (((0, _N),),), (lambda v: None,))
        return ring.dmatmul("w", scr["xb"], _N, split=split)

    with pytest.raises(AssertionError, match=form):
        _run(_tiles(dense_acc_pair=True, dmm_unroll=0), layer)


# ---------------------------------------------------------------------------
# claim: a module takes its tiles itself
# ---------------------------------------------------------------------------
def _claim_layer(ring: stream.TwoRing, scr: dict[str, Any]) -> Any:
    """`dmatmul('w', ...)` rebuilt from `claim` + `dtake_group` + `give`."""
    g0, give = ring.claim("w")
    t_k = ring.sp.tk["w"]
    ws = ring.dtake_group([("w", g0 + u) for u in range(6)])
    xs = [scr["xb"][:, pl.ds(u * t_k, t_k)] for u in range(6)]
    acc = scr["acc"]
    for c0, cw in ((0, 512), (512, _N - 512)):
        p = ring.mm(xs[0], ws[0][:, c0 : c0 + cw])
        for u in range(1, 6):
            p = p + ring.mm(xs[u], ws[u][:, c0 : c0 + cw])
        acc[:, c0 : c0 + cw] = p
    for u in range(6):
        give(u)
    return acc[:, :_N]


@pytest.mark.parametrize("dwait_hoist", [False, True])
@pytest.mark.parametrize(
    "lazy,rel_fetch", [(False, False), (False, True), (True, True)]
)
def test_claim_matches_dmatmul(dwait_hoist: bool, lazy: bool, rel_fetch: bool):
    """`claim` consumes and refills the stream exactly as `dmatmul` does: same
    outputs bit for bit, same primitive histogram."""
    kw = dict(dense_acc_pair=True, dmm_unroll=8, dwait_hoist=dwait_hoist)
    kw["rel_fetch"] = rel_fetch
    out0, ref, call0, args = _run(_tiles(**kw), lazy=lazy)
    out1, _, call1, _ = _run(_tiles(**kw), _claim_layer, lazy=lazy)
    np.testing.assert_allclose(out0, ref, atol=1e-2, rtol=1e-2)
    np.testing.assert_array_equal(out1, out0)
    assert prim_counts(call1, *args) == prim_counts(call0, *args)


def test_claim_advances_cursor_and_checks_order():
    ex = ExecSpec(B=4, T=1, R=1, ctx=128, tiles=_tiles(), interpret=True)
    ms = ModelSpec(
        "claim_order",
        config=FakeGeometry(L=_L, H=_K),
        layer_type=LayerType("lt", (_WideTileModule(),)),
    )
    sp = vmem.build_scratch(ms, ex)[0].stream
    ring = stream.TwoRing(sp, {}, {})
    ring.begin_layer(1)
    g0, _ = ring.claim("w")
    assert g0 == sp.nt["w"] and ring.cursor == 2 * sp.nt["w"]
    with pytest.raises(AssertionError, match=r"claim\('w'\) called at layer tile"):
        ring.claim("w")


# ---------------------------------------------------------------------------
# segment(defer_flush=True): the lazy refill issued later
# ---------------------------------------------------------------------------
def _segment_layer(lazy: bool, defer: bool) -> LayerFn:
    """One segment around `dmatmul('w')`; with `defer`, its refill is issued
    after other work (a store to `part`)."""

    def layer(ring: stream.TwoRing, scr: dict[str, Any]) -> Any:
        with ring.segment(lazy=lazy, defer_flush=defer):
            y = ring.dmatmul("w", scr["xb"], _N)
        flush = ring.take_deferred_flush() if defer else (lambda: None)
        scr["part"][:, :_N] = y
        flush()
        return scr["part"][:, :_N]

    return layer


@pytest.mark.parametrize("lazy,rel_fetch", [(True, False), (True, True), (False, True)])
def test_deferred_flush_matches_segment_exit(lazy: bool, rel_fetch: bool):
    """The deferred closure issues exactly the refill the exit would have
    issued: same outputs bit for bit, same primitive histogram."""
    kw = dict(dense_acc_pair=True, dmm_unroll=8, rel_fetch=rel_fetch)
    out0, ref, call0, args = _run(
        _tiles(**kw), _segment_layer(lazy, False), own_segment=True
    )
    out1, _, call1, _ = _run(_tiles(**kw), _segment_layer(lazy, True), own_segment=True)
    np.testing.assert_allclose(out0, ref, atol=1e-2, rtol=1e-2)
    np.testing.assert_array_equal(out1, out0)
    assert prim_counts(call1, *args) == prim_counts(call0, *args)


def _host_ring() -> stream.TwoRing:
    ex = ExecSpec(B=4, T=1, R=1, ctx=128, tiles=_tiles(), interpret=True)
    ms = ModelSpec(
        "deferred_flush",
        config=FakeGeometry(L=_L, H=_K),
        layer_type=LayerType("lt", (_WideTileModule(),)),
    )
    ring = stream.TwoRing(vmem.build_scratch(ms, ex)[0].stream, {}, {})
    ring.begin_layer(0)
    return ring


def test_deferred_flush_must_be_taken_and_issued_once():
    ring = _host_ring()
    with ring.segment(lazy=True, defer_flush=True):
        pass
    with pytest.raises(AssertionError, match="never taken"):
        ring.assert_flushed()
    with (
        pytest.raises(AssertionError, match="deferred flush was taken"),
        ring.segment(lazy=True),
    ):
        pass

    ring = _host_ring()
    with ring.segment(lazy=True, defer_flush=True):
        pass
    flush = ring.take_deferred_flush()
    with pytest.raises(AssertionError, match="never issued"):
        ring.assert_flushed()
    with pytest.raises(AssertionError, match="never issued"):
        ring.begin_layer(1)
    flush()
    ring.assert_flushed()
    with pytest.raises(AssertionError, match="issued twice"):
        flush()
    with pytest.raises(AssertionError, match="without a preceding"):
        ring.take_deferred_flush()
