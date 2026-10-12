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
"""Optional layer-schedule fields of the generic builder (`KernelContext`).

A two-slot model (a linear mixer and a linear FFN, each followed by an
all-reduce) runs through `kernel.make_step` on an 8-rank CPU interpret mesh
with each schedule option on and off. Options that only move work are
bit-identical to the default schedule; every run is also checked against a
host reference.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from tests.megakernel.core.megakernel_fakes import TINY_TILES, FakeGeometry
from tests.megakernel.core.test_stream_schedule import prim_counts
from vllm_torchtpu.megakernel.core import kernel
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.core.ops import BF16, F32, I32, rmsnorm, round_bf16
from vllm_torchtpu.megakernel.core.spec import (
    LayerType,
    ModelSpec,
    Module,
    SmallField,
    WeightDecl,
)

_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"
_CHILD_TIMEOUT_S = 1500
_R, _H, _NI, _L, _B = 8, 1024, 512, 3, 4


def _spawn_with_cpu_devices(R: int, tag: str) -> None:
    flags = [
        f
        for f in os.environ.get("XLA_FLAGS", "").split()
        if not f.startswith(_DEVICE_COUNT_FLAG)
    ]
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        MALLOC_ARENA_MAX="2",
        XLA_FLAGS=" ".join([*flags, f"{_DEVICE_COUNT_FLAG}={R}"]),
    )
    out = subprocess.run(
        [sys.executable or "python3", __file__, tag],
        env=env,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
    )
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-5000:]


class _LinMixer(Module):
    """`y = (norm(x) @ w_in) @ w_out * out_s` (rows-local, then all-reduced)."""

    name = "lin_mixer"
    reduction = "allreduce"
    input_norm = "in_norm"

    def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
        return [
            WeightDecl("w_in", K=g.H, N=_NI, dtype=BF16, stream_ring="static"),
            WeightDecl("w_out", K=_NI, N=g.H, dtype=BF16, stream_ring="static"),
        ]

    def smalls(self, g: FakeGeometry, ex: ExecSpec) -> list[SmallField]:
        return [SmallField("in_norm", g.H), SmallField("out_s", g.H)]

    def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
        return {"acc": g.H, "xb2": _NI}

    def emit(self, ctx: Any, lyr: Any, slot: Any) -> Any:
        del lyr
        scr = ctx.scr
        with ctx.stream.segment(lazy=True):
            y = ctx.stream.dmatmul("w_in", scr["xb"], _NI, hook=ctx.layer_hook)
            scr["xb2"][:, :_NI] = round_bf16(ctx.arp_scale(y)).astype(BF16)
            if ctx.ar0 is not None:
                split = ctx.ar0("out_s")
                ctx.stream.dmatmul("w_out", scr["xb2"], ctx.g.H, split=split)
                return None
            yo = ctx.stream.dmatmul("w_out", scr["xb2"], ctx.g.H)
        return yo * ctx.sm(slot, "out_s")


class _HookMixer(_LinMixer):
    """`_LinMixer` that declares the ARPIPE hook (`Module.layer_schedule`)."""

    layer_schedule = frozenset({"layer_hook"})


class _Ar0Mixer(_LinMixer):
    """`_LinMixer` that declares the ARPIPE hook and AR0S."""

    layer_schedule = frozenset({"layer_hook", "ar0"})


class _LinFFN(Module):
    """`y = norm(x) @ w_f`; option `defer`: the lazy refill is left to the
    all-reduce (`deferred_fill`)."""

    name = "lin_ffn"
    reduction = "allreduce"
    input_norm = "post_norm"
    OPTIONS = {"defer": False}

    def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
        return [WeightDecl("w_f", K=g.H, N=g.H, dtype=BF16, stream_ring="static")]

    def smalls(self, g: FakeGeometry, ex: ExecSpec) -> list[SmallField]:
        return [SmallField("post_norm", g.H)]

    def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
        return {"acc": g.H}

    def emit(self, ctx: Any, lyr: Any, slot: Any) -> Any:
        del lyr, slot
        defer = bool(self.options["defer"])
        with ctx.stream.segment(lazy=True, defer_flush=defer):
            y = ctx.stream.dmatmul("w_f", ctx.scr["xb"], ctx.g.H)
        if defer:
            ctx.deferred_fill = ctx.stream.take_deferred_flush()
        return y


def _spec(mixer: Module, ffn: Module) -> ModelSpec:
    g = FakeGeometry(L=_L, H=_H)
    return ModelSpec(
        "layer_schedule", config=g, layer_type=LayerType("lt", (mixer, ffn))
    )


def _tiles(**kw: Any) -> Tiles:
    return dataclasses.replace(
        TINY_TILES,
        bank=256 << 10,
        nbd=8,
        war=1,
        mb8=2,
        dense_acc_pair=True,
        dmm_unroll=8,
        **kw,
    )


def _inputs(info: dict[str, Any], g: FakeGeometry, ex: ExecSpec, seed: int = 3):
    wsh = info["wshape"]
    ks = jax.random.split(jax.random.PRNGKey(seed), 6)
    R = ex.R
    pos_pad = jnp.pad(
        jnp.arange(ex.B, dtype=I32), (0, ex.MB - ex.B), constant_values=-1
    )
    pos = kernel.fold(jnp.broadcast_to(pos_pad[None, None, :], (R, 1, ex.MB)))
    hid0 = jax.random.normal(ks[0], (ex.M, g.H), dtype=BF16)
    hid = kernel.fold(jnp.broadcast_to(hid0[None], (R, ex.M, g.H)))
    raw = {
        "w_in": jax.random.normal(ks[1], (R, g.L, g.H, _NI), dtype=BF16) * 0.05,
        "w_out": jax.random.normal(ks[2], (R, g.L, _NI, g.H), dtype=BF16) * 0.05,
        "w_f": jax.random.normal(ks[3], (R, g.L, g.H, g.H), dtype=BF16) * 0.02,
    }
    sm0 = jax.random.normal(ks[4], (g.L, 1, info["plan"].smalls_width), dtype=F32)
    sm0 = sm0 * 0.1
    fn0 = jax.random.normal(ks[5], (1, g.H), dtype=F32) * 0.1
    weights = {k: v.reshape((R, *wsh[k][0])) for k, v in raw.items()}
    weights["smalls"] = jnp.broadcast_to(sm0[None], (R, *wsh["smalls"][0]))
    weights["final_norm"] = jnp.broadcast_to(fn0[None], (R, *wsh["final_norm"][0]))
    return (pos, hid, weights), (hid0, raw, sm0, fn0)


def _reference(spec: ModelSpec, info: dict[str, Any], host: Any) -> np.ndarray:
    hid0, raw, sm0, fn0 = host
    g = spec.config
    lay = info["plan"].smalls_layout
    zc = bool(g.zero_centered_norm)

    def field(lyr: int, name: str) -> Any:
        o, w = lay[name]
        return sm0[lyr, 0, o : o + w]

    res = hid0[:_B].astype(F32)
    for lyr in range(g.L):
        nx = rmsnorm(res, field(lyr, "in_norm"), g.eps, zero_centered=zc)
        tot = 0.0
        for r in range(_R):
            h = round_bf16(
                nx.astype(BF16).astype(F32) @ raw["w_in"][r, lyr].astype(F32)
            )
            part = (h @ raw["w_out"][r, lyr].astype(F32)) * field(lyr, "out_s")
            tot = tot + round_bf16(part)
        res = round_bf16(res + tot)
        nx = rmsnorm(res, field(lyr, "post_norm"), g.eps, zero_centered=zc)
        tot = 0.0
        for r in range(_R):
            part = nx.astype(BF16).astype(F32) @ raw["w_f"][r, lyr].astype(F32)
            tot = tot + round_bf16(part)
        res = round_bf16(res + tot)
    out = rmsnorm(res, fn0[0], g.eps, zero_centered=zc).astype(BF16)
    return np.asarray(out.astype(F32))


def _run(spec: ModelSpec, tiles: Tiles, *, counts: bool = False):
    ex = ExecSpec(B=_B, T=1, R=_R, ctx=128, tiles=tiles, interpret=True)
    mesh = Mesh(jax.devices("cpu")[:_R], ("tp",))
    step, info = kernel.make_step(mesh, spec, ex)
    args, host = _inputs(info, spec.config, info["exec"])
    if counts:
        return prim_counts(step, *args, {}), info
    out, _ = jax.block_until_ready(step(*args, {}))
    out = np.asarray(kernel.unfold(out, _R)[:, :_B].astype(F32))
    for r in range(1, _R):
        np.testing.assert_array_equal(out[r], out[0])
    ref = _reference(spec, info, host)
    np.testing.assert_allclose(out[0], ref, atol=3e-2, rtol=3e-2)
    return out[0], info


# ---------------------------------------------------------------------------
# deferred_fill: a module's lazy refill issued inside its slot's all-reduce
# ---------------------------------------------------------------------------
def _check_deferred_fill() -> None:
    for hier in (1, 2):
        tiles = _tiles(hier=hier)
        base, _ = _run(_spec(_LinMixer(), _LinFFN()), tiles)
        defer_spec = _spec(_LinMixer(), _LinFFN(defer=True))
        out, _ = _run(defer_spec, tiles)
        np.testing.assert_array_equal(out, base)
        c0, _ = _run(_spec(_LinMixer(), _LinFFN()), tiles, counts=True)
        c1, _ = _run(defer_spec, tiles, counts=True)
        assert c1 == c0, (hier, c0 - c1, c1 - c0)


def test_deferred_fill_issued_inside_the_all_reduce():
    """`ctx.deferred_fill`: same outputs and the same primitives as the
    segment-exit refill, on hier=1 and hier=2."""
    if len(jax.devices("cpu")) < _R:
        _spawn_with_cpu_devices(_R, "deferred_fill")
        return
    _check_deferred_fill()


# ---------------------------------------------------------------------------
# ARPIPE: the FFN all-reduce's last round deferred into the next input projection
# ---------------------------------------------------------------------------
_ARP_CASES = (
    dict(hier=2, arpipe=True),
    dict(hier=2, arpipe=True, hx=True),
    dict(hier=1, arpipe=True, arpipe_h1=True),
)


def _check_arpipe() -> None:
    for kw in _ARP_CASES:
        base_t = _tiles(
            **{k: v for k, v in kw.items() if k not in ("arpipe", "arpipe_h1")}
        )
        base, _ = _run(_spec(_HookMixer(), _LinFFN()), base_t)
        out, info = _run(_spec(_HookMixer(), _LinFFN()), _tiles(**kw))
        sched = info["schedule"]
        assert sched.arpipe and sched.hook_kind == "w_in", (kw, sched)
        assert sched.hook_tiles == (0, 1) and sched.hook_hpt == 2, sched
        assert (kw["hier"] == 1) == info["core_plan"].comm.ARP_H1
        assert "arp" in info["vmem_shapes"]
        # the deferred input norm changes rounding (unscaled input, scaled
        # after the projection): close to the reference, not bit-identical
        assert not np.array_equal(out, base), kw
        # with the FFN's refill also deferred into its (now deferred) all-reduce
        out2, _ = _run(_spec(_HookMixer(), _LinFFN(defer=True)), _tiles(**kw))
        np.testing.assert_array_equal(out2, out)


def _check_arpipe_needs_module_support() -> None:
    """A first slot that does not declare the hook keeps the default schedule
    (bit-identical), as does `arpipe` on `hier=1` without `arpipe_h1`."""
    base, _ = _run(_spec(_LinMixer(), _LinFFN()), _tiles(hier=2))
    out, info = _run(_spec(_LinMixer(), _LinFFN()), _tiles(hier=2, arpipe=True))
    assert not info["schedule"].arpipe and "arp" not in info["vmem_shapes"]
    np.testing.assert_array_equal(out, base)
    base1, _ = _run(_spec(_HookMixer(), _LinFFN()), _tiles(hier=1))
    out1, info1 = _run(_spec(_HookMixer(), _LinFFN()), _tiles(hier=1, arpipe=True))
    assert not info1["schedule"].arpipe
    np.testing.assert_array_equal(out1, base1)


def test_arpipe_defers_the_last_all_reduce_into_the_input_projection():
    """ARPIPE on hier=2 (with and without hx) and hier=1: engaged, close to
    the host reference, and unchanged by also deferring the FFN refill."""
    if len(jax.devices("cpu")) < _R:
        _spawn_with_cpu_devices(_R, "arpipe")
        return
    _check_arpipe()


def test_arpipe_needs_module_support_and_arpipe_h1():
    if len(jax.devices("cpu")) < _R:
        _spawn_with_cpu_devices(_R, "arpipe_gate")
        return
    _check_arpipe_needs_module_support()


# ---------------------------------------------------------------------------
# AR0S: the mixer all-reduce's round 0 sent from inside its output projection
# ---------------------------------------------------------------------------
_AR0_CASES = (
    dict(hier=2, ar0s=True),
    dict(hier=2, ar0s=True, hx=True),
    dict(hier=2, ar0s=True, arpipe=True),
    dict(hier=1, ar0s=True, arpipe=True, arpipe_h1=True),
)


def _check_ar0s() -> None:
    """AR0S moves the payload packing and sends into the output projection:
    bit-identical to the same schedule without it."""
    for kw in _AR0_CASES:
        out, info = _run(_spec(_Ar0Mixer(), _LinFFN()), _tiles(**kw))
        assert info["schedule"].ar0s and info["core_plan"].comm.AR0S, kw
        assert info["schedule"].arpipe == bool(kw.get("arpipe")), kw
        no_ar0 = {k: v for k, v in kw.items() if k != "ar0s"}
        base, binfo = _run(_spec(_Ar0Mixer(), _LinFFN()), _tiles(**no_ar0))
        assert not binfo["schedule"].ar0s
        np.testing.assert_array_equal(out, base)
    # hier=1 without ARPIPE, or a mixer that does not declare it: off
    _, info = _run(_spec(_Ar0Mixer(), _LinFFN()), _tiles(hier=1, ar0s=True))
    assert not info["schedule"].ar0s
    _, info = _run(_spec(_HookMixer(), _LinFFN()), _tiles(hier=2, ar0s=True))
    assert not info["schedule"].ar0s


def test_ar0s_sends_round0_from_the_output_projection():
    if len(jax.devices("cpu")) < _R:
        _spawn_with_cpu_devices(_R, "ar0s")
        return
    _check_ar0s()


_CHECKS = {
    "ar0s": _check_ar0s,
    "arpipe": _check_arpipe,
    "arpipe_gate": _check_arpipe_needs_module_support,
    "deferred_fill": _check_deferred_fill,
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        _CHECKS[sys.argv[1]]()
