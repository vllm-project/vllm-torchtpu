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
"""Unit tests executing `TwoRing` and `AllReduceRing` in Pallas `interpret=True`."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tests.megakernel.megakernel_fakes import TINY_TILES, FakeGeometry
from vllm_torchtpu.megakernel import comm, kernel, planner, stream, vmem
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.ops import BF16, F32, I32, rmsnorm, round_bf16
from vllm_torchtpu.megakernel.spec import (
    LayerType,
    ModelSpec,
    Module,
    SmallField,
    SwitchSlot,
    WeightDecl,
)


class _TwoTileModule(Module):
    """Minimal module declaring two static streamed weights for `TwoRing` testing."""

    name = "two_tile"
    reduction = "allreduce"

    def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
        return [
            WeightDecl("w1", K=256, N=128, dtype=BF16, stream_ring="static"),
            WeightDecl("w2", K=128, N=256, dtype=BF16, stream_ring="static"),
        ]

    def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
        return {"acc": 256, "xb2": 128}


class _AltTileModule(Module):
    """Alternate branch module for testing hybrid `SwitchSlot` prefetch in `TwoRing`."""

    name = "alt_tile"
    reduction = "allreduce"

    def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
        return [
            WeightDecl("w_alt", K=256, N=256, dtype=BF16, stream_ring="static"),
        ]

    def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
        return {"acc": 256}


def test_tworing_stream_and_lazy_segment_in_pallas():
    """Execute `TwoRing` (`fetch`, `begin_layer`, `segment`, `dmatmul`) in Pallas."""
    # bank=64 KiB: w1 (256x128 bf16) and w2 (128x256 bf16) are 1 tile each
    tiles = dataclasses.replace(TINY_TILES, bank=64 << 10, nbd=4)
    ex = ExecSpec(B=4, T=1, R=1, ctx=128, tiles=tiles, interpret=True)
    g = FakeGeometry(L=2, H=256)
    ms = ModelSpec(
        "tworing_test", config=g, layer_type=LayerType("lt", slots=(_TwoTileModule(),))
    )
    cp, scratch = vmem.build_scratch(ms, ex)
    sp = cp.stream
    assert sp.nt["w1"] == 1 and sp.nt["w2"] == 1
    assert sp.T_TOTAL == 2 * (1 + 1)

    s_names = list(scratch.keys())

    def pallas_body(x_ref, w1_ref, w2_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        ring = stream.TwoRing(
            sp, {"w1": w1_ref, "w2": w2_ref}, scr, jnp.int32(0), interpret=True
        )
        for i in range(sp.NBD):
            ring.fetch(jnp.int32(i))

        scr["xb"][...] = x_ref[...].astype(BF16)

        def layer_fn(lyr, _):
            ring.begin_layer(lyr)
            with ring.segment(lazy=True):
                h = ring.dmatmul("w1", scr["xb"], 128)
                scr["xb2"][:, :128] = h.astype(BF16)
                y = ring.dmatmul("w2", scr["xb2"], 256)
            scr["xb"][...] = (scr["xb"][...].astype(F32) + y).astype(BF16)
            return 0

        lax.fori_loop(0, g.L, layer_fn, 0)
        out_ref[...] = scr["xb"][...].astype(F32)

    wsh = planner.weight_shapes(ms, ex, cp.plan)
    k0, k1, k2 = jax.random.split(jax.random.PRNGKey(0), 3)
    x = jax.random.normal(k0, (ex.MB, 256), dtype=BF16)
    w1_full = jax.random.normal(k1, (g.L, 256, 128), dtype=BF16) * 0.1
    w2_full = jax.random.normal(k2, (g.L, 128, 256), dtype=BF16) * 0.1
    w1_tiled = w1_full.reshape(wsh["w1"][0])
    w2_tiled = w2_full.reshape(wsh["w2"][0])

    call = pl.pallas_call(
        pallas_body,
        out_shape=jax.ShapeDtypeStruct((ex.MB, 256), F32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[scratch[k] for k in s_names],
        interpret=pltpu.InterpretParams(),
    )
    out = jax.jit(call)(x, w1_tiled, w2_tiled)

    # Pure-JAX reference
    ref = x.astype(F32)
    for lyr in range(g.L):
        h = (
            (ref.astype(BF16).astype(F32) @ w1_full[lyr].astype(F32))
            .astype(BF16)
            .astype(F32)
        )
        y = h @ w2_full[lyr].astype(F32)
        ref = (ref.astype(BF16).astype(F32) + y).astype(BF16).astype(F32)

    assert jnp.allclose(out, ref, atol=1e-2, rtol=1e-2)


def test_tworing_hybrid_switch_slot_in_pallas():
    """Verify `TwoRing.fetch` and `cond_branches` decode a 2-branch `SwitchSlot`."""
    tiles = dataclasses.replace(TINY_TILES, bank=64 << 10, nbd=4)
    ex = ExecSpec(B=4, T=1, R=1, ctx=128, tiles=tiles, interpret=True)
    g = FakeGeometry(L=2, att_every=2, H=256)
    sw = SwitchSlot(
        "mixer", branches=(_AltTileModule(), _TwoTileModule()), branch_counts=(1, 1)
    )
    ms = ModelSpec("hybrid_tworing", config=g, layer_type=LayerType("lt", slots=(sw,)))
    cp, scratch = vmem.build_scratch(ms, ex)
    sp = cp.stream
    s_names = list(scratch.keys())

    def pallas_body(x_ref, w_alt_ref, w1_ref, w2_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        W = {"w_alt": w_alt_ref, "w1": w1_ref, "w2": w2_ref}
        ring = stream.TwoRing(sp, W, scr, jnp.int32(0), interpret=True)
        for i in range(sp.NBD):
            ring.fetch(jnp.int32(i))

        scr["xb"][...] = x_ref[...].astype(BF16)

        def layer_fn(lyr, _):
            ring.begin_layer(lyr)

            def b0():
                with ring.segment(lazy=False):
                    return ring.dmatmul("w_alt", scr["xb"], 256)

            def b1():
                with ring.segment(lazy=False):
                    h = ring.dmatmul("w1", scr["xb"], 128)
                    scr["xb2"][:, :128] = h.astype(BF16)
                    return ring.dmatmul("w2", scr["xb2"], 256)

            y = ring.cond_branches(lyr % 2 == 0, b0, b1)
            scr["xb"][...] = (scr["xb"][...].astype(F32) + y).astype(BF16)
            return 0

        lax.fori_loop(0, g.L, layer_fn, 0)
        out_ref[...] = scr["xb"][...].astype(F32)

    wsh = planner.weight_shapes(ms, ex, cp.plan)
    k0, k1, k2, k3 = jax.random.split(jax.random.PRNGKey(7), 4)
    x = jax.random.normal(k0, (ex.MB, 256), dtype=BF16)
    w_alt_full = jax.random.normal(k1, (1, 256, 256), dtype=BF16) * 0.1
    w1_full = jax.random.normal(k2, (1, 256, 128), dtype=BF16) * 0.1
    w2_full = jax.random.normal(k3, (1, 128, 256), dtype=BF16) * 0.1

    call = pl.pallas_call(
        pallas_body,
        out_shape=jax.ShapeDtypeStruct((ex.MB, 256), F32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[scratch[k] for k in s_names],
        interpret=pltpu.InterpretParams(),
    )
    out = jax.jit(call)(
        x,
        w_alt_full.reshape(wsh["w_alt"][0]),
        w1_full.reshape(wsh["w1"][0]),
        w2_full.reshape(wsh["w2"][0]),
    )

    # Layer 0 runs _AltTileModule; Layer 1 runs _TwoTileModule
    ref = x.astype(F32)
    y0 = ref.astype(BF16).astype(F32) @ w_alt_full[0].astype(F32)
    ref = (ref + y0).astype(BF16).astype(F32)
    h1 = (ref @ w1_full[0].astype(F32)).astype(BF16).astype(F32)
    y1 = h1 @ w2_full[0].astype(F32)
    ref = (ref + y1).astype(BF16).astype(F32)

    assert jnp.allclose(out, ref, atol=1e-2, rtol=1e-2)


def test_tworing_consumption_order_and_nesting_guards():
    """Verify `TwoRing` rejects out-of-order `dmatmul` calls and nested `segment()`."""
    tiles = dataclasses.replace(TINY_TILES, bank=64 << 10, nbd=4)
    ex = ExecSpec(B=4, T=1, R=1, ctx=128, tiles=tiles, interpret=True)
    g = FakeGeometry(L=2, H=256)
    ms = ModelSpec(
        "guard_test", config=g, layer_type=LayerType("lt", slots=(_TwoTileModule(),))
    )
    cp, scratch = vmem.build_scratch(ms, ex)
    sp = cp.stream
    s_names = list(scratch.keys())

    def bad_order_body(x_ref, w1_ref, w2_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        ring = stream.TwoRing(
            sp, {"w1": w1_ref, "w2": w2_ref}, scr, jnp.int32(0), interpret=True
        )
        ring.begin_layer(jnp.int32(0))
        # Calling "w2" before "w1" violates declaration order ("w1", "w2")
        ring.dmatmul("w2", x_ref, 256)
        out_ref[...] = scr["acc"][:, :256]

    call_bad_order = pl.pallas_call(
        bad_order_body,
        out_shape=jax.ShapeDtypeStruct((ex.MB, 256), F32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[scratch[k] for k in s_names],
        interpret=pltpu.InterpretParams(),
    )
    wsh = planner.weight_shapes(ms, ex, cp.plan)
    x = jnp.zeros((ex.MB, 256), dtype=BF16)
    w1 = jnp.zeros(wsh["w1"][0], dtype=BF16)
    w2 = jnp.zeros(wsh["w2"][0], dtype=BF16)
    with pytest.raises(AssertionError, match="Stream consumption order mismatch"):
        jax.jit(call_bad_order)(x, w1, w2)

    # Nested segment() raises AssertionError
    ring_dummy = stream.TwoRing(sp, {}, {}, jnp.int32(0), interpret=True)
    with (
        pytest.raises(AssertionError, match="Nested stream.segment"),
        ring_dummy.segment(lazy=False),
        ring_dummy.segment(lazy=False),
    ):
        pass


@pytest.mark.parametrize(
    "R,hier,rsag",
    [
        (2, 0, False),
        (2, 0, True),
        (8, 0, True),
        (8, 1, False),
        (8, 2, False),
        (8, 3, False),
    ],
)
def test_allreduce_ring_modes_in_pallas(R: int, hier: int, rsag: bool):
    """Execute `AllReduceRing` across `R` ranks and 2 layers in Pallas."""
    devs = jax.devices("cpu")
    assert len(devs) >= R, (
        f"Expected >= {R} CPU devices (check XLA_FLAGS), found {len(devs)}"
    )

    tiles = dataclasses.replace(TINY_TILES, hier=hier, rsag=rsag, war=1, mb8=2)
    ex = ExecSpec(B=4, T=1, R=R, ctx=128, tiles=tiles, interpret=True)
    g = FakeGeometry(L=2, H=1024)
    ms = ModelSpec(
        "ar_test", config=g, layer_type=LayerType("lt", slots=(_TwoTileModule(),))
    )
    cp, scratch = vmem.build_scratch(ms, ex)
    s_names = list(scratch.keys())

    def pallas_body(x_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        me = lax.axis_index("tp").astype(I32)
        ar = comm.AllReduceRing(cp.comm, scr, me)

        if R > 1:
            barrier = pltpu.get_barrier_semaphore()
            for off in range(1, R):
                pl.semaphore_signal(
                    barrier,
                    1,
                    device_id=(me ^ off,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(barrier, R - 1)

        scr["res"][...] = jnp.zeros((ex.MB, g.H), F32)

        def layer_fn(lyr, _):
            first = lyr == 0
            s0 = (lyr + 1).astype(F32)
            s1 = (lyr + 3).astype(F32)
            tot0 = ar.allreduce(0, (x_ref[...] * s0).astype(BF16).astype(F32), first)
            ar.add_residual(tot0)
            tot1 = ar.allreduce(1, (x_ref[...] * s1).astype(BF16).astype(F32), first)
            ar.add_residual(tot1)
            return 0

        lax.fori_loop(0, 2, layer_fn, 0)
        ar.drain_sends()
        out_ref[...] = scr["res"][...]

    call = pl.pallas_call(
        pallas_body,
        out_shape=jax.ShapeDtypeStruct((ex.MB, g.H), F32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[scratch[k] for k in s_names],
        compiler_params=pltpu.CompilerParams(collective_id=0),
        interpret=pltpu.InterpretParams(detect_races=True),
    )

    mesh = Mesh(devs[:R], ("tp",))
    step = jax.jit(
        jax.shard_map(
            lambda x: call(x[0])[None],
            mesh=mesh,
            in_specs=(P("tp"),),
            out_specs=P("tp"),
            check_vma=False,
        )
    )

    x_all = jax.random.normal(
        jax.random.PRNGKey(42), (R, ex.MB, g.H), dtype=BF16
    ).astype(F32)
    out_all = jax.block_until_ready(step(x_all))

    expected = jnp.zeros((ex.B, g.H), dtype=F32)
    for lyr in range(2):
        for mult in (lyr + 1, lyr + 3):
            term = jnp.sum(
                (x_all[:, : ex.B] * float(mult)).astype(BF16).astype(F32), axis=0
            )
            expected = round_bf16(expected + round_bf16(term))
    for r in range(R):
        assert jnp.allclose(out_all[r, : ex.B], expected, atol=4e-1, rtol=2e-2)
    assert ipc.races is not None and not ipc.races.races_found


class _SingleARLinearModule(Module):
    """1-AR-per-layer module exercised through `kernel.make_step`."""

    name = "single_ar_linear"
    reduction = "allreduce"
    input_norm = "in_norm"

    def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
        return [
            WeightDecl("w1", K=g.H, N=128, dtype=BF16, stream_ring="static"),
            WeightDecl("w2", K=128, N=g.H, dtype=BF16, stream_ring="static"),
        ]

    def smalls(self, g: FakeGeometry, ex: ExecSpec) -> list[SmallField]:
        return [SmallField("in_norm", g.H)]

    def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
        return {"acc": g.H, "xb2": 128}

    def emit(self, ctx: Any, lyr: Any, slot: Any) -> Any:
        del lyr, slot
        with ctx.stream.segment(lazy=True):
            h = ctx.stream.dmatmul("w1", ctx.scr["xb"], 128)
            ctx.scr["xb2"][:, :128] = h.astype(BF16)
            return ctx.stream.dmatmul("w2", ctx.scr["xb2"], ctx.g.H)


def test_allreduce_ring_single_slot_layer_parity_alternation():
    """Drive 1-AR-per-layer alternation through `kernel.make_step` with odd `L=3`."""
    R = 2
    devs = jax.devices("cpu")
    assert len(devs) >= R
    tiles = dataclasses.replace(
        TINY_TILES, bank=64 << 10, nbd=4, hier=0, rsag=False, war=1, mb8=2
    )
    ex = ExecSpec(
        B=4,
        T=1,
        R=R,
        ctx=128,
        tiles=tiles,
        interpret=pltpu.InterpretParams(detect_races=True),
    )
    g = FakeGeometry(L=3, H=256)
    ms = ModelSpec(
        "ar_1slot",
        config=g,
        layer_type=LayerType("lt", slots=(_SingleARLinearModule(),)),
    )
    mesh = Mesh(devs[:R], ("tp",))
    step, info = kernel.make_step(mesh, ms, ex)
    assert info["plan"].slot_ar_index == (0,)

    wsh = info["wshape"]
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(99), 5)
    pos_pad = jnp.pad(
        jnp.arange(ex.B, dtype=I32), (0, ex.MB - ex.B), constant_values=-1
    )
    pos = kernel.fold(jnp.broadcast_to(pos_pad[None, None, :], (R, 1, ex.MB)))
    hid_0 = jax.random.normal(k0, (ex.M, g.H), dtype=BF16)
    hid = kernel.fold(jnp.broadcast_to(hid_0[None], (R, ex.M, g.H)))
    w1_raw = jax.random.normal(k1, (R, g.L, g.H, 128), dtype=BF16) * 0.1
    w2_raw = jax.random.normal(k2, (R, g.L, 128, g.H), dtype=BF16) * 0.1
    sm_0 = jax.random.normal(k3, (g.L, 1, info["plan"].smalls_width), dtype=F32) * 0.1
    fn_0 = jax.random.normal(k4, (1, g.H), dtype=F32) * 0.1
    weights = {
        "w1": w1_raw.reshape((R, *wsh["w1"][0])),
        "w2": w2_raw.reshape((R, *wsh["w2"][0])),
        "smalls": jnp.broadcast_to(sm_0[None], (R, *wsh["smalls"][0])),
        "final_norm": jnp.broadcast_to(fn_0[None], (R, *wsh["final_norm"][0])),
    }
    out_h, _ = jax.block_until_ready(step(pos, hid, weights, {}))
    assert ipc.races is not None and not ipc.races.races_found
    out_u = kernel.unfold(out_h, R)[:, : ex.B]

    res = hid_0[: ex.B].astype(F32)
    for lyr in range(g.L):
        nx = rmsnorm(
            res, sm_0[lyr, 0, : g.H], g.eps, zero_centered=bool(g.zero_centered_norm)
        ).astype(BF16)
        parts = []
        for r in range(R):
            h = (nx.astype(F32) @ w1_raw[r, lyr].astype(F32)).astype(BF16).astype(F32)
            parts.append((h @ w2_raw[r, lyr].astype(F32)).astype(BF16).astype(F32))
        res = round_bf16(res + round_bf16(parts[0] + parts[1]))
    ref = rmsnorm(res, fn_0[0], g.eps, zero_centered=bool(g.zero_centered_norm)).astype(
        BF16
    )
    for r in range(R):
        assert jnp.allclose(out_u[r], ref, atol=2e-2, rtol=2e-2)


def test_allreduce_ring_detect_races_catches_slot_reuse():
    """Negative test: reusing AR slot 0 on every layer (`L=2, R=2`) triggers a race."""
    R = 2
    devs = jax.devices("cpu")
    assert len(devs) >= R
    tiles = dataclasses.replace(TINY_TILES, hier=0, rsag=False, war=1, mb8=2)
    ex = ExecSpec(B=4, T=1, R=R, ctx=128, tiles=tiles, interpret=True)
    g = FakeGeometry(L=2, H=1024)
    ms = ModelSpec(
        "ar_race_neg", config=g, layer_type=LayerType("lt", slots=(_TwoTileModule(),))
    )
    cp, scratch = vmem.build_scratch(ms, ex)
    s_names = list(scratch.keys())

    def racy_body(x_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        me = lax.axis_index("tp").astype(I32)
        ar = comm.AllReduceRing(cp.comm, scr, me)
        barrier = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(
            barrier, 1, device_id=(me ^ 1,), device_id_type=pl.DeviceIdType.MESH
        )
        pl.semaphore_wait(barrier, 1)
        scr["res"][...] = jnp.zeros((ex.MB, g.H), F32)

        def layer_fn(lyr, _):
            # Buggy: always uses slot 0 instead of alternating `lyr & 1`
            val = (x_ref[...] * (lyr + 1).astype(F32)).astype(BF16).astype(F32)
            tot = ar.allreduce(0, val, lyr == 0)
            ar.add_residual(tot)
            return 0

        lax.fori_loop(0, g.L, layer_fn, 0)
        ar.drain_sends()
        out_ref[...] = scr["res"][...]

    call = pl.pallas_call(
        racy_body,
        out_shape=jax.ShapeDtypeStruct((ex.MB, g.H), F32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[scratch[k] for k in s_names],
        compiler_params=pltpu.CompilerParams(collective_id=0),
        interpret=pltpu.InterpretParams(detect_races=True),
    )
    mesh = Mesh(devs[:R], ("tp",))
    step = jax.jit(
        jax.shard_map(
            lambda x: call(x[0])[None],
            mesh=mesh,
            in_specs=(P("tp"),),
            out_specs=P("tp"),
            check_vma=False,
        )
    )
    x_all = jax.random.normal(
        jax.random.PRNGKey(11), (R, ex.MB, g.H), dtype=BF16
    ).astype(F32)
    _ = jax.block_until_ready(step(x_all))
    assert ipc.races is not None and ipc.races.races_found


@pytest.mark.parametrize(
    "R,hier,rsag,war,H,match",
    [
        (8, 0, True, 1, 512, "multiple of 128"),
        (8, 0, True, 0, 512, "multiple of 128"),
        (8, -1, False, 1, 1024, "unsupported all-reduce mode"),
        (8, 99, False, 1, 1024, "unsupported all-reduce mode"),
        (4, 1, False, 1, 1024, "unsupported all-reduce mode"),
        (4, 2, False, 1, 1024, "unsupported all-reduce mode"),
        (6, 3, False, 1, 1024, "unsupported all-reduce mode"),
        (8, 2, False, 1, 128, "hier=2 requires HC2=64 divisible by 128"),
    ],
)
def test_comm_params_rejects_invalid_config(
    R: int, hier: int, rsag: bool, war: int, H: int, match: str
):
    """Verify comm_params raises MegakernelConfigError on invalid configs."""
    tiles = dataclasses.replace(TINY_TILES, hier=hier, rsag=rsag, war=war, mb8=2)
    ex = ExecSpec(B=4, T=1, R=R, ctx=128, tiles=tiles, interpret=True)
    ms = ModelSpec(
        "bad_comm",
        config=FakeGeometry(L=2, H=H),
        layer_type=LayerType("lt", slots=(_TwoTileModule(),)),
    )
    with pytest.raises(MegakernelConfigError, match=match):
        vmem.build_scratch(ms, ex)


def test_borrow_epool_rejected_inside_expert_stream():
    """Verify `TwoRing.borrow_epool` is rejected while `expert_stream` is active."""
    ring = object.__new__(stream.TwoRing)
    ring._ering_busy = False
    ring.scr = {"epool": "EPOOL"}
    assert ring.borrow_epool() == "EPOOL"
    with ring.expert_stream():
        with pytest.raises(AssertionError, match="dynamic expert streaming"):
            ring.borrow_epool()
        with (
            pytest.raises(AssertionError, match="not re-entrant"),
            ring.expert_stream(),
        ):
            pass
    assert ring.borrow_epool() == "EPOOL"
