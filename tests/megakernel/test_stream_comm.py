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
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tests.megakernel.megakernel_fakes import TINY_TILES, FakeGeometry
from vllm_torchtpu.megakernel import comm, planner, stream, vmem
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.ops import BF16, F32, I32, rnd
from vllm_torchtpu.megakernel.spec import (
    LayerType,
    ModelSpec,
    Module,
    SwitchSlot,
    WeightDecl,
)

_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"
_CHILD_TIMEOUT_S = 540


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


class _SecondTileModule(Module):
    """Second allreduce module so `_execute_allreduce_ring` has 2 allreduce slots."""

    name = "second_tile"
    reduction = "allreduce"


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


def _execute_allreduce_ring(
    R: int,
    *,
    hier: int,
    rsag: bool,
    hier_group: int = 8,
    war: int = 1,
    H: int = 1024,
    B: int = 4,
    seed: int = 42,
    heavy_tail: bool = False,
) -> tuple[jax.Array, jax.Array, ExecSpec, vmem.CorePlan, dict[str, object]]:
    """Run a 2-layer, 2-slot `AllReduceRing` kernel across `R` CPU devices."""
    devs = jax.devices("cpu")
    assert len(devs) >= R, (
        f"Expected >= {R} CPU devices (check XLA_FLAGS), found {len(devs)}"
    )

    tiles = dataclasses.replace(
        TINY_TILES, hier=hier, hier_group=hier_group, rsag=rsag, war=war, mb8=2
    )
    ex = ExecSpec(B=B, T=1, R=R, ctx=128, tiles=tiles, interpret=True)
    g = FakeGeometry(L=2, H=H)
    ms = ModelSpec(
        "ar_test",
        config=g,
        layer_type=LayerType("lt", slots=(_TwoTileModule(), _SecondTileModule())),
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
            tot0 = ar.allreduce(0, x_ref[...], first)
            ar.add_residual(tot0)
            tot1 = ar.allreduce(1, x_ref[...], first)
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
        interpret=pltpu.InterpretParams(),
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

    k_main, k_tail = jax.random.split(jax.random.PRNGKey(seed))
    x_raw = jax.random.normal(k_main, (R, ex.MB, g.H), dtype=BF16).astype(F32)
    if heavy_tail:
        spikes = jax.random.bernoulli(k_tail, 0.05, (R, ex.MB, g.H)).astype(F32) * 8.0
        x_raw = (x_raw + spikes).astype(BF16).astype(F32)
        if ex.B < ex.MB:
            x_raw = x_raw.at[:, ex.B :].set(F32(77.0))
    x_all = x_raw
    out_all = step(x_all)
    return x_all, out_all, ex, cp, scratch


def _expected_residual(x_all: jax.Array, B: int) -> jax.Array:
    expected_one = jnp.sum(x_all[:, :B].astype(BF16).astype(F32), axis=0)
    expected = jnp.zeros_like(expected_one)
    for _ in range(4):
        expected = rnd(expected + rnd(expected_one))
    return expected


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
    x_all, out_all, ex, _, _ = _execute_allreduce_ring(R, hier=hier, rsag=rsag)
    expected = _expected_residual(x_all, ex.B)
    for r in range(R):
        assert jnp.allclose(out_all[r, : ex.B], expected, atol=2e-1, rtol=2e-2)


def test_allreduce_ring_hier4_single_group_matches_rsag_bitwise():
    """Verify 1-group `hier=4` (`R=8, hier_group=8`) matches `rsag=True` bitwise."""
    _, out_rsag, ex, _, _ = _execute_allreduce_ring(8, hier=0, rsag=True)
    _, out_hier4, _, cp, scratch = _execute_allreduce_ring(
        8, hier=4, rsag=False, hier_group=8
    )
    assert cp.comm.HIER == 4
    assert cp.comm.HIER_GROUP == 8
    assert cp.comm.N_HOSTS == 1
    assert "ih_recv" not in scratch
    assert "s2_send" not in scratch
    assert jnp.array_equal(out_hier4[:, : ex.B], out_rsag[:, : ex.B])


def _check_hier4_32_ranks_in_process() -> None:
    x_all, out_all, ex, cp, scratch = _execute_allreduce_ring(
        32, hier=4, rsag=False, hier_group=8
    )
    assert cp.comm.HIER == 4
    assert cp.comm.HIER_GROUP == 8
    assert cp.comm.N_HOSTS == 4
    assert "ih_recv" in scratch and "s2_send" in scratch
    expected = _expected_residual(x_all, ex.B)
    host0_only = _expected_residual(x_all[:8], ex.B)
    for r in range(1, 32):
        assert jnp.array_equal(out_all[r, : ex.B], out_all[0, : ex.B])
    assert jnp.allclose(out_all[0, : ex.B], expected, atol=2e-1, rtol=2e-2)
    assert not jnp.allclose(out_all[0, : ex.B], host0_only, atol=2e-1, rtol=2e-2)


def test_allreduce_ring_hier4_multi_group_and_32_ranks():
    """Verify multi-group `hier=4` at `R=8, hier_group=4` and `R=32, hier_group=8`."""
    x_all, out_all, ex, cp, scratch = _execute_allreduce_ring(
        8, hier=4, rsag=False, hier_group=4
    )
    assert cp.comm.HIER == 4
    assert cp.comm.HIER_GROUP == 4
    assert cp.comm.N_HOSTS == 2
    assert "ih_recv" in scratch and "s2_send" in scratch
    expected = _expected_residual(x_all, ex.B)
    group0_only = _expected_residual(x_all[:4], ex.B)
    for r in range(1, 8):
        assert jnp.array_equal(out_all[r, : ex.B], out_all[0, : ex.B])
    assert jnp.allclose(out_all[0, : ex.B], expected, atol=2e-1, rtol=2e-2)
    assert not jnp.allclose(out_all[0, : ex.B], group0_only, atol=2e-1, rtol=2e-2)

    flags = [
        f
        for f in os.environ.get("XLA_FLAGS", "").split()
        if not f.startswith(_DEVICE_COUNT_FLAG)
    ]
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        MALLOC_ARENA_MAX="2",
        PYTHONPATH=os.pathsep.join(
            p for p in (repo_root, os.environ.get("PYTHONPATH", "")) if p
        ),
        XLA_FLAGS=" ".join([*flags, f"{_DEVICE_COUNT_FLAG}=32"]),
    )
    out = subprocess.run(
        [sys.executable or "python3", __file__, "hier4_r32"],
        env=env,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
    )
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-4000:]


def test_allreduce_ring_hier4_two_groups_with_folded_payload():
    """`hier=4, hier_group=4` at `R=8` on the folded `[PR, PC]` payload."""
    x_all, out_all, ex, cp, _ = _execute_allreduce_ring(
        8, hier=4, rsag=False, hier_group=4, war=0, H=2048
    )
    assert not cp.comm.WAR and cp.comm.AR_C == 1024
    for r in range(1, 8):
        assert jnp.array_equal(out_all[r, : ex.B], out_all[0, : ex.B])
    expected = _expected_residual(x_all, ex.B)
    assert jnp.allclose(out_all[0, : ex.B], expected, atol=2e-1, rtol=2e-2)


@pytest.mark.parametrize(
    "R,bad_group,H", [(8, 3, 3072), (8, 16, 2048), (8, 1, 2048), (24, 8, 1024)]
)
def test_allreduce_ring_hier4_rejects_bad_hier_group(R: int, bad_group: int, H: int):
    """Verify `build_scratch` rejects invalid `hier_group` values for `hier=4`."""
    tiles = dataclasses.replace(TINY_TILES, hier=4, hier_group=bad_group, war=1, mb8=2)
    ex = ExecSpec(B=4, T=1, R=R, ctx=128, tiles=tiles, interpret=True)
    g = FakeGeometry(L=2, H=H)
    ms = ModelSpec(
        "ar_bad_group", config=g, layer_type=LayerType("lt", slots=(_TwoTileModule(),))
    )
    with pytest.raises(MegakernelConfigError, match="unsupported all-reduce mode"):
        vmem.build_scratch(ms, ex)
    tiles_cw = dataclasses.replace(TINY_TILES, hier=4, hier_group=8, war=1, mb8=2)
    ex_cw = ExecSpec(B=4, T=1, R=8, ctx=128, tiles=tiles_cw, interpret=True)
    ms_cw = ModelSpec(
        "ar_bad_cw",
        config=FakeGeometry(L=2, H=512),
        layer_type=LayerType("lt", slots=(_TwoTileModule(), _SecondTileModule())),
    )
    with pytest.raises(MegakernelConfigError, match="multiple of 128"):
        vmem.build_scratch(ms_cw, ex_cw)


def test_allreduce_ring_hier4_rejects_single_allreduce_slot():
    """Verify `hier=4` rejects 1 all-reduce slot for both 1-host and 2-host meshes."""
    build_scratch = vmem.build_scratch
    g = FakeGeometry(L=2, H=2048)
    lt = LayerType(name="single_ar", slots=(_TwoTileModule(),))
    spec = ModelSpec("single_ar", config=g, layer_type=lt)
    for hg in (4, 8):
        tiles = dataclasses.replace(TINY_TILES, hier=4, hier_group=hg, war=1, mb8=2)
        ex = ExecSpec(B=4, T=1, R=8, ctx=128, tiles=tiles, interpret=True)
        with pytest.raises(
            MegakernelConfigError, match="hier=4 needs 2 all-reduce slots"
        ):
            build_scratch(spec, ex)

    tiles_h0 = dataclasses.replace(TINY_TILES, hier=0, hier_group=1, war=1, mb8=2)
    ex_h0 = ExecSpec(B=4, T=1, R=8, ctx=128, tiles=tiles_h0, interpret=True)
    spec_h0 = ModelSpec("single_ar", config=g, layer_type=lt)
    build_scratch(spec_h0, ex_h0)


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


def test_allreduce_ring_odd_batches_and_heavy_tail_float64_bounds():
    """Verify odd `B in (1, 3, 5)` and heavy-tail inputs obey float64 BF16 bounds."""
    u_bf16 = 2.0**-8
    configs = [
        (0, True, 8, 1, 1024, 1),
        (3, False, 8, 1, 1024, 2),
        (4, False, 8, 1, 1024, 1),
        (4, False, 4, 1, 1024, 2),
        (4, False, 4, 0, 2048, 2),
    ]
    for seed in (0, 17, 42):
        for b_cnt in (1, 3, 5):
            for hier, rsag, hg, war, h_dim, n_stage in configs:
                x_all, out_all, ex, _, _ = _execute_allreduce_ring(
                    8,
                    hier=hier,
                    rsag=rsag,
                    hier_group=hg,
                    war=war,
                    H=h_dim,
                    B=b_cnt,
                    seed=seed,
                    heavy_tail=True,
                )
                x_f64 = np.asarray(x_all[:, : ex.B], dtype=np.float64)
                out_f64 = np.asarray(out_all[:, : ex.B], dtype=np.float64)
                host_sum_f64 = 4.0 * np.sum(x_f64, axis=0)
                step_mag = float(np.max(np.sum(np.abs(x_f64), axis=0)))
                max_abs_bound = (4 * n_stage + 10) * u_bf16 * step_mag
                for r in range(8):
                    max_abs = float(np.max(np.abs(out_f64[r] - host_sum_f64)))
                    assert max_abs <= max_abs_bound, (
                        f"seed={seed} B={b_cnt} hier={hier} hg={hg} war={war} r={r}: "
                        f"max_abs={max_abs:.4e} > {max_abs_bound:.4e}"
                    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hier4_r32":
        _check_hier4_32_ranks_in_process()
