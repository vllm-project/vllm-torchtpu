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
"""Unit tests executing `TwoRing` and combined `AllReduceRing` in Pallas."""

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
from vllm_torchtpu.megakernel import collectives, planner, pool_alias, stream, vmem
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.ops import BF16, F32, I32, round_bf16
from vllm_torchtpu.megakernel.spec import (
    LayerType,
    ModelSpec,
    Module,
    SwitchSlot,
    WeightDecl,
)


class _TwoTileModule(Module):
    """Module with a 2-tile weight (`w1`) and a 1-tile weight (`w2`) at 64 KiB bank."""

    name = "two_tile"
    reduction = "allreduce"

    def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
        return [
            WeightDecl("w1", K=256, N=256, dtype=BF16, stream_ring="static"),
            WeightDecl("w2", K=256, N=128, dtype=BF16, stream_ring="static"),
        ]

    def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
        return {"acc": 256, "xb2": 256}


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


class _WideTileModule(Module):
    """Module with `n_t = 6` tiles and `N = 640 > 512` for `dmatmul_pair` testing."""

    name = "wide_tile"
    reduction = "allreduce"

    def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
        return [
            WeightDecl("w_wide", K=768, N=640, dtype=BF16, stream_ring="static"),
        ]

    def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
        return {"acc": 640}


class _ExpertAndStaticModule(Module):
    """Module declaring both a static (`dpool`) and a dynamic (`epool`) weight."""

    name = "expert_and_static"
    reduction = "allreduce"

    def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
        return [
            WeightDecl("w_dense", K=256, N=128, dtype=BF16, stream_ring="static"),
            WeightDecl("w_exp", K=128, N=256, dtype=BF16, stream_ring="dynamic"),
        ]

    def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
        return {"acc": 256}


def _run_wrapping_tworing(
    *, lazy: bool, rel_fetch: bool
) -> tuple[jax.Array, jax.Array]:
    # bank=64 KiB, nbd=2, L=3: w1 (256x256 -> 2 tiles), w2 (256x128 -> 1 tile).
    # T_TOTAL = 9 tiles across NBD = 2 banks (bank 0 reused 4x, bank 1 reused 3x).
    tiles = dataclasses.replace(TINY_TILES, bank=64 << 10, nbd=2, rel_fetch=rel_fetch)
    ex = ExecSpec(B=4, T=1, R=1, ctx=128, tiles=tiles, interpret=True)
    g = FakeGeometry(L=3, H=256)
    ms = ModelSpec(
        "tworing_test", config=g, layer_type=LayerType("lt", slots=(_TwoTileModule(),))
    )
    cp, scratch = vmem.build_scratch(ms, ex)
    sp = cp.stream
    assert sp.NBD == 2
    assert sp.nt["w1"] == 2 and sp.nt["w2"] == 1
    assert sp.T_TOTAL == 3 * (2 + 1)

    s_names = list(scratch.keys())

    def pallas_body(x_ref, w1_ref, w2_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        ring = stream.TwoRing(sp, {"w1": w1_ref, "w2": w2_ref}, scr, interpret=True)
        for i in range(sp.NBD):
            ring.fetch(jnp.int32(i))

        scr["xb"][...] = x_ref[...].astype(BF16)

        def layer_fn(lyr, _):
            ring.begin_layer(lyr)
            with ring.segment(lazy=lazy):
                h = ring.dmatmul("w1", scr["xb"], 256)
                scr["xb2"][:, :256] = h.astype(BF16)
            with ring.segment(lazy=lazy):
                y = ring.dmatmul("w2", scr["xb2"], 128)
            scr["xb"][:, :128] = (scr["xb"][:, :128].astype(F32) + y).astype(BF16)
            return 0

        lax.fori_loop(0, g.L, layer_fn, 0)
        out_ref[...] = scr["xb"][...].astype(F32)

    wsh = planner.weight_shapes(ms, ex, cp.plan)
    k0, k1, k2 = jax.random.split(jax.random.PRNGKey(0), 3)
    x = jax.random.normal(k0, (ex.MB, 256), dtype=BF16)
    w1_full = jax.random.normal(k1, (g.L, 256, 256), dtype=BF16) * 0.1
    w2_full = jax.random.normal(k2, (g.L, 256, 128), dtype=BF16) * 0.1
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
        interpret=pltpu.InterpretParams(detect_races=True),
    )
    out = jax.block_until_ready(jax.jit(call)(x, w1_tiled, w2_tiled))
    assert ipc.races is not None and not ipc.races.races_found

    ref = x.astype(F32)
    for lyr in range(g.L):
        h = (
            (ref.astype(BF16).astype(F32) @ w1_full[lyr].astype(F32))
            .astype(BF16)
            .astype(F32)
        )
        y = h @ w2_full[lyr].astype(F32)
        ref = ref.at[:, :128].set(
            (ref[:, :128].astype(BF16).astype(F32) + y).astype(BF16).astype(F32)
        )
    return out, ref


@pytest.mark.parametrize("lazy", [False, True])
def test_tworing_stream_and_lazy_segment_in_pallas(lazy: bool):
    """Execute wrapping `TwoRing` (`nbd=2`, `T_TOTAL=9`) across `lazy`/`rel_fetch`."""
    out_abs, ref = _run_wrapping_tworing(lazy=lazy, rel_fetch=False)
    out_rel, _ = _run_wrapping_tworing(lazy=lazy, rel_fetch=True)
    assert jnp.allclose(out_abs, ref, atol=1e-2, rtol=1e-2)
    assert jnp.array_equal(out_rel, out_abs)


def _run_wrapping_hybrid_switch_slot(
    *, lazy: bool, rel_fetch: bool
) -> tuple[jax.Array, jax.Array]:
    # bank=64 KiB, nbd=2, L=4, att_every=2:
    # Branch 0 (_AltTileModule: 2 tiles) + Branch 1 (_TwoTileModule: 3 tiles).
    # T_TOTAL = 10 tiles across NBD = 2 banks (each bank reused 4x >= 3x).
    tiles = dataclasses.replace(TINY_TILES, bank=64 << 10, nbd=2, rel_fetch=rel_fetch)
    ex = ExecSpec(B=4, T=1, R=1, ctx=128, tiles=tiles, interpret=True)
    g = FakeGeometry(L=4, att_every=2, H=256)
    sw = SwitchSlot(
        "mixer", branches=(_AltTileModule(), _TwoTileModule()), branch_counts=(1, 1)
    )
    ms = ModelSpec("hybrid_tworing", config=g, layer_type=LayerType("lt", slots=(sw,)))
    cp, scratch = vmem.build_scratch(ms, ex)
    sp = cp.stream
    assert sp.NBD == 2 and sp.T_TOTAL == 10
    s_names = list(scratch.keys())

    def pallas_body(x_ref, w_alt_ref, w1_ref, w2_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        W = {"w_alt": w_alt_ref, "w1": w1_ref, "w2": w2_ref}
        ring = stream.TwoRing(sp, W, scr, interpret=True)
        for i in range(sp.NBD):
            ring.fetch(jnp.int32(i))

        scr["xb"][...] = x_ref[...].astype(BF16)

        def layer_fn(lyr, _):
            ring.begin_layer(lyr)

            def b0():
                with ring.segment(lazy=lazy):
                    return ring.dmatmul("w_alt", scr["xb"], 256)[:, :128]

            def b1():
                with ring.segment(lazy=lazy):
                    h = ring.dmatmul("w1", scr["xb"], 256)
                    scr["xb2"][:, :256] = h.astype(BF16)
                with ring.segment(lazy=lazy):
                    return ring.dmatmul("w2", scr["xb2"], 128)

            y = ring.cond_branches(lyr % 2 == 0, b0, b1)
            scr["xb"][:, :128] = (scr["xb"][:, :128].astype(F32) + y).astype(BF16)
            return 0

        lax.fori_loop(0, g.L, layer_fn, 0)
        out_ref[...] = scr["xb"][...].astype(F32)

    wsh = planner.weight_shapes(ms, ex, cp.plan)
    k0, k1, k2, k3 = jax.random.split(jax.random.PRNGKey(7), 4)
    x = jax.random.normal(k0, (ex.MB, 256), dtype=BF16)
    w_alt_full = jax.random.normal(k1, (2, 256, 256), dtype=BF16) * 0.1
    w1_full = jax.random.normal(k2, (2, 256, 256), dtype=BF16) * 0.1
    w2_full = jax.random.normal(k3, (2, 256, 128), dtype=BF16) * 0.1

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
        interpret=pltpu.InterpretParams(detect_races=True),
    )
    out = jax.block_until_ready(
        jax.jit(call)(
            x,
            w_alt_full.reshape(wsh["w_alt"][0]),
            w1_full.reshape(wsh["w1"][0]),
            w2_full.reshape(wsh["w2"][0]),
        )
    )
    assert ipc.races is not None and not ipc.races.races_found

    ref = x.astype(F32)
    for cyc in range(2):
        y0 = (ref.astype(BF16).astype(F32) @ w_alt_full[cyc].astype(F32))[:, :128]
        ref = ref.at[:, :128].set(
            (ref[:, :128].astype(BF16).astype(F32) + y0).astype(BF16).astype(F32)
        )
        h1 = (
            (ref.astype(BF16).astype(F32) @ w1_full[cyc].astype(F32))
            .astype(BF16)
            .astype(F32)
        )
        y1 = h1 @ w2_full[cyc].astype(F32)
        ref = ref.at[:, :128].set(
            (ref[:, :128].astype(BF16).astype(F32) + y1).astype(BF16).astype(F32)
        )
    return out, ref


@pytest.mark.parametrize("lazy", [False, True])
def test_tworing_hybrid_switch_slot_in_pallas(lazy: bool):
    """Verify wrapping `SwitchSlot` prefetch across `lazy` and `rel_fetch`."""
    out_abs, ref = _run_wrapping_hybrid_switch_slot(lazy=lazy, rel_fetch=False)
    out_rel, _ = _run_wrapping_hybrid_switch_slot(lazy=lazy, rel_fetch=True)
    assert jnp.allclose(out_abs, ref, atol=1e-2, rtol=1e-2)
    assert jnp.array_equal(out_rel, out_abs)


def test_tworing_single_branch_cond_branches_in_pallas():
    """Verify `cond_branches` restores `_branch_offsets` for single-branch specs."""
    tiles = dataclasses.replace(TINY_TILES, bank=64 << 10, nbd=2)
    ex = ExecSpec(B=4, T=1, R=1, ctx=128, tiles=tiles, interpret=True)
    g = FakeGeometry(L=2, H=256)
    ms = ModelSpec(
        "single_branch_cond",
        config=g,
        layer_type=LayerType("lt", slots=(_TwoTileModule(),)),
    )
    cp, scratch = vmem.build_scratch(ms, ex)
    sp = cp.stream
    s_names = list(scratch.keys())

    def pallas_body(x_ref, w1_ref, w2_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        ring = stream.TwoRing(sp, {"w1": w1_ref, "w2": w2_ref}, scr, interpret=True)
        for i in range(sp.NBD):
            ring.fetch(jnp.int32(i))
        scr["xb"][...] = x_ref[...].astype(BF16)

        def layer_fn(lyr, _):
            ring.begin_layer(lyr)

            def arm_even():
                h = ring.dmatmul("w1", scr["xb"], 256)
                scr["xb2"][:, :256] = h.astype(BF16)
                return ring.dmatmul("w2", scr["xb2"], 128)

            def arm_odd():
                h = ring.dmatmul("w1", scr["xb"], 256) * 0.5
                scr["xb2"][:, :256] = h.astype(BF16)
                return ring.dmatmul("w2", scr["xb2"], 128)

            y = ring.cond_branches(lyr % 2 == 0, arm_even, arm_odd)
            scr["xb"][:, :128] = (scr["xb"][:, :128].astype(F32) + y).astype(BF16)
            return 0

        lax.fori_loop(0, g.L, layer_fn, 0)
        out_ref[...] = scr["xb"][...].astype(F32)

    wsh = planner.weight_shapes(ms, ex, cp.plan)
    k0, k1, k2 = jax.random.split(jax.random.PRNGKey(19), 3)
    x = jax.random.normal(k0, (ex.MB, 256), dtype=BF16)
    w1_full = jax.random.normal(k1, (g.L, 256, 256), dtype=BF16) * 0.1
    w2_full = jax.random.normal(k2, (g.L, 256, 128), dtype=BF16) * 0.1

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
        interpret=pltpu.InterpretParams(detect_races=True),
    )
    out = jax.block_until_ready(
        jax.jit(call)(
            x,
            w1_full.reshape(wsh["w1"][0]),
            w2_full.reshape(wsh["w2"][0]),
        )
    )
    assert ipc.races is not None and not ipc.races.races_found

    ref = x.astype(F32)
    for lyr in range(g.L):
        scale = 1.0 if lyr % 2 == 0 else 0.5
        h = (
            ((ref.astype(BF16).astype(F32) @ w1_full[lyr].astype(F32)) * scale)
            .astype(BF16)
            .astype(F32)
        )
        y = h @ w2_full[lyr].astype(F32)
        ref = ref.at[:, :128].set(
            (ref[:, :128].astype(BF16).astype(F32) + y).astype(BF16).astype(F32)
        )
    assert jnp.allclose(out, ref, atol=1e-2, rtol=1e-2)


def test_tworing_consumption_order_and_nesting_guards():
    """Verify `TwoRing` schedule order, nested `segment`, and lazy depth guards."""
    tiles = dataclasses.replace(TINY_TILES, bank=64 << 10, nbd=2)
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
        ring = stream.TwoRing(sp, {"w1": w1_ref, "w2": w2_ref}, scr, interpret=True)
        ring.begin_layer(jnp.int32(0))
        # Calling "w2" before "w1" violates declaration order ("w1", "w2")
        ring.dmatmul("w2", x_ref, 128)
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

    # Lazy segment consuming 2 + 1 = 3 > NBD=2 tiles raises at trace time
    def oversized_lazy_body(x_ref, w1_ref, w2_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        ring = stream.TwoRing(sp, {"w1": w1_ref, "w2": w2_ref}, scr, interpret=True)
        ring.begin_layer(jnp.int32(0))
        with ring.segment(lazy=True):
            ring.dmatmul("w1", x_ref, 256)
            ring.dmatmul("w2", x_ref, 128)
        out_ref[...] = scr["acc"][:, :256]

    call_oversized_lazy = pl.pallas_call(
        oversized_lazy_body,
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
    with pytest.raises(
        AssertionError,
        match=(
            r"Lazy stream\.segment\(\) exceeded ring depth on "
            r"dmatmul\('w2'\): segment tiles 3 > NBD=2"
        ),
    ):
        jax.jit(call_oversized_lazy)(x, w1, w2)

    # Nested segment() raises AssertionError
    ring_dummy = stream.TwoRing(sp, {}, {}, interpret=True)
    with (
        pytest.raises(AssertionError, match="Nested stream.segment"),
        ring_dummy.segment(lazy=False),
        ring_dummy.segment(lazy=False),
    ):
        pass

    # dmatmul(lazy=...) disagreeing with enclosing segment(lazy=...) raises ValueError
    for seg_lazy in (False, True):
        ring_mismatch = stream.TwoRing(sp, {}, {}, interpret=True)
        with (
            pytest.raises(ValueError, match="disagrees with enclosing stream.segment"),
            ring_mismatch.segment(lazy=seg_lazy),
        ):
            ring_mismatch.dmatmul("w1", None, 256, g0=0, lazy=not seg_lazy)


@pytest.mark.parametrize(
    "dense_acc_pair,dmm_unroll,rel_fetch",
    [
        (True, 0, False),
        (True, 0, True),
        (True, 2, False),
        (False, 6, False),
    ],
)
def test_tworing_dmatmul_pair_and_unroll_in_pallas(
    dense_acc_pair: bool, dmm_unroll: int, rel_fetch: bool
):
    """Verify `dmatmul_pair` (`n_t=6`, `N=640 > 512`) matches the JAX reference."""
    # row_bytes for N=640 bf16 is 128 * 640 * 2 = 160 KiB -> t_k=128, n_t=6 at K=768.
    tiles = dataclasses.replace(
        TINY_TILES,
        bank=160 << 10,
        nbd=6,
        dense_acc_pair=dense_acc_pair,
        dmm_unroll=dmm_unroll,
        rel_fetch=rel_fetch,
    )
    ex = ExecSpec(B=4, T=1, R=1, ctx=128, tiles=tiles, interpret=True)
    g = FakeGeometry(L=2, H=768)
    ms = ModelSpec(
        "dmatmul_pair_test",
        config=g,
        layer_type=LayerType("lt", slots=(_WideTileModule(),)),
    )
    cp, scratch = vmem.build_scratch(ms, ex)
    sp = cp.stream
    assert sp.nt["w_wide"] == 6 and sp.tk["w_wide"] == 128
    s_names = list(scratch.keys())

    def pallas_body(x_ref, w_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        ring = stream.TwoRing(sp, {"w_wide": w_ref}, scr, interpret=True)
        for i in range(sp.NBD):
            ring.fetch(jnp.int32(i))
        scr["xb"][...] = x_ref[...].astype(BF16)

        def layer_fn(lyr, _):
            ring.begin_layer(lyr)
            y = ring.dmatmul("w_wide", scr["xb"], 640)
            scr["xb"][:, :640] = (scr["xb"][:, :640].astype(F32) + y).astype(BF16)
            return 0

        lax.fori_loop(0, g.L, layer_fn, 0)
        out_ref[...] = scr["xb"][:, :640].astype(F32)

    wsh = planner.weight_shapes(ms, ex, cp.plan)
    k0, k1 = jax.random.split(jax.random.PRNGKey(31), 2)
    x = jax.random.normal(k0, (ex.MB, 768), dtype=BF16)
    w_full = jax.random.normal(k1, (g.L, 768, 640), dtype=BF16) * 0.05

    call = pl.pallas_call(
        pallas_body,
        out_shape=jax.ShapeDtypeStruct((ex.MB, 640), F32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[scratch[k] for k in s_names],
        interpret=pltpu.InterpretParams(detect_races=True),
    )
    out = jax.block_until_ready(jax.jit(call)(x, w_full.reshape(wsh["w_wide"][0])))
    assert ipc.races is not None and not ipc.races.races_found

    ref = x.astype(F32)
    for lyr in range(g.L):
        y = ref.astype(BF16).astype(F32) @ w_full[lyr].astype(F32)
        ref = ref.at[:, :640].set(
            (ref[:, :640].astype(BF16).astype(F32) + y).astype(BF16).astype(F32)
        )
    assert jnp.allclose(out, ref[:, :640], atol=1e-2, rtol=1e-2)


def test_tworing_untyped_dview_and_eview_slice_offsets(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify untyped (`typed=False`) `dview` and `eview` bank row offsets and sizes."""
    tiles = dataclasses.replace(TINY_TILES, bank=64 << 10, nbd=3, ebank=64 << 10, nbe=2)
    ex = ExecSpec(B=4, T=1, R=1, ctx=128, tiles=tiles, interpret=False)
    g = FakeGeometry(L=2, H=256)
    ms = ModelSpec(
        "untyped_views",
        config=g,
        layer_type=LayerType("lt", slots=(_ExpertAndStaticModule(),)),
    )
    cp, _ = vmem.build_scratch(ms, ex)
    sp = cp.stream
    assert not sp.typed

    class _FakePoolRef:
        def __init__(self, name: str):
            self.name = name
            self.at = self

        def __getitem__(self, idx: Any):
            return (self.name, idx)

    captured: list[tuple[Any, tuple[int, ...], jnp.dtype]] = []

    def _fake_view(ref: Any, shape: tuple[int, ...], dtype: Any):
        dt = jnp.dtype(dtype)
        captured.append((ref, shape, dt))
        return ("view", ref, shape, dt)

    monkeypatch.setattr(pool_alias, "view", _fake_view)
    scr = {"dpool": _FakePoolRef("dpool"), "epool": _FakePoolRef("epool")}
    ring = stream.TwoRing(sp, {}, scr, interpret=False)

    dv = ring.dview("w_dense", 2)
    ev = ring.eview("w_exp", 1)

    d_shp = sp.vshape["w_dense"]
    d_rows = pool_alias.storage_size(d_shp, jnp.dtype(BF16)) // 128
    assert dv[0] == "view"
    assert dv[1][0] == "dpool"
    assert dv[1][1].start == 2 * sp.BROWS and dv[1][1].size == d_rows
    assert dv[2] == d_shp and dv[3] == jnp.dtype(BF16)

    e_shp = sp.vshape["w_exp"]
    e_rows = pool_alias.storage_size(e_shp, jnp.dtype(BF16)) // 128
    assert ev[0] == "view"
    assert ev[1][0] == "epool"
    assert ev[1][1].start == 1 * sp.EBROWS and ev[1][1].size == e_rows
    assert ev[2] == e_shp and ev[3] == jnp.dtype(BF16)
    assert len(captured) == 2


@pytest.mark.tpu_test
def test_tworing_untyped_dpool_and_epool_on_tpu():
    """Execute `TwoRing` with `interpret=False` (`typed=False`) on TPU."""
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend for untyped pool_alias.view execution")

    tiles = dataclasses.replace(TINY_TILES, bank=64 << 10, nbd=2, ebank=64 << 10, nbe=2)
    ex = ExecSpec(B=4, T=1, R=1, ctx=128, tiles=tiles, interpret=False)
    g = FakeGeometry(L=2, H=256)
    ms = ModelSpec(
        "tpu_untyped_tworing",
        config=g,
        layer_type=LayerType("lt", slots=(_ExpertAndStaticModule(),)),
    )
    cp, scratch = vmem.build_scratch(ms, ex)
    sp = cp.stream
    assert not sp.typed
    s_names = list(scratch.keys())

    def pallas_body(x_ref, wd_ref, we_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        ring = stream.TwoRing(
            sp, {"w_dense": wd_ref, "w_exp": we_ref}, scr, interpret=False
        )
        for i in range(sp.NBD):
            ring.fetch(jnp.int32(i))
        scr["xb"][...] = x_ref[...].astype(BF16)

        def layer_fn(lyr, _):
            ring.begin_layer(lyr)
            h = ring.dmatmul("w_dense", scr["xb"], 128)
            with ring.expert_stream():
                bank = lyr % sp.NBE
                ring.ecopy("w_exp", we_ref.at[lyr, 0], bank).start()
                ring.ecopy("w_exp", we_ref.at[0, 0], bank).wait()
                ew = ring.eview("w_exp", bank)
                y = ring.mm(h.astype(BF16), ew[...])
            scr["xb"][...] = (scr["xb"][...].astype(F32) + y).astype(BF16)
            return 0

        lax.fori_loop(0, g.L, layer_fn, 0)
        out_ref[...] = scr["xb"][...].astype(F32)

    wsh = planner.weight_shapes(ms, ex, cp.plan)
    k0, k1, k2 = jax.random.split(jax.random.PRNGKey(23), 3)
    x = jax.random.normal(k0, (ex.MB, 256), dtype=BF16)
    wd_full = jax.random.normal(k1, (g.L, 256, 128), dtype=BF16) * 0.1
    we_full = jax.random.normal(k2, (g.L, 128, 256), dtype=BF16) * 0.1

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
    )
    out = jax.block_until_ready(
        jax.jit(call)(
            x,
            wd_full.reshape(wsh["w_dense"][0]),
            we_full.reshape(wsh["w_exp"][0]),
        )
    )

    ref = x.astype(F32)
    for lyr in range(g.L):
        h = (ref.astype(BF16).astype(F32) @ wd_full[lyr].astype(F32)).astype(BF16)
        y = h.astype(F32) @ we_full[lyr].astype(F32)
        ref = (ref + y).astype(BF16).astype(F32)
    assert jnp.allclose(out, ref, atol=1e-2, rtol=1e-2)


def test_tworing_combined_with_allreduce_ring_in_pallas():
    """Execute `TwoRing.dmatmul` with `AllReduceRing.allreduce` across 2 ranks."""
    R = 2
    devs = jax.devices("cpu")
    assert len(devs) >= R

    tiles = dataclasses.replace(
        TINY_TILES, bank=64 << 10, nbd=2, hier=0, rsag=False, war=1, mb8=2
    )
    ex = ExecSpec(B=4, T=1, R=R, ctx=128, tiles=tiles, interpret=True)
    g = FakeGeometry(L=2, H=256)
    ms = ModelSpec(
        "combined_tworing_ar",
        config=g,
        layer_type=LayerType("lt", slots=(_AltTileModule(),)),
    )
    cp, scratch = vmem.build_scratch(ms, ex)
    sp = cp.stream
    s_names = list(scratch.keys())

    def pallas_body(x_ref, w_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        me = lax.axis_index("tp").astype(I32)
        ring = stream.TwoRing(sp, {"w_alt": w_ref}, scr, interpret=True)
        ar = collectives.AllReduceRing(cp.comm, scr, me)

        barrier = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(
            barrier, 1, device_id=(me ^ 1,), device_id_type=pl.DeviceIdType.MESH
        )
        pl.semaphore_wait(barrier, 1)

        for i in range(sp.NBD):
            ring.fetch(jnp.int32(i))
        scr["res"][...] = jnp.zeros((ex.MB, g.H), F32)
        scr["xb"][...] = x_ref[...].astype(BF16)

        def layer_fn(lyr, _):
            ring.begin_layer(lyr)
            y = ring.dmatmul("w_alt", scr["xb"], 256)
            tot = ar.allreduce(lyr & 1, y, lyr < 2)
            ar.add_residual(tot)
            return 0

        lax.fori_loop(0, g.L, layer_fn, 0)
        ar.drain_sends()
        out_ref[...] = scr["res"][...]

    call = pl.pallas_call(
        pallas_body,
        out_shape=jax.ShapeDtypeStruct((ex.MB, g.H), F32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[scratch[k] for k in s_names],
        compiler_params=pltpu.CompilerParams(collective_id=0),
        interpret=pltpu.InterpretParams(detect_races=True),
    )

    wsh = planner.weight_shapes(ms, ex, cp.plan)
    k0, k1 = jax.random.split(jax.random.PRNGKey(53), 2)
    x_all = jax.random.normal(k0, (R, ex.MB, g.H), dtype=BF16)
    w_all = jax.random.normal(k1, (R, g.L, 256, 256), dtype=BF16) * 0.1
    w_tiled = w_all.reshape((R, *wsh["w_alt"][0]))

    mesh = Mesh(devs[:R], ("tp",))
    step = jax.jit(
        jax.shard_map(
            lambda x, w: call(x[0], w[0])[None],
            mesh=mesh,
            in_specs=(P("tp"), P("tp")),
            out_specs=P("tp"),
            check_vma=False,
        )
    )
    out_all = jax.block_until_ready(step(x_all, w_tiled))
    assert ipc.races is not None and not ipc.races.races_found

    expected = jnp.zeros((ex.B, g.H), dtype=F32)
    for lyr in range(g.L):
        partials = jnp.stack(
            [
                x_all[r, : ex.B].astype(F32) @ w_all[r, lyr].astype(F32)
                for r in range(R)
            ],
            axis=0,
        )
        term = jnp.sum(partials.astype(BF16).astype(F32), axis=0)
        expected = round_bf16(expected + round_bf16(term))
    for r in range(R):
        assert jnp.allclose(out_all[r, : ex.B], expected, atol=2e-1, rtol=2e-2)


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
