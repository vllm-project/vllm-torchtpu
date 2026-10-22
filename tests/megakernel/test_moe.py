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
"""Unit tests for `MoEBlock` declarations and Pallas `interpret=True` execution."""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TINY_TILES, FakeGeometry, FakeMixerBlock, MiniCtx
from jax import lax
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import blocks, comm, planner, stream, vmem
from vllm_torchtpu.megakernel.blocks import MoEBlock
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8, I32, rmsnorm, rnd
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec

BLOCKS_DIR = pathlib.Path(blocks.__file__).resolve().parent

# Row-count names that must never be compared to a literal inside `blocks/`.
_ROW_NAMES = {"B", "NS", "MR", "MB", "M", "CB"}
# Remaining literal row-count comparisons are hardware / layout limits, not
# per-bucket tuning (each one with its reason).
_ALLOWED = {
    ("moe/topk.py", "MR <= 128"): "lane top-k keeps all MR rows in one 128-lane vreg",
    (
        "gqa/params.py",
        "ex.NS <= 64",
    ): "GQ packs 128 // NS lanes per sequence (m | l) field",
}


def _row_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and node.id in _ROW_NAMES:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in _ROW_NAMES:
        return node.attr
    return None


def _is_num(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    )


def test_blocks_have_no_row_count_literal_comparisons():
    """Blocks read per-bucket tuning from `Tiles`, never `B <= 40`-style literals."""
    assert (BLOCKS_DIR / "moe" / "block.py").is_file()
    found = []
    for path in sorted(BLOCKS_DIR.rglob("*.py")):
        rel = path.relative_to(BLOCKS_DIR).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            ops = [node.left, *node.comparators]
            for a, b in zip(ops, ops[1:]):
                pair = (_row_name(a) and _is_num(b)) or (_row_name(b) and _is_num(a))
                if pair:
                    key = (rel, ast.unparse(node))
                    if key not in _ALLOWED:
                        found.append(f"{rel}:{node.lineno}: {key[1]}")
    assert not found, "row-count literal comparisons in blocks/:\n" + "\n".join(found)


def test_moe_block_declarations_and_emit_in_pallas_interpret(fake_ex: ExecSpec):
    """Verify `MoEBlock` declarations drive `planner`/`vmem` and `emit` in Pallas."""
    g_decl = FakeGeometry(L=2, att_every=1, H=256)
    moe_decl = MoEBlock()
    ms_decl = ModelSpec(
        "moe_decl_test",
        config=g_decl,
        layer_type=LayerType("lt", slots=(FakeMixerBlock(), moe_decl)),
    )

    p_decl = planner.plan(ms_decl, fake_ex)
    assert p_decl.static_kinds == ("att_in", "att_out", "router", "sh13", "sh2")
    assert p_decl.dynamic_kinds == ("e13", "e2")
    assert p_decl.w_names == (
        "att_in",
        "att_out",
        "router",
        "sh13",
        "sh2",
        "e13",
        "e2",
        "es",
        "smalls",
        "final_norm",
    )

    cp_decl, scr_decl = vmem.build_scratch(ms_decl, fake_ex, p_decl)
    assert (
        "wn" in scr_decl
        and "elist" in scr_decl
        and "es" in scr_decl
        and "st" in scr_decl
    )
    assert cp_decl.block("moe").EW == g_decl.E // fake_ex.R

    ex = ExecSpec(
        B=4, T=1, R=1, ctx=256, stack_e=True, tiles=TINY_TILES, interpret=True
    )
    g = FakeGeometry(
        L=1,
        att_every=1,
        H=256,
        NH=2,
        NKV=1,
        HD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=128,
        moe_tp=True,
    )
    moe = MoEBlock()
    ms = ModelSpec("moe_exec", config=g, layer_type=LayerType("lt", slots=(moe,)))
    cp, scratch = vmem.build_scratch(ms, ex)
    p = cp.plan
    wsh = planner.weight_shapes(ms, ex, p)
    osh = planner.kernel_out_shapes(ms, ex, p)
    s_names = list(scratch.keys())

    assert moe.input_norm == "post_norm"

    def pallas_body(pos_ref, hid_ref, *rest):
        n_w, n_o = len(p.w_names), len(p.o_names)
        W = dict(zip(p.w_names, rest[:n_w]))
        O = dict(zip(p.o_names, rest[n_w : n_w + n_o]))  # noqa: E741
        scr = dict(zip(s_names, rest[n_w + n_o :]))
        me = jnp.int32(0)
        ring = stream.TwoRing(cp.stream, W, scr, me, interpret=True)
        ar = comm.AllReduceRing(cp.comm, scr, me)
        ctx = MiniCtx(
            g=g,
            ex=ex,
            core_plan=cp,
            pos_ref=pos_ref,
            W=W,
            ST={},
            O=O,
            scr=scr,
            me=me,
            stream=ring,
            comm=ar,
        )

        for i in range(cp.stream.NBD):
            ring.fetch(jnp.int32(i))
        pltpu.make_async_copy(
            W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        ).start()
        moe.prologue(ctx)
        scr["res"][...] = hid_ref[: ex.MB].astype(F32)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        moe.init_buffers(ctx)

        pltpu.make_async_copy(
            W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        ).wait()
        ring.begin_layer(jnp.int32(0))
        ctx.apply_input_norm(moe.input_norm, jnp.int32(0))
        part_moe = moe.emit(ctx, jnp.int32(0), jnp.int32(0))
        ar.add_residual(ar.allreduce(0, part_moe, jnp.bool_(True)))
        O["hidden"][: ex.MB] = scr["res"][...].astype(BF16)
        if ex.MB < ex.M:
            O["hidden"][ex.MB :] = jnp.zeros((ex.M - ex.MB, g.H), BF16)

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.SMEM),
        pl.BlockSpec(memory_space=pltpu.VMEM),
    ] + [pl.BlockSpec(memory_space=pltpu.HBM) for _ in p.w_names]
    out_specs = [pl.BlockSpec(memory_space=pltpu.VMEM) for _ in p.o_names]
    out_shape = [jax.ShapeDtypeStruct(*osh[k]) for k in p.o_names]

    call = pl.pallas_call(
        pallas_body,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        interpret=pltpu.InterpretParams(),
    )

    pos = jnp.array([4, 8, 12, 16], dtype=I32)
    hid = jnp.ones((ex.M, g.H), dtype=BF16) * 0.1
    w_args = [jnp.zeros(wsh[k][0], dtype=wsh[k][1]) for k in p.w_names]
    (hid_out,) = jax.jit(call)(pos, hid, *w_args)
    assert hid_out.shape == (ex.M, g.H)
    assert jnp.all(jnp.isfinite(hid_out))


@pytest.mark.parametrize(
    "moe_tp,R,pipe",
    [
        (False, 1, False),
        (True, 1, False),
        (True, 1, True),
        (False, 2, False),
        (False, 2, True),
        (True, 2, True),
    ],
)
def test_moe_block_numerical_parity(moe_tp: bool, R: int, pipe: bool):
    """Compare `MoEBlock.emit` in Pallas `interpret=True` against a JAX reference."""
    devs = jax.devices("cpu")
    assert len(devs) >= R

    H, E, TOPK, IR, ISR = 128, 128, 2, 128, 128
    ER = E // R
    EW = E if moe_tp else ER
    tiles = dataclasses.replace(TINY_TILES, war=1, mb8=2).with_blocks(
        {"moe": {"pipe": pipe}}
    )
    ex = ExecSpec(B=2, T=1, R=R, ctx=128, stack_e=True, tiles=tiles, interpret=True)
    g = FakeGeometry(
        L=1,
        att_every=1,
        H=H,
        NH=2,
        NKV=1,
        HD=128,
        E=E,
        TOPK=TOPK,
        I=IR * R if moe_tp else IR,
        IS=ISR * R,
        moe_tp=moe_tp,
    )
    moe = MoEBlock()
    ms = ModelSpec("moe_num", config=g, layer_type=LayerType("lt", slots=(moe,)))
    cp, scratch = vmem.build_scratch(ms, ex)
    assert cp.block("moe").PIPE is pipe
    p = cp.plan
    wsh = planner.weight_shapes(ms, ex, p)
    osh = planner.kernel_out_shapes(ms, ex, p)
    s_names = list(scratch.keys())

    keys = jax.random.split(jax.random.PRNGKey(101), 12)
    hid = jax.random.normal(keys[0], (ex.M, H), dtype=BF16) * 0.5
    post_norm = jax.random.uniform(keys[1], (H,), dtype=F32, minval=-0.2, maxval=0.2)

    # Construct router weights so token 0 strongly prefers experts on rank 0
    # and token 1 strongly prefers experts on rank (R - 1), exercising both
    # ranks and the multi-hit PIPE loop.
    x_norm0 = rmsnorm(
        hid[: ex.B].astype(F32), post_norm, g.eps, zero_centered=g.zero_centered_norm
    ).astype(BF16)
    router_f32 = jax.random.normal(keys[2], (H, E), dtype=F32) * 0.05
    b0_dir = x_norm0[0].astype(F32)
    b1_dir = x_norm0[1].astype(F32)
    e_r0_a, e_r0_b = 2, 5
    e_r1_a, e_r1_b = (R - 1) * ER + 3, (R - 1) * ER + 7
    router_f32 = router_f32.at[:, e_r0_a].add(b0_dir * 2.0)
    router_f32 = router_f32.at[:, e_r0_b].add(b0_dir * 1.5)
    router_f32 = router_f32.at[:, e_r1_a].add(b1_dir * 2.0)
    router_f32 = router_f32.at[:, e_r1_b].add(b1_dir * 1.5)
    router = router_f32.astype(BF16)

    sh_gate = jax.random.normal(keys[3], (H,), dtype=F32) * 0.2
    sh13 = (jax.random.normal(keys[4], (R, H, 2 * ISR), dtype=F32) * 0.4).astype(FP8)
    sh13_s = jax.random.uniform(
        keys[5], (R, 2 * ISR), dtype=F32, minval=0.05, maxval=0.15
    )
    sh2 = (jax.random.normal(keys[6], (R, ISR, H), dtype=F32) * 0.4).astype(FP8)
    sh2_s = jax.random.uniform(keys[7], (R, H), dtype=F32, minval=0.05, maxval=0.15)
    e13 = (jax.random.normal(keys[8], (R, EW, H, 2 * IR), dtype=F32) * 0.4).astype(FP8)
    e13_s = jax.random.uniform(
        keys[9], (R, EW, 2 * IR), dtype=F32, minval=0.05, maxval=0.15
    )
    e2 = (jax.random.normal(keys[10], (R, EW, IR, H), dtype=F32) * 0.4).astype(FP8)
    e2_s = jax.random.uniform(keys[11], (R, EW, H), dtype=F32, minval=0.05, maxval=0.15)

    lay = p.smalls_layout
    sm = jnp.zeros((R, 1, 1, p.smalls_width), dtype=F32)
    for name, val in (
        ("post_norm", jnp.broadcast_to(post_norm[None, :], (R, H))),
        ("sh13_s", sh13_s),
        ("sh2_s", sh2_s),
        ("sh_gate", jnp.broadcast_to(sh_gate[None, :], (R, H))),
    ):
        o, w = lay[name]
        sm = sm.at[:, 0, 0, o : o + w].set(val)

    w_by_name = {
        "router": jnp.broadcast_to(router[None, ...], (R, 1, H, E)).reshape(
            R, *wsh["router"][0]
        ),
        "sh13": sh13.reshape(R, *wsh["sh13"][0]),
        "sh2": sh2.reshape(R, *wsh["sh2"][0]),
        "e13": e13.reshape(R, *wsh["e13"][0]),
        "e2": e2.reshape(R, *wsh["e2"][0]),
        "es": jnp.concatenate([e13_s, e2_s], axis=-1).reshape(R, *wsh["es"][0]),
        "smalls": sm,
        "final_norm": jnp.zeros((R, *wsh["final_norm"][0]), dtype=F32),
    }

    def pallas_body(pos_ref, hid_ref, *rest):
        n_w, n_o = len(p.w_names), len(p.o_names)
        W = dict(zip(p.w_names, rest[:n_w]))
        O = dict(zip(p.o_names, rest[n_w : n_w + n_o]))  # noqa: E741
        scr = dict(zip(s_names, rest[n_w + n_o :]))
        me = lax.axis_index("tp").astype(I32) if R > 1 else jnp.int32(0)
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

        ring = stream.TwoRing(cp.stream, W, scr, me, interpret=True)
        ar = comm.AllReduceRing(cp.comm, scr, me)
        ctx = MiniCtx(
            g=g,
            ex=ex,
            core_plan=cp,
            pos_ref=pos_ref,
            W=W,
            ST={},
            O=O,
            scr=scr,
            me=me,
            stream=ring,
            comm=ar,
        )

        for i in range(cp.stream.NBD):
            ring.fetch(jnp.int32(i))
        pltpu.make_async_copy(
            W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        ).start()
        moe.prologue(ctx)
        scr["res"][...] = hid_ref[: ex.MB].astype(F32)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        moe.init_buffers(ctx)

        pltpu.make_async_copy(
            W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        ).wait()
        ring.begin_layer(jnp.int32(0))
        ctx.apply_input_norm(moe.input_norm, jnp.int32(0))
        part_moe = moe.emit(ctx, jnp.int32(0), jnp.int32(0))
        ar.add_residual(ar.allreduce(0, part_moe, jnp.bool_(True)))
        ar.drain_sends()
        O["hidden"][: ex.MB] = scr["res"][...].astype(BF16)
        if ex.MB < ex.M:
            O["hidden"][ex.MB :] = jnp.zeros((ex.M - ex.MB, H), BF16)

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.SMEM),
        pl.BlockSpec(memory_space=pltpu.VMEM),
    ] + [pl.BlockSpec(memory_space=pltpu.HBM) for _ in p.w_names]
    out_specs = [pl.BlockSpec(memory_space=pltpu.VMEM) for _ in p.o_names]
    out_shape = [jax.ShapeDtypeStruct(*osh[k]) for k in p.o_names]

    call = pl.pallas_call(
        pallas_body,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        compiler_params=pltpu.CompilerParams(collective_id=0) if R > 1 else None,
        interpret=pltpu.InterpretParams(),
    )

    pos = jnp.array([4, 8], dtype=I32)
    w_all = [w_by_name[k] for k in p.w_names]
    if R == 1:
        (hid_out,) = jax.jit(call)(pos, hid, *[w[0] for w in w_all])
        out_ranks = hid_out[None, : ex.B]
    else:
        mesh = Mesh(devs[:R], ("tp",))
        step = jax.jit(
            jax.shard_map(
                lambda pos_in, hid_in, *ws: call(
                    pos_in, hid_in[0], *[w[0] for w in ws]
                )[0][None],
                mesh=mesh,
                in_specs=(P(), P("tp"), *(P("tp") for _ in p.w_names)),
                out_specs=P("tp"),
                check_vma=False,
            )
        )
        hid_rep = jnp.broadcast_to(hid[None, ...], (R, ex.M, H))
        out_ranks = step(pos, hid_rep, *w_all)[:, : ex.B]

    # Pure-JAX MoE reference: RMSNorm -> router softmax + top-k -> shared + routed
    res = hid[: ex.B].astype(F32)
    x_moe = rmsnorm(res, post_norm, g.eps, zero_centered=g.zero_centered_norm)
    xb_moe = x_moe.astype(BF16).astype(F32)

    logits = xb_moe @ router.astype(F32)
    probs = jax.nn.softmax(logits, axis=-1)
    top_vals, top_idx = lax.top_k(probs, TOPK)
    top_w = top_vals / jnp.sum(top_vals, axis=-1, keepdims=True)

    sg = jax.nn.sigmoid(jnp.sum(x_moe * sh_gate[None, :], axis=-1, keepdims=True))
    partials = []
    for r in range(R):
        g13 = (xb_moe @ sh13[r].astype(F32)) * sh13_s[r]
        mid_sh = (
            rnd(rnd(jax.nn.silu(rnd(g13[:, :ISR]))) * rnd(g13[:, ISR:]))
            .astype(BF16)
            .astype(F32)
        )
        part_r = (mid_sh @ sh2[r].astype(F32)) * sh2_s[r] * sg
        for b in range(ex.B):
            for k in range(TOPK):
                e = int(top_idx[b, k])
                if not moe_tp and (e // ER) != r:
                    continue
                w_slot = e if moe_tp else (e - r * ER)
                h_e = (xb_moe[b : b + 1] @ e13[r, w_slot].astype(F32)) * e13_s[
                    r, w_slot
                ]
                mid_e = (
                    rnd(rnd(jax.nn.silu(rnd(h_e[:, :IR]))) * rnd(h_e[:, IR:]))
                    .astype(BF16)
                    .astype(F32)
                )
                y_e = rnd((mid_e @ e2[r, w_slot].astype(F32)) * e2_s[r, w_slot])
                part_r = part_r.at[b : b + 1].add(top_w[b, k] * y_e)
        partials.append(part_r.astype(BF16).astype(F32))

    ref_out = (res + jnp.sum(jnp.stack(partials, axis=0), axis=0)).astype(BF16)
    for r in range(R):
        assert jnp.allclose(
            out_ranks[r].astype(F32), ref_out.astype(F32), atol=2e-2, rtol=2e-2
        )


def test_moe_topk_packed_and_route_vec_compile_time_bounds():
    """Verify compile-time checks guard bit-packed top-k and resolve `route_vec`."""
    moe = MoEBlock()
    base_tiles = dataclasses.replace(TINY_TILES, bank=512 << 10)
    g_ok = FakeGeometry(
        L=1,
        att_every=1,
        H=256,
        I=256,
        IS=1024,
        E=512,
        TOPK=8,
        moe_tp=False,
        e_slots=72,
    )
    ex_ok = ExecSpec(
        R=8,
        B=4,
        T=1,
        tiles=base_tiles.with_blocks({"moe": {"topk": True, "route_vec": True}}),
    )
    ms_ok = ModelSpec("moe_ok", config=g_ok, layer_type=LayerType("lt", slots=(moe,)))
    plan_ok = planner.plan(ms_ok, ex_ok)
    mp_ok = plan_ok.block_params["moe"]
    assert mp_ok.topk_packed is True
    assert mp_ok.route_vec is True

    # Non-power-of-two E=384 <= 512 is valid for topk=True.
    g_e384 = dataclasses.replace(g_ok, E=384, e_slots=0)
    ms_e384 = ModelSpec(
        "moe_e384", config=g_e384, layer_type=LayerType("lt", slots=(moe,))
    )
    ex_topk = ExecSpec(
        R=8,
        B=4,
        T=1,
        tiles=base_tiles.with_blocks({"moe": {"topk": True, "route_vec": False}}),
    )
    assert planner.plan(ms_e384, ex_topk).block_params["moe"].topk_packed is True

    # E > 512 (e.g. 513, 1024) overflows the 9-bit float32 mantissa pack when topk=True.
    for bad_e in (513, 1024):
        g_bad_topk = dataclasses.replace(g_ok, E=bad_e, e_slots=0)
        ms_bad_topk = ModelSpec(
            f"moe_e{bad_e}", config=g_bad_topk, layer_type=LayerType("lt", slots=(moe,))
        )
        with pytest.raises(MegakernelConfigError, match="topk_packed"):
            planner.plan(ms_bad_topk, ex_topk)

    # `route_vec` resolves to False (quiet fallback) when preconditions are not met:
    # - `e_slots=80` (128 redundant experts on R=8 -> R*EXTRA+R+1 = 137 > 128)
    # - `e_slots=96` (R*EXTRA+R+1 = 265 > 128)
    # - `E=2048, B=4` (E // 128 + 1 = 17 > MR=8)
    # - `e_slots=0` (ecopy=False)
    ex_rvec = ExecSpec(
        R=8,
        B=4,
        T=1,
        tiles=base_tiles.with_blocks({"moe": {"topk": False, "route_vec": True}}),
    )
    for fallback_g in (
        dataclasses.replace(g_ok, E=512, e_slots=80),
        dataclasses.replace(g_ok, E=512, e_slots=96),
        dataclasses.replace(g_ok, E=2048, e_slots=260),
        dataclasses.replace(g_ok, E=512, e_slots=0),
    ):
        ms_fb = ModelSpec(
            "moe_fb", config=fallback_g, layer_type=LayerType("lt", slots=(moe,))
        )
        assert planner.plan(ms_fb, ex_rvec).block_params["moe"].route_vec is False


def _simulate_expert_ring_no_collision(mp, n_hit: int = 8) -> None:
    """Simulate `ring.py` / `experts.py` `e_ring` (`NBE`) and `es` (`NSS`) lifetimes."""
    nbe, nss, te, t13, t2 = mp.NBE, mp.NSS, mp.TE, mp.T13, mp.T2
    assert nbe >= 1 and nss >= 1 and te == t13 + t2
    e_inflight: dict[int, int] = {}
    es_inflight: dict[int, int] = {}

    def fetch_unit(u: int, t: int | None = None) -> None:
        if u >= n_hit * te:
            return
        if t is not None:
            nt = (t + nbe) % te
            assert (u - nt) % te == 0, f"misaligned unit {u} for tile {nt}"
            if mp.PIPE_EP and nbe >= te:
                assert nt == t, (
                    "PIPE_EP requires (t + NBE) % TE == t, "
                    f"got t={t}, NBE={nbe}, TE={te}"
                )
        slot = u % nbe
        assert slot not in e_inflight, (
            f"e_ring slot {slot} collision: {e_inflight[slot]} vs {u}"
        )
        e_inflight[slot] = u
        if u % te == 0:
            exp_i = u // te
            es_slot = exp_i % nss
            assert es_slot not in es_inflight, (
                f"es slot {es_slot} collision: {es_inflight[es_slot]} vs {exp_i}"
            )
            es_inflight[es_slot] = exp_i

    # `route_fetch` primes units `0 .. min(NBE, n_hit * TE) - 1`; `fetch_e`
    # starts `es[(u // TE) % NSS]` whenever `u % TE == 0`.
    for u0 in range(min(nbe, n_hit * te)):
        fetch_unit(u0)

    if mp.PIPE:
        assert t13 == 1 and t2 == 1
        assert e_inflight.pop(0 % nbe, None) == 0
        assert es_inflight.get(0 % nss) == 0
        fetch_unit(0 + nbe, t=0)
        for i in range(n_hit - 1):
            u13 = (i + 1) * te
            u2 = i * te + t13
            assert e_inflight.pop(u13 % nbe, None) == u13
            assert es_inflight.get((i + 1) % nss) == i + 1
            assert e_inflight.pop(u2 % nbe, None) == u2
            assert es_inflight.pop(i % nss, None) == i
            fetch_unit(u13 + nbe, t=0)
            fetch_unit(u2 + nbe, t=t13)
        last = n_hit - 1
        u2_last = last * te + t13
        assert e_inflight.pop(u2_last % nbe, None) == u2_last
        assert es_inflight.pop(last % nss, None) == last
        fetch_unit(u2_last + nbe, t=t13)
    elif mp.PIPE_EP:
        assert t13 == 2 * t2
        for t in range(t13):
            assert e_inflight.pop(t % nbe, None) == t
            if mp.group_rel:
                if t % 2 == 1:
                    fetch_unit(t - 1 + nbe, t=t - 1)
                    fetch_unit(t + nbe, t=t)
                elif t == t13 - 1:
                    fetch_unit(t + nbe, t=t)
            else:
                fetch_unit(t + nbe, t=t)
        assert es_inflight.get(0 % nss) == 0
        for i in range(n_hit - 1):
            # In PIPE_EP, all `e2` tiles of expert `i` and all `e13` tiles of
            # expert `i + 1` must already be in flight at the start of `body_ep(i)`.
            for s_step in range(t2):
                u2 = i * te + t13 + s_step
                u13_0 = (i + 1) * te + 2 * s_step
                u13_1 = (i + 1) * te + 2 * s_step + 1
                assert e_inflight.get(u2 % nbe) == u2, (
                    f"unit {u2} not prefetched before body_ep({i})"
                )
                assert e_inflight.get(u13_0 % nbe) == u13_0, (
                    f"unit {u13_0} not prefetched before body_ep({i})"
                )
                assert e_inflight.get(u13_1 % nbe) == u13_1, (
                    f"unit {u13_1} not prefetched before body_ep({i})"
                )
            for s_step in range(t2):
                u2 = i * te + t13 + s_step
                t0, t1 = 2 * s_step, 2 * s_step + 1
                u13_0 = (i + 1) * te + t0
                u13_1 = (i + 1) * te + t1
                assert e_inflight.pop(u2 % nbe, None) == u2
                assert e_inflight.pop(u13_0 % nbe, None) == u13_0
                fetch_unit(u2 + nbe, t=t13 + s_step)
                fetch_unit(u13_0 + nbe, t=t0)
                assert e_inflight.pop(u13_1 % nbe, None) == u13_1
                fetch_unit(u13_1 + nbe, t=t1)
            # `scr["es"][s_cur, :, BN:]` is read after the `s_step` loop, then
            # `es_wait(i + 1)` waits on expert `i + 1`'s scales.
            assert es_inflight.pop(i % nss, None) == i
            assert es_inflight.get((i + 1) % nss) == i + 1
        i_last = n_hit - 1
        for s_step in range(t2):
            u2 = i_last * te + t13 + s_step
            assert e_inflight.pop(u2 % nbe, None) == u2
            if mp.group_rel:
                if s_step % 2 == 1:
                    fetch_unit(u2 - 1 + nbe, t=t13 + s_step - 1)
                    fetch_unit(u2 + nbe, t=t13 + s_step)
                elif s_step == t2 - 1:
                    fetch_unit(u2 + nbe, t=t13 + s_step)
            else:
                fetch_unit(u2 + nbe, t=t13 + s_step)
        assert es_inflight.pop(i_last % nss, None) == i_last
    else:
        for i in range(n_hit):
            base = i * te
            for t in range(t13):
                u = base + t
                assert e_inflight.pop(u % nbe, None) == u
                if mp.unroll and mp.group_rel and t13 > 1:
                    if t % 2 == 1:
                        fetch_unit(u - 1 + nbe, t=t - 1)
                        fetch_unit(u + nbe, t=t)
                    elif t == t13 - 1:
                        fetch_unit(u + nbe, t=t)
                else:
                    fetch_unit(u + nbe, t=t if mp.unroll and t13 > 1 else None)
            assert es_inflight.get(i % nss) == i
            u2_base = base + t13
            if t2 == 1:
                assert e_inflight.pop(u2_base % nbe, None) == u2_base
                assert es_inflight.pop(i % nss, None) == i
                fetch_unit(u2_base + nbe)
            else:
                for t in range(t2):
                    u = u2_base + t
                    assert e_inflight.pop(u % nbe, None) == u
                    if mp.unroll and mp.group_rel:
                        if t % 2 == 1:
                            fetch_unit(u - 1 + nbe, t=t13 + t - 1)
                            fetch_unit(u + nbe, t=t13 + t)
                        elif t == t2 - 1:
                            fetch_unit(u + nbe, t=t13 + t)
                    else:
                        fetch_unit(u + nbe, t=t13 + t if mp.unroll else None)
                assert es_inflight.pop(i % nss, None) == i
    assert not e_inflight and not es_inflight


def test_moe_deployment_ring_invariants():
    """Verify MoE ring schedules and PIPE / PIPE_EP / LAZY_MOE across real profiles."""
    from vllm_torchtpu.megakernel.models.qwen35 import Qwen35Geometry, qwen35_spec
    from vllm_torchtpu.megakernel.models.qwen35_candidate30 import candidate30_tiles
    from vllm_torchtpu.megakernel.models.qwen35_profile import baseline_tiles

    seen_pipe = False
    seen_pipe_ep = False
    seen_lazy = False
    sample_pipe_ep_mp = None
    for moe_tp, e_slots in ((False, 0), (False, 72), (True, 0)):
        g = Qwen35Geometry(L=4, moe_tp=moe_tp, e_slots=e_slots)
        ms = qwen35_spec(g)
        for T in (1, 4):
            for B in (1, 4, 8, 16, 20, 24, 32, 40, 48, 56, 64):
                if B % T != 0:
                    continue
                for tile_fn in (baseline_tiles, candidate30_tiles):
                    tiles = tile_fn(B, moe_tp=moe_tp, R=8)
                    ex = ExecSpec(R=8, B=B, T=T, tiles=tiles)
                    cp, _ = vmem.build_scratch(ms, ex)
                    mp = cp.block("moe")
                    assert mp.NSS >= 3 + (mp.NBE - 1) // mp.TE
                    assert mp.NBE >= 2
                    assert not (mp.PIPE and mp.PIPE_EP)
                    if mp.PIPE:
                        assert mp.T13 == 1 and mp.T2 == 1
                    if mp.PIPE_EP:
                        assert mp.T13 == 2 * mp.T2
                        assert mp.NBE >= mp.TE and mp.NBE % mp.TE == 0
                        sample_pipe_ep_mp = mp
                    if mp.LAZY_MOE:
                        assert cp.plan.ex.resolved_tiles.nbd >= mp.N_MOE
                    seen_pipe = seen_pipe or mp.PIPE
                    seen_pipe_ep = seen_pipe_ep or mp.PIPE_EP
                    seen_lazy = seen_lazy or mp.LAZY_MOE
                    _simulate_expert_ring_no_collision(mp, n_hit=8)
    assert seen_pipe and seen_pipe_ep and seen_lazy, (
        seen_pipe,
        seen_pipe_ep,
        seen_lazy,
    )
    assert sample_pipe_ep_mp is not None
    # Negative self-tests: mutating NSS or NBE below the schedule requirements
    # or breaking PIPE_EP's NBE % TE == 0 invariant must fail the simulator.
    with pytest.raises(AssertionError, match="es slot .* collision"):
        _simulate_expert_ring_no_collision(
            dataclasses.replace(sample_pipe_ep_mp, NSS=sample_pipe_ep_mp.NSS - 1),
            n_hit=8,
        )
    with pytest.raises(AssertionError, match="es slot .* collision"):
        _simulate_expert_ring_no_collision(
            dataclasses.replace(sample_pipe_ep_mp, T13=2, T2=1, TE=3, NBE=12, NSS=3),
            n_hit=8,
        )
    with pytest.raises(AssertionError, match="not prefetched before body_ep"):
        _simulate_expert_ring_no_collision(
            dataclasses.replace(sample_pipe_ep_mp, NBE=sample_pipe_ep_mp.TE - 1),
            n_hit=8,
        )
    with pytest.raises(AssertionError, match="PIPE_EP requires"):
        _simulate_expert_ring_no_collision(
            dataclasses.replace(
                sample_pipe_ep_mp, NBE=sample_pipe_ep_mp.TE + 1, NSS=10
            ),
            n_hit=8,
        )


@pytest.mark.parametrize(
    "case_name,pipe,pipe_ep,e_slots,route_vec",
    [
        ("pipe_single_tile", True, False, 0, False),
        ("pipe_ep_group_rel", False, True, 0, False),
        ("pipe_ep_eplb_route_vec", False, True, 70, True),
    ],
)
def test_moe_dynamic_dma_race_invariants(
    case_name: str,
    pipe: bool,
    pipe_ep: bool,
    e_slots: int,
    route_vec: bool,
):
    """Run MoEBlock under Pallas vector-clock race detection (`on_wait` vs `eager`)."""
    del case_name
    R = 2
    devs = jax.devices("cpu")
    assert len(devs) >= R

    H, E, TOPK, IR, ISR = 256, 128, 2, 128, 128
    ER = E // R
    EW = e_slots if e_slots > 0 else ER
    ebank = (32 << 10) if pipe_ep else (256 << 10)
    tiles = dataclasses.replace(
        TINY_TILES,
        bank=512 << 10,
        nbd=8,
        nbe=6,
        ebank=ebank,
        war=1,
        mb8=2,
    ).with_blocks(
        {
            "moe": {
                "pipe": pipe,
                "pipe_ep": pipe_ep,
                "moe_unroll": pipe_ep,
                "r2_direct_store": pipe_ep,
                "route_vec": route_vec,
                "topk_lanes": True,
            }
        }
    )
    g = FakeGeometry(
        L=1,
        att_every=1,
        H=H,
        NH=2,
        NKV=1,
        HD=128,
        E=E,
        TOPK=TOPK,
        I=IR,
        IS=ISR * R,
        moe_tp=False,
        e_slots=e_slots,
    )
    moe = MoEBlock()
    ms = ModelSpec("moe_race", config=g, layer_type=LayerType("lt", slots=(moe,)))

    def _run_mode(dma_mode: str):
        ex = ExecSpec(
            B=4,
            T=1,
            R=R,
            ctx=128,
            stack_e=True,
            tiles=tiles,
            interpret=pltpu.InterpretParams(
                detect_races=True,
                dma_execution_mode=dma_mode,
            ),
        )
        cp, scratch = vmem.build_scratch(ms, ex)
        assert cp.block("moe").route_vec is route_vec
        if pipe_ep:
            assert cp.block("moe").PIPE_EP is True
        p = cp.plan
        wsh = planner.weight_shapes(ms, ex, p)
        osh = planner.kernel_out_shapes(ms, ex, p)
        s_names = list(scratch.keys())

        keys = jax.random.split(jax.random.PRNGKey(202), 12)
        hid = jax.random.normal(keys[0], (ex.M, H), dtype=BF16) * 0.5
        post_norm = jax.random.uniform(
            keys[1], (H,), dtype=F32, minval=-0.2, maxval=0.2
        )
        router = jax.random.normal(keys[2], (H, E), dtype=BF16) * 0.1
        sh_gate = jax.random.normal(keys[3], (H,), dtype=F32) * 0.2
        sh13 = (jax.random.normal(keys[4], (R, H, 2 * ISR), dtype=F32) * 0.4).astype(
            FP8
        )
        sh13_s = jax.random.uniform(
            keys[5], (R, 2 * ISR), dtype=F32, minval=0.05, maxval=0.15
        )
        sh2 = (jax.random.normal(keys[6], (R, ISR, H), dtype=F32) * 0.4).astype(FP8)
        sh2_s = jax.random.uniform(keys[7], (R, H), dtype=F32, minval=0.05, maxval=0.15)
        e13 = (jax.random.normal(keys[8], (R, EW, H, 2 * IR), dtype=F32) * 0.4).astype(
            FP8
        )
        e13_s = jax.random.uniform(
            keys[9], (R, EW, 2 * IR), dtype=F32, minval=0.05, maxval=0.15
        )
        e2 = (jax.random.normal(keys[10], (R, EW, IR, H), dtype=F32) * 0.4).astype(FP8)
        e2_s = jax.random.uniform(
            keys[11], (R, EW, H), dtype=F32, minval=0.05, maxval=0.15
        )

        lay = p.smalls_layout
        sm = jnp.zeros((R, 1, 1, p.smalls_width), dtype=F32)
        sm_items = [
            ("post_norm", jnp.broadcast_to(post_norm[None, :], (R, H))),
            ("sh13_s", sh13_s),
            ("sh2_s", sh2_s),
            ("sh_gate", jnp.broadcast_to(sh_gate[None, :], (R, H))),
        ]
        if e_slots > 0:
            ecopy_tbl = jnp.zeros((R, E), dtype=F32)
            # Place a replica of expert 0 on rank 1's extra slot 0 (code=1 + 1*extra)
            extra = EW - ER
            ecopy_tbl = ecopy_tbl.at[:, 0].set(float(1 + 1 * extra))
            sm_items.append(("ecopy", ecopy_tbl))
        for name, val in sm_items:
            o, w = lay[name]
            sm = sm.at[:, 0, 0, o : o + w].set(val)

        w_by_name = {
            "router": jnp.broadcast_to(router[None, ...], (R, 1, H, E)).reshape(
                R, *wsh["router"][0]
            ),
            "sh13": sh13.reshape(R, *wsh["sh13"][0]),
            "sh2": sh2.reshape(R, *wsh["sh2"][0]),
            "e13": e13.reshape(R, *wsh["e13"][0]),
            "e2": e2.reshape(R, *wsh["e2"][0]),
            "es": jnp.concatenate([e13_s, e2_s], axis=-1).reshape(R, *wsh["es"][0]),
            "smalls": sm,
            "final_norm": jnp.zeros((R, *wsh["final_norm"][0]), dtype=F32),
        }

        def pallas_body(pos_ref, hid_ref, *rest):
            n_w, n_o = len(p.w_names), len(p.o_names)
            W = dict(zip(p.w_names, rest[:n_w]))
            O = dict(zip(p.o_names, rest[n_w : n_w + n_o]))  # noqa: E741
            scr = dict(zip(s_names, rest[n_w + n_o :]))
            me = lax.axis_index("tp").astype(I32)
            barrier = pltpu.get_barrier_semaphore()
            for off in range(1, R):
                pl.semaphore_signal(
                    barrier,
                    1,
                    device_id=(me ^ off,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(barrier, R - 1)

            ring = stream.TwoRing(cp.stream, W, scr, me, interpret=True)
            ar = comm.AllReduceRing(cp.comm, scr, me)
            ctx = MiniCtx(
                g=g,
                ex=ex,
                core_plan=cp,
                pos_ref=pos_ref,
                W=W,
                ST={},
                O=O,
                scr=scr,
                me=me,
                stream=ring,
                comm=ar,
            )

            for i in range(cp.stream.NBD):
                ring.fetch(jnp.int32(i))
            pltpu.make_async_copy(
                W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
            ).start()
            moe.prologue(ctx)
            scr["res"][...] = hid_ref[: ex.MB].astype(F32)
            scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
            moe.init_buffers(ctx)

            pltpu.make_async_copy(
                W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
            ).wait()
            ring.begin_layer(jnp.int32(0))
            ctx.apply_input_norm(moe.input_norm, jnp.int32(0))
            part_moe = moe.emit(ctx, jnp.int32(0), jnp.int32(0))
            ar.add_residual(ar.allreduce(0, part_moe, jnp.bool_(True)))
            ar.drain_sends()
            O["hidden"][: ex.MB] = scr["res"][...].astype(BF16)
            if ex.MB < ex.M:
                O["hidden"][ex.MB :] = jnp.zeros((ex.M - ex.MB, H), BF16)

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ] + [pl.BlockSpec(memory_space=pltpu.HBM) for _ in p.w_names]
        out_specs = [pl.BlockSpec(memory_space=pltpu.VMEM) for _ in p.o_names]
        out_shape = [jax.ShapeDtypeStruct(*osh[k]) for k in p.o_names]

        call = pl.pallas_call(
            pallas_body,
            out_shape=out_shape,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=[scratch[k] for k in s_names],
            compiler_params=pltpu.CompilerParams(collective_id=0),
            interpret=ex.interpret,
        )
        mesh = Mesh(devs[:R], ("tp",))
        step = jax.jit(
            jax.shard_map(
                lambda pos_in, hid_in, *ws: call(
                    pos_in, hid_in[0], *[w[0] for w in ws]
                )[0][None],
                mesh=mesh,
                in_specs=(P(), P("tp"), *(P("tp") for _ in p.w_names)),
                out_specs=P("tp"),
                check_vma=False,
            )
        )
        pos = jnp.array([1, 2, 3, 4], dtype=I32)
        hid_rep = jnp.broadcast_to(hid[None, ...], (R, ex.M, H))
        w_all = [w_by_name[k] for k in p.w_names]
        out = jax.block_until_ready(step(pos, hid_rep, *w_all))
        assert ipc.races is not None
        assert not ipc.races.races_found, (
            f"Data race detected in MoEBlock under dma_execution_mode={dma_mode!r}"
        )
        return np.asarray(out[:, : ex.B]).copy()

    out_wait = _run_mode("on_wait")
    out_eager = _run_mode("eager")
    np.testing.assert_array_equal(
        out_eager.view(np.uint8),
        out_wait.view(np.uint8),
    )
