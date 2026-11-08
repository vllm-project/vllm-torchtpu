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
"""Unit tests for `MoEModule` declarations and Pallas `interpret=True` execution."""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import pathlib
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tests.megakernel.megakernel_fakes import (
    TINY_TILES,
    FakeFFNModule,
    FakeGeometry,
    FakeMixerModule,
    MiniCtx,
)
from vllm_torchtpu.megakernel import modules, planner
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.modules import MoEModule
from vllm_torchtpu.megakernel.modules.moe import dedup, ring
from vllm_torchtpu.megakernel.modules.moe.context import MoEEmit
from vllm_torchtpu.megakernel.modules.moe.params import (
    MoEDebug,
    MoETunables,
    derive_moe_params,
)
from vllm_torchtpu.megakernel.modules.moe.routing import route
from vllm_torchtpu.megakernel.ops import (
    BF16,
    F32,
    FP8,
    I32,
    rmsnorm,
    round_bf16,
    round_up,
)
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec


def _build_moe_test_scratch(
    ms: ModelSpec, ex: ExecSpec, p: planner.Plan | None = None
) -> tuple[Any, dict[str, Any]]:
    """Build core + module scratch specs for standalone `MoEModule` Pallas tests."""
    if p is None:
        p = planner.plan(ms, ex)
    g = ms.config
    mb, sw = ex.MB, p.shared_widths
    nbe = ex.resolved_tiles.nbe
    scratch: dict[str, Any] = {
        "res": pltpu.VMEM((mb, g.H), F32),
        "xb": pltpu.VMEM((mb, g.H), BF16),
        "xb2": pltpu.VMEM((mb, round_up(sw["xb2"], 128)), BF16),
        "acc": pltpu.VMEM((mb, round_up(sw["acc"], 128)), F32),
        "ybuf": pltpu.VMEM((mb, round_up(sw["ybuf"], 128)), F32),
        "mix": pltpu.VMEM((mb, round_up(sw["mix"], 128)), F32),
        "part": pltpu.VMEM((mb, g.H), F32),
        "sm": pltpu.VMEM((2, 1, p.smalls_width), F32),
        "sm_sem": pltpu.SemaphoreType.DMA((2,)),
    }
    for k in p.static_kinds:
        _, n_k, dt_k = p.kinds[k]
        scratch[f"_w_{k}"] = pltpu.VMEM((p.tk[k], n_k), dt_k)
        scratch[f"_w_sem_{k}"] = pltpu.SemaphoreType.DMA(())
    for k in p.dynamic_kinds:
        _, n_k, dt_k = p.kinds[k]
        scratch[f"_ebuf_{k}"] = pltpu.VMEM((nbe, p.tk[k], n_k), dt_k)
        scratch[f"_esem_{k}"] = pltpu.SemaphoreType.DMA((nbe,))
    for m in ms.all_modules():
        reqs = m.scratch(g, ex, p)
        items = (
            list(reqs.items())
            if isinstance(reqs, dict)
            else [(r.name, r.spec) for r in reqs]
        )
        for name, spec in items:
            scratch[name] = spec
    cp = SimpleNamespace(
        plan=p,
        stream=SimpleNamespace(L=g.L, tk=p.tk, nt=p.nt, lay=p.smalls_layout),
        comm=None,
        module=p.module_params.__getitem__,
    )
    return cp, scratch


class _FakeMoEStream:
    """Minimal stream stub for `MoEModule.emit` (`dmatmul`, `mm`, `ecopy`, `eview`)."""

    def __init__(self, sp: Any, W: dict[str, Any], scr: dict[str, Any]) -> None:
        self.sp = sp
        self.W = W
        self.scr = scr

    def expert_stream(self) -> Any:
        return contextlib.nullcontext()

    def segment(self, *, lazy: bool = False) -> Any:
        del lazy
        return contextlib.nullcontext()

    def mm(self, a: Any, b: Any, dims: Any = (((1,), (0,)), ((), ()))) -> Any:
        return lax.dot_general(a, b.astype(BF16), dims, preferred_element_type=F32)

    def dmatmul(self, kind: str, x_ref: Any, n_dim: int) -> Any:
        tk = self.sp.tk[kind]
        nt = self.sp.nt[kind]
        x = x_ref[...]
        acc = jnp.zeros((x.shape[0], n_dim), dtype=F32)
        w_buf = self.scr[f"_w_{kind}"]
        w_sem = self.scr[f"_w_sem_{kind}"]
        for t in range(nt):
            cp = pltpu.make_async_copy(self.W[kind].at[0, t], w_buf, w_sem)
            cp.start()
            cp.wait()
            acc = acc + self.mm(x[:, t * tk : (t + 1) * tk], w_buf[...])
        self.scr["acc"][:, :n_dim] = acc
        return self.scr["acc"][:, :n_dim]

    def ecopy(self, kind: str, src: Any, bank: Any) -> Any:
        return pltpu.make_async_copy(
            src,
            self.scr[f"_ebuf_{kind}"].at[bank],
            self.scr[f"_esem_{kind}"].at[bank],
        )

    def eview(self, kind: str, bank: Any) -> Any:
        return self.scr[f"_ebuf_{kind}"].at[bank]


MODULES_DIR = pathlib.Path(modules.__file__).resolve().parent

# Row-count names that must never be compared to a literal inside `modules/`.
_ROW_NAMES = {"B", "NS", "MR", "MB", "M", "CB"}
# Remaining literal row-count comparisons are hardware / layout limits, not
# per-bucket tuning (each one with its reason).
_ALLOWED_EXPRS = {
    "MR <= 128": "lane top-k keeps all MR rows in one 128-lane vreg",
    "ex.NS <= 64": "sequence metadata packs 128 // NS lanes per (m | l) field",
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


def test_modules_have_no_row_count_literal_comparisons():
    """Modules read per-bucket tuning from `Tiles`, never `B <= 40`-style literals."""
    py_files = sorted(MODULES_DIR.rglob("*.py"))
    assert py_files
    found = []
    for path in py_files:
        rel = path.relative_to(MODULES_DIR).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            ops = [node.left, *node.comparators]
            for a, b in zip(ops, ops[1:]):
                pair = (_row_name(a) and _is_num(b)) or (_row_name(b) and _is_num(a))
                if pair:
                    expr = ast.unparse(node)
                    if expr not in _ALLOWED_EXPRS:
                        found.append(f"{rel}:{node.lineno}: {expr}")
    assert not found, "row-count literal comparisons in modules/:\n" + "\n".join(found)


def test_moe_module_declarations_and_emit_in_pallas_interpret(fake_ex: ExecSpec):
    """Verify `MoEModule` declarations drive `planner`/`vmem` and `emit` in Pallas."""
    g_decl = FakeGeometry(L=2, att_every=1, H=256)
    moe_decl = MoEModule()
    ms_decl = ModelSpec(
        "moe_decl_test",
        config=g_decl,
        layer_type=LayerType("lt", slots=(FakeMixerModule(), moe_decl)),
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

    cp_decl, scr_decl = _build_moe_test_scratch(ms_decl, fake_ex, p_decl)
    assert (
        "wn" in scr_decl
        and "elist" in scr_decl
        and "es" in scr_decl
        and "st" in scr_decl
    )
    assert cp_decl.module("moe").EW == g_decl.E // fake_ex.R

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
    moe = MoEModule()
    ms = ModelSpec("moe_exec", config=g, layer_type=LayerType("lt", slots=(moe,)))
    cp, scratch = _build_moe_test_scratch(ms, ex)
    p = cp.plan
    wsh = planner.weight_shapes(ms, ex, p)
    osh = planner.kernel_out_shapes(ms, ex, p)
    s_names = list(scratch.keys())

    assert moe.input_norm == "post_norm"

    def pallas_body(pos_ref, hid_ref, *rest):
        n_w, n_o = len(p.w_names), len(p.o_names)
        W = dict(zip(p.w_names, rest[:n_w]))
        out_refs = dict(zip(p.o_names, rest[n_w : n_w + n_o]))
        scr = dict(zip(s_names, rest[n_w + n_o :]))
        me = jnp.int32(0)
        ring = _FakeMoEStream(cp.stream, W, scr)
        ctx = MiniCtx(
            g=g,
            ex=ex,
            core_plan=cp,
            pos_ref=pos_ref,
            W=W,
            ST={},
            O=out_refs,
            scr=scr,
            me=me,
            stream=ring,
            comm=None,
        )

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
        moe.on_layer_start(ctx, jnp.int32(0), jnp.int32(0))
        ctx.apply_input_norm(moe.input_norm, jnp.int32(0))
        part_moe = moe.emit(ctx, jnp.int32(0), jnp.int32(0))
        scr["res"][...] = scr["res"][...] + part_moe.astype(BF16).astype(F32)
        out_refs["hidden"][: ex.MB] = scr["res"][...].astype(BF16)
        if ex.MB < ex.M:
            out_refs["hidden"][ex.MB :] = jnp.zeros((ex.M - ex.MB, g.H), BF16)

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
def test_moe_module_numerical_parity(moe_tp: bool, R: int, pipe: bool):
    """Compare `MoEModule.emit` in Pallas `interpret=True` against a JAX reference."""
    devs = jax.devices("cpu")
    assert len(devs) >= R

    H, E, TOPK, IR, ISR = 128, 128, 2, 128, 128
    ER = E // R
    EW = E if moe_tp else ER
    tiles = dataclasses.replace(TINY_TILES, war=1, mb8=2).with_modules(
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
    moe = MoEModule()
    ms = ModelSpec("moe_num", config=g, layer_type=LayerType("lt", slots=(moe,)))
    cp, scratch = _build_moe_test_scratch(ms, ex)
    assert cp.module("moe").PIPE is pipe
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
        out_refs = dict(zip(p.o_names, rest[n_w : n_w + n_o]))
        scr = dict(zip(s_names, rest[n_w + n_o :]))
        me = lax.axis_index("tp").astype(I32) if R > 1 else jnp.int32(0)
        ring = _FakeMoEStream(cp.stream, W, scr)
        ctx = MiniCtx(
            g=g,
            ex=ex,
            core_plan=cp,
            pos_ref=pos_ref,
            W=W,
            ST={},
            O=out_refs,
            scr=scr,
            me=me,
            stream=ring,
            comm=None,
        )

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
        moe.on_layer_start(ctx, jnp.int32(0), jnp.int32(0))
        ctx.apply_input_norm(moe.input_norm, jnp.int32(0))
        part_moe = moe.emit(ctx, jnp.int32(0), jnp.int32(0))
        out_refs["hidden"][: ex.MB] = part_moe.astype(BF16)
        if ex.MB < ex.M:
            out_refs["hidden"][ex.MB :] = jnp.zeros((ex.M - ex.MB, H), BF16)

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
        (part_out,) = jax.jit(call)(pos, hid, *[w[0] for w in w_all])
        part_ranks = part_out[None, : ex.B]
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
        part_ranks = step(pos, hid_rep, *w_all)[:, : ex.B]

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
            round_bf16(
                round_bf16(jax.nn.silu(round_bf16(g13[:, :ISR])))
                * round_bf16(g13[:, ISR:])
            )
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
                    round_bf16(
                        round_bf16(jax.nn.silu(round_bf16(h_e[:, :IR])))
                        * round_bf16(h_e[:, IR:])
                    )
                    .astype(BF16)
                    .astype(F32)
                )
                y_e = round_bf16((mid_e @ e2[r, w_slot].astype(F32)) * e2_s[r, w_slot])
                part_r = part_r.at[b : b + 1].add(top_w[b, k] * y_e)
        partials.append(part_r.astype(BF16).astype(F32))

    for r in range(R):
        assert jnp.allclose(
            part_ranks[r].astype(F32), partials[r], atol=2e-2, rtol=2e-2
        )
    ref_out = (res + jnp.sum(jnp.stack(partials, axis=0), axis=0)).astype(BF16)
    actual_out = (res + jnp.sum(part_ranks.astype(F32), axis=0)).astype(BF16)
    assert jnp.allclose(
        actual_out.astype(F32), ref_out.astype(F32), atol=2e-2, rtol=2e-2
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
    """Run MoEModule under Pallas vector-clock race detection (`on_wait` vs `eager`)."""
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
    ).with_modules(
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
    moe = MoEModule()
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
        cp, scratch = _build_moe_test_scratch(ms, ex)
        assert cp.module("moe").route_vec is route_vec
        if pipe_ep:
            assert cp.module("moe").PIPE_EP is True
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
            out_refs = dict(zip(p.o_names, rest[n_w : n_w + n_o]))
            scr = dict(zip(s_names, rest[n_w + n_o :]))
            me = lax.axis_index("tp").astype(I32)
            ring = _FakeMoEStream(cp.stream, W, scr)
            ctx = MiniCtx(
                g=g,
                ex=ex,
                core_plan=cp,
                pos_ref=pos_ref,
                W=W,
                ST={},
                O=out_refs,
                scr=scr,
                me=me,
                stream=ring,
                comm=None,
            )

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
            # `MoEModule.prologue` starts the async DMA into `scr["ecopy"].at[0]`
            # on `scr["ec_sem"].at[0]`; `kernel.py` calls `b.on_layer_start`
            # inside `smalls_sync` at the start of every layer to wait on
            # `scr["ec_sem"].at[slot]` before `po2c_pass1` reads `scr["ecopy"]`.
            moe.on_layer_start(ctx, jnp.int32(0), jnp.int32(0))
            ctx.apply_input_norm(moe.input_norm, jnp.int32(0))
            part_moe = moe.emit(ctx, jnp.int32(0), jnp.int32(0))
            out_refs["hidden"][: ex.MB] = part_moe.astype(BF16)
            if ex.MB < ex.M:
                out_refs["hidden"][ex.MB :] = jnp.zeros((ex.M - ex.MB, H), BF16)

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
            f"Data race detected in MoEModule under dma_execution_mode={dma_mode!r}"
        )
        return np.asarray(out[:, : ex.B]).copy()

    out_wait = _run_mode("on_wait")
    out_eager = _run_mode("eager")
    np.testing.assert_array_equal(
        out_eager.view(np.uint8),
        out_wait.view(np.uint8),
    )


def _sync_copy(src, dst, sem) -> None:
    pltpu.make_async_copy(src, dst, sem).start()
    pltpu.make_async_copy(src, dst, sem).wait()


@pytest.mark.parametrize(
    ("ecopy", "route_vec", "ecopy_opt"),
    [
        (False, False, 0),
        (True, False, 0),
        (True, False, 1),
        (True, True, 0),
    ],
    ids=["local_dedup", "ecopy_opt0", "ecopy_opt1", "ecopy_route_vec"],
)
def test_moe_routing_and_ring_helpers_in_pallas_interpret(
    ecopy: bool, route_vec: bool, ecopy_opt: int
):
    """Execute `MoEEmit.build`, `route_topk`, `dedup_*`, `route`, and `ring.py`."""
    R, B, H, E, TOPK, IR, ISR = 2, 4, 256, 128, 2, 128, 128
    e_slots = 70 if ecopy else 0
    tiles = dataclasses.replace(TINY_TILES, bank=512 << 10, ebank=256 << 10, nbe=4)
    ex = ExecSpec(B=B, T=1, R=R, ctx=128, stack_e=True, tiles=tiles, interpret=True)
    g = FakeGeometry(
        L=1,
        att_every=1,
        H=H,
        NH=4,
        NKV=2,
        E=E,
        TOPK=TOPK,
        I=IR,
        IS=ISR * R,
        e_slots=e_slots,
    )
    ms = ModelSpec(
        "moe_r_ring",
        config=g,
        layer_type=LayerType("lt", slots=(FakeMixerModule(), FakeFFNModule())),
    )
    cp, scratch = _build_moe_test_scratch(ms, ex)
    p = cp.plan
    tt = MoETunables(
        topk=True, topk_lanes=True, route_vec=route_vec, ecopy_opt=ecopy_opt
    )
    mp = derive_moe_params(
        g, ex, p, tt, MoEDebug(), moe_tp=False, e_slots=e_slots, module_name="fake_moe"
    )
    assert mp.ECOPY is ecopy and mp.route_vec is route_vec
    nb_picks = round_up(B * TOPK, 128)
    scratch["hitv"] = pltpu.VMEM((ex.MR, 128), I32)
    scratch["hits"] = pltpu.SMEM((ex.MR, 128), I32)
    scratch["elist"] = pltpu.SMEM((nb_picks,), I32)
    scratch["seen"] = pltpu.SMEM((8, 128), I32)
    scratch["misc_sem"] = pltpu.SemaphoreType.DMA(())
    scratch["st"] = pltpu.SMEM((1,), I32)
    scratch["es"] = pltpu.VMEM((mp.NSS, 1, 2 * IR + H), F32)
    scratch["es_sem"] = pltpu.SemaphoreType.DMA((mp.NSS,))
    if ecopy:
        scratch["ecopy"] = pltpu.SMEM((2, 1, E), F32)
        scratch["ec_sem"] = pltpu.SemaphoreType.DMA((2,))
        for k_ec in ("wlist", "cplist", "cpchoice"):
            scratch[k_ec] = pltpu.SMEM((nb_picks,), I32)
        scratch["rload"] = pltpu.SMEM((R,), I32)
        if ecopy_opt == 1:
            scratch["ulist"] = pltpu.SMEM((nb_picks,), I32)

    wsh = planner.weight_shapes(ms, ex, p)
    s_names = list(scratch.keys())

    def pallas_body(pexp_ref, *rest):
        n_w = len(p.w_names)
        W = dict(zip(p.w_names, rest[:n_w]))
        wn_out_ref, nhit_out_ref, e13_out_ref = rest[n_w : n_w + 3]
        scr = dict(zip(s_names, rest[n_w + 3 :]))
        me = jnp.int32(0)
        two_ring = _FakeMoEStream(cp.stream, W, scr)
        ctx = MiniCtx(g, ex, cp, None, W, {}, {}, scr, me, stream=two_ring, comm=None)
        scr["hitv"][...] = jnp.zeros((ex.MR, 128), I32)
        _sync_copy(scr["hitv"].at[pl.ds(0, 8)], scr["seen"], scr["misc_sem"])
        _sync_copy(W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0])
        if ecopy:
            o_ec = cp.stream.lay["ecopy"][0]
            _sync_copy(
                W["smalls"].at[0, :, pl.ds(o_ec, E)],
                scr["ecopy"].at[0],
                scr["ec_sem"].at[0],
            )

        m = MoEEmit.build(ctx, mp, MoEDebug())
        with two_ring.expert_stream():
            n_hit = route(m, jnp.int32(0), pexp_ref[...])
            ring.es_wait(m, jnp.int32(0))
            t0 = ring.etake(m, "e13", 0)
            e13_out_ref[...] = t0[...]
            ring.egive(m, 0, jnp.int32(0), t=0)
            t1 = ring.etake(m, "e2", 1)
            _ = t1[...]
            ring.egive_group(m, [(1, 1)], jnp.int32(0))
        wn_out_ref[...] = scr["wn"][...]
        nhit_out_ref[0] = n_hit

    call = pl.pallas_call(
        pallas_body,
        out_shape=[
            jax.ShapeDtypeStruct((ex.MR, E), F32),
            jax.ShapeDtypeStruct((1,), I32),
            jax.ShapeDtypeStruct((H, 2 * IR), wsh["e13"][1]),
        ],
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)]
        + [pl.BlockSpec(memory_space=pltpu.HBM) for _ in p.w_names],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        scratch_shapes=[scratch[k] for k in s_names],
        interpret=pltpu.InterpretParams(),
    )

    pexp = jnp.ones((ex.MR, E), dtype=F32) * 0.01
    pexp = pexp.at[0, 2].set(5.0).at[0, 5].set(4.0)
    pexp = pexp.at[1, 3].set(5.0).at[1, 66].set(4.0)
    pexp = pexp.at[2, 2].set(5.0).at[2, 3].set(4.0)
    pexp = pexp.at[3, 5].set(5.0).at[3, 66].set(4.0)

    w_in = {k: jnp.zeros(wsh[k][0], dtype=wsh[k][1]) for k in p.w_names}
    if ecopy:
        o_ec, w_ec = cp.stream.lay["ecopy"]
        ec_tbl = jnp.zeros((w_ec,), dtype=F32).at[66].set(1.0)
        w_in["smalls"] = w_in["smalls"].at[0, 0, o_ec : o_ec + w_ec].set(ec_tbl)

    wn_out, nhit_out, e13_out = jax.jit(call)(pexp, *[w_in[k] for k in p.w_names])
    assert int(nhit_out[0]) >= 3
    np.testing.assert_allclose(jnp.sum(wn_out[:B], axis=-1), jnp.ones((B,)), atol=1e-5)
    assert e13_out.shape == (H, 2 * IR)
    assert callable(dedup.po2c)
