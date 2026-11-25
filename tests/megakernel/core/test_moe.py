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

from tests.megakernel.core.megakernel_fakes import (
    TINY_TILES,
    FakeGeometry,
    FakeMixerModule,
    MiniCtx,
)
from vllm_torchtpu.megakernel.core import modules, planner
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import DebugOptions, ExecSpec
from vllm_torchtpu.megakernel.core.modules import MoEDebug, MoEModule
from vllm_torchtpu.megakernel.core.modules.moe.params import (
    MoETunables,
    derive_moe_params,
)
from vllm_torchtpu.megakernel.core.ops import (
    BF16,
    F32,
    FP8,
    I32,
    rmsnorm,
    round_bf16,
    round_up,
)
from vllm_torchtpu.megakernel.core.spec import ExtraInput, LayerType, ModelSpec, Spec


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
        self.cur_layer: Any = jnp.int32(0)

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
            cp = pltpu.make_async_copy(self.W[kind].at[self.cur_layer, t], w_buf, w_sem)
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


def _run_moe_numerical_parity_case(
    *,
    moe_tp: bool,
    R: int,
    pipe: bool,
    pipe_ep: bool = False,
    moe_unroll: bool = False,
    moe_group_rows: int = 40,
    H: int = 128,
    IR: int = 128,
    ebank: int = 256 << 10,
    nbe: int = 4,
    e_slots: int = 0,
    L: int = 1,
    stack_e: bool = True,
    weight_dtypes: dict[str, str] | None = None,
    interpret: bool = True,
) -> None:
    """Compare `MoEModule.emit` in Pallas against `MoEModule.reference` / EPLB ref."""
    devs = jax.devices("cpu") if interpret else jax.devices("tpu")
    assert len(devs) >= R

    E, TOPK, ISR = 128, 2, 128
    ER = E // R
    EW = e_slots if e_slots > 0 else (E if moe_tp else ER)
    EXTRA = EW - ER
    ecopy = not moe_tp and e_slots > ER
    tiles = dataclasses.replace(
        TINY_TILES, bank=512 << 10, ebank=ebank, nbe=nbe, war=1, mb8=2
    ).with_modules(
        {
            "moe": {
                "pipe": pipe,
                "pipe_ep": pipe_ep,
                "moe_unroll": moe_unroll,
                "moe_group_rows": moe_group_rows,
            }
        }
    )
    ex = ExecSpec(
        B=2, T=1, R=R, ctx=128, stack_e=stack_e, tiles=tiles, interpret=interpret
    )
    g = FakeGeometry(
        L=L,
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
        e_slots=e_slots,
    )
    moe = MoEModule(weight_dtypes=weight_dtypes)
    sh13_fp8 = moe.weight_dtypes["sh13"] == "fp8"
    sh2_fp8 = moe.weight_dtypes["sh2"] == "fp8"
    e_fp8 = moe.weight_dtypes["e13"] == "fp8"
    ms = ModelSpec("moe_num", config=g, layer_type=LayerType("lt", slots=(moe,)))
    cp, scratch = _build_moe_test_scratch(ms, ex)
    mp = cp.module("moe")
    assert mp.PIPE is pipe
    assert mp.PIPE_EP is pipe_ep
    assert mp.ECOPY is ecopy
    assert mp.stack_e is stack_e
    assert mp.unroll is bool(moe_unroll or pipe_ep)
    assert mp.group_rel is (moe_group_rows >= ex.B)
    if H == 512 and IR == 256 and ebank == (64 << 10):
        assert mp.T13 > 1 and mp.T2 > 1
    if pipe_ep:
        assert mp.T13 == 2 * mp.T2
    p = cp.plan
    w_decls = {w.name: w for w in moe.weights(g, ex)}
    assert w_decls["e13"].layout == ("stacked" if stack_e else "per_layer")
    if not e_fp8:
        assert "es" not in p.w_names and "es" not in scratch
    wsh = planner.weight_shapes(ms, ex, p)
    s_names = list(scratch.keys())

    with jax.default_device(jax.devices("cpu")[0]):
        keys = jax.random.split(jax.random.PRNGKey(101 + L * 17 + e_slots), 12)
        hid = jax.random.normal(keys[0], (ex.M, H), dtype=BF16) * 0.5
        post_norm = jax.random.uniform(
            keys[1], (L, H), dtype=F32, minval=-0.2, maxval=0.2
        )
        router_noise = jax.random.normal(keys[2], (L, H, E), dtype=F32) * 0.02
        sh_gate = jax.random.normal(keys[3], (L, H), dtype=F32) * 0.2
        sh13 = (
            jax.random.normal(keys[4], (R, L, H, 2 * ISR), dtype=F32)
            * (0.4 if sh13_fp8 else 0.12)
        ).astype(FP8 if sh13_fp8 else BF16)
        sh13_s = jax.random.uniform(
            keys[5], (R, L, 2 * ISR), dtype=F32, minval=0.05, maxval=0.15
        )
        sh2 = (
            jax.random.normal(keys[6], (R, L, ISR, H), dtype=F32)
            * (0.4 if sh2_fp8 else 0.12)
        ).astype(FP8 if sh2_fp8 else BF16)
        sh2_s = jax.random.uniform(
            keys[7], (R, L, H), dtype=F32, minval=0.05, maxval=0.15
        )
        e13 = (
            jax.random.normal(keys[8], (R, L, EW, H, 2 * IR), dtype=F32)
            * (0.4 if e_fp8 else 0.15)
        ).astype(FP8 if e_fp8 else BF16)
        e13_s = jax.random.uniform(
            keys[9], (R, L, EW, 2 * IR), dtype=F32, minval=0.15, maxval=0.35
        )
        e2 = (
            jax.random.normal(keys[10], (R, L, EW, IR, H), dtype=F32)
            * (0.4 if e_fp8 else 0.15)
        ).astype(FP8 if e_fp8 else BF16)
        e2_s = jax.random.uniform(
            keys[11], (R, L, EW, H), dtype=F32, minval=0.15, maxval=0.35
        )

        # Construct per-layer router weights (and replica weights + `ecopy` table
        # when `ecopy=True`). When `ecopy=True`, both tokens pick from rank 0's
        # experts `{2, 5, 8}` (`token 0 -> (2, 5)`, `token 1 -> (2, 8)`), and each
        # layer replicates a different rank-0 expert onto rank 1 (`layer 0`:
        # expert 5 -> rank 1 slot `ER + 0`; `layer 1`: expert 8 -> rank 1 slot
        # `ER + 1`), exercising the `on_layer_start` double-buffer (`slot=0` then
        # `slot=1`).
        res = hid[: ex.B].astype(F32)
        routers = []
        ecopy_tbl = jnp.zeros((R, L, E), dtype=F32)
        partials_by_layer: list[list[jax.Array]] = []
        ref_outs_by_layer: list[jax.Array] = []
        mod_ref_deltas: list[jax.Array | None] = []

        for lyr in range(L):
            x_moe = rmsnorm(
                res,
                post_norm[lyr],
                g.eps,
                zero_centered=g.zero_centered_norm,
            )
            xb_moe = x_moe.astype(BF16).astype(F32)
            x0, x1 = xb_moe[0], xb_moe[1]
            u0 = x0 - (jnp.dot(x0, x1) / jnp.dot(x1, x1)) * x1
            u1 = x1 - (jnp.dot(x1, x0) / jnp.dot(x0, x0)) * x0
            d0 = u0 / jnp.dot(x0, u0)
            d1 = u1 / jnp.dot(x1, u1)

            r_l = router_noise[lyr]
            if ecopy:
                r_l = r_l.at[:, 2].add((d0 + d1) * 4.0)
                r_l = r_l.at[:, 5].add(d0 * 3.0)
                r_l = r_l.at[:, 8].add(d1 * 3.0)
                rep_e = 5 if (lyr % 2 == 0) else 8
                rep_extra = lyr % EXTRA
                rep_slot = ER + rep_extra
                rep_code = 1 + 1 * EXTRA + rep_extra
                ecopy_tbl = ecopy_tbl.at[:, lyr, rep_e].set(float(rep_code))
                e13 = e13.at[1, lyr, rep_slot].set(e13[0, lyr, rep_e])
                e13_s = e13_s.at[1, lyr, rep_slot].set(e13_s[0, lyr, rep_e])
                e2 = e2.at[1, lyr, rep_slot].set(e2[0, lyr, rep_e])
                e2_s = e2_s.at[1, lyr, rep_slot].set(e2_s[0, lyr, rep_e])
            else:
                e_r0_a, e_r0_b = 2, 5
                e_r1_a, e_r1_b = (R - 1) * ER + 3, (R - 1) * ER + 7
                r_l = r_l.at[:, e_r0_a].add(d0 * 4.0)
                r_l = r_l.at[:, e_r0_b].add(d0 * 3.0)
                r_l = r_l.at[:, e_r1_a].add(d1 * 4.0)
                r_l = r_l.at[:, e_r1_b].add(d1 * 3.0)
                rep_e = -1
                rep_slot = -1

            router_l = r_l.astype(BF16)
            routers.append(router_l)

            logits = xb_moe @ router_l.astype(F32)
            probs = jax.nn.softmax(logits, axis=-1)
            top_vals, top_idx = lax.top_k(probs, TOPK)
            if ecopy:
                assert list(map(int, top_idx[0])) == [2, 5]
                assert list(map(int, top_idx[1])) == [2, 8]
            top_w = top_vals / jnp.sum(top_vals, axis=-1, keepdims=True)
            sg = jax.nn.sigmoid(
                jnp.sum(x_moe * sh_gate[lyr][None, :], axis=-1, keepdims=True)
            )
            partials_l: list[jax.Array] = []
            for r in range(R):
                g13 = xb_moe @ sh13[r, lyr].astype(F32)
                if sh13_fp8:
                    g13 = g13 * sh13_s[r, lyr]
                mid_sh = (
                    round_bf16(
                        round_bf16(jax.nn.silu(round_bf16(g13[:, :ISR])))
                        * round_bf16(g13[:, ISR:])
                    )
                    .astype(BF16)
                    .astype(F32)
                )
                ysh = mid_sh @ sh2[r, lyr].astype(F32)
                if sh2_fp8:
                    ysh = ysh * sh2_s[r, lyr]
                part_r = ysh * sg
                for b in range(ex.B):
                    for k in range(TOPK):
                        e = int(top_idx[b, k])
                        if ecopy and e == rep_e:
                            owner_r, w_slot = 1, rep_slot
                        else:
                            owner_r = r if moe_tp else (e // ER)
                            w_slot = e if moe_tp else (e - owner_r * ER)
                        if owner_r != r:
                            continue
                        h_e = xb_moe[b : b + 1] @ e13[r, lyr, w_slot].astype(F32)
                        if e_fp8:
                            h_e = h_e * e13_s[r, lyr, w_slot]
                        mid_e = (
                            round_bf16(
                                round_bf16(jax.nn.silu(round_bf16(h_e[:, :IR])))
                                * round_bf16(h_e[:, IR:])
                            )
                            .astype(BF16)
                            .astype(F32)
                        )
                        y_raw = mid_e @ e2[r, lyr, w_slot].astype(F32)
                        y_e = round_bf16(
                            y_raw * e2_s[r, lyr, w_slot] if e_fp8 else y_raw
                        )
                        part_r = part_r.at[b : b + 1].add(top_w[b, k] * y_e)
                partials_l.append(part_r.astype(BF16).astype(F32))
            partials_by_layer.append(partials_l)
            ref_outs_by_layer.append(
                (res + jnp.sum(jnp.stack(partials_l, axis=0), axis=0)).astype(BF16)
            )
            if not ecopy:
                ref_params: dict[str, Any] = {
                    "router": router_l,
                    "sh_gate": jnp.broadcast_to(sh_gate[lyr][None, :], (R, H)),
                    "sh13": sh13[:, lyr],
                    "sh2": sh2[:, lyr],
                    "e13": e13[:, lyr],
                    "e2": e2[:, lyr],
                }
                if sh13_fp8:
                    ref_params["sh13_s"] = sh13_s[:, lyr]
                if sh2_fp8:
                    ref_params["sh2_s"] = sh2_s[:, lyr]
                if e_fp8:
                    ref_params["e13_s"] = e13_s[:, lyr]
                    ref_params["e2_s"] = e2_s[:, lyr]
                ref_delta, ref_st = moe.reference(ref_params, x_moe, g=g, ex=ex)
                assert ref_st is None
                mod_ref_deltas.append(ref_delta)
            else:
                mod_ref_deltas.append(None)

        router = jnp.stack(routers, axis=0)
        lay = p.smalls_layout
        sm = jnp.zeros((R, L, 1, p.smalls_width), dtype=F32)
        sm_items = [
            ("post_norm", jnp.broadcast_to(post_norm[None, ...], (R, L, H))),
            ("sh_gate", jnp.broadcast_to(sh_gate[None, ...], (R, L, H))),
        ]
        if sh13_fp8:
            sm_items.append(("sh13_s", sh13_s))
        if sh2_fp8:
            sm_items.append(("sh2_s", sh2_s))
        if ecopy:
            sm_items.append(("ecopy", ecopy_tbl))
        for name, val in sm_items:
            o, w = lay[name]
            sm = sm.at[:, :, 0, o : o + w].set(val)

        w_by_name: dict[str, Any] = {
            "router": jnp.broadcast_to(router[None, ...], (R, L, H, E)).reshape(
                R, *wsh["router"][0]
            ),
            "sh13": sh13.reshape(R, *wsh["sh13"][0]),
            "sh2": sh2.reshape(R, *wsh["sh2"][0]),
            "e13": (
                e13.reshape(R, *wsh["e13"][0])
                if stack_e
                else tuple(
                    e13[:, lyr].reshape(R, *wsh["e13"][lyr].shape) for lyr in range(L)
                )
            ),
            "e2": (
                e2.reshape(R, *wsh["e2"][0])
                if stack_e
                else tuple(
                    e2[:, lyr].reshape(R, *wsh["e2"][lyr].shape) for lyr in range(L)
                )
            ),
            "smalls": sm,
            "final_norm": jnp.zeros((R, *wsh["final_norm"][0]), dtype=F32),
        }
        if e_fp8:
            w_by_name["es"] = jnp.concatenate([e13_s, e2_s], axis=-1).reshape(
                R, *wsh["es"][0]
            )

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

        for lyr in range(L):
            slot = lyr % 2
            l_i32 = jnp.int32(lyr)
            s_i32 = jnp.int32(slot)
            ring.cur_layer = l_i32
            pltpu.make_async_copy(
                W["smalls"].at[lyr], scr["sm"].at[slot], scr["sm_sem"].at[slot]
            ).wait()
            if lyr + 1 < L:
                pltpu.make_async_copy(
                    W["smalls"].at[lyr + 1],
                    scr["sm"].at[1 - slot],
                    scr["sm_sem"].at[1 - slot],
                ).start()
            moe.on_layer_start(ctx, l_i32, s_i32)
            ctx.apply_input_norm(moe.input_norm, s_i32)
            part_moe = moe.emit(ctx, l_i32, s_i32)
            out_refs["hidden"][lyr, : ex.MB] = part_moe.astype(BF16)
            if ex.MB < ex.M:
                out_refs["hidden"][lyr, ex.MB :] = jnp.zeros((ex.M - ex.MB, H), BF16)

    def _is_tupled(v: Any) -> bool:
        return isinstance(v, tuple) and bool(v) and isinstance(v[0], Spec)

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.SMEM),
        pl.BlockSpec(memory_space=pltpu.VMEM),
    ] + [
        tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in wsh[k])
        if _is_tupled(wsh[k])
        else pl.BlockSpec(memory_space=pltpu.HBM)
        for k in p.w_names
    ]
    out_specs = [pl.BlockSpec(memory_space=pltpu.VMEM) for _ in p.o_names]
    out_shape = [jax.ShapeDtypeStruct((L, ex.M, H), BF16)]

    call = pl.pallas_call(
        pallas_body,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        interpret=pltpu.InterpretParams() if interpret else False,
    )

    pos = jnp.array([4, 8], dtype=I32)
    w_all = [w_by_name[k] for k in p.w_names]
    if R == 1:
        w_r0 = [tuple(t[0] for t in w) if isinstance(w, tuple) else w[0] for w in w_all]
        (part_out,) = jax.jit(call)(pos, hid, *w_r0)
        part_ranks = part_out[None, :, : ex.B]
    else:
        mesh = Mesh(devs[:R], ("tp",))
        w_part = tuple(
            tuple(P("tp") for _ in wsh[k]) if _is_tupled(wsh[k]) else P("tp")
            for k in p.w_names
        )
        step = jax.jit(
            jax.shard_map(
                lambda pos_in, hid_in, *ws: call(
                    pos_in,
                    hid_in[0],
                    *[
                        tuple(t[0] for t in w) if isinstance(w, tuple) else w[0]
                        for w in ws
                    ],
                )[0][None],
                mesh=mesh,
                in_specs=(P(), P("tp"), *w_part),
                out_specs=P("tp"),
                check_vma=False,
            )
        )
        hid_rep = jnp.broadcast_to(hid[None, ...], (R, ex.M, H))
        part_ranks = step(pos, hid_rep, *w_all)[:, :, : ex.B]

    with jax.default_device(jax.devices("cpu")[0]):
        part_ranks = jnp.asarray(np.asarray(part_ranks))
        for lyr in range(L):
            for r in range(R):
                assert float(jnp.max(jnp.abs(partials_by_layer[lyr][r]))) >= 0.08
                assert jnp.allclose(
                    part_ranks[r, lyr].astype(F32),
                    partials_by_layer[lyr][r],
                    atol=2e-2,
                    rtol=2e-2,
                )
            summed_delta = jnp.sum(part_ranks[:, lyr].astype(F32), axis=0)
            if mod_ref_deltas[lyr] is not None:
                np.testing.assert_allclose(
                    summed_delta,
                    mod_ref_deltas[lyr].astype(F32),
                    atol=2e-2,
                    rtol=2e-2,
                )
            actual_out = (res + summed_delta).astype(BF16)
            assert jnp.allclose(
                actual_out.astype(F32),
                ref_outs_by_layer[lyr].astype(F32),
                atol=2e-2,
                rtol=2e-2,
            )


@pytest.mark.parametrize(
    (
        "moe_tp",
        "R",
        "pipe",
        "pipe_ep",
        "moe_unroll",
        "moe_group_rows",
        "H",
        "IR",
        "ebank",
        "nbe",
        "e_slots",
        "L",
    ),
    [
        (False, 1, False, False, False, 40, 128, 128, 256 << 10, 4, 0, 1),
        (True, 1, False, False, False, 40, 128, 128, 256 << 10, 4, 0, 1),
        (True, 1, True, False, False, 40, 128, 128, 256 << 10, 4, 0, 1),
        (False, 2, False, False, False, 40, 128, 128, 256 << 10, 4, 0, 1),
        (False, 2, True, False, False, 40, 128, 128, 256 << 10, 4, 0, 1),
        (True, 2, True, False, False, 40, 128, 128, 256 << 10, 4, 0, 1),
        (False, 2, True, False, False, 40, 128, 128, 256 << 10, 4, 70, 2),
        (False, 1, False, True, True, 40, 256, 128, 32 << 10, 6, 0, 1),
        (False, 1, False, False, False, 40, 512, 256, 64 << 10, 8, 0, 1),
        (False, 1, False, False, True, 40, 512, 256, 64 << 10, 8, 0, 1),
        (False, 1, False, False, True, 0, 512, 256, 64 << 10, 8, 0, 1),
    ],
    ids=[
        "ep_r1_unpiped_1t",
        "tp_r1_unpiped_1t",
        "tp_r1_pipe",
        "ep_r2_unpiped_1t",
        "ep_r2_pipe",
        "tp_r2_pipe",
        "ep_r2_ecopy_l2",
        "ep_r1_pipe_ep",
        "ep_r1_multitile_rolled",
        "ep_r1_multitile_unroll_group_rel",
        "ep_r1_multitile_unroll_no_group_rel",
    ],
)
def test_moe_module_numerical_parity(
    moe_tp: bool,
    R: int,
    pipe: bool,
    pipe_ep: bool,
    moe_unroll: bool,
    moe_group_rows: int,
    H: int,
    IR: int,
    ebank: int,
    nbe: int,
    e_slots: int,
    L: int,
):
    """Compare `MoEModule.emit` in Pallas `interpret=True` against a JAX reference."""
    _run_moe_numerical_parity_case(
        moe_tp=moe_tp,
        R=R,
        pipe=pipe,
        pipe_ep=pipe_ep,
        moe_unroll=moe_unroll,
        moe_group_rows=moe_group_rows,
        H=H,
        IR=IR,
        ebank=ebank,
        nbe=nbe,
        e_slots=e_slots,
        L=L,
        interpret=True,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    (
        "moe_tp",
        "R",
        "pipe",
        "pipe_ep",
        "moe_unroll",
        "moe_group_rows",
        "H",
        "IR",
        "ebank",
        "nbe",
        "e_slots",
        "L",
    ),
    [
        (True, 1, True, False, False, 40, 128, 128, 256 << 10, 4, 0, 1),
        (False, 1, False, True, True, 40, 256, 128, 32 << 10, 6, 0, 1),
        (False, 1, False, False, False, 40, 512, 256, 64 << 10, 8, 0, 1),
        (False, 1, False, False, True, 40, 512, 256, 64 << 10, 8, 0, 1),
        (False, 1, False, False, True, 0, 512, 256, 64 << 10, 8, 0, 1),
        (False, 2, True, False, False, 40, 128, 128, 256 << 10, 4, 70, 2),
    ],
    ids=[
        "tpu_pipe",
        "tpu_pipe_ep",
        "tpu_multitile_rolled",
        "tpu_multitile_unroll_group_rel",
        "tpu_multitile_unroll_no_group_rel",
        "tpu_ep_r2_ecopy_l2",
    ],
)
def test_moe_module_numerical_parity_on_tpu(
    moe_tp: bool,
    R: int,
    pipe: bool,
    pipe_ep: bool,
    moe_unroll: bool,
    moe_group_rows: int,
    H: int,
    IR: int,
    ebank: int,
    nbe: int,
    e_slots: int,
    L: int,
):
    """Run MoE numerical parity on TPU (`interpret=False`)."""
    try:
        tpu_devs = jax.devices("tpu")
    except RuntimeError:
        tpu_devs = []
    if len(tpu_devs) < R:
        pytest.skip(f"{R} TPU device(s) required for interpret=False Pallas execution")
    with contextlib.nullcontext() if R > 1 else jax.default_device(tpu_devs[0]):
        _run_moe_numerical_parity_case(
            moe_tp=moe_tp,
            R=R,
            pipe=pipe,
            pipe_ep=pipe_ep,
            moe_unroll=moe_unroll,
            moe_group_rows=moe_group_rows,
            H=H,
            IR=IR,
            ebank=ebank,
            nbe=nbe,
            e_slots=e_slots,
            L=L,
            interpret=False,
        )


@pytest.mark.parametrize(
    (
        "case_name",
        "pipe",
        "pipe_ep",
        "moe_unroll",
        "moe_group_rows",
        "H",
        "IR",
        "e_slots",
        "route_vec",
    ),
    [
        ("pipe_single_tile", True, False, False, 40, 256, 128, 0, False),
        ("pipe_ep_group_rel", False, True, True, 40, 256, 128, 0, False),
        ("pipe_ep_eplb_route_vec", False, True, True, 40, 256, 128, 70, True),
        ("multitile_rolled", False, False, False, 40, 512, 256, 0, False),
        ("multitile_unroll_group_rel", False, False, True, 40, 512, 256, 0, False),
        ("multitile_unroll_no_group_rel", False, False, True, 0, 512, 256, 0, False),
    ],
)
def test_moe_dynamic_dma_race_invariants(
    case_name: str,
    pipe: bool,
    pipe_ep: bool,
    moe_unroll: bool,
    moe_group_rows: int,
    H: int,
    IR: int,
    e_slots: int,
    route_vec: bool,
):
    """Run MoEModule under Pallas vector-clock race detection (`on_wait` vs `eager`)."""
    del case_name
    R = 2
    devs = jax.devices("cpu")
    assert len(devs) >= R

    E, TOPK, ISR = 128, 2, 128
    ER = E // R
    EW = e_slots if e_slots > 0 else ER
    ebank = (256 << 10) if pipe else ((32 << 10) if pipe_ep else (64 << 10))
    nbe = 6 if (pipe or pipe_ep) else 8
    tiles = dataclasses.replace(
        TINY_TILES,
        bank=512 << 10,
        nbd=8,
        nbe=nbe,
        ebank=ebank,
        war=1,
        mb8=2,
    ).with_modules(
        {
            "moe": {
                "pipe": pipe,
                "pipe_ep": pipe_ep,
                "moe_unroll": moe_unroll,
                "moe_group_rows": moe_group_rows,
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
        mp = cp.module("moe")
        assert mp.route_vec is route_vec
        assert mp.PIPE is pipe
        assert mp.PIPE_EP is pipe_ep
        assert mp.unroll is bool(moe_unroll or pipe_ep)
        assert mp.group_rel is (moe_group_rows >= ex.B)
        if not pipe and not pipe_ep:
            assert mp.T13 > 1 and mp.T2 > 1
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


def test_moe_module_option_consistency_precedence_and_decls():
    """Verify `MoEModule` `moe_tp`/`e_slots`/`geometry` precedence and decls."""
    # Default precedence when neither explicit options nor geometry are provided.
    m_def = MoEModule()
    assert m_def.moe_tp is True
    assert m_def.e_slots == 0
    assert dict(m_def.options) == {"shared_gate": True}
    assert m_def.cache_decl() is None

    hf_cfg = {
        "hidden_size": 128,
        "num_experts": 128,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "shared_expert_intermediate_size": 128,
    }
    g_def = FakeGeometry(
        L=1,
        att_every=1,
        H=128,
        NH=2,
        NKV=1,
        HD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=128,
        moe_tp=True,
        e_slots=0,
    )
    b_def = MoEModule.bind("mlp")
    assert (
        m_def.options
        == MoEModule(geometry=g_def).options
        == b_def.resolve(hf_cfg).options
        == b_def.to_module(hf_cfg).options
    )
    b_wd = MoEModule.bind("mlp", weight_dtypes={"sh13": "bf16"})
    assert (
        MoEModule(weight_dtypes={"sh13": "bf16"}).options
        == b_wd.resolve(hf_cfg).options
        == b_wd.to_module(hf_cfg).options
    )

    # Geometry precedence when explicit values are omitted (`None`).
    g_ep = FakeGeometry(
        L=1,
        att_every=1,
        H=128,
        NH=2,
        NKV=1,
        HD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=128,
        moe_tp=False,
        e_slots=70,
    )
    m_geom = MoEModule(geometry=g_ep)
    assert m_geom.moe_tp is False
    assert m_geom.e_slots == 70
    assert m_geom.options["moe_tp"] is False
    assert m_geom.options["e_slots"] == 70
    assert m_def._check_option_consistency(g_ep) == (False, 70)

    # Agreeing explicit and geometry values.
    m_agree = MoEModule(moe_tp=False, e_slots=70, geometry=g_ep)
    assert m_agree.moe_tp is False
    assert m_agree.e_slots == 70
    assert m_agree._check_option_consistency(g_ep) == (False, 70)
    b_ep = MoEModule.bind("mlp", moe_tp=False, e_slots=70)
    assert (
        m_geom.options
        == m_agree.options
        == b_ep.resolve(hf_cfg).options
        == b_ep.to_module(hf_cfg).options
    )

    # Explicit values take precedence when geometry lacks `moe_tp` / `e_slots`.
    g_plain = SimpleNamespace(
        L=1, att_every=1, H=128, E=128, TOPK=2, I=128, IS=128, eps=1e-6
    )
    m_exp = MoEModule(moe_tp=False, e_slots=70)
    assert m_exp.moe_tp is False and m_exp.e_slots == 70
    assert m_exp._check_option_consistency(g_plain) == (False, 70)
    assert m_def._check_option_consistency(g_plain) == (True, 0)

    # Invalid option types / negative `e_slots` / conflicting values fail closed.
    for bad_kw in (
        {"e_slots": True},
        {"e_slots": -1},
        {"moe_tp": "yes"},
        {"shared_gate": "yes"},
    ):
        with pytest.raises(MegakernelConfigError):
            MoEModule(**bad_kw)  # type: ignore[arg-type]
        with pytest.raises(MegakernelConfigError):
            MoEModule.bind("mlp", **bad_kw)
    with pytest.raises(MegakernelConfigError, match="moe_tp=True.*moe_tp=False"):
        MoEModule(moe_tp=True, geometry=g_ep)
    with pytest.raises(MegakernelConfigError, match="e_slots=64.*e_slots=70"):
        MoEModule(e_slots=64, geometry=g_ep)

    ex = ExecSpec(B=2, T=1, R=2, ctx=128, stack_e=True, tiles=TINY_TILES)
    m_tp_only = MoEModule(moe_tp=True)
    with pytest.raises(MegakernelConfigError, match="moe_tp=True.*moe_tp=False"):
        m_tp_only.hf_params(g=g_ep)
    m_slots_only = MoEModule(e_slots=64)
    with pytest.raises(MegakernelConfigError, match="e_slots=64.*e_slots=70"):
        m_slots_only.weights(g_ep, ex)
    g_bad_div = dataclasses.replace(g_def, E=127)
    with pytest.raises(MegakernelConfigError, match="multiple of R=2"):
        m_def.weights(g_bad_div, ex)
    with pytest.raises(MegakernelConfigError, match="multiple of R=2"):
        m_def.smalls(g_bad_div, ex)
    with pytest.raises(MegakernelConfigError, match="multiple of R=2"):
        m_def.shared_widths(g_bad_div, ex)
    with pytest.raises(MegakernelConfigError, match="redundant expert slots"):
        m_def.weights(dataclasses.replace(g_def, e_slots=70), ex)

    # `E % 128 == 0` and `E <= 1024` seen-table envelope checks in `derive_moe_params`.
    ex_r1 = ExecSpec(B=2, T=1, R=1, ctx=128, stack_e=True, tiles=TINY_TILES)
    ms_r1 = ModelSpec(
        "moe_env", config=g_def, layer_type=LayerType("lt", slots=(m_def,))
    )
    p_r1 = planner.plan(ms_r1, ex_r1)
    with pytest.raises(MegakernelConfigError, match="E=64 must be a multiple of 128"):
        derive_moe_params(
            dataclasses.replace(g_def, E=64),
            ex_r1,
            p_r1,
            MoETunables(topk=False),
            MoEDebug(),
            moe_tp=True,
            e_slots=0,
        )
    with pytest.raises(MegakernelConfigError, match="E=2048 exceeds seen-table"):
        derive_moe_params(
            dataclasses.replace(g_def, E=2048),
            ex_r1,
            p_r1,
            MoETunables(topk=False),
            MoEDebug(),
            moe_tp=True,
            e_slots=0,
        )

    # `hf_params()` and `reference()` through `MoEModule`.
    ex_ref = ExecSpec(B=2, T=1, R=1, ctx=128, stack_e=True, tiles=TINY_TILES)
    decls = {d.name: d for d in m_agree.hf_params()}
    assert {
        "router",
        "sh_gate_proj",
        "sh_up_proj",
        "sh2",
        "e_gate_proj",
        "e_up_proj",
        "e2",
        "sh_gate",
    } <= set(decls)
    assert decls["e_gate_proj"].transform.shard_axis == 0
    assert decls["e2"].transform.shard_axis == 0

    g_ref = g_def
    m_ref = MoEModule(moe_tp=True, e_slots=0, geometry=g_ref)
    keys = jax.random.split(jax.random.PRNGKey(303), 8)
    x_in = jax.random.normal(keys[0], (ex_ref.B, g_ref.H), dtype=BF16) * 0.5
    params = {
        "post_norm": jnp.full((1, g_ref.H), 0.2, dtype=F32),
        "router": jax.random.normal(keys[1], (1, g_ref.H, g_ref.E), dtype=BF16) * 0.2,
        "sh_gate": jax.random.normal(keys[2], (1, g_ref.H), dtype=F32) * 0.2,
        "sh13": (
            jax.random.normal(keys[3], (1, g_ref.H, 2 * g_ref.IS), dtype=F32) * 0.3
        ).astype(FP8),
        "sh13_s": jnp.full((1, 2 * g_ref.IS), 0.3, dtype=F32),
        "sh2": (
            jax.random.normal(keys[4], (1, g_ref.IS, g_ref.H), dtype=F32) * 0.3
        ).astype(FP8),
        "sh2_s": jnp.full((1, g_ref.H), 0.3, dtype=F32),
        "e13": (
            jax.random.normal(keys[5], (1, g_ref.E, g_ref.H, 2 * g_ref.I), dtype=F32)
            * 0.3
        ).astype(FP8),
        "e13_s": jnp.full((1, g_ref.E, 2 * g_ref.I), 0.3, dtype=F32),
        "e2": (
            jax.random.normal(keys[6], (1, g_ref.E, g_ref.I, g_ref.H), dtype=F32) * 0.3
        ).astype(FP8),
        "e2_s": jnp.full((1, g_ref.E, g_ref.H), 0.3, dtype=F32),
    }
    ref_out, ref_st = m_ref.reference(params, x_in, ex=ex_ref)
    assert ref_st is None
    assert ref_out.shape == (ex_ref.B, g_ref.H)
    assert float(jnp.max(jnp.abs(ref_out.astype(F32)))) >= 0.05
    with pytest.raises(MegakernelConfigError, match="moe_tp=True.*moe_tp=False"):
        m_ref.reference(params, x_in, g=g_ep, ex=ex_ref)


@pytest.mark.parametrize(
    ("case_id", "kw"),
    [
        (
            "bf16_weights_pipe",
            dict(
                moe_tp=True,
                R=1,
                pipe=True,
                weight_dtypes={
                    "sh13": "bf16",
                    "sh2": "bf16",
                    "e13": "bf16",
                    "e2": "bf16",
                },
            ),
        ),
        (
            "bf16_weights_multitile_unroll",
            dict(
                moe_tp=False,
                R=1,
                pipe=False,
                moe_unroll=True,
                H=512,
                IR=256,
                ebank=128 << 10,
                nbe=8,
                weight_dtypes={
                    "sh13": "bf16",
                    "sh2": "bf16",
                    "e13": "bf16",
                    "e2": "bf16",
                },
            ),
        ),
        (
            "ep_r1_unstacked_multitile_group_rel",
            dict(
                moe_tp=False,
                R=1,
                pipe=False,
                moe_unroll=True,
                moe_group_rows=40,
                H=512,
                IR=256,
                ebank=64 << 10,
                nbe=8,
                L=2,
                stack_e=False,
            ),
        ),
    ],
    ids=[
        "bf16_weights_pipe",
        "bf16_weights_multitile_unroll",
        "ep_r1_unstacked_multitile_group_rel",
    ],
)
def test_moe_module_bf16_weights_and_unstacked_parity(case_id: str, kw: dict[str, Any]):
    """Verify `weight_dtypes='bf16'` (no `es`) and `stack_e=False` (`per_layer`)."""
    del case_id
    _run_moe_numerical_parity_case(**kw, interpret=True)


def test_moe_module_debug_forced_routing_and_row_mask_in_pallas():
    """Verify `MoEDebug(forced_routing=True, row_mask=True)` hooks on `MoEModule`."""
    L, H, E, TOPK, IR, ISR = 2, 128, 128, 2, 128, 128
    tiles = dataclasses.replace(TINY_TILES, bank=512 << 10, ebank=256 << 10, nbe=4)
    ex = ExecSpec(
        B=2,
        T=1,
        R=1,
        ctx=128,
        stack_e=True,
        tiles=tiles,
        debug=DebugOptions(
            modules={"moe": MoEDebug(forced_routing=True, row_mask=True)}
        ),
        interpret=True,
    )
    g = FakeGeometry(
        L=L,
        att_every=1,
        H=H,
        NH=2,
        NKV=1,
        HD=128,
        E=E,
        TOPK=TOPK,
        I=IR,
        IS=ISR,
        moe_tp=True,
    )
    moe = MoEModule()
    assert moe.extra_inputs(g, ex) == [
        ExtraInput("rbias", "hbm"),
        ExtraInput("moe_rows", "smem", (2,), I32),
    ]
    ms = ModelSpec("moe_dbg", config=g, layer_type=LayerType("lt", slots=(moe,)))
    cp, scratch = _build_moe_test_scratch(ms, ex)
    assert "rb" in scratch and "rb_sem" in scratch
    p = cp.plan
    wsh = planner.weight_shapes(ms, ex, p)
    s_names = list(scratch.keys())

    keys = jax.random.split(jax.random.PRNGKey(909), 6)
    hid = jax.random.normal(keys[0], (ex.M, H), dtype=BF16) * 0.5
    # Router logits are uniform (0.0 -> equal pexp -> normalized top-2 weight 0.5 each),
    # while `rbias` forces row 0 -> (5, 9) and row 1 -> (20, 21) on layer 0, and
    # row 0 -> (12, 15) and row 1 -> (30, 31) on layer 1. `moe_rows = [1, 0]`
    # suppresses row 1's experts from `elist`, so only row 0's forced experts run.
    e13 = (jax.random.normal(keys[1], (L, E, H, 2 * IR), dtype=F32) * 0.35).astype(FP8)
    e13_s = jnp.full((L, E, 2 * IR), 0.25, dtype=F32)
    e2 = (jax.random.normal(keys[2], (L, E, IR, H), dtype=F32) * 0.35).astype(FP8)
    e2_s = jnp.full((L, E, H), 0.25, dtype=F32)
    sm = jnp.zeros(wsh["smalls"][0], dtype=F32)
    o_pn, w_pn = p.smalls_layout["post_norm"]
    sm = sm.at[:, 0, o_pn : o_pn + w_pn].set(0.2)

    rbias = jnp.zeros((L, ex.M, E), dtype=F32)
    rbias = rbias.at[0, 0, 5].set(10.0).at[0, 0, 9].set(8.0)
    rbias = rbias.at[0, 1, 20].set(10.0).at[0, 1, 21].set(8.0)
    rbias = rbias.at[1, 0, 12].set(10.0).at[1, 0, 15].set(8.0)
    rbias = rbias.at[1, 1, 30].set(10.0).at[1, 1, 31].set(8.0)
    moe_rows = jnp.array([1, 0], dtype=I32)

    w_in = {k: jnp.zeros(wsh[k][0], dtype=wsh[k][1]) for k in p.w_names}
    w_in["e13"] = e13.reshape(wsh["e13"][0])
    w_in["e2"] = e2.reshape(wsh["e2"][0])
    w_in["es"] = jnp.concatenate([e13_s, e2_s], axis=-1).reshape(wsh["es"][0])
    w_in["smalls"] = sm

    def pallas_body(pos_ref, hid_ref, rbias_ref, moe_rows_ref, *rest):
        n_w, n_o = len(p.w_names), len(p.o_names)
        W = dict(zip(p.w_names, rest[:n_w]))
        out_refs = dict(zip(p.o_names, rest[n_w : n_w + n_o]))
        scr = dict(zip(s_names, rest[n_w + n_o :]))
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
            me=jnp.int32(0),
            stream=ring,
            comm=None,
            extra={"rbias": rbias_ref, "moe_rows": moe_rows_ref},
        )
        pltpu.make_async_copy(
            W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        ).start()
        moe.prologue(ctx)
        scr["res"][...] = hid_ref[: ex.MB].astype(F32)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        moe.init_buffers(ctx)
        for lyr in range(L):
            slot = lyr % 2
            ring.cur_layer = jnp.int32(lyr)
            pltpu.make_async_copy(
                W["smalls"].at[lyr], scr["sm"].at[slot], scr["sm_sem"].at[slot]
            ).wait()
            if lyr + 1 < L:
                pltpu.make_async_copy(
                    W["smalls"].at[lyr + 1],
                    scr["sm"].at[1 - slot],
                    scr["sm_sem"].at[1 - slot],
                ).start()
            moe.on_layer_start(ctx, jnp.int32(lyr), jnp.int32(slot))
            ctx.apply_input_norm(moe.input_norm, jnp.int32(slot))
            part_moe = moe.emit(ctx, jnp.int32(lyr), jnp.int32(slot))
            out_refs["hidden"][lyr, : ex.MB] = part_moe.astype(BF16)
            if ex.MB < ex.M:
                out_refs["hidden"][lyr, ex.MB :] = jnp.zeros((ex.M - ex.MB, H), BF16)

    call = pl.pallas_call(
        pallas_body,
        out_shape=[jax.ShapeDtypeStruct((L, ex.M, H), BF16)],
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ]
        + [pl.BlockSpec(memory_space=pltpu.HBM) for _ in p.w_names],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        scratch_shapes=[scratch[k] for k in s_names],
        interpret=pltpu.InterpretParams(),
    )
    pos = jnp.array([1, 2], dtype=I32)
    (out_part,) = jax.jit(call)(
        pos, hid, rbias, moe_rows, *[w_in[k] for k in p.w_names]
    )

    x_moe = rmsnorm(
        hid[: ex.B].astype(F32), jnp.full((H,), 0.2, F32), g.eps, zero_centered=True
    )
    xb0 = x_moe[0:1].astype(BF16).astype(F32)
    for lyr, forced_pair in ((0, (5, 9)), (1, (12, 15))):
        exp_row0 = jnp.zeros((1, H), dtype=F32)
        for e_id in forced_pair:
            h_e = (xb0 @ e13[lyr, e_id].astype(F32)) * e13_s[lyr, e_id]
            mid_e = (
                round_bf16(
                    round_bf16(jax.nn.silu(round_bf16(h_e[:, :IR])))
                    * round_bf16(h_e[:, IR:])
                )
                .astype(BF16)
                .astype(F32)
            )
            y_e = round_bf16((mid_e @ e2[lyr, e_id].astype(F32)) * e2_s[lyr, e_id])
            exp_row0 = exp_row0 + 0.5 * y_e
        assert float(jnp.max(jnp.abs(exp_row0))) >= 0.05
        np.testing.assert_allclose(
            out_part[lyr, 0:1].astype(F32), exp_row0, atol=2e-2, rtol=2e-2
        )
        # Row 1's forced experts were masked out of `elist` by `moe_rows = [1, 0]`
        # (and `sh13`/`sh2` are zero), so row 1 receives 0 contribution!
        np.testing.assert_allclose(out_part[lyr, 1].astype(F32), 0.0, atol=1e-6)
