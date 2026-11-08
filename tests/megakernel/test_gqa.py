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
"""Unit tests for `GQAModule` declarations and Pallas `interpret=True` execution."""

from __future__ import annotations

import contextlib
import dataclasses
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tests.megakernel.megakernel_fakes import (
    TINY_TILES,
    FakeFFNModule,
    FakeGeometry,
    MiniCtx,
)
from vllm_torchtpu.megakernel import planner
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.modules.gqa import GQAModule
from vllm_torchtpu.megakernel.modules.gqa.context import GQAEmit
from vllm_torchtpu.megakernel.modules.gqa.merge import split_merge
from vllm_torchtpu.megakernel.ops import BF16, F32, I32, round_bf16, round_up
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec

_TINY_GQA_TILES = TINY_TILES.with_modules({"gqa": {"kv_blk": 256, "nkv": 3}})


def _build_gqa_test_scratch(
    ms: ModelSpec, ex: ExecSpec, p: planner.Plan | None = None
) -> tuple[Any, dict[str, Any]]:
    """Build core + module scratch specs for standalone `GQAModule` Pallas tests."""
    if p is None:
        p = planner.plan(ms, ex)
    g = ms.config
    mb, sw = ex.MB, p.shared_widths
    scratch: dict[str, Any] = {
        "res": pltpu.VMEM((mb, g.H), F32),
        "xb": pltpu.VMEM((mb, g.H), BF16),
        "xb2": pltpu.VMEM((mb, round_up(sw["xb2"], 128)), BF16),
        "acc": pltpu.VMEM((mb, round_up(sw["acc"], 128)), F32),
        "ybuf": pltpu.VMEM((mb, round_up(sw["ybuf"], 128)), F32),
        "mix": pltpu.VMEM((mb, round_up(sw["mix"], 128)), F32),
        "sm": pltpu.VMEM((2, 1, p.smalls_width), F32),
        "sm_sem": pltpu.SemaphoreType.DMA((2,)),
    }
    for k in p.static_kinds:
        _, n_k, dt_k = p.kinds[k]
        scratch[f"_w_{k}"] = pltpu.VMEM((p.tk[k], n_k), dt_k)
        scratch[f"_w_sem_{k}"] = pltpu.SemaphoreType.DMA(())
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
        stream=SimpleNamespace(
            L=g.L,
            tk=p.tk,
            nt=p.nt,
            lay=p.smalls_layout,
            module_ord=lambda lyr, _name: lyr,
        ),
        comm=None,
        module=p.module_params.__getitem__,
    )
    return cp, scratch


class _FakeGQAStream:
    """Minimal stream stub implementing `segment` and `dmatmul` for `GQAModule.emit`."""

    def __init__(self, sp: Any, W: dict[str, Any], scr: dict[str, Any]) -> None:
        self.sp = sp
        self.W = W
        self.scr = scr

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


def test_gqa_module_declarations_and_emit_in_pallas_interpret(fake_ex: ExecSpec):
    """Verify `GQAModule` declarations drive `planner`/`vmem` and `emit` in Pallas."""
    fake_ex = dataclasses.replace(fake_ex, tiles=_TINY_GQA_TILES)
    g_decl = FakeGeometry(L=2, att_every=1, H=256)
    gqa_decl = GQAModule()
    ffn_decl = FakeFFNModule()
    ms_decl = ModelSpec(
        "gqa_ffn_test",
        config=g_decl,
        layer_type=LayerType("lt", slots=(gqa_decl, ffn_decl)),
    )

    p_decl = planner.plan(ms_decl, fake_ex)
    assert p_decl.static_kinds == ("att_in", "att_out", "router", "sh13", "sh2")
    assert p_decl.dynamic_kinds == ("e13", "e2")
    assert p_decl.s_names == ("kc", "vc")
    assert p_decl.o_names == ("hidden", "knew", "vnew")

    cp_decl, scr_decl = _build_gqa_test_scratch(ms_decl, fake_ex, p_decl)
    # GQA scratch allocated, no GDN scratch allocated
    assert "kbuf" in scr_decl and "vbuf" in scr_decl and "wl_cnt" in scr_decl
    assert "recb" not in scr_decl and "sbuf" not in scr_decl and "convb" not in scr_decl
    assert cp_decl.module("gqa").NAI == (2 * (g_decl.NH // fake_ex.R) + 2) * g_decl.HD

    ex = ExecSpec(
        B=4, T=1, R=1, ctx=256, stack_e=True, tiles=_TINY_GQA_TILES, interpret=True
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
    gqa = GQAModule()
    ffn = FakeFFNModule()
    ms = ModelSpec(
        "gqa_ffn_exec", config=g, layer_type=LayerType("lt", slots=(gqa, ffn))
    )
    cp, scratch = _build_gqa_test_scratch(ms, ex)
    p = cp.plan
    wsh = planner.weight_shapes(ms, ex, p)
    ssh = planner.state_shapes(ms, ex, p)
    osh = planner.kernel_out_shapes(ms, ex, p)
    s_names = list(scratch.keys())

    assert gqa.input_norm == "in_norm"
    assert ffn.input_norm == "post_norm"

    def pallas_body(pos_ref, hid_ref, *rest):
        n_w, n_s, n_o = len(p.w_names), len(p.s_names), len(p.o_names)
        W = dict(zip(p.w_names, rest[:n_w]))
        ST = dict(zip(p.s_names, rest[n_w : n_w + n_s]))
        out_refs = dict(zip(p.o_names, rest[n_w + n_s : n_w + n_s + n_o]))
        scr = dict(zip(s_names, rest[n_w + n_s + n_o :]))
        me = jnp.int32(0)
        ring = _FakeGQAStream(cp.stream, W, scr)
        ctx = MiniCtx(
            g=g,
            ex=ex,
            core_plan=cp,
            pos_ref=pos_ref,
            W=W,
            ST=ST,
            O=out_refs,
            scr=scr,
            me=me,
            stream=ring,
            comm=None,
        )

        pltpu.make_async_copy(
            W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        ).start()
        gqa.prologue(ctx)
        scr["res"][...] = hid_ref[: ex.MB].astype(F32)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        gqa.init_buffers(ctx)

        pltpu.make_async_copy(
            W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        ).wait()
        ctx.apply_input_norm(gqa.input_norm, jnp.int32(0))
        part_att = gqa.emit(ctx, jnp.int32(0), jnp.int32(0))
        scr["res"][...] = scr["res"][...] + part_att.astype(F32)
        out_refs["hidden"][: ex.MB] = scr["res"][...].astype(BF16)
        if ex.MB < ex.M:
            out_refs["hidden"][ex.MB :] = jnp.zeros((ex.M - ex.MB, g.H), BF16)

    in_specs = (
        [pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec(memory_space=pltpu.VMEM)]
        + [pl.BlockSpec(memory_space=pltpu.HBM) for _ in p.w_names]
        + [pl.BlockSpec(memory_space=pltpu.HBM) for _ in p.s_names]
    )
    out_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM)
        if k == "hidden"
        else pl.BlockSpec(memory_space=pltpu.HBM)
        for k in p.o_names
    ]
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
    s_args = [jnp.zeros(ssh[k][0], dtype=ssh[k][1]) for k in p.s_names]
    outs = jax.jit(call)(pos, hid, *w_args, *s_args)
    hid_out, knew_out, vnew_out = outs
    assert hid_out.shape == (ex.M, g.H)
    assert jnp.all(jnp.isfinite(hid_out))
    assert knew_out.shape == osh["knew"][0]
    assert vnew_out.shape == osh["vnew"][0]

    # Verify host_update_states scatters knew/vnew into kc/vc
    updated = gqa.host_update_states(
        g,
        ex,
        pos,
        {"kc": s_args[0], "vc": s_args[1]},
        {"knew": knew_out, "vnew": vnew_out},
    )
    assert updated["kc"].shape == ssh["kc"][0]
    assert updated["vc"].shape == ssh["vc"][0]


@pytest.mark.parametrize("expect_gq", [True, False])
def test_gqa_split_merge_gq_and_slab_in_pallas_interpret(expect_gq: bool):
    """Verify `split_merge` (`gq_merge_start` and `slab_merge_start`) in Pallas."""
    tiles = dataclasses.replace(_TINY_GQA_TILES, bank=512 << 10).with_modules(
        {"gqa": {"gqa_qv": expect_gq, "gq_pack": expect_gq}}
    )
    ex = ExecSpec(B=8, T=4, R=2, ctx=256, stack_e=True, tiles=tiles, interpret=True)
    g = FakeGeometry(
        L=1,
        att_every=1,
        H=256,
        NH=8,
        NKV=1,
        HD=256,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=True,
    )
    gqa = GQAModule()
    ms = ModelSpec(
        "gqa_split",
        config=g,
        layer_type=LayerType("lt", slots=(gqa, FakeFFNModule())),
    )
    cp, scratch = _build_gqa_test_scratch(ms, ex)
    ap = cp.module("gqa")
    assert ap.SPLIT is True and ap.GQ is expect_gq
    s_names = list(scratch.keys())

    def body(
        am_in_ref,
        al_in_ref,
        aacc_in_ref,
        gate_in_ref,
        am_out_ref,
        al_out_ref,
        aacc_out_ref,
        gate_out_ref,
        *scr_refs,
    ):
        scr = dict(zip(s_names, scr_refs))
        scr["am"][...] = am_in_ref[...]
        scr["al"][...] = al_in_ref[...]
        scr["aacc"][...] = aacc_in_ref[...]
        scr["gate"][...] = gate_in_ref[...]
        if not ap.GQ:
            scr["pso"][...] = jnp.zeros(scr["pso"].shape, F32)

        class _FakePeerComm:
            def group_copies(self, src_fn, dst_ref, ssem, rsem, G: int):
                del dst_ref, ssem, rsem, G

                class _Copy:
                    def start(self):
                        scr["pri"][1] = src_fn(1)[...]

                    def wait_recv(self):
                        return None

                    def wait_send(self):
                        return None

                return [_Copy()]

        ctx = MiniCtx(
            g=g,
            ex=ex,
            core_plan=cp,
            pos_ref=None,
            W={},
            ST={},
            O={},
            scr=scr,
            me=jnp.int32(0),
            stream=_FakeGQAStream(cp.stream, {}, scr),
            comm=_FakePeerComm(),
        )
        cx = GQAEmit.build(ctx, ap, jnp.int32(0), gqa.name)
        split_merge(cx)
        am_out_ref[...] = scr["am"][...]
        al_out_ref[...] = scr["al"][...]
        aacc_out_ref[...] = scr["aacc"][...]
        gate_out_ref[...] = scr["gate"][...]

    am_sh = scratch["am"].shape
    al_sh = scratch["al"].shape
    aacc_sh = scratch["aacc"].shape
    gate_sh = scratch["gate"].shape
    call = pl.pallas_call(
        body,
        out_shape=[
            jax.ShapeDtypeStruct(am_sh, F32),
            jax.ShapeDtypeStruct(al_sh, F32),
            jax.ShapeDtypeStruct(aacc_sh, F32),
            jax.ShapeDtypeStruct(gate_sh, F32),
        ],
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 4,
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 4,
        scratch_shapes=[scratch[k] for k in s_names],
        interpret=pltpu.InterpretParams(),
    )
    am_in = jnp.ones(am_sh, dtype=F32) * 0.5
    al_in = jnp.ones(al_sh, dtype=F32) * 2.0
    aacc_in = jnp.ones(aacc_sh, dtype=F32) * 1.25
    gate_in = jnp.zeros(gate_sh, dtype=F32)
    am_out, al_out, aacc_out, gate_out = jax.jit(call)(am_in, al_in, aacc_in, gate_in)
    assert jnp.allclose(gate_out[: ex.B], round_bf16(jax.nn.sigmoid(gate_in[: ex.B])))
    if not expect_gq:
        assert jnp.allclose(am_out[:, : ap.nh], 0.5)
        assert jnp.allclose(al_out[:, : ap.nh], 4.0)
        assert jnp.allclose(aacc_out[:, : ap.nh], 2.5)
    else:
        assert jnp.all(jnp.isfinite(aacc_out))
