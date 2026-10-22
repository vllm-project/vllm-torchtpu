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
"""Unit tests for `GQABlock` declarations and Pallas `interpret=True` execution."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from conftest import TINY_TILES, FakeGeometry, MiniCtx
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel import comm, planner, stream, vmem
from vllm_torchtpu.megakernel.blocks import GQABlock, MoEBlock
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.ops import BF16, F32, I32
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec

_TINY_GQA_TILES = TINY_TILES.with_blocks({"gqa": {"kv_blk": 256, "nkv": 3}})


def test_gqa_block_declarations_and_emit_in_pallas_interpret(fake_ex: ExecSpec):
    """Verify `GQABlock` declarations drive `planner`/`vmem` and `emit` in Pallas."""
    fake_ex = dataclasses.replace(fake_ex, tiles=_TINY_GQA_TILES)
    g_decl = FakeGeometry(L=2, att_every=1, H=256)
    gqa_decl = GQABlock()
    moe_decl = MoEBlock()
    ms_decl = ModelSpec(
        "gqa_moe_test",
        config=g_decl,
        layer_type=LayerType("lt", slots=(gqa_decl, moe_decl)),
    )

    p_decl = planner.plan(ms_decl, fake_ex)
    assert p_decl.static_kinds == ("att_in", "att_out", "router", "sh13", "sh2")
    assert p_decl.dynamic_kinds == ("e13", "e2")
    assert p_decl.s_names == ("kc", "vc")
    assert p_decl.o_names == ("hidden", "knew", "vnew")

    cp_decl, scr_decl = vmem.build_scratch(ms_decl, fake_ex, p_decl)
    # GQA scratch allocated, no GDN scratch allocated
    assert "kbuf" in scr_decl and "vbuf" in scr_decl and "wl_cnt" in scr_decl
    assert "recb" not in scr_decl and "sbuf" not in scr_decl and "convb" not in scr_decl
    assert cp_decl.block("gqa").NAI == (2 * (g_decl.NH // fake_ex.R) + 2) * g_decl.HD

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
    gqa = GQABlock()
    moe = MoEBlock()
    ms = ModelSpec(
        "gqa_moe_exec", config=g, layer_type=LayerType("lt", slots=(gqa, moe))
    )
    cp, scratch = vmem.build_scratch(ms, ex)
    p = cp.plan
    wsh = planner.weight_shapes(ms, ex, p)
    ssh = planner.state_shapes(ms, ex, p)
    osh = planner.kernel_out_shapes(ms, ex, p)
    s_names = list(scratch.keys())

    assert gqa.input_norm == "in_norm"
    assert moe.input_norm == "post_norm"

    def pallas_body(pos_ref, hid_ref, *rest):
        n_w, n_s, n_o = len(p.w_names), len(p.s_names), len(p.o_names)
        W = dict(zip(p.w_names, rest[:n_w]))
        ST = dict(zip(p.s_names, rest[n_w : n_w + n_s]))
        O = dict(zip(p.o_names, rest[n_w + n_s : n_w + n_s + n_o]))  # noqa: E741
        scr = dict(zip(s_names, rest[n_w + n_s + n_o :]))
        me = jnp.int32(0)
        ring = stream.TwoRing(cp.stream, W, scr, me, interpret=True)
        ar = comm.AllReduceRing(cp.comm, scr, me)
        ctx = MiniCtx(
            g=g,
            ex=ex,
            core_plan=cp,
            pos_ref=pos_ref,
            W=W,
            ST=ST,
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
        gqa.prologue(ctx)
        moe.prologue(ctx)
        scr["res"][...] = hid_ref[: ex.MB].astype(F32)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        gqa.init_buffers(ctx)
        moe.init_buffers(ctx)

        pltpu.make_async_copy(
            W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        ).wait()
        ring.begin_layer(jnp.int32(0))
        ctx.apply_input_norm(gqa.input_norm, jnp.int32(0))
        part_att = gqa.emit(ctx, jnp.int32(0), jnp.int32(0))
        ar.add_residual(ar.allreduce(0, part_att, jnp.bool_(True)))
        ctx.apply_input_norm(moe.input_norm, jnp.int32(0))
        part_moe = moe.emit(ctx, jnp.int32(0), jnp.int32(0))
        ar.add_residual(ar.allreduce(1, part_moe, jnp.bool_(True)))
        O["hidden"][: ex.MB] = scr["res"][...].astype(BF16)
        if ex.MB < ex.M:
            O["hidden"][ex.MB :] = jnp.zeros((ex.M - ex.MB, g.H), BF16)

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
