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
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tests.megakernel.core.megakernel_fakes import (
    TINY_TILES,
    FakeFFNModule,
    FakeGeometry,
    MiniCtx,
)
from vllm_torchtpu.megakernel.core import planner
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.modules.gqa import GQAModule
from vllm_torchtpu.megakernel.core.modules.gqa.context import GQAEmit
from vllm_torchtpu.megakernel.core.modules.gqa.merge import split_merge
from vllm_torchtpu.megakernel.core.modules.gqa.new_token import att_new
from vllm_torchtpu.megakernel.core.modules.gqa.params import GQATunables
from vllm_torchtpu.megakernel.core.ops import (
    BF16,
    F32,
    FP8,
    I32,
    rmsnorm,
    round_bf16,
    round_up,
)
from vllm_torchtpu.megakernel.core.spec import LayerType, ModelSpec

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

    def dmatmul(
        self, kind: str, x_ref: Any, n_dim: int, *, hook: Any = None, split: Any = None
    ) -> Any:
        # The layer-schedule inputs (ARPIPE `hook`, AR0S `split`) stay off here.
        assert hook is None and split is None
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
        gqa.epilogue(ctx)
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

    keys = jax.random.split(jax.random.PRNGKey(41), 9)
    pos = jnp.array([4, 8, 12, 16], dtype=I32)
    hid = jax.random.normal(keys[0], (ex.M, g.H), dtype=BF16) * 0.6
    w_by_name = {k: jnp.zeros(wsh[k][0], dtype=wsh[k][1]) for k in p.w_names}
    w_att_in = (jax.random.normal(keys[1], wsh["att_in"][0], dtype=BF16) * 0.4).astype(
        FP8
    )
    w_att_out = (
        jax.random.normal(keys[2], wsh["att_out"][0], dtype=BF16) * 0.5
    ).astype(FP8)
    w_by_name["att_in"] = w_att_in
    w_by_name["att_out"] = w_att_out

    lay = p.smalls_layout
    in_norm_w = jax.random.normal(keys[3], (1, g.H), dtype=F32) * 0.2 + 0.1
    in_s = jax.random.uniform(
        keys[4], (1, lay["in_s"][1]), dtype=F32, minval=0.3, maxval=0.8
    )
    out_s = jax.random.uniform(
        keys[5], (1, lay["out_s"][1]), dtype=F32, minval=0.3, maxval=0.8
    )
    qn = jax.random.normal(keys[6], (1, g.HD), dtype=F32) * 0.2 + 1.0
    kn = jax.random.normal(keys[7], (1, g.HD), dtype=F32) * 0.2 + 1.0
    sm = jnp.zeros(wsh["smalls"][0], dtype=F32)
    for fname, val in (
        ("in_norm", in_norm_w),
        ("in_s", in_s),
        ("out_s", out_s),
        ("qn", qn),
        ("kn", kn),
    ):
        o_sm, w_sm = lay[fname]
        sm = sm.at[:, 0, o_sm : o_sm + w_sm].set(val)
    w_by_name["smalls"] = sm

    k_kc, k_vc = jax.random.split(keys[8], 2)
    kc_in = jax.random.normal(k_kc, ssh["kc"][0], dtype=BF16) * 0.6
    vc_in = jax.random.normal(k_vc, ssh["vc"][0], dtype=BF16) * 0.6 + jnp.asarray(
        0.5, dtype=BF16
    )
    w_args = [w_by_name[k] for k in p.w_names]
    s_args = [kc_in, vc_in]
    outs = jax.jit(call)(pos, hid, *w_args, *s_args)
    hid_out, knew_out, vnew_out = outs
    assert hid_out.shape == (ex.M, g.H)
    assert jnp.all(jnp.isfinite(hid_out))
    assert knew_out.shape == osh["knew"][0]
    assert vnew_out.shape == osh["vnew"][0]

    # Compare emitted attention delta and updated KV cache against `gqa.reference`.
    x_norm = rmsnorm(
        hid[: ex.B].astype(F32), in_norm_w[0], g.eps, zero_centered=g.zero_centered_norm
    )
    ref_params = {
        "att_in": w_att_in.reshape(1, 1, g.H, -1),
        "att_out": w_att_out.reshape(1, 1, -1, g.H),
        "in_s": in_s,
        "out_s": out_s,
        "qn": qn,
        "kn": kn,
    }
    ref_delta, ref_st = gqa.reference(
        ref_params,
        x_norm,
        {"kc": kc_in[None], "vc": vc_in[None]},
        g=g,
        ex=ex,
        pos=pos,
        paged_kv=False,
    )
    actual_delta = hid_out[: ex.B].astype(F32) - hid[: ex.B].astype(F32)
    ref_delta_f32 = np.asarray(ref_delta.astype(F32))
    assert float(np.max(np.abs(ref_delta_f32))) >= 0.2
    np.testing.assert_allclose(
        np.asarray(actual_delta), ref_delta_f32, rtol=2e-2, atol=2e-2
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            np.asarray(actual_delta),
            np.zeros_like(ref_delta_f32),
            rtol=2e-2,
            atol=2e-2,
        )

    # Verify host_update_states scatters knew/vnew into kc/vc and matches reference.
    updated = gqa.host_update_states(
        g,
        ex,
        pos,
        {"kc": kc_in, "vc": vc_in},
        {"knew": knew_out, "vnew": vnew_out},
    )
    assert updated["kc"].shape == ssh["kc"][0]
    assert updated["vc"].shape == ssh["vc"][0]
    np.testing.assert_allclose(
        np.asarray(updated["kc"].astype(F32)),
        np.asarray(ref_st["kc"][0].astype(F32)),
        rtol=1e-2,
        atol=1e-2,
    )
    np.testing.assert_allclose(
        np.asarray(updated["vc"].astype(F32)),
        np.asarray(ref_st["vc"][0].astype(F32)),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize("expect_gq", [True, False])
def test_gqa_split_merge_gq_and_slab_in_pallas_interpret(expect_gq: bool):
    """Verify `split_merge` (`gq_merge_start` / `slab_merge_start`) + `att_new`."""
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
        pri_out_ref,
        am_out_ref,
        al_out_ref,
        aacc_out_ref,
        gate_out_ref,
        mix_out_ref,
        *scr_refs,
    ):
        scr = dict(zip(s_names, scr_refs))
        scr["am"][...] = am_in_ref[...]
        scr["al"][...] = al_in_ref[...]
        scr["aacc"][...] = aacc_in_ref[...]
        scr["gate"][...] = gate_in_ref[...]
        scr["knew"][...] = jnp.zeros(scr["knew"].shape, scr["knew"].dtype)
        scr["vnew"][...] = jnp.zeros(scr["vnew"].shape, scr["vnew"].dtype)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        scr["qg"][...] = jnp.zeros(scr["qg"].shape, F32)
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
        pri_out_ref[...] = scr["pri"][...]
        am_out_ref[...] = scr["am"][...]
        al_out_ref[...] = scr["al"][...]
        aacc_out_ref[...] = scr["aacc"][...]
        gate_out_ref[...] = scr["gate"][...]
        att_new(cx)
        mix_out_ref[...] = scr["mix"][:, : ap.nh * ap.HD]

    pri_sh = scratch["pri"].shape
    am_sh = scratch["am"].shape
    al_sh = scratch["al"].shape
    aacc_sh = scratch["aacc"].shape
    gate_sh = scratch["gate"].shape
    mix_sh = (ex.MB, ap.nh * ap.HD)
    call = pl.pallas_call(
        body,
        out_shape=[
            jax.ShapeDtypeStruct(pri_sh, F32),
            jax.ShapeDtypeStruct(am_sh, F32),
            jax.ShapeDtypeStruct(al_sh, F32),
            jax.ShapeDtypeStruct(aacc_sh, F32),
            jax.ShapeDtypeStruct(gate_sh, F32),
            jax.ShapeDtypeStruct(mix_sh, F32),
        ],
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 4,
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 6,
        scratch_shapes=[scratch[k] for k in s_names],
        interpret=pltpu.InterpretParams(),
    )

    # Build distinct (m, l, acc) per sequence, peer group (0 vs 1), token, and head.
    NS, T, G, nh, HD = ex.NS, ex.T, ap.G, ap.nh, ap.HD
    QG, QR, MLW = ap.QG, ap.QR, ap.MLW
    sq_idx = np.arange(NS, dtype=np.float32)[:, None, None, None]
    g_idx = np.arange(G, dtype=np.float32)[None, :, None, None]
    t_idx = np.arange(T, dtype=np.float32)[None, None, :, None]
    h_idx = np.arange(nh, dtype=np.float32)[None, None, None, :]
    col_idx = np.arange(HD, dtype=np.float32)[None, None, None, None, :]

    # Positive `m` (> 0) with a nontrivial gap between group 0 and group 1 so
    # `exp(m0 - m)` and `exp(m1 - m)` are both strictly inside (0, 1) across rows.
    m_logical = (
        1.2
        + 0.35 * sq_idx
        + (0.9 - 0.55 * h_idx) * (2.0 * g_idx - 1.0)
        + 0.15 * t_idx
        + 0.08 * h_idx
    )
    l_logical = 1.5 + 0.4 * sq_idx + 0.7 * g_idx + 0.2 * t_idx + 0.1 * h_idx
    acc_logical = (
        (0.6 + 0.3 * sq_idx + 0.8 * g_idx - 0.12 * t_idx + 0.17 * h_idx)[..., None]
        + 0.01 * np.sin(col_idx * 0.13 + (g_idx + 1.0)[..., None])
    ).astype(np.float32)

    if expect_gq:
        am_rows = m_logical.reshape(NS, G * QG, 1)
        al_rows = l_logical.reshape(NS, G * QG, 1)
        aacc_rows = acc_logical.reshape(NS, G * QG, HD)
    else:
        am_rows = np.swapaxes(m_logical, 1, 2).reshape(NS, T * QR, 1)
        al_rows = np.swapaxes(l_logical, 1, 2).reshape(NS, T * QR, 1)
        aacc_rows = np.swapaxes(acc_logical, 1, 2).reshape(NS, T * QR, HD)

    am_in = jnp.broadcast_to(jnp.asarray(am_rows, dtype=F32), am_sh)
    al_in = jnp.broadcast_to(jnp.asarray(al_rows, dtype=F32), al_sh)
    aacc_in = jnp.asarray(aacc_rows, dtype=F32)
    gate_in = jnp.linspace(-0.8, 0.9, gate_sh[0] * gate_sh[1], dtype=F32).reshape(
        gate_sh
    )

    pri_out, am_out, al_out, aacc_out, gate_out, mix_out = jax.jit(call)(
        am_in, al_in, aacc_in, gate_in
    )
    expected_gate = np.asarray(round_bf16(jax.nn.sigmoid(gate_in[: ex.B])), np.float32)
    np.testing.assert_allclose(
        np.asarray(gate_out[: ex.B]), expected_gate, rtol=1e-5, atol=1e-5
    )

    # Analytic NumPy LSE merge across groups g=0 and g=1: shape (NS, T, nh, ...).
    m0, m1 = m_logical[:, 0], m_logical[:, 1]
    l0, l1 = l_logical[:, 0], l_logical[:, 1]
    acc0, acc1 = acc_logical[:, 0], acc_logical[:, 1]
    m_merged = np.maximum(m0, m1)
    r0 = np.exp(m0 - m_merged)
    r1 = np.exp(m1 - m_merged)
    l_merged = l0 * r0 + l1 * r1
    acc_merged = acc0 * r0[..., None] + acc1 * r1[..., None]

    if expect_gq:
        # Check `gq_merge_start` packing into `pri[g, :NS]` and `pri[g, NS]`.
        pri_np = np.asarray(pri_out)
        for g_i in range(G):
            for sq in range(NS):
                np.testing.assert_allclose(
                    pri_np[g_i, sq],
                    acc_logical[sq, g_i].reshape(QG, HD),
                    rtol=1e-6,
                    atol=1e-6,
                )
                m_band = pri_np[g_i, NS, :, sq * MLW : (sq + 1) * MLW]
                l_band = pri_np[g_i, NS, :, 128 + sq * MLW : 128 + (sq + 1) * MLW]
                np.testing.assert_allclose(
                    m_band,
                    np.broadcast_to(m_logical[sq, g_i].reshape(QG, 1), (QG, MLW)),
                    rtol=1e-6,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    l_band,
                    np.broadcast_to(l_logical[sq, g_i].reshape(QG, 1), (QG, MLW)),
                    rtol=1e-6,
                    atol=1e-6,
                )
    else:
        for b in range(ex.B):
            sq, tb = divmod(b, T)
            q0 = tb * QR
            np.testing.assert_allclose(
                np.asarray(am_out[sq, q0 : q0 + nh, 0]),
                m_merged[sq, tb],
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.asarray(al_out[sq, q0 : q0 + nh, 0]),
                l_merged[sq, tb],
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.asarray(aacc_out[sq, q0 : q0 + nh]),
                acc_merged[sq, tb],
                rtol=1e-5,
                atol=1e-5,
            )

    # Check post-`att_new` output (`mix_out`) against the analytic LSE merge +
    # causal zero-score new-token steps (`t_tok + 1` steps with `s_j = 0 < m_merged`).
    n_new = (np.arange(T, dtype=np.float32) + 1.0)[None, :, None]
    l_final = l_merged + n_new * np.exp(-m_merged)
    oh_ref = np.asarray(
        round_bf16(jnp.asarray(acc_merged / l_final[..., None])), dtype=np.float32
    ).reshape(ex.B, nh * HD)
    mix_ref = np.asarray(
        round_bf16(jnp.asarray(oh_ref * expected_gate)), dtype=np.float32
    )
    np.testing.assert_allclose(
        np.asarray(mix_out[: ex.B]), mix_ref, rtol=1e-2, atol=1e-2
    )

    # Negative controls: wrong peer index (merging group 0 with itself) and
    # inverted LSE rescale (`exp(m - m_g)`) must both fail.
    l_wrong_peer = 2.0 * l0 + n_new * np.exp(-m0)
    mix_wrong_peer = ((2.0 * acc0) / l_wrong_peer[..., None]).reshape(
        ex.B, nh * HD
    ) * expected_gate
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            np.asarray(mix_out[: ex.B]), mix_wrong_peer, rtol=1e-2, atol=1e-2
        )

    r0_inv = np.exp(m_merged - m0)
    r1_inv = np.exp(m_merged - m1)
    l_inv = l0 * r0_inv + l1 * r1_inv + n_new * np.exp(-m_merged)
    acc_inv = acc0 * r0_inv[..., None] + acc1 * r1_inv[..., None]
    mix_inv = (acc_inv / l_inv[..., None]).reshape(ex.B, nh * HD) * expected_gate
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            np.asarray(mix_out[: ex.B]), mix_inv, rtol=1e-2, atol=1e-2
        )


@pytest.mark.parametrize(("T", "NS"), [(1, 8), (2, 8), (1, 20), (2, 20)])
def test_gqa_host_update_states_single_scatter_vs_loop_and_numpy(T: int, NS: int):
    """Verify `GQAModule.host_update_states` single-scatter (`NS > kv_scatter_vec_ns`)
    matches the `dynamic_update_slice` loop and a NumPy scatter at `T=1` and `T>1`."""
    assert GQATunables.NUMERICS["kv_scatter_vec_ns"] == "exact"
    B = NS * T
    L, ctx, HD = 2, 64, 128
    g = FakeGeometry(L=L, att_every=1, H=256, NH=2, NKV=1, HD=HD)
    gqa = GQAModule(geometry=g)

    ex_default = ExecSpec(B=B, T=T, R=1, ctx=ctx, tiles=_TINY_GQA_TILES)
    ex_vec0 = ExecSpec(
        B=B,
        T=T,
        R=1,
        ctx=ctx,
        tiles=_TINY_GQA_TILES.with_modules({"gqa": {"kv_scatter_vec_ns": 0}}),
    )
    ex_loop = ExecSpec(
        B=B,
        T=T,
        R=1,
        ctx=ctx,
        tiles=_TINY_GQA_TILES.with_modules({"gqa": {"kv_scatter_vec_ns": NS + 4}}),
    )
    assert gqa.tunables(ex_default).kv_scatter_vec_ns == 16
    assert gqa.tunables(ex_vec0).kv_scatter_vec_ns == 0
    assert gqa.tunables(ex_loop).kv_scatter_vec_ns == NS + 4

    k0, k1, k2, k3 = jax.random.split(jax.random.PRNGKey(90 + T * 31 + NS), 4)
    kc_init = jax.random.normal(k0, (L, NS, ctx, HD), dtype=BF16) * 0.5
    vc_init = jax.random.normal(k1, (L, NS, ctx, HD), dtype=BF16) * 0.5
    knew = jax.random.normal(k2, (L, ex_default.MR, HD), dtype=F32) * 0.7
    vnew = jax.random.normal(k3, (L, ex_default.MR, HD), dtype=F32) * 0.7

    base_pos = np.arange(NS, dtype=np.int32) * 2 + 3
    pos_np = np.asarray(
        [base_pos[s] + t for s in range(NS) for t in range(T)], dtype=np.int32
    )
    pos = jnp.asarray(pos_np, dtype=I32)

    states = {"kc": kc_init, "vc": vc_init}
    raw_out = {"knew": knew, "vnew": vnew}
    out_default = gqa.host_update_states(g, ex_default, pos, states, raw_out)
    out_vec0 = gqa.host_update_states(g, ex_vec0, pos, states, raw_out)
    out_loop = gqa.host_update_states(g, ex_loop, pos, states, raw_out)

    # NumPy scatter oracle.
    kc_exp = np.asarray(kc_init).copy()
    vc_exp = np.asarray(vc_init).copy()
    knew_bf16 = np.asarray(knew[:, :B].astype(BF16))
    vnew_bf16 = np.asarray(vnew[:, :B].astype(BF16))
    for b in range(B):
        s_idx = b if T == 1 else b // T
        kc_exp[:, s_idx, pos_np[b], :] = knew_bf16[:, b, :]
        vc_exp[:, s_idx, pos_np[b], :] = vnew_bf16[:, b, :]

    for out in (out_default, out_vec0, out_loop):
        np.testing.assert_array_equal(
            np.asarray(out["kc"]).view(np.uint16), kc_exp.view(np.uint16)
        )
        np.testing.assert_array_equal(
            np.asarray(out["vc"]).view(np.uint16), vc_exp.view(np.uint16)
        )
    assert not np.array_equal(
        np.asarray(out_vec0["kc"]).view(np.uint16),
        np.asarray(kc_init).view(np.uint16),
    )
