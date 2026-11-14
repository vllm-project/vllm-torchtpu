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
"""Model-free unit tests for `GDNModule` declarations, states, and Pallas parity."""

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
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.modules.gdn import GDNModule
from vllm_torchtpu.megakernel.core.ops import BF16, F32, FP8, rmsnorm, round_up
from vllm_torchtpu.megakernel.core.spec import LayerType, ModelSpec, SwitchSlot


def _build_gdn_test_scratch(
    ms: ModelSpec, ex: ExecSpec, p: planner.Plan | None = None
) -> tuple[Any, dict[str, Any]]:
    """Build core + module scratch specs for standalone `GDNModule` Pallas tests."""
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


class _FakeGDNStream:
    """Minimal stream stub implementing `segment` and `dmatmul` for `GDNModule.emit`."""

    def __init__(self, sp: Any, W: dict[str, Any], scr: dict[str, Any]) -> None:
        self.sp = sp
        self.W = W
        self.scr = scr

    def segment(self, *, lazy: bool = False) -> Any:
        del lazy
        return contextlib.nullcontext()

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
            acc = acc + lax.dot_general(
                x[:, t * tk : (t + 1) * tk],
                w_buf[...].astype(BF16),
                (((1,), (0,)), ((), ())),
                preferred_element_type=F32,
            )
        self.scr["acc"][:, :n_dim] = acc
        return self.scr["acc"][:, :n_dim]


def test_gdn_module_declarations_bind_and_options():
    """`GDNModule` declares weights, smalls, states, outputs, and `cache_decl`."""
    cfg = {
        "hidden_size": 256,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    b = GDNModule.bind("linear_attn", gate="silu", rename={"lin_q": "q_ssm"})
    mod = b.to_module(cfg)
    decls = mod.hf_params()
    names = {d.name for d in decls}
    assert "q_ssm" in names and "lin_out" in names and "conv" in names
    cdecl = mod.cache_decl()
    assert cdecl.conv_taps == 3 and cdecl.KD == 128 and cdecl.VD == 128

    mod_bf16 = GDNModule(
        geometry=FakeGeometry(L=2, att_every=2, H=256),
        weight_dtypes={"lin_in": "bf16", "lin_out": "fp8"},
    )
    ex = ExecSpec(B=4, T=1, R=2, ctx=256, tiles=TINY_TILES)
    w_decls = {w.name: w for w in mod_bf16.weights(mod_bf16.geometry, ex)}
    assert w_decls["lin_in"].dtype == BF16 and w_decls["lin_out"].dtype == FP8

    with pytest.raises(MegakernelConfigError, match="gate"):
        GDNModule(gate="invalid_gate")
    with pytest.raises(MegakernelConfigError, match="gate"):
        GDNModule.bind("linear_attn", gate="sigmoidd")
    with pytest.raises(MegakernelConfigError, match="gate"):
        GDNModule.bind("linear_attn", gdn_gate="sigmoidd")
    with pytest.raises(MegakernelConfigError, match="weight_dtypes"):
        GDNModule(weight_dtypes={"bad_weight": "bf16"})

    b_sig = GDNModule.bind("linear_attn", gdn_gate="sigmoid")
    mod_sig = b_sig.to_module(cfg)
    assert mod_sig.gate == "sigmoid"


@pytest.mark.parametrize(
    ("T", "B", "chunked"),
    [(1, 4, False), (4, 4, False), (4, 8, True)],
    ids=["decode_t1", "verify_t4_rec", "verify_t4_chunked"],
)
def test_gdn_emit_in_pallas_interpret_matches_reference(T: int, B: int, chunked: bool):
    """`GDNModule.emit` in Pallas interpret mode matches `GDNModule.reference`."""
    tiles = dataclasses.replace(
        TINY_TILES, bank=256 << 10, wide_rows=4 if chunked else 16
    ).with_modules({"gdn": {"gdn_rows": 8, "gdn_bufs": 2 if chunked else 1}})
    ex = ExecSpec(B=B, T=T, R=1, ctx=256, stack_e=True, tiles=tiles, interpret=True)
    g = FakeGeometry(
        L=1,
        att_every=1,
        H=256,
        NH=2,
        NKV=1,
        HD=128,
        KH=2,
        VH=2,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=128,
        moe_tp=True,
    )
    gdn = GDNModule()
    mixer = SwitchSlot("mixer", branches=(gdn,), branch_counts=(1,))
    ms = ModelSpec(
        "gdn_exec", config=g, layer_type=LayerType("lt", slots=(mixer, FakeFFNModule()))
    )
    cp, scratch = _build_gdn_test_scratch(ms, ex)
    p = cp.plan
    assert p.module_params["gdn"].GCH is chunked
    wsh = planner.weight_shapes(ms, ex, p)
    ssh = planner.state_shapes(ms, ex, p)
    osh = planner.kernel_out_shapes(ms, ex, p)
    s_names = list(scratch.keys())

    def pallas_body(pos_ref, hid_ref, *rest):
        n_w, n_s, n_o = len(p.w_names), len(p.s_names), len(p.o_names)
        W = dict(zip(p.w_names, rest[:n_w]))
        ST = dict(zip(p.s_names, rest[n_w : n_w + n_s]))
        out_refs = dict(zip(p.o_names, rest[n_w + n_s : n_w + n_s + n_o]))
        scr = dict(zip(s_names, rest[n_w + n_s + n_o :]))
        me = jnp.int32(0)
        ring = _FakeGDNStream(cp.stream, W, scr)
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
        gdn.prologue(ctx)
        scr["res"][...] = hid_ref[: ex.MB].astype(F32)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        gdn.init_buffers(ctx)
        pltpu.make_async_copy(
            W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        ).wait()
        ctx.apply_input_norm(gdn.input_norm, jnp.int32(0))
        part = gdn.emit(ctx, jnp.int32(0), jnp.int32(0))
        scr["res"][...] = scr["res"][...] + part.astype(F32)
        gdn.epilogue(ctx)
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
    call = pl.pallas_call(
        pallas_body,
        out_shape=[jax.ShapeDtypeStruct(*osh[k]) for k in p.o_names],
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        interpret=pltpu.InterpretParams(),
    )

    key = jax.random.PRNGKey(17 + T + B)
    k_h, k_in, k_out, k_sm, k_c, k_r = jax.random.split(key, 6)
    pos = jnp.zeros((ex.B,), dtype=jnp.int32)
    hid = jax.random.normal(k_h, (ex.M, g.H), dtype=BF16) * 0.5
    W_in = {k: jnp.zeros(wsh[k][0], dtype=wsh[k][1]) for k in p.w_names}
    W_in["lin_in"] = (
        jax.random.normal(k_in, wsh["lin_in"][0], dtype=BF16) * 0.3
    ).astype(FP8)
    W_in["lin_out"] = (
        jax.random.normal(k_out, wsh["lin_out"][0], dtype=BF16) * 0.3
    ).astype(FP8)
    sm = jnp.zeros(wsh["smalls"][0], dtype=F32)
    for fname, val in (
        ("in_norm", 0.3),
        ("in_s", 0.2),
        ("out_s", 0.2),
        ("conv", 0.35),
        ("alog", -1.0),
        ("dtb", 0.2),
        ("gnorm", 1.0),
    ):
        o_sm, w_sm = cp.stream.lay[fname]
        sm = sm.at[:, :, o_sm : o_sm + w_sm].set(val)
    sm = sm + jax.random.normal(k_sm, sm.shape, dtype=F32) * 0.03
    W_in["smalls"] = sm

    ST_in = {k: jnp.zeros(ssh[k][0], dtype=ssh[k][1]) for k in p.s_names}
    ST_in["conv"] = jax.random.normal(k_c, ssh["conv"][0], dtype=F32) * 0.3
    ST_in["rec"] = jax.random.normal(k_r, ssh["rec"][0], dtype=BF16) * 0.3

    outs = dict(
        zip(
            p.o_names,
            call(
                pos,
                hid,
                *[W_in[k] for k in p.w_names],
                *[ST_in[k] for k in p.s_names],
            ),
        )
    )
    in_norm_w = sm[0, 0, cp.stream.lay["in_norm"][0] : sum(cp.stream.lay["in_norm"])]
    norm_x = rmsnorm(hid[:B].astype(F32), in_norm_w, g.eps, zero_centered=True)
    ref_params = {
        "lin_in": W_in["lin_in"].reshape(1, 1, g.H, -1),
        "lin_out": W_in["lin_out"].reshape(1, 1, -1, g.H),
    }
    for fname in ("in_s", "out_s", "conv", "alog", "dtb", "gnorm"):
        o_sm, w_sm = cp.stream.lay[fname]
        ref_params[fname] = sm[None, :, 0, o_sm : o_sm + w_sm]
    ref_st = mod_st = gdn.unpack_kernel_states(ST_in, g, ex)
    if T == 1:
        c_tb = mod_st["conv"][:, :, : 3 * B, :].reshape(1, 1, 3, B, -1)
        ref_st = {
            "conv": jnp.swapaxes(c_tb, 2, 3),
            "rec": mod_st["rec"].reshape(1, 1, B, 2, 128, 128),
        }
    else:
        ref_st = {
            "conv": mod_st["conv"][:, :, :, :3, :],
            "rec": mod_st["rec"].reshape(1, 1, ex.NS, 2, 128, 128),
        }
    ref_part, ref_out_st = gdn.reference(ref_params, norm_x, ref_st, g=g, ex=ex)
    assert float(np.max(np.abs(np.asarray(ref_part, dtype=np.float32)))) >= 0.2
    ref_hid = (hid[:B].astype(F32) + ref_part.astype(F32)).astype(BF16)
    np.testing.assert_allclose(
        outs["hidden"][:B].astype(F32), ref_hid.astype(F32), rtol=2e-2, atol=2e-2
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            outs["hidden"][:B].astype(F32), hid[:B].astype(F32), rtol=2e-2, atol=2e-2
        )

    conv_key = "conv_ckpt" if T > 1 else "conv"
    rec_key = "rec_ckpt" if T > 1 else "rec"
    conv_valid = slice(0, 3 * B) if T == 1 else (slice(None), slice(0, 3 + T))
    np.testing.assert_allclose(
        outs[conv_key][0][conv_valid].astype(F32),
        ref_out_st[conv_key][0][conv_valid].astype(F32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        outs[rec_key][0].astype(F32),
        ref_out_st[rec_key][0].astype(F32),
        rtol=2e-2,
        atol=2e-2,
    )
    rec_in_cmp = (
        jnp.repeat(ST_in["rec"][0].reshape(ex.NS, 2, 128, 128), T, axis=0).reshape(
            -1, 128, 128
        )
        if T > 1
        else ST_in["rec"][0]
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            outs[rec_key][0].astype(F32), rec_in_cmp.astype(F32), rtol=2e-2, atol=2e-2
        )

    # Also compare unpacked logical conv and rec states via `next_logical_states`.
    host_st = gdn.host_update_states(g, ex, pos, ST_in, outs)
    unpacked_st = gdn.unpack_kernel_states(host_st, g, ex)
    next_logical = gdn.next_logical_states(unpacked_st, g, ex)
    ref_next_logical = gdn.next_logical_states(
        {
            conv_key: ref_out_st[conv_key][:, None],
            rec_key: ref_out_st[rec_key][:, None],
        },
        g,
        ex,
    )
    np.testing.assert_allclose(
        next_logical["conv"].astype(F32),
        ref_next_logical["conv"].astype(F32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        next_logical["rec"].astype(F32),
        ref_next_logical["rec"].astype(F32),
        rtol=2e-2,
        atol=2e-2,
    )
