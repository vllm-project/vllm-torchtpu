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
"""Standalone unit tests for slice L-6b (`modules/gqa` Pallas core helpers)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tests.megakernel.megakernel_fakes import (
    TINY_TILES,
    FakeGeometry,
    FakeMixerModule,
    MiniCtx,
)
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, register_tunables
from vllm_torchtpu.megakernel.modules.gqa.attention import (
    att_core,
    item_back,
    item_front,
)
from vllm_torchtpu.megakernel.modules.gqa.context import GQAEmit
from vllm_torchtpu.megakernel.modules.gqa.kv import (
    KVRing,
    build_worklist,
    make_kv_ring,
)
from vllm_torchtpu.megakernel.modules.gqa.new_token import att_new
from vllm_torchtpu.megakernel.modules.gqa.params import (
    GQATunables,
    derive_params,
    gqa_scratch,
)
from vllm_torchtpu.megakernel.modules.gqa.reference import gqa_reference
from vllm_torchtpu.megakernel.modules.gqa.staging import (
    att_q,
    kv_new,
    q_exchange_start,
    q_gather_finish,
    rms_zc,
    rope,
)
from vllm_torchtpu.megakernel.ops import (
    BF16,
    F32,
    I32,
    rmsnorm,
    rope_partial,
    round_bf16,
)
from vllm_torchtpu.megakernel.planner import plan
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec


class _FakeStreamParams:
    def __init__(self, lay: dict[str, tuple[int, int]]) -> None:
        self.lay = lay

    def module_ord(self, lyr: Any, module_name: str) -> Any:
        del module_name
        return lyr


class _FakeStream:
    """Minimal stream stub exercising `dmatmul` and `mm` inside Pallas."""

    def __init__(self, sp: _FakeStreamParams, w_refs: dict[str, Any]) -> None:
        self.sp = sp
        self._w = w_refs

    def dmatmul(self, kind: str, x: Any, n: int) -> Any:
        del n
        return lax.dot_general(
            x[...].astype(F32),
            self._w[kind][0].astype(F32),
            (((1,), (0,)), ((), ())),
            preferred_element_type=F32,
        )

    def mm(self, a: Any, b: Any, dims: Any) -> Any:
        return lax.dot_general(
            a.astype(F32),
            b.astype(F32),
            dims,
            preferred_element_type=F32,
        )


class _FakeCorePlan:
    def __init__(self, sp: _FakeStreamParams) -> None:
        self.stream = sp


def test_gqa_staging_rms_zc_and_rope_helpers():
    """Verify `rms_zc` and `rope` (`rope_fp32=False` vs `True`) helpers."""
    g = FakeGeometry(L=1, att_every=1, H=256, NH=2, NKV=1, HD=128, ROT=32)
    ms = ModelSpec("m", config=g, layer_type=LayerType("lt", (FakeMixerModule(),)))
    ex = ExecSpec(B=4, T=1, R=1, ctx=256, tiles=TINY_TILES)
    p = plan(ms, ex)
    ap_bf16 = derive_params(g, ex, p, GQATunables(kv_blk=128, nkv=2, rope_fp32=False))
    ap_fp32 = derive_params(g, ex, p, GQATunables(kv_blk=128, nkv=2, rope_fp32=True))

    x = jnp.arange(2 * 128, dtype=F32).reshape(2, 128) * 0.01
    w = jnp.ones((128,), dtype=F32) * 0.2
    np.testing.assert_allclose(
        rms_zc(ap_bf16, x, w), rmsnorm(x, w, g.eps, zero_centered=True)
    )
    cos = jnp.cos(jnp.arange(16, dtype=F32))[None, :]
    sin = jnp.sin(jnp.arange(16, dtype=F32))[None, :]
    np.testing.assert_allclose(
        rope(ap_bf16, x, cos, sin),
        rope_partial(x, cos, sin, g.ROT, round_intermediates=True),
    )
    np.testing.assert_allclose(
        rope(ap_fp32, x, cos, sin),
        rope_partial(x, cos, sin, g.ROT, round_intermediates=False),
    )


@pytest.mark.parametrize(
    ("T", "pair_kv", "gqa_qv"),
    [(1, False, False), (2, True, True)],
    ids=["decode_t1_single", "verify_t2_pair_qv"],
)
def test_gqa_pallas_core_helpers_match_gqa_reference(
    T: int, pair_kv: bool, gqa_qv: bool
):
    """Execute GQA Pallas core helpers and compare against `gqa_reference`."""
    B = 4
    ctx_len = 256
    g = FakeGeometry(L=1, att_every=1, H=256, NH=2, NKV=1, HD=128, ROT=32)
    tt = GQATunables(kv_blk=128, nkv=2, split=False, pair_kv=pair_kv, gqa_qv=gqa_qv)
    register_tunables("fake_gqa_ref", GQATunables)
    ex = ExecSpec(B=B, T=T, R=1, ctx=ctx_len, tiles=TINY_TILES, interpret=True)
    ms = ModelSpec(
        "gqa_core", config=g, layer_type=LayerType("lt", (FakeMixerModule(),))
    )
    p = plan(ms, ex)
    ap = derive_params(g, ex, p, tt, in_scaled=False, out_scaled=False)

    scratch = gqa_scratch(ex, ap)
    scratch.update(
        {
            "xb": pltpu.VMEM((ex.MB, g.H), BF16),
            "ybuf": pltpu.VMEM((ex.MB, ap.NAI), F32),
            "mix": pltpu.VMEM((ex.MB, ap.nh * g.HD), F32),
            "sm": pltpu.VMEM((2, 1, p.smalls_width), F32),
        }
    )
    s_names = list(scratch.keys())
    sp = _FakeStreamParams(p.smalls_layout)
    cp_stub = _FakeCorePlan(sp)

    def pallas_body(
        pos_ref,
        x_ref,
        w_in_ref,
        w_out_ref,
        sm_ref,
        kc_ref,
        vc_ref,
        out_mix_ref,
        out_knew_ref,
        out_vnew_ref,
        *scr_refs,
    ):
        scr = dict(zip(s_names, scr_refs))
        w_refs = {"att_in": w_in_ref, "att_out": w_out_ref}
        st_refs = {"kc": kc_ref, "vc": vc_ref}
        out_refs = {"knew": out_knew_ref, "vnew": out_vnew_ref}
        f_stream = _FakeStream(sp, w_refs)
        ctx = MiniCtx(
            g=g,
            ex=ex,
            core_plan=cp_stub,
            pos_ref=pos_ref,
            W=w_refs,
            ST=st_refs,
            O=out_refs,
            scr=scr,
            me=jnp.int32(0),
            stream=f_stream,
            comm=None,
        )
        scr["sm"][0] = sm_ref[0]
        scr["xb"][: ex.B] = x_ref[: ex.B].astype(BF16)
        if ex.B < ex.MB:
            scr["xb"][ex.B :] = jnp.zeros((ex.MB - ex.B, g.H), BF16)
        scr["knew"][...] = jnp.zeros((ex.MR, ap.HD), F32)
        scr["vnew"][...] = jnp.zeros((ex.MR, ap.HD), F32)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        if ap.QV:
            scr["qg"][...] = jnp.zeros(scr["qg"].shape, F32)
            half_ = ap.ROT // 2
            inv_ = 1.0 / (
                ap.theta
                ** (lax.broadcasted_iota(I32, (1, half_), 1).astype(F32) * 2.0 / ap.ROT)
            )
            rows_ = lax.broadcasted_iota(I32, (ex.MR, half_), 0)
            pf = jnp.zeros((ex.MR, half_), F32)
            for b in range(ex.B):
                pf = jnp.where(rows_ == b, pos_ref[b].astype(F32), pf)
            ang_ = pf * inv_
            scr["rope"][0, :, :half_] = round_bf16(jnp.cos(ang_))
            scr["rope"][1, :, :half_] = round_bf16(jnp.sin(ang_))

        build_worklist(ctx, ap)
        cx = GQAEmit.build(ctx, ap, jnp.int32(0), "fake_gqa")
        kv_ring = make_kv_ring(cx)
        assert isinstance(kv_ring, KVRing)
        ybuf, kn, half, koff, rcs = att_q(cx, jnp.int32(0))
        qx = q_exchange_start(cx)
        kv_out = kv_new(cx, ybuf, kn, half, koff, rcs)
        if ap.SPLIT or ex.VER:
            q_gather_finish(cx, qx)
        # Exercise item_front / item_back via att_core
        assert callable(item_front) and callable(item_back)
        att_core(cx, kv_ring)
        att_new(cx)
        for d in kv_out:
            d.wait()
        yo = f_stream.dmatmul(
            "att_out", scr["mix"][:, : ap.nh * ap.HD].astype(BF16), g.H
        )
        out_mix_ref[...] = yo[: ex.B].astype(BF16)

    call = pl.pallas_call(
        pallas_body,
        out_shape=[
            jax.ShapeDtypeStruct((B, g.H), BF16),
            jax.ShapeDtypeStruct((1, ex.MR, g.HD), F32),
            jax.ShapeDtypeStruct((1, ex.MR, g.HD), F32),
        ],
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        scratch_shapes=[scratch[k] for k in s_names],
        interpret=pltpu.InterpretParams(),
    )

    keys = jax.random.split(jax.random.PRNGKey(31 + T), 6)
    pos = jnp.array([140 + (b % T) for b in range(B)], dtype=I32)
    x_in = jax.random.normal(keys[0], (ex.MB, g.H), dtype=BF16) * 0.2
    w_in = jax.random.normal(keys[1], (1, g.H, ap.NAI), dtype=BF16) * 0.1
    w_out = jax.random.normal(keys[2], (1, ap.nh * g.HD, g.H), dtype=BF16) * 0.1
    qn = jax.random.normal(keys[3], (1, g.HD), dtype=F32) * 0.1 + 1.0
    kn = jax.random.normal(keys[4], (1, g.HD), dtype=F32) * 0.1 + 1.0
    sm = jnp.zeros((1, 1, p.smalls_width), dtype=F32)
    for fname, val in (("qn", qn), ("kn", kn)):
        o_sm, w_sm = p.smalls_layout[fname]
        sm = sm.at[:, 0, o_sm : o_sm + w_sm].set(val)
    kc = jax.random.normal(keys[5], (1, ex.NS, ctx_len, g.HD), dtype=BF16) * 0.1
    vc = jax.random.normal(keys[5], (1, ex.NS, ctx_len, g.HD), dtype=BF16) * 0.05

    yo_out, knew_out, vnew_out = jax.jit(call)(pos, x_in, w_in, w_out, sm, kc, vc)
    ref_delta, ref_st = gqa_reference(
        {"att_in": w_in, "att_out": w_out, "qn": qn, "kn": kn},
        x_in[:B],
        {"kc": kc, "vc": vc},
        g=g,
        ex=ex,
        pos=pos,
        module_name="fake_gqa_ref",
    )
    np.testing.assert_allclose(
        yo_out.astype(F32), ref_delta.astype(F32), rtol=2e-2, atol=2e-2
    )
    np.testing.assert_allclose(
        knew_out[0, :T].astype(BF16),
        ref_st["kc"][0, 0, pos[0] : pos[0] + T],
        rtol=1e-2,
        atol=1e-2,
    )
    np.testing.assert_allclose(
        vnew_out[0, :T].astype(BF16),
        ref_st["vc"][0, 0, pos[0] : pos[0] + T],
        rtol=1e-2,
        atol=1e-2,
    )
