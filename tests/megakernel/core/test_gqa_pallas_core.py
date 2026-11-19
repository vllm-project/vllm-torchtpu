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
"""Standalone unit tests for GQA Pallas KV ring, staging, and attention helpers."""

from __future__ import annotations

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
    FakeMixerModule,
    MiniCtx,
)
from vllm_torchtpu.megakernel.core import pool
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.modules.gqa.attention import att_core
from vllm_torchtpu.megakernel.core.modules.gqa.context import GQAEmit
from vllm_torchtpu.megakernel.core.modules.gqa.kv import (
    KVRing,
    build_worklist,
    make_kv_ring,
)
from vllm_torchtpu.megakernel.core.modules.gqa.new_token import att_new
from vllm_torchtpu.megakernel.core.modules.gqa.params import (
    GQATunables,
    derive_params,
    gqa_scratch,
)
from vllm_torchtpu.megakernel.core.modules.gqa.reference import gqa_reference
from vllm_torchtpu.megakernel.core.modules.gqa.staging import (
    att_q,
    gqa_rms,
    kv_new,
    q_exchange_start,
    q_gather_finish,
    rope,
)
from vllm_torchtpu.megakernel.core.ops import (
    BF16,
    F32,
    FP8,
    I32,
    rmsnorm,
    rope_partial,
    round_bf16,
)
from vllm_torchtpu.megakernel.core.planner import plan
from vllm_torchtpu.megakernel.core.pool import CacheExec, PoolLayout
from vllm_torchtpu.megakernel.core.spec import LayerType, ModelSpec


class _FakeStreamParams:
    def __init__(self, lay: dict[str, tuple[int, int]]) -> None:
        self.lay = lay

    def module_ord(self, lyr: Any, module_name: str) -> Any:
        del module_name
        return lyr


class _FakeStream:
    """Minimal stream stub exercising `dmatmul`, `mm`, and `borrow_epool`."""

    def __init__(
        self,
        sp: _FakeStreamParams,
        w_refs: dict[str, Any],
        scr: dict[str, Any],
        *,
        interpret: bool,
    ) -> None:
        self.sp = sp
        self._w = w_refs
        self._scr = scr
        self.interpret = interpret

    def dmatmul(self, kind: str, x: Any, n: int) -> Any:
        del n
        x_val = x[...]
        w_val = self._w[kind][0]
        if self.interpret:
            x_val = x_val.astype(F32)
            w_val = w_val.astype(F32)
        return lax.dot_general(
            x_val,
            w_val,
            (((1,), (0,)), ((), ())),
            preferred_element_type=F32,
        )

    def mm(self, a: Any, b: Any, dims: Any) -> Any:
        if self.interpret and a.dtype != b.dtype:
            b = b.astype(a.dtype)
        return lax.dot_general(
            a,
            b,
            dims,
            preferred_element_type=F32,
        )

    def borrow_epool(self) -> Any:
        return self._scr["epool"]


class _FakeCorePlan:
    def __init__(self, sp: _FakeStreamParams) -> None:
        self.stream = sp


def test_gqa_staging_gqa_rms_and_rope_helpers():
    """Verify `gqa_rms` and `rope` (`rope_fp32` flag mapping and rotation formula)."""
    g = FakeGeometry(L=1, att_every=1, H=256, NH=2, NKV=1, HD=128, ROT=32)
    ms = ModelSpec("m", config=g, layer_type=LayerType("lt", (FakeMixerModule(),)))
    ex = ExecSpec(B=4, T=1, R=1, ctx=256, tiles=TINY_TILES)
    p = plan(ms, ex)
    ap_bf16 = derive_params(g, ex, p, GQATunables(kv_blk=128, nkv=2, rope_fp32=False))
    ap_fp32 = derive_params(g, ex, p, GQATunables(kv_blk=128, nkv=2, rope_fp32=True))

    x = jnp.arange(2 * 128, dtype=F32).reshape(2, 128) * 0.01
    w = jnp.ones((128,), dtype=F32) * 0.2
    np.testing.assert_allclose(
        gqa_rms(ap_bf16, x, w), rmsnorm(x, w, g.eps, zero_centered=True)
    )
    cos = jnp.cos(jnp.arange(16, dtype=F32))[None, :]
    sin = jnp.sin(jnp.arange(16, dtype=F32))[None, :]
    half = g.ROT // 2
    x0, x1 = x[:, :half], x[:, half : g.ROT]
    direct_rot = jnp.concatenate(
        [x0 * cos - x1 * sin, x1 * cos + x0 * sin, x[:, g.ROT :]], axis=-1
    )
    np.testing.assert_allclose(
        rope(ap_bf16, x, cos, sin),
        rope_partial(x, cos, sin, g.ROT, round_intermediates=True),
    )
    np.testing.assert_allclose(
        rope(ap_fp32, x, cos, sin),
        direct_rot,
        rtol=1e-6,
        atol=1e-6,
    )


def _run_gqa_pallas_core_case(
    *,
    T: int,
    paged_kv: bool,
    kv_ering: bool,
    pair_kv: bool,
    gqa_qv: bool,
    interpret: bool,
    kv_blk: int = 128,
    kv_sub: int = 0,
    amask: int = 1,
) -> None:
    """Run GQA Pallas core helpers and compare against `gqa_reference`."""
    B = 4
    ctx_len = 256
    g = FakeGeometry(L=1, att_every=1, H=256, NH=1, NKV=1, HD=256, ROT=32)
    tt = GQATunables(
        kv_blk=kv_blk,
        kv_sub=kv_sub,
        amask=amask,
        nkv=2,
        split=False,
        pair_kv=pair_kv,
        gqa_qv=gqa_qv,
        kv_ering=kv_ering,
    )
    tiles = TINY_TILES.with_modules({"gqa": tt})
    cache = (
        CacheExec(paged_kv=True, page_size=128, num_pages=32)
        if paged_kv
        else CacheExec()
    )
    ex = ExecSpec(
        B=B,
        T=T,
        R=1,
        ctx=ctx_len,
        stack_e=True,
        tiles=tiles,
        cache=cache,
        interpret=interpret,
    )
    ms = ModelSpec(
        "gqa_core",
        config=g,
        layer_type=LayerType("lt", (FakeMixerModule(), FakeFFNModule())),
    )
    p = plan(ms, ex)
    ap = derive_params(g, ex, p, tt, in_scaled=False, out_scaled=False)
    assert ap.KVE is (paged_kv and kv_ering)

    scratch = gqa_scratch(ex, ap)
    scratch.update(
        {
            "xb": pltpu.VMEM((ex.MB, g.H), BF16),
            "ybuf": pltpu.VMEM((ex.MB, ap.NAI), F32),
            "mix": pltpu.VMEM((ex.MB, ap.nh * g.HD), F32),
            "sm": pltpu.VMEM((2, 1, p.smalls_width), F32),
        }
    )
    if ap.KVE:
        nbe = ex.resolved_tiles.nbe
        scratch["epool"] = (
            pltpu.VMEM((nbe, 2, g.HD // 4, 4, ap.BLK), FP8)
            if interpret
            else pltpu.VMEM((nbe * ap.EBROWS, 128), jnp.uint8)
        )
    s_names = list(scratch.keys())
    sp = _FakeStreamParams(p.smalls_layout)
    cp_stub = _FakeCorePlan(sp)
    kv_out_dt = FP8 if paged_kv else F32

    def pallas_body(
        pos_ref,
        pt_or_dummy_ref,
        x_ref,
        w_in_ref,
        w_out_ref,
        sm_ref,
        st0_ref,
        st1_ref,
        out_mix_ref,
        out_knew_ref,
        out_vnew_ref,
        *scr_refs,
    ):
        scr = dict(zip(s_names, scr_refs))
        w_refs = {"att_in": w_in_ref, "att_out": w_out_ref}
        if paged_kv:
            st_refs = {"page_table": pt_or_dummy_ref}
            raw_refs: Any = (st0_ref,)
        else:
            st_refs = {"kc": st0_ref, "vc": st1_ref}
            raw_refs = None
        out_refs = {"knew": out_knew_ref, "vnew": out_vnew_ref}
        f_stream = _FakeStream(sp, w_refs, scr, interpret=interpret)
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
            raw_refs=raw_refs,
        )
        scr["sm"][0] = sm_ref[0]
        scr["xb"][: ex.B] = x_ref[: ex.B].astype(BF16)
        if ex.B < ex.MB:
            scr["xb"][ex.B :] = jnp.zeros((ex.MB - ex.B, g.H), BF16)
        scr["knew"][...] = jnp.zeros((ex.MR, ap.HD), F32)
        scr["vnew"][...] = jnp.zeros((ex.MR, ap.HD), F32)
        if paged_kv:
            scr["knew_o"][...] = jnp.zeros((ex.MR, ap.HD), FP8)
            scr["vnew_o"][...] = jnp.zeros((ex.MR, ap.HD), FP8)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        if ex.VER or ap.QV:
            scr["qg"][...] = jnp.zeros(scr["qg"].shape, F32)
        if ap.QV:
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
        cx = GQAEmit.build(ctx, ap, jnp.int32(0), "gqa")
        kv_ring = make_kv_ring(cx)
        assert isinstance(kv_ring, KVRing)
        ybuf, kn, half, koff, rcs = att_q(cx, jnp.int32(0))
        qx = q_exchange_start(cx)
        kv_out = kv_new(cx, ybuf, kn, half, koff, rcs)
        if ap.SPLIT or ex.VER:
            q_gather_finish(cx, qx)
        att_core(cx, kv_ring)
        att_new(cx)
        for d in kv_out:
            d.wait()
        yo = f_stream.dmatmul(
            "att_out", scr["mix"][:, : ap.nh * ap.HD].astype(BF16), g.H
        )
        out_mix_ref[: ex.B] = yo[: ex.B].astype(BF16)
        if ex.B < ex.MB:
            out_mix_ref[ex.B :] = jnp.zeros((ex.MB - ex.B, g.H), BF16)

    call = pl.pallas_call(
        pallas_body,
        out_shape=[
            jax.ShapeDtypeStruct((ex.MB, g.H), BF16),
            jax.ShapeDtypeStruct((1, ex.MR, g.HD), kv_out_dt),
            jax.ShapeDtypeStruct((1, ex.MR, g.HD), kv_out_dt),
        ],
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
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
        interpret=pltpu.InterpretParams() if interpret else False,
    )

    cpu_dev = jax.devices("cpu")[0]
    with jax.default_device(cpu_dev):
        keys = jax.random.split(
            jax.random.PRNGKey(
                31 + T + 10 * int(paged_kv) + 20 * int(kv_ering) + 40 * int(pair_kv)
            ),
            7,
        )
        # Sequence 0 has plen=140 (2 blocks of 128), sequence 1 has plen=60 (1 block),
        # so when T=2 (NS=2) total worklist items = 2 + 1 = 3 (odd!), exercising
        # both `item_pair` and the odd-item tail in `att_core` when `pair_kv=True`.
        seq_plens = [140, 60, 140, 60]
        pos = jnp.array(
            [seq_plens[b // T] + (b % T) for b in range(B)],
            dtype=I32,
        )
        x_in = jax.random.normal(keys[0], (ex.MB, g.H), dtype=BF16) * 0.8
        w_in = jax.random.normal(keys[1], (1, g.H, ap.NAI), dtype=BF16) * 0.4
        w_out = jax.random.normal(keys[2], (1, ap.nh * g.HD, g.H), dtype=BF16) * 0.5
        qn = jax.random.normal(keys[3], (1, g.HD), dtype=F32) * 0.2 + 1.0
        kn = jax.random.normal(keys[4], (1, g.HD), dtype=F32) * 0.2 + 1.0
        sm = jnp.zeros((1, 1, p.smalls_width), dtype=F32)
        for fname, val in (("qn", qn), ("kn", kn)):
            o_sm, w_sm = p.smalls_layout[fname]
            sm = sm.at[:, 0, o_sm : o_sm + w_sm].set(val)
        kc = jax.random.normal(keys[5], (1, ex.NS, ctx_len, g.HD), dtype=BF16) * 0.6
        # Give `vc` a non-zero mean + variation so cached KV contributes O(1) to `yo`.
        vc = jax.random.normal(
            keys[6], (1, ex.NS, ctx_len, g.HD), dtype=BF16
        ) * 0.6 + jnp.asarray(0.5, dtype=BF16)

        ref_ex = ExecSpec(
            B=B,
            T=T,
            R=1,
            ctx=ctx_len,
            stack_e=True,
            tiles=tiles,
            cache=cache,
            interpret=True,
        )
        ref_delta, ref_st = gqa_reference(
            {"att_in": w_in, "att_out": w_out, "qn": qn, "kn": kn},
            x_in[:B],
            {"kc": kc, "vc": vc},
            g=g,
            ex=ref_ex,
            pos=pos,
        )
        ref_zero_vc, _ = gqa_reference(
            {"att_in": w_in, "att_out": w_out, "qn": qn, "kn": kn},
            x_in[:B],
            {"kc": kc, "vc": jnp.zeros_like(vc)},
            g=g,
            ex=ref_ex,
            pos=pos,
        )

        if paged_kv:
            lay = PoolLayout(num_pools=1, slots_per_pool=0, hd=g.HD, kvh=1)
            pt, ckpt, roff, _ = pool.allocate_test_layout(lay, ref_ex)
            kc_fp8 = jnp.clip(kc.astype(F32), -448.0, 448.0).astype(FP8)
            vc_fp8 = jnp.clip(vc.astype(F32), -448.0, 448.0).astype(FP8)
            nan_fp8 = lax.bitcast_convert_type(jnp.uint8(0x7F), FP8)
            for sq in range(ex.NS):
                p_end = int(pos[sq * T])
                vc_fp8 = vc_fp8.at[:, sq, p_end:, :].set(nan_fp8)
            raw_pools = pool.states_to_pool(
                lay,
                ref_ex,
                {"kc": kc_fp8[:, None], "vc": vc_fp8[:, None]},
                pt,
                ckpt,
                roff,
            )
            pt_arg = pt
            st0_arg = raw_pools[0]
            st1_arg = raw_pools[0]
        else:
            pt_arg = jnp.zeros((ex.NS, 2), dtype=I32)
            st0_arg = kc
            # Poison DirectHBM `vc` from each sequence's `plen` on with NaN to
            # verify that `item_front` masks V rows from `plen` on before `pr @ vj`.
            vc_poisoned = vc
            for sq in range(ex.NS):
                p_end = int(pos[sq * T])
                vc_poisoned = vc_poisoned.at[:, sq, p_end:, :].set(
                    jnp.asarray(jnp.nan, dtype=BF16)
                )
            st1_arg = vc_poisoned

    run_dev = cpu_dev if interpret else jax.devices("tpu")[0]
    pos_d, pt_d, x_d, w_in_d, w_out_d, sm_d, st0_d, st1_d = [
        jax.device_put(arr, run_dev)
        for arr in (pos, pt_arg, x_in, w_in, w_out, sm, st0_arg, st1_arg)
    ]
    yo_full, knew_out, vnew_out = jax.jit(call)(
        pos_d, pt_d, x_d, w_in_d, w_out_d, sm_d, st0_d, st1_d
    )
    yo_out = np.asarray(yo_full[:B].astype(F32))
    ref_delta_f32 = np.asarray(ref_delta.astype(F32))
    ref_zero_vc_f32 = np.asarray(ref_zero_vc.astype(F32))

    # Ensure output magnitude is O(1) and the tolerance catches broken attention.
    max_ref = float(np.max(np.abs(ref_delta_f32)))
    assert max_ref >= 0.5
    atol = 5e-3 * max_ref
    np.testing.assert_allclose(yo_out, ref_delta_f32, rtol=2e-2, atol=atol)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            yo_out, np.zeros_like(ref_delta_f32), rtol=2e-2, atol=atol
        )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(yo_out, ref_zero_vc_f32, rtol=2e-2, atol=atol)

    # Check every sequence's new K/V rows (`knew` / `vnew`).
    for sq in range(ex.NS):
        p_sq = int(pos[sq * T])
        np.testing.assert_allclose(
            np.asarray(
                knew_out[0, sq * T : (sq + 1) * T].astype(BF16), dtype=np.float32
            ),
            np.asarray(ref_st["kc"][0, sq, p_sq : p_sq + T], dtype=np.float32),
            rtol=1e-2,
            atol=1e-2,
        )
        np.testing.assert_allclose(
            np.asarray(
                vnew_out[0, sq * T : (sq + 1) * T].astype(BF16), dtype=np.float32
            ),
            np.asarray(ref_st["vc"][0, sq, p_sq : p_sq + T], dtype=np.float32),
            rtol=1e-2,
            atol=1e-2,
        )


@pytest.mark.parametrize(
    ("T", "paged_kv", "kv_ering", "pair_kv", "gqa_qv", "kv_blk", "kv_sub", "amask"),
    [
        (1, False, False, False, False, 128, 0, 1),
        (2, False, False, True, True, 128, 0, 0),
        (1, False, False, False, False, 256, 128, 2),
        (2, False, False, True, True, 256, 128, 3),
        (1, False, False, False, False, 256, 128, 4),
        (1, True, False, False, False, 128, 0, 0),
        (2, True, True, True, True, 128, 0, 1),
        (2, True, False, False, True, 256, 128, 2),
    ],
    ids=[
        "decode_t1_direct_hbm_amask1",
        "verify_t2_direct_hbm_odd_items_amask0",
        "decode_t1_direct_hbm_sub128_amask2",
        "verify_t2_direct_hbm_sub128_amask3",
        "decode_t1_direct_hbm_sub128_amask4",
        "decode_t1_paged_kv_no_kve_amask0",
        "verify_t2_paged_kv_kve_odd_items_amask1",
        "verify_t2_paged_kv_sub128_amask2",
    ],
)
def test_gqa_pallas_core_helpers_match_gqa_reference(
    T: int,
    paged_kv: bool,
    kv_ering: bool,
    pair_kv: bool,
    gqa_qv: bool,
    kv_blk: int,
    kv_sub: int,
    amask: int,
):
    """Execute GQA Pallas core helpers in CPU interpret mode vs `gqa_reference`."""
    _run_gqa_pallas_core_case(
        T=T,
        paged_kv=paged_kv,
        kv_ering=kv_ering,
        pair_kv=pair_kv,
        gqa_qv=gqa_qv,
        interpret=True,
        kv_blk=kv_blk,
        kv_sub=kv_sub,
        amask=amask,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("T", "paged_kv", "kv_ering", "pair_kv", "gqa_qv", "kv_blk", "kv_sub", "amask"),
    [
        (1, False, False, False, False, 128, 0, 1),
        (1, False, False, False, False, 256, 128, 2),
        (2, False, False, True, True, 256, 128, 3),
        (1, False, False, False, False, 256, 128, 4),
        (2, True, False, True, True, 128, 0, 0),
        (2, True, True, True, True, 128, 0, 1),
        (2, True, False, False, True, 256, 128, 2),
    ],
    ids=[
        "tpu_direct_hbm_t1_amask1",
        "tpu_direct_hbm_t1_sub128_amask2",
        "tpu_direct_hbm_t2_sub128_amask3",
        "tpu_direct_hbm_t1_sub128_amask4",
        "tpu_paged_kv_no_kve_t2_amask0",
        "tpu_paged_kv_kve_t2_amask1",
        "tpu_paged_kv_no_kve_t2_sub128_amask2",
    ],
)
def test_gqa_pallas_core_helpers_on_tpu(
    T: int,
    paged_kv: bool,
    kv_ering: bool,
    pair_kv: bool,
    gqa_qv: bool,
    kv_blk: int,
    kv_sub: int,
    amask: int,
):
    """Execute GQA Pallas core helpers on TPU (`interpret=False`) vs CPU reference."""
    try:
        tpu_devs = jax.devices("tpu")
    except RuntimeError:
        tpu_devs = []
    if not tpu_devs:
        pytest.skip("TPU device required for interpret=False GQA Pallas core test")
    _run_gqa_pallas_core_case(
        T=T,
        paged_kv=paged_kv,
        kv_ering=kv_ering,
        pair_kv=pair_kv,
        gqa_qv=gqa_qv,
        interpret=False,
        kv_blk=kv_blk,
        kv_sub=kv_sub,
        amask=amask,
    )
