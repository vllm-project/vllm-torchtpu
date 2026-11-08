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
"""Unit tests for `modules/gdn/{states,conv,recurrence}` Pallas helpers."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tests.megakernel.megakernel_fakes import (
    TINY_TILES,
    FakeFFNModule,
    FakeGeometry,
    FakeMixerModule,
    MiniCtx,
)
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.modules.gdn.conv import conv_step, gdn_in
from vllm_torchtpu.megakernel.modules.gdn.params import (
    GDNParams,
    GDNTunables,
    derive_params,
)
from vllm_torchtpu.megakernel.modules.gdn.recurrence import (
    gdn_rec,
    gdn_rec_chunked,
)
from vllm_torchtpu.megakernel.modules.gdn.reference import gdn_reference
from vllm_torchtpu.megakernel.modules.gdn.states import (
    conv_out_src,
    gdn_accept,
    gdn_init_logical_states,
    gdn_next_logical_states,
    gdn_pack_logical_states,
    gdn_unpack_kernel_states,
    rec_drain,
    rec_in,
)
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8, round_up
from vllm_torchtpu.megakernel.planner import plan
from vllm_torchtpu.megakernel.pool import (
    CacheExec,
    PoolBackend,
    PoolLayout,
    allocate_test_layout,
    states_to_pool,
)
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec, SmallField


def test_gdn_logical_state_pack_unpack_and_accept():
    """GDN state helpers pack/unpack rank-folded states and select accepted tokens."""
    g = FakeGeometry(
        L=2, att_every=2, H=256, KH=4, VH=4, KD=128, VD=128, NH=4, NKV=2, HD=128
    )
    ex_d = ExecSpec(B=4, T=1, R=2, ctx=256, tiles=TINY_TILES)
    st_d = gdn_init_logical_states(
        jax.random.PRNGKey(0), g, ex_d, n_layers=1, conv_taps=3
    )
    assert st_d["conv"].shape == (2, 1, 4, 3, 768)
    assert st_d["rec"].shape == (2, 1, 4, 2, 128, 128)
    packed_d = gdn_pack_logical_states(st_d, g, ex_d, conv_taps=3)
    unpacked_d = gdn_unpack_kernel_states(packed_d, g, ex_d, conv_taps=3)
    next_d = gdn_next_logical_states(unpacked_d, g, ex_d, conv_taps=3)
    np.testing.assert_allclose(next_d["conv"], st_d["conv"], rtol=0, atol=0)
    np.testing.assert_allclose(next_d["rec"], st_d["rec"], rtol=0, atol=0)

    ex_v = ExecSpec(B=8, T=4, R=2, ctx=256, tiles=TINY_TILES)
    st_v = gdn_init_logical_states(
        jax.random.PRNGKey(1), g, ex_v, n_layers=1, conv_taps=3
    )
    packed_v = gdn_pack_logical_states(st_v, g, ex_v, conv_taps=3)
    c_ckpt = jnp.arange(2 * 2 * 8 * 768, dtype=F32).reshape(2, 2, 8, 768)
    r_ckpt = jnp.arange(2 * (2 * 4 * 2) * 128 * 128, dtype=F32).astype(BF16)
    r_ckpt = r_ckpt.reshape(2, 2 * 4 * 2, 128, 128)
    r_ckpt_unflat = r_ckpt.reshape(2, 2, 4, 2, 128, 128)
    ckpts = {"conv_ckpt": c_ckpt, "rec_ckpt": r_ckpt}
    acc = jnp.array([2, 4], dtype=jnp.int32)
    accepted = gdn_accept(packed_v, ckpts, acc, T=4, vh=2, taps=3)
    np.testing.assert_array_equal(accepted["conv"][:, 0, :3], c_ckpt[:, 0, 2:5])
    np.testing.assert_array_equal(accepted["conv"][:, 1, :3], c_ckpt[:, 1, 4:7])
    acc_rec = accepted["rec"].reshape(2, 2, 2, 128, 128)
    np.testing.assert_array_equal(acc_rec[:, 0], r_ckpt_unflat[:, 0, 1])
    np.testing.assert_array_equal(acc_rec[:, 1], r_ckpt_unflat[:, 1, 3])

    # acc == 0 keeps the sequence's previous rec state
    acc_zero = jnp.array([0, 3], dtype=jnp.int32)
    accepted_z = gdn_accept(packed_v, ckpts, acc_zero, T=4, vh=2, taps=3)
    np.testing.assert_array_equal(accepted_z["conv"][:, 0, :3], c_ckpt[:, 0, 0:3])
    acc_z_rec = accepted_z["rec"].reshape(2, 2, 2, 128, 128)
    prev_rec = packed_v["rec"].reshape(2, 2, 2, 128, 128)
    np.testing.assert_array_equal(acc_z_rec[:, 0], prev_rec[:, 0])
    np.testing.assert_array_equal(acc_z_rec[:, 1], r_ckpt_unflat[:, 1, 2])

    # Unpadded 5-D logical layout infers `taps=3` when `taps=None`
    c_ckpt_5d = c_ckpt[:, None, :, :7, :]
    r_ckpt_5d = r_ckpt[:, None]
    accepted_5d = gdn_accept(
        st_v, {"conv_ckpt": c_ckpt_5d, "rec_ckpt": r_ckpt_5d}, acc, T=4, vh=2
    )
    np.testing.assert_array_equal(accepted_5d["conv"][:, 0, 0, :3], c_ckpt[:, 0, 2:5])
    acc_5d_rec = accepted_5d["rec"].reshape(2, 1, 2, 2, 128, 128)
    np.testing.assert_array_equal(acc_5d_rec[:, 0, 0], r_ckpt_unflat[:, 0, 1])
    np.testing.assert_array_equal(acc_5d_rec[:, 0, 1], r_ckpt_unflat[:, 1, 3])

    with pytest.raises(ValueError, match="requires explicit `taps`"):
        gdn_accept(packed_v, ckpts, acc, T=4, vh=2, taps=None)


def _gdn_core_scratch(gp: GDNParams, ex: ExecSpec, sm_width: int) -> dict[str, Any]:
    scr: dict[str, Any] = {
        "xb": pltpu.VMEM((ex.MB, gp.H), BF16),
        "ybuf": pltpu.VMEM((ex.MB, gp.NLI), F32),
        "mix": pltpu.VMEM((ex.MB, gp.Z), F32),
        "sm": pltpu.VMEM((1, 1, sm_width), F32),
        "recb": (
            pltpu.VMEM((gp.NRB, gp.CBV, gp.KD, gp.VD), BF16)
            if gp.GCH
            else pltpu.VMEM((ex.B * gp.vh, gp.KD, gp.VD), BF16)
        ),
        "sbuf": pltpu.VMEM((gp.vh, gp.KD, gp.VD), F32),
        "convb": pltpu.VMEM((ex.NS, gp.HR, gp.QKV) if ex.VER else (gp.CR, gp.QKV), F32),
        "st_sem": pltpu.SemaphoreType.DMA((4,)),
    }
    if ex.VER and not gp.GCH:
        scr["ub"] = pltpu.VMEM((ex.NS, round_up(ex.T, 8), gp.QKV), F32)
        scr["rin_sem"] = pltpu.SemaphoreType.DMA((ex.NS,))
    if gp.GCH:
        scr["rec_sem"] = pltpu.SemaphoreType.DMA((2, gp.NRB))
        if gp.GPR:
            scr["gpr_rec"] = pltpu.VMEM(
                (
                    gp.NRB,
                    gp.CB // ex.T,
                    gp.REC_PAGES,
                    *gp.pool_layout.kv_page_shape,
                ),
                FP8,
            )
            scr["gpr_conv"] = pltpu.VMEM(
                (ex.NS, gp.GCONV_ROWS, *gp.pool_layout.kv_page_shape[2:]), FP8
            )
            if not gp.conv_is_f32:
                scr["convb16"] = pltpu.VMEM((ex.NS, gp.HR, gp.QKV), BF16)
    return scr


class _FakeGDNStream:
    """Minimal stream stub providing `dmatmul` and `sp.lay` for GDN Pallas tests."""

    def __init__(self, w_in_ref: Any, lay: dict[str, tuple[int, int]]) -> None:
        self._w_in_ref = w_in_ref
        self.sp = SimpleNamespace(lay=lay)

    def dmatmul(self, kind: str, x_vmem: Any, n_cols: int) -> Any:
        assert kind == "lin_in"
        return x_vmem[...].astype(F32) @ self._w_in_ref[0, :, :n_cols].astype(F32)


def _run_gdn_pallas_case(
    *,
    T: int,
    B: int,
    chunked: bool,
    head_major: bool,
    gdn_bufs: int,
    gpr: bool = False,
    conv_dtype: Any = F32,
    interpret: bool = True,
) -> None:
    tiles = dataclasses.replace(
        TINY_TILES, bank=512 << 10, wide_rows=4 if chunked else 64
    )
    g = FakeGeometry(
        L=2 if gpr else 1,
        att_every=2 if gpr else 1,
        H=512,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        NH=2,
        NKV=1,
        HD=256,
    )
    pl_layout = None
    ex = ExecSpec(
        B=B,
        T=T,
        R=1,
        ctx=256,
        stack_e=True,
        tiles=tiles,
        interpret=interpret,
    )
    if gpr:
        pl_layout = PoolLayout(
            num_pools=1,
            slots_per_pool=1,
            kh=g.KH,
            vh=g.VH,
            rec_pages_per_slot=4,
            conv_pages_per_slot=1,
            conv_taps=3,
            conv_dtype=conv_dtype,
        )
        ex = dataclasses.replace(
            ex,
            cache=CacheExec(
                paged_kv=True,
                page_size=pl_layout.page_size,
                num_pages=pl_layout.min_num_pages(ex),
                pool_gather=True,
            ),
        )

    class _GDNSmallMixer(FakeMixerModule):
        input_norm = None

        def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
            del ex
            qkv = 2 * g.KH * g.KD + g.VH * g.VD
            return [
                SmallField("conv", 4 * qkv),
                SmallField("alog", 128),
                SmallField("dtb", 128),
                SmallField("gnorm", g.VD),
            ]

    ms = ModelSpec(
        "gdn_core",
        config=g,
        layer_type=LayerType("lt", slots=(_GDNSmallMixer(), FakeFFNModule())),
        cache_backend=(
            PoolBackend(
                layout=pl_layout,
                kernel_names=("gather_pool_states", "scatter_pool_states"),
            )
            if pl_layout is not None
            else None
        ),
    )
    p = plan(ms, ex)
    tt = GDNTunables(
        gdn_rows=8 if chunked else 0,
        gdn_bufs=gdn_bufs if chunked else 1,
        gdn_head_major=head_major,
    )
    gp = derive_params(
        g, ex, p, tt, "fake_gqa", gate="silu", in_scaled=False, out_scaled=False
    )
    assert gp.GCH is chunked and gp.head_major is head_major and gp.GPR is gpr

    cp = SimpleNamespace(plan=p, stream=SimpleNamespace(lay=p.smalls_layout), comm=None)
    scratch = _gdn_core_scratch(gp, ex, p.smalls_width)
    s_names = list(scratch.keys())
    conv_out_key = "conv_ckpt" if ex.VER else "conv"
    rec_out_key = "rec_ckpt" if ex.VER else "rec"
    conv_out_dt = BF16 if (gp.GPR and not gp.conv_is_f32) else F32
    conv_sh = (1, ex.NS, gp.HR, gp.QKV) if ex.VER else (1, gp.CR, gp.QKV)
    rec_out_sh = (1, ex.B * gp.vh, gp.KD, gp.VD)

    if gpr:

        def pallas_body(
            x_ref,
            w_in_ref,
            sm_ref,
            pool_ref,
            ckpt_ref,
            roff_ref,
            mix_out_ref,
            conv_out_ref,
            rec_out_ref,
            *scr_refs,
        ):
            scr = dict(zip(s_names, scr_refs))
            scr["xb"][...] = x_ref[...]
            scr["sm"][...] = sm_ref[...]
            scr["mix"][...] = jnp.zeros_like(scr["mix"][...])
            st_refs = {"ckpt_indices": ckpt_ref, "read_offsets": roff_ref}
            out_refs = {conv_out_key: conv_out_ref, rec_out_key: rec_out_ref}
            fake_stream = _FakeGDNStream(w_in_ref, p.smalls_layout)
            ctx = MiniCtx(
                g=g,
                ex=ex,
                core_plan=cp,
                pos_ref=None,
                W={},
                ST=st_refs,
                O=out_refs,
                scr=scr,
                me=jnp.int32(0),
                stream=fake_stream,
                comm=None,
                raw_refs=[pool_ref],
            )
            o = jnp.int32(0)
            slot = jnp.int32(0)
            ybuf = gdn_in(ctx, gp, o, slot)
            conv_step(ctx, gp, o, slot, ybuf)
            gdn_rec_chunked(ctx, gp, o, slot)
            pltpu.make_async_copy(
                conv_out_src(ctx, gp),
                out_refs[conv_out_key].at[o],
                scr["st_sem"].at[2],
            ).wait()
            rec_drain(ctx, gp, o)
            mix_out_ref[...] = scr["mix"][...]

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ]
    else:

        def pallas_body(
            x_ref,
            w_in_ref,
            sm_ref,
            conv_in_ref,
            rec_in_ref,
            mix_out_ref,
            conv_out_ref,
            rec_out_ref,
            *scr_refs,
        ):
            scr = dict(zip(s_names, scr_refs))
            scr["xb"][...] = x_ref[...]
            scr["sm"][...] = sm_ref[...]
            scr["mix"][...] = jnp.zeros_like(scr["mix"][...])
            st_refs = {"conv": conv_in_ref, "rec": rec_in_ref}
            out_refs = {conv_out_key: conv_out_ref, rec_out_key: rec_out_ref}
            fake_stream = _FakeGDNStream(w_in_ref, p.smalls_layout)
            ctx = MiniCtx(
                g=g,
                ex=ex,
                core_plan=cp,
                pos_ref=None,
                W={},
                ST=st_refs,
                O=out_refs,
                scr=scr,
                me=jnp.int32(0),
                stream=fake_stream,
                comm=None,
            )
            o = jnp.int32(0)
            slot = jnp.int32(0)
            ybuf = gdn_in(ctx, gp, o, slot)
            u = conv_step(ctx, gp, o, slot, ybuf)
            if gp.GCH:
                gdn_rec_chunked(ctx, gp, o, slot)
            else:
                for d in rec_in(ctx, gp, o):
                    d.wait()
                gdn_rec(ctx, gp, u, ybuf, slot)
                pltpu.make_async_copy(
                    scr["recb"], out_refs[rec_out_key].at[o], scr["st_sem"].at[3]
                ).start()
            pltpu.make_async_copy(
                conv_out_src(ctx, gp),
                out_refs[conv_out_key].at[o],
                scr["st_sem"].at[2],
            ).wait()
            if gp.GCH:
                rec_drain(ctx, gp, o)
            else:
                pltpu.make_async_copy(
                    scr["recb"], out_refs[rec_out_key].at[o], scr["st_sem"].at[3]
                ).wait()
            mix_out_ref[...] = scr["mix"][...]

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]

    call = pl.pallas_call(
        pallas_body,
        out_shape=[
            jax.ShapeDtypeStruct((ex.MB, gp.Z), F32),
            jax.ShapeDtypeStruct(conv_sh, conv_out_dt),
            jax.ShapeDtypeStruct(rec_out_sh, BF16),
        ],
        in_specs=in_specs,
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        scratch_shapes=[scratch[k] for k in s_names],
        interpret=pltpu.InterpretParams() if interpret else False,
    )

    keys = jax.random.split(
        jax.random.PRNGKey(70 + T + B + int(head_major) + int(gpr)), 7
    )
    x_in = jax.random.normal(keys[0], (ex.MB, g.H), dtype=BF16) * 0.6
    w_in = jax.random.normal(keys[1], (1, g.H, gp.NLI), dtype=BF16) * 0.25
    w_out_eye = jnp.broadcast_to(jnp.eye(gp.Z, g.H, dtype=BF16), (1, gp.Z, g.H))
    sm = jnp.zeros((1, 1, p.smalls_width), dtype=F32)
    for fname, base in (("conv", 0.35), ("alog", -1.0), ("dtb", 0.2), ("gnorm", 1.2)):
        o_sm, w_sm = p.smalls_layout[fname]
        sm = sm.at[:, :, o_sm : o_sm + w_sm].set(base)
    sm = sm + jax.random.normal(keys[2], sm.shape, dtype=F32) * 0.05

    st_raw = gdn_init_logical_states(keys[3], g, ex, n_layers=1, conv_taps=3)
    st_log = {
        "conv": st_raw["conv"] * 8.0,
        "rec": (st_raw["rec"].astype(F32) * 8.0).astype(BF16),
    }

    if gpr:
        assert pl_layout is not None
        pt, ckpt, roff, _ = allocate_test_layout(pl_layout, ex)
        # Populate pool from st_log, then mark the last sequence inactive
        st_for_pool = {
            **st_log,
            "kc": jnp.zeros(
                (1, 1, ex.NS, ex.ctx, pl_layout.kvh, pl_layout.hd), dtype=FP8
            ),
            "vc": jnp.zeros(
                (1, 1, ex.NS, ex.ctx, pl_layout.kvh, pl_layout.hd), dtype=FP8
            ),
        }
        pools = states_to_pool(
            pl_layout, ex, st_for_pool, pt, ckpt[..., :1], jnp.maximum(roff, 0)
        )
        ckpt_masked = ckpt.at[:, -1, :].set(0)
        mix_out, conv_out, rec_out = jax.jit(call)(
            x_in, w_in, sm, pools[0], ckpt_masked, roff
        )
        conv_ref_in = (
            st_log["conv"].astype(BF16).astype(F32)
            if not gp.conv_is_f32
            else st_log["conv"]
        )
        st_ref_in = {
            "conv": conv_ref_in.at[:, :, -1].set(0.0),
            "rec": st_log["rec"].at[:, :, -1].set(0.0),
        }
    else:
        st_packed = gdn_pack_logical_states(st_log, g, ex, conv_taps=3)
        mix_out, conv_out, rec_out = jax.jit(call)(
            x_in, w_in, sm, st_packed["conv"], st_packed["rec"]
        )
        st_ref_in = st_log

    ref_params = {"lin_in": w_in, "lin_out": w_out_eye}
    for fname in ("conv", "alog", "dtb", "gnorm"):
        o_sm, w_sm = p.smalls_layout[fname]
        ref_params[fname] = sm[:, 0, o_sm : o_sm + w_sm]
    with jax.default_device(jax.devices("cpu")[0]):
        ref_out, ref_st = gdn_reference(ref_params, x_in[:B], st_ref_in, g=g, ex=ex)

    assert float(jnp.max(jnp.abs(ref_out))) >= 0.25
    assert float(jnp.max(jnp.abs(ref_st[rec_out_key][0].astype(F32)))) >= 0.25

    ref_conv = ref_st[conv_out_key][0]
    if conv_out_dt == BF16:
        ref_conv = ref_conv.astype(BF16)
    # `rec_out`, `mix_out`, and the newly projected `conv_out` rows
    # (`round_bf16(xb @ w_in)`) are quantized to BF16 (unit roundoff 2**-7 =
    # 7.8125e-3), so MXU vs CPU f32 matmuls or JIT-fused vs eager F32 reductions
    # before `round_bf16` can differ by 1-2 BF16 ULPs. Shifted (`T == 1`) or
    # gathered (`T > 1`) history rows are pure copies and must match bit-for-bit
    # (`rtol=0, atol=0`).
    bf16_2ulp = 2.0 * (2.0**-7)
    conv_new_rtol = 1e-5 if interpret else bf16_2ulp
    conv_new_atol = 1e-5 if interpret else 1e-3
    if T == 1:
        np.testing.assert_allclose(
            conv_out[0, : 2 * B].astype(F32),
            ref_conv[: 2 * B].astype(F32),
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            conv_out[0, 2 * B :].astype(F32),
            ref_conv[2 * B :].astype(F32),
            rtol=conv_new_rtol,
            atol=conv_new_atol,
        )
    else:
        np.testing.assert_allclose(
            conv_out[0, :, :3].astype(F32),
            ref_conv[:, :3].astype(F32),
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            conv_out[0, :, 3:].astype(F32),
            ref_conv[:, 3:].astype(F32),
            rtol=conv_new_rtol,
            atol=conv_new_atol,
        )
    np.testing.assert_allclose(
        mix_out[:B].astype(F32),
        ref_out.astype(F32),
        rtol=bf16_2ulp,
        atol=1e-2,
    )
    np.testing.assert_allclose(
        rec_out[0].astype(F32),
        ref_st[rec_out_key][0].astype(F32),
        rtol=bf16_2ulp,
        atol=1e-2,
    )
    # Must-fail control: a dropped delta update (keeping initial rec state) must fail
    if T == 1:
        init_rec = st_ref_in["rec"][0, 0].reshape(B * gp.vh, gp.KD, gp.VD).astype(F32)
    else:
        init_rec = (
            jnp.repeat(st_ref_in["rec"][0, 0], T, axis=0)
            .reshape(B * gp.vh, gp.KD, gp.VD)
            .astype(F32)
        )
    assert not jnp.allclose(rec_out[0].astype(F32), init_rec, rtol=bf16_2ulp, atol=1e-2)


@pytest.mark.parametrize(
    ("T", "B", "chunked", "head_major", "gdn_bufs"),
    [
        (1, 4, False, False, 1),
        (1, 16, True, False, 2),
        (4, 4, False, False, 1),
        (4, 8, True, False, 2),
        (4, 8, True, True, 2),
        (4, 20, True, False, 3),
        (4, 24, True, False, 2),
    ],
    ids=[
        "decode_t1_rec",
        "decode_t1_wide_chunked",
        "verify_t4_rec",
        "verify_t4_chunked",
        "verify_t4_chunked_head_major",
        "verify_t4_b20_multi_chunk_tail",
        "verify_t4_b24_ring_nrb2",
    ],
)
def test_gdn_pallas_core_conv_and_recurrence_match_reference(
    T: int, B: int, chunked: bool, head_major: bool, gdn_bufs: int
):
    """Execute `gdn_in`, `conv_step`, `gdn_rec`, and `gdn_rec_chunked` in Pallas."""
    _run_gdn_pallas_case(
        T=T,
        B=B,
        chunked=chunked,
        head_major=head_major,
        gdn_bufs=gdn_bufs,
        gpr=False,
        interpret=True,
    )


@pytest.mark.parametrize("conv_dtype", [BF16, F32], ids=["bf16_conv", "f32_conv"])
def test_gdn_pallas_core_gpr_gather_matches_reference(conv_dtype: Any):
    """Verify in-kernel gather (`GPR=True`) with bf16/f32 conv and an inactive seq."""
    _run_gdn_pallas_case(
        T=4,
        B=8,
        chunked=True,
        head_major=False,
        gdn_bufs=2,
        gpr=True,
        conv_dtype=conv_dtype,
        interpret=True,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("T", "B", "chunked", "head_major", "gdn_bufs", "gpr", "conv_dtype"),
    [
        (1, 4, False, False, 1, False, F32),
        (4, 4, False, False, 1, False, F32),
        (1, 16, True, False, 2, False, F32),
        (4, 8, True, True, 2, False, F32),
        (4, 16, True, False, 2, False, F32),
        (4, 16, True, False, 2, True, BF16),
        (4, 16, True, False, 2, True, F32),
    ],
    ids=[
        "tpu_decode_t1_rec",
        "tpu_verify_t4_rec",
        "tpu_decode_t1_wide_chunked",
        "tpu_verify_t4_chunked_head_major",
        "tpu_chunked_hbm",
        "tpu_gpr_bf16_conv",
        "tpu_gpr_f32_conv",
    ],
)
def test_gdn_pallas_core_on_tpu(
    T: int,
    B: int,
    chunked: bool,
    head_major: bool,
    gdn_bufs: int,
    gpr: bool,
    conv_dtype: Any,
):
    """Run GDN Pallas conv and recurrence paths on TPU (`interpret=False`)."""
    try:
        tpu_devs = jax.devices("tpu")
    except RuntimeError:
        tpu_devs = []
    if not tpu_devs:
        pytest.skip("TPU device required for interpret=False Pallas execution")
    with jax.default_device(tpu_devs[0]):
        _run_gdn_pallas_case(
            T=T,
            B=B,
            chunked=chunked,
            head_major=head_major,
            gdn_bufs=gdn_bufs,
            gpr=gpr,
            conv_dtype=conv_dtype,
            interpret=False,
        )
