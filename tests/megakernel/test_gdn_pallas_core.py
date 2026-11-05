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
"""Standalone unit tests for slice L-7b (`modules/gdn/{states,conv,recurrence}`)."""

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
from vllm_torchtpu.megakernel.ops import BF16, F32, round_up
from vllm_torchtpu.megakernel.planner import plan
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
    ckpts = {"conv_ckpt": c_ckpt, "rec_ckpt": r_ckpt}
    acc = jnp.array([2, 4], dtype=jnp.int32)
    accepted = gdn_accept(packed_v, ckpts, acc, T=4, vh=2, taps=3)
    np.testing.assert_array_equal(accepted["conv"][:, 0, :3], c_ckpt[:, 0, 2:5])
    np.testing.assert_array_equal(accepted["conv"][:, 1, :3], c_ckpt[:, 1, 4:7])

    # Unpadded 5-D logical layout infers `taps=3` when `taps=None`
    c_ckpt_5d = c_ckpt[:, None, :, :7, :]
    r_ckpt_5d = r_ckpt[:, None]
    accepted_5d = gdn_accept(
        st_v, {"conv_ckpt": c_ckpt_5d, "rec_ckpt": r_ckpt_5d}, acc, T=4, vh=2
    )
    np.testing.assert_array_equal(accepted_5d["conv"][:, 0, 0, :3], c_ckpt[:, 0, 2:5])

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
    return scr


class _FakeGDNStream:
    """Minimal stream stub providing `dmatmul` and `sp.lay` for GDN Pallas tests."""

    def __init__(self, w_in_ref: Any, lay: dict[str, tuple[int, int]]) -> None:
        self._w_in_ref = w_in_ref
        self.sp = SimpleNamespace(lay=lay)

    def dmatmul(self, kind: str, x_vmem: Any, n_cols: int) -> Any:
        assert kind == "lin_in"
        return x_vmem[...].astype(F32) @ self._w_in_ref[0, :, :n_cols].astype(F32)


@pytest.mark.parametrize(
    ("T", "B", "chunked", "head_major"),
    [
        (1, 4, False, False),
        (4, 4, False, False),
        (4, 8, True, False),
        (4, 8, True, True),
    ],
    ids=[
        "decode_t1_rec",
        "verify_t4_rec",
        "verify_t4_chunked",
        "verify_t4_chunked_head_major",
    ],
)
def test_gdn_pallas_core_conv_and_recurrence_match_reference(
    T: int, B: int, chunked: bool, head_major: bool
):
    """Execute `gdn_in`, `conv_step`, `gdn_rec`, and `gdn_rec_chunked` in Pallas."""
    tiles = dataclasses.replace(
        TINY_TILES, bank=512 << 10, wide_rows=4 if chunked else 16
    )
    ex = ExecSpec(B=B, T=T, R=1, ctx=256, stack_e=True, tiles=tiles, interpret=True)
    g = FakeGeometry(
        L=1, att_every=1, H=256, KH=2, VH=4, KD=128, VD=128, NH=2, NKV=1, HD=128
    )

    class _GDNSmallMixer(FakeMixerModule):
        input_norm = None

        def smalls(self, geom: Any, ex_spec: ExecSpec) -> list[SmallField]:
            del geom, ex_spec
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
    )
    p = plan(ms, ex)
    tt = GDNTunables(
        gdn_rows=8 if chunked else 0,
        gdn_bufs=2 if chunked else 1,
        gdn_head_major=head_major,
    )
    gp = derive_params(
        g, ex, p, tt, "fake_gqa", gate="silu", in_scaled=False, out_scaled=False
    )
    assert gp.GCH is chunked and gp.head_major is head_major

    cp = SimpleNamespace(plan=p, stream=SimpleNamespace(lay=p.smalls_layout), comm=None)
    scratch = _gdn_core_scratch(gp, ex, p.smalls_width)
    s_names = list(scratch.keys())
    conv_out_key = "conv_ckpt" if ex.VER else "conv"
    rec_out_key = "rec_ckpt" if ex.VER else "rec"

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
            conv_out_src(ctx, gp), out_refs[conv_out_key].at[o], scr["st_sem"].at[2]
        ).wait()
        if gp.GCH:
            rec_drain(ctx, gp, o)
        else:
            pltpu.make_async_copy(
                scr["recb"], out_refs[rec_out_key].at[o], scr["st_sem"].at[3]
            ).wait()
        mix_out_ref[...] = scr["mix"][...]

    conv_sh = (1, ex.NS, gp.HR, gp.QKV) if ex.VER else (1, gp.CR, gp.QKV)
    rec_out_sh = (1, ex.B * gp.vh, gp.KD, gp.VD)
    call = pl.pallas_call(
        pallas_body,
        out_shape=[
            jax.ShapeDtypeStruct((ex.MB, gp.Z), F32),
            jax.ShapeDtypeStruct(conv_sh, F32),
            jax.ShapeDtypeStruct(rec_out_sh, BF16),
        ],
        in_specs=[
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

    keys = jax.random.split(jax.random.PRNGKey(70 + T + B + int(head_major)), 7)
    x_in = jax.random.normal(keys[0], (ex.MB, g.H), dtype=BF16) * 0.2
    w_in = jax.random.normal(keys[1], (1, g.H, gp.NLI), dtype=BF16) * 0.1
    w_out_eye = jnp.broadcast_to(jnp.eye(gp.Z, g.H, dtype=BF16), (1, gp.Z, g.H))
    sm = jnp.zeros((1, 1, p.smalls_width), dtype=F32)
    for fname, base in (("conv", 0.1), ("alog", -1.0), ("dtb", 0.1), ("gnorm", 1.0)):
        o_sm, w_sm = p.smalls_layout[fname]
        sm = sm.at[:, :, o_sm : o_sm + w_sm].set(base)
    sm = sm + jax.random.normal(keys[2], sm.shape, dtype=F32) * 0.01

    st_log = gdn_init_logical_states(keys[3], g, ex, n_layers=1, conv_taps=3)
    st_packed = gdn_pack_logical_states(st_log, g, ex, conv_taps=3)
    mix_out, conv_out, rec_out = jax.jit(call)(
        x_in, w_in, sm, st_packed["conv"], st_packed["rec"]
    )
    assert conv_out.shape == conv_sh

    ref_params = {"lin_in": w_in, "lin_out": w_out_eye}
    for fname in ("conv", "alog", "dtb", "gnorm"):
        o_sm, w_sm = p.smalls_layout[fname]
        ref_params[fname] = sm[:, 0, o_sm : o_sm + w_sm]
    _, ref_st = gdn_reference(ref_params, x_in[:B], st_log, g=g, ex=ex)
    np.testing.assert_allclose(
        rec_out[0].astype(F32),
        ref_st[rec_out_key][0].astype(F32),
        rtol=2e-2,
        atol=2e-2,
    )
    assert jnp.all(jnp.isfinite(mix_out[:B]))
