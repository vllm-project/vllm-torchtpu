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
"""Unit tests for Qwen3.5 megakernel `GDNBlock`, `GDNPoolDMA`, `GprRecIn`, and GDN pool I/O."""

from __future__ import annotations

import dataclasses

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import torch

from vllm_torchtpu.kernels.qwen35_megakernel import (
    GDNBlock,
    GDNPoolDMA,
    WeightRings,
    megakernel as K,
    pool_io,
)
from vllm_torchtpu.kernels.qwen35_megakernel.megakernel import BF16, F32, rnd
from vllm_torchtpu.kernels.qwen35_megakernel.vllm_adapter import pack_weights

from qwen35_megakernel_test_utils import (
    jnp_leaves,
    make_test_ctx_and_inputs,
    make_tiny_backbone,
    ref_dmatmul,
    rms_ref,
    setup_8_cpu_devices,
)


def test_standalone_gdn_block_interpret():
    """Test GDNBlock standalone in pl.pallas_call(interpret=True) vs pure-JAX reference."""
    c, _, ctx, w_sh, st_sh, lay, w_vals, st_vals, pos, hid = (
        make_test_ctx_and_inputs()
    )
    mesh1 = Mesh(np.array(setup_8_cpu_devices()[:1]), ("tp",))

    def gdn_kernel(*refs):
        ctx.bind_refs(refs)
        rings = WeightRings(ctx)
        gdn_dma = GDNPoolDMA(ctx)
        gdn_blk = GDNBlock(ctx, rings, gdn_dma)
        scr = ctx.scr
        for i in range(ctx.t_gdn):
            rings.fetch(jnp.int32(i))
        cp_sm = pltpu.make_async_copy(
            ctx.W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        )
        cp_sm.start()
        cp_sm.wait()
        scr["res"][...] = jnp.where(ctx.live, ctx.hid_ref[: ctx.MB].astype(F32), 0.0)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        scr["xb"][...] = ctx.rms_zc(scr["res"][...], ctx.sm(0, "in_norm")).astype(BF16)
        yo = gdn_blk(jnp.int32(0), jnp.int32(0), jnp.int32(0))
        pltpu.make_async_copy(
            gdn_dma.conv_out_src(), ctx.O["conv"].at[0], scr["st_sem"].at[2]
        ).wait()
        pltpu.make_async_copy(
            scr["recb"], ctx.O["rec"].at[0], scr["st_sem"].at[3]
        ).wait()
        ctx.out_ref[: ctx.MB] = yo.astype(BF16)

    call = pl.pallas_call(
        gdn_kernel,
        out_shape=ctx.out_shape,
        in_specs=ctx.in_specs,
        out_specs=ctx.out_specs,
        scratch_shapes=[ctx.scratch[k] for k in ctx.s_names],
        interpret=pltpu.InterpretParams(),
    )
    args = (pos, hid, *[w_vals[k] for k in w_sh], *[st_vals[k] for k in st_sh])
    hid_gdn, conv_gdn, rec_gdn, _, _ = jax.shard_map(
        lambda *a: call(*a),
        mesh=mesh1,
        in_specs=tuple(P() for _ in args),
        out_specs=tuple(P() for _ in ctx.out_shape),
        check_vma=False,
    )(*args)

    def sm_val(l_idx, name, width=None):
        o, n = lay[name]
        w = n if width is None else width
        return w_vals["smalls"][l_idx, :, o : o + w]

    xb_ref = rms_ref(hid[: c.B].astype(F32), sm_val(0, "in_norm"), c.eps).astype(BF16)
    y_in = rnd(
        ref_dmatmul(xb_ref, w_vals["lin_in"][0], ctx.MB) * sm_val(0, "in_s", c.NLI)
    )
    conv0 = st_vals["conv"][0]
    ow, _ = lay["conv"]
    taps = [
        w_vals["smalls"][0, :, ow + t * c.QKV : ow + (t + 1) * c.QKV] for t in range(4)
    ]
    x_conv = y_in[:, : c.QKV]
    s0, s1, s2 = conv0[0 : c.B], conv0[c.B : 2 * c.B], conv0[2 * c.B : 3 * c.B]
    u_ref = rnd(jax.nn.silu(s0 * taps[0] + s1 * taps[1] + s2 * taps[2] + x_conv * taps[3]))
    ref_conv = jnp.concatenate([s1, s2, x_conv], axis=0)
    alog = sm_val(0, "alog")[0, : c.vh]
    dtb = sm_val(0, "dtb")[0, : c.vh]
    gnw = sm_val(0, "gnorm")[0, :].astype(BF16)
    expand = c.vh // c.kh
    rec_state = st_vals["rec"][0]
    mix_list, rec_list = [], []
    for b in range(c.B):
        q = u_ref[b, : c.kh * c.KD].reshape(c.kh, c.KD)
        k = u_ref[b, c.kh * c.KD : 2 * c.kh * c.KD].reshape(c.kh, c.KD)
        v = u_ref[b, 2 * c.kh * c.KD :].reshape(c.vh, c.VD)
        q = q * lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6) * (c.KD**-0.5)
        k = k * lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        qc = jnp.repeat(q[:, :, None], expand, axis=0)
        kc = jnp.repeat(k[:, :, None], expand, axis=0)
        beta = rnd(jax.nn.sigmoid(y_in[b, c.BA : c.BA + c.vh]))
        a_val = y_in[b, c.BA + 64 : c.BA + 64 + c.vh]
        g = -jnp.exp(alog) * jax.nn.softplus(a_val + dtb)
        sb = rec_state[b * c.vh : (b + 1) * c.vh].astype(F32) * jnp.exp(g)[:, None, None]
        pred = jnp.sum(sb * kc, axis=1)
        delta = (v - pred) * beta[:, None]
        sb = sb + kc * delta[:, None, :]
        out = rnd(jnp.sum(sb * qc, axis=1))
        rec_list.append(sb.astype(BF16))
        val = out * lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + c.eps)
        val = (gnw[None, :] * val.astype(BF16)).astype(F32)
        zz = y_in[b, c.QKV : c.QKV + c.Z].reshape(c.vh, c.VD)
        mix_list.append(rnd(val * jax.nn.silu(zz)).reshape(c.Z))
    ref_rec = jnp.concatenate(rec_list, axis=0)
    ref_mix = jnp.stack(mix_list, axis=0).astype(BF16)
    ref_yo = (
        ref_dmatmul(ref_mix, w_vals["lin_out"][0], ctx.MB) * sm_val(0, "out_s")
    ).astype(BF16)

    assert float(jnp.max(jnp.abs(conv_gdn[0, : 3 * c.B] - ref_conv))) == 0.0
    assert float(jnp.max(jnp.abs(rec_gdn[0].astype(F32) - ref_rec.astype(F32)))) == 0.0
    assert (
        float(jnp.max(jnp.abs(hid_gdn[: c.B].astype(F32) - ref_yo.astype(F32)))) <= 4e-3
    )


def test_scatter_pool_states_direct_conv_aligned_conv_rows():
    """Regression test for `scatter_pool_states` when `qkv_128 = 16` (`QKV = 2048`):
    `conv_rows = 48 == conv_rows_pad` while `(4 * qkv_128) % 16 == 0` (`64 != 48`)."""
    c = K.MKConfig(
        R=1,
        B=2,
        T=2,
        L=2,
        att_every=2,
        H=512,
        NH=2,
        NKV=1,
        HD=256,
        KH=4,
        VH=8,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        ctx=256,
        paged_kv=True,
        num_pages=32,
        page_size=128,
        max_pages_per_seq=2,
        ckpt_k=2,
    )
    assert c.QKV == 2048  # qkv_128 = 16 -> conv_rows = 48 == conv_rows_pad
    rng = np.random.default_rng(7)
    raw = (jnp.zeros((c.num_pages, 2, 64, 4, c.page_size), dtype=K.FP8),)
    pos = jnp.array([4, 5], dtype=jnp.int32)
    pt = jnp.array([[1, 0]], dtype=jnp.int32)
    ckpt = jnp.arange(4, 4 + c.NS * c.T, dtype=jnp.int32).reshape(1, c.NS, c.T)
    knew = jnp.asarray(rng.standard_normal((1, c.MR, c.HD)), dtype=K.FP8)
    vnew = jnp.asarray(rng.standard_normal((1, c.MR, c.HD)), dtype=K.FP8)
    conv_out = jnp.asarray(
        rng.standard_normal((1, c.NS, c.HR, c.QKV)), dtype=jnp.bfloat16
    )
    rec_out = jnp.asarray(
        rng.standard_normal((1, c.B * c.vh, c.KD, c.VD)), dtype=jnp.bfloat16
    )

    raw_out = pool_io.scatter_pool_states(
        c, raw, pos, pt, knew, vnew, conv_out, rec_out, ckpt, interpret=True
    )
    for t in range(c.T):
        roff = jnp.full((1, c.NS), t, dtype=jnp.int32)
        g_conv, g_rec = pool_io.gather_gdn_states(
            c, raw_out, ckpt, roff, interpret=True
        )
        np.testing.assert_array_equal(
            np.asarray(g_conv[:, :, :3, :], dtype=np.float32),
            np.asarray(conv_out[:, :, 1 + t : 4 + t, :], dtype=np.float32),
        )
        for s in range(c.NS):
            b = s * c.T + t
            np.testing.assert_array_equal(
                np.asarray(g_rec[:, s * c.vh : (s + 1) * c.vh], dtype=np.float32),
                np.asarray(rec_out[:, b * c.vh : (b + 1) * c.vh], dtype=np.float32),
            )


def test_gpr_three_stage_matches_fused_at_tp16_checkpoint_geometry():
    """3-stage pool I/O of a > 16-row verify bucket (GatedDeltaNet inputs read straight from the
    checkpoint pages (pool_gather / GPR), partial conv-page scatter) against fused pool I/O at
    the TP=16 checkpoint geometry (vh=4: 2 rec pages + 1 conv page per slot)."""
    c = dataclasses.replace(
        K.TINY_EP,
        R=1,
        B=20,
        T=4,
        L=2,
        att_every=2,
        H=2048,
        NH=1,
        NKV=1,
        KH=1,
        VH=4,
        I=128,
        IS=128,
        E=128,
        TOPK=2,
        e_slots=128,
        paged_kv=True,
        page_size=128,
        max_pages_per_seq=8,
        ctx=1024,
    )
    assert (c.rec_pages, c.gdn_pages) == (2, 3)
    pt, ckpt, roff, min_pages = pool_io.default_pool_layout(c)
    c = dataclasses.replace(c, num_pages=min_pages)
    tiles = dataclasses.replace(K.default_tiles(c), kv_blk=512, kv_sub=256)
    assert tiles.gdn_rows > 0
    torch.manual_seed(0)
    w = jnp_leaves(pack_weights(make_tiny_backbone(c, e_slots=c.EW), c, 0, tiles))
    st_shapes = K.state_shapes(dataclasses.replace(c, paged_kv=False))
    St = {
        k: (
            jax.random.normal(jax.random.PRNGKey(20 + i), (c.R, *s), jnp.float32)
            * (0.1 if k in ("conv", "rec") else 0.5)
        ).astype(dt)
        for i, (k, (s, dt)) in enumerate(st_shapes.items())
    }
    raw = pool_io.states_to_pool(c, St, pt, ckpt, roff)
    pos = jnp.asarray(
        [10 + 37 * s + t for s in range(c.NS) for t in range(c.T)], jnp.int32
    )
    hid = jax.random.normal(jax.random.PRNGKey(7), (c.M, c.H), jnp.float32)
    hid = jnp.where(jnp.arange(c.M)[:, None] < c.B, hid, 0.0).astype(jnp.bfloat16)
    mesh = Mesh(np.array(setup_8_cpu_devices()[:1]), ("tp",))

    outs = {}
    for fused in (True, False):
        fn, _, _, _ = pool_io.make_flat_pooled_step(
            mesh,
            c,
            tiles,
            fused_pool_io=fused,
            interpret=True,
            skip=frozenset({"experts"}),
        )
        out = fn(
            pos, hid, *w, *[jnp.array(r, copy=True) for r in raw], pt, ckpt, roff
        )
        outs[fused] = (
            np.asarray(out[0][: c.B], np.float32),
            np.asarray(jax.lax.bitcast_convert_type(out[1], jnp.uint8)),
        )
    (h_f, p_f), (h_3, p_3) = outs[True], outs[False]
    assert np.isfinite(h_f).all()
    np.testing.assert_allclose(h_3, h_f, rtol=1e-2, atol=1e-2)

    kv_pages = c.NS * (c.ctx // c.page_size)
    assert np.array_equal(p_3[:kv_pages], p_f[:kv_pages])
    cp_rows = ((3 * c.QKV // 128 + 15) // 16 * 16) // 2

    def rec_f32(pages):
        u = np.ascontiguousarray(np.moveaxis(pages, -2, -1)).view(np.uint32)[..., 0]
        return np.stack([(u & 0xFFFF) << 16, (u >> 16) << 16]).view(np.float32)

    for slot in np.unique(np.asarray(ckpt)):
        p0 = int(slot) * c.gdn_pages
        np.testing.assert_allclose(
            rec_f32(p_3[p0 : p0 + c.rec_pages]),
            rec_f32(p_f[p0 : p0 + c.rec_pages]),
            rtol=1e-2,
            atol=1e-3,
        )
        conv = p0 + c.rec_pages
        assert np.array_equal(p_3[conv, 0, :cp_rows], p_f[conv, 0, :cp_rows]), int(slot)
