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
"""M2 verification for `GDNBlock` and full hybrid `models.qwen35` (`att_every=4`)."""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import pathlib

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax.experimental import topologies
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import kernel, planner
from vllm_torchtpu.megakernel.blocks.gdn import GDNBlock
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES, baseline_tiles
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8, I32, rmsnorm, rnd, rup
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec


def _can_compile_tpu7x() -> bool:
    try:
        topologies.get_topology_desc(platform="tpu", topology_name="tpu7x", **{"1d": 8})
        return True
    except Exception:
        return False


def test_m2_recorded_full_ep_hybrid_build_invariants():
    """Verify full hybrid `FULL_EP` (`L=60, att_every=4`) matches frozen constants."""
    g = qwen35.FULL_EP
    assert g.L == 60 and g.att_every == 4 and g.NGDN == 45 and g.NATT == 15

    ex_4k = ExecSpec(B=4, T=1, R=8, ctx=4096, tiles=baseline_tiles(4, R=8))
    _, info_4k = qwen35.build(g, ex_4k)
    p = info_4k["plan"]

    assert p.tk == {
        "lin_in": 512,
        "lin_out": 512,
        "att_in": 512,
        "att_out": 512,
        "router": 2048,
        "sh13": 4096,
        "sh2": 128,
        "e13": 512,
        "e2": 256,
    }
    assert p.nt == {
        "lin_in": 8,
        "lin_out": 2,
        "att_in": 8,
        "att_out": 2,
        "router": 2,
        "sh13": 1,
        "sh2": 1,
        "e13": 8,
        "e2": 4,
    }
    assert p.w_names == (
        "lin_in",
        "lin_out",
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
    assert p.s_names == ("conv", "rec", "kc", "vc")
    assert p.o_names == ("hidden", "conv", "rec", "knew", "vnew")
    assert info_4k["T_TOTAL"] == 840
    # M4b: 49,760,256; +2,048 B of always-allocated M5 GQA scratch (`rcs`;
    # `rope` is allocated only under `gqa_qv`)
    assert info_4k["vmem_scratch_bytes"] == 49_762_304
    assert info_4k["vmem_scratch_bytes"] <= ex_4k.resolved_tiles.vmem_mib * (1 << 20)
    assert "recb" in info_4k["vmem_shapes"]
    assert "sbuf" in info_4k["vmem_shapes"]
    assert "convb" in info_4k["vmem_shapes"]

    ex_ver = ExecSpec(B=16, T=4, R=8, ctx=4096, tiles=baseline_tiles(16, R=8))
    _, info_ver = qwen35.build(g, ex_ver)
    assert info_ver["plan"].o_names == (
        "hidden",
        "conv_ckpt",
        "rec_ckpt",
        "knew",
        "vnew",
    )

    ex_10k = ExecSpec(B=4, T=1, R=8, ctx=10240, tiles=baseline_tiles(4, R=8))
    _, info_10k = qwen35.build(g, ex_10k)
    assert info_10k["vmem_scratch_bytes"] == 49_762_304


def test_m2_gdn_block_invariants_and_layout():
    """Verify `GDNBlock.plan_params` enforces invariants and exposes `lin_in_layout`."""
    g = qwen35.FULL_EP
    ex = ExecSpec(B=4, T=1, R=8, ctx=4096)
    lay = GDNBlock.lin_in_layout(g, ex)
    assert (lay.kh, lay.vh, lay.QKV, lay.Z, lay.BA, lay.A_OFF, lay.NLI) == (
        2,
        8,
        1536,
        1024,
        2560,
        2624,
        2688,
    )
    p = planner.plan(qwen35.qwen35_spec(g), ex)
    assert p.block_params["gdn"].l2_eps == 1e-6

    # vh > 64 violates 64-lane b|a packing in ybuf[:, BA:BA+128]
    g_vh_over_64 = dataclasses.replace(g, KH=64, VH=128)
    ex_r1 = ExecSpec(B=4, T=1, R=1, ctx=256)
    with pytest.raises(AssertionError):
        planner.plan(qwen35.qwen35_spec(g_vh_over_64), ex_r1)

    # VH % KH != 0 violates grouped-query GDN expansion
    g_bad_ratio = dataclasses.replace(g, KH=24, VH=32)
    with pytest.raises(AssertionError):
        planner.plan(qwen35.qwen35_spec(g_bad_ratio), ex)

    # KD not a multiple of 128 violates TPU 8x128 tile alignment
    g_bad_kd = dataclasses.replace(g, KD=64)
    with pytest.raises(AssertionError):
        planner.plan(qwen35.qwen35_spec(g_bad_kd), ex)


@pytest.mark.parametrize(
    "B,T,gdn_rows",
    [
        (2, 1, 0),
        (4, 2, 0),
        pytest.param(20, 1, 8, marks=pytest.mark.slow),
        pytest.param(20, 2, 8, marks=pytest.mark.slow),
    ],
)
def test_m2_gdn_block_standalone_interpret(B: int, T: int, gdn_rows: int):
    """Test standalone `GDNBlock` across `T=1`, `T=2` verify, and `B=20` `GCH=True`."""
    devs = jax.devices("cpu")
    mesh = Mesh(devs[:1], ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    rep = NamedSharding(mesh, P())

    g = qwen35.Qwen35Geometry(
        L=2,
        att_every=2,
        H=256,
        NH=2,
        NKV=1,
        HD=128,
        ROT=32,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
    )
    spec = ModelSpec(
        name="gdn_standalone",
        config=g,
        layer_type=LayerType(name="gdn_layer", slots=(GDNBlock(),)),
    )
    tiles = dataclasses.replace(
        TINY_TILES, bank=256 << 10, inplace=not T > 1
    ).with_blocks({"gdn": {"gdn_rows": gdn_rows, "gdn_bufs": 2}})
    ex = ExecSpec(B=B, T=T, R=1, ctx=128, tiles=tiles, interpret=True)
    p = planner.plan(spec, ex)
    gp = p.block_params["gdn"]
    assert (B > 16 and gdn_rows > 0) == gp.GCH

    NS, M = ex.NS, ex.M
    L, H, KD, VD = g.L, g.H, g.KD, g.VD
    lin_lay = GDNBlock.lin_in_layout(g, ex)
    kh, vh, QKV, Z, BA, NLI = (
        lin_lay.kh,
        lin_lay.vh,
        lin_lay.QKV,
        lin_lay.Z,
        lin_lay.BA,
        lin_lay.NLI,
    )
    CR, HR = rup(3 * B, 8), rup(3 + T, 8)

    keys = jax.random.split(jax.random.PRNGKey(42 + B * 10 + T), 16)
    it = iter(keys)
    pos = jax.device_put(jnp.arange(B, dtype=I32), rep)
    hid_l = jax.random.normal(next(it), (B, H), dtype=BF16) * 0.1
    hid_pad = jnp.pad(hid_l, ((0, M - B), (0, 0)))
    hid_mk = jax.device_put(hid_pad, tp)

    in_norm = jax.random.normal(next(it), (L, H), dtype=F32) * 0.05
    final_norm = jax.random.normal(next(it), (H,), dtype=F32) * 0.05
    lin_in = (jax.random.normal(next(it), (1, L, H, NLI), dtype=F32) * 0.2).astype(FP8)
    lin_in_s = (
        jnp.abs(jax.random.normal(next(it), (1, L, NLI), dtype=F32)) * 0.02 + 0.01
    )
    lin_out = (jax.random.normal(next(it), (1, L, Z, H), dtype=F32) * 0.2).astype(FP8)
    lin_out_s = jnp.abs(jax.random.normal(next(it), (1, L, H), dtype=F32)) * 0.02 + 0.01
    conv_w = jax.random.normal(next(it), (1, L, 4 * QKV), dtype=F32) * 0.05
    alog = jax.random.normal(next(it), (1, L, vh), dtype=F32) * 0.1 - 1.0
    dtb = jax.random.normal(next(it), (1, L, vh), dtype=F32) * 0.1
    gnorm = jnp.ones((1, L, VD), dtype=F32) + (
        jax.random.normal(next(it), (1, L, VD), dtype=F32) * 0.05
    )

    conv_l = jax.random.normal(next(it), (1, L, NS, 3, QKV), dtype=F32) * 0.05
    rec_l = jax.random.normal(next(it), (1, L, NS, vh, KD, VD), dtype=BF16) * 0.05

    lay = p.smalls_layout
    sm = jnp.zeros((1, L, 1, p.smalls_width), dtype=F32)

    def put_sm(name, arr):
        nonlocal sm
        o, w = lay[name]
        if arr.ndim == 2:
            arr = arr[None, :, :]
        vw = min(w, arr.shape[-1])
        sm = sm.at[:, :, 0, o : o + vw].set(arr[:, :, :vw])

    put_sm("in_norm", in_norm)
    put_sm("in_s", lin_in_s)
    put_sm("out_s", lin_out_s)
    put_sm("conv", conv_w)
    put_sm("alog", alog)
    put_sm("dtb", dtb)
    put_sm("gnorm", gnorm)

    def tile_w(arr, kind):
        *lead, K_, N_ = arr.shape
        return arr.reshape((*lead, p.nt[kind], p.tk[kind], N_))

    w_mk = {
        "lin_in": jax.device_put(tile_w(lin_in, "lin_in"), tp),
        "lin_out": jax.device_put(tile_w(lin_out, "lin_out"), tp),
        "smalls": jax.device_put(sm, tp),
        "final_norm": jax.device_put(final_norm[None, None, :], tp),
    }
    if T == 1:
        conv_3b = jnp.swapaxes(conv_l, 2, 3).reshape(1, L, 3 * B, QKV)
        conv_p = jnp.pad(conv_3b, ((0, 0), (0, 0), (0, CR - 3 * B), (0, 0)))
    else:
        conv_p = jnp.pad(conv_l, ((0, 0), (0, 0), (0, 0), (0, HR - 3), (0, 0)))
    st_mk = {
        "conv": jax.device_put(kernel.fold(conv_p), tp),
        "rec": jax.device_put(kernel.fold(rec_l.reshape(1, L, NS * vh, KD, VD)), tp),
    }

    step, _ = kernel.make_step(mesh, spec, ex)
    out_hid, out_st = step(pos, hid_mk, w_mk, st_mk)

    # Pure-JAX reference for L GDN-only layers
    res = hid_l.astype(F32)
    expand = vh // kh
    ref_conv = (
        jnp.zeros((L, CR, QKV), dtype=F32)
        if T == 1
        else jnp.zeros((L, NS, HR, QKV), dtype=F32)
    )
    ref_rec = jnp.zeros((L, B * vh, KD, VD), dtype=BF16)
    for lyr in range(L):
        xb = (
            rmsnorm(res, in_norm[lyr], g.eps, zero_centered=g.zero_centered_norm)
            .astype(BF16)
            .astype(F32)
        )
        ybuf = rnd((xb @ lin_in[0, lyr].astype(F32)) * lin_in_s[0, lyr])
        taps = [conv_w[0, lyr, t * QKV : (t + 1) * QKV] for t in range(4)]
        if T == 1:
            s0, s1, s2 = (
                conv_l[0, lyr, :, 0, :],
                conv_l[0, lyr, :, 1, :],
                conv_l[0, lyr, :, 2, :],
            )
            x_c = ybuf[:, :QKV]
            u = rnd(
                jax.nn.silu(s0 * taps[0] + s1 * taps[1] + s2 * taps[2] + x_c * taps[3])
            )
            ref_conv = ref_conv.at[lyr, : 3 * B].set(
                jnp.concatenate([s1, s2, x_c], axis=0)
            )
        else:
            us = []
            for s_idx in range(NS):
                hist = jnp.zeros((HR, QKV), dtype=F32)
                hist = hist.at[0:3].set(conv_l[0, lyr, s_idx])
                hist = hist.at[3 : 3 + T].set(ybuf[s_idx * T : (s_idx + 1) * T, :QKV])
                ref_conv = ref_conv.at[lyr, s_idx].set(hist)
                us.append(
                    rnd(
                        jax.nn.silu(
                            hist[0:T] * taps[0]
                            + hist[1 : 1 + T] * taps[1]
                            + hist[2 : 2 + T] * taps[2]
                            + hist[3 : 3 + T] * taps[3]
                        )
                    )
                )
            u = jnp.concatenate(us, axis=0)

        mix_rows = []
        rec_rows = []
        rec_prev = None
        for b in range(B):
            s_idx, t_idx = divmod(b, T)
            st_cur = (
                rec_l[0, lyr, s_idx].astype(F32) if t_idx == 0 else rec_prev.astype(F32)
            )
            q = u[b, : kh * KD].reshape(kh, KD)
            k = u[b, kh * KD : 2 * kh * KD].reshape(kh, KD)
            v = u[b, 2 * kh * KD :].reshape(vh, VD)
            q = (
                q
                * lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + gp.l2_eps)
                * (KD**-0.5)
            )
            k = k * lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + gp.l2_eps)
            qc = jnp.repeat(q, expand, axis=0)[:, :, None]
            kc_ = jnp.repeat(k, expand, axis=0)[:, :, None]
            beta = rnd(jax.nn.sigmoid(ybuf[b, BA : BA + vh]))
            a_val = ybuf[b, lin_lay.A_OFF : lin_lay.A_OFF + vh]
            decay = -jnp.exp(alog[0, lyr]) * jax.nn.softplus(a_val + dtb[0, lyr])
            st_cur = st_cur * jnp.exp(decay)[:, None, None]
            pred = jnp.sum(st_cur * kc_, axis=1)
            delta = (v - pred) * beta[:, None]
            st_cur = st_cur + kc_ * delta[:, None, :]
            out_b = rnd(jnp.sum(st_cur * qc, axis=1))
            rec_prev = st_cur.astype(BF16)
            rec_rows.append(rec_prev)
            val = out_b * lax.rsqrt(
                jnp.mean(out_b * out_b, axis=-1, keepdims=True) + g.eps
            )
            val = (gnorm[0, lyr].astype(BF16)[None, :] * val.astype(BF16)).astype(F32)
            zz = ybuf[b, QKV : QKV + Z].reshape(vh, VD)
            mix_rows.append(rnd(val * jax.nn.silu(zz)).reshape(Z))
        ref_rec = ref_rec.at[lyr].set(jnp.concatenate(rec_rows, axis=0))
        mix_arr = jnp.stack(mix_rows, axis=0).astype(BF16).astype(F32)
        yo = (mix_arr @ lin_out[0, lyr].astype(F32)) * lin_out_s[0, lyr]
        res = rnd(res + yo.astype(BF16).astype(F32))

    ref_hid = rmsnorm(
        res, final_norm, g.eps, zero_centered=g.zero_centered_norm
    ).astype(BF16)

    conv_key = "conv" if T == 1 else "conv_ckpt"
    rec_key = "rec" if T == 1 else "rec_ckpt"
    assert jnp.allclose(
        out_hid[:B].astype(F32), ref_hid.astype(F32), atol=1e-2, rtol=1e-2
    )
    assert jnp.allclose(
        out_st[conv_key].astype(F32), ref_conv.astype(F32), atol=1e-2, rtol=1e-2
    )
    assert jnp.allclose(
        out_st[rec_key].astype(F32), ref_rec.astype(F32), atol=1e-2, rtol=1e-2
    )
    assert out_st["conv"].shape == st_mk["conv"].shape
    assert out_st["rec"].shape == st_mk["rec"].shape


@pytest.mark.parametrize(
    "R,B,T,moe_tp,gdn_rows",
    [
        (1, 2, 1, False, 0),
        (1, 4, 2, True, 0),
        (2, 2, 1, False, 0),
        pytest.param(2, 4, 2, False, 0, marks=pytest.mark.slow),
        pytest.param(2, 20, 1, False, 8, marks=pytest.mark.slow),
    ],
)
def test_m2_qwen35_hybrid_pack_and_step_vs_logical_reference(
    R: int,
    B: int,
    T: int,
    moe_tp: bool,
    gdn_rows: int,
):
    """Verify hybrid `att_every=4` (3 GDN + 1 GQA) matches `reference_step`."""
    devs = jax.devices("cpu")
    assert len(devs) >= R, f"Requires {R} CPU devices, found {len(devs)}"

    g = qwen35.Qwen35Geometry(
        L=4,
        att_every=4,
        H=256,
        NH=2 * R,
        NKV=1,
        HD=128,
        ROT=32,
        KH=2 * R,
        VH=4 * R,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=moe_tp,
    )
    tiles = dataclasses.replace(
        TINY_TILES, bank=256 << 10, inplace=R == 1 and not T > 1
    ).with_blocks(
        {"gqa": {"split": True}, "gdn": {"gdn_rows": gdn_rows, "gdn_bufs": 2}}
    )
    ex = ExecSpec(B=B, T=T, R=R, ctx=256, tiles=tiles, interpret=True)

    key = jax.random.PRNGKey(200 + R * 10 + T + gdn_rows)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(key, g, ex)
    ref_hid, ref_st = qwen35.reference_step(g, ex, pos, hid_l, w_l, st_l)

    mesh = Mesh(devs[:R], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh
    )
    in_conv_shape = st_mk["conv"].shape
    in_rec_shape = st_mk["rec"].shape
    step, _ = qwen35.make_step(mesh, g, ex, donate_states=True)
    out_hid, out_st = step(pos_mk, hid_mk, w_mk, st_mk)

    assert out_st["conv"].shape == in_conv_shape
    assert out_st["rec"].shape == in_rec_shape

    out_hid_r = kernel.unfold(out_hid, R)
    for r in range(R):
        assert jnp.allclose(
            out_hid_r[r, :B].astype(F32),
            ref_hid.astype(F32),
            atol=2e-2,
            rtol=2e-2,
        )

    conv_key = "conv" if T == 1 else "conv_ckpt"
    rec_key = "rec" if T == 1 else "rec_ckpt"
    for k, tol in ((conv_key, 2e-2), (rec_key, 2e-2), ("kc", 2e-2), ("vc", 2e-2)):
        got = kernel.unfold(out_st[k], R).astype(F32)
        exp = ref_st[k].astype(F32)
        assert jnp.allclose(got, exp, atol=tol, rtol=tol), k

    if T > 1:
        acc = jnp.asarray([1, 2], dtype=I32)
        vh = g.vh(R)
        jit_accept = jax.jit(lambda st, a: GDNBlock.accept(st, st, a, T=T, vh=vh))
        acc_folded = jit_accept(out_st, acc)
        acc_unfolded = qwen35.accept_states(g, ex, ref_st, [1, 2])
        assert jnp.allclose(
            kernel.unfold(acc_folded["conv"], R)[:, :, :, :3, :],
            acc_unfolded["conv"][:, :, :, :3, :],
            atol=2e-2,
            rtol=2e-2,
        )
        assert jnp.allclose(
            kernel.unfold(acc_folded["rec"], R).astype(F32),
            acc_unfolded["rec"].astype(F32),
            atol=2e-2,
            rtol=2e-2,
        )


@pytest.mark.slow
@pytest.mark.parametrize("T", [1, 2])
def test_m2_qwen35_hybrid_r8_multi_rank_interpret(T: int):
    """Verify 4-layer hybrid (`3 GDN + 1 GQA`) at `R=8` with `split=True, hier=1`."""
    devs = jax.devices("cpu")
    assert len(devs) >= 8, f"Requires 8 CPU devices, found {len(devs)}"

    g = qwen35.Qwen35Geometry(
        L=4,
        att_every=4,
        H=1024,
        NH=8,
        NKV=2,
        HD=128,
        ROT=32,
        KH=8,
        VH=16,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=1024,
        moe_tp=False,
    )
    tiles = dataclasses.replace(
        TINY_TILES, bank=256 << 10, hier=1, mb8=2, inplace=False
    ).with_blocks({"gqa": {"split": True}})
    ex = ExecSpec(B=2 * T, T=T, R=8, ctx=256, tiles=tiles, interpret=True)

    key = jax.random.PRNGKey(909 + T)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(key, g, ex)
    ref_hid, ref_st = qwen35.reference_step(g, ex, pos, hid_l, w_l, st_l)

    mesh = Mesh(devs[:8], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh
    )
    step, _ = qwen35.make_step(mesh, g, ex, donate_states=True)
    out_hid, out_st = step(pos_mk, hid_mk, w_mk, st_mk)

    out_hid_r = kernel.unfold(out_hid, 8)
    for r in range(8):
        assert jnp.allclose(
            out_hid_r[r, : ex.B].astype(F32),
            ref_hid.astype(F32),
            atol=3e-2,
            rtol=3e-2,
        )
    conv_key = "conv" if T == 1 else "conv_ckpt"
    rec_key = "rec" if T == 1 else "rec_ckpt"
    for k, tol in ((conv_key, 3e-2), (rec_key, 3e-2), ("kc", 3e-2), ("vc", 3e-2)):
        got = kernel.unfold(out_st[k], 8).astype(F32)
        exp = ref_st[k].astype(F32)
        assert jnp.allclose(got, exp, atol=tol, rtol=tol), k


@pytest.mark.skipif(
    not _can_compile_tpu7x(), reason="TPU7x AOT topology compiler not available"
)
@pytest.mark.parametrize("B", [4, 32])
def test_m2_tpu7x_aot_compile_full_ep_hybrid(B: int):
    """Verify hybrid `FULL_EP` (`att_every=4`) lowers and AOT-compiles for `tpu7x`."""
    topo = topologies.get_topology_desc(
        platform="tpu", topology_name="tpu7x", **{"1d": 8}
    )
    mesh = Mesh(topo.devices, ("tp",))
    g = dataclasses.replace(qwen35.FULL_EP, L=4, att_every=4)
    ex = ExecSpec(B=B, T=1, R=8, ctx=2048, tiles=baseline_tiles(B, R=8))

    step, info = qwen35.make_step(mesh, g, ex, donate_states=True)
    wsh = info["wshape"]
    ssh = info["sshape"]

    rep = NamedSharding(mesh, P())
    tp = NamedSharding(mesh, P("tp"))

    pos = jax.ShapeDtypeStruct((ex.B,), jnp.int32, sharding=rep)
    hid = jax.ShapeDtypeStruct((ex.R * ex.M, g.H), jnp.bfloat16, sharding=tp)
    weights = {
        k: tuple(
            jax.ShapeDtypeStruct((ex.R, *s.shape), s.dtype, sharding=tp) for s in v
        )
        if isinstance(v, tuple) and hasattr(v[0], "shape")
        else jax.ShapeDtypeStruct((ex.R, *v[0]), v[1], sharding=tp)
        for k, v in wsh.items()
    }
    states = {
        k: jax.ShapeDtypeStruct((ex.R * v[0][0], *v[0][1:]), v[1], sharding=tp)
        for k, v in ssh.items()
    }

    compiled = step.lower(pos, hid, weights, states).compile()
    hlo_text = compiled.as_text()
    assert "HloModule" in hlo_text
    assert "qwen3_5_decode_megakernel" in hlo_text


@pytest.mark.skipif(
    not os.environ.get("MK_FROZEN_QWEN35_DIR"),
    reason="Set MK_FROZEN_QWEN35_DIR to frozen qwen35 path for bit-exact check",
)
@pytest.mark.parametrize(
    "B,T,gdn_rows",
    [
        (2, 1, 0),
        (4, 2, 0),
        pytest.param(20, 1, 8, marks=pytest.mark.slow),
        pytest.param(20, 2, 8, marks=pytest.mark.slow),
    ],
)
def test_m2_bit_exact_vs_frozen_qwen35_hybrid(B: int, T: int, gdn_rows: int):
    """Opt-in bit-exact check vs frozen `qwen35_megakernel` at `att_every=4`."""
    import sys

    frozen_dir = pathlib.Path(os.environ["MK_FROZEN_QWEN35_DIR"])
    if str(frozen_dir) not in sys.path:
        sys.path.insert(0, str(frozen_dir))
    spec_f = importlib.util.spec_from_file_location(
        "_frozen_qwen35_mk", frozen_dir / "megakernel.py"
    )
    assert spec_f is not None and spec_f.loader is not None
    frozen_mk = importlib.util.module_from_spec(spec_f)
    sys.modules[spec_f.name] = frozen_mk
    spec_f.loader.exec_module(frozen_mk)

    devs = jax.devices("cpu")
    mesh = Mesh(devs[:1], ("tp",))
    tp = NamedSharding(mesh, P("tp"))

    g = qwen35.Qwen35Geometry(
        L=4,
        att_every=4,
        H=256,
        NH=2,
        NKV=1,
        HD=128,
        ROT=32,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
    )
    tiles_lib = dataclasses.replace(
        TINY_TILES, bank=256 << 10, inplace=not T > 1
    ).with_blocks({"gdn": {"gdn_rows": gdn_rows, "gdn_bufs": 2}})
    ex = ExecSpec(B=B, T=T, R=1, ctx=256, tiles=tiles_lib, interpret=True)
    pos, hid_lib, w_lib, st_lib = qwen35.random_inputs(
        jax.random.PRNGKey(B * 10 + T), g, ex, mesh=mesh
    )
    step_lib, info_lib = qwen35.make_step(mesh, g, ex)
    out_hid_lib, out_st_lib = step_lib(pos, hid_lib, w_lib, st_lib)

    tiles_f = frozen_mk.Tiles(
        bank=256 << 10,
        nbd=10,
        nbe=6,
        ebank=256 << 10,
        vmem_mib=32,
        inplace=not T > 1,
        kv_blk=256,
        nkv=3,
        gdn_rows=gdn_rows,
        gdn_bufs=2,
    )
    c_f = frozen_mk.MKConfig(
        R=1,
        B=B,
        T=T,
        L=4,
        att_every=4,
        H=256,
        NH=2,
        NKV=1,
        HD=128,
        ROT=32,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        ctx=256,
        stack_e=False,
    )
    f_lay, f_S = frozen_mk.smalls_layout(c_f)
    l_lay = info_lib["plan"].smalls_layout

    sm_f = jnp.zeros((ex.R, g.L, 1, f_S), dtype=F32)
    for name, (lo, w) in l_lay.items():
        fo, _ = f_lay[name]
        sm_f = sm_f.at[:, :, 0, fo : fo + w].set(w_lib["smalls"][:, :, 0, lo : lo + w])

    w_f = {**w_lib, "smalls": jax.device_put(sm_f, tp)}
    st_f = dict(st_lib)
    step_f, _ = frozen_mk.make_step(mesh, c_f, tiles_f, interpret=True)
    out_hid_f, out_st_f = jax.jit(step_f)(pos, hid_lib, w_f, st_f)

    assert jnp.array_equal(out_hid_lib, out_hid_f)
    conv_key = "conv" if T == 1 else "conv_ckpt"
    rec_key = "rec" if T == 1 else "rec_ckpt"
    assert jnp.array_equal(out_st_lib[conv_key], out_st_f["conv"]), conv_key
    assert jnp.array_equal(out_st_lib[rec_key], out_st_f["rec"]), rec_key
    assert jnp.array_equal(out_st_lib["kc"], out_st_f["kc"]), "kc"
    assert jnp.array_equal(out_st_lib["vc"], out_st_f["vc"]), "vc"
