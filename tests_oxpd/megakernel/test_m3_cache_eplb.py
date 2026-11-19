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
"""M3a tests: Paged KV in `GQABlock`, 128K `page_table` SMEM, and KV page helpers."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES, baseline_tiles


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _cpu_mesh(R: int) -> Mesh:
    devs = jax.devices("cpu")
    if len(devs) < R:
        pytest.skip(f"Need at least {R} CPU devices, have {len(devs)}")
    return Mesh(np.array(devs[:R]), ("tp",))


def _tiny_geom(
    *,
    L: int = 4,
    att_every: int = 4,
    NKV: int = 2,
    moe_tp: bool = False,
    e_slots: int = 0,
) -> qwen35.Qwen35Geometry:
    return qwen35.Qwen35Geometry(
        L=L,
        H=256,
        I=256,
        IS=256,
        E=128,
        TOPK=2,
        NH=4,
        NKV=NKV,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        att_every=att_every,
        moe_tp=moe_tp,
        e_slots=e_slots,
    )


def _tiny_exec(
    *,
    R: int = 2,
    B: int = 2,
    T: int = 1,
    ctx: int = 256,
    paged_kv: bool = True,
    num_pages: int = 64,
    max_pages_per_seq: int = 0,
    stack_e: bool = True,
    ecopy_opt: int = 0,
    amask: int = 1,
    kv_sub: int = 0,
) -> mk.ExecSpec:
    tiles = replace(TINY_TILES, bank=512 << 10).with_blocks(
        {
            "gqa": {
                "kv_blk": min(TINY_TILES.block("gqa").kv_blk, ctx),
                "kv_sub": kv_sub,
                "amask": amask,
            },
            "moe": {"ecopy_opt": ecopy_opt},
        }
    )
    return mk.ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=ctx,
        stack_e=stack_e,
        interpret=True,
        tiles=tiles,
        cache=mk.CacheExec(
            paged_kv=paged_kv, num_pages=num_pages, max_pages_per_seq=max_pages_per_seq
        ),
    )


def _pack_kv_to_raw_pools(
    g: qwen35.Qwen35Geometry,
    ex: mk.ExecSpec,
    kc: jax.Array,
    vc: jax.Array,
    *,
    pt: np.ndarray | None = None,
) -> tuple[tuple[jax.Array, ...], jax.Array]:
    """Pack `[R, NATT, NS, ctx_tokens, HD]` `kc`/`vc` into `g.NATT` raw pools."""
    ctx_tokens = kc.shape[3]
    pages_per_seq = ctx_tokens // mk.cache_exec(ex).page_size
    pt_cols = max(mk.pool.resolved_max_pages_per_seq(ex), pages_per_seq)
    if pt is None:
        pt = np.zeros((ex.NS, pt_cols), dtype=np.int32)
        rng = np.random.default_rng(1337)
        # Reserve pages 0..3 as unreferenced sentinels and permute remaining pages
        perm = rng.permutation(mk.cache_exec(ex).num_pages - 4).astype(np.int32) + 4
        for s in range(ex.NS):
            pt[s, :pages_per_seq] = perm[s * pages_per_seq : (s + 1) * pages_per_seq]

    raw_list = []
    for g_idx in range(g.NATT):
        pool_g = np.zeros(
            (
                ex.R,
                mk.cache_exec(ex).num_pages,
                2,
                g.HD // 4,
                4,
                mk.cache_exec(ex).page_size,
            ),
            dtype=jnp.dtype(jnp.float8_e4m3fn),
        )
        kc_g = np.asarray(
            jnp.clip(kc[:, g_idx], -448.0, 448.0).astype(jnp.float8_e4m3fn)
        )
        vc_g = np.asarray(
            jnp.clip(vc[:, g_idx], -448.0, 448.0).astype(jnp.float8_e4m3fn)
        )
        for r in range(ex.R):
            for s in range(ex.NS):
                for p in range(pages_per_seq):
                    kp = int(pt[s, p])
                    t0 = p * mk.cache_exec(ex).page_size
                    k_sl = kc_g[r, s, t0 : t0 + mk.cache_exec(ex).page_size]
                    v_sl = vc_g[r, s, t0 : t0 + mk.cache_exec(ex).page_size]
                    pool_g[r, kp] = np.asarray(mk.kv_to_raw_page(k_sl, v_sl))
        raw_list.append(
            jnp.asarray(
                pool_g.reshape(
                    ex.R * mk.cache_exec(ex).num_pages,
                    2,
                    g.HD // 4,
                    4,
                    mk.cache_exec(ex).page_size,
                )
            )
        )
    return tuple(raw_list), jnp.asarray(pt)


def test_raw_page_pack_unpack_roundtrip():
    """Verify per-page KV, SSM, and conv raw page pack/unpack helpers round-trip."""
    key = jax.random.key(101)
    k1, k2 = jax.random.split(key, 2)

    # 1. KV page [128, 256] fp8 -> [2, 64, 4, 128] fp8 -> [128, 256] fp8
    k_page = (jax.random.normal(k1, (128, 256), dtype=jnp.float32) * 0.2).astype(
        jnp.float8_e4m3fn
    )
    v_page = (jax.random.normal(k2, (128, 256), dtype=jnp.float32) * 0.2).astype(
        jnp.float8_e4m3fn
    )
    raw_kv = mk.kv_to_raw_page(k_page, v_page)
    assert raw_kv.shape == (2, 64, 4, 128)
    k_rt, v_rt = mk.raw_page_to_kv(raw_kv)
    np.testing.assert_array_equal(
        np.asarray(k_rt).view(np.uint8), np.asarray(k_page).view(np.uint8)
    )
    np.testing.assert_array_equal(
        np.asarray(v_rt).view(np.uint8), np.asarray(v_page).view(np.uint8)
    )

    # 2. SSM state [vh=2, KD=128, VD=128] bf16 -> [4, 2, 64, 4, 128] fp8 -> bf16
    rec = (jax.random.normal(k1, (2, 128, 128), dtype=jnp.float32) * 0.1).astype(
        jnp.bfloat16
    )
    raw_ssm = mk.ssm_to_raw_pages(rec)
    assert raw_ssm.shape == (4, 2, 64, 4, 128)
    rec_rt = mk.raw_pages_to_ssm(raw_ssm, vh=2, KD=128, VD=128)
    np.testing.assert_array_equal(np.asarray(rec_rt), np.asarray(rec))

    # 3. Conv state [3, QKV=512] f32 -> [2, 64, 4, 128] fp8 -> [3, 512] f32
    conv = (
        (jax.random.normal(k1, (3, 512), dtype=jnp.float32) * 0.1)
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )
    raw_conv = mk.conv_to_raw_page(conv)
    assert raw_conv.shape == (2, 64, 4, 128)
    conv_rt = mk.raw_page_to_conv(raw_conv, QKV=512)
    np.testing.assert_array_equal(np.asarray(conv_rt), np.asarray(conv))


@pytest.mark.parametrize("att_every,use_2d_ckpt", [(4, False), (3, True)])
def test_states_to_pool_roundtrip(att_every: int, use_2d_ckpt: bool):
    """Verify `states_to_pool` <-> `pool_to_states` round-trip for 3D and 2D ckpt."""
    L = 4 if att_every == 4 else 6
    g = _tiny_geom(L=L, att_every=att_every)
    ex = _tiny_exec(R=2, B=2, T=1, ctx=256, paged_kv=True, num_pages=96)
    _, _, _, st_l = qwen35.random_logical_inputs(jax.random.key(7 + att_every), g, ex)

    st_in = dict(st_l)
    st_in["kc"] = st_l["kc"].astype(jnp.float8_e4m3fn)
    st_in["vc"] = st_l["vc"].astype(jnp.float8_e4m3fn)
    st_in["conv"] = st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32)

    pt, ckpt, roff, min_np = mk.default_pool_layout(g, ex)
    assert mk.cache_exec(ex).num_pages >= min_np
    # Pages 0..5*G_PER_ATT-1 are reserved; active sequences start at page >= 5*G_PER_ATT
    assert int(jnp.min(pt[:, : ex.ctx // mk.cache_exec(ex).page_size])) >= 5 * (
        att_every - 1
    )
    assert int(jnp.min(ckpt)) >= 1
    if use_2d_ckpt:
        ckpt = ckpt[0]
        roff = roff[0]

    raw_pools = mk.states_to_pool(g, ex, st_in, pt, ckpt, roff)
    st_out = mk.pool_to_states(g, ex, raw_pools, pt, ckpt)

    np.testing.assert_array_equal(
        np.asarray(st_out["kc"]).view(np.uint8),
        np.asarray(st_in["kc"]).view(np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(st_out["vc"]).view(np.uint8),
        np.asarray(st_in["vc"]).view(np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(st_out["conv"][:, :, :, 0]), np.asarray(st_in["conv"])
    )
    vh = g.vh(ex.R)
    np.testing.assert_array_equal(
        np.asarray(st_out["rec"].reshape(ex.R, g.NGDN, ex.NS, vh, g.KD, g.VD)),
        np.asarray(st_in["rec"]),
    )


@pytest.mark.parametrize(
    "T,NS,att_every",
    [(1, 1, 4), (1, 2, 4), (1, 3, 3), (1, 4, 4), (1, 5, 3), (2, 2, 4)],
)
def test_gather_gdn_states_cpu_interpret(T: int, NS: int, att_every: int):
    """Verify `gather_gdn_states` across `NS=1..5` (`n_pairs>1`) and `from_ckpt`."""
    L = 4 if att_every == 4 else 6
    B = NS * T
    g = _tiny_geom(L=L, att_every=att_every)
    ex = _tiny_exec(R=2, B=B, T=T, ctx=256, paged_kv=True, num_pages=96)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.key(15 + T + NS), g, ex
    )
    st_quant = dict(
        st_l,
        conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32),
    )
    _, _, _, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_l, st_quant, g, ex, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    # Test 2D ckpt_indices when att_every == 3 (G_PER_ATT = 2 != 3)
    ckpt_use = ckpt[0] if att_every == 3 else ckpt
    roff_use = roff[0] if att_every == 3 else roff
    raw_pools = mk.states_to_pool(g, ex, st_quant, pt, ckpt_use, roff_use)
    raw_r0 = tuple(mk.unfold(r, ex.R)[0] for r in raw_pools)

    conv_g, rec_g = mk.gather_gdn_states(g, ex, raw_r0, ckpt_use, roff_use)
    conv_exp = mk.unfold(st_mk["conv"], ex.R)[0]
    rec_exp = mk.unfold(st_mk["rec"], ex.R)[0]
    np.testing.assert_array_equal(np.asarray(conv_g), np.asarray(conv_exp))
    np.testing.assert_array_equal(np.asarray(rec_g), np.asarray(rec_exp))

    if NS >= 2:
        # 1. Explicit active_seqs mask: zero out sequence 1
        act_mask = jnp.asarray([1] + [0] + [1] * (NS - 2), dtype=jnp.int32)
        _, rec_act = mk.gather_gdn_states(g, ex, raw_r0, ckpt_use, roff_use, act_mask)
        vh = g.vh(ex.R)
        rec_act_seq = rec_act.reshape(g.NGDN, ex.NS, vh, g.KD, g.VD)
        np.testing.assert_array_equal(
            np.asarray(rec_act_seq[:, 0]),
            np.asarray(rec_exp.reshape(g.NGDN, ex.NS, vh, g.KD, g.VD)[:, 0]),
        )
        np.testing.assert_array_equal(
            np.asarray(rec_act_seq[:, 1]),
            np.zeros_like(np.asarray(rec_act_seq[:, 1])),
        )
        # 2. active_seqs=None (default "from_ckpt") and explicit "from_ckpt":
        # setting sequence 1's slot to 0 marks it inactive
        ckpt_inact = (
            ckpt_use.at[:, 1, :].set(0)
            if ckpt_use.ndim == 3
            else ckpt_use.at[1, :].set(0)
        )
        _, rec_fc = mk.gather_gdn_states(
            g, ex, raw_r0, ckpt_inact, roff_use, "from_ckpt"
        )
        _, rec_def = mk.gather_gdn_states(g, ex, raw_r0, ckpt_inact, roff_use)
        np.testing.assert_array_equal(np.asarray(rec_fc), np.asarray(rec_act))
        np.testing.assert_array_equal(np.asarray(rec_def), np.asarray(rec_act))
        with pytest.raises(ValueError, match="Unsupported active_seqs mode"):
            mk.gather_gdn_states(g, ex, raw_r0, ckpt_use, roff_use, "invalid")
        with pytest.raises(ValueError, match="maps to pages"):
            mk.gather_gdn_states(
                g, ex, raw_r0, ckpt_use + mk.cache_exec(ex).num_pages, roff_use
            )
        if NS == 2 and T == 1 and att_every == 4:
            from jax.experimental import pallas as pl

            with (
                pl.enable_debug_checks(True),
                pytest.raises(
                    jax.errors.JaxRuntimeError,
                    match="GDN gather checkpoint slot maps outside",
                ),
            ):
                c_bad, _ = jax.jit(
                    lambda ck: mk.gather_gdn_states(g, ex, raw_r0, ck, roff_use)
                )(ckpt_use + mk.cache_exec(ex).num_pages)
                c_bad.block_until_ready()


@pytest.mark.parametrize("T,B", [(1, 2), (2, 4)])
def test_paged_gqa_make_step_cpu_interpret(T: int, B: int):
    """Verify `GQABlock` with `paged_kv=True`, `G=2`, `NATT=2`, and permuted pages."""
    mesh = _cpu_mesh(2)
    # L=8, att_every=4 -> NATT=2 attention pools; NKV=1 with R=2 -> G=2 (SPLIT=True)
    g = _tiny_geom(L=8, att_every=4, NKV=1)
    ex = _tiny_exec(R=2, B=B, T=T, ctx=512, paged_kv=True, num_pages=64)
    _, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.key(20 + T), g, ex)
    # Sequence 0 is in KV block 1 (page 2, pos >= 300); sequence 1 in block 0 (page 1)
    pos = (
        jnp.asarray([300, 150], dtype=jnp.int32)
        if T == 1
        else jnp.asarray([300, 301, 150, 151], dtype=jnp.int32)
    )

    ref_out, ref_st = qwen35.reference_step(g, ex, pos, hid_l, w_l, st_l, paged_kv=True)

    raw_pools, pt = _pack_kv_to_raw_pools(g, ex, st_l["kc"], st_l["vc"])
    st_paged_l = {**st_l, "raw": raw_pools, "page_table": pt}
    _, hid_mk, w_mk, st_mk_paged = qwen35.pack_inputs(
        pos, hid_l, w_l, st_paged_l, g, ex, mesh=mesh
    )
    mk_paged_step, _ = qwen35.make_step(mesh, g, ex, donate_states=False)
    out_hid, mk_st = mk_paged_step(pos, hid_mk, w_mk, st_mk_paged)
    out_b = mk.unfold(out_hid, ex.R)[0, :B]

    # Also run unpaged megakernel to verify parity across all NATT=2 pools
    ex_unp = _tiny_exec(R=2, B=B, T=T, ctx=512, paged_kv=False)
    _, hid_unp, w_unp, st_mk_unp = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex_unp, mesh=mesh
    )
    mk_unp_step, _ = qwen35.make_step(mesh, g, ex_unp, donate_states=False)
    out_unp, st_unp = mk_unp_step(pos, hid_unp, w_unp, st_mk_unp)
    assert jnp.allclose(
        out_hid.astype(jnp.float32), out_unp.astype(jnp.float32), atol=5e-2, rtol=5e-2
    )

    assert "raw" not in mk_st
    assert "page_table" not in mk_st
    assert mk_st["knew"].dtype == jnp.float8_e4m3fn
    assert mk_st["vnew"].dtype == jnp.float8_e4m3fn
    if T == 1:
        # Trace non-interpret TPU path (Ref.bitcast(U32).reshape(HD // 4, BLK)) on CPU
        mk_non_interp, _ = qwen35.make_step(
            mesh, g, replace(ex, interpret=False), donate_states=False
        )
        jaxpr_non_interp = jax.make_jaxpr(mk_non_interp)(pos, hid_mk, w_mk, st_mk_paged)
        assert "decode_megakernel" in str(jaxpr_non_interp)

        # Verify 3D page_table (R, NS, W) at R=2 and donate_states=True (read-only
        # raw pools and page_table must NOT be invalidated while conv/rec are donated)
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        tp_s = NamedSharding(mesh, P("tp"))
        pt_3d = jnp.broadcast_to(
            pt[None, :, : mk.pool.resolved_max_pages_per_seq(ex)],
            (ex.R, ex.NS, mk.pool.resolved_max_pages_per_seq(ex)),
        )
        st_don = {
            "conv": jax.device_put(np.asarray(st_mk_paged["conv"]).copy(), tp_s),
            "rec": jax.device_put(np.asarray(st_mk_paged["rec"]).copy(), tp_s),
            "raw": tuple(
                jax.device_put(np.asarray(r).copy(), tp_s) for r in st_mk_paged["raw"]
            ),
            "page_table": jax.device_put(np.asarray(pt_3d).copy(), tp_s),
        }
        mk_don_step, _ = qwen35.make_step(mesh, g, ex, donate_states=True)
        out_don, mk_st_don = mk_don_step(pos, hid_mk, w_mk, st_don)
        np.testing.assert_array_equal(np.asarray(out_don), np.asarray(out_hid))
        assert "raw" not in mk_st_don and "page_table" not in mk_st_don
        assert st_don["conv"].is_deleted() and st_don["rec"].is_deleted()
        assert not any(r.is_deleted() for r in st_don["raw"])
        assert not st_don["page_table"].is_deleted()
    knew_u = mk.unfold(mk_st["knew"], ex.R)
    vnew_u = mk.unfold(mk_st["vnew"], ex.R)
    kc_unp = mk.unfold(st_unp["kc"], ex.R)
    vc_unp = mk.unfold(st_unp["vc"], ex.R)
    assert knew_u.shape == (ex.R, g.NATT, ex.MR, g.HD)
    # Verify knew/vnew across all NATT=2 attention layers and both sequences
    for a_idx in range(g.NATT):
        for s_idx in range(ex.NS):
            b0 = s_idx * T
            p0 = int(pos[b0])
            got_k_fp8 = knew_u[:, a_idx, b0 : b0 + T]
            got_v_fp8 = vnew_u[:, a_idx, b0 : b0 + T]
            unp_k_fp8 = jnp.clip(
                kc_unp[:, a_idx, s_idx, p0 : p0 + T], -448.0, 448.0
            ).astype(jnp.float8_e4m3fn)
            unp_v_fp8 = jnp.clip(
                vc_unp[:, a_idx, s_idx, p0 : p0 + T], -448.0, 448.0
            ).astype(jnp.float8_e4m3fn)
            if a_idx == 0:
                assert jnp.array_equal(got_k_fp8, unp_k_fp8)
                assert jnp.array_equal(got_v_fp8, unp_v_fp8)
            ref_k = ref_st["kc"][:, a_idx, s_idx, p0 : p0 + T].astype(jnp.float32)
            ref_v = ref_st["vc"][:, a_idx, s_idx, p0 : p0 + T].astype(jnp.float32)
            assert jnp.allclose(
                got_k_fp8.astype(jnp.float32), ref_k, atol=2.5e-1, rtol=1.5e-1
            )
            assert jnp.allclose(
                got_v_fp8.astype(jnp.float32), ref_v, atol=2.5e-1, rtol=1.5e-1
            )

    assert jnp.allclose(
        out_b.astype(jnp.float32), ref_out.astype(jnp.float32), atol=5e-2, rtol=5e-2
    )


@pytest.mark.parametrize("amask", [0, 1, 2])
def test_paged_kv_fp8_nan_padding_guard(amask: int):
    """Verify `amask in (0, 1, 2)` masks out `0x7f` (`NaN`) via the `uint32` path."""
    mesh = _cpu_mesh(2)
    g = _tiny_geom(L=2, att_every=1)
    ex = _tiny_exec(
        R=2, B=2, T=1, ctx=256, paged_kv=True, num_pages=32, amask=amask, kv_sub=128
    )
    _, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.key(44), g, ex)
    pos = jnp.asarray([180, 60], dtype=jnp.int32)

    nan_fp8 = jax.lax.bitcast_convert_type(jnp.uint8(0x7F), jnp.float8_e4m3fn)
    assert np.isnan(np.asarray(nan_fp8, dtype=np.float32))

    kc_fp8 = st_l["kc"].astype(jnp.float8_e4m3fn)
    vc_fp8 = st_l["vc"].astype(jnp.float8_e4m3fn)
    idx = jnp.arange(ex.ctx)[None, None, None, :, None]
    plen = pos[None, None, :, None, None]
    kc_nan = jnp.where(idx < plen, kc_fp8, nan_fp8)
    vc_nan = jnp.where(idx < plen, vc_fp8, nan_fp8)

    raw_clean, pt = _pack_kv_to_raw_pools(g, ex, st_l["kc"], st_l["vc"])
    raw_nan, _ = _pack_kv_to_raw_pools(g, ex, kc_nan, vc_nan, pt=np.asarray(pt))

    _, hid_mk, w_mk, st_clean_mk = qwen35.pack_inputs(
        pos,
        hid_l,
        w_l,
        {**st_l, "raw": raw_clean, "page_table": pt},
        g,
        ex,
        mesh=mesh,
    )
    _, _, _, st_nan_mk = qwen35.pack_inputs(
        pos,
        hid_l,
        w_l,
        {**st_l, "raw": raw_nan, "page_table": pt},
        g,
        ex,
        mesh=mesh,
    )

    attn_step, _ = qwen35.make_step(mesh, g, ex, donate_states=False)
    out_clean, _ = attn_step(pos, hid_mk, w_mk, st_clean_mk)
    out_nan, _ = attn_step(pos, hid_mk, w_mk, st_nan_mk)
    assert jnp.all(jnp.isfinite(out_nan))
    np.testing.assert_array_equal(np.asarray(out_nan), np.asarray(out_clean))

    if amask == 1:
        for bad_amask in (3, 4):
            ex_bad = _tiny_exec(
                R=2, B=2, T=1, ctx=256, paged_kv=True, num_pages=32, amask=bad_amask
            )
            with pytest.raises(AssertionError, match="paged_kv requires amask"):
                qwen35.build(g, ex_bad)


def test_long_context_128k_page_table_plan_and_interpret():
    """Verify `ctx=131072` SMEM budgets, SMEM limit check, and `eff_ctx > ex.ctx`."""
    g_full = qwen35.Qwen35Geometry()
    t64 = baseline_tiles(64, R=8)
    ex_4k_direct = mk.ExecSpec(
        R=8, B=64, T=1, ctx=4096, tiles=t64, cache=mk.CacheExec(paged_kv=False)
    )
    ex_4k_paged = mk.ExecSpec(
        R=8,
        B=64,
        T=1,
        ctx=4096,
        tiles=t64,
        cache=mk.CacheExec(paged_kv=True, num_pages=65536),
    )
    ex_128k_paged = mk.ExecSpec(
        R=8,
        B=64,
        T=1,
        ctx=131072,
        tiles=t64,
        cache=mk.CacheExec(paged_kv=True, num_pages=65536),
    )
    assert mk.pool.resolved_max_pages_per_seq(ex_128k_paged) == 1024

    spec_full = qwen35.qwen35_spec(g_full)
    cp_4k_direct, _ = mk.vmem.build_scratch(spec_full, ex_4k_direct)
    cp_4k_paged, _ = mk.vmem.build_scratch(spec_full, ex_4k_paged)
    cp_128k_paged, _ = mk.vmem.build_scratch(spec_full, ex_128k_paged)

    def _buf_bytes(cp_info, name: str) -> int:
        if name not in cp_info.vmem_shapes:
            return 0
        shape, dt = cp_info.vmem_shapes[name]
        return int(np.prod(shape)) * jnp.dtype(dt).itemsize

    expected_direct_minus_paged = (
        _buf_bytes(cp_4k_direct, "kbuf")
        + _buf_bytes(cp_4k_direct, "vbuf")
        - _buf_bytes(cp_128k_paged, "kvbuf")
        - _buf_bytes(cp_128k_paged, "knew_o")
        - _buf_bytes(cp_128k_paged, "vnew_o")
    )
    assert cp_128k_paged.vmem_bytes == cp_4k_paged.vmem_bytes
    assert (
        cp_4k_direct.vmem_bytes - cp_128k_paged.vmem_bytes
        == expected_direct_minus_paged
    )
    assert cp_128k_paged.smem_shapes["page_table"] == ((64, 1024), jnp.int32)
    # SMEM = page_table (64 x 1024 i32) + per-item KV page list `wl_kp`
    # (NWL * BLK / page_size i32, added in M5) + small worklist/MoE tables.
    pt_bytes = 64 * 1024 * 4
    nwl = cp_128k_paged.smem_shapes["wl_b"][0][0]
    wl_kp_bytes = nwl * (2048 // 128) * 4
    assert cp_128k_paged.smem_shapes["wl_kp"] == ((nwl * 16,), jnp.int32)
    assert cp_128k_paged.smem_bytes == sum(
        int(np.prod(sh)) * jnp.dtype(dt).itemsize
        for sh, dt in cp_128k_paged.smem_shapes.values()
    )
    other = cp_128k_paged.smem_bytes - pt_bytes - wl_kp_bytes
    assert 0 < other < 64 * 1024, other

    # SMEM budget check raises MegakernelConfigError when page_table exceeds
    # tiles.smem_mib (1 MiB)
    ex_smem_overflow = mk.ExecSpec(
        R=8,
        B=64,
        T=1,
        ctx=131072,
        tiles=t64,
        cache=mk.CacheExec(paged_kv=True, num_pages=65536, max_pages_per_seq=65536),
    )
    with pytest.raises(mk.MegakernelConfigError, match="exceeds tiles.smem_mib budget"):
        mk.vmem.build_scratch(spec_full, ex_smem_overflow)

    # Verify eff_ctx = max(ctx, max_pages_per_seq * page_size) does not truncate > ctx
    # including when max_pages_per_seq % P_BLK != 0 (max_pages_per_seq=3, P_BLK=2)
    mesh = _cpu_mesh(2)
    g_tiny = _tiny_geom(L=4, att_every=4)
    ex_512 = _tiny_exec(R=2, B=2, T=1, ctx=512, paged_kv=True, num_pages=64)
    ex_eff = _tiny_exec(
        R=2,
        B=2,
        T=1,
        ctx=256,
        paged_kv=True,
        num_pages=64,
        max_pages_per_seq=3,  # eff_ctx = 3 * 128 = 384 > ctx = 256; 3 % P_BLK(2) == 1
    )
    _, hid_l, w_l, st_512 = qwen35.random_logical_inputs(
        jax.random.key(55), g_tiny, ex_512
    )
    # Sequence 0 is at pos=320 (> ex_eff.ctx=256, inside page 2 of eff_ctx=384)
    pos = jnp.asarray([320, 140], dtype=jnp.int32)
    ref_512, _ = qwen35.reference_step(
        g_tiny, ex_512, pos, hid_l, w_l, st_512, paged_kv=True
    )
    st_trunc = {
        **st_512,
        "kc": st_512["kc"][:, :, :, :256, :],
        "vc": st_512["vc"][:, :, :, :256, :],
    }
    ref_trunc, _ = qwen35.reference_step(
        g_tiny, ex_eff, jnp.minimum(pos, 255), hid_l, w_l, st_trunc, paged_kv=True
    )
    assert not jnp.allclose(
        ref_512[0].astype(jnp.float32), ref_trunc[0].astype(jnp.float32), atol=1e-3
    )

    raw_pools, pt_full = _pack_kv_to_raw_pools(
        g_tiny, ex_eff, st_512["kc"], st_512["vc"]
    )
    pt = pt_full[:, :3]
    assert pt.shape == (2, 3)

    with pytest.raises(ValueError, match="requires 'raw' and 'page_table'"):
        qwen35.pack_inputs(pos, hid_l, w_l, st_512, g_tiny, ex_eff, mesh=mesh)
    with pytest.raises(ValueError, match="page_table has shape"):
        qwen35.pack_inputs(
            pos,
            hid_l,
            w_l,
            {**st_512, "raw": raw_pools, "page_table": pt[:, :2]},
            g_tiny,
            ex_eff,
            mesh=mesh,
        )

    _, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos,
        hid_l,
        w_l,
        {**st_512, "raw": raw_pools, "page_table": pt},
        g_tiny,
        ex_eff,
        mesh=mesh,
    )
    step, info = qwen35.make_step(mesh, g_tiny, ex_eff)
    assert info["smem_shapes"]["page_table"] == ((2, 3), "int32")
    with pytest.raises(ValueError, match="SMEM state 'page_table' has shape"):
        step(pos, hid_mk, w_mk, {**st_mk, "page_table": jnp.zeros((2, 2), jnp.int32)})
    out_hid, _ = step(pos, hid_mk, w_mk, st_mk)
    out_b = mk.unfold(out_hid, ex_eff.R)[0, : ex_eff.B]
    assert jnp.allclose(
        out_b.astype(jnp.float32), ref_512.astype(jnp.float32), atol=3e-2, rtol=3e-2
    )


@pytest.mark.parametrize("T,B", [(1, 2), (2, 4)])
def test_pallas_writer_pooled_step_cpu_interpret(T: int, B: int):
    """Verify `PallasWriter`, `make_flat_pooled_step`, and scattered states."""
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))
    g = _tiny_geom(L=4, att_every=4)
    ex = _tiny_exec(R=2, B=B, T=T, ctx=256, paged_kv=True, num_pages=96)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.key(20 + T), g, ex)
    st_l = dict(
        st_l,
        conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32),
    )

    ref_out, ref_st = qwen35.reference_step(
        g, ex, pos, hid_l, w_l, st_l, paged_kv=True, pool_io=True
    )

    _, hid_mk, w_mk, st_mk_paged = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, min_np = mk.default_pool_layout(g, ex)
    assert mk.cache_exec(ex).num_pages >= min_np
    raw_pools = tuple(
        jax.device_put(r, tp) for r in mk.states_to_pool(g, ex, st_l, pt, ckpt, roff)
    )

    mk_paged_step, _ = qwen35.make_step(mesh, g, ex, donate_states=False)
    _, mk_st = mk_paged_step(pos, hid_mk, w_mk, st_mk_paged)

    spec = qwen35.qwen35_spec(g)
    pooled_step, info = mk.PallasWriter().make_step(mesh, spec, ex, donate=False)
    out_hid, raw_out = pooled_step(pos, hid_mk, w_mk, raw_pools, pt, ckpt, roff)
    out_b = mk.unfold(out_hid, ex.R)[0, :B]

    ex_direct = _tiny_exec(R=2, B=B, T=T, ctx=256, paged_kv=False)
    st_quant = dict(
        st_l,
        kc=jnp.clip(st_l["kc"], -448.0, 448.0)
        .astype(jnp.float8_e4m3fn)
        .astype(jnp.bfloat16),
        vc=jnp.clip(st_l["vc"], -448.0, 448.0)
        .astype(jnp.float8_e4m3fn)
        .astype(jnp.bfloat16),
        conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32),
    )
    _, _, _, st_mk_q = qwen35.pack_inputs(
        pos, hid_l, w_l, st_quant, g, ex_direct, mesh=mesh
    )
    direct_step, _ = mk.DirectHBM().make_step(mesh, spec, ex_direct, donate=False)
    dir_hid, _ = direct_step(pos, hid_mk, w_mk, st_mk_q)
    assert jnp.allclose(
        out_hid.astype(jnp.float32), dir_hid.astype(jnp.float32), atol=2e-2, rtol=2e-2
    )
    assert jnp.allclose(
        out_b.astype(jnp.float32), ref_out.astype(jnp.float32), atol=3e-2, rtol=3e-2
    )

    st_after = mk.pool_to_states(g, ex, raw_out, pt, ckpt)
    mk_knew = mk.unfold(mk_st["knew"], ex.R)
    mk_vnew = mk.unfold(mk_st["vnew"], ex.R)
    for b in range(B):
        s_idx, _ = divmod(b, T)
        p_b = int(pos[b])
        np.testing.assert_array_equal(
            np.asarray(st_after["kc"][:, :, s_idx, p_b, :]).view(np.uint8),
            np.asarray(mk_knew[:, :, b, :]).view(np.uint8),
        )
        np.testing.assert_array_equal(
            np.asarray(st_after["vc"][:, :, s_idx, p_b, :]).view(np.uint8),
            np.asarray(mk_vnew[:, :, b, :]).view(np.uint8),
        )

    # Verify BOTH rec AND conv states written by scatter_pool_states
    QKV = g.QKV(ex.R)
    if T == 1:
        mk_rec = mk.unfold(mk_st["rec"], ex.R)
        np.testing.assert_array_equal(np.asarray(st_after["rec"]), np.asarray(mk_rec))
        assert jnp.allclose(
            st_after["rec"].astype(jnp.float32),
            ref_st["rec"].astype(jnp.float32),
            atol=3e-2,
            rtol=3e-2,
        )
        mk_conv = (
            mk.unfold(mk_st["conv"], ex.R)[:, :, : 3 * B, :]
            .reshape(ex.R, g.NGDN, 3, B, QKV)
            .swapaxes(2, 3)
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        np.testing.assert_array_equal(
            np.asarray(st_after["conv"][:, :, :, 0]), np.asarray(mk_conv)
        )
        ref_conv = (
            ref_st["conv"][:, :, : 3 * B, :]
            .reshape(ex.R, g.NGDN, 3, B, QKV)
            .swapaxes(2, 3)
        )
        assert jnp.allclose(
            st_after["conv"][:, :, :, 0], ref_conv, atol=3e-2, rtol=3e-2
        )
    else:
        mk_rec_ckpt = mk.unfold(mk_st["rec_ckpt"], ex.R)
        np.testing.assert_array_equal(
            np.asarray(st_after["rec"]), np.asarray(mk_rec_ckpt)
        )
        assert jnp.allclose(
            st_after["rec"].astype(jnp.float32),
            ref_st["rec_ckpt"].astype(jnp.float32),
            atol=3e-2,
            rtol=3e-2,
        )
        mk_conv_ckpt = mk.unfold(mk_st["conv_ckpt"], ex.R)
        for t_idx in range(T):
            exp_taps = (
                mk_conv_ckpt[:, :, :, 1 + t_idx : 4 + t_idx, :]
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
            np.testing.assert_array_equal(
                np.asarray(st_after["conv"][:, :, :, t_idx]), np.asarray(exp_taps)
            )

    # Verify PallasWriter (states_dict path) and make_flat_pooled_step match bit-exact
    out_pw, st_dict_out = pooled_step(
        pos,
        hid_mk,
        w_mk,
        {
            "raw": raw_pools,
            "page_table": pt,
            "ckpt_indices": ckpt,
            "read_offsets": roff,
        },
    )
    np.testing.assert_array_equal(np.asarray(out_pw), np.asarray(out_hid))
    for r_a, r_b in zip(st_dict_out["raw"], raw_out):
        np.testing.assert_array_equal(
            np.asarray(r_a).view(np.uint8), np.asarray(r_b).view(np.uint8)
        )

    # Verify make_flat_pooled_step for both T=1 and T=2 (VER=True)
    flat_step, _, _, _ = mk.make_flat_pooled_step(mesh, spec, ex, donate_pools=False)
    w_flat = [w_mk[k] for k in info["plan"].w_names]
    flat_res = flat_step(pos, hid_mk, *w_flat, *raw_pools, pt, ckpt, roff)
    np.testing.assert_array_equal(np.asarray(flat_res[0]), np.asarray(out_hid))
    for r_a, r_b in zip(flat_res[1:], raw_out):
        np.testing.assert_array_equal(
            np.asarray(r_a).view(np.uint8), np.asarray(r_b).view(np.uint8)
        )


@pytest.mark.parametrize("NS,att_every", [(1, 4), (2, 4), (3, 3), (4, 4), (5, 4)])
def test_scatter_pool_states_multi_ns_and_from_ckpt(NS: int, att_every: int):
    """Verify `scatter_pool_states` across `NS=1..5`, `att_every`, and `from_ckpt`."""
    L = 4 if att_every == 4 else 6
    g = _tiny_geom(L=L, att_every=att_every)
    ex = _tiny_exec(R=2, B=NS, T=1, ctx=256, paged_kv=True, num_pages=128)
    pos, _, _, st_l = qwen35.random_logical_inputs(
        jax.random.key(30 + NS + att_every), g, ex
    )
    pt, ckpt, roff, min_np = mk.default_pool_layout(g, ex)
    assert mk.cache_exec(ex).num_pages >= min_np
    ckpt_use = ckpt[0] if att_every == 3 else ckpt
    roff_use = roff[0] if att_every == 3 else roff
    raw_pools = mk.states_to_pool(g, ex, st_l, pt, ckpt_use, roff_use)
    raw_r0 = tuple(mk.unfold(r, ex.R)[0] for r in raw_pools)

    k1, k2, k3, k4 = jax.random.split(jax.random.key(90 + NS), 4)
    knew = jax.random.normal(k1, (g.NATT, ex.MR, g.HD), dtype=jnp.float32).astype(
        jnp.float8_e4m3fn
    )
    vnew = jax.random.normal(k2, (g.NATT, ex.MR, g.HD), dtype=jnp.float32).astype(
        jnp.float8_e4m3fn
    )
    QKV = g.QKV(ex.R)
    vh = g.vh(ex.R)
    CR = mk.rup(3 * NS, 8)
    conv_new = (
        jax.random.normal(k3, (g.NGDN, CR, QKV), dtype=jnp.float32)
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )
    rec_new = jax.random.normal(
        k4, (g.NGDN, NS * vh, g.KD, g.VD), dtype=jnp.float32
    ).astype(jnp.bfloat16)

    raw_sc = mk.scatter_pool_states(
        g, ex, raw_r0, pos, pt, knew, vnew, conv_new, rec_new, ckpt_use
    )
    raw_sc_full = tuple(jnp.tile(r, (ex.R, 1, 1, 1, 1)) for r in raw_sc)
    st_after = mk.pool_to_states(g, ex, raw_sc_full, pt, ckpt_use)
    np.testing.assert_array_equal(np.asarray(st_after["rec"][0]), np.asarray(rec_new))
    exp_conv = conv_new[:, : 3 * NS, :].reshape(g.NGDN, 3, NS, QKV).swapaxes(1, 2)
    np.testing.assert_array_equal(
        np.asarray(st_after["conv"][0, :, :, 0]), np.asarray(exp_conv)
    )

    if NS >= 2:
        # Test inactive sequence 1 via explicit mask AND via "from_ckpt" (default)
        act_mask = jnp.asarray([1, 0] + [1] * (NS - 2), dtype=jnp.int32)
        raw_act = mk.scatter_pool_states(
            g,
            ex,
            raw_r0,
            pos,
            pt,
            knew,
            vnew,
            conv_new,
            rec_new,
            ckpt_use,
            act_mask,
        )
        ckpt_inact = (
            ckpt_use.at[:, 1, :].set(0)
            if ckpt_use.ndim == 3
            else ckpt_use.at[1, :].set(0)
        )
        raw_fc = mk.scatter_pool_states(
            g,
            ex,
            raw_r0,
            pos,
            pt,
            knew,
            vnew,
            conv_new,
            rec_new,
            ckpt_inact,
            "from_ckpt",
        )
        raw_default = mk.scatter_pool_states(
            g,
            ex,
            raw_r0,
            pos,
            pt,
            knew,
            vnew,
            conv_new,
            rec_new,
            ckpt_inact,
            None,
        )
        for r_a, r_b, r_d in zip(raw_act, raw_fc, raw_default):
            np.testing.assert_array_equal(
                np.asarray(r_a).view(np.uint8), np.asarray(r_b).view(np.uint8)
            )
            np.testing.assert_array_equal(
                np.asarray(r_a).view(np.uint8), np.asarray(r_d).view(np.uint8)
            )
        if NS == 2 and att_every == 4:
            from jax.experimental import pallas as pl

            with (
                pl.enable_debug_checks(True),
                pytest.raises(
                    jax.errors.JaxRuntimeError,
                    match="GDN scatter checkpoint slot maps outside",
                ),
            ):
                r_bad = jax.jit(
                    lambda ck: mk.scatter_pool_states(
                        g,
                        ex,
                        raw_r0,
                        pos,
                        pt,
                        knew,
                        vnew,
                        conv_new,
                        rec_new,
                        ck,
                    )
                )(ckpt_use + mk.cache_exec(ex).num_pages)
                r_bad[0].block_until_ready()


def test_scatter_pool_states_top_of_pool_t2_no_corruption():
    """Verify T>1 per-token and contiguous scatter at top of pool."""
    g = _tiny_geom(L=2, att_every=2)  # G_PER_ATT=1 -> 5 pages per slot
    # With num_pages=64 and T=2: (num_pages - 5*T, num_pages - 5] is (54, 59].
    # Slot 11 has page base 11*5 = 55 (pages 55..59), right in (54, 59]!
    # Neighboring slot 10 has page base 10*5 = 50 (pages 50..54).
    ex = _tiny_exec(R=2, B=4, T=2, ctx=256, paged_kv=True, num_pages=64)
    pos, _, _, st_l = qwen35.random_logical_inputs(jax.random.key(77), g, ex)
    pt, _, roff, _ = mk.default_pool_layout(g, ex)
    # 2D ckpt_indices: seq 0 uses slots (8, 9), seq 1 uses slots (10, 11)
    # where slot 11 -> pages 55..59 (in (num_pages - 5*T, num_pages - 5]).
    ckpt_2d = jnp.asarray([[8, 9], [10, 11]], dtype=jnp.int32)
    raw_pools = mk.states_to_pool(g, ex, st_l, pt, ckpt_2d, roff[0])
    raw_r0 = tuple(mk.unfold(r, ex.R)[0] for r in raw_pools)

    k1, k2, k3, k4 = jax.random.split(jax.random.key(78), 4)
    knew = jax.random.normal(k1, (g.NATT, ex.MR, g.HD), dtype=jnp.float32).astype(
        jnp.float8_e4m3fn
    )
    vnew = jax.random.normal(k2, (g.NATT, ex.MR, g.HD), dtype=jnp.float32).astype(
        jnp.float8_e4m3fn
    )
    QKV = g.QKV(ex.R)
    vh = g.vh(ex.R)
    conv_new = (
        jax.random.normal(k3, (g.NGDN, 2, 3 + ex.T, QKV), dtype=jnp.float32)
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )
    rec_new = jax.random.normal(
        k4, (g.NGDN, ex.NS * ex.T * vh, g.KD, g.VD), dtype=jnp.float32
    ).astype(jnp.bfloat16)

    def _assert_rec_and_conv(st_after: dict):
        np.testing.assert_array_equal(
            np.asarray(st_after["rec"][0]), np.asarray(rec_new)
        )
        for t_idx in range(ex.T):
            exp_taps = (
                conv_new[:, :, 1 + t_idx : 4 + t_idx, :]
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
            np.testing.assert_array_equal(
                np.asarray(st_after["conv"][0, :, :, t_idx]), np.asarray(exp_taps)
            )

    # 1. Test 2D ckpt_indices (always takes per-token branch)
    raw_sc_2d = mk.scatter_pool_states(
        g, ex, raw_r0, pos, pt, knew, vnew, conv_new, rec_new, ckpt_2d
    )
    raw_sc_2d_full = tuple(jnp.tile(r, (ex.R, 1, 1, 1, 1)) for r in raw_sc_2d)
    st_after_2d = mk.pool_to_states(g, ex, raw_sc_2d_full, pt, ckpt_2d)
    _assert_rec_and_conv(st_after_2d)

    # 2. Test 3D non-contiguous ckpt_indices with slot 11 at t=0 and slot 10 at t=1
    ckpt_3d_noncontig = jnp.asarray([[[8, 9], [11, 10]]], dtype=jnp.int32)
    raw_sc_3d = mk.scatter_pool_states(
        g, ex, raw_r0, pos, pt, knew, vnew, conv_new, rec_new, ckpt_3d_noncontig
    )
    raw_sc_3d_full = tuple(jnp.tile(r, (ex.R, 1, 1, 1, 1)) for r in raw_sc_3d)
    st_after_3d = mk.pool_to_states(g, ex, raw_sc_3d_full, pt, ckpt_3d_noncontig)
    _assert_rec_and_conv(st_after_3d)

    # 3. Test 3D contiguous ckpt_indices at top of pool (slots [10, 11] -> pages 50..59,
    # exercising span = 5*T = 10 at num_pages - 5*T = 54 without corrupting [8, 9])
    ckpt_3d_contig_top = jnp.asarray([[[8, 9], [10, 11]]], dtype=jnp.int32)
    raw_sc_3d_contig = mk.scatter_pool_states(
        g, ex, raw_r0, pos, pt, knew, vnew, conv_new, rec_new, ckpt_3d_contig_top
    )
    raw_sc_3d_contig_full = tuple(
        jnp.tile(r, (ex.R, 1, 1, 1, 1)) for r in raw_sc_3d_contig
    )
    st_after_3d_contig = mk.pool_to_states(
        g, ex, raw_sc_3d_contig_full, pt, ckpt_3d_contig_top
    )
    _assert_rec_and_conv(st_after_3d_contig)

    # 4. Verify out-of-bounds ckpt slot (slot 12 -> pages 60..64) raises ValueError
    with pytest.raises(ValueError, match="exceeds pool capacity"):
        mk.scatter_pool_states(
            g,
            ex,
            raw_r0,
            pos,
            pt,
            knew,
            vnew,
            conv_new,
            rec_new,
            jnp.asarray([[8, 9], [10, 12]], dtype=jnp.int32),
        )

    # 5. Also trace non-interpret (TPU Ref.bitcast().reshape() path) via make_jaxpr
    ex_tpu_trace = replace(ex, interpret=False)
    jaxpr_sc = jax.make_jaxpr(
        lambda r0, p, pt_i, kn, vn, cn, rn, ck: mk.scatter_pool_states(
            g, ex_tpu_trace, r0, p, pt_i, kn, vn, cn, rn, ck
        )
    )(raw_r0, pos, pt, knew, vnew, conv_new, rec_new, ckpt_2d)
    assert "qwen3_5_scatter_pool_states" in str(jaxpr_sc)


def test_donate_pools_all_call_shapes_no_pool_copy():
    """Verify `donate_pools=True` donates `raw_pools` across all 4 call shapes."""
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))

    # 1-3: Mixed GDN+GQA model (states_dict, positional no-act, positional with act)
    g = _tiny_geom(L=4, att_every=4)
    ex = _tiny_exec(R=2, B=2, T=1, ctx=256, paged_kv=True, num_pages=96)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.key(61), g, ex)
    _, hid_mk, w_mk, _ = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    spec = qwen35.qwen35_spec(g)
    pw_step, _ = mk.PallasWriter().make_step(mesh, spec, ex, donate=True)

    def _fresh_pools(target_g=g):
        return tuple(
            jax.device_put(np.asarray(r).copy(), tp)
            for r in mk.states_to_pool(target_g, ex, st_l, pt, ckpt, roff)
        )

    # Shape 1: states_dict call (used by PallasWriter)
    r1 = _fresh_pools()
    st_d = {
        "raw": r1,
        "page_table": pt,
        "ckpt_indices": ckpt,
        "read_offsets": roff,
    }
    lowered_1 = pw_step.lower(pos, hid_mk, w_mk, st_d)
    assert "jax.buffer_donor = true" in lowered_1.as_text()
    compiled_hlo_1 = lowered_1.compile().as_text()
    ok_real, msg_real = mk.check_pooled_step_hlo(
        compiled_hlo_1, page_size=mk.cache_exec(ex).page_size, kvh=1, hd=g.HD
    )
    assert ok_real, msg_real
    out_1, _ = pw_step(pos, hid_mk, w_mk, st_d)
    assert all(buf.is_deleted() for buf in r1)

    # Verify 3D page_table (R, NS, W) at R=2 on PallasWriter
    r_3d = _fresh_pools()
    pt_3d = jax.device_put(
        jnp.broadcast_to(pt[None, ...], (ex.R, *pt.shape)),
        tp,
    )
    out_3d, _ = pw_step(pos, hid_mk, w_mk, r_3d, pt_3d, ckpt, roff)
    np.testing.assert_array_equal(np.asarray(out_3d), np.asarray(out_1))
    assert all(buf.is_deleted() for buf in r_3d)

    # Also trace gather_gdn_states with interpret=False via make_jaxpr
    ex_non_interp = replace(ex, interpret=False)
    raw_r0 = tuple(mk.unfold(r, ex.R)[0] for r in _fresh_pools())
    jaxpr_gather = jax.make_jaxpr(
        lambda r0, ck, ro: mk.gather_gdn_states(g, ex_non_interp, r0, ck, ro)
    )(raw_r0, ckpt, roff)
    assert "qwen3_5_gather_gdn_states" in str(jaxpr_gather)

    # Shape 2: GDN positional without active_seqs (reuses compiled _jitted_step)
    r2 = _fresh_pools()
    hlo_2 = pw_step.lower(pos, hid_mk, w_mk, r2, pt, ckpt, roff).as_text()
    assert "jax.buffer_donor = true" in hlo_2
    _, _ = pw_step(pos, hid_mk, w_mk, r2, pt, ckpt, roff)
    assert all(buf.is_deleted() for buf in r2)

    # Shape 3: GDN positional with active_seqs array
    r3 = _fresh_pools()
    act = jnp.asarray([1, 1], dtype=jnp.int32)
    hlo_3 = pw_step.lower(pos, hid_mk, w_mk, r3, pt, ckpt, roff, act).as_text()
    assert "jax.buffer_donor = true" in hlo_3
    _, _ = pw_step(pos, hid_mk, w_mk, r3, pt, ckpt, roff, act)
    assert all(buf.is_deleted() for buf in r3)

    # Shape 4: Attention-only (NGDN=0) positional call
    g_att = _tiny_geom(L=1, att_every=1)
    _, hid_a, w_a, st_a = qwen35.random_logical_inputs(jax.random.key(62), g_att, ex)
    _, hid_mk_a, w_mk_a, _ = qwen35.pack_inputs(
        pos, hid_a, w_a, st_a, g_att, ex, mesh=mesh, build_default_pool=True
    )
    pt_a, ckpt_a, roff_a, _ = mk.default_pool_layout(g_att, ex)
    r4 = tuple(
        jax.device_put(np.asarray(r).copy(), tp)
        for r in mk.states_to_pool(g_att, ex, st_a, pt_a, ckpt_a, roff_a)
    )
    att_step, _ = qwen35.make_pooled_step(mesh, g_att, ex, donate_pools=True)
    hlo_4 = att_step.lower(pos, hid_mk_a, w_mk_a, r4, pt_a).as_text()
    assert "jax.buffer_donor = true" in hlo_4
    _, _ = att_step(pos, hid_mk_a, w_mk_a, r4, pt_a)
    assert all(buf.is_deleted() for buf in r4)

    # Also verify check_pooled_step_hlo negative cases
    bad_single_cc, _ = mk.check_pooled_step_hlo("tpu_custom_call scatter_pool_states")
    assert not bad_single_cc
    bad_copy, _ = mk.check_pooled_step_hlo(
        "tpu_custom_call tpu_custom_call scatter_pool_states "
        "%c = f8e4m3fn[64,2,64,4,128] copy(%p)"
    )
    assert not bad_copy


@pytest.mark.parametrize("T,B", [(1, 2), (2, 4)])
def test_unstacked_expert_weights_equivalence(T: int, B: int):
    """Verify `stack_e=False` (per-layer tuples, 3D/5D, flat step) is bit-exact."""
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))
    g = _tiny_geom(L=4, att_every=4)
    ex_stacked = _tiny_exec(
        R=2, B=B, T=T, ctx=256, paged_kv=True, num_pages=96, stack_e=True
    )
    ex_unstacked = replace(ex_stacked, stack_e=False)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.key(66 + T), g, ex_stacked
    )

    _, hid_mk, w_stacked, _ = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex_stacked, mesh=mesh, build_default_pool=True
    )
    # Pass per-layer expert tuples directly into pack_inputs without a stacked tensor
    w_l_per_layer = dict(
        w_l,
        e13=tuple(w_l["e13"][:, lyr] for lyr in range(g.L)),
        e2=tuple(w_l["e2"][:, lyr] for lyr in range(g.L)),
    )
    _, _, w_unstacked, _ = qwen35.pack_inputs(
        pos,
        hid_l,
        w_l_per_layer,
        st_l,
        g,
        ex_unstacked,
        mesh=mesh,
        build_default_pool=True,
    )
    assert isinstance(w_unstacked["e13"], tuple) and len(w_unstacked["e13"]) == g.L
    assert isinstance(w_unstacked["e2"], tuple) and len(w_unstacked["e2"]) == g.L

    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex_stacked)
    raw_pools = tuple(
        jax.device_put(r, tp)
        for r in mk.states_to_pool(g, ex_stacked, st_l, pt, ckpt, roff)
    )

    spec = qwen35.qwen35_spec(g)
    step_stacked, _ = mk.PallasWriter().make_step(mesh, spec, ex_stacked, donate=False)
    step_unstacked, info_u = mk.PallasWriter().make_step(
        mesh, spec, ex_unstacked, donate=False
    )
    out_s, raw_s = step_stacked(pos, hid_mk, w_stacked, raw_pools, pt, ckpt, roff)
    out_u, raw_u = step_unstacked(pos, hid_mk, w_unstacked, raw_pools, pt, ckpt, roff)

    np.testing.assert_array_equal(np.asarray(out_u), np.asarray(out_s))
    for r_u, r_s in zip(raw_u, raw_s):
        np.testing.assert_array_equal(
            np.asarray(r_u).view(np.uint8), np.asarray(r_s).view(np.uint8)
        )

    if T == 1:
        # Exercise 3D per-rank `(EW, K, N)` (global `(R * EW, K, N)`) reshape branch
        # and `make_flat_pooled_step` with `stack_e=False`.
        e13_3d = tuple(
            jax.device_put(mk.fold(w_l["e13"][:, lyr]), tp) for lyr in range(g.L)
        )
        e2_3d = tuple(
            jax.device_put(mk.fold(w_l["e2"][:, lyr]), tp) for lyr in range(g.L)
        )
        assert e13_3d[0].ndim == 3 and e2_3d[0].ndim == 3
        w_unstacked_3d = dict(w_unstacked, e13=e13_3d, e2=e2_3d)

        w_flat_u: list[jax.Array] = []
        for k in info_u["plan"].w_names:
            if k in ("e13", "e2"):
                w_flat_u.extend(w_unstacked_3d[k])
            else:
                w_flat_u.append(w_unstacked_3d[k])

        flat_u, _, _, _ = mk.make_flat_pooled_step(
            mesh, spec, ex_unstacked, donate_pools=False
        )
        flat_res = flat_u(pos, hid_mk, *w_flat_u, *raw_pools, pt, ckpt, roff)
        np.testing.assert_array_equal(np.asarray(flat_res[0]), np.asarray(out_s))
        for r_f, r_s in zip(flat_res[1:], raw_s):
            np.testing.assert_array_equal(
                np.asarray(r_f).view(np.uint8), np.asarray(r_s).view(np.uint8)
            )

        # Verify unpack_local_weight rejects transposed non-expert and expert tensors
        with pytest.raises(mk.MegakernelConfigError, match="local shard has shape"):
            mk.kernel.unpack_local_weight(
                jnp.zeros((1, 4, 32, 16, 128), dtype=jnp.bfloat16),
                (4, 16, 128, 32),
                name="router",
                allow_expert_reshape=False,
            )
        with pytest.raises(mk.MegakernelConfigError, match="local shard has shape"):
            mk.kernel.unpack_local_weight(
                jnp.zeros((8, 64, 256), dtype=jnp.float8_e4m3fn),
                (8, 2, 128, 64),
                name="e13",
                allow_expert_reshape=True,
            )


@pytest.mark.parametrize(
    "T,B",
    # T=2 re-checks the T=1 EPLB path (EPLB is also pinned by the golden harness).
    [(1, 2), pytest.param(2, 4, marks=pytest.mark.slow)],
)
def test_eplb_ecopy_stack_e_and_split_hf_weights(T: int, B: int):
    """Verify `split_hf_weights` (R=1 vs R=2), `encode_ecopy`, EPLB, and TP MoE."""
    # Verify encode_ecopy warns when an expert has > 1 secondary replicas
    with pytest.warns(UserWarning, match="multiple secondary replicas"):
        mk.encode_ecopy(
            E=6,
            ER=2,
            EW=3,
            moe_tp=False,
            cores=[[0, 1, 2], [2, 3, 0], [4, 5, 0]],
        )
    # Verify encode_ecopy warns when a redundant slot is placed on the same rank
    with pytest.warns(UserWarning, match="same rank as its primary slot"):
        mk.encode_ecopy(
            E=4,
            ER=2,
            EW=3,
            moe_tp=False,
            cores=[[0, 1, 0], [2, 3, 1]],
        )

    mesh = _cpu_mesh(2)
    g_ep = _tiny_geom(L=4, att_every=4, moe_tp=False, e_slots=0)
    g_eplb = _tiny_geom(L=4, att_every=4, moe_tp=False, e_slots=68)
    g_tp = _tiny_geom(L=4, att_every=4, moe_tp=True, e_slots=0)

    ex_base = _tiny_exec(R=2, B=B, T=T, ctx=256, paged_kv=False)
    ex_opt0_base = _tiny_exec(R=2, B=B, T=T, ctx=256, paged_kv=False, ecopy_opt=0)
    ex_opt0 = replace(
        ex_opt0_base,
        tiles=ex_opt0_base.resolved_tiles.with_blocks({"moe": {"route_vec": False}}),
    )
    ex_po2c_base = _tiny_exec(R=2, B=B, T=T, ctx=256, paged_kv=False, ecopy_opt=1)
    ex_po2c = replace(
        ex_po2c_base,
        tiles=ex_po2c_base.resolved_tiles.with_blocks({"moe": {"route_vec": False}}),
    )
    ex_vec = replace(
        ex_po2c_base,
        tiles=ex_po2c_base.resolved_tiles.with_blocks({"moe": {"route_vec": True}}),
    )
    ex_eplb_unstacked = replace(ex_vec, stack_e=False)

    key = jax.random.key(202 + T)
    k_hf, k_in = jax.random.split(key)
    hf_w = qwen35.random_hf_weights(k_hf, g_ep)

    # Bias router strongly toward experts 0 and 1 (replicated onto rank 1 slots 64..65
    # on layer 0) and replicate KV head 0 across both KV heads so R=1 (NKV=1) and R=2
    # (NKV=2) execute the exact same mathematical model.
    router_biased = hf_w["router"].at[:, :, :2].add(jnp.bfloat16(100.0))
    hf_w = dict(
        hf_w,
        router=router_biased,
        k_proj=jnp.repeat(hf_w["k_proj"][:, :1], 2, axis=1),
        k_proj_s=jnp.repeat(hf_w["k_proj_s"][:, :1], 2, axis=1),
        v_proj=jnp.repeat(hf_w["v_proj"][:, :1], 2, axis=1),
        v_proj_s=jnp.repeat(hf_w["v_proj_s"][:, :1], 2, axis=1),
        kn=jnp.repeat(hf_w["kn"][:, :1], 2, axis=1),
    )

    pos, hid_l, _, st_l = qwen35.random_logical_inputs(k_in, g_ep, ex_base)
    st_l = dict(
        st_l,
        kc=jnp.repeat(st_l["kc"][:1], 2, axis=0),
        vc=jnp.repeat(st_l["vc"][:1], 2, axis=0),
    )

    # Pass only unfused HF projection keys to verify real HF tensor splitting
    fused_keys = (
        "att_q",
        "att_q_s",
        "att_kv",
        "att_kv_s",
        "sh13",
        "sh13_s",
        "e13",
        "e13_s",
    )
    hf_unfused = {k: v for k, v in hf_w.items() if k not in fused_keys}
    w_ep_l = qwen35.split_hf_weights(hf_unfused, g_ep, ex_base)
    w_eplb_l = qwen35.split_hf_weights(hf_unfused, g_eplb, ex_base)
    w_tp_l = qwen35.split_hf_weights(hf_unfused, g_tp, ex_base)
    assert "ecopy" in w_eplb_l
    assert jnp.any(w_eplb_l["ecopy"] > 0)

    # Unsharded R=1 forward pass comparison for split_hf_weights
    g_r1 = replace(g_ep, NKV=1)
    ex_r1 = replace(ex_base, R=1)
    hf_r1 = dict(
        hf_unfused,
        k_proj=hf_unfused["k_proj"][:, :1],
        k_proj_s=hf_unfused["k_proj_s"][:, :1],
        v_proj=hf_unfused["v_proj"][:, :1],
        v_proj_s=hf_unfused["v_proj_s"][:, :1],
        kn=hf_unfused["kn"][:, :1],
    )
    w_r1_l = qwen35.split_hf_weights(hf_r1, g_r1, ex_r1)
    qk = (g_ep.KH // ex_base.R) * g_ep.KD
    st_r1 = {
        "kc": st_l["kc"][:1],
        "vc": st_l["vc"][:1],
        "conv": jnp.concatenate(
            [
                st_l["conv"][0, :, :, :, :qk],
                st_l["conv"][1, :, :, :, :qk],
                st_l["conv"][0, :, :, :, qk : 2 * qk],
                st_l["conv"][1, :, :, :, qk : 2 * qk],
                st_l["conv"][0, :, :, :, 2 * qk :],
                st_l["conv"][1, :, :, :, 2 * qk :],
            ],
            axis=-1,
        )[None, ...],
        "rec": jnp.concatenate([st_l["rec"][0], st_l["rec"][1]], axis=2)[None, ...],
    }
    ref_r1, _ = qwen35.reference_step(g_r1, ex_r1, pos, hid_l, w_r1_l, st_r1)

    # Directly verify GDN `lin_in` & `lin_in_s` layout against unsharded HF tensors
    lay_gdn = mk.GDNBlock.lin_in_layout(g_ep, ex_base)
    vh = lay_gdn.vh
    assert w_ep_l["lin_in"].shape == (ex_base.R, g_ep.NGDN, g_ep.H, lay_gdn.NLI)
    assert w_ep_l["lin_in_s"].shape == (ex_base.R, g_ep.NGDN, lay_gdn.NLI)
    exp_a = jnp.transpose(
        hf_unfused["lin_a"].reshape(g_ep.NGDN, g_ep.H, ex_base.R, vh),
        (2, 0, 1, 3),
    )
    exp_b = jnp.transpose(
        hf_unfused["lin_b"].reshape(g_ep.NGDN, g_ep.H, ex_base.R, vh),
        (2, 0, 1, 3),
    )
    np.testing.assert_array_equal(
        np.asarray(w_ep_l["lin_in"][..., lay_gdn.BA : lay_gdn.BA + vh]).view(np.uint8),
        np.asarray(exp_b).view(np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(w_ep_l["lin_in"][..., lay_gdn.A_OFF : lay_gdn.A_OFF + vh]).view(
            np.uint8
        ),
        np.asarray(exp_a).view(np.uint8),
    )
    assert jnp.all(
        w_ep_l["lin_in"][..., lay_gdn.BA + vh : lay_gdn.A_OFF].astype(jnp.float32) == 0
    )
    assert jnp.all(
        w_ep_l["lin_in"][..., lay_gdn.A_OFF + vh : lay_gdn.NLI].astype(jnp.float32) == 0
    )

    ref_ep, _ = qwen35.reference_step(g_ep, ex_base, pos, hid_l, w_ep_l, st_l)
    ref_tp, _ = qwen35.reference_step(g_tp, ex_base, pos, hid_l, w_tp_l, st_l)
    assert jnp.allclose(
        ref_ep.astype(jnp.float32), ref_r1.astype(jnp.float32), atol=1e-2, rtol=1e-2
    )

    def run_cfg(g_cfg, ex_cfg, w_logical):
        p_in, h_mk, w_mk, s_mk = qwen35.pack_inputs(
            pos, hid_l, w_logical, st_l, g_cfg, ex_cfg, mesh=mesh
        )
        step, _ = qwen35.make_step(mesh, g_cfg, ex_cfg)
        out_h, _ = step(p_in, h_mk, w_mk, s_mk)
        return mk.unfold(out_h, ex_cfg.R)[0, :B]

    out_ep = run_cfg(g_ep, ex_base, w_ep_l)
    out_eplb_opt0 = run_cfg(g_eplb, ex_opt0, w_eplb_l)
    out_eplb_opt1 = run_cfg(g_eplb, ex_po2c, w_eplb_l)
    out_eplb_vec = run_cfg(g_eplb, ex_vec, w_eplb_l)
    out_eplb_unstacked = run_cfg(g_eplb, ex_eplb_unstacked, w_eplb_l)
    out_tp = run_cfg(g_tp, ex_base, w_tp_l)

    # Prove that replica routing actually occurred: corrupting redundant slots [64:68]
    # changes the output of ecopy_opt=0, ecopy_opt=1 (scalar), and route_vec=True (VEC).
    er = g_eplb.ER(ex_base.R)
    w_eplb_zero_rep = dict(
        w_eplb_l,
        e2=w_eplb_l["e2"].at[:, :, er:].set(jnp.zeros_like(w_eplb_l["e2"][:, :, er:])),
        e2_s=w_eplb_l["e2_s"]
        .at[:, :, er:]
        .set(jnp.zeros_like(w_eplb_l["e2_s"][:, :, er:])),
    )
    for ex_opt, ref_opt_out in (
        (ex_opt0, out_eplb_opt0),
        (ex_po2c, out_eplb_opt1),
        (ex_vec, out_eplb_vec),
    ):
        out_eplb_zero_rep = run_cfg(g_eplb, ex_opt, w_eplb_zero_rep)
        assert not jnp.allclose(
            out_eplb_zero_rep.astype(jnp.float32),
            ref_opt_out.astype(jnp.float32),
            atol=1e-3,
        )

    np.testing.assert_array_equal(
        np.asarray(out_eplb_unstacked), np.asarray(out_eplb_vec)
    )
    for out_cand in (out_eplb_opt0, out_eplb_opt1, out_eplb_vec):
        assert jnp.allclose(
            out_cand.astype(jnp.float32),
            out_ep.astype(jnp.float32),
            atol=1.6e-2,
            rtol=1.6e-2,
        )

    if T == 1:
        # Also test randomized routing across all experts (no router bias) comparing
        # route_vec=True vs route_vec=False (ecopy_opt=0 and ecopy_opt=1)
        hf_rand = qwen35.random_hf_weights(jax.random.key(299), g_ep)
        hf_rand_unfused = {k: v for k, v in hf_rand.items() if k not in fused_keys}
        w_eplb_rand = qwen35.split_hf_weights(hf_rand_unfused, g_eplb, ex_base)
        out_rand_opt0 = run_cfg(g_eplb, ex_opt0, w_eplb_rand)
        out_rand_opt1 = run_cfg(g_eplb, ex_po2c, w_eplb_rand)
        out_rand_vec = run_cfg(g_eplb, ex_vec, w_eplb_rand)
        assert jnp.allclose(
            out_rand_vec.astype(jnp.float32),
            out_rand_opt0.astype(jnp.float32),
            atol=1.6e-2,
            rtol=1.6e-2,
        )
        assert jnp.allclose(
            out_rand_vec.astype(jnp.float32),
            out_rand_opt1.astype(jnp.float32),
            atol=1.6e-2,
            rtol=1.6e-2,
        )

    assert jnp.allclose(
        out_ep.astype(jnp.float32), ref_ep.astype(jnp.float32), atol=3e-2, rtol=3e-2
    )
    assert jnp.allclose(
        out_ep.astype(jnp.float32), ref_r1.astype(jnp.float32), atol=3e-2, rtol=3e-2
    )
    assert jnp.allclose(
        out_tp.astype(jnp.float32), ref_tp.astype(jnp.float32), atol=3e-2, rtol=3e-2
    )
    tp_ep_diff = float(
        jnp.max(jnp.abs(out_tp.astype(jnp.float32) - out_ep.astype(jnp.float32)))
    )
    assert tp_ep_diff <= 1.6e-2, f"TP vs EP max diff too large: {tp_ep_diff}"


import sys  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from frozen_utils import DEFAULT_FROZEN_DIR, load_frozen_qwen35  # noqa: E402


@pytest.mark.skipif(
    not os.path.isdir(os.environ.get("MK_FROZEN_QWEN35_DIR", DEFAULT_FROZEN_DIR)),
    reason="Set MK_FROZEN_QWEN35_DIR to canonical c196d2500 frozen qwen35 path",
)
@pytest.mark.parametrize("T,B", [(1, 2)])
def test_m3_bit_exact_vs_frozen_pool_io_and_eplb(T: int, B: int):
    """Bit-exact check vs canonical `c196d2500` frozen `pool_io` and `R=2` EPLB."""
    frozen_mk, frozen_pio = load_frozen_qwen35()

    # 1. Pooled step (`pool_io.make_pooled_step`) bit-exact comparison
    mesh1 = _cpu_mesh(1)
    tp1 = NamedSharding(mesh1, P("tp"))

    g_pool = replace(_tiny_geom(L=4, att_every=4, e_slots=0), NKV=1)
    # Library defaults (no flag overrides) must be bit-exact vs frozen.
    ex_pool = _tiny_exec(
        R=1, B=B, T=T, ctx=256, paged_kv=True, num_pages=96, stack_e=False
    )
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.key(303 + T), g_pool, ex_pool
    )
    _, hid_mk, w_lib, _ = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g_pool, ex_pool, mesh=mesh1, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g_pool, ex_pool)
    raw_pools = tuple(
        jax.device_put(r, tp1)
        for r in mk.states_to_pool(g_pool, ex_pool, st_l, pt, ckpt, roff)
    )

    step_lib, info_lib = qwen35.make_pooled_step(
        mesh1, g_pool, ex_pool, donate_pools=False
    )
    out_hid_lib, raw_out_lib = step_lib(pos, hid_mk, w_lib, raw_pools, pt, ckpt, roff)

    tiles_f = replace(
        frozen_mk.TINY_TILES,
        bank=512 << 10,
        kv_blk=min(frozen_mk.TINY_TILES.kv_blk, 256),
    )
    c_f = frozen_mk.MKConfig(
        R=1,
        B=B,
        T=T,
        L=4,
        att_every=4,
        H=256,
        NH=4,
        NKV=1,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        e_slots=0,
        ctx=256,
        paged_kv=True,
        num_pages=96,
        max_pages_per_seq=2,
        stack_e=False,
    )
    f_lay, f_S = frozen_mk.smalls_layout(c_f)
    l_lay = info_lib["plan"].smalls_layout
    sm_f = jnp.zeros((ex_pool.R, g_pool.L, 1, f_S), dtype=jnp.float32)
    for name, (lo, w) in l_lay.items():
        fo, _ = f_lay[name]
        sm_f = sm_f.at[:, :, 0, fo : fo + w].set(w_lib["smalls"][:, :, 0, lo : lo + w])

    w_f = {**w_lib, "smalls": jax.device_put(sm_f, tp1)}
    step_f, _ = frozen_pio.make_pooled_step(
        mesh1, c_f, tiles_f, interpret=True, fused_pool_io=False
    )
    out_hid_f, raw_out_f = jax.jit(step_f)(pos, hid_mk, w_f, raw_pools, pt, ckpt, roff)

    np.testing.assert_array_equal(np.asarray(out_hid_lib), np.asarray(out_hid_f))
    for r_lib, r_f in zip(raw_out_lib, raw_out_f):
        np.testing.assert_array_equal(
            np.asarray(r_lib).view(np.uint8),
            np.asarray(r_f).view(np.uint8),
        )

    # 2. Multi-rank (R=2) EPLB bit-exact comparison with non-zero ecopy
    mesh2 = _cpu_mesh(2)
    tp2 = NamedSharding(mesh2, P("tp"))
    g_eplb = _tiny_geom(L=4, att_every=4, moe_tp=False, e_slots=68)
    ex_eplb = _tiny_exec(
        R=2, B=B, T=T, ctx=256, paged_kv=False, ecopy_opt=1, stack_e=False
    )
    pos_e, hid_le, w_le, st_le = qwen35.random_logical_inputs(
        jax.random.key(404 + T), g_eplb, ex_eplb
    )
    assert "ecopy" in w_le and jnp.any(w_le["ecopy"] > 0)
    # Bias router strongly toward experts 0 and 1 (replicated on rank 1 slots 64..65)
    # so Po2C deterministically routes tokens to secondary replicas.
    w_le = dict(
        w_le,
        router=w_le["router"].at[:, :, :2].add(jnp.bfloat16(100.0)),
    )
    _, hid_mke, w_libe, st_mke = qwen35.pack_inputs(
        pos_e, hid_le, w_le, st_le, g_eplb, ex_eplb, mesh=mesh2
    )

    step_lib_e, info_lib_e = qwen35.make_step(mesh2, g_eplb, ex_eplb)
    out_hid_lib_e, out_st_lib_e = step_lib_e(pos_e, hid_mke, w_libe, st_mke)

    # Assert that secondary replica slots were actually exercised
    er_e = g_eplb.ER(ex_eplb.R)
    w_le_zero_rep = dict(
        w_le,
        e2=w_le["e2"].at[:, :, er_e:].set(jnp.zeros_like(w_le["e2"][:, :, er_e:])),
    )
    _, _, w_libe_zero, _ = qwen35.pack_inputs(
        pos_e, hid_le, w_le_zero_rep, st_le, g_eplb, ex_eplb, mesh=mesh2
    )
    out_hid_zero_rep, _ = step_lib_e(pos_e, hid_mke, w_libe_zero, st_mke)
    assert not jnp.allclose(
        out_hid_zero_rep.astype(jnp.float32),
        out_hid_lib_e.astype(jnp.float32),
        atol=1e-3,
    )

    c_fe = replace(c_f, R=2, NKV=2, e_slots=68, paged_kv=False)
    tiles_fe = replace(tiles_f, ecopy_opt=1)
    f_lay_e, f_Se = frozen_mk.smalls_layout(c_fe)
    l_lay_e = info_lib_e["plan"].smalls_layout
    sm_fe = jnp.zeros((ex_eplb.R, g_eplb.L, 1, f_Se), dtype=jnp.float32)
    for name, (lo, w) in l_lay_e.items():
        fo, _ = f_lay_e[name]
        sm_fe = sm_fe.at[:, :, 0, fo : fo + w].set(
            w_libe["smalls"][:, :, 0, lo : lo + w]
        )
    w_fe = {**w_libe, "smalls": jax.device_put(sm_fe, tp2)}
    step_fe, _ = frozen_mk.make_step(mesh2, c_fe, tiles_fe, interpret=True)
    out_hid_fe, out_st_fe = jax.jit(step_fe)(pos_e, hid_mke, w_fe, st_mke)

    np.testing.assert_array_equal(
        np.asarray(out_hid_lib_e).view(np.uint8),
        np.asarray(out_hid_fe).view(np.uint8),
    )
    conv_key = "conv" if T == 1 else "conv_ckpt"
    rec_key = "rec" if T == 1 else "rec_ckpt"
    np.testing.assert_array_equal(
        np.asarray(out_st_lib_e[conv_key]).view(np.uint8),
        np.asarray(out_st_fe["conv"]).view(np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(out_st_lib_e[rec_key]).view(np.uint8),
        np.asarray(out_st_fe["rec"]).view(np.uint8),
    )
    for k in ("kc", "vc"):
        np.testing.assert_array_equal(
            np.asarray(out_st_lib_e[k]).view(np.uint8),
            np.asarray(out_st_fe[k]).view(np.uint8),
        )


@pytest.mark.parametrize("T,B", [(1, 20), (2, 20)])
def test_gdn_gch_non_multiple_of_8_tail_and_padded_mr(T: int, B: int):
    """Verify GCH chunked recurrence when `B < MR` and tail chunk `TR % 8 != 0`."""
    mesh = _cpu_mesh(2)
    g = _tiny_geom(L=4, att_every=4)
    # B=20 -> MR=24 (B < MR), WIDE=True, gdn_rows=8 -> CB=8, NF=2, TR=4 (n % 8 != 0)
    ex_base = _tiny_exec(R=2, B=B, T=T, ctx=256, paged_kv=False)
    ex_gch = replace(
        ex_base,
        tiles=ex_base.resolved_tiles.with_blocks(
            {"gdn": {"gdn_rows": 8, "gdn_bufs": 2}}
        ),
    )
    ex_noch = replace(
        ex_base, tiles=ex_base.resolved_tiles.with_blocks({"gdn": {"gdn_rows": 0}})
    )
    assert ex_gch.B < ex_gch.MR

    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.key(601 + T), g, ex_gch
    )
    ref_out, _ = qwen35.reference_step(g, ex_gch, pos, hid_l, w_l, st_l)

    pos_in, hid_gch, w_gch, st_gch = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex_gch, mesh=mesh
    )
    _, hid_noch, w_noch, st_noch = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex_noch, mesh=mesh
    )
    step_gch, info_gch = qwen35.make_step(mesh, g, ex_gch, donate_states=False)
    gp = info_gch["plan"].block_params["gdn"]
    assert gp.GCH and gp.CB == 8 and gp.NF == 2 and gp.TR == 4
    step_noch, _ = qwen35.make_step(mesh, g, ex_noch, donate_states=False)
    out_gch, st_out_gch = step_gch(pos_in, hid_gch, w_gch, st_gch)
    out_noch, st_out_noch = step_noch(pos_in, hid_noch, w_noch, st_noch)

    np.testing.assert_array_equal(
        np.asarray(mk.unfold(out_gch, 2)[:, :B]).view(np.uint8),
        np.asarray(mk.unfold(out_noch, 2)[:, :B]).view(np.uint8),
    )
    conv_key = "conv" if T == 1 else "conv_ckpt"
    rec_key = "rec" if T == 1 else "rec_ckpt"
    np.testing.assert_array_equal(
        np.asarray(st_out_gch[conv_key]).view(np.uint8),
        np.asarray(st_out_noch[conv_key]).view(np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(st_out_gch[rec_key]).view(np.uint8),
        np.asarray(st_out_noch[rec_key]).view(np.uint8),
    )
    out_r0 = mk.unfold(out_gch, 2)[0, :B]
    assert jnp.allclose(
        out_r0.astype(jnp.float32), ref_out.astype(jnp.float32), atol=2e-2, rtol=2e-2
    )


@pytest.mark.parametrize("natt", [1, 2, 3])
@pytest.mark.parametrize("T,B", [(1, 3), (2, 4)])
def test_scatter_pool_states_odd_and_even_natt(natt: int, T: int, B: int):
    """Verify `scatter_pool_states` with odd (`1, 3`) and even (`2`) `NATT`."""
    g = _tiny_geom(L=4 * natt, att_every=4)
    assert natt == g.NATT
    ex = _tiny_exec(R=2, B=B, T=T, ctx=256, paged_kv=True, num_pages=160)
    pos, _, _, st_l = qwen35.random_logical_inputs(
        jax.random.key(700 + 10 * natt + T), g, ex
    )
    # Exercise both same-page and cross-page token positions when T > 1
    if T > 1:
        pos = jnp.asarray([10, 11, 127, 128][:B], dtype=jnp.int32)
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    raw_init = mk.states_to_pool(g, ex, st_l, pt, ckpt, roff)

    k_k, k_v, k_c, k_r = jax.random.split(jax.random.key(750 + natt), 4)
    knew_l = (
        jax.random.normal(k_k, (ex.R, g.NATT, ex.MR, g.HD), dtype=jnp.float32) * 2.0
    ).astype(mk.ops.FP8)
    vnew_l = (
        jax.random.normal(k_v, (ex.R, g.NATT, ex.MR, g.HD), dtype=jnp.float32) * 2.0
    ).astype(mk.ops.FP8)
    lay = mk.GDNBlock.lin_in_layout(g, ex)
    hr = mk.ops.rup(3 + T, 8)
    cr = mk.ops.rup(3 * B, 8)
    conv_new = (
        jax.random.normal(
            k_c,
            (ex.R, g.NGDN, cr, lay.QKV)
            if T == 1
            else (ex.R, g.NGDN, ex.NS, hr, lay.QKV),
            dtype=jnp.float32,
        )
        * 0.2
    )
    rec_new = jax.random.normal(
        k_r, (ex.R, g.NGDN, B * lay.vh, lay.KD, lay.VD), dtype=jnp.bfloat16
    ) * jnp.bfloat16(0.2)

    # Scatter per rank and unpack via pool_to_states
    raw_updated = []
    for r in range(ex.R):
        pools_r = [
            raw_init[g_i].reshape(
                ex.R, mk.cache_exec(ex).num_pages, 2, 64, 4, mk.cache_exec(ex).page_size
            )[r]
            for g_i in range(g.NATT)
        ]
        out_r = mk.scatter_pool_states(
            g,
            ex,
            pools_r,
            pos,
            pt,
            knew_l[r],
            vnew_l[r],
            conv_new[r],
            rec_new[r],
            ckpt,
            interpret=True,
        )
        raw_updated.append(out_r)
    raw_folded = tuple(
        jnp.stack([raw_updated[r][g_i] for r in range(ex.R)], axis=0).reshape(
            ex.R * mk.cache_exec(ex).num_pages, 2, 64, 4, mk.cache_exec(ex).page_size
        )
        for g_i in range(g.NATT)
    )
    unpacked = mk.pool_to_states(g, ex, raw_folded, pt, ckpt)

    for b in range(B):
        s_i, p_i = b // T, int(pos[b])
        np.testing.assert_array_equal(
            np.asarray(unpacked["kc"][:, :, s_i, p_i, :]).view(np.uint8),
            np.asarray(knew_l[:, :, b, :]).view(np.uint8),
        )
        np.testing.assert_array_equal(
            np.asarray(unpacked["vc"][:, :, s_i, p_i, :]).view(np.uint8),
            np.asarray(vnew_l[:, :, b, :]).view(np.uint8),
        )
    np.testing.assert_array_equal(
        np.asarray(unpacked["rec"]).view(np.uint8),
        np.asarray(rec_new).view(np.uint8),
    )


@pytest.mark.parametrize("T,B", [(1, 2), (2, 4)])
def test_gather_gdn_states_cold_start_negative_read_offsets(T: int, B: int):
    """Verify `read_offsets < 0` produces exact zero GDN states and no stale leak."""
    mesh = _cpu_mesh(2)
    # Use L=8, att_every=4 (NATT=2, G_PER_ATT=3, NGDN=6) so g_idx >= 1 and o >= 2
    # reuse double-buffered conv_buf/rec_buf slots.
    g = _tiny_geom(L=8, att_every=4)
    ex = _tiny_exec(R=2, B=B, T=T, ctx=256, paged_kv=True, num_pages=160)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.key(810 + T), g, ex)
    pt, ckpt, roff_zero, _ = mk.default_pool_layout(g, ex)
    raw_pools = mk.states_to_pool(g, ex, st_l, pt, ckpt, roff_zero)

    # Corrupt sequence 0's checkpoint pages in pool with 0x7f (NaN in BF16/F32)
    # while sequence 1 retains real state at roff=0.
    raw_corrupted = []
    for g_i in range(g.NATT):
        p_arr = (
            np.asarray(raw_pools[g_i])
            .reshape(
                ex.R, mk.cache_exec(ex).num_pages, 2, 64, 4, mk.cache_exec(ex).page_size
            )
            .copy()
        )
        for sub in range(g.att_every - 1):
            slot_s0 = int(ckpt[sub, 0, 0])
            p_arr[:, slot_s0 * 5 : (slot_s0 + 1) * 5].view(np.uint8)[:] = 0x7F
        raw_corrupted.append(
            jnp.asarray(
                p_arr.reshape(
                    ex.R * mk.cache_exec(ex).num_pages,
                    2,
                    64,
                    4,
                    mk.cache_exec(ex).page_size,
                )
            )
        )

    # Sequence 0 is cold-start (roff = -1), sequence 1 has real state (roff = 0)
    roff_mixed = jnp.asarray([[-1, 0], [-1, 0], [-1, 0]], dtype=jnp.int32)
    lay = mk.GDNBlock.lin_in_layout(g, ex)
    vh = lay.vh

    for r in range(ex.R):
        pools_r = [
            raw_corrupted[g_i].reshape(
                ex.R, mk.cache_exec(ex).num_pages, 2, 64, 4, mk.cache_exec(ex).page_size
            )[r]
            for g_i in range(g.NATT)
        ]
        conv_g, rec_g = mk.gather_gdn_states(
            g, ex, pools_r, ckpt, roff_mixed, interpret=True
        )
        assert jnp.all(jnp.isfinite(conv_g))
        assert jnp.all(jnp.isfinite(rec_g))
        # Sequence 0 must have exact zero rec and conv state across all 6 GDN layers
        np.testing.assert_array_equal(
            np.asarray(rec_g[:, 0:vh], dtype=np.float32),
            np.zeros((g.NGDN, vh, lay.KD, lay.VD), dtype=np.float32),
        )
        if T == 1:
            for tap in range(3):
                np.testing.assert_array_equal(
                    np.asarray(conv_g[:, tap * B + 0]),
                    np.zeros((g.NGDN, lay.QKV), dtype=np.float32),
                )
        else:
            np.testing.assert_array_equal(
                np.asarray(conv_g[:, 0]),
                np.zeros_like(np.asarray(conv_g[:, 0])),
            )
        # Sequence 1 must match the real state from raw_pools (roff=0)
        pools_r_clean = [
            raw_pools[g_i].reshape(
                ex.R, mk.cache_exec(ex).num_pages, 2, 64, 4, mk.cache_exec(ex).page_size
            )[r]
            for g_i in range(g.NATT)
        ]
        conv_ref, rec_ref = mk.gather_gdn_states(
            g, ex, pools_r_clean, ckpt, roff_zero, interpret=True
        )
        np.testing.assert_array_equal(
            np.asarray(rec_g[:, vh : 2 * vh]).view(np.uint8),
            np.asarray(rec_ref[:, vh : 2 * vh]).view(np.uint8),
        )
        if T == 1:
            for tap in range(3):
                np.testing.assert_array_equal(
                    np.asarray(conv_g[:, tap * B + 1]),
                    np.asarray(conv_ref[:, tap * B + 1]),
                )
        else:
            np.testing.assert_array_equal(
                np.asarray(conv_g[:, 1]),
                np.asarray(conv_ref[:, 1]),
            )

    if T == 1:
        # Also test non-uniform roff across subs when coal_rec=False (NS * vh > 128)
        # so rec_buf[o & 1] is shared across o and o - 2:
        # sub=0 (o=0) has roff=0 (non-zero state), sub=2 (o=2, same slot 0) has roff=-1.
        g_uncoal = replace(g, KH=4, VH=16)  # vh = 8 -> with NS=17, NS * vh = 136 > 128
        ex_uncoal = _tiny_exec(R=2, B=17, T=1, ctx=256, paged_kv=True, num_pages=400)
        _, _, _, st_u = qwen35.random_logical_inputs(
            jax.random.key(820), g_uncoal, ex_uncoal
        )
        pt_u, ckpt_u, roff_u0, _ = mk.default_pool_layout(g_uncoal, ex_uncoal)
        raw_u = mk.states_to_pool(g_uncoal, ex_uncoal, st_u, pt_u, ckpt_u, roff_u0)
        roff_np = np.zeros((3, 17), dtype=np.int32)
        roff_np[2, 0] = -1  # sub=2 (o=2, reusing slot 0) has roff=-1 for seq 0
        roff_non_unif = jnp.asarray(roff_np)
        vh_u = g_uncoal.VH // ex_uncoal.R
        assert ex_uncoal.NS * vh_u > 128  # coal_rec is False
        pools_u0 = [
            raw_u[g_i].reshape(
                ex_uncoal.R,
                mk.cache_exec(ex_uncoal).num_pages,
                2,
                64,
                4,
                mk.cache_exec(ex_uncoal).page_size,
            )[0]
            for g_i in range(g_uncoal.NATT)
        ]
        _, rec_u = mk.gather_gdn_states(
            g_uncoal, ex_uncoal, pools_u0, ckpt_u, roff_non_unif, interpret=True
        )
        assert jnp.any(rec_u[0, 0:vh_u] != 0)
        np.testing.assert_array_equal(
            np.asarray(rec_u[2, 0:vh_u], dtype=np.float32),
            np.zeros((vh_u, g_uncoal.KD, g_uncoal.VD), dtype=np.float32),
        )

    # End-to-end make_pooled_step parity against reference_step with zeroed seq-0 state
    tp_s = NamedSharding(mesh, P("tp"))
    rep_s = NamedSharding(mesh, P())
    st_ref_mixed = {
        **st_l,
        "conv": st_l["conv"]
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
        .at[:, :, 0]
        .set(0.0),
        "rec": st_l["rec"].at[:, :, 0].set(jnp.bfloat16(0.0)),
    }
    ref_mixed, _ = qwen35.reference_step(
        g, ex, pos, hid_l, w_l, st_ref_mixed, paged_kv=True
    )
    _, hid_mk, w_mk, _ = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    raw_sharded = tuple(jax.device_put(r, tp_s) for r in raw_corrupted)
    step_pooled, _ = qwen35.make_pooled_step(mesh, g, ex, donate_pools=False)
    out_pooled, _ = step_pooled(
        jax.device_put(pos, rep_s),
        hid_mk,
        w_mk,
        raw_sharded,
        jax.device_put(pt, rep_s),
        jax.device_put(ckpt, rep_s),
        jax.device_put(roff_mixed, rep_s),
    )
    out_r0 = mk.unfold(out_pooled, ex.R)[0, :B]
    assert jnp.allclose(
        out_r0.astype(jnp.float32), ref_mixed.astype(jnp.float32), atol=2e-2, rtol=2e-2
    )


def _rec_page_bf16_bits(pages_u8: np.ndarray) -> np.ndarray:
    """Recover bf16 bit patterns from fp8-encoded GDN recurrent-state pages.

    Pages store u32 words (two bf16) as `bitcast(u32 -> 4 x fp8).swapaxes(1, 2)`,
    i.e. the 4 bytes of a word sit on the size-4 axis of `[..., 2, 64, 4, 128]`.
    """
    words = np.ascontiguousarray(np.moveaxis(pages_u8, -2, -1)).view(np.uint32)
    return words.view(np.uint16)


def _pool_page_kinds(
    pt: Any, ckpt: Any, *, g_per_att: int, pages_per_slot: int = 5, rec_pages: int = 4
) -> dict[int, str]:
    """Classify the pages referenced by a layout as `kv`, `rec` or `conv`.

    Uses the real unified-pool layout (`qwen35_pool`): KV pages come from the
    page table; GDN checkpoint slot `slot` of linear-attention group `sub` owns
    pages `base .. base + pages_per_slot - 1` with `base = slot * pages_per_slot`
    for 3D `ckpt_indices` and `slot * pages_per_slot * g_per_att + sub *
    pages_per_slot` for 2D, of which the first `rec_pages` hold recurrent state
    and the last one the conv taps.
    """
    kinds: dict[int, str] = {int(p): "kv" for p in np.asarray(pt).reshape(-1)}
    ck = np.asarray(ckpt)
    bases = []
    if ck.ndim == 3:
        bases = [int(v) * pages_per_slot for v in ck.reshape(-1)]
    else:
        for v in ck.reshape(-1):
            for sub in range(g_per_att):
                bases.append(int(v) * pages_per_slot * g_per_att + sub * pages_per_slot)
    for base in bases:
        for k in range(pages_per_slot):
            kind = "rec" if k < rec_pages else "conv"
            prev = kinds.setdefault(base + k, kind)
            assert prev == kind, f"page {base + k} is both {prev} and {kind}"
    return kinds


def _rec_pool_max_ulp(r_a: Any, r_b: Any, num_pages: int, kinds: dict[int, str]) -> int:
    """Max bf16 ULP distance between two pools; asserts all diffs are rec-state.

    `kinds` comes from `_pool_page_kinds` (the real layout). Every differing
    page must be a GDN recurrent-state page: KV pages, conv pages and
    unreferenced pages must match bytewise, and no recurrent-state value may
    flip sign.
    """
    a = np.asarray(r_a).view(np.uint8)
    b = np.asarray(r_b).view(np.uint8)
    worst = 0
    for gp_ in np.unique(np.argwhere(a != b)[:, 0]):
        local = int(gp_) % num_pages
        kind = kinds.get(local, "unreferenced")
        assert kind == "rec", f"{kind} page {local} differs"
        ba = _rec_page_bf16_bits(a[gp_]).astype(np.int32)
        bb = _rec_page_bf16_bits(b[gp_]).astype(np.int32)
        same_sign = (ba >> 15) == (bb >> 15)
        assert np.all(same_sign | (ba == bb)), f"sign flip in page {local}"
        worst = max(worst, int(np.max(np.abs(ba - bb))))
    return worst


def _pool_diff_summary(
    raw_a: Any, raw_b: Any, num_pages: int, kinds: dict[int, str]
) -> dict[str, Any]:
    """Per page kind: number of differing pages (+ max bf16 ULP for rec pages)."""
    out: dict[str, Any] = {}
    for r_a, r_b in zip(raw_a, raw_b):
        a = np.asarray(r_a).view(np.uint8)
        b = np.asarray(r_b).view(np.uint8)
        for gp_ in np.unique(np.argwhere(a != b)[:, 0]):
            kind = kinds.get(int(gp_) % num_pages, "unreferenced")
            n, ulp = out.get(kind, (0, 0))
            if kind == "rec":
                ba = _rec_page_bf16_bits(a[gp_]).astype(np.int32)
                bb = _rec_page_bf16_bits(b[gp_]).astype(np.int32)
                ulp = max(ulp, int(np.max(np.abs(ba - bb))))
            out[kind] = (n + 1, ulp)
    return out


def _hidden_row_ulp(out_a: Any, out_b: Any, R: int) -> np.ndarray:
    """Per-row max bf16 ULP distance between two hidden outputs."""
    a = np.asarray(mk.unfold(out_a, R)).view(np.uint16).astype(np.int32)
    b = np.asarray(mk.unfold(out_b, R)).view(np.uint16).astype(np.int32)
    return np.abs(a - b).max(axis=tuple(i for i in range(a.ndim) if i != 1))


def _assert_hidden_equal_per_row(
    out_a: Any, out_b: Any, R: int, pools: tuple | None = None
) -> None:
    """Hidden outputs bitwise equal; on mismatch report per-row max bf16 ULP.

    `pools=(raw_a, raw_b, num_pages, kinds)` adds a `_pool_diff_summary`.
    """
    d = _hidden_row_ulp(out_a, out_b, R)
    if d.any():
        extra = f"; pools: {_pool_diff_summary(*pools)}" if pools else ""
        raise AssertionError(f"hidden per-row max bf16 ULP: {d.tolist()}{extra}")


# Fused vs split at T > 1 on CPU interpret: for some seeds a few elements of the
# layer-0 GDN output epilogue differ by <= 7 bf16 ULP while the layer-0 GDN state
# is bitwise equal (not routing: forced routing and skipping experts keep it);
# later GDN layers amplify this. The same configs (test tiles seeds 0/2/4 and
# qwen35-candidate30-fused tiles B=8 seeds 0/2/4, B=24 seeds 0-2, T=4) are
# bitwise equal on TPU v7x for hidden and every pool page (see
# `test_fused_vs_split_t4_tpu_bitwise`, the real bitwise exactness check), so
# this is a CPU-interpret codegen artifact whose exact ULP magnitude depends on
# host CPU microarchitecture:
#   - Intel Xeon @ 2.20GHz: ("test", 4, 8, 2) -> 6 ULP, ("test", 4, 8, 4) -> 177,
#     ("c30f", 8, 0) -> 11, ("c30f", 8, 1) -> 0 (bitwise), ("c30f", 24, 0) -> 36
#   - AMD EPYC 7B12:        ("test", 4, 8, 2) -> 6 ULP, ("test", 4, 8, 4) -> 12,
#     ("c30f", 8, 0) -> 11, ("c30f", 8, 1) -> 2,           ("c30f", 24, 0) -> 36
# Every T > 1 CPU case uses one host-independent bound (256 bf16 ULP, ~1.4x the
# largest measured value across hosts); T = 1 stays strictly bitwise (0 ULP).
CPU_FUSED_T_GT_1_HIDDEN_ULP_BOUND = 256


def _xfail_on_cpu_fused_artifact(
    out_f: Any, out_s: Any, R: int, pools: tuple, case: tuple
) -> None:
    """Pass if hidden is bitwise; else XFAIL within the T>1 CPU bound."""
    d = _hidden_row_ulp(out_f, out_s, R)
    if not d.any():
        return
    bound = CPU_FUSED_T_GT_1_HIDDEN_ULP_BOUND
    msg = (
        f"hidden per-row max bf16 ULP {d.tolist()} (case {case}, bound {bound}); "
        f"pools: {_pool_diff_summary(*pools)}"
    )
    assert int(d.max()) <= bound, msg
    pytest.xfail(
        f"CPU-interpret fused T>1 artifact "
        f"(test_fused_vs_split_t4_tpu_bitwise checks TPU bitwise exactness): {msg}"
    )


def _assert_pools_equal_up_to_rec_ulp(
    r_a: Any, r_b: Any, num_pages: int, kinds: dict[int, str], *, max_ulp: int
) -> None:
    """Pools equal bytewise, except GDN rec-state pages within `max_ulp` bf16 ULP."""
    worst = _rec_pool_max_ulp(r_a, r_b, num_pages, kinds)
    assert worst <= max_ulp, (worst, max_ulp)


def test_pool_page_kinds_follow_layout():
    """`_pool_page_kinds` tells KV pages from GDN slot pages (not `page % 5`)."""
    g = _tiny_geom(L=4, att_every=4)
    ex = _tiny_exec(R=2, B=4, T=2, ctx=256, paged_kv=True, num_pages=512)
    pt, ckpt, _, _ = mk.default_pool_layout(g, ex)
    kinds = _pool_page_kinds(pt, ckpt, g_per_att=g.att_every - 1)
    kv = {p for p, k in kinds.items() if k == "kv"}
    rec = {p for p, k in kinds.items() if k == "rec"}
    conv = {p for p, k in kinds.items() if k == "conv"}
    assert kv and rec and conv and not (kv & rec) and not (kv & conv)
    # The old `page % 5 < 4` rule would call some KV pages recurrent state.
    assert any(p % 5 < 4 for p in kv)
    assert all(kinds[p + 4] == "conv" for p in rec if p % 5 == 0)


# Fused vs split on WIDE buckets (CPU interpret, B=56, T=1): recurrent-state
# pages measure 12 bf16 ULP on Intel Xeon @ 2.20GHz and 7 bf16 ULP on AMD EPYC
# 7B12 while hidden output is bitwise equal. Use one host-independent bound with
# margin above the largest measured value.
WIDE_FUSED_REC_ULP_BOUND = 24


@pytest.mark.parametrize(
    "T,B,seed",
    [
        (1, 2, 0),
        (2, 4, 0),
        (4, 8, 0),
        # Extra seeds: the T > 1 bounds below are empirical.
        (2, 4, 1),
        (2, 4, 2),
        (4, 8, 1),
        # CPU-interpret artifact (see CPU_FUSED_T_GT_1_HIDDEN_ULP_BOUND; bitwise
        # on TPU via test_fused_vs_split_t4_tpu_bitwise): seed 2 differs on
        # hidden rows 2-3 by up to 6 bf16 ULP, seed 4 on rows 3-7 by up to 177
        # ULP (12 ULP on AMD EPYC 7B12). Reported as bounded XFAIL with per-row
        # ULP and the per-kind pool diff.
        (4, 8, 2),
        (4, 8, 4),
        pytest.param(1, 56, 0, marks=pytest.mark.slow),
    ],
)
def test_fused_pool_io_bit_exact_and_hlo(T: int, B: int, seed: int):
    """Verify 1-launch `fused_pool_io=True` is bit-exact vs split `PallasWriter`."""
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))
    g = _tiny_geom(L=4, att_every=4)
    # B=56: read slots span 3 * 56 = 168 plus disjoint write slots -> 2048 pages.
    ex_base = _tiny_exec(
        R=2,
        B=B,
        T=T,
        ctx=256,
        paged_kv=True,
        num_pages=512 if B <= 8 else 2048,
        stack_e=False,
    )
    # The fused GDN path always runs the head-major recurrence, so the split
    # reference opts into `gdn_head_major` too. conv_stage=True is tested below.
    ex = mk.with_cache(
        replace(
            ex_base,
            tiles=ex_base.resolved_tiles.with_blocks(
                {"gdn": {"conv_stage": False, "gdn_head_major": True}}
            ),
        ),
        pool_gather=False,
    )
    spec = qwen35.qwen35_spec(g)

    if (T, B) == (1, 2):
        # Verify should_fuse_pool_io defaults to False, and auto bucket policy
        for b_chk, exp_fused in (
            (8, True),
            (24, True),
            (32, False),
            (40, False),
            (48, False),
            (56, True),
            (64, True),
        ):
            ex_b = replace(ex, B=b_chk)
            assert mk.should_fuse_pool_io(spec, ex_b) is False
            assert mk.should_fuse_pool_io(spec, ex_b, fused_pool_io="auto") is exp_fused
            assert mk.should_fuse_pool_io(spec, ex_b, fused_pool_io=True) is True
            assert mk.should_fuse_pool_io(spec, ex_b, fused_pool_io=False) is False

    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.key(710 + T * 10 + B + 1000 * seed), g, ex
    )
    if T > 1:
        # Exercise cross-page KV boundary (pos 127 -> page 0, pos 128 -> page 1)
        pos_list = []
        for s_i in range(ex.NS):
            base_p = 126 if s_i == 0 else (20 + 10 * s_i)
            pos_list.extend(base_p + t_i for t_i in range(T))
        pos = jnp.asarray(pos_list, dtype=jnp.int32)
    st_l = dict(
        st_l,
        conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32),
    )
    _, hid_mk, w_mk, _ = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    if T == 1:
        # Exercise 2-column [read_slot, write_slot] ckpt_indices with roff=-1 on
        # seq 1. Write slots must not alias any read slot (as in serving): the
        # fused path interleaves per-sequence reads and writes in one kernel.
        n_read = int(ckpt.max()) - int(ckpt.min()) + 1
        ckpt = jnp.concatenate([ckpt, ckpt + n_read + 4], axis=-1)
        roff = roff.at[:, 1].set(-1)

    def _fresh_pools():
        return tuple(
            jax.device_put(np.asarray(r).copy(), tp)
            for r in mk.states_to_pool(
                g,
                ex,
                st_l,
                pt,
                ckpt[..., :T],
                jnp.maximum(roff, 0),
            )
        )

    step_split, _ = mk.PallasWriter(fused_pool_io=False).make_step(
        mesh, spec, ex, donate=False
    )
    step_fused, _ = mk.PallasWriter(fused_pool_io=True).make_step(
        mesh, spec, ex, donate=True
    )

    pools_s = _fresh_pools()
    pools_f = _fresh_pools()
    hlo_f = step_fused.lower(pos, hid_mk, w_mk, pools_f, pt, ckpt, roff).as_text("hlo")
    ok_f, msg_f = mk.check_pooled_step_hlo(
        hlo_f, page_size=mk.cache_exec(ex).page_size, kvh=1, hd=g.HD, fused_pool_io=True
    )
    assert ok_f, msg_f

    out_s, raw_s = step_split(pos, hid_mk, w_mk, pools_s, pt, ckpt, roff)
    out_f, raw_f = step_fused(pos, hid_mk, w_mk, pools_f, pt, ckpt, roff)

    kinds = _pool_page_kinds(pt, ckpt, g_per_att=g.att_every - 1)
    pools_cmp = (raw_f, raw_s, mk.cache_exec(ex).num_pages, kinds)
    if T == 1:
        _assert_hidden_equal_per_row(out_f, out_s, ex.R, pools=pools_cmp)
    else:
        _xfail_on_cpu_fused_artifact(
            out_f, out_s, ex.R, pools_cmp, case=("test", T, B, seed)
        )
    rec_ulp = max(1, T - 1)
    if ex.WIDE:
        # Known open deviation (root cause under investigation): on WIDE buckets
        # recurrent-state pages differ from split by a few bf16 ULP on CPU
        # interpret, although hidden output is bitwise equal. Both sides run the
        # head-major recurrence (the split reference opts into gdn_head_major),
        # so this is not head-major vs key-major. Bounded at the measured value
        # and reported as XFAIL instead of widening a tolerance to fit.
        worst = max(
            _rec_pool_max_ulp(r_f, r_s, mk.cache_exec(ex).num_pages, kinds)
            for r_f, r_s in zip(raw_f, raw_s)
        )
        assert worst <= WIDE_FUSED_REC_ULP_BOUND, (worst, WIDE_FUSED_REC_ULP_BOUND)
        if worst:
            pytest.xfail(
                f"fused WIDE rec-state deviation vs split: {worst} bf16 ULP "
                f"(bound {WIDE_FUSED_REC_ULP_BOUND}, open)"
            )
    for r_f, r_s in zip(raw_f, raw_s):
        if T == 1:
            np.testing.assert_array_equal(
                np.asarray(r_f).view(np.uint8),
                np.asarray(r_s).view(np.uint8),
            )
        else:
            # Known deviation (open; empirical: seeds 0-2 at T=2, 0-1 at T=4):
            # for T > 1 a few bf16 GDN recurrent-state values of the t >= 1
            # checkpoints differ from split by up to T - 1 ULP on CPU interpret;
            # hidden output and KV / conv pages match exactly for these seeds.
            _assert_pools_equal_up_to_rec_ulp(
                r_f, r_s, mk.cache_exec(ex).num_pages, kinds, max_ulp=rec_ulp
            )

    if seed == 0 and (T, B) in ((1, 2), (2, 4)):
        # Verify conv_stage=True (conv_fast VMEM staging) on CPU interpret
        ex_cstage = mk.with_cache(
            replace(
                ex, tiles=ex.resolved_tiles.with_blocks({"gdn": {"conv_stage": True}})
            ),
            fused_pool_io=True,
        )
        step_cstage, _ = mk.PallasWriter(fused_pool_io=True).make_step(
            mesh, spec, ex_cstage, donate=True
        )
        out_cs, raw_cs = step_cstage(pos, hid_mk, w_mk, _fresh_pools(), pt, ckpt, roff)
        np.testing.assert_array_equal(
            np.asarray(out_cs).view(np.uint8),
            np.asarray(out_s).view(np.uint8),
        )
        # conv_stage only changes how the conv taps are staged (same bf16 values,
        # rows past `conv_live` zero), so it must match conv_stage=False fused
        # byte-for-byte, and split up to the known T > 1 rec-state ULP deviation.
        for r_cs, r_f0, r_s in zip(raw_cs, raw_f, raw_s):
            np.testing.assert_array_equal(
                np.asarray(r_cs).view(np.uint8),
                np.asarray(r_f0).view(np.uint8),
            )
            _assert_pools_equal_up_to_rec_ulp(
                r_cs, r_s, mk.cache_exec(ex).num_pages, kinds, max_ulp=rec_ulp
            )

    if (T, B) == (1, 2):
        # Verify act_mode="ref" with explicit active_seqs mask [1, 0]
        act_mask = jnp.asarray([1, 0], dtype=jnp.int32)
        ex_ref = mk.with_cache(ex, act_mode="ref")
        step_s_ref, _ = mk.PallasWriter(fused_pool_io=False).make_step(
            mesh, spec, ex_ref, donate=False
        )
        step_f_ref, _ = mk.PallasWriter(fused_pool_io=True).make_step(
            mesh, spec, ex_ref, donate=True
        )
        out_sr, raw_sr = step_s_ref(
            pos, hid_mk, w_mk, _fresh_pools(), pt, ckpt, roff, act_mask
        )
        out_fr, raw_fr = step_f_ref(
            pos, hid_mk, w_mk, _fresh_pools(), pt, ckpt, roff, act_mask
        )
        np.testing.assert_array_equal(
            np.asarray(out_fr).view(np.uint8),
            np.asarray(out_sr).view(np.uint8),
        )
        for r_fr, r_sr in zip(raw_fr, raw_sr):
            np.testing.assert_array_equal(
                np.asarray(r_fr).view(np.uint8),
                np.asarray(r_sr).view(np.uint8),
            )

        # Also trace non-interpret TPU path for fused_pool_io=True
        ex_tpu = mk.with_cache(
            replace(ex, interpret=False), fused_pool_io=True, ckpt_k=2
        )
        step_tpu, _ = mk.PallasWriter(fused_pool_io=True).make_step(
            mesh, spec, ex_tpu, donate=True
        )
        jaxpr_tpu = jax.make_jaxpr(
            lambda p, h, w, r, pt_i, ck_i, ro_i: step_tpu(p, h, w, r, pt_i, ck_i, ro_i)
        )(pos, hid_mk, w_mk, _fresh_pools(), pt, ckpt, roff)
        jaxpr_str = str(jaxpr_tpu)
        assert "decode_megakernel" in jaxpr_str
        assert "gather_gdn_states" not in jaxpr_str
        assert "scatter_pool_states" not in jaxpr_str


@pytest.mark.parametrize(
    "B,seed",
    [
        # CPU-interpret artifact (see CPU_FUSED_T_GT_1_HIDDEN_ULP_BOUND; bitwise
        # on TPU via test_fused_vs_split_t4_tpu_bitwise): B=8 seed 1 is bitwise
        # on Intel Xeon @ 2.20GHz and differs by 2 bf16 ULP on AMD EPYC 7B12;
        # B=8 seed 0 differs on hidden row 7 by 11 bf16 ULP; B=24 seed 0 differs
        # on rows 6-7 by 36 ULP. Any non-zero CPU diff within the bound is
        # reported as XFAIL.
        (8, 1),
        (8, 0),
        pytest.param(24, 0, marks=pytest.mark.slow),
    ],
)
def test_candidate30_fused_profile_vs_split(B: int, seed: int):
    """Fused vs split for the real `qwen35-candidate30-fused` exec config (T=4).

    The ExecSpec comes from `get_profile(...).exec_for(...)` (only ring / bank
    sizes shrunk to the CPU geometry), so it exercises the profile's own tile
    flags (`conv_stage`, `gdn_head_major`, `kv_sub`, ...). Same policy as
    `test_fused_pool_io_bit_exact_and_hlo`: hidden bitwise, pools bytewise
    except GDN rec-state pages within `T - 1` bf16 ULP (WIDE: the measured
    WIDE bound, reported as XFAIL).
    """
    T = 4
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))
    g = _tiny_geom(L=4, att_every=4)
    spec = qwen35.qwen35_spec(g)
    prof = mk.get_profile("qwen35-candidate30-fused")
    base = mk.ExecSpec(
        R=2,
        B=B,
        T=T,
        ctx=256,
        interpret=True,
        cache=mk.CacheExec(paged_kv=True, num_pages=512 if B <= 8 else 2048),
    )
    ex = prof.exec_for(base, B=B, T=T, g=g)
    t = ex.resolved_tiles
    t = replace(
        t,
        bank=512 << 10,
        nbd=10,
        nbe=max(6, min(t.nbe, 12)),
        ebank=256 << 10,
        vmem_mib=32,
    ).with_blocks(
        {"gqa": {"kv_blk": 256, "kv_sub": 128 if t.block("gqa").kv_sub else 0}}
    )
    ex = mk.with_cache(replace(ex, tiles=t), pool_gather=False)
    assert mk.should_fuse_pool_io(spec, ex, fused_pool_io=prof.fused_pool_io)
    assert t.block("gdn").conv_stage and t.block("gdn").gdn_head_major

    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.key(810 + B + 1000 * seed), g, ex
    )
    pos_list = []
    for s_i in range(ex.NS):
        base_p = 126 if s_i == 0 else (20 + 10 * s_i)
        pos_list.extend(base_p + t_i for t_i in range(T))
    pos = jnp.asarray(pos_list, dtype=jnp.int32)
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    _, hid_mk, w_mk, _ = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)

    def _fresh_pools():
        return tuple(
            jax.device_put(np.asarray(r).copy(), tp)
            for r in mk.states_to_pool(g, ex, st_l, pt, ckpt, roff)
        )

    step_s, _ = mk.PallasWriter(fused_pool_io=False).make_step(
        mesh, spec, ex, donate=False
    )
    step_f, _ = mk.PallasWriter(fused_pool_io=True).make_step(
        mesh, spec, ex, donate=True
    )
    out_s, raw_s = step_s(pos, hid_mk, w_mk, _fresh_pools(), pt, ckpt, roff)
    out_f, raw_f = step_f(pos, hid_mk, w_mk, _fresh_pools(), pt, ckpt, roff)
    kinds = _pool_page_kinds(pt, ckpt, g_per_att=g.att_every - 1)
    _xfail_on_cpu_fused_artifact(
        out_f,
        out_s,
        ex.R,
        (raw_f, raw_s, mk.cache_exec(ex).num_pages, kinds),
        case=("c30f", B, seed),
    )
    worst = max(
        _rec_pool_max_ulp(r_f, r_s, mk.cache_exec(ex).num_pages, kinds)
        for r_f, r_s in zip(raw_f, raw_s)
    )
    if ex.WIDE:
        assert worst <= WIDE_FUSED_REC_ULP_BOUND, (worst, WIDE_FUSED_REC_ULP_BOUND)
        if worst:
            pytest.xfail(f"fused WIDE rec-state deviation vs split: {worst} bf16 ULP")
    assert worst <= T - 1, (worst, T - 1)


def test_pool_gather_verify_wide_cpu_interpret():
    """Verify `pool_gather` (`GPR`), `KVE`, and `GQ` against split 3-stage reference."""
    R, B, T = 2, 24, 4
    mesh = _cpu_mesh(R)
    tp = NamedSharding(mesh, P("tp"))
    g = _tiny_geom(L=4, att_every=4)
    ex_gpr_base = _tiny_exec(R=R, B=B, T=T, ctx=256, paged_kv=True, num_pages=512)
    # KVE is opt-in (`kv_ering`); GPR is opt-in (`pool_gather=True`).
    ex_gpr = mk.with_cache(
        replace(
            ex_gpr_base,
            tiles=ex_gpr_base.resolved_tiles.with_blocks({"gqa": {"kv_ering": True}}),
        ),
        pool_gather=True,
    )
    ex_split = mk.with_cache(ex_gpr, pool_gather=False)
    spec = qwen35.qwen35_spec(g)
    step_gpr, info_gpr = mk.PallasWriter(fused_pool_io=False).make_step(
        mesh, spec, ex_gpr, donate=False
    )
    step_split, info_split = mk.PallasWriter(fused_pool_io=False).make_step(
        mesh, spec, ex_split, donate=False
    )
    assert info_gpr["plan"].block_params["gdn"].GPR is True
    assert info_split["plan"].block_params["gdn"].GPR is False
    # KVE is enabled in CPU interpret mode when kv_ering=True
    assert info_gpr["plan"].block_params["gqa"].KVE is True
    assert "kvbuf" not in info_gpr["vmem_shapes"]
    assert info_gpr["plan"].s_names == (
        "ckpt_indices",
        "read_offsets",
        "raw",
        "page_table",
    )
    assert info_gpr["plan"].o_names == (
        "hidden",
        "conv_ckpt",
        "rec_ckpt",
        "knew",
        "vnew",
    )
    assert info_gpr["oshape"]["conv_ckpt"][1] == jnp.bfloat16
    assert info_gpr["oshape"]["knew"][1] == jnp.float8_e4m3fn
    assert info_gpr["oshape"]["vnew"][1] == jnp.float8_e4m3fn

    # Trace non-interpret TPU path for GPR (megakernel + scatter, no separate gather)
    ex_tpu = replace(ex_gpr, interpret=False)
    step_gpr_tpu, _ = mk.PallasWriter(fused_pool_io=False).make_step(
        mesh, spec, ex_tpu, donate=True
    )
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.key(824), g, ex_gpr)
    st_l = dict(
        st_l,
        conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32),
    )
    _, hid_mk, w_mk, _ = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex_gpr, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex_gpr)

    def _fresh_pools(roff_init=roff):
        return tuple(
            jax.device_put(np.asarray(r).copy(), tp)
            for r in mk.states_to_pool(
                g, ex_gpr, st_l, pt, ckpt, jnp.maximum(roff_init, 0)
            )
        )

    pools = _fresh_pools()
    jaxpr_gpr = str(
        jax.make_jaxpr(
            lambda p, h, w, r, pt_i, ck_i, ro_i: step_gpr_tpu(
                p, h, w, r, pt_i, ck_i, ro_i
            )
        )(pos, hid_mk, w_mk, pools, pt, ckpt, roff)
    )
    assert "decode_megakernel" in jaxpr_gpr
    assert "gather_gdn_states" not in jaxpr_gpr
    assert "scatter_pool_states" in jaxpr_gpr

    out_gpr, raw_gpr = step_gpr(pos, hid_mk, w_mk, _fresh_pools(), pt, ckpt, roff)
    out_split, raw_split = step_split(pos, hid_mk, w_mk, _fresh_pools(), pt, ckpt, roff)
    np.testing.assert_array_equal(
        np.asarray(out_gpr).view(np.uint8),
        np.asarray(out_split).view(np.uint8),
    )
    for r_g, r_s in zip(raw_gpr, raw_split):
        np.testing.assert_array_equal(
            np.asarray(r_g).view(np.uint8),
            np.asarray(r_s).view(np.uint8),
        )

    # Verify GPR cold-start (read_offsets = -1 on sequence 0 with 0x7F NaN-corrupted
    # initial pool pages for sequence 0) matches split 3-stage bit-for-bit.
    roff_cold = roff.at[:, 0].set(-1)
    pools_cold_gpr = []
    pools_cold_split = []
    for r_arr in _fresh_pools():
        p_np = (
            np.asarray(r_arr)
            .reshape(
                R,
                mk.cache_exec(ex_gpr).num_pages,
                2,
                64,
                4,
                mk.cache_exec(ex_gpr).page_size,
            )
            .copy()
        )
        for sub in range(g.att_every - 1):
            s0_slot = int(ckpt[sub, 0, 0])
            p_np[:, s0_slot * 5 : (s0_slot + 1) * 5].view(np.uint8)[:] = 0x7F
        corrupted = p_np.reshape(
            R * mk.cache_exec(ex_gpr).num_pages,
            2,
            64,
            4,
            mk.cache_exec(ex_gpr).page_size,
        )
        pools_cold_gpr.append(jax.device_put(corrupted.copy(), tp))
        pools_cold_split.append(jax.device_put(corrupted.copy(), tp))
    out_gpr_cold, raw_gpr_cold = step_gpr(
        pos, hid_mk, w_mk, tuple(pools_cold_gpr), pt, ckpt, roff_cold
    )
    out_split_cold, raw_split_cold = step_split(
        pos, hid_mk, w_mk, tuple(pools_cold_split), pt, ckpt, roff_cold
    )
    assert jnp.all(jnp.isfinite(out_gpr_cold))
    np.testing.assert_array_equal(
        np.asarray(out_gpr_cold).view(np.uint8),
        np.asarray(out_split_cold).view(np.uint8),
    )
    for r_g, r_s in zip(raw_gpr_cold, raw_split_cold):
        np.testing.assert_array_equal(
            np.asarray(r_g).view(np.uint8),
            np.asarray(r_s).view(np.uint8),
        )

    # Verify KVE (kv_ering=True vs False) and GQ (gq_pack=True vs False) on
    # R=4, NKV=2, NH=16, T=4, B=8: G=2, nh=4 -> G*nh == QR == 8 and
    # QG = T*nh = 16 (QG % 16 == 0), NS=2, HD=256 -> the GQ gate holds.
    mesh4 = _cpu_mesh(4)
    g_gq = replace(
        _tiny_geom(L=4, att_every=4, NKV=2), NH=16, KH=4, VH=8, IS=512
    )  # IS=512: ISR = IS / R = 128
    ex_gq_on = _tiny_exec(
        R=4, B=8, T=4, ctx=256, paged_kv=True, num_pages=256, stack_e=False
    )
    ex_gq_on = mk.with_cache(
        replace(
            ex_gq_on,
            tiles=ex_gq_on.resolved_tiles.with_blocks(
                {
                    "gqa": {
                        "gqa_qv": True,
                        "gq_pack": True,
                        "kv_ering": True,
                        "split": True,
                    }
                }
            ),
        ),
        pool_gather=False,
    )
    ex_gq_off = replace(
        ex_gq_on,
        tiles=ex_gq_on.resolved_tiles.with_blocks(
            {"gqa": {"gqa_qv": False, "gq_pack": False, "kv_ering": False}}
        ),
    )
    ex_kve = replace(
        ex_gq_off,
        tiles=ex_gq_off.resolved_tiles.with_blocks({"gqa": {"kv_ering": True}}),
    )
    spec_gq = qwen35.qwen35_spec(g_gq)
    steps, infos = {}, {}
    for name, ex_v in (("on", ex_gq_on), ("off", ex_gq_off), ("kve", ex_kve)):
        steps[name], infos[name] = mk.PallasWriter(fused_pool_io=False).make_step(
            mesh4, spec_gq, ex_v, donate=False
        )
    ap = {k: v["plan"].block_params["gqa"] for k, v in infos.items()}
    assert (ap["on"].GQ, ap["on"].KVE) == (True, True)
    assert (ap["off"].GQ, ap["off"].KVE) == (False, False)
    assert (ap["kve"].GQ, ap["kve"].KVE) == (False, True)

    pos4, hid_l4, w_l4, st_l4 = qwen35.random_logical_inputs(
        jax.random.key(844), g_gq, ex_gq_on
    )
    _, hid_mk4, w_mk4, _ = qwen35.pack_inputs(
        pos4, hid_l4, w_l4, st_l4, g_gq, ex_gq_on, mesh=mesh4, build_default_pool=True
    )
    pt4, ckpt4, roff4, _ = mk.default_pool_layout(g_gq, ex_gq_on)
    tp4 = NamedSharding(mesh4, P("tp"))

    def _pools4():
        return tuple(
            jax.device_put(np.asarray(r).copy(), tp4)
            for r in mk.states_to_pool(g_gq, ex_gq_on, st_l4, pt4, ckpt4, roff4)
        )

    outs = {
        k: step(pos4, hid_mk4, w_mk4, _pools4(), pt4, ckpt4, roff4)
        for k, step in steps.items()
    }
    # KVE only changes where KV blocks are staged: numerics class `exact`.
    np.testing.assert_array_equal(
        np.asarray(outs["kve"][0]).view(np.uint8),
        np.asarray(outs["off"][0]).view(np.uint8),
    )
    for r_k, r_off in zip(outs["kve"][1], outs["off"][1]):
        np.testing.assert_array_equal(
            np.asarray(r_k).view(np.uint8), np.asarray(r_off).view(np.uint8)
        )
    # GQ packs G*nh query heads per MXU tile and merges partial softmax in a
    # different order: numerics class `approx` (a few bf16 ULP on CPU).
    np.testing.assert_allclose(
        np.asarray(outs["on"][0]).astype(np.float32),
        np.asarray(outs["off"][0]).astype(np.float32),
        atol=2e-2,
        rtol=2e-2,
    )
