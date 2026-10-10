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
"""CPU unit tests for Qwen3.5 megakernel bucket policy, per-bucket tiles, HLO fallback,
and wide verify bucket numerical parity (`B <= 24` fused, `32..48` 3-stage GPR, `56..64` wide fused)."""

from __future__ import annotations

import dataclasses
import importlib

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.qwen35_megakernel import megakernel as K
from vllm_torchtpu.kernels.qwen35_megakernel import pool_io
from vllm_torchtpu.kernels.qwen35_megakernel import vllm_adapter as VA

from qwen35_megakernel_test_utils import (
    bare_mk,
    unpack_conv_checkpoint,
    unpack_rec_checkpoint,
)

VERIFY_ENV = (
    "VLLM_TPU_QWEN35_MK_VERIFY_BUCKETS",
    "VLLM_TPU_QWEN35_MK_VERIFY_48",
    "VLLM_TPU_QWEN35_MK_VERIFY_64",
)
TILE_ENV = (
    "VLLM_TPU_QWEN35_MK_ROUTE_VEC",
    "VLLM_TPU_QWEN35_MK_DENSE_ACC",
    "VLLM_TPU_QWEN35_MK_R2_DIRECT",
    "VLLM_TPU_QWEN35_MK_AMASK",
    "VLLM_TPU_QWEN35_MK_ECOPY_OPT",
    "VLLM_TPU_QWEN35_MK_NBE_SMALL",
)
ALL_VERIFY = (4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64)

# The v21 per-bucket tiles at R=8: B -> (nbd, nbe in EP mode, nbe in TP mode, nkv, hier, rsag).
R8_TILES = {
    4: (14, 20, 10, 3, 1, False),
    8: (14, 20, 10, 3, 2, True),
    12: (14, 12, 10, 4, 2, True),
    16: (14, 12, 10, 4, 2, True),
    20: (12, 12, 10, 3, 2, True),
    24: (12, 12, 10, 3, 2, True),
    32: (10, 12, 10, 3, 2, True),
    40: (8, 12, 10, 3, 2, True),
    48: (8, 12, 10, 3, 2, True),
    56: (8, 12, 10, 3, 2, True),
    64: (7, 12, 10, 3, 2, True),
}


@pytest.fixture
def reload_adapter(monkeypatch):
    """Re-import vllm_adapter under the test's env; restore the default module afterwards."""
    for name in VERIFY_ENV:
        monkeypatch.delenv(name, raising=False)
    yield lambda: importlib.reload(VA)
    monkeypatch.undo()
    importlib.reload(VA)


def test_buckets_parsing_and_verify_buckets(reload_adapter, monkeypatch):
    mod = reload_adapter()
    assert mod._DEFAULT_VERIFY_BUCKETS == ALL_VERIFY
    assert mod.parse_mk_verify_buckets() == ALL_VERIFY
    assert mod.parse_mk_verify_buckets("4,8,64") == (4, 8, 64)
    assert mod.parse_mk_verify_buckets("0") == ()
    assert mod.parse_mk_buckets("8,16,32,64") == (8, 16, 32, 64)
    assert mod.parse_mk_buckets("0") == ()
    assert mod.select_bucket(12, (16, 32, 48, 64)) == 16
    assert mod.select_bucket(28, (16, 32, 48, 64)) == 32
    assert mod.select_bucket(44, (16, 32, 48, 64)) == 48
    assert mod.select_bucket(60, (16, 32, 48, 64)) == 64

    monkeypatch.setenv("VLLM_TPU_QWEN35_MK_VERIFY_64", "0")
    assert reload_adapter()._DEFAULT_VERIFY_BUCKETS == ALL_VERIFY[:9]
    monkeypatch.setenv("VLLM_TPU_QWEN35_MK_VERIFY_48", "0")
    assert reload_adapter()._DEFAULT_VERIFY_BUCKETS == ALL_VERIFY[:7]


@pytest.mark.parametrize("b", sorted(R8_TILES))
def test_default_tiles_r8_match_v21(b):
    nbd, nbe_ep, nbe_tp, nkv, hier, rsag = R8_TILES[b]
    for base, nbe in ((K.FULL_EP, nbe_ep), (K.FULL, nbe_tp)):
        t = K.default_tiles(dataclasses.replace(base, B=b, T=4))
        assert (t.nbd, t.nbe, t.nkv, t.hier, t.rsag) == (nbd, nbe, nkv, hier, rsag), b
        assert t.ecopy_opt == 1 and t.amask == 1 and t.kv_sub == 1024
        assert t.route_vec and t.dense_acc_pair and t.r2_direct_store
        assert t.kv_ering and t.pair_kv


def test_tp8_geometry_is_unpadded_and_tp16_scratch_fits():
    """R=8 keeps v21's checkpoint geometry and unpadded widths; at R=16 the B > 32 verify
    buckets (flat rsag all-reduce) need no more VMEM scratch than at R=8."""
    for base in (K.FULL_EP, K.FULL):
        c8 = dataclasses.replace(base, paged_kv=True)
        assert (c8.rec_pages, c8.gdn_pages) == (4, 5)
        assert c8.ISR == c8.IS // 8 and c8.IR == (c8.I // 8 if c8.moe_tp else c8.I)
    for b in (40, 64):
        for base in (K.FULL_EP, K.FULL):
            vm = {}
            for r in (8, 16):
                fused = b >= 56
                c = dataclasses.replace(
                    base, R=r, B=b, T=4, paged_kv=True, fused_pool_io=fused, ckpt_k=4
                )
                tiles = K.default_tiles(c)
                _, info = K.build(c, tiles, pool_gather=not fused)
                vm[r] = info["vmem_scratch_bytes"]
                assert vm[r] <= tiles.vmem_mib * 2**20, (r, b)
            assert vm[16] <= vm[8], (b, vm)


def test_fused_pool_io_bucket_selection_and_hlo_guard(monkeypatch):
    mk = bare_mk(monkeypatch)
    assert (mk.fused_pool_io_max_b, mk.fused_pool_io_min_b_hi) == (24, 56)
    assert [b for b in ALL_VERIFY if not mk._bucket_is_fused(b)] == [32, 40, 48]
    mk.fused_pool_io_min_b_hi = VA.parse_mk_fused_pool_io_min_b_hi("0")
    assert [b for b in ALL_VERIFY if mk._bucket_is_fused(b)] == [4, 8, 12, 16, 20, 24]
    mk.fused_pool_io_max_b = VA.parse_mk_fused_pool_io_max_b("32")
    assert [b for b in ALL_VERIFY if mk._bucket_is_fused(b)][-1] == 32
    assert VA.parse_mk_fused_pool_io(None) is True
    assert VA.parse_mk_fused_pool_io("1") is True
    assert VA.parse_mk_fused_pool_io("0") is False
    assert VA.parse_mk_fused_pool_io("off") is False
    assert VA.parse_mk_zero_copy(None) is False

    good_hlo = """
    HloModule fused_mk
    ENTRY main {
      %p0 = f8e4m3fn[4096,2,64,4,128]{4,3,2,1,0:T(8,128)} parameter(0)
      %call = bf16[16,4096]{1,0:T(8,128)S(1)} custom-call(%p0), custom_call_target="tpu_custom_call"
      ROOT %tup = (bf16[16,4096]{1,0:T(8,128)}, f8e4m3fn[4096,2,64,4,128]{4,3,2,1,0:T(8,128)}) tuple(%call, %p0)
    }
    """
    ok, reason = pool_io.check_fused_pool_hlo(good_hlo, page_size=128)
    assert ok is True and reason == "ok"

    bad_vmem_hlo = """
    HloModule bad_mk
    ENTRY main {
      %p0 = f8e4m3fn[64,2,64,4,128]{4,3,2,1,0:T(8,128)} parameter(0)
      %cs = (f8e4m3fn[64,2,64,4,128]{4,3,2,1,0:T(8,128)S(1)}, f8e4m3fn[64,2,64,4,128]{4,3,2,1,0:T(8,128)}, u32[]{:S(2)}) copy-start(%p0)
      %cd = f8e4m3fn[64,2,64,4,128]{4,3,2,1,0:T(8,128)S(1)} copy-done(%cs)
      %call = bf16[16,4096]{1,0:T(8,128)S(1)} custom-call(%cd), custom_call_target="tpu_custom_call"
      ROOT %tup = (bf16[16,4096]{1,0:T(8,128)}, f8e4m3fn[64,2,64,4,128]{4,3,2,1,0:T(8,128)}) tuple(%call, %p0)
    }
    """
    ok_bad, reason_bad = pool_io.check_fused_pool_hlo(bad_vmem_hlo, page_size=128)
    assert ok_bad is False and ("copy-start" in reason_bad or "S(1)" in reason_bad)


def test_tile_env_overrides(monkeypatch):
    for name in TILE_ENV:
        monkeypatch.delenv(name, raising=False)
    base = K.default_tiles(dataclasses.replace(K.FULL_EP, B=8))
    assert VA.apply_mk_tile_env_overrides(base, 8) == base
    monkeypatch.setenv("VLLM_TPU_QWEN35_MK_ROUTE_VEC", "0")
    monkeypatch.setenv("VLLM_TPU_QWEN35_MK_DENSE_ACC", "off")
    monkeypatch.setenv("VLLM_TPU_QWEN35_MK_AMASK", "2")
    monkeypatch.setenv("VLLM_TPU_QWEN35_MK_NBE_SMALL", "12")
    t8 = VA.apply_mk_tile_env_overrides(base, 8)
    assert (t8.route_vec, t8.dense_acc_pair, t8.r2_direct_store) == (False, False, True)
    assert (t8.amask, t8.nbe) == (2, 12)
    assert VA.apply_mk_tile_env_overrides(base, 12).nbe == base.nbe


def test_fused_pool_io_fallback_rebuilds_only_fused_buckets(monkeypatch):
    mk = bare_mk(monkeypatch, buckets=(8,), verify_buckets=(4, 32, 64))
    built = []

    def fake_build(b, tokens_per_seq=1, out_pad_rows=None):
        fused = bool(mk.fused_pool_io and mk._bucket_is_fused(b))
        mk._bucket_fused[(tokens_per_seq, b)] = fused
        built.append((tokens_per_seq, b, fused))
        return object()

    mk._build_bucket_op = fake_build
    mk._build_all_buckets()
    assert built == [(1, 8, True), (4, 4, True), (4, 32, False), (4, 64, True)]
    kept = mk._bucket_steps[(4, 32)]
    assert mk._bucket_steps[8] is mk._bucket_steps[(1, 8)]

    built.clear()
    mk.fused_pool_io = False
    mk._build_all_buckets(only_fused=True)
    assert built == [(1, 8, False), (4, 4, False), (4, 64, False)]
    assert mk._bucket_steps[(4, 32)] is kept
    assert not any(mk._bucket_fused.values())


def test_build_bucket_op_names_and_hlo_fallback(monkeypatch):
    """Op names carry `_3stage` exactly for 3-stage buckets, including a bucket that falls
    back from fused pool I/O when the HLO guard fails."""
    c = dataclasses.replace(
        K.FULL_EP,
        stack_e=False,
        e_slots=72,
        paged_kv=True,
        num_pages=4096,
        max_pages_per_seq=64,
        ctx=8192,
    )
    n_w, n_p = 20, c.NATT
    names, made = [], []

    def fake_step(mesh, cfg, tiles, fused_pool_io=None):
        made.append((cfg.B, bool(fused_pool_io), cfg.fused_pool_io))
        n_in = 2 + n_w + n_p + 3
        return (lambda *a: a), (P("tp"),) * n_in, (P("tp"),) * (1 + n_p), ()

    def fake_callable(name, fn, **kw):
        names.append(name)
        return fn

    monkeypatch.setattr(VA.pool_io, "make_flat_pooled_step", fake_step)
    monkeypatch.setattr(VA, "sharded_jax_callable", fake_callable)
    monkeypatch.setattr(torch, "compile", lambda f, **kw: f)
    for mode, hlo_ok in (("utok", True), ("utfb", False)):
        names.clear()
        made.clear()
        mk = bare_mk(
            monkeypatch,
            mode=mode,
            base_cfg=c,
            mesh=None,
            w_flat=[None] * n_w,
            raw_pools=[None] * n_p,
            max_pages_per_seq=c.max_pages_per_seq,
            zero_copy=False,
            _hlo_checked=False,
        )
        mk._verify_fused_pool_hlo = lambda *a, ok=hlo_ok: (ok, "forced")
        for b in (4, 32, 64):
            mk._build_bucket_op(b, tokens_per_seq=4, out_pad_rows=b)
        suffix = {b: n.endswith("_3stage") for b, n in zip((4, 32, 64), names)}
        if hlo_ok:
            assert suffix == {4: False, 32: True, 64: False}
            assert mk._bucket_fused == {(4, 4): True, (4, 32): False, (4, 64): True}
            assert mk.fused_pool_io
        else:
            assert suffix == {4: True, 32: True, 64: True}
            assert made[:2] == [(4, True, True), (4, False, False)]
            assert not any(mk._bucket_fused.values()) and not mk.fused_pool_io


def test_wide_verify_buckets_numerical_parity_and_tail_sequences():
    """Verify numerical parity across small fused (`B=16`), 3-stage GPR (`B=32, 40`), and wide
    fused (`B=64`) verify buckets for a multi-layer hybrid GDN + full-attention stack
    (`L=4, att_every=4, T=4`), including active sequences placed in tail slots `s >= 8`
    (`b = 32..63`) and mixed `read_offsets >= 0` / `read_offsets == -1`."""
    mesh = Mesh(np.array(jax.devices("cpu")[:1]), ("tp",))
    c_base = K.MKConfig(
        R=1,
        B=16,
        L=4,
        att_every=4,
        H=512,
        NH=4,
        NKV=1,
        HD=256,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=128,
        ctx=256,
        T=4,
        paged_kv=True,
        num_pages=384,
        page_size=128,
        max_pages_per_seq=2,
        ckpt_k=4,
    )
    tiles_base = dataclasses.replace(
        K.default_tiles(c_base),
        nbd=4,
        nbe=4,
        nkv=2,
        hier=0,
        rsag=False,
        kv_blk=256,
        kv_sub=128,
    )
    shapes = K.weight_shapes(c_base, tiles_base)
    rng = np.random.default_rng(42)
    w_leaves = [
        jnp.asarray(rng.standard_normal(s).astype(np.float32) * 0.02, dtype=dt)
        for k in K.W_NAMES
        for s, dt in (shapes[k],)
    ]

    raw_init_u8 = rng.integers(
        0, 40, size=(c_base.num_pages, 2, 64, 4, c_base.page_size), dtype=np.uint8
    )
    raw_init = jnp.asarray(raw_init_u8).view(K.FP8)

    n_act = 2
    act_tokens = rng.standard_normal((n_act * 4, c_base.H)).astype(np.float32) * 0.1
    act_roff = np.array([0, -1], dtype=np.int32)
    act_pos = np.array([[16, 17, 18, 19], [0, 1, 2, 3]], dtype=np.int32)
    act_pt = np.array([[1, 2], [3, 4]], dtype=np.int32)
    act_ckpt = np.zeros((3, n_act, 4), dtype=np.int32)
    slot = 16
    for sub in range(3):
        for s in range(n_act):
            for k in range(4):
                act_ckpt[sub, s, k] = slot
                slot += 1

    def run_bucket(b: int, place_at_tail: bool):
        fused = (b <= 24) or (b >= 56)
        c = dataclasses.replace(c_base, B=b, fused_pool_io=fused)
        tiles = dataclasses.replace(
            K.default_tiles(c),
            nbd=4,
            nbe=4,
            nkv=2,
            hier=0,
            rsag=False,
            kv_blk=256,
            kv_sub=128,
        )
        ns = b // 4
        s0 = (ns - n_act) if place_at_tail else 0
        x = np.zeros((b, c.H), dtype=np.float32)
        x[s0 * 4 : (s0 + n_act) * 4] = act_tokens
        pos = np.full((b,), -1, dtype=np.int32)
        pos[s0 * 4 : (s0 + n_act) * 4] = act_pos.reshape(-1)
        pt = np.zeros((ns, c.max_pages_per_seq), dtype=np.int32)
        pt[s0 : s0 + n_act] = act_pt
        ckpt = np.zeros((3, ns, 4), dtype=np.int32)
        ckpt[:, s0 : s0 + n_act, :] = act_ckpt
        roff = np.full((3, ns), -1, dtype=np.int32)
        roff[:, s0 : s0 + n_act] = act_roff[None, :]

        step, _, _, _ = pool_io.make_flat_pooled_step(
            mesh,
            c,
            tiles,
            interpret=True,
            skip=frozenset({"experts"}),
            fused_pool_io=fused,
        )
        outs = step(
            jnp.asarray(pos),
            jnp.asarray(x, dtype=K.BF16),
            *w_leaves,
            raw_init,
            jnp.asarray(pt),
            jnp.asarray(ckpt),
            jnp.asarray(roff),
        )
        out_act = np.asarray(outs[0][:b], dtype=np.float32)[s0 * 4 : (s0 + n_act) * 4]
        raw_u8 = np.asarray(outs[1].view(jnp.uint8))
        return out_act, raw_u8

    ref_out, ref_raw = run_bucket(16, place_at_tail=False)
    ref_32 = None
    kv_pages = act_pt[:, 0]

    for b, tail in ((32, False), (40, True), (64, True)):
        out, raw = run_bucket(b, place_at_tail=tail)
        fused = (b <= 24) or (b >= 56)
        assert np.all(np.isfinite(out)), f"Non-finite output in B={b}"
        np.testing.assert_array_equal(
            raw[kv_pages], ref_raw[kv_pages], err_msg=f"KV page mismatch at B={b}"
        )
        if fused:
            np.testing.assert_array_equal(
                out, ref_out, err_msg=f"Fused hidden mismatch at B={b} (tail={tail})"
            )
        else:
            if ref_32 is None:
                ref_32 = (out, raw)
            else:
                np.testing.assert_array_equal(
                    out, ref_32[0], err_msg=f"3-stage GPR hidden mismatch at B={b}"
                )
            np.testing.assert_allclose(out, ref_out, atol=2e-2, rtol=2e-2)

        for sub in range(3):
            for s in range(n_act):
                for t in range(4):
                    p0 = int(act_ckpt[sub, s, t]) * c_base.gdn_pages
                    rec_val = unpack_rec_checkpoint(
                        raw[p0 : p0 + c_base.rec_pages], c_base.vh
                    )
                    ref_rec = unpack_rec_checkpoint(
                        ref_raw[p0 : p0 + c_base.rec_pages], c_base.vh
                    )
                    conv_val = unpack_conv_checkpoint(
                        raw[p0 + c_base.rec_pages], c_base.QKV
                    )
                    ref_conv = unpack_conv_checkpoint(
                        ref_raw[p0 + c_base.rec_pages], c_base.QKV
                    )
                    if fused:
                        np.testing.assert_array_equal(rec_val, ref_rec)
                        np.testing.assert_array_equal(conv_val, ref_conv)
                    else:
                        ref32_rec = unpack_rec_checkpoint(
                            ref_32[1][p0 : p0 + c_base.rec_pages], c_base.vh
                        )
                        ref32_conv = unpack_conv_checkpoint(
                            ref_32[1][p0 + c_base.rec_pages], c_base.QKV
                        )
                        np.testing.assert_array_equal(rec_val, ref32_rec)
                        np.testing.assert_array_equal(conv_val, ref32_conv)
                        np.testing.assert_allclose(
                            rec_val, ref_rec, atol=2e-2, rtol=2e-2
                        )
                        np.testing.assert_allclose(
                            conv_val, ref_conv, atol=2e-2, rtol=2e-2
                        )


def test_wide_verify_buckets_split_gq_pack_parity():
    """Verify multi-rank `R=2, G=2, SPLIT=True, GQ=True` (`MLW = 128 // NS = 12, 8` at
    `B = 40, 64`) numerical parity and tail-sequence handling (`s >= 8`, `b = 32..63`)."""
    mesh = Mesh(np.array(jax.devices("cpu")[:2]), ("tp",))
    # R=2, NKV=1 -> G=2 (SPLIT=True); NH=8 -> nh=4, T=4 -> QG=16 (GQ=True, MLW=128//NS).
    c_base = K.MKConfig(
        R=2,
        B=16,
        L=2,
        att_every=2,
        H=512,
        NH=8,
        NKV=1,
        HD=256,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        ctx=256,
        T=4,
        paged_kv=True,
        num_pages=192,
        page_size=128,
        max_pages_per_seq=2,
        ckpt_k=4,
    )
    tiles_base = dataclasses.replace(
        K.default_tiles(c_base),
        nbd=4,
        nbe=4,
        nkv=2,
        hier=0,
        rsag=False,
        kv_blk=256,
        kv_sub=128,
    )
    shapes = K.weight_shapes(c_base, tiles_base)
    rng = np.random.default_rng(99)
    w_leaves = []
    for k in K.W_NAMES:
        s, dt = shapes[k]
        arr = rng.standard_normal((c_base.R, *s)).astype(np.float32) * 0.02
        w_leaves.append(K.fold(jnp.asarray(arr, dtype=dt)))

    raw_init_u8 = rng.integers(
        0,
        40,
        size=(c_base.R * c_base.num_pages, 2, 64, 4, c_base.page_size),
        dtype=np.uint8,
    )
    raw_init = jnp.asarray(raw_init_u8).view(K.FP8)

    n_act = 2
    act_tokens = rng.standard_normal((n_act * 4, c_base.H)).astype(np.float32) * 0.1
    act_roff = np.array([0, -1], dtype=np.int32)
    act_pos = np.array([[16, 17, 18, 19], [0, 1, 2, 3]], dtype=np.int32)
    act_pt = np.array([[1, 2], [3, 4]], dtype=np.int32)
    act_ckpt = np.arange(16, 16 + n_act * 4, dtype=np.int32).reshape(1, n_act, 4)

    def run_bucket(b: int, place_at_tail: bool):
        fused = (b <= 24) or (b >= 56)
        c = dataclasses.replace(c_base, B=b, fused_pool_io=fused)
        tiles = dataclasses.replace(
            K.default_tiles(c),
            nbd=4,
            nbe=4,
            nkv=2,
            hier=0,
            rsag=False,
            kv_blk=256,
            kv_sub=128,
        )
        ns = b // 4
        s0 = (ns - n_act) if place_at_tail else 0
        x = np.zeros((b, c.H), dtype=np.float32)
        x[s0 * 4 : (s0 + n_act) * 4] = act_tokens
        pos = np.full((b,), -1, dtype=np.int32)
        pos[s0 * 4 : (s0 + n_act) * 4] = act_pos.reshape(-1)
        pt = np.zeros((ns, c.max_pages_per_seq), dtype=np.int32)
        pt[s0 : s0 + n_act] = act_pt
        ckpt = np.zeros((1, ns, 4), dtype=np.int32)
        ckpt[:, s0 : s0 + n_act, :] = act_ckpt
        roff = np.full((1, ns), -1, dtype=np.int32)
        roff[:, s0 : s0 + n_act] = act_roff[None, :]

        step, _, _, _ = pool_io.make_flat_pooled_step(
            mesh,
            c,
            tiles,
            interpret=True,
            skip=frozenset({"experts"}),
            fused_pool_io=fused,
        )
        outs = step(
            jnp.tile(jnp.asarray(pos), (c.R,)),
            jnp.tile(jnp.asarray(x, dtype=K.BF16), (c.R, 1)),
            *w_leaves,
            raw_init,
            jnp.tile(jnp.asarray(pt), (c.R, 1)),
            jnp.tile(jnp.asarray(ckpt), (c.R, 1, 1)),
            jnp.tile(jnp.asarray(roff), (c.R, 1)),
        )
        out_r0 = np.asarray(outs[0][:b], dtype=np.float32)[s0 * 4 : (s0 + n_act) * 4]
        out_r1 = np.asarray(outs[0][c.M : c.M + b], dtype=np.float32)[
            s0 * 4 : (s0 + n_act) * 4
        ]
        out_ranks = np.stack([out_r0, out_r1], axis=0)
        raw_u8 = np.asarray(outs[1].view(jnp.uint8)).reshape(
            c.R, c.num_pages, 2, 64, 4, c.page_size
        )
        return out_ranks, raw_u8

    ref_out, ref_raw = run_bucket(16, place_at_tail=False)
    kv_pages = act_pt[:, 0]

    for b in (40, 64):
        out, raw = run_bucket(b, place_at_tail=True)
        fused = b >= 56
        np.testing.assert_array_equal(raw[:, kv_pages], ref_raw[:, kv_pages])
        if fused:
            np.testing.assert_array_equal(out, ref_out)
        else:
            np.testing.assert_allclose(out, ref_out, atol=2e-2, rtol=2e-2)
        for s in range(n_act):
            for t in range(4):
                p0 = int(act_ckpt[0, s, t]) * c_base.gdn_pages
                for r in range(c_base.R):
                    rec_val = unpack_rec_checkpoint(
                        raw[r, p0 : p0 + c_base.rec_pages], c_base.vh
                    )
                    ref_rec = unpack_rec_checkpoint(
                        ref_raw[r, p0 : p0 + c_base.rec_pages], c_base.vh
                    )
                    conv_val = unpack_conv_checkpoint(
                        raw[r, p0 + c_base.rec_pages], c_base.QKV
                    )
                    ref_conv = unpack_conv_checkpoint(
                        ref_raw[r, p0 + c_base.rec_pages], c_base.QKV
                    )
                    if fused:
                        np.testing.assert_array_equal(rec_val, ref_rec)
                        np.testing.assert_array_equal(conv_val, ref_conv)
                    else:
                        np.testing.assert_allclose(
                            rec_val, ref_rec, atol=2e-2, rtol=2e-2
                        )
                        np.testing.assert_allclose(
                            conv_val, ref_conv, atol=2e-2, rtol=2e-2
                        )
