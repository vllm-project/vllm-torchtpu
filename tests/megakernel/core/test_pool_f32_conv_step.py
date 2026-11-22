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
"""Pooled step tests for `f32` and `bf16` conv pools across split, GPR, and fused I/O.

The step reads and writes the conv window as `f32` or `bf16` across split pool
I/O, in-kernel `GPR` gather, and experimental 1-launch `fused_pool_io`, keeping
the kernel's `f32` conv state bit-exact (no `bf16` round trip) when configured.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tests.megakernel.core.megakernel_synth import TINY_TILES, synth_hybrid_spec
from vllm_torchtpu.megakernel import core as mk
from vllm_torchtpu.megakernel.core import pool
from vllm_torchtpu.megakernel.core.ops import BF16, F32, I32
from vllm_torchtpu.megakernel.core.pool.config import resolved_max_pages_per_seq
from vllm_torchtpu.megakernel.core.reference import spec_reference_step


@pytest.fixture(autouse=True)
def _use_cpu_default_device(request: pytest.FixtureRequest):
    if request.node.get_closest_marker("tpu_test") is not None:
        yield
        return
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _cpu_mesh(R: int) -> Mesh:
    devs = jax.devices("cpu")
    if len(devs) < R:
        pytest.skip(f"Need at least {R} CPU devices, have {len(devs)}")
    return Mesh(np.array(devs[:R]), ("tp",))


def _tiny_geom() -> mk.ModelGeometry:
    return mk.ModelGeometry(
        L=4,
        H=256,
        I=256,
        IS=256,
        E=128,
        TOPK=2,
        NH=4,
        NKV=2,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        att_every=4,
        moe_tp=False,
        e_slots=0,
        theta=10_000_000.0,
        zero_centered_norm=True,
    )


def _tiny_exec(R: int, B: int, T: int) -> mk.ExecSpec:
    tiles = replace(TINY_TILES, bank=512 << 10)
    return mk.ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=256,
        stack_e=True,
        interpret=True,
        tiles=tiles,
        cache=mk.CacheExec(paged_kv=True, num_pages=96),
    )


def _with_conv_dtype(
    spec: mk.ModelSpec, dt: Any
) -> tuple[mk.ModelSpec, pool.PoolLayout]:
    assert spec.cache_backend is not None
    lay = replace(spec.cache_backend.layout, conv_dtype=dt)
    pb = spec.cache_backend.with_layout(lay)
    assert pb.supports_fused_io
    return replace(spec, cache_backend=pb), lay


def _non_rec_pool_bytes_equal(
    raw_a: tuple[jax.Array, ...],
    raw_b: tuple[jax.Array, ...],
    lay: pool.PoolLayout,
    R: int,
    ckpt: jax.Array,
    active_seqs: jax.Array | None = None,
) -> None:
    """Assert raw byte equality on all KV, conv, and unreferenced/inactive pages.

    On AVX512 XLA:CPU, fused (`[1, KD, VD]`) vs chunked split (`[CB, KD, VD]`)
    GDN recurrence can differ by 1 bf16 ULP inside active `rec` checkpoint
    pages; `st['rec']` is checked separately at 1-ULP bf16 tolerance while all
    KV pages, conv pages, and inactive/unreferenced pages must match bit-exact.
    """
    ck_np = np.asarray(ckpt)
    if ck_np.ndim == 4:
        ck_np = ck_np[0]
    ndim = ck_np.ndim
    n_s = ck_np.shape[1] if ndim == 3 else ck_np.shape[0]
    act = (
        np.ones((n_s,), dtype=bool)
        if active_seqs is None
        else (np.asarray(active_seqs) != 0)
    )
    rec_pages: set[int] = set()
    for s_i in range(n_s):
        if not act[s_i]:
            continue
        if ndim == 3:
            for sub in range(ck_np.shape[0]):
                for sl in np.unique(ck_np[sub, s_i]):
                    if int(sl) >= 0:
                        pb = int(lay.page_base(int(sl), sub, 3))
                        rec_pages.update(range(pb, pb + lay.rec_pages_per_slot))
        else:
            for sl in np.unique(ck_np[s_i]):
                if int(sl) >= 0:
                    for sub in range(lay.slots_per_pool):
                        pb = int(lay.page_base(int(sl), sub, 2))
                        rec_pages.update(range(pb, pb + lay.rec_pages_per_slot))
    for p_idx, (ra, rb) in enumerate(zip(raw_a, raw_b)):
        n_pages = ra.shape[0] // R
        keep = np.array([pg not in rec_pages for pg in range(n_pages)], dtype=bool)
        ua = np.asarray(ra).view(np.uint8).reshape(R, n_pages, -1)[:, keep]
        ub = np.asarray(rb).view(np.uint8).reshape(R, n_pages, -1)[:, keep]
        np.testing.assert_array_equal(ua, ub, err_msg=f"pool {p_idx} non-rec mismatch")


def _wide_conv_geom() -> mk.ModelGeometry:
    """Per-rank `QKV = 1536` (`qkv_128 = 12`, wide TP=8 conv width):
    `GCONV_ROWS = 96 > 64` (2-DMA K+V GPR path) and `conv8_rows = 96 <= 128`."""
    return replace(_tiny_geom(), KH=4, VH=16)


@pytest.mark.parametrize("T,B", [(1, 2), (2, 4)])
def test_f32_conv_pooled_step_matches_reference_and_keeps_conv_exact(T: int, B: int):
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))
    g = _tiny_geom()
    ex = _tiny_exec(R=2, B=B, T=T)
    spec = synth_hybrid_spec(g, R=2)
    pos, hid_l, w_l, st_l = mk.random_spec_inputs(jax.random.key(70 + T), spec, ex)
    conv = np.asarray(st_l["conv"], np.float32)
    assert not np.array_equal(
        conv, np.asarray(st_l["conv"].astype(jnp.bfloat16), np.float32)
    )

    # Use `pool_io=False` so `ref_st` keeps unrounded `f32` conv taps.
    ref_out, ref_st = spec_reference_step(
        spec, g, ex, pos, hid_l, w_l, st_l, paged_kv=True, pool_io=False
    )
    _, hid_mk, w_mk, st_mk_paged = mk.pack_spec_inputs(
        pos, hid_l, w_l, st_l, spec, ex, mesh=mesh
    )
    assert spec.cache_backend is not None
    play_base = spec.cache_backend.layout
    pt_base, ckpt_base, roff_base, _ = pool.allocate_test_layout(play_base, ex)
    st_mk_paged[mk.POOL_STATE_NAME] = tuple(
        jax.device_put(r, tp)
        for r in pool.states_to_pool(play_base, ex, st_l, pt_base, ckpt_base, roff_base)
    )
    st_mk_paged["page_table"] = jax.device_put(
        jnp.tile(pt_base[:, : resolved_max_pages_per_seq(ex)], (ex.R, 1)), tp
    )
    _, mk_st = mk.make_step(mesh, spec, ex, donate_states=False)[0](
        pos, hid_mk, w_mk, st_mk_paged
    )

    errs = {}
    for name, dt in (("f32", jnp.float32), ("bf16", jnp.bfloat16)):
        spec_dt, lay = _with_conv_dtype(spec, dt)
        lay.validate(ex)
        pt, ckpt, roff, min_np = pool.allocate_test_layout(lay, ex)
        assert mk.cache_exec(ex).num_pages >= min_np
        st_in = (
            st_l
            if name == "f32"
            else dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
        )
        raw_pools = tuple(
            jax.device_put(r, tp)
            for r in pool.states_to_pool(lay, ex, st_in, pt, ckpt, roff)
        )
        step, _ = mk.PallasWriter().make_step(mesh, spec_dt, ex, donate=False)
        out_hid, raw_out = step(pos, hid_mk, w_mk, raw_pools, pt, ckpt, roff)
        out_b = mk.unfold(out_hid, ex.R)[0, :B].astype(jnp.float32)
        assert jnp.allclose(out_b, ref_out.astype(jnp.float32), atol=3e-2, rtol=3e-2)
        st_after = pool.pool_to_states(lay, ex, raw_out, pt, ckpt)

        QKV = g.QKV(ex.R)
        if T == 1:
            kern = (
                mk.unfold(mk_st["conv"], ex.R)[:, :, : 3 * B, :]
                .reshape(ex.R, g.NGDN, 3, B, QKV)
                .swapaxes(2, 3)
            )
            got = st_after["conv"][:, :, :, 0]
            ref = ref_st["conv"][:, :, : 3 * B, :].reshape(ex.R, g.NGDN, 3, B, QKV)
            ref = ref.swapaxes(2, 3)
        else:
            kern = mk.unfold(mk_st["conv_ckpt"], ex.R)[
                :, :, :, 1 + T - 1 : 4 + T - 1, :
            ]
            got = st_after["conv"][:, :, :, T - 1]
            ref = ref_st["conv_ckpt"][:, :, :, 1 + T - 1 : 4 + T - 1, :]
        if name == "f32":
            np.testing.assert_array_equal(np.asarray(got), np.asarray(kern))
        else:
            assert not np.array_equal(np.asarray(got), np.asarray(kern))
        n_pres = 3 - T
        errs[name] = (
            float(jnp.max(jnp.abs(got[..., :n_pres, :] - ref[..., :n_pres, :]))),
            float(jnp.mean(jnp.abs(got - ref))),
        )
    assert errs["f32"][0] == 0.0 and errs["bf16"][0] > 0.0, errs
    assert errs["f32"][1] < errs["bf16"][1], errs


@pytest.mark.parametrize(
    ("T", "wide_qkv", "conv_dtype"),
    [
        (1, False, BF16),
        (4, False, BF16),
        (4, True, BF16),
        (1, False, F32),
        (1, True, F32),
        (4, False, F32),
        (4, True, F32),
    ],
    ids=[
        "t1_narrow_bf16",
        "t4_narrow_bf16",
        "t4_wide_bf16",
        "t1_narrow_f32",
        "t1_wide_f32",
        "t4_narrow_f32",
        "t4_wide_f32",
    ],
)
def test_conv_fused_matches_split(T: int, wide_qkv: bool, conv_dtype: Any):
    """Fused pool I/O (`T in (1, 4)`, `conv_stage in (False, True)`) matches
    head-major split pool I/O: bitwise on `f32`-conv hidden outputs (at `T=1`
    and `t=0` of `T=4`) and on KV/conv pool bytes, within 8 `bf16` ULPs on CPU
    for `bf16`-conv hidden outputs across `L=4` (TPU bitwise coverage for both
    `bf16` and `f32` conv is in `test_fused_pool_io_tpu.py`)."""
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))
    g = _wide_conv_geom() if wide_qkv else _tiny_geom()
    B = 2 * T
    # `wide_rows=0` ensures split activates `GCH=True` (`gdn_rec_chunked`), so
    # `gdn_head_major=True` actually runs head-major recurrence matching fused.
    tiles = replace(TINY_TILES, bank=512 << 10, wide_rows=0).with_modules(
        {"gdn": {"conv_stage": False, "gdn_head_major": True}}
    )
    ex_split = mk.ExecSpec(
        R=2,
        B=B,
        T=T,
        ctx=256,
        stack_e=False,
        interpret=True,
        tiles=tiles,
        cache=mk.CacheExec(
            paged_kv=True,
            num_pages=512,
            fused_pool_io=False,
            pool_gather=False,
        ),
    )
    ex_fused = mk.with_cache(ex_split, fused_pool_io=True, pool_gather=False)
    ex_fused_cs = replace(
        ex_fused,
        tiles=tiles.with_modules({"gdn": {"conv_stage": True, "gdn_head_major": True}}),
    )

    spec_base = synth_hybrid_spec(g, R=2)
    pos, hid_l, w_l, st_l = mk.random_spec_inputs(
        jax.random.key(710 + 10 * T + B + 2000), spec_base, ex_split
    )
    assert not np.array_equal(
        np.asarray(st_l["conv"], np.float32),
        np.asarray(st_l["conv"].astype(jnp.bfloat16), np.float32),
    )
    if jnp.dtype(conv_dtype) == jnp.dtype(BF16):
        st_l = dict(st_l, conv=st_l["conv"].astype(BF16).astype(F32))
    _, hid_mk, w_mk, _ = mk.pack_spec_inputs(
        pos, hid_l, w_l, st_l, spec_base, ex_split, mesh=mesh
    )

    spec_dt, lay = _with_conv_dtype(spec_base, conv_dtype)
    pt, ckpt, roff, min_np = pool.allocate_test_layout(lay, ex_split)
    assert mk.cache_exec(ex_split).num_pages >= min_np
    raw_init = [
        jax.device_put(r, tp)
        for r in pool.states_to_pool(lay, ex_split, st_l, pt, ckpt, roff)
    ]

    step_split, info_split = mk.PallasWriter().make_step(
        mesh, spec_dt, ex_split, donate=False, fused_pool_io=False
    )
    assert not info_split["use_fused"] and not info_split["use_gpr"]
    assert info_split["plan"].module_params["gdn"].GCH
    hid_split, raw_split = step_split(
        pos, hid_mk, w_mk, tuple(raw_init), pt, ckpt, roff
    )
    st_split = pool.pool_to_states(lay, ex_split, raw_split, pt, ckpt)

    raw_fused_in = tuple(jax.device_put(np.asarray(r).copy(), tp) for r in raw_init)
    step_fused, info_fused = mk.PallasWriter().make_step(
        mesh, spec_dt, ex_fused, donate=True, fused_pool_io=True
    )
    assert info_fused["use_fused"]
    hid_fused, raw_fused = step_fused(pos, hid_mk, w_mk, raw_fused_in, pt, ckpt, roff)
    st_fused = pool.pool_to_states(lay, ex_split, raw_fused, pt, ckpt)

    raw_cs_in = tuple(jax.device_put(np.asarray(r).copy(), tp) for r in raw_init)
    step_cs, info_cs = mk.PallasWriter().make_step(
        mesh, spec_dt, ex_fused_cs, donate=True, fused_pool_io=True
    )
    assert info_cs["use_fused"]
    hid_cs, raw_cs = step_cs(pos, hid_mk, w_mk, raw_cs_in, pt, ckpt, roff)

    np.testing.assert_array_equal(
        np.asarray(hid_cs).view(np.uint8), np.asarray(hid_fused).view(np.uint8)
    )
    for r_cs, r_f in zip(raw_cs, raw_fused):
        np.testing.assert_array_equal(
            np.asarray(r_cs).view(np.uint8), np.asarray(r_f).view(np.uint8)
        )

    is_f32_conv = jnp.dtype(conv_dtype) == jnp.dtype(F32)
    if T == 1:
        if is_f32_conv:
            np.testing.assert_array_equal(
                np.asarray(hid_fused).view(np.uint8),
                np.asarray(hid_split).view(np.uint8),
            )
            for k in ("kc", "vc", "conv"):
                np.testing.assert_array_equal(
                    np.asarray(st_fused[k]).view(np.uint8),
                    np.asarray(st_split[k]).view(np.uint8),
                    err_msg=f"mismatch on {k}",
                )
            np.testing.assert_allclose(
                np.asarray(st_fused["rec"], np.float32),
                np.asarray(st_split["rec"], np.float32),
                rtol=2**-7,
                atol=1e-4,
            )
            _non_rec_pool_bytes_equal(raw_fused, raw_split, lay, ex_split.R, ckpt)
        else:
            # On Intel XLA:CPU, batched `(B, QKV)`/`[CB, KD, VD]` split vs
            # per-sequence `(1, QKV)`/`[1, KD, VD]` fused GDN recurrence can
            # differ by 1 `bf16` ULP in layer 0. Later layers inherit it: bound
            # `hid` at 8 `bf16` ULPs (`<= 4` observed), later-layer conv at
            # `2**-4`, `rec` at `2**-7`, and the KV entry written at `pos` at
            # 1 fp8 ULP. Layer-0 conv (written before the recurrence), padded
            # rows and every KV entry other than the one written stay bitwise.
            # TPU bitwise coverage is in `test_fused_pool_io_tpu.py`.
            np.testing.assert_array_equal(
                np.asarray(mk.unfold(hid_fused, 2)[:, B:]).view(np.uint8),
                np.asarray(mk.unfold(hid_split, 2)[:, B:]).view(np.uint8),
            )
            np.testing.assert_allclose(
                np.asarray(hid_fused, np.float32),
                np.asarray(hid_split, np.float32),
                rtol=8 * 2**-7,
                atol=1e-3,
            )
            np.testing.assert_array_equal(
                np.asarray(st_fused["conv"][:, 0]).view(np.uint8),
                np.asarray(st_split["conv"][:, 0]).view(np.uint8),
            )
            np.testing.assert_allclose(
                np.asarray(st_fused["conv"], np.float32),
                np.asarray(st_split["conv"], np.float32),
                rtol=2**-4,
                atol=2e-3,
            )
            np.testing.assert_allclose(
                np.asarray(st_fused["rec"], np.float32),
                np.asarray(st_split["rec"], np.float32),
                rtol=2**-7,
                atol=1e-3,
            )
            pos_np = np.asarray(pos, dtype=np.int32)
            for k in ("kc", "vc"):
                f_u8 = np.asarray(st_fused[k]).view(np.uint8)
                s_u8 = np.asarray(st_split[k]).view(np.uint8)
                for s_i in range(B):
                    p0 = int(pos_np[s_i])
                    np.testing.assert_array_equal(
                        f_u8[:, :, s_i, :p0], s_u8[:, :, s_i, :p0]
                    )
                    np.testing.assert_array_equal(
                        f_u8[:, :, s_i, p0 + 1 :], s_u8[:, :, s_i, p0 + 1 :]
                    )
                np.testing.assert_allclose(
                    np.asarray(st_fused[k].astype(F32)),
                    np.asarray(st_split[k].astype(F32)),
                    rtol=0.15,
                    atol=2.5e-2,
                )
    else:
        # Causal `t == 0` (`f32` conv) outputs and layer-0 conv across all `T`
        # checkpoints are bit-exact on XLA:CPU; `t >= 1` tokens can inherit 1
        # bf16 ULP from batched `(B, QKV)` vs single-row `(1, QKV)` CPU FMA /
        # recurrence (full `T=4` bit-exactness on TPU is in
        # `test_fused_pool_io_tpu.py`).
        hid_fu_b = np.asarray(mk.unfold(hid_fused, 2)[:, :B], np.float32)
        hid_sp_b = np.asarray(mk.unfold(hid_split, 2)[:, :B], np.float32)
        if is_f32_conv:
            np.testing.assert_array_equal(
                np.asarray(mk.unfold(hid_fused, 2)[:, :B:T]).view(np.uint8),
                np.asarray(mk.unfold(hid_split, 2)[:, :B:T]).view(np.uint8),
            )
        else:
            np.testing.assert_allclose(
                hid_fu_b[:, ::T],
                hid_sp_b[:, ::T],
                rtol=8 * 2**-7,
                atol=1e-3,
            )
        np.testing.assert_allclose(hid_fu_b, hid_sp_b, rtol=2.5e-2, atol=2.5e-2)
        assert (
            float(np.mean(np.isclose(hid_fu_b, hid_sp_b, rtol=2**-4, atol=3e-3)))
            >= 0.99
        )
        np.testing.assert_array_equal(
            np.asarray(st_fused["conv"][:, 0]).view(np.uint8),
            np.asarray(st_split["conv"][:, 0]).view(np.uint8),
        )
        np.testing.assert_allclose(
            np.asarray(st_fused["conv"], np.float32),
            np.asarray(st_split["conv"], np.float32),
            rtol=2.5e-2,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            np.asarray(st_fused["rec"], np.float32),
            np.asarray(st_split["rec"], np.float32),
            rtol=2.5e-2,
            atol=2e-2,
        )
        pos_np = np.asarray(pos, dtype=np.int32)
        for k in ("kc", "vc"):
            f_u8 = np.asarray(st_fused[k]).view(np.uint8)
            s_u8 = np.asarray(st_split[k]).view(np.uint8)
            for s_i in range(B // T):
                p0 = int(pos_np[s_i * T])
                np.testing.assert_array_equal(
                    f_u8[:, :, s_i, : p0 + 1], s_u8[:, :, s_i, : p0 + 1]
                )
                np.testing.assert_array_equal(
                    f_u8[:, :, s_i, p0 + T :], s_u8[:, :, s_i, p0 + T :]
                )
            np.testing.assert_allclose(
                np.asarray(st_fused[k].astype(F32)),
                np.asarray(st_split[k].astype(F32)),
                rtol=0.15,
                atol=2.5e-2,
            )
    conv_out = np.asarray(st_fused["conv"], np.float32)
    conv_bf16 = np.asarray(st_fused["conv"].astype(jnp.bfloat16), np.float32)
    if jnp.dtype(conv_dtype) == jnp.dtype(F32):
        assert not np.array_equal(conv_out, conv_bf16)
    else:
        np.testing.assert_array_equal(conv_out, conv_bf16)


def test_fused_pool_io_rejects_vh_equal_kh_with_config_error():
    """`VH == KH` (`2 * kh > vh`) raises `MegakernelConfigError`, not raw `assert`."""
    mesh = _cpu_mesh(2)
    g_eq = replace(_tiny_geom(), KH=4, VH=4)
    spec_eq = synth_hybrid_spec(g_eq, R=2)
    ex_fused = mk.with_cache(_tiny_exec(R=2, B=2, T=1), fused_pool_io=True)
    with pytest.raises(mk.MegakernelConfigError, match=r"2 \* kh <= vh"):
        mk.build(spec_eq, ex_fused)
    with pytest.raises(mk.MegakernelConfigError, match=r"2 \* kh <= vh"):
        mk.PallasWriter().make_step(
            mesh, spec_eq, ex_fused, donate=True, fused_pool_io=True
        )


def test_fused_pool_io_boundary_paths_match_split_bitwise():
    """Exercise slow cross-page KV RMW (`kv_mode == 2`), `NATT=2` multi-pool
    `bswitch`, `SPLIT=True` (`G=2`), non-contiguous `ckpt` (`out_mode == 2`),
    cold-start (`roff == -1`), inactive `active_seqs`, `NS > NGI=8` input refill
    with `ckpt_2d`/`roff_1d`, and `detect_races=True`."""
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))
    tiles = replace(TINY_TILES, bank=512 << 10, wide_rows=0).with_modules(
        {"gdn": {"conv_stage": False, "gdn_head_major": True}}
    )

    # Part 1: T=4, NS=3 (B=12), L=4, att_every=2 (NATT=2), NKV=1 (G=2 -> SPLIT=True),
    # seq 0 straddling page boundary pos=[126..129] + non-contiguous ckpt,
    # seq 1 cold-start roff=-1, seq 2 inactive via active_seqs=[1, 1, 0].
    g_split_att = replace(_tiny_geom(), L=4, att_every=2, NH=4, NKV=1)
    spec_s, lay_s = _with_conv_dtype(synth_hybrid_spec(g_split_att, R=2), F32)
    assert lay_s.num_pools == 2 and lay_s.slots_per_pool == 1
    ex_s = mk.ExecSpec(
        R=2,
        B=12,
        T=4,
        ctx=256,
        stack_e=True,
        interpret=pltpu.InterpretParams(detect_races=True),
        tiles=tiles,
        cache=mk.CacheExec(
            paged_kv=True, num_pages=256, fused_pool_io=False, pool_gather=False
        ),
    )
    pos_base, hid_l, w_l, st_l = mk.random_spec_inputs(
        jax.random.key(801), spec_s, ex_s
    )
    del pos_base
    pos_cross = jnp.array(
        [126, 127, 128, 129, 16, 17, 18, 19, 32, 33, 34, 35], dtype=I32
    )
    _, hid_mk, w_mk, _ = mk.pack_spec_inputs(
        pos_cross, hid_l, w_l, st_l, spec_s, ex_s, mesh=mesh
    )
    pt, ckpt, roff, _ = pool.allocate_test_layout(lay_s, ex_s)
    # Non-contiguous checkpoint slots for seq 0 (`out_mode == 2`) without colliding.
    max_slot = int(jnp.max(ckpt))
    ckpt_nc = ckpt.at[0, 0, :].set(
        jnp.array([max_slot + 2, max_slot + 5, max_slot + 8, max_slot + 11], dtype=I32)
    )
    roff_cs = roff.at[:, 1].set(-1)
    act_mask = jnp.array([1, 1, 0], dtype=I32)
    # Zero cold-start seq 1's initial logical GDN state for `states_to_pool` baseline.
    st_init = dict(
        st_l,
        conv=st_l["conv"].at[:, :, 1].set(0.0),
        rec=st_l["rec"].at[:, :, 1].set(0.0),
    )
    raw_init = [
        jax.device_put(r, tp)
        for r in pool.states_to_pool(
            lay_s, ex_s, st_init, pt, ckpt_nc, jnp.maximum(roff_cs, 0)
        )
    ]

    step_sp, info_sp = mk.PallasWriter().make_step(
        mesh, spec_s, ex_s, donate=False, fused_pool_io=False
    )
    assert info_sp["plan"].module_params["gqa"].SPLIT
    assert info_sp["plan"].module_params["gdn"].GCH
    hid_sp, raw_sp = jax.block_until_ready(
        step_sp(
            pos_cross, hid_mk, w_mk, tuple(raw_init), pt, ckpt_nc, roff_cs, act_mask
        )
    )
    assert ipc.races is not None and not ipc.races.races_found

    ex_fu = mk.with_cache(ex_s, fused_pool_io=True, pool_gather=False)
    raw_fu_in = tuple(jax.device_put(np.asarray(r).copy(), tp) for r in raw_init)
    step_fu, info_fu = mk.PallasWriter().make_step(
        mesh, spec_s, ex_fu, donate=True, fused_pool_io=True
    )
    assert info_fu["use_fused"] and info_fu["plan"].module_params["gqa"].SPLIT
    hid_fu, raw_fu = jax.block_until_ready(
        step_fu(pos_cross, hid_mk, w_mk, raw_fu_in, pt, ckpt_nc, roff_cs, act_mask)
    )
    assert ipc.races is not None and not ipc.races.races_found

    # Active rows 0..7 match; KV, conv, and inactive seq 2's pool pages match
    # bit-for-bit, and active `rec` matches within 1 bf16 ULP on XLA:CPU.
    np.testing.assert_allclose(
        np.asarray(mk.unfold(hid_fu, 2)[:, :8], np.float32),
        np.asarray(mk.unfold(hid_sp, 2)[:, :8], np.float32),
        rtol=2**-5,
        atol=2e-3,
    )
    _non_rec_pool_bytes_equal(raw_fu, raw_sp, lay_s, ex_s.R, ckpt_nc, act_mask)
    st_fu = pool.pool_to_states(lay_s, ex_s, raw_fu, pt, ckpt_nc)
    st_sp = pool.pool_to_states(lay_s, ex_s, raw_sp, pt, ckpt_nc)
    for k in ("kc", "vc", "conv"):
        np.testing.assert_array_equal(
            np.asarray(st_fu[k]).view(np.uint8), np.asarray(st_sp[k]).view(np.uint8)
        )
    np.testing.assert_allclose(
        np.asarray(st_fu["rec"], np.float32),
        np.asarray(st_sp["rec"], np.float32),
        rtol=2**-7,
        atol=1e-4,
    )
    # Inactive seq 2's recurrent and conv state in the pool is untouched vs `raw_init`.
    st_init_p = pool.pool_to_states(lay_s, ex_s, tuple(raw_init), pt, ckpt_nc)
    s2_r0 = 2 * ex_s.T * lay_s.vh_per_rank(ex_s.R)
    np.testing.assert_array_equal(
        np.asarray(st_fu["rec"][:, :, s2_r0 : s2_r0 + lay_s.vh_per_rank(ex_s.R)]).view(
            np.uint8
        ),
        np.asarray(
            st_init_p["rec"][:, :, s2_r0 : s2_r0 + lay_s.vh_per_rank(ex_s.R)]
        ).view(np.uint8),
    )

    # Part 2: T=1, NS=10 > NGI=8 (input refill `s + NGI < NS` + `NS > NGO=2`),
    # with 2D `ckpt` (`(NS, 1)`) and 1D `roff` (`(NS,)`) on 1-slot-per-pool layout.
    g_refill = replace(_tiny_geom(), L=2, att_every=2)
    spec_r, lay_r = _with_conv_dtype(synth_hybrid_spec(g_refill, R=2), BF16)
    ex_r = mk.ExecSpec(
        R=2,
        B=10,
        T=1,
        ctx=256,
        stack_e=True,
        interpret=pltpu.InterpretParams(detect_races=True),
        tiles=tiles,
        cache=mk.CacheExec(
            paged_kv=True,
            num_pages=256,
            fused_pool_io=False,
            pool_gather=False,
            ckpt_2d=True,
            roff_1d=True,
        ),
    )
    pos_r, hid_r, w_r, st_r = mk.random_spec_inputs(jax.random.key(802), spec_r, ex_r)
    st_r = dict(st_r, conv=st_r["conv"].astype(BF16).astype(F32))
    _, hid_r_mk, w_r_mk, _ = mk.pack_spec_inputs(
        pos_r, hid_r, w_r, st_r, spec_r, ex_r, mesh=mesh
    )
    pt_r, ckpt_3d, roff_2d, _ = pool.allocate_test_layout(lay_r, ex_r)
    ckpt_2d, roff_1d = ckpt_3d[0], roff_2d[0]
    raw_r_init = [
        jax.device_put(r, tp)
        for r in pool.states_to_pool(lay_r, ex_r, st_r, pt_r, ckpt_2d, roff_1d)
    ]
    step_r_sp, _ = mk.PallasWriter().make_step(
        mesh, spec_r, ex_r, donate=False, fused_pool_io=False
    )
    hid_r_sp, raw_r_sp = jax.block_until_ready(
        step_r_sp(pos_r, hid_r_mk, w_r_mk, tuple(raw_r_init), pt_r, ckpt_2d, roff_1d)
    )
    assert ipc.races is not None and not ipc.races.races_found

    ex_r_fu = mk.with_cache(ex_r, fused_pool_io=True, pool_gather=False)
    raw_r_fu_in = tuple(jax.device_put(np.asarray(r).copy(), tp) for r in raw_r_init)
    step_r_fu, info_r_fu = mk.PallasWriter().make_step(
        mesh, spec_r, ex_r_fu, donate=True, fused_pool_io=True
    )
    assert info_r_fu["plan"].module_params["gdn"].NGI == 8 < ex_r.NS
    assert not info_r_fu["plan"].module_params["gdn"].ckpt_is_3d
    assert not info_r_fu["plan"].module_params["gdn"].roff_is_2d
    hid_r_fu, raw_r_fu = jax.block_until_ready(
        step_r_fu(pos_r, hid_r_mk, w_r_mk, raw_r_fu_in, pt_r, ckpt_2d, roff_1d)
    )
    assert ipc.races is not None and not ipc.races.races_found
    np.testing.assert_array_equal(
        np.asarray(hid_r_fu).view(np.uint8), np.asarray(hid_r_sp).view(np.uint8)
    )
    _non_rec_pool_bytes_equal(raw_r_fu, raw_r_sp, lay_r, ex_r.R, ckpt_2d)
    st_r_fu = pool.pool_to_states(lay_r, ex_r, raw_r_fu, pt_r, ckpt_2d)
    st_r_sp = pool.pool_to_states(lay_r, ex_r, raw_r_sp, pt_r, ckpt_2d)
    for k in ("kc", "vc", "conv"):
        np.testing.assert_array_equal(
            np.asarray(st_r_fu[k]).view(np.uint8),
            np.asarray(st_r_sp[k]).view(np.uint8),
        )
    np.testing.assert_allclose(
        np.asarray(st_r_fu["rec"], np.float32),
        np.asarray(st_r_sp["rec"], np.float32),
        rtol=2**-7,
        atol=1e-4,
    )


@pytest.mark.parametrize("wide_qkv", [False, True])
def test_f32_conv_gpr_matches_split_bitwise(wide_qkv: bool):
    """In-kernel gather (`GPR`, `T=4`) with `conv_dtype=f32` matches split pool I/O
    bit-for-bit on both `GCONV_ROWS=32 <= 64` (single-DMA K half) and
    `GCONV_ROWS=96 > 64` (two-DMA K+V halves)."""
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))
    g = _wide_conv_geom() if wide_qkv else _tiny_geom()
    T, B = 4, 8
    tiles = replace(TINY_TILES, bank=512 << 10)
    ex_split = mk.ExecSpec(
        R=2,
        B=B,
        T=T,
        ctx=256,
        stack_e=True,
        interpret=True,
        tiles=tiles,
        cache=mk.CacheExec(
            paged_kv=True,
            num_pages=192,
            fused_pool_io=False,
            pool_gather=False,
        ),
    )
    ex_gpr = mk.with_cache(ex_split, fused_pool_io=False, pool_gather=True)

    spec_base = synth_hybrid_spec(g, R=2)
    pos, hid_l, w_l, st_l = mk.random_spec_inputs(
        jax.random.key(410 + int(wide_qkv)), spec_base, ex_split
    )
    _, hid_mk, w_mk, _ = mk.pack_spec_inputs(
        pos, hid_l, w_l, st_l, spec_base, ex_split, mesh=mesh
    )

    spec_f32, lay = _with_conv_dtype(spec_base, F32)
    pt, ckpt, roff, min_np = pool.allocate_test_layout(lay, ex_split)
    assert mk.cache_exec(ex_split).num_pages >= min_np
    raw_init = tuple(
        jax.device_put(r, tp)
        for r in pool.states_to_pool(lay, ex_split, st_l, pt, ckpt, roff)
    )

    step_split, info_split = mk.PallasWriter().make_step(
        mesh, spec_f32, ex_split, donate=False, fused_pool_io=False
    )
    assert not info_split["use_gpr"]
    hid_split, raw_split = step_split(pos, hid_mk, w_mk, raw_init, pt, ckpt, roff)
    st_split = pool.pool_to_states(lay, ex_split, raw_split, pt, ckpt)

    step_gpr, info_gpr = mk.PallasWriter().make_step(
        mesh, spec_f32, ex_gpr, donate=False, fused_pool_io=False
    )
    assert info_gpr["use_gpr"]
    assert info_gpr["oshape"]["conv_ckpt"][1] == jnp.float32
    assert info_gpr["plan"].output_decls["conv_ckpt"].dtype == jnp.float32
    gdn_rows = info_gpr["plan"].module_params["gdn"].GCONV_ROWS
    assert gdn_rows == (96 if wide_qkv else 32)
    hid_gpr, raw_gpr = step_gpr(pos, hid_mk, w_mk, raw_init, pt, ckpt, roff)
    st_gpr = pool.pool_to_states(lay, ex_split, raw_gpr, pt, ckpt)

    np.testing.assert_array_equal(np.asarray(hid_gpr), np.asarray(hid_split))
    for k in ("kc", "vc", "conv", "rec"):
        np.testing.assert_array_equal(
            np.asarray(st_gpr[k]), np.asarray(st_split[k]), err_msg=f"mismatch on {k}"
        )
    conv_out = np.asarray(st_gpr["conv"], np.float32)
    assert not np.array_equal(
        conv_out, np.asarray(st_gpr["conv"].astype(jnp.bfloat16), np.float32)
    )


def _poison_stale_conv_rows(
    raw_pools: list[jax.Array] | tuple[jax.Array, ...],
    lay: pool.PoolLayout,
    ckpt: jax.Array,
    tp: NamedSharding,
    *,
    live_rows: int = 36,
) -> tuple[jax.Array, ...]:
    """Fill u32 rows `live_rows..127` of each conv page with `0xFF` (`f32` NaN)."""
    ckpt_np = np.asarray(ckpt)
    if ckpt_np.ndim == 4:
        ckpt_np = ckpt_np[0]
    ndim = ckpt_np.ndim
    conv_pages: set[int] = set()
    if ndim == 3:
        for sub in range(ckpt_np.shape[0]):
            for slot in np.unique(ckpt_np[sub]):
                if int(slot) >= 0:
                    conv_pages.add(
                        int(lay.page_base(int(slot), sub, 3)) + lay.conv_page
                    )
    else:
        for slot in np.unique(ckpt_np):
            if int(slot) >= 0:
                for sub in range(lay.slots_per_pool):
                    conv_pages.add(
                        int(lay.page_base(int(slot), sub, 2)) + lay.conv_page
                    )
    assert conv_pages
    r_cnt = len(tp.device_set)
    out = []
    for r in raw_pools:
        np_pages = r.shape[0] // r_cnt
        arr_u8 = (
            np.asarray(r).view(np.uint8).copy().reshape(r_cnt, np_pages, *r.shape[1:])
        )
        for pb in conv_pages:
            arr_u8[:, pb, 0, live_rows:64, :, :] = 0xFF
            arr_u8[:, pb, 1, 0:64, :, :] = 0xFF
            assert np.isnan(arr_u8[:, pb, 0, live_rows:64].view(np.float32)).all()
            assert np.isnan(arr_u8[:, pb, 1, 0:64].view(np.float32)).all()
        out.append(jax.device_put(arr_u8.reshape(r.shape).view(r.dtype), tp))
    return tuple(out)


@pytest.mark.parametrize("T", [1, 4])
def test_f32_conv_ignores_poisoned_stale_rows_36_to_127(T: int):
    """The stock GDN kernel writes only u32 rows `0..35` (`3 * 1536 / 128 = 36`
    granules) of the conv page; rows `36..127` may contain stale KV bytes including
    `f32` NaN (`0xFFFFFFFF`). Verify split, fused (`conv_stage` False/True), and
    GPR (`T=4`) produce finite outputs bit-identical to a clean zero-padded pool."""
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))
    g = _wide_conv_geom()
    B = 2 * T
    tiles = replace(TINY_TILES, bank=512 << 10).with_modules(
        {"gdn": {"conv_stage": False, "gdn_head_major": True}}
    )
    ex_split = mk.ExecSpec(
        R=2,
        B=B,
        T=T,
        ctx=256,
        stack_e=False,
        interpret=True,
        tiles=tiles,
        cache=mk.CacheExec(
            paged_kv=True,
            num_pages=512,
            fused_pool_io=False,
            pool_gather=False,
        ),
    )
    spec_base = synth_hybrid_spec(g, R=2)
    pos, hid_l, w_l, st_l = mk.random_spec_inputs(
        jax.random.key(910 + T), spec_base, ex_split
    )
    _, hid_mk, w_mk, _ = mk.pack_spec_inputs(
        pos, hid_l, w_l, st_l, spec_base, ex_split, mesh=mesh
    )
    spec_f32, lay = _with_conv_dtype(spec_base, F32)
    assert lay.conv_rows(ex_split.R) == 36
    pt, ckpt, roff, _ = pool.allocate_test_layout(lay, ex_split)
    raw_clean = tuple(
        jax.device_put(r, tp)
        for r in pool.states_to_pool(lay, ex_split, st_l, pt, ckpt, roff)
    )
    raw_poison = _poison_stale_conv_rows(raw_clean, lay, ckpt, tp, live_rows=36)

    modes: list[tuple[str, mk.ExecSpec, bool]] = [
        ("split", ex_split, False),
        (
            "fused",
            mk.with_cache(ex_split, fused_pool_io=True, pool_gather=False),
            True,
        ),
        (
            "fused_conv_stage",
            replace(
                mk.with_cache(ex_split, fused_pool_io=True, pool_gather=False),
                tiles=tiles.with_modules(
                    {"gdn": {"conv_stage": True, "gdn_head_major": True}}
                ),
            ),
            True,
        ),
    ]
    if T > 1:
        modes.append(
            (
                "gpr",
                mk.with_cache(ex_split, fused_pool_io=False, pool_gather=True),
                False,
            )
        )

    for label, ex_m, use_fused in modes:
        step, info_m = mk.PallasWriter().make_step(
            mesh, spec_f32, ex_m, donate=use_fused, fused_pool_io=use_fused
        )
        assert info_m["use_fused"] is use_fused, label
        assert info_m["use_gpr"] is (label == "gpr"), label
        c_in = tuple(jax.device_put(np.asarray(r).copy(), tp) for r in raw_clean)
        p_in = tuple(jax.device_put(np.asarray(r).copy(), tp) for r in raw_poison)
        hid_c, out_c = step(pos, hid_mk, w_mk, c_in, pt, ckpt, roff)
        hid_p, out_p = step(pos, hid_mk, w_mk, p_in, pt, ckpt, roff)
        assert np.isfinite(np.asarray(hid_p, np.float32)).all(), label
        np.testing.assert_array_equal(
            np.asarray(hid_p).view(np.uint8),
            np.asarray(hid_c).view(np.uint8),
            err_msg=f"{label}: hidden mismatch with poisoned stale conv rows",
        )
        st_c = pool.pool_to_states(lay, ex_split, out_c, pt, ckpt)
        st_p = pool.pool_to_states(lay, ex_split, out_p, pt, ckpt)
        for k in ("kc", "vc", "conv", "rec"):
            assert np.isfinite(np.asarray(st_p[k], np.float32)).all(), f"{label}:{k}"
            np.testing.assert_array_equal(
                np.asarray(st_p[k]).view(np.uint8),
                np.asarray(st_c[k]).view(np.uint8),
                err_msg=f"{label}:{k} mismatch with poisoned stale conv rows",
            )


@pytest.mark.parametrize("conv_dtype", [BF16, F32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "T,fused,gpr",
    [(1, True, False), (4, True, False), (4, False, True)],
)
def test_conv_non_interpret_traces_cleanly(
    T: int, fused: bool, gpr: bool, conv_dtype: Any
):
    """Non-interpret (`interpret=False`) jaxpr tracing of `FUSED` (`T=1, 4`) and
    `GPR` (`T=4`) across `bf16` and `f32` conv with wide `QKV=1536`
    (`GCONV_ROWS=96`, `conv8_rows=96`) exercises the TPU `Ref.bitcast` and 2-DMA
    `gpr_conv_copy` branches cleanly (device Mosaic lowering + execution is
    covered by `test_fused_pool_io_tpu.py`)."""
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))
    g = _wide_conv_geom()
    B = 2 * T
    ex = mk.ExecSpec(
        R=2,
        B=B,
        T=T,
        ctx=256,
        stack_e=True,
        interpret=False,
        tiles=replace(TINY_TILES, bank=512 << 10),
        cache=mk.CacheExec(
            paged_kv=True,
            num_pages=192,
            fused_pool_io=fused,
            pool_gather=gpr,
        ),
    )
    ex_host = mk.with_cache(
        replace(ex, interpret=True), fused_pool_io=False, pool_gather=False
    )
    spec_base = synth_hybrid_spec(g, R=2)
    pos, hid_l, w_l, st_l = mk.random_spec_inputs(
        jax.random.key(500 + T + int(fused)), spec_base, ex_host
    )
    _, hid_mk, w_mk, _ = mk.pack_spec_inputs(
        pos, hid_l, w_l, st_l, spec_base, ex_host, mesh=mesh
    )
    spec_dt, lay = _with_conv_dtype(spec_base, conv_dtype)
    pt, ckpt, roff, _ = pool.allocate_test_layout(lay, ex_host)
    raw_init = tuple(
        jax.device_put(r, tp)
        for r in pool.states_to_pool(lay, ex_host, st_l, pt, ckpt, roff)
    )
    step, info = mk.PallasWriter().make_step(
        mesh, spec_dt, ex, donate=fused, fused_pool_io=fused
    )
    assert info["use_fused"] is fused
    assert info["use_gpr"] is gpr
    jaxpr = jax.make_jaxpr(
        lambda p, h, w, r, pt_i, ck_i, ro_i: step(p, h, w, r, pt_i, ck_i, ro_i)
    )(pos, hid_mk, w_mk, raw_init, pt, ckpt, roff)
    jaxpr_str = str(jaxpr)
    assert "decode_megakernel" in jaxpr_str
    assert "gather_gdn_states" not in jaxpr_str


def test_model_spec_with_layout_clones_gdn_module_without_mutating_shared_pool_layout():
    """Replacing `cache_backend` with a new `PoolLayout` (or calling
    `ModelSpec.with_layout`) clones `GDNModule` instances so building a bf16
    step after an f32 step never mutates the f32 spec's `GDNModule.pool_layout`.
    """
    mesh = _cpu_mesh(2)
    g = _tiny_geom()
    ex = mk.with_cache(_tiny_exec(R=2, B=4, T=2), pool_gather=True)
    spec_bf16 = synth_hybrid_spec(g, R=2)
    spec_f32, f32_lay = _with_conv_dtype(spec_bf16, jnp.float32)
    spec_f32_direct = spec_bf16.with_layout(f32_lay)
    gdn_direct = next(m for m in spec_f32_direct.all_modules() if m.name == "gdn")
    assert gdn_direct is not next(m for m in spec_bf16.all_modules() if m.name == "gdn")
    assert gdn_direct.pool_layout is f32_lay

    gdn_bf16 = next(m for m in spec_bf16.all_modules() if m.name == "gdn")
    gdn_f32 = next(m for m in spec_f32.all_modules() if m.name == "gdn")
    assert gdn_bf16 is not gdn_f32
    assert gdn_bf16.pool_layout is spec_bf16.cache_backend.layout
    assert gdn_f32.pool_layout is f32_lay
    assert jnp.dtype(gdn_bf16.conv_dtype(g)) == jnp.dtype(jnp.bfloat16)
    assert jnp.dtype(gdn_f32.conv_dtype(g)) == jnp.dtype(jnp.float32)

    mk.PallasWriter().make_step(mesh, spec_f32, ex)
    mk.PallasWriter().make_step(mesh, spec_bf16, ex)
    assert gdn_f32.pool_layout is f32_lay
    assert jnp.dtype(gdn_f32.conv_dtype(g)) == jnp.dtype(jnp.float32)
    assert jnp.dtype(gdn_bf16.conv_dtype(g)) == jnp.dtype(jnp.bfloat16)
