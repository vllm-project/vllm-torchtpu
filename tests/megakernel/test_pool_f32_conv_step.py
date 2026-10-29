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
"""A Qwen3.5 pooled step over a pool that stores the conv state as f32.

The step reads and writes the conv window as f32 (the dtype the serving
framework's stock GDN kernel uses), on the split pool I/O path; it must match
the reference step and keep the kernel's conv state exactly (no bf16 round
trip), where a bf16-conv pool rounds it.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import pool
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _cpu_mesh(R: int) -> Mesh:
    devs = jax.devices("cpu")
    if len(devs) < R:
        pytest.skip(f"Need at least {R} CPU devices, have {len(devs)}")
    return Mesh(np.array(devs[:R]), ("tp",))


def _tiny_geom() -> qwen35.Qwen35Geometry:
    return qwen35.Qwen35Geometry(
        L=4, H=256, I=256, IS=256, E=128, TOPK=2, NH=4, NKV=2, HD=256, ROT=64,
        KH=2, VH=4, KD=128, VD=128, att_every=4, moe_tp=False, e_slots=0,
    )  # fmt: skip


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


def _with_conv_dtype(spec, dt):
    lay = replace(spec.cache_backend.layout, conv_dtype=dt)
    return replace(spec, cache_backend=spec.cache_backend.with_layout(lay)), lay


@pytest.mark.parametrize("T,B", [(1, 2), (2, 4)])
def test_f32_conv_pooled_step_matches_reference_and_keeps_conv_exact(T: int, B: int):
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))
    g = _tiny_geom()
    ex = _tiny_exec(R=2, B=B, T=T)
    pos, hid_l, w_l, st_l = mk.random_logical_inputs(jax.random.key(70 + T), g, ex)
    conv = np.asarray(st_l["conv"], np.float32)
    assert not np.array_equal(
        conv, np.asarray(st_l["conv"].astype(jnp.bfloat16), np.float32)
    )

    ref_out, ref_st = qwen35.reference_step(
        g, ex, pos, hid_l, w_l, st_l, paged_kv=True, pool_io=True
    )
    _, hid_mk, w_mk, st_mk_paged = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    _, mk_st = qwen35.make_step(mesh, g, ex, donate_states=False)[0](
        pos, hid_mk, w_mk, st_mk_paged
    )

    spec = qwen35.qwen35_spec(g)
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
            # The pool keeps the kernel's f32 conv state bit-for-bit.
            np.testing.assert_array_equal(np.asarray(got), np.asarray(kern))
        errs[name] = float(jnp.max(jnp.abs(got - ref)))
    assert errs["f32"] <= errs["bf16"], errs


def _with_f32_in_kernel_backend(spec):
    lay = replace(spec.cache_backend.layout, conv_dtype=jnp.float32)
    pb = spec.cache_backend.with_layout(lay)
    pb = replace(
        pb,
        supports_fused_io=True,
        can_pool_gather=lambda ex_p: ex_p.T > 1,
    )
    return replace(spec, cache_backend=pb), lay


def _wide_conv_geom() -> qwen35.Qwen35Geometry:
    """Per-rank `QKV = 1536` (`qkv_128 = 12`, matching Qwen3.5 TP=8 conv width):
    `GCONV_ROWS = 96 > 64` (2-DMA K+V GPR path) and `conv8_rows = 96 <= 128`."""
    return replace(_tiny_geom(), KH=4, VH=16)


@pytest.mark.parametrize("T", [1, 4])
@pytest.mark.parametrize("wide_qkv", [False, True])
def test_f32_conv_fused_matches_split_bitwise(T: int, wide_qkv: bool):
    """Fused pool I/O (`T in (1, 4)`, `conv_stage in (False, True)`) with
    `conv_dtype=f32` matches split pool I/O on hidden outputs and pool states
    (`kc`, `vc`, `conv` bit-exact; full `f32` conv precision preserved;
    `conv_stage=True` bit-exact with `conv_stage=False`)."""
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))
    g = _wide_conv_geom() if wide_qkv else _tiny_geom()
    B = 2 * T
    # The fused GDN path always runs the head-major recurrence, so the split
    # reference opts into `gdn_head_major` too.
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
    ex_fused = mk.with_cache(ex_split, fused_pool_io=True, pool_gather=False)
    ex_fused_cs = replace(
        ex_fused,
        tiles=tiles.with_modules({"gdn": {"conv_stage": True, "gdn_head_major": True}}),
    )

    pos, hid_l, w_l, st_l = mk.random_logical_inputs(
        jax.random.key(710 + 10 * T + B + 2000), g, ex_split
    )
    assert not np.array_equal(
        np.asarray(st_l["conv"], np.float32),
        np.asarray(st_l["conv"].astype(jnp.bfloat16), np.float32),
    )
    _, hid_mk, w_mk, _ = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex_split, mesh=mesh, build_default_pool=True
    )

    spec_f32, lay = _with_f32_in_kernel_backend(qwen35.qwen35_spec(g))
    pt, ckpt, roff, min_np = pool.allocate_test_layout(lay, ex_split)
    assert mk.cache_exec(ex_split).num_pages >= min_np
    raw_init = [
        jax.device_put(r, tp)
        for r in pool.states_to_pool(lay, ex_split, st_l, pt, ckpt, roff)
    ]

    step_split, info_split = mk.PallasWriter().make_step(
        mesh, spec_f32, ex_split, donate=False, fused_pool_io=False
    )
    assert not info_split["use_fused"] and not info_split["use_gpr"]
    hid_split, raw_split = step_split(
        pos, hid_mk, w_mk, tuple(raw_init), pt, ckpt, roff
    )
    st_split = pool.pool_to_states(lay, ex_split, raw_split, pt, ckpt)

    raw_fused_in = tuple(jax.device_put(np.asarray(r).copy(), tp) for r in raw_init)
    step_fused, info_fused = mk.PallasWriter().make_step(
        mesh, spec_f32, ex_fused, donate=True, fused_pool_io=True
    )
    assert info_fused["use_fused"]
    hid_fused, raw_fused = step_fused(
        pos, hid_mk, w_mk, raw_fused_in, pt, ckpt, roff
    )
    st_fused = pool.pool_to_states(lay, ex_split, raw_fused, pt, ckpt)

    raw_cs_in = tuple(jax.device_put(np.asarray(r).copy(), tp) for r in raw_init)
    step_cs, info_cs = mk.PallasWriter().make_step(
        mesh, spec_f32, ex_fused_cs, donate=True, fused_pool_io=True
    )
    assert info_cs["use_fused"]
    hid_cs, raw_cs = step_cs(pos, hid_mk, w_mk, raw_cs_in, pt, ckpt, roff)

    # conv_stage=True (f32 conv_fast VMEM staging) matches conv_stage=False bytewise.
    np.testing.assert_array_equal(
        np.asarray(hid_cs).view(np.uint8), np.asarray(hid_fused).view(np.uint8)
    )
    for r_cs, r_f in zip(raw_cs, raw_fused):
        np.testing.assert_array_equal(
            np.asarray(r_cs).view(np.uint8), np.asarray(r_f).view(np.uint8)
        )

    np.testing.assert_array_equal(np.asarray(hid_fused), np.asarray(hid_split))
    for k in ("kc", "vc", "conv"):
        np.testing.assert_array_equal(
            np.asarray(st_fused[k]), np.asarray(st_split[k]), err_msg=f"mismatch on {k}"
        )
    if T == 1:
        np.testing.assert_array_equal(
            np.asarray(st_fused["rec"]), np.asarray(st_split["rec"])
        )
        for r_f, r_s in zip(raw_fused, raw_split):
            np.testing.assert_array_equal(
                np.asarray(r_f).view(np.uint8), np.asarray(r_s).view(np.uint8)
            )
    else:
        rec_f_u16 = np.asarray(st_fused["rec"]).view(np.uint16).astype(np.int32)
        rec_s_u16 = np.asarray(st_split["rec"]).view(np.uint16).astype(np.int32)
        assert int(np.max(np.abs(rec_f_u16 - rec_s_u16))) <= T - 1
    # Full f32 conv precision is preserved (not rounded to bf16).
    conv_out = np.asarray(st_fused["conv"], np.float32)
    assert not np.array_equal(
        conv_out, np.asarray(st_fused["conv"].astype(jnp.bfloat16), np.float32)
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

    pos, hid_l, w_l, st_l = mk.random_logical_inputs(
        jax.random.key(410 + int(wide_qkv)), g, ex_split
    )
    _, hid_mk, w_mk, _ = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex_split, mesh=mesh, build_default_pool=True
    )

    spec_f32, lay = _with_f32_in_kernel_backend(qwen35.qwen35_spec(g))
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


@pytest.mark.parametrize(
    "T,fused,gpr",
    [(1, True, False), (4, True, False), (4, False, True)],
)
def test_f32_conv_non_interpret_lowers_cleanly(T: int, fused: bool, gpr: bool):
    """Non-interpret (`interpret=False`) lowering of `FUSED` (`T=1, 4`) and `GPR`
    (`T=4`) with wide `QKV=1536` (`GCONV_ROWS=96`, `conv8_rows=96`) traces the
    TPU `Ref.bitcast(F32)` and 2-DMA `gpr_conv_copy` branches cleanly."""
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
    pos, hid_l, w_l, st_l = mk.random_logical_inputs(
        jax.random.key(500 + T + int(fused)), g, ex_host
    )
    _, hid_mk, w_mk, _ = mk.pack_inputs(
        pos,
        hid_l,
        w_l,
        st_l,
        g,
        ex_host,
        mesh=mesh,
        build_default_pool=True,
    )
    spec_f32, lay = _with_f32_in_kernel_backend(qwen35.qwen35_spec(g))
    pt, ckpt, roff, _ = pool.allocate_test_layout(lay, ex_host)
    raw_init = tuple(
        jax.device_put(r, tp)
        for r in pool.states_to_pool(lay, ex_host, st_l, pt, ckpt, roff)
    )
    step, info = mk.PallasWriter().make_step(
        mesh, spec_f32, ex, donate=fused, fused_pool_io=fused
    )
    assert info["use_fused"] is fused
    assert info["use_gpr"] is gpr
    jaxpr = jax.make_jaxpr(
        lambda p, h, w, r, pt_i, ck_i, ro_i: step(p, h, w, r, pt_i, ck_i, ro_i)
    )(pos, hid_mk, w_mk, raw_init, pt, ckpt, roff)
    jaxpr_str = str(jaxpr)
    assert "decode_megakernel" in jaxpr_str
    assert "gather_gdn_states" not in jaxpr_str

