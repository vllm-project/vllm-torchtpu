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
"""On-TPU (`interpret=False`) tests for `fused_pool_io=True` vs split pool I/O."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tests.megakernel.conftest import host_device, tpu_mesh
from tests.megakernel.core.megakernel_synth import TINY_TILES, synth_hybrid_spec
from vllm_torchtpu.megakernel import core as mk
from vllm_torchtpu.megakernel.core import pool
from vllm_torchtpu.megakernel.core.cache_hlo import check_pooled_step_hlo
from vllm_torchtpu.megakernel.core.ops import BF16, F32, I32
from vllm_torchtpu.megakernel.core.reference import spec_reference_step


def _hybrid_geom(
    *,
    L: int = 4,
    att_every: int = 4,
    R: int = 2,
    KH: int = 2,
    VH: int = 4,
) -> Any:
    return mk.ModelGeometry(
        L=L,
        att_every=att_every,
        H=256,
        NH=max(2, 2 * R),
        NKV=min(2, R),
        HD=256,
        ROT=64,
        KH=max(KH, R),
        VH=max(VH, 2 * max(KH, R)),
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        e_slots=0,
        theta=10_000_000.0,
        zero_centered_norm=True,
    )


def _with_conv_dtype(
    spec: mk.ModelSpec, conv_dtype: Any
) -> tuple[mk.ModelSpec, pool.PoolLayout]:
    assert spec.cache_backend is not None
    lay = replace(spec.cache_backend.layout, conv_dtype=conv_dtype)
    pb = spec.cache_backend.with_layout(lay)
    assert pb.supports_fused_io
    return replace(spec, cache_backend=pb), lay


def _run_split_and_fused_on_tpu(
    mesh: Mesh,
    spec_base: mk.ModelSpec,
    g: mk.ModelGeometry,
    *,
    T: int,
    B: int,
    conv_dtype: Any,
    seed: int,
    pos_override: jax.Array | None = None,
    conv_stage: bool = False,
    num_pages: int = 256,
) -> dict[str, Any]:
    """Run split (`fused_pool_io=False`) and fused (`fused_pool_io=True`) steps on
    identical inputs on TPU (`interpret=False`) and return unpacked results."""
    # Fused GDN recurrence always runs head-major; `wide_rows=0` activates
    # `GCH=True` (`gdn_rec_chunked`) on split so `gdn_head_major=True` takes effect.
    tiles = replace(TINY_TILES, bank=512 << 10, wide_rows=0).with_modules(
        {
            "gqa": {"kv_blk": 128, "nkv": 2},
            "gdn": {"conv_stage": conv_stage, "gdn_head_major": True},
        }
    )
    ex_split = mk.ExecSpec(
        R=mesh.devices.size,
        B=B,
        T=T,
        ctx=256,
        stack_e=True,
        interpret=False,
        tiles=tiles,
        cache=mk.CacheExec(
            paged_kv=True,
            num_pages=num_pages,
            fused_pool_io=False,
            pool_gather=False,
        ),
    )
    ex_fused = mk.with_cache(ex_split, fused_pool_io=True, pool_gather=False)
    ex_ref = replace(ex_split, interpret=True)

    spec_dt, lay = _with_conv_dtype(spec_base, conv_dtype)
    lay.validate(ex_split)

    host_dev = host_device(mesh.devices.flat[0])
    with jax.default_device(host_dev):
        pos, hid_l, w_l, st_l = mk.random_spec_inputs(
            jax.random.key(seed), spec_dt, ex_ref
        )
        if pos_override is not None:
            pos = jnp.asarray(pos_override, dtype=I32)
        # Keep non-zero low f32 mantissa bits in conv so f32 vs bf16 differs.
        conv_full = (
            jax.random.normal(jax.random.key(seed + 777), st_l["conv"].shape, dtype=F32)
            * 0.1
        )
        assert not np.array_equal(
            np.asarray(conv_full, dtype=np.float32),
            np.asarray(conv_full.astype(BF16), dtype=np.float32),
        )
        is_f32_conv = jnp.dtype(conv_dtype) == jnp.dtype(F32)
        if is_f32_conv:
            st_l = dict(st_l, conv=conv_full)
        else:
            st_l = dict(st_l, conv=conv_full.astype(BF16).astype(F32))

        ref_hid, ref_st = spec_reference_step(
            spec_dt,
            g,
            ex_ref,
            pos,
            hid_l,
            w_l,
            st_l,
            paged_kv=True,
            pool_io=not is_f32_conv,
        )
        pos_mk, hid_mk, w_mk, _ = mk.pack_spec_inputs(
            pos, hid_l, w_l, st_l, spec_dt, ex_ref
        )
        pt, ckpt, roff, min_np = pool.allocate_test_layout(lay, ex_ref)
        assert num_pages >= min_np, (num_pages, min_np)
        raw_host = [
            np.asarray(r).copy()
            for r in pool.states_to_pool(lay, ex_ref, st_l, pt, ckpt, roff)
        ]

    tp = NamedSharding(mesh, P("tp"))
    rep = NamedSharding(mesh, P())
    pos_tpu = jax.device_put(pos_mk, rep)
    hid_tpu = jax.device_put(hid_mk, tp)
    w_tpu = {
        k: tuple(jax.device_put(x, tp) for x in v)
        if isinstance(v, tuple)
        else jax.device_put(v, tp)
        for k, v in w_mk.items()
    }
    pt_tpu = jax.device_put(pt, rep)
    ckpt_tpu = jax.device_put(ckpt, rep)
    roff_tpu = jax.device_put(roff, rep)

    raw_split_in = tuple(jax.device_put(r.copy(), tp) for r in raw_host)
    step_split, info_split = mk.PallasWriter().make_step(
        mesh, spec_dt, ex_split, donate=False, fused_pool_io=False
    )
    assert not info_split["use_fused"] and not info_split["use_gpr"]
    assert info_split["plan"].module_params["gdn"].GCH
    hid_split, raw_split = step_split(
        pos_tpu, hid_tpu, w_tpu, raw_split_in, pt_tpu, ckpt_tpu, roff_tpu
    )
    jax.block_until_ready((hid_split, raw_split))

    raw_fused_in = tuple(jax.device_put(r.copy(), tp) for r in raw_host)
    step_fused, info_fused = mk.PallasWriter().make_step(
        mesh, spec_dt, ex_fused, donate=True, fused_pool_io=True
    )
    assert info_fused["use_fused"]
    compiled_fused = step_fused.lower(
        pos_tpu, hid_tpu, w_tpu, raw_fused_in, pt_tpu, ckpt_tpu, roff_tpu
    ).compile()
    compiled_hlo = compiled_fused.as_text()
    assert compiled_hlo is not None
    ok_hlo, msg_hlo = check_pooled_step_hlo(
        compiled_hlo, layout=lay, fused_pool_io=True
    )
    assert ok_hlo, msg_hlo

    # The compiled executable takes the step's internal arguments, so run the
    # public step; `compiled_fused` above is only for the HLO check.
    hid_fused, raw_fused = step_fused(
        pos_tpu, hid_tpu, w_tpu, raw_fused_in, pt_tpu, ckpt_tpu, roff_tpu
    )
    jax.block_until_ready((hid_fused, raw_fused))

    raw_split_np = [np.asarray(r).copy() for r in raw_split]
    raw_fused_np = [np.asarray(r).copy() for r in raw_fused]
    with jax.default_device(host_dev):
        st_init = pool.pool_to_states(
            lay, ex_ref, [jnp.asarray(r) for r in raw_host], pt, ckpt
        )
        st_split = pool.pool_to_states(
            lay, ex_ref, [jnp.asarray(r) for r in raw_split_np], pt, ckpt
        )
        st_fused = pool.pool_to_states(
            lay, ex_ref, [jnp.asarray(r) for r in raw_fused_np], pt, ckpt
        )

    return {
        "lay": lay,
        "pos": np.asarray(pos, dtype=np.int32),
        "hid_split": np.asarray(
            mk.unfold(hid_split, ex_split.R)[:, :B], dtype=np.float32
        ),
        "hid_fused": np.asarray(
            mk.unfold(hid_fused, ex_fused.R)[:, :B], dtype=np.float32
        ),
        "raw_split": raw_split_np,
        "raw_fused": raw_fused_np,
        "ref_hid": np.asarray(ref_hid, dtype=np.float32),
        "ref_st": ref_st,
        "st_init": st_init,
        "st_split": st_split,
        "st_fused": st_fused,
    }


def _assert_fused_matches_split(res: dict[str, Any], *, conv_dtype: Any) -> None:
    # Hidden output parity (fused vs split, and vs reference).
    np.testing.assert_array_equal(res["hid_fused"], res["hid_split"])
    np.testing.assert_allclose(
        res["hid_fused"][0], res["ref_hid"], rtol=3e-2, atol=3e-2
    )

    # Full raw pool byte equality across every pool (including unreferenced pages,
    # null pages, and conv rows past `conv_rows`).
    assert len(res["raw_fused"]) == len(res["raw_split"])
    for pool_idx, (rf, rs) in enumerate(zip(res["raw_fused"], res["raw_split"])):
        np.testing.assert_array_equal(
            np.asarray(rf).view(np.uint8),
            np.asarray(rs).view(np.uint8),
            err_msg=f"fused vs split raw pool[{pool_idx}] bytes mismatch",
        )

    # Pooled KV (`kc`, `vc`) and GDN (`conv`, `rec`) states match split bit-for-bit.
    for key in ("kc", "vc", "conv", "rec"):
        f_u8 = np.asarray(res["st_fused"][key]).view(np.uint8)
        s_u8 = np.asarray(res["st_split"][key]).view(np.uint8)
        i_u8 = np.asarray(res["st_init"][key]).view(np.uint8)
        np.testing.assert_array_equal(
            f_u8, s_u8, err_msg=f"fused vs split pool state mismatch on {key!r}"
        )
        assert not np.array_equal(f_u8, i_u8), f"pool state {key!r} was not updated"

    if jnp.dtype(conv_dtype) == jnp.dtype(F32):
        conv_f32 = np.asarray(res["st_fused"]["conv"], dtype=np.float32)
        assert not np.array_equal(
            conv_f32, np.asarray(res["st_fused"]["conv"].astype(BF16), dtype=np.float32)
        ), "f32 conv lost sub-bf16 mantissa precision in fused pool I/O"


@pytest.mark.tpu_test
@pytest.mark.parametrize("T", [1, 4], ids=["t1", "t4"])
@pytest.mark.parametrize(
    ("conv_dtype", "conv_stage"),
    [(BF16, False), (F32, False), (F32, True)],
    ids=["bf16_conv", "f32_conv", "f32_conv_staged"],
)
def test_fused_vs_split_pool_io_bf16_and_f32_conv_on_tpu(
    T: int, conv_dtype: Any, conv_stage: bool
) -> None:
    """`fused_pool_io=True` matches split pool I/O on TPU (`interpret=False`)
    across `bf16` and `f32` conv (`conv_stage` off and on) at `T=1` and `T=4`."""
    mesh = tpu_mesh(R=2)
    g = _hybrid_geom(L=4, att_every=4, R=2, KH=2, VH=4)
    spec = synth_hybrid_spec(g, R=2)
    B = 2 * T
    res = _run_split_and_fused_on_tpu(
        mesh,
        spec,
        g,
        T=T,
        B=B,
        conv_dtype=conv_dtype,
        conv_stage=conv_stage,
        seed=500
        + 10 * T
        + int(jnp.dtype(conv_dtype) == jnp.dtype(F32))
        + 2 * int(conv_stage),
    )
    _assert_fused_matches_split(res, conv_dtype=conv_dtype)


@pytest.mark.tpu_test
@pytest.mark.parametrize("conv_dtype", [BF16, F32], ids=["bf16_conv", "f32_conv"])
def test_fused_vs_split_page_crossing_verify_on_tpu(conv_dtype: Any) -> None:
    """`T=4` verify step where sequence 0 crosses a `128`-token KV page boundary
    (`pos = [126, 127, 128, 129]`, exercising slow per-token KV RMW `kv_mode == 2`
    alongside sequence 1 on fast KV RMW `kv_mode == 1`) matches split on TPU."""
    mesh = tpu_mesh(R=2)
    g = _hybrid_geom(L=4, att_every=4, R=2, KH=2, VH=4)
    spec = synth_hybrid_spec(g, R=2)
    assert spec.cache_backend is not None
    page_size = spec.cache_backend.layout.page_size
    assert page_size == 128

    T, NS = 4, 2
    B = NS * T
    # Seq 0 spans [126, 127] (page 0) -> [128, 129] (page 1); seq 1 stays in page 0.
    pos_cross = jnp.asarray(
        [page_size - 2, page_size - 1, page_size, page_size + 1, 32, 33, 34, 35],
        dtype=I32,
    )
    res = _run_split_and_fused_on_tpu(
        mesh,
        spec,
        g,
        T=T,
        B=B,
        conv_dtype=conv_dtype,
        pos_override=pos_cross,
        seed=610 + int(jnp.dtype(conv_dtype) == jnp.dtype(F32)),
    )
    _assert_fused_matches_split(res, conv_dtype=conv_dtype)

    # Check that both sides of the page boundary (126..127 and 128..129) updated.
    kc_init = np.asarray(res["st_init"]["kc"]).view(np.uint8)
    kc_fused = np.asarray(res["st_fused"]["kc"]).view(np.uint8)
    for tok_pos in (page_size - 2, page_size - 1, page_size, page_size + 1):
        assert not np.array_equal(
            kc_fused[:, :, 0, tok_pos, :], kc_init[:, :, 0, tok_pos, :]
        ), f"KV cache at page-crossing position {tok_pos} was not written"


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("L", "att_every"),
    [(4, 2), (8, 4)],
    ids=["natt2_slots1", "natt2_slots3"],
)
@pytest.mark.parametrize("T", [1, 4], ids=["t1", "t4"])
@pytest.mark.parametrize("conv_dtype", [BF16, F32], ids=["bf16_conv", "f32_conv"])
def test_fused_vs_split_multi_pool_natt_gt_1_on_tpu(
    L: int, att_every: int, T: int, conv_dtype: Any
) -> None:
    """`fused_pool_io=True` matches split on multi-pool `synth_hybrid_spec`
    geometries with `NATT = 2 > 1` (`num_pools = 2`, `slots_per_pool` in `{1, 3}`)."""
    mesh = tpu_mesh(R=2)
    g = _hybrid_geom(L=L, att_every=att_every, R=2, KH=2, VH=4)
    assert g.NATT == 2 and g.NGDN == L - 2
    spec = synth_hybrid_spec(g, R=2)
    assert spec.cache_backend is not None
    assert spec.cache_backend.layout.num_pools == 2
    assert spec.cache_backend.layout.slots_per_pool == att_every - 1

    B = 2 * T
    res = _run_split_and_fused_on_tpu(
        mesh,
        spec,
        g,
        T=T,
        B=B,
        conv_dtype=conv_dtype,
        num_pages=384,
        seed=700 + 20 * L + 5 * T + int(jnp.dtype(conv_dtype) == jnp.dtype(F32)),
    )
    assert res["st_fused"]["kc"].shape[1] == 2
    assert res["st_fused"]["rec"].shape[1] == g.NGDN
    _assert_fused_matches_split(res, conv_dtype=conv_dtype)
