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
"""CPU unit tests for 32-rank TPxDCP megakernel runtime admission and mesh gates."""

from __future__ import annotations

import dataclasses
import logging
import types
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

from vllm_torchtpu.kernels.experimental.batched_rpa.utils import cp_local_cache_len
from vllm_torchtpu.megakernel import (
    kernel,
    make_pooled_step,
    pack_inputs,
    planner,
    random_logical_inputs,
)
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import qwen35, qwen35_pool, qwen38_moe
from vllm_torchtpu.megakernel.models.qwen35_profile import (
    DEFAULT_HF_CONFIG,
    TINY_TILES,
    baseline_tiles,
)
from vllm_torchtpu.megakernel.models.qwen38_moe_profile import (
    DEFAULT_HF_CONFIG as QWEN38_HF_CONFIG,
)
from vllm_torchtpu.megakernel.models.qwen38_moe_profile import (
    tp32_dcp8_tiles,
)
from vllm_torchtpu.megakernel.pool.config import (
    POOL_STATE_NAME,
    cache_exec,
    check_dcp_interleave,
    with_cache,
)
from vllm_torchtpu.megakernel_runtime import driver as driver_mod

QWEN38_NESTED_HF_CONFIG: dict[str, Any] = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "text_config": QWEN38_HF_CONFIG,
}


def _fake_mesh(r_cnt: int, *, interleaved_hosts: bool = False) -> Any:
    devs = []
    for r in range(r_cnt):
        pid = (r % (r_cnt // 8)) if (interleaved_hosts and r_cnt >= 16) else (r // 8)
        devs.append(
            types.SimpleNamespace(
                id=r,
                process_index=pid,
                device_kind="TPU7x",
            )
        )
    arr = np.empty((r_cnt,), dtype=object)
    for r, d in enumerate(devs):
        arr[r] = d
    return types.SimpleNamespace(devices=arr, axis_names=("tp",))


def _fake_runner(
    *,
    hf_cfg: dict[str, Any] = DEFAULT_HF_CONFIG,
    r_cnt: int = 8,
    pcp: int = 1,
    dcp: int = 1,
    replicated_dcp_kv: bool = False,
    interleaved_hosts: bool = False,
    num_layers: int | None = None,
) -> Any:
    tc = hf_cfg.get("text_config")
    root = tc if tc is not None else hf_cfg
    l_cnt = int(root["num_hidden_layers"]) if num_layers is None else num_layers
    e_cnt = int(root["num_experts"])
    routed = types.SimpleNamespace(w13_weight=torch.empty((e_cnt // r_cnt, 16, 16)))
    mlp = types.SimpleNamespace(experts=routed)
    layers = [types.SimpleNamespace(mlp=mlp) for _ in range(l_cnt)]
    backbone = types.SimpleNamespace(
        layers=layers,
        norm=types.SimpleNamespace(),
        aux_hidden_state_layers=(),
        embed_tokens=types.SimpleNamespace(),
    )
    model = types.SimpleNamespace(
        model=backbone,
        config=types.SimpleNamespace(**hf_cfg),
    )
    mc = types.SimpleNamespace(hf_config=hf_cfg)
    pc = types.SimpleNamespace(
        tensor_parallel_size=r_cnt,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
        cp_kv_cache_interleave_size=128,
    )
    vc = types.SimpleNamespace(
        model_config=mc,
        parallel_config=pc,
        additional_config={"megakernel": {"enabled": True, "mode": "ep"}},
    )
    return types.SimpleNamespace(
        model=model,
        model_config=mc,
        parallel_config=pc,
        vllm_config=vc,
        mesh=_fake_mesh(r_cnt, interleaved_hosts=interleaved_hosts),
        _unified_kv_layout=True,
        supports_mm_inputs=False,
        lora_config=None,
        _pp_is_first=True,
        _pp_is_last=True,
        _pcp_enabled=(pcp > 1),
        _dcp_enabled=(dcp > 1),
        _mk_replicated_dcp_kv=replicated_dcp_kv,
    )


def test_baseline_tiles_r32_selects_hypercube_hier3():
    """At R=32, baseline_tiles and tp32_dcp8_tiles use hier=3 (rsag=False)."""
    for b in (1, 4, 8, 16, 24, 32):
        t8 = baseline_tiles(b, R=8)
        t32 = baseline_tiles(b, R=32)
        if t8.hier != 0:
            assert t32.hier == 3
            assert not t32.rsag
        else:
            assert t32.hier == 0
        t_dcp = tp32_dcp8_tiles(b, R=32)
        if b <= 16:
            assert t_dcp.hier == 3
            assert not t_dcp.rsag
    for b in (1, 8, 16):
        t4 = baseline_tiles(b, R=4)
        assert t4.hier == 0


def test_validate_runner_hf_config_admits_qwen35_and_qwen38():
    """Driver config validation admits both Qwen3.5 (60L) and Qwen3.8 (92L) configs."""
    r35 = _fake_runner(hf_cfg=DEFAULT_HF_CONFIG, r_cnt=8)
    r38 = _fake_runner(hf_cfg=QWEN38_HF_CONFIG, r_cnt=32)
    assert driver_mod.MegakernelDriver._validate_runner_hf_config(
        r35, model_def=qwen35.Qwen35
    )
    assert driver_mod.MegakernelDriver._validate_runner_hf_config(
        r38, model_def=qwen38_moe.Qwen38Moe
    )
    assert not driver_mod.MegakernelDriver._validate_runner_hf_config(
        r38, model_def=qwen35.Qwen35
    )
    assert not driver_mod.MegakernelDriver._validate_runner_hf_config(
        r35, model_def=qwen38_moe.Qwen38Moe
    )


@pytest.mark.parametrize(
    "hf_cfg,r_cnt,pcp,dcp,replicated_kv,interleaved,expect_admitted",
    [
        (DEFAULT_HF_CONFIG, 8, 1, 1, False, False, True),
        (DEFAULT_HF_CONFIG, 32, 1, 1, False, False, True),
        (QWEN38_HF_CONFIG, 32, 1, 1, False, False, True),
        (QWEN38_NESTED_HF_CONFIG, 32, 1, 1, False, False, True),
        (QWEN38_HF_CONFIG, 32, 1, 8, True, False, True),
        (QWEN38_HF_CONFIG, 32, 1, 8, False, False, True),
        (QWEN38_HF_CONFIG, 32, 1, 4, True, False, False),
        (QWEN38_HF_CONFIG, 32, 2, 1, True, False, False),
        (QWEN38_HF_CONFIG, 24, 1, 1, False, False, False),
        (QWEN38_HF_CONFIG, 32, 1, 8, True, True, False),
    ],
)
def test_maybe_create_tpdcp_admission_matrix(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    hf_cfg: dict[str, Any],
    r_cnt: int,
    pcp: int,
    dcp: int,
    replicated_kv: bool,
    interleaved: bool,
    expect_admitted: bool,
):
    """Verify maybe_create admission across R, PCP, DCP, KV, and mesh order."""
    monkeypatch.setattr(
        driver_mod,
        "logger",
        logging.getLogger("vllm_torchtpu.megakernel_runtime.driver"),
    )
    monkeypatch.setattr(driver_mod, "_get_pallas_mesh_fn", lambda: None)
    monkeypatch.setattr(driver_mod, "_tp_rank", lambda: 0)
    monkeypatch.setattr(
        driver_mod.MegakernelDriver,
        "_prepare_and_pack_weights",
        lambda self, **kw: {"w_flat": []},
    )

    runner = _fake_runner(
        hf_cfg=hf_cfg,
        r_cnt=r_cnt,
        pcp=pcp,
        dcp=dcp,
        replicated_dcp_kv=replicated_kv,
        interleaved_hosts=interleaved,
    )
    with caplog.at_level(logging.WARNING):
        res = driver_mod.maybe_create(runner)
    if expect_admitted:
        assert isinstance(res, driver_mod.MegakernelDriver)
        num_layers = int((hf_cfg.get("text_config") or hf_cfg)["num_hidden_layers"])
        if num_layers == 92:
            assert res.spec.name == "qwen3_8_moe"
            assert res.spec.cache_backend.layout.rec_pages_per_slot == 4
        else:
            assert res.spec.name == "qwen3_5"
    else:
        assert res is None


def test_check_dcp_interleave():
    """C1: check_dcp_interleave accepts matching interleave, rejects mismatched."""
    check_dcp_interleave(128, 128)
    with pytest.raises(MegakernelConfigError, match="cp_kv_cache_interleave_size"):
        check_dcp_interleave(1, 128)
    with pytest.raises(MegakernelConfigError, match="cp_kv_cache_interleave_size"):
        check_dcp_interleave(384, 128)


def test_dcp_sharded_rejects_split_off():
    """C2: dcp_kv_sharded with G > 1 rejects split=False."""
    g = qwen35.Qwen35Geometry(
        L=2,
        att_every=2,
        H=1024,
        NH=8,
        NKV=2,
        HD=256,
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
        TINY_TILES,
        bank=256 << 10,
        hier=3,
        mb8=2,
        inplace=False,
    ).with_modules({"gqa": {"split": False}})
    ex = with_cache(
        ExecSpec(B=2, T=1, R=8, ctx=1024, tiles=tiles, interpret=True),
        paged_kv=True,
        num_pages=128,
        page_size=128,
        max_pages_per_seq=2,
        dcp_kv_sharded=True,
        dcp_size=4,
    )
    ms = g.to_model_spec(R=8)
    with pytest.raises(
        MegakernelConfigError,
        match="dcp_kv_sharded needs GQA split=True and paged_kv=True",
    ):
        planner.plan(ms, ex)


def test_dcp_sharded_pool_matches_stock_local_cache_len():
    """Worklist per-rank token count matches stock cp_local_cache_len."""
    page_size = 128
    G = 8
    rng = np.random.default_rng(1234)
    lengths = rng.integers(1, 8192, size=30)
    for L in lengths:
        nb = int(-(-L // page_size))
        for r in range(G):
            cnt = max(nb - r + G - 1, 0) // G
            tokens_wl = sum(
                max(0, min(page_size, int(L) - (r + j * G) * page_size))
                for j in range(cnt)
            )
            expected = int(cp_local_cache_len(int(L), G, r, page_size))
            assert tokens_wl == expected, (
                f"L={L} rank={r}: wl={tokens_wl} != stock={expected}"
            )

    # Negative control: an off-by-one-rank owner map must disagree
    partial_L = 3 * page_size + 42
    r = 3
    shifted_tokens = sum(
        max(0, min(page_size, partial_L - (((r + 1) % G) + j * G) * page_size))
        for j in range(cnt)
    )
    assert shifted_tokens != int(cp_local_cache_len(partial_L, G, r, page_size))


@pytest.mark.parametrize("fused_pool_io", [False, True])
@pytest.mark.parametrize("nkv", [1, 2])  # G = 8 // nkv; nkv=2 makes G != R
def test_dcp_sharded_pool_roundtrip(fused_pool_io: bool, nkv: int):
    """Verify pool round trip and step vs reference_step under dcp_kv_sharded=True."""
    F32 = jnp.float32

    g = qwen35.Qwen35Geometry(
        L=2,
        att_every=2,
        H=1024,
        NH=8,
        NKV=nkv,
        HD=256,
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
    tiles = (
        dataclasses.replace(
            TINY_TILES,
            bank=256 << 10,
            hier=3,
            mb8=2,
            inplace=False,
        )
        .with_modules({"gqa": {"split": False}})
        .with_modules({"gqa": {"split": True}})
    )
    ex = with_cache(
        ExecSpec(B=2, T=1, R=8, ctx=1024, tiles=tiles, interpret=True),
        paged_kv=True,
        num_pages=128,
        page_size=128,
        max_pages_per_seq=2,
        dcp_kv_sharded=True,
        dcp_size=8 // nkv,
        fused_pool_io=fused_pool_io,
    )
    ms = g.to_model_spec(R=8)
    p = planner.plan(ms, ex)
    ap = p.module_params["gqa"]
    assert ap.DCP_SHARDED is True
    assert ap.BLK == 128
    assert ap.SUB == 128

    pt, ckpt, roff, _ = qwen35_pool.default_pool_layout(g, ex)
    pos, hid_l, w_l, st_l = random_logical_inputs(jax.random.PRNGKey(0), g, ex)
    pos = 900 + jnp.arange(ex.NS, dtype=jnp.int32)
    pools = qwen35_pool.states_to_pool(g, ex, st_l, pt, ckpt, roff)
    rec_st = qwen35_pool.pool_to_states(g, ex, pools, pt, ckpt)
    assert jnp.allclose(
        rec_st["kc"].astype(F32), st_l["kc"].astype(F32), atol=5e-2, rtol=5e-2
    )
    assert jnp.allclose(
        rec_st["vc"].astype(F32), st_l["vc"].astype(F32), atol=5e-2, rtol=5e-2
    )

    devs = jax.devices("cpu")
    if len(devs) < 8:
        pytest.skip(
            "Test requires 8 CPU devices "
            "(set XLA_FLAGS=--xla_force_host_platform_device_count=8)"
        )

    mesh = Mesh(devs[:8], ("tp",))
    ref_hid, _ = qwen35.reference_step(
        g, ex, pos, hid_l, w_l, st_l, paged_kv=True, pool_io=True
    )
    pos_mk, hid_mk, w_mk, st_mk = pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    step, _ = qwen35.make_pooled_step(
        mesh, g, ex, fused_pool_io=fused_pool_io, donate_pools=fused_pool_io
    )
    step_pools = (
        tuple(jnp.array(p, copy=True) for p in st_mk[POOL_STATE_NAME])
        if fused_pool_io
        else st_mk[POOL_STATE_NAME]
    )
    out_hid, _ = step(
        pos_mk,
        hid_mk,
        w_mk,
        step_pools,
        st_mk["page_table"],
        ckpt,
        roff,
    )
    hid_r = kernel.unfold(out_hid, 8)[:, : ex.B]
    assert jnp.allclose(
        hid_r[0].astype(jnp.float32),
        ref_hid.astype(jnp.float32),
        atol=5e-2,
        rtol=5e-2,
    )
    assert not jnp.allclose(
        hid_r[0].astype(jnp.float32),
        hid_l.astype(jnp.float32),
        atol=5e-2,
        rtol=5e-2,
    )

    # Negative control: permuted page table with distinct pages exceeds tolerance
    bad_pt = (st_mk["page_table"] + 17) % cache_exec(ex).num_pages
    bad_pools = (
        tuple(jnp.array(p, copy=True) for p in st_mk[POOL_STATE_NAME])
        if fused_pool_io
        else st_mk[POOL_STATE_NAME]
    )
    out_bad, _ = step(
        pos_mk,
        hid_mk,
        w_mk,
        bad_pools,
        bad_pt,
        ckpt,
        roff,
    )
    hid_bad = kernel.unfold(out_bad, 8)[:, : ex.B]
    assert not jnp.allclose(
        hid_bad[0].astype(jnp.float32),
        ref_hid.astype(jnp.float32),
        atol=5e-2,
        rtol=5e-2,
    )


def test_dcp_kv_pool_subgroup_rank_indexing():
    """Verify DCP KV subgroup rank indexing (r % G) for G < R and straddling T=2."""
    devs = jax.devices("cpu")
    if len(devs) < 8:
        pytest.skip(
            "Test requires 8 CPU devices "
            "(set XLA_FLAGS=--xla_force_host_platform_device_count=8)"
        )

    g = qwen38_moe.Qwen38MoeGeometry(
        L=2,
        att_every=2,
        H=1024,
        NH=8,
        NKV=2,
        HD=256,
        ROT=64,
        KH=8,
        VH=16,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=512,
        moe_tp=False,
    )
    tiles = dataclasses.replace(
        TINY_TILES,
        bank=256 << 10,
        hier=3,
        mb8=2,
        inplace=False,
    ).with_modules({"gqa": {"split": True}})
    ex_split = with_cache(
        ExecSpec(B=4, T=2, R=8, ctx=1024, tiles=tiles, interpret=True),
        paged_kv=True,
        num_pages=128,
        page_size=128,
        max_pages_per_seq=2,
        dcp_kv_sharded=True,
        dcp_size=4,
        fused_pool_io=False,
    )
    ex_fused = with_cache(ex_split, fused_pool_io=True)
    pt, ckpt, roff, _ = qwen35_pool.default_pool_layout(g, ex_split)
    _, hid_l, w_l, st_l = random_logical_inputs(jax.random.PRNGKey(7), g, ex_split)
    # Straddle page boundaries at blk >= pt_cols (2): blk = 1..2 and blk = 2..3
    pos = jnp.array([255, 256, 383, 384], dtype=jnp.int32)

    mesh = Mesh(devs[:8], ("tp",))
    ref_hid, ref_st = qwen38_moe.Qwen38Moe.reference_step(
        g, ex_split, pos, hid_l, w_l, st_l, paged_kv=True, pool_io=True
    )
    pos_mk, hid_mk, w_mk, st_mk = pack_inputs(
        pos, hid_l, w_l, st_l, g, ex_split, mesh=mesh, build_default_pool=True
    )
    ms = g.to_model_spec(R=ex_split.R)
    step_split, _ = make_pooled_step(
        mesh, ms, ex_split, fused_pool_io=False, donate_pools=False
    )
    step_fused, _ = make_pooled_step(
        mesh, ms, ex_fused, fused_pool_io=True, donate_pools=True
    )
    out_hid_split, out_pools_split = step_split(
        pos_mk,
        hid_mk,
        w_mk,
        st_mk[POOL_STATE_NAME],
        st_mk["page_table"],
        ckpt,
        roff,
    )
    fused_pools_in = tuple(jnp.array(p, copy=True) for p in st_mk[POOL_STATE_NAME])
    out_hid_fused, out_pools_fused = step_fused(
        pos_mk,
        hid_mk,
        w_mk,
        fused_pools_in,
        st_mk["page_table"],
        ckpt,
        roff,
    )
    for out_hid, out_pools in (
        (out_hid_split, out_pools_split),
        (out_hid_fused, out_pools_fused),
    ):
        hid_r = kernel.unfold(out_hid, 8)[:, : ex_split.B]
        for r_chk in (0, 4):
            assert jnp.allclose(
                hid_r[r_chk].astype(jnp.float32),
                ref_hid.astype(jnp.float32),
                atol=5e-2,
                rtol=5e-2,
            )
        rec_after = qwen35_pool.pool_to_states(g, ex_split, out_pools, pt, ckpt)
        for p_tok in (255, 256, 383, 384):
            s_idx = 0 if p_tok < 300 else 1
            for key in ("kc", "vc"):
                got = rec_after[key][:, s_idx, :, p_tok].astype(jnp.float32)
                exp = ref_st[key][:, s_idx, :, p_tok].astype(jnp.float32)
                assert jnp.allclose(
                    got / jnp.linalg.norm(got),
                    exp / jnp.linalg.norm(exp),
                    atol=5e-2,
                    rtol=5e-2,
                )
        for s_idx, p_untouched in ((0, 100), (1, 300)):
            assert jnp.array_equal(
                rec_after["kc"][:, s_idx, :, p_untouched],
                st_l["kc"][:, s_idx, :, p_untouched].astype(rec_after["kc"].dtype),
            )
            assert jnp.array_equal(
                rec_after["vc"][:, s_idx, :, p_untouched],
                st_l["vc"][:, s_idx, :, p_untouched].astype(rec_after["vc"].dtype),
            )
