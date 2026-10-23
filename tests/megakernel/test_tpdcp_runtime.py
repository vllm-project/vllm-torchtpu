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

import logging
import types
from typing import Any

import numpy as np
import pytest
import torch

from vllm_torchtpu.megakernel.models.qwen35_profile import (
    DEFAULT_HF_CONFIG,
    baseline_tiles,
)
from vllm_torchtpu.megakernel.models.qwen38_profile import (
    DEFAULT_HF_CONFIG as QWEN38_HF_CONFIG,
)
from vllm_torchtpu.megakernel.models.qwen38_profile import (
    tp32_dcp8_tiles,
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
    assert driver_mod.Qwen35Megakernel._validate_runner_hf_config(r35)
    assert driver_mod.Qwen38Megakernel._validate_runner_hf_config(r38)
    assert not driver_mod.Qwen35Megakernel._validate_runner_hf_config(r38)
    assert not driver_mod.Qwen38Megakernel._validate_runner_hf_config(r35)


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
    import vllm.distributed as vllm_dist

    monkeypatch.setattr(vllm_dist, "get_tensor_model_parallel_rank", lambda: 0)
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
        assert res is not None
        num_layers = int((hf_cfg.get("text_config") or hf_cfg)["num_hidden_layers"])
        if num_layers == 92:
            assert isinstance(res, driver_mod.Qwen38Megakernel)
            assert res.spec.name == "qwen3_8"
            assert res.spec.cache_backend.layout.rec_pages_per_slot == 4
        else:
            assert isinstance(res, driver_mod.Qwen35Megakernel)
            assert res.spec.name == "qwen3_5"
    else:
        assert res is None


def test_dcp_sharded_kv_pool_roundtrip_and_worklist():
    """Verify pool round trip and GQAParams under dcp_kv_sharded=True."""
    import dataclasses

    import jax
    import jax.numpy as jnp

    F32 = jnp.float32
    from jax.sharding import Mesh

    from vllm_torchtpu.megakernel import kernel, planner
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec
    from vllm_torchtpu.megakernel.models import qwen35, qwen35_pool
    from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES
    from vllm_torchtpu.megakernel.pool.config import POOL_STATE_NAME, with_cache

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
    ).with_blocks({"gqa": {"split": True}})
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
    p = planner.plan(ms, ex, fit_vmem=False)
    ap = p.block_params["gqa"]
    assert ap.DCP_SHARDED is True
    assert ap.BLK == 128
    assert ap.SUB == 128

    pt, ckpt, roff, _ = qwen35_pool.default_pool_layout(g, ex)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.PRNGKey(0), g, ex)
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
    if len(devs) >= 8:
        mesh = Mesh(devs[:8], ("tp",))
        ref_hid, _ = qwen35.reference_step(
            g, ex, pos, hid_l, w_l, st_l, paged_kv=True, pool_io=True
        )
        pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
            pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
        )
        step, _ = qwen35.make_pooled_step(mesh, g, ex, fused_pool_io=False)
        out_hid, _ = step(
            pos_mk,
            hid_mk,
            w_mk,
            st_mk[POOL_STATE_NAME],
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
        pos_neg = jnp.array([128, -512], dtype=jnp.int32)
        _, pools_neg = step(
            pos_neg,
            hid_mk,
            w_mk,
            st_mk[POOL_STATE_NAME],
            st_mk["page_table"],
            ckpt,
            roff,
            "all",
        )
        p10 = int(pt[1, 0])
        assert jnp.array_equal(pools_neg[0][0, p10], st_mk[POOL_STATE_NAME][0][0, p10])
