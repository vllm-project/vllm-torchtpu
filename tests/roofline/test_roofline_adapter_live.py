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
"""Live integration probe against megakernel.planner on Qwen3.5 FULL and FULL_EP."""

from __future__ import annotations

import dataclasses
import importlib

import pytest

from vllm_torchtpu.roofline import (
    Dtypes,
    ModelGeometry,
    Parallel,
    StepShape,
    ep_busiest_experts,
    floor_from_plan,
    pass_costs,
    reconcile_plan_vs_standalone,
)
from vllm_torchtpu.roofline.adapter import _dtype_bytes

QWEN35_GEOM = ModelGeometry(
    hidden=4096,
    num_layers=60,
    layer_types=tuple(["linear_attention"] * 45 + ["full_attention"] * 15),
    num_heads=32,
    num_kv_heads=2,
    head_dim=256,
    attn_output_gate=True,
    linear_k_heads=16,
    linear_v_heads=64,
    linear_k_dim=128,
    linear_v_dim=128,
    conv_kernel=4,
    num_experts=512,
    top_k=10,
    moe_inter=1024,
    shared_inter=1024,
    vocab=248320,
    mtp_layers=1,
)


def test_live_qwen35_full_and_full_ep_plan_reconciliation() -> None:
    pytest.importorskip("vllm_torchtpu.megakernel.planner")
    exec_spec_mod = importlib.import_module("vllm_torchtpu.megakernel.exec_spec")
    qwen35 = importlib.import_module("vllm_torchtpu.megakernel.models.qwen35")
    prof_qwen35 = importlib.import_module(
        "vllm_torchtpu.megakernel.models.qwen35_profile"
    )

    ExecSpec = exec_spec_mod.ExecSpec
    baseline_tiles = prof_qwen35.baseline_tiles

    ex_full = ExecSpec(
        B=1,
        T=1,
        R=8,
        ctx=2048,
        tiles=dataclasses.replace(baseline_tiles(1, R=8), hier=0),
    )
    _, info_full = qwen35.build(qwen35.FULL, ex_full)
    p_full = info_full["plan"]

    mods_by_name = {m.name: m for m in p_full.model_spec.all_modules()}
    gqa_cd = mods_by_name["gqa"].cache_decl(p_full.model_spec.config)
    gdn_cd = mods_by_name["gdn"].cache_decl(p_full.model_spec.config)
    dt_full = Dtypes(
        kv=_dtype_bytes(gqa_cd.page_dtype),
        state=_dtype_bytes(gdn_cd.rec_dtype),
        conv=_dtype_bytes(gdn_cd.conv_dtype),
    )

    sf_full = floor_from_plan(
        p_full,
        ex_full,
        geom=QWEN35_GEOM,
        dt=dt_full,
        include_plan_extras=False,
    )
    full_mods = {m.name: m for m in sf_full.modules}
    for name in ("gqa", "gdn", "moe"):
        assert name in full_mods and full_mods[name].hbm_bytes > 0.0

    rows = reconcile_plan_vs_standalone(
        p_full,
        ex_full,
        QWEN35_GEOM,
        shape=StepShape(batch=1, rows_per_seq=1, context=2048),
        dt=dt_full,
        include_plan_extras=False,
    )
    row_map = {r.module: r for r in rows}
    assert set(row_map) == {"gqa", "gdn", "moe", "lm_head"}
    for r in rows:
        assert r.delta_hbm_bytes == pytest.approx(0.0, abs=1e-6), r
        assert r.delta_flops == pytest.approx(0.0, abs=1e-6), r

    ex_ep = ExecSpec(
        B=1,
        T=1,
        R=8,
        ctx=2048,
        tiles=dataclasses.replace(baseline_tiles(1, R=8), hier=0),
    )
    _, info_ep = qwen35.build(qwen35.FULL_EP, ex_ep)
    p_ep = info_ep["plan"]
    sf_ep = floor_from_plan(
        p_ep,
        ex_ep,
        geom=QWEN35_GEOM,
        dt=dt_full,
        include_plan_extras=False,
    )
    ep_mods = {m.name: m for m in sf_ep.modules}
    for name in ("gqa", "gdn", "moe"):
        assert name in ep_mods and ep_mods[name].hbm_bytes > 0.0
    std_ep = {
        m.name: m
        for m in pass_costs(
            QWEN35_GEOM,
            Parallel(tp=8, kv_split=4, kv_merge="exchange", moe="ep"),
            StepShape(batch=1, rows_per_seq=1, context=2048),
            dt_full,
        )
    }
    assert ep_busiest_experts(512, 10, 1, 8) == pytest.approx(2.979897, abs=1e-6)
    assert ep_mods["moe"].hbm_bytes == pytest.approx(std_ep["moe"].hbm_bytes, abs=1.0)
