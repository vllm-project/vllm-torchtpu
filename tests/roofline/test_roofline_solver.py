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
"""Unit tests for mesh/prefill-decode solver and CLI solve subcommand."""

from __future__ import annotations

import json
import math
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING, Literal

import pytest

import vllm_torchtpu.roofline as rf
from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    FloorOptions,
    KvFitInput,
    Parallel,
    ScenarioOverrides,
    SeqKvLayout,
    SolveOptions,
    SpecShape,
    cli,
    hbm_breakdown_per_core_bytes,
    hbm_footprint_per_core_bytes,
    solve_serving_plan,
)
from vllm_torchtpu.roofline.memory import (
    default_rpa_page_size,
    kv_capacity_and_hit_rate,
    smem_max_seqs,
)
from vllm_torchtpu.roofline.partition import (
    count_full_attn_layers,
)
from vllm_torchtpu.roofline.solver import (
    EXAMPLE_SCENARIOS,
    _candidate_decode_parallels,
    _candidate_moe_dispatches,
    _candidate_prefill_parallels,
    _pareto_filter,
    _valid_gdn_degree,
    _valid_head_tp,
    _valid_raiden_gdn_pair,
)

if TYPE_CHECKING:
    from pathlib import Path

QWEN35_CFG: dict[str, object] = {
    "text_config": {
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 15,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
        "dtype": "bfloat16",
    }
}


def _check_dp_moe_decode(chip: rf.ChipSpec, q35_geom: rf.ModelGeometry) -> None:
    """Verify that DP > 1 MoE decode candidates price cross-DP collectives."""
    dp_cands = [
        p
        for p in _candidate_decode_parallels(q35_geom, rf.TOPOLOGIES["2x2x1"])
        if p.dp > 1
    ]
    assert dp_cands
    ep_degree = 8
    for par_dp in dp_cands:
        assert par_dp.moe == "ep"
        assert par_dp.ep == ep_degree
        assert par_dp.moe_dispatch in ("ag_ep", "a2a_push")
        opts = FloorOptions(dt=rf.Dtypes(kv=1.0), topology=rf.TOPOLOGIES["2x2x1"])
        sf_dp = rf.step_floor(
            q35_geom,
            chip,
            par_dp,
            rf.StepShape(batch=4, context=1024),
            opts,
        )
        sf_dp_ar = rf.step_floor(
            q35_geom,
            chip,
            dc_replace(par_dp, moe_dispatch="allreduce"),
            rf.StepShape(batch=4, context=1024),
            opts,
        )
        assert sf_dp.chain_s > 0.0
        if par_dp.moe_dispatch == "ag_ep":
            assert sf_dp_ar.chain_s == pytest.approx(sf_dp.chain_s)


def _check_mixed_prefill_dp2(chip: rf.ChipSpec) -> None:
    """Verify that mixed prefill work scales as 1/dp per replica."""
    dp2_geom = rf.from_hf_config(
        {
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "intermediate_size": 4096,
            "vocab_size": 32768,
        }
    )
    dp2_scen = rf.TrafficScenario(
        name="custom",
        isl=4096,
        osl=512,
        concurrencies=(8,),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
    )
    expected_dp_2x2x1 = 8
    dp2_plan = rf.solve_serving_plan(
        dp2_geom,
        rf.Dtypes(kv=1.0),
        chip,
        dp2_scen,
        SolveOptions(mode="mixed", max_chips=4),
    )
    assert dp2_plan.recommended is not None
    assert dp2_plan.recommended.decode_pool.dp == expected_dp_2x2x1
    par_dp2 = rf.Parallel(tp=1, dp=8, mesh_axes=(2, 2, 2, 1))
    opts_dp2 = FloorOptions(dt=rf.Dtypes(kv=1.0), topology=rf.TOPOLOGIES["2x2x1"])
    sf_d_dp2 = rf.step_floor(
        dp2_geom,
        chip,
        par_dp2,
        rf.StepShape(batch=1, context=4352),
        opts_dp2,
    )
    sf_m_dp2 = rf.step_floor(
        dp2_geom,
        chip,
        par_dp2,
        rf.StepShape(batch=1, context=4352, prefill_tokens=2048),
        opts_dp2,
    )
    assert sf_d_dp2.serial_estimate_s > 0.0
    assert sf_m_dp2.serial_estimate_s > 0.0
    exp_p_mix = float(1 * 2) / float(512 + 1 * 2)
    assert dp2_plan.recommended.p_mixed == pytest.approx(exp_p_mix, rel=1e-5)


def _check_cli_solve(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that CLI solve subcommand produces structured JSON payload."""
    cfg_path = tmp_path / "q35_config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    rc = cli.main(
        [
            "solve",
            "--model",
            str(cfg_path),
            "--chip",
            "tpu7x",
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "auto",
            "--max-chips",
            "8",
            "--example-slo-tier",
            "interactive",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommended"] is not None


def _check_pp_prefill_ttft(chip: rf.ChipSpec) -> None:
    """Verify that PP > 1 prefill TTFT includes pipeline bubble stage fills."""
    pp_geom = rf.from_hf_config(
        {
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "intermediate_size": 4096,
            "vocab_size": 32768,
        }
    )
    for p_chunk, n_chk in ((2048, 1), (256, 4)):
        chk_scen = rf.TrafficScenario(
            name="custom",
            isl=1024,
            osl=256,
            concurrencies=(1,),
            prefill_chunk=p_chunk,
            prefix_hit_rate=0.0,
            accept_len=1.0,
            think_time_s=0.0,
            kv_working_set_turns=1.0,
        )
        chk_disagg = rf.solve_serving_plan(
            pp_geom,
            rf.Dtypes(kv=1.0),
            chip,
            chk_scen,
            SolveOptions(mode="disagg", max_chips=8),
        )
        pp_pts = [
            p
            for p in chk_disagg.candidates
            if p.prefill_pool is not None and p.prefill_pool.pp > 1
        ]
        assert pp_pts
        p0 = pp_pts[0]
        assert p0.prefill_pool is not None
        sf_st = rf.step_floor(
            pp_geom,
            chip,
            rf.Parallel(
                tp=p0.prefill_pool.tp,
                pp=p0.prefill_pool.pp,
                pcp=p0.prefill_pool.pcp,
                moe="ep" if p0.prefill_pool.ep > 1 else "tp_slice",
                ep=p0.prefill_pool.ep,
                mesh_axes=p0.prefill_pool.mesh_axes,
            ),
            rf.StepShape(
                batch=0, rows_per_seq=0, context=0, state_writes=0, prefill_tokens=1024
            ),
            FloorOptions(
                dt=rf.Dtypes(kv=1.0),
                topology=rf.TOPOLOGIES[p0.prefill_pool.topology],
            ),
        )
        bubble_mult = 1.0 + float(p0.prefill_pool.pp - 1) / float(n_chk)
        assert p0.p90_ttft_ms == pytest.approx(
            sf_st.serial_estimate_s * bubble_mult * 1e3
        )


def test_solver_hybrid_moe_mixed_vs_disagg_and_crossover(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test mixed vs disaggregated solver behavior and MTP cycle pricing."""
    q35_geom = rf.from_hf_config(QWEN35_CFG)
    chip = rf.CHIPS["tpu7x"]
    scen = EXAMPLE_SCENARIOS["chat-8k1k"]
    mixed_plan = rf.solve_serving_plan(
        q35_geom,
        rf.Dtypes(kv=1.0),
        chip,
        scen,
        SolveOptions(mode="mixed", max_chips=8),
    )
    disagg_plan = rf.solve_serving_plan(
        q35_geom,
        rf.Dtypes(kv=1.0),
        chip,
        scen,
        SolveOptions(mode="disagg", max_chips=8),
    )
    min_conc = 16
    min_mixed_fraction = 0.10
    mixed_c16 = [p for p in mixed_plan.frontier if p.concurrency >= min_conc]
    assert mixed_c16
    assert any(p.p_mixed >= min_mixed_fraction for p in mixed_c16)
    assert disagg_plan.recommended is not None
    assert disagg_plan.recommended.p_mixed == 0.0

    # coding-agent (accept_len=3.4) with 1 MTP layer prices k=3 window.
    ca_scen = rf.build_example_scenario(
        "coding-agent", overrides=ScenarioOverrides(concurrencies=(1,))
    )
    ca_plan = rf.solve_serving_plan(
        q35_geom,
        rf.Dtypes(kv=1.0),
        chip,
        ca_scen,
        SolveOptions(mode="disagg", max_chips=8),
    )
    assert ca_plan.recommended is not None
    rec_d = ca_plan.recommended.decode_pool
    sf_k3 = rf.spec_cycle_floor(
        q35_geom,
        chip,
        rf.Parallel(
            tp=rec_d.tp,
            dp=rec_d.dp,
            moe="ep" if rec_d.ep > 1 else "tp_slice",
            ep=rec_d.ep,
            moe_dispatch=rec_d.moe_dispatch,
            kv_split=rec_d.kv_split,
            mesh_axes=rec_d.mesh_axes,
        ),
        SpecShape(
            batch=max(1, (1 + rec_d.dp - 1) // rec_d.dp),
            context=ca_scen.isl + ca_scen.osl // 2,
            k=3,
        ),
        FloorOptions(
            dt=rf.Dtypes(kv=1.0),
            topology=rf.TOPOLOGIES[rec_d.topology],
        ),
    )
    assert ca_plan.recommended.p90_tpot_ms == pytest.approx(
        sf_k3.serial_estimate_s * 1e3 / ca_scen.accept_len, rel=1e-4
    )

    _check_dp_moe_decode(chip, q35_geom)

    _check_mixed_prefill_dp2(chip)

    _check_pp_prefill_ttft(chip)

    hybrid_2t_cfg = {
        "text_config": {
            "hidden_size": 8192,
            "num_hidden_layers": 92,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 23,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "attn_output_gate": True,
            "linear_num_key_heads": 32,
            "linear_num_value_heads": 128,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 2048,
            "shared_expert_intermediate_size": 2048,
            "vocab_size": 248320,
        }
    }
    hybrid_2t_geom = rf.from_hf_config(hybrid_2t_cfg)
    hybrid_2t_dt = rf.Dtypes(
        w_dense=1.0, w_expert=0.5625, w_aux=1.0, kv=1.0, state=2.0, lm_head=2.0
    )
    topo_32 = rf.TOPOLOGIES["2x2x4"]
    par_tp32 = rf.Parallel(
        tp=32, dp=1, moe="ep", ep=32, moe_dispatch="ag_ep", kv_split=8
    )
    par_tp8_dp4 = rf.Parallel(
        tp=8, dp=4, moe="ep", ep=32, moe_dispatch="ag_ep", kv_split=2
    )
    opts_32 = FloorOptions(dt=hybrid_2t_dt, topology=topo_32)
    sf_tp32_c4 = rf.step_floor(
        hybrid_2t_geom,
        chip,
        par_tp32,
        rf.StepShape(batch=1, context=131072),
        opts_32,
    )
    sf_dp4_c4 = rf.step_floor(
        hybrid_2t_geom,
        chip,
        par_tp8_dp4,
        rf.StepShape(batch=1, context=131072),
        opts_32,
    )
    assert sf_tp32_c4.serial_estimate_s < sf_dp4_c4.serial_estimate_s

    sf_tp32_c256 = rf.step_floor(
        hybrid_2t_geom,
        chip,
        par_tp32,
        rf.StepShape(batch=256, context=8192),
        opts_32,
    )
    sf_dp4_c256 = rf.step_floor(
        hybrid_2t_geom,
        chip,
        par_tp8_dp4,
        rf.StepShape(batch=64, context=8192),
        opts_32,
    )
    assert sf_dp4_c256.serial_estimate_s < sf_tp32_c256.serial_estimate_s

    gpt_oss_cfg = {
        "hidden_size": 2880,
        "num_hidden_layers": 36,
        "layer_types": ["sliding_attention", "full_attention"] * 18,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "sliding_window": 128,
        "num_local_experts": 128,
        "num_experts_per_tok": 4,
        "intermediate_size": 2880,
        "vocab_size": 201088,
        "dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "mxfp4",
            "modules_to_not_convert": ["self_attn", "lm_head"],
        },
    }
    oss_geom = rf.from_hf_config(gpt_oss_cfg)
    oss_dt = rf.dtypes_from_hf_config(gpt_oss_cfg)
    max_oss_chips = 12
    oss_disagg = rf.solve_serving_plan(
        oss_geom,
        oss_dt,
        chip,
        scen,
        SolveOptions(mode="disagg", max_chips=max_oss_chips),
    )
    assert oss_disagg.recommended is not None
    assert oss_disagg.recommended.total_chips <= max_oss_chips
    assert any(
        p.prefill_pool is not None
        and (
            p.prefill_pool.instances != p.decode_pool.instances
            or p.prefill_pool.topology != p.decode_pool.topology
        )
        for p in oss_disagg.frontier
    )

    infeasible_scen = rf.TrafficScenario(
        name="custom",
        isl=8192,
        osl=1024,
        concurrencies=(1, 4),
        tpot_slo_ms=0.05,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
    )
    inf_plan = rf.solve_serving_plan(
        q35_geom,
        rf.Dtypes(kv=1.0),
        chip,
        infeasible_scen,
        SolveOptions(mode="mixed", max_chips=4),
    )
    assert any("INFEASIBLE_SLO" in w for w in inf_plan.warnings)

    _check_cli_solve(tmp_path, capsys)


def test_physical_kv_page_capacity_and_unpruned_regimes() -> None:
    """Test physical KV page table capacity scaling and unpruned disagg regimes."""
    chip = rf.CHIPS["tpu7x"]
    q35_geom = rf.from_hf_config(QWEN35_CFG)
    q35_dt = rf.Dtypes(
        w_dense=1.0, w_expert=1.0, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0
    )

    hbm_diff_threshold = 0.80e9
    hbm_mtp1 = rf.hbm_footprint_per_core_bytes(
        q35_geom, q35_dt, rf.Parallel(tp=8, kv_split=1, moe="ep"), SeqKvLayout(0, 0)
    )
    hbm_mtp0 = rf.hbm_footprint_per_core_bytes(
        dc_replace(q35_geom, mtp_layers=0),
        q35_dt,
        rf.Parallel(tp=8, kv_split=1, moe="ep"),
        SeqKvLayout(0, 0),
    )
    assert (hbm_mtp1 - hbm_mtp0) > hbm_diff_threshold
    hbm_phys = rf.hbm_footprint_per_core_bytes(
        q35_geom,
        q35_dt,
        rf.Parallel(tp=8, kv_split=1, moe="ep"),
        SeqKvLayout(1, 8192, physical_kv_layout=True),
    )
    hbm_log = rf.hbm_footprint_per_core_bytes(
        q35_geom,
        q35_dt,
        rf.Parallel(tp=8, kv_split=1, moe="ep"),
        SeqKvLayout(1, 8192, physical_kv_layout=False),
    )
    assert (hbm_phys - hbm_log) == pytest.approx(15.0 * 8192.0 * 512.0, abs=2.0)

    accept_len = 3.0
    target_concurrency = 64
    min_pp_stages = 8
    scen = rf.TrafficScenario(
        name="custom",
        isl=1024,
        osl=1024,
        concurrencies=(target_concurrency,),
        prefix_hit_rate=0.5,
        accept_len=accept_len,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
    )
    disagg = rf.solve_serving_plan(
        q35_geom,
        q35_dt,
        chip,
        scen,
        SolveOptions(mode="disagg", max_chips=12),
    )
    assert all(p.concurrency == target_concurrency for p in disagg.frontier)
    assert any(
        p.prefill_pool is not None and p.prefill_pool.pp >= min_pp_stages
        for p in disagg.candidates
    )
    assert any(
        p.decode_pool.moe_dispatch == "a2a_push" and p.decode_pool.dp in (4, 8)
        for p in disagg.candidates
    )

    dsa_78l_cfg = {
        "hidden_size": 6144,
        "num_hidden_layers": 78,
        "layer_types": ["dsa"] * 78,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "shared_expert_intermediate_size": 2048,
        "first_k_dense_replace": 3,
        "dense_intermediate_size": 12288,
        "vocab_size": 154880,
        "num_nextn_predict_layers": 1,
    }
    dsa_78l_geom = rf.from_hf_config(dsa_78l_cfg)
    dsa_78l_dt = rf.Dtypes(
        w_dense=1.0, w_expert=1.0, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0
    )
    accept_len = 3.0
    tp_16 = 16
    kv_split_16 = 16
    dsa_c48 = rf.solve_serving_plan(
        dsa_78l_geom,
        dsa_78l_dt,
        chip,
        rf.TrafficScenario(
            name="dsa-single-turn",
            isl=137168,
            osl=894,
            concurrencies=(48,),
            prefix_hit_rate=0.95,
            accept_len=accept_len,
            think_time_s=0.0,
            kv_working_set_turns=1.0,
            spec_tokens=3,
        ),
        SolveOptions(mode="mixed", max_chips=8, gpu_memory_utilization=0.85),
    )
    tp16_pts = [
        p
        for p in dsa_c48.candidates
        if p.decode_pool.tp == tp_16
        and p.decode_pool.ep == 1
        and p.decode_pool.max_num_seqs is None
    ]
    best_tp16 = min(tp16_pts, key=lambda p: p.p90_tpot_ms)
    assert best_tp16.decode_pool.kv_split == kv_split_16

    w4_dsa_geom = dc_replace(
        dsa_78l_geom,
        hidden=7168,
        num_layers=61,
        layer_types=("dsa",) * 61,
        num_heads=128,
        num_kv_heads=128,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        v_head_dim=128,
        index_n_heads=64,
        vocab=129280,
    )
    w4_dsa_dt = rf.Dtypes(
        w_dense=1.0, w_expert=0.5625, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0
    )
    w4_dsa_c64 = rf.solve_serving_plan(
        w4_dsa_geom,
        w4_dsa_dt,
        chip,
        rf.TrafficScenario(
            name="w4-dsa-c64",
            isl=131072,
            osl=935,
            concurrencies=(64,),
            prefix_hit_rate=0.95,
            accept_len=accept_len,
            think_time_s=0.0,
            kv_working_set_turns=1.0,
            spec_tokens=3,
            kv_offload_gb=48.0,
        ),
        SolveOptions(mode="mixed", max_chips=8, gpu_memory_utilization=0.85),
    )
    expected_tp_2 = 2
    expected_dp_8 = 8
    expected_dp_16 = 16
    ep_degree_16 = 16
    tp2_dp8 = [
        p
        for p in w4_dsa_c64.candidates
        if p.decode_pool.tp == expected_tp_2
        and p.decode_pool.dp == expected_dp_8
        and p.decode_pool.ep == ep_degree_16
    ]
    tp1_dp16 = [
        p
        for p in w4_dsa_c64.candidates
        if p.decode_pool.tp == 1
        and p.decode_pool.dp == expected_dp_16
        and p.decode_pool.ep == ep_degree_16
    ]
    assert tp2_dp8
    assert tp1_dp16
    assert min(p.p90_tpot_ms for p in tp2_dp8) < min(p.p90_tpot_ms for p in tp1_dp16)


def test_engine_memory_utilization_and_explicit_scenario_inputs(tmp_path: Path) -> None:
    """Test engine GPU memory utilization clipping and CLI explicit scenario flags."""
    chip = rf.CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = rf.Dtypes(kv=1.0, w_aux=2.0)

    with pytest.raises(TypeError):
        rf.TrafficScenario(name="x", isl=1024, osl=128)

    util_045 = 0.45
    util_092 = 0.92
    plan_45 = rf.solve_serving_plan(
        geom,
        dt,
        chip,
        rf.TrafficScenario(
            name="x",
            isl=1024,
            osl=128,
            prefix_hit_rate=0.5,
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(1,),
            kv_working_set_turns=1.0,
        ),
        SolveOptions(mode="mixed", max_chips=4, gpu_memory_utilization=util_045),
    )
    assert plan_45.gpu_memory_utilization == util_045
    assert len(plan_45.frontier) == 0

    plan_92 = rf.solve_serving_plan(
        geom,
        dt,
        chip,
        rf.TrafficScenario(
            name="x",
            isl=1024,
            osl=128,
            prefix_hit_rate=0.5,
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(1,),
            kv_working_set_turns=1.0,
        ),
        SolveOptions(mode="mixed", max_chips=4, gpu_memory_utilization=util_092),
    )
    assert plan_92.gpu_memory_utilization == util_092
    assert len(plan_92.frontier) > 0

    scen_c0 = rf.TrafficScenario(
        name="closed",
        isl=4096,
        osl=1024,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(128,),
        kv_working_set_turns=1.0,
    )
    scen_think = rf.TrafficScenario(
        name="open",
        isl=4096,
        osl=1024,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=15.0,
        concurrencies=(128,),
        kv_working_set_turns=1.0,
    )
    p_c0 = rf.solve_serving_plan(
        geom, dt, chip, scen_c0, SolveOptions(mode="mixed", max_chips=4)
    )
    p_think = rf.solve_serving_plan(
        geom, dt, chip, scen_think, SolveOptions(mode="mixed", max_chips=4)
    )
    assert p_c0.recommended is not None
    assert p_think.recommended is not None
    assert p_think.recommended.p90_tpot_ms <= p_c0.recommended.p90_tpot_ms

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["solve", "--config", str(cfg_path)])
    assert excinfo.value.code != 0

    rc_ex = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--concurrency",
            "1",
            "--json",
        ]
    )
    assert rc_ex == 0

    rc_exp = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--isl",
            "1024",
            "--osl",
            "128",
            "--prefix-hit-rate",
            "0.5",
            "--accept-len",
            "1.0",
            "--think-time-s",
            "0.0",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--concurrency",
            "1",
            "--json",
        ]
    )
    assert rc_exp == 0


def test_runtime_legality_heads_pp_tp_pcp_ep_gdn_and_smem_cap() -> None:
    """Test parallelism legality checks across attention heads, GDN, and SMEM."""
    assert count_full_attn_layers(rf.from_hf_config(QWEN35_CFG)) > 0
    assert _valid_head_tp(rf.from_hf_config(QWEN35_CFG), 1)
    assert _pareto_filter(()) == ()
    uneven_geom = rf.from_hf_config(
        {
            "hidden_size": 3584,
            "num_hidden_layers": 8,
            "num_attention_heads": 28,
            "num_key_value_heads": 6,
            "head_dim": 128,
            "intermediate_size": 7168,
            "vocab_size": 32000,
        }
    )
    invalid_tp_4 = 4
    invalid_tp_8 = 8
    dec_pars = _candidate_decode_parallels(uneven_geom, rf.TOPOLOGIES["2x2x1"])
    assert all(28 % p.tp == 0 and (6 % p.tp == 0 or p.tp % 6 == 0) for p in dec_pars)
    assert all(p.tp not in (invalid_tp_4, invalid_tp_8) for p in dec_pars)

    min_pcp_2 = 2
    pre_pars = _candidate_prefill_parallels(uneven_geom, rf.TOPOLOGIES["2x2x1"])
    assert all(not (p.pp > 1 and p.tp > 1) for p in pre_pars)
    assert any(p.pp > 1 and p.tp == 1 and p.pcp >= min_pcp_2 for p in pre_pars)

    hybrid_8l = rf.from_hf_config(
        {
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "linear_num_key_heads": 8,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 32000,
        }
    )
    max_disagg_pp = 2
    disagg_pre = _candidate_prefill_parallels(
        hybrid_8l, rf.TOPOLOGIES["2x2x1"], for_disagg=True
    )
    assert all(p.pp <= max_disagg_pp for p in disagg_pre)
    assert not _valid_gdn_degree(dc_replace(hybrid_8l, linear_v_heads=64), 6)
    assert not _valid_raiden_gdn_pair(
        hybrid_8l, rf.Parallel(tp=2, pcp=1), rf.Parallel(tp=8, pcp=1)
    )
    assert _valid_raiden_gdn_pair(
        hybrid_8l, rf.Parallel(tp=1, pcp=2, pp=2), rf.Parallel(tp=8, pcp=1)
    )

    moe_geom = rf.from_hf_config(QWEN35_CFG)
    moe_pre = _candidate_prefill_parallels(moe_geom, rf.TOPOLOGIES["2x2x1"])
    for p in moe_pre:
        if p.pcp > 1:
            assert p.moe == "ep"
            assert p.ep == p.tp * p.pcp

    isl_262144 = 262144
    page_size_16 = 16
    smem_seqs_8 = 8
    assert default_rpa_page_size(isl_262144) == page_size_16
    assert smem_max_seqs(isl_262144, 1, page_size_16) == smem_seqs_8


def test_disagg_capacity_clamped_hit_rate_and_prefill_dispatch() -> None:
    """Test disagg capacity clamping, prefix hit degradation, and prefill dispatch."""
    chip = rf.CHIPS["tpu7x"]
    chat_scen = EXAMPLE_SCENARIOS["chat-8k1k"]
    cap_init, h_init = kv_capacity_and_hit_rate(
        rf.from_hf_config(QWEN35_CFG),
        rf.Dtypes(kv=1.0),
        rf.Parallel(tp=1),
        1,
        KvFitInput(
            b_ws=1,
            ctx_full=1024,
            hbm_limit_b=1e12,
            w_base_b=0.0,
            b_kv_phys_core=1.0,
            offload_b_per_core=0.0,
            prefix_hit_rate=chat_scen.prefix_hit_rate,
            kv_working_set_turns=chat_scen.kv_working_set_turns,
        ),
    )
    assert cap_init > 0.0
    assert h_init == chat_scen.prefix_hit_rate
    moe_cfg = {
        "hidden_size": 2048,
        "num_hidden_layers": 8,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 32,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "vocab_size": 32000,
    }
    geom = rf.from_hf_config(moe_cfg)
    dt = rf.Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    scen_small = rf.TrafficScenario(
        name="small-ws",
        isl=8192,
        osl=256,
        prefix_hit_rate=0.9,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(8,),
        kv_working_set_turns=0.01,
    )
    scen_huge = dc_replace(scen_small, name="huge-ws", kv_working_set_turns=5000.0)
    plan_small = rf.solve_serving_plan(
        geom, dt, chip, scen_small, SolveOptions(mode="disagg", max_chips=8)
    )
    plan_huge = rf.solve_serving_plan(
        geom, dt, chip, scen_huge, SolveOptions(mode="disagg", max_chips=8)
    )
    assert plan_small.recommended is not None
    assert plan_huge.recommended is not None
    by_pair_small = {
        (
            p.prefill_pool.topology,
            p.prefill_pool.tp,
            p.prefill_pool.pcp,
            p.prefill_pool.pp,
            p.decode_pool.topology,
            p.decode_pool.tp,
            p.decode_pool.dp,
            p.decode_pool.kv_split,
        ): p
        for p in plan_small.candidates
        if p.prefill_pool is not None
    }
    matched = 0
    for pt_h in plan_huge.candidates:
        if pt_h.prefill_pool is None:
            continue
        k = (
            pt_h.prefill_pool.topology,
            pt_h.prefill_pool.tp,
            pt_h.prefill_pool.pcp,
            pt_h.prefill_pool.pp,
            pt_h.decode_pool.topology,
            pt_h.decode_pool.tp,
            pt_h.decode_pool.dp,
            pt_h.decode_pool.kv_split,
        )
        pt_s = by_pair_small.get(k)
        if pt_s is not None:
            matched += 1
            assert pt_s.kv_cache_capacity_tokens is not None
            assert pt_h.effective_prefix_hit_rate < pt_s.effective_prefix_hit_rate
            assert pt_h.p90_ttft_ms > pt_s.p90_ttft_ms
    assert matched >= 1
    pcp_ep_pts = [
        p
        for p in plan_small.candidates
        if p.prefill_pool is not None
        and p.prefill_pool.pcp > 1
        and p.prefill_pool.tp == 1
        and p.prefill_pool.ep > 1
    ]
    assert pcp_ep_pts
    assert all(
        p.prefill_pool.moe_dispatch == "ag_ep" for p in pcp_ep_pts if p.prefill_pool
    )

    scen_think = dc_replace(
        scen_small,
        concurrencies=(64,),
        think_time_s=5.0,
        ttft_slo_ms=5000.0,
        tpot_slo_ms=200.0,
    )
    plan_think = rf.solve_serving_plan(
        geom, dt, chip, scen_think, SolveOptions(mode="auto", max_chips=4)
    )
    assert plan_think.recommended is not None
    assert plan_think.recommended.slo_met

    scen_capped_ws = dc_replace(
        scen_small,
        concurrencies=(64,),
        kv_working_set_turns=200.0,
        max_num_seqs=1,
    )
    scen_single_ws = dc_replace(
        scen_small,
        concurrencies=(1,),
        kv_working_set_turns=200.0,
        max_num_seqs=1,
    )
    expected_tp_8 = 8
    pt_c64 = next(
        p
        for p in rf.solve_serving_plan(
            geom,
            dt,
            chip,
            scen_capped_ws,
            SolveOptions(mode="mixed", max_chips=4),
        ).candidates
        if p.decode_pool.tp == expected_tp_8 and p.decode_pool.dp == 1
    )
    pt_c1 = next(
        p
        for p in rf.solve_serving_plan(
            geom,
            dt,
            chip,
            scen_single_ws,
            SolveOptions(mode="mixed", max_chips=4),
        ).candidates
        if p.decode_pool.tp == expected_tp_8 and p.decode_pool.dp == 1
    )
    assert pt_c64.effective_prefix_hit_rate < pt_c1.effective_prefix_hit_rate


def test_solver_excludes_impossible_dp1_ep_and_gdn_dcp() -> None:
    """Reject cross-DP EP dispatches at dp=1 and kv_split>1 on KV-less pure GDN."""
    assert "a2a_push" not in _candidate_moe_dispatches("ep", dp=1)
    assert "ag_ep" not in _candidate_moe_dispatches("ep", dp=1)
    pure_gdn = rf.from_hf_config(
        {
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 0,
            "head_dim": 128,
            "linear_num_key_heads": 8,
            "linear_num_value_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "intermediate_size": 4096,
            "vocab_size": 32000,
        }
    )
    dec = _candidate_decode_parallels(pure_gdn, rf.TOPOLOGIES["2x2x1"])
    assert dec
    assert all(p.kv_split == 1 for p in dec)


def test_solver_rejects_subhost_max_chips() -> None:
    """Reject max_chips below minimum provisionable 2x2x1 slice (4 chips)."""
    geom = rf.from_hf_config(QWEN35_CFG)
    scen = rf.build_example_scenario(overrides=ScenarioOverrides(concurrencies=(1,)))
    for bad_chips in (1, 2):
        with pytest.raises(ValueError, match="2x2x1"):
            rf.solve_serving_plan(
                geom,
                rf.Dtypes(),
                rf.CHIPS["tpu7x"],
                scen,
                SolveOptions(max_chips=bad_chips),
            )


def test_hbm_breakdown_matches_footprint() -> None:
    """HbmBreakdown total matches hbm_footprint_per_core_bytes across geometries."""
    gqa_cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 32000,
    }
    mla_cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 30,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
        "kv_lora_rank": 512,
        "q_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "vocab_size": 32000,
    }

    geoms = [
        rf.from_hf_config(gqa_cfg),
        rf.from_hf_config(mla_cfg),
        rf.from_hf_config(QWEN35_CFG),
    ]
    dt = Dtypes(kv=1.0)
    par = Parallel(tp=4, dp=1, ep=1, pp=1, pcp=1)
    seq_layout = SeqKvLayout(seqs_per_dp=16, context=2048)

    for geom in geoms:
        bd = hbm_breakdown_per_core_bytes(geom, dt, par, seq_layout)
        footprint = hbm_footprint_per_core_bytes(geom, dt, par, seq_layout)
        assert bd.total == footprint
        assert math.isclose(bd.weights + bd.kv + bd.state, bd.total, rel_tol=1e-9)
        assert bd.weights > 0
        assert bd.kv >= 0
        if geom.linear_k_heads:
            assert bd.state > 0
        else:
            assert bd.state == 0.0


@pytest.mark.parametrize("mode", ["mixed", "disagg", "auto"])
def test_search_stats_funnel_is_monotone(
    mode: Literal["mixed", "disagg", "auto"],
) -> None:
    """Search funnel is monotonically non-increasing and frontier is positive."""
    chip = CHIPS["tpu7x"]
    gqa_cfg = {
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 32000,
    }
    geom = rf.from_hf_config(gqa_cfg)
    dt = Dtypes(kv=1.0)
    scenario = dc_replace(
        EXAMPLE_SCENARIOS["chat-8k1k"],
        concurrencies=(1, 4),
        ttft_slo_ms=5000.0,
        tpot_slo_ms=500.0,
    )
    # Disaggregated serving needs >= 2 provisioned 2x2x1 slices (8 chips).
    max_chips = 8 if mode == "disagg" else 4
    splan = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        SolveOptions(mode=mode, max_chips=max_chips),
    )
    assert splan.search is not None
    stats = splan.search
    assert stats.enumerated >= stats.hbm_fit >= stats.slo_met >= stats.frontier >= 1
    assert stats.frontier == len(splan.frontier)


def test_decode_floor_matches_binding() -> None:
    """Every candidate has decode_floor matching binding resource."""
    chip = CHIPS["tpu7x"]
    gqa_cfg = {
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 32000,
    }
    geom = rf.from_hf_config(gqa_cfg)
    dt = Dtypes(kv=1.0)
    scenario = dc_replace(
        EXAMPLE_SCENARIOS["chat-8k1k"],
        concurrencies=(1, 4),
    )
    splan = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=4),
    )
    for pt in splan.candidates:
        assert pt.decode_floor is not None
        df = pt.decode_floor
        expected_hard = max(
            df.hbm_ms, df.matmul_ms, df.ici_ms, df.chain_ms, df.hop_latency_ms
        )
        assert df.hard_floor_ms <= expected_hard + 1e-9
        if df.binding == "hbm":
            assert df.hard_floor_ms <= df.hbm_ms + 1e-9
        elif df.binding == "matmul":
            assert math.isclose(df.matmul_ms, df.hard_floor_ms, rel_tol=1e-5)


def test_explicit_gqa_token_stays_gqa_when_kv_lora_rank_positive() -> None:
    """Verify that explicit GQA token is sized as GQA when kv_lora_rank > 0."""
    dsa_base = {
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "layer_types": ["dsa"] * 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 64,
        "kv_lora_rank": 512,
        "q_lora_rank": 1024,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 2048,
        "intermediate_size": 8192,
        "vocab_size": 32000,
    }
    geom_idx = rf.from_hf_config(dsa_base)
    geom_no_idx = dc_replace(geom_idx, index_n_heads=0)
    geom_gqa_tok = dc_replace(geom_idx, layer_types=("gqa",) * 4)
    dt = rf.Dtypes(w_dense=1.0, kv=1.0)
    par = rf.Parallel(tp=4, dp=1, ep=1, pp=1, pcp=1)
    seq_layout = SeqKvLayout(4, 2048)

    bd_idx = hbm_breakdown_per_core_bytes(geom_idx, dt, par, seq_layout)
    bd_no_idx = hbm_breakdown_per_core_bytes(geom_no_idx, dt, par, seq_layout)
    bd_gqa_tok = hbm_breakdown_per_core_bytes(geom_gqa_tok, dt, par, seq_layout)

    expected_idx_w = float(4 * 4096 * (64 // 4) * 128) * 1.0
    assert bd_idx.weights - bd_no_idx.weights == pytest.approx(expected_idx_w, rel=1e-6)
    assert bd_gqa_tok.weights < bd_idx.weights
    assert bd_gqa_tok.kv > 0.0
