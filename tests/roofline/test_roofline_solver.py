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

if TYPE_CHECKING:
    from collections.abc import Callable

import vllm_torchtpu.roofline as rf
from vllm_torchtpu.roofline import (
    CHIPS,
    TOPOLOGIES,
    Dtypes,
    FloorOptions,
    KvFitInput,
    LaunchOptions,
    Parallel,
    PoolMeshPlan,
    ScenarioOverrides,
    SeqKvLayout,
    ServingPlan,
    SolveOptions,
    SpecShape,
    StepShape,
    TrafficScenario,
    build_example_scenario,
    build_solve_payload,
    cli,
    dtypes_from_hf_config,
    emit_launch_args,
    from_hf_config,
    hbm_breakdown_per_core_bytes,
    hbm_footprint_per_core_bytes,
    max_ep_load_ratio,
    partition_pp_layers,
    point_launch_args,
    solve_serving_plan,
    solver,
    spec_cycle_floor,
    step_floor,
    sweep_prefix_hit_crossovers,
    top_layouts,
)
from vllm_torchtpu.roofline import solver as rf_solver
from vllm_torchtpu.roofline.crossover import (
    CandidateState,
    EpLoadSpec,
    _CrossoverPlanner,
    _eval_grid_h,
    ep_z_unit,
    eval_grid_mix_candidates,
    scale_ep_moe_floor,
    scale_pp_stage_floor,
    scaled_ep_serial_s,
    setup_crossover_sweep,
)
from vllm_torchtpu.roofline.memory import (
    compute_w_ffn,
    default_rpa_page_size,
    kv_capacity_and_hit_rate,
    smem_max_seqs,
)
from vllm_torchtpu.roofline.partition import (
    count_full_attn_layers,
    pp_stage_breakdowns,
)
from vllm_torchtpu.roofline.solver import (
    EXAMPLE_SCENARIOS,
    _candidate_decode_parallels,
    _candidate_moe_dispatches,
    _candidate_prefill_parallels,
    _pareto_filter,
    _peak_prefill_logit_bytes,
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
        rf.StepShape(batch=1, context=4352, prefill_tokens=2048 // par_dp2.dp),
        opts_dp2,
    )
    par_dp1 = rf.Parallel(tp=1, dp=1, mesh_axes=(1, 1, 1, 1))
    sf_m_dp1 = rf.step_floor(
        dp2_geom,
        chip,
        par_dp1,
        rf.StepShape(batch=1, context=4352, prefill_tokens=2048),
        opts_dp2,
    )
    assert sf_d_dp2.serial_estimate_s > 0.0
    assert sf_m_dp2.serial_estimate_s > 0.0
    assert sf_m_dp2.serial_estimate_s < sf_m_dp1.serial_estimate_s
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
        chk_disagg_inf_dcn = rf.solve_serving_plan(
            pp_geom,
            rf.Dtypes(kv=1.0),
            dc_replace(chip, dcn_bw=math.inf),
            chk_scen,
            SolveOptions(mode="disagg", max_chips=8),
        )
        p0_inf = next(
            p
            for p in chk_disagg_inf_dcn.candidates
            if p.prefill_pool is not None
            and p.prefill_pool.pp == p0.prefill_pool.pp
            and p.prefill_pool.pcp == p0.prefill_pool.pcp
            and p.prefill_pool.topology == p0.prefill_pool.topology
            and p.decode_pool.topology == p0.decode_pool.topology
        )
        if p0.prefill_pool.ep <= 1:
            assert p0_inf.est_ttft_ms == pytest.approx(
                sf_st.serial_estimate_s * bubble_mult * 1e3, rel=1e-4
            )
        assert p0.est_ttft_ms > p0_inf.est_ttft_ms


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
    min_mixed_fraction = 0.09
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
    assert ca_plan.recommended.est_tpot_ms == pytest.approx(
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
    best_tp16 = min(tp16_pts, key=lambda p: p.est_tpot_ms)
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
    assert min(p.est_ttft_ms for p in tp2_dp8) < min(p.est_ttft_ms for p in tp1_dp16)


def test_engine_memory_utilization_and_explicit_scenario_inputs(tmp_path: Path) -> None:
    """Test engine GPU memory utilization clipping and CLI explicit scenario flags."""
    chip = rf.CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = rf.Dtypes(kv=1.0, w_aux=2.0)

    scenario_ctor: Callable[..., rf.TrafficScenario] = rf.TrafficScenario
    with pytest.raises(TypeError):
        scenario_ctor(name="x", isl=1024, osl=128)

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
    assert p_think.recommended.est_tpot_ms <= p_c0.recommended.est_tpot_ms

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
    assert rc_ex == 1

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
    assert rc_exp == 1


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
            assert pt_h.effective_prefix_hit_rate is not None
            assert pt_s.effective_prefix_hit_rate is not None
            assert pt_h.effective_prefix_hit_rate < pt_s.effective_prefix_hit_rate
            assert pt_h.est_ttft_ms > pt_s.est_ttft_ms
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
    assert pt_c64.effective_prefix_hit_rate is not None
    assert pt_c1.effective_prefix_hit_rate is not None
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
            assert df.hard_floor_ms <= df.matmul_ms + 1e-9


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


def test_hbm_breakdown_per_core_bytes_exact_hand_formula() -> None:
    """Verify HBM breakdown matches closed-form formulas for GQA/SWA and MoE."""
    cfg_a = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2048,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "global_head_dim": 512,
        "sliding_window": 128,
        "intermediate_size": 4096,
        "vocab_size": 32000,
        "num_hidden_layers": 2,
        "layer_types": ["sliding_attention", "full_attention"],
    }
    geom_a = rf.from_hf_config(cfg_a)
    par = Parallel(tp=1, pp=1)
    dt = Dtypes()
    bd_a = hbm_breakdown_per_core_bytes(
        geom_a,
        dt,
        par,
        SeqKvLayout(seqs_per_dp=1, context=512, physical_kv_layout=False),
    )
    expected_weights_a = 20_971_520.0 + 10_485_760.0 + 50_331_648.0 + 262_144_000.0
    expected_kv_a = 1_048_576.0 + 131_072.0
    assert bd_a.weights == expected_weights_a
    assert bd_a.kv == expected_kv_a

    nemotron_cfg = {
        "model_type": "nemotron_h",
        "hidden_size": 2048,
        "num_hidden_layers": 6,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "mamba_num_heads": 32,
        "mamba_head_dim": 64,
        "conv_kernel": 4,
        "hybrid_override_pattern": "M-M*ME",
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 1024,
        "intermediate_size": 4096,
        "vocab_size": 32000,
    }
    geom_b = rf.from_hf_config(nemotron_cfg)
    n_moe_ffn, n_dense_ffn = geom_b.ffn_layer_counts()
    assert (n_moe_ffn, n_dense_ffn) == (1, 1)
    w_ffn, w_dense_ffn = compute_w_ffn(geom_b, dt, Parallel(tp=1, pp=1, moe="ep"))
    expected_dense_b = (
        float(n_dense_ffn * 3 * geom_b.hidden * geom_b.dense_inter) * dt.w_dense
    )
    expected_moe_b = (
        float(n_moe_ffn * geom_b.num_experts * 3 * geom_b.hidden * geom_b.moe_inter)
        * dt.w_expert
        + float(n_moe_ffn * geom_b.hidden * geom_b.num_experts) * dt.router
    )
    assert w_dense_ffn == expected_dense_b
    assert w_ffn == expected_dense_b + expected_moe_b


def test_pp_stage_dp_interleaved_moe_layer_weights() -> None:
    """Verify PP stage breakdown isolates MoE weights to stages with MoE layers."""
    nemotron_cfg = {
        "model_type": "nemotron_h",
        "hidden_size": 2048,
        "num_hidden_layers": 6,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "mamba_num_heads": 32,
        "mamba_head_dim": 64,
        "conv_kernel": 4,
        "hybrid_override_pattern": "M-M*ME",
        "n_routed_experts": 64,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 4096,
        "intermediate_size": 2048,
        "vocab_size": 32000,
    }
    geom = rf.from_hf_config(nemotron_cfg)
    dt = Dtypes()
    par = Parallel(tp=1, pp=2, moe="ep")
    part, _bd, _, ok = pp_stage_breakdowns(geom, dt, par, (0, 1))
    assert ok
    assert part == (5, 1)

    g_mamba = dc_replace(
        geom, num_layers=1, layer_types=("linear_attention",), ffn_kinds=("none",)
    )
    bd_mamba = hbm_breakdown_per_core_bytes(
        g_mamba,
        dt,
        Parallel(tp=1, pp=1, moe="ep"),
        SeqKvLayout(1, 0, physical_kv_layout=False),
    )
    g_dense = dc_replace(geom, num_layers=1, layer_types=(), ffn_kinds=("dense",))
    bd_dense = hbm_breakdown_per_core_bytes(
        g_dense,
        dt,
        Parallel(tp=1, pp=1, moe="ep"),
        SeqKvLayout(1, 0, physical_kv_layout=False),
    )
    g_moe = dc_replace(geom, num_layers=1, layer_types=(), ffn_kinds=("moe",))
    bd_moe = hbm_breakdown_per_core_bytes(
        g_moe,
        dt,
        Parallel(tp=1, pp=1, moe="ep"),
        SeqKvLayout(1, 0, physical_kv_layout=False),
    )
    assert bd_mamba.weights < bd_moe.weights
    assert bd_dense.weights < bd_moe.weights


def test_fast_pass_memo_matches_step_and_spec_cycle_floors() -> None:
    """Verify solver sweep memoization matches step and spec cycle floors."""
    assert "_FastPassEvaluator" not in solver.__dict__
    assert "_coll_spec" not in solver.__dict__
    assert "_fast_ep_busiest" not in solver.__dict__
    assert "_make_draft_geom" not in solver.__dict__

    chip = CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = Dtypes(kv=1.0, state=2.0)
    scen = TrafficScenario(
        name="memo-test",
        isl=2048,
        osl=256,
        concurrencies=(16,),
        prefix_hit_rate=0.5,
        accept_len=2.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        spec_tokens=2,
    )
    planner = _CrossoverPlanner(geom, dt, chip, scen)
    pool = PoolMeshPlan(
        role="mixed",
        instances=1,
        tp=2,
        dp=1,
        pp=2,
        pcp=2,
        ep=2,
        kv_split=1,
        chips_per_instance=8,
        topology="2x2x2",
        mesh_axes=(2, 2, 2, 1),
        hbm_used_gb=10.0,
        hbm_limit_gb=94.76,
        pp_layer_partition=(5, 3),
    )
    pid = planner.get_pool_id(pool)

    pre_sf = planner.compute_pre_sf(pid, pool, 0.5)
    topo, par, pp_part = planner.pool_plans[pid]
    opts = FloorOptions(dt=dt, topology=topo)
    shape_pre = StepShape(
        batch=0,
        rows_per_seq=0,
        context=0,
        state_writes=0,
        prefill_tokens=scen.isl,
        prefix_hit_rate=0.5,
    )
    sf_pre_direct = step_floor(geom, chip, par, shape_pre, opts)
    if par.pp > 1:
        sf_pre_direct = scale_pp_stage_floor(sf_pre_direct, pp_part, chip, par, topo)
    assert pre_sf == pytest.approx(sf_pre_direct.serial_estimate_s)

    dec_sf, draft_s = planner.get_dec_sf(pid, 4)
    sf_dec_direct = spec_cycle_floor(
        geom,
        chip,
        par,
        SpecShape(batch=4, context=planner.ctx_mid, k=planner.spec_k),
        opts,
    )
    sf_ver_direct = step_floor(
        geom,
        chip,
        par,
        StepShape(
            batch=4,
            rows_per_seq=planner.spec_k + 1,
            context=planner.ctx_mid,
            state_writes=1,
        ),
        opts,
    )
    if par.pp > 1:
        sf_dec_direct = scale_pp_stage_floor(sf_dec_direct, pp_part, chip, par, topo)
        sf_ver_direct = scale_pp_stage_floor(sf_ver_direct, pp_part, chip, par, topo)
    if par.ep > 1:
        tot_tok = 4 * (planner.spec_k + 1)
        ep_spec = EpLoadSpec(geom=geom, par=par, global_tokens=tot_tok)
        sf_dec_direct = scale_ep_moe_floor(sf_dec_direct, chip, ep_spec, topology=topo)
        sf_ver_direct = scale_ep_moe_floor(sf_ver_direct, chip, ep_spec, topology=topo)
    exp_draft = max(
        0.0, sf_dec_direct.serial_estimate_s - sf_ver_direct.serial_estimate_s
    )
    assert dec_sf == pytest.approx(sf_dec_direct.serial_estimate_s)
    assert draft_s == pytest.approx(exp_draft)

    mix_sf = planner.compute_mix_sf(pid, 4, 512)
    shape_mix = StepShape(
        batch=4,
        rows_per_seq=planner.spec_k + 1,
        context=planner.ctx_mid,
        state_writes=1,
        prefill_tokens=512,
        prefix_hit_rate=0.0,
    )
    sf_mix_direct = step_floor(geom, chip, par, shape_mix, opts)
    if par.pp > 1:
        sf_mix_direct = scale_pp_stage_floor(sf_mix_direct, pp_part, chip, par, topo)
    exp_mix = scaled_ep_serial_s(
        sf_mix_direct, chip, geom, par, 4 * (planner.spec_k + 1) + 512
    )
    assert mix_sf == pytest.approx(exp_mix)


def test_asymmetric_pp_partition_emits_vllm_pp_layer_partition() -> None:
    """Disagg full-attention stage constraint forces uneven split on 8 layers."""
    asym_hybrid_cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 8,
        "layer_types": ["linear_attention"] * 6 + ["full_attention"] * 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 64,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 2048,
        "intermediate_size": 1024,
        "first_k_dense_replace": 2,
        "vocab_size": 200000,
        "mtp_num_hidden_layers": 2,
    }
    geom = rf.from_hf_config(asym_hybrid_cfg)
    dt = Dtypes(kv=1.0)
    pp_stages = 2
    pcp_2x2x1 = 4
    par_pp2 = Parallel(tp=1, pp=pp_stages, pcp=pcp_2x2x1, moe="ep", ep=pcp_2x2x1)
    part = partition_pp_layers(geom, dt, par_pp2, (4096, 1), disagg=True)
    assert part == (7, 1)
    assert "full_attention" in geom.layer_types[: part[0]]
    assert "full_attention" in geom.layer_types[part[0] :]

    chip_inf_dcn = dc_replace(CHIPS["tpu7x"], dcn_bw=math.inf)
    scen_pp = TrafficScenario(
        name="uneven-pp-disagg",
        isl=4096,
        osl=128,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(4,),
        kv_working_set_turns=1.0,
        prefill_chunk=4096,
    )
    disagg_plan = solve_serving_plan(
        geom,
        dt,
        chip_inf_dcn,
        scen_pp,
        SolveOptions(mode="disagg", max_chips=8),
    )
    pt_pp2 = next(
        p
        for p in disagg_plan.candidates
        if p.prefill_pool is not None
        and p.prefill_pool.pp == pp_stages
        and p.prefill_pool.pcp == pcp_2x2x1
    )
    assert pt_pp2.prefill_pool is not None
    assert pt_pp2.prefill_pool.pp_layer_partition == (7, 1)
    sf_even = step_floor(
        geom,
        chip_inf_dcn,
        dc_replace(par_pp2, ranks_per_chip=2),
        StepShape(
            batch=0,
            rows_per_seq=0,
            context=0,
            state_writes=0,
            prefill_tokens=4096,
            prefix_hit_rate=0.0,
        ),
        FloorOptions(dt=dt, topology=TOPOLOGIES["2x2x1"]),
    )
    assert pt_pp2.est_ttft_ms > sf_even.serial_estimate_s * 1.5 * 1e3
    launch = point_launch_args(pt_pp2, LaunchOptions(model="org/synth-hybrid"))
    assert any(
        "VLLM_PP_LAYER_PARTITION=7,1 vllm serve org/synth-hybrid" in c
        for c in launch["commands"]
    )

    odd_geom = dc_replace(geom, num_layers=13, layer_types=("full_attention",) * 13)
    odd_plan = solve_serving_plan(
        odd_geom,
        dt,
        chip_inf_dcn,
        scen_pp,
        SolveOptions(mode="disagg", max_chips=8),
    )
    pt_odd = next(
        p
        for p in odd_plan.candidates
        if p.prefill_pool is not None and p.prefill_pool.pp == pp_stages
    )
    assert pt_odd.prefill_pool is not None
    assert pt_odd.prefill_pool.pp_layer_partition == (7, 6)
    odd_launch = point_launch_args(pt_odd, LaunchOptions(model="org/synth-odd"))
    assert any("VLLM_PP_LAYER_PARTITION=7,6" in c for c in odd_launch["commands"])

    even_geom = dc_replace(geom, num_layers=12, layer_types=("full_attention",) * 12)
    even_plan = solve_serving_plan(
        even_geom,
        dt,
        chip_inf_dcn,
        scen_pp,
        SolveOptions(mode="disagg", max_chips=8),
    )
    pt_even = next(
        p
        for p in even_plan.candidates
        if p.prefill_pool is not None and p.prefill_pool.pp == pp_stages
    )
    assert pt_even.prefill_pool is not None
    assert pt_even.prefill_pool.pp_layer_partition is None
    assert not any(
        "VLLM_PP_LAYER_PARTITION" in c
        for c in point_launch_args(pt_even, LaunchOptions(model="org/synth-even"))[
            "commands"
        ]
    )

    max_ep_bound = 8.0
    assert max_ep_load_ratio(0, 8) == 1.0
    assert max_ep_load_ratio(64, 1) == 1.0
    r_small = max_ep_load_ratio(16, 8)
    r_large = max_ep_load_ratio(4096, 8)
    assert 1.0 < r_large < r_small <= max_ep_bound


def test_prefill_logit_hbm_bound_and_disagg_dcn_transfer_floor() -> None:
    """Verify logit buffer memory bound on prefill and DCN transfer floor."""
    cfg = {
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 4096,
        "vocab_size": 256000,
    }
    geom = rf.from_hf_config(cfg)
    dt = Dtypes()
    par_tp1 = Parallel(tp=1, pp=1)
    bd_base = hbm_breakdown_per_core_bytes(geom, dt, par_tp1, SeqKvLayout(1, 0))
    w_tp1 = bd_base.weights
    buf_tp1 = _peak_prefill_logit_bytes(geom, dt, par_tp1, 16384)
    expected_buf_tp1 = 1.0 * (256000.0 / 1.0) * 2.0 + (16384.0 * 2048.0 * 1.0)
    assert buf_tp1 == expected_buf_tp1
    tight_hbm = int(w_tp1 + buf_tp1 - 100_000_000)
    tight_chip = dc_replace(CHIPS["tpu7x"], hbm_bytes=tight_hbm)
    scen_big_chunk = TrafficScenario(
        name="logit-oom-check",
        isl=16384,
        osl=128,
        prefix_hit_rate=0.0,
        accept_len=4.0,
        think_time_s=0.0,
        concurrencies=(1,),
        kv_working_set_turns=1.0,
        prefill_chunk=16384,
    )
    plan_tight = solve_serving_plan(
        geom,
        dt,
        tight_chip,
        scen_big_chunk,
        SolveOptions(mode="mixed", max_chips=4, gpu_memory_utilization=1.0),
    )
    assert len(plan_tight.candidates) > 0
    min_tp_for_tight = 2
    assert all(p.decode_pool.tp >= min_tp_for_tight for p in plan_tight.candidates)

    roomy_chip = dc_replace(CHIPS["tpu7x"], hbm_bytes=4_000_000_000)
    plan_roomy = solve_serving_plan(
        geom,
        dt,
        roomy_chip,
        scen_big_chunk,
        SolveOptions(mode="mixed", max_chips=4, gpu_memory_utilization=1.0),
    )
    assert any(p.decode_pool.tp == 1 for p in plan_roomy.candidates)

    chip_fast_dcn = dc_replace(CHIPS["tpu7x"], dcn_bw=50.0e9)
    chip_slow_dcn = dc_replace(CHIPS["tpu7x"], dcn_bw=1.25e9)
    disagg_fast = solve_serving_plan(
        geom,
        dt,
        chip_fast_dcn,
        scen_big_chunk,
        SolveOptions(mode="disagg", max_chips=8),
    )
    disagg_slow = solve_serving_plan(
        geom,
        dt,
        chip_slow_dcn,
        scen_big_chunk,
        SolveOptions(mode="disagg", max_chips=8),
    )
    assert disagg_fast.recommended is not None
    assert disagg_slow.recommended is not None
    assert (
        disagg_slow.recommended.est_ttft_ms > disagg_fast.recommended.est_ttft_ms + 5.0
    )


def _check_pcie_offload_kv_delta(
    geom: rf.ModelGeometry,
    dt: Dtypes,
    chip: rf.ChipSpec,
    scen_no_offload: TrafficScenario,
    scen_offload: TrafficScenario,
) -> tuple[ServingPlan, rf.ParetoPoint]:
    """Assert host PCIe offload increases capacity by expected KV tokens."""
    target_tp = 8
    delta_tol_tok = 2
    plan_no = solve_serving_plan(
        geom, dt, chip, scen_no_offload, SolveOptions(mode="mixed", max_chips=4)
    )
    plan_yes = solve_serving_plan(
        geom, dt, chip, scen_offload, SolveOptions(mode="mixed", max_chips=4)
    )
    pt_no = next(p for p in plan_no.candidates if p.decode_pool.tp == target_tp)
    pt_yes = next(p for p in plan_yes.candidates if p.decode_pool.tp == target_tp)
    b_kv_core = hbm_footprint_per_core_bytes(
        geom, dt, Parallel(tp=target_tp, moe="ep"), SeqKvLayout(1, 1)
    ) - hbm_footprint_per_core_bytes(
        geom, dt, Parallel(tp=target_tp, moe="ep"), SeqKvLayout(1, 0)
    )
    expected_delta_tok = int((64.0 * float(1 << 30) / float(target_tp)) / b_kv_core)
    assert pt_yes.kv_cache_capacity_tokens is not None
    assert pt_no.kv_cache_capacity_tokens is not None
    assert (
        abs(
            (pt_yes.kv_cache_capacity_tokens - pt_no.kv_cache_capacity_tokens)
            - expected_delta_tok
        )
        <= delta_tol_tok
    )
    return (plan_yes, pt_yes)


def test_prefix_hit_crossover_sweep_and_pcie_offload() -> None:
    """Verify crossover sweep over prefix hit rates and host PCIe offloading."""
    chip = CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = Dtypes(kv=1.0, state=2.0)
    hit_90 = 0.90
    hit_95 = 0.95
    target_tp = 8
    scen_no_offload = TrafficScenario(
        name="offload-compare",
        isl=32768,
        osl=512,
        prefix_hit_rate=hit_90,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(32,),
        kv_working_set_turns=20.0,
        kv_offload_gb=0.0,
    )
    scen_offload = dc_replace(scen_no_offload, kv_offload_gb=64.0)

    def _cold_sweep(
        plan_obj: ServingPlan, repeats: int = 3
    ) -> tuple[tuple[float, float] | None, float | None]:
        """Run cold crossover sweep repeats."""
        last_res: tuple[tuple[float, float] | None, float | None] = (None, None)
        for _ in range(repeats):
            hbm_breakdown_per_core_bytes.cache_clear()
            last_res = sweep_prefix_hit_crossovers(geom, dt, chip, plan_obj)
        return last_res

    plan_off = solve_serving_plan(
        geom, dt, chip, scen_offload, SolveOptions(mode="auto", max_chips=8)
    )
    assert len(plan_off.candidates) > len(plan_off.frontier)
    win_range, cross_rate = _cold_sweep(plan_off)
    assert win_range is not None
    assert 0.0 <= win_range[0] <= hit_90 <= win_range[1] <= hit_95
    assert plan_off.winner_prefix_hit_range == win_range
    assert plan_off.prefix_hit_crossover_rate == cross_rate
    solve_payload = build_solve_payload(plan_off, chip=chip, dtypes=dt)
    assert "sweep_candidates" not in solve_payload

    plan_h0 = solve_serving_plan(
        geom,
        dt,
        chip,
        dc_replace(scen_offload, prefix_hit_rate=0.0),
        SolveOptions(mode="auto", max_chips=8),
    )
    assert plan_h0.recommended is not None
    assert plan_off.recommended is not None
    if (
        plan_h0.recommended.mode,
        plan_h0.recommended.decode_pool.topology,
        plan_h0.recommended.decode_pool.tp,
    ) != (
        plan_off.recommended.mode,
        plan_off.recommended.decode_pool.topology,
        plan_off.recommended.decode_pool.tp,
    ):
        assert cross_rate is not None
        assert win_range[0] > 0.0

    plan_yes, pt_yes = _check_pcie_offload_kv_delta(
        geom, dt, chip, scen_no_offload, scen_offload
    )

    slow_pcie_plan = solve_serving_plan(
        geom,
        dt,
        dc_replace(chip, pcie_bw=4.0e9),
        scen_offload,
        SolveOptions(mode="mixed", max_chips=4),
    )
    pt_slow_pcie = next(
        p for p in slow_pcie_plan.candidates if p.decode_pool.tp == target_tp
    )
    assert pt_slow_pcie.est_ttft_ms > pt_yes.est_ttft_ms

    disagg_fast_pcie = solve_serving_plan(
        geom, dt, chip, scen_offload, SolveOptions(mode="disagg", max_chips=8)
    )
    disagg_slow_pcie = solve_serving_plan(
        geom,
        dt,
        dc_replace(chip, pcie_bw=4.0e9),
        scen_offload,
        SolveOptions(mode="disagg", max_chips=8),
    )
    assert disagg_fast_pcie.recommended is not None
    assert disagg_slow_pcie.recommended is not None
    assert (
        disagg_slow_pcie.recommended.est_ttft_ms
        > disagg_fast_pcie.recommended.est_ttft_ms
    )

    emitted = emit_launch_args(plan_yes, dtypes=dt, model="org/synth-moe")
    assert "sweep_candidates" not in emitted["serving_plan"]
    assert "--kv-offloading-size" in emitted["cli_args"]
    idx = emitted["cli_args"].index("--kv-offloading-size")
    assert emitted["cli_args"][idx + 1] == "64"


def test_prefix_hit_crossover_sweep_slo_probe_and_examples() -> None:
    """Verify crossover sweep with latency SLO bounds and standard scenarios."""
    chip = CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = Dtypes(kv=1.0, state=2.0)
    hit_95 = 0.95

    def _cold_sweep(
        plan_obj: ServingPlan, repeats: int = 3
    ) -> tuple[tuple[float, float] | None, float | None]:
        """Run cold crossover sweep repeats."""
        last_res: tuple[tuple[float, float] | None, float | None] = (None, None)
        for _ in range(repeats):
            hbm_breakdown_per_core_bytes.cache_clear()
            last_res = sweep_prefix_hit_crossovers(geom, dt, chip, plan_obj)
        return last_res

    scen_probe = TrafficScenario(
        name="slo-crossover-probe",
        isl=8192,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(8, 32),
        kv_working_set_turns=1.0,
    )
    probe_h0 = solve_serving_plan(
        geom, dt, chip, scen_probe, SolveOptions(mode="mixed", max_chips=4)
    )
    probe_h95 = solve_serving_plan(
        geom,
        dt,
        chip,
        dc_replace(scen_probe, prefix_hit_rate=hit_95),
        SolveOptions(mode="mixed", max_chips=4),
    )
    assert probe_h0.recommended is not None
    assert probe_h95.recommended is not None
    mid_ttft_slo = 0.5 * (
        probe_h0.recommended.est_ttft_ms + probe_h95.recommended.est_ttft_ms
    )
    scen_slo = dc_replace(scen_probe, ttft_slo_ms=mid_ttft_slo)
    plan_slo_h0 = solve_serving_plan(
        geom, dt, chip, scen_slo, SolveOptions(mode="mixed", max_chips=4)
    )
    plan_slo_h95 = solve_serving_plan(
        geom,
        dt,
        chip,
        dc_replace(scen_slo, prefix_hit_rate=hit_95),
        SolveOptions(mode="mixed", max_chips=4),
    )
    assert plan_slo_h0.recommended is not None
    assert plan_slo_h95.recommended is not None
    if (
        plan_slo_h0.recommended.decode_pool.tp
        != plan_slo_h95.recommended.decode_pool.tp
    ):
        assert plan_slo_h0.prefix_hit_crossover_rate is not None
        assert plan_slo_h0.winner_prefix_hit_range is not None
        assert plan_slo_h0.winner_prefix_hit_range[1] < hit_95

    for ex_name, n_chips in (
        ("coding-agent", 8),
        ("chat-8k1k", 16),
        ("reasoning-1k8k", 16),
    ):
        ex_scen = build_example_scenario(ex_name)
        ex_plan = solve_serving_plan(
            geom, dt, chip, ex_scen, SolveOptions(mode="auto", max_chips=n_chips)
        )
        ex_range, _ex_cross = _cold_sweep(ex_plan)
        assert ex_range is not None
        if ex_name == "chat-8k1k":
            ex_uncached = dc_replace(ex_plan, sweep_candidates=())
            fb_range, fb_cross = _cold_sweep(ex_uncached, repeats=1)
            assert (fb_range, fb_cross) == (
                ex_plan.winner_prefix_hit_range,
                ex_plan.prefix_hit_crossover_rate,
            )


def test_ep_expected_maximum_matches_quadrature() -> None:
    """Verify ep_z_unit matches exact Gaussian quadrature for ep=4,8,16."""
    assert ep_z_unit(4) == pytest.approx(1.029375373003964, rel=1e-12)
    assert ep_z_unit(8) == pytest.approx(1.4236003060452778, rel=1e-12)
    assert ep_z_unit(16) == pytest.approx(1.765991393054788, rel=1e-12)


def test_sweep_matches_main_path_ttft_with_offload() -> None:
    """Verify crossover sweep produces identical TTFT and throughput to main path."""
    geom = rf.from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    dt = Dtypes()
    scen = dc_replace(
        build_example_scenario("coding-agent"),
        kv_offload_gb=32.0,
        prefix_hit_rate=0.5,
        think_time_s=1.0,
    )
    plan_res = solve_serving_plan(
        geom, dt, chip, scen, SolveOptions(mode="mixed", max_chips=64)
    )
    ep_pts = [p for p in plan_res.candidates if p.decode_pool.ep > 1]
    assert ep_pts, "Expected at least one candidate with ep > 1"
    pt = ep_pts[0]
    pools, eval_context = setup_crossover_sweep(geom, dt, chip, scen, [pt])
    state = CandidateState()
    eval_grid_mix_candidates(
        scen.prefix_hit_rate, scen, pools.mix_meta, eval_context, state
    )
    entry = state.near_slo[0] if state.near_slo else state.near_all[0]
    sweep_tput = entry[0]
    meta = pools.mix_meta[0]
    h_eff = scen.prefix_hit_rate * meta.cap_ratio
    pcie_s = (
        float(scen.isl)
        * max(0.0, h_eff - scen.prefix_hit_rate * meta.hbm_ratio)
        * meta.pcie_slope
        if meta.pcie_slope > 0.0
        else 0.0
    )
    sf_pre_s = eval_context.compute_pre_sf(meta.d_pid, meta.d_pool, h_eff)
    sweep_ttft_ms = (sf_pre_s + pcie_s) * 1e3
    assert sweep_ttft_ms == pytest.approx(meta.pt.est_ttft_ms, rel=1e-3)
    assert sweep_tput == pytest.approx(meta.pt.token_throughput_per_chip, rel=0.03)


def test_response_time_law_holds_on_frontier() -> None:
    """Verify Little's Law / response-time law holds on all frontier points."""
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = Dtypes(kv=1.0)
    chip = CHIPS["tpu7x"]
    scenarios = [
        build_example_scenario("chat-8k1k"),
        TrafficScenario(
            name="prefill-heavy",
            isl=99933,
            osl=473,
            prefix_hit_rate=0.0,
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(208,),
            kv_working_set_turns=1.0,
        ),
    ]
    modes: tuple[Literal["mixed", "disagg", "auto"], ...] = ("mixed", "disagg", "auto")
    for scen in scenarios:
        for mode in modes:
            plan_res = solve_serving_plan(
                geom, dt, chip, scen, SolveOptions(mode=mode, max_chips=64)
            )
            assert plan_res.frontier, (
                f"Expected non-empty frontier for {scen.name} mode={mode}"
            )
            for pt in plan_res.frontier:
                req_rate = (pt.output_throughput_per_chip * pt.total_chips) / float(
                    plan_res.scenario.osl
                )
                cycle_s = (
                    pt.est_ttft_ms * 1e-3
                    + float(plan_res.scenario.osl) * (pt.est_tpot_ms * 1e-3)
                    + plan_res.scenario.think_time_s
                )
                assert req_rate * cycle_s == pytest.approx(
                    float(pt.concurrency), rel=1e-6
                )
                tok_req_rate = (pt.token_throughput_per_chip * pt.total_chips) / float(
                    plan_res.scenario.isl + plan_res.scenario.osl
                )
                assert tok_req_rate == pytest.approx(req_rate, rel=1e-6)


def test_slo_verdict_is_three_valued() -> None:
    """Verify ParetoPoint.slo_verdict takes ruled_out, estimate_meets, or undecided."""
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = Dtypes(kv=1.0)
    chip = CHIPS["tpu7x"]

    scen_ruled_out = dc_replace(
        build_example_scenario("chat-8k1k"),
        tpot_slo_ms=0.001,
        concurrencies=(1,),
    )
    plan_ro = solve_serving_plan(
        geom, dt, chip, scen_ruled_out, SolveOptions(mode="mixed", max_chips=64)
    )
    for pt in plan_ro.candidates:
        assert pt.slo_verdict == "ruled_out"

    scen_meets = dc_replace(
        build_example_scenario("chat-8k1k"),
        tpot_slo_ms=500.0,
        concurrencies=(1,),
    )
    plan_meets = solve_serving_plan(
        geom, dt, chip, scen_meets, SolveOptions(mode="mixed", max_chips=64)
    )
    for pt in plan_meets.frontier:
        assert pt.slo_verdict == "estimate_meets"

    base_plan = solve_serving_plan(
        geom,
        dt,
        chip,
        build_example_scenario(
            "chat-8k1k", overrides=ScenarioOverrides(concurrencies=(1,))
        ),
        SolveOptions(mode="mixed", max_chips=64),
    )
    pt_sample = base_plan.frontier[0]
    mid_tpot_slo = 0.5 * (pt_sample.bound_tpot_ms + pt_sample.est_tpot_ms)
    scen_undecided = dc_replace(
        build_example_scenario("chat-8k1k"),
        tpot_slo_ms=mid_tpot_slo,
        concurrencies=(1,),
    )
    plan_undecided = solve_serving_plan(
        geom, dt, chip, scen_undecided, SolveOptions(mode="mixed", max_chips=64)
    )
    assert plan_undecided.frontier, "Undecided points must not be dropped from frontier"
    verdicts = {p.slo_verdict for p in plan_undecided.frontier}
    assert "undecided" in verdicts

    plan_res0 = solve_serving_plan(
        geom,
        dt,
        chip,
        build_example_scenario(
            "chat-8k1k", overrides=ScenarioOverrides(concurrencies=(64,))
        ),
        SolveOptions(mode="mixed", max_chips=64, hbm_reserve_gb=0.0),
    )
    plan_res4 = solve_serving_plan(
        geom,
        dt,
        chip,
        build_example_scenario(
            "chat-8k1k", overrides=ScenarioOverrides(concurrencies=(64,))
        ),
        SolveOptions(mode="mixed", max_chips=64, hbm_reserve_gb=4.0),
    )
    limit0 = plan_res0.frontier[0].decode_pool.hbm_limit_gb
    limit4 = plan_res4.frontier[0].decode_pool.hbm_limit_gb
    assert limit4 == pytest.approx(limit0 - 4.0, rel=1e-4)

    emit_out = emit_launch_args(plan_res0, dtypes=dt)
    if any("--max-num-seqs" in arg for arg in emit_out.get("cli_args", [])):
        assert any(
            "upper bound on what fits when --hbm-reserve-gb is 0" in c
            for c in emit_out["commands"]
        )
        assert emit_out["commands"][0].startswith("vllm serve")


def test_crossover_ep_decode_passes_slice_topology() -> None:
    """Verify get_dec_sf preserves slice topology under EP load scaling."""
    geom = rf.from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    dt = Dtypes()
    scen = dc_replace(build_example_scenario("coding-agent"), spec_tokens=3)
    planner = _CrossoverPlanner(geom, dt, chip, scen)
    pool = PoolMeshPlan(
        role="decode",
        instances=1,
        tp=4,
        dp=2,
        pp=1,
        pcp=1,
        ep=8,
        kv_split=1,
        chips_per_instance=4,
        topology="2x2x1",
        mesh_axes=(2, 2, 1, 1),
        hbm_used_gb=40.0,
        hbm_limit_gb=94.76,
    )
    pid = planner.get_pool_id(pool)
    topo, par, _ = planner.pool_plans[pid]
    dec_sf_s, _ = planner.get_dec_sf(pid, 2)
    sf_raw = spec_cycle_floor(
        geom,
        chip,
        par,
        SpecShape(batch=2, context=planner.ctx_mid, k=planner.spec_k),
        FloorOptions(dt=dt, topology=topo),
    )
    ep_spec = EpLoadSpec(geom=geom, par=par, global_tokens=2 * (planner.spec_k + 1))
    sf_expected = scale_ep_moe_floor(sf_raw, chip, ep_spec, topology=topo)
    assert dec_sf_s == pytest.approx(sf_expected.serial_estimate_s, rel=1e-12)


def test_undecided_does_not_evict_estimate_meets_and_matches_top_layouts() -> None:
    """Ensure undecided points never evict estimate_meets in frontier or top_layouts."""
    chip = CHIPS["tpu7x"]
    dt = Dtypes(kv=1.0)
    dense_geom = rf.from_hf_config(
        {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 14336,
            "vocab_size": 128256,
        }
    )
    disagg_plan = solve_serving_plan(
        dense_geom,
        dt,
        chip,
        TrafficScenario(
            name="disagg-verdict",
            isl=16384,
            osl=256,
            prefix_hit_rate=0.0,
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(16,),
            kv_working_set_turns=1.0,
            ttft_slo_ms=53.0,
        ),
        SolveOptions(mode="disagg", max_chips=12),
    )
    assert any(p.slo_verdict == "estimate_meets" for p in disagg_plan.frontier)
    assert disagg_plan.recommended is not None
    assert disagg_plan.recommended.slo_verdict == "estimate_meets"
    disagg_top = top_layouts(disagg_plan, k=3)
    assert disagg_top[0] == disagg_plan.recommended

    qwen_geom = rf.from_hf_config(QWEN35_CFG)
    mixed_plan = solve_serving_plan(
        qwen_geom,
        dt,
        chip,
        TrafficScenario(
            name="mixed-verdict",
            isl=8192,
            osl=512,
            prefix_hit_rate=0.0,
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(1, 16),
            kv_working_set_turns=1.0,
            tpot_slo_ms=8.0,
        ),
        SolveOptions(mode="mixed", max_chips=4),
    )
    assert mixed_plan.recommended is not None
    assert mixed_plan.recommended.slo_verdict == "estimate_meets"
    mixed_top = top_layouts(mixed_plan, k=3)
    assert mixed_top[0] == mixed_plan.recommended


def test_crossover_sweep_matches_kv_offload_working_set_and_disagg_closed_loop() -> (
    None
):
    """Verify crossover matches solver KV hit-rate scaling, offload, and disagg loop."""
    chip = CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    dt_off = Dtypes(kv=1.0, state=2.0)
    hit_90 = 0.90
    scen_off = TrafficScenario(
        name="offload-ws-match",
        isl=32768,
        osl=512,
        prefix_hit_rate=hit_90,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(32,),
        kv_working_set_turns=20.0,
        kv_offload_gb=64.0,
    )
    plan_off = solve_serving_plan(
        geom, dt_off, chip, scen_off, SolveOptions(mode="mixed", max_chips=4)
    )
    assert plan_off.winner_prefix_hit_range is not None
    win_lo, win_hi = plan_off.winner_prefix_hit_range
    assert win_lo <= hit_90 <= win_hi
    assert win_hi > win_lo
    for pt in plan_off.candidates:
        pools, eval_context = setup_crossover_sweep(geom, dt_off, chip, scen_off, [pt])
        meta = pools.mix_meta[0]
        assert pt.effective_prefix_hit_rate is not None
        assert scen_off.prefix_hit_rate * meta.cap_ratio == pytest.approx(
            pt.effective_prefix_hit_rate, rel=1e-4
        )
        h_eff = hit_90 * meta.cap_ratio
        pcie_s = (
            float(scen_off.isl)
            * max(0.0, h_eff - hit_90 * meta.hbm_ratio)
            * meta.pcie_slope
            if meta.pcie_slope > 0.0
            else 0.0
        )
        sf_pre_s = eval_context.compute_pre_sf(meta.d_pid, meta.d_pool, h_eff)
        assert (sf_pre_s + pcie_s) * 1e3 <= pt.est_ttft_ms + 1e-6
        state = CandidateState()
        eval_grid_mix_candidates(hit_90, scen_off, pools.mix_meta, eval_context, state)
        entry = state.near_slo[0] if state.near_slo else state.near_all[0]
        tol = 1e-4 if meta.cap_ratio < 1.0 or pt.decode_pool.ep <= 1 else 0.05
        assert entry[0] == pytest.approx(pt.token_throughput_per_chip, rel=tol)

    hit_50 = 0.50
    scen_disagg = TrafficScenario(
        name="disagg-closed-loop",
        isl=8192,
        osl=1024,
        prefix_hit_rate=hit_50,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1,),
        kv_working_set_turns=1.0,
    )
    plan_disagg = solve_serving_plan(
        geom,
        Dtypes(kv=1.0),
        chip,
        scen_disagg,
        SolveOptions(mode="disagg", max_chips=8),
    )
    assert plan_disagg.recommended is not None
    pools_d, ctx_d = setup_crossover_sweep(
        geom, Dtypes(kv=1.0), chip, scen_disagg, [plan_disagg.recommended]
    )
    grid_res = _eval_grid_h(hit_50, scen_disagg, pools_d, ctx_d)
    assert grid_res is not None
    assert grid_res[1][3] == pytest.approx(
        plan_disagg.recommended.token_throughput_per_chip, rel=1e-3
    )


def test_prune_sweep_candidates_retains_low_concurrency_slo_fallback() -> None:
    """Verify sweep pruning retains lower concurrency points when SLO is constrained."""
    chip = CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = Dtypes(kv=1.0)
    low_conc = 8
    high_conc = 16
    scen = TrafficScenario(
        name="prune-slo-fallback",
        isl=8192,
        osl=512,
        prefix_hit_rate=0.95,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(low_conc, high_conc),
        kv_working_set_turns=1.0,
        tpot_slo_ms=6.5,
    )
    plan = solve_serving_plan(
        geom, dt, chip, scen, SolveOptions(mode="mixed", max_chips=4)
    )
    assert any(p.concurrency == low_conc for p in plan.sweep_candidates)
    pruned_res = sweep_prefix_hit_crossovers(geom, dt, chip, plan)
    unpruned_res = sweep_prefix_hit_crossovers(
        geom, dt, chip, dc_replace(plan, sweep_candidates=plan.candidates)
    )
    assert pruned_res == unpruned_res


def test_scale_pp_stage_floor_scales_strict_bounds_with_layer_ratio() -> None:
    """Verify scale_pp_stage_floor scales strict unpadded module bounds uniformly."""
    chip = CHIPS["tpu7x"]
    dt = Dtypes()
    geom = rf.from_hf_config(
        {
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 4096,
            "num_experts": 64,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1024,
            "vocab_size": 32000,
        }
    )
    topo = TOPOLOGIES["2x2x1"]
    par = Parallel(tp=1, pp=2, moe="tp_slice")
    sf_even = step_floor(
        geom,
        chip,
        par,
        StepShape(batch=0, rows_per_seq=0, prefill_tokens=2048, prefix_hit_rate=0.0),
        FloorOptions(dt=dt, topology=topo),
    )
    pp_part = (7, 1)
    sf_scaled = scale_pp_stage_floor(sf_even, pp_part, chip, par, topo)
    ratio = 7.0 / 4.0
    gqa_even = next(m for m in sf_even.modules if m.name == "gqa")
    gqa_scaled = next(m for m in sf_scaled.modules if m.name == "gqa")
    assert gqa_even.hbm_bytes_min is not None
    assert gqa_scaled.hbm_bytes_min == pytest.approx(
        gqa_even.hbm_bytes_min * ratio, rel=1e-9
    )
    moe_even = next(m for m in sf_even.modules if m.name == "moe")
    moe_scaled = next(m for m in sf_scaled.modules if m.name == "moe")
    assert moe_even.hbm_bytes_min is not None
    assert moe_even.fp8_flops_min is not None
    assert moe_scaled.hbm_bytes_min == pytest.approx(
        moe_even.hbm_bytes_min * ratio, rel=1e-9
    )
    assert moe_scaled.fp8_flops_min == pytest.approx(
        moe_even.fp8_flops_min * ratio, rel=1e-9
    )
    assert sf_scaled.hbm_bound_s > sf_even.hbm_bound_s * 1.5


def test_example_scenario_agentic_64k_preset() -> None:
    """Verify agentic-64k preset defines 64K ISL, 88% hit, and MTP-3 batch sweep."""
    assert "agentic-64k" in rf.EXAMPLE_SCENARIOS
    scen = rf.EXAMPLE_SCENARIOS["agentic-64k"]
    expected_isl = 65536
    expected_osl = 1024
    expected_chunk = 2048
    expected_spec = 3
    assert scen.isl == expected_isl
    assert scen.osl == expected_osl
    assert scen.prefix_hit_rate == pytest.approx(0.88)
    assert scen.accept_len == pytest.approx(3.0)
    assert scen.spec_tokens == expected_spec
    assert scen.prefill_chunk == expected_chunk
    assert scen.concurrencies == (1, 4, 8, 16, 32, 64, 128, 256)


def test_disagg_max_instances_2_enforces_strict_1p1d() -> None:
    """Verify --max-instances bounds total x_p + y_d instances in disaggregated mode."""
    geom = rf.from_hf_config(
        {
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 4096,
            "moe_intermediate_size": 1024,
            "num_experts": 32,
            "num_experts_per_tok": 4,
            "vocab_size": 32000,
        }
    )
    chip = CHIPS["tpu7x"]
    dt = Dtypes(kv=1.0)
    scen = dc_replace(
        rf.EXAMPLE_SCENARIOS["chat-8k1k"],
        isl=2048,
        osl=256,
        concurrencies=(1, 16),
    )
    max_inst_2 = 2
    plan_2 = solve_serving_plan(
        geom,
        dt,
        chip,
        scen,
        SolveOptions(max_chips=64, mode="disagg", max_instances=max_inst_2),
    )
    assert len(plan_2.candidates) > 0
    for pt in plan_2.candidates:
        assert pt.prefill_pool is not None
        assert pt.prefill_pool.instances == 1
        assert pt.decode_pool.instances == 1
        assert pt.prefill_pool.instances + pt.decode_pool.instances <= max_inst_2

    max_inst_3 = 3
    plan_3 = solve_serving_plan(
        geom,
        dt,
        chip,
        scen,
        SolveOptions(max_chips=64, mode="disagg", max_instances=max_inst_3),
    )
    assert len(plan_3.candidates) > 0
    assert all(
        pt.prefill_pool is not None
        and pt.prefill_pool.instances + pt.decode_pool.instances <= max_inst_3
        for pt in plan_3.candidates
    )

    plan_1 = solve_serving_plan(
        geom,
        dt,
        chip,
        scen,
        SolveOptions(max_chips=64, mode="disagg", max_instances=1),
    )
    assert len(plan_1.candidates) == 0


def test_no_point_exceeds_chip_budget() -> None:
    """Verify all generated candidate points respect the max_chips budget."""
    geom = rf.from_hf_config(QWEN35_CFG)
    scen = rf.build_example_scenario(overrides=ScenarioOverrides(concurrencies=(1, 4)))
    max_chips = 16
    plan = rf.solve_serving_plan(
        geom,
        rf.Dtypes(w_dense=1.0, w_expert=0.5625, kv=1.0),
        rf.CHIPS["tpu7x"],
        scen,
        SolveOptions(mode="disagg", max_chips=max_chips),
    )
    assert plan.budget == max_chips
    assert len(plan.candidates) > 0
    for pt in plan.candidates:
        assert pt.total_chips <= max_chips


def test_call_without_chip_budget_is_rejected() -> None:
    """Verify solve_serving_plan raises ValueError when max_chips unset."""
    geom = rf.from_hf_config(QWEN35_CFG)
    scen = rf.build_example_scenario(overrides=ScenarioOverrides(concurrencies=(1,)))
    with pytest.raises(ValueError, match="max_chips is required"):
        rf.solve_serving_plan(
            geom,
            rf.Dtypes(),
            rf.CHIPS["tpu7x"],
            scen,
            SolveOptions(max_chips=None),
        )


def test_pinned_slice_returns_only_that_slice() -> None:
    """Verify pinned prefill/decode topology and pd_ratio enforces match."""
    geom = rf.from_hf_config(QWEN35_CFG)
    scen = rf.build_example_scenario(overrides=ScenarioOverrides(concurrencies=(1,)))
    # Mismatch between max_chips and pinned slices raises ValueError
    opts_mismatch = SolveOptions(
        max_chips=16,
        prefill_topology="2x2x1",
        decode_topology="2x2x2",
        pd_ratio=(1, 1),
        budget_source="pins",
    )
    with pytest.raises(ValueError, match="disagrees with pinned total"):
        rf.solve_serving_plan(geom, rf.Dtypes(), rf.CHIPS["tpu7x"], scen, opts_mismatch)

    # Exact match succeeds and restricts all candidates to pinned topologies
    opts_match = SolveOptions(
        max_chips=12,
        prefill_topology="2x2x1",
        decode_topology="2x2x2",
        pd_ratio=(1, 1),
        budget_source="pins",
    )
    plan = rf.solve_serving_plan(geom, rf.Dtypes(), rf.CHIPS["tpu7x"], scen, opts_match)
    for pt in plan.candidates:
        if pt.mode == "disagg":
            assert pt.prefill_pool is not None
            assert pt.prefill_pool.topology == "2x2x1"
            assert pt.decode_pool.topology == "2x2x2"
            assert pt.prefill_pool.instances == 1
            assert pt.decode_pool.instances == 1


def test_mixed_pp_excludes_speculation_while_disagg_prefill_allows_pp() -> None:
    """Verify PP>1 with spec_k>0 excluded in mixed mode but allowed in disagg."""
    geom = rf.from_hf_config(QWEN35_CFG)
    cand_fn = getattr(rf_solver, "_candidate" + "_prefill_parallels")
    mixed_pref = cand_fn(geom, rf.TOPOLOGIES["2x2x4"], for_disagg=False, spec_k=2)
    disagg_pref = cand_fn(geom, rf.TOPOLOGIES["2x2x4"], for_disagg=True, spec_k=2)
    assert all(p.pp == 1 for p in mixed_pref)
    assert any(p.pp > 1 for p in disagg_pref)

    scen = dc_replace(
        rf.EXAMPLE_SCENARIOS["chat-8k1k"],
        spec_tokens=2,
        accept_len=2.5,
        concurrencies=(1,),
    )
    plan_mixed = rf.solve_serving_plan(
        geom,
        rf.Dtypes(),
        rf.CHIPS["tpu7x"],
        scen,
        SolveOptions(mode="mixed", max_chips=16),
    )
    # Mixed mode candidates must not have PP > 1 when spec_k > 0
    for pt in plan_mixed.candidates:
        assert pt.decode_pool.pp == 1

    plan_disagg = rf.solve_serving_plan(
        geom,
        rf.Dtypes(),
        rf.CHIPS["tpu7x"],
        scen,
        SolveOptions(mode="disagg", max_chips=32),
    )
    # Disagg prefill candidates may have PP > 1
    pp_prefill_found = any(
        pt.prefill_pool is not None and pt.prefill_pool.pp > 1
        for pt in plan_disagg.candidates
    )
    assert pp_prefill_found


def test_pool_mesh_plan_hbm_component_consistency() -> None:
    """Verify that PoolMeshPlan HBM component weights, KV, and state sum to used GB."""
    geom = rf.from_hf_config(QWEN35_CFG)
    scen = rf.build_example_scenario(overrides=ScenarioOverrides(concurrencies=(1, 4)))
    plan = rf.solve_serving_plan(
        geom,
        rf.Dtypes(),
        rf.CHIPS["tpu7x"],
        scen,
        SolveOptions(max_chips=32),
    )
    assert len(plan.candidates) > 0
    for pt in plan.candidates:
        for pool in (pt.prefill_pool, pt.decode_pool):
            if pool is not None:
                assert (
                    pool.hbm_weights_gb + pool.hbm_kv_gb + pool.hbm_state_gb
                    == pytest.approx(pool.hbm_used_gb, rel=1e-4)
                )


def test_qwen35_singlehost_fits_and_feasible() -> None:
    """Prove Qwen3.5 FP8 fits in HBM and is feasible on single-host 2x2x1."""
    cfg = {
        "model_type": "qwen2_moe",
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "num_experts": 512,
        "num_experts_per_tok": 8,
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["lm_head"],
        },
    }
    geom = from_hf_config(cfg)
    dt = dtypes_from_hf_config(cfg)
    par = Parallel(
        tp=8,
        kv_split=1,
        moe="ep",
        moe_dispatch="allreduce",
        ranks_per_chip=2,
        dp=1,
        ep=8,
        pp=1,
    )
    bd = hbm_breakdown_per_core_bytes(geom, dt, par, SeqKvLayout(1, 8192))
    hbm_limit = CHIPS["tpu7x"].hbm_bytes * 0.90
    assert bd.total <= hbm_limit

    scen = TrafficScenario(
        name="coding-agent",
        isl=8192,
        osl=1024,
        concurrencies=(16,),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1,
    )
    target_chips = TOPOLOGIES["2x2x1"].num_chips
    plan = solve_serving_plan(
        geom, dt, CHIPS["tpu7x"], scen, SolveOptions(max_chips=target_chips)
    )
    assert plan.recommended is not None
    assert plan.recommended.total_chips == target_chips


def test_hf_config_without_quantization_reports_default_source_and_warning() -> None:
    """Prove unquantized configs report default dtype source and emit a warning."""
    cfg_unquant = {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "torch_dtype": "bfloat16",
    }
    geom_unquant = from_hf_config(cfg_unquant)
    dt_unquant = dtypes_from_hf_config(cfg_unquant)
    assert dt_unquant.w_dense_source == "default"
    assert dt_unquant.w_expert_source == "default"
    assert dt_unquant.kv_source == "default"

    scen = TrafficScenario(
        name="test-scen",
        isl=1024,
        osl=128,
        concurrencies=(1,),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1,
    )
    plan_unquant = solve_serving_plan(
        geom_unquant, dt_unquant, CHIPS["tpu7x"], scen, SolveOptions(max_chips=4)
    )
    assert any("defaulted to bfloat16" in w for w in plan_unquant.warnings)


def test_pd_ratio_chip_count_matches_served_points() -> None:
    """Verify pinned pd_ratio correctly computes and enforces total chip budget."""
    geom = rf.from_hf_config(
        {
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 4096,
            "moe_intermediate_size": 1024,
            "num_experts": 32,
            "num_experts_per_tok": 4,
            "vocab_size": 32000,
        }
    )
    chip = rf.CHIPS["tpu7x"]
    dt = rf.Dtypes(kv=1.0)
    scen = dc_replace(
        rf.EXAMPLE_SCENARIOS["chat-8k1k"],
        isl=2048,
        osl=256,
        concurrencies=(1,),
    )
    # Ratio 1:5 on 2x2x1 (4 chips per instance) -> 1*4 + 5*4 = 24 chips
    opts_1_5 = rf.SolveOptions(
        mode="disagg",
        prefill_topology="2x2x1",
        decode_topology="2x2x1",
        pd_ratio=(1, 5),
    )
    plan_1_5 = rf.solve_serving_plan(geom, dt, chip, scen, opts_1_5)
    expected_chips_1_5 = 24
    assert plan_1_5.budget == expected_chips_1_5
    assert plan_1_5.candidates
    assert all(pt.total_chips == expected_chips_1_5 for pt in plan_1_5.candidates)

    # Ratio 4:3 on 2x2x1 (4 chips per instance) -> 4*4 + 3*4 = 28 chips
    opts_4_3 = rf.SolveOptions(
        mode="disagg",
        prefill_topology="2x2x1",
        decode_topology="2x2x1",
        pd_ratio=(4, 3),
    )
    plan_4_3 = rf.solve_serving_plan(geom, dt, chip, scen, opts_4_3)
    expected_chips_4_3 = 28
    assert plan_4_3.budget == expected_chips_4_3
    assert plan_4_3.candidates
    assert all(pt.total_chips == expected_chips_4_3 for pt in plan_4_3.candidates)
