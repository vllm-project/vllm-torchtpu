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
import time
from dataclasses import replace as dc_replace
from pathlib import Path

import pytest

import vllm_torchtpu.roofline as rf
from vllm_torchtpu.roofline import cli
from vllm_torchtpu.roofline.solver import (
    _candidate_decode_parallels,
    _candidate_prefill_parallels,
)

try:
    from vllm_torchtpu.roofline.solver import (
        _default_rpa_page_size,
        _smem_max_seqs,
        _valid_gdn_degree,
        _valid_raiden_gdn_pair,
    )
except ImportError:
    _default_rpa_page_size = None
    _smem_max_seqs = None
    _valid_gdn_degree = None
    _valid_raiden_gdn_pair = None
import math

from vllm_torchtpu.roofline import CHIPS, Dtypes, Parallel
from vllm_torchtpu.roofline.solver import (
    hbm_breakdown_per_core_bytes,
    hbm_footprint_per_core_bytes,
    solve_serving_plan,
)

EXAMPLE_ACCEPT_LEN: float = 3.0

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


def test_solver_hybrid_moe_mixed_vs_disagg_and_crossover(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    q35_geom = rf.from_hf_config(QWEN35_CFG)
    chip = rf.CHIPS["tpu7x"]
    scen = rf.EXAMPLE_SCENARIOS["chat-8k1k"]
    mixed_plan = rf.solve_serving_plan(
        q35_geom, rf.Dtypes(kv=1.0), chip, scen, mode="mixed", max_chips=8
    )
    disagg_plan = rf.solve_serving_plan(
        q35_geom, rf.Dtypes(kv=1.0), chip, scen, mode="disagg", max_chips=8
    )
    mixed_c16 = [p for p in mixed_plan.frontier if p.concurrency >= 16]
    assert mixed_c16 and any(p.p_mixed >= 0.10 for p in mixed_c16)
    assert disagg_plan.recommended is not None
    assert disagg_plan.recommended.p_mixed == 0.0

    # coding-agent (accept_len=3.4) with 1 MTP layer prices k=3 window.
    ca_scen = rf.build_example_scenario("coding-agent", concurrencies=(1,))
    ca_plan = rf.solve_serving_plan(
        q35_geom, rf.Dtypes(kv=1.0), chip, ca_scen, mode="disagg", max_chips=8
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
        max(1, (1 + rec_d.dp - 1) // rec_d.dp),
        ca_scen.isl + ca_scen.osl // 2,
        3,
        rf.Dtypes(kv=1.0),
        topology=rf.TOPOLOGIES[rec_d.topology],
    )
    assert ca_plan.recommended.p90_tpot_ms == pytest.approx(
        sf_k3.serial_floor_s * 1e3 / ca_scen.accept_len, rel=1e-4
    )

    # DP > 1 MoE decode candidates price cross-DP EP collectives even under allreduce.
    dp_cands = [
        p
        for p in _candidate_decode_parallels(q35_geom, rf.TOPOLOGIES["2x2x1"])
        if p.dp > 1
    ]
    assert dp_cands
    for par_dp in dp_cands:
        assert (
            par_dp.moe == "ep"
            and par_dp.ep == 8
            and par_dp.moe_dispatch in ("ag_ep", "a2a_push")
        )
        sf_dp = rf.step_floor(
            q35_geom,
            chip,
            par_dp,
            rf.StepShape(batch=4, context=1024),
            rf.Dtypes(kv=1.0),
            topology=rf.TOPOLOGIES["2x2x1"],
        )
        sf_dp_ar = rf.step_floor(
            q35_geom,
            chip,
            dc_replace(par_dp, moe_dispatch="allreduce"),
            rf.StepShape(batch=4, context=1024),
            rf.Dtypes(kv=1.0),
            topology=rf.TOPOLOGIES["2x2x1"],
        )
        assert sf_dp.chain_s > 0.0
        if par_dp.moe_dispatch == "ag_ep":
            assert sf_dp_ar.chain_s == pytest.approx(sf_dp.chain_s)

    # Mixed prefill work scales as 1/dp (chunk_tok per replica).
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
    dp2_plan = rf.solve_serving_plan(
        dp2_geom, rf.Dtypes(kv=1.0), chip, dp2_scen, mode="mixed", max_chips=1
    )
    assert dp2_plan.recommended is not None and dp2_plan.recommended.decode_pool.dp == 2
    par_dp2 = rf.Parallel(tp=1, dp=2, mesh_axes=(1, 1, 1, 1))
    sf_d_dp2 = rf.step_floor(
        dp2_geom,
        chip,
        par_dp2,
        rf.StepShape(batch=4, context=4352),
        rf.Dtypes(kv=1.0),
        topology=rf.TOPOLOGIES["1x1x1"],
    )
    sf_m_dp2 = rf.step_floor(
        dp2_geom,
        chip,
        par_dp2,
        rf.StepShape(batch=4, context=4352, prefill_tokens=2048),
        rf.Dtypes(kv=1.0),
        topology=rf.TOPOLOGIES["1x1x1"],
    )
    assert sf_d_dp2.serial_floor_s > 0.0 and sf_m_dp2.serial_floor_s > 0.0
    exp_p_mix = float(4 * 2) / float(512 + 4 * 2)
    assert dp2_plan.recommended.p_mixed == pytest.approx(exp_p_mix, rel=1e-5)

    # PP > 1 prefill TTFT includes stage fill for n_chunks=1 and n_chunks=4.
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
            pp_geom, rf.Dtypes(kv=1.0), chip, chk_scen, mode="disagg", max_chips=8
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
            rf.Dtypes(kv=1.0),
            topology=rf.TOPOLOGIES[p0.prefill_pool.topology],
        )
        bubble_mult = 1.0 + float(p0.prefill_pool.pp - 1) / float(n_chk)
        chk_disagg_inf_dcn = rf.solve_serving_plan(
            pp_geom,
            rf.Dtypes(kv=1.0),
            dc_replace(chip, dcn_bw=math.inf),
            chk_scen,
            mode="disagg",
            max_chips=8,
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
            assert p0_inf.p90_ttft_ms == pytest.approx(
                sf_st.serial_floor_s * bubble_mult * 1e3, rel=1e-4
            )
        assert p0.p90_ttft_ms > p0_inf.p90_ttft_ms

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
    sf_tp32_c4 = rf.step_floor(
        hybrid_2t_geom,
        chip,
        par_tp32,
        rf.StepShape(batch=4, context=131072),
        hybrid_2t_dt,
        topology=topo_32,
    )
    sf_dp4_c4 = rf.step_floor(
        hybrid_2t_geom,
        chip,
        par_tp8_dp4,
        rf.StepShape(batch=1, context=131072),
        hybrid_2t_dt,
        topology=topo_32,
    )
    assert sf_tp32_c4.serial_floor_s < sf_dp4_c4.serial_floor_s

    sf_tp32_c256 = rf.step_floor(
        hybrid_2t_geom,
        chip,
        par_tp32,
        rf.StepShape(batch=256, context=8192),
        hybrid_2t_dt,
        topology=topo_32,
    )
    sf_dp4_c256 = rf.step_floor(
        hybrid_2t_geom,
        chip,
        par_tp8_dp4,
        rf.StepShape(batch=64, context=8192),
        hybrid_2t_dt,
        topology=topo_32,
    )
    assert sf_dp4_c256.serial_floor_s < sf_tp32_c256.serial_floor_s

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
    oss_disagg = rf.solve_serving_plan(
        oss_geom, oss_dt, chip, scen, mode="disagg", max_chips=7
    )
    assert oss_disagg.recommended is not None
    assert oss_disagg.recommended.total_chips <= 7
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
        q35_geom, rf.Dtypes(kv=1.0), chip, infeasible_scen, mode="mixed", max_chips=4
    )
    assert any("INFEASIBLE_SLO" in w for w in inf_plan.warnings)

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


def test_physical_kv_page_capacity_and_unpruned_regimes() -> None:
    chip = rf.CHIPS["tpu7x"]
    q35_geom = rf.from_hf_config(QWEN35_CFG)
    q35_dt = rf.Dtypes(
        w_dense=1.0, w_expert=1.0, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0
    )

    hbm_mtp1 = rf.hbm_footprint_per_core_bytes(
        q35_geom, q35_dt, rf.Parallel(tp=8, kv_split=1, moe="ep"), 0, 0
    )
    hbm_mtp0 = rf.hbm_footprint_per_core_bytes(
        dc_replace(q35_geom, mtp_layers=0),
        q35_dt,
        rf.Parallel(tp=8, kv_split=1, moe="ep"),
        0,
        0,
    )
    assert (hbm_mtp1 - hbm_mtp0) > 0.80e9
    hbm_phys = rf.hbm_footprint_per_core_bytes(
        q35_geom,
        q35_dt,
        rf.Parallel(tp=8, kv_split=1, moe="ep"),
        1,
        8192,
        physical_kv_layout=True,
    )
    hbm_log = rf.hbm_footprint_per_core_bytes(
        q35_geom,
        q35_dt,
        rf.Parallel(tp=8, kv_split=1, moe="ep"),
        1,
        8192,
        physical_kv_layout=False,
    )
    assert (hbm_phys - hbm_log) == pytest.approx(15.0 * 8192.0 * 512.0, abs=2.0)

    scen = rf.TrafficScenario(
        name="custom",
        isl=1024,
        osl=1024,
        concurrencies=(64,),
        prefix_hit_rate=0.5,
        accept_len=EXAMPLE_ACCEPT_LEN,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
    )
    disagg = rf.solve_serving_plan(
        q35_geom, q35_dt, chip, scen, mode="disagg", max_chips=12
    )
    assert all(p.concurrency == 64 for p in disagg.frontier)
    assert any(
        p.prefill_pool is not None and p.prefill_pool.pp >= 8 for p in disagg.candidates
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
            accept_len=EXAMPLE_ACCEPT_LEN,
            think_time_s=0.0,
            kv_working_set_turns=1.0,
            spec_tokens=3,
        ),
        mode="mixed",
        max_chips=8,
        gpu_memory_utilization=0.85,
    )
    tp16_pts = [
        p
        for p in dsa_c48.candidates
        if p.decode_pool.tp == 16
        and p.decode_pool.ep == 1
        and p.decode_pool.max_num_seqs is None
    ]
    best_tp16 = min(tp16_pts, key=lambda p: p.p90_tpot_ms)
    assert best_tp16.decode_pool.kv_split == 16

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
            accept_len=EXAMPLE_ACCEPT_LEN,
            think_time_s=0.0,
            kv_working_set_turns=1.0,
            spec_tokens=3,
            kv_offload_gb=48.0,
        ),
        mode="mixed",
        max_chips=8,
        gpu_memory_utilization=0.85,
    )
    tp2_dp8 = [
        p
        for p in w4_dsa_c64.candidates
        if p.decode_pool.tp == 2 and p.decode_pool.dp == 8 and p.decode_pool.ep == 16
    ]
    tp1_dp16 = [
        p
        for p in w4_dsa_c64.candidates
        if p.decode_pool.tp == 1 and p.decode_pool.dp == 16 and p.decode_pool.ep == 16
    ]
    assert tp2_dp8 and tp1_dp16
    assert min(p.p90_tpot_ms for p in tp2_dp8) < min(p.p90_tpot_ms for p in tp1_dp16)


def test_engine_memory_utilization_and_explicit_scenario_inputs(tmp_path: Path) -> None:
    chip = rf.CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = rf.Dtypes(kv=1.0, w_aux=2.0)

    with pytest.raises(TypeError):
        rf.TrafficScenario(name="x", isl=1024, osl=128)

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
        mode="mixed",
        max_chips=4,
        gpu_memory_utilization=0.45,
    )
    assert plan_45.gpu_memory_utilization == 0.45
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
        mode="mixed",
        max_chips=4,
        gpu_memory_utilization=0.92,
    )
    assert plan_92.gpu_memory_utilization == 0.92
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
    p_c0 = rf.solve_serving_plan(geom, dt, chip, scen_c0, mode="mixed", max_chips=4)
    p_think = rf.solve_serving_plan(
        geom, dt, chip, scen_think, mode="mixed", max_chips=4
    )
    assert p_c0.recommended is not None and p_think.recommended is not None
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


def test_hbm_breakdown_matches_footprint():
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

    for geom in geoms:
        bd = hbm_breakdown_per_core_bytes(geom, dt, par, seqs_per_dp=16, context=2048)
        footprint = hbm_footprint_per_core_bytes(
            geom, dt, par, seqs_per_dp=16, context=2048
        )
        assert bd.total == footprint
        assert math.isclose(bd.weights + bd.kv + bd.state, bd.total, rel_tol=1e-9)
        assert bd.weights > 0
        assert bd.kv >= 0
        if geom.linear_k_heads:
            assert bd.state > 0
        else:
            assert bd.state == 0.0


@pytest.mark.parametrize("mode", ["mixed", "disagg", "auto"])
def test_search_stats_funnel_is_monotone(mode: str):
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
        rf.EXAMPLE_SCENARIOS["chat-8k1k"],
        concurrencies=(1, 4),
        ttft_slo_ms=5000.0,
        tpot_slo_ms=500.0,
    )
    plan = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        max_chips=4,
        mode=mode,
    )
    assert plan.search is not None
    s = plan.search
    assert s.enumerated >= s.hbm_fit >= s.slo_met >= s.frontier >= 1
    assert s.frontier == len(plan.frontier)


def test_decode_floor_matches_binding():
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
        rf.EXAMPLE_SCENARIOS["chat-8k1k"],
        concurrencies=(1, 4),
    )
    plan = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        max_chips=4,
        mode="mixed",
    )
    for pt in plan.candidates:
        assert pt.decode_floor is not None
        df = pt.decode_floor
        expected_hard = max(
            df.hbm_ms, df.matmul_ms, df.ici_ms, df.chain_ms, df.hop_latency_ms
        )
        assert math.isclose(df.hard_floor_ms, expected_hard, rel_tol=1e-5)
        if df.binding == "hbm":
            assert math.isclose(df.hbm_ms, df.hard_floor_ms, rel_tol=1e-5)
        elif df.binding == "matmul":
            assert math.isclose(df.matmul_ms, df.hard_floor_ms, rel_tol=1e-5)


def test_runtime_legality_heads_pp_tp_pcp_ep_gdn_and_smem_cap() -> None:
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
    dec_pars = _candidate_decode_parallels(uneven_geom, rf.TOPOLOGIES["2x2x1"])
    assert all(28 % p.tp == 0 and (6 % p.tp == 0 or p.tp % 6 == 0) for p in dec_pars)
    assert all(p.tp not in (4, 8) for p in dec_pars)

    pre_pars = _candidate_prefill_parallels(uneven_geom, rf.TOPOLOGIES["1x2x1"])
    assert all(not (p.pp > 1 and p.tp > 1) for p in pre_pars)
    assert any(p.pp == 2 and p.tp == 1 and p.pcp == 2 for p in pre_pars)

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
    disagg_pre = _candidate_prefill_parallels(
        hybrid_8l, rf.TOPOLOGIES["2x2x1"], for_disagg=True
    )
    assert all(p.pp <= 2 for p in disagg_pre)
    assert not _valid_gdn_degree(dc_replace(hybrid_8l, linear_v_heads=64), 6)
    assert not _valid_raiden_gdn_pair(
        hybrid_8l, rf.Parallel(tp=2, pcp=1), rf.Parallel(tp=8, pcp=1)
    )
    assert _valid_raiden_gdn_pair(
        hybrid_8l, rf.Parallel(tp=1, pcp=2, pp=2), rf.Parallel(tp=8, pcp=1)
    )

    moe_geom = rf.from_hf_config(QWEN35_CFG)
    moe_pre = _candidate_prefill_parallels(moe_geom, rf.TOPOLOGIES["1x2x1"])
    for p in moe_pre:
        if p.pcp > 1:
            assert p.moe == "ep" and p.ep == p.tp * p.pcp

    assert _default_rpa_page_size(262144) == 16
    assert _smem_max_seqs(262144, 1, 16) == 8


def test_disagg_capacity_clamped_hit_rate_and_prefill_dispatch() -> None:
    chip = rf.CHIPS["tpu7x"]
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
        geom, dt, chip, scen_small, mode="disagg", max_chips=4
    )
    plan_huge = rf.solve_serving_plan(
        geom, dt, chip, scen_huge, mode="disagg", max_chips=4
    )
    assert plan_small.recommended is not None and plan_huge.recommended is not None
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
        geom, dt, chip, scen_think, mode="auto", max_chips=4
    )
    assert plan_think.recommended is not None and plan_think.recommended.slo_met

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
    pt_c64 = next(
        p
        for p in rf.solve_serving_plan(
            geom, dt, chip, scen_capped_ws, mode="mixed", max_chips=1
        ).candidates
        if p.decode_pool.tp == 2 and p.decode_pool.dp == 1
    )
    pt_c1 = next(
        p
        for p in rf.solve_serving_plan(
            geom, dt, chip, scen_single_ws, mode="mixed", max_chips=1
        ).candidates
        if p.decode_pool.tp == 2 and p.decode_pool.dp == 1
    )
    assert pt_c64.effective_prefix_hit_rate < pt_c1.effective_prefix_hit_rate


def test_hbm_breakdown_dsa_indexer_weights_and_mla_gqa_parity() -> None:
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

    bd_idx = hbm_breakdown_per_core_bytes(geom_idx, dt, par, 4, 2048)
    bd_no_idx = hbm_breakdown_per_core_bytes(geom_no_idx, dt, par, 4, 2048)
    bd_gqa_tok = hbm_breakdown_per_core_bytes(geom_gqa_tok, dt, par, 4, 2048)

    expected_idx_w = float(4 * 4096 * (64 // 4) * 128) * 1.0
    assert bd_idx.weights - bd_no_idx.weights == pytest.approx(expected_idx_w, rel=1e-6)
    assert bd_gqa_tok == bd_idx


def test_asymmetric_pp_partition_emits_vllm_pp_layer_partition() -> None:
    # 1. Disagg full-attention stage constraint forces uneven (7, 1) split on 8 layers
    # and scales bottleneck stage prefill floor by 7 / 4 = 1.75x, while retaining
    # full-attention MTP weights even though layer 0 is linear_attention.
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
    dt = rf.Dtypes(kv=1.0)
    par_pp2 = rf.Parallel(tp=1, pp=2, pcp=2, moe="ep", ep=2)
    part = rf.partition_pp_layers(
        geom,
        dt,
        par_pp2,
        context=4096,
        seqs_per_dp=1,
        disagg=True,
    )
    assert part == (7, 1)
    assert "full_attention" in geom.layer_types[: part[0]]
    assert "full_attention" in geom.layer_types[part[0] :]

    # Verify solve_serving_plan wires uneven PP into PoolMeshPlan, scales TTFT by
    # max(part) / ceil(L / pp) = 7 / 4, and emits VLLM_PP_LAYER_PARTITION=7,1.
    chip_inf_dcn = dc_replace(rf.CHIPS["tpu7x"], dcn_bw=math.inf)
    scen_pp = rf.TrafficScenario(
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
    disagg_plan = rf.solve_serving_plan(
        geom, dt, chip_inf_dcn, scen_pp, mode="disagg", max_chips=4
    )
    pt_pp2 = next(
        p
        for p in disagg_plan.candidates
        if p.prefill_pool is not None
        and p.prefill_pool.pp == 2
        and p.prefill_pool.pcp == 2
    )
    assert pt_pp2.prefill_pool is not None
    assert pt_pp2.prefill_pool.pp_layer_partition == (7, 1)
    sf_even = rf.step_floor(
        geom,
        chip_inf_dcn,
        dc_replace(par_pp2, ranks_per_chip=2),
        rf.StepShape(
            batch=0,
            rows_per_seq=0,
            context=0,
            state_writes=0,
            prefill_tokens=4096,
            prefix_hit_rate=0.0,
        ),
        dt,
        topology=rf.TOPOLOGIES["1x2x1"],
    )
    # 7-layer heavy stage + EP load ratio makes TTFT strictly > 1.5x the 4-layer floor.
    assert pt_pp2.p90_ttft_ms > sf_even.serial_floor_s * 1.5 * 1e3
    launch = rf.point_launch_args(pt_pp2, model="org/synth-hybrid")
    assert (
        "VLLM_PP_LAYER_PARTITION=7,1 vllm serve org/synth-hybrid"
        in launch["commands"][0]
    )

    # 2. Odd layer count L=13 on PP=2 picks (7, 6) instead of putting 7 layers on
    # the heavy lm_head + MTP stage, while L=12 uniform stays (6, 6) without env prefix.
    odd_geom = dc_replace(
        geom,
        num_layers=13,
        layer_types=("full_attention",) * 13,
    )
    odd_plan = rf.solve_serving_plan(
        odd_geom, dt, chip_inf_dcn, scen_pp, mode="disagg", max_chips=4
    )
    pt_odd = next(
        p
        for p in odd_plan.candidates
        if p.prefill_pool is not None and p.prefill_pool.pp == 2
    )
    assert pt_odd.prefill_pool is not None
    assert pt_odd.prefill_pool.pp_layer_partition == (7, 6)
    odd_launch = rf.point_launch_args(pt_odd, model="org/synth-odd")
    assert "VLLM_PP_LAYER_PARTITION=7,6" in odd_launch["commands"][0]

    even_geom = dc_replace(geom, num_layers=12, layer_types=("full_attention",) * 12)
    even_plan = rf.solve_serving_plan(
        even_geom, dt, chip_inf_dcn, scen_pp, mode="disagg", max_chips=4
    )
    pt_even = next(
        p
        for p in even_plan.candidates
        if p.prefill_pool is not None and p.prefill_pool.pp == 2
    )
    assert pt_even.prefill_pool is not None
    assert pt_even.prefill_pool.pp_layer_partition is None
    assert (
        "VLLM_PP_LAYER_PARTITION"
        not in rf.point_launch_args(pt_even, model="org/synth-even")["commands"][0]
    )

    # 3. Direct unit check on max_ep_load_ratio boundary and decay.
    assert rf.max_ep_load_ratio(0, 8) == 1.0
    assert rf.max_ep_load_ratio(64, 1) == 1.0
    r_small = rf.max_ep_load_ratio(16, 8)
    r_large = rf.max_ep_load_ratio(4096, 8)
    assert 1.0 < r_large < r_small <= 8.0


def test_prefill_logit_hbm_bound_and_disagg_dcn_transfer_floor() -> None:
    large_vocab_cfg = {
        "hidden_size": 2048,
        "num_hidden_layers": 8,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 4096,
        "vocab_size": 262144,
        "mtp_num_hidden_layers": 3,
    }
    geom = rf.from_hf_config(large_vocab_cfg)
    dt = rf.Dtypes(w_dense=1.0, lm_head=2.0, kv=1.0, act="a16")
    # At b_res=1, TP1 weights + KV + MTP (~2.87 GB) fit inside 2.91 GB, while the
    # active-chunk hidden tensor + sampled-row vocab logits (~69 MB) push TP1 past
    # 2.91 GB without falsely charging 8.59 GB (chunk_rows * vocab) at 4.0 GB!
    bd_tp1 = rf.hbm_breakdown_per_core_bytes(
        geom, dt, rf.Parallel(tp=1), 1, 16384 + 128
    )
    tight_hbm = int(bd_tp1.total + 30_000_000)
    tight_chip = dc_replace(rf.CHIPS["tpu7x"], hbm_bytes=tight_hbm)
    scen_big_chunk = rf.TrafficScenario(
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
    plan_tight = rf.solve_serving_plan(
        geom,
        dt,
        tight_chip,
        scen_big_chunk,
        mode="mixed",
        max_chips=4,
        gpu_memory_utilization=1.0,
    )
    assert len(plan_tight.candidates) > 0
    assert all(p.decode_pool.tp >= 2 for p in plan_tight.candidates)

    # Verify TP1 IS admitted at 4.0 GB because compute_selected_logits gathers
    # sampled_rows before lm_head rather than materializing [chunk_rows, vocab].
    roomy_chip = dc_replace(rf.CHIPS["tpu7x"], hbm_bytes=4_000_000_000)
    plan_roomy = rf.solve_serving_plan(
        geom,
        dt,
        roomy_chip,
        scen_big_chunk,
        mode="mixed",
        max_chips=4,
        gpu_memory_utilization=1.0,
    )
    assert any(p.decode_pool.tp == 1 for p in plan_roomy.candidates)

    chip_fast_dcn = dc_replace(rf.CHIPS["tpu7x"], dcn_bw=50.0e9)
    chip_slow_dcn = dc_replace(rf.CHIPS["tpu7x"], dcn_bw=1.25e9)
    disagg_fast = rf.solve_serving_plan(
        geom, dt, chip_fast_dcn, scen_big_chunk, mode="disagg", max_chips=4
    )
    disagg_slow = rf.solve_serving_plan(
        geom, dt, chip_slow_dcn, scen_big_chunk, mode="disagg", max_chips=4
    )
    assert disagg_fast.recommended is not None and disagg_slow.recommended is not None
    assert (
        disagg_slow.recommended.p90_ttft_ms > disagg_fast.recommended.p90_ttft_ms + 5.0
    )


def test_prefix_hit_crossover_sweep_under_50ms_and_pcie_offload() -> None:
    chip = rf.CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = rf.Dtypes(kv=1.0, state=2.0)
    scen_no_offload = rf.TrafficScenario(
        name="offload-compare",
        isl=32768,
        osl=512,
        prefix_hit_rate=0.90,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(32,),
        kv_working_set_turns=20.0,
        kv_offload_gb=0.0,
    )
    scen_offload = dc_replace(scen_no_offload, kv_offload_gb=64.0)

    def _cold_sweep(
        plan_obj: rf.ServingPlan, repeats: int = 3
    ) -> tuple[tuple[float, float] | None, float | None, float]:
        best_ms = math.inf
        last_res: tuple[tuple[float, float] | None, float | None] = (None, None)
        for _ in range(repeats):
            rf.solver._EP_BUSIEST_CACHE.clear()
            rf.solver._ep_z_unit.cache_clear()
            rf.solver._coll_spec.cache_clear()
            rf.solver._make_draft_geom.cache_clear()
            rf.solver._classify_layers_cached.cache_clear()
            rf.hbm_breakdown_per_core_bytes.cache_clear()
            t0 = time.perf_counter()
            last_res = rf.sweep_prefix_hit_crossovers(geom, dt, chip, plan_obj)
            best_ms = min(best_ms, (time.perf_counter() - t0) * 1e3)
        return last_res[0], last_res[1], best_ms

    plan_off = rf.solve_serving_plan(
        geom, dt, chip, scen_offload, mode="auto", max_chips=8
    )
    assert len(plan_off.candidates) > len(plan_off.frontier)
    win_range, cross_rate, sweep_ms = _cold_sweep(plan_off)
    assert sweep_ms < 50.0
    assert win_range is not None
    assert 0.0 <= win_range[0] <= 0.90 <= win_range[1] <= 0.95
    assert plan_off.winner_prefix_hit_range == win_range
    assert plan_off.prefix_hit_crossover_rate == cross_rate
    solve_payload = rf.build_solve_payload(plan_off, chip=chip, dtypes=dt)
    assert "sweep_candidates" not in solve_payload

    # Verify full-candidate sweep agrees with a fresh solve_serving_plan at h=0.0.
    plan_h0 = rf.solve_serving_plan(
        geom,
        dt,
        chip,
        dc_replace(scen_offload, prefix_hit_rate=0.0),
        mode="auto",
        max_chips=8,
    )
    assert plan_h0.recommended is not None and plan_off.recommended is not None
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

    plan_no = rf.solve_serving_plan(
        geom, dt, chip, scen_no_offload, mode="mixed", max_chips=4
    )
    plan_yes = rf.solve_serving_plan(
        geom, dt, chip, scen_offload, mode="mixed", max_chips=4
    )
    pt_no = next(p for p in plan_no.candidates if p.decode_pool.tp == 8)
    pt_yes = next(p for p in plan_yes.candidates if p.decode_pool.tp == 8)
    # 64 GiB summed across TP=8 adds exactly 8 GiB per rank / b_kv_phys_core tokens.
    b_kv_core = rf.hbm_footprint_per_core_bytes(
        geom, dt, rf.Parallel(tp=8, moe="ep"), 1, 1
    ) - rf.hbm_footprint_per_core_bytes(geom, dt, rf.Parallel(tp=8, moe="ep"), 1, 0)
    expected_delta_tok = int((64.0 * float(1 << 30) / 8.0) / b_kv_core)
    assert pt_yes.kv_cache_capacity_tokens is not None
    assert pt_no.kv_cache_capacity_tokens is not None
    assert (
        abs(
            (pt_yes.kv_cache_capacity_tokens - pt_no.kv_cache_capacity_tokens)
            - expected_delta_tok
        )
        <= 2
    )
    # Slower PCIe increases TTFT on both mixed and disagg pools when host
    # offload hits read over PCIe.
    slow_pcie_plan = rf.solve_serving_plan(
        geom,
        dt,
        dc_replace(chip, pcie_bw=4.0e9),
        scen_offload,
        mode="mixed",
        max_chips=4,
    )
    pt_slow_pcie = next(p for p in slow_pcie_plan.candidates if p.decode_pool.tp == 8)
    assert pt_slow_pcie.p90_ttft_ms > pt_yes.p90_ttft_ms

    disagg_fast_pcie = rf.solve_serving_plan(
        geom, dt, chip, scen_offload, mode="disagg", max_chips=8
    )
    disagg_slow_pcie = rf.solve_serving_plan(
        geom,
        dt,
        dc_replace(chip, pcie_bw=4.0e9),
        scen_offload,
        mode="disagg",
        max_chips=8,
    )
    assert (
        disagg_fast_pcie.recommended is not None
        and disagg_slow_pcie.recommended is not None
    )
    assert (
        disagg_slow_pcie.recommended.p90_ttft_ms
        > disagg_fast_pcie.recommended.p90_ttft_ms
    )

    emitted = rf.emit_launch_args(plan_yes, dtypes=dt, model="org/synth-moe")
    assert "sweep_candidates" not in emitted["serving_plan"]
    assert "--kv-offloading-size" in emitted["cli_args"]
    idx = emitted["cli_args"].index("--kv-offloading-size")
    assert emitted["cli_args"][idx + 1] == "64"

    # R1: TTFT-SLO filtering at h0=0.0 must still evaluate full HBM-fit candidates
    # across higher hit rates where prefill shrinks under the TTFT SLO.
    scen_probe = rf.TrafficScenario(
        name="slo-crossover-probe",
        isl=8192,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(8, 32),
        kv_working_set_turns=1.0,
    )
    probe_h0 = rf.solve_serving_plan(
        geom, dt, chip, scen_probe, mode="mixed", max_chips=4
    )
    probe_h95 = rf.solve_serving_plan(
        geom,
        dt,
        chip,
        dc_replace(scen_probe, prefix_hit_rate=0.95),
        mode="mixed",
        max_chips=4,
    )
    assert probe_h0.recommended is not None and probe_h95.recommended is not None
    mid_ttft_slo = 0.5 * (
        probe_h0.recommended.p90_ttft_ms + probe_h95.recommended.p90_ttft_ms
    )
    scen_slo = dc_replace(scen_probe, ttft_slo_ms=mid_ttft_slo)
    plan_slo_h0 = rf.solve_serving_plan(
        geom, dt, chip, scen_slo, mode="mixed", max_chips=4
    )
    plan_slo_h95 = rf.solve_serving_plan(
        geom,
        dt,
        chip,
        dc_replace(scen_slo, prefix_hit_rate=0.95),
        mode="mixed",
        max_chips=4,
    )
    assert plan_slo_h0.recommended is not None and plan_slo_h95.recommended is not None
    if (
        plan_slo_h0.recommended.decode_pool.tp,
        plan_slo_h0.recommended.concurrency,
    ) != (
        plan_slo_h95.recommended.decode_pool.tp,
        plan_slo_h95.recommended.concurrency,
    ):
        assert plan_slo_h0.prefix_hit_crossover_rate is not None
        assert plan_slo_h0.winner_prefix_hit_range is not None
        assert plan_slo_h0.winner_prefix_hit_range[1] < 0.95

    # R2: Built-in multi-concurrency scenarios stay under 50 ms cold at 8 and 16 chips.
    for ex_name, n_chips in (
        ("coding-agent", 8),
        ("chat-8k1k", 16),
        ("reasoning-1k8k", 16),
    ):
        ex_scen = rf.build_example_scenario(ex_name)
        ex_plan = rf.solve_serving_plan(
            geom, dt, chip, ex_scen, mode="auto", max_chips=n_chips
        )
        ex_range, _ex_cross, ex_ms = _cold_sweep(ex_plan)
        assert ex_ms < 50.0
        assert ex_range is not None
        if ex_name == "chat-8k1k":
            ex_uncached = dc_replace(ex_plan, sweep_candidates=())
            fb_range, fb_cross, _ = _cold_sweep(ex_uncached, repeats=1)
            assert (fb_range, fb_cross) == (
                ex_plan.winner_prefix_hit_range,
                ex_plan.prefix_hit_crossover_rate,
            )
