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
"""Unit tests for physics-only mesh/prefill-decode solver and CLI solve subcommand."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import replace as dc_replace
from pathlib import Path

import pytest

import vllm_torchtpu.roofline as rf
from tests.roofline.conftest import (
    LOCKED_FIXTURE_FILES,
    load_locked_bundle,
    load_locked_config,
)
from vllm_torchtpu.roofline import (
    FloorOptions,
    KvFitInput,
    ScenarioOverrides,
    SeqKvLayout,
    SolveOptions,
    SpecShape,
    cli,
    hbm_footprint_per_core_bytes,
)
from vllm_torchtpu.roofline.constants import UNIT_GIGA
from vllm_torchtpu.roofline.geometry import _from_hf_config_dry_run
from vllm_torchtpu.roofline.memory import (
    compute_w_attn,
    compute_w_ffn,
    kv_capacity_and_hit_rate,
)
from vllm_torchtpu.roofline.solver import (
    EXAMPLE_SCENARIOS,
    _candidate_decode_parallels,
    _candidate_prefill_parallels,
    _decode_steps_per_req,
    _find_prefill_options,
    _SolveContext,
    _valid_gdn_degree,
    _valid_head_tp,
)


def _check_dp_moe_decode(chip: rf.ChipSpec, q35_geom: rf.ModelGeometry) -> None:
    """Verify DP > 1 MoE decode enumerates EP push and subgroup options."""
    dp_cands = [
        p
        for p in _candidate_decode_parallels(q35_geom, rf.TOPOLOGIES["2x2x1"])
        if p.dp > 1 and p.pp == 1 and p.pcp == 1
    ]
    assert dp_cands
    ep_degree = 8
    ep8_cands = [p for p in dp_cands if p.moe == "ep" and p.ep == ep_degree]
    assert ep8_cands
    opts = FloorOptions(dtypes=rf.Dtypes(kv=1.0), topology=rf.TOPOLOGIES["2x2x1"])
    for par_dp in ep8_cands:
        assert par_dp.moe_dispatch == "a2a_push"
        sf_dp = rf.step_floor(
            q35_geom,
            chip,
            par_dp,
            rf.StepShape(batch=4, context=1024),
            opts,
        )
        assert sf_dp.hard_floor_s > 0.0


def _check_mixed_prefill_dp2(chip: rf.ChipSpec) -> None:
    """Verify mixed prefill work scales as 1/dp per replica."""
    dp2_geom = rf.from_hf_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
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
    dp8_pt = next(
        p
        for p in dp2_plan.candidates
        if p.decode_pool.dp == expected_dp_2x2x1
        and p.decode_pool.pp == 1
        and p.decode_pool.pcp == 1
    )
    # 1 seq/replica * 2 prefill chunks amortized over osl - 1 = 511 decode steps.
    exp_p_mix = min(1.0, float(1 * 2) / _decode_steps_per_req(dp2_scen.osl, 1.0))
    assert dp8_pt.p_mixed == pytest.approx(exp_p_mix, rel=1e-5)


def _check_cli_solve(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI solve produces structured JSON with ordered candidates and pools."""
    cfg_path = tmp_path / "q35_config.json"
    cfg_path.write_text(json.dumps(load_locked_config("qwen3_5_397b_fp8.json")))
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
            "--sort-by",
            "interactivity",
            "--top-k",
            "3",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommended"] is not None
    assert payload["prefill_pools"]
    assert payload["decode_pools"]
    assert payload["sort_by"] == "interactivity"


def _check_pp_prefill_ttft(chip: rf.ChipSpec) -> None:
    """Verify PP > 1 prefill TTFT includes pipeline bubble stage fills."""
    pp_geom = rf.from_hf_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
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
        pp_pools = [r for r in chk_disagg.prefill_pools if r.pool.pp > 1]
        assert pp_pools
        rec0 = pp_pools[0]
        sf_st = rf.step_floor(
            pp_geom,
            chip,
            rf.Parallel(
                tp=rec0.pool.tp,
                dp=rec0.pool.dp,
                pp=rec0.pool.pp,
                pcp=rec0.pool.pcp,
                kv_split=rec0.pool.kv_split,
                moe="ep" if rec0.pool.ep > 1 else "tp_slice",
                ep=rec0.pool.ep,
                mesh_axes=rec0.pool.mesh_axes,
            ),
            rf.StepShape(
                batch=0, rows_per_seq=0, context=0, state_writes=0, prefill_tokens=1024
            ),
            FloorOptions(
                dtypes=rf.Dtypes(kv=1.0),
                topology=rf.TOPOLOGIES[rec0.pool.topology],
            ),
        )
        bubble_mult = 1.0 + float(rec0.pool.pp - 1) / float(n_chk)
        assert rec0.ttft_ms == pytest.approx(
            sf_st.serial_estimate_s * bubble_mult * 1e3
        )


def test_solver_hybrid_moe_mixed_vs_disagg_and_crossover(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test mixed vs disaggregated solver behavior and MTP cycle pricing."""
    q35_geom = rf.from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    chip = rf.CHIPS["tpu7x"]
    scenario = EXAMPLE_SCENARIOS["chat-8k1k"]
    mixed_plan = rf.solve_serving_plan(
        q35_geom,
        rf.Dtypes(kv=1.0),
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=8),
    )
    disagg_plan = rf.solve_serving_plan(
        q35_geom,
        rf.Dtypes(kv=1.0),
        chip,
        dc_replace(scenario, concurrencies=(1, 16)),
        SolveOptions(mode="disagg", max_chips=8, include_pairs=True),
    )
    min_conc = 16
    min_mixed_fraction = 0.10
    mixed_c16 = [p for p in mixed_plan.frontier if p.concurrency >= min_conc]
    assert mixed_c16
    assert any(p.p_mixed >= min_mixed_fraction for p in mixed_c16)
    assert disagg_plan.recommended is not None
    assert disagg_plan.recommended.p_mixed == 0.0
    assert disagg_plan.prefill_pools
    assert disagg_plan.decode_pools

    # coding-agent (accept_len=3.0) with 1 MTP layer prices k=3 window.
    ca_scen = rf.build_example_scenario(
        "coding-agent", overrides=ScenarioOverrides(concurrencies=(1,))
    )
    ca_plan = rf.solve_serving_plan(
        q35_geom,
        rf.Dtypes(kv=1.0),
        chip,
        ca_scen,
        SolveOptions(mode="disagg", max_chips=8, include_pairs=True),
    )
    assert ca_plan.recommended is not None
    rec_d = ca_plan.recommended.decode_pool
    h_eff = ca_plan.recommended.effective_prefix_hit_rate or 0.0
    sf_k3 = rf.spec_cycle_floor(
        q35_geom,
        chip,
        rf.Parallel(
            tp=rec_d.tp,
            dp=rec_d.dp,
            pp=rec_d.pp,
            pcp=rec_d.pcp,
            moe="ep" if rec_d.ep > 1 else "tp_slice",
            ep=rec_d.ep,
            moe_dispatch=rec_d.moe_dispatch,
            kv_split=rec_d.kv_split,
            mesh_axes=rec_d.mesh_axes,
        ),
        SpecShape(
            batch=max(1, (1 + rec_d.dp - 1) // rec_d.dp),
            context=ca_scen.isl + ca_scen.osl // 2,
            spec_k=3,
            shared_prefix_tokens=round(h_eff * ca_scen.isl),
        ),
        FloorOptions(
            dtypes=rf.Dtypes(kv=1.0),
            topology=rf.TOPOLOGIES[rec_d.topology],
        ),
    )
    assert ca_plan.recommended.est_tpot_ms == pytest.approx(
        sf_k3.serial_estimate_s * 1e3 / ca_scen.accept_len, rel=1e-4
    )

    _check_dp_moe_decode(chip, q35_geom)
    _check_mixed_prefill_dp2(chip)
    _check_pp_prefill_ttft(chip)

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


def test_logical_kv_capacity_and_unpruned_physics_regimes() -> None:
    """Verify logical KV sizing, MTP weights, and unpruned PP/DCP/DP regimes."""
    chip = rf.CHIPS["tpu7x"]
    q35_geom = rf.from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
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
        SeqKvLayout(1, 8192),
    )
    assert (hbm_phys - hbm_log) == pytest.approx(15.0 * 8192.0 * 512.0, abs=2.0)

    target_concurrency = 64
    min_pp_stages = 8
    scenario = rf.TrafficScenario(
        name="custom",
        isl=1024,
        osl=1024,
        concurrencies=(target_concurrency,),
        prefix_hit_rate=0.5,
        accept_len=3.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
    )
    disagg = rf.solve_serving_plan(
        q35_geom,
        q35_dt,
        chip,
        scenario,
        SolveOptions(mode="disagg", max_chips=12),
    )
    # Physics allows PP >= 8 in prefill regardless of spec_k, plus DP in (4, 8) decode.
    assert disagg.candidates == ()
    assert any(r.pool.pp >= min_pp_stages for r in disagg.prefill_pools)
    assert any(
        r.pool.moe_dispatch == "a2a_push" and r.pool.dp in (4, 8)
        for r in disagg.decode_pools
    )

    dsa_78l_cfg = load_locked_config("glm_5_2_fp8.json")
    dsa_78l_geom = rf.from_hf_config(dsa_78l_cfg)
    dsa_78l_dt = rf.Dtypes(
        w_dense=1.0, w_expert=1.0, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0
    )
    tp_16 = 16
    kv_split_16 = 16
    expected_c48 = 48
    dsa_c48 = rf.solve_serving_plan(
        dsa_78l_geom,
        dsa_78l_dt,
        chip,
        rf.TrafficScenario(
            name="dsa-single-turn",
            isl=137168,
            osl=894,
            concurrencies=(expected_c48,),
            prefix_hit_rate=0.95,
            accept_len=3.0,
            think_time_s=0.0,
            kv_working_set_turns=1.0,
            spec_tokens=3,
        ),
        SolveOptions(mode="mixed", max_chips=8, hbm_reserve_gb=15.0),
    )
    tp16_pts = [
        p
        for p in dsa_c48.candidates
        if p.decode_pool.tp == tp_16
        and p.decode_pool.ep == 1
        and p.decode_pool.pp == 1
        and p.decode_pool.pcp == 1
        and p.decode_pool.resident_batch_cap == expected_c48
    ]
    assert tp16_pts
    best_tp16 = min(tp16_pts, key=lambda p: p.est_tpot_ms)
    assert best_tp16.decode_pool.kv_split == kv_split_16


def test_hbm_reserve_and_explicit_scenario_inputs(tmp_path: Path) -> None:
    """Test HBM reserve clipping and CLI explicit scenario flags."""
    chip = rf.CHIPS["tpu7x"]
    geometry = rf.from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    dtypes = rf.Dtypes(kv=1.0, w_aux=2.0)

    with pytest.raises(TypeError):
        partial_kwargs: dict[str, object] = {"name": "x", "isl": 1024, "osl": 128}
        rf.TrafficScenario(**partial_kwargs)  # type: ignore[arg-type]

    reserve_tight = 95.0
    plan_tight = rf.solve_serving_plan(
        geometry,
        dtypes,
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
        SolveOptions(mode="mixed", max_chips=4, hbm_reserve_gb=reserve_tight),
    )
    assert plan_tight.hbm_reserve_gb == reserve_tight
    assert len(plan_tight.candidates) == 0

    plan_full = rf.solve_serving_plan(
        geometry,
        dtypes,
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
        SolveOptions(mode="mixed", max_chips=4, hbm_reserve_gb=0.0),
    )
    assert plan_full.hbm_reserve_gb == 0.0
    assert len(plan_full.candidates) > 0

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
    scen_think = dc_replace(scen_c0, name="open", think_time_s=15.0)
    p_c0 = rf.solve_serving_plan(
        geometry, dtypes, chip, scen_c0, SolveOptions(mode="mixed", max_chips=4)
    )
    p_think = rf.solve_serving_plan(
        geometry, dtypes, chip, scen_think, SolveOptions(mode="mixed", max_chips=4)
    )
    assert p_c0.recommended is not None
    assert p_think.recommended is not None
    assert p_think.recommended.est_tpot_ms <= p_c0.recommended.est_tpot_ms

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(load_locked_config("qwen3_5_397b_fp8.json")))
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["solve", "--config", str(cfg_path)])
    assert excinfo.value.code != 0

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
            "--hbm-reserve-gb",
            "0",
            "--json",
        ]
    )
    assert rc_exp == 0


def test_physical_legality_heads_pp_tp_pcp_ep_and_gdn() -> None:
    """Verify physical head divisibility and unlocked 2D/subgroup parallel axes."""
    assert _valid_head_tp(
        rf.from_hf_config(load_locked_config("qwen3_5_397b_fp8.json")), 1
    )
    uneven_geom = rf.from_hf_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
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
    # Physics allows PP > 1 with TP > 1 on decode; PCP > 1 and DP > 1 on prefill.
    assert all(p.pcp == 1 for p in dec_pars)
    assert any(p.pp > 1 and p.tp > 1 for p in dec_pars)
    pre_pars = _candidate_prefill_parallels(uneven_geom, rf.TOPOLOGIES["2x2x1"])
    assert any(p.pp > 1 and p.pcp > 1 for p in pre_pars)
    assert any(p.dp > 1 for p in pre_pars)

    hybrid_8l = rf.from_hf_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
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
    assert not _valid_gdn_degree(dc_replace(hybrid_8l, linear_v_heads=64), 6)
    hybrid_pre = _candidate_prefill_parallels(hybrid_8l, rf.TOPOLOGIES["2x2x1"])
    assert any(p.pp == 8 // 1 for p in hybrid_pre)


def test_disagg_capacity_clamped_hit_rate_and_prefill_dispatch() -> None:
    """Test disagg capacity clamping, prefix hit degradation, and EP dispatch."""
    chip = rf.CHIPS["tpu7x"]
    chat_scen = EXAMPLE_SCENARIOS["chat-8k1k"]
    cap_init, h_init = kv_capacity_and_hit_rate(
        rf.from_hf_config(load_locked_config("qwen3_5_397b_fp8.json")),
        rf.Dtypes(kv=1.0),
        rf.Parallel(tp=1),
        1,
        KvFitInput(
            working_set_batch=1,
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
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForCausalLM"],
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
    geometry = rf.from_hf_config(moe_cfg)
    dtypes = rf.Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
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
        geometry,
        dtypes,
        chip,
        scen_small,
        SolveOptions(mode="disagg", max_chips=8, include_pairs=True),
    )
    plan_huge = rf.solve_serving_plan(
        geometry,
        dtypes,
        chip,
        scen_huge,
        SolveOptions(mode="disagg", max_chips=8, include_pairs=True),
    )
    assert plan_small.recommended is not None
    assert plan_huge.recommended is not None
    by_pair_small = {
        (
            p.prefill_pool.topology,
            p.prefill_pool.tp,
            p.prefill_pool.dp,
            p.prefill_pool.pcp,
            p.prefill_pool.pp,
            p.prefill_pool.ep,
            p.prefill_pool.kv_split,
            p.decode_pool.topology,
            p.decode_pool.tp,
            p.decode_pool.dp,
            p.decode_pool.pcp,
            p.decode_pool.pp,
            p.decode_pool.ep,
            p.decode_pool.kv_split,
        ): p
        for p in plan_small.candidates
        if p.prefill_pool is not None
    }
    matched_pp1 = 0
    for pt_h in plan_huge.candidates:
        if pt_h.prefill_pool is None:
            continue
        k = (
            pt_h.prefill_pool.topology,
            pt_h.prefill_pool.tp,
            pt_h.prefill_pool.dp,
            pt_h.prefill_pool.pcp,
            pt_h.prefill_pool.pp,
            pt_h.prefill_pool.ep,
            pt_h.prefill_pool.kv_split,
            pt_h.decode_pool.topology,
            pt_h.decode_pool.tp,
            pt_h.decode_pool.dp,
            pt_h.decode_pool.pcp,
            pt_h.decode_pool.pp,
            pt_h.decode_pool.ep,
            pt_h.decode_pool.kv_split,
        )
        pt_s = by_pair_small.get(k)
        if pt_s is not None:
            assert pt_s.kv_cache_capacity_tokens is not None
            assert pt_h.effective_prefix_hit_rate is not None
            assert pt_s.effective_prefix_hit_rate is not None
            assert pt_h.effective_prefix_hit_rate < pt_s.effective_prefix_hit_rate
            if pt_h.prefill_pool.pp == 1:
                matched_pp1 += 1
                assert pt_h.est_ttft_ms > pt_s.est_ttft_ms
    assert matched_pp1 >= 1

    scen_high_ws = dc_replace(
        scen_small, concurrencies=(64,), kv_working_set_turns=200.0
    )
    scen_low_ws = dc_replace(scen_small, concurrencies=(1,), kv_working_set_turns=200.0)
    expected_tp_8 = 8
    pt_c64 = next(
        p
        for p in rf.solve_serving_plan(
            geometry,
            dtypes,
            chip,
            scen_high_ws,
            SolveOptions(mode="mixed", max_chips=4),
        ).candidates
        if p.decode_pool.tp == expected_tp_8 and p.decode_pool.dp == 1
    )
    pt_c1 = next(
        p
        for p in rf.solve_serving_plan(
            geometry, dtypes, chip, scen_low_ws, SolveOptions(mode="mixed", max_chips=4)
        ).candidates
        if p.decode_pool.tp == expected_tp_8 and p.decode_pool.dp == 1
    )
    assert pt_c64.effective_prefix_hit_rate is not None
    assert pt_c1.effective_prefix_hit_rate is not None
    assert pt_c64.effective_prefix_hit_rate < pt_c1.effective_prefix_hit_rate


def test_solver_excludes_gdn_dcp_on_kvless_model() -> None:
    """Reject kv_split > 1 on KV-less pure recurrent models."""
    pure_gdn = rf.from_hf_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
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
    geometry = rf.from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    scenario = rf.build_example_scenario(
        overrides=ScenarioOverrides(concurrencies=(1,))
    )
    for bad_chips in (1, 2):
        with pytest.raises(ValueError, match="2x2x1"):
            rf.solve_serving_plan(
                geometry,
                rf.Dtypes(),
                rf.CHIPS["tpu7x"],
                scenario,
                SolveOptions(max_chips=bad_chips),
            )


def test_swa_kv_capacity_not_linearized_beyond_window() -> None:
    """Verify SWA KV stays capped at min(ctx, sliding_window) in capacity fit."""
    sw_window = 512
    ctx_long = 131_072
    isl_long = 130_048
    osl_long = 1024
    tp_8 = 8
    conc_one = 1
    hit_rate_target = 0.8
    no_offload = 0.0
    single_turn = 1.0
    hybrid_swa_geom = _from_hf_config_dry_run(
        {
            "model_type": "synthetic_swa",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "layer_types": ["sliding_attention"] * 15 + ["full_attention"],
            "sliding_window": sw_window,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 4096,
            "vocab_size": 32000,
        }
    )
    pure_swa_geom = dc_replace(
        hybrid_swa_geom,
        layer_types=("sliding_attention",) * 16,
    )
    dtypes = rf.Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    chip = rf.CHIPS["tpu7x"]
    parallel = rf.Parallel(tp=tp_8, ranks_per_chip=chip.cores_per_chip)
    w_base_b = hbm_footprint_per_core_bytes(
        hybrid_swa_geom, dtypes, parallel, SeqKvLayout(0, 0)
    )
    kv_at_sw = (
        hbm_footprint_per_core_bytes(
            hybrid_swa_geom, dtypes, parallel, SeqKvLayout(conc_one, sw_window)
        )
        - w_base_b
    )
    kv_past_sw = (
        hbm_footprint_per_core_bytes(
            hybrid_swa_geom,
            dtypes,
            parallel,
            SeqKvLayout(conc_one, sw_window + conc_one),
        )
        - w_base_b
    )
    full_kv_per_tok = kv_past_sw - kv_at_sw
    swa_fixed_per_seq = kv_at_sw - float(sw_window) * full_kv_per_tok
    hbm_limit_b = float(chip.hbm_bytes)
    scenario = rf.TrafficScenario(
        name="swa-long",
        isl=isl_long,
        osl=osl_long,
        concurrencies=(conc_one,),
        prefix_hit_rate=hit_rate_target,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=single_turn,
    )
    plan = rf.solve_serving_plan(
        hybrid_swa_geom,
        dtypes,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=chip.cores_per_chip * 2),
    )
    pt_tp8 = next(
        p
        for p in plan.candidates
        if p.decode_pool.tp == tp_8
        and p.decode_pool.dp == conc_one
        and p.decode_pool.pp == conc_one
        and p.decode_pool.pcp == conc_one
        and p.decode_pool.kv_split == conc_one
    )
    expected_cap_tok = int(
        (hbm_limit_b - w_base_b - swa_fixed_per_seq) / full_kv_per_tok
    )
    assert pt_tp8.kv_cache_capacity_tokens == expected_cap_tok

    pure_plan = rf.solve_serving_plan(
        pure_swa_geom,
        dtypes,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=chip.cores_per_chip * 2),
    )
    pure_pt = next(
        p
        for p in pure_plan.candidates
        if p.decode_pool.tp == tp_8
        and p.decode_pool.dp == conc_one
        and p.decode_pool.pp == conc_one
        and p.decode_pool.pcp == conc_one
        and p.decode_pool.kv_split == conc_one
    )
    assert pure_pt.kv_cache_capacity_tokens is None
    assert pure_pt.effective_prefix_hit_rate == pytest.approx(hit_rate_target)

    cap_short, h_short = kv_capacity_and_hit_rate(
        hybrid_swa_geom,
        dtypes,
        parallel,
        conc_one,
        KvFitInput(
            working_set_batch=conc_one,
            ctx_full=sw_window,
            hbm_limit_b=hbm_limit_b,
            w_base_b=w_base_b,
            b_kv_phys_core=full_kv_per_tok,
            offload_b_per_core=no_offload,
            prefix_hit_rate=hit_rate_target,
            kv_working_set_turns=single_turn,
        ),
        fixed_kv_per_seq_bytes=swa_fixed_per_seq,
    )
    cap_long, h_long = kv_capacity_and_hit_rate(
        hybrid_swa_geom,
        dtypes,
        parallel,
        conc_one,
        KvFitInput(
            working_set_batch=conc_one,
            ctx_full=ctx_long,
            hbm_limit_b=hbm_limit_b,
            w_base_b=w_base_b,
            b_kv_phys_core=full_kv_per_tok,
            offload_b_per_core=no_offload,
            prefix_hit_rate=hit_rate_target,
            kv_working_set_turns=single_turn,
        ),
        fixed_kv_per_seq_bytes=swa_fixed_per_seq,
    )
    assert cap_long == pytest.approx(cap_short)
    assert h_long == pytest.approx(h_short)


def test_prefill_options_keep_all_legal_variants() -> None:
    """Verify _find_prefill_options keeps all legal single-stage variants."""
    geometry = rf.from_hf_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 4096,
            "vocab_size": 32000,
        }
    )
    dtypes = rf.Dtypes(w_dense=1.0, kv=1.0)
    chip = rf.CHIPS["tpu7x"]
    topology = rf.TOPOLOGIES["2x2x1"]
    scenario = rf.build_example_scenario("chat-8k1k")
    hbm_limit_b = float(chip.hbm_bytes)
    ctx_full = scenario.isl + scenario.osl
    eval_context = _SolveContext(
        geometry=geometry,
        dtypes=dtypes,
        chip=chip,
        scenario=scenario,
        mode="disagg",
        max_chips=topology.num_chips,
        max_instances=None,
        hbm_reserve_gb=0.0,
        hbm_limit_b=hbm_limit_b,
        hbm_limit_gb=hbm_limit_b / UNIT_GIGA,
        effective_accept_len=1.0,
        spec_k=0,
        ctx_full=ctx_full,
        ctx_mid=scenario.isl + scenario.osl // 2,
        eff_isl=scenario.isl,
        n_chunks=max(1, scenario.isl // scenario.prefill_chunk),
        c_max=max(scenario.concurrencies),
    )
    options, _ = _find_prefill_options(eval_context, [topology])
    span_target = topology.num_cores
    single_stage_variants = [
        opt
        for opt in options
        if opt.parallel.tp * opt.parallel.pcp == span_target
        and opt.parallel.pp == 1
        and opt.parallel.dp == 1
        and opt.parallel.kv_split == 1
    ]
    expected_pcp_tp_pairs = {(8, 1), (4, 2), (2, 4), (1, 8)}
    assert {(opt.parallel.tp, opt.parallel.pcp) for opt in single_stage_variants} == (
        expected_pcp_tp_pairs
    )
    assert len(single_stage_variants) == len(expected_pcp_tp_pairs)


def test_prefill_enumerates_physical_ep_and_tp_slice_moe() -> None:
    """Verify MoE legality depends on physical divisibility, not block-quant padding."""
    base_moe = rf.from_hf_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_experts": 16,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1024,
            "shared_expert_intermediate_size": 512,
            "vocab_size": 32000,
        }
    )
    topology = rf.TOPOLOGIES["2x2x1"]
    tp_8 = 8
    tp_4 = 4
    pcp_2 = 2
    block_128 = 128
    # Divisible by tp=8 (1088 = 8 * 136) even when not divisible by 128.
    moe_div_tp_not_128 = tp_8 * block_128 + 64
    geom_partial_blk = dc_replace(
        base_moe, moe_inter=moe_div_tp_not_128, weight_block_size=(block_128, block_128)
    )
    pars_partial = _candidate_prefill_parallels(geom_partial_blk, topology)
    modes_tp8 = {
        (p.moe, p.ep)
        for p in pars_partial
        if p.pp == 1 and p.pcp == 1 and p.dp == 1 and p.tp == tp_8
    }
    assert ("tp_slice", 1) in modes_tp8
    assert ("ep", tp_8) in modes_tp8
    assert ("ep", tp_4) in modes_tp8

    # Physically indivisible by tp=8 (1028 % 8 == 4) excludes tp_slice at tp=8.
    geom_indiv = dc_replace(base_moe, moe_inter=1028)
    pars_indiv = _candidate_prefill_parallels(geom_indiv, topology)
    modes_indiv = {
        (p.moe, p.ep)
        for p in pars_indiv
        if p.pp == 1 and p.pcp == 1 and p.dp == 1 and p.tp == tp_8
    }
    assert ("tp_slice", 1) not in modes_indiv
    assert ("ep", tp_8) in modes_indiv

    # PCP > 1 allows both EP and TP-slice physically when divisible.
    pars_pcp = [
        (p.moe, p.ep)
        for p in _candidate_prefill_parallels(base_moe, topology)
        if p.pp == 1 and p.pcp == pcp_2 and p.dp == 1 and p.tp == tp_4
    ]
    assert ("ep", tp_4 * pcp_2) in pars_pcp
    assert ("tp_slice", 1) in pars_pcp


def test_solver_think_time_preserves_working_set_and_resident_cap() -> None:
    """Verify KvFitInput uses c/dp working set and records resident_batch_cap."""
    chip = rf.CHIPS["tpu7x"]
    geometry = rf.from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    dtypes = rf.Dtypes(
        w_dense=1.0, w_expert=1.0, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0
    )
    base_scen = rf.TrafficScenario(
        name="think-ws",
        isl=32768,
        osl=512,
        prefix_hit_rate=0.9,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(512,),
        kv_working_set_turns=1.0,
    )
    think_scen = dc_replace(base_scen, think_time_s=60.0)
    plan_tt0 = rf.solve_serving_plan(
        geometry, dtypes, chip, base_scen, SolveOptions(mode="mixed", max_chips=4)
    )
    plan_tt60 = rf.solve_serving_plan(
        geometry, dtypes, chip, think_scen, SolveOptions(mode="mixed", max_chips=4)
    )
    expected_tp = 8
    pt_tt0 = next(
        p
        for p in plan_tt0.candidates
        if p.decode_pool.tp == expected_tp
        and p.decode_pool.dp == 1
        and p.decode_pool.pp == 1
        and p.decode_pool.pcp == 1
        and p.decode_pool.ep == expected_tp
        and p.decode_pool.kv_split == 1
    )
    pt_tt60 = next(
        p
        for p in plan_tt60.candidates
        if p.decode_pool.tp == expected_tp
        and p.decode_pool.dp == 1
        and p.decode_pool.pp == 1
        and p.decode_pool.pcp == 1
        and p.decode_pool.ep == expected_tp
        and p.decode_pool.kv_split == 1
    )
    assert pt_tt0.decode_pool.resident_batch_cap is not None
    assert pt_tt60.decode_pool.resident_batch_cap is not None
    assert (
        pt_tt60.decode_pool.resident_batch_cap < pt_tt0.decode_pool.resident_batch_cap
    )
    assert pt_tt60.effective_prefix_hit_rate is not None
    assert pt_tt0.effective_prefix_hit_rate is not None
    assert pt_tt0.effective_prefix_hit_rate < base_scen.prefix_hit_rate
    assert pt_tt60.effective_prefix_hit_rate < base_scen.prefix_hit_rate
    # Higher think time lowers resident batch (freeing minor state/act HBM) while
    # preserving the c/dp=512 working-set denominator so the hit rate stays clamped.
    assert pt_tt60.effective_prefix_hit_rate == pytest.approx(
        pt_tt0.effective_prefix_hit_rate, rel=2e-2
    )


def test_disagg_think_time_preserves_decode_working_set_hit_rate_clamp() -> None:
    """Verify disagg decode KvFitInput clamps hit rate by c/(y_d*dp) on GLM-5.2."""
    bundle = load_locked_bundle("glm_5_2_fp8.json")
    chip = rf.CHIPS["tpu7x"]
    geometry = bundle.geometry
    dtypes = bundle.dtypes
    base_scen = rf.TrafficScenario(
        name="disagg-think-ws",
        isl=32768,
        osl=256,
        prefix_hit_rate=0.9,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(256,),
        kv_working_set_turns=1.0,
    )
    think_scen = dc_replace(base_scen, think_time_s=60.0)
    opts = SolveOptions(
        mode="disagg",
        max_chips=24,
        topologies=(rf.TOPOLOGIES["2x2x2"],),
        include_pairs=True,
    )
    plan_tt0 = rf.solve_serving_plan(geometry, dtypes, chip, base_scen, opts)
    plan_tt60 = rf.solve_serving_plan(geometry, dtypes, chip, think_scen, opts)
    clamped_ceiling = 0.85

    def _dec_key(record: rf.PoolFloorRecord) -> tuple[object, ...]:
        """Extract structural decode pool identity for tt=0 vs tt=60 matching."""
        d = record.pool
        return (d.topology, d.tp, d.dp, d.ep, d.pp, d.pcp, d.kv_split)

    tt60_by_key = {
        _dec_key(r): r.effective_prefix_hit_rate
        for r in plan_tt60.decode_pools
        if r.effective_prefix_hit_rate is not None
    }
    matched = [
        (r.effective_prefix_hit_rate, tt60_by_key[_dec_key(r)])
        for r in plan_tt0.decode_pools
        if _dec_key(r) in tt60_by_key
        and r.effective_prefix_hit_rate is not None
        and r.effective_prefix_hit_rate < clamped_ceiling
    ]
    assert matched
    for h0, h60 in matched:
        assert h60 == pytest.approx(h0, abs=1e-6)

    def _pair_key(pt: rf.ParetoPoint) -> tuple[object, ...]:
        """Extract structural paired-candidate identity for tt=0 vs tt=60 matching."""
        p = pt.prefill_pool
        d = pt.decode_pool
        assert p is not None
        return (
            p.instances,
            p.tp,
            p.dp,
            p.ep,
            p.pp,
            p.pcp,
            d.instances,
            d.tp,
            d.dp,
            d.ep,
            d.pp,
            d.kv_split,
        )

    tt60_pairs = {_pair_key(pt): pt for pt in plan_tt60.candidates}
    paired_multi_dec = [
        (pt, tt60_pairs[_pair_key(pt)])
        for pt in plan_tt0.candidates
        if pt.decode_pool.instances > 1
        and pt.effective_prefix_hit_rate is not None
        and pt.effective_prefix_hit_rate < clamped_ceiling
        and _pair_key(pt) in tt60_pairs
    ]
    assert paired_multi_dec
    for p0, p60 in paired_multi_dec[:20]:
        assert p60.effective_prefix_hit_rate == pytest.approx(
            p0.effective_prefix_hit_rate, abs=1e-6
        )
        assert p0.kv_cache_capacity_tokens is not None
        assert p60.kv_cache_capacity_tokens is not None
        assert p60.kv_cache_capacity_tokens >= p0.kv_cache_capacity_tokens


def test_moe_tp_slice_hbm_is_unpadded_and_pp_uses_heaviest_stage() -> None:
    """Verify MoE TP-slice HBM uses unpadded shard width and PP uses ceil(L/pp)."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    geometry = bundle.geometry
    dtypes = bundle.dtypes
    w_tp8, _ = compute_w_ffn(geometry, dtypes, rf.Parallel(tp=8, moe="tp_slice"))
    w_tp16, _ = compute_w_ffn(geometry, dtypes, rf.Parallel(tp=16, moe="tp_slice"))
    # Router weights are replicated; routed + shared halve cleanly from TP8 to TP16.
    w_router = (
        float(geometry.num_layers * geometry.hidden * geometry.num_experts)
        * dtypes.router
    )
    assert (w_tp16 - w_router) == pytest.approx(0.5 * (w_tp8 - w_router), rel=1e-9)

    w_attn_pp1 = compute_w_attn(
        geometry, dtypes, rf.Parallel(tp=1, pp=1), rf.count_layer_types(geometry)
    )
    w_attn_pp8 = compute_w_attn(
        geometry, dtypes, rf.Parallel(tp=1, pp=8), rf.count_layer_types(geometry)
    )
    expected_pp8_scale = 8.0 / 60.0
    assert w_attn_pp8 == pytest.approx(w_attn_pp1 * expected_pp8_scale, rel=1e-9)


def test_solver_allows_all_physical_multihost_dp_and_pp_layouts() -> None:
    """Verify multi-host physical enumeration keeps DP, PPxTP, and prefill PCPxDCP."""
    geometry = load_locked_bundle("qwen3_5_397b_fp8.json").geometry
    topo_4host = rf.TOPOLOGIES["2x2x4"]
    dec_pars = _candidate_decode_parallels(geometry, topo_4host)
    pre_pars = _candidate_prefill_parallels(geometry, topo_4host)
    assert dec_pars
    assert pre_pars
    assert all(p.pcp == 1 for p in dec_pars)
    assert len(dec_pars) < len(pre_pars)
    hybrid_dp = 2
    assert any(p.dp == hybrid_dp for p in dec_pars)
    assert any(p.pp > 1 and p.tp > 1 for p in dec_pars)
    assert any(p.pcp > 1 and p.kv_split > 1 for p in pre_pars)
    assert any(p.dp > 1 for p in pre_pars)
    assert any(p.moe == "ep" and 1 < p.ep < p.tp for p in dec_pars)
    assert any(p.moe == "tp_slice" and p.dp > 1 for p in dec_pars)
    assert all(p.tp * p.dp * p.pp * p.pcp == topo_4host.num_cores for p in dec_pars)
    assert all(p.pp <= geometry.num_layers for p in dec_pars)


def test_solver_long_context_bounded_only_by_physical_hbm() -> None:
    """Verify long-context capacity depends only on physical HBM."""
    chip = rf.CHIPS["tpu7x"]
    geometry = rf.from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    dtypes = rf.Dtypes(
        w_dense=1.0, w_expert=1.0, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0
    )
    scenario = rf.TrafficScenario(
        name="long-ctx-physical-hbm",
        isl=65536 - 512,
        osl=512,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(32,),
        kv_working_set_turns=1.0,
    )
    serving_plan = rf.solve_serving_plan(
        geometry, dtypes, chip, scenario, SolveOptions(mode="mixed", max_chips=4)
    )
    assert serving_plan.candidates
    tp_target = 8
    point = next(
        cand
        for cand in serving_plan.candidates
        if cand.decode_pool.tp == tp_target
        and cand.decode_pool.dp == 1
        and cand.decode_pool.pp == 1
        and cand.decode_pool.pcp == 1
        and cand.decode_pool.kv_split == 1
    )
    assert point.decode_pool.resident_batch_cap == 32
    assert point.decode_pool.hbm_limit_gb == pytest.approx(chip.hbm_bytes / UNIT_GIGA)
    assert point.decode_pool.hbm_used_gb <= point.decode_pool.hbm_limit_gb


def test_solver_allows_pcp_and_pp_when_spec_k_exceeds_one() -> None:
    """Verify physics solver keeps prefill PCP > 1 and PP > 1 when spec_k > 1."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    geometry = bundle.geometry
    dtypes = bundle.dtypes
    chip = rf.CHIPS["tpu7x"]
    scen_k3 = rf.TrafficScenario(
        name="pcp-pp-spec-k3",
        isl=8192,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=3.0,
        think_time_s=0.0,
        concurrencies=(8,),
        kv_working_set_turns=1.0,
        spec_tokens=3,
    )
    plan_disagg = rf.solve_serving_plan(
        geometry, dtypes, chip, scen_k3, SolveOptions(mode="disagg", max_chips=8)
    )
    assert plan_disagg.prefill_pools
    assert plan_disagg.decode_pools
    assert any(r.pool.pcp > 1 for r in plan_disagg.prefill_pools)
    assert any(r.pool.pp > 1 for r in plan_disagg.prefill_pools)
    assert all(r.pool.pcp == 1 for r in plan_disagg.decode_pools)
    assert any(r.pool.pp > 1 for r in plan_disagg.decode_pools)
    assert any(r.is_pareto for r in plan_disagg.prefill_pools)
    assert any(r.is_pareto for r in plan_disagg.decode_pools)


def test_cli_ici_bw_override_scales_ici_bw_and_pool_tables(
    locked_config_path: Callable[[str], Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI --ici-bw-gbps scales ICI bandwidth without altering d2d_bw."""
    cfg_path = locked_config_path("qwen3_5_397b_fp8.json")
    rc_default = cli.main(
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
            "16",
            "--json",
        ]
    )
    assert rc_default == 0
    default_payload = json.loads(capsys.readouterr().out)
    rc_fast = cli.main(
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
            "16",
            "--ici-bw-gbps",
            "400",
            "--json",
        ]
    )
    assert rc_fast == 0
    fast_payload = json.loads(capsys.readouterr().out)
    rc_slow = cli.main(
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
            "16",
            "--ici-bw-gbps",
            "50",
            "--json",
        ]
    )
    assert rc_slow == 0
    slow_payload = json.loads(capsys.readouterr().out)
    assert fast_payload["candidates"]
    assert slow_payload["candidates"]
    assert fast_payload["hbm_reserve_gb"] == 0.0
    assert slow_payload["hbm_reserve_gb"] == 0.0
    assert (
        fast_payload["recommended"]["token_throughput_per_chip"]
        >= slow_payload["recommended"]["token_throughput_per_chip"]
    )
    tp_eight = 8
    default_tp8 = next(
        p
        for p in default_payload["candidates"]
        if p["decode_pool"]["tp"] == tp_eight
        and p["decode_pool"]["dp"] == 1
        and p["decode_pool"]["ep"] == tp_eight
        and p["decode_pool"]["kv_split"] == 1
    )
    fast_tp8 = next(
        p
        for p in fast_payload["candidates"]
        if p["decode_pool"]["tp"] == tp_eight
        and p["decode_pool"]["dp"] == 1
        and p["decode_pool"]["ep"] == tp_eight
        and p["decode_pool"]["kv_split"] == 1
    )
    slow_tp8 = next(
        p
        for p in slow_payload["candidates"]
        if p["decode_pool"]["tp"] == tp_eight
        and p["decode_pool"]["dp"] == 1
        and p["decode_pool"]["ep"] == tp_eight
        and p["decode_pool"]["kv_split"] == 1
    )
    assert (
        fast_tp8["est_tpot_ms"] < default_tp8["est_tpot_ms"] < slow_tp8["est_tpot_ms"]
    )

    rc_disagg_txt = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "disagg",
            "--max-chips",
            "4",
            "--concurrency",
            "1",
            "--top-k",
            "2",
        ]
    )
    assert rc_disagg_txt == 0
    table_out = capsys.readouterr().out
    assert "prefill" in table_out
    assert "decode" in table_out
    assert "pareto" in table_out


def test_decode_pool_record_counts_decode_steps_after_first_token() -> None:
    """Verify standalone decode pool records use max(0, osl - 1) decode steps."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    geometry = bundle.geometry
    dtypes = bundle.dtypes
    chip = rf.CHIPS["tpu7x"]
    topo = rf.TOPOLOGIES["2x2x1"]
    osl_two = 2
    osl_one = 1
    target_tp_ep = 8
    target_conc = 8
    ms_per_s = 1000.0
    scen_two = rf.TrafficScenario(
        name="dec-pool-osl2",
        isl=1024,
        osl=osl_two,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(target_conc,),
        kv_working_set_turns=1.0,
    )
    plan_two = rf.solve_serving_plan(
        geometry,
        dtypes,
        chip,
        scen_two,
        SolveOptions(mode="disagg", max_chips=8, topologies=(topo,)),
    )
    assert plan_two.decode_pools
    rec_two = next(
        rec
        for rec in plan_two.decode_pools
        if rec.pool.tp == target_tp_ep
        and rec.pool.dp == 1
        and rec.pool.ep == target_tp_ep
    )
    assert rec_two.tpot_ms is not None
    assert rec_two.tpot_ms > 0.0
    # osl=2 has exactly 1 decode step after the first token: rate = c_served / step_s.
    res_batch = rec_two.pool.resident_batch_cap or 1
    expected_rate = min(float(target_conc), float(res_batch)) / (
        rec_two.tpot_ms / ms_per_s
    )
    assert rec_two.request_rate_per_instance == pytest.approx(expected_rate, rel=1e-6)

    scen_one = rf.TrafficScenario(
        name="dec-pool-osl1",
        isl=1024,
        osl=osl_one,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(target_conc,),
        kv_working_set_turns=1.0,
    )
    plan_one = rf.solve_serving_plan(
        geometry,
        dtypes,
        chip,
        scen_one,
        SolveOptions(
            mode="disagg", max_chips=8, topologies=(topo,), include_pairs=True
        ),
    )
    assert plan_one.decode_pools == ()
    assert plan_one.prefill_pools
    assert plan_one.candidates
    for point in plan_one.candidates:
        assert math.isfinite(point.token_throughput_per_chip)
        assert math.isfinite(point.output_throughput_per_chip)
        assert math.isfinite(point.est_ttft_ms)
        assert math.isfinite(point.est_tpot_ms)
    plan_mixed_one = rf.solve_serving_plan(
        geometry,
        dtypes,
        chip,
        scen_one,
        SolveOptions(mode="mixed", max_chips=4, topologies=(topo,)),
    )
    assert plan_mixed_one.candidates
    assert all(pt.p_mixed == 0.0 for pt in plan_mixed_one.candidates)


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
def test_residual_sp_pool_collapse_lowers_prefill_floor_and_keeps_one_candidate(
    fixture_name: str,
) -> None:
    """Verify SP residual collapses with replicated into one pool candidate."""
    bundle = load_locked_bundle(fixture_name)
    chip = rf.CHIPS["tpu7x"]
    topo = rf.TOPOLOGIES["2x2x1"]
    scenario = rf.TrafficScenario(
        name="sp-residual-collapse",
        isl=8192,
        osl=512,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(8,),
        kv_working_set_turns=1.0,
    )
    plan = rf.solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        chip,
        scenario,
        SolveOptions(mode="disagg", topologies=(topo,)),
    )
    tp8_ep8_prefill = [
        rec
        for rec in plan.prefill_pools
        if rec.pool.tp == 8
        and rec.pool.ep == 8
        and rec.pool.dp == 1
        and rec.pool.pp == 1
        and rec.pool.pcp == 1
        and rec.pool.kv_split == 1
    ]
    assert len(tp8_ep8_prefill) == 1
    rec = tp8_ep8_prefill[0]
    assert rec.pool.residual == "sp"
    rep_floor = rf.step_floor(
        bundle.geometry,
        chip,
        rf.Parallel(tp=8, ep=8, moe="ep", residual="replicated"),
        rf.StepShape(batch=1, rows_per_seq=2048, context=8192),
        FloorOptions(dtypes=bundle.dtypes, topology=topo),
    )
    sp_floor = rf.step_floor(
        bundle.geometry,
        chip,
        rf.Parallel(tp=8, ep=8, moe="ep", residual="sp"),
        rf.StepShape(batch=1, rows_per_seq=2048, context=8192),
        FloorOptions(dtypes=bundle.dtypes, topology=topo),
    )
    assert sp_floor.hard_floor_s < rep_floor.hard_floor_s
    # PP > 1 or TP == 1 always selects replicated residual.
    for pool_rec in (*plan.prefill_pools, *plan.decode_pools):
        if pool_rec.pool.pp > 1 or pool_rec.pool.tp == 1:
            assert pool_rec.pool.residual == "replicated"


@pytest.mark.parametrize("osl", [1, 2, 16, 1024])
@pytest.mark.parametrize("accept_len", [1.0, 3.0])
def test_mixed_overflow_and_disagg_closed_loop_conservation(
    osl: int,
    accept_len: float,
) -> None:
    """Verify mixed prefill-overflow and prefill-starved disagg Little's Law."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    chip = rf.CHIPS["tpu7x"]
    topo = rf.TOPOLOGIES["2x2x1"]
    scenario = rf.TrafficScenario(
        name="closed-loop-conservation",
        isl=32768,
        osl=osl,
        prefix_hit_rate=0.0,
        accept_len=accept_len,
        think_time_s=0.0,
        concurrencies=(32,),
        kv_working_set_turns=1.0,
        spec_tokens=3 if accept_len > 1.0 else None,
    )
    mixed_plan = rf.solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        chip,
        scenario,
        SolveOptions(mode="mixed", topologies=(topo,)),
    )
    assert mixed_plan.candidates
    for point in mixed_plan.candidates:
        assert math.isfinite(point.est_ttft_ms)
        assert math.isfinite(point.est_tpot_ms)
        assert point.est_ttft_ms > 0.0
        if osl == 1:
            assert point.p_mixed == 0.0

    disagg_plan = rf.solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        chip,
        scenario,
        SolveOptions(
            mode="disagg",
            max_chips=12 if (osl == 16 and accept_len == 1.0) else 8,
            topologies=(topo,),
            include_pairs=True,
        ),
    )
    assert disagg_plan.candidates
    dec_steps = _decode_steps_per_req(osl, accept_len)
    for point in disagg_plan.candidates:
        assert math.isfinite(point.est_ttft_ms)
        assert math.isfinite(point.token_throughput_per_chip)
        rate_req_s = (
            point.output_throughput_per_chip * float(point.total_chips) / float(osl)
        )
        cycle_s = point.est_ttft_ms / 1000.0 + dec_steps * (
            point.est_tpot_ms * accept_len / 1000.0
        )
        assert float(point.concurrency) / rate_req_s <= cycle_s * (1.0 + 1e-6)


def test_cli_validation_rejects_invalid_flags_and_writes_warnings_to_stderr(
    locked_config_path: Callable[[str], Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI rejects non-positive flags and emits warnings to stderr."""
    cfg_path = locked_config_path("qwen3_5_397b_fp8.json")
    base_solve = [
        "solve",
        "--config",
        str(cfg_path),
        "--example-scenario",
        "chat-8k1k",
        "--max-chips",
        "4",
    ]
    for bad_argv, expected_err in (
        ([*base_solve, "--ici-bw-gbps", "0"], "ici_bw"),
        ([*base_solve, "--ici-bw-gbps", "-10"], "ici_bw"),
        ([*base_solve, "--max-instances", "0"], "--max-instances"),
        (
            [
                "solve",
                "--config",
                str(cfg_path),
                "--max-chips",
                "4",
                "--example-slo-tier",
                "interactive",
            ],
            "--example-slo-tier",
        ),
        (
            ["plan", "--config", str(cfg_path), "--tp", "8", "--concurrency", ""],
            "--concurrency",
        ),
    ):
        rc_bad = cli.main(bad_argv)
        captured_bad = capsys.readouterr()
        assert rc_bad == 2
        assert expected_err in captured_bad.err

    with pytest.raises(ValueError, match="prefix_hit_rate"):
        rf.TrafficScenario(
            name="nan-isl",
            isl=1024,
            osl=128,
            prefix_hit_rate=float("nan"),
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(1,),
            kv_working_set_turns=1.0,
        )

    rc = cli.main(
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
            "16",
            "--tpot-slo-ms",
            "0.01",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "WARNING:" in captured.err
    assert "WARNING:" not in captured.out


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
def test_locked_fixtures_mixed_and_disagg_end_to_end(fixture_name: str) -> None:
    """Verify all locked basis fixtures solve cleanly across mixed and disagg modes."""
    bundle = load_locked_bundle(fixture_name)
    chip = rf.CHIPS["tpu7x"]
    scenario = rf.TrafficScenario(
        name="locked-matrix",
        isl=4096,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=2.0,
        think_time_s=1.0,
        concurrencies=(1, 16),
        kv_working_set_turns=1.0,
    )
    mixed = rf.solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=8),
    )
    assert mixed.candidates
    disagg = rf.solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        chip,
        scenario,
        SolveOptions(
            mode="disagg",
            max_chips=16,
            topologies=(rf.TOPOLOGIES["2x2x1"], rf.TOPOLOGIES["2x2x2"]),
            include_pairs=True,
        ),
    )
    assert disagg.candidates
    assert any(
        pt.prefill_pool is not None and pt.decode_pool is not None
        for pt in disagg.candidates
    )


def test_pp_mixed_step_conserves_prefill_work_across_microbatches() -> None:
    """Verify PP > 1 mixed steps split uncached prefill tokens across microbatches."""
    from vllm_torchtpu.roofline.solver import _eval_pp_step_serial, _SolveContext

    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    chip = rf.CHIPS["tpu7x"]
    scenario = rf.TrafficScenario(
        name="pp-mixed-prefill",
        isl=8192,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(4,),
        kv_working_set_turns=1.0,
    )
    hbm_limit_b = float(chip.hbm_bytes)
    eval_context = _SolveContext(
        geometry=bundle.geometry,
        dtypes=bundle.dtypes,
        chip=chip,
        scenario=scenario,
        mode="mixed",
        max_chips=8,
        max_instances=1,
        hbm_reserve_gb=0.0,
        hbm_limit_b=hbm_limit_b,
        hbm_limit_gb=hbm_limit_b / UNIT_GIGA,
        effective_accept_len=1.0,
        spec_k=0,
        ctx_full=scenario.isl + scenario.osl,
        ctx_mid=scenario.isl + scenario.osl // 2,
        eff_isl=scenario.isl,
        n_chunks=1,
        c_max=4,
        shared_prefix_tokens=0,
        collective_latency_s=0.0,
        sort_by="throughput",
    )
    mixed_shape = rf.StepShape(
        batch=4,
        rows_per_seq=1,
        context=4096,
        state_writes=1,
        prefill_tokens=4096,
        prefix_hit_rate=0.0,
    )
    pp1_s, _ = _eval_pp_step_serial(
        eval_context,
        rf.TOPOLOGIES["2x2x1"],
        rf.Parallel(tp=8, pp=1, ep=8, moe="ep"),
        mixed_shape,
        local_batch=4,
    )
    pp2_s, _ = _eval_pp_step_serial(
        eval_context,
        rf.TOPOLOGIES["2x2x2"],
        rf.Parallel(tp=8, pp=2, ep=8, moe="ep"),
        mixed_shape,
        local_batch=4,
    )
    assert pp2_s < 0.75 * pp1_s
    assert pp2_s >= 0.45 * pp1_s
