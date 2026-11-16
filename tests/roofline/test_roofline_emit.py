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
"""CPU unit tests for vLLM serve flag emission and Pareto ceiling export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING, Literal

import pytest

from vllm_torchtpu.roofline import (
    CHIPS,
    SOLVE_SCHEMA,
    Dtypes,
    LaunchOptions,
    ParetoPoint,
    PoolMeshPlan,
    ServingPlan,
    SolveOptions,
    TrafficScenario,
    build_solve_payload,
    cli,
    dtypes_from_hf_config,
    emit_ceiling_rows,
    emit_launch_args,
    format_parallelism,
    from_hf_config,
    parse_layout,
    point_launch_args,
    pool_cli_args,
    pool_label,
    solve_serving_plan,
    top_layouts,
)
from vllm_torchtpu.roofline import emit as rf_emit
from vllm_torchtpu.roofline.emit import (
    _filter_eligible,
    _point_struct_key,
    _pool_key,
    top_layout_roles,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class _PoolConfig:
    """Typed synthetic pool configuration for flag-emission tests."""

    instances: int = 1
    topology: str = "2x2x2"
    chips: int = 8
    tp: int = 2
    dp: int = 4
    ep: int = 8
    pp: int = 1
    pcp: int = 1
    kv_split: int = 1
    moe_dispatch: Literal["allreduce", "a2a_push", "ag_ep", "fused_rs"] = "allreduce"


def _make_pareto_point(
    pool: PoolMeshPlan,
    conc: int,
    tput: float,
    inter: float,
    *,
    total_chips: int = 8,
) -> ParetoPoint:
    """Create a synthetic Pareto point for testing layout selection."""
    return ParetoPoint(
        mode="mixed",
        concurrency=conc,
        total_chips=total_chips,
        prefill_pool=None,
        decode_pool=pool,
        est_ttft_ms=100.0,
        est_tpot_ms=1000.0 / inter,
        est_interactivity_tok_s=inter,
        output_throughput_per_chip=tput * 0.1,
        token_throughput_per_chip=tput,
        p_mixed=0.05,
        binding="hbm",
        slo_met=True,
    )


def _toy_moe_config_dict() -> dict[str, object]:
    """Return sample MoE model configuration dictionary for CLI emission tests."""
    return {
        "model_type": "qwen3_moe",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 12288,
        "moe_intermediate_size": 1536,
        "shared_expert_intermediate_size": 3072,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "vocab_size": 151936,
        "dtype": "bfloat16",
        "quantization_config": {"quant_method": "fp8"},
    }


def _sample_pool(
    role: Literal["mixed", "prefill", "decode"] = "decode",
    cfg: _PoolConfig | None = None,
) -> PoolMeshPlan:
    """Build synthetic PoolMeshPlan fixture with typed parallel dimensions."""
    spec = cfg or _PoolConfig()
    return PoolMeshPlan(
        role=role,
        instances=spec.instances,
        topology=spec.topology,
        chips_per_instance=spec.chips,
        tp=spec.tp,
        dp=spec.dp,
        ep=spec.ep,
        pp=spec.pp,
        pcp=spec.pcp,
        kv_split=spec.kv_split,
        mesh_axes=(2, 2, 2, 2),
        hbm_used_gb=42.0,
        hbm_limit_gb=83.25,
        moe_dispatch=spec.moe_dispatch,
    )


def test_pool_cli_args_vllm_layouts() -> None:
    """Verify mapping from PoolMeshPlan topologies to vLLM server CLI arguments."""
    dp_ep_pool = _sample_pool(cfg=_PoolConfig(tp=2, dp=4, ep=8, pp=2))
    vllm_flags = pool_cli_args(
        dp_ep_pool,
        kv_dtype="fp8",
        gpu_memory_utilization=0.9,
    )
    assert vllm_flags == [
        "--tensor-parallel-size",
        "2",
        "--data-parallel-size",
        "4",
        "--kv-cache-dtype",
        "fp8",
        "--pipeline-parallel-size",
        "2",
        "--enable-expert-parallel",
        "--gpu-memory-utilization",
        "0.9",
    ]

    cp_pool = _sample_pool(cfg=_PoolConfig(tp=8, dp=1, ep=1, pcp=2, kv_split=4))
    cp_vllm = pool_cli_args(cp_pool, kv_dtype="auto")
    assert cp_vllm == [
        "--tensor-parallel-size",
        "8",
        "--data-parallel-size",
        "1",
        "--kv-cache-dtype",
        "auto",
        "--prefill-context-parallel-size",
        "2",
        "--decode-context-parallel-size",
        "4",
    ]


def test_emit_launch_args_mixed_and_disagg() -> None:
    """Verify emission of launch commands for mixed and disaggregated serving plans."""
    scen = TrafficScenario(
        name="chat-8k1k",
        isl=8192,
        osl=1024,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(16, 64),
        kv_working_set_turns=1.0,
    )
    mixed_pool = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x1", chips=4, tp=8, dp=1, ep=8),
    )
    mixed_pt = ParetoPoint(
        mode="mixed",
        concurrency=16,
        total_chips=4,
        prefill_pool=None,
        decode_pool=mixed_pool,
        est_ttft_ms=180.0,
        est_tpot_ms=12.0,
        est_interactivity_tok_s=83.3,
        output_throughput_per_chip=320.0,
        token_throughput_per_chip=2880.0,
        p_mixed=0.15,
        binding="hbm",
        slo_met=True,
    )
    mixed_plan = ServingPlan(
        model_layers=32,
        chip="tpu7x",
        scenario=scen,
        mode="mixed",
        recommended=mixed_pt,
        frontier=(mixed_pt,),
        warnings=(),
        gpu_memory_utilization=0.9,
    )
    mixed_out = emit_launch_args(
        mixed_plan,
        dtypes=Dtypes(kv=1.0),
        model="Qwen/Qwen3.5-397B-A17B-FP8",
    )
    assert mixed_out["prefill_cli_args"] == []
    assert len(mixed_out["commands"]) == 1
    assert mixed_out["commands"][0].startswith(
        "vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --tensor-parallel-size 8"
    )

    prefill_pool = _sample_pool(
        role="prefill",
        cfg=_PoolConfig(
            instances=1, topology="2x2x1", chips=4, tp=4, dp=1, ep=1, pcp=2
        ),
    )
    decode_pool = _sample_pool(
        role="decode",
        cfg=_PoolConfig(instances=2, topology="2x2x2", chips=8, tp=2, dp=4, ep=8),
    )
    disagg_pt = ParetoPoint(
        mode="disagg",
        concurrency=64,
        total_chips=20,
        prefill_pool=prefill_pool,
        decode_pool=decode_pool,
        est_ttft_ms=140.0,
        est_tpot_ms=9.5,
        est_interactivity_tok_s=105.2,
        output_throughput_per_chip=336.8,
        token_throughput_per_chip=3031.2,
        p_mixed=0.0,
        binding="ici",
        slo_met=True,
    )
    disagg_plan = ServingPlan(
        model_layers=32,
        chip="tpu7x",
        scenario=scen,
        mode="disagg",
        recommended=disagg_pt,
        frontier=(disagg_pt,),
        warnings=(),
        gpu_memory_utilization=0.9,
    )
    disagg_out = emit_launch_args(
        disagg_plan,
        dtypes=Dtypes(kv=1.0),
        model="Qwen/Qwen3.5-397B-A17B-FP8",
    )
    assert "--prefill-context-parallel-size" in disagg_out["prefill_cli_args"]
    assert any(c.startswith("# provision: prefill") for c in disagg_out["commands"])
    assert any(c.startswith("# provision: decode") for c in disagg_out["commands"])
    assert any(
        c.startswith("vllm serve") and "--prefill-context-parallel-size" in c
        for c in disagg_out["commands"]
    )
    assert any(
        c.startswith("vllm serve") and "--decode-context-parallel-size" not in c
        for c in disagg_out["commands"]
    )
    assert any(c.startswith("# note:") for c in disagg_out["commands"])


def test_emit_launch_args_infeasible_plan_returns_empty_and_cli_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that infeasible SLO plans exit non-zero and write stderr warnings."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(_toy_moe_config_dict()), encoding="utf-8")

    rc_emit = cli.main(
        [
            "emit",
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
            "0.001",
        ]
    )
    captured = capsys.readouterr()
    assert rc_emit == 1
    assert captured.out == ""
    assert "WARNING:" in captured.err

    rc_ceil = cli.main(
        [
            "ceiling",
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
            "0.001",
            "--json",
        ]
    )
    captured_ceil = capsys.readouterr()
    assert rc_ceil == 1
    assert captured_ceil.out.strip() == "[]"
    assert "WARNING:" in captured_ceil.err


def test_emit_ceiling_rows_and_cli_subcommands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify Pareto ceiling row formatting and CLI subcommand JSON output."""
    scen = TrafficScenario(
        name="chat-8k1k",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 8, 32),
        kv_working_set_turns=1.0,
    )
    splan = solve_serving_plan(
        from_hf_config(_toy_moe_config_dict()),
        Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0),
        CHIPS["tpu7x"],
        scen,
        SolveOptions(mode="mixed", max_chips=4),
    )
    rows = emit_ceiling_rows(splan)
    assert len(rows) >= 1
    assert list(rows[0].keys())[:5] == [
        "est_interactivity_tok_s",
        "token_throughput_per_chip",
        "output_throughput_per_chip",
        "est_ttft_ms",
        "est_tpot_ms",
    ]

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(_toy_moe_config_dict()), encoding="utf-8")

    rc_vllm = cli.main(
        [
            "emit",
            "--model",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--concurrency",
            "8,32",
            "--json",
        ]
    )
    out_vllm = json.loads(capsys.readouterr().out)
    assert rc_vllm == 0
    assert "--tensor-parallel-size" in out_vllm["cli_args"]
    assert "--gpu-memory-utilization" in out_vllm["cli_args"]
    assert "env" not in out_vllm

    rc_ceil = cli.main(
        [
            "ceiling",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--concurrency",
            "8,32",
            "--json",
        ]
    )
    out_ceil = json.loads(capsys.readouterr().out)
    assert rc_ceil == 0
    assert len(out_ceil) >= 1
    assert "est_interactivity_tok_s" in out_ceil[0]


def test_resident_batch_cap_emits_max_num_seqs_without_tput_inflation() -> None:
    """Verify resident batch cap adds --max-num-seqs without inflating tput."""
    dense_cfg = {
        "hidden_size": 2048,
        "num_hidden_layers": 8,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 4096,
        "vocab_size": 32000,
    }
    geom = from_hf_config(dense_cfg)
    dt = Dtypes(kv=1.0)
    chip = CHIPS["tpu7x"]
    cap_concurrency = 15
    overflow_concurrency = 256
    expected_tp = 8
    expected_max_model_len = 131328
    scen = TrafficScenario(
        name="long-ctx-cap",
        isl=131072,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(cap_concurrency, overflow_concurrency),
        kv_working_set_turns=1.0,
    )
    splan = solve_serving_plan(
        geom,
        dt,
        chip,
        scen,
        SolveOptions(mode="mixed", max_chips=4),
    )
    dp1_pts = {
        p.concurrency: p
        for p in splan.candidates
        if p.decode_pool.tp == expected_tp
        and p.decode_pool.dp == 1
        and p.decode_pool.pp == 1
        and p.decode_pool.pcp == 1
        and p.decode_pool.kv_split == 1
    }
    assert cap_concurrency in dp1_pts
    assert overflow_concurrency in dp1_pts
    pt_15 = dp1_pts[cap_concurrency]
    pt_256 = dp1_pts[overflow_concurrency]
    assert pt_256.decode_pool.max_num_seqs == cap_concurrency
    assert pt_256.decode_pool.max_model_len == expected_max_model_len
    flags = pool_cli_args(pt_256.decode_pool, kv_dtype="fp8")
    assert "--max-model-len" in flags
    assert flags[flags.index("--max-model-len") + 1] == str(expected_max_model_len)
    assert "--max-num-seqs" in flags
    assert flags[flags.index("--max-num-seqs") + 1] == str(cap_concurrency)
    assert pt_256.token_throughput_per_chip == pytest.approx(
        pt_15.token_throughput_per_chip, rel=1e-6
    )
    assert pt_256.est_ttft_ms > pt_15.est_ttft_ms


def test_top_layouts_distinct_and_led_by_recommended() -> None:
    """Top-K layouts are distinct, SLO-met, sorted by throughput, led by recommended."""
    chip = CHIPS["tpu7x"]
    gqa_cfg = {
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 32000,
    }
    geom = from_hf_config(gqa_cfg)
    dt = Dtypes(kv=1.0)
    scenario = TrafficScenario(
        name="chat-8k1k",
        isl=8192,
        osl=1024,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 2, 4),
        kv_working_set_turns=1.0,
    )
    splan = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=4),
    )
    top_k = 3
    top = top_layouts(splan, k=top_k)
    assert 1 <= len(top) <= top_k
    assert splan.recommended is not None
    assert top[0] == splan.recommended

    keys: set[tuple[object, ...]] = set()
    for pt in top:
        assert pt.slo_met
        key = (
            pt.mode,
            _pool_key(pt.prefill_pool),
            _pool_key(pt.decode_pool),
        )
        assert key not in keys
        keys.add(key)

    tputs = [pt.token_throughput_per_chip for pt in top]
    assert tputs == sorted(tputs, reverse=True)


def test_top_layouts_empty_without_slo_met() -> None:
    """Top-K layouts return empty tuple if no candidate meets SLOs."""
    chip = CHIPS["tpu7x"]
    gqa_cfg = {
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 32000,
    }
    geom = from_hf_config(gqa_cfg)
    dt = Dtypes(kv=1.0)
    scenario = TrafficScenario(
        name="custom",
        isl=8192,
        osl=1024,
        concurrencies=(1,),
        tpot_slo_ms=0.0001,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
    )
    splan = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=4),
    )
    top = top_layouts(splan, k=3)
    assert top == ()


def test_solve_json_v1_envelope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Solve --json outputs schema v1 with provenance, stats, and top_k launch."""
    cfg = {
        "model_type": "llama",
        "torch_dtype": "bfloat16",
        "num_hidden_layers": 8,
        "hidden_size": 1024,
        "intermediate_size": 2048,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": 32000,
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    rc = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--chip",
            "tpu7x",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--concurrency",
            "1,2",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == SOLVE_SCHEMA
    assert out["provenance"]["calibrated"] is False
    assert "search" in out
    assert out["search"]["enumerated"] >= 1
    assert "top_k" in out
    assert len(out["top_k"]) >= 1
    assert "launch" in out["top_k"][0]
    assert out["top_k"][0]["launch"]["commands"][0].startswith("vllm serve")
    assert "recommended" in out
    assert "frontier" in out
    assert "candidates" in out


def test_top_layouts_structural_diversity_and_interactivity_anchor() -> None:
    """Verify structural diversity across top layouts with interactivity anchor."""
    scen = TrafficScenario(
        name="chat-8k1k",
        isl=4096,
        osl=512,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 16, 64),
        kv_working_set_turns=1.0,
    )
    p_a1 = dc_replace(
        _sample_pool(
            role="mixed",
            cfg=_PoolConfig(topology="2x2x2", chips=8, tp=2, dp=8, ep=16),
        ),
        moe_dispatch="ag_ep",
    )
    p_a2 = dc_replace(
        _sample_pool(
            role="mixed",
            cfg=_PoolConfig(topology="2x2x2", chips=8, tp=2, dp=8, ep=16),
        ),
        moe_dispatch="a2a_push",
    )
    p_b = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x2", chips=8, tp=4, dp=4, ep=16),
    )
    p_c = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x2", chips=8, tp=8, dp=2, ep=16),
    )
    p_d = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x2", chips=8, tp=16, dp=1, ep=16),
    )

    a1 = _make_pareto_point(p_a1, 64, 5000.0, 40.0)
    a2 = _make_pareto_point(p_a2, 64, 4900.0, 39.0)
    b = _make_pareto_point(p_b, 32, 4200.0, 65.0)
    c = _make_pareto_point(p_c, 16, 3800.0, 80.0)
    d = _make_pareto_point(p_d, 1, 2500.0, 130.0)
    splan = ServingPlan(
        model_layers=32,
        chip="tpu7x",
        scenario=scen,
        mode="mixed",
        recommended=a1,
        frontier=(d, c, b, a1),
        warnings=(),
        candidates=(a1, a2, b, c, d),
    )

    top_k = 3
    top = top_layouts(splan, k=top_k)
    assert top == (a1, b, d)
    cmds = {
        tuple(
            point_launch_args(pt, LaunchOptions(kv_dtype="fp8", model="m"))["commands"]
        )
        for pt in top
    }
    assert len(cmds) == top_k

    payload = build_solve_payload(
        splan,
        chip=CHIPS["tpu7x"],
        dtypes=Dtypes(kv=1.0),
        model="m",
        top_k=top_k,
    )
    roles = [entry["frontier_role"] for entry in payload["top_k"]]
    margins = [entry["margin_vs_rank1_pct"] for entry in payload["top_k"]]
    assert roles == ["max_throughput", "balanced_knee", "max_interactivity"]
    assert margins == [0.0, -16.0, -50.0]


def test_emit_omits_mtp_speculative_config_when_mtp_layers_zero() -> None:
    """Verify emit_launch_args omits --speculative-config when MTP layers is 0."""
    scen = TrafficScenario(
        name="custom",
        isl=1024,
        osl=128,
        concurrencies=(16,),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        spec_tokens=3,
    )
    pool = PoolMeshPlan(
        role="mixed",
        instances=1,
        tp=8,
        dp=1,
        pp=1,
        pcp=1,
        ep=1,
        kv_split=1,
        chips_per_instance=8,
        topology="2x2x2",
        mesh_axes=(2, 2, 2, 1),
        hbm_used_gb=10.0,
        hbm_limit_gb=94.76,
    )
    pt = ParetoPoint(
        mode="mixed",
        concurrency=16,
        total_chips=8,
        prefill_pool=None,
        decode_pool=pool,
        est_ttft_ms=50.0,
        est_tpot_ms=20.0,
        est_interactivity_tok_s=50.0,
        output_throughput_per_chip=100.0,
        token_throughput_per_chip=1000.0,
        p_mixed=0.0,
        binding="hbm",
        slo_met=True,
    )
    plan = ServingPlan(
        model_layers=32,
        chip="tpu7x",
        scenario=scen,
        mode="mixed",
        recommended=pt,
        frontier=(pt,),
        warnings=(),
        candidates=(pt,),
        mtp_layers=0,
        spec_tokens=3,
    )
    emitted = emit_launch_args(plan, dtypes=Dtypes(kv=1.0), model="test-model")
    assert "--speculative-config" not in emitted["cli_args"]
    assert "--speculative-config" not in emitted["commands"][0]
    assert any("omitting --speculative-config" in w for w in emitted["warnings"])


def test_emit_dispatch_env_and_enable_ep_for_fused_rs() -> None:
    """Verify ag_ep omits phantom env and fused_rs emits --enable-expert-parallel."""
    ag_pool = PoolMeshPlan(
        role="mixed",
        instances=1,
        topology="2x2x2",
        chips_per_instance=8,
        tp=4,
        dp=4,
        ep=16,
        pp=1,
        pcp=1,
        kv_split=1,
        mesh_axes=(2, 2, 2, 1),
        hbm_used_gb=10.0,
        hbm_limit_gb=94.76,
        moe_dispatch="ag_ep",
    )
    ag_pt = ParetoPoint(
        mode="mixed",
        concurrency=16,
        total_chips=8,
        prefill_pool=None,
        decode_pool=ag_pool,
        est_ttft_ms=50.0,
        est_tpot_ms=10.0,
        est_interactivity_tok_s=100.0,
        output_throughput_per_chip=100.0,
        token_throughput_per_chip=1000.0,
        p_mixed=0.0,
        binding="hbm",
        slo_met=True,
    )
    ag_cmd = point_launch_args(ag_pt, LaunchOptions(model="m"))["commands"][0]
    assert "ENABLE_PIPELINED_COLLECTIVE_AND_COMPUTE" not in ag_cmd

    rs_pool = PoolMeshPlan(
        role="mixed",
        instances=1,
        topology="2x2x1",
        chips_per_instance=4,
        tp=8,
        dp=1,
        ep=1,
        pp=1,
        pcp=1,
        kv_split=1,
        mesh_axes=(2, 2, 1, 1),
        hbm_used_gb=10.0,
        hbm_limit_gb=94.76,
        moe_dispatch="fused_rs",
    )
    assert "--enable-expert-parallel" in pool_cli_args(rs_pool)
    rs_pt = ParetoPoint(
        mode="mixed",
        concurrency=8,
        total_chips=4,
        prefill_pool=None,
        decode_pool=rs_pool,
        est_ttft_ms=50.0,
        est_tpot_ms=10.0,
        est_interactivity_tok_s=100.0,
        output_throughput_per_chip=100.0,
        token_throughput_per_chip=1000.0,
        p_mixed=0.0,
        binding="hbm",
        slo_met=True,
    )
    rs_cmd = point_launch_args(rs_pt, LaunchOptions(model="m"))["commands"][0]
    assert "TPU_MOE_SEQUENCE_PARALLEL=1" in rs_cmd
    assert "--enable-expert-parallel" in rs_cmd


def _synthetic_pareto_point(
    mode: Literal["mixed", "disagg"],
    pools: tuple[PoolMeshPlan | None, PoolMeshPlan],
    conc: int,
    tput: float,
    inter: float,
) -> ParetoPoint:
    """Create a synthetic SLO-met ParetoPoint fixture from prefill/decode pools."""
    pref, dec = pools
    chips = (
        pref.instances * pref.chips_per_instance if pref is not None else 0
    ) + dec.instances * dec.chips_per_instance
    return ParetoPoint(
        mode=mode,
        concurrency=conc,
        total_chips=chips,
        prefill_pool=pref,
        decode_pool=dec,
        est_ttft_ms=100.0,
        est_tpot_ms=1000.0 / inter,
        est_interactivity_tok_s=inter,
        output_throughput_per_chip=tput * 0.1,
        token_throughput_per_chip=tput,
        p_mixed=0.0,
        binding="hbm",
        slo_met=True,
        slo_verdict="estimate_meets",
    )


def _agentic_scenario(name: str = "agentic-64k") -> TrafficScenario:
    """Build synthetic TrafficScenario fixture for Top-K diversity tests."""
    return TrafficScenario(
        name=name,
        isl=65536,
        osl=1024,
        prefix_hit_rate=0.88,
        accept_len=3.0,
        think_time_s=0.0,
        concurrencies=(1, 64, 256),
        kv_working_set_turns=1.0,
    )


def test_top_layouts_k5_surfaces_diverse_frontiers_and_pp8() -> None:
    """Verify k=5 surfaces PP>=8 and diverse archetypes behind dense anchors."""
    expected_top_5 = 5
    expected_top_3 = 3
    pp_8 = 8
    scen = _agentic_scenario()
    m_base = _sample_pool(
        role="mixed", cfg=_PoolConfig(topology="2x2x2", chips=8, tp=2, dp=8, ep=16)
    )
    m_dcp2 = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x2", chips=8, tp=2, dp=4, ep=16, kv_split=2),
    )
    m_dcp4 = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x2", chips=8, tp=2, dp=2, ep=16, kv_split=4),
    )
    m_dcp8 = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x2", chips=8, tp=2, dp=1, ep=16, kv_split=8),
    )
    m_fast = _sample_pool(
        role="mixed", cfg=_PoolConfig(topology="2x2x2", chips=8, tp=16, dp=1, ep=16)
    )
    p_pp8 = _sample_pool(
        role="prefill",
        cfg=_PoolConfig(topology="2x2x1", chips=4, tp=1, dp=1, ep=1, pp=pp_8, pcp=1),
    )
    p_pcp8 = _sample_pool(
        role="prefill",
        cfg=_PoolConfig(topology="2x2x1", chips=4, tp=1, dp=1, ep=8, pp=1, pcp=8),
    )
    d_tp8 = _sample_pool(
        role="decode", cfg=_PoolConfig(topology="2x2x2", chips=8, tp=8, dp=2, ep=16)
    )

    c_base = _synthetic_pareto_point("mixed", (None, m_base), 256, 10000.0, 50.0)
    c_d2 = _synthetic_pareto_point("mixed", (None, m_dcp2), 256, 9900.0, 51.0)
    c_d4 = _synthetic_pareto_point("mixed", (None, m_dcp4), 256, 9800.0, 52.0)
    c_d8 = _synthetic_pareto_point("mixed", (None, m_dcp8), 256, 9700.0, 53.0)
    c_pp8 = _synthetic_pareto_point("disagg", (p_pp8, d_tp8), 64, 7000.0, 220.0)
    c_pcp8 = _synthetic_pareto_point("disagg", (p_pcp8, d_tp8), 64, 6500.0, 260.0)
    c_fast = _synthetic_pareto_point("mixed", (None, m_fast), 1, 1500.0, 900.0)

    splan = ServingPlan(
        model_layers=60,
        chip="tpu7x",
        scenario=scen,
        mode="auto",
        recommended=c_base,
        frontier=(c_fast, c_pcp8, c_pp8, c_d8, c_d4, c_d2, c_base),
        warnings=(),
        candidates=(c_base, c_d2, c_d4, c_d8, c_pp8, c_pcp8, c_fast),
    )
    top5 = top_layouts(splan)
    assert len(top5) == expected_top_5
    assert c_base in top5
    assert c_fast in top5
    assert c_pp8 in top5
    assert c_pcp8 in top5
    assert len(top_layouts(splan, k=expected_top_3)) == expected_top_3


def test_top_layouts_k5_surfaces_pp8_over_pp2_pcp4() -> None:
    """Verify k=5 still surfaces PP8 when Rank-1 is disagg PP2+PCP4 across 8+ keys."""
    expected_top_5 = 5
    pp_8 = 8
    m_base = _sample_pool(
        role="mixed", cfg=_PoolConfig(topology="2x2x2", chips=8, tp=2, dp=8, ep=16)
    )
    p_pp2 = _sample_pool(
        role="prefill",
        cfg=_PoolConfig(topology="2x2x1", chips=4, tp=1, dp=1, ep=4, pp=2, pcp=4),
    )
    p_pp8 = _sample_pool(
        role="prefill",
        cfg=_PoolConfig(topology="2x2x1", chips=4, tp=1, dp=1, ep=1, pp=pp_8, pcp=1),
    )
    p_pcp8 = _sample_pool(
        role="prefill",
        cfg=_PoolConfig(topology="2x2x1", chips=4, tp=1, dp=1, ep=8, pp=1, pcp=8),
    )
    d_tp4 = _sample_pool(
        role="decode", cfg=_PoolConfig(topology="2x2x1", chips=4, tp=4, dp=2, ep=8)
    )
    d_tp4_dcp2 = _sample_pool(
        role="decode",
        cfg=_PoolConfig(topology="2x2x1", chips=4, tp=4, dp=1, ep=8, kv_split=2),
    )
    d_tp8 = _sample_pool(
        role="decode", cfg=_PoolConfig(topology="2x2x2", chips=8, tp=8, dp=2, ep=16)
    )
    d_tp8_dcp2 = _sample_pool(
        role="decode",
        cfg=_PoolConfig(topology="2x2x2", chips=8, tp=8, dp=1, ep=16, kv_split=2),
    )
    d_tp16 = _sample_pool(
        role="decode",
        cfg=_PoolConfig(instances=2, topology="2x2x2", chips=8, tp=16, dp=1, ep=16),
    )
    a_pp2 = _synthetic_pareto_point("disagg", (p_pp2, d_tp8), 256, 12000.0, 80.0)
    a_pp2_b = _synthetic_pareto_point(
        "disagg", (p_pp2, d_tp8_dcp2), 128, 11200.0, 105.0
    )
    a_pp2_c = _synthetic_pareto_point("disagg", (p_pp2, d_tp4), 128, 10400.0, 130.0)
    a_pp2_d = _synthetic_pareto_point("disagg", (p_pp2, d_tp4_dcp2), 64, 9600.0, 155.0)
    b_mix = _synthetic_pareto_point("mixed", (None, m_base), 64, 8500.0, 180.0)
    b_pp8 = _synthetic_pareto_point("disagg", (p_pp8, d_tp8), 64, 7800.0, 240.0)
    b_pp2_mid = _synthetic_pareto_point("disagg", (p_pp2, d_tp16), 32, 6200.0, 480.0)
    b_fast = _synthetic_pareto_point("disagg", (p_pcp8, d_tp16), 1, 2000.0, 950.0)
    pp2_frontier = (
        b_fast,
        b_pp2_mid,
        b_pp8,
        b_mix,
        a_pp2_d,
        a_pp2_c,
        a_pp2_b,
        a_pp2,
    )
    splan_pp2 = ServingPlan(
        model_layers=60,
        chip="tpu7x",
        scenario=_agentic_scenario(),
        mode="auto",
        recommended=a_pp2,
        frontier=pp2_frontier,
        warnings=(),
        candidates=tuple(reversed(pp2_frontier)),
    )
    top5_pp2 = top_layouts(splan_pp2, k=expected_top_5)
    assert len(top5_pp2) == expected_top_5
    assert b_pp8 in top5_pp2
    assert all(pt in pp2_frontier for pt in top5_pp2)


def test_top_layouts_k5_selects_actual_frontier_over_off_frontier_candidates() -> None:
    """Verify k=5 never replaces a frontier layout point with a dominated candidate."""
    expected_top_5 = 5
    scen = _agentic_scenario(name="frontier_only")
    m_f1 = _sample_pool("mixed", _PoolConfig(topology="2x2x2", chips=8, tp=2, dp=8))
    m_f2 = _sample_pool("mixed", _PoolConfig(topology="2x2x2", chips=8, tp=4, dp=4))
    m_f3 = _sample_pool("mixed", _PoolConfig(topology="2x2x2", chips=8, tp=8, dp=2))
    m_f4 = _sample_pool(
        "mixed", _PoolConfig(topology="2x2x2", chips=8, tp=4, dp=2, kv_split=2)
    )
    m_f5 = _sample_pool("mixed", _PoolConfig(topology="2x2x2", chips=8, tp=16, dp=1))
    m_off = _sample_pool(
        "mixed", _PoolConfig(topology="2x2x2", chips=8, tp=2, dp=4, kv_split=2)
    )

    f1 = _synthetic_pareto_point("mixed", (None, m_f1), 256, 10000.0, 90.0)
    # Same structural key as f2, higher throughput at c=256, strictly dominated by f1.
    f2_off_hi = _synthetic_pareto_point("mixed", (None, m_f2), 256, 9800.0, 60.0)
    f2_on = _synthetic_pareto_point("mixed", (None, m_f2), 64, 9000.0, 140.0)
    # Wholly non-frontier structural key, higher throughput than f2_on, dominated by f1.
    g_off = _synthetic_pareto_point("mixed", (None, m_off), 256, 9700.0, 65.0)
    f3 = _synthetic_pareto_point("mixed", (None, m_f3), 32, 7500.0, 260.0)
    f4 = _synthetic_pareto_point("mixed", (None, m_f4), 16, 5200.0, 480.0)
    f5 = _synthetic_pareto_point("mixed", (None, m_f5), 1, 2200.0, 920.0)

    frontier = (f5, f4, f3, f2_on, f1)
    splan = ServingPlan(
        model_layers=60,
        chip="tpu7x",
        scenario=scen,
        mode="mixed",
        recommended=f1,
        frontier=frontier,
        warnings=(),
        candidates=(f1, f2_off_hi, g_off, f2_on, f3, f4, f5),
    )
    top5 = top_layouts(splan, k=expected_top_5)
    assert len(top5) == expected_top_5
    assert all(pt in frontier for pt in top5)
    assert f2_off_hi not in top5
    assert g_off not in top5
    assert f2_on in top5


def test_result_rows_print_chips_slices_and_counts() -> None:
    """Verify pool_label unconditionally formats ASCII '1x ' prefix and total chips."""
    pool = {
        "instances": 1,
        "topology": "2x2x1",
        "tp": 4,
        "dp": 1,
        "ep": 4,
    }
    label = pool_label(pool)
    assert label.startswith("1x ")
    assert "2x2x1" in label
    assert "TP4" in label
    assert "\u00d7" not in label

    pool_multi = {
        "instances": 4,
        "topology": "2x2x2",
        "tp": 8,
        "dp": 4,
        "ep": 16,
    }
    label_multi = pool_label(pool_multi)
    assert label_multi.startswith("4x ")
    assert "2x2x2" in label_multi
    assert "TP8" in label_multi


def test_top_layouts_retains_near_ties() -> None:
    """Verify top_layouts retains candidates within near_tie_rel of kth throughput."""
    sample_traffic_scenario = _agentic_scenario()
    p_a1 = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x2", chips=8, tp=2, dp=4, ep=8),
    )
    p_a2 = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x2", chips=8, tp=2, dp=4, ep=16),
    )
    p_b = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x2", chips=8, tp=4, dp=2, ep=8),
    )
    p_c = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x2", chips=8, tp=8, dp=1, ep=8),
    )

    a1 = _make_pareto_point(p_a1, 64, 5000.0, 40.0)
    a2 = _make_pareto_point(p_a2, 64, 4900.0, 39.0)
    b = _make_pareto_point(p_b, 32, 4000.0, 60.0)
    c = _make_pareto_point(p_c, 16, 3900.0, 70.0)

    splan = ServingPlan(
        model_layers=32,
        chip="tpu7x",
        scenario=sample_traffic_scenario,
        mode="mixed",
        recommended=a1,
        frontier=(b, a1),
        warnings=(),
        candidates=(a1, a2, b, c),
    )

    k_val = 2
    selected_strict = top_layouts(splan, k=k_val, near_tie_rel=0.0)
    assert len(selected_strict) == k_val

    selected = top_layouts(splan, k=k_val, near_tie_rel=0.10)
    assert len(selected) > k_val
    assert a2 in selected or c in selected


def test_emit_launch_args_config_path_not_in_serve_command(
    tmp_path: Path,
) -> None:
    """Verify config path never leaks into emitted vllm serve command."""
    sample_traffic_scenario = _agentic_scenario()
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        '{"model_type": "llama", "num_hidden_layers": 32}', encoding="utf-8"
    )
    pool = _sample_pool(
        role="mixed", cfg=_PoolConfig(topology="2x2x1", chips=4, tp=4, dp=1, ep=1)
    )
    pt = _make_pareto_point(pool, 16, 1000.0, 50.0, total_chips=4)
    splan = ServingPlan(
        model_layers=32,
        chip="tpu7x",
        scenario=sample_traffic_scenario,
        mode="mixed",
        recommended=pt,
        frontier=(pt,),
        warnings=(),
        candidates=(pt,),
    )
    out = emit_launch_args(splan, model=None)
    for cmd in out["commands"]:
        assert str(cfg_file) not in cmd


def test_pool_label_joins_axes_with_x_and_omits_ones() -> None:
    """Verify format_parallelism and pool_label omit size-1 axes and join with x."""
    sample_pool = {
        "instances": 1,
        "topology": "2x2x1",
        "chips_per_instance": 4,
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "pp": 1,
        "pcp": 1,
        "kv_split": 4,
    }
    assert format_parallelism(sample_pool) == "TP8 x EP8 x DCP4"
    label = pool_label(sample_pool)
    assert label.endswith(": TP8 x EP8 x DCP4")

    all_one_pool = {
        "instances": 1,
        "topology": "1x1x1",
        "chips_per_instance": 1,
        "tp": 1,
        "dp": 1,
        "ep": 1,
        "pp": 1,
        "pcp": 1,
        "kv_split": 1,
    }
    assert format_parallelism(all_one_pool) == "TP1"


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


def test_top_layouts_include_best_layout_by_tpot_bound() -> None:
    """Verify top_layouts includes best candidate by TPOT bound under tight latency."""
    raw_text_config = QWEN35_CFG["text_config"]
    assert isinstance(raw_text_config, dict)
    config = {
        **QWEN35_CFG,
        "text_config": {
            **raw_text_config,
            "quantization_config": {"quant_method": "fp8"},
        },
    }
    geometry = from_hf_config(config)
    dtypes = dtypes_from_hf_config(config)
    scenario = TrafficScenario(
        name="tight-latency-agent",
        isl=105000,
        osl=1000,
        prefix_hit_rate=0.95,
        accept_len=3.39,
        think_time_s=0.0,
        concurrencies=(8,),
        kv_working_set_turns=1.0,
    )
    options = SolveOptions(mode="mixed", max_chips=4)
    plan = solve_serving_plan(geometry, dtypes, CHIPS["tpu7x"], scenario, options)

    top_candidates = top_layouts(plan, 5)
    eligible_candidates = _filter_eligible(plan.candidates)
    assert eligible_candidates
    minimum_bound = min(point.bound_tpot_ms for point in eligible_candidates)
    roles_by_point = {
        point: role for point, role, _ in top_layout_roles(top_candidates)
    }
    best_bound_points = [
        point
        for point in top_candidates
        if point.bound_tpot_ms == pytest.approx(minimum_bound, rel=1e-9)
    ]
    assert best_bound_points
    assert any(
        roles_by_point[point]
        in {"max_throughput", "max_interactivity", "min_tpot_bound"}
        for point in best_bound_points
    )


def test_top_layouts_never_repeat_a_layout() -> None:
    """Verify top_layouts returns unique structural keys preserving prefix layouts."""
    raw_text_config = QWEN35_CFG["text_config"]
    assert isinstance(raw_text_config, dict)
    config = {
        **QWEN35_CFG,
        "text_config": {
            **raw_text_config,
            "quantization_config": {"quant_method": "fp8"},
        },
    }
    geometry = from_hf_config(config)
    dtypes = dtypes_from_hf_config(config)

    scenario_tight = TrafficScenario(
        name="tight-latency-agent",
        isl=105000,
        osl=1000,
        prefix_hit_rate=0.95,
        accept_len=3.39,
        think_time_s=0.0,
        concurrencies=(8,),
        kv_working_set_turns=1.0,
    )
    plan_tight = solve_serving_plan(
        geometry,
        dtypes,
        CHIPS["tpu7x"],
        scenario_tight,
        SolveOptions(mode="mixed", max_chips=4),
    )

    scenario_multi = TrafficScenario(
        name="multi-concurrency-agent",
        isl=105000,
        osl=1000,
        prefix_hit_rate=0.95,
        accept_len=3.39,
        think_time_s=0.0,
        concurrencies=(1, 8, 64),
        kv_working_set_turns=1.0,
    )
    # Chip budget of 4 provides 10 distinct eligible layouts (>= 6 required).
    plan_multi = solve_serving_plan(
        geometry,
        dtypes,
        CHIPS["tpu7x"],
        scenario_multi,
        SolveOptions(max_chips=4),
    )

    for evaluated_plan in (plan_tight, plan_multi):
        eligible = _filter_eligible(evaluated_plan.candidates)
        minimum_bound = min(point.bound_tpot_ms for point in eligible)
        lowest_bound_keys = {
            _point_struct_key(point)
            for point in eligible
            if point.bound_tpot_ms == minimum_bound
        }

        for candidate_count in range(1, 7):
            top_candidates = top_layouts(evaluated_plan, candidate_count)
            top_keys = [_point_struct_key(point) for point in top_candidates]

            assert len(top_candidates) <= candidate_count
            assert len(set(top_keys)) == len(top_keys)

            rank1_key = _point_struct_key(top_layouts(evaluated_plan, 1)[0])
            assert rank1_key in top_keys

            min_count_for_pair = 2
            min_count_for_bound = 3
            if candidate_count >= min_count_for_pair:
                top2_keys = {
                    _point_struct_key(point) for point in top_layouts(evaluated_plan, 2)
                }
                assert top2_keys.issubset(set(top_keys))

            if candidate_count >= min_count_for_bound:
                assert bool(lowest_bound_keys & set(top_keys))


def test_bound_best_step_changes_at_most_one_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify bound-best insertion changes at most one entry in top layouts."""
    raw_text_config = QWEN35_CFG["text_config"]
    assert isinstance(raw_text_config, dict)
    config = {
        **QWEN35_CFG,
        "text_config": {
            **raw_text_config,
            "quantization_config": {"quant_method": "fp8"},
        },
    }
    geometry = from_hf_config(config)
    dtypes = dtypes_from_hf_config(config)
    scenario = TrafficScenario(
        name="multi-concurrency-agent",
        isl=105000,
        osl=1000,
        prefix_hit_rate=0.95,
        accept_len=3.39,
        think_time_s=0.0,
        concurrencies=(1, 8, 64),
        kv_working_set_turns=1.0,
    )
    plan = solve_serving_plan(
        geometry, dtypes, CHIPS["tpu7x"], scenario, SolveOptions(max_chips=4)
    )

    for relative_tolerance in (0.0, 0.10):
        for candidate_count in range(3, 7):
            top_with_bound_best = top_layouts(
                plan, candidate_count, near_tie_rel=relative_tolerance
            )
            keys_with_bound_best = {
                _point_struct_key(point) for point in top_with_bound_best
            }

            monkeypatch.setattr(rf_emit, "MIN_TOP_LAYOUTS_FOR_BOUND_BEST", 10**6)
            top_without_bound_best = top_layouts(
                plan, candidate_count, near_tie_rel=relative_tolerance
            )
            keys_without_bound_best = {
                _point_struct_key(point) for point in top_without_bound_best
            }
            monkeypatch.undo()

            keys_leaving = keys_without_bound_best - keys_with_bound_best
            keys_entering = keys_with_bound_best - keys_without_bound_best

            if relative_tolerance == 0.0:
                assert len(keys_leaving) <= 1
                assert len(keys_entering) <= 1
            else:
                assert len(keys_leaving) == 0
                assert len(keys_entering) <= 1


def test_parse_layout_rejects_malformed_text() -> None:
    """Verify parse_layout raises descriptive ValueError on malformed inputs."""
    with pytest.raises(ValueError, match="unknown parallelism axis in layout: 'FOO8'"):
        parse_layout("FOO8")

    with pytest.raises(ValueError, match="axis given twice in layout: 'TP8'"):
        parse_layout("TP8 x TP8")

    with pytest.raises(ValueError, match="layout needs a decode part: 'P: TP8'"):
        parse_layout("P: TP8")

    with pytest.raises(
        ValueError, match="parallelism degree must be at least 1: 'TP0'"
    ):
        parse_layout("TP0")

    with pytest.raises(ValueError, match="empty layout text"):
        parse_layout("")
