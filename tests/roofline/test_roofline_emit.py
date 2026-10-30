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
from pathlib import Path
from typing import Literal

import pytest

from dataclasses import replace as dc_replace
from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    ParetoPoint,
    PoolMeshPlan,
    SOLVE_SCHEMA,
    ServingPlan,
    TrafficScenario,
    cli,
    emit_ceiling_rows,
    emit_launch_args,
    from_hf_config,
    pool_cli_args,
    solve_serving_plan,
    top_layouts,
)
from vllm_torchtpu.roofline.emit import _pool_key


def _toy_moe_config_dict() -> dict:
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
    *,
    role: Literal["mixed", "prefill", "decode"] = "decode",
    instances: int = 1,
    topology: str = "2x2x2",
    chips: int = 8,
    tp: int = 2,
    dp: int = 4,
    ep: int = 8,
    pp: int = 1,
    pcp: int = 1,
    kv_split: int = 1,
) -> PoolMeshPlan:
    return PoolMeshPlan(
        role=role,
        instances=instances,
        topology=topology,
        chips_per_instance=chips,
        tp=tp,
        dp=dp,
        ep=ep,
        pp=pp,
        pcp=pcp,
        kv_split=kv_split,
        mesh_axes=(2, 2, 2, 2),
        hbm_used_gb=42.0,
        hbm_limit_gb=83.25,
    )


def test_pool_cli_args_vllm_layouts() -> None:
    dp_ep_pool = _sample_pool(tp=2, dp=4, ep=8, pp=2)
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

    cp_pool = _sample_pool(tp=8, dp=1, ep=1, pcp=2, kv_split=4)
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
    mixed_pool = _sample_pool(role="mixed", topology="2x2x1", chips=4, tp=8, dp=1, ep=8)
    mixed_pt = ParetoPoint(
        mode="mixed",
        concurrency=16,
        total_chips=4,
        prefill_pool=None,
        decode_pool=mixed_pool,
        p90_ttft_ms=180.0,
        p90_tpot_ms=12.0,
        p90_interactivity_tok_s=83.3,
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
        instances=1,
        topology="2x2x1",
        chips=4,
        tp=4,
        dp=1,
        ep=1,
        pcp=2,
    )
    decode_pool = _sample_pool(
        role="decode", instances=2, topology="2x2x2", chips=8, tp=2, dp=4, ep=8
    )
    disagg_pt = ParetoPoint(
        mode="disagg",
        concurrency=64,
        total_chips=20,
        prefill_pool=prefill_pool,
        decode_pool=decode_pool,
        p90_ttft_ms=140.0,
        p90_tpot_ms=9.5,
        p90_interactivity_tok_s=105.2,
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
    assert len(disagg_out["commands"]) == 2
    assert disagg_out["commands"][0].startswith("# prefill (1x 2x2x1): vllm serve")
    assert disagg_out["commands"][1].startswith("# decode (2x 2x2x2): vllm serve")


def test_emit_launch_args_infeasible_plan_returns_empty_and_cli_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
        mode="mixed",
        max_chips=4,
    )
    rows = emit_ceiling_rows(splan)
    assert len(rows) >= 1
    assert list(rows[0].keys())[:5] == [
        "p90_interactivity_tok_s",
        "token_throughput_per_chip",
        "output_throughput_per_chip",
        "p90_ttft_ms",
        "p90_tpot_ms",
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
    assert "p90_interactivity_tok_s" in out_ceil[0]


def test_top_layouts_distinct_and_led_by_recommended():
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
    scenario = dc_replace(
        TrafficScenario(
            name="chat-8k1k",
            isl=8192,
            osl=1024,
            prefix_hit_rate=0.0,
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(1, 2, 4),
            kv_working_set_turns=1.0,
        ),
    )
    plan = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        max_chips=4,
        mode="mixed",
    )
    top = top_layouts(plan, k=3)
    assert 1 <= len(top) <= 3
    assert plan.recommended is not None
    assert top[0] == plan.recommended

    keys = set()
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


def test_top_layouts_empty_without_slo_met():
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
    plan = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        max_chips=4,
        mode="mixed",
    )
    top = top_layouts(plan, k=3)
    assert top == ()


def test_solve_json_v1_envelope(tmp_path, monkeypatch, capsys):
    """solve --json outputs schema v1 with provenance, search stats, and top_k launch."""
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
    cfg_path.write_text(json.dumps(cfg))

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
