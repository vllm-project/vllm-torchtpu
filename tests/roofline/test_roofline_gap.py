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
"""Unit tests for measured-vs-floor gap decomposition and CLI gap subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import vllm_torchtpu.roofline as rf
from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    ModelGeometry,
    Parallel,
    StepShape,
    compare_measured,
    step_floor,
)
from vllm_torchtpu.roofline.cli import main as cli_main

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


def test_compare_measured_qwen35_tp8_decomposition() -> None:
    sf = step_floor(
        QWEN35_GEOM,
        CHIPS["tpu7x"],
        Parallel(tp=8, kv_split=4),
        StepShape(batch=1, rows_per_seq=1, context=1024),
        Dtypes(kv=1.0, state=2.0, lm_head=1.0),
    )
    measured = {
        "gqa": 0.080,
        "gdn": 0.250,
        "moe": 0.500,
        "lm_head": 0.050,
        "step_ms": 1.000,
    }
    rep = compare_measured(sf, CHIPS["tpu7x"], measured)
    assert rep.status == "OK"
    assert rep.unattributed_ms == pytest.approx(0.120, rel=1e-4)
    assert rep.total_gap_ms == pytest.approx(1.000 - sf.hard_floor_s * 1e3, rel=1e-4)
    shares = sum(m.gap_share for m in rep.modules)
    assert 0.0 < shares < 1.0


def test_compare_measured_flags_sub_floor_model_bug() -> None:
    sf = step_floor(
        QWEN35_GEOM,
        CHIPS["tpu7x"],
        Parallel(tp=8, kv_split=4),
        StepShape(batch=1, rows_per_seq=1, context=1024),
        Dtypes(kv=1.0, state=2.0),
    )
    rep = compare_measured(sf, CHIPS["tpu7x"], {"moe": 0.100, "step_ms": 0.900})
    moe_gap = next(m for m in rep.modules if m.name == "moe")
    assert moe_gap.status == "SUB_FLOOR_MODEL_BUG"
    assert rep.status == "SUB_FLOOR_MODEL_BUG"


def test_cli_gap_json_and_fail_on_sub_floor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "text_config": {
                    "hidden_size": 4096,
                    "num_hidden_layers": 60,
                    "layer_types": ["linear_attention"] * 45 + ["full_attention"] * 15,
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
                }
            }
        )
    )
    meas_path = tmp_path / "measured.json"
    meas_path.write_text(json.dumps({"moe": 0.050, "step_ms": 0.200}))
    rc = cli_main(
        [
            "gap",
            "--config",
            str(cfg_path),
            "--measured",
            str(meas_path),
            "--tp",
            "8",
            "--kv-split",
            "4",
            "--fail-on-sub-floor",
            "--json",
        ]
    )
    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "SUB_FLOOR_MODEL_BUG"


def test_compare_measured_dense_mlp_aliases() -> None:
    dense_geom = ModelGeometry(
        hidden=4096,
        num_layers=4,
        layer_types=("full_attention",) * 4,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_output_gate=False,
        linear_k_heads=0,
        linear_v_heads=0,
        linear_k_dim=0,
        linear_v_dim=0,
        conv_kernel=4,
        num_experts=0,
        top_k=0,
        moe_inter=0,
        shared_inter=0,
        vocab=32000,
        mtp_layers=0,
        dense_inter=11008,
    )
    sf = step_floor(
        dense_geom,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        StepShape(batch=1, rows_per_seq=1, context=1024),
    )
    for alias in ("mlp", "dense_mlp", "ffn"):
        rep = compare_measured(
            sf,
            CHIPS["tpu7x"],
            {"gqa": 0.050, alias: 0.100, "lm_head": 0.050, "step_ms": 0.250},
        )
        mlp_gap = next(m for m in rep.modules if m.name == "mlp")
        assert mlp_gap.status == "OK"
        assert mlp_gap.measured_ms == pytest.approx(0.100)


def test_gap_reports_collective_row() -> None:
    sf = step_floor(
        QWEN35_GEOM,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        StepShape(batch=1, rows_per_seq=4, context=1024),
        Dtypes(),
        collective_latency_s=15.4e-6,
    )
    rep = compare_measured(
        sf,
        CHIPS["tpu7x"],
        {"all-reduce": 1.200, "ar_sum": 0.900, "step_ms": 3.500},
        collective_latency_s=15.4e-6,
    )
    coll = next(m for m in rep.modules if m.name == "collectives")
    assert coll.floor_ms == pytest.approx(sf.chain_s * 1e3, rel=1e-9)
    assert coll.measured_ms == pytest.approx(2.100, rel=1e-9)
    assert coll.status == "OK"


def test_gap_lists_unmapped_keys() -> None:
    sf = step_floor(
        QWEN35_GEOM,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        StepShape(batch=1, rows_per_seq=1, context=1024),
    )
    rep = compare_measured(
        sf,
        CHIPS["tpu7x"],
        {"gqa": 0.100, "mystery_op": 0.050, "host_wait": 0.020, "step_ms": 1.500},
    )
    assert rep.unmapped_keys == ("mystery_op", "host_wait")


def test_strict_sub_floor_uses_min_experts() -> None:
    sf = step_floor(
        QWEN35_GEOM,
        CHIPS["tpu7x"],
        Parallel(tp=8, moe="ep"),
        StepShape(batch=4, rows_per_seq=1, context=1024),
        Dtypes(),
    )
    moe_mod = next(m for m in sf.modules if m.name == "moe")
    avg_floor_ms = (moe_mod.hbm_bytes / CHIPS["tpu7x"].hbm_bw) * 1e3
    min_floor_ms = (moe_mod.strict_hbm_bytes / CHIPS["tpu7x"].hbm_bw) * 1e3
    assert min_floor_ms < avg_floor_ms
    mid_ms = 0.5 * (min_floor_ms + avg_floor_ms)
    rep_ok = compare_measured(sf, CHIPS["tpu7x"], {"moe": mid_ms, "step_ms": 2.000})
    moe_ok = next(m for m in rep_ok.modules if m.name == "moe")
    assert moe_ok.status == "OK"

    rep_bug = compare_measured(
        sf, CHIPS["tpu7x"], {"moe": min_floor_ms * 0.95, "step_ms": 2.000}
    )
    moe_bug = next(m for m in rep_bug.modules if m.name == "moe")
    assert moe_bug.status == "SUB_FLOOR_MODEL_BUG"


def test_plan_from_serve_args_stock_recipe() -> None:
    argv = [
        "--tensor-parallel-size=8",
        "--data-parallel-size=1",
        "--quantization=fp8",
        "--max-num-seqs=8",
        "--kv-cache-dtype=fp8",
        "--enable-expert-parallel",
        "--block-size",
        "256",
    ]
    env = {
        "USE_MOE_FUSED_EP_KERNEL": "0",
        "TPU_MOE_SKIP_PADDED_TOKENS": "1",
    }
    par, dt = rf.plan_from_serve_args(argv, env)
    assert par == Parallel(tp=8, moe="ep")
    assert dt.kv == 1.0


def test_gap_cli_rejects_conflicting_plan_flags(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "text_config": {
                    "hidden_size": 4096,
                    "num_hidden_layers": 60,
                    "layer_types": ["linear_attention"] * 45 + ["full_attention"] * 15,
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
                }
            }
        )
    )
    meas_path = tmp_path / "measured.json"
    meas_path.write_text(json.dumps({"moe": 0.800, "step_ms": 1.800}))
    serve_path = tmp_path / "serve.json"
    serve_path.write_text(
        json.dumps(
            {
                "argv": [
                    "--tensor-parallel-size=8",
                    "--kv-cache-dtype=fp8",
                    "--enable-expert-parallel",
                ],
                "env": {"USE_MOE_FUSED_EP_KERNEL": "0"},
            }
        )
    )
    rc = cli_main(
        [
            "gap",
            "--config",
            str(cfg_path),
            "--measured",
            str(meas_path),
            "--serve-args",
            str(serve_path),
            "--moe",
            "tp_slice",
            "--json",
        ]
    )
    assert rc == 2


def test_plan_from_serve_args_kv_dtype_absent_is_model_dtype() -> None:
    par, dt = rf.plan_from_serve_args(["--tensor-parallel-size=8"], {})
    assert par == Parallel(tp=8)
    assert dt.kv == 2.0


def test_gap_reports_fixed_cost_rows() -> None:
    sf = step_floor(
        QWEN35_GEOM,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        StepShape(batch=1, rows_per_seq=1, context=1024),
    )
    base = compare_measured(sf, CHIPS["tpu7x"], {"moe": 0.500, "step_ms": 6.000})
    rep = compare_measured(
        sf,
        CHIPS["tpu7x"],
        {"moe": 0.500, "scale_relayout": 1.90, "logits_ag": 1.28, "step_ms": 6.000},
    )
    rows = {m.name: m for m in rep.modules}
    for name, ms in (("scale_relayout", 1.90), ("logits_ag", 1.28)):
        assert rows[name].floor_ms == 0.0
        assert rows[name].measured_ms == pytest.approx(ms)
    assert rep.unmapped_keys == ()
    assert base.unattributed_ms - rep.unattributed_ms == pytest.approx(1.90 + 1.28)


def test_gap_without_step_time_keeps_unmapped_time_unattributed() -> None:
    sf = step_floor(
        QWEN35_GEOM,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        StepShape(batch=1, rows_per_seq=1, context=1024),
        Dtypes(),
    )
    rep = compare_measured(sf, CHIPS["tpu7x"], {"moe": 0.500, "mystery_op": 5.000})
    assert rep.unmapped_keys == ("mystery_op",)
    assert rep.measured_step_ms == pytest.approx(5.500)
    assert rep.unattributed_ms == pytest.approx(5.000)


def test_plan_from_serve_args_reads_speculative_window() -> None:
    spec = '{"method": "mtp", "num_speculative_tokens": 3}'
    assert rf.recipe.spec_window_from_serve_args(["--speculative-config", spec]) == 3
    assert rf.recipe.spec_window_from_serve_args([f"--speculative-config={spec}"]) == 3
    assert rf.recipe.spec_window_from_serve_args(["--tensor-parallel-size=8"]) == 0

    # vLLM --enable-expert-parallel derives ep = tp * dp when dp > 1.
    par_ep32, _ = rf.plan_from_serve_args(
        [
            "--tensor-parallel-size=8",
            "--data-parallel-size=4",
            "--enable-expert-parallel",
        ],
        {},
    )
    assert (
        par_ep32.moe == "ep" and par_ep32.ep == 32 and par_ep32.moe_dispatch == "ag_ep"
    )

    # SGLang --enable-dp-attention divides --tp-size by --dp-size.
    rt_par, rt_dt = rf.plan_from_serve_args(
        [
            "--tp-size",
            "8",
            "--dp-size",
            "2",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--enable-dp-attention",
            "--ep-size",
            "8",
        ],
        {},
    )
    assert rt_par.tp == 4 and rt_par.dp == 2 and rt_par.ep == 8
    assert rt_par.moe == "ep" and rt_par.moe_dispatch == "ag_ep"
    assert rt_dt.kv == pytest.approx(1.0)

    # --kv-cache-dtype auto resolves to model torch_dtype (2.0 B bf16), fp8 to 1.0 B.
    par_auto, dt_auto = rf.plan_from_serve_args(
        ["--tensor-parallel-size=8", "--kv-cache-dtype=auto"], {}
    )
    assert par_auto.tp == 8
    assert dt_auto.kv == pytest.approx(2.0)

    _, dt_fp8 = rf.plan_from_serve_args(
        ["--tensor-parallel-size=8", "--kv-cache-dtype=fp8"], {}
    )
    assert dt_fp8.kv == pytest.approx(1.0)

    par_sgl, dt_sgl = rf.plan_from_serve_args(
        [
            "--tp-size",
            "8",
            "--ep-size",
            "8",
            "--dp-size",
            "2",
            "--enable-ep-moe",
            "--kv-cache-dtype",
            "auto",
        ],
        {"SGLANG_FUSED_RS": "1"},
    )
    assert par_sgl.tp == 8 and par_sgl.ep == 8 and par_sgl.dp == 2
    assert par_sgl.moe == "ep" and par_sgl.moe_dispatch == "fused_rs"
    assert dt_sgl.kv == pytest.approx(2.0)

    # 10-family gap aliases map cleanly without landing in unmapped_keys.
    ten_cfg = {
        "hidden_size": 2048,
        "num_hidden_layers": 6,
        "layer_types": ["gqa", "mla", "dsa", "swa", "gdn", "kda"],
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "q_lora_rank": 1024,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "index_n_heads": 8,
        "index_head_dim": 64,
        "index_topk": 512,
        "sliding_window": 256,
        "linear_num_key_heads": 8,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "n_routed_experts": 64,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 512,
        "intermediate_size": 4096,
        "first_k_dense_replace": 1,
        "num_nextn_predict_layers": 1,
        "per_layer_input_dim": 256,
        "vocab_size": 32000,
    }
    ten_geom = rf.from_hf_config(ten_cfg)
    sf_ten = rf.spec_cycle_floor(
        ten_geom,
        CHIPS["tpu7x"],
        rf.Parallel(tp=4, moe="ep"),
        batch=2,
        context=2048,
        k=1,
    )
    rep_ten = rf.compare_measured(
        sf_ten,
        CHIPS["tpu7x"],
        {
            "mla_attention": 0.20,
            "sparse_mla": 0.20,
            "sliding_attention": 0.15,
            "kda_attention": 0.25,
            "per_layer_embedding": 0.10,
            "mtp": 0.15,
            "step_ms": 2.50,
        },
    )
    assert rep_ten.unmapped_keys == ()
    assert rep_ten.unattributed_ms > 0.0


def test_memory_utilization_from_serve_args() -> None:
    assert rf.memory_utilization_from_serve_args([]) == 0.90
    assert (
        rf.memory_utilization_from_serve_args(["--gpu-memory-utilization=0.92"]) == 0.92
    )
    assert (
        rf.memory_utilization_from_serve_args(["--gpu-memory-utilization", "0.85"])
        == 0.85
    )
    assert rf.memory_utilization_from_serve_args(["--mem-fraction-static=0.88"]) == 0.88
    assert (
        rf.memory_utilization_from_serve_args(["--mem-fraction-static", "0.80"]) == 0.80
    )
    with pytest.raises(ValueError):
        rf.memory_utilization_from_serve_args(["--gpu-memory-utilization=1.5"])
