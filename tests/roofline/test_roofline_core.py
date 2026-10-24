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
"""Unit tests for vllm_torchtpu.roofline core accounting and sub-tile rules."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import vllm_torchtpu.roofline as rf
from vllm_torchtpu.roofline import (
    CHIPS,
    ChipSpec,
    Dtypes,
    Parallel,
    StepShape,
    experts_touched,
    from_hf_config,
    pass_costs,
    plan,
    spec_cycle_floor,
    step_floor,
)

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
    }
}

DOC_DTYPES = Dtypes(kv=2.0, w_aux=1.0)


def test_qwen35_batch1_bytes_match_worked_example() -> None:
    geom = from_hf_config(QWEN35_CFG)
    mods = {
        m.name: m
        for m in pass_costs(
            geom,
            Parallel(tp=8),
            StepShape(batch=1, rows_per_seq=1, context=90036),
            DOC_DTYPES,
        )
    }
    assert mods["gqa"].hbm_bytes == pytest.approx(220_200_960 + 1_382_952_960, abs=1.0)
    assert mods["gdn"].hbm_bytes == pytest.approx(663_828_480 + 48_015_360, abs=1.0)
    assert mods["moe"].hbm_bytes == pytest.approx(
        943_718_400 + 94_371_840 + 251_658_240, abs=1.0
    )
    assert mods["lm_head"].hbm_bytes == pytest.approx(254_279_680, abs=1.0)
    total = sum(m.hbm_bytes for m in mods.values())
    assert total == pytest.approx(3_859_025_920, abs=1.0)


def test_fused_path_splits_kv_in_group() -> None:
    geom = from_hf_config(QWEN35_CFG)
    mods = {
        m.name: m
        for m in pass_costs(
            geom,
            Parallel(tp=8, kv_split=4),
            StepShape(batch=1, rows_per_seq=1, context=90036),
            DOC_DTYPES,
        )
    }
    assert mods["gqa"].hbm_bytes == pytest.approx(220_200_960 + 345_738_240, abs=1.0)
    total = sum(m.hbm_bytes for m in mods.values())
    assert total == pytest.approx(2_821_811_200, abs=1.0)


def test_hard_floor_is_max_of_resources() -> None:
    geom = from_hf_config(QWEN35_CFG)
    sf = step_floor(
        geom,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        StepShape(batch=1, rows_per_seq=1, context=90036),
        DOC_DTYPES,
    )
    assert sf.hard_floor_s == max(sf.hbm_s, sf.mm_s, sf.ici_s)
    assert sf.binding == "hbm"
    assert sf.hbm_s == pytest.approx(3_859_025_920 / 3.7e12, rel=1e-12)
    assert sf.mm_s == pytest.approx(6.8988973870882745e-06, rel=1e-9)

    tiny_mm = ChipSpec("tiny_mm", 100, 1e18, 1e6, 1e6, 100)
    sf_mm = step_floor(
        geom, tiny_mm, Parallel(tp=8), StepShape(batch=1, context=10), DOC_DTYPES
    )
    assert sf_mm.binding == "matmul"


def test_chip_table_matches_jax_tpu_info() -> None:
    jti = pytest.importorskip("jax._src.tpu_info")

    info = jti.get_tpu_info_for_chip(jti.ChipVersion.TPU_7X, 1)
    c = CHIPS["tpu7x"]
    assert c.hbm_bytes == info.hbm_capacity_bytes
    assert c.hbm_bw == pytest.approx(info.mem_bw_bytes_per_second)
    assert c.bf16_flops == pytest.approx(info.bf16_ops_per_second)
    assert c.fp8_flops == pytest.approx(info.fp8_ops_per_second)
    assert c.vmem_bytes == info.vmem_capacity_bytes


def test_collective_payload_and_count() -> None:
    geom = from_hf_config(QWEN35_CFG)
    mods1 = pass_costs(
        geom, Parallel(tp=8), StepShape(batch=1, rows_per_seq=1), Dtypes()
    )
    assert sum(m.n_collectives for m in mods1) == 122
    assert sum(m.ici_bytes for m in mods1) == pytest.approx(6_319_872.0)
    mods4 = pass_costs(
        geom, Parallel(tp=8), StepShape(batch=1, rows_per_seq=4), Dtypes()
    )
    assert sum(m.ici_bytes for m in mods4) == pytest.approx(25_279_488.0)
    mods_tp1 = pass_costs(
        geom, Parallel(tp=1), StepShape(batch=1, rows_per_seq=1), Dtypes()
    )
    assert sum(m.n_collectives for m in mods_tp1) == 0
    assert sum(m.ici_bytes for m in mods_tp1) == 0.0


def test_spec_rows_and_tokens_per_step() -> None:
    geom = from_hf_config(QWEN35_CFG)
    rows = plan(
        geom,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        DOC_DTYPES,
        context=90036,
        concurrencies=[1, 4],
        spec_window=3,
        acceptance_length=3.39,
    )
    assert [r.rows for r in rows] == [4, 16]
    assert rows[0].tpot_floor_ms == pytest.approx(
        rows[0].hard_floor_ms / 3.39, rel=1e-12
    )
    assert experts_touched(512, 10, 4) == pytest.approx(38.84330928325653, rel=1e-12)
    cyc = spec_cycle_floor(
        geom,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        batch=1,
        context=90036,
        k=3,
        dt=DOC_DTYPES,
    )
    assert sum(m.hbm_bytes for m in cyc.modules) == pytest.approx(
        7_813_585_288.75, abs=1.0
    )


def test_core_has_no_engine_imports() -> None:
    pkg_dir = Path(__file__).resolve().parents[2] / "src/vllm_torchtpu/roofline"
    banned = {
        "vllm",
        "torch",
        "torch_tpu",
        "jax",
        "jaxlib",
        "numpy",
        "vllm_torchtpu",
    }
    for py_file in sorted(pkg_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                assert node.module.split(".")[0] not in banned


def test_qwen38_tp32_subtile_replication() -> None:
    qwen38_cfg = {
        "text_config": {
            "hidden_size": 6144,
            "num_hidden_layers": 64,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
            "num_attention_heads": 64,
            "num_key_value_heads": 16,
            "head_dim": 128,
            "attn_output_gate": True,
            "linear_num_key_heads": 16,
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
    geom = from_hf_config(qwen38_cfg)
    mods = {
        m.name: m
        for m in pass_costs(
            geom,
            Parallel(tp=32),
            StepShape(batch=1, rows_per_seq=1, context=1024),
            DOC_DTYPES,
        )
    }
    expected_gdn_w = 48 * (6144 * (768 + 512) + 6144 * 8 + 768 * 4 + 512 * 6144)
    expected_gdn_st = 2 * 48 * (4 * 128 * 128 * 4 + 768 * 3 * 2)
    assert mods["gdn"].hbm_bytes == pytest.approx(
        expected_gdn_w + expected_gdn_st, abs=1.0
    )
    ed1 = experts_touched(512, 10, 1)
    expected_moe_w = 64 * (ed1 * 3 * 6144 * 64 + 3 * 6144 * 128 + 6144 * 512 * 2)
    assert mods["moe"].hbm_bytes == pytest.approx(expected_moe_w, abs=1.0)

    dense_att_cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
        "vocab_size": 32000,
    }
    d_geom = from_hf_config(dense_att_cfg)
    d_mods = {
        m.name: m
        for m in pass_costs(
            d_geom, Parallel(tp=8), StepShape(batch=1, rows_per_seq=1), Dtypes()
        )
    }
    assert "gdn" not in d_mods
    assert "mlp" in d_mods


def test_non_hybrid_models_omit_absent_blocks_and_interval_layer_types() -> None:
    interval_cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 8,
        "full_attention_interval": 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "intermediate_size": 11008,
        "vocab_size": 32000,
    }
    geom = from_hf_config(interval_cfg)
    assert geom.layer_types.count("full_attention") == 2
    assert geom.layer_types.count("linear_attention") == 6

    pure_att_geom = from_hf_config({**interval_cfg, "full_attention_interval": 0})
    pure_att_mods = {
        m.name: m
        for m in pass_costs(pure_att_geom, Parallel(tp=8), StepShape(), Dtypes())
    }
    assert "gdn" not in pure_att_mods and "gqa" in pure_att_mods
    pure_lin_geom = from_hf_config(
        {**interval_cfg, "layer_types": ["linear_attention"] * 8}
    )
    pure_lin_mods = {
        m.name: m
        for m in pass_costs(pure_lin_geom, Parallel(tp=8), StepShape(), Dtypes())
    }
    assert "gqa" not in pure_lin_mods and "gdn" in pure_lin_mods


def test_ep_busiest_rank_closed_form() -> None:
    expected = (
        (1, 2.979897),
        (4, 8.031072),
        (16, 22.468799),
        (64, 50.896276),
        (256, 63.999832),
    )
    prev = 0.0
    for rows, val in expected:
        got = rf.ep_busiest_experts(512, 10, rows, 8)
        assert got == pytest.approx(val, abs=1e-6)
        assert got > prev
        assert got <= 64.0
        prev = got
    assert rf.ep_busiest_experts(512, 10, 4, 1) == pytest.approx(
        experts_touched(512, 10, 4), rel=1e-12
    )
    assert rf.ep_busiest_experts(0, 10, 4, 8) == 0.0


def test_stock_plan_ep_fp8kv_batch1_bytes() -> None:
    geom = from_hf_config(QWEN35_CFG)
    mods = pass_costs(
        geom,
        Parallel(tp=8, moe="ep"),
        StepShape(batch=1, rows_per_seq=1, context=90036),
        Dtypes(),
    )
    by_name = {m.name: m for m in mods}
    assert by_name["moe"].hbm_bytes == pytest.approx(2_595_776_984.77, abs=1.0)
    assert sum(m.hbm_bytes for m in mods) == pytest.approx(4_476_803_544.77, abs=1.0)


def test_sequence_parallel_replicates_shared_expert() -> None:
    geom = from_hf_config(QWEN35_CFG)
    shape = StepShape(batch=1, rows_per_seq=1, context=90036)
    stock = {
        m.name: m for m in pass_costs(geom, Parallel(tp=8, moe="ep"), shape, Dtypes())
    }
    sp = {
        m.name: m
        for m in pass_costs(
            geom, Parallel(tp=8, moe="ep", residual="sp"), shape, Dtypes()
        )
    }
    assert sp["moe"].hbm_bytes - stock["moe"].hbm_bytes == pytest.approx(
        660_602_880.0, abs=1.0
    )


def test_library_defaults_batch1_bytes() -> None:
    geom = from_hf_config(QWEN35_CFG)
    mods = pass_costs(
        geom,
        Parallel(tp=8, kv_split=4),
        StepShape(batch=1, rows_per_seq=1, context=90036),
        Dtypes(state=2.0),
    )
    assert sum(m.hbm_bytes for m in mods) == pytest.approx(2_628_574_720.0, abs=1.0)


def test_ar_algorithm_wire_and_hops() -> None:
    geom = from_hf_config(QWEN35_CFG)
    shape = StepShape(batch=1, rows_per_seq=4)
    cases = (
        ("one_shot", 122, 29_492_736.0, 122),
        ("two_shot", 122, 8_676_864.0, 243),
        ("ring", 122, 8_676_864.0, 1701),
    )
    for algo, count, wire, hops in cases:
        mods = pass_costs(geom, Parallel(tp=8, ar_algo=algo), shape, Dtypes())
        colls = [c for m in mods for c in m.collectives]
        assert len(colls) == count
        assert sum(c.wire_bytes for c in colls) == pytest.approx(wire, abs=1.0)
        assert sum(c.hops for c in colls) == hops


def test_sequence_parallel_collective_count() -> None:
    geom = from_hf_config(QWEN35_CFG)
    mods = pass_costs(
        geom,
        Parallel(tp=8, moe="ep", residual="sp"),
        StepShape(batch=1, rows_per_seq=4),
        Dtypes(),
    )
    colls = [c for m in mods for c in m.collectives]
    assert len(colls) == 303
    assert sum(c.wire_bytes for c in colls) == pytest.approx(16_217_600.0, abs=1.0)
    assert sum(c.hops for c in colls) == 303


def test_a2a_push_wire_bytes() -> None:
    geom = from_hf_config(QWEN35_CFG)
    shape = StepShape(batch=1, rows_per_seq=4)
    rep = pass_costs(
        geom,
        Parallel(tp=8, moe="ep", moe_dispatch="a2a_push"),
        shape,
        Dtypes(),
    )
    rep_c = [c for m in rep for c in m.collectives]
    assert len(rep_c) == 242
    assert sum(c.wire_bytes for c in rep_c) == pytest.approx(21_751_296.0, abs=1.0)
    sp = pass_costs(
        geom,
        Parallel(tp=8, moe="ep", moe_dispatch="a2a_push", residual="sp"),
        shape,
        Dtypes(),
    )
    sp_c = [c for m in sp for c in m.collectives]
    assert len(sp_c) == 243
    assert sum(c.wire_bytes for c in sp_c) == pytest.approx(13_207_040.0, abs=1.0)


def test_kv_split_adds_merge_collectives() -> None:
    geom = from_hf_config(QWEN35_CFG)
    shape = StepShape(batch=1, rows_per_seq=4)
    ag_rs = pass_costs(
        geom, Parallel(tp=8, kv_split=4, kv_merge="ag_rs"), shape, Dtypes()
    )
    c1 = [c for m in ag_rs for c in m.collectives]
    assert len(c1) == 167
    assert sum(c.wire_bytes for c in c1) == pytest.approx(30_610_176.0, abs=1.0)
    xchg = pass_costs(
        geom, Parallel(tp=8, kv_split=4, kv_merge="exchange"), shape, Dtypes()
    )
    c2 = [c for m in xchg for c in m.collectives]
    assert len(c2) == 137
    assert sum(c.wire_bytes for c in c2) == pytest.approx(30_967_296.0, abs=1.0)


def test_chain_term_binds_with_latency() -> None:
    geom = from_hf_config(QWEN35_CFG)
    shape = StepShape(batch=1, rows_per_seq=4, context=0)
    sf0 = step_floor(geom, CHIPS["tpu7x"], Parallel(tp=8), shape, Dtypes(), 0.0)
    assert sf0.chain_s == pytest.approx(9.027743733e-05, rel=1e-6)
    assert sf0.binding == "hbm"
    sf1 = step_floor(geom, CHIPS["tpu7x"], Parallel(tp=8), shape, Dtypes(), 15.4e-6)
    assert sf1.chain_s == pytest.approx(1.969077437e-03, rel=1e-6)
    assert sf1.binding == "chain"
    assert sf1.hard_floor_s == pytest.approx(sf1.chain_s, rel=1e-12)


def test_a16_routes_fp8_weights_to_bf16_peak() -> None:
    geom = from_hf_config(QWEN35_CFG)
    shape = StepShape(batch=1, rows_per_seq=1, context=0)
    m_a8 = pass_costs(geom, Parallel(tp=8), shape, Dtypes(act="a8"))
    m_a16 = pass_costs(geom, Parallel(tp=8), shape, Dtypes(act="a16"))
    assert sum(m.fp8_flops for m in m_a8) > 0.0
    assert sum(m.fp8_flops for m in m_a16) == 0.0
    tot_a8 = sum(m.fp8_flops + m.bf16_flops for m in m_a8)
    tot_a16 = sum(m.fp8_flops + m.bf16_flops for m in m_a16)
    assert tot_a16 == pytest.approx(tot_a8, rel=1e-12)


def test_invalid_plans_raise() -> None:
    geom = from_hf_config(QWEN35_CFG)
    shape = StepShape(batch=1, rows_per_seq=1)
    for bad in (
        Parallel(tp=8, kv_split=3),
        Parallel(tp=8, moe="tp_slice", residual="sp"),
        Parallel(tp=8, moe="tp_slice", moe_dispatch="a2a_push"),
        Parallel(tp=1, moe="ep", residual="sp"),
    ):
        with pytest.raises(ValueError):
            pass_costs(geom, bad, shape, Dtypes())
