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
"""Unit tests for per-pass module byte, FLOP, and collective accounting."""

from __future__ import annotations

import pytest

import vllm_torchtpu.roofline as rf
from vllm_torchtpu.roofline import (
    Dtypes,
    Parallel,
    StepShape,
    experts_touched,
    from_hf_config,
    pass_costs,
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


def test_tp32_shared_expert_shard_width() -> None:
    synthetic_tp32_cfg = {
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
    geom = from_hf_config(synthetic_tp32_cfg)
    mods = {
        m.name: m
        for m in pass_costs(
            geom,
            Parallel(tp=32),
            StepShape(batch=1, rows_per_seq=1, context=1024),
            DOC_DTYPES,
        )
    }
    expected_gdn_w = 69 * (8192 * (768 + 512) + 8192 * 8 + 768 * 4 + 512 * 8192)
    expected_gdn_st = 2 * 69 * (4 * 128 * 128 * 4 + 768 * 3 * 2)
    assert mods["gdn"].hbm_bytes == pytest.approx(
        expected_gdn_w + expected_gdn_st, abs=1.0
    )
    ed1 = experts_touched(512, 10, 1)
    expected_moe_w = 92 * (ed1 * 3 * 8192 * 64 + 3 * 8192 * 64 + 8192 * 512 * 2)
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
        "linear_conv_kernel_dim": 4,
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


def test_uneven_pp8_and_gdn_recurrence_flops() -> None:
    geom = from_hf_config(QWEN35_CFG)
    m_dcp1 = {
        m.name: m
        for m in pass_costs(
            geom,
            Parallel(tp=8, kv_split=1, moe="ep"),
            StepShape(batch=1, rows_per_seq=1, context=8192),
            Dtypes(kv=1.0),
        )
    }
    m_dcp1_ctx0 = {
        m.name: m
        for m in pass_costs(
            geom,
            Parallel(tp=8, kv_split=1, moe="ep"),
            StepShape(batch=1, rows_per_seq=1, context=0),
            Dtypes(kv=1.0),
        )
    }
    kv_only_b = m_dcp1["gqa"].hbm_bytes - m_dcp1_ctx0["gqa"].hbm_bytes
    assert kv_only_b == pytest.approx(15.0 * 8192.0 * 512.0, abs=1.0)
    assert m_dcp1_ctx0["gdn"].bf16_flops > 45.0 * 4.0 * 8.0 * 128.0 * 128.0

    m_pp8 = {
        m.name: m
        for m in pass_costs(
            geom,
            Parallel(tp=1, pp=8, pcp=2),
            StepShape(
                batch=0, rows_per_seq=0, context=0, state_writes=0, prefill_tokens=8192
            ),
            Dtypes(kv=1.0),
        )
    }
    m_pp1 = {
        m.name: m
        for m in pass_costs(
            geom,
            Parallel(tp=1, pp=1, pcp=2),
            StepShape(
                batch=0, rows_per_seq=0, context=0, state_writes=0, prefill_tokens=8192
            ),
            Dtypes(kv=1.0),
        )
    }
    assert m_pp8["moe"].fp8_flops / m_pp1["moe"].fp8_flops == pytest.approx(
        8.0 / 60.0, rel=1e-6
    )

    assert rf.ep_busiest_experts(4, 2, 8, 8) == pytest.approx(1.0)
    m_ep2_tp8 = {
        m.name: m
        for m in pass_costs(
            geom,
            Parallel(tp=8, ep=2, moe="ep"),
            StepShape(batch=1, rows_per_seq=1, context=0),
            Dtypes(kv=1.0),
        )
    }
    m_ep8_tp8 = {
        m.name: m
        for m in pass_costs(
            geom,
            Parallel(tp=8, ep=8, moe="ep"),
            StepShape(batch=1, rows_per_seq=1, context=0),
            Dtypes(kv=1.0),
        )
    }
    assert m_ep2_tp8["moe"].hbm_bytes < m_ep8_tp8["moe"].hbm_bytes
    assert m_ep2_tp8["moe"].fp8_flops == pytest.approx(
        m_ep8_tp8["moe"].fp8_flops, rel=1e-9
    )


def test_latent_moe_and_compressed_tensors_w4() -> None:
    latent_w4_cfg = {
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_routed_experts": 384,
        "num_selected_experts": 8,
        "latent_moe_dim": 3584,
        "n_shared_experts": 1,
        "shared_expert_intermediate_size": 2048,
        "vocab_size": 163840,
        "dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 4,
                        "strategy": "group",
                        "group_size": 32,
                        "scale_bytes": 2,
                    },
                }
            },
            "ignore": ["lm_head", "self_attn"],
        },
    }
    g = from_hf_config(latent_w4_cfg)
    assert g.top_k == 8
    assert g.moe_inter == 3584
    dt = rf.dtypes_from_hf_config(latent_w4_cfg)
    assert dt.w_expert == pytest.approx(0.5625, abs=1e-6)
    pre_shape = StepShape(
        batch=0,
        rows_per_seq=0,
        context=4096,
        state_writes=0,
        prefill_tokens=4096,
        prefix_hit_rate=0.5,
    )
    mla_p1 = {
        m.name: m
        for m in pass_costs(g, Parallel(tp=8, pcp=1), pre_shape, Dtypes(kv=1.0))
    }["mla"]
    mla_p2 = {
        m.name: m
        for m in pass_costs(g, Parallel(tp=8, pcp=2), pre_shape, Dtypes(kv=1.0))
    }["mla"]
    assert mla_p2.bf16_flops == pytest.approx(0.5 * mla_p1.bf16_flops, rel=1e-6)


def test_pcp_moe_global_rows_and_flops_conservation() -> None:
    moe_cfg = {
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 0,
        "vocab_size": 32000,
    }
    geom = from_hf_config(moe_cfg)
    dt = Dtypes(w_expert=1.0, w_dense=2.0, router=1.0, act="a8")
    pre_shape = StepShape(
        batch=0,
        rows_per_seq=0,
        context=0,
        state_writes=0,
        prefill_tokens=16,
        prefix_hit_rate=0.0,
    )
    par_pcp2 = Parallel(tp=2, pcp=2, moe="ep", ep=4, dp=1)
    par_tp4 = Parallel(tp=4, pcp=1, moe="ep", ep=4, dp=1)
    moe_pcp2 = {m.name: m for m in pass_costs(geom, par_pcp2, pre_shape, dt)}["moe"]
    moe_tp4 = {m.name: m for m in pass_costs(geom, par_tp4, pre_shape, dt)}["moe"]
    assert moe_pcp2.hbm_bytes == pytest.approx(moe_tp4.hbm_bytes, rel=1e-6)
    router_elems = float(4 * 2048 * 64)
    routed_fp8_pcp2 = moe_pcp2.fp8_flops - 2.0 * 8.0 * router_elems
    routed_fp8_tp4 = moe_tp4.fp8_flops - 2.0 * 16.0 * router_elems
    assert routed_fp8_pcp2 == pytest.approx(routed_fp8_tp4, rel=1e-6)

    mix_shape = StepShape(
        batch=4,
        rows_per_seq=1,
        context=512,
        state_writes=1,
        prefill_tokens=16,
        prefix_hit_rate=0.0,
    )
    par_mix = Parallel(tp=2, pcp=2, dp=2, moe="ep", ep=8, moe_dispatch="ag_ep")
    moe_mix = {m.name: m for m in pass_costs(geom, par_mix, mix_shape, dt)}["moe"]
    exp_ed = rf.ep_busiest_experts(64, 8, (4 + 8 * 2) * 2, 8)
    exp_w_bytes = exp_ed * float(4 * 3 * 2048 * 1024) * 1.0 + router_elems * 1.0
    assert moe_mix.hbm_bytes == pytest.approx(exp_w_bytes, rel=1e-6)
