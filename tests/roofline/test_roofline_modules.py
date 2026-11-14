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

import re
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from vllm_torchtpu.roofline.modules import ModuleCost

import vllm_torchtpu.roofline as rf
from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    Parallel,
    SeqKvLayout,
    StepShape,
    dtypes_from_hf_config,
    from_hf_config,
    pass_costs,
)
from vllm_torchtpu.roofline.constants import (
    DEFAULT_VPU_TO_BF16_RATIO,
    VECTOR_MAX_ELEMENTS_PER_WORD,
    VECTOR_SUB_BYTE_ELEMENTS_PER_WORD,
)
from vllm_torchtpu.roofline.geometry import layer_ffn_kinds
from vllm_torchtpu.roofline.memory import compute_w_ffn
from vllm_torchtpu.roofline.modules import (
    ep_busiest_experts,
    experts_touched,
    kv_bytes_per_token_per_layer,
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
    """Verify batch-1 HBM byte footprint matches worked example for Qwen3.5."""
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
    # GQA weights: 6 layers * (hidden*q_out + 2*hidden*kvh*hd + nq*hd*hidden)
    # * w_dense + kv_tokens * b_kv_phys
    assert mods["gqa"].hbm_bytes == pytest.approx(220_200_960 + 1_382_952_960, abs=1.0)
    # GDN weights: 42 layers * (q_dim + k_dim + v_dim + beta) * hidden * w_dense
    # + state bytes
    assert mods["gdn"].hbm_bytes == pytest.approx(663_828_480 + 48_015_360, abs=1.0)
    # MoE weights: gate + routed experts + shared expert
    assert mods["moe"].hbm_bytes == pytest.approx(
        943_718_400 + 94_371_840 + 251_658_240, abs=1.0
    )
    # LM head weights: vocab_size * hidden * w_dense
    assert mods["lm_head"].hbm_bytes == pytest.approx(254_279_680, abs=1.0)
    total = sum(m.hbm_bytes for m in mods.values())
    # Total worked example HBM bytes for batch 1
    assert total == pytest.approx(3_859_025_920, abs=1.0)


def test_fused_path_splits_kv_in_group() -> None:
    """Verify KV cache bytes reduction when KV split is enabled."""
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
    # Fused path splits KV cache across groups
    assert mods["gqa"].hbm_bytes == pytest.approx(220_200_960 + 345_738_240, abs=1.0)
    total = sum(m.hbm_bytes for m in mods.values())
    # Fused path total batch 1 HBM bytes
    assert total == pytest.approx(2_821_811_200, abs=1.0)


def test_collective_payload_and_count() -> None:
    """Verify collective communication count and wire payload totals."""
    expected_batch1_collectives = 122
    geom = from_hf_config(QWEN35_CFG)
    mods1 = pass_costs(
        geom, Parallel(tp=8), StepShape(batch=1, rows_per_seq=1), Dtypes()
    )
    assert sum(m.n_collectives for m in mods1) == expected_batch1_collectives
    # Expected batch 1 collective payload wire bytes: 122 colls * rows * hidden * 2
    assert sum(m.ici_bytes for m in mods1) == pytest.approx(6_319_872.0)
    mods4 = pass_costs(
        geom, Parallel(tp=8), StepShape(batch=1, rows_per_seq=4), Dtypes()
    )
    # Expected batch 4 collective payload wire bytes: 4 * batch 1 payload
    assert sum(m.ici_bytes for m in mods4) == pytest.approx(25_279_488.0)
    mods_tp1 = pass_costs(
        geom, Parallel(tp=1), StepShape(batch=1, rows_per_seq=1), Dtypes()
    )
    assert sum(m.n_collectives for m in mods_tp1) == 0
    assert sum(m.ici_bytes for m in mods_tp1) == 0.0


def test_tp32_shared_expert_shard_width() -> None:
    """Verify shared expert sharding and state bytes under TP32."""
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
    """Verify layer type counts and absent block filtering for non-hybrid models."""
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
    expected_full_blocks = 2
    expected_linear_blocks = 6
    geom = from_hf_config(interval_cfg)
    assert geom.layer_types.count("full_attention") == expected_full_blocks
    assert geom.layer_types.count("linear_attention") == expected_linear_blocks

    pure_att_geom = from_hf_config({**interval_cfg, "full_attention_interval": 0})
    pure_att_mods = {
        m.name: m
        for m in pass_costs(pure_att_geom, Parallel(tp=8), StepShape(), Dtypes())
    }
    assert "gdn" not in pure_att_mods
    assert "gqa" in pure_att_mods
    pure_lin_geom = from_hf_config(
        {**interval_cfg, "layer_types": ["linear_attention"] * 8}
    )
    pure_lin_mods = {
        m.name: m
        for m in pass_costs(pure_lin_geom, Parallel(tp=8), StepShape(), Dtypes())
    }
    assert "gqa" not in pure_lin_mods
    assert "gdn" in pure_lin_mods


def test_ep_busiest_rank_closed_form() -> None:
    """Verify closed-form expected busiest rank values under uniform EP routing."""
    experts_per_rank = 64.0
    expected = (
        (1, 2.979897),
        (4, 8.031072),
        (16, 22.468799),
        (64, 50.896276),
        (256, 63.999832),
    )
    prev = 0.0
    for rows, val in expected:
        got = ep_busiest_experts(512, 10, rows, 8)
        assert got == pytest.approx(val, abs=1e-6)
        assert got > prev
        assert got <= experts_per_rank
        prev = got
    assert ep_busiest_experts(512, 10, 4, 1) == pytest.approx(
        experts_touched(512, 10, 4), rel=1e-12
    )
    assert ep_busiest_experts(0, 10, 4, 8) == 0.0


def test_stock_plan_ep_fp8kv_batch1_bytes() -> None:
    """Verify batch-1 HBM bytes for standard EP plan with FP8 KV cache."""
    geom = from_hf_config(QWEN35_CFG)
    mods = pass_costs(
        geom,
        Parallel(tp=8, moe="ep"),
        StepShape(batch=1, rows_per_seq=1, context=90036),
        Dtypes(),
    )
    by_name = {m.name: m for m in mods}
    # EP batch 1 MoE weights in bytes
    assert by_name["moe"].hbm_bytes == pytest.approx(2_595_776_984.77, abs=1.0)
    # EP batch 1 total HBM bytes
    assert sum(m.hbm_bytes for m in mods) == pytest.approx(5_168_280_024.77, abs=1.0)
    assert by_name["gqa"].hbm_bytes_min is not None
    logical_hbm = sum(m.hbm_bytes for m in mods) - (
        by_name["gqa"].hbm_bytes - by_name["gqa"].hbm_bytes_min
    )
    # Logical HBM excluding physical KV padding
    assert logical_hbm == pytest.approx(4_476_803_544.77, abs=1.0)


def test_sequence_parallel_replicates_shared_expert() -> None:
    """Verify sequence parallel replicates shared expert weights."""
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
    # SP shared expert replication delta in bytes
    assert sp["moe"].hbm_bytes - stock["moe"].hbm_bytes == pytest.approx(
        660_602_880.0, abs=1.0
    )


def test_library_defaults_batch1_bytes() -> None:
    """Verify total HBM bytes with library default parameters."""
    geom = from_hf_config(QWEN35_CFG)
    mods = pass_costs(
        geom,
        Parallel(tp=8, kv_split=4),
        StepShape(batch=1, rows_per_seq=1, context=90036),
        Dtypes(state=2.0),
    )
    # Library defaults batch 1 total HBM bytes
    assert sum(m.hbm_bytes for m in mods) == pytest.approx(2_801_443_840.0, abs=1.0)
    # Library defaults batch 1 strict HBM bytes
    assert sum(m.strict_hbm_bytes for m in mods) == pytest.approx(
        2_628_574_720.0, abs=1.0
    )


def test_ar_algorithm_wire_and_hops() -> None:
    """Verify wire bytes and hop counts across all-reduce algorithms."""
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
    """Verify total collective count and hops under sequence parallel."""
    geom = from_hf_config(QWEN35_CFG)
    mods = pass_costs(
        geom,
        Parallel(tp=8, moe="ep", residual="sp"),
        StepShape(batch=1, rows_per_seq=4),
        Dtypes(),
    )
    expected_sp_colls = 303
    colls = [c for m in mods for c in m.collectives]
    assert len(colls) == expected_sp_colls
    # SP all-reduce total wire bytes across layers
    assert sum(c.wire_bytes for c in colls) == pytest.approx(16_217_600.0, abs=1.0)
    assert sum(c.hops for c in colls) == expected_sp_colls


def test_a2a_push_wire_bytes() -> None:
    """Verify all-to-all push wire bytes across residual modes."""
    expected_rep_colls = 242
    expected_sp_colls = 243
    geom = from_hf_config(QWEN35_CFG)
    shape = StepShape(batch=1, rows_per_seq=4)
    replicated_costs = pass_costs(
        geom,
        Parallel(tp=8, moe="ep", moe_dispatch="a2a_push"),
        shape,
        Dtypes(),
    )
    replicated_collectives = [c for m in replicated_costs for c in m.collectives]
    assert len(replicated_collectives) == expected_rep_colls
    # Replicated residual all-to-all push wire bytes
    assert sum(c.wire_bytes for c in replicated_collectives) == pytest.approx(
        21_751_296.0, abs=1.0
    )
    sp = pass_costs(
        geom,
        Parallel(tp=8, moe="ep", moe_dispatch="a2a_push", residual="sp"),
        shape,
        Dtypes(),
    )
    sp_c = [c for m in sp for c in m.collectives]
    assert len(sp_c) == expected_sp_colls
    # Sequence parallel residual all-to-all push wire bytes
    assert sum(c.wire_bytes for c in sp_c) == pytest.approx(13_207_040.0, abs=1.0)


def test_kv_split_adds_merge_collectives() -> None:
    """Verify merge collective addition when KV split is active."""
    expected_kv_split_colls = 167
    expected_xchg_colls = 137
    geom = from_hf_config(QWEN35_CFG)
    shape = StepShape(batch=1, rows_per_seq=4)
    ag_rs = pass_costs(
        geom, Parallel(tp=8, kv_split=4, kv_merge="ag_rs"), shape, Dtypes()
    )
    c1 = [c for m in ag_rs for c in m.collectives]
    assert len(c1) == expected_kv_split_colls
    # KV split all-gather / reduce-scatter merge wire bytes
    assert sum(c.wire_bytes for c in c1) == pytest.approx(30_610_176.0, abs=1.0)
    xchg = pass_costs(
        geom, Parallel(tp=8, kv_split=4, kv_merge="exchange"), shape, Dtypes()
    )
    c2 = [c for m in xchg for c in m.collectives]
    assert len(c2) == expected_xchg_colls
    # KV split exchange merge wire bytes
    assert sum(c.wire_bytes for c in c2) == pytest.approx(30_967_296.0, abs=1.0)


def test_a16_routes_fp8_weights_to_bf16_peak() -> None:
    """Verify A16 activation routing, VPU peak ratio, and sub-channel scale FLOPs."""
    assert pytest.approx(1.0 / 128.0) == DEFAULT_VPU_TO_BF16_RATIO
    assert CHIPS["tpu7x"].vpu_flops == pytest.approx(
        1.155e15 * DEFAULT_VPU_TO_BF16_RATIO
    )
    geom = from_hf_config(QWEN35_CFG)
    shape = StepShape(batch=1, rows_per_seq=1, context=0)
    m_a8 = pass_costs(geom, Parallel(tp=8), shape, Dtypes(act="a8"))
    m_a16 = pass_costs(geom, Parallel(tp=8), shape, Dtypes(act="a16"))
    assert sum(m.fp8_flops for m in m_a8) > 0.0
    assert sum(m.fp8_flops for m in m_a16) == 0.0
    tot_a8 = sum(m.fp8_flops + m.bf16_flops for m in m_a8)
    tot_a16 = sum(m.fp8_flops + m.bf16_flops for m in m_a16)
    assert tot_a16 == pytest.approx(tot_a8, rel=1e-12)
    geom_sub = dc_replace(geom, weight_block_size=(128, 128))
    m_sub = pass_costs(geom_sub, Parallel(tp=8), shape, Dtypes(act="a8"))
    assert sum(m.vpu_flops for m in m_sub) > sum(m.vpu_flops for m in m_a8)


def test_invalid_plans_raise() -> None:
    """Verify invalid parallelism configurations raise ValueError."""
    geom = from_hf_config(QWEN35_CFG)
    shape = StepShape(batch=1, rows_per_seq=1)
    for bad in (
        Parallel(tp=8, kv_split=3),
        Parallel(tp=8, moe="tp_slice", residual="sp"),
        Parallel(tp=8, moe="tp_slice", moe_dispatch="a2a_push"),
        Parallel(tp=1, moe="ep", residual="sp"),
    ):
        with pytest.raises(ValueError, match=re.compile(r"Parallel.*(must|requires)")):
            pass_costs(geom, bad, shape, Dtypes())


def test_uneven_pp8_and_gdn_recurrence_flops() -> None:
    """Verify pipeline parallel scaling and GDN recurrence FLOP calculations."""
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
    assert kv_only_b == pytest.approx(15.0 * 8192.0 * 1024.0, abs=1.0)
    kv_only_min_b = m_dcp1["gqa"].strict_hbm_bytes - m_dcp1_ctx0["gqa"].strict_hbm_bytes
    assert kv_only_min_b == pytest.approx(15.0 * 8192.0 * 512.0, abs=1.0)
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

    assert ep_busiest_experts(4, 2, 8, 8) == pytest.approx(1.0)
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
    """Verify latent MoE dimensions and 4-bit compressed tensor calculations."""
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
    expected_top_k = 8
    expected_moe_inter = 3584
    g = from_hf_config(latent_w4_cfg)
    assert g.top_k == expected_top_k
    assert g.moe_inter == expected_moe_inter
    dt = dtypes_from_hf_config(latent_w4_cfg)
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
    """Verify global row scaling and FLOP conservation under PCP and EP."""
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
    exp_ed = ep_busiest_experts(64, 8, (4 + 8 * 2) * 2, 8)
    exp_w_bytes = exp_ed * float(4 * 3 * 2048 * 1024) * 1.0 + router_elems * 1.0
    assert moe_mix.hbm_bytes == pytest.approx(exp_w_bytes, rel=1e-6)


def test_nope_mla_zero_rope_dim_and_compress_ratios() -> None:
    """Verify NoPE-MLA costing with zero rope dim and compression ratio."""
    nope_mla_cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "layer_types": ["mla"] * 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "q_lora_rank": 1024,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 0,
        "v_head_dim": 128,
        "intermediate_size": 8192,
        "vocab_size": 64000,
    }
    geom_uncomp = rf.from_hf_config(nope_mla_cfg)
    assert geom_uncomp.qk_rope_head_dim == 0
    geom_comp = rf.from_hf_config(
        {**nope_mla_cfg, "compress_ratios": [4.0, 0.5, 1.0, 1.0]}
    )
    assert geom_comp.compress_ratios == (4.0, 0.5, 1.0, 1.0)

    par = Parallel(tp=4)
    dt = Dtypes(kv=1.0)
    shape = StepShape(batch=4, rows_per_seq=1, context=4096)
    mods_u = {m.name: m for m in pass_costs(geom_uncomp, par, shape, dt)}
    mods_c = {m.name: m for m in pass_costs(geom_comp, par, shape, dt)}
    mods_ctx0 = {
        m.name: m
        for m in pass_costs(
            geom_uncomp, par, StepShape(batch=4, rows_per_seq=1, context=0), dt
        )
    }
    kv_traffic_u = mods_u["mla"].hbm_bytes - mods_ctx0["mla"].hbm_bytes
    kv_traffic_c = mods_c["mla"].hbm_bytes - mods_ctx0["mla"].hbm_bytes
    assert kv_traffic_c == pytest.approx(kv_traffic_u * (2.75 / 4.0), rel=1e-9)

    bd_u = rf.hbm_breakdown_per_core_bytes(geom_uncomp, dt, par, SeqKvLayout(4, 4096))
    bd_c = rf.hbm_breakdown_per_core_bytes(geom_comp, dt, par, SeqKvLayout(4, 4096))
    assert bd_c.kv == pytest.approx(bd_u.kv * (2.75 / 4.0), rel=1e-9)

    with pytest.raises(ValueError, match="compress_ratios"):
        rf.from_hf_config({**nope_mla_cfg, "compress_ratios": [2.0, 2.0]})
    with pytest.raises(ValueError, match="compress_ratios"):
        rf.from_hf_config({**nope_mla_cfg, "compress_ratios": [2.0, 0.0, 1.0, 1.0]})

    with pytest.raises(ValueError, match="qk_rope_head_dim"):
        pass_costs(
            rf.from_hf_config({**nope_mla_cfg, "qk_rope_head_dim": -1}),
            par,
            shape,
            dt,
        )
    with pytest.raises(ValueError, match="qk_rope_head_dim"):
        rf.hbm_breakdown_per_core_bytes(
            rf.from_hf_config({**nope_mla_cfg, "qk_rope_head_dim": -1}),
            dt,
            par,
            SeqKvLayout(1, 1024),
        )


def test_mxu_m_tile_systolic_duty_cycle_on_sparse_moe() -> None:
    """Verify MXU M-tile systolic duty cycle scaling on sparse MoE."""
    geom = from_hf_config(QWEN35_CFG)
    par = Parallel(tp=8)
    dt = Dtypes(w_expert=1.0, act="a8")
    shape = StepShape(batch=1, rows_per_seq=1, context=0)
    mods_unpad = {m.name: m for m in pass_costs(geom, par, shape, dt, mxu_m_tile=1)}
    mods_pad = {m.name: m for m in pass_costs(geom, par, shape, dt, mxu_m_tile=256)}
    assert mods_pad["moe"].hbm_bytes == pytest.approx(
        mods_unpad["moe"].hbm_bytes, rel=1e-12
    )
    shared_fp8 = 2.0 * 1.0 * float(60 * 3 * 4096 * (1024 // 8))
    routed_unpad = mods_unpad["moe"].fp8_flops - shared_fp8
    routed_pad = mods_pad["moe"].fp8_flops - shared_fp8
    assert routed_pad == pytest.approx(routed_unpad * 256.0, rel=1e-9)
    assert routed_pad < routed_unpad * 256.0 * (20.0 / 10.0)

    par_ep = Parallel(tp=8, ep=8, moe="ep")
    ep_unpad = {m.name: m for m in pass_costs(geom, par_ep, shape, dt, mxu_m_tile=1)}
    ep_pad = {m.name: m for m in pass_costs(geom, par_ep, shape, dt, mxu_m_tile=256)}
    ep_routed_unpad = ep_unpad["moe"].fp8_flops - shared_fp8
    ep_routed_pad = ep_pad["moe"].fp8_flops - shared_fp8
    assert ep_routed_pad == pytest.approx(ep_routed_unpad * 256.0, rel=1e-9)


def test_block_parallel_sp_collectives_per_moe_dispatch() -> None:
    """Verify SP residual collectives per dispatch mode under block parallelism."""
    cfg = {
        "architectures": ["Qwen2MoeForCausalLM"],
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_hidden_layers": 2,
        "num_experts": 16,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 1024,
        "intermediate_size": 4096,
        "vocab_size": 32000,
        "block_parallel": 2,
    }
    geom = from_hf_config(cfg)
    shape = StepShape(batch=1, rows_per_seq=4)
    dt = Dtypes()

    par_ar = Parallel(tp=8, residual="sp", moe="ep", moe_dispatch="allreduce")
    costs_ar = pass_costs(geom, par_ar, shape, dt)
    attn_ar = next(m for m in costs_ar if m.name == "gqa")
    moe_ar = next(m for m in costs_ar if m.name == "moe")
    assert "out_rs" not in {c.name for c in attn_ar.collectives}
    assert "dispatch_ag" not in {c.name for c in moe_ar.collectives}
    assert "combine_rs" in {c.name for c in moe_ar.collectives}

    par_push = Parallel(tp=8, residual="sp", moe="ep", moe_dispatch="a2a_push")
    costs_push = pass_costs(geom, par_push, shape, dt)
    attn_push = next(m for m in costs_push if m.name == "gqa")
    moe_push = next(m for m in costs_push if m.name == "moe")
    assert "out_rs" in {c.name for c in attn_push.collectives}
    assert "dispatch_push" in {c.name for c in moe_push.collectives}

    par_ag = Parallel(tp=8, residual="sp", moe="ep", moe_dispatch="ag_ep")
    costs_ag = pass_costs(geom, par_ag, shape, dt)
    attn_ag = next(m for m in costs_ag if m.name == "gqa")
    moe_ag = next(m for m in costs_ag if m.name == "moe")
    assert "out_rs" in {c.name for c in attn_ag.collectives}
    assert "dispatch_ag" in {c.name for c in moe_ag.collectives}


def test_step_ffn_layers_match_capacity_ffn_layers() -> None:
    """Verify pass_costs FFN accounting agrees with compute_w_ffn across hybrids."""
    llama4_cfg: dict[str, object] = {
        "model_type": "llama4_text",
        "hidden_size": 5120,
        "num_hidden_layers": 6,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "intermediate_size": 8192,
        "moe_intermediate_size": 8192,
        "num_local_experts": 16,
        "num_experts_per_tok": 1,
        "interleave_moe_layer_step": 2,
        "vocab_size": 202048,
    }
    nemotron_cfg: dict[str, object] = {
        "model_type": "nemotron_h",
        "hidden_size": 4096,
        "num_hidden_layers": 6,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "conv_kernel": 4,
        "hybrid_override_pattern": "M-M*ME",
        "n_routed_experts": 16,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 131072,
    }
    dt = Dtypes()
    par = Parallel(tp=1)
    shape = StepShape(batch=4096, rows_per_seq=1, context=16)

    expected_three = 3
    geom_l4 = from_hf_config(llama4_cfg)
    assert layer_ffn_kinds(geom_l4).count("dense") == expected_three
    assert layer_ffn_kinds(geom_l4).count("moe") == expected_three
    assert geom_l4.ffn_layer_counts() == (expected_three, expected_three)
    mods_l4 = pass_costs(geom_l4, par, shape, dt)
    step_ffn_l4 = sum(m.hbm_bytes for m in mods_l4 if m.name in ("mlp", "moe"))
    w_ffn_l4, _ = compute_w_ffn(geom_l4, dt, par)
    assert step_ffn_l4 == pytest.approx(w_ffn_l4, abs=1.0)
    # Llama 4 FFN total bytes
    assert step_ffn_l4 == pytest.approx(6_417_776_640.0, abs=1.0)

    geom_nh = from_hf_config(nemotron_cfg)
    expected_four = 4
    assert layer_ffn_kinds(geom_nh).count("dense") == 1
    assert layer_ffn_kinds(geom_nh).count("moe") == 1
    assert layer_ffn_kinds(geom_nh).count("none") == expected_four
    assert geom_nh.ffn_layer_counts() == (1, 1)
    mods_nh = pass_costs(geom_nh, par, shape, dt)
    step_ffn_nh = sum(m.hbm_bytes for m in mods_nh if m.name in ("mlp", "moe"))
    w_ffn_nh, _ = compute_w_ffn(geom_nh, dt, par)
    assert step_ffn_nh == pytest.approx(w_ffn_nh, abs=1.0)
    # Nemotron hybrid FFN total bytes
    assert step_ffn_nh == pytest.approx(503_447_552.0, abs=1.0)


def test_kv_bytes_single_source_physical_and_logical() -> None:
    """Verify kv_bytes_per_token_per_layer physical padding and unpadded bounds."""
    geom = from_hf_config(QWEN35_CFG)
    par = Parallel(tp=8)
    dt = Dtypes(kv=1.0)
    expected_phys = 1024.0
    expected_min = 512.0
    assert kv_bytes_per_token_per_layer(geom, dt, par, "gqa", physical=True) == (
        pytest.approx(expected_phys)
    )
    assert kv_bytes_per_token_per_layer(geom, dt, par, "gqa", physical=False) == (
        pytest.approx(expected_min)
    )
    fam_cfg = {
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "layer_types": ["gqa", "swa", "mla", "dsa"],
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "sliding_window": 256,
        "q_lora_rank": 512,
        "kv_lora_rank": 300,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "index_n_heads": 8,
        "index_head_dim": 64,
        "index_topk": 256,
        "intermediate_size": 4096,
        "vocab_size": 32000,
    }
    fam_geom = from_hf_config(fam_cfg)
    mods = {
        m.name: m
        for m in pass_costs(
            fam_geom,
            Parallel(tp=8),
            StepShape(batch=2, rows_per_seq=1, context=1024),
            Dtypes(kv=1.0),
        )
    }
    for fam in ("gqa", "swa", "mla", "dsa"):
        mod = mods[fam]
        assert mod.hbm_bytes_min is not None
        assert mod.hbm_bytes > mod.hbm_bytes_min


def test_dense_four_bit_weights_charge_unpack() -> None:
    """Verify 4-bit dense weights charge vector unpack operations."""
    geom = from_hf_config(QWEN35_CFG)
    assert geom.weight_block_size == (1, 1)
    par = Parallel(tp=8)
    shape = StepShape(batch=1, rows_per_seq=1, context=1024)

    def get_gqa(w_dense: float) -> ModuleCost:
        """Return GQA module cost for dense weight byte size."""
        mods = {
            m.name: m
            for m in pass_costs(
                geom,
                par,
                shape,
                Dtypes(w_dense=w_dense),
            )
        }
        return mods["gqa"]

    m_half = get_gqa(0.5)
    m_one = get_gqa(1.0)
    m_two = get_gqa(2.0)

    # With weight_block_size == (1, 1), subchannel scaling is 0 for all dtypes.
    # Weight elements is the byte delta between 1.0 and 0.5 byte/elem over 0.5.
    weight_elems = (m_one.hbm_bytes - m_half.hbm_bytes) / 0.5
    assert (m_half.vpu_flops - m_one.vpu_flops) == pytest.approx(2.0 * weight_elems)
    assert (m_half.strict_vpu_flops - m_one.strict_vpu_flops) == pytest.approx(
        2.0
        * weight_elems
        * VECTOR_MAX_ELEMENTS_PER_WORD
        / VECTOR_SUB_BYTE_ELEMENTS_PER_WORD
    )

    # For w_dense >= 1.0, sub-byte unpack term is 0.0 and no other VPU term
    # depends on w_dense.
    assert (m_one.strict_vpu_flops - m_two.strict_vpu_flops) == pytest.approx(0.0)


def test_routed_four_bit_experts_use_sub_byte_packing_in_strict_count() -> None:
    """Verify 4-bit routed experts scale unpack by word packing ratio."""
    geom = from_hf_config(QWEN35_CFG)
    parallel = Parallel(tp=8)
    shape = StepShape(batch=1, rows_per_seq=1, context=0)

    def get_moe_cost(expert_weight_bytes: float) -> ModuleCost:
        """Return MoE module cost for expert weight byte size."""
        module_costs = {
            module.name: module
            for module in pass_costs(
                geom,
                parallel,
                shape,
                Dtypes(w_expert=expert_weight_bytes),
            )
        }
        return module_costs["moe"]

    moe_half = get_moe_cost(0.5)
    moe_one = get_moe_cost(1.0)

    routed_elems_min = (moe_one.hbm_bytes_min - moe_half.hbm_bytes_min) / 0.5
    assert (moe_half.strict_vpu_flops - moe_one.strict_vpu_flops) == pytest.approx(
        2.0
        * routed_elems_min
        * VECTOR_MAX_ELEMENTS_PER_WORD
        / VECTOR_SUB_BYTE_ELEMENTS_PER_WORD
    )


def test_latent_projection_weights_charge_sub_byte_unpack() -> None:
    """Verify latent projection weights charge sub-byte unpack flops."""
    latent_hidden = 512
    raw_text_config = QWEN35_CFG["text_config"]
    assert isinstance(raw_text_config, dict)
    config = {
        **QWEN35_CFG,
        "text_config": {
            **raw_text_config,
            "routed_expert_hidden_size": latent_hidden,
        },
    }
    geom = from_hf_config(config)
    assert geom.routed_expert_hidden == latent_hidden

    parallel = Parallel(tp=8)
    shape = StepShape(batch=1, rows_per_seq=1, context=0)

    def get_moe_module(dense_weight_bytes: float) -> ModuleCost:
        """Return MoE module cost for dense weight byte size."""
        module_costs = {
            module.name: module
            for module in pass_costs(
                geom,
                parallel,
                shape,
                Dtypes(w_dense=dense_weight_bytes, w_expert=1.0),
            )
        }
        return module_costs["moe"]

    moe_half = get_moe_module(0.5)
    moe_one = get_moe_module(1.0)

    dense_elems = (moe_one.hbm_bytes - moe_half.hbm_bytes) / 0.5
    assert (moe_half.vpu_flops - moe_one.vpu_flops) == pytest.approx(2.0 * dense_elems)
