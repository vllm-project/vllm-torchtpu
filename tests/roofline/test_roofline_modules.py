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

import math
import re
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

import pytest
from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from vllm_torchtpu.roofline.modules import ModuleCost

import vllm_torchtpu.roofline as rf
from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    ModelGeometry,
    Parallel,
    SeqKvLayout,
    StepShape,
    dtypes_from_hf_config,
    from_hf_config,
    pass_costs,
    step_floor,
)
from vllm_torchtpu.roofline import chips as rf_chips
from vllm_torchtpu.roofline.constants import (
    BF16_BYTES,
    DEFAULT_VECTOR_TO_BF16_RATIO,
    FP8_BYTES,
    MATMUL_FLOP_PER_MAC,
    VECTOR_MAX_ELEMENTS_PER_WORD,
    VECTOR_SUB_BYTE_ELEMENTS_PER_WORD,
)
from vllm_torchtpu.roofline.modules import (
    ep_busiest_experts,
    experts_touched,
)
from vllm_torchtpu.roofline.step import combine_modules

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
    assert mods["gqa"].hbm_bytes == pytest.approx(220200960 + 1382952960, abs=1.0)
    assert mods["gdn"].hbm_bytes == pytest.approx(663828480 + 48015360, abs=1.0)
    assert mods["moe"].hbm_bytes == pytest.approx(
        943718400 + 94371840 + 251658240, abs=1.0
    )
    assert mods["lm_head"].hbm_bytes == pytest.approx(254279680, abs=1.0)
    total = sum(m.hbm_bytes for m in mods.values())
    assert total == pytest.approx(3859025920, abs=1.0)


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
    assert mods["gqa"].hbm_bytes == pytest.approx(220200960 + 345738240, abs=1.0)
    total = sum(m.hbm_bytes for m in mods.values())
    assert total == pytest.approx(2821811200, abs=1.0)


def test_collective_payload_and_count() -> None:
    """Verify collective communication count and wire payload totals."""
    expected_batch1_collectives = 122
    geom = from_hf_config(QWEN35_CFG)
    mods1 = pass_costs(
        geom, Parallel(tp=8), StepShape(batch=1, rows_per_seq=1), Dtypes()
    )
    assert sum(m.n_collectives for m in mods1) == expected_batch1_collectives
    assert sum(m.ici_bytes for m in mods1) == pytest.approx(6319872.0)
    mods4 = pass_costs(
        geom, Parallel(tp=8), StepShape(batch=1, rows_per_seq=4), Dtypes()
    )
    assert sum(m.ici_bytes for m in mods4) == pytest.approx(25279488.0)
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
        assert got == pytest.approx(val, abs=1e-06)
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
    assert by_name["moe"].hbm_bytes == pytest.approx(2595776984.77, abs=1.0)
    assert sum(m.hbm_bytes for m in mods) == pytest.approx(5168280024.77, abs=1.0)
    assert by_name["gqa"].hbm_bytes_min is not None
    logical_hbm = sum(m.hbm_bytes for m in mods) - (
        by_name["gqa"].hbm_bytes - by_name["gqa"].hbm_bytes_min
    )
    assert logical_hbm == pytest.approx(4476803544.77, abs=1.0)


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
    assert sp["moe"].hbm_bytes - stock["moe"].hbm_bytes == pytest.approx(
        660602880.0, abs=1.0
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
    assert sum(m.hbm_bytes for m in mods) == pytest.approx(2801443840.0, abs=1.0)
    assert sum(m.strict_hbm_bytes for m in mods) == pytest.approx(2628574720.0, abs=1.0)


def test_ar_algorithm_wire_and_hops() -> None:
    """Verify wire bytes and hop counts across all-reduce algorithms."""
    geom = from_hf_config(QWEN35_CFG)
    shape = StepShape(batch=1, rows_per_seq=4)
    cases = (
        ("one_shot", 122, 29492736.0, 122),
        ("two_shot", 122, 8676864.0, 243),
        ("ring", 122, 8676864.0, 1701),
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
    # Embed uses ReduceScatter under SP instead of AllReduce (-200704 wire bytes).
    assert sum(c.wire_bytes for c in colls) == pytest.approx(16016896.0, abs=1.0)
    assert sum(c.hops for c in colls) == expected_sp_colls


def test_a2a_push_wire_bytes() -> None:
    """Verify all-to-all push wire bytes across residual modes."""
    expected_rep_colls = 242
    expected_sp_colls = 243
    geom = from_hf_config(QWEN35_CFG)
    shape = StepShape(batch=1, rows_per_seq=4)
    replicated_costs = pass_costs(
        geom, Parallel(tp=8, moe="ep", moe_dispatch="a2a_push"), shape, Dtypes()
    )
    replicated_collectives = [c for m in replicated_costs for c in m.collectives]
    assert len(replicated_collectives) == expected_rep_colls
    assert sum(c.wire_bytes for c in replicated_collectives) == pytest.approx(
        21751296.0, abs=1.0
    )
    sp = pass_costs(
        geom,
        Parallel(tp=8, moe="ep", moe_dispatch="a2a_push", residual="sp"),
        shape,
        Dtypes(),
    )
    sp_c = [c for m in sp for c in m.collectives]
    assert len(sp_c) == expected_sp_colls
    # Embed uses ReduceScatter under SP instead of AllReduce (-200704 wire bytes).
    assert sum(c.wire_bytes for c in sp_c) == pytest.approx(13006336.0, abs=1.0)


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
    assert sum(c.wire_bytes for c in c1) == pytest.approx(30610176.0, abs=1.0)
    xchg = pass_costs(
        geom, Parallel(tp=8, kv_split=4, kv_merge="exchange"), shape, Dtypes()
    )
    c2 = [c for m in xchg for c in m.collectives]
    assert len(c2) == expected_xchg_colls
    assert sum(c.wire_bytes for c in c2) == pytest.approx(30967296.0, abs=1.0)


def test_a16_routes_fp8_weights_to_bf16_peak() -> None:
    """Verify A16 activation routing, vector peak ratio, and sub-channel scale FLOPs."""
    assert pytest.approx(1.0 / 128.0) == DEFAULT_VECTOR_TO_BF16_RATIO
    assert CHIPS["tpu7x"].vector_flops == pytest.approx(
        1155000000000000.0 * DEFAULT_VECTOR_TO_BF16_RATIO
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
    assert sum(m.vector_flops for m in m_sub) > sum(m.vector_flops for m in m_a8)


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
        with pytest.raises(ValueError, match=re.compile("Parallel.*(must|requires)")):
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
        8.0 / 60.0, rel=1e-06
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
        m_ep8_tp8["moe"].fp8_flops, rel=1e-09
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
                    }
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
    assert dt.w_expert == pytest.approx(0.5625, abs=1e-06)
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
    assert mla_p2.bf16_flops == pytest.approx(0.5 * mla_p1.bf16_flops, rel=1e-06)


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
    assert moe_pcp2.hbm_bytes == pytest.approx(moe_tp4.hbm_bytes, rel=1e-06)
    router_elems = float(4 * 2048 * 64)
    routed_fp8_pcp2 = moe_pcp2.fp8_flops - 2.0 * 8.0 * router_elems
    routed_fp8_tp4 = moe_tp4.fp8_flops - 2.0 * 16.0 * router_elems
    assert routed_fp8_pcp2 == pytest.approx(routed_fp8_tp4, rel=1e-06)
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
    assert moe_mix.hbm_bytes == pytest.approx(exp_w_bytes, rel=1e-06)


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
    assert kv_traffic_c == pytest.approx(kv_traffic_u * (2.75 / 4.0), rel=1e-09)
    bd_u = rf.hbm_breakdown_per_core_bytes(geom_uncomp, dt, par, SeqKvLayout(4, 4096))
    bd_c = rf.hbm_breakdown_per_core_bytes(geom_comp, dt, par, SeqKvLayout(4, 4096))
    assert bd_c.kv == pytest.approx(bd_u.kv * (2.75 / 4.0), rel=1e-09)
    with pytest.raises(ValueError, match="compress_ratios"):
        rf.from_hf_config({**nope_mla_cfg, "compress_ratios": [2.0, 2.0]})
    with pytest.raises(ValueError, match="compress_ratios"):
        rf.from_hf_config({**nope_mla_cfg, "compress_ratios": [2.0, 0.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="qk_rope_head_dim"):
        pass_costs(
            rf.from_hf_config({**nope_mla_cfg, "qk_rope_head_dim": -1}), par, shape, dt
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
    assert routed_pad == pytest.approx(routed_unpad * 256.0, rel=1e-09)
    assert routed_pad < routed_unpad * 256.0 * (20.0 / 10.0)
    par_ep = Parallel(tp=8, ep=8, moe="ep")
    ep_unpad = {m.name: m for m in pass_costs(geom, par_ep, shape, dt, mxu_m_tile=1)}
    ep_pad = {m.name: m for m in pass_costs(geom, par_ep, shape, dt, mxu_m_tile=256)}
    ep_routed_unpad = ep_unpad["moe"].fp8_flops - shared_fp8
    ep_routed_pad = ep_pad["moe"].fp8_flops - shared_fp8
    assert ep_routed_pad == pytest.approx(ep_routed_unpad * 256.0, rel=1e-09)


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
    assert (m_half.vector_flops - m_one.vector_flops) == pytest.approx(
        2.0 * weight_elems
    )
    assert (m_half.strict_vector_flops - m_one.strict_vector_flops) == pytest.approx(
        2.0
        * weight_elems
        * VECTOR_MAX_ELEMENTS_PER_WORD
        / VECTOR_SUB_BYTE_ELEMENTS_PER_WORD
    )

    # For w_dense >= 1.0, sub-byte unpack term is 0.0 and no other vector term
    # depends on w_dense.
    assert (m_one.strict_vector_flops - m_two.strict_vector_flops) == pytest.approx(0.0)


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
    assert (
        moe_half.strict_vector_flops - moe_one.strict_vector_flops
    ) == pytest.approx(
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
    assert (moe_half.vector_flops - moe_one.vector_flops) == pytest.approx(
        2.0 * dense_elems
    )


def _dsa_base_geom(
    layer_types: tuple[str, ...],
    *,
    num_layers: int | None = None,
) -> ModelGeometry:
    """Build a compact DeepSeek-style MLA/DSA test geometry."""
    return ModelGeometry(
        hidden=2048,
        num_layers=len(layer_types) if num_layers is None else num_layers,
        layer_types=layer_types,
        num_heads=32,
        num_kv_heads=0,
        head_dim=128,
        attn_output_gate=False,
        linear_k_heads=0,
        linear_v_heads=0,
        linear_k_dim=0,
        linear_v_dim=0,
        conv_kernel=0,
        num_experts=0,
        top_k=0,
        moe_inter=0,
        shared_inter=0,
        dense_inter=4096,
        first_k_dense_replace=len(layer_types) if num_layers is None else num_layers,
        vocab=32000,
        mtp_layers=0,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_n_heads=8,
        index_head_dim=64,
        index_topk=512,
    )


def test_dsa_layers_not_double_counted_conservation() -> None:
    """Verify DSA layers split cleanly from plain MLA without double counting."""
    par = Parallel(tp=4)
    shape = StepShape(batch=2, rows_per_seq=1, context=1024)
    dt = Dtypes(w_dense=1.0, kv=1.0, act="a8")

    pure_mla = {
        m.name: m for m in pass_costs(_dsa_base_geom(("mla", "mla")), par, shape, dt)
    }
    pure_dsa = {
        m.name: m for m in pass_costs(_dsa_base_geom(("dsa", "dsa")), par, shape, dt)
    }
    mixed = {
        m.name: m
        for m in pass_costs(
            _dsa_base_geom(("mla", "mla", "dsa", "dsa")), par, shape, dt
        )
    }

    assert "mla" not in pure_dsa
    assert "dsa" in pure_dsa
    assert "mla" in mixed
    assert "dsa" in mixed
    assert mixed["mla"].hbm_bytes == pytest.approx(pure_mla["mla"].hbm_bytes)
    assert mixed["dsa"].hbm_bytes == pytest.approx(pure_dsa["dsa"].hbm_bytes)
    assert mixed["mla"].strict_fp8_flops == pytest.approx(
        pure_mla["mla"].strict_fp8_flops
    )
    assert mixed["dsa"].strict_fp8_flops == pytest.approx(
        pure_dsa["dsa"].strict_fp8_flops
    )

    # Full 61-layer DSA model decode floor matches exact 1x DSA weight + KV footprint.
    full_layers = 61
    geom_v32 = dc_replace(
        _dsa_base_geom(("dsa",) * full_layers),
        hidden=7168,
        num_heads=128,
        dense_inter=2048,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
    )
    sf_v32 = step_floor(
        geom_v32, CHIPS["tpu7x"], Parallel(tp=8), StepShape(batch=1, context=4096)
    )
    v32_mods = {m.name: m for m in sf_v32.modules}
    assert "mla" not in v32_mods
    nq_tp = 128 // 8
    idx_heads_tp = 64 // 8
    one_dsa_w = (
        7168 * 1536
        + 1536 * nq_tp * (128 + 64)
        + 7168 * (512 + 64)
        + 512 * nq_tp * (128 + 128)
        + (nq_tp * 128) * 7168
        + 1536 * idx_heads_tp * 128
        + 7168 * 128
        + 7168 * idx_heads_tp
    )
    exp_strict_dsa_hbm = float(full_layers) * (
        one_dsa_w * Dtypes().w_dense
        + 2048 * (512 + 64) * Dtypes().kv
        + 4096 * 128 * Dtypes().kv
    )
    assert v32_mods["dsa"].strict_hbm_bytes == pytest.approx(
        exp_strict_dsa_hbm, rel=1e-9
    )


def test_mla_prefill_strict_uses_non_absorbed_min_and_dsa_topk_cap() -> None:
    """Verify strict MLA prefill picks non-absorbed min cleanly and caps DSA pairs."""
    tp = 4
    geom_mla = _dsa_base_geom(("mla", "mla"))
    par = Parallel(tp=tp)
    dt = Dtypes(w_dense=1.0, kv=1.0, act="a8")
    nq = geom_mla.num_heads // tp
    n_layers = float(geom_mla.num_layers)
    q_w = geom_mla.hidden * geom_mla.q_lora_rank + geom_mla.q_lora_rank * nq * (
        geom_mla.qk_nope_head_dim + geom_mla.qk_rope_head_dim
    )
    kv_a = geom_mla.hidden * (geom_mla.kv_lora_rank + geom_mla.qk_rope_head_dim)
    ukv_layer = float(
        geom_mla.kv_lora_rank * nq * (geom_mla.qk_nope_head_dim + geom_mla.v_head_dim)
    )
    o_w = float((nq * geom_mla.v_head_dim) * geom_mla.hidden)
    base_elems = n_layers * float(q_w + kv_a) + n_layers * o_w
    ukv_elems = n_layers * ukv_layer

    # Uncached prefill: non-absorbed wins and ukv is not double-counted.
    uncached = StepShape(
        batch=0, rows_per_seq=0, context=512, prefill_tokens=512, prefix_hit_rate=0.0
    )
    mla_uncached = {m.name: m for m in pass_costs(geom_mla, par, uncached, dt)}["mla"]
    pairs_512 = 0.5 * 512.0 * 512.0
    score_w_mha = float(
        nq
        * (geom_mla.qk_nope_head_dim + geom_mla.qk_rope_head_dim + geom_mla.v_head_dim)
    )
    score_w_abs = float(nq * (2 * geom_mla.kv_lora_rank + geom_mla.qk_rope_head_dim))
    exp_mha_fp8 = MATMUL_FLOP_PER_MAC * 512.0 * (base_elems + ukv_elems) + (
        pairs_512 * 2.0 * score_w_mha * n_layers
    )
    exp_abs_fp8 = MATMUL_FLOP_PER_MAC * 512.0 * (base_elems + ukv_elems) + (
        pairs_512 * 2.0 * score_w_abs * n_layers
    )
    assert mla_uncached.strict_fp8_flops == pytest.approx(exp_mha_fp8, rel=1e-9)
    assert mla_uncached.strict_fp8_flops < exp_abs_fp8
    assert mla_uncached.strict_bf16_flops == 0.0

    # Large cached prefix + tiny prefill: absorbed wins over recomputing cached ukv.
    hit_heavy = StepShape(
        batch=0,
        rows_per_seq=0,
        context=8192,
        prefill_tokens=8192,
        prefix_hit_rate=0.999,
    )
    mla_hit = {m.name: m for m in pass_costs(geom_mla, par, hit_heavy, dt)}["mla"]
    cached_pref = round(8192 * 0.999)
    eff_pref = math.ceil(8192 * (1.0 - 0.999))
    hit_pairs = float(eff_pref) * (float(cached_pref) + 0.5 * float(eff_pref))
    exp_hit_abs = MATMUL_FLOP_PER_MAC * float(eff_pref) * (base_elems + ukv_elems) + (
        hit_pairs * 2.0 * score_w_abs * n_layers
    )
    exp_hit_mha = MATMUL_FLOP_PER_MAC * (
        float(eff_pref) * base_elems + float(cached_pref + eff_pref) * ukv_elems
    ) + (hit_pairs * 2.0 * score_w_mha * n_layers)
    assert mla_hit.strict_fp8_flops == pytest.approx(exp_hit_abs, rel=1e-9)
    assert mla_hit.strict_fp8_flops < exp_hit_mha

    # DSA ramp + topk cap with PCP=2: sum_{pos=257..1024} min(pos, 512) / 2.
    geom_dsa = _dsa_base_geom(("dsa", "dsa"))
    idx_heads = max(1, geom_dsa.index_n_heads // tp)
    idx_w = float(
        geom_dsa.q_lora_rank * idx_heads * geom_dsa.index_head_dim
        + geom_dsa.hidden * geom_dsa.index_head_dim
        + geom_dsa.hidden * idx_heads
    )
    dsa_base = base_elems + n_layers * idx_w
    dsa_shape = StepShape(
        batch=0,
        rows_per_seq=0,
        context=2048,
        prefill_tokens=1024,
        prefix_hit_rate=0.25,
    )
    dsa_mod = {
        m.name: m for m in pass_costs(geom_dsa, Parallel(tp=tp, pcp=2), dsa_shape, dt)
    }["dsa"]
    c_prefix = round(1024 * 0.25)
    p_rank = math.ceil(1024 * 0.75 / 2)
    topk = geom_dsa.index_topk
    ramp_len = topk - c_prefix
    cap_len = (c_prefix + p_rank * 2) - topk
    dsa_pairs_per_rank = (
        float(ramp_len) * (float(c_prefix) + 0.5 * float(ramp_len))
        + float(cap_len * topk)
    ) / 2.0
    key_tokens_per_rank = float(c_prefix) / 2.0 + float(p_rank)
    exp_dsa_strict = (
        MATMUL_FLOP_PER_MAC * float(p_rank) * dsa_base
        + MATMUL_FLOP_PER_MAC * key_tokens_per_rank * ukv_elems
        + dsa_pairs_per_rank * 2.0 * score_w_mha * n_layers
    )
    assert dsa_mod.strict_fp8_flops == pytest.approx(exp_dsa_strict, rel=1e-9)


def test_pcp_kv_ag_uses_kv_and_latent_dims() -> None:
    """Verify pcp_kv_ag transfers KV or latent bytes instead of query heads."""
    tp = 2
    pcp = 4
    par = Parallel(tp=tp, pcp=pcp)
    shape = StepShape(
        batch=0, rows_per_seq=0, context=1024, prefill_tokens=128, prefix_hit_rate=0.0
    )
    global_pre_toks = 128.0
    num_heads = 32
    num_kv_heads = 4
    kvh_tp = num_kv_heads // tp

    def _pcp_kv_ag_payload(
        layer_type: str,
        dt: Dtypes,
        *,
        dims: tuple[int, int, int, int, int] = (128, 0, 0, 0, 0),
    ) -> float:
        """Extract single-layer pcp_kv_ag collective payload bytes."""
        head_dim, global_head_dim, sliding_window, kv_lora_rank, rope_dim = dims
        geom = ModelGeometry(
            hidden=2048,
            num_layers=1,
            layer_types=(layer_type,),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            attn_output_gate=False,
            linear_k_heads=0,
            linear_v_heads=0,
            linear_k_dim=0,
            linear_v_dim=0,
            conv_kernel=0,
            num_experts=0,
            top_k=0,
            moe_inter=0,
            shared_inter=0,
            dense_inter=4096,
            first_k_dense_replace=1,
            vocab=32000,
            mtp_layers=0,
            global_head_dim=global_head_dim,
            sliding_window=sliding_window,
            q_lora_rank=1024 if layer_type in ("mla", "dsa") else 0,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=128 if layer_type in ("mla", "dsa") else 0,
            qk_rope_head_dim=rope_dim,
            v_head_dim=128 if layer_type in ("mla", "dsa") else 0,
            index_n_heads=8 if layer_type == "dsa" else 0,
            index_head_dim=64 if layer_type == "dsa" else 0,
            index_topk=256 if layer_type == "dsa" else 0,
        )
        mods = {m.name: m for m in pass_costs(geom, par, shape, dt)}
        mod_key = (
            "gqa"
            if layer_type == "full_attention"
            else ("swa" if layer_type == "sliding_attention" else layer_type)
        )
        pcp_colls = [c for c in mods[mod_key].collectives if c.name == "pcp_kv_ag"]
        assert len(pcp_colls) == 1
        return pcp_colls[0].payload

    assert _pcp_kv_ag_payload(
        "full_attention", Dtypes(kv=1.0), dims=(128, 256, 0, 0, 0)
    ) == pytest.approx(global_pre_toks * float(2 * kvh_tp * 256) * 1.0)
    assert _pcp_kv_ag_payload(
        "full_attention", Dtypes(kv=2.0), dims=(128, 256, 0, 0, 0)
    ) == pytest.approx(global_pre_toks * float(2 * kvh_tp * 256) * 2.0)
    assert _pcp_kv_ag_payload(
        "sliding_attention", Dtypes(kv=1.0), dims=(128, 0, 128, 0, 0)
    ) == pytest.approx(global_pre_toks * float(2 * kvh_tp * 128) * 1.0)
    assert _pcp_kv_ag_payload(
        "mla", Dtypes(kv=1.0), dims=(128, 0, 0, 512, 64)
    ) == pytest.approx(global_pre_toks * float(512 + 64) * 1.0)
    assert _pcp_kv_ag_payload(
        "dsa", Dtypes(kv=2.0), dims=(128, 0, 0, 512, 64)
    ) == pytest.approx(global_pre_toks * float(512 + 64) * 2.0)


def test_moe_strict_bound_never_exceeds_serial_or_valid_schedule() -> None:
    """Verify EP busiest floor and strict LHS padding stay valid lower bounds."""
    moe_geom = ModelGeometry(
        hidden=2048,
        num_layers=2,
        layer_types=("full_attention", "full_attention"),
        num_heads=16,
        num_kv_heads=4,
        head_dim=128,
        attn_output_gate=False,
        linear_k_heads=0,
        linear_v_heads=0,
        linear_k_dim=0,
        linear_v_dim=0,
        conv_kernel=0,
        num_experts=16,
        top_k=1,
        moe_inter=512,
        shared_inter=0,
        dense_inter=0,
        first_k_dense_replace=0,
        vocab=32000,
        mtp_layers=0,
    )
    sf_k1 = step_floor(
        moe_geom,
        CHIPS["tpu7x"],
        Parallel(tp=1, ep=8, moe="ep", dp=1),
        StepShape(batch=1, rows_per_seq=1, context=512),
    )
    assert sf_k1.hbm_bound_s <= sf_k1.hbm_s
    assert sf_k1.hard_floor_s <= sf_k1.serial_estimate_s

    # ep_busiest_experts checks single-token upper cap and remainder pigeonhole min.
    non_div_e = 10
    non_div_ranks = 4
    busiest_k1 = ep_busiest_experts(non_div_e, 1, 1, non_div_ranks)
    assert busiest_k1 == pytest.approx(1.0, rel=1e-12)
    non_div_k_all = 10
    busiest_all = ep_busiest_experts(non_div_e, non_div_k_all, 1, non_div_ranks)
    pigeonhole_all = float(math.ceil(min(non_div_e, non_div_k_all) / non_div_ranks))
    assert busiest_all >= pigeonhole_all
    assert busiest_all == pytest.approx(pigeonhole_all, rel=1e-12)

    # M=33, K=8, E=16, L=32 allows a valid 9-tile (288-row) schedule.
    lhs_floor = 32
    m_rows = 33
    top_k = 8
    geom_m33 = dc_replace(moe_geom, top_k=top_k, moe_inter=512)
    dt = Dtypes(w_dense=1.0, w_expert=1.0, router=1.0, act="a8")
    moe_m33 = {
        m.name: m
        for m in pass_costs(
            geom_m33,
            Parallel(tp=1),
            StepShape(batch=m_rows, rows_per_seq=1, context=0),
            dt,
            mxu_lhs_rows_fp8=lhs_floor,
        )
    }["moe"]
    router_fp8 = (
        MATMUL_FLOP_PER_MAC
        * float(m_rows * geom_m33.num_layers)
        * float(geom_m33.hidden * geom_m33.num_experts)
    )
    routed_strict_fp8 = moe_m33.strict_fp8_flops - router_fp8
    per_row_routed_fp8 = MATMUL_FLOP_PER_MAC * float(
        geom_m33.num_layers * 3 * geom_m33.hidden * geom_m33.moe_inter
    )
    valid_9_tile_schedule_fp8 = 9.0 * float(lhs_floor) * per_row_routed_fp8
    assert routed_strict_fp8 == pytest.approx(
        float(m_rows * top_k) * per_row_routed_fp8, rel=1e-9
    )
    assert routed_strict_fp8 <= valid_9_tile_schedule_fp8


def test_latent_moe_dense_projections_unpadded() -> None:
    """Verify latent MoE dense up/down projections stay outside grouped padding."""
    latent_hidden = 256
    tp = 8
    batch_rows = 4
    lhs_floor = 32
    raw_text = QWEN35_CFG["text_config"]
    assert isinstance(raw_text, dict)
    geom = dc_replace(
        from_hf_config({**QWEN35_CFG, "text_config": {**raw_text}}),
        routed_expert_hidden=latent_hidden,
        shared_inter=1024,
    )
    n_moe = len(
        [
            idx
            for idx in range(geom.num_layers)
            if idx >= geom.first_k_dense_replace and geom.num_experts > 0
        ]
    )
    par = Parallel(tp=tp, moe="tp_slice")
    shape = StepShape(batch=batch_rows, rows_per_seq=1, context=0)
    dt = Dtypes(w_dense=1.0, w_expert=1.0, router=1.0, act="a8")

    moe_pad = {
        m.name: m
        for m in pass_costs(
            geom, par, shape, dt, mxu_m_tile=128, mxu_lhs_rows_fp8=lhs_floor
        )
    }["moe"]
    moe_unpad = {
        m.name: m
        for m in pass_costs(geom, par, shape, dt, mxu_m_tile=1, mxu_lhs_rows_fp8=1)
    }["moe"]

    routed_base_fp8 = (
        MATMUL_FLOP_PER_MAC
        * float(batch_rows * n_moe)
        * float(geom.top_k * 3 * latent_hidden)
        * (float(geom.moe_inter) / float(tp))
    )
    latent_fp8 = (
        MATMUL_FLOP_PER_MAC
        * float(batch_rows * n_moe)
        * float(2 * geom.hidden * latent_hidden / tp)
    )
    shared_fp8 = (
        MATMUL_FLOP_PER_MAC
        * float(batch_rows * n_moe)
        * float(3 * geom.hidden * (geom.shared_inter // tp))
    )
    router_fp8 = (
        MATMUL_FLOP_PER_MAC
        * float(batch_rows * n_moe)
        * float(geom.hidden * geom.num_experts)
    )
    strict_mult = float(lhs_floor) / float(batch_rows)
    assert moe_unpad.strict_fp8_flops == pytest.approx(
        routed_base_fp8 + latent_fp8 + shared_fp8 + router_fp8, rel=1e-9
    )
    assert moe_pad.strict_fp8_flops == pytest.approx(
        routed_base_fp8 * strict_mult + latent_fp8 + shared_fp8 + router_fp8, rel=1e-9
    )
    assert (moe_pad.strict_fp8_flops - moe_unpad.strict_fp8_flops) == pytest.approx(
        routed_base_fp8 * (strict_mult - 1.0), rel=1e-9
    )

    ed_nom = experts_touched(geom.num_experts, geom.top_k, batch_rows)
    m_per_touched = float(batch_rows * geom.top_k) / ed_nom
    nom_mult = float(math.ceil(m_per_touched / 128.0) * 128.0) / m_per_touched
    assert moe_pad.fp8_flops == pytest.approx(
        routed_base_fp8 * nom_mult + latent_fp8 + shared_fp8 + router_fp8, rel=1e-9
    )


def test_moe_lhs_floor_follows_precision_bucket() -> None:
    """Verify strict MoE LHS row floor tracks active matmul precision bucket."""
    tp = 8
    raw_text = QWEN35_CFG["text_config"]
    assert isinstance(raw_text, dict)
    geom = dc_replace(
        from_hf_config({**QWEN35_CFG, "text_config": {**raw_text}}),
        shared_inter=0,
    )
    n_moe = geom.num_layers
    par = Parallel(tp=tp, moe="tp_slice")
    shape = StepShape(batch=1, rows_per_seq=1, context=0)

    routed_base = (
        MATMUL_FLOP_PER_MAC
        * float(n_moe * geom.top_k * 3 * geom.hidden)
        * (float(geom.moe_inter) / float(tp))
    )
    router_base = MATMUL_FLOP_PER_MAC * float(n_moe * geom.hidden * geom.num_experts)

    bf16_floor_16 = 16
    bf16_floor_8 = 8
    fp8_floor_32 = 32
    fp8_floor_64 = 64

    dt_w8a16 = Dtypes(w_expert=FP8_BYTES, w_dense=2.0, router=2.0, act="a16")
    w8a16_default = {
        m.name: m
        for m in pass_costs(
            geom,
            par,
            shape,
            dt_w8a16,
            mxu_lhs_rows_bf16=bf16_floor_16,
            mxu_lhs_rows_fp8=fp8_floor_32,
        )
    }["moe"]
    w8a16_fp8_changed = {
        m.name: m
        for m in pass_costs(
            geom,
            par,
            shape,
            dt_w8a16,
            mxu_lhs_rows_bf16=bf16_floor_16,
            mxu_lhs_rows_fp8=fp8_floor_64,
        )
    }["moe"]
    w8a16_bf16_changed = {
        m.name: m
        for m in pass_costs(
            geom,
            par,
            shape,
            dt_w8a16,
            mxu_lhs_rows_bf16=bf16_floor_8,
            mxu_lhs_rows_fp8=fp8_floor_32,
        )
    }["moe"]

    assert w8a16_default.strict_fp8_flops == 0.0
    assert w8a16_default.strict_bf16_flops == pytest.approx(
        routed_base * float(bf16_floor_16) + router_base, rel=1e-9
    )
    assert w8a16_fp8_changed.strict_bf16_flops == pytest.approx(
        w8a16_default.strict_bf16_flops, rel=1e-12
    )
    assert w8a16_bf16_changed.strict_bf16_flops == pytest.approx(
        routed_base * float(bf16_floor_8) + router_base, rel=1e-9
    )

    dt_w8a8 = Dtypes(w_expert=FP8_BYTES, w_dense=2.0, router=2.0, act="a8")
    w8a8_default = {
        m.name: m
        for m in pass_costs(
            geom,
            par,
            shape,
            dt_w8a8,
            mxu_lhs_rows_bf16=bf16_floor_16,
            mxu_lhs_rows_fp8=fp8_floor_32,
        )
    }["moe"]
    w8a8_bf16_changed = {
        m.name: m
        for m in pass_costs(
            geom,
            par,
            shape,
            dt_w8a8,
            mxu_lhs_rows_bf16=bf16_floor_8,
            mxu_lhs_rows_fp8=fp8_floor_32,
        )
    }["moe"]
    assert w8a8_default.strict_fp8_flops == pytest.approx(
        routed_base * float(fp8_floor_32), rel=1e-9
    )
    assert w8a8_bf16_changed.strict_fp8_flops == pytest.approx(
        w8a8_default.strict_fp8_flops, rel=1e-12
    )


def test_sp_zero_expert_skips_logits_ag_and_linear_state_mixed_io() -> None:
    """Verify SP emits embed_rs, skips logits_ag on dense, tracks linear state."""
    tp_4 = 4
    dense_geom = ModelGeometry(
        hidden=1024,
        num_layers=2,
        layer_types=("full_attention", "full_attention"),
        num_heads=8,
        num_kv_heads=2,
        head_dim=128,
        attn_output_gate=False,
        linear_k_heads=0,
        linear_v_heads=0,
        linear_k_dim=0,
        linear_v_dim=0,
        conv_kernel=0,
        num_experts=0,
        top_k=0,
        moe_inter=0,
        shared_inter=0,
        dense_inter=2048,
        first_k_dense_replace=2,
        vocab=32000,
        mtp_layers=0,
    )
    sp_mods = {
        m.name: m
        for m in pass_costs(
            dense_geom,
            Parallel(tp=tp_4, residual="sp", moe_dispatch="allreduce"),
            StepShape(batch=4, rows_per_seq=1, context=128),
            Dtypes(),
        )
    }
    embed_names = [c.name for c in sp_mods["embed"].collectives]
    assert embed_names == ["embed_rs"]
    mlp_coll_names = [c.name for c in sp_mods["mlp"].collectives]
    assert "logits_ag" not in mlp_coll_names

    # Linear attention state IO covers decode seqs and skips unhit prefill prefix reads.
    linear_k_h = 4
    linear_v_h = 8
    linear_kd = 64
    linear_vd = 64
    conv_k = 4
    hybrid_geom = ModelGeometry(
        hidden=1024,
        num_layers=1,
        layer_types=("linear_attention",),
        num_heads=8,
        num_kv_heads=2,
        head_dim=128,
        attn_output_gate=False,
        linear_k_heads=linear_k_h,
        linear_v_heads=linear_v_h,
        linear_k_dim=linear_kd,
        linear_v_dim=linear_vd,
        conv_kernel=conv_k,
        num_experts=0,
        top_k=0,
        moe_inter=0,
        shared_inter=0,
        dense_inter=2048,
        first_k_dense_replace=1,
        vocab=32000,
        mtp_layers=0,
    )
    dt = Dtypes(w_dense=2.0, w_aux=2.0, state=4.0, conv=2.0)
    pre_toks = 16
    ctx = 64
    dec_batch = 2

    # Unhit prefill (batch=0, prefix_hit_rate=0.0): 1 state write, zero prefix read.
    unhit_mod = {
        m.name: m
        for m in pass_costs(
            hybrid_geom,
            Parallel(tp=1),
            StepShape(
                batch=0,
                rows_per_seq=0,
                prefill_tokens=pre_toks,
                context=ctx,
                prefix_hit_rate=0.0,
            ),
            dt,
        )
    }["gdn"]
    # Hit prefill (prefix_hit_rate=1.0): 1 read + 1 write = 2 state transfers.
    hit_mod = {
        m.name: m
        for m in pass_costs(
            hybrid_geom,
            Parallel(tp=1),
            StepShape(
                batch=0,
                rows_per_seq=0,
                prefill_tokens=pre_toks,
                context=ctx,
                prefix_hit_rate=1.0,
            ),
            dt,
        )
    }["gdn"]
    # Mixed step (batch=2 decode + 16 prefill, state_writes=1): 2*(1+1) + 1 state IOs.
    mixed_mod = {
        m.name: m
        for m in pass_costs(
            hybrid_geom,
            Parallel(tp=1),
            StepShape(
                batch=dec_batch,
                rows_per_seq=1,
                prefill_tokens=pre_toks,
                context=ctx,
                prefix_hit_rate=0.0,
                state_writes=1,
            ),
            dt,
        )
    }["gdn"]
    qkv_r = 2 * linear_k_h * linear_kd + linear_v_h * linear_vd
    one_state_bytes = (
        float(linear_v_h * linear_kd * linear_vd) * dt.state
        + float(qkv_r * (conv_k - 1)) * dt.conv
    )
    tol = 1e-12
    assert (hit_mod.hbm_bytes - unhit_mod.hbm_bytes) == pytest.approx(
        one_state_bytes, rel=tol
    )
    expected_mixed_extra_io = float(dec_batch * 2) * one_state_bytes
    assert (mixed_mod.hbm_bytes - unhit_mod.hbm_bytes) == pytest.approx(
        expected_mixed_extra_io, rel=tol
    )


_V5E_TEST_CTX = 128


def test_pass_costs_and_combine_modules_on_tpu5e_with_fp8_kv() -> None:
    """Verify pass_costs and combine_modules handle FP8 KV on TPU_V5E."""
    v5e_chip = rf_chips.from_jax("tpu5e", pltpu.ChipVersion.TPU_V5E)
    assert v5e_chip.fp8_flops == 0.0
    geom = from_hf_config(QWEN35_CFG)
    par = Parallel(tp=1)
    mods = pass_costs(
        geom,
        par,
        StepShape(batch=1, rows_per_seq=1, context=_V5E_TEST_CTX),
        Dtypes(w_dense=BF16_BYTES, w_expert=BF16_BYTES, kv=FP8_BYTES),
        chip=v5e_chip,
    )
    assert all(mod.strict_fp8_flops == 0.0 for mod in mods)
    sf = combine_modules(mods, v5e_chip, parallel=par)
    assert sf.hard_floor_s > 0.0
