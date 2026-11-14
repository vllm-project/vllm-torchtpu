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
"""Unit tests for step floors, speculative cycles, and multihost chains."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

import vllm_torchtpu.roofline as rf
from vllm_torchtpu.roofline import (
    CHIPS,
    TOPOLOGIES,
    ChipSpec,
    Dtypes,
    FloorOptions,
    ModelGeometry,
    Parallel,
    SeqKvLayout,
    ServingRequest,
    SpecShape,
    StepShape,
    from_hf_config,
    pass_costs,
    plan,
    spec_cycle_floor,
    step_floor,
    torus_collective_s,
)
from vllm_torchtpu.roofline import constants as rf_constants
from vllm_torchtpu.roofline.collectives import submesh_for_group
from vllm_torchtpu.roofline.hf_loader import load_hf_config
from vllm_torchtpu.roofline.modules import experts_touched
from vllm_torchtpu.roofline.partition import partition_pp_layers

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

DOC_DTYPES = Dtypes(kv=2.0, w_aux=1.0)
EXAMPLE_FP8_DTYPES = Dtypes()
EXAMPLE_ACCEPT_LEN = 3.0
EXAMPLE_COLLECTIVE_LATENCY_S = 10e-6
WIDE_GEOM = ModelGeometry(
    hidden=8192,
    num_layers=4,
    layer_types=("linear_attention",) * 3 + ("full_attention",),
    num_heads=64,
    num_kv_heads=4,
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
    mtp_layers=0,
)
TP32 = Parallel(tp=32)


def test_hard_floor_is_max_of_resources() -> None:
    """Verify hard step floor selects maximum of resource, latency, and chain bounds."""
    geom = from_hf_config(QWEN35_CFG)
    sf = step_floor(
        geom,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        StepShape(batch=1, rows_per_seq=1, context=90036),
        FloorOptions(dt=DOC_DTYPES),
    )
    assert sf.hard_floor_s == pytest.approx(sf.resource_bound_s, rel=1e-12)
    assert sf.binding == "hbm"
    assert sf.hbm_s == pytest.approx(3_859_025_920 / 3.7e12, rel=1e-12)
    sf_unpad = step_floor(
        geom,
        replace(CHIPS["tpu7x"], mxu_m_tile=1),
        Parallel(tp=8),
        StepShape(batch=1, rows_per_seq=1, context=90036),
        FloorOptions(dt=DOC_DTYPES),
    )
    assert sf_unpad.matmul_s == pytest.approx(6.91932419228308e-06, rel=1e-9)
    assert sf.matmul_s == pytest.approx(0.00021617862158358742, rel=1e-9)

    tiny_mm = ChipSpec("tiny_mm", 100, 1e18, 1e6, 1e6, 100)
    sf_mm = step_floor(
        geom,
        tiny_mm,
        Parallel(tp=8),
        StepShape(batch=1, context=10),
        FloorOptions(dt=DOC_DTYPES),
    )
    assert sf_mm.binding == "matmul"


def test_spec_rows_and_tokens_per_step() -> None:
    """Verify speculative decode rows and tokens per step across concurrencies."""
    geom = from_hf_config(QWEN35_CFG)
    rows = plan(
        geom,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        DOC_DTYPES,
        ServingRequest(
            context=90036,
            concurrencies=[1, 4],
            spec_window=3,
            acceptance_length=EXAMPLE_ACCEPT_LEN,
        ),
    )
    assert [r.rows for r in rows] == [4, 16]
    assert rows[0].tpot_floor_ms == pytest.approx(
        rows[0].hard_floor_ms / EXAMPLE_ACCEPT_LEN, rel=1e-12
    )
    assert experts_touched(512, 10, 4) == pytest.approx(38.84330928325653, rel=1e-12)
    cyc = spec_cycle_floor(
        geom,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        SpecShape(batch=1, context=90036, k=3),
        FloorOptions(dt=DOC_DTYPES),
    )
    assert sum(m.hbm_bytes for m in cyc.modules) == pytest.approx(
        7_741_562_248.75, abs=1.0
    )


def test_chain_term_binds_with_latency() -> None:
    """Verify chain term binding under non-zero collective latency."""
    geom = from_hf_config(QWEN35_CFG)
    shape = StepShape(batch=1, rows_per_seq=4, context=0)
    sf0 = step_floor(
        geom,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        shape,
        FloorOptions(dt=EXAMPLE_FP8_DTYPES),
    )
    assert sf0.chain_s == pytest.approx(0.00090253728, rel=1e-6)
    assert sf0.binding == "hbm"
    sf1 = step_floor(
        geom,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        shape,
        FloorOptions(
            dt=EXAMPLE_FP8_DTYPES,
            collective_latency_s=EXAMPLE_COLLECTIVE_LATENCY_S,
        ),
    )
    expected_hops = sum(
        c.hops for m in sf0.modules for c in m.collectives if c.group > 1
    )
    assert sf1.chain_s == pytest.approx(
        sf0.chain_s + expected_hops * EXAMPLE_COLLECTIVE_LATENCY_S, rel=1e-6
    )
    assert sf1.hard_floor_s == sf0.hard_floor_s
    assert sf1.serial_estimate_s == pytest.approx(
        sf0.serial_estimate_s + expected_hops * EXAMPLE_COLLECTIVE_LATENCY_S, rel=1e-6
    )


def test_hf_config_quantization_dtypes_and_10_families() -> None:
    """Verify quantization dtype extraction across 10 layer families."""
    dt_fp8 = rf.dtypes_from_hf_config(
        {
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": ["lm_head", "mlp.gate", "mtp.fc"],
            },
        }
    )
    assert dt_fp8.w_dense == pytest.approx(1.0 + 4.0 / (128.0 * 128.0))
    assert dt_fp8.lm_head == pytest.approx(2.0)
    assert dt_fp8.w_aux == pytest.approx(2.0)
    # Block-scaled FP8 (1.000244 B < 2.0 B) routes linears to fp8_flops.
    fp8_mods = {
        m.name: m
        for m in pass_costs(
            from_hf_config(QWEN35_CFG),
            Parallel(tp=8),
            StepShape(
                batch=0, rows_per_seq=0, context=0, state_writes=0, prefill_tokens=2048
            ),
            dt_fp8,
        )
    }
    exact_1b_mods = {
        m.name: m
        for m in pass_costs(
            from_hf_config(QWEN35_CFG),
            Parallel(tp=8),
            StepShape(
                batch=0, rows_per_seq=0, context=0, state_writes=0, prefill_tokens=2048
            ),
            Dtypes(w_dense=1.0, w_expert=1.0, w_aux=2.0, act="a8"),
        )
    }
    min_fp8_flops = 8.0e11
    assert fp8_mods["gqa"].fp8_flops == pytest.approx(exact_1b_mods["gqa"].fp8_flops)
    assert fp8_mods["gqa"].fp8_flops > min_fp8_flops

    # linear_attn.in_proj_ba keeps dense FP8; lm_head stays bf16 unless enabled.
    dt_lin_ign = rf.dtypes_from_hf_config(
        {
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": ["model.layers.0.linear_attn.in_proj_ba"],
            },
        }
    )
    assert dt_lin_ign.w_dense == pytest.approx(1.0 + 4.0 / (128.0 * 128.0))
    assert dt_lin_ign.lm_head == pytest.approx(2.0)
    assert dt_lin_ign.w_aux == pytest.approx(2.0)
    dt_lm_q = rf.dtypes_from_hf_config(
        {
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "quantize_lm_head": True,
            },
        }
    )
    assert dt_lm_q.lm_head == pytest.approx(1.0 + 4.0 / (128.0 * 128.0))

    dt_modelopt = rf.dtypes_from_hf_config(
        {
            "torch_dtype": "bfloat16",
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
        }
    )
    assert dt_modelopt.w_expert == pytest.approx(0.5625)

    dt_nvfp4 = rf.dtypes_from_hf_config(
        {
            "torch_dtype": "bfloat16",
            "quantization_config": {"quant_method": "nvfp4", "group_size": 16},
        }
    )
    assert dt_nvfp4.w_expert == pytest.approx(0.5625)

    dt_mxfp4 = rf.dtypes_from_hf_config(
        {
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "mxfp4",
                "modules_to_not_convert": ["self_attn", "lm_head"],
            },
        }
    )
    assert dt_mxfp4.w_expert == pytest.approx(0.53125)
    assert dt_mxfp4.w_dense == pytest.approx(2.0)

    ten_family_cfg = {
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
        "linear_conv_kernel_dim": 4,
        "n_routed_experts": 64,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 512,
        "intermediate_size": 4096,
        "first_k_dense_replace": 1,
        "num_nextn_predict_layers": 2,
        "per_layer_input_dim": 256,
        "vocab_size": 32000,
    }
    expected_mtp_layers = 2
    geom = from_hf_config(ten_family_cfg)
    assert geom.mtp_layers == expected_mtp_layers
    mods = {
        m.name: m
        for m in pass_costs(
            geom,
            Parallel(tp=4, moe="ep"),
            StepShape(batch=2, rows_per_seq=1, context=2048),
            Dtypes(),
        )
    }
    for expected in ("gqa", "mla", "dsa", "swa", "gdn", "kda", "mlp", "moe", "ple"):
        assert expected in mods
        assert mods[expected].hbm_bytes > 0.0
    cyc = spec_cycle_floor(
        geom,
        CHIPS["tpu7x"],
        Parallel(tp=4, moe="ep"),
        SpecShape(batch=2, context=2048, k=geom.mtp_layers),
        FloorOptions(dt=EXAMPLE_FP8_DTYPES),
    )
    assert any(m.name == "draft_fc" for m in cyc.modules)

    with pytest.raises(ValueError, match="Unknown layer_type"):
        from_hf_config(
            {
                "hidden_size": 1024,
                "num_hidden_layers": 2,
                "layer_types": ["full_attention", "alien_block"],
                "num_attention_heads": 8,
                "vocab_size": 1000,
            }
        )


def test_pcp_dcp_mutual_exclusion_and_mixed_step() -> None:
    """Verify PCP/DCP exclusion and cost calculation for mixed steps."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        Parallel(tp=8, kv_split=4, pcp=2)

    geom = from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    par_p1 = Parallel(tp=8, moe="ep", pcp=1)
    par_p4 = Parallel(tp=8, moe="ep", pcp=4)
    opts_fp8 = FloorOptions(dt=EXAMPLE_FP8_DTYPES)
    sf_miss = step_floor(
        geom,
        chip,
        par_p1,
        StepShape(
            batch=0,
            rows_per_seq=0,
            state_writes=0,
            prefill_tokens=8192,
            prefix_hit_rate=0.0,
        ),
        opts_fp8,
    )
    sf_hit = step_floor(
        geom,
        chip,
        par_p1,
        StepShape(
            batch=0,
            rows_per_seq=0,
            state_writes=0,
            prefill_tokens=8192,
            prefix_hit_rate=0.92,
        ),
        opts_fp8,
    )
    sf_pcp = step_floor(
        geom,
        chip,
        par_p4,
        StepShape(
            batch=0,
            rows_per_seq=0,
            state_writes=0,
            prefill_tokens=8192,
            prefix_hit_rate=0.0,
        ),
        opts_fp8,
    )
    chip_unpad = replace(chip, mxu_m_tile=1)
    sf_miss_u = step_floor(
        geom,
        chip_unpad,
        par_p1,
        StepShape(
            batch=0,
            rows_per_seq=0,
            state_writes=0,
            prefill_tokens=8192,
            prefix_hit_rate=0.0,
        ),
        opts_fp8,
    )
    sf_hit_u = step_floor(
        geom,
        chip_unpad,
        par_p1,
        StepShape(
            batch=0,
            rows_per_seq=0,
            state_writes=0,
            prefill_tokens=8192,
            prefix_hit_rate=0.92,
        ),
        opts_fp8,
    )
    sf_pcp_u = step_floor(
        geom,
        chip_unpad,
        par_p4,
        StepShape(
            batch=0,
            rows_per_seq=0,
            state_writes=0,
            prefill_tokens=8192,
            prefix_hit_rate=0.0,
        ),
        opts_fp8,
    )
    assert sf_hit_u.matmul_s < sf_miss_u.matmul_s * 0.20
    assert sf_pcp_u.matmul_s < sf_miss_u.matmul_s * 0.40
    assert sf_hit.matmul_s < sf_miss.matmul_s
    assert sf_pcp.matmul_s < sf_miss.matmul_s

    sf_dec = step_floor(
        geom,
        chip,
        par_p1,
        StepShape(batch=16, rows_per_seq=1, context=8192),
        opts_fp8,
    )
    sf_mix = step_floor(
        geom,
        chip,
        par_p1,
        StepShape(batch=16, rows_per_seq=1, context=8192, prefill_tokens=2048),
        opts_fp8,
    )
    assert sf_mix.serial_estimate_s > sf_dec.serial_estimate_s * 1.5


def test_slice_topology_wraparound_and_d2d_ar_floor() -> None:
    """Verify slice topology torus hop counts, wraparound, and D2D all-reduce floors."""
    assert set(TOPOLOGIES) == {
        "2x2x1",
        "2x2x2",
        "2x2x4",
        "2x4x4",
        "4x4x4",
    }
    assert not TOPOLOGIES["2x2x4"].has_torus_wraparound
    assert not TOPOLOGIES["2x4x4"].has_torus_wraparound
    assert TOPOLOGIES["4x4x4"].has_torus_wraparound

    chip = CHIPS["tpu7x"]
    t_d2d = torus_collective_s("ar", 32 * 1024, (2, 1, 1, 1), chip, TOPOLOGIES["2x2x1"])
    t_host = torus_collective_s(
        "ar", 32 * 1024, (2, 2, 2, 1), chip, TOPOLOGIES["2x2x1"]
    )
    t_strided = torus_collective_s(
        "ar", 32 * 1024, (2, 1, 1, 4), chip, TOPOLOGIES["2x2x4"]
    )
    assert 0.0 < t_d2d < t_host < t_strided

    # Identical 8-rank submesh (2, 1, 1, 4): wrapped 4x4x4 gets 2 lanes vs 1.
    open_z4 = torus_collective_s(
        "ar", 64 * 1024 * 1024, (2, 1, 1, 4), chip, TOPOLOGIES["2x2x4"]
    )
    wrap_z4 = torus_collective_s(
        "ar", 64 * 1024 * 1024, (2, 1, 1, 4), chip, TOPOLOGIES["4x4x4"]
    )
    assert wrap_z4 < open_z4

    # EP128 on 4x4x4 with TP8 submesh (2, 4, 1, 1) packs EP against (2, 4, 4, 4).
    par_ep128 = Parallel(
        tp=8, dp=16, moe="ep", ep=128, moe_dispatch="ag_ep", mesh_axes=(2, 4, 1, 1)
    )
    tp_group_size = 8
    sf_128 = step_floor(
        WIDE_GEOM,
        chip,
        par_ep128,
        StepShape(batch=4, context=1024),
        FloorOptions(dt=EXAMPLE_FP8_DTYPES, topology=TOPOLOGIES["4x4x4"]),
    )
    exp_chain_128 = sum(
        torus_collective_s(
            c.kind,
            c.payload,
            (2, 4, 1, 1) if c.group == tp_group_size else (2, 4, 4, 4),
            chip,
            TOPOLOGIES["4x4x4"],
        )
        for m in sf_128.modules
        for c in m.collectives
        if c.group > 1
    )
    assert exp_chain_128 > 0.0
    assert sf_128.chain_s == pytest.approx(exp_chain_128, rel=1e-6)
    assert submesh_for_group(128, TOPOLOGIES["4x4x4"], 2, (2, 4, 1, 1)) == (2, 4, 4, 4)
    with pytest.raises(ValueError, match="exceeds topology"):
        submesh_for_group(64, TOPOLOGIES["2x2x2"], 2)


def test_spec_cycle_verify_writes_state_once() -> None:
    """Verify speculative verify step performs state writes exactly once."""
    geom = from_hf_config(QWEN35_CFG)
    par = Parallel(tp=8, kv_split=4, moe="ep")
    opts_doc = FloorOptions(dt=DOC_DTYPES)
    cyc = spec_cycle_floor(
        geom, CHIPS["tpu7x"], par, SpecShape(batch=4, context=4096, k=3), opts_doc
    )
    ver_once = step_floor(
        geom,
        CHIPS["tpu7x"],
        par,
        StepShape(batch=4, rows_per_seq=4, context=4096, state_writes=1),
        opts_doc,
    )
    ver_k1 = step_floor(
        geom,
        CHIPS["tpu7x"],
        par,
        StepShape(batch=4, rows_per_seq=4, context=4096, state_writes=4),
        opts_doc,
    )
    gdn_cyc = next(m for m in cyc.modules if m.name == "gdn")
    gdn_once = next(m for m in ver_once.modules if m.name == "gdn")
    gdn_k1 = next(m for m in ver_k1.modules if m.name == "gdn")
    assert gdn_cyc.hbm_bytes == pytest.approx(gdn_once.hbm_bytes, abs=1e-6)
    assert gdn_cyc.hbm_bytes < gdn_k1.hbm_bytes


def test_assumed_hop_latency_is_reported_outside_the_bound() -> None:
    """Verify assumed hop latency is reported outside the hard resource bound."""
    geom = from_hf_config(QWEN35_CFG)
    inf_chip = replace(
        CHIPS["tpu7x"],
        hbm_bw=1e30,
        fp8_flops=1e30,
        bf16_flops=1e30,
        vpu_flops=1e30,
        ici_bw=1e30,
        ici_link_bw=1e30,
    )
    par = Parallel(tp=8)
    shape = StepShape(batch=1, rows_per_seq=1, context=0)
    sf = step_floor(
        geom,
        inf_chip,
        par,
        shape,
        FloorOptions(topology=TOPOLOGIES["2x2x1"]),
    )
    assert sf.latency_assumed_s > 0.0
    assert sf.serial_estimate_s >= sf.latency_assumed_s
    assert sf.coll_bound_s < sf.latency_assumed_s
    assert sf.hard_floor_s < sf.latency_assumed_s


def test_mla_and_ple_missing_field_validations() -> None:
    """Verify MLA and PLE config field validations raise informative ValueError."""
    bad_mla_kv = ModelGeometry(
        hidden=4096,
        num_layers=2,
        layer_types=("mla", "mla"),
        num_heads=32,
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
        vocab=1000,
        mtp_layers=0,
        kv_lora_rank=0,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
    )
    with pytest.raises(ValueError, match="kv_lora_rank"):
        pass_costs(
            bad_mla_kv,
            Parallel(tp=1),
            StepShape(batch=1, rows_per_seq=1),
            EXAMPLE_FP8_DTYPES,
        )

    bad_ple = ModelGeometry(
        hidden=4096,
        num_layers=2,
        layer_types=("ple",),
        num_heads=32,
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
        vocab=1000,
        mtp_layers=0,
        ple_dim=0,
    )
    with pytest.raises(ValueError, match="ple_dim"):
        pass_costs(
            bad_ple,
            Parallel(tp=1),
            StepShape(batch=1, rows_per_seq=1),
            EXAMPLE_FP8_DTYPES,
        )


def test_hf_loader_dtype_validations() -> None:
    """Verify HF loader dtype and quantization config error validations."""
    expected_w_dense = 1.0
    expected_lm_head = 2.0
    expected_fp32_state = 4.0
    with pytest.raises(ValueError, match="dtype"):
        rf.dtypes_from_hf_config({"hidden_size": 4096})

    dt_q_default = rf.dtypes_from_hf_config(
        {"quantization_config": {"quant_method": "fp8"}}
    )
    assert dt_q_default.w_dense == expected_w_dense
    assert dt_q_default.lm_head == expected_lm_head
    assert dt_q_default.state == pytest.approx(expected_fp32_state)

    dt_awq = rf.dtypes_from_hf_config(
        {
            "dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "scale_dtype": "bfloat16",
            },
        }
    )
    assert dt_awq.act == "a16"
    assert dt_awq.state == pytest.approx(expected_fp32_state)

    with pytest.raises(ValueError, match="Unknown base dtype"):
        rf.dtypes_from_hf_config({"dtype": "unknown_type"})

    with pytest.raises(ValueError, match="group_size"):
        rf.dtypes_from_hf_config(
            {
                "dtype": "bfloat16",
                "quantization_config": {"bits": 4},
            }
        )

    with pytest.raises(ValueError, match="scale_bytes"):
        rf.dtypes_from_hf_config(
            {
                "dtype": "bfloat16",
                "quantization_config": {"bits": 4, "group_size": 32},
            }
        )


def test_serving_plan_dp_chip_normalization() -> None:
    """Verify serving.plan normalizes throughput by all DP/PP/TP chip ranks."""
    geom = from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    rows_dp2 = plan(
        geom,
        chip,
        Parallel(tp=1, dp=2, ranks_per_chip=2),
        Dtypes(),
        ServingRequest(context=512, concurrencies=[1]),
    )
    c1 = rows_dp2[0]
    expected = (1 * 1.0) / (c1.hard_floor_ms / 1e3) / 1.0
    assert c1.output_tput_ceil_per_chip == pytest.approx(expected, rel=1e-6)


def test_topology_none_uses_torus_collective_schedule() -> None:
    """Verify topology=None prices collectives via torus_collective_s."""
    geom = from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    par = Parallel(tp=8)
    shape = StepShape(batch=1, rows_per_seq=4, context=0)
    sf_none = step_floor(geom, chip, par, shape, FloorOptions(dt=EXAMPLE_FP8_DTYPES))
    expected_chain_s = sum(
        torus_collective_s(
            c.kind,
            c.payload,
            submesh_for_group(c.group, None, par.ranks_per_chip, par.mesh_axes),
            chip,
            None,
        )
        for m in sf_none.modules
        for c in m.collectives
        if c.group > 1
    )
    assert sf_none.chain_s == pytest.approx(expected_chain_s, rel=1e-9)


def test_block_parallel_and_multilayer_mtp_draft_cost() -> None:
    """Verify block-parallel and multi-layer MTP draft module costing."""
    synth_moe_mtp_cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 8,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_experts": 64,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 1024,
        "mtp_num_hidden_layers": 2,
        "mtp_intermediate_size": 6144,
        "block_parallel": True,
        "vocab_size": 64000,
    }
    geom_bp = from_hf_config(synth_moe_mtp_cfg)
    geom_seq = replace(geom_bp, block_parallel=1)
    exp_block_parallel = 2
    exp_mtp_inter = 6144
    assert geom_bp.block_parallel == exp_block_parallel
    assert geom_bp.mtp_inter == exp_mtp_inter

    par = Parallel(tp=8)
    dt = Dtypes()
    opts_dt = FloorOptions(dt=dt)
    mods_seq = pass_costs(geom_seq, par, StepShape(batch=4, rows_per_seq=1), dt)
    mods_bp = pass_costs(geom_bp, par, StepShape(batch=4, rows_per_seq=1), dt)
    exp_coll_diff = 8
    assert (
        sum(m.n_collectives for m in mods_seq) - sum(m.n_collectives for m in mods_bp)
        == exp_coll_diff
    )

    spec = SpecShape(batch=4, context=2048, k=2)
    cyc_dense_mtp = spec_cycle_floor(geom_bp, CHIPS["tpu7x"], par, spec, opts_dt)
    cyc_moe_mtp = spec_cycle_floor(
        replace(geom_bp, mtp_inter=0),
        CHIPS["tpu7x"],
        par,
        spec,
        opts_dt,
    )
    assert any(m.name == "draft_mlp" for m in cyc_dense_mtp.modules)
    assert not any(m.name == "draft_moe" for m in cyc_dense_mtp.modules)
    assert any(m.name == "draft_moe" for m in cyc_moe_mtp.modules)

    cyc_mtp1 = spec_cycle_floor(
        replace(geom_bp, mtp_layers=1),
        CHIPS["tpu7x"],
        par,
        spec,
        opts_dt,
    )
    draft_mlp_2 = next(m for m in cyc_dense_mtp.modules if m.name == "draft_mlp")
    draft_mlp_1 = next(m for m in cyc_mtp1.modules if m.name == "draft_mlp")
    draft_fc_2 = next(m for m in cyc_dense_mtp.modules if m.name == "draft_fc")
    draft_fc_1 = next(m for m in cyc_mtp1.modules if m.name == "draft_fc")
    assert draft_mlp_2.hbm_bytes == pytest.approx(draft_mlp_1.hbm_bytes, rel=1e-9)
    assert draft_fc_2.hbm_bytes == pytest.approx(draft_fc_1.hbm_bytes, rel=1e-9)

    seq_layout = SeqKvLayout(1, 1024)
    bd_mtp2 = rf.hbm_breakdown_per_core_bytes(geom_bp, dt, par, seq_layout)
    bd_mtp1 = rf.hbm_breakdown_per_core_bytes(
        replace(geom_bp, mtp_layers=1), dt, par, seq_layout
    )
    bd_mtp0 = rf.hbm_breakdown_per_core_bytes(
        replace(geom_bp, mtp_layers=0), dt, par, seq_layout
    )
    assert (bd_mtp2.weights - bd_mtp0.weights) == pytest.approx(
        2.0 * (bd_mtp1.weights - bd_mtp0.weights), rel=1e-9
    )

    bd_moe = rf.hbm_breakdown_per_core_bytes(
        replace(geom_bp, mtp_inter=0), dt, par, seq_layout
    )
    assert bd_mtp2.weights < bd_moe.weights


def test_bound_never_exceeds_estimate_across_families() -> None:
    """Verify cut/unpadded bounds stay <= step resource times and estimates."""
    chip = CHIPS["tpu7x"]
    chip_hi = replace(chip, ici_hop_latency_s=100.0e-6)
    mla_cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 16,
        "layer_types": ["mla"] * 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "q_lora_rank": 1024,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "intermediate_size": 8192,
        "vocab_size": 32000,
    }
    dense_cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 12288,
        "vocab_size": 32000,
    }
    cases = (
        (
            from_hf_config(QWEN35_CFG),
            Parallel(tp=8, moe="ep", moe_dispatch="ag_ep"),
            StepShape(batch=8, rows_per_seq=1, context=8192),
            rf.dtypes_from_hf_config(QWEN35_CFG),
        ),
        (
            from_hf_config(dense_cfg),
            Parallel(tp=8),
            StepShape(batch=8, rows_per_seq=1, context=4096),
            Dtypes(),
        ),
        (
            from_hf_config(mla_cfg),
            Parallel(tp=8),
            StepShape(batch=8, rows_per_seq=1, context=4096),
            Dtypes(kv=1.0),
        ),
    )
    topo = TOPOLOGIES["2x2x1"]
    for geom, par, shape, dt in cases:
        sf = step_floor(geom, chip, par, shape, FloorOptions(dt=dt, topology=topo))
        sf_hi = step_floor(
            geom, chip_hi, par, shape, FloorOptions(dt=dt, topology=topo)
        )
        assert sf.hbm_bound_s <= sf.hbm_s + 1e-12
        assert sf.matmul_bound_s <= sf.matmul_s + 1e-12
        assert sf.coll_bound_s <= sf.chain_s + 1e-12
        assert sf.hard_floor_s == pytest.approx(
            max(sf.hbm_bound_s, sf.matmul_bound_s, sf.vpu_bound_s, sf.coll_bound_s),
            rel=1e-12,
        )
        assert sf.hard_floor_s <= sf.serial_estimate_s + 1e-12
        assert sf_hi.hard_floor_s >= sf.hard_floor_s - 1e-12
        if sf.lat_s > 0.0:
            assert sf_hi.latency_assumed_s > sf.latency_assumed_s


def test_single_expert_routing_cannot_beat_bound() -> None:
    """Verify best-case single-expert routing time at batch 8 is >= hard_floor_s."""
    geom = from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    dt = rf.dtypes_from_hf_config(QWEN35_CFG)
    shape = StepShape(batch=8, rows_per_seq=1, context=90036)
    sf = step_floor(
        geom,
        chip,
        Parallel(tp=8),
        shape,
        FloorOptions(dt=dt, topology=TOPOLOGIES["2x2x1"]),
    )
    single_expert_hbm_s = sum(m.strict_hbm_bytes for m in sf.modules) / chip.hbm_bw
    assert single_expert_hbm_s == pytest.approx(0.004269709145945946, rel=1e-9)
    assert single_expert_hbm_s >= sf.hard_floor_s - 1e-15
    # Concentrated top-10 routing at batch=8 pads each active expert
    # to 32 FP8 LHS rows (4x).
    expected_matmul_bound_s = (
        sum(m.strict_fp8_flops for m in sf.modules) / chip.fp8_flops
        + sum(m.strict_bf16_flops for m in sf.modules) / chip.bf16_flops
    )
    assert sf.matmul_bound_s == pytest.approx(expected_matmul_bound_s, rel=1e-9)
    unpadded_matmul_bound_s = 6.861009232900433e-05
    assert sf.matmul_bound_s > unpadded_matmul_bound_s


def test_bound_never_exceeds_estimate_without_topology() -> None:
    """Verify topology=None keeps chain_s and serial_estimate_s above cut bounds."""
    chip = CHIPS["tpu7x"]
    par = Parallel(tp=8, ar_algo="two_shot")
    shape = StepShape(batch=28, rows_per_seq=4, context=0)
    dense_cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 12288,
        "vocab_size": 32000,
    }
    for cfg in (dense_cfg, QWEN35_CFG):
        sf = step_floor(
            from_hf_config(cfg), chip, par, shape, FloorOptions(dt=Dtypes())
        )
        assert sf.coll_bound_s <= sf.chain_s
        assert sf.hard_floor_s <= sf.serial_estimate_s


def test_hf_loader_weight_only_and_local_path_fail_closed(
    tmp_path: Path,
) -> None:
    """Verify weight-only quant defaults to a16 and missing local paths fail fast."""
    dt_ct_w8 = rf.dtypes_from_hf_config(
        {
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "weights": {"num_bits": 8, "type": "int"},
                        "input_activations": None,
                    }
                },
            },
        }
    )
    assert dt_ct_w8.act == "a16"
    missing_json = tmp_path / "missing" / "config.json"
    with pytest.raises(FileNotFoundError, match="Could not resolve HF config path"):
        rf.load_hf_config(str(missing_json), offline=False)


def test_partition_pp_layers_rejects_infeasible_disagg_hybrid() -> None:
    """Verify disagg PP partitioning rejects hybrid models with too few full layers."""
    hybrid_end_cfg = {
        "hidden_size": 2048,
        "num_hidden_layers": 6,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "full_attention",
        ],
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "linear_num_key_heads": 8,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "intermediate_size": 4096,
        "vocab_size": 32000,
    }
    geom = from_hf_config(hybrid_end_cfg)
    with pytest.raises(ValueError, match="full-attention"):
        partition_pp_layers(geom, Dtypes(), Parallel(tp=1, pp=3), disagg=True)


def test_load_hf_config_offline_hub_cache_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify load_hf_config resolves from local HF_HUB_CACHE offline snapshot."""
    snapshot_dir = (
        tmp_path / "models--TestOrg--TestModel" / "snapshots" / "test_snapshot_hash"
    )
    snapshot_dir.mkdir(parents=True)
    cfg_file = snapshot_dir / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "intermediate_size": 5632,
                "num_hidden_layers": 28,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    loaded = load_hf_config("TestOrg/TestModel", offline=True)
    assert loaded["model_type"] == "qwen2"
    expected_hidden_size = 2048
    assert loaded["hidden_size"] == expected_hidden_size


def test_load_hf_config_rejects_multi_segment_relative_path() -> None:
    """Verify load_hf_config rejects relative paths with != 2 segments as repo IDs."""
    with pytest.raises(FileNotFoundError):
        load_hf_config("foo/bar/baz")
    with pytest.raises(FileNotFoundError):
        load_hf_config("single_segment")
    with pytest.raises(FileNotFoundError):
        load_hf_config("./relative/path/config.json")


def test_serving_plan_divides_by_ranks_per_chip() -> None:
    """Verify serving.plan normalizes chip count by parallel.ranks_per_chip."""
    geom = from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    req = ServingRequest(context=512, concurrencies=[1])
    rows_r1 = plan(
        geom,
        chip,
        Parallel(tp=2, dp=2, ranks_per_chip=1),
        Dtypes(),
        req,
    )
    rows_r2 = plan(
        geom,
        chip,
        Parallel(tp=2, dp=2, ranks_per_chip=2),
        Dtypes(),
        req,
    )
    assert rows_r2[0].output_tput_ceil_per_chip == pytest.approx(
        rows_r1[0].output_tput_ceil_per_chip * 2.0, rel=1e-6
    )


def test_coll_bound_is_the_cut_bound_sum() -> None:
    """Verify coll_bound_s is the cut bound sum and ignores hop latency."""
    geom = from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    par = Parallel(tp=8)
    shape = StepShape(batch=1, rows_per_seq=1, context=0)
    opts = FloorOptions(
        dt=EXAMPLE_FP8_DTYPES,
        topology=TOPOLOGIES["2x2x1"],
        collective_latency_s=50e-6,
    )
    sf = step_floor(geom, chip, par, shape, opts)
    sf_zero_hop = step_floor(
        geom, replace(chip, ici_hop_latency_s=0.0), par, shape, opts
    )
    assert sf.coll_bound_s > 0.0
    assert sf.hard_floor_s >= sf.coll_bound_s - 1e-12
    assert sf.coll_bound_s == pytest.approx(sf_zero_hop.coll_bound_s, rel=1e-12)


def test_hard_floor_ignores_assumed_hop_latency() -> None:
    """Verify hard_floor_s and coll_bound_s ignore assumed hop latency."""
    geom = from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    par = Parallel(tp=8)
    shape = StepShape(batch=1, rows_per_seq=1, context=0)
    opts = FloorOptions(topology=TOPOLOGIES["2x2x1"])
    sf_default = step_floor(geom, chip, par, shape, opts)
    slow_chip = replace(chip, ici_hop_latency_s=1.0)
    sf_slow = step_floor(geom, slow_chip, par, shape, opts)
    assert sf_slow.hard_floor_s == pytest.approx(sf_default.hard_floor_s, rel=1e-12)
    assert sf_slow.coll_bound_s == pytest.approx(sf_default.coll_bound_s, rel=1e-12)
    assert sf_slow.serial_estimate_s > sf_default.serial_estimate_s


def test_vector_bound_uses_packed_element_rate() -> None:
    """Verify vpu_bound_s divides strict VPU FLOPs by packed element rate."""
    geom = from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    par = Parallel(tp=8)
    shape = StepShape(batch=1, rows_per_seq=1, context=0)
    sf = step_floor(geom, chip, par, shape, FloorOptions(dt=Dtypes(w_expert=0.5)))
    expected_vpu_bound = sum(m.strict_vpu_flops for m in sf.modules) / (
        rf_constants.VECTOR_MAX_ELEMENTS_PER_WORD * chip.vpu_flops
    )
    assert sf.vpu_bound_s == pytest.approx(expected_vpu_bound)


def test_strict_counts_ignore_kernel_row_tile() -> None:
    """Verify strict matmul, vector, HBM and hard floors ignore mxu_m_tile."""
    geometry = from_hf_config(QWEN35_CFG)
    parallel = Parallel(tp=8)
    shape = StepShape(batch=1, rows_per_seq=1, context=0)
    chips_by_tile = {
        tile: replace(CHIPS["tpu7x"], mxu_m_tile=tile) for tile in (1, 256, 512)
    }

    floors_moe = {
        tile: step_floor(geometry, chip, parallel, shape)
        for tile, chip in chips_by_tile.items()
    }
    for field_name in (
        "matmul_bound_s",
        "vpu_bound_s",
        "hbm_bound_s",
        "hard_floor_s",
    ):
        values = [getattr(floors_moe[tile], field_name) for tile in (1, 256, 512)]
        assert values[0] == pytest.approx(values[1], rel=1e-12)
        assert values[1] == pytest.approx(values[2], rel=1e-12)

    llama_config = {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "num_attention_heads": 32,
        "intermediate_size": 11008,
        "vocab_size": 32000,
    }
    dense_geometry = from_hf_config(llama_config)
    floors_dense = {
        tile: step_floor(dense_geometry, chip, parallel, shape)
        for tile, chip in chips_by_tile.items()
    }
    for field_name in (
        "matmul_bound_s",
        "vpu_bound_s",
        "hbm_bound_s",
        "hard_floor_s",
    ):
        values = [getattr(floors_dense[tile], field_name) for tile in (1, 256, 512)]
        assert values[0] == pytest.approx(values[1], rel=1e-12)
        assert values[1] == pytest.approx(values[2], rel=1e-12)


def test_vpu_floor_and_lhs_row_tiles_bound_step() -> None:
    """Verify vpu_bound_s and padded LHS row tiles bound the step floor."""
    geom = from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    par = Parallel(tp=8)
    shape = StepShape(batch=1, rows_per_seq=1, context=0)
    sf = step_floor(geom, chip, par, shape, FloorOptions(dt=EXAMPLE_FP8_DTYPES))
    assert sf.vpu_s > 0.0
    assert sf.vpu_bound_s > 0.0
    assert sf.hard_floor_s >= sf.vpu_bound_s - 1e-12
    moe_mod = next(m for m in sf.modules if m.name == "moe")
    assert moe_mod.strict_fp8_flops > 0.0
    assert moe_mod.strict_vpu_flops > 0.0
