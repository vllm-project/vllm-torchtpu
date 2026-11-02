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

from dataclasses import replace

import pytest

import vllm_torchtpu.roofline as rf
from vllm_torchtpu.roofline import (
    CHIPS,
    TOPOLOGIES,
    ChipSpec,
    Dtypes,
    ModelGeometry,
    Parallel,
    StepShape,
    experts_touched,
    from_hf_config,
    pass_costs,
    plan,
    spec_cycle_floor,
    step_floor,
    submesh_for_group,
    torus_collective_s,
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
    assert sf.mm_s == pytest.approx(6.91932419228308e-06, rel=1e-9)

    tiny_mm = ChipSpec("tiny_mm", 100, 1e18, 1e6, 1e6, 100)
    sf_mm = step_floor(
        geom, tiny_mm, Parallel(tp=8), StepShape(batch=1, context=10), DOC_DTYPES
    )
    assert sf_mm.binding == "matmul"


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
        acceptance_length=EXAMPLE_ACCEPT_LEN,
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
        batch=1,
        context=90036,
        k=3,
        dt=DOC_DTYPES,
    )
    assert sum(m.hbm_bytes for m in cyc.modules) == pytest.approx(
        7_741_562_248.75, abs=1.0
    )


def test_chain_term_binds_with_latency() -> None:
    geom = from_hf_config(QWEN35_CFG)
    shape = StepShape(batch=1, rows_per_seq=4, context=0)
    sf0 = step_floor(
        geom, CHIPS["tpu7x"], Parallel(tp=8), shape, EXAMPLE_FP8_DTYPES, 0.0
    )
    assert sf0.chain_s == pytest.approx(0.0001259664, rel=1e-6)
    assert sf0.binding == "hbm"
    sf1 = step_floor(
        geom,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        shape,
        EXAMPLE_FP8_DTYPES,
        collective_latency_s=EXAMPLE_COLLECTIVE_LATENCY_S,
    )
    expected_hops = sum(
        c.hops for m in sf0.modules for c in m.collectives if c.group > 1
    )
    assert sf1.chain_s == pytest.approx(
        sf0.chain_s + expected_hops * EXAMPLE_COLLECTIVE_LATENCY_S, rel=1e-6
    )
    assert sf1.hard_floor_s == sf0.hard_floor_s
    assert sf1.serial_floor_s == pytest.approx(
        sf0.serial_floor_s + expected_hops * EXAMPLE_COLLECTIVE_LATENCY_S, rel=1e-6
    )


def test_hf_config_quantization_dtypes_and_10_families() -> None:
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
    assert fp8_mods["gqa"].fp8_flops == pytest.approx(exact_1b_mods["gqa"].fp8_flops)
    assert fp8_mods["gqa"].fp8_flops > 8.0e11

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
    geom = from_hf_config(ten_family_cfg)
    assert geom.mtp_layers == 2
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
        assert expected in mods and mods[expected].hbm_bytes > 0.0
    cyc = spec_cycle_floor(
        geom,
        CHIPS["tpu7x"],
        Parallel(tp=4, moe="ep"),
        batch=2,
        context=2048,
        k=geom.mtp_layers,
        dt=EXAMPLE_FP8_DTYPES,
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
    with pytest.raises(ValueError, match="mutually exclusive"):
        Parallel(tp=8, kv_split=4, pcp=2)

    geom = from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    par_p1 = Parallel(tp=8, moe="ep", pcp=1)
    par_p4 = Parallel(tp=8, moe="ep", pcp=4)
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
        dt=EXAMPLE_FP8_DTYPES,
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
        dt=EXAMPLE_FP8_DTYPES,
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
        dt=EXAMPLE_FP8_DTYPES,
    )
    assert sf_hit.mm_s < sf_miss.mm_s * 0.20
    assert sf_pcp.mm_s < sf_miss.mm_s * 0.40

    sf_dec = step_floor(
        geom,
        chip,
        par_p1,
        StepShape(batch=16, rows_per_seq=1, context=8192),
        dt=EXAMPLE_FP8_DTYPES,
    )
    sf_mix = step_floor(
        geom,
        chip,
        par_p1,
        StepShape(batch=16, rows_per_seq=1, context=8192, prefill_tokens=2048),
        dt=EXAMPLE_FP8_DTYPES,
    )
    assert sf_mix.serial_floor_s > sf_dec.serial_floor_s * 1.5


def test_slice_topology_wraparound_and_d2d_ar_floor() -> None:
    assert set(TOPOLOGIES) == {
        "1x1x1",
        "1x2x1",
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
    sf_128 = step_floor(
        WIDE_GEOM,
        chip,
        par_ep128,
        StepShape(batch=4, context=1024),
        dt=EXAMPLE_FP8_DTYPES,
        topology=TOPOLOGIES["4x4x4"],
    )
    exp_chain_128 = sum(
        torus_collective_s(
            c.kind,
            c.payload,
            (2, 4, 1, 1) if c.group == 8 else (2, 4, 4, 4),
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
    geom = from_hf_config(QWEN35_CFG)
    par = Parallel(tp=8, kv_split=4, moe="ep")
    cyc = spec_cycle_floor(
        geom, CHIPS["tpu7x"], par, batch=4, context=4096, k=3, dt=DOC_DTYPES
    )
    ver_once = step_floor(
        geom,
        CHIPS["tpu7x"],
        par,
        StepShape(batch=4, rows_per_seq=4, context=4096, state_writes=1),
        DOC_DTYPES,
    )
    ver_k1 = step_floor(
        geom,
        CHIPS["tpu7x"],
        par,
        StepShape(batch=4, rows_per_seq=4, context=4096, state_writes=4),
        DOC_DTYPES,
    )
    gdn_cyc = next(m for m in cyc.modules if m.name == "gdn")
    gdn_once = next(m for m in ver_once.modules if m.name == "gdn")
    gdn_k1 = next(m for m in ver_k1.modules if m.name == "gdn")
    assert gdn_cyc.hbm_bytes == pytest.approx(gdn_once.hbm_bytes, abs=1e-6)
    assert gdn_cyc.hbm_bytes < gdn_k1.hbm_bytes


def test_dependent_collectives_latency_bound() -> None:
    geom = from_hf_config(QWEN35_CFG)
    inf_chip = replace(
        CHIPS["tpu7x"],
        hbm_bw=1e30,
        fp8_flops=1e30,
        bf16_flops=1e30,
        ici_bw=1e30,
        ici_link_bw=1e30,
    )
    par = Parallel(tp=8)
    shape = StepShape(batch=1, rows_per_seq=1, context=0)
    sf = step_floor(geom, inf_chip, par, shape, Dtypes(), topology=TOPOLOGIES["2x2x1"])
    assert sf.resource_bound_s < 1e-12
    assert sf.latency_bound_s > 0.0
    assert sf.hard_floor_s == pytest.approx(sf.latency_bound_s, rel=1e-12)
    assert sf.hard_floor_s == max(sf.resource_bound_s, sf.latency_bound_s, sf.chain_s)
    assert sf.binding == "chain"


def test_mla_and_ple_missing_field_validations() -> None:
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
    with pytest.raises(ValueError, match="dtype"):
        rf.dtypes_from_hf_config({"hidden_size": 4096})

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
