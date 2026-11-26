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
from pathlib import Path

import pytest

import vllm_torchtpu.roofline as rf
from tests.roofline.conftest import (
    LOCKED_FIXTURE_FILES,
    load_locked_bundle,
    load_locked_config,
)
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
from vllm_torchtpu.roofline.collectives import CollectiveConfig, make, submesh_for_group
from vllm_torchtpu.roofline.geometry import _from_hf_config_dry_run
from vllm_torchtpu.roofline.modules import (
    ModuleCost,
    _dsa_indexer_layer_weight_elems_strict,
    dsa_indexer_layer_weight_elems,
    experts_touched,
)
from vllm_torchtpu.roofline.occupancy import ep_busiest_experts
from vllm_torchtpu.roofline.step import combine_modules

DOC_DTYPES = Dtypes(kv=2.0, w_aux=1.0)
EXAMPLE_FP8_DTYPES = Dtypes()
EXAMPLE_ACCEPT_LEN = 3.0
EXAMPLE_COLLECTIVE_LATENCY_S = 1e-05
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


def test_hard_floor_is_max_of_resources() -> None:
    """Verify hard step floor selects maximum of resource, latency, and chain bounds."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json"))
    floor_result = step_floor(
        geometry,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        StepShape(batch=1, rows_per_seq=1, context=90036),
        FloorOptions(dtypes=DOC_DTYPES),
    )
    assert floor_result.hard_floor_s == pytest.approx(
        floor_result.resource_bound_s, rel=1e-12
    )
    assert floor_result.binding == "hbm"
    assert floor_result.hbm_s == pytest.approx(3859025920 / 3700000000000.0, rel=1e-12)
    sf_unpad = step_floor(
        geometry,
        replace(CHIPS["tpu7x"], mxu_m_tile=1),
        Parallel(tp=8),
        StepShape(batch=1, rows_per_seq=1, context=90036),
        FloorOptions(dtypes=DOC_DTYPES),
    )
    assert sf_unpad.matmul_s == pytest.approx(6.91932419228308e-06, rel=1e-09)
    assert floor_result.matmul_s == pytest.approx(0.00021617862158358742, rel=1e-09)
    tiny_mm = ChipSpec("tiny_mm", 100, 1e18, 1000000.0, 1000000.0, 100)
    sf_mm = step_floor(
        geometry,
        tiny_mm,
        Parallel(tp=8),
        StepShape(batch=1, context=10),
        FloorOptions(dtypes=DOC_DTYPES),
    )
    assert sf_mm.binding == "matmul"


def test_spec_rows_and_tokens_per_step() -> None:
    """Verify speculative decode rows and tokens per step across concurrencies."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json"))
    rows = plan(
        geometry,
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
        geometry,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        SpecShape(batch=1, context=90036, spec_k=3),
        FloorOptions(dtypes=DOC_DTYPES),
    )
    assert sum(m.hbm_bytes for m in cyc.modules) == pytest.approx(
        7741562248.75, abs=1.0
    )


def test_chain_term_binds_with_latency() -> None:
    """Verify chain term binding under non-zero collective latency."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json"))
    shape = StepShape(batch=1, rows_per_seq=4, context=0)
    sf0 = step_floor(
        geometry,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        shape,
        FloorOptions(dtypes=EXAMPLE_FP8_DTYPES),
    )
    assert sf0.chain_s == pytest.approx(0.00086420352, rel=1e-06)
    assert sf0.binding == "hbm"
    sf1 = step_floor(
        geometry,
        CHIPS["tpu7x"],
        Parallel(tp=8),
        shape,
        FloorOptions(
            dtypes=EXAMPLE_FP8_DTYPES, collective_latency_s=EXAMPLE_COLLECTIVE_LATENCY_S
        ),
    )
    expected_hops = sum(
        c.hops for m in sf0.modules for c in m.collectives if c.group > 1
    )
    assert sf1.chain_s == pytest.approx(
        sf0.chain_s + expected_hops * EXAMPLE_COLLECTIVE_LATENCY_S, rel=1e-06
    )
    assert sf1.hard_floor_s == sf0.hard_floor_s
    assert sf1.serial_estimate_s == pytest.approx(
        sf0.serial_estimate_s + expected_hops * EXAMPLE_COLLECTIVE_LATENCY_S, rel=1e-06
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
    fp8_mods = {
        m.name: m
        for m in pass_costs(
            from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json")),
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
            from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json")),
            Parallel(tp=8),
            StepShape(
                batch=0, rows_per_seq=0, context=0, state_writes=0, prefill_tokens=2048
            ),
            Dtypes(w_dense=1.0, w_expert=1.0, w_aux=2.0, act="a8"),
        )
    }
    min_fp8_flops = 800000000000.0
    assert fp8_mods["gqa"].fp8_flops == pytest.approx(exact_1b_mods["gqa"].fp8_flops)
    assert fp8_mods["gqa"].fp8_flops > min_fp8_flops
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
        "model_type": "synthetic_ten_family",
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
    geometry = _from_hf_config_dry_run(ten_family_cfg)
    assert geometry.mtp_layers == expected_mtp_layers
    mods = {
        m.name: m
        for m in pass_costs(
            geometry,
            Parallel(tp=4, moe="ep"),
            StepShape(batch=2, rows_per_seq=1, context=2048),
            Dtypes(),
        )
    }
    for expected in ("gqa", "mla", "dsa", "swa", "gdn", "kda", "mlp", "moe", "ple"):
        assert expected in mods
        assert mods[expected].hbm_bytes > 0.0
    cyc = spec_cycle_floor(
        geometry,
        CHIPS["tpu7x"],
        Parallel(tp=4, moe="ep"),
        SpecShape(batch=2, context=2048, spec_k=geometry.mtp_layers),
        FloorOptions(dtypes=EXAMPLE_FP8_DTYPES),
    )
    assert any(m.name == "draft_fc" for m in cyc.modules)
    with pytest.raises(ValueError, match="Unknown layer_type"):
        from_hf_config(
            {
                "model_type": "qwen3_5_moe",
                "hidden_size": 1024,
                "num_hidden_layers": 2,
                "layer_types": ["full_attention", "alien_block"],
                "num_attention_heads": 8,
                "vocab_size": 1000,
            }
        )


def test_pcp_dcp_mutual_exclusion_and_mixed_step() -> None:
    """Verify PCP+DCP coexistence shards prefill KV and mixed steps price cleanly."""
    pcp_dcp = Parallel(tp=8, kv_split=4, pcp=2)
    assert (pcp_dcp.pcp, pcp_dcp.kv_split) == (2, 4)
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json"))
    chip = CHIPS["tpu7x"]
    par_p1 = Parallel(tp=8, moe="ep", pcp=1)
    par_p4 = Parallel(tp=8, moe="ep", pcp=4)
    opts_fp8 = FloorOptions(dtypes=EXAMPLE_FP8_DTYPES)
    sf_miss = step_floor(
        geometry,
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
        geometry,
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
        geometry,
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
        geometry,
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
        geometry,
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
        geometry,
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
    assert sf_hit_u.matmul_s < sf_miss_u.matmul_s * 0.2
    assert sf_pcp_u.matmul_s < sf_miss_u.matmul_s * 0.4
    assert sf_hit.matmul_s < sf_miss.matmul_s
    assert sf_pcp.matmul_s < sf_miss.matmul_s
    decode_step_floor = step_floor(
        geometry,
        chip,
        par_p1,
        StepShape(batch=16, rows_per_seq=1, context=8192),
        opts_fp8,
    )
    sf_mix = step_floor(
        geometry,
        chip,
        par_p1,
        StepShape(batch=16, rows_per_seq=1, context=8192, prefill_tokens=2048),
        opts_fp8,
    )
    assert sf_mix.serial_estimate_s > decode_step_floor.serial_estimate_s * 1.5


def test_slice_topology_wraparound_and_d2d_ar_floor() -> None:
    """Verify slice topology torus hop counts, wraparound, and D2D all-reduce floors."""
    assert set(TOPOLOGIES) == {"2x2x1", "2x2x2", "2x2x4", "2x4x4", "4x4x4"}
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
    open_z4 = torus_collective_s(
        "ar", 64 * 1024 * 1024, (2, 1, 1, 4), chip, TOPOLOGIES["2x2x4"]
    )
    wrap_z4 = torus_collective_s(
        "ar", 64 * 1024 * 1024, (2, 1, 1, 4), chip, TOPOLOGIES["4x4x4"]
    )
    assert wrap_z4 < open_z4
    par_ep128 = Parallel(
        tp=8, dp=16, moe="ep", ep=128, moe_dispatch="ag_ep", mesh_axes=(2, 4, 1, 1)
    )
    tp_group_size = 8
    sf_128 = step_floor(
        WIDE_GEOM,
        chip,
        par_ep128,
        StepShape(batch=4, context=1024),
        FloorOptions(dtypes=EXAMPLE_FP8_DTYPES, topology=TOPOLOGIES["4x4x4"]),
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
    assert sf_128.chain_s == pytest.approx(exp_chain_128, rel=1e-06)
    assert submesh_for_group(128, TOPOLOGIES["4x4x4"], 2, (2, 4, 1, 1)) == (2, 4, 4, 4)
    with pytest.raises(ValueError, match="exceeds topology"):
        submesh_for_group(64, TOPOLOGIES["2x2x2"], 2)


def test_spec_cycle_verify_writes_state_once() -> None:
    """Verify speculative verify step performs state writes exactly once."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json"))
    parallel = Parallel(tp=8, kv_split=4, moe="ep")
    opts_doc = FloorOptions(dtypes=DOC_DTYPES)
    cyc = spec_cycle_floor(
        geometry,
        CHIPS["tpu7x"],
        parallel,
        SpecShape(batch=4, context=4096, spec_k=3),
        opts_doc,
    )
    ver_once = step_floor(
        geometry,
        CHIPS["tpu7x"],
        parallel,
        StepShape(batch=4, rows_per_seq=4, context=4096, state_writes=1),
        opts_doc,
    )
    ver_k1 = step_floor(
        geometry,
        CHIPS["tpu7x"],
        parallel,
        StepShape(batch=4, rows_per_seq=4, context=4096, state_writes=4),
        opts_doc,
    )
    gdn_cyc = next(m for m in cyc.modules if m.name == "gdn")
    gdn_once = next(m for m in ver_once.modules if m.name == "gdn")
    gdn_k1 = next(m for m in ver_k1.modules if m.name == "gdn")
    assert gdn_cyc.hbm_bytes == pytest.approx(gdn_once.hbm_bytes, abs=1e-06)
    assert gdn_cyc.hbm_bytes < gdn_k1.hbm_bytes


def test_assumed_hop_latency_is_reported_outside_the_bound() -> None:
    """Verify assumed hop latency is reported outside the hard resource bound."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json"))
    inf_chip = replace(
        CHIPS["tpu7x"],
        hbm_bw=1e30,
        fp8_flops=1e30,
        bf16_flops=1e30,
        vector_flops=1e30,
        ici_bw=1e30,
        ici_link_bw=1e30,
    )
    parallel = Parallel(tp=8)
    shape = StepShape(batch=1, rows_per_seq=1, context=0)
    floor_result = step_floor(
        geometry,
        inf_chip,
        parallel,
        shape,
        FloorOptions(topology=TOPOLOGIES["2x2x1"]),
    )
    assert floor_result.latency_assumed_s > 0.0
    assert floor_result.serial_estimate_s >= floor_result.latency_assumed_s
    assert floor_result.coll_bound_s < floor_result.latency_assumed_s
    assert floor_result.hard_floor_s < floor_result.latency_assumed_s


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
            {"dtype": "bfloat16", "quantization_config": {"bits": 4}}
        )
    with pytest.raises(ValueError, match="scale_bytes"):
        rf.dtypes_from_hf_config(
            {"dtype": "bfloat16", "quantization_config": {"bits": 4, "group_size": 32}}
        )


def test_serving_plan_dp_throughput_invariance_on_locked_fixtures() -> None:
    """Verify DP=1 and DP=2 achieve identical per-chip output throughput."""
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    chip = CHIPS["tpu7x"]
    for fixture_name, repo in (
        ("qwen3_5_397b_fp8.json", "Qwen/Qwen3.5-397B-A17B-FP8"),
        ("glm_5_2_fp8.json", "zai-org/GLM-5.2-FP8"),
    ):
        config = json.loads((fixtures_dir / fixture_name).read_text())
        geometry = from_hf_config(config, model_ref=repo)
        dtypes = rf.dtypes_from_hf_config(config)
        req = ServingRequest(context=4096, concurrencies=[4])
        row_dp1 = plan(
            geometry, chip, Parallel(tp=8, dp=1, moe="tp_slice"), dtypes, req
        )[0]
        row_dp2 = plan(
            geometry, chip, Parallel(tp=8, dp=2, moe="tp_slice"), dtypes, req
        )[0]
        assert row_dp1.hard_floor_ms == pytest.approx(row_dp2.hard_floor_ms, rel=1e-9)
        assert row_dp2.output_tput_ceil_per_chip == pytest.approx(
            row_dp1.output_tput_ceil_per_chip, rel=1e-9
        )


def test_lm_head_sampled_rows_and_pp_invariance() -> None:
    """Verify lm_head prices sampled rows only (including MTP) without PP division."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json"))
    dtypes = Dtypes(w_dense=1.0, w_expert=1.0, lm_head=2.0, act="a8")
    pre_short = {
        m.name: m
        for m in pass_costs(
            geometry,
            Parallel(tp=8, pp=1),
            StepShape(batch=1, rows_per_seq=0, prefill_tokens=128),
            dtypes,
        )
    }["lm_head"]
    pre_long = {
        m.name: m
        for m in pass_costs(
            geometry,
            Parallel(tp=8, pp=1),
            StepShape(batch=1, rows_per_seq=0, prefill_tokens=4096),
            dtypes,
        )
    }["lm_head"]
    assert pre_short.bf16_flops == pytest.approx(pre_long.bf16_flops, rel=1e-9)

    dec_k0 = {
        m.name: m
        for m in pass_costs(
            geometry,
            Parallel(tp=8, pp=1),
            StepShape(batch=4, rows_per_seq=1, context=1024),
            dtypes,
        )
    }["lm_head"]
    dec_k3 = {
        m.name: m
        for m in pass_costs(
            geometry,
            Parallel(tp=8, pp=1),
            StepShape(batch=4, rows_per_seq=4, context=1024),
            dtypes,
        )
    }["lm_head"]
    dec_pp4 = {
        m.name: m
        for m in pass_costs(
            geometry,
            Parallel(tp=8, pp=4),
            StepShape(batch=4, rows_per_seq=1, context=1024),
            dtypes,
        )
    }["lm_head"]
    assert dec_k3.bf16_flops == pytest.approx(4.0 * dec_k0.bf16_flops, rel=1e-9)
    assert dec_pp4.hbm_bytes == pytest.approx(dec_k0.hbm_bytes, rel=1e-9)


def test_dsa_indexer_replicated_index_share_and_scoring_flops() -> None:
    """Verify GLM-5.2 IndexShare scales indexer weights and charges scoring FLOPs."""
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    glm_cfg = json.loads((fixtures_dir / "glm_5_2_fp8.json").read_text())
    geometry = from_hf_config(glm_cfg, model_ref="zai-org/GLM-5.2-FP8")
    dtypes = rf.dtypes_from_hf_config(glm_cfg)
    short_dsa = {
        m.name: m
        for m in pass_costs(
            geometry,
            Parallel(tp=8),
            StepShape(batch=2, rows_per_seq=1, context=1024),
            dtypes,
        )
    }["dsa"]
    long_dsa = {
        m.name: m
        for m in pass_costs(
            geometry,
            Parallel(tp=8),
            StepShape(batch=2, rows_per_seq=1, context=8192),
            dtypes,
        )
    }["dsa"]
    assert long_dsa.fp8_flops > short_dsa.fp8_flops
    assert long_dsa.strict_fp8_flops > short_dsa.strict_fp8_flops

    all_full_cfg = {**glm_cfg, "indexer_types": ["full"] * 78}
    all_full_geom = from_hf_config(all_full_cfg)
    all_full_dsa = {
        m.name: m
        for m in pass_costs(
            all_full_geom,
            Parallel(tp=8),
            StepShape(batch=2, rows_per_seq=1, context=8192),
            dtypes,
        )
    }["dsa"]
    assert all_full_dsa.hbm_bytes > long_dsa.hbm_bytes

    # Check replicated indexer weight bytes vs. TP-sharded strict bytes at context=0.
    idx_full_layers = geometry.indexer_types.count("full")
    assert idx_full_layers > 0
    for tp in (4, 8):
        dsa_zero_ctx = {
            m.name: m
            for m in pass_costs(
                geometry,
                Parallel(tp=tp),
                StepShape(batch=1, rows_per_seq=1, context=0),
                dtypes,
            )
        }["dsa"]
        expected_delta = (
            idx_full_layers
            * (
                dsa_indexer_layer_weight_elems(geometry)
                - _dsa_indexer_layer_weight_elems_strict(geometry, tp)
            )
            * dtypes.w_dense
        )
        assert expected_delta > 0.0
        assert (
            dsa_zero_ctx.hbm_bytes - dsa_zero_ctx.strict_hbm_bytes
        ) == pytest.approx(expected_delta, rel=1e-9)


def test_locked_fixture_quantization_and_hc_lowrank() -> None:
    """Verify ModelOpt NVFP4, FP8, and hyper-connection low-rank costs."""
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    for nvfp4_file in ("qwen3_5_397b_nvfp4.json", "glm_5_2_nvfp4.json"):
        config = json.loads((fixtures_dir / nvfp4_file).read_text())
        dtypes = rf.dtypes_from_hf_config(config)
        assert dtypes.act == "a4"
        assert dtypes.w_dense == pytest.approx(2.0)
        assert dtypes.w_expert == pytest.approx(0.5625)
        assert dtypes.w_aux == pytest.approx(2.0)

    dt_modelopt_fp8 = rf.dtypes_from_hf_config(
        {
            "torch_dtype": "bfloat16",
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"},
        }
    )
    assert dt_modelopt_fp8.w_expert == pytest.approx(1.0)
    assert dt_modelopt_fp8.w_dense == pytest.approx(1.0)

    hc_geom = replace(
        from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json")),
        hc_mult=4,
        hc_lowrank=64,
    )
    hc_mods = {
        m.name: m
        for m in pass_costs(
            hc_geom,
            Parallel(tp=8),
            StepShape(batch=2, rows_per_seq=1, context=1024),
            Dtypes(w_dense=2.0, w_expert=1.0),
        )
    }
    assert "ple" not in hc_mods
    assert "hc_proj" in hc_mods
    assert hc_mods["hc_proj"].hbm_bytes > 0.0


def test_topology_none_uses_torus_collective_schedule() -> None:
    """Verify topology=None prices collectives via torus_collective_s."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json"))
    chip = CHIPS["tpu7x"]
    parallel = Parallel(tp=8)
    shape = StepShape(batch=1, rows_per_seq=4, context=0)
    sf_none = step_floor(
        geometry, chip, parallel, shape, FloorOptions(dtypes=EXAMPLE_FP8_DTYPES)
    )
    expected_chain_s = sum(
        torus_collective_s(
            c.kind,
            c.payload,
            submesh_for_group(
                c.group, None, parallel.ranks_per_chip, parallel.mesh_axes
            ),
            chip,
            None,
        )
        for m in sf_none.modules
        for c in m.collectives
        if c.group > 1
    )
    assert sf_none.chain_s == pytest.approx(expected_chain_s, rel=1e-09)


def test_coll_bound_is_the_cut_bound_sum() -> None:
    """Verify coll_bound_s is the cut bound sum and ignores hop latency."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json"))
    chip = CHIPS["tpu7x"]
    parallel = Parallel(tp=8)
    shape = StepShape(batch=1, rows_per_seq=1, context=0)
    opts = FloorOptions(
        dtypes=EXAMPLE_FP8_DTYPES,
        topology=TOPOLOGIES["2x2x1"],
        collective_latency_s=50e-6,
    )
    floor_result = step_floor(geometry, chip, parallel, shape, opts)
    sf_zero_hop = step_floor(
        geometry, replace(chip, ici_hop_latency_s=0.0), parallel, shape, opts
    )
    assert floor_result.coll_bound_s > 0.0
    assert floor_result.hard_floor_s >= floor_result.coll_bound_s - 1e-12
    assert floor_result.coll_bound_s == pytest.approx(
        sf_zero_hop.coll_bound_s, rel=1e-12
    )


def test_hard_floor_ignores_assumed_hop_latency() -> None:
    """Verify hard_floor_s and coll_bound_s ignore assumed hop latency."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json"))
    chip = CHIPS["tpu7x"]
    parallel = Parallel(tp=8)
    shape = StepShape(batch=1, rows_per_seq=1, context=0)
    opts = FloorOptions(topology=TOPOLOGIES["2x2x1"])
    sf_default = step_floor(geometry, chip, parallel, shape, opts)
    slow_chip = replace(chip, ici_hop_latency_s=1.0)
    sf_slow = step_floor(geometry, slow_chip, parallel, shape, opts)
    assert sf_slow.hard_floor_s == pytest.approx(sf_default.hard_floor_s, rel=1e-12)
    assert sf_slow.coll_bound_s == pytest.approx(sf_default.coll_bound_s, rel=1e-12)
    assert sf_slow.serial_estimate_s > sf_default.serial_estimate_s


def test_vector_bound_uses_fast_bracket_end_and_bounds_estimate() -> None:
    """Verify strict vector_bound_s <= vector_s across the entire public bracket."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json"))
    chip = CHIPS["tpu7x"]
    parallel = Parallel(tp=8)
    shape = StepShape(batch=1, rows_per_seq=1, context=0)
    opts = FloorOptions(dtypes=Dtypes(w_expert=0.5))
    sf_default = step_floor(geometry, chip, parallel, shape, opts)
    fast_ratio = rf_constants.VECTOR_TO_BF16_RATIO_BRACKET[1]
    fast_chip = replace(
        chip,
        vector_flops=chip.bf16_flops * fast_ratio,
        vector_flops_overridden=True,
    )
    sf_fast = step_floor(geometry, fast_chip, parallel, shape, opts)
    assert sf_default.vector_bound_s == pytest.approx(sf_fast.vector_bound_s)
    for ratio in (
        rf_constants.VECTOR_TO_BF16_RATIO_BRACKET[0],
        rf_constants.DEFAULT_VECTOR_TO_BF16_RATIO,
        rf_constants.VECTOR_TO_BF16_RATIO_BRACKET[1],
    ):
        probe_chip = replace(chip, vector_flops=chip.bf16_flops * ratio)
        sf_probe = step_floor(geometry, probe_chip, parallel, shape, opts)
        assert sf_default.vector_bound_s <= sf_probe.vector_s


def test_strict_counts_ignore_kernel_row_tile() -> None:
    """Verify strict matmul, vector, HBM and hard floors ignore mxu_m_tile."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json"))
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
        "vector_bound_s",
        "hbm_bound_s",
        "hard_floor_s",
    ):
        values = [getattr(floors_moe[tile], field_name) for tile in (1, 256, 512)]
        assert values[0] == pytest.approx(values[1], rel=1e-12)
        assert values[1] == pytest.approx(values[2], rel=1e-12)

    llama_config = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForCausalLM"],
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
        "vector_bound_s",
        "hbm_bound_s",
        "hard_floor_s",
    ):
        values = [getattr(floors_dense[tile], field_name) for tile in (1, 256, 512)]
        assert values[0] == pytest.approx(values[1], rel=1e-12)
        assert values[1] == pytest.approx(values[2], rel=1e-12)


def test_serving_plan_bounds_ep_chips_and_auto_topology() -> None:
    """Verify serving.plan validates spec bounds, EP chip count, and auto topology."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json"))
    chip = CHIPS["tpu7x"]
    dtypes = Dtypes()
    bad_spec_window = -1
    with pytest.raises(ValueError, match="spec_window"):
        plan(
            geometry,
            chip,
            Parallel(tp=8),
            dtypes,
            ServingRequest(
                context=512, concurrencies=(1,), spec_window=bad_spec_window
            ),
        )
    with pytest.raises(ValueError, match="acceptance_length"):
        plan(
            geometry,
            chip,
            Parallel(tp=8),
            dtypes,
            ServingRequest(
                context=512, concurrencies=(1,), spec_window=0, acceptance_length=2.0
            ),
        )
    spec_win = 3
    with pytest.raises(ValueError, match="acceptance_length"):
        plan(
            geometry,
            chip,
            Parallel(tp=8),
            dtypes,
            ServingRequest(
                context=512,
                concurrencies=(1,),
                spec_window=spec_win,
                acceptance_length=None,
            ),
        )
    below_min_al = 0.5
    above_max_al = float(spec_win + 2)
    for bad_al in (below_min_al, above_max_al):
        with pytest.raises(ValueError, match="acceptance_length"):
            plan(
                geometry,
                chip,
                Parallel(tp=8),
                dtypes,
                ServingRequest(
                    context=512,
                    concurrencies=(1,),
                    spec_window=spec_win,
                    acceptance_length=bad_al,
                ),
            )

    # Standalone EP=8 with TP=1 spans 8 ranks (4 chips on 2-core chips), not 1 rank.
    concurrency = 8
    ctx = 256
    ep_ranks = 8
    ranks_per_chip = rf_constants.DEFAULT_CORES_PER_CHIP
    ep_chips = float(ep_ranks) / float(ranks_per_chip)
    ep_only = Parallel(tp=1, ep=ep_ranks, moe="ep", dp=1, ranks_per_chip=ranks_per_chip)
    rows = plan(
        geometry,
        chip,
        ep_only,
        dtypes,
        ServingRequest(
            context=ctx, concurrencies=(concurrency,), collective_latency_us=1.0
        ),
    )
    sf_explicit = step_floor(
        geometry,
        chip,
        ep_only,
        StepShape(batch=concurrency, rows_per_seq=1, context=ctx),
        FloorOptions(
            dtypes=dtypes,
            collective_latency_s=rf_constants.UNIT_MICRO,
            topology=TOPOLOGIES["2x2x1"],
        ),
    )
    tol = 1e-9
    expected_tput = float(concurrency) / sf_explicit.hard_floor_s / ep_chips
    assert rows[0].hard_floor_ms == pytest.approx(
        sf_explicit.hard_floor_s * rf_constants.UNIT_MILLIS_PER_SECOND, rel=tol
    )
    assert rows[0].output_tput_ceil_per_chip == pytest.approx(expected_tput, rel=tol)


def test_spec_cycle_floor_preserves_structural_fields_and_intra_chip_zero_latency() -> (
    None
):
    """Verify draft structural fields hold and intra-chip hops add zero latency."""
    base_geom = ModelGeometry(
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
        num_experts=16,
        top_k=2,
        moe_inter=512,
        shared_inter=0,
        dense_inter=0,
        first_k_dense_replace=0,
        vocab=32000,
        mtp_layers=1,
        mtp_inter=0,
        routed_expert_hidden=256,
        hc_mult=1,
        global_head_dim=256,
        ple_dim=64,
        compress_ratios=(0.5,),
    )
    hc_geom = replace(base_geom, hc_mult=3)
    chip = CHIPS["tpu7x"]
    batch_n = 2
    ctx_n = 256
    spec_k = 2
    spec = SpecShape(batch=batch_n, context=ctx_n, spec_k=spec_k)

    # TP=2 fits inside 1 chip (ranks_per_chip=2) so ICI hops == 0 and latency == 0.
    intra_chip = Parallel(tp=2, ranks_per_chip=2)
    lat_s = rf_constants.UNIT_MICRO
    sf_base = spec_cycle_floor(
        base_geom,
        chip,
        intra_chip,
        spec,
        FloorOptions(collective_latency_s=lat_s, topology=TOPOLOGIES["2x2x1"]),
    )
    sf_hc = spec_cycle_floor(
        hc_geom,
        chip,
        intra_chip,
        spec,
        FloorOptions(collective_latency_s=lat_s, topology=TOPOLOGIES["2x2x1"]),
    )
    assert sf_base.latency_assumed_s == 0.0
    assert sf_hc.vector_s > sf_base.vector_s

    # Verify draft_moe runs cleanly and multi-chip tp adds hop latency.
    inter_chip = Parallel(tp=8, ranks_per_chip=2)
    sf_multi = spec_cycle_floor(
        hc_geom,
        chip,
        inter_chip,
        spec,
        FloorOptions(collective_latency_s=lat_s, topology=TOPOLOGIES["2x2x1"]),
    )
    draft_moe_mods = [m for m in sf_multi.modules if m.name == "draft_moe"]
    assert len(draft_moe_mods) == 1
    assert sf_multi.latency_assumed_s > 0.0


def test_hf_loader_heterogeneous_groups_block_scale_and_kv_guards() -> None:
    """Verify mixed config_groups, FP4 group_size, block scale_dtype, and KV guards."""
    block_sz = 128
    group_sz = 128
    eight_bits = 8
    four_bits = 4
    tol = 1e-12

    # Mixed config_groups: Linear in FP8 with block scale_dtype, MoE in 4-bit grouped.
    mixed_cfg: dict[str, object] = {
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "weight_block_size": [block_sz, block_sz],
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "format": "float-quantized",
                    "weights": {
                        "num_bits": eight_bits,
                        "type": "float",
                        "scale_dtype": "float8_e4m3fn",
                    },
                    "input_activations": {"num_bits": eight_bits, "type": "float"},
                },
                "group_1": {
                    "targets": ["MoE"],
                    "weights": {
                        "num_bits": four_bits,
                        "group_size": group_sz,
                        "scale_dtype": "bfloat16",
                    },
                },
            },
        },
    }
    mixed_dt = rf.dtypes_from_hf_config(mixed_cfg)
    expected_fp8_e4m3_block = rf_constants.FP8_BYTES + (
        rf_constants.FP8_BYTES / float(block_sz * block_sz)
    )
    expected_int4_bf16_scale = 0.5 + rf_constants.BF16_BYTES / float(group_sz)
    assert mixed_dt.w_dense == pytest.approx(expected_fp8_e4m3_block, rel=tol)
    assert mixed_dt.w_expert == pytest.approx(expected_int4_bf16_scale, rel=tol)
    assert mixed_dt.w_dense_dtype == "fp8_block"
    assert mixed_dt.w_expert_dtype == "compressed-tensors_int4_packed"

    # compressed-tensors FP8 resolves w_expert_dtype to "fp8", not int4.
    ct_fp8_cfg: dict[str, object] = {
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "format": "fp8-quantized",
                    "weights": {"num_bits": eight_bits, "type": "float"},
                    "input_activations": {"num_bits": eight_bits, "type": "float"},
                }
            },
        },
    }
    ct_fp8_dt = rf.dtypes_from_hf_config(ct_fp8_cfg)
    assert ct_fp8_dt.w_dense_dtype == "fp8"
    assert ct_fp8_dt.w_expert_dtype == "fp8"
    assert ct_fp8_dt.w_expert == pytest.approx(rf_constants.FP8_BYTES, rel=tol)

    # NVFP4 / MXFP4 configs succeed without a group_size field.
    for fp4_method in ("nvfp4", "mxfp4"):
        fp4_cfg: dict[str, object] = {
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": fp4_method,
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {"num_bits": four_bits},
                    }
                },
            },
        }
        fp4_dt = rf.dtypes_from_hf_config(fp4_cfg)
        assert fp4_dt.w_expert_dtype == fp4_method

    # Mapping kv_cache_scheme parses FP8 and 4-bit/FP4 KV cleanly.
    kv_map_cfg: dict[str, object] = {
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "fp8",
            "kv_cache_scheme": {"type": "fp8", "num_bits": eight_bits},
        },
    }
    kv_map_dt = rf.dtypes_from_hf_config(kv_map_cfg)
    assert kv_map_dt.kv == pytest.approx(rf_constants.FP8_BYTES, rel=tol)
    assert kv_map_dt.kv_dtype == "fp8"

    for fp4_kv in ("fp4", "nvfp4", {"type": "nvfp4", "num_bits": four_bits}):
        dt_fp4_kv = rf.dtypes_from_hf_config(
            {
                "torch_dtype": "bfloat16",
                "quantization_config": {
                    "quant_method": "fp8",
                    "kv_cache_scheme": fp4_kv,
                },
            }
        )
        assert dt_fp4_kv.kv == pytest.approx(rf_constants.FP4_BYTES, rel=tol)
        assert dt_fp4_kv.kv_dtype == "fp4"


def test_block_parallel_and_multilayer_mtp_draft_cost() -> None:
    """Verify block-parallel and multi-layer MTP draft module costing."""
    synth_moe_mtp_cfg = {
        "model_type": "qwen3_5_moe",
        "hidden_size": 4096,
        "num_hidden_layers": 8,
        "layer_types": ["full_attention"] * 8,
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
    geom_bp = _from_hf_config_dry_run(synth_moe_mtp_cfg)
    geom_seq = replace(geom_bp, block_parallel=1)
    assert geom_bp.block_parallel == 2
    assert geom_bp.mtp_inter == 6144
    parallel = Parallel(tp=8)
    dtypes = Dtypes()
    opts_dt = FloorOptions(dtypes=dtypes)
    mods_seq = pass_costs(
        geom_seq, parallel, StepShape(batch=4, rows_per_seq=1), dtypes
    )
    mods_bp = pass_costs(geom_bp, parallel, StepShape(batch=4, rows_per_seq=1), dtypes)
    assert (
        sum(module.n_collectives for module in mods_seq)
        - sum(module.n_collectives for module in mods_bp)
        == 8
    )
    spec = SpecShape(batch=4, context=2048, spec_k=2)
    cyc_dense_mtp = spec_cycle_floor(geom_bp, CHIPS["tpu7x"], parallel, spec, opts_dt)
    cyc_moe_mtp = spec_cycle_floor(
        replace(geom_bp, mtp_inter=0), CHIPS["tpu7x"], parallel, spec, opts_dt
    )
    assert any(module.name == "draft_mlp" for module in cyc_dense_mtp.modules)
    assert not any(module.name == "draft_moe" for module in cyc_dense_mtp.modules)
    assert any(module.name == "draft_moe" for module in cyc_moe_mtp.modules)
    cyc_mtp1 = spec_cycle_floor(
        replace(geom_bp, mtp_layers=1), CHIPS["tpu7x"], parallel, spec, opts_dt
    )
    draft_mlp_2 = next(
        module for module in cyc_dense_mtp.modules if module.name == "draft_mlp"
    )
    draft_mlp_1 = next(
        module for module in cyc_mtp1.modules if module.name == "draft_mlp"
    )
    draft_fc_2 = next(
        module for module in cyc_dense_mtp.modules if module.name == "draft_fc"
    )
    draft_fc_1 = next(
        module for module in cyc_mtp1.modules if module.name == "draft_fc"
    )
    assert draft_mlp_2.hbm_bytes == pytest.approx(draft_mlp_1.hbm_bytes, rel=1e-09)
    assert draft_fc_2.hbm_bytes == pytest.approx(draft_fc_1.hbm_bytes, rel=1e-09)
    seq_layout = SeqKvLayout(1, 1024)
    bd_mtp2 = rf.hbm_breakdown_per_core_bytes(geom_bp, dtypes, parallel, seq_layout)
    bd_mtp1 = rf.hbm_breakdown_per_core_bytes(
        replace(geom_bp, mtp_layers=1), dtypes, parallel, seq_layout
    )
    bd_mtp0 = rf.hbm_breakdown_per_core_bytes(
        replace(geom_bp, mtp_layers=0), dtypes, parallel, seq_layout
    )
    assert bd_mtp2.weights - bd_mtp0.weights == pytest.approx(
        2.0 * (bd_mtp1.weights - bd_mtp0.weights), rel=1e-09
    )
    bd_moe = rf.hbm_breakdown_per_core_bytes(
        replace(geom_bp, mtp_inter=0), dtypes, parallel, seq_layout
    )
    assert bd_mtp2.weights < bd_moe.weights


@pytest.mark.parametrize("fixture_name", ("glm_5_2_fp8.json", "glm_5_2_nvfp4.json"))
def test_spec_cycle_floor_dsa_shared_indexer_and_pp_zero_draft_p2p(
    fixture_name: str,
) -> None:
    """Verify GLM-5.2 draft DSA respects shared indexer and omits draft pp_p2p."""
    bundle = load_locked_bundle(fixture_name)
    chip = CHIPS["tpu7x"]
    spec = SpecShape(batch=16, context=8192, spec_k=1)
    cyc = spec_cycle_floor(
        bundle.geometry,
        chip,
        Parallel(tp=8, ep=8, moe="ep"),
        spec,
        FloorOptions(dtypes=bundle.dtypes, topology=TOPOLOGIES["2x2x1"]),
    )
    mods_by_name = {module.name: module for module in cyc.modules}
    target_dsa = mods_by_name["dsa"]
    draft_dsa = mods_by_name["draft_dsa"]
    # Draft layer shares the indexer (idx_scale=0), so 1 draft layer stays
    # well below 2x of the average per-layer target DSA HBM footprint.
    avg_target_layer_hbm = target_dsa.hbm_bytes / float(bundle.geometry.num_layers)
    assert draft_dsa.hbm_bytes < 2.0 * avg_target_layer_hbm

    cyc_pp = spec_cycle_floor(
        bundle.geometry,
        chip,
        Parallel(tp=4, pp=4),
        SpecShape(batch=4, context=2048, spec_k=3),
        FloorOptions(dtypes=bundle.dtypes, topology=TOPOLOGIES["2x2x2"]),
    )
    for module in cyc_pp.modules:
        if module.name.startswith("draft_"):
            assert all(coll.name != "pp_p2p" for coll in module.collectives)


def test_joint_orthogonal_placement_avoids_same_axis_collision() -> None:
    """Verify joint placement assigns TP4 and PCP4 on 2x2x2 to disjoint axes."""
    unit_chip = replace(
        CHIPS["tpu7x"],
        name="unit_ici",
        ici_bw=300.0e9,
        ici_link_bw=100.0e9,
        d2d_bw=600.0e9,
        ici_hop_latency_s=0.0,
    )
    topo = TOPOLOGIES["2x2x2"]
    coll_cfg = CollectiveConfig(algo="one_shot", ranks_per_chip=2)
    out_ar = make("out_ar", "ar", 100.0, 4, coll_cfg)
    pcp_ag = make("pcp_kv_ag", "ag", 200.0, 4, coll_cfg)
    mod_tp = ModuleCost(
        name="attn_tp",
        hbm_bytes=0.0,
        fp8_flops=0.0,
        bf16_flops=0.0,
        collectives=(out_ar,),
        unique_collectives=((out_ar, 1),),
    )
    mod_pcp = ModuleCost(
        name="attn_pcp",
        hbm_bytes=0.0,
        fp8_flops=0.0,
        bf16_flops=0.0,
        collectives=(pcp_ag,),
        unique_collectives=((pcp_ag, 1),),
    )
    floor_joint = combine_modules(
        (mod_tp, mod_pcp),
        unit_chip,
        topology=topo,
        parallel=Parallel(tp=4, pcp=4),
    )
    # Disjoint placement finishes in <= 1.50 ns (never 2.00 ns collided on one axis).
    assert floor_joint.coll_bound_s <= 1.5e-9 + 1e-18


def test_hf_loader_quantization_heterogeneous_int4_kv_and_local_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify W4A4 a4, heterogeneous groups, int4 KV, unknown KV, and local path."""
    hetero_cfg: dict[str, object] = {
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {
                "group_dense": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 8, "type": "float"},
                    "input_activations": {"num_bits": 8, "type": "float"},
                },
                "group_experts": {
                    "targets": ["experts"],
                    "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                    "input_activations": {"num_bits": 4, "type": "float"},
                },
            },
        },
    }
    hetero_dt = rf.dtypes_from_hf_config(hetero_cfg)
    assert hetero_dt.w_dense == pytest.approx(rf_constants.FP8_BYTES)
    assert hetero_dt.w_dense_dtype == "fp8"
    assert hetero_dt.w_expert == pytest.approx(rf_constants.NVFP4_BYTES)
    assert hetero_dt.w_expert_dtype == "nvfp4"
    assert hetero_dt.act == "a4"

    int4_kv_dt = rf.dtypes_from_hf_config(
        {
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "fp8",
                "kv_cache_scheme": {"type": "int", "num_bits": 4},
            },
        }
    )
    assert int4_kv_dt.kv == pytest.approx(rf_constants.FP4_BYTES)
    assert int4_kv_dt.kv_dtype == "int4"

    for bad_kv in ("v8_custom", "unknown_kv_scheme"):
        with pytest.raises(ValueError, match="Unsupported kv_cache_scheme"):
            rf.dtypes_from_hf_config(
                {
                    "torch_dtype": "bfloat16",
                    "quantization_config": {
                        "quant_method": "fp8",
                        "kv_cache_scheme": bad_kv,
                    },
                }
            )

    local_model_dir = tmp_path / "models" / "qwen_local"
    local_model_dir.mkdir(parents=True)
    (local_model_dir / "config.json").write_text(
        json.dumps(load_locked_config("qwen3_5_397b_fp8.json")),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    loaded_cfg, loaded_ident = rf.load_hf_config_with_identity("models/qwen_local")
    assert loaded_cfg["model_type"] == "qwen3_5_moe"
    assert loaded_ident.model_type == "qwen3_5_moe_text"


def test_ep_busiest_experts_large_n_and_plan_rejects_nonpositive_concurrency() -> None:
    """Verify large-N log-space binomial PMF stays finite and plan guards conc <= 0."""
    busiest = ep_busiest_experts(
        num_experts=256,
        top_k=8,
        rows=2000,
        ranks=64,
    )
    assert busiest > 0.0
    assert busiest <= 256.0 / 64.0

    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    with pytest.raises(ValueError, match="concurrency must be positive"):
        plan(
            bundle.geometry,
            CHIPS["tpu7x"],
            Parallel(tp=8),
            bundle.dtypes,
            ServingRequest(context=2048, concurrencies=[0]),
        )


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
@pytest.mark.parametrize(
    ("topology_name", "tp_degree", "ep_degree"),
    (
        ("2x2x1", 8, 8),
        ("2x2x2", 8, 16),
        ("2x2x4", 8, 32),
        ("2x4x4", 8, 64),
        ("4x4x4", 8, 128),
    ),
)
def test_locked_fixtures_across_all_topologies_respect_floor_ordering(
    fixture_name: str,
    topology_name: str,
    tp_degree: int,
    ep_degree: int,
) -> None:
    """Verify hard_floor <= envelope_floor <= serial_estimate across all 4x5 cases."""
    bundle = load_locked_bundle(fixture_name)
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES[topology_name]
    dp_degree = max(1, ep_degree // tp_degree)
    parallel = Parallel(tp=tp_degree, dp=dp_degree, ep=ep_degree, moe="ep")
    floor = step_floor(
        bundle.geometry,
        chip,
        parallel,
        StepShape(batch=8, rows_per_seq=1, context=4096),
        FloorOptions(dtypes=bundle.dtypes, topology=topology),
    )
    assert 0.0 < floor.hard_floor_s <= floor.envelope_floor_s + 1e-12
    assert floor.envelope_floor_s <= floor.serial_estimate_s + 1e-12
    rows = plan(
        bundle.geometry,
        chip,
        parallel,
        bundle.dtypes,
        ServingRequest(context=4096, concurrencies=[8]),
        topologies={topology_name: topology},
    )
    assert len(rows) == 1
    assert rows[0].hard_floor_ms > 0.0
    assert rows[0].output_tput_ceil_per_chip > 0.0


def test_spec_cycle_floor_reduces_kv_read_with_shared_prefix_tokens() -> None:
    """Verify spec_cycle_floor forwards shared_prefix_tokens to reduce KV bytes."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    chip = CHIPS["tpu7x"]
    parallel = Parallel(tp=8, ep=8, moe="ep")
    opts = FloorOptions(dtypes=bundle.dtypes, topology=TOPOLOGIES["2x2x1"])
    no_share = rf.spec_cycle_floor(
        bundle.geometry,
        chip,
        parallel,
        rf.SpecShape(batch=8, spec_k=3, context=8192, shared_prefix_tokens=0),
        opts,
    )
    with_share = rf.spec_cycle_floor(
        bundle.geometry,
        chip,
        parallel,
        rf.SpecShape(batch=8, spec_k=3, context=8192, shared_prefix_tokens=4096),
        opts,
    )
    assert with_share.hard_floor_s < no_share.hard_floor_s
    assert with_share.serial_estimate_s == pytest.approx(no_share.serial_estimate_s)
    no_share_hbm_min = sum(m.strict_hbm_bytes for m in no_share.modules)
    with_share_hbm_min = sum(m.strict_hbm_bytes for m in with_share.modules)
    assert with_share_hbm_min < no_share_hbm_min


def test_unpadded_prefill_capacity_bound_keeps_sampled_lm_head_rows() -> None:
    """Unpadded prefill capacity bound charges 1 sampled LM-head row, not isl rows."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    geometry, dtypes = bundle.geometry, bundle.dtypes
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES["2x2x1"]
    parallel = Parallel(
        tp=8,
        dp=1,
        ep=8,
        moe="ep",
        moe_dispatch="ag_ep",
        ranks_per_chip=topology.cores_per_chip,
    )
    short_shape = StepShape(
        batch=0, rows_per_seq=1, context=0, prefill_tokens=1024, prefix_hit_rate=0.0
    )
    long_shape = StepShape(
        batch=0, rows_per_seq=1, context=0, prefill_tokens=8192, prefix_hit_rate=0.0
    )
    short_lm = next(
        m
        for m in pass_costs(
            geometry, parallel, short_shape, dtypes, unpadded_linear=True, chip=chip
        )
        if m.name == "lm_head"
    )
    long_lm = next(
        m
        for m in pass_costs(
            geometry, parallel, long_shape, dtypes, unpadded_linear=True, chip=chip
        )
        if m.name == "lm_head"
    )
    assert short_lm.bf16_flops > 0.0
    assert short_lm.ici_bytes > 0.0
    assert short_lm.bf16_flops == pytest.approx(long_lm.bf16_flops, rel=0, abs=0)
    assert short_lm.ici_bytes == pytest.approx(long_lm.ici_bytes, rel=0, abs=0)
    zero_lm = next(
        m
        for m in pass_costs(
            geometry,
            parallel,
            StepShape(batch=0, rows_per_seq=0, context=0, prefill_tokens=0),
            dtypes,
            unpadded_linear=True,
            chip=chip,
        )
        if m.name == "lm_head"
    )
    assert zero_lm.bf16_flops == 0.0
    assert zero_lm.vector_flops == 0.0
    assert zero_lm.ici_bytes == 0.0
    unp_short = rf.unpadded_strict_resource_times(
        geometry,
        chip,
        parallel,
        short_shape,
        rf.UnpaddedFloorOptions(dtypes=dtypes, topology=topology, zero_weights=True),
    )
    unp_long = rf.unpadded_strict_resource_times(
        geometry,
        chip,
        parallel,
        long_shape,
        rf.UnpaddedFloorOptions(dtypes=dtypes, topology=topology, zero_weights=True),
    )
    assert all(val >= 0.0 for val in unp_short)
    assert unp_long[1] > unp_short[1]


def test_routed_moe_min_unpack_vector_flops_matches_module_layout() -> None:
    """Shared routed-MoE unpack helper matches pass_costs and gates on active_rows."""
    from vllm_torchtpu.roofline.modules import (
        routed_moe_min_unpack_vector_flops,
    )

    bundle = load_locked_bundle("qwen3_5_397b_nvfp4.json")
    geometry, dtypes = bundle.geometry, bundle.dtypes
    chip = CHIPS["tpu7x"]
    parallel = Parallel(
        tp=8,
        dp=1,
        ep=8,
        moe="ep",
        moe_dispatch="ag_ep",
        ranks_per_chip=chip.cores_per_chip,
    )
    assert (
        routed_moe_min_unpack_vector_flops(
            geometry, parallel, dtypes, chip=chip, active_rows=0
        )
        == 0.0
    )
    unpack_active = routed_moe_min_unpack_vector_flops(
        geometry, parallel, dtypes, chip=chip, active_rows=4
    )
    assert unpack_active > 0.0
    moe_with_rows = next(
        m
        for m in pass_costs(
            geometry,
            parallel,
            StepShape(batch=4, rows_per_seq=1, context=1024),
            dtypes,
            unpadded_linear=True,
            chip=chip,
        )
        if m.name == "moe"
    )
    moe_zero_rows = next(
        m
        for m in pass_costs(
            geometry,
            parallel,
            StepShape(batch=0, rows_per_seq=0, context=0, prefill_tokens=0),
            dtypes,
            unpadded_linear=True,
            chip=chip,
        )
        if m.name == "moe"
    )
    assert moe_zero_rows.strict_vector_flops == pytest.approx(
        moe_zero_rows.strict_vector_flops
    )
    assert moe_with_rows.strict_vector_flops >= unpack_active
