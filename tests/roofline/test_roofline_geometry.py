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
"""Unit tests for TPU chip specifications, model geometry, and import isolation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from jax.experimental.pallas import tpu as pltpu

import vllm_torchtpu.roofline as rf
from vllm_torchtpu.roofline import CHIPS, chips
from vllm_torchtpu.roofline.geometry import from_hf_config

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


def test_chip_table_matches_jax_tpu_info() -> None:
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    c = CHIPS["tpu7x"]
    assert c.hbm_bytes == info.hbm_capacity_bytes
    assert c.hbm_bw == info.mem_bw_bytes_per_second
    assert c.bf16_flops == info.bf16_ops_per_second
    assert c.fp8_flops == info.fp8_ops_per_second
    assert c.vmem_bytes == info.vmem_capacity_bytes
    assert (c.ici_link_bw, c.ici_bw, c.ici_hop_latency_s) == (100.0e9, 300.0e9, 1.0e-6)


def test_core_has_no_engine_imports() -> None:
    pkg_dir = Path(__file__).resolve().parents[2] / "src/vllm_torchtpu/roofline"
    banned = {
        "vllm",
        "torch",
        "torch_tpu",
        "numpy",
        "vllm_torchtpu",
    }
    for py_file in sorted(pkg_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        for node in ast.walk(tree):
            if type(node) is ast.Import:
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned
            elif type(node) is ast.ImportFrom and node.level == 0 and node.module:
                assert node.module.split(".")[0] not in banned


def test_from_hf_config_dsa_and_latent_moe_aliases() -> None:
    # Synthetic sparse-MLA / latent-MoE configuration exercising HF config aliases.
    synthetic_mla_cfg = {
        "hidden_size": 6144,
        "num_hidden_layers": 78,
        "layer_types": ["dsa"] * 78,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "num_experts": 256,
        "num_selected_experts": 8,
        "latent_moe_dim": 2048,
        "dense_intermediate_size": 12288,
        "first_k_dense_replace": 3,
        "vocab_size": 154880,
    }
    geom = from_hf_config(synthetic_mla_cfg)
    assert geom.dense_inter == 12288
    assert geom.moe_inter == 2048
    assert geom.top_k == 8


def test_linear_conv_kernel_dim_validation() -> None:
    # Model with linear attention but no linear_conv_kernel_dim must raise ValueError
    cfg_bad = {
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 4,
        "num_attention_heads": 32,
    }
    with pytest.raises(ValueError, match="linear_conv_kernel_dim"):
        from_hf_config(cfg_bad)

    # Model without linear attention defaults conv_kernel to 0
    cfg_non_linear = {
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "layer_types": ["full_attention"] * 4,
        "num_attention_heads": 32,
    }
    geom = from_hf_config(cfg_non_linear)
    assert geom.conv_kernel == 0


def test_chips_stdlib_fallback_when_jax_absent() -> None:
    jax_spec = CHIPS["tpu7x"]
    orig_pltpu = chips.pltpu
    try:
        chips.pltpu = None
        fallback = chips._from_jax(
            "tpu7x",
            "tpu7x",
            cores_per_chip=2,
            ici_link_bw=100.0e9,
            ici_bw=300.0e9,
            ici_hop_latency_s=1.0e-6,
        )
    finally:
        chips.pltpu = orig_pltpu
    assert fallback == jax_spec


def test_supports_validator_fails_closed_on_unknown_arch() -> None:
    for bad_cfg in (
        {},
        {
            "model_type": "mamba2",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "vocab_size": 32000,
        },
        {
            "model_type": "rwkv",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "vocab_size": 32000,
        },
        {
            "model_type": "falcon_mamba",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "vocab_size": 32000,
        },
        {
            "model_type": "bert",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "vocab_size": 30522,
        },
        {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "ssm_cfg": {"d_state": 16},
            "vocab_size": 32000,
        },
        {
            "hidden_size": 2048,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "layer_types": ["mamba", "full_attention"],
            "vocab_size": 32000,
        },
        {
            "hidden_size": 2048,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "layer_types": ["retnet_retention", "full_attention"],
            "vocab_size": 32000,
        },
        {
            "vision_config": {"hidden_size": 1024},
            "text_config": None,
        },
        {
            "vision_config": {"hidden_size": 1024},
        },
    ):
        with pytest.raises(rf.UnsupportedArchitectureError):
            rf.supports(bad_cfg)
        with pytest.raises(rf.UnsupportedArchitectureError):
            rf.from_hf_config(bad_cfg)


def test_kimi_linear_and_hybrid_multimodel_configs() -> None:
    kimi_cfg = {
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
        "hidden_size": 4096,
        "num_hidden_layers": 8,
        "num_attention_heads": 32,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 0,
        "v_head_dim": 128,
        "full_attention_interval": 4,
        "linear_attn_config": {
            "num_heads": 32,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
        },
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 1024,
        "vocab_size": 163840,
    }
    kg = rf.from_hf_config(kimi_cfg)
    assert kg.layer_types.count("kda") == 6
    assert kg.layer_types.count("mla") == 2
    assert kg.conv_kernel == 4
    assert kg.linear_k_heads == 32
    assert kg.linear_k_dim == 128

    llama4_cfg = {
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
    lg = rf.from_hf_config(llama4_cfg)
    assert lg.n_moe_layers == 3

    gemma4_cfg = {
        "model_type": "gemma4_text",
        "hidden_size": 3840,
        "num_hidden_layers": 6,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "global_head_dim": 512,
        "sliding_window": 1024,
        "sliding_window_pattern": 6,
        "intermediate_size": 15360,
        "vocab_size": 262144,
    }
    gg = rf.from_hf_config(gemma4_cfg)
    assert gg.global_head_dim == 512
    assert gg.layer_types.count("full_attention") == 1
    assert gg.layer_types.count("sliding_attention") == 5

    minimax_cfg = {
        "model_type": "minimax_text_01",
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "attn_type_list": [0, 0, 0, 1],
        "num_local_experts": 16,
        "num_experts_per_tok": 2,
        "intermediate_size": 4096,
        "vocab_size": 100000,
    }
    mg = rf.from_hf_config(minimax_cfg)
    assert mg.layer_types == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )
    assert mg.conv_kernel == 0

    nemotron_cfg = {
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
    ng = rf.from_hf_config(nemotron_cfg)
    assert ng.layer_types.count("ffn_only") == 2
    assert ng.n_moe_layers == 1
    assert ng.n_dense_layers == 1

    with pytest.raises(FileNotFoundError, match="Offline mode"):
        rf.load_hf_config("nonexistent-org/nonexistent-model-6a", offline=True)
