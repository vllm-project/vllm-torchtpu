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
"""Unit tests for TPU chip specifications, model geometry, and stdlib isolation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from vllm_torchtpu.roofline import CHIPS
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
    jti = pytest.importorskip("jax._src.tpu_info")

    info = jti.get_tpu_info_for_chip(jti.ChipVersion.TPU_7X, 1)
    c = CHIPS["tpu7x"]
    assert c.hbm_bytes == info.hbm_capacity_bytes
    assert c.hbm_bw == pytest.approx(info.mem_bw_bytes_per_second)
    assert c.bf16_flops == pytest.approx(info.bf16_ops_per_second)
    assert c.fp8_flops == 2.307e15
    assert c.fp8_flops == pytest.approx(info.fp8_ops_per_second, rel=1e-2)
    assert c.vmem_bytes == info.vmem_capacity_bytes


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
