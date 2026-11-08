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

from vllm_torchtpu.roofline import CHIPS, chips
from vllm_torchtpu.roofline.geometry import from_hf_config

_EXPECTED_DENSE_INTER: int = 12288
_EXPECTED_MOE_INTER: int = 2048
_EXPECTED_TOP_K: int = 8

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
    """Verify CHIPS['tpu7x'] parameters exactly match JAX's TPU info table."""
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    c = CHIPS["tpu7x"]
    assert c.hbm_bytes == info.hbm_capacity_bytes
    assert c.hbm_bw == info.mem_bw_bytes_per_second
    assert c.bf16_flops == info.bf16_ops_per_second
    assert c.fp8_flops == info.fp8_ops_per_second
    assert c.vmem_bytes == info.vmem_capacity_bytes
    assert (c.ici_link_bw, c.ici_bw, c.ici_hop_latency_s) == (100.0e9, 300.0e9, 1.0e-6)


def test_core_has_no_engine_imports() -> None:
    """Ensure roofline core package has zero dependencies on torch or vllm."""
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
    """Exercise HF config alias resolution for DSA and latent MoE dimensions."""
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
    assert geom.dense_inter == _EXPECTED_DENSE_INTER
    assert geom.moe_inter == _EXPECTED_MOE_INTER
    assert geom.top_k == _EXPECTED_TOP_K


def test_linear_conv_kernel_dim_validation() -> None:
    """Validate linear conv kernel dimension for linear attention models."""
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


def test_chips_stdlib_fallback_when_jax_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fallback chip specs match JAX TPU table when JAX Pallas is absent."""
    jax_spec = CHIPS["tpu7x"]
    monkeypatch.setattr(chips, "pltpu", None)
    fallback = chips.from_jax(
        "tpu7x",
        "tpu7x",
        ici_link_bw=100.0e9,
        ici_bw=300.0e9,
        ici_hop_latency_s=1.0e-6,
    )
    assert fallback.hbm_bytes == jax_spec.hbm_bytes
    assert fallback.hbm_bw == jax_spec.hbm_bw
    assert fallback.bf16_flops == jax_spec.bf16_flops
    assert fallback.fp8_flops == jax_spec.fp8_flops
    assert fallback.vmem_bytes == jax_spec.vmem_bytes
    assert fallback.cores_per_chip == jax_spec.cores_per_chip
    assert fallback.ici_link_bw == jax_spec.ici_link_bw
    assert fallback.ici_bw == jax_spec.ici_bw
    assert fallback.ici_hop_latency_s == jax_spec.ici_hop_latency_s
