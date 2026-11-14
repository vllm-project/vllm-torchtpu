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
import re
from pathlib import Path
from types import MappingProxyType

import pytest

from vllm_torchtpu.roofline import CHIPS, TOPOLOGIES, ChipOverrides, chips
from vllm_torchtpu.roofline.constants import (
    DEFAULT_ICI_BW,
    DEFAULT_ICI_HOP_LATENCY_S,
    DEFAULT_ICI_LINK_BW,
)
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


def test_topologies_exclude_subhost_shapes() -> None:
    """Verify TOPOLOGIES includes only provisionable TPU slice topologies."""
    expected_topologies = {"2x2x1", "2x2x2", "2x2x4", "2x4x4", "4x4x4"}
    assert "1x1x1" not in TOPOLOGIES
    assert "1x2x1" not in TOPOLOGIES
    assert set(TOPOLOGIES) == expected_topologies


def test_chip_table_matches_jax_tpu_info() -> None:
    """Verify CHIPS['tpu7x'] parameters exactly match JAX's TPU info table."""
    pltpu = pytest.importorskip("jax.experimental.pallas.tpu")
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    chip = CHIPS["tpu7x"]
    assert chip.hbm_bytes == info.hbm_capacity_bytes
    assert chip.hbm_bw == info.mem_bw_bytes_per_second
    assert chip.bf16_flops == info.bf16_ops_per_second
    assert chip.fp8_flops == info.fp8_ops_per_second
    assert chip.vmem_bytes == info.vmem_capacity_bytes
    assert (chip.ici_link_bw, chip.ici_bw, chip.ici_hop_latency_s) == (
        DEFAULT_ICI_LINK_BW,
        DEFAULT_ICI_BW,
        DEFAULT_ICI_HOP_LATENCY_S,
    )


def test_core_has_no_engine_imports() -> None:
    """Ensure roofline core package has zero dependencies on torch or vllm."""
    pkg_dir = Path(__file__).resolve().parents[2] / "src/vllm_torchtpu/roofline"
    banned = {"vllm", "torch", "torch_tpu", "numpy", "vllm_torchtpu"}
    for py_file in sorted(pkg_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                assert node.module.split(".")[0] not in banned


def test_from_hf_config_dsa_and_latent_moe_aliases() -> None:
    """Exercise HF config alias resolution for DSA and latent MoE dimensions."""
    expected_dense_inter = 12288
    expected_moe_inter = 2048
    expected_top_k = 8
    synthetic_mla_cfg = MappingProxyType(
        {
            "hidden_size": 6144,
            "num_hidden_layers": 78,
            "layer_types": ("dsa",) * 78,
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
    )
    geom = from_hf_config(synthetic_mla_cfg)
    assert geom.dense_inter == expected_dense_inter
    assert geom.moe_inter == expected_moe_inter
    assert geom.top_k == expected_top_k


def test_from_hf_config_sliding_window_kda_and_zero_kv_heads() -> None:
    """Verify uniform sliding window, capability-based KDA, and explicit 0 KV heads."""
    swa_geom = from_hf_config(
        {
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "sliding_window": 4096,
            "vocab_size": 32000,
        }
    )
    assert swa_geom.layer_types == ("sliding_attention",) * 4
    kda_geom = from_hf_config(
        {
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 0,
            "full_attention_interval": 4,
            "linear_attn_config": {
                "num_heads": 16,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
            },
            "vocab_size": 32000,
        }
    )
    assert kda_geom.num_kv_heads == 0
    assert kda_geom.layer_types == ("kda", "kda", "kda", "full_attention")


def test_linear_conv_kernel_dim_validation() -> None:
    """Validate linear conv kernel dimension for linear attention models."""
    cfg_bad = {
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 4,
        "num_attention_heads": 32,
    }
    with pytest.raises(ValueError, match="linear_conv_kernel_dim"):
        from_hf_config(cfg_bad)
    cfg_non_linear = {
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "layer_types": ["full_attention"] * 4,
        "num_attention_heads": 32,
    }
    geom = from_hf_config(cfg_non_linear)
    assert geom.conv_kernel == 0


def test_chips_stdlib_fallback_when_jax_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify fallback chip specs match JAX TPU table when JAX Pallas is absent."""
    jax_spec = CHIPS["tpu7x"]
    monkeypatch.setattr(chips, "pltpu", None)
    fallback = chips.from_jax("tpu7x", "tpu7x", ChipOverrides())
    assert fallback.hbm_bytes == jax_spec.hbm_bytes
    assert fallback.hbm_bw == jax_spec.hbm_bw
    assert fallback.bf16_flops == jax_spec.bf16_flops
    assert fallback.fp8_flops == jax_spec.fp8_flops
    assert fallback.vmem_bytes == jax_spec.vmem_bytes
    assert fallback.cores_per_chip == jax_spec.cores_per_chip
    assert fallback.ici_link_bw == jax_spec.ici_link_bw
    assert fallback.ici_bw == jax_spec.ici_bw
    assert fallback.ici_hop_latency_s == jax_spec.ici_hop_latency_s


def test_chips_sources_contain_no_legacy_tpu_v5_or_v4_references() -> None:
    """Verify constant catalog contains no legacy TPU v4/v5 mentions."""
    pattern = re.compile(r"(?i)\bv5[ep]?\b|\bv4\b|TPU v[1-5]")
    for chip in CHIPS.values():
        cat = chip.catalog()
        for key, entry in cat.items():
            source = str(entry.get("source", ""))
            desc = str(entry.get("description", ""))
            assert not pattern.search(source), (
                f"legacy reference in {key}.source: {source}"
            )
            assert not pattern.search(desc), (
                f"legacy reference in {key}.description: {desc}"
            )
