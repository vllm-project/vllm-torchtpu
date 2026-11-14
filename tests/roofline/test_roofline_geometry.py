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

import vllm_torchtpu.roofline as rf
from vllm_torchtpu.roofline import CHIPS, TOPOLOGIES, ChipOverrides, chips
from vllm_torchtpu.roofline import constants as rf_constants
from vllm_torchtpu.roofline import geometry as rf_geometry
from vllm_torchtpu.roofline.constants import (
    DEFAULT_ICI_BW,
    DEFAULT_ICI_HOP_LATENCY_S,
    DEFAULT_ICI_LINK_BW,
)
from vllm_torchtpu.roofline.geometry import from_hf_config
from vllm_torchtpu.roofline.partition import count_full_attn_layers

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
    assert (
        chip.ici_link_bw,
        chip.ici_bw,
        chip.ici_hop_latency_s,
    ) == (
        DEFAULT_ICI_LINK_BW,
        DEFAULT_ICI_BW,
        DEFAULT_ICI_HOP_LATENCY_S,
    )


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


def test_chips_stdlib_fallback_when_jax_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fallback chip specs match JAX TPU table when JAX Pallas is absent."""
    jax_spec = CHIPS["tpu7x"]
    monkeypatch.setattr(chips, "pltpu", None)
    fallback = chips.from_jax(
        "tpu7x",
        "tpu7x",
        ChipOverrides(),
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


def test_supports_validator_fails_closed_on_unknown_arch() -> None:
    """Verify supports rejects unsupported or invalid architectures."""
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
    """Verify parsing of Kimi linear, Llama4, Gemma4, MiniMax, and Nemotron configs."""
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
    exp_kda_layers = 6
    exp_mla_layers = 2
    exp_conv_kernel = 4
    exp_k_heads = 32
    exp_k_dim = 128
    assert kg.layer_types.count("kda") == exp_kda_layers
    assert kg.layer_types.count("mla") == exp_mla_layers
    assert kg.conv_kernel == exp_conv_kernel
    assert kg.linear_k_heads == exp_k_heads
    assert kg.linear_k_dim == exp_k_dim

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
    exp_moe_layers = 3
    assert lg.n_moe_layers == exp_moe_layers

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
    exp_global_head_dim = 512
    exp_full_layers = 1
    exp_sliding_layers = 5
    assert gg.global_head_dim == exp_global_head_dim
    assert gg.layer_types.count("full_attention") == exp_full_layers
    assert gg.layer_types.count("sliding_attention") == exp_sliding_layers

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
    exp_ffn_only = 2
    assert ng.layer_types.count("ffn_only") == exp_ffn_only
    assert ng.n_moe_layers == 1
    assert ng.n_dense_layers == 1

    with pytest.raises(FileNotFoundError, match="Offline mode"):
        rf.load_hf_config("nonexistent-org/nonexistent-model-6a", offline=True)


def test_ffn_only_hybrid_override_excluded_from_full_attn_count() -> None:
    """Verify ffn_only layers from hybrid_override are excluded from full-attn count."""
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
    exp_ffn_count = 2
    assert ng.layer_types.count("ffn_only") == exp_ffn_count
    assert ng.ffn_layer_counts() == (1, 1)
    assert ng.ffn_kinds == ("none", "dense", "none", "none", "none", "moe")
    assert count_full_attn_layers(ng) == 1


def test_dcn_per_chip_derives_from_host_bandwidth() -> None:
    """Verify per-chip DCN bandwidth is derived from host Gbit/s and chip count."""
    chip = CHIPS["tpu7x"]
    # 400 Gbit/s host NIC / 8 bits / 4 chips per host = 12.5 GB/s
    expected_dcn_bw = (400.0e9 / 8.0) / 4.0
    assert chip.dcn_bw == pytest.approx(expected_dcn_bw)
    low = chips.from_jax(
        "tpu7x",
        "tpu7x",
        ChipOverrides(dcn_host_gbit_per_s=200.0, chips_per_host=4),
    )
    high = chips.from_jax(
        "tpu7x",
        "tpu7x",
        ChipOverrides(dcn_host_gbit_per_s=800.0, chips_per_host=4),
    )
    # 200 Gbit/s host NIC / 8 bits / 4 chips = 6.25 GB/s
    assert low.dcn_bw == pytest.approx((200.0e9 / 8.0) / 4.0)
    # 800 Gbit/s host NIC / 8 bits / 4 chips = 25.0 GB/s
    assert high.dcn_bw == pytest.approx((800.0e9 / 8.0) / 4.0)


def test_every_constant_has_kind_unit_scope_source() -> None:
    """Verify every catalog constant carries value, unit, scope, kind, and source."""
    cat = CHIPS["tpu7x"].catalog()
    expected_kinds = {
        "hbm_bytes": "fact",
        "hbm_bw": "fact",
        "bf16_flops": "fact",
        "fp8_flops": "fact",
        "vmem_bytes": "fact",
        "cores_per_chip": "fact",
        "ici_link_bw": "derived",
        "d2d_bw": "derived",
        "ici_bw": "derived",
        "dcn_bw": "derived",
        "vpu_flops": "derived",
        "mxu_lhs_rows_bf16": "fact",
        "mxu_lhs_rows_fp8": "fact",
        "ici_hop_latency_s": "assumed",
        "pcie_bw": "assumed",
        "mxu_m_tile": "assumed",
        "dcn_host_gbit_per_s": "assumed",
        "chips_per_host": "assumed",
        "page_table_byte_budget": "assumed",
        "rpa_page_size_tokens": "assumed",
        "default_gpu_memory_utilization": "config",
    }
    for key, exp_kind in expected_kinds.items():
        assert key in cat, f"missing catalog key {key}"
        entry = cat[key]
        assert entry["kind"] == exp_kind
        assert entry["scope"] in {"core", "chip", "host", "link"}
        unit = entry["unit"]
        source = entry["source"]
        assert isinstance(unit, str)
        assert bool(unit.strip())
        assert isinstance(source, str)
        assert bool(source.strip())
        assert "value" in entry
        if exp_kind == "assumed":
            assert entry["bracket"] is not None
    # 400 Gbit/s host NIC / 8 bits / 4 chips per host = 12.5 GB/s
    assert cat["dcn_bw"]["value"] == pytest.approx((400.0e9 / 8.0) / 4.0)
    assert cat["vpu_flops"]["value"] == pytest.approx(CHIPS["tpu7x"].vpu_flops)
    assert cat["mxu_lhs_rows_bf16"]["value"] == rf_constants.DEFAULT_MXU_LHS_ROWS_BF16
    assert cat["mxu_lhs_rows_fp8"]["value"] == rf_constants.DEFAULT_MXU_LHS_ROWS_FP8
    bracket = cat["dcn_bw"]["bracket"]
    assert isinstance(bracket, tuple)
    assert list(bracket) == [(200.0e9 / 8.0) / 4.0, (800.0e9 / 8.0) / 4.0]


def test_second_chip_entry_loads_without_code_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fallback table entry loads via from_jax without code edits."""
    synth_hbm_bytes = 34_400_000_000
    synth_hbm_bw = 1.64e12
    synth_bf16_flops = 0.918e15
    synth_fp8_flops = 1.836e15
    synth_vmem_bytes = 134_217_728
    synth_link_bw = 50.0e9
    synth_dcn_host_gbit = 200.0
    synth_chips_per_host = 4
    monkeypatch.setitem(
        chips.FALLBACK_TPU_SPECS,
        "synth_chip",
        (
            synth_hbm_bytes,
            synth_hbm_bw,
            synth_bf16_flops,
            synth_fp8_flops,
            synth_vmem_bytes,
        ),
    )
    spec = chips.from_jax(
        "synth_chip",
        "synth_chip",
        ChipOverrides(
            cores_per_chip=1,
            ici_link_bw=synth_link_bw,
            dcn_host_gbit_per_s=synth_dcn_host_gbit,
            chips_per_host=synth_chips_per_host,
        ),
    )
    assert spec.name == "synth_chip"
    assert spec.hbm_bytes == synth_hbm_bytes
    assert spec.cores_per_chip == 1
    # 1 core * 300 GB/s = 300 GB/s
    expected_synth_d2d = float(spec.cores_per_chip) * DEFAULT_ICI_BW
    assert spec.d2d_bw == pytest.approx(expected_synth_d2d)
    assert spec.ici_bw == pytest.approx(DEFAULT_ICI_BW)
    # 200 Gbit/s host / 8 bits / 4 chips = 6.25 GB/s
    expected_synth_dcn = (synth_dcn_host_gbit * 1.0e9 / 8.0) / synth_chips_per_host
    assert spec.dcn_bw == pytest.approx(expected_synth_dcn)
    cat = spec.catalog()
    assert cat["hbm_bytes"]["value"] == synth_hbm_bytes
    assert cat["dcn_bw"]["value"] == pytest.approx(expected_synth_dcn)


def test_supports_closed_world_model_type_allowlist() -> None:
    """Assert closed-world allowlist accepts head & served types and rejects others."""
    # 1. Verify head-test and served model families are accepted
    head_test_model_types = (
        "llama",
        "llama4_text",
        "qwen3_moe",
        "qwen3_5_moe",
        "qwen3_8_moe",
        "deepseek_v4",
        "glm5",
        "kimi_linear",
        "gpt_oss",
        "gemma4_text",
        "minimax_text_01",
    )
    for model_type in head_test_model_types:
        sample_config: dict[str, object] = {
            "model_type": model_type,
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "vocab_size": 32000,
        }
        if model_type == "kimi_linear":
            sample_config["linear_attn_config"] = {
                "num_heads": 16,
                "head_dim": 128,
            }
        rf.supports(sample_config)

    # Hybrid SSM with GDN keys must be accepted
    nemotron_valid: dict[str, object] = {
        "model_type": "nemotron_h",
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "mamba_num_heads": 64,
        "vocab_size": 32000,
    }
    rf.supports(nemotron_valid)
    expected_chips = frozenset(rf.CHIPS)
    assert expected_chips == rf.SUPPORTED_CHIPS

    # Served multimodal and custom causal architecture class names must pass supports()
    served_arch_configs: tuple[dict[str, object], ...] = (
        {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 4096,
                "num_hidden_layers": 4,
                "num_attention_heads": 32,
                "vocab_size": 248320,
            },
        },
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 4096,
                "num_hidden_layers": 4,
                "num_attention_heads": 32,
                "vocab_size": 248320,
            },
        },
        {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "hidden_size": 4096,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "vocab_size": 151936,
        },
        {
            "architectures": ["KimiK3ForConditionalGeneration"],
            "model_type": "kimi_k3",
            "hidden_size": 7168,
            "num_hidden_layers": 4,
            "num_attention_heads": 64,
            "vocab_size": 163840,
        },
    )
    for served_cfg in served_arch_configs:
        rf.supports(served_cfg)

    # 2. Verify unlisted, pure-SSM, and non-causal types raise
    # UnsupportedArchitectureError
    rejected_configs: tuple[dict[str, object], ...] = (
        {
            "model_type": "bert",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "vocab_size": 30522,
        },
        {
            "model_type": "roberta",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "vocab_size": 50265,
        },
        {
            "model_type": "t5",
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "vocab_size": 32128,
        },
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
            "vocab_size": 32000,
        },
        {
            "model_type": "falcon_mamba",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "vocab_size": 32000,
        },
        {
            "model_type": "unknown_future_ssm",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "vocab_size": 32000,
        },
        # Hybrid SSM without GDN keys must fail closed
        {
            "model_type": "nemotron_h",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "vocab_size": 32000,
        },
    )
    for bad_cfg in rejected_configs:
        with pytest.raises(rf.UnsupportedArchitectureError):
            rf.supports(bad_cfg)


def test_explicit_gqa_token_stays_gqa_when_kv_lora_rank_positive() -> None:
    """Verify count_layer_types keeps explicit gqa tokens when kv_lora_rank > 0."""
    geom = from_hf_config(
        {
            "hidden_size": 4096,
            "num_hidden_layers": 4,
            "layer_types": ("gqa", "gqa", "dsa", "full_attention"),
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 32000,
            "kv_lora_rank": 512,
            "index_topk": 2048,
        }
    )
    counts = rf_geometry.count_layer_types(geom)
    expected_gqa_count = 2
    expected_mla_count = 2
    expected_dsa_count = 2
    expected_swa_count = 0
    expected_gdn_count = 0
    expected_kda_count = 0
    expected_total_count = 4
    assert counts.gqa == expected_gqa_count
    assert counts.mla == expected_mla_count
    assert counts.dsa == expected_dsa_count
    assert counts.swa == expected_swa_count
    assert counts.gdn == expected_gdn_count
    assert counts.kda == expected_kda_count
    assert counts.attn_total == expected_total_count


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
