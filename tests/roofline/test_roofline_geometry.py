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

import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

import vllm_torchtpu.roofline as rf
from tests.roofline.conftest import NEMOTRON_CFG
from vllm_torchtpu.roofline import (
    CHIPS,
    TOPOLOGIES,
    ChipOverrides,
    SliceTopology,
    chips,
    count_layer_types,
)
from vllm_torchtpu.roofline import constants as rf_constants
from vllm_torchtpu.roofline import geometry as rf_geometry
from vllm_torchtpu.roofline.constants import (
    BASE_DTYPE_BYTES,
    BITS_PER_BYTE,
    DEFAULT_ICI_BW,
    DEFAULT_ICI_HOP_LATENCY_S,
    DEFAULT_ICI_LINK_BW,
    DEFAULT_VECTOR_TO_BF16_RATIO,
    FP8_BYTES,
    TPU7X_ICI_LINKS_PER_CHIP,
    UNIT_GIGA,
)
from vllm_torchtpu.roofline.geometry import (
    UnsupportedArchitectureError,
    from_hf_config,
    supports,
)
from vllm_torchtpu.roofline.partition import count_full_attn_layers


def test_topologies_exclude_subhost_shapes() -> None:
    """Verify TOPOLOGIES includes only provisionable TPU slice topologies."""
    expected_topologies = {"2x2x1", "2x2x2", "2x2x4", "2x4x4", "4x4x4"}
    assert "1x1x1" not in TOPOLOGIES
    assert "1x2x1" not in TOPOLOGIES
    assert set(TOPOLOGIES) == expected_topologies


def test_chip_table_matches_jax_tpu_info() -> None:
    """Verify CHIPS['tpu7x'] parameters exactly match JAX's TPU info table."""
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    chip = CHIPS["tpu7x"]
    assert chip.hbm_bytes == info.hbm_capacity_bytes
    assert chip.hbm_bw == info.mem_bw_bytes_per_second
    assert chip.bf16_flops == info.bf16_ops_per_second
    assert chip.fp8_flops == info.fp8_ops_per_second
    assert chip.vmem_bytes == info.vmem_capacity_bytes
    assert chip.cores_per_chip == (
        pltpu.ChipVersion.TPU_7X.num_physical_tensor_cores_per_chip
    )
    assert chip.mxu_m_tile == info.mxu_column_size
    assert chip.mxu_lhs_rows_bf16 == info.get_sublane_tiling(jnp.bfloat16)
    assert chip.mxu_lhs_rows_fp8 == info.get_sublane_tiling(jnp.float8_e4m3fn)
    assert chip.num_lanes == info.num_lanes
    assert chip.num_sublanes == info.num_sublanes
    assert chip.physical_alignment_elems == info.num_lanes
    assert chip.smem_page_table_bytes == (
        info.smem_capacity_bytes // rf_constants.SMEM_PAGE_TABLE_DIVISOR
    )
    assert chip.int8_flops == info.int8_ops_per_second
    assert chip.int4_flops == info.int4_ops_per_second
    assert chip.has_sparse_core is (info.sparse_core is not None)
    if info.sparse_core is not None:
        assert chip.dma_granule_bytes == info.sparse_core.dma_granule_size_bytes
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


def test_from_jax_hydrates_live_tpu6e_capabilities() -> None:
    """Verify from_jax hydrates live per-chip JAX TPU capabilities for TPU_V6E."""
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)
    v6e = chips.from_jax("tpu6e", pltpu.ChipVersion.TPU_V6E, ChipOverrides())
    assert v6e.hbm_bytes == info.hbm_capacity_bytes
    assert v6e.hbm_bw == info.mem_bw_bytes_per_second
    assert v6e.bf16_flops == info.bf16_ops_per_second
    assert v6e.fp8_flops == info.fp8_ops_per_second
    assert v6e.vmem_bytes == info.vmem_capacity_bytes
    assert v6e.cores_per_chip == int(
        pltpu.ChipVersion.TPU_V6E.num_physical_tensor_cores_per_chip
    )
    assert v6e.mxu_m_tile == info.mxu_column_size
    assert v6e.mxu_lhs_rows_bf16 == int(info.get_sublane_tiling(jnp.bfloat16))
    assert v6e.mxu_lhs_rows_fp8 == int(info.get_sublane_tiling(jnp.float8_e4m3fn))
    assert v6e.num_lanes == int(info.num_lanes)
    assert v6e.num_sublanes == int(info.num_sublanes)
    assert v6e.physical_alignment_elems == int(info.num_lanes)
    assert v6e.smem_page_table_bytes == (
        info.smem_capacity_bytes // rf_constants.SMEM_PAGE_TABLE_DIVISOR
    )
    assert v6e.ici_bw == pytest.approx(
        float(TPU7X_ICI_LINKS_PER_CHIP) * DEFAULT_ICI_LINK_BW
    )
    assert info.sparse_core is not None
    assert v6e.dma_granule_bytes == int(info.sparse_core.dma_granule_size_bytes)
    assert v6e.has_sparse_core is True
    assert v6e.int8_flops == info.int8_ops_per_second
    assert v6e.int4_flops == info.int4_ops_per_second


def test_from_jax_hydrates_tpu5e_without_sparse_core() -> None:
    """Verify from_jax sets has_sparse_core=False and dma_granule_bytes=0 on TPU_V5E."""
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V5E, 1)
    assert info.sparse_core is None
    v5e = chips.from_jax("tpu5e", pltpu.ChipVersion.TPU_V5E, ChipOverrides())
    assert v5e.has_sparse_core is False
    assert v5e.dma_granule_bytes == 0
    assert v5e.fp8_flops == 0.0
    assert v5e.int8_flops == info.int8_ops_per_second
    assert v5e.int4_flops == info.int4_ops_per_second


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

    ng = rf.from_hf_config(NEMOTRON_CFG)
    exp_ffn_only = 2
    assert ng.layer_types.count("ffn_only") == exp_ffn_only
    assert ng.n_moe_layers == 1
    assert ng.n_dense_layers == 1

    with pytest.raises(FileNotFoundError, match="Offline mode"):
        rf.load_hf_config("nonexistent-org/nonexistent-model-6a", offline=True)


def test_ffn_only_hybrid_override_excluded_from_full_attn_count() -> None:
    """Verify ffn_only layers from hybrid_override are excluded from full-attn count."""
    ng = rf.from_hf_config(NEMOTRON_CFG)
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
        pltpu.ChipVersion.TPU_7X,
        ChipOverrides(dcn_host_gbit_per_s=200.0, chips_per_host=4),
    )
    high = chips.from_jax(
        "tpu7x",
        pltpu.ChipVersion.TPU_7X,
        ChipOverrides(dcn_host_gbit_per_s=800.0, chips_per_host=4),
    )
    # 200 Gbit/s host NIC / 8 bits / 4 chips = 6.25 GB/s
    assert low.dcn_bw == pytest.approx((200.0e9 / 8.0) / 4.0)
    # 800 Gbit/s host NIC / 8 bits / 4 chips = 25.0 GB/s
    assert high.dcn_bw == pytest.approx((800.0e9 / 8.0) / 4.0)


def test_every_constant_has_kind_unit_scope_source() -> None:
    """Verify every catalog constant carries value, unit, scope, kind, and source."""
    tpu_info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
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
        "vector_flops": "derived",
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
    assert cat["vector_flops"]["value"] == pytest.approx(CHIPS["tpu7x"].vector_flops)
    assert cat["mxu_lhs_rows_bf16"]["value"] == tpu_info.get_sublane_tiling(
        jnp.bfloat16
    )
    assert cat["mxu_lhs_rows_fp8"]["value"] == tpu_info.get_sublane_tiling(
        jnp.float8_e4m3fn
    )
    bracket = cat["dcn_bw"]["bracket"]
    assert isinstance(bracket, tuple)
    assert list(bracket) == [(200.0e9 / 8.0) / 4.0, (800.0e9 / 8.0) / 4.0]


def test_second_chip_entry_loads_without_code_change() -> None:
    """Verify a second JAX ChipVersion loads via from_jax without code edits."""
    tpu6e_info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)
    synth_link_bw = 50.0e9
    synth_dcn_host_gbit = 200.0
    synth_chips_per_host = 4
    spec = chips.from_jax(
        "tpu6e",
        pltpu.ChipVersion.TPU_V6E,
        ChipOverrides(
            ici_link_bw=synth_link_bw,
            dcn_host_gbit_per_s=synth_dcn_host_gbit,
            chips_per_host=synth_chips_per_host,
        ),
    )
    assert spec.name == "tpu6e"
    assert spec.hbm_bytes == tpu6e_info.hbm_capacity_bytes
    assert spec.cores_per_chip == 1
    expected_d2d = float(TPU7X_ICI_LINKS_PER_CHIP) * synth_link_bw
    assert spec.d2d_bw == pytest.approx(expected_d2d)
    assert spec.ici_bw == pytest.approx(DEFAULT_ICI_BW)
    expected_dcn = (synth_dcn_host_gbit * 1.0e9 / 8.0) / synth_chips_per_host
    assert spec.dcn_bw == pytest.approx(expected_dcn)
    cat = spec.catalog()
    assert cat["hbm_bytes"]["value"] == tpu6e_info.hbm_capacity_bytes
    assert cat["dcn_bw"]["value"] == pytest.approx(expected_dcn)


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


SAMPLING_TOP_K_MOE_CFG: dict[str, object] = {
    "hidden_size": 4096,
    "num_hidden_layers": 8,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_experts": 896,
    "top_k": 50,
    "moe_intermediate_size": 1024,
    "vocab_size": 163840,
}


@pytest.mark.parametrize(
    "routed_experts_key",
    [
        "num_experts_per_tok",
        "num_experts_per_token",
        "experts_per_token",
        "num_selected_experts",
    ],
)
def test_config_with_experts_per_token_and_sampling_top_k_reads_routed_count(
    routed_experts_key: str,
) -> None:
    """Read each routed-experts key, not a sampling top_k in the same config."""
    expected_routed_experts_per_token = 16
    routed_and_sampling_cfg = {
        **SAMPLING_TOP_K_MOE_CFG,
        routed_experts_key: expected_routed_experts_per_token,
    }
    geom = from_hf_config(routed_and_sampling_cfg)
    assert geom.top_k == expected_routed_experts_per_token


def test_nested_text_config_with_sampling_top_k_reads_routed_count() -> None:
    """Read the routed experts per token from a nested text_config."""
    expected_routed_experts_per_token = 16
    nested_cfg = {
        "text_config": {
            **SAMPLING_TOP_K_MOE_CFG,
            "num_experts_per_token": expected_routed_experts_per_token,
        }
    }
    geom = from_hf_config(nested_cfg)
    assert geom.top_k == expected_routed_experts_per_token


def test_moe_config_without_routed_experts_key_is_rejected() -> None:
    """Reject a MoE config that names no routed experts per token."""
    with pytest.raises(ValueError, match="routed experts per token"):
        from_hf_config(SAMPLING_TOP_K_MOE_CFG)


def _verify_dsa_alias_output_gate_and_unmodeled_fail_closed() -> None:
    """Verify DSA alias counting, output_gate_type parsing, and fail-closed guards."""
    num_dsa_layers = 4
    dsa_alias_cfg: dict[str, object] = {
        "hidden_size": 4096,
        "num_hidden_layers": num_dsa_layers,
        "layer_types": ["deepseek_sparse_attention"] * num_dsa_layers,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "vocab_size": 32000,
    }
    supports(dsa_alias_cfg)
    dsa_geom = from_hf_config(dsa_alias_cfg)
    counts = count_layer_types(dsa_geom)
    assert counts.mla == num_dsa_layers
    assert counts.dsa == num_dsa_layers
    assert counts.attn_total == num_dsa_layers

    base_gqa_cfg: dict[str, object] = {
        "hidden_size": 2048,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "vocab_size": 32000,
    }
    assert (
        from_hf_config({**base_gqa_cfg, "output_gate_type": "sigmoid"}).attn_output_gate
        is True
    )
    assert (
        from_hf_config({**base_gqa_cfg, "output_gate_type": "none"}).attn_output_gate
        is False
    )
    assert (
        from_hf_config({**base_gqa_cfg, "output_gate_type": ""}).attn_output_gate
        is False
    )

    o_lora_cfg: dict[str, object] = {
        **dsa_alias_cfg,
        "layer_types": ["dsa"] * num_dsa_layers,
        "o_lora_rank": 512,
    }
    with pytest.raises(UnsupportedArchitectureError, match="o_lora_rank"):
        supports(o_lora_cfg)
    with pytest.raises(UnsupportedArchitectureError, match="o_lora_rank"):
        from_hf_config(o_lora_cfg)

    hybrid_swa_indexer_cfg: dict[str, object] = {
        **base_gqa_cfg,
        "sliding_window": 4096,
        "index_topk": 2048,
        "kv_lora_rank": 0,
    }
    with pytest.raises(UnsupportedArchitectureError, match="kv_lora_rank"):
        supports(hybrid_swa_indexer_cfg)
    with pytest.raises(UnsupportedArchitectureError, match="kv_lora_rank"):
        from_hf_config(hybrid_swa_indexer_cfg)

    valid_mla_dsa_with_sw_cfg: dict[str, object] = {
        **dsa_alias_cfg,
        "layer_types": ["dsa"] * num_dsa_layers,
        "sliding_window": 4096,
    }
    supports(valid_mla_dsa_with_sw_cfg)
    assert (
        count_layer_types(from_hf_config(valid_mla_dsa_with_sw_cfg)).dsa
        == num_dsa_layers
    )


test_deepseek_sparse_attention_alias_and_output_gate_type_and_unmodeled_fail_closed = (
    _verify_dsa_alias_output_gate_and_unmodeled_fail_closed
)


def test_geometry_boundary_guards_and_multimodal_first_k_dense_replace() -> None:
    """Verify num_heads<=0, sliding_window_pattern==1, and multimodal MoE offset."""
    zero_heads_cfg: dict[str, object] = {
        "hidden_size": 2048,
        "num_hidden_layers": 2,
        "num_attention_heads": 0,
        "vocab_size": 32000,
    }
    with pytest.raises(ValueError, match="num_attention_heads"):
        from_hf_config(zero_heads_cfg)

    num_sw_layers = 4
    sw_pat_one_cfg: dict[str, object] = {
        "hidden_size": 2048,
        "num_hidden_layers": num_sw_layers,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "sliding_window": 4096,
        "sliding_window_pattern": 1,
        "vocab_size": 32000,
    }
    assert (
        from_hf_config(sw_pat_one_cfg).layer_types
        == ("full_attention",) * num_sw_layers
    )

    num_mm_layers = 6
    top_first_k = 2
    expected_moe_layers = num_mm_layers - top_first_k
    mm_cfg: dict[str, object] = {
        "first_k_dense_replace": top_first_k,
        "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": num_mm_layers,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "num_experts": 16,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 1024,
            "vocab_size": 32000,
        },
    }
    mm_geom = from_hf_config(mm_cfg)
    assert mm_geom.first_k_dense_replace == top_first_k
    assert mm_geom.n_dense_layers == top_first_k
    assert mm_geom.n_moe_layers == expected_moe_layers
    assert (
        mm_geom.ffn_kinds == ("dense",) * top_first_k + ("moe",) * expected_moe_layers
    )


def test_slice_topology_validates_axes_and_chip_overrides_decouples_ici() -> None:
    """Verify SliceTopology axis validation, decoupled ICI BW, catalog, and dtypes."""
    bad_axes: tuple[int, ...] = (2, 2)
    with pytest.raises(ValueError, match="chip_axes"):
        SliceTopology("bad_axes", bad_axes)
    bad_wrapped: tuple[bool, ...] = (True, False)
    with pytest.raises(ValueError, match="wrapped_axes"):
        SliceTopology("bad_wrapped", (2, 2, 1), wrapped_axes=bad_wrapped)

    custom_d2d = 900.0e9
    overridden_chip = chips.from_jax(
        "tpu7x", pltpu.ChipVersion.TPU_7X, ChipOverrides(d2d_bw=custom_d2d)
    )
    assert overridden_chip.d2d_bw == custom_d2d
    assert overridden_chip.ici_bw == DEFAULT_ICI_BW
    cat = overridden_chip.catalog()
    assert "dedicated" in str(cat["d2d_bw"]["description"]).lower()
    assert "across all 6 ici links" not in str(cat["d2d_bw"]["description"]).lower()
    assert "6 * ici_link_bw / cores_per_chip" in str(cat["ici_bw"]["source"])
    assert "Roofline default 0.92 (vLLM CacheConfig default is 0.90)" in str(
        cat["default_gpu_memory_utilization"]["source"]
    )
    for fp8_alias in ("float8_e8m0fnu", "e8m0", "float8_e5m2"):
        assert BASE_DTYPE_BYTES[fp8_alias] == FP8_BYTES


def test_from_jax_applies_chip_overrides() -> None:
    """Verify from_jax applies keyword and ChipOverrides rates over JAX info."""
    tpu_info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    custom_cores = 4
    custom_link_bw = 120.0 * UNIT_GIGA
    custom_hop_s = 2.5e-6
    base_overrides = ChipOverrides(
        dcn_host_gbit_per_s=800.0,
        chips_per_host=4,
    )
    spec_kw = chips.from_jax(
        "tpu7x",
        pltpu.ChipVersion.TPU_7X,
        overrides=base_overrides,
        cores_per_chip=custom_cores,
        ici_link_bw=custom_link_bw,
        ici_hop_latency_s=custom_hop_s,
    )
    expected_d2d = float(TPU7X_ICI_LINKS_PER_CHIP) * custom_link_bw
    assert spec_kw.hbm_bytes == tpu_info.hbm_capacity_bytes
    assert spec_kw.cores_per_chip == custom_cores
    assert spec_kw.ici_hop_latency_s == pytest.approx(custom_hop_s, rel=1e-12)
    assert spec_kw.d2d_bw == pytest.approx(expected_d2d, rel=1e-12)
    assert spec_kw.ici_bw == pytest.approx(expected_d2d / custom_cores, rel=1e-12)
    assert spec_kw.dcn_bw == pytest.approx(
        (800.0 * UNIT_GIGA / BITS_PER_BYTE) / 4.0, rel=1e-12
    )
    assert spec_kw.vector_flops == pytest.approx(
        float(tpu_info.bf16_ops_per_second) * DEFAULT_VECTOR_TO_BF16_RATIO, rel=1e-12
    )

    explicit = ChipOverrides(
        d2d_bw=500.0 * UNIT_GIGA,
        ici_bw=220.0 * UNIT_GIGA,
        dcn_bw=40.0 * UNIT_GIGA,
        vector_flops=12.0e12,
    )
    spec_exp = chips.from_jax(
        "tpu7x", pltpu.ChipVersion.TPU_7X, overrides=explicit, ici_bw=210.0e9
    )
    assert spec_exp.d2d_bw == pytest.approx(500.0 * UNIT_GIGA, rel=1e-12)
    assert spec_exp.ici_bw == pytest.approx(210.0e9, rel=1e-12)
    assert spec_exp.dcn_bw == pytest.approx(40.0 * UNIT_GIGA, rel=1e-12)
    assert spec_exp.vector_flops == pytest.approx(12.0e12, rel=1e-12)


def test_from_jax_rejects_non_chip_version() -> None:
    """Verify from_jax raises TypeError when version is not a ChipVersion."""
    with pytest.raises(TypeError, match="ChipVersion"):
        chips.from_jax("unknown_tpu", "unknown_tpu")
