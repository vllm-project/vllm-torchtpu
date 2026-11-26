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
from typing import cast

import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

from tests.roofline.conftest import load_locked_config
from vllm_torchtpu.roofline import (
    CHIPS,
    TOPOLOGIES,
    ChipOverrides,
    SliceTopology,
    chips,
    count_layer_types,
)
from vllm_torchtpu.roofline.constants import (
    BASE_DTYPE_BYTES,
    DEFAULT_ICI_BW,
    DEFAULT_ICI_HOP_LATENCY_S,
    DEFAULT_ICI_LINK_BW,
    DEFAULT_ICI_LINKS_PER_CHIP,
    FP8_BYTES,
)
from vllm_torchtpu.roofline.geometry import (
    UnsupportedArchitectureError,
    _from_hf_config_dry_run,
    from_hf_config,
    supports,
)


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
    assert chip.int8_flops == info.int8_ops_per_second
    assert chip.int4_flops == info.int4_ops_per_second
    assert chip.has_sparse_core is (info.sparse_core is not None)
    if info.sparse_core is not None:
        assert chip.dma_granule_bytes == info.sparse_core.dma_granule_size_bytes
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
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    assert node.module.split(".")[0] not in banned
                elif node.level > 1:
                    is_parent_envs = (
                        node.level == 2
                        and node.module is None
                        and all(alias.name == "envs" for alias in node.names)
                    )
                    assert is_parent_envs, (
                        f"Disallowed parent relative import in {py_file.name}: "
                        f"level={node.level} module={node.module!r}"
                    )


def test_from_hf_config_dsa_and_latent_moe_aliases() -> None:
    """Exercise HF config alias resolution for DSA and latent MoE dimensions."""
    expected_dense_inter = 12288
    expected_moe_inter = 2048
    expected_top_k = 8
    synthetic_mla_cfg = MappingProxyType(
        {
            "model_type": "glm_moe_dsa",
            "architectures": ["GlmMoeDsaForCausalLM"],
            "hidden_size": 6144,
            "num_hidden_layers": 78,
            "layer_types": ("dsa",) * 78,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "head_dim": 192,
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
    geometry = from_hf_config(synthetic_mla_cfg)
    assert geometry.dense_inter == expected_dense_inter
    assert geometry.moe_inter == expected_moe_inter
    assert geometry.top_k == expected_top_k
    assert geometry.head_dim == 192


def test_from_hf_config_sliding_window_kda_and_zero_kv_heads() -> None:
    """Verify sliding_window pattern guard, KDA parsing, and explicit 0 KV heads."""
    with pytest.raises(ValueError, match="sliding_window"):
        from_hf_config(
            {
                "model_type": "qwen3_5_moe",
                "hidden_size": 2048,
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "sliding_window": 4096,
                "vocab_size": 32000,
            }
        )
    disabled_swa_geom = from_hf_config(
        {
            "model_type": "qwen3_5_moe",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "sliding_window": 4096,
            "use_sliding_window": False,
            "vocab_size": 32000,
        }
    )
    assert disabled_swa_geom.layer_types == ("full_attention",) * 4
    kda_geom = _from_hf_config_dry_run(
        {
            "model_type": "custom_kda",
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
        "model_type": "qwen3_5_moe",
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 4,
        "num_attention_heads": 32,
    }
    with pytest.raises(ValueError, match="linear_conv_kernel_dim"):
        from_hf_config(cfg_bad)
    cfg_non_linear = {
        "model_type": "qwen3_5_moe",
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "layer_types": ["full_attention"] * 4,
        "num_attention_heads": 32,
    }
    geometry = from_hf_config(cfg_non_linear)
    assert geometry.conv_kernel == 0


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
    assert v6e.ici_bw == pytest.approx(
        float(DEFAULT_ICI_LINKS_PER_CHIP) * DEFAULT_ICI_LINK_BW
    )
    assert info.sparse_core is not None
    assert v6e.dma_granule_bytes == int(info.sparse_core.dma_granule_size_bytes)
    assert v6e.has_sparse_core is True
    assert v6e.int8_flops == info.int8_ops_per_second
    assert v6e.int4_flops == info.int4_ops_per_second


def test_from_jax_override_disables_sparse_core_cleanly() -> None:
    """Verify ChipOverrides(has_sparse_core=False) zeroes dma_granule_bytes."""
    no_sc_chip = chips.from_jax(
        "tpu7x_no_sc",
        pltpu.ChipVersion.TPU_7X,
        ChipOverrides(has_sparse_core=False),
    )
    assert no_sc_chip.has_sparse_core is False
    assert no_sc_chip.dma_granule_bytes == 0
    assert no_sc_chip.fp8_flops == CHIPS["tpu7x"].fp8_flops


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
    "model_type": "qwen3_5_moe",
    "architectures": ["Qwen3_5MoeForCausalLM"],
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
    geometry = from_hf_config(routed_and_sampling_cfg)
    assert geometry.top_k == expected_routed_experts_per_token


def test_nested_text_config_with_sampling_top_k_reads_routed_count() -> None:
    """Read the routed experts per token from a nested text_config."""
    expected_routed_experts_per_token = 16
    nested_cfg = {
        "text_config": {
            **SAMPLING_TOP_K_MOE_CFG,
            "num_experts_per_token": expected_routed_experts_per_token,
        }
    }
    geometry = from_hf_config(nested_cfg)
    assert geometry.top_k == expected_routed_experts_per_token


def test_moe_config_without_routed_experts_key_is_rejected() -> None:
    """Reject a MoE config that names no routed experts per token."""
    with pytest.raises(ValueError, match="routed experts per token"):
        from_hf_config(SAMPLING_TOP_K_MOE_CFG)


def test_deepseek_sparse_attention_alias_and_sliding_window_mla() -> None:
    """Verify deepseek_sparse_attention alias and MLA+DSA with sliding_window."""
    glm_cfg = load_locked_config("glm_5_2_fp8.json")
    num_dsa_layers = 4
    dsa_alias_cfg: dict[str, object] = {
        **glm_cfg,
        "num_hidden_layers": num_dsa_layers,
        "layer_types": ["deepseek_sparse_attention"] * num_dsa_layers,
        "indexer_types": ["full"] * num_dsa_layers,
        "mlp_layer_types": ["moe"] * num_dsa_layers,
        "first_k_dense_replace": 0,
    }
    supports(dsa_alias_cfg)
    dsa_geom = from_hf_config(dsa_alias_cfg)
    counts = count_layer_types(dsa_geom)
    assert (counts.mla, counts.dsa, counts.attn_total) == (
        num_dsa_layers,
        num_dsa_layers,
        num_dsa_layers,
    )

    valid_mla_dsa_with_sw_cfg = {
        **dsa_alias_cfg,
        "layer_types": ["dsa"] * num_dsa_layers,
        "sliding_window": 4096,
    }
    supports(valid_mla_dsa_with_sw_cfg)
    assert (
        count_layer_types(from_hf_config(valid_mla_dsa_with_sw_cfg)).dsa
        == num_dsa_layers
    )


@pytest.mark.parametrize(
    ("gate_value", "expected_gate"),
    [("sigmoid", True), ("none", False), ("", False)],
)
def test_output_gate_type_parsing(gate_value: str, expected_gate: bool) -> None:
    """Verify output_gate_type strings map cleanly to attn_output_gate booleans."""
    base_gqa_cfg: dict[str, object] = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "vocab_size": 32000,
        "output_gate_type": gate_value,
    }
    assert from_hf_config(base_gqa_cfg).attn_output_gate is expected_gate


@pytest.mark.parametrize(
    ("override", "match_text"),
    [
        ({"o_lora_rank": 512}, "o_lora_rank"),
        (
            {"sliding_window": 4096, "index_topk": 2048, "kv_lora_rank": 0},
            "kv_lora_rank",
        ),
    ],
)
def test_unmodeled_attention_variants_fail_closed(
    override: dict[str, object],
    match_text: str,
) -> None:
    """Verify o_lora_rank and non-MLA sliding_window+index_topk fail closed."""
    base_cfg: dict[str, object] = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "vocab_size": 32000,
        **override,
    }
    with pytest.raises(UnsupportedArchitectureError, match=match_text):
        supports(base_cfg)
    with pytest.raises(UnsupportedArchitectureError, match=match_text):
        from_hf_config(base_cfg)


def test_compress_ratios_longer_than_num_layers_fails_closed() -> None:
    """Verify compress_ratios longer than num_hidden_layers raises ValueError."""
    base_cfg: dict[str, object] = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "compress_ratios": [1.0, 2.0, 4.0],
        "vocab_size": 32000,
    }
    with pytest.raises(ValueError, match="compress_ratios length"):
        from_hf_config(base_cfg)


def test_geometry_boundary_guards_and_multimodal_first_k_dense_replace() -> None:
    """Verify num_heads<=0, sliding_window_pattern==1, and multimodal MoE offset."""
    zero_heads_cfg: dict[str, object] = {
        "model_type": "qwen3_5_moe",
        "hidden_size": 2048,
        "num_hidden_layers": 2,
        "num_attention_heads": 0,
        "vocab_size": 32000,
    }
    with pytest.raises(ValueError, match="num_attention_heads"):
        from_hf_config(zero_heads_cfg)

    num_sw_layers = 4
    sw_pat_one_cfg: dict[str, object] = {
        "model_type": "qwen3_5_moe",
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
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "first_k_dense_replace": top_first_k,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
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
    bad_axes = cast(tuple[int, int, int], (2, 2))
    with pytest.raises(ValueError, match="chip_axes"):
        SliceTopology("bad_axes", bad_axes)
    bad_wrapped = cast(tuple[bool, bool, bool], (True, False))
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
    assert "default_gpu_memory_utilization" not in cat
    assert "rpa_page_size_tokens" not in cat
    for fp8_alias in ("float8_e8m0fnu", "e8m0", "float8_e5m2"):
        assert BASE_DTYPE_BYTES[fp8_alias] == FP8_BYTES


@pytest.mark.parametrize(
    "bad_cfg",
    [
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
            "layer_types": ["retnet_retention", "full_attention"],
            "vocab_size": 32000,
        },
        {"vision_config": {"hidden_size": 1024}, "text_config": None},
        {"vision_config": {"hidden_size": 1024}},
    ],
)
def test_supports_validator_fails_closed_on_unknown_arch(
    bad_cfg: dict[str, object],
) -> None:
    """Verify supports and from_hf_config reject unsupported or invalid configs."""
    with pytest.raises(UnsupportedArchitectureError):
        supports(bad_cfg)
    with pytest.raises(UnsupportedArchitectureError):
        from_hf_config(bad_cfg)


def test_ffn_only_hybrid_override_excluded_from_full_attn_count() -> None:
    """Verify ffn_only layers from hybrid_override do not count as attention."""
    hybrid_override_cfg = {
        "model_type": "qwen3_5_moe",
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
    geometry = _from_hf_config_dry_run(hybrid_override_cfg)
    exp_ffn_count = 2
    assert geometry.layer_types.count("ffn_only") == exp_ffn_count
    assert geometry.ffn_layer_counts() == (1, 1)
    assert geometry.ffn_kinds == ("none", "dense", "none", "none", "none", "moe")
    assert count_layer_types(geometry).gqa == 1
