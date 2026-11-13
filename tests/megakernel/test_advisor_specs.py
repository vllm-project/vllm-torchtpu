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
"""Specifications and profile tests for the TPU megakernel advisor."""

from __future__ import annotations

import ast
import dataclasses
import json
import re
from pathlib import Path
from unittest import mock

import pytest

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import (
    chips,
    constants,
    jax_bridge,
    models,
    specs,
)
from vllm_torchtpu.megakernel.core import errors

SOURCE_PREFIXES = ("FACT:", "DERIVED:", "ASSUMED:", "CHOSEN:")
EXPECTED_VECTOR_REGISTERS_PER_CORE = 64
EXPECTED_VECTOR_REGISTER_BYTES = 4096
EXPECTED_VECTOR_LOADS_PER_CYCLE = 3
EXPECTED_VECTOR_STORES_PER_CYCLE = 2
EXPECTED_VECTOR_PORT_OPS_PER_CYCLE = 4
EXPECTED_CLOCK_HZ = 2.2e9
PUBLIC_SCALING_BOOK_TPU_CHAPTER_URL = "jax-ml.github.io/scaling-book/tpus"
CHIP_WITHOUT_VECTOR_REGISTER_FIGURES = "v6e"
UNSET_SOURCE_PREFIX = "CHOSEN: 0 (unset)"
EXPECTED_VECTOR_LOAD_CAP_TEXT = (
    "Vector register load cap: 3 x 4,096 B x 2.2e9 Hz = 27.03 TB/s per core. "
    "It is above the HBM rate, so it does not bind while weights stream from HBM."
)

DEEPSEEK_V3_CFG: dict[str, object] = {
    "model_type": "deepseek_v3",
    "num_hidden_layers": 61,
    "first_k_dense_replace": 3,
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "quantization_config": {"quant_method": "fp8"},
}

QWEN35_397B_CFG: dict[str, object] = {
    "model_type": "qwen3_5_moe",
    "num_hidden_layers": 60,
    "hidden_size": 4096,
    "intermediate_size": 1024,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "full_attention_interval": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 64,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "quantization_config": {"quant_method": "fp8"},
}

KIMI_K3_PRO_CFG: dict[str, object] = {
    "model_type": "kimi_linear",
    "num_hidden_layers": 93,
    "first_k_dense_replace": 1,
    "hidden_size": 7168,
    "intermediate_size": 33792,
    "moe_intermediate_size": 3072,
    "shared_expert_intermediate_size": 6144,
    "routed_expert_hidden_size": 3584,
    "num_attention_heads": 96,
    "num_key_value_heads": 96,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "num_experts": 896,
    "num_experts_per_token": 16,
    "linear_attn_config": {
        "kda_layers": list(range(69)),
        "num_heads": 96,
        "head_k_dim": 128,
        "head_v_dim": 128,
        "short_conv_kernel_size": 4,
    },
    "quantization_config": {"quant_method": "fp8", "expert_dtype": "mxfp4"},
}


def test_tpu7x_dual_core_defaults() -> None:
    """Verify TPU v7x resolves to dual-core defaults."""
    chip_spec = chips.resolve_chip_spec("tpu7x")
    assert chip_spec.name == "tpu7x"
    expected_devices, expected_cores = 2, 1
    assert chip_spec.devices_per_chip == expected_devices
    assert chip_spec.cores_per_device == expected_cores
    expected_usable_vmem = 62 * (1 << 20)  # 62 MiB usable VMEM
    expected_capacity_vmem = 64 * (1 << 20)  # 64 MiB total VMEM
    assert chip_spec.vmem_usable_bytes == expected_usable_vmem
    assert chip_spec.vmem_capacity_bytes == expected_capacity_vmem


def test_single_core_escape_hatch() -> None:
    """Verify single-core escape hatch preserves 1-core spec on TPU v7x."""
    chip_spec = chips.resolve_chip_spec("tpu7x", megacore=False)
    assert chip_spec.name == "tpu7x"
    assert chip_spec.cores_per_device == 1
    expected_usable_vmem = 62 * (1 << 20)  # 62 MiB usable VMEM
    expected_capacity_vmem = 64 * (1 << 20)  # 64 MiB total VMEM
    assert chip_spec.vmem_usable_bytes == expected_usable_vmem
    assert chip_spec.vmem_capacity_bytes == expected_capacity_vmem


def test_auto_chip_detect_tpu7x_two_cores_per_chip_no_error() -> None:
    """Verify --chip auto resolves tpu7x cleanly even when JAX sees 2 cores/chip."""
    with (
        mock.patch.object(chips, "detect_tpu_from_sysfs", return_value=None),
        mock.patch.object(jax_bridge, "probe_jax_tpu_info", return_value=("7x", 2)),
    ):
        chip_spec = chips.resolve_chip_spec("auto")
    assert chip_spec.name == "tpu7x"
    assert chip_spec.cores_per_device == 1
    expected_devices = 2
    assert chip_spec.devices_per_chip == expected_devices


def test_all_preset_chips_load() -> None:
    """Verify all built-in chip specs load and have valid physical parameters."""
    for name in ("tpu7x", "v6e"):
        chip_spec = chips.resolve_chip_spec(name)
        assert chip_spec.vmem_capacity_bytes > 0
        assert chip_spec.hbm_capacity_bytes > 0
        assert chip_spec.hbm_bw_bytes_per_s > 0
        assert chip_spec.bf16_flops_per_s > 0

    for name in ("v5p", "v5e", "v4"):
        with pytest.raises(chips.UnsupportedChipError):
            chips.resolve_chip_spec(name)


def test_chip_spec_json_override(tmp_path: Path) -> None:
    """Verify loading a custom chip spec JSON overrides hardware parameters."""
    custom_json = tmp_path / "custom_chip.json"
    payload = {
        "name": "custom_v7",
        "cores_per_device": 1,
        "vmem_capacity_bytes": 64 * (1 << 20),  # 64 MiB
        "vmem_usable_bytes": 58 * (1 << 20),  # 58 MiB
        "hbm_capacity_bytes": 128 * (1 << 30),  # 128 GiB
        "hbm_bw_bytes_per_s": 2400e9,
        "bf16_flops_per_s": 1150e12,
        "fp8_flops_per_s": 2300e12,
        "ici_egress_bw_bytes_per_s": 450e9,
        "ici_link_bw_bytes_per_s": 150e9,
        "collective_latency_us_bracket": [1.2, 1.2],
        "sources": {"spec": "custom user chip spec JSON"},
    }
    custom_json.write_text(json.dumps(payload), encoding="utf-8")
    loaded = chips.resolve_chip_spec("custom", chip_spec_path=custom_json)
    assert loaded.name == "custom_v7"
    assert loaded.vmem_usable_bytes == payload["vmem_usable_bytes"]
    assert loaded.cores_per_device == 1


def test_resolve_chip_spec_unknown_raises() -> None:
    """Verify resolving an unknown chip name raises MegakernelConfigError."""
    with pytest.raises(errors.MegakernelConfigError, match="Unknown chip="):
        chips.resolve_chip_spec("nonexistent_chip_xyz")


def test_all_preset_models_resolve() -> None:
    """Verify all registered model presets resolve to valid ModelProfile."""
    for slug in models.MODEL_PRESETS:
        profile = models.resolve_model_profile(slug)
        assert profile.name == slug
        assert profile.num_layers > 0
        assert profile.hidden_size > 0
        assert profile.num_attention_heads > 0


def test_hf_config_autodiscovery_deepseek_v3() -> None:
    """Verify DeepSeek-V3 HF config auto-discovery sets MLA parameters."""
    profile = models.model_profile_from_hf_config(DEEPSEEK_V3_CFG)
    assert profile.num_layers == DEEPSEEK_V3_CFG["num_hidden_layers"]
    exp_moe, exp_experts, exp_top_k, exp_q_lora, exp_kv_lora = 58, 256, 8, 1536, 512
    assert profile.num_moe_layers == exp_moe
    assert profile.num_experts == exp_experts
    assert profile.top_k == exp_top_k
    assert profile.q_lora_rank == exp_q_lora
    assert profile.kv_lora_rank == exp_kv_lora


def test_hf_config_autodiscovery_qwen3_next_hybrid() -> None:
    """Verify Qwen3.5 hybrid linear-attention config autodiscovery."""
    profile = models.model_profile_from_hf_config(QWEN35_397B_CFG)
    assert profile.num_layers == QWEN35_397B_CFG["num_hidden_layers"]
    exp_attn, exp_lin, exp_experts, exp_top_k = 15, 45, 512, 10
    assert profile.num_attention_layers == exp_attn
    assert profile.num_linear_layers == exp_lin
    assert profile.num_experts == exp_experts
    assert profile.top_k == exp_top_k


def test_hf_config_autodiscovery_dense_llama() -> None:
    """Verify dense LLaMA style config autodiscovery."""
    cfg = {
        "model_type": "llama",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    profile = models.model_profile_from_hf_config(cfg)
    assert profile.num_layers == cfg["num_hidden_layers"]
    assert profile.num_attention_layers == cfg["num_hidden_layers"]
    assert profile.num_linear_layers == 0
    assert profile.num_experts == 0


def test_hf_config_autodiscovery_kimi_k25_native_int4() -> None:
    """Verify Kimi-K3 nested linear_attn_config and mxfp4 expert quantization."""
    profile = models.model_profile_from_hf_config(KIMI_K3_PRO_CFG)
    assert profile.num_layers == KIMI_K3_PRO_CFG["num_hidden_layers"]
    exp_lin, exp_routed_h = 69, 3584
    assert profile.num_linear_layers == exp_lin
    assert profile.routed_expert_hidden_size == exp_routed_h
    assert profile.expert_dtype == "mxfp4"


def test_hf_config_autodiscovery_glm5_mla_dsa() -> None:
    """Verify GLM-5 MLA/DSA architecture config autodiscovery."""
    cfg = {
        "model_type": "glm_moe",
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "num_attention_heads": 96,
        "num_key_value_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "num_experts": 256,
        "num_experts_per_tok": 8,
    }
    profile = models.model_profile_from_hf_config(cfg)
    assert profile.num_layers == cfg["num_hidden_layers"]
    exp_moe, exp_experts = 75, 256
    assert profile.num_moe_layers == exp_moe
    assert profile.num_experts == exp_experts


def test_preset_models_json_schema() -> None:
    """Verify models.json and deployments.json load and conform to schema."""
    presets_dir = Path(models.__file__).resolve().parent / "data" / "presets"
    models_json = presets_dir / "models.json"
    deployments_json = presets_dir / "deployments.json"
    assert models_json.is_file()
    assert deployments_json.is_file()
    models_data = json.loads(models_json.read_text(encoding="utf-8"))
    assert "presets" in models_data
    min_presets = 10
    assert len(models_data["presets"]) >= min_presets
    for moe_slug in ("mixtral-8x22b", "llama-4-maverick", "gpt-oss-120b"):
        assert models_data["presets"][moe_slug]["moe_intermediate_size"] > 0
    deployments_data = json.loads(deployments_json.read_text(encoding="utf-8"))
    assert "deployments" in deployments_data


def test_detect_sysfs_and_hf_snapshot_accept_injected_paths(tmp_path: Path) -> None:
    """Verify public sysfs and HF snapshot path-injection helpers exist and work."""
    pci_dev = tmp_path / "pci" / "0000:00:04.0"
    pci_dev.mkdir(parents=True)
    (pci_dev / "vendor").write_text("0x1ae0\n", encoding="utf-8")
    (pci_dev / "device").write_text("0x0076\n", encoding="utf-8")
    detected = chips.detect_tpu_from_sysfs(tmp_path / "pci")
    assert detected == "tpu7x"
    chip_spec = chips.resolve_chip_spec(detected)
    expected_devices = 2
    assert (chip_spec.name, chip_spec.devices_per_chip) == ("tpu7x", expected_devices)

    hf_home = tmp_path / "hf_home"
    repo_dir = hf_home / "hub" / "models--testorg--testmodel"
    snap_dir = repo_dir / "snapshots" / "abc123rev"
    snap_dir.mkdir(parents=True)
    cfg_file = snap_dir / "config.json"
    cfg_payload = {
        "model_type": "llama",
        "num_hidden_layers": 16,
        "hidden_size": 2048,
        "num_attention_heads": 16,
    }
    cfg_file.write_text(json.dumps(cfg_payload), encoding="utf-8")
    ref_main = repo_dir / "refs" / "main"
    ref_main.parent.mkdir(parents=True, exist_ok=True)
    ref_main.write_text("abc123rev\n", encoding="utf-8")

    resolved = models.resolve_hf_snapshot_config("testorg/testmodel", hf_home=hf_home)
    assert resolved == cfg_file
    profile = models.resolve_model_profile(resolved)
    assert profile.num_layers == cfg_payload["num_hidden_layers"]


def test_tpu7x_int8_peak_equals_bf16_rate() -> None:
    """Verify tpu7x int8 peak FLOP/s equals the BF16 rate (1155 TFLOP/s)."""
    c7x = chips.resolve_chip_spec("tpu7x")
    assert c7x.int8_flops_per_s == c7x.bf16_flops_per_s
    assert c7x.peak_flops("int8") == c7x.bf16_flops_per_s


def test_chip_sources_cover_every_field() -> None:
    """Verify every built-in ChipSpec value field has a prefixed source entry."""
    skip = {"name", "topologies", "sources", "tpu_info"}
    value_fields = {f.name for f in dataclasses.fields(chips.ChipSpec)} - skip
    for name, spec in chips.BUILTIN_CHIP_SPECS.items():
        assert spec.sources is not None, f"{name} missing sources"
        assert value_fields <= set(spec.sources.keys()), (
            f"{name} missing field sources: {value_fields - set(spec.sources.keys())}"
        )
        for key in value_fields:
            src_val = spec.sources[key]
            assert src_val.startswith(SOURCE_PREFIXES), (
                f"{name}.{key} source missing class prefix: {src_val!r}"
            )


def test_jax_bridge_checks_every_exposed_field() -> None:
    """Verify verify_chip_spec_against_jax checks all 6 tpu_info fields."""
    for name in ("tpu7x", "v6e"):
        cs = chips.resolve_chip_spec(name)
        assert chips.verify_chip_spec_against_jax(cs) == []

    for name in ("v5p", "v5e", "v4"):
        with pytest.raises(chips.UnsupportedChipError):
            chips.resolve_chip_spec(name)

    bad_7x = dataclasses.replace(
        chips.resolve_chip_spec("tpu7x"), int8_flops_per_s=2.31e15
    )
    mismatches = chips.verify_chip_spec_against_jax(bad_7x)
    assert any("int8_flops_per_s" in m for m in mismatches)


def test_dense_prefix_predicate_matches_non_homogeneous_reasons() -> None:
    """Verify dense_prefix_moe is derived when 0 < moe_layers < num_layers."""
    cfg = {
        **DEEPSEEK_V3_CFG,
        "num_hidden_layers": 4,
        "first_k_dense_replace": 1,
    }
    model_profile = models.model_profile_from_hf_config(cfg)
    assert (model_profile.num_layers, model_profile.num_moe_layers) == (4, 3)
    assert "dense_prefix_moe" in model_profile.layer_families


def test_load_measurements_bundled_uses_requested_chip() -> None:
    """Verify load_measurements('bundled', chip_name=...) uses requested chip."""
    m7x = specs.load_measurements("bundled", chip_name="tpu7x")
    assert m7x.source_kind == "architectural_prior"
    with pytest.raises(errors.MegakernelConfigError, match="v6e"):
        specs.load_measurements("bundled", chip_name="v6e")


def test_hf_config_autodiscovery_deepseek_v4_csa_and_mhc() -> None:
    """Verify deepseek_v4 CSA layer token, compress_ratios, and mhc_num_streams."""
    cfg: dict[str, object] = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 4,
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "moe_intermediate_size": 2048,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "layer_types": ["mla", "csa", "csa", "csa"],
        "compress_ratios": [1.0, 0.1, 0.1, 0.1],
        "mhc_num_streams": 4,
    }
    model_profile = models.model_profile_from_hf_config(cfg)
    assert model_profile.architecture_unverified is False
    assert model_profile.kv_compress_factor == pytest.approx(0.325)
    assert model_profile.residual_streams == cfg["mhc_num_streams"]
    assert "sparse_csa" in model_profile.layer_families
    assert "hyper_connection" in model_profile.layer_families


def test_hf_config_autodiscovery_qwen4_exp_hyper_connections() -> None:
    """Verify qwen4_exp HCA, indexed_attention, and hc_mult residual streams."""
    cfg: dict[str, object] = {
        "model_type": "qwen4_exp",
        "num_hidden_layers": 4,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "layer_types": ["hca", "indexed_attention", "gdn", "gdn"],
        "hc_mult": 4,
    }
    model_profile = models.model_profile_from_hf_config(cfg)
    assert model_profile.architecture_unverified is False
    assert model_profile.residual_streams == cfg["hc_mult"]
    assert "hyper_connection" in model_profile.layer_families


def test_hf_config_compress_ratios_divisors_and_geometry_validation() -> None:
    """Verify divisor compress_ratios invert and invalid ratios/heads fail closed."""
    cfg_div = {
        **DEEPSEEK_V3_CFG,
        "num_hidden_layers": 4,
        "first_k_dense_replace": 0,
        "compress_ratios": [1.0, 10.0, 10.0, 10.0],
        "quantization_config": {"quant_method": "gptq", "bits": 8},
    }
    model_profile = models.model_profile_from_hf_config(cfg_div)
    assert model_profile.kv_compress_factor == pytest.approx(0.325)
    assert model_profile.weight_dtype == "int8"

    with pytest.raises(models.UnsupportedArchitectureError, match="compress_ratios"):
        models.model_profile_from_hf_config(
            {**cfg_div, "compress_ratios": [1.0, -2.0, 0.5, 0.5]}
        )

    with pytest.raises(models.UnsupportedArchitectureError, match="num_kv_heads"):
        models.model_profile_from_hf_config(
            {**DEEPSEEK_V3_CFG, "num_key_value_heads": 256}
        )

    with pytest.raises(models.UnsupportedArchitectureError, match="head_dim"):
        models.model_profile_from_hf_config(
            {
                "model_type": "llama",
                "num_hidden_layers": 4,
                "hidden_size": 256,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": 6,
            }
        )


def test_fail_closed_spec_loaders_and_specs_bounds(tmp_path: Path) -> None:
    """Verify chip overrides, measurements, RingPlan, and EfficiencySpec fail closed."""
    bad_chip = tmp_path / "bad_chip.json"
    bad_chip.write_text(
        json.dumps(
            {
                "vmem_capacity_bytes": 64 * (1 << 20),
                "hbm_capacity_bytes": 100 * (1 << 30),
                "hbm_bw_bytes_per_s": 3.0e12,
                "bf16_flops_per_s": 1.0e15,
                "collective_latency_us_bracket": [1.0, 2.0, 3.0],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(errors.MegakernelConfigError, match="2-element"):
        chips.load_chip_spec_file(bad_chip)

    with pytest.raises(errors.MegakernelConfigError, match="numeric"):
        specs.load_measurements(
            {
                "schema": constants.MEASUREMENTS_SCHEMA,
                "launch_gap_us": {"p50": "not-a-float"},
            }
        )

    with pytest.raises(errors.MegakernelConfigError, match=r"RingPlan\.dense_slots"):
        specs.RingPlan(dense_slots=0)

    with pytest.raises(errors.MegakernelConfigError, match=r"EfficiencySpec\.eta_hbm"):
        specs.EfficiencySpec(eta_hbm=1.5)


def _extract_assign_names(node: ast.stmt) -> list[str]:
    """Return assigned top-level variable names from an AST statement."""
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def test_no_private_underscore_module_constants() -> None:
    """Verify advisor Python modules define zero _UPPER_CASE module constants."""
    advisor_dir = Path(models.__file__).resolve().parent
    pattern = re.compile(r"_[A-Z][A-Z0-9_]*")
    offenders = [
        f"{py_path.name}:{name}"
        for py_path in sorted(advisor_dir.glob("*.py"))
        for node in ast.parse(
            py_path.read_text(encoding="utf-8"), filename=str(py_path)
        ).body
        for name in _extract_assign_names(node)
        if pattern.fullmatch(name)
    ]
    assert offenders == [], f"Found private module constants: {offenders}"


def test_unlisted_model_type_fails_closed() -> None:
    """Verify unlisted model_type fails closed with UnsupportedArchitectureError."""
    with pytest.raises(
        models.UnsupportedArchitectureError, match="Unsupported architecture"
    ):
        models.model_profile_from_hf_config(
            {
                "model_type": "unlisted_custom_architecture",
                "num_hidden_layers": 4,
                "hidden_size": 256,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": 64,
            }
        )


def test_deepseek_v4_zero_compress_ratio_swa_layers() -> None:
    """Verify compress_ratios 0/0.0 treats SWA layers as uncompressed (1.0)."""
    cfg = {
        **DEEPSEEK_V3_CFG,
        "compress_ratios": [0, 2, 0.0, 2.0],
    }
    profile = models.model_profile_from_hf_config(cfg)
    expected_ratio = 0.75
    assert profile.kv_compress_factor == pytest.approx(expected_ratio)


def test_custom_chip_spec_rejects_unsupported_megacore(tmp_path: Path) -> None:
    """Verify custom chip spec with supports_megacore=False rejects megacore=True."""
    custom_spec_file = tmp_path / "custom_chip.json"
    custom_spec_file.write_text(
        json.dumps(
            {
                "name": "custom_single_core",
                "hbm_capacity_bytes": 32 * 1024 * 1024 * 1024,
                "hbm_bw_bytes_per_s": 1.6e12,
                "vmem_capacity_bytes": 32 * 1024 * 1024,
                "bf16_flops_per_s": 5.0e14,
                "collective_latency_us_bracket": [1.0, 2.0],
                "supports_megacore": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(chips.UnsupportedChipError, match="supports_megacore"):
        chips.resolve_chip_spec(
            "custom_single_core",
            chip_spec_path=custom_spec_file,
            megacore=True,
        )


def test_all_preset_models_preserve_support_status() -> None:
    """Verify all 15 preset models preserve within_core_limits vs hypothetical."""
    expected_within_core = {
        "qwen3.5-0.8b",
        "qwen3.5-397b",
        "qwen3.8-2.4t",
        "mistral-large",
        "llama-3.1-405b",
        "mixtral-8x22b",
        "llama-4-maverick",
        "gpt-oss-120b",
        "qwen3-30b-a3b",
    }
    total_presets = 15
    assert len(models.MODEL_PRESETS) == total_presets
    for name in models.MODEL_PRESETS:
        model_profile = models.resolve_model_profile(name)
        has_unmodeled = any(
            fam not in specs.SUPPORTED_ONCHIP_FAMILIES
            for fam in model_profile.layer_families
        )
        if name in expected_within_core:
            assert not has_unmodeled, f"{name} should have all supported layer families"
        else:
            assert has_unmodeled, f"{name} should decline to hypothetical"


def test_dtype_source_provenance_and_default_fallback() -> None:
    """Verify dtype_source provenance across presets, quant_config, and defaults."""
    p_q35 = models.resolve_model_profile("qwen3.5-397b")
    assert p_q35.dtype_source == "preset (qwen3.5-397b)"

    p_flag = models.resolve_model_profile("qwen3.5-397b", weight_dtype="bf16")
    assert p_flag.dtype_source == "flag"
    assert p_flag.weight_dtype == "bf16"

    cfg_quant = {
        "model_type": "qwen2",
        "num_hidden_layers": 4,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "quantization_config": {"quant_method": "fp8"},
    }
    p_quant = models.model_profile_from_hf_config(cfg_quant)
    assert p_quant.dtype_source == "quantization_config"

    cfg_torch_dt = {
        "model_type": "qwen2",
        "num_hidden_layers": 4,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "torch_dtype": "bfloat16",
    }
    p_torch_dt = models.model_profile_from_hf_config(cfg_torch_dt)
    assert p_torch_dt.dtype_source == "default (torch_dtype; no quantization_config)"

    cfg_bare = {
        "model_type": "qwen2",
        "num_hidden_layers": 4,
        "hidden_size": 1024,
        "num_attention_heads": 8,
    }
    p_bare = models.model_profile_from_hf_config(cfg_bare)
    assert p_bare.dtype_source == "default (bf16 fallback; no quantization_config)"


def test_quantization_peak_flops_and_hf_subchannel_parsing() -> None:
    """Verify W8A16/W8A8 peak FLOP rates and HuggingFace sub-channel config parsing."""
    c7x = chips.resolve_chip_spec("tpu7x")
    cv6e = chips.resolve_chip_spec("v6e")

    assert c7x.peak_flops("fp8", "bf16") == c7x.bf16_flops_per_s
    assert c7x.peak_flops("fp8", "fp8") == c7x.fp8_flops_per_s
    assert c7x.peak_flops("bf16", "bf16") == c7x.bf16_flops_per_s
    assert cv6e.peak_flops("fp8", "bf16") == cv6e.bf16_flops_per_s
    assert cv6e.peak_flops("fp8", "fp8") == cv6e.fp8_flops_per_s

    cfg_subchannel = {
        "model_type": "qwen2",
        "num_hidden_layers": 4,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "activation_scheme": "none",
        },
    }
    p_subchannel = models.model_profile_from_hf_config(cfg_subchannel)
    expected_block_k = 128
    assert p_subchannel.weight_block_k == expected_block_k
    assert p_subchannel.act_dtype == "bf16"
    assert p_subchannel.weight_dtype == "fp8"


def test_vector_register_constants_are_public_values() -> None:
    """Verify TPU 7x vector register fields, provenance, and derived load cap."""
    chip_spec = chips.resolve_chip_spec("tpu7x")
    assert chip_spec.vector_registers_per_core == EXPECTED_VECTOR_REGISTERS_PER_CORE
    assert chip_spec.vector_register_bytes == EXPECTED_VECTOR_REGISTER_BYTES
    assert chip_spec.vector_loads_per_cycle == EXPECTED_VECTOR_LOADS_PER_CYCLE
    assert chip_spec.vector_stores_per_cycle == EXPECTED_VECTOR_STORES_PER_CYCLE
    assert chip_spec.vector_port_ops_per_cycle == EXPECTED_VECTOR_PORT_OPS_PER_CYCLE
    assert chip_spec.clock_hz == EXPECTED_CLOCK_HZ

    new_field_names = (
        "vector_registers_per_core",
        "vector_register_bytes",
        "vector_loads_per_cycle",
        "vector_stores_per_cycle",
        "vector_port_ops_per_cycle",
        "clock_hz",
    )
    assert chip_spec.sources is not None
    for field_name in new_field_names:
        assert field_name in chip_spec.sources
        field_source_text = chip_spec.sources[field_name]
        assert PUBLIC_SCALING_BOOK_TPU_CHAPTER_URL in field_source_text

    calculated_cap = (
        chip_spec.vector_loads_per_cycle
        * chip_spec.vector_register_bytes
        * chip_spec.clock_hz
    )
    assert chip_spec.vector_load_cap_bytes_per_s == calculated_cap
    assert chip_spec.vector_load_cap_bytes_per_s > chip_spec.hbm_bw_bytes_per_s

    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    table_text = advisor.format_regime_table(sweep)
    html_text = advisor.render_advisor_html(sweep)
    assert EXPECTED_VECTOR_LOAD_CAP_TEXT in table_text
    assert EXPECTED_VECTOR_LOAD_CAP_TEXT in html_text

    unset_chip_spec = chips.resolve_chip_spec(CHIP_WITHOUT_VECTOR_REGISTER_FIGURES)
    assert not unset_chip_spec.vector_load_cap_bytes_per_s
    assert not advisor.report.vector_register_load_cap_line(unset_chip_spec)
    assert unset_chip_spec.sources is not None
    for field_name in new_field_names:
        assert unset_chip_spec.sources[field_name].startswith(UNSET_SOURCE_PREFIX)
