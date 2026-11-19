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
import copy
import dataclasses
import json
import re
from pathlib import Path
from unittest import mock

import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import (
    chips,
    constants,
    models,
    specs,
)
from vllm_torchtpu.megakernel.core import errors

SOURCE_PREFIXES = ("FACT:", "DERIVED:", "ASSUMED:", "CHOSEN:")


SHIPPED_PRESET_COUNT = 3
NEGATIVE_CONTROL_SNAPSHOTS = frozenset({"kimi-k3.1-pro", "qwen3.8-2.4t", "deepseek-v3"})
HF_CONFIG_DIR = Path(__file__).resolve().parent / "data" / "advisor" / "hf_configs"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_config_snapshots() -> dict[str, dict[str, object]]:
    """Load all saved HuggingFace config snapshots keyed by file stem."""
    return {
        snapshot_path.stem: json.loads(snapshot_path.read_text(encoding="utf-8"))
        for snapshot_path in sorted(HF_CONFIG_DIR.glob("*.json"))
    }


def _load_preset_snapshots() -> dict[str, dict[str, object]]:
    """Load saved HuggingFace config snapshots for shipped presets only."""
    return {
        slug: payload
        for slug, payload in _load_config_snapshots().items()
        if slug in models.MODEL_PRESETS
    }


# Shared HF config fixtures for custom architectures removed from models.json
# (tested strictly via model_profile_from_hf_config / --model <config.json>).
# 61 layers: 3 uncompressed dense-prefix layers + 58 sparse-CSA layers at ~1/12
# KV retention gives average retention (3 + 58 * 0.08176) / 61 = 0.126921.
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
    "attn_output_gate": True,
    "quantization_config": {"quant_method": "fp8"},
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


def test_auto_chip_detect_tpu7x_two_cores_per_chip_no_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify --chip auto detection, CPU error, and Megacore rejection."""
    with (
        mock.patch.object(chips, "detect_tpu_from_sysfs", return_value=None),
        mock.patch.object(chips, "probe_jax_tpu_info", return_value=("7x", 2)),
    ):
        chip_spec = chips.resolve_chip_spec("auto")
    assert chip_spec.name == "tpu7x"
    assert chip_spec.cores_per_device == 1
    expected_devices = 2
    assert chip_spec.devices_per_chip == expected_devices

    with (
        mock.patch.object(chips, "detect_tpu_from_sysfs", return_value=None),
        pytest.raises(errors.MegakernelConfigError, match="requires TPU"),
    ):
        chips.resolve_chip_spec("auto")

    fake_7x = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: fake_7x)
    with mock.patch.object(chips, "detect_tpu_from_sysfs", return_value=None):
        auto_cs = chips.resolve_chip_spec("auto")
    assert auto_cs.name == "tpu7x"
    assert auto_cs.cores_per_device == 1

    with pytest.raises(errors.MegakernelConfigError, match="Megacore"):
        chips.resolve_chip_spec("tpu7x", megacore=True)


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


def test_tpu7x_collective_latency_bracket_is_clean_round_prior() -> None:
    """Verify tpu7x collective latency bracket uses clean round prior bounds."""
    chip_spec = chips.resolve_chip_spec("tpu7x")
    expected_bracket = (1.0, 5.0)
    assert chip_spec.collective_latency_us_bracket == expected_bracket


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


def test_hf_config_autodiscovery_qwen3_next_hybrid() -> None:
    """Verify Qwen3.5 hybrid linear-attention config autodiscovery."""
    profile = models.model_profile_from_hf_config(QWEN35_397B_CFG)
    assert profile.num_layers == QWEN35_397B_CFG["num_hidden_layers"]
    exp_attn, exp_lin, exp_experts, exp_top_k = 15, 45, 512, 10
    assert profile.num_attention_layers == exp_attn
    assert profile.num_linear_layers == exp_lin
    assert profile.num_experts == exp_experts
    assert profile.top_k == exp_top_k


def test_hf_config_autodiscovery_kimi_k25_native_int4() -> None:
    """Verify Kimi-K3 nested linear_attn_config and mxfp4 expert quantization."""
    k3_cfg = _load_preset_snapshots()["kimi-k3"]["config"]
    assert isinstance(k3_cfg, dict)
    profile = models.model_profile_from_hf_config(k3_cfg)
    exp_layers, exp_lin, exp_routed_h = 93, 69, 3584
    assert profile.name == "kimi-k3"
    assert profile.num_layers == exp_layers
    assert profile.num_linear_layers == exp_lin
    assert profile.routed_expert_hidden_size == exp_routed_h
    assert profile.weight_dtype == "bf16"
    assert profile.expert_dtype == "mxfp4"


def test_hf_config_autodiscovery_glm5_mla_dsa() -> None:
    """Verify GLM-5.2 MLA/DSA architecture config autodiscovery."""
    glm_cfg = _load_preset_snapshots()["glm-5.2"]["config"]
    assert isinstance(glm_cfg, dict)
    profile = models.model_profile_from_hf_config(glm_cfg)
    exp_layers, exp_moe, exp_experts, exp_topk, exp_idx = 78, 75, 256, 2048, 21
    assert profile.name == "glm-5.2"
    assert profile.num_layers == exp_layers
    assert profile.num_moe_layers == exp_moe
    assert profile.num_experts == exp_experts
    assert profile.index_topk == exp_topk
    assert profile.num_indexer_layers == exp_idx


def test_preset_models_json_schema() -> None:
    """Verify models.json and deployments.json load and conform to schema."""
    presets_dir = Path(models.__file__).resolve().parent / "data" / "presets"
    models_json = presets_dir / "models.json"
    deployments_json = presets_dir / "deployments.json"
    assert models_json.is_file()
    assert deployments_json.is_file()
    models_data = json.loads(models_json.read_text(encoding="utf-8"))
    assert "presets" in models_data
    assert len(models_data["presets"]) == SHIPPED_PRESET_COUNT
    for moe_slug in ("qwen3.5-397b", "kimi-k3", "glm-5.2"):
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
    cfg_file.write_text(json.dumps(QWEN35_397B_CFG), encoding="utf-8")
    ref_main = repo_dir / "refs" / "main"
    ref_main.parent.mkdir(parents=True, exist_ok=True)
    ref_main.write_text("abc123rev\n", encoding="utf-8")

    resolved = models.resolve_hf_snapshot_config("testorg/testmodel", hf_home=hf_home)
    assert resolved == cfg_file
    profile = models.resolve_model_profile(resolved)
    assert profile.name == "qwen3.5-397b"
    assert profile.num_layers == QWEN35_397B_CFG["num_hidden_layers"]


def test_tpu7x_int8_peak_equals_bf16_rate() -> None:
    """Verify tpu7x int8 peak FLOP/s equals the BF16 rate (1155 TFLOP/s)."""
    c7x = chips.resolve_chip_spec("tpu7x")
    assert c7x.int8_flops_per_s == c7x.bf16_flops_per_s
    assert c7x.peak_flops("int8") == c7x.bf16_flops_per_s


def test_chip_sources_cover_every_field() -> None:
    """Verify every built-in ChipSpec value field has a prefixed source entry."""
    skip = {"name", "topologies", "sources"}
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


def test_dense_prefix_predicate_matches_non_homogeneous_reasons() -> None:
    """Verify dense_prefix_moe is derived when 0 < moe_layers < num_layers."""
    glm_cfg = _load_preset_snapshots()["glm-5.2"]["config"]
    assert isinstance(glm_cfg, dict)
    model_profile = models.model_profile_from_hf_config(glm_cfg)
    expected_layers, expected_moe_layers = 78, 75
    assert (model_profile.num_layers, model_profile.num_moe_layers) == (
        expected_layers,
        expected_moe_layers,
    )
    assert "dense_prefix_moe" in model_profile.layer_families


def test_load_measurements_bundled_uses_requested_chip() -> None:
    """Verify load_measurements('bundled', chip_name=...) uses requested chip."""
    m7x = specs.load_measurements("bundled", chip_name="tpu7x")
    assert m7x.source_kind == "architectural_prior"
    with pytest.raises(errors.MegakernelConfigError, match="v6e"):
        specs.load_measurements("bundled", chip_name="v6e")


def test_hf_config_compress_ratios_divisors_and_geometry_validation() -> None:
    """Verify invalid attention head and head_dim geometry fail closed."""
    with pytest.raises(models.UnsupportedArchitectureError, match="num_kv_heads"):
        models.model_profile_from_hf_config(
            {**QWEN35_397B_CFG, "num_key_value_heads": 64}
        )

    with pytest.raises(models.UnsupportedArchitectureError, match="head_dim"):
        models.model_profile_from_hf_config({**QWEN35_397B_CFG, "head_dim": 6})


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
    """Verify advisor Python modules define zero private `_`-prefixed constants."""
    advisor_dir = Path(models.__file__).resolve().parent
    offenders = [
        f"{py_path.name}:{name}"
        for py_path in sorted(advisor_dir.glob("*.py"))
        for node in ast.parse(
            py_path.read_text(encoding="utf-8"), filename=str(py_path)
        ).body
        for name in _extract_assign_names(node)
        if name.startswith("_") and not name.startswith("__")
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
    """Verify every shipped preset contains only priced layer families."""
    assert len(models.MODEL_PRESETS) == SHIPPED_PRESET_COUNT
    allowed_families = specs.SUPPORTED_ONCHIP_FAMILIES | specs.STRUCTURAL_LAYER_TAGS
    for name in models.MODEL_PRESETS:
        model_profile = models.resolve_model_profile(name)
        unmodeled = [
            fam for fam in model_profile.layer_families if fam not in allowed_families
        ]
        assert unmodeled == [], f"{name} has unpriced families: {unmodeled}"


def test_dtype_source_provenance_and_default_fallback() -> None:
    """Verify dtype_source provenance across presets, quant_config, and defaults."""
    p_q35 = models.resolve_model_profile("qwen3.5-397b")
    assert p_q35.dtype_source == "preset (qwen3.5-397b)"

    p_flag = models.resolve_model_profile("qwen3.5-397b", weight_dtype="bf16")
    assert p_flag.dtype_source == "flag"
    assert p_flag.weight_dtype == "bf16"

    p_quant = models.model_profile_from_hf_config(QWEN35_397B_CFG)
    assert p_quant.dtype_source == "quantization_config"

    cfg_torch_dt = {
        k: v for k, v in QWEN35_397B_CFG.items() if k != "quantization_config"
    }
    cfg_torch_dt["torch_dtype"] = "bfloat16"
    p_torch_dt = models.parse_hf_config_profile(cfg_torch_dt, name="q35")
    assert p_torch_dt.dtype_source == "default (torch_dtype; no quantization_config)"
    with pytest.raises(models.UnsupportedArchitectureError):
        models.model_profile_from_hf_config(cfg_torch_dt)

    cfg_bare = {k: v for k, v in QWEN35_397B_CFG.items() if k != "quantization_config"}
    p_bare = models.parse_hf_config_profile(cfg_bare, name="q35")
    assert p_bare.dtype_source == "default (bf16 fallback; no quantization_config)"
    with pytest.raises(models.UnsupportedArchitectureError):
        models.model_profile_from_hf_config(cfg_bare)


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
        **QWEN35_397B_CFG,
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


def test_kimi_k3_declares_eight_expert_owner_groups() -> None:
    """Verify Kimi-K3 decode and saturation presets declare 8 expert owner groups."""
    kimi_preset = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    assert kimi_preset.decode is not None
    assert kimi_preset.decode.megakernel is not None
    # Kimi-K3's 32-rank megakernel shards each expert across 4 TP ranks (32 // 4 == 8).
    tp_ranks_per_expert_group = 4
    expected_owner_groups = kimi_preset.decode.ep // tp_ranks_per_expert_group
    assert kimi_preset.decode.megakernel["expert_owner_groups"] == expected_owner_groups
    assert kimi_preset.saturation is not None
    assert kimi_preset.saturation.megakernel is not None
    assert (
        kimi_preset.saturation.megakernel["expert_owner_groups"]
        == expected_owner_groups
    )


@pytest.mark.parametrize("bad_groups", [0, True, "8", 6, 64])
def test_parse_parallel_layout_rejects_invalid_expert_owner_groups(
    bad_groups: object,
) -> None:
    """Reject non-int, out-of-range, or ep-non-dividing expert_owner_groups."""
    ep_size = 32
    with pytest.raises(errors.MegakernelConfigError, match="expert_owner_groups"):
        models.parse_parallel_layout(
            {
                "tp": 32,
                "ep": ep_size,
                "megakernel": {
                    "source": "unit test invalid owner groups",
                    "scopes": ["whole_step"],
                    "ranks": 32,
                    "expert_owner_groups": bad_groups,
                },
            }
        )


def test_validate_declared_layouts_rejects_non_dividing_or_unknown_owner_groups() -> (
    None
):
    """Reject owner_groups that do not divide num_experts or unknown preset slugs."""
    non_dividing_layout = models.parse_parallel_layout(
        {
            "tp": 6,
            "ep": 6,
            "megakernel": {
                "source": "unit test non-dividing owner groups",
                "scopes": ["whole_step"],
                "ranks": 6,
                "expert_owner_groups": 6,
            },
        }
    )
    prefill_layout = models.parse_parallel_layout({"tp": 32, "ep": 16, "pcp": 32})
    bad_preset = models.DeploymentPreset(
        decode=non_dividing_layout,
        prefill=prefill_layout,
    )
    with pytest.raises(errors.MegakernelConfigError, match="divide num_experts"):
        models.validate_declared_layouts("kimi-k3", bad_preset)
    with pytest.raises(KeyError):
        models.validate_declared_layouts("unknown-preset-slug", bad_preset)


def test_declared_collectives_and_expert_modes_validate_schema() -> None:
    """Validate declared collective steps, expert row rules, and 6-axis pcp keys."""
    kimi_preset = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    assert kimi_preset.decode is not None
    assert kimi_preset.decode.megakernel is not None
    kimi_decode_coll = advisor.parse_declared_collectives(
        kimi_preset.decode.megakernel["collectives"]
    )
    assert kimi_decode_coll is not None
    expected_kimi_all_reduces = 94
    assert kimi_decode_coll.all_reduces == expected_kimi_all_reduces
    assert [step.kind for step in kimi_decode_coll.steps] == [
        "routed_transport",
        "dcp_gather",
    ]

    qwen_preset = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    assert qwen_preset.decode is not None
    assert qwen_preset.decode.megakernel is not None
    raw_rules = qwen_preset.decode.megakernel["expert_mode_by_rows"]
    parsed_rules = advisor.parse_expert_mode_rules(raw_rules)
    tp_row_limit = 12
    ep_row_limit = 800
    assert [(rule.max_rows, rule.mode) for rule in parsed_rules] == [
        (tp_row_limit, "tp"),
        (ep_row_limit, "ep"),
    ]
    qwen_decode_coll = advisor.parse_declared_collectives(
        qwen_preset.decode.megakernel["collectives"]
    )
    assert qwen_decode_coll is not None
    assert [
        (step.kind, step.group_size, step.calls_per_layer)
        for step in qwen_decode_coll.steps
    ] == [
        ("split_kv", 4, 2),
    ]

    for bad_collectives in (
        {"all_reduces": -1},
        {"all_reduces": True},
        {"steps": [{"kind": "unknown_op", "group_size": 4}]},
        {"steps": [{"kind": "split_kv", "group_size": 6}]},
        {"steps": [{"kind": "split_kv", "group_size": 1}]},
        {"steps": [{"kind": "split_kv", "group_size": 4, "calls_per_layer": 0}]},
    ):
        with pytest.raises(errors.MegakernelConfigError):
            models.parse_parallel_layout(
                {
                    "tp": 8,
                    "ep": 8,
                    "megakernel": {
                        "source": "invalid collective schema check",
                        "scopes": ["whole_step"],
                        "ranks": 8,
                        "collectives": bad_collectives,
                    },
                }
            )

    for bad_rules in (
        [],
        [{"max_rows": 0, "mode": "tp"}],
        [{"max_rows": 16, "mode": "tp"}, {"max_rows": 12, "mode": "ep"}],
        [{"max_rows": 12, "mode": "tp"}, {"max_rows": 12, "mode": "ep"}],
        [{"max_rows": 12, "mode": "invalid_mode"}],
    ):
        with pytest.raises(errors.MegakernelConfigError):
            models.parse_parallel_layout(
                {
                    "tp": 8,
                    "ep": 8,
                    "megakernel": {
                        "source": "invalid expert_mode_by_rows check",
                        "scopes": ["whole_step"],
                        "ranks": 8,
                        "expert_mode_by_rows": bad_rules,
                    },
                }
            )

    pcp_one_layout = models.parse_parallel_layout(
        {
            "tp": 32,
            "ep": 16,
            "pcp": 1,
            "megakernel": {
                "source": "pcp=1 prefill layout",
                "scopes": ["whole_step"],
                "phases": ["prefill"],
                "ranks": 32,
            },
        }
    )
    pcp_two_layout = models.parse_parallel_layout(
        {
            "tp": 32,
            "ep": 16,
            "pcp": 2,
            "megakernel": {
                "source": "pcp=2 prefill layout",
                "scopes": ["whole_step"],
                "phases": ["prefill"],
                "ranks": 32,
            },
        }
    )
    distinct_pcp_preset = models.DeploymentPreset(
        decode=models.parse_parallel_layout({"tp": 32, "ep": 32, "dcp": 32}),
        prefill=pcp_one_layout,
        megakernel_layouts=(pcp_two_layout,),
    )
    models.validate_declared_layouts("kimi-k3", distinct_pcp_preset)

    duplicate_pcp_preset = models.DeploymentPreset(
        decode=models.parse_parallel_layout({"tp": 32, "ep": 32, "dcp": 32}),
        prefill=pcp_two_layout,
        megakernel_layouts=(pcp_two_layout,),
    )
    with pytest.raises(errors.MegakernelConfigError, match="Duplicate megakernel"):
        models.validate_declared_layouts("kimi-k3", duplicate_pcp_preset)


def test_hf_config_autodiscovery_glm_moe_dsa_indexer_schedule() -> None:
    """Verify glm_moe_dsa HF config parses DSA dimensions, schedules, and errors."""
    num_layers = 78
    num_full_indexers = 21
    indexer_types = ["full"] * num_full_indexers + ["shared"] * (
        num_layers - num_full_indexers
    )
    glm_config: dict[str, object] = {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": num_layers,
        "first_k_dense_replace": 3,
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "n_shared_experts": 1,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "index_topk": 2048,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "indexer_types": indexer_types,
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    }
    profile = models.model_profile_from_hf_config(glm_config, name="glm-5.2-hf")
    assert profile.name == "glm-5.2"
    assert models.has_dsa_profile(profile)
    assert profile.num_indexer_layers == num_full_indexers
    assert profile.index_topk == glm_config["index_topk"]
    assert profile.index_head_dim == glm_config["index_head_dim"]
    assert profile.index_n_heads == glm_config["index_n_heads"]
    assert "dsa" in profile.layer_families
    assert "dsa" in specs.SUPPORTED_ONCHIP_FAMILIES
    assert not profile.is_periodic
    expected_record_bytes = (
        (
            profile.index_head_dim
            + (profile.index_head_dim // constants.INDEXER_SCALE_BLOCK_SIZE)
            + constants.WORD_ALIGN_BYTES
            - 1
        )
        // constants.WORD_ALIGN_BYTES
    ) * constants.WORD_ALIGN_BYTES
    assert models.indexer_record_bytes(profile.index_head_dim) == expected_record_bytes

    # Numeric index_topk_pattern entries (>0 marks full, 0 marks shared).
    numeric_config = {
        **glm_config,
        "index_topk_pattern": [1] * num_full_indexers
        + [0] * (num_layers - num_full_indexers),
    }
    numeric_config.pop("indexer_types")
    numeric_profile = models.model_profile_from_hf_config(
        numeric_config, name="glm-dsa-numeric"
    )
    assert numeric_profile.num_indexer_layers == num_full_indexers
    assert not numeric_profile.is_periodic

    # Periodic frequency schedule with skip offset resolves 21 scheduled layers.
    skip_offset = 15
    topk_freq = 3
    freq_config = {
        **glm_config,
        "index_topk_freq": topk_freq,
        "index_skip_topk_offset": skip_offset,
    }
    freq_config.pop("indexer_types")
    freq_profile = models.model_profile_from_hf_config(freq_config, name="glm-dsa-freq")
    expected_freq_layers = (num_layers - skip_offset) // topk_freq
    assert freq_profile.num_indexer_layers == expected_freq_layers
    assert not freq_profile.is_periodic

    # Malformed explicit schedules fail fast with UnsupportedArchitectureError.
    with pytest.raises(models.UnsupportedArchitectureError):
        models.model_profile_from_hf_config(
            {**glm_config, "indexer_types": ["full", "shared"]},
            name="glm-bad-len",
        )
    with pytest.raises(models.UnsupportedArchitectureError):
        models.model_profile_from_hf_config(
            {**glm_config, "indexer_types": ["typo"] * num_layers},
            name="glm-bad-token",
        )
    with pytest.raises(models.UnsupportedArchitectureError):
        models.model_profile_from_hf_config(
            {**glm_config, "indexer_types": ["shared"] * num_layers},
            name="glm-all-shared",
        )


def test_preset_glm52_matches_glm_moe_dsa_hf_config() -> None:
    """Verify shipped glm-5.2 preset matches zai-org/GLM-5.2-FP8 geometry."""
    profile = models.resolve_model_profile("glm-5.2")
    # zai-org/GLM-5.2-FP8 config.json has 78 layers: 21 full + 57 shared indexers.
    shared_indexer_layers = 57
    assert profile.model_type == "glm_moe_dsa"
    assert models.has_dsa_profile(profile)
    assert profile.intermediate_size == 2 * profile.hidden_size
    assert profile.q_lora_rank == 4 * profile.kv_lora_rank
    assert profile.qk_nope_head_dim == 3 * profile.qk_rope_head_dim
    assert profile.index_topk == 4 * profile.kv_lora_rank
    assert profile.index_head_dim == constants.INDEXER_SCALE_BLOCK_SIZE
    assert profile.index_n_heads == profile.num_attention_heads // 2
    assert profile.num_indexer_layers == profile.num_layers - shared_indexer_layers
    assert profile.weight_block_k == constants.INDEXER_SCALE_BLOCK_SIZE
    assert "dsa" in profile.layer_families
    assert "dense_prefix_moe" in profile.layer_families


def test_all_shipped_presets_cite_retrievable_hf_config() -> None:
    """Verify every shipped preset cites a concrete HuggingFace config.json."""
    assert len(models.MODEL_PRESETS) == SHIPPED_PRESET_COUNT
    for removed in ("deepseek-v4", "qwen4-exp"):
        assert removed not in models.MODEL_PRESETS
        assert removed not in models.REGISTERED_MODEL_ALIASES
    for slug, source in models.MODEL_PRESET_SOURCES.items():
        assert source.startswith("HuggingFace "), f"{slug}: {source}"
        assert source.endswith("config.json"), f"{slug}: {source}"
        assert "/" in source, f"{slug}: missing org/repo in {source}"


def test_presets_match_cited_config() -> None:
    """Verify every shipped preset matches its full authentic config snapshot."""
    all_snapshots = _load_config_snapshots()
    snapshots = _load_preset_snapshots()
    assert set(snapshots) == set(models.MODEL_PRESETS)
    assert set(all_snapshots) - set(snapshots) == NEGATIVE_CONTROL_SNAPSHOTS
    ignored_fields = {"name", "dtype_source", "kv_dtype", "state_dtype"}
    mismatches: list[str] = []
    for preset_name, snapshot_payload in snapshots.items():
        config_mapping = snapshot_payload["config"]
        assert isinstance(config_mapping, dict)
        text_subconfig = config_mapping.get("text_config")
        text_mapping = text_subconfig if isinstance(text_subconfig, dict) else {}
        assert models.MODEL_PRESET_DTYPE_BASIS[preset_name] == "config"
        has_intermediate_key = (
            "intermediate_size" in config_mapping or "intermediate_size" in text_mapping
        )
        parsed_profile = advisor.model_profile_from_hf_config(config_mapping)
        preset_profile = advisor.MODEL_PRESETS[preset_name]
        for profile_field in dataclasses.fields(models.ModelProfile):
            field_name = profile_field.name
            if field_name in ignored_fields:
                continue
            if field_name == "intermediate_size" and not has_intermediate_key:
                continue
            preset_value = getattr(preset_profile, field_name)
            config_value = getattr(parsed_profile, field_name)
            if field_name == "model_type":
                preset_value = str(preset_value).removesuffix("_text")
                config_value = str(config_value).removesuffix("_text")
            if preset_value != config_value:
                detail = f"preset={preset_value} config={config_value}"
                mismatches.append(f"{preset_name}.{field_name}: {detail}")
    assert mismatches == []


def test_every_preset_has_snapshot_or_is_listed_unverified() -> None:
    """Verify every preset and negative-control snapshot cites a repo revision."""
    revision_pattern = re.compile(r"[0-9a-f]{40}")
    for preset_name in advisor.MODEL_PRESETS:
        assert models.MODEL_PRESET_DTYPE_BASIS[preset_name] == "config"
        snapshot_path = HF_CONFIG_DIR / f"{preset_name}.json"
        assert snapshot_path.is_file()
        snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        repo_id = str(snapshot_payload["repo_id"])
        revision = str(snapshot_payload["revision"])
        assert repo_id in models.MODEL_PRESET_SOURCES[preset_name]
        assert revision_pattern.fullmatch(revision) is not None
    for neg_name in NEGATIVE_CONTROL_SNAPSHOTS:
        neg_path = HF_CONFIG_DIR / f"{neg_name}.json"
        assert neg_path.is_file()
        neg_payload = json.loads(neg_path.read_text(encoding="utf-8"))
        assert "/" in str(neg_payload["repo_id"])
        assert revision_pattern.fullmatch(str(neg_payload["revision"])) is not None


def test_config_path_honors_quantization_ignore_list() -> None:
    """Verify HF config parsing honors per-module quantization skip lists."""
    snapshots = _load_preset_snapshots()
    kimi_config = snapshots["kimi-k3"]["config"]
    assert isinstance(kimi_config, dict)
    kimi_profile = advisor.model_profile_from_hf_config(kimi_config)
    assert kimi_profile.weight_dtype == "bf16"
    assert kimi_profile.expert_dtype == "mxfp4"
    assert kimi_profile.act_dtype == "bf16"

    for preset_name in ("qwen3.5-397b", "glm-5.2"):
        model_config = snapshots[preset_name]["config"]
        assert isinstance(model_config, dict)
        model_profile = advisor.model_profile_from_hf_config(model_config)
        assert model_profile.weight_dtype == "fp8"
        assert model_profile.expert_dtype == "fp8"

    mla_config = copy.deepcopy(snapshots["glm-5.2"]["config"])
    assert isinstance(mla_config, dict)
    mla_config["quantization_config"] = {
        "quant_method": "fp8",
        "modules_to_not_convert": [
            "model.layers.*.self_attn.q_a_proj",
            "model.layers.*.self_attn.kv_a_proj_with_mqa",
        ],
    }
    mla_profile = models.parse_hf_config_profile(mla_config, name="glm-mla-bf16")
    assert mla_profile.weight_dtype == "bf16"
    assert mla_profile.expert_dtype == "fp8"
    with pytest.raises(models.UnsupportedArchitectureError):
        advisor.model_profile_from_hf_config(mla_config)

    qwen_base = snapshots["qwen3.5-397b"]["config"]
    assert isinstance(qwen_base, dict)
    qwen_text = qwen_base["text_config"]
    assert isinstance(qwen_text, dict)
    qwen_layer_types = qwen_text["layer_types"]
    assert isinstance(qwen_layer_types, list)
    qwen_full_modules = [
        f"model.language_model.layers.{layer_idx}.self_attn.q_proj"
        for layer_idx, layer_type in enumerate(qwen_layer_types)
        if layer_type == "full_attention"
    ]
    qwen_partial = copy.deepcopy(qwen_base)
    qwen_partial["quantization_config"]["modules_to_not_convert"] = [
        "model.language_model.layers.3.self_attn.q_proj"
    ]
    qwen_partial_profile = advisor.model_profile_from_hf_config(qwen_partial)
    assert qwen_partial_profile.weight_dtype == "fp8"
    assert qwen_partial_profile.expert_dtype == "fp8"

    qwen_all_skip = copy.deepcopy(qwen_base)
    qwen_all_skip["quantization_config"]["modules_to_not_convert"] = qwen_full_modules
    qwen_skip_profile = models.parse_hf_config_profile(qwen_all_skip, name="q35-bf16")
    assert qwen_skip_profile.weight_dtype == "bf16"
    assert qwen_skip_profile.expert_dtype == "fp8"
    with pytest.raises(models.UnsupportedArchitectureError):
        advisor.model_profile_from_hf_config(qwen_all_skip)

    kv_only_config = copy.deepcopy(snapshots["glm-5.2"]["config"])
    assert isinstance(kv_only_config, dict)
    kv_only_config["quantization_config"] = {
        "quant_method": "fp8",
        "modules_to_not_convert": ["kv_a_proj_with_mqa"],
    }
    kv_only_profile = advisor.model_profile_from_hf_config(kv_only_config)
    assert kv_only_profile.weight_dtype == "fp8"
    assert kv_only_profile.expert_dtype == "fp8"


def test_config_path_resolves_to_matching_preset(tmp_path: Path) -> None:
    """Verify config file and directory paths resolve to matching shipped presets."""
    snapshots = _load_config_snapshots()
    config_files: dict[str, Path] = {}
    for index, (preset_name, snapshot_payload) in enumerate(snapshots.items()):
        checkpoint_dir = tmp_path / f"checkpoint-{index}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config_file = checkpoint_dir / "config.json"
        config_file.write_text(json.dumps(snapshot_payload["config"]), encoding="utf-8")
        config_files[preset_name] = config_file
        assert advisor.resolve_model_profile(str(config_file)).name == preset_name
        assert (
            advisor.resolve_model_profile(str(config_file.parent)).name == preset_name
        )

    workload = advisor.WorkloadSpec(
        phase="decode", batch=1, context=4096, tp=32, ep=32, dcp=32
    )
    config_profile = advisor.resolve_model_profile(str(config_files["kimi-k3"]))
    config_report = advisor.advise(config_profile, workload, chip="tpu7x")
    preset_report = advisor.advise("kimi-k3", workload, chip="tpu7x")
    config_whole_step = next(
        scope_estimate
        for scope_estimate in config_report.scopes
        if scope_estimate.scope == "whole_step"
    )
    preset_whole_step = next(
        scope_estimate
        for scope_estimate in preset_report.scopes
        if scope_estimate.scope == "whole_step"
    )
    assert config_whole_step.support == "declared"
    assert config_whole_step.support == preset_whole_step.support
    assert config_whole_step.step_achievable_us == preset_whole_step.step_achievable_us

    dict_profile = advisor.resolve_model_profile(snapshots["kimi-k3"]["config"])
    assert dict_profile.name == "kimi-k3"
    dict_report = advisor.advise(dict_profile, workload, chip="tpu7x")
    dict_whole_step = next(
        scope_estimate
        for scope_estimate in dict_report.scopes
        if scope_estimate.scope == "whole_step"
    )
    assert dict_whole_step.support == "declared"
    assert dict_whole_step.step_achievable_us == preset_whole_step.step_achievable_us

    requantized_kimi = copy.deepcopy(snapshots["kimi-k3"]["config"])
    assert isinstance(requantized_kimi, dict)
    text_config = requantized_kimi["text_config"]
    assert isinstance(text_config, dict)
    quant_config = text_config["quantization_config"]
    assert isinstance(quant_config, dict)
    quant_config["format"] = "nvfp4-pack-quantized"
    config_groups = quant_config["config_groups"]
    assert isinstance(config_groups, dict)
    group_zero = config_groups["group_0"]
    assert isinstance(group_zero, dict)
    group_zero["format"] = "nvfp4-pack-quantized"

    neutral_dir = tmp_path / "neutral-checkpoint"
    neutral_dir.mkdir(parents=True, exist_ok=True)
    neutral_file = neutral_dir / "config.json"
    neutral_file.write_text(json.dumps(requantized_kimi), encoding="utf-8")
    requantized_profile = advisor.resolve_model_profile(str(neutral_file))
    assert requantized_profile.expert_dtype == "nvfp4"
    assert models.match_preset_by_geometry(requantized_profile) is None
    assert requantized_profile.name != "kimi-k3"
    requantized_report = advisor.advise(requantized_profile, workload, chip="tpu7x")
    requantized_whole_step = next(
        scope_estimate
        for scope_estimate in requantized_report.scopes
        if scope_estimate.scope == "whole_step"
    )
    assert requantized_whole_step.support != "declared"


def test_declared_blocks_cite_openable_source() -> None:
    """Verify each declared source_ref is a repo file, https URL, or unpublished."""
    seen_blocks = 0
    for deployment_preset in advisor.DEPLOYMENT_PRESETS.values():
        candidate_layouts = (
            deployment_preset.decode,
            deployment_preset.prefill,
            deployment_preset.saturation,
            *deployment_preset.megakernel_layouts,
        )
        for layout in candidate_layouts:
            if layout is None or layout.megakernel is None:
                continue
            seen_blocks += 1
            megakernel_block = layout.megakernel
            source_ref = megakernel_block.get("source_ref")
            assert isinstance(source_ref, str)
            assert source_ref
            assert (
                source_ref == "unpublished"
                or source_ref.startswith("https://")
                or (REPO_ROOT / source_ref).exists()
            )
    assert seen_blocks >= 1


def test_preset_loaders_reject_malformed_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify config and preset parsers fail loudly on malformed fields."""
    bad_skip_cfg = copy.deepcopy(QWEN35_397B_CFG)
    bad_skip_cfg["quantization_config"] = {
        "quant_method": "fp8",
        "ignore": ["re:[unclosed"],
    }
    with pytest.raises(
        models.UnsupportedArchitectureError,
        match="Invalid quantization skip pattern",
    ):
        advisor.model_profile_from_hf_config(bad_skip_cfg)

    bad_deployments = {
        "deployments": {
            "broken-notes": {
                "decode": {"tp": 8, "ep": 1},
                "prefill": {"tp": 8, "ep": 1},
                "unmodeled_notes": "not-a-list",
            }
        }
    }
    (tmp_path / "deployments.json").write_text(
        json.dumps(bad_deployments), encoding="utf-8"
    )
    bad_models = {"presets": {"broken-basis": {"dtype_basis": "bogus"}}}
    (tmp_path / "models.json").write_text(json.dumps(bad_models), encoding="utf-8")
    monkeypatch.setattr(models, "PRESETS_DIR", tmp_path)
    with pytest.raises(
        errors.MegakernelConfigError,
        match="unmodeled_notes must be a list of strings",
    ):
        models.load_deployment_presets()
    with pytest.raises(
        errors.MegakernelConfigError,
        match="dtype_basis must be one of",
    ):
        models.load_preset_dtype_basis()


def test_shipped_presets_are_the_three_validated_models() -> None:
    """Verify models.json ships only the three validated presets and their aliases."""
    expected_presets = {"qwen3.5-397b", "kimi-k3", "glm-5.2"}
    expected_aliases = {
        "qwen3.5-397b": "qwen3.5-397b",
        "qwen3.5-397b-a17b": "qwen3.5-397b",
        "qwen3.5-397b-a17b-fp8": "qwen3.5-397b",
        "kimi-k3": "kimi-k3",
        "glm-5.2": "glm-5.2",
        "glm-5.2-fp8": "glm-5.2",
        "glm5.2": "glm-5.2",
    }
    assert set(models.MODEL_PRESETS) == expected_presets
    assert dict(models.REGISTERED_MODEL_ALIASES) == expected_aliases
    assert not hasattr(models, "RECOGNIZED_MODEL_TYPES")
    for preset_name in ("qwen3.5-397b", "kimi-k3", "glm-5.2"):
        snapshot = json.loads(
            (HF_CONFIG_DIR / f"{preset_name}.json").read_text(encoding="utf-8")
        )
        direct = models.model_profile_from_hf_config(snapshot["config"])
        assert direct.name == preset_name
        models.supports_hf_config(snapshot["config"])


def test_authentic_kept_configs_resolve_to_shipped_presets() -> None:
    """Verify each kept authentic HF config snapshot resolves to its shipped preset."""
    for preset_name in ("qwen3.5-397b", "kimi-k3", "glm-5.2"):
        snapshot = json.loads(
            (HF_CONFIG_DIR / f"{preset_name}.json").read_text(encoding="utf-8")
        )
        resolved = advisor.resolve_model_profile(snapshot["config"])
        assert resolved.name == preset_name


def test_unsupported_authentic_config_fails_closed() -> None:
    """Verify authentic negative-control HF configs fail closed listing the 3 repos."""
    expected_repos = (
        "Qwen/Qwen3.5-397B-A17B-FP8",
        "moonshotai/Kimi-K3",
        "zai-org/GLM-5.2-FP8",
    )
    for neg_name in ("kimi-k3.1-pro", "qwen3.8-2.4t", "deepseek-v3"):
        snapshot = json.loads(
            (HF_CONFIG_DIR / f"{neg_name}.json").read_text(encoding="utf-8")
        )
        with pytest.raises(models.UnsupportedArchitectureError) as exc_info:
            advisor.resolve_model_profile(snapshot["config"])
        err_text = str(exc_info.value)
        for repo in expected_repos:
            assert repo in err_text, f"missing {repo} in {err_text!r}"
        assert "README.md" in err_text
        with pytest.raises(models.UnsupportedArchitectureError):
            models.model_profile_from_hf_config(snapshot["config"])
        with pytest.raises(models.UnsupportedArchitectureError):
            models.supports_hf_config(snapshot["config"])


def test_perturbed_supported_config_fails_closed() -> None:
    """Verify perturbing num_hidden_layers on any kept config fails closed."""
    for preset_name in ("qwen3.5-397b", "kimi-k3", "glm-5.2"):
        snapshot = json.loads(
            (HF_CONFIG_DIR / f"{preset_name}.json").read_text(encoding="utf-8")
        )
        perturbed = copy.deepcopy(snapshot["config"])
        target_cfg = perturbed.get("text_config", perturbed)
        target_cfg["num_hidden_layers"] = int(target_cfg["num_hidden_layers"]) + 2
        with pytest.raises(models.UnsupportedArchitectureError):
            advisor.resolve_model_profile(perturbed)
        with pytest.raises(models.UnsupportedArchitectureError):
            models.model_profile_from_hf_config(perturbed)


def test_builtin_chip_facts_follow_jax_tpu_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build_builtin_chip_spec reads live JAX TpuInfo and stays JSON-clean."""
    real_get = pltpu.get_tpu_info_for_chip
    delta_bw = 123_000_000_000.0
    delta_bf16 = 45_000_000_000_000.0

    def _perturbed(
        chip_version: pltpu.ChipVersion, cores_per_device: int
    ) -> pltpu.TpuInfo:
        """Return real TpuInfo with perturbed bandwidth and BF16 FLOP/s."""
        info = real_get(chip_version, cores_per_device)
        return dataclasses.replace(
            info,
            mem_bw_bytes_per_second=info.mem_bw_bytes_per_second + delta_bw,
            bf16_ops_per_second=info.bf16_ops_per_second + delta_bf16,
        )

    monkeypatch.setattr(pltpu, "get_tpu_info_for_chip", _perturbed)
    baseline_info = real_get(pltpu.ChipVersion.TPU_7X, 1)
    built = chips.build_builtin_chip_spec("tpu7x")
    assert built.hbm_bw_bytes_per_s == pytest.approx(
        float(baseline_info.mem_bw_bytes_per_second) + delta_bw
    )
    assert built.bf16_flops_per_s == pytest.approx(
        float(baseline_info.bf16_ops_per_second) + delta_bf16
    )
    assert built.vector_f32_ops_per_s == pytest.approx(
        built.bf16_flops_per_s / float(baseline_info.num_lanes)
    )
    assert "tpu_info" not in {f.name for f in dataclasses.fields(chips.ChipSpec)}
    assert json.dumps(dataclasses.asdict(built))
    with pytest.raises(chips.UnsupportedChipError, match="matmul_row_tile_by_dtype"):
        built.matmul_row_tile("unknown_dtype")


def test_row_tiles_follow_jax_sublane_tiling() -> None:
    """Verify built-in row tiles match JAX get_sublane_tiling and 4-bit rule."""
    jax_dtypes = {
        "bf16": jnp.dtype(jnp.bfloat16),
        "fp16": jnp.dtype(jnp.float16),
        "fp8": jnp.dtype(jnp.float8_e4m3fn),
        "int8": jnp.dtype(jnp.int8),
        "fp32": jnp.dtype(jnp.float32),
    }
    chip_versions = {
        "tpu7x": pltpu.ChipVersion.TPU_7X,
        "v6e": pltpu.ChipVersion.TPU_V6E,
    }
    for chip_name, version in chip_versions.items():
        spec = chips.BUILTIN_CHIP_SPECS[chip_name]
        info = pltpu.get_tpu_info_for_chip(version, 1)
        assert spec.matmul_row_tile_by_dtype is not None
        for label, jax_dt in jax_dtypes.items():
            expected_tile = int(info.get_sublane_tiling(jax_dt))
            assert spec.matmul_row_tile(label) == expected_tile
        expected_fp4_tile = (
            spec.matmul_row_tile("fp8")
            if spec.native_ocp_fp8
            else spec.matmul_row_tile("bf16")
        )
        for fp4_label in ("mxfp4", "nvfp4", "fp4"):
            assert spec.matmul_row_tile(fp4_label) == expected_fp4_tile
        for accepted_dtype in constants.DTYPE_BYTES:
            assert accepted_dtype in spec.matmul_row_tile_by_dtype


def test_chip_fields_without_jax_source_are_enumerated() -> None:
    """Verify non-JAX chip source fields match the exact hand-written enumeration."""
    expected_non_jax_fields = {
        "vmem_usable_bytes",
        "ici_egress_bw_bytes_per_s",
        "ici_link_bw_bytes_per_s",
        "collective_latency_s",
        "collective_latency_us_bracket",
        "dcn_bw_gbps",
        "dcn_bw_gbps_bracket",
        "min_slice_cores",
        "pcie_bw_bytes_per_s",
        "torus_ndim",
        "vector_registers_per_core",
        "vector_register_bytes",
        "vector_loads_per_cycle",
        "vector_stores_per_cycle",
        "vector_port_ops_per_cycle",
        "clock_hz",
    }
    skip_fields = {"name", "topologies", "sources"}
    value_fields = {f.name for f in dataclasses.fields(chips.ChipSpec)} - skip_fields
    for chip_name, spec in chips.BUILTIN_CHIP_SPECS.items():
        assert spec.sources is not None
        non_jax_fields = {
            field_name
            for field_name in value_fields
            if "jax.experimental.pallas.tpu" not in spec.sources.get(field_name, "")
        }
        assert non_jax_fields == expected_non_jax_fields, (
            f"{chip_name} non-JAX source fields mismatch: "
            f"extra={sorted(non_jax_fields - expected_non_jax_fields)}, "
            f"missing={sorted(expected_non_jax_fields - non_jax_fields)}"
        )
