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
import itertools
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, get_args

import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import tpu_info
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import (
    chips,
    constants,
    models,
    specs,
)
from vllm_torchtpu.megakernel.advisor.models import _resolve_linear_schedule
from vllm_torchtpu.megakernel.core import errors
from vllm_torchtpu.megakernel.core import exec_spec as core_exec_spec
from vllm_torchtpu.megakernel.core.dma import vmem as core_vmem
from vllm_torchtpu.megakernel.core.profiles import default as core_default_profiles
from vllm_torchtpu.megakernel.core.runtime import constants as core_rt_constants

if TYPE_CHECKING:
    from collections.abc import Callable

SOURCE_PREFIXES = ("FACT:", "DERIVED:", "ASSUMED:", "CHOSEN:")


SHIPPED_PRESET_COUNT = 5
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


def load_qwen35_snapshot_config() -> dict[str, object]:
    """Load a fresh copy of the authentic Qwen3.5-397B HF snapshot config."""
    payload = json.loads(
        (HF_CONFIG_DIR / "qwen3.5-397b.json").read_text(encoding="utf-8")
    )
    config_mapping = payload["config"]
    assert isinstance(config_mapping, dict)
    return copy.deepcopy(config_mapping)


@pytest.mark.parametrize("megacore_kw", [{}, {"megacore": False}])
def test_tpu7x_single_core_defaults(megacore_kw: dict[str, bool]) -> None:
    """Verify TPU v7x resolves to single-core defaults regardless of megacore=False."""
    chip_spec = chips.resolve_chip_spec("tpu7x", **megacore_kw)
    assert chip_spec.name == "tpu7x"
    expected_devices, expected_cores = 2, 1
    assert chip_spec.devices_per_chip == expected_devices
    assert chip_spec.cores_per_device == expected_cores
    expected_usable_vmem = 62 * (1 << 20)
    expected_capacity_vmem = 64 * (1 << 20)
    assert chip_spec.vmem_usable_bytes == expected_usable_vmem
    assert chip_spec.vmem_capacity_bytes == expected_capacity_vmem


def test_auto_chip_detect_tpu7x_two_cores_per_chip_no_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `--chip auto` delegates to `tpu_info` without claiming PJRT."""
    monkeypatch.setattr(tpu_info, "get_tpu_type", lambda: "v7x-8")
    chip_spec = chips.resolve_chip_spec("auto")
    assert chip_spec.name == "tpu7x"
    assert chip_spec.cores_per_device == 1
    expected_devices = 2
    assert chip_spec.devices_per_chip == expected_devices

    monkeypatch.setattr(tpu_info, "get_tpu_type", lambda: "v6e-8")
    v6e_spec = chips.resolve_chip_spec("auto")
    assert v6e_spec.name == "v6e"
    assert v6e_spec.cores_per_device == 1
    assert v6e_spec.devices_per_chip == 1

    def _fail_get_chip_version(
        _accelerator_type: str | None = None,
    ) -> pltpu.ChipVersion:
        """Fail if get_chip_version runs when get_tpu_type returns empty."""
        pytest.fail("get_chip_version must not run on non-TPU hosts")

    monkeypatch.setattr(tpu_info, "get_chip_version", _fail_get_chip_version)
    for empty_type in (None, ""):
        monkeypatch.setattr(tpu_info, "get_tpu_type", lambda val=empty_type: val)
        with pytest.raises(chips.UnsupportedChipError, match="requires TPU"):
            chips.resolve_chip_spec("auto")

    with pytest.raises(errors.MegakernelConfigError, match="Megacore"):
        chips.resolve_chip_spec("tpu7x", megacore=True)


@pytest.mark.parametrize(
    "bad_tpu_type",
    [
        "v99-8",
        "unknown-4",
        "v7x_bogus-8",
        "v17x-8",
        "foo7x-8",
        "7xpro-16",
        "   ",
    ],
)
def test_auto_chip_rejects_unrecognized_tpu_type(
    monkeypatch: pytest.MonkeyPatch,
    bad_tpu_type: str,
) -> None:
    """Verify chip='auto' fails closed when tpu_info falls back on unknown types."""
    monkeypatch.setattr(tpu_info, "get_tpu_type", lambda: bad_tpu_type)
    expected_match = "requires TPU" if not bad_tpu_type.strip() else "Unrecognized TPU"
    with pytest.raises(chips.UnsupportedChipError, match=expected_match):
        chips.resolve_chip_spec("auto")


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


def test_resolve_chip_spec_unknown_raises() -> None:
    """Verify resolving an unknown chip name raises MegakernelConfigError."""
    with pytest.raises(errors.MegakernelConfigError, match="Unknown chip="):
        chips.resolve_chip_spec("nonexistent_chip_xyz")


def test_shipped_presets_geometry_families_and_sources() -> None:
    """Verify shipped presets resolve, price all families, and cite HF configs."""
    assert len(models.MODEL_PRESETS) == SHIPPED_PRESET_COUNT
    allowed_families = specs.SUPPORTED_ONCHIP_FAMILIES | specs.STRUCTURAL_LAYER_TAGS
    for slug in models.MODEL_PRESETS:
        profile = models.resolve_model_profile(slug)
        assert profile.name == slug
        assert profile.num_layers > 0
        assert profile.hidden_size > 0
        assert profile.num_attention_heads > 0
        unmodeled = [
            fam for fam in profile.layer_families if fam not in allowed_families
        ]
        assert unmodeled == [], f"{slug} has unpriced families: {unmodeled}"
        source = models.MODEL_PRESET_SOURCES[slug]
        assert source.startswith("HuggingFace "), f"{slug}: {source}"
        assert source.endswith("config.json"), f"{slug}: {source}"
        assert "/" in source, f"{slug}: missing org/repo in {source}"


@pytest.mark.parametrize("preset_slug", sorted(models.MODEL_PRESETS))
def test_authentic_hf_snapshots_autodiscovery_and_lock_in(preset_slug: str) -> None:
    """Verify authentic HF snapshots match preset geometry, dtypes, and families."""
    snapshot = _load_preset_snapshots()[preset_slug]["config"]
    assert isinstance(snapshot, dict)
    preset = models.MODEL_PRESETS[preset_slug]
    profile = models.model_profile_from_hf_config(snapshot)
    assert profile.name == preset_slug
    assert profile.num_layers == preset.num_layers
    assert (
        profile.num_attention_layers + profile.num_linear_layers == profile.num_layers
    )
    assert profile.num_linear_layers == preset.num_linear_layers
    assert profile.num_moe_layers == preset.num_moe_layers
    assert profile.num_experts == preset.num_experts
    assert profile.num_indexer_layers == preset.num_indexer_layers
    assert profile.weight_dtype == preset.weight_dtype
    assert profile.expert_dtype == preset.expert_dtype
    assert profile.intermediate_size == preset.intermediate_size
    assert profile.has_attn_output_gate is preset.has_attn_output_gate
    assert profile.weight_block_k == preset.weight_block_k
    if profile.num_linear_layers > 0:
        expected_state_dtype = models.default_serving_state_dtype(profile.model_type)
        assert profile.state_dtype == expected_state_dtype
        assert profile.state_dtype == preset.state_dtype
    if models.has_dsa_profile(profile):
        assert "dsa" in profile.layer_families


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
    for moe_slug in (
        "qwen3.5-397b",
        "kimi-k3",
        "glm-5.2",
        "qwen3.8-flash-next",
        "deepseek-v4.1-flash",
    ):
        assert models_data["presets"][moe_slug]["moe_intermediate_size"] > 0
    deployments_data = json.loads(deployments_json.read_text(encoding="utf-8"))
    assert "deployments" in deployments_data


def test_hf_snapshot_accepts_injected_paths(tmp_path: Path) -> None:
    """Verify HF snapshot path-injection helper resolves cached config.json."""
    qwen_cfg = load_qwen35_snapshot_config()
    hf_home = tmp_path / "hf_home"
    repo_dir = hf_home / "hub" / "models--testorg--testmodel"
    snap_dir = repo_dir / "snapshots" / "abc123rev"
    snap_dir.mkdir(parents=True)
    cfg_file = snap_dir / "config.json"
    cfg_file.write_text(json.dumps(qwen_cfg), encoding="utf-8")
    ref_main = repo_dir / "refs" / "main"
    ref_main.parent.mkdir(parents=True, exist_ok=True)
    ref_main.write_text("abc123rev\n", encoding="utf-8")

    resolved = models.resolve_hf_snapshot_config("testorg/testmodel", hf_home=hf_home)
    assert resolved == cfg_file
    profile = models.resolve_model_profile(resolved)
    assert profile.name == "qwen3.5-397b"
    assert profile.num_layers == models.MODEL_PRESETS["qwen3.5-397b"].num_layers


def test_peak_flops_all_dtypes_and_subchannel_parsing() -> None:
    """Verify peak_flops across all DTYPE_BYTES entries, unknown dtypes, and quant."""
    tpu7x_spec = chips.resolve_chip_spec("tpu7x")
    v6e_spec = chips.resolve_chip_spec("v6e")

    fp8_aliases = ("fp8", "fp8_e4m3", "fp8_e4m3fn", "fp8_e5m2")
    fp4_aliases = ("nvfp4", "mxfp4", "fp4")
    bf16_aliases = ("bf16", "fp16")
    for chip_spec in chips.BUILTIN_CHIP_SPECS.values():
        for dtype_name in constants.DTYPE_BYTES:
            if dtype_name in fp8_aliases or dtype_name in fp4_aliases:
                assert (
                    chip_spec.peak_flops(dtype_name, "fp8") == chip_spec.fp8_flops_per_s
                )
                assert (
                    chip_spec.peak_flops(dtype_name, "bf16")
                    == chip_spec.bf16_flops_per_s
                )
            elif dtype_name in ("int8", "int4"):
                assert chip_spec.peak_flops(dtype_name) == chip_spec.int8_flops_per_s
                assert (
                    chip_spec.peak_flops(dtype_name, "fp8") == chip_spec.fp8_flops_per_s
                )
            elif dtype_name in bf16_aliases:
                assert chip_spec.peak_flops(dtype_name) == chip_spec.bf16_flops_per_s
            elif dtype_name == "fp32":
                assert chip_spec.peak_flops(dtype_name) == pytest.approx(
                    constants.FP32_TO_BF16_FLOPS_RATIO * chip_spec.bf16_flops_per_s
                )
    assert (
        tpu7x_spec.peak_flops(" fp8_e4m3fn ", " fp8_e5m2 ")
        == tpu7x_spec.fp8_flops_per_s
    )
    assert v6e_spec.peak_flops("int4") == v6e_spec.int8_flops_per_s
    for bad_weight, bad_act in (("fp7", None), ("fp8", "fp7")):
        with pytest.raises(errors.MegakernelConfigError):
            tpu7x_spec.peak_flops(bad_weight, bad_act)

    subchannel_cfg = load_qwen35_snapshot_config()
    subchannel_cfg["quantization_config"] = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "activation_scheme": "none",
    }
    subchannel_profile = models.model_profile_from_hf_config(subchannel_cfg)
    expected_block_k = 128
    assert subchannel_profile.weight_block_k == expected_block_k
    assert subchannel_profile.act_dtype == "bf16"
    assert subchannel_profile.weight_dtype == "fp8"

    awq_cfg = load_qwen35_snapshot_config()
    awq_cfg["quantization_config"] = {"quant_method": "awq", "bits": 4}
    with pytest.raises(models.UnsupportedArchitectureError):
        models.parse_hf_config_profile(awq_cfg, name="q35-awq")


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


def test_dense_prefix_derives_dense_prefix_moe_family() -> None:
    """Verify dense_prefix_moe is derived iff 0 < moe_layers < num_layers."""
    for preset_name, payload in _load_preset_snapshots().items():
        config_obj = payload["config"]
        assert isinstance(config_obj, dict)
        model_profile = models.model_profile_from_hf_config(config_obj)
        expected_dense_prefix = (
            0 < model_profile.num_moe_layers < model_profile.num_layers
        )
        assert (
            "dense_prefix_moe" in model_profile.layer_families
        ) is expected_dense_prefix, (
            f"{preset_name}: layer_families={model_profile.layer_families}"
        )


def test_load_measurements_bundled_uses_requested_chip() -> None:
    """Verify load_measurements('bundled', chip_name=...) uses requested chip."""
    tpu7x_measurements = specs.load_measurements("bundled", chip_name="tpu7x")
    assert tpu7x_measurements.source_kind == "architectural_prior"
    with pytest.raises(errors.MegakernelConfigError, match="v6e"):
        specs.load_measurements("bundled", chip_name="v6e")


def test_hf_config_geometry_validation_fails_closed() -> None:
    """Verify invalid geometry, int parsing, linear counts, and dtypes fail closed."""
    bad_kv_cfg = load_qwen35_snapshot_config()
    bad_kv_text = bad_kv_cfg["text_config"]
    assert isinstance(bad_kv_text, dict)
    bad_kv_text["num_key_value_heads"] = 64
    with pytest.raises(models.UnsupportedArchitectureError, match="num_kv_heads"):
        models.model_profile_from_hf_config(bad_kv_cfg)

    bad_head_cfg = load_qwen35_snapshot_config()
    bad_head_text = bad_head_cfg["text_config"]
    assert isinstance(bad_head_text, dict)
    bad_head_text["head_dim"] = 6
    with pytest.raises(models.UnsupportedArchitectureError, match="head_dim"):
        models.model_profile_from_hf_config(bad_head_cfg)

    bad_int_cfg = load_qwen35_snapshot_config()
    bad_int_text = bad_int_cfg["text_config"]
    assert isinstance(bad_int_text, dict)
    bad_int_text["num_hidden_layers"] = "sixty"
    with pytest.raises(models.UnsupportedArchitectureError, match="Invalid integer"):
        models.model_profile_from_hf_config(bad_int_cfg)

    for preset_slug in models.MODEL_PRESETS:
        with pytest.raises(
            errors.MegakernelConfigError, match="Unsupported weight_dtype"
        ):
            models.resolve_model_profile(preset_slug, weight_dtype="fp7")

    with pytest.raises(errors.MegakernelConfigError, match="num_linear_layers"):
        models.finalize_model_profile(
            dataclasses.replace(
                models.MODEL_PRESETS["qwen3.5-397b"],
                num_layers=10,
                num_linear_layers=15,
            )
        )


def test_fail_closed_spec_loaders_and_specs_bounds() -> None:
    """Verify measurements, RingPlan, and EfficiencySpec bounds fail closed."""
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


def test_dtype_source_provenance_and_default_fallback() -> None:
    """Verify dtype_source provenance across presets, quant_config, and defaults."""
    qwen_profile = models.resolve_model_profile("qwen3.5-397b")
    assert qwen_profile.dtype_source == "preset (qwen3.5-397b)"

    flag_profile = models.resolve_model_profile("qwen3.5-397b", weight_dtype="bf16")
    assert flag_profile.dtype_source == "flag"
    assert flag_profile.weight_dtype == "bf16"

    qwen_cfg = load_qwen35_snapshot_config()
    quant_profile = models.model_profile_from_hf_config(qwen_cfg)
    assert quant_profile.dtype_source == "quantization_config"

    cfg_bare = load_qwen35_snapshot_config()
    cfg_bare.pop("quantization_config", None)
    bare_text = cfg_bare["text_config"]
    assert isinstance(bare_text, dict)
    bare_text.pop("dtype", None)
    bare_text.pop("torch_dtype", None)
    bare_profile = models.parse_hf_config_profile(cfg_bare, name="q35")
    assert (
        bare_profile.dtype_source == "default (bf16 fallback; no quantization_config)"
    )
    with pytest.raises(models.UnsupportedArchitectureError):
        models.model_profile_from_hf_config(cfg_bare)

    cfg_torch_dt = copy.deepcopy(cfg_bare)
    torch_text = cfg_torch_dt["text_config"]
    assert isinstance(torch_text, dict)
    torch_text["torch_dtype"] = "bfloat16"
    torch_dtype_profile = models.parse_hf_config_profile(cfg_torch_dt, name="q35")
    assert (
        torch_dtype_profile.dtype_source
        == "default (torch_dtype; no quantization_config)"
    )
    with pytest.raises(models.UnsupportedArchitectureError):
        models.model_profile_from_hf_config(cfg_torch_dt)


def test_hf_config_autodiscovery_glm_moe_dsa_indexer_schedule() -> None:
    """Verify glm_moe_dsa HF config parses DSA dimensions, schedules, and errors."""
    glm_config = copy.deepcopy(_load_preset_snapshots()["glm-5.2"]["config"])
    assert isinstance(glm_config, dict)
    num_layers = int(glm_config["num_hidden_layers"])
    num_full_indexers = 21

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
    for head_dim, expected_record_bytes in (
        (0, 0),
        (64, 68),
        (128, 132),
        (129, 132),
        (256, 260),
    ):
        assert models.indexer_record_bytes(head_dim) == expected_record_bytes

    numeric_config = copy.deepcopy(glm_config)
    numeric_config.pop("indexer_types", None)
    numeric_config["index_topk_pattern"] = [1] * num_full_indexers + [0] * (
        num_layers - num_full_indexers
    )
    numeric_profile = models.model_profile_from_hf_config(
        numeric_config, name="glm-dsa-numeric"
    )
    assert numeric_profile.num_indexer_layers == num_full_indexers
    assert not numeric_profile.is_periodic

    freq_config = copy.deepcopy(glm_config)
    freq_config.pop("indexer_types", None)
    freq_config.pop("index_topk_pattern", None)
    topk_freq = int(freq_config["index_topk_freq"])
    freq_profile = models.model_profile_from_hf_config(freq_config, name="glm-dsa-freq")
    assert freq_profile.num_indexer_layers == num_full_indexers
    assert not freq_profile.is_periodic

    no_prefix_profile = models.parse_hf_config_profile(
        {**freq_config, "index_skip_topk_offset": 0},
        name="glm-dsa-freq-no-prefix",
    )
    assert no_prefix_profile.num_indexer_layers == num_layers // topk_freq

    for bad_schedule, bad_name in (
        (["full", "shared"], "glm-bad-len"),
        (["typo"] * num_layers, "glm-bad-token"),
        (["shared"] * num_layers, "glm-all-shared"),
    ):
        with pytest.raises(models.UnsupportedArchitectureError):
            models.model_profile_from_hf_config(
                {**glm_config, "indexer_types": bad_schedule},
                name=bad_name,
            )


RUNTIME_DTYPE_OVERRIDES: dict[str, dict[str, str]] = {
    "qwen3.5-397b": {"kv_dtype": "fp8", "state_dtype": "bf16"},
    "kimi-k3": {"kv_dtype": "fp8", "state_dtype": "fp32"},
    "glm-5.2": {"kv_dtype": "fp8"},
    "qwen3.8-flash-next": {"kv_dtype": "fp8", "state_dtype": "fp32"},
    "deepseek-v4.1-flash": {"kv_dtype": "fp8"},
}


def test_presets_match_cited_config_strict_no_dtype_mask() -> None:
    """Verify presets match full HF snapshots with only explicit audited overrides."""
    all_snapshots = _load_config_snapshots()
    snapshots = _load_preset_snapshots()
    assert set(snapshots) == set(models.MODEL_PRESETS)
    assert set(all_snapshots) - set(snapshots) == NEGATIVE_CONTROL_SNAPSHOTS
    ignored_fields = {"name", "dtype_source"}
    mismatches: list[str] = []
    for preset_name, snapshot_payload in snapshots.items():
        config_mapping = snapshot_payload["config"]
        assert isinstance(config_mapping, dict)
        assert models.MODEL_PRESET_DTYPE_BASIS[preset_name] == "config"
        parsed_profile = advisor.model_profile_from_hf_config(config_mapping)
        preset_profile = advisor.MODEL_PRESETS[preset_name]
        runtime_overrides = RUNTIME_DTYPE_OVERRIDES.get(preset_name, {})
        for override_field, override_value in runtime_overrides.items():
            actual_preset = getattr(preset_profile, override_field)
            actual_parsed = getattr(parsed_profile, override_field)
            if actual_preset != override_value or actual_parsed != override_value:
                mismatches.append(
                    f"{preset_name}.{override_field}: "
                    f"preset={actual_preset} parsed={actual_parsed} "
                    f"expected_override={override_value}"
                )
        for profile_field in dataclasses.fields(models.ModelProfile):
            field_name = profile_field.name
            if field_name in ignored_fields or field_name in runtime_overrides:
                continue
            preset_value = getattr(preset_profile, field_name)
            config_value = getattr(parsed_profile, field_name)
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

    q38_config = snapshots["qwen3.8-flash-next"]["config"]
    assert isinstance(q38_config, dict)
    q38_profile = advisor.model_profile_from_hf_config(q38_config)
    assert q38_profile.weight_dtype == "bf16"
    assert q38_profile.expert_dtype == "fp8"
    assert q38_profile.act_dtype == "bf16"
    assert q38_profile.expert_act_dtype == "fp8"

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


def test_preset_loaders_reject_malformed_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify config and preset parsers fail loudly on malformed fields."""
    bad_skip_cfg = load_qwen35_snapshot_config()
    bad_skip_cfg["quantization_config"] = {
        "quant_method": "fp8",
        "ignore": ["re:[unclosed"],
    }
    with pytest.raises(
        models.UnsupportedArchitectureError,
        match="Invalid quantization skip pattern",
    ):
        advisor.model_profile_from_hf_config(bad_skip_cfg)

    with pytest.raises(errors.MegakernelConfigError, match="integer"):
        models.parse_parallel_layout({"tp": True, "ep": 8})

    corrupt_cfg = tmp_path / "corrupt.json"
    corrupt_cfg.write_text("{not-json", encoding="utf-8")
    with pytest.raises(errors.MegakernelConfigError, match="Invalid JSON"):
        models.resolve_model_profile(corrupt_cfg)

    bad_deployments = {
        "deployments": {
            "broken-windows": {
                "decode": {"tp": 8, "ep": 1},
                "prefill": {"tp": 8, "ep": 1},
                "prefill_windows": "not-a-sequence",
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
        match="prefill_windows must be a sequence of integers",
    ):
        models.load_deployment_presets()
    with pytest.raises(
        errors.MegakernelConfigError,
        match="dtype_basis must be one of",
    ):
        models.load_model_presets()


def test_shipped_presets_are_the_three_validated_models() -> None:
    """Verify models.json ships only the validated presets and their aliases."""
    expected_presets = {
        "qwen3.5-397b",
        "kimi-k3",
        "glm-5.2",
        "qwen3.8-flash-next",
        "deepseek-v4.1-flash",
    }
    expected_aliases = {
        "qwen3.5-397b": "qwen3.5-397b",
        "qwen3.5-397b-a17b": "qwen3.5-397b",
        "qwen3.5-397b-a17b-fp8": "qwen3.5-397b",
        "kimi-k3": "kimi-k3",
        "glm-5.2": "glm-5.2",
        "glm-5.2-fp8": "glm-5.2",
        "glm5.2": "glm-5.2",
        "qwen3.8-flash-next": "qwen3.8-flash-next",
        "qwen3.8-flash-next-fp8": "qwen3.8-flash-next",
        "qwen3.8-flash": "qwen3.8-flash-next",
        "deepseek-v4.1-flash": "deepseek-v4.1-flash",
        "deepseek-v41-flash": "deepseek-v4.1-flash",
    }
    assert set(models.MODEL_PRESETS) == expected_presets
    assert dict(models.REGISTERED_MODEL_ALIASES) == expected_aliases
    for preset_name in expected_presets:
        snapshot = json.loads(
            (HF_CONFIG_DIR / f"{preset_name}.json").read_text(encoding="utf-8")
        )
        direct = models.model_profile_from_hf_config(snapshot["config"])
        assert direct.name == preset_name
        resolved = advisor.resolve_model_profile(snapshot["config"])
        assert resolved.name == preset_name
        models.supports_hf_config(snapshot["config"])


def test_unsupported_authentic_config_fails_closed() -> None:
    """Verify negative-control HF configs fail closed listing shipped repos."""
    expected_repos = (
        "Qwen/Qwen3.5-397B-A17B-FP8",
        "moonshotai/Kimi-K3",
        "zai-org/GLM-5.2-FP8",
        "Qwen/Qwen3.8-Flash-Next-FP8",
        "deepseek-ai/DeepSeek-V4.1-Flash",
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
    for bad_target in ("/nonexistent/path/kimi-k3", "bogus-org/kimi-k3"):
        with pytest.raises(models.UnsupportedArchitectureError):
            models.resolve_model_profile(bad_target)


@pytest.mark.parametrize("preset_name", sorted(models.MODEL_PRESETS))
@pytest.mark.parametrize(
    "perturb_field",
    [
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "output_gate",
        "quantization_config",
    ],
)
def test_perturbed_supported_config_fails_closed(
    preset_name: str,
    perturb_field: str,
) -> None:
    """Verify perturbing geometry, output gate, or quant fails closed."""
    snapshot = _load_preset_snapshots()[preset_name]
    config_obj = snapshot["config"]
    assert isinstance(config_obj, dict)
    perturbed = copy.deepcopy(config_obj)
    target_cfg = perturbed.get("text_config", perturbed)
    assert isinstance(target_cfg, dict)
    if perturb_field == "num_hidden_layers":
        target_cfg["num_hidden_layers"] = int(target_cfg["num_hidden_layers"]) + 2
    elif perturb_field == "hidden_size":
        target_cfg["hidden_size"] = int(target_cfg["hidden_size"]) + 128
    elif perturb_field == "intermediate_size":
        target_cfg["intermediate_size"] = (
            int(target_cfg.get("intermediate_size") or 0) + 1024
        )
    elif perturb_field == "num_attention_heads":
        target_cfg["num_attention_heads"] = int(target_cfg["num_attention_heads"]) * 2
    elif perturb_field == "output_gate":
        if "output_gate_type" in target_cfg:
            cur_gate = str(target_cfg["output_gate_type"]).strip().lower()
            target_cfg["output_gate_type"] = "none" if cur_gate != "none" else "sigmoid"
        else:
            gate_key = (
                "mla_use_output_gate"
                if "mla_use_output_gate" in target_cfg
                else "attn_output_gate"
            )
            target_cfg[gate_key] = not bool(target_cfg.get(gate_key, False))
    else:
        perturbed.pop("quantization_config", None)
        target_cfg["quantization_config"] = {"quant_method": "awq", "bits": 4}
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
        for fp4_label in ("int4", "mxfp4", "nvfp4", "fp4"):
            assert spec.matmul_row_tile(fp4_label) == expected_fp4_tile
        assert spec.sources is not None
        assert "int4" in spec.sources["matmul_row_tile_by_dtype"]
        for accepted_dtype in sorted(
            set(constants.DTYPE_BYTES) - constants.KV_ONLY_DTYPES
        ):
            assert accepted_dtype in spec.matmul_row_tile_by_dtype
        for kv_only_dtype in sorted(constants.KV_ONLY_DTYPES):
            assert kv_only_dtype not in spec.matmul_row_tile_by_dtype


def test_chip_fields_without_jax_source_are_enumerated() -> None:
    """Verify non-JAX chip source fields match the exact hand-written enumeration."""
    expected_non_jax_fields = {
        "vmem_usable_bytes",
        "vmem_fit_headroom_bytes",
        "ici_egress_bw_bytes_per_s",
        "ici_link_bw_bytes_per_s",
        "collective_latency_s",
        "collective_latency_us_bracket",
        "dcn_bw_gbps",
        "dcn_bw_gbps_bracket",
        "pcie_h2d_bw_bytes_per_s",
        "pcie_h2d_bw_bracket_bytes_per_s",
        "min_slice_cores",
        "vector_register_bytes",
        "vector_loads_per_cycle",
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


def test_scaling_book_priors_and_hbm_cap_provenance() -> None:
    """Verify Scaling Book priors, collective latency bracket, and HBM cap."""
    scaling_book_fields = (
        "vector_register_bytes",
        "vector_loads_per_cycle",
        "clock_hz",
    )
    for field_name in scaling_book_fields:
        src_text = constants.DEFAULT_SOURCES_7X[field_name]
        assert src_text.startswith("ASSUMED:"), f"{field_name}: {src_text}"
        assert "jax-ml.github.io/scaling-book/tpus" in src_text

    expected_bracket = (1.0, 5.0)
    assert (
        chips.resolve_chip_spec("tpu7x").collective_latency_us_bracket
        == expected_bracket
    )

    expected_hbm_cap = 0.92
    actual_hbm_cap = constants.HBM_UTILIZATION_CAP
    assert actual_hbm_cap == expected_hbm_cap
    assert specs.WorkloadSpec().hbm_utilization_cap == actual_hbm_cap

    assert type(constants.GAIN_BASIS_CALIBRATED_E2E) is str
    assert type(constants.GAIN_BASIS_DEVICE_ONLY) is str
    assert {
        constants.GAIN_BASIS_CALIBRATED_E2E,
        constants.GAIN_BASIS_DEVICE_ONLY,
    } == set(get_args(specs.GainBasis))


def test_qwen35_and_kimi_k3_hf_config_lock_in_fields() -> None:
    """Verify HF parser applies serving state dtype, MLA gate, and block size."""
    snapshots = _load_preset_snapshots()
    qwen_config = snapshots["qwen3.5-397b"]["config"]
    assert isinstance(qwen_config, dict)
    qwen_text = qwen_config["text_config"]
    assert isinstance(qwen_text, dict)
    assert qwen_text["mamba_ssm_dtype"] == "float32"

    qwen_profile = models.model_profile_from_hf_config(qwen_config)
    assert qwen_profile.state_dtype == "bf16"
    assert qwen_profile.intermediate_size == 0
    assert qwen_profile.model_type == "qwen3_5_moe_text"
    expected_linear_dims = (128, 128, 4)
    assert (
        qwen_profile.linear_key_dim,
        qwen_profile.linear_value_dim,
        qwen_profile.linear_conv_kernel,
    ) == expected_linear_dims

    kimi_profile = models.model_profile_from_hf_config(snapshots["kimi-k3"]["config"])
    expected_kimi_block_k = 32
    assert kimi_profile.state_dtype == "fp32"
    assert kimi_profile.has_attn_output_gate is True
    assert kimi_profile.weight_block_k == expected_kimi_block_k


def test_hub_param_totals_manifest_matches_snapshots() -> None:
    """Verify checked-in Hub safetensors manifest aligns with model snapshots."""
    manifest_path = HF_CONFIG_DIR.parent / "hub_param_totals.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "advisor.hub_param_totals.v1"
    entries = manifest["models"]
    assert isinstance(entries, dict)
    assert set(entries) == set(models.MODEL_PRESETS)

    sha_pattern = re.compile(r"[0-9a-f]{40}")
    snapshots = _load_preset_snapshots()
    for slug, record in entries.items():
        assert isinstance(record, dict)
        repo_id = str(record["repo_id"])
        assert repo_id in models.MODEL_PRESET_SOURCES[slug]
        hub_sha = str(record["hub_sha"])
        assert sha_pattern.fullmatch(hub_sha) is not None
        snapshot_revision = str(record["snapshot_revision"])
        assert sha_pattern.fullmatch(snapshot_revision) is not None
        assert snapshot_revision == str(snapshots[slug]["revision"])
        assert hub_sha == snapshot_revision
        safetensors_total = record["safetensors_total"]
        assert type(safetensors_total) is int
        assert safetensors_total > 0
        out_of_scope = record["out_of_scope_components"]
        assert isinstance(out_of_scope, list)
        assert len(out_of_scope) > 0
        assert isinstance(record["out_of_scope_note"], str)
        assert record["out_of_scope_note"]


def test_preset_loader_requires_mandatory_geometry_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify models.json preset loader rejects missing geometry keys."""
    base_entry = {
        "num_layers": 60,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_kv_heads": 2,
        "head_dim": 256,
        "dtype_basis": "config",
    }
    monkeypatch.setattr(models, "PRESETS_DIR", tmp_path)
    for missing_key in (
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "num_kv_heads",
        "head_dim",
    ):
        entry = {k: v for k, v in base_entry.items() if k != missing_key}
        payload = {"presets": {"broken": entry}, "aliases": {"broken": "broken"}}
        (tmp_path / "models.json").write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises((errors.MegakernelConfigError, KeyError)):
            models.load_model_presets()
    profile_ctor: Callable[..., models.ModelProfile] = models.ModelProfile
    with pytest.raises(TypeError):
        profile_ctor(name="broken", num_layers=1)


def test_resolve_model_profile_dict_and_distinct_expert_dtype_overrides() -> None:
    """Verify dict resolution admits before overrides and preserves expert_dtype."""
    qwen_cfg = _load_preset_snapshots()["qwen3.5-397b"]["config"]
    assert isinstance(qwen_cfg, dict)

    dict_override = models.resolve_model_profile(qwen_cfg, weight_dtype="bf16")
    assert dict_override.name == "qwen3.5-397b"
    assert dict_override.weight_dtype == "bf16"
    assert dict_override.expert_dtype == "bf16"
    assert dict_override.dtype_source == "flag"

    direct_override = models.model_profile_from_hf_config(qwen_cfg, weight_dtype="bf16")
    assert direct_override.name == "qwen3.5-397b"
    assert direct_override.weight_dtype == "bf16"

    kimi_override = models.resolve_model_profile("kimi-k3", weight_dtype="fp8")
    assert kimi_override.weight_dtype == "fp8"
    assert kimi_override.expert_dtype == "mxfp4"

    kimi_explicit = models.resolve_model_profile(
        "kimi-k3", weight_dtype="fp8", expert_dtype="fp8"
    )
    assert kimi_explicit.weight_dtype == "fp8"
    assert kimi_explicit.expert_dtype == "fp8"


def test_load_measurements_rejects_zero_step_coverage() -> None:
    """Verify measurements parser rejects step_coverage <= 0 like WorkloadSpec."""
    with pytest.raises(errors.MegakernelConfigError, match=r"\(0, 1\]"):
        specs.load_measurements(
            {
                "schema": constants.MEASUREMENTS_SCHEMA,
                "step_coverage": {"fused_eligible_fraction": 0.0},
            }
        )


def test_chip_spec_loader_and_overrides_fail_closed() -> None:
    """Verify ChipSpec validation fails closed, supports replace, and scales."""
    tpu7x_spec = chips.resolve_chip_spec("tpu7x")
    with pytest.raises(errors.MegakernelConfigError, match="vector_f32_ops_per_s"):
        chips.resolve_chip_spec(
            dataclasses.replace(tpu7x_spec, vector_f32_ops_per_s=0.0)
        )
    resolve_fn: Callable[..., chips.ChipSpec] = chips.resolve_chip_spec
    for removed_kwarg in (
        {"ici_bw_gbps": 200.0},
        {"collective_latency_us": 2.5},
    ):
        with pytest.raises(TypeError):
            resolve_fn("tpu7x", **removed_kwarg)

    overridden = chips.resolve_chip_spec(
        dataclasses.replace(
            tpu7x_spec,
            ici_egress_bw_bytes_per_s=200.0e9,
            collective_latency_s=2.5e-6,
        )
    )
    expected_ici_egress = 200.0e9
    expected_latency_s = 2.5e-6
    assert overridden.ici_egress_bw_bytes_per_s == pytest.approx(expected_ici_egress)
    assert overridden.collective_latency_s == pytest.approx(expected_latency_s)

    single_mc = dataclasses.replace(
        tpu7x_spec, name="custom_mc", supports_megacore=True
    )
    dual_mc = chips.resolve_chip_spec(single_mc, megacore=True)
    expected_cores = 2
    assert dual_mc.cores_per_device == expected_cores
    assert dual_mc.ici_link_bw_bytes_per_s == pytest.approx(
        single_mc.ici_link_bw_bytes_per_s
    )
    assert dual_mc.vector_load_cap_bytes_per_s == pytest.approx(
        single_mc.vector_load_cap_bytes_per_s * expected_cores
    )


@pytest.mark.parametrize(
    "workload_kwargs",
    [
        {"batch": 0},
        {"batch": -4},
        {"batch": True},
        {"tp": 0},
        {"pp": 0},
        {"dp": 0},
        {"ep": 0},
        {"kv_split": 0},
        {"dcp": 0},
        {"pcp": 0},
        {"isl": True},
        {"isl": -1},
        {"prefix_hit_rate": -0.1},
        {"prefix_hit_rate": 1.0},
        {"prefix_hit_rate": float("nan")},
        {"dma_buffers": 0},
        {"dma_buffers": 99},
        {"hbm_utilization_cap": 0.0},
        {"hbm_utilization_cap": 1.5},
        {"hbm_utilization_cap": float("nan")},
        {"vmem_budget_mib": 0.0},
        {"vmem_budget_mib": -10.0},
        {"vmem_budget_mib": 1.0},
        {"vmem_budget_mib": float("nan")},
        {"max_rows_per_call": -1},
        {"layer_groups": (0, 60)},
        {"layer_groups": (True,)},
    ],
)
def test_workload_spec_and_measurements_bounds_fail_closed(
    workload_kwargs: dict[str, object],
) -> None:
    """Verify WorkloadSpec scalar bounds and measurements validation fail closed."""
    workload_ctor: Callable[..., specs.WorkloadSpec] = specs.WorkloadSpec
    with pytest.raises(errors.MegakernelConfigError):
        workload_ctor(**workload_kwargs)


def test_measurements_loader_rejects_invalid_entries_and_parses_bytes() -> None:
    """Verify measurements loader rejects malformed entries and parses bytes."""
    bad_payloads: list[dict[str, object]] = [
        {"collectives": [{"op": "all-reduce", "group_size": 1, "latency_us": 2.0}]},
        {"collectives": [{"op": "all-reduce", "group_size": 8}]},
        {
            "collectives": [
                {"op": "all-reduce", "group_size": 8, "latency_us": 2.0, "bytes": -1}
            ]
        },
        {
            "collectives": [
                {"op": "all-reduce", "group_size": 8, "latency_us": 2.0, "bytes": True}
            ]
        },
        {"launch_gap_us": {"p50": 5.0, "p90": 1.0}},
        {"launch_gap_us": {"p90": 3.0}},
        {"host_head_us": {"p50": 4.0, "p90": 2.0}},
    ]
    for payload in bad_payloads:
        with pytest.raises(errors.MegakernelConfigError):
            specs.load_measurements(
                {"schema": constants.MEASUREMENTS_SCHEMA, **payload}
            )

    expected_bytes = 32768
    parsed = specs.load_measurements(
        {
            "schema": constants.MEASUREMENTS_SCHEMA,
            "collectives": [
                {
                    "op": "all_reduce",
                    "group_size": 8,
                    "bytes": expected_bytes,
                    "latency_us": 5.0,
                    "n": 4,
                },
                {
                    "op": "all_gather",
                    "group_size": 8,
                    "latency_us": 3.0,
                    "n": 4,
                },
            ],
        }
    )
    assert parsed.collective_calls[0].bytes == expected_bytes
    assert parsed.collective_calls[1].bytes is None


def test_parallel_layout_submodule_label_and_bounds() -> None:
    """Verify ParallelLayout submodule_label uses gcd and rejects invalid bounds."""
    layout = models.ParallelLayout(tp=6, ep=4, dp=2)
    assert layout.submodule_label() == "Attn: TP6 x DP2 \u00b7 MoE: TP3 x EP4"
    for bad_layout in (
        {"tp": 0, "ep": 1},
        {"tp": 4, "ep": -2},
        {"tp": 4, "ep": 1, "dp": 0},
        {"tp": 4, "ep": 1, "pp": 0},
        {"tp": 4, "ep": 1, "dcp": 0},
        {"tp": 4, "ep": 1, "pcp": 0},
        {"tp": 4, "ep": 1, "spec_tokens": -1},
        {"tp": 4, "ep": 1, "layer_groups": (0,)},
    ):
        with pytest.raises(errors.MegakernelConfigError):
            models.parse_parallel_layout(bad_layout)


def two_run_periodic(flags: list[bool]) -> bool:
    """Return whether flags repeats a cycle of at most two contiguous runs."""
    total = len(flags)
    max_runs = 2
    for period in range(1, total + 1):
        if total % period or flags[:period] * (total // period) != flags:
            continue
        runs = 1 + sum(flags[idx] != flags[idx - 1] for idx in range(1, period))
        if runs <= max_runs:
            return True
    return False


def test_linear_schedule_periodicity_matches_brute_force() -> None:
    """Verify linear schedule periodicity matches brute-force cycle check."""
    min_layers = 2
    max_layers = 10
    patterns = [
        list(flags)
        for total_layers in range(min_layers, max_layers + 1)
        for flags in itertools.product((False, True), repeat=total_layers)
        if 0 < sum(flags) < total_layers
    ]

    layer_mismatches: list[str] = []
    for flags in patterns:
        total_layers = len(flags)
        count_linear = sum(flags)
        expected = (count_linear, two_run_periodic(flags))
        layer_types = tuple(
            "linear_attention" if is_lin else "full_attention" for is_lin in flags
        )
        got = _resolve_linear_schedule({}, {}, total_layers, layer_types)
        if got != expected:
            pattern_str = "".join("L" if is_lin else "F" for is_lin in flags)
            layer_mismatches.append(f"{pattern_str}: got={got}, expected={expected}")
    assert not layer_mismatches, f"layer_types mismatches: {layer_mismatches[:5]}"

    kda_mismatches: list[str] = []
    for flags in patterns:
        total_layers = len(flags)
        count_linear = sum(flags)
        expected = (count_linear, two_run_periodic(flags))
        kda_indices = [idx for idx, is_lin in enumerate(flags) if is_lin]
        got = _resolve_linear_schedule(
            {}, {"kda_layers": kda_indices}, total_layers, None
        )
        if got != expected:
            pattern_str = "".join("L" if is_lin else "F" for is_lin in flags)
            kda_mismatches.append(f"{pattern_str}: got={got}, expected={expected}")
    assert not kda_mismatches, f"kda_layers mismatches: {kda_mismatches[:5]}"

    bad_layer_count = 4
    for bad_indices in ([1, 1], [bad_layer_count], [-1]):
        with pytest.raises(models.UnsupportedArchitectureError):
            _resolve_linear_schedule(
                {}, {"kda_layers": bad_indices}, bad_layer_count, None
            )


def test_quant_scheme_without_snapshot_fixture_is_rejected() -> None:
    """Verify unsupported quant schemes and preset dtype overrides fail closed."""
    snapshot_schemes: set[tuple[str, str | None]] = set()
    for slug in advisor.MODEL_PRESETS:
        payload = json.loads(
            (HF_CONFIG_DIR / f"{slug}.json").read_text(encoding="utf-8")
        )
        config_obj = payload["config"]
        assert isinstance(config_obj, dict)
        text_obj = config_obj.get("text_config")
        quant_cfg = config_obj.get("quantization_config") or (
            text_obj.get("quantization_config") if isinstance(text_obj, dict) else None
        )
        assert isinstance(quant_cfg, dict)
        raw_format = quant_cfg.get("format")
        snapshot_schemes.add(
            (
                str(quant_cfg["quant_method"]).strip().lower(),
                str(raw_format).strip().lower() if raw_format is not None else None,
            )
        )
    assert snapshot_schemes == models.SUPPORTED_QUANT_SCHEMES

    unsupported_qconfigs: tuple[dict[str, object], ...] = (
        {"quant_method": "awq", "bits": 4},
        {"quant_method": "gptq", "bits": 4},
        {"quant_method": "compressed-tensors", "format": "int-quantized"},
        {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized"},
        {"quant_method": "bitsandbytes", "load_in_8bit": True},
    )
    for bad_qconfig in unsupported_qconfigs:
        cfg = load_qwen35_snapshot_config()
        cfg["quantization_config"] = bad_qconfig
        with pytest.raises(models.UnsupportedArchitectureError):
            models.parse_hf_config_profile(cfg, name="x")

    for slug in advisor.MODEL_PRESETS:
        for bad_dtype in ("int8", "int4", "nvfp4", "fp16"):
            with pytest.raises(models.UnsupportedArchitectureError):
                models.resolve_model_profile(slug, weight_dtype=bad_dtype)
            with pytest.raises(models.UnsupportedArchitectureError):
                models.resolve_model_profile(slug, expert_dtype=bad_dtype)

    allowed_override_dtypes = (
        {"bf16"}
        | {preset.weight_dtype for preset in advisor.MODEL_PRESETS.values()}
        | {
            preset.expert_dtype
            for preset in advisor.MODEL_PRESETS.values()
            if preset.expert_dtype
        }
    )
    for slug in advisor.MODEL_PRESETS:
        for ok_dtype in sorted(allowed_override_dtypes):
            weight_overridden = models.resolve_model_profile(
                slug, weight_dtype=ok_dtype
            )
            assert weight_overridden.weight_dtype == ok_dtype
            expert_overridden = models.resolve_model_profile(
                slug, expert_dtype=ok_dtype
            )
            assert expert_overridden.expert_dtype == ok_dtype


def test_workload_spec_rejects_unknown_literals_and_bad_scalars() -> None:
    """Verify WorkloadSpec literal/bool/int guards and strict step_time_us."""
    workload_ctor: Callable[..., specs.WorkloadSpec] = specs.WorkloadSpec
    bad_workload_kwargs: tuple[dict[str, object], ...] = (
        {"phase": "bogus"},
        {"moe_mode": "bogus"},
        {"objective": "bogus"},
        {"dispatch_regime": "bogus"},
        {"vmem_alias": "false"},
        {"vmem_fit_headroom": 1},
        {"batch": True},
        {"batch": 2.5},
    )
    for bad_kwargs in bad_workload_kwargs:
        with pytest.raises(errors.MegakernelConfigError):
            workload_ctor(**bad_kwargs)

    with pytest.raises(errors.MegakernelConfigError):
        specs.load_measurements(
            {
                "schema": constants.MEASUREMENTS_SCHEMA,
                "step_time_us": {"p50": 0.0, "p90": 0.0, "n": 1},
            }
        )


def test_qwen35_default_serving_state_dtype_is_bf16() -> None:
    """Verify Qwen3.5 resolves bf16 serving recurrent cache while Kimi-K3 uses fp32."""
    qwen_preset = models.resolve_model_profile("qwen3.5-397b")
    assert qwen_preset.state_dtype == "bf16"
    assert qwen_preset.kv_dtype == "fp8"

    qwen_cfg = load_qwen35_snapshot_config()
    qwen_text_cfg = qwen_cfg["text_config"]
    assert isinstance(qwen_text_cfg, dict)
    assert qwen_text_cfg["mamba_ssm_dtype"] == "float32"

    qwen_hf = models.model_profile_from_hf_config(qwen_cfg)
    assert qwen_hf.state_dtype == "bf16"
    assert qwen_hf.kv_dtype == "fp8"
    assert models.resolve_model_profile(qwen_cfg).state_dtype == "bf16"

    qwen_override = models.resolve_model_profile("qwen3.5-397b", state_dtype="fp32")
    assert qwen_override.state_dtype == "fp32"
    qwen_hf_override = models.model_profile_from_hf_config(qwen_cfg, state_dtype="fp32")
    assert qwen_hf_override.state_dtype == "fp32"

    kimi_preset = models.resolve_model_profile("kimi-k3")
    assert kimi_preset.state_dtype == "fp32"
    kimi_cfg = _load_preset_snapshots()["kimi-k3"]["config"]
    assert isinstance(kimi_cfg, dict)
    assert models.model_profile_from_hf_config(kimi_cfg).state_dtype == "fp32"
    assert models.resolve_model_profile(kimi_cfg).state_dtype == "fp32"


def test_core_limits_matches_live_core_and_drops_supported_chips() -> None:
    """Verify CORE_LIMITS reads live megakernel/core constants and drops dead table."""
    assert not hasattr(constants, "SUPPORTED_CHIPS")
    assert not hasattr(chips, "SUPPORTED_CHIPS")
    assert "SUPPORTED_CHIPS" not in chips.__all__

    core_tiles = core_exec_spec.Tiles()
    limits = constants.CORE_LIMITS
    expected_ranks = (1, 2, 4, 8, 16)
    assert limits == constants.CoreLimits()
    assert limits.max_batch_rows == core_exec_spec.MAX_BATCH_ROWS
    assert constants.CORE_LIMITS.verified_ranks == expected_ranks
    assert limits.chip_kinds == tuple(core_default_profiles.CHIP_BUDGETS.keys())
    expected_ar_slots = 2
    assert limits.max_ar_slots_per_layer == expected_ar_slots
    assert limits.dense_ring_slots == core_tiles.nbd
    assert limits.expert_ring_slots == core_tiles.nbe
    assert limits.dense_slot_bytes == core_tiles.bank
    assert limits.expert_slot_bytes == core_tiles.ebank
    assert limits.vmem_mib == core_tiles.vmem_mib
    assert limits.smem_mib == core_tiles.smem_mib
    assert limits.wide_rows == core_tiles.wide_rows
    assert limits.default_fit_max_rows == core_vmem.DEFAULT_FIT_MAX_ROWS
    assert limits.fit_vmem_headroom_mib == core_vmem.FIT_VMEM_HEADROOM_MIB
    assert limits.default_decode_bucket_top == max(core_rt_constants.DEFAULT_BUCKETS)
    assert limits.default_verify_bucket_top == max(
        core_rt_constants.DEFAULT_VERIFY_BUCKETS
    )
