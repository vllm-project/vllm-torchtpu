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
"""Authentic HuggingFace fixture tests for canonical model identity and geometry."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from tests.roofline.conftest import (
    LOCKED_FIXTURE_FILES,
    load_locked_bundle,
    load_locked_config,
)
from vllm_torchtpu.roofline import (
    HF_REPO_TO_CANONICAL_ID,
    SUPPORTED_MODELS,
    from_hf_config,
    resolve_model_identity,
)
from vllm_torchtpu.roofline.geometry import UnsupportedArchitectureError, supports

SUPPORTED_FIXTURES: tuple[tuple[str, str, str], ...] = (
    ("qwen3_5_397b_fp8.json", "qwen3_5_moe", "Qwen/Qwen3.5-397B-A17B-FP8"),
    ("qwen3_5_397b_nvfp4.json", "qwen3_5_moe", "nvidia/Qwen3.5-397B-A17B-NVFP4"),
    ("glm_5_2_fp8.json", "glm_moe_dsa", "zai-org/GLM-5.2-FP8"),
    ("glm_5_2_nvfp4.json", "glm_moe_dsa", "nvidia/GLM-5.2-NVFP4"),
)

_EXPECTED_FIXTURE_GEOMETRY: dict[str, tuple[int, int, int, int, int, int]] = {
    "qwen3_5_397b_fp8.json": (60, 4096, 32, 2, 256, 512),
    "qwen3_5_397b_nvfp4.json": (60, 4096, 32, 2, 256, 512),
    "glm_5_2_fp8.json": (78, 6144, 64, 64, 192, 256),
    "glm_5_2_nvfp4.json": (78, 6144, 64, 64, 192, 256),
}

SYNTHETIC_NEGATIVE_CONTROLS: tuple[tuple[dict[str, object], str], ...] = (
    (
        {
            "model_type": "unsupported_synthetic_arch",
            "architectures": ["UnsupportedSyntheticForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 8,
            "num_attention_heads": 32,
        },
        "unsupported_synthetic_arch",
    ),
    (
        {
            "model_type": "qwen4_exp",
            "architectures": ["Qwen4ExpForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 8,
            "num_attention_heads": 32,
        },
        "qwen4_exp",
    ),
    (
        {
            "model_type": "deepseek_v4",
            "architectures": ["DeepseekV4ForCausalLM"],
            "hidden_size": 6144,
            "num_hidden_layers": 8,
            "num_attention_heads": 64,
        },
        "deepseek_v4",
    ),
    (
        {
            "model_type": "deepseek_v41",
            "architectures": ["DeepseekV41ForCausalLM"],
            "hidden_size": 6144,
            "num_hidden_layers": 8,
            "num_attention_heads": 64,
        },
        "deepseek_v41",
    ),
    (
        {
            "model_type": "minimax_m3_vl",
            "architectures": ["MiniMaxM3VLForConditionalGeneration"],
            "hidden_size": 4096,
            "num_hidden_layers": 8,
            "num_attention_heads": 32,
        },
        "minimax_m3_vl",
    ),
    (
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "is_causal": False,
            "hidden_size": 4096,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
        },
        "Non-causal",
    ),
)


def _text_config_copy(config: Mapping[str, object]) -> dict[str, object]:
    """Return a mutable copy of a locked fixture's nested text_config mapping."""
    text_cfg = config.get("text_config")
    assert isinstance(text_cfg, Mapping)
    return dict(text_cfg)


def test_every_supported_model_has_authentic_fixture_and_repo_mapping() -> None:
    """Verify every SUPPORTED_MODELS entry has checked-in authentic fixtures."""
    assert tuple(name for name, _, _ in SUPPORTED_FIXTURES) == LOCKED_FIXTURE_FILES
    covered_ids = {canonical_id for _, canonical_id, _ in SUPPORTED_FIXTURES}
    assert covered_ids == set(SUPPORTED_MODELS)
    for spec in SUPPORTED_MODELS.values():
        assert spec.hf_repos
        for repo in spec.hf_repos:
            assert HF_REPO_TO_CANONICAL_ID[repo.lower()] == spec.canonical_id


@pytest.mark.parametrize(
    ("fixture_name", "expected_id", "hf_repo"),
    SUPPORTED_FIXTURES,
)
def test_supported_authentic_hf_fixtures_resolve_and_extract_geometry(
    fixture_name: str,
    expected_id: str,
    hf_repo: str,
) -> None:
    """Verify canonical identity, repo matching, and geometry extraction."""
    bundle = load_locked_bundle(fixture_name)
    assert bundle.identity.canonical_id == expected_id
    identity = resolve_model_identity(bundle.config, model_ref=hf_repo)
    assert identity.canonical_id == expected_id
    assert identity.hf_repo == hf_repo

    geometry = from_hf_config(bundle.config, model_ref=hf_repo)
    assert geometry == bundle.geometry
    assert geometry.model_id == expected_id
    expected_tuple = _EXPECTED_FIXTURE_GEOMETRY[fixture_name]
    assert (
        geometry.num_layers,
        geometry.hidden,
        geometry.num_heads,
        geometry.num_kv_heads,
        geometry.head_dim,
        geometry.num_experts,
    ) == expected_tuple

    wrong_repo = (
        "zai-org/GLM-5.2-FP8"
        if expected_id != "glm_moe_dsa"
        else "Qwen/Qwen3.5-397B-A17B-FP8"
    )
    with pytest.raises(UnsupportedArchitectureError, match="does not match"):
        resolve_model_identity(bundle.config, model_ref=wrong_repo)


@pytest.mark.parametrize(
    ("fixture_name", "hf_repo"),
    [
        ("glm_5_2_fp8.json", "zai-org/GLM-5.2-FP8"),
        ("glm_5_2_nvfp4.json", "nvidia/GLM-5.2-NVFP4"),
    ],
)
def test_glm_5_2_authentic_fixture_parses_index_share_and_mlp_layer_types(
    fixture_name: str,
    hf_repo: str,
) -> None:
    """Verify GLM-5.2 parses IndexShare indexer_types and mlp_layer_types."""
    expected_layers = 78
    expected_full_idx = 21
    expected_shared_idx = 57
    expected_dense_ffn = 3
    expected_moe_ffn = 75
    geometry = from_hf_config(load_locked_config(fixture_name), model_ref=hf_repo)
    assert geometry.num_layers == expected_layers
    assert len(geometry.indexer_types) == expected_layers
    assert geometry.indexer_types.count("full") == expected_full_idx
    assert geometry.indexer_types.count("shared") == expected_shared_idx
    assert geometry.index_share_for_mtp_iteration is True
    assert geometry.ffn_layer_counts() == (expected_moe_ffn, expected_dense_ffn)


@pytest.mark.parametrize(("config", "match_text"), SYNTHETIC_NEGATIVE_CONTROLS)
def test_negative_control_synthetic_configs_fail_closed(
    config: dict[str, object],
    match_text: str,
) -> None:
    """Verify unmodeled synthetic model types and non-causal drafts fail closed."""
    with pytest.raises(UnsupportedArchitectureError, match=match_text):
        from_hf_config(config)


@pytest.mark.parametrize(
    ("config", "match_text"),
    [
        (
            {
                "model_type": "muse_spark",
                "architectures": ["MuseSparkForCausalLM"],
                "hidden_size": 8192,
                "num_hidden_layers": 62,
                "num_attention_heads": 128,
            },
            "muse_spark",
        ),
        (
            {
                "model_type": "llama4_text",
                "hidden_size": 5120,
                "num_hidden_layers": 48,
                "num_attention_heads": 40,
            },
            "llama4_text",
        ),
        (
            {
                "model_type": "gemma2",
                "hidden_size": 3584,
                "num_hidden_layers": 42,
                "num_attention_heads": 16,
            },
            "gemma2",
        ),
        (
            {
                "model_type": "gemma4_text",
                "hidden_size": 4096,
                "num_hidden_layers": 48,
                "num_attention_heads": 32,
            },
            "gemma4_text",
        ),
        (
            {
                "model_type": "qwen2",
                "hidden_size": 3584,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
            },
            "qwen2",
        ),
        (
            {
                "hidden_size": 4096,
                "num_hidden_layers": 8,
                "num_attention_heads": 32,
            },
            "missing required 'model_type'",
        ),
        (
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["DFlash2DraftModel"],
                "hidden_size": 4096,
                "num_hidden_layers": 4,
                "num_attention_heads": 32,
            },
            "architectures",
        ),
    ],
)
def test_unsupported_omitted_model_type_and_bad_architecture_fail_closed(
    config: dict[str, object],
    match_text: str,
) -> None:
    """Verify unsupported model_types, omitted model_type, and bad arch fail."""
    with pytest.raises(UnsupportedArchitectureError, match=match_text):
        from_hf_config(config)


def test_unlisted_repo_conflicting_aliases_interval_and_dedicated_mtp_fail_closed() -> (
    None
):
    """Verify unlisted repos, alias conflicts, bad intervals, and ded MTP fail."""
    qwen_config = load_locked_config("qwen3_5_397b_fp8.json")
    with pytest.raises(UnsupportedArchitectureError, match="Unlisted HuggingFace repo"):
        from_hf_config(qwen_config, model_ref="someorg/NotARealRepo")

    glm_config = load_locked_config("glm_5_2_nvfp4.json")
    bad_alias = dict(glm_config, num_experts=128)
    with pytest.raises(ValueError, match="Conflicting HF config alias keys"):
        from_hf_config(bad_alias)

    qwen_text = _text_config_copy(qwen_config)
    bad_conflict_interval = {
        **qwen_config,
        "text_config": {**qwen_text, "full_attention_interval": 2},
    }
    with pytest.raises(ValueError, match="full_attention_interval"):
        from_hf_config(bad_conflict_interval)

    qwen_no_layers = _text_config_copy(qwen_config)
    qwen_no_layers.pop("layer_types", None)
    bad_interval = {
        **qwen_config,
        "text_config": {**qwen_no_layers, "full_attention_interval": 0},
    }
    with pytest.raises(ValueError, match="full_attention_interval"):
        from_hf_config(bad_interval)

    bad_mtp_emb = {
        **qwen_config,
        "text_config": {
            **qwen_text,
            "mtp_use_dedicated_embeddings": True,
        },
    }
    with pytest.raises(UnsupportedArchitectureError, match="Dedicated MTP embeddings"):
        from_hf_config(bad_mtp_emb)


def test_spoofed_sliding_window_and_kda_under_listed_model_type_fail_closed() -> None:
    """Verify per-family allowlist blocks spoofed SWA, KDA, or MLA configs."""
    qwen_base = load_locked_config("qwen3_5_397b_fp8.json")
    qwen_text = _text_config_copy(qwen_base)
    raw_num_layers = qwen_text["num_hidden_layers"]
    assert isinstance(raw_num_layers, int)
    num_layers = raw_num_layers

    spoofed_swa = {
        **qwen_base,
        "text_config": {
            **qwen_text,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ]
            * (num_layers // 4),
            "sliding_window": 256,
        },
    }
    with pytest.raises(
        UnsupportedArchitectureError, match="does not allow layer kinds"
    ):
        supports(spoofed_swa)
    with pytest.raises(
        UnsupportedArchitectureError, match="does not allow layer kinds"
    ):
        from_hf_config(spoofed_swa)

    qwen_kda_text = _text_config_copy(qwen_base)
    qwen_kda_text.pop("layer_types", None)
    spoofed_kda = {
        **qwen_base,
        "text_config": {
            **qwen_kda_text,
            "full_attention_interval": 4,
            "linear_attn_config": {
                "num_heads": 16,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
            },
        },
    }
    with pytest.raises(
        UnsupportedArchitectureError, match="does not allow layer kinds"
    ):
        supports(spoofed_kda)
    with pytest.raises(
        UnsupportedArchitectureError, match="does not allow layer kinds"
    ):
        from_hf_config(spoofed_kda)

    spoofed_mla = {**qwen_base, "text_config": {**qwen_text, "kv_lora_rank": 512}}
    with pytest.raises(
        UnsupportedArchitectureError, match="does not allow layer kinds"
    ):
        supports(spoofed_mla)
    with pytest.raises(
        UnsupportedArchitectureError, match="does not allow layer kinds"
    ):
        from_hf_config(spoofed_mla)
