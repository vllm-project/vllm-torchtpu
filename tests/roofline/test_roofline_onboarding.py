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
"""Onboarding audit, key-tracker, param-count oracle, and layer-selection tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.roofline.conftest import (
    load_locked_bundle,
    load_locked_config,
    override_locked_config,
)
from vllm_torchtpu.roofline import (
    Dtypes,
    KeyTracker,
    Parallel,
    SeqKvLayout,
    audit_hf_config,
    cli,
    count_layer_types,
    from_hf_config,
    param_counts,
    pass_costs,
)
from vllm_torchtpu.roofline.constants import (
    GLU_MATMUL_COUNT,
    NON_GATED_MATMUL_COUNT,
)
from vllm_torchtpu.roofline.geometry import _from_hf_config_dry_run
from vllm_torchtpu.roofline.memory import (
    compute_w_attn,
    hbm_breakdown_per_core_bytes,
)
from vllm_torchtpu.roofline.modules import StepShape, kv_bytes_per_token_per_layer
from vllm_torchtpu.roofline.solver import _valid_gdn_degree

_UNIT_BYTE_TOL = 1e-6
_KDA_EXPECTED_LAYERS = 6
_KDA_EXPECTED_GQA = 2
_KDA_VALID_TP = 8
_KDA_SUB_TP = 4
_KDA_PAIR_TP = 2
_KDA_BAD_DIV_TP = 3
_KDA_OVER_HEADS_TP = 16
_INVALID_FFN_MATRICES = 4
_EIGHT_LAYER_COUNT = 8
_FOUR_LAYER_COUNT = 4
_THREE_LAYER_COUNT = 3
_TWO_LAYER_COUNT = 2
_CONTEXT_1024 = 1024


def _glm_mla_moe_cfg() -> dict[str, object]:
    """Build a synthetic GLM-style MLA+MoE HF config dictionary."""
    return {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 64,
        "kv_lora_rank": 128,
        "q_lora_rank": 256,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 32,
        "v_head_dim": 64,
        "intermediate_size": 2048,
        "moe_intermediate_size": 512,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "first_k_dense_replace": 1,
        "vocab_size": 32000,
        "dtype": "bfloat16",
        "bos_token_id": 1,
        "transformers_version": "4.48.0",
        "use_cache": True,
    }


def _dense_qwen_cfg() -> dict[str, object]:
    """Build a compact dense Qwen3.5-typed config for FFN and tying tests."""
    return {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "hidden_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "intermediate_size": 2048,
        "vocab_size": 32000,
        "dtype": "bfloat16",
    }


def _mla_dense_8layer_cfg() -> dict[str, object]:
    """Build an 8-layer MLA+DSA-capable dense config for layer-routing tests."""
    return {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "hidden_size": 1024,
        "num_hidden_layers": _EIGHT_LAYER_COUNT,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 64,
        "kv_lora_rank": 512,
        "q_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "index_n_heads": 8,
        "index_head_dim": 64,
        "index_topk": 256,
        "intermediate_size": 2048,
        "vocab_size": 32000,
    }


def test_inspect_zero_drift_key_tracker_flags_unmapped_structural_key() -> None:
    """Verify KeyTracker catches unknown structural and guarded SSM keys."""
    assert KeyTracker.__name__ == "KeyTracker"
    clean_cfg = _glm_mla_moe_cfg()
    clean_audit = audit_hf_config(clean_cfg)
    assert clean_audit.onboarding_tier == "tier0_ready"
    assert clean_audit.model_type_status == "listed"
    assert clean_audit.unmapped_structural_keys == ()
    assert clean_audit.ignored_non_text_keys == ()

    drifted_cfg: dict[str, object] = {
        **_glm_mla_moe_cfg(),
        "custom_sparse_router_dim": 64,
        "ssm_cfg": {"d_state": 16},
        "state_size": 16,
    }
    audit = audit_hf_config(drifted_cfg)
    assert audit.onboarding_tier == "tier1_config_needed"
    assert audit.blocking_reasons
    assert audit.model_type_status == "listed"
    assert "custom_sparse_router_dim" in audit.unmapped_structural_keys
    assert "ssm_cfg" in audit.unmapped_structural_keys
    assert "state_size" in audit.unmapped_structural_keys
    for excluded in (
        "bos_token_id",
        "transformers_version",
        "hidden_size",
        "kv_lora_rank",
    ):
        assert excluded not in audit.unmapped_structural_keys

    mm_cfg: dict[str, object] = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "text_config": _glm_mla_moe_cfg(),
        "vision_config": {"hidden_size": 1024},
    }
    mm_audit = audit_hf_config(mm_cfg)
    assert mm_audit.onboarding_tier == "tier0_ready"
    assert mm_audit.ignored_non_text_keys == ("vision_config",)
    assert mm_audit.unmapped_structural_keys == ()

    flat_mm_cfg: dict[str, object] = {
        **_glm_mla_moe_cfg(),
        "vision_config": {"hidden_size": 1024},
    }
    flat_mm_audit = audit_hf_config(flat_mm_cfg)
    assert flat_mm_audit.onboarding_tier == "tier0_ready"
    assert flat_mm_audit.ignored_non_text_keys == ("vision_config",)
    assert flat_mm_audit.unmapped_structural_keys == ()


def test_param_counts_single_source_matches_memory_bytes_at_unit_dtype() -> None:
    """Verify param_counts.total_params matches HBM weights at unit dtype/parallel."""
    unit_dtypes = Dtypes(
        w_dense=1.0,
        w_expert=1.0,
        w_aux=1.0,
        kv=1.0,
        state=1.0,
        conv=1.0,
        router=1.0,
        lm_head=1.0,
    )
    unit_parallel = Parallel(tp=1, pp=1, ep=1)
    empty_kv = SeqKvLayout(seqs_per_dp=0, context=0)

    configs: list[tuple[str, dict[str, object]]] = [
        (
            "gqa_dense_tied",
            {**_dense_qwen_cfg(), "tie_word_embeddings": True},
        ),
        (
            "mla_dsa_moe_hc_mtp",
            {
                **_glm_mla_moe_cfg(),
                "layer_types": ["mla", "dsa", "mla", "dsa"],
                "index_n_heads": 8,
                "index_head_dim": 64,
                "index_topk": 256,
                "hc_mult": 4,
                "mtp_num_hidden_layers": 2,
            },
        ),
        (
            "qwen3_5_authentic",
            load_locked_config("qwen3_5_397b_fp8.json"),
        ),
        (
            "glm_5_2_authentic",
            load_locked_config("glm_5_2_fp8.json"),
        ),
    ]

    for label, config in configs:
        geometry = from_hf_config(config)
        param_breakdown = param_counts(geometry)
        hbm = hbm_breakdown_per_core_bytes(
            geometry, unit_dtypes, unit_parallel, empty_kv
        )
        assert abs(param_breakdown.total_params - hbm.weights) <= _UNIT_BYTE_TOL, label


def _assert_hybrid_moe_param_conservation() -> None:
    """Assert closed-form conservation for locked Qwen3.5-397B hybrid MoE+MTP."""
    hybrid_geometry = from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    hybrid_params = param_counts(hybrid_geometry)
    exp_attn = 15.0 * (4096 * (32 * 256 * 2 + 32 * 256 + 2 * 2 * 256))
    exp_qkv_r = 2 * 16 * 128 + 64 * 128
    exp_z_r = 64 * 128
    exp_rec = 45.0 * (
        4096 * (exp_qkv_r + exp_z_r) + exp_z_r * 4096 + 4096 * 2 * 64 + exp_qkv_r * 4
    )
    exp_routed = 60.0 * 512 * 3 * 4096 * 1024
    exp_shared = 60.0 * 3 * 4096 * 1024
    exp_router = 60.0 * 4096 * 512
    exp_emb = float(248320 * 4096)
    exp_head = float(248320 * 4096)
    exp_mtp = (
        2.0 * 4096 * 4096
        + exp_attn / 15.0
        + (exp_routed + exp_shared + exp_router) / 60.0
    )
    exp_total = (
        exp_attn
        + exp_rec
        + exp_routed
        + exp_shared
        + exp_router
        + exp_emb
        + exp_head
        + exp_mtp
    )
    exp_inactive = (60.0 + 1.0) * (512 - 10) * 3.0 * 4096 * 1024
    assert hybrid_params.attention == pytest.approx(exp_attn)
    assert hybrid_params.linear_attention == pytest.approx(exp_rec)
    assert hybrid_params.moe_routed_ffn == pytest.approx(exp_routed)
    assert hybrid_params.moe_shared_ffn == pytest.approx(exp_shared)
    assert hybrid_params.router == pytest.approx(exp_router)
    assert hybrid_params.embedding == pytest.approx(exp_emb)
    assert hybrid_params.lm_head == pytest.approx(exp_head)
    assert hybrid_params.mtp == pytest.approx(exp_mtp)
    assert hybrid_params.total_params == pytest.approx(exp_total)
    assert hybrid_params.active_params_per_token == pytest.approx(
        exp_total - exp_inactive
    )


def _assert_mla_moe_param_conservation() -> None:
    """Assert closed-form conservation for an MLA+MoE model with dense prefix."""
    mla_geom = from_hf_config(_glm_mla_moe_cfg())
    mla_params = param_counts(mla_geom)
    exp_mla_attn = 4.0 * (
        1024 * 256
        + 1024 * (128 + 32)
        + 256 * 16 * (64 + 32)
        + 128 * 16 * (64 + 64)
        + 16 * 64 * 1024
    )
    exp_mla_dense = 1.0 * 3 * 1024 * 2048
    exp_mla_routed = 3.0 * 8 * 3 * 1024 * 512
    exp_mla_shared = 3.0 * 3 * 1024 * 512
    exp_mla_router = 3.0 * 1024 * 8
    exp_mla_emb = float(32000 * 1024)
    exp_mla_total = (
        exp_mla_attn
        + exp_mla_dense
        + exp_mla_routed
        + exp_mla_shared
        + exp_mla_router
        + 2.0 * exp_mla_emb
    )
    assert mla_params.attention == pytest.approx(exp_mla_attn)
    assert mla_params.dense_ffn == pytest.approx(exp_mla_dense)
    assert mla_params.moe_routed_ffn == pytest.approx(exp_mla_routed)
    assert mla_params.active_params_per_token == pytest.approx(
        exp_mla_total - 3.0 * (8 - 2) * 3 * 1024 * 512
    )


def _assert_tied_and_ffn_matrix_conservation() -> None:
    """Assert parameter and MLP byte differences for tied embeddings and 2-mat FFN."""
    base = _dense_qwen_cfg()
    tied_params = param_counts(from_hf_config({**base, "tie_word_embeddings": True}))
    untied_params = param_counts(from_hf_config({**base, "tie_word_embeddings": False}))
    assert tied_params.embedding == pytest.approx(32000 * 512)
    assert tied_params.lm_head == 0.0
    assert tied_params.active_params_per_token == pytest.approx(
        tied_params.total_params
    )
    assert untied_params.total_params - tied_params.total_params == pytest.approx(
        32000 * 512
    )

    two_mat_geom = from_hf_config({**base, "ffn_num_matrices": NON_GATED_MATMUL_COUNT})
    gated_geom = from_hf_config(base)
    assert two_mat_geom.ffn_num_matrices == NON_GATED_MATMUL_COUNT
    assert gated_geom.ffn_num_matrices == GLU_MATMUL_COUNT
    gated_ratio = GLU_MATMUL_COUNT / NON_GATED_MATMUL_COUNT
    assert param_counts(gated_geom).dense_ffn / param_counts(
        two_mat_geom
    ).dense_ffn == (pytest.approx(gated_ratio))
    two_mat_mods = {
        m.name: m
        for m in pass_costs(two_mat_geom, Parallel(tp=1), StepShape(batch=1), Dtypes())
    }
    gated_mods = {
        m.name: m
        for m in pass_costs(gated_geom, Parallel(tp=1), StepShape(batch=1), Dtypes())
    }
    assert gated_mods["mlp"].hbm_bytes / two_mat_mods["mlp"].hbm_bytes == pytest.approx(
        gated_ratio
    )


def test_param_counts_conservation_reference_models() -> None:
    """Verify closed-form parameter equations across hybrid, MLA, tied, and 2-mat."""
    _assert_hybrid_moe_param_conservation()
    _assert_mla_moe_param_conservation()
    _assert_tied_and_ffn_matrix_conservation()


def test_memory_pp_worst_stage_head_mtp_and_replicated_dsa_indexer() -> None:
    """Verify PP>1 keeps whole head/MTP on worst stage and DSA indexer replicates."""
    glm_bundle = load_locked_bundle("glm_5_2_fp8.json")
    glm_geom = glm_bundle.geometry
    dtypes = glm_bundle.dtypes

    # Replicated DSA indexer weights across TP + IndexShare (21 full vs 78 total).
    counts = count_layer_types(glm_geom)
    w_tp1 = compute_w_attn(glm_geom, dtypes, Parallel(tp=1, pp=1), counts)
    w_tp4 = compute_w_attn(glm_geom, dtypes, Parallel(tp=4, pp=1), counts)
    idx_elem_per_full = float(
        glm_geom.q_lora_rank * glm_geom.index_n_heads * glm_geom.index_head_dim
        + glm_geom.hidden * glm_geom.index_head_dim
        + glm_geom.hidden * glm_geom.index_n_heads
    )
    expected_idx_bytes = 21.0 * idx_elem_per_full * dtypes.w_dense
    assert w_tp4 > (w_tp1 / 4.0)
    assert (w_tp4 - (w_tp1 - expected_idx_bytes) / 4.0) >= expected_idx_bytes

    # PP>1 does not divide embedding/lm_head/MTP by PP.
    q35_bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    q35_geom = q35_bundle.geometry
    q35_dtypes = q35_bundle.dtypes
    hbm_pp1 = hbm_breakdown_per_core_bytes(
        q35_geom, q35_dtypes, Parallel(tp=1, pp=1), SeqKvLayout(0, 0)
    )
    hbm_pp4 = hbm_breakdown_per_core_bytes(
        q35_geom, q35_dtypes, Parallel(tp=1, pp=4), SeqKvLayout(0, 0)
    )
    # Worst stage holds max(emb, lm_head + mtp), strictly larger than total / 4.
    assert hbm_pp4.weights > hbm_pp1.weights / 4.0


def test_partial_dsa_layers_and_kda_without_kimi_name() -> None:
    """Verify partial DSA KV/weight accounting and non-substring KDA checks."""
    base_mla = _mla_dense_8layer_cfg()
    geom_partial = from_hf_config({**base_mla, "dsa_layers": [0, 1, 2, 3]})
    geom_indexer = from_hf_config({**base_mla, "indexer_layers": [4, 5, 6, 7]})
    geom_full_dsa = from_hf_config(base_mla)

    counts_partial = count_layer_types(geom_partial)
    counts_indexer = count_layer_types(geom_indexer)
    counts_full = count_layer_types(geom_full_dsa)
    assert (counts_partial.mla, counts_partial.dsa) == (
        _EIGHT_LAYER_COUNT,
        _FOUR_LAYER_COUNT,
    )
    assert (counts_indexer.mla, counts_indexer.dsa) == (
        _EIGHT_LAYER_COUNT,
        _FOUR_LAYER_COUNT,
    )
    assert (counts_full.mla, counts_full.dsa) == (
        _EIGHT_LAYER_COUNT,
        _EIGHT_LAYER_COUNT,
    )

    dtypes = Dtypes(kv=1.0)
    parallel = Parallel(tp=1)
    layout = SeqKvLayout(seqs_per_dp=_TWO_LAYER_COUNT, context=_CONTEXT_1024)
    hbm_partial = hbm_breakdown_per_core_bytes(geom_partial, dtypes, parallel, layout)
    hbm_full = hbm_breakdown_per_core_bytes(geom_full_dsa, dtypes, parallel, layout)
    mla_only_b = kv_bytes_per_token_per_layer(
        geom_partial, dtypes, parallel, "mla", physical=layout.physical_kv_layout
    )
    dsa_b = kv_bytes_per_token_per_layer(
        geom_partial, dtypes, parallel, "dsa", physical=layout.physical_kv_layout
    )
    assert hbm_partial.kv < hbm_full.kv
    assert hbm_full.kv - hbm_partial.kv == pytest.approx(
        float(_FOUR_LAYER_COUNT * _TWO_LAYER_COUNT * _CONTEXT_1024)
        * (dsa_b - mla_only_b)
    )

    attn_costs = {
        module.name: module.hbm_bytes
        for module in pass_costs(
            geom_partial, parallel, StepShape(batch=1, context=0), dtypes
        )
        if module.name in ("mla", "dsa")
    }
    expected_attn_bytes = compute_w_attn(geom_partial, dtypes, parallel, counts_partial)
    assert set(attn_costs) == {"mla", "dsa"}
    assert sum(attn_costs.values()) == pytest.approx(expected_attn_bytes)

    kda_cfg: dict[str, object] = {
        "architectures": ["CustomHybridCausalLM"],
        "model_type": "custom_hybrid_kda",
        "hidden_size": 512,
        "num_hidden_layers": _EIGHT_LAYER_COUNT,
        "num_attention_heads": _KDA_VALID_TP,
        "num_key_value_heads": _KDA_EXPECTED_GQA,
        "head_dim": 64,
        "full_attention_interval": 4,
        "linear_attn_config": {
            "num_heads": _KDA_VALID_TP,
            "head_dim": 64,
            "short_conv_kernel_size": 4,
        },
        "intermediate_size": 1024,
        "vocab_size": 32000,
    }
    kda_geom = _from_hf_config_dry_run(kda_cfg)
    kda_counts = count_layer_types(kda_geom)
    assert kda_counts.kda == _KDA_EXPECTED_LAYERS
    assert kda_counts.gqa == _KDA_EXPECTED_GQA
    assert _valid_gdn_degree(kda_geom, _KDA_VALID_TP)
    assert not _valid_gdn_degree(kda_geom, _KDA_BAD_DIV_TP)
    assert not _valid_gdn_degree(kda_geom, _KDA_OVER_HEADS_TP)
    assert _valid_gdn_degree(kda_geom, 1)
    assert _valid_gdn_degree(kda_geom, _KDA_PAIR_TP)
    assert _valid_gdn_degree(kda_geom, _KDA_SUB_TP)


@pytest.mark.parametrize(
    ("overrides", "expected_matrices"),
    [
        ({"ffn_num_matrices": NON_GATED_MATMUL_COUNT}, NON_GATED_MATMUL_COUNT),
        ({"ffn_num_matrices": GLU_MATMUL_COUNT}, GLU_MATMUL_COUNT),
        ({"is_gated_mlp": False}, NON_GATED_MATMUL_COUNT),
        ({"is_gated_mlp": True}, GLU_MATMUL_COUNT),
    ],
)
def test_ffn_matrix_count_overrides_and_moe_two_matrix(
    overrides: dict[str, object],
    expected_matrices: int,
) -> None:
    """Verify explicit ffn_num_matrices/is_gated_mlp controls and 2-matrix MoE."""
    geometry = from_hf_config({**_dense_qwen_cfg(), **overrides})
    assert geometry.ffn_num_matrices == expected_matrices

    with pytest.raises(ValueError, match="Unsupported ffn_num_matrices"):
        from_hf_config(
            {
                **_dense_qwen_cfg(),
                "ffn_num_matrices": _INVALID_FFN_MATRICES,
            }
        )

    moe_gated = from_hf_config(_glm_mla_moe_cfg())
    moe_non_gated = from_hf_config(
        {**_glm_mla_moe_cfg(), "ffn_num_matrices": NON_GATED_MATMUL_COUNT}
    )
    ratio = NON_GATED_MATMUL_COUNT / GLU_MATMUL_COUNT
    gated_counts = param_counts(moe_gated)
    non_gated_counts = param_counts(moe_non_gated)
    assert non_gated_counts.moe_routed_ffn / gated_counts.moe_routed_ffn == (
        pytest.approx(ratio)
    )
    assert non_gated_counts.moe_shared_ffn / gated_counts.moe_shared_ffn == (
        pytest.approx(ratio)
    )


def test_dsa_layer_selection_masks_aliases_and_errors() -> None:
    """Verify bool masks, MLA/DSA/full-attention aliases, and invalid dsa_layers."""
    base_mla = _mla_dense_8layer_cfg()
    bool_mask = [True, False, True, False, False, False, False, False]
    mask_geom = from_hf_config({**base_mla, "dsa_layers": bool_mask})
    assert mask_geom.layer_types == (
        "dsa",
        "mla",
        "dsa",
        "mla",
        "mla",
        "mla",
        "mla",
        "mla",
    )
    mask_counts = count_layer_types(mask_geom)
    assert (mask_counts.mla, mask_counts.dsa) == (_EIGHT_LAYER_COUNT, _TWO_LAYER_COUNT)
    conflict_geom = from_hf_config(
        {
            **base_mla,
            "layer_types": ["dsa", "mla", "dsa", "mla", "mla", "mla", "mla", "mla"],
            "dsa_layers": [1, 3],
        }
    )
    assert conflict_geom.layer_types == (
        "mla",
        "dsa",
        "mla",
        "dsa",
        "mla",
        "mla",
        "mla",
        "mla",
    )

    alias_types = [
        "mla_attention",
        "dsa_attention",
        "sparse_mla",
        "attention",
        "mla",
        "dsa",
        "full_attention",
        "mla_attention",
    ]
    alias_geom = from_hf_config(
        {**base_mla, "layer_types": alias_types, "dsa_layers": [1, 2, 5]}
    )
    assert alias_geom.layer_types == (
        "mla",
        "dsa",
        "dsa",
        "mla",
        "mla",
        "dsa",
        "mla",
        "mla",
    )
    alias_counts = count_layer_types(alias_geom)
    assert (alias_counts.mla, alias_counts.dsa) == (
        _EIGHT_LAYER_COUNT,
        _THREE_LAYER_COUNT,
    )

    for bad_dsa_layers in (
        [True, False],
        [True, 1],
        ["0"],
        [-1],
        [_EIGHT_LAYER_COUNT],
    ):
        with pytest.raises(ValueError, match="dsa_layers"):
            from_hf_config({**base_mla, "dsa_layers": bad_dsa_layers})


def test_memory_tied_embeddings_replicated_on_pipeline_stages() -> None:
    """Verify tied embeddings charge an lm_head replica when PP > 1."""
    tied_cfg = override_locked_config(
        "qwen3_5_397b_fp8.json", {"tie_word_embeddings": True}
    )
    untied_cfg = override_locked_config(
        "qwen3_5_397b_fp8.json", {"tie_word_embeddings": False}
    )
    tied_geom = from_hf_config(tied_cfg)
    untied_geom = from_hf_config(untied_cfg)
    dtypes = load_locked_bundle("qwen3_5_397b_fp8.json").dtypes
    empty_kv = SeqKvLayout(0, 0)
    pp1_tied = hbm_breakdown_per_core_bytes(
        tied_geom, dtypes, Parallel(tp=8, pp=1), empty_kv
    )
    pp1_untied = hbm_breakdown_per_core_bytes(
        untied_geom, dtypes, Parallel(tp=8, pp=1), empty_kv
    )
    assert pp1_tied.weights < pp1_untied.weights

    pp4_tied = hbm_breakdown_per_core_bytes(
        tied_geom, dtypes, Parallel(tp=8, pp=4), empty_kv
    )
    pp4_untied = hbm_breakdown_per_core_bytes(
        untied_geom, dtypes, Parallel(tp=8, pp=4), empty_kv
    )
    assert pp4_tied.weights == pytest.approx(pp4_untied.weights, rel=1e-9)


def test_audit_hf_config_blocked_tier_returns_none_payload_fields() -> None:
    """Verify blocked audit tiers emit None geometry/dtypes/counts/params."""
    mutated = override_locked_config(
        "qwen3_5_397b_fp8.json", {"unknown_structural_knob": 42}
    )
    audit = audit_hf_config(mutated)
    assert audit.onboarding_tier == "tier1_config_needed"
    assert "unknown_structural_knob" in audit.unmapped_structural_keys
    assert audit.geometry is None
    assert audit.dtypes is None
    assert audit.layer_counts is None
    assert audit.param_counts is None


def test_cli_inspect_text_output_and_fail_closed_tier2_blocker(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify inspect text output for locked configs and fail-closed unlisted."""
    expected_error_rc = 2
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    q35_path = fixtures_dir / "qwen3_5_397b_fp8.json"
    rc_ok = cli.main(["inspect", "--config", str(q35_path)])
    assert rc_ok == 0
    out_ok = capsys.readouterr().out
    assert "onboarding_tier: tier0_ready" in out_ok
    assert "canonical_id: qwen3_5_moe" in out_ok
    assert "param_counts:" in out_ok

    rc_json = cli.main(["inspect", "--config", str(q35_path), "--json"])
    assert rc_json == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "vllm_torchtpu.roofline.inspect/v1"
    assert payload["onboarding_tier"] == "tier0_ready"

    unlisted_path = tmp_path / "unsupported_synthetic_arch.json"
    unlisted_path.write_text(
        json.dumps(
            {
                "model_type": "unsupported_synthetic_arch",
                "architectures": ["UnsupportedSyntheticForCausalLM"],
                "hidden_size": 4096,
                "num_hidden_layers": 4,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 64000,
                "torch_dtype": "bfloat16",
                "o_lora_rank": 16,
            }
        ),
        encoding="utf-8",
    )
    rc_blocked = cli.main(["inspect", "--config", str(unlisted_path)])
    assert rc_blocked == expected_error_rc
    out_blocked = capsys.readouterr().out
    assert "onboarding_tier: tier2_primitive_needed" in out_blocked
    assert "model_type_status: unlisted_blocked" in out_blocked
