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

import pytest

from tests.roofline.conftest import SMALL_HYBRID_MOE_CFG
from vllm_torchtpu.roofline import (
    Dtypes,
    KeyTracker,
    Parallel,
    SeqKvLayout,
    audit_hf_config,
    count_layer_types,
    from_hf_config,
    param_counts,
    pass_costs,
)
from vllm_torchtpu.roofline.constants import (
    GLU_MATMUL_COUNT,
    NON_GATED_MATMUL_COUNT,
)
from vllm_torchtpu.roofline.memory import (
    compute_w_attn,
    hbm_breakdown_per_core_bytes,
)
from vllm_torchtpu.roofline.modules import StepShape, kv_bytes_per_token_per_layer
from vllm_torchtpu.roofline.solver import (
    _valid_gdn_degree,
    _valid_raiden_gdn_pair,
)

_UNIT_BYTE_TOL = 1e-6
_KDA_EXPECTED_LAYERS = 6
_KDA_EXPECTED_GQA = 2
_NON_KDA_GDN_LAYERS = 3
_KDA_VALID_TP = 8
_KDA_SUB_TP = 4
_KDA_BAD_DIV_TP = 3
_KDA_OVER_HEADS_TP = 16
_INVALID_FFN_MATRICES = 4
_EIGHT_LAYER_COUNT = 8
_FOUR_LAYER_COUNT = 4
_THREE_LAYER_COUNT = 3
_TWO_LAYER_COUNT = 2
_CONTEXT_1024 = 1024


def _mla_moe_cfg() -> dict[str, object]:
    """Build a synthetic DeepSeek-style MLA+MoE HF config dictionary."""
    return {
        "model_type": "deepseek_v3",
        "architectures": ["DeepseekV3ForCausalLM"],
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


def _opt_dense_cfg(*, model_type: str = "opt") -> dict[str, object]:
    """Build a synthetic dense config matching non-gated OPT MLP shape."""
    return {
        "model_type": model_type,
        "hidden_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "intermediate_size": 2048,
        "vocab_size": 50272,
        "dtype": "bfloat16",
    }


def _mla_dense_8layer_cfg() -> dict[str, object]:
    """Build an 8-layer MLA+DSA-capable dense config for layer-routing tests."""
    return {
        "model_type": "deepseek_v32",
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
    clean_cfg = _mla_moe_cfg()
    clean_audit = audit_hf_config(clean_cfg)
    assert clean_audit.onboarding_tier == "tier0_ready"
    assert clean_audit.model_type_status == "listed"
    assert clean_audit.unmapped_structural_keys == ()
    assert clean_audit.ignored_non_text_keys == ()

    drifted_cfg: dict[str, object] = {
        **_mla_moe_cfg(),
        "custom_sparse_router_dim": 64,
        "ssm_cfg": {"d_state": 16},
        "state_size": 16,
    }
    audit = audit_hf_config(drifted_cfg)
    assert audit.onboarding_tier == "tier0_ready"
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
        "model_type": "deepseek_v3",
        "text_config": _mla_moe_cfg(),
        "vision_config": {"hidden_size": 1024},
    }
    mm_audit = audit_hf_config(mm_cfg)
    assert mm_audit.onboarding_tier == "tier0_ready"
    assert mm_audit.ignored_non_text_keys == ("vision_config",)
    assert mm_audit.unmapped_structural_keys == ()

    flat_mm_cfg: dict[str, object] = {
        **_mla_moe_cfg(),
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
            {
                "model_type": "llama",
                "hidden_size": 512,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "intermediate_size": 1536,
                "vocab_size": 32000,
                "tie_word_embeddings": True,
                "per_layer_input_dim": 128,
            },
        ),
        (
            "mla_dsa_moe_hc_mtp",
            {
                **_mla_moe_cfg(),
                "layer_types": ["mla", "dsa", "mla", "dsa"],
                "index_n_heads": 8,
                "index_head_dim": 64,
                "index_topk": 256,
                "hc_mult": 4,
                "mtp_num_hidden_layers": 2,
            },
        ),
        (
            "hybrid_gdn_moe_mtp",
            dict(SMALL_HYBRID_MOE_CFG),
        ),
        (
            "kda_moe",
            {
                "model_type": "kimi_linear",
                "hidden_size": 512,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "kv_lora_rank": 128,
                "qk_nope_head_dim": 64,
                "qk_rope_head_dim": 0,
                "v_head_dim": 64,
                "full_attention_interval": 4,
                "linear_attn_config": {
                    "num_heads": 8,
                    "head_dim": 64,
                    "short_conv_kernel_size": 4,
                },
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 512,
                "routed_expert_hidden_size": 256,
                "vocab_size": 32000,
                "mtp_num_hidden_layers": 1,
                "mtp_intermediate_size": 512,
            },
        ),
        ("opt_two_matrix", _opt_dense_cfg()),
    ]

    for label, cfg in configs:
        geom = from_hf_config(cfg)
        param_breakdown = param_counts(geom)
        hbm = hbm_breakdown_per_core_bytes(geom, unit_dtypes, unit_parallel, empty_kv)
        assert abs(param_breakdown.total_params - hbm.weights) <= _UNIT_BYTE_TOL, label


def _assert_hybrid_moe_param_conservation() -> None:
    """Assert closed-form conservation for a hybrid GQA+GDN MoE+MTP model."""
    hybrid_geom = from_hf_config(SMALL_HYBRID_MOE_CFG)
    hybrid_params = param_counts(hybrid_geom)
    # 2 full GQA layers + 6 GDN layers, 8 MoE FFN layers, 1 MTP layer.
    exp_attn = 2.0 * (4096 * (32 * 256 * 2 + 32 * 256 + 2 * 2 * 256))
    exp_qkv_r = 2 * 16 * 128 + 64 * 128
    exp_z_r = 64 * 128
    exp_rec = 6.0 * (
        4096 * (exp_qkv_r + exp_z_r) + exp_z_r * 4096 + 4096 * 2 * 64 + exp_qkv_r * 4
    )
    exp_routed = 8.0 * 16 * 3 * 4096 * 1024
    exp_shared = 8.0 * 3 * 4096 * 1024
    exp_router = 8.0 * 4096 * 16
    exp_emb = float(248320 * 4096)
    exp_head = float(248320 * 4096)
    exp_mtp = (
        2.0 * 4096 * 4096
        + exp_attn / 2.0
        + (exp_routed + exp_shared + exp_router) / 8.0
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
    exp_inactive = (8.0 + 1.0) * (16 - 4) * 3.0 * 4096 * 1024
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
    mla_geom = from_hf_config(_mla_moe_cfg())
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


def _assert_tied_and_opt_param_conservation() -> None:
    """Assert parameter and MLP byte differences for tied embeddings and OPT FFN."""
    llama_base: dict[str, object] = {
        "model_type": "llama",
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "intermediate_size": 1024,
        "vocab_size": 32000,
    }
    tied_params = param_counts(
        from_hf_config({**llama_base, "tie_word_embeddings": True})
    )
    untied_params = param_counts(
        from_hf_config({**llama_base, "tie_word_embeddings": False})
    )
    assert tied_params.embedding == pytest.approx(32000 * 512)
    assert tied_params.lm_head == 0.0
    assert tied_params.active_params_per_token == pytest.approx(
        tied_params.total_params
    )
    assert untied_params.total_params - tied_params.total_params == pytest.approx(
        32000 * 512
    )

    opt_geom = from_hf_config(_opt_dense_cfg(model_type="opt"))
    gated_geom = from_hf_config(_opt_dense_cfg(model_type="llama"))
    gemma_geom = from_hf_config(
        {**_opt_dense_cfg(model_type="gemma2"), "hidden_act": "gelu_pytorch_tanh"}
    )
    assert opt_geom.ffn_num_matrices == NON_GATED_MATMUL_COUNT
    assert gated_geom.ffn_num_matrices == GLU_MATMUL_COUNT
    assert gemma_geom.ffn_num_matrices == GLU_MATMUL_COUNT
    gated_ratio = GLU_MATMUL_COUNT / NON_GATED_MATMUL_COUNT
    assert param_counts(gated_geom).dense_ffn / param_counts(opt_geom).dense_ffn == (
        pytest.approx(gated_ratio)
    )
    opt_mods = {
        m.name: m
        for m in pass_costs(opt_geom, Parallel(tp=1), StepShape(batch=1), Dtypes())
    }
    gated_mods = {
        m.name: m
        for m in pass_costs(gated_geom, Parallel(tp=1), StepShape(batch=1), Dtypes())
    }
    assert gated_mods["mlp"].hbm_bytes / opt_mods["mlp"].hbm_bytes == pytest.approx(
        gated_ratio
    )


def test_param_counts_conservation_reference_models() -> None:
    """Verify closed-form parameter equations across hybrid, MLA, tied, and OPT."""
    _assert_hybrid_moe_param_conservation()
    _assert_mla_moe_param_conservation()
    _assert_tied_and_opt_param_conservation()


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
    mla_only_b = kv_bytes_per_token_per_layer(geom_partial, dtypes, parallel, "mla")
    dsa_b = kv_bytes_per_token_per_layer(geom_partial, dtypes, parallel, "dsa")
    assert hbm_partial.kv < hbm_full.kv
    assert hbm_full.kv - hbm_partial.kv == pytest.approx(
        float(_FOUR_LAYER_COUNT * _TWO_LAYER_COUNT * _CONTEXT_1024)
        * (dsa_b - mla_only_b)
    )

    attn_costs = {
        mod.name: mod.hbm_bytes
        for mod in pass_costs(
            geom_partial, parallel, StepShape(batch=1, context=0), dtypes
        )
        if mod.name in ("mla", "dsa")
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
    kda_geom = from_hf_config(kda_cfg, allow_unlisted_model_type=True)
    kda_counts = count_layer_types(kda_geom)
    assert kda_counts.kda == _KDA_EXPECTED_LAYERS
    assert kda_counts.gqa == _KDA_EXPECTED_GQA
    assert _valid_gdn_degree(kda_geom, _KDA_VALID_TP)
    assert not _valid_gdn_degree(kda_geom, _KDA_BAD_DIV_TP)
    assert not _valid_gdn_degree(kda_geom, _KDA_OVER_HEADS_TP)
    assert _valid_raiden_gdn_pair(
        kda_geom, Parallel(tp=_KDA_VALID_TP), Parallel(tp=_KDA_SUB_TP)
    )
    assert not _valid_raiden_gdn_pair(
        kda_geom, Parallel(tp=_KDA_SUB_TP), Parallel(tp=_KDA_VALID_TP)
    )
    assert not _valid_raiden_gdn_pair(
        kda_geom, Parallel(tp=_KDA_VALID_TP), Parallel(tp=_KDA_BAD_DIV_TP)
    )

    non_kda_kimi_cfg: dict[str, object] = {
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
        "hidden_size": 512,
        "num_hidden_layers": _FOUR_LAYER_COUNT,
        "num_attention_heads": _KDA_VALID_TP,
        "num_key_value_heads": _KDA_EXPECTED_GQA,
        "head_dim": 64,
        "full_attention_interval": 4,
        "linear_num_key_heads": _KDA_SUB_TP,
        "linear_num_value_heads": _KDA_VALID_TP,
        "linear_key_head_dim": 64,
        "linear_value_head_dim": 64,
        "linear_conv_kernel_dim": 4,
        "intermediate_size": 1024,
        "vocab_size": 32000,
    }
    generic_lin_counts = count_layer_types(from_hf_config(non_kda_kimi_cfg))
    assert generic_lin_counts.kda == 0
    assert generic_lin_counts.gdn == _NON_KDA_GDN_LAYERS


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
    geom = from_hf_config({**_opt_dense_cfg(model_type="llama"), **overrides})
    assert geom.ffn_num_matrices == expected_matrices

    with pytest.raises(ValueError, match="Unsupported ffn_num_matrices"):
        from_hf_config(
            {
                **_opt_dense_cfg(model_type="llama"),
                "ffn_num_matrices": _INVALID_FFN_MATRICES,
            }
        )

    moe_gated = from_hf_config(_mla_moe_cfg())
    moe_non_gated = from_hf_config(
        {**_mla_moe_cfg(), "ffn_num_matrices": NON_GATED_MATMUL_COUNT}
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
    mask_counts = count_layer_types(
        from_hf_config({**base_mla, "dsa_layers": bool_mask})
    )
    assert (mask_counts.mla, mask_counts.dsa) == (_EIGHT_LAYER_COUNT, _TWO_LAYER_COUNT)

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
