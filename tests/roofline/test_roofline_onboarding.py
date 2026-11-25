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
"""Model onboarding layer-selection tests."""

from __future__ import annotations

import pytest

from vllm_torchtpu.roofline import count_layer_types, from_hf_config

_EIGHT_LAYER_COUNT = 8
_THREE_LAYER_COUNT = 3
_TWO_LAYER_COUNT = 2


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

    # Conflicting index list against explicit dsa layer_types overrides to
    # exact indices.
    override_geom = from_hf_config(
        {
            **base_mla,
            "layer_types": ["dsa"] * _EIGHT_LAYER_COUNT,
            "dsa_layers": [0, 7],
        }
    )
    assert override_geom.layer_types == (
        "dsa",
        "mla",
        "mla",
        "mla",
        "mla",
        "mla",
        "mla",
        "dsa",
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
