# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Native ML4 metadata conversion without weights or an engine."""

import json

import pytest
from vllm.transformers_utils.config import get_config_parser
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
)

from vllm_torchtpu.models.vllm.mistral_large4.checkpoint_config import (
    adapt_mistral_large4_config,
    get_layer_specs,
)


@pytest.fixture
def native_config():
    return {
        "dim": 128,
        "hidden_dim": 256,
        "n_layers": 4,
        "n_heads": 2,
        "n_kv_heads": 2,
        "head_dim": 128,
        "vocab_size": 256,
        "norm_eps": 1e-5,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "nope": "rope,nope",
        "q_lora_rank": None,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
        "lerope": True,
        "ssmax_scaling": True,
        "softmax_tanh_gating": 30.0,
        "yarn": {
            "factor": 8,
            "original_max_position_embeddings": 512,
            "apply_scale": False,
            "beta": 32,
            "alpha": 1,
        },
        "kv_sharing": {"source_layers_str": "0,1,2,1"},
        "dsa": {
            "topk": 2048,
            "n_heads": 32,
            "head_dim": 128,
            "rope_head_dim": 64,
            "match_nope": True,
            "ssmax_scaling": True,
            "norm_eps": 1e-6,
            "sharing": "full,shared",
            "fp8_indexer": True,
            "fp8_kv_cache": True,
        },
        "moe": {
            "route_every_n": 1,
            "first_k_dense_replace": 1,
            "num_experts_per_tok": 4,
            "num_experts": 8,
            "expert_hidden_dim": 256,
            "routed_scale": 1.0,
            "num_shared_experts": 1,
            "num_expert_groups": 1,
            "num_expert_groups_per_tok": 1,
        },
        "quantization": {
            "qformat_weight": "fp8_e4m3",
            "qscheme_act": "BLOCK",
            "qscheme_weight": "BLOCK",
            "block_size_out": 128,
            "block_size_in": 128,
        },
        "vision_encoder": {
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "patch_size": 14,
            "spatial_merge_size": 2,
        },
    }


def test_native_metadata_and_registered_parser(native_config, tmp_path):
    original = json.dumps(native_config, sort_keys=True)
    (tmp_path / "params.json").write_text(json.dumps(native_config))
    raw, config = get_config_parser("mistral").parse(tmp_path, False)
    assert raw == native_config
    assert json.dumps(native_config, sort_keys=True) == original
    text = config.get_text_config()
    assert text.model_type == "mistral_large_4"
    assert config.architectures == ["MistralLarge4ForConditionalGeneration"]
    assert text.architectures == ["MistralLarge4ForCausalLM"]
    assert text.hidden_size == 128
    assert text.rope_parameters["attention_factor"] == 1.0
    assert text.quantization_config["weight_block_size"] == [128, 128]
    specs = get_layer_specs(text)
    assert [s.use_rope for s in specs] == [True, False, True, False]
    assert [s.has_indexer for s in specs] == [True, False, True, False]
    assert [s.is_moe for s in specs] == [False, True, True, True]
    assert specs[3].shares_kv and specs[3].kv_source_layer == 1
    converter = MODEL_ARCH_CONFIG_CONVERTORS[config.model_type](config, text)
    assert converter.is_deepseek_mla()


def test_standard_mistral_parser_is_preserved(tmp_path):
    native = {
        "dim": 128,
        "hidden_dim": 256,
        "n_layers": 2,
        "n_heads": 2,
        "n_kv_heads": 2,
        "vocab_size": 256,
        "norm_eps": 1e-5,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "dtype": "bfloat16",
    }
    (tmp_path / "params.json").write_text(json.dumps(native))
    _, config = get_config_parser("mistral").parse(tmp_path, False)
    assert config.architectures == ["MistralForCausalLM"]
    assert config.hidden_size == 128


@pytest.mark.parametrize("sources", ["0,1", "0,1,3,1", "0,1,2,0"])
def test_invalid_shared_cache_layout(native_config, sources):
    native_config["kv_sharing"]["source_layers_str"] = sources
    with pytest.raises(ValueError, match="patterns|KV sources|RoPE"):
        adapt_mistral_large4_config(native_config)
