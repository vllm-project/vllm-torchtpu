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
"""Kimi-K3.1's registration with vLLM: its config and architecture, and the
index-key cache module of its MLA layers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

FULL = [*range(3, 92, 4), 92]  # kernel MLA layers; the config's 1-based list below
DSA = dict(
    index_topk=2048,
    index_num_heads=32,
    index_head_dim=128,
    index_rope_dim=0,
    index_k_norm_type="layernorm",
    indexer_use_fp8=False,
    index_topk_freq=1,
    index_skip_topk_offset=1,
    dsa_layers=[*range(4, 93, 4), 93],
)
CONFIG = {
    "model_type": "kimi_k31",
    "architectures": ["KimiK31ForConditionalGeneration"],
    "text_config": {
        "model_type": "kimi_linear",
        "linear_attn_config": {
            "full_attn_layers": [*range(4, 93, 4), 93],
            "kda_layers": [n for n in range(1, 94) if n not in DSA["dsa_layers"]],
        },
        "dsa_config": DSA,
    },
}


def test_kimi_k31_config_is_registered(tmp_path) -> None:
    from vllm.transformers_utils.config import get_config

    from vllm_torchtpu.models.vllm import register_models

    register_models()
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    config = get_config(str(tmp_path), trust_remote_code=False)
    assert config.model_type == "kimi_k31"
    assert config.architectures == ["KimiK31ForConditionalGeneration"]
    assert dict(config.text_config.dsa_config)["index_topk"] == 2048


def test_kimi_k31_architecture_is_registered() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from vllm_torchtpu import gdn_pool_layout
    from vllm_torchtpu.distributed.kv_transfer import tpu_connector
    from vllm_torchtpu.megakernel import backend
    from vllm_torchtpu.models.vllm import register_models

    register_models()
    name = "KimiK31ForConditionalGeneration"
    assert name in ModelRegistry.get_supported_archs()
    assert name in MODELS_CONFIG_MAP
    assert name in gdn_pool_layout.POOLED_GDN_ARCHITECTURES
    assert name in backend.ARCHITECTURES
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(architecture=name))
    assert tpu_connector._is_kimi(vllm_config)


def test_index_key_cache_registers_without_a_spec() -> None:
    from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackend
    from vllm_torchtpu.models.vllm.kimi_k3.k31 import KimiIndexer

    context: dict = {}
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=context)
    )
    prefix = "language_model.model.layers.3.self_attn.indexer"
    indexer = KimiIndexer(vllm_config, prefix)
    assert context[f"{prefix}.k_cache"] is indexer.k_cache
    assert indexer.k_cache.get_kv_cache_spec(vllm_config) is None
    assert indexer.k_cache.get_attn_backend() is PallasMLAttentionBackend
    with pytest.raises(ValueError, match="Duplicate layer name"):
        KimiIndexer(vllm_config, prefix)


def test_kimi_k31_is_served_by_the_megakernel_only() -> None:
    from vllm_torchtpu.models.vllm.kimi_k3.k31 import KimiK31ForConditionalGeneration

    vllm_config = SimpleNamespace(load_config=SimpleNamespace(load_format="auto"))
    with pytest.raises(NotImplementedError, match="megakernel"):
        KimiK31ForConditionalGeneration(vllm_config=vllm_config)
