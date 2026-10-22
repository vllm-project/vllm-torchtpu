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
"""Tests for Gemma4 multimodal patch and mm-encoder-tp-mode support on TPU."""

from dataclasses import dataclass
from types import SimpleNamespace

try:
    import pytest

    pytestmark = pytest.mark.cpu_test
except ImportError:
    pytest = None

from vllm_torchtpu.models.vllm.gemma4_mm_patch import (
    _is_gemma4_mm_model,
    maybe_patch_gemma4_mm,
    patch_gemma4_mm_model_classes,
    refresh_gemma4_mm,
)
from vllm_torchtpu.patch_registry import PATCHES


@dataclass(frozen=True)
class DummyModelInfo:
    supports_multimodal: bool = True
    supports_multimodal_encoder_tp_data: bool = False


def test_is_gemma4_mm_model_detection():
    # None config
    assert not _is_gemma4_mm_model(None)

    # Non-Gemma model
    qwen_config = SimpleNamespace(
        architectures=["Qwen2_5_VLForConditionalGeneration"],
        hf_config=SimpleNamespace(model_type="qwen2_5_vl"),
        model="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    assert not _is_gemma4_mm_model(qwen_config)

    # Text-only Gemma4 model
    text_gemma = SimpleNamespace(
        architectures=["Gemma4ForCausalLM"],
        hf_config=SimpleNamespace(model_type="gemma4"),
        model="google/gemma-4-26b-it",
    )
    assert not _is_gemma4_mm_model(text_gemma)

    # Singular architecture attribute
    singular_mm = SimpleNamespace(
        architecture="Gemma4ForConditionalGeneration",
        model="google/gemma-4-26b-it",
    )
    assert _is_gemma4_mm_model(singular_mm)

    singular_text = SimpleNamespace(
        architecture="Gemma4ForCausalLM",
        model="google/gemma-4-26b-it",
    )
    assert not _is_gemma4_mm_model(singular_text)

    # Gemma4 multimodal model architectures
    mm_gemma = SimpleNamespace(
        architectures=["Gemma4ForConditionalGeneration"],
        hf_config=SimpleNamespace(
            model_type="gemma4",
            architectures=["Gemma4ForConditionalGeneration"],
            vision_config=SimpleNamespace(hidden_size=1152),
        ),
        model="google/gemma-4-26b-it",
    )
    assert _is_gemma4_mm_model(mm_gemma)

    unified_gemma = SimpleNamespace(
        architectures=["Gemma4UnifiedForConditionalGeneration"],
        hf_config=SimpleNamespace(
            model_type="gemma4",
            architectures=["Gemma4UnifiedForConditionalGeneration"],
        ),
        model="google/gemma-4-unified",
    )
    assert _is_gemma4_mm_model(unified_gemma)

    # Config with only hf_config.vision_config
    vision_only_gemma = SimpleNamespace(
        architectures=[],
        hf_config=SimpleNamespace(
            model_type="gemma4",
            vision_config=SimpleNamespace(hidden_size=1152),
        ),
        model="custom-gemma4",
    )
    assert _is_gemma4_mm_model(vision_only_gemma)


def test_patch_gemma4_mm_model_classes():
    # Verify idempotent class patching
    patch_gemma4_mm_model_classes()
    patch_gemma4_mm_model_classes()

    try:
        from vllm.model_executor.models.gemma4_mm import (
            Gemma4ForConditionalGeneration,
        )

        assert getattr(
            Gemma4ForConditionalGeneration, "supports_encoder_tp_data", False
        )
    except ImportError:
        pass


def test_maybe_patch_gemma4_mm_returns_false_for_non_matching_model():
    non_gemma_config = SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        hf_config=SimpleNamespace(model_type="llama"),
        model="meta-llama/Llama-3-8B",
    )
    assert maybe_patch_gemma4_mm(non_gemma_config) is False
    assert maybe_patch_gemma4_mm(None) is False


def test_maybe_patch_gemma4_mm_applies_successfully():
    model_info = DummyModelInfo(supports_multimodal_encoder_tp_data=False)
    multimodal_config = SimpleNamespace(mm_encoder_tp_mode="weights")
    model_config = SimpleNamespace(
        architectures=["Gemma4ForConditionalGeneration"],
        hf_config=SimpleNamespace(model_type="gemma4"),
        model="google/gemma-4-26b-it",
        _model_info=model_info,
        multimodal_config=multimodal_config,
    )

    result = maybe_patch_gemma4_mm(model_config)
    assert result is None

    # Model info should be updated even if frozen
    assert model_config._model_info.supports_multimodal_encoder_tp_data is True


def test_patch_registry_contains_gemma4_mm_patch():
    targets = [p.target for p in PATCHES]
    assert "vllm_torchtpu.models.vllm.gemma4_mm_patch:maybe_patch_gemma4_mm" in targets

    patch = next(
        p
        for p in PATCHES
        if p.target == "vllm_torchtpu.models.vllm.gemma4_mm_patch:maybe_patch_gemma4_mm"
    )
    assert patch.stages == ("platform_activation", "model_load")
    assert patch.model_config is True
    assert (
        patch.refresh == "vllm_torchtpu.models.vllm.gemma4_mm_patch:refresh_gemma4_mm"
    )

    # Verify refresh callable works without error
    refresh_gemma4_mm()


def test_patch_gemma4_layer_config():
    import sys
    import types

    from vllm_torchtpu.models.vllm.gemma4_mm_patch import patch_gemma4_layer_config

    # Mock vllm.transformers_utils.configs.gemma4 if not present in test environment
    if "vllm.transformers_utils.configs.gemma4" not in sys.modules:
        vllm_mock = types.ModuleType("vllm")
        tu_mock = types.ModuleType("vllm.transformers_utils")
        cfg_mock = types.ModuleType("vllm.transformers_utils.configs")
        g4_cfg_mock = types.ModuleType("vllm.transformers_utils.configs.gemma4")
        g4_cfg_mock.gemma4_layer_config = (
            lambda text_config, layer_idx: SimpleNamespace(
                head_dim=256, num_key_value_heads=8
            )
        )
        cfg_mock.gemma4 = g4_cfg_mock
        tu_mock.configs = cfg_mock
        vllm_mock.transformers_utils = tu_mock
        sys.modules["vllm"] = vllm_mock
        sys.modules["vllm.transformers_utils"] = tu_mock
        sys.modules["vllm.transformers_utils.configs"] = cfg_mock
        sys.modules["vllm.transformers_utils.configs.gemma4"] = g4_cfg_mock

    patch_gemma4_layer_config()
    patch_gemma4_layer_config()

    import vllm.transformers_utils.configs.gemma4 as g4_cfg_mod

    text_cfg = SimpleNamespace(
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        head_dim=256,
        num_key_value_heads=8,
        global_head_dim=512,
        num_global_key_value_heads=2,
        attention_k_eq_v=True,
    )
    l0 = g4_cfg_mod.gemma4_layer_config(text_cfg, 0)
    assert l0.head_dim == 256
    assert l0.num_key_value_heads == 8

    l5 = g4_cfg_mod.gemma4_layer_config(text_cfg, 5)
    assert l5.head_dim == 512
    assert l5.num_key_value_heads == 2

    # Also test Gemma4Attention __init__ patching
    g4_mod_mock = types.ModuleType("vllm.model_executor.models.gemma4")
    g4_mod_mock.extract_layer_index = lambda prefix: 5

    class MockAttention:
        def __init__(
            self,
            config,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            *args,
            **kwargs,
        ):
            self.head_dim = head_dim
            self.num_kv_heads = num_kv_heads

    g4_mod_mock.Gemma4Attention = MockAttention
    sys.modules["vllm.model_executor.models.gemma4"] = g4_mod_mock
    if "vllm.model_executor.models" in sys.modules:
        setattr(sys.modules["vllm.model_executor.models"], "gemma4", g4_mod_mock)  # noqa: B010

    patch_gemma4_layer_config()
    attn = g4_mod_mock.Gemma4Attention(
        text_cfg, 1024, 8, 8, 256, prefix="model.layers.5.self_attn"
    )
    assert attn.head_dim == 512
    assert attn.num_kv_heads == 2
