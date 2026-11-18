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
"""Registration and FP8 quantization config resolution tests for DeepSeek-V4.1."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models import registry as vllm_registry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.models.deepseek_v41 import quant_config as upstream_qcfg
from vllm_torchtpu.layers.adapter import quantization as tpu_quant
from vllm_torchtpu.layers.adapter.quantization.deepseek_v4_fp8 import (
    VllmDeepseekV4Fp8Config,
)
from vllm_torchtpu.models import vllm as vllm_models


@contextmanager
def _preserve_registry_state():
    """Restore global vLLM ModelRegistry and MODELS_CONFIG_MAP after test."""
    saved_models = dict(ModelRegistry.models)
    saved_configs = dict(MODELS_CONFIG_MAP)
    try:
        yield
    finally:
        ModelRegistry.models.clear()
        ModelRegistry.models.update(saved_models)
        MODELS_CONFIG_MAP.clear()
        MODELS_CONFIG_MAP.update(saved_configs)


def _has_real_module(module_path: str) -> bool:
    """Check module existence prefix-by-prefix without raising on missing parent."""
    parts = module_path.split(".")
    for depth in range(1, len(parts) + 1):
        if importlib.util.find_spec(".".join(parts[:depth])) is None:
            return False
    return True


def _ensure_target_module_cls(
    monkeypatch: pytest.MonkeyPatch,
    module_path: str,
    class_name: str,
) -> type[nn.Module]:
    """Return real model class if merged, else install a stub module for resolution."""
    if _has_real_module(module_path):
        mod = importlib.import_module(module_path)
        return cast(type[nn.Module], getattr(mod, class_name))

    parts = module_path.split(".")
    for depth in range(1, len(parts)):
        pkg_name = ".".join(parts[:depth])
        if pkg_name not in sys.modules:
            pkg = ModuleType(pkg_name)
            pkg.__path__ = []
            monkeypatch.setitem(sys.modules, pkg_name, pkg)

    stub_cls = type(class_name, (nn.Module,), {})
    stub_mod = ModuleType(module_path)
    setattr(stub_mod, class_name, stub_cls)
    monkeypatch.setitem(sys.modules, module_path, stub_mod)
    return stub_cls


def _dummy_model_config() -> ModelConfig:
    return cast(
        ModelConfig,
        SimpleNamespace(
            model_impl="auto",
            runner_type="generate",
            convert_type="none",
        ),
    )


@pytest.mark.parametrize(
    ("arch", "module_path", "class_name", "in_spec_decode"),
    [
        (
            "DeepseekV41ForCausalLM",
            "vllm_torchtpu.models.vllm.deepseek_v41.model",
            "DeepseekV41ForCausalLM",
            False,
        ),
        (
            "DSparkV41DraftModel",
            "vllm_torchtpu.models.vllm.deepseek_v41.dspark",
            "DSparkDeepseekV41ForCausalLM",
            True,
        ),
    ],
    ids=["target_causal_lm", "dspark_draft_model"],
)
def test_register_models_resolves_deepseek_v41_architectures(
    monkeypatch: pytest.MonkeyPatch,
    arch: str,
    module_path: str,
    class_name: str,
    in_spec_decode: bool,
) -> None:
    """Verify register_models registers DeepSeek-V4.1 target and DSpark models."""
    upstream_table = (
        vllm_registry._SPECULATIVE_DECODING_MODELS
        if in_spec_decode
        else vllm_registry._VLLM_MODELS
    )
    assert arch in upstream_table
    expected_cls = _ensure_target_module_cls(monkeypatch, module_path, class_name)

    with _preserve_registry_state():
        vllm_models.register_models()
        resolved_cls, resolved_arch = ModelRegistry.resolve_model_cls(
            [arch], _dummy_model_config()
        )
        assert resolved_arch == arch
        assert resolved_cls is expected_cls


def test_config_hooks_refresh_deepseek_v41_arch_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _TPUDeepseekV41Config refreshes model_arch_config after FP8 rewrite."""
    with _preserve_registry_state():
        vllm_models.register_models()

        called: list[tuple[object, object]] = []
        monkeypatch.setattr(
            vllm_models,
            "update_gqa_head_padding_config",
            lambda m_c, p_c: called.append((m_c, p_c)),
        )
        qwen_cfg = MODELS_CONFIG_MAP["Qwen2ForCausalLM"]
        dummy_vcfg = cast(
            VllmConfig, SimpleNamespace(model_config="mc", parallel_config="pc")
        )
        qwen_cfg.verify_and_update_config(dummy_vcfg)
        assert called == [("mc", "pc")]

        dsv41_cfg = MODELS_CONFIG_MAP["DeepseekV41ForCausalLM"]
        hf_q = {
            "quant_method": "fp8",
            "weight_block_size": [32, 32],
            "expert_dtype": "fp4",
        }
        hf_c = SimpleNamespace(
            model_type="deepseek_v41", quantization_config=dict(hf_q)
        )
        stale_arch = SimpleNamespace(quantization_config=dict(hf_q))
        dsv41_mc = SimpleNamespace(
            hf_config=hf_c,
            hf_text_config=hf_c,
            model_arch_config=stale_arch,
            get_model_arch_config=lambda: SimpleNamespace(
                quantization_config=dict(hf_c.quantization_config)
            ),
        )
        dsv41_cfg.verify_and_update_model_config(cast(ModelConfig, dsv41_mc))
        assert (
            dsv41_mc.hf_config.quantization_config["quant_method"] == "deepseek_v4_fp8"
        )
        assert (
            dsv41_mc.model_arch_config.quantization_config["quant_method"]
            == "deepseek_v4_fp8"
        )


def test_deepseek_v4_fp8_quantization_resolution() -> None:
    """Verify deepseek_v41 fp8 config overrides to deepseek_v4_fp8 on TPU."""
    hf_quant_cfg = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
    }
    hf_cfg = SimpleNamespace(
        model_type="deepseek_v41",
        quantization_config=hf_quant_cfg,
    )
    resolved_method = upstream_qcfg.DeepseekV4FP8Config.override_quantization_method(
        hf_quant_cfg, user_quant=None, hf_config=cast(Any, hf_cfg)
    )
    assert resolved_method == "deepseek_v4_fp8"
    vcfg = SimpleNamespace(
        model_config=SimpleNamespace(
            model="deepseek-v41",
            revision=None,
            dtype=torch.bfloat16,
            quantization=resolved_method,
            quantization_config=None,
            hf_config=hf_cfg,
        ),
        load_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
        additional_config={},
    )
    q_cfg = tpu_quant.get_tpu_quantization_config(cast(VllmConfig, vcfg))
    assert isinstance(q_cfg, VllmDeepseekV4Fp8Config)
