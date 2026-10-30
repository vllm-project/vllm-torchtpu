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
"""Registration, quantization config resolution, and E2E tests for DeepSeek-V4.1."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models import registry as vllm_registry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.models.deepseek_v41 import quant_config as upstream_qcfg

from tests.models.vllm.deepseek_v41.test_model_dspark import (
    _make_vllm_config,
    _patch_submodules,
    _zero_hc_params,
    test_model_and_dspark_tpu_forward,
)
from vllm_torchtpu.layers.adapter import quantization as tpu_quant
from vllm_torchtpu.layers.adapter.quantization.deepseek_v4_fp8 import (
    VllmDeepseekV4Fp8Config,
)
from vllm_torchtpu.models import vllm as vllm_models
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    DeepseekV41ForCausalLM,
    DSparkDeepseekV41ForCausalLM,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import model as model_mod


def _registry_mc(vcfg: SimpleNamespace) -> ModelConfig:
    mc = vcfg.model_config
    mc.model_impl = "auto"
    mc.runner_type = "generate"
    mc.convert_type = "none"
    return cast(ModelConfig, mc)


def test_register_models_and_config_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify register_models registers DeepSeek-V4.1 and DSparkV41 architectures."""
    assert "DeepseekV41ForCausalLM" in vllm_registry._VLLM_MODELS
    assert "DSparkV41DraftModel" in vllm_registry._SPECULATIVE_DECODING_MODELS
    vllm_models.register_models()
    mc = _registry_mc(_make_vllm_config())
    cls_main, _ = ModelRegistry.resolve_model_cls(["DeepseekV41ForCausalLM"], mc)
    cls_draft, _ = ModelRegistry.resolve_model_cls(["DSparkV41DraftModel"], mc)
    assert cls_main is DeepseekV41ForCausalLM
    assert cls_draft is DSparkDeepseekV41ForCausalLM

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
    hf_q = {"quant_method": "fp8", "weight_block_size": [32, 32], "expert_dtype": "fp4"}
    hf_c = SimpleNamespace(model_type="deepseek_v41", quantization_config=dict(hf_q))
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
    assert dsv41_mc.hf_config.quantization_config["quant_method"] == "deepseek_v4_fp8"
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


def test_registered_e2e_smoke_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve registered DeepseekV41ForCausalLM and DSparkV41DraftModel and run CPU."""
    _patch_submodules(monkeypatch)
    monkeypatch.setattr(
        model_mod.upstream.EngramLayout,
        "from_config",
        classmethod(lambda cls, cfg: None),
    )
    monkeypatch.setattr(
        model_mod.upstream,
        "is_forward_context_available",
        lambda: False,
    )
    vllm_models.register_models()
    vcfg = _make_vllm_config(
        spec_method="dspark",
        num_spec=2,
        enable_prefix_caching=True,
        with_engram=False,
    )
    mc = _registry_mc(vcfg)
    cls_main, _ = ModelRegistry.resolve_model_cls(["DeepseekV41ForCausalLM"], mc)
    cls_draft, _ = ModelRegistry.resolve_model_cls(["DSparkV41DraftModel"], mc)
    lm = cast(DeepseekV41ForCausalLM, cls_main(vllm_config=cast(VllmConfig, vcfg)))
    dm = cast(
        DSparkDeepseekV41ForCausalLM, cls_draft(vllm_config=cast(VllmConfig, vcfg))
    )
    dm.model.embed_tokens = lm.model.embed_tokens
    _zero_hc_params(lm.model.layers)
    _zero_hc_params(dm.model.layers)
    lm.process_weights_after_loading()
    dm.process_weights_after_loading()
    lm.set_aux_hidden_state_layers((2, 3))
    ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    pos = torch.arange(4, dtype=torch.int32)
    out_main, aux = cast(tuple[torch.Tensor, list[torch.Tensor]], lm(ids, pos))
    out_draft = cast(torch.Tensor, dm(ids, pos))
    assert out_main.shape == (4, 16) and len(aux) == 2
    assert out_draft.shape == (4, 16)


@pytest.mark.tpu_test
def test_registered_e2e_smoke_tpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # pragma: no cover
    """Resolve registered DeepseekV41ForCausalLM and DSparkV41DraftModel on TPU7x."""
    vllm_models.register_models()
    mc = _registry_mc(_make_vllm_config())
    cls_main, _ = ModelRegistry.resolve_model_cls(["DeepseekV41ForCausalLM"], mc)
    cls_draft, _ = ModelRegistry.resolve_model_cls(["DSparkV41DraftModel"], mc)
    assert cls_main is DeepseekV41ForCausalLM
    assert cls_draft is DSparkDeepseekV41ForCausalLM
    test_model_and_dspark_tpu_forward(monkeypatch)
