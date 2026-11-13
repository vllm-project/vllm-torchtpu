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
"""CPU unit tests for the Qwen3-Omni Talker and Code Predictor TPU patches."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorConfig,
)
from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorBaseModel,
    CodePredictorWrapper,
    CodePredictorWrapperConfig,
    _RMSNorm,
    _RotaryEmbedding,
)
from vllm_omni.model_executor.models.qwen3_omni import (
    qwen3_omni as qwen3_omni_mod,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)

from tests_omni.omni_test_utils import init_test_weights, use_native_ops_on_cpu
from vllm_torchtpu.omni.patches.qwen3omni_talker import (
    _TALKER_PATCHED_ATTR,
    maybe_patch_qwen3_omni_talker,
)

pytestmark = [pytest.mark.omni]

CP_CONFIG = Qwen3OmniMoeTalkerCodePredictorConfig(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    vocab_size=64,
    num_code_groups=4,
)


@pytest.fixture
def restore_talker_patch() -> Iterator[None]:
    rms_own = _RMSNorm.__dict__.get("forward_tpu")
    rotary_own = _RotaryEmbedding.__dict__.get("forward_tpu")
    base_fwd = CodePredictorBaseModel.__dict__.get("forward")
    wrapper_fwd = CodePredictorWrapper.__dict__.get("forward")
    talker_pre = Qwen3OmniMoeForConditionalGeneration.__dict__.get("talker_preprocess")
    try:
        yield
    finally:
        for cls, attr, val in (
            (_RMSNorm, "forward_tpu", rms_own),
            (_RotaryEmbedding, "forward_tpu", rotary_own),
            (CodePredictorBaseModel, "forward", base_fwd),
            (CodePredictorWrapper, "forward", wrapper_fwd),
            (Qwen3OmniMoeForConditionalGeneration, "talker_preprocess", talker_pre),
        ):
            if val is None:
                if attr in cls.__dict__:
                    delattr(cls, attr)
            else:
                setattr(cls, attr, val)


def test_maybe_patch_qwen3_omni_talker_filters_and_is_idempotent(
    restore_talker_patch: None,
) -> None:
    assert maybe_patch_qwen3_omni_talker(None) is False
    assert maybe_patch_qwen3_omni_talker(["LlamaForCausalLM"]) is False

    archs = ["Qwen3OmniMoeForConditionalGeneration"]
    assert maybe_patch_qwen3_omni_talker(archs) is True
    assert maybe_patch_qwen3_omni_talker(archs) is False

    assert _RMSNorm.forward_tpu is _RMSNorm.forward_native
    assert getattr(_RotaryEmbedding.forward_tpu, _TALKER_PATCHED_ATTR, False)
    assert getattr(CodePredictorBaseModel.forward, _TALKER_PATCHED_ATTR, False)
    assert getattr(CodePredictorWrapper.forward, _TALKER_PATCHED_ATTR, False)
    assert getattr(
        Qwen3OmniMoeForConditionalGeneration.talker_preprocess,
        _TALKER_PATCHED_ATTR,
        False,
    )


def test_patched_code_predictor_components_run_on_cpu(
    omni_ctx: None,
    restore_talker_patch: None,
) -> None:
    archs = ["Qwen3OmniMoeForConditionalGeneration"]
    assert maybe_patch_qwen3_omni_talker(archs) is True

    # 1. _RMSNorm.forward_tpu and _RotaryEmbedding.forward_tpu on real classes
    norm = _RMSNorm(hidden_size=32, eps=1e-6)
    x = torch.randn(1, 3, 32)
    assert torch.equal(norm.forward_tpu(x), norm.forward_native(x))

    rotary = _RotaryEmbedding(CP_CONFIG)
    pos_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    cos, sin = rotary.forward_tpu(x, pos_ids)
    assert cos.shape == (1, 3, 8)
    assert sin.shape == (1, 3, 8)

    # 2. Real CodePredictorWrapper with tiny config (both float16 and float32)
    torch.manual_seed(0)
    wrapper = CodePredictorWrapper(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace()),
        cp_config=CP_CONFIG,
        wrapper_config=CodePredictorWrapperConfig(
            use_parallel_embedding=False,
            use_projection=True,
            use_cuda_graphs=False,
            return_proj_buf=True,
        ),
        talker_hidden_size=48,
    ).eval()
    init_test_weights(wrapper, seed=0)
    use_native_ops_on_cpu(wrapper)
    wrapper.set_sampling_params(do_sample=False)

    layer0_code = torch.tensor([[3]], dtype=torch.long)
    layer0_embed = torch.randn(1, 1, 48)
    last_talker_hidden = torch.randn(1, 1, 48)

    with torch.no_grad():
        codes, proj_buf = wrapper(
            layer0_code=layer0_code,
            layer0_embed=layer0_embed,
            last_talker_hidden=last_talker_hidden,
            do_sample=False,
        )
    assert codes.shape == (1, CP_CONFIG.num_code_groups, 1)
    assert proj_buf.shape == (1, CP_CONFIG.num_code_groups + 1, 32)
    assert int(codes[0, 0, 0]) == 3

    # Verify CodePredictorBaseModel fp16 forward skips autocast on CPU/TPU
    base_embeds = torch.randn(1, 2, 32, dtype=torch.float16)
    base_pos = torch.tensor([[0, 1]], dtype=torch.long)
    with (
        patch.object(torch.amp, "autocast", side_effect=AssertionError("no autocast")),
        torch.no_grad(),
    ):
        base_out = wrapper.model.to(torch.float16)(base_embeds, base_pos)
    assert base_out.shape == (1, 2, 32)
    assert base_out.dtype == torch.float16


def test_patched_talker_preprocess_detaches_codes(
    restore_talker_patch: None,
) -> None:
    def _fake_orig(self, input_ids, input_embeds=None, *args, **kwargs):
        return (
            input_ids,
            input_embeds,
            {"codes": {"audio": torch.ones((2, 16), dtype=torch.long)}},
        )

    with patch.object(
        qwen3_omni_mod.Qwen3OmniMoeForConditionalGeneration,
        "talker_preprocess",
        _fake_orig,
    ):
        assert (
            maybe_patch_qwen3_omni_talker(["Qwen3OmniMoeForConditionalGeneration"])
            is True
        )
        dummy_self = SimpleNamespace()
        in_ids = torch.zeros(2, dtype=torch.long)
        ids, embeds, update_dict = (
            Qwen3OmniMoeForConditionalGeneration.talker_preprocess(
                dummy_self, in_ids, torch.ones(2, 8)
            )
        )
        assert torch.equal(ids, in_ids)
        assert embeds.shape == (2, 8)
        assert update_dict["codes"]["audio"].device.type == "cpu"
