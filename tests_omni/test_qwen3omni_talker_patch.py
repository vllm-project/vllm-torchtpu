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
"""Registry gating and wiring of the Qwen3-Omni Talker and Code Predictor patches."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm_omni.model_executor.models.common import qwen3_code_predictor as cp
from vllm_omni.model_executor.models.qwen3_omni import (
    qwen3_omni as qwen3_omni_mod,
)

from vllm_torchtpu.omni.patches.qwen3omni_talker import (
    _TALKER_PATCHED_ATTR,
    maybe_patch_qwen3_omni_talker,
)

pytestmark = pytest.mark.omni


def _model_config(*architectures: str) -> SimpleNamespace:
    """The only ModelConfig field the callback reads."""
    return SimpleNamespace(architectures=list(architectures))


OMNI_TALKER = _model_config("Qwen3OmniMoeForConditionalGeneration")


def test_non_omni_config_is_not_consumed():
    before = cp.CodePredictorWrapper.forward
    assert maybe_patch_qwen3_omni_talker(None) is False
    assert maybe_patch_qwen3_omni_talker(_model_config("LlamaForCausalLM")) is False
    assert (
        maybe_patch_qwen3_omni_talker(
            _model_config("Qwen3OmniMoeTalkerForConditionalGeneration")
        )
        is False
    )
    assert cp.CodePredictorWrapper.forward is before


def test_maybe_patch_qwen3_omni_talker_code_predictor_and_preprocess(monkeypatch):
    class DummyRMSNorm:
        def forward_native(self, x):
            return x * 2.0

    class DummyRotary:
        def __init__(self):
            self.cos_cached = torch.arange(8, dtype=torch.float32).unsqueeze(-1)
            self.sin_cached = -torch.arange(8, dtype=torch.float32).unsqueeze(-1)

    class DummyLayer(nn.Module):
        def forward(self, hidden_states, position_embeddings):
            cos, sin = position_embeddings
            return hidden_states + cos + sin

    class DummyBaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = lambda x, pos: (
                torch.ones_like(x),
                torch.zeros_like(x),
            )
            self.layers = nn.ModuleList([DummyLayer()])
            self.norm = nn.Identity()

    class DummyWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.ModuleList([nn.Linear(4, 4)])
            self.model = nn.Module()
            self.model.codec_embedding = nn.ModuleList([nn.Embedding(8, 4)])
            self.model.param = nn.Parameter(torch.ones(1))
            self._compiled_model_fwd = None

        def forward(
            self,
            layer0_code,
            layer0_embed,
            last_talker_hidden,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
            generator=None,
            generators=None,
        ):
            return layer0_code + 1, layer0_embed + last_talker_hidden

    class DummyOmniModel:
        def talker_preprocess(self, input_ids, input_embeds=None, *args, **kwargs):
            assert input_ids is not None
            return (
                input_ids,
                input_embeds,
                {"codes": {"audio": torch.ones((input_ids.shape[0], 16))}},
            )

    monkeypatch.setattr(cp, "_RMSNorm", DummyRMSNorm)
    monkeypatch.setattr(cp, "_RotaryEmbedding", DummyRotary)
    monkeypatch.setattr(cp, "CodePredictorBaseModel", DummyBaseModel)
    monkeypatch.setattr(cp, "CodePredictorWrapper", DummyWrapper)
    monkeypatch.setattr(
        qwen3_omni_mod, "Qwen3OmniMoeForConditionalGeneration", DummyOmniModel
    )

    assert maybe_patch_qwen3_omni_talker(OMNI_TALKER) is True
    patched_wrapper_fwd = DummyWrapper.forward
    assert getattr(patched_wrapper_fwd, _TALKER_PATCHED_ATTR, False)

    # 1. _RMSNorm.forward_tpu delegates to forward_native
    norm = DummyRMSNorm()
    x = torch.tensor([1.0, 2.0])
    assert torch.equal(norm.forward_tpu(x), torch.tensor([2.0, 4.0]))

    # 2. _RotaryEmbedding.forward_tpu indexes cos_cached / sin_cached
    rot = DummyRotary()
    cos, sin = rot.forward_tpu(
        torch.zeros(2, 1, dtype=torch.bfloat16), torch.tensor([1, 3])
    )
    assert cos.dtype == torch.bfloat16 and sin.dtype == torch.bfloat16
    assert torch.equal(cos.float().squeeze(-1), torch.tensor([1.0, 3.0]))
    assert torch.equal(sin.float().squeeze(-1), torch.tensor([-1.0, -3.0]))

    # 3. CodePredictorBaseModel.forward runs without unsupported autocast
    base = DummyBaseModel()
    out_base = base(torch.zeros(2, 4, dtype=torch.float16), torch.tensor([0, 1]))
    assert out_base.dtype == torch.float16
    assert torch.equal(out_base.float(), torch.ones(2, 4))

    # 4. CodePredictorWrapper.forward preserves CPU behavior
    wrapper = DummyWrapper()
    codes, summed = wrapper(torch.tensor([[1]]), torch.ones(1, 4), torch.ones(1, 4))
    assert torch.equal(codes, torch.tensor([[2]]))
    assert torch.equal(summed, torch.full((1, 4), 2.0))

    # 5. talker_preprocess synthesizes input_ids when only input_embeds is passed
    omni = DummyOmniModel()
    embeds = torch.randn(3, 8)
    ids, out_embeds, upd = omni.talker_preprocess(None, embeds)
    assert ids.shape == (3,)
    assert out_embeds is embeds
    assert upd["codes"]["audio"].device.type == "cpu"

    # Idempotent: a second application keeps the same wrapper.
    assert maybe_patch_qwen3_omni_talker(OMNI_TALKER) is True
    assert DummyWrapper.forward is patched_wrapper_fwd
