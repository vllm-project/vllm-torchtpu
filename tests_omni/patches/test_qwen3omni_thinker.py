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
"""Unit tests for src/vllm_torchtpu/omni/patches/qwen3omni_thinker.py."""

from types import SimpleNamespace

import pytest
import torch
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
)

from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_CHUNK_LENGTHS_PATCHED,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = pytest.mark.omni

THINKER = SimpleNamespace(architectures=["Qwen3OmniMoeForConditionalGeneration"])


def test_non_omni_config_is_not_patched() -> None:
    before = Qwen3OmniMoeAudioEncoder.forward
    assert maybe_patch_qwen3_omni_thinker(None) is False
    for arch in ("LlamaForCausalLM", "Qwen3OmniMoeThinkerForConditionalGeneration"):
        assert (
            maybe_patch_qwen3_omni_thinker(SimpleNamespace(architectures=[arch]))
            is False
        )
    assert Qwen3OmniMoeAudioEncoder.forward is before


def test_patch_moves_chunk_lengths_to_cpu_and_cnn_lengths_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cls = Qwen3OmniMoeAudioEncoder
    monkeypatch.setattr(cls, "forward", lambda self, x, lens, after: lens)
    monkeypatch.setattr(cls, "_get_cnn_output_lengths", lambda self, x: x)

    assert maybe_patch_qwen3_omni_thinker(THINKER) is True
    patched = cls.forward
    assert getattr(patched, _AUDIO_CHUNK_LENGTHS_PATCHED, False)

    class TrackedTensor(torch.Tensor):
        cpu_called: bool = False

        def cpu(self, *args, **kwargs):
            self.cpu_called = True
            return super().cpu(*args, **kwargs)

    enc = SimpleNamespace(conv2d1=SimpleNamespace(weight=torch.empty(0, device="meta")))
    lens = torch.tensor([800]).as_subclass(TrackedTensor)
    out_lens = patched(enc, torch.randn(128, 800), lens, torch.tensor([104]))
    assert lens.cpu_called is True
    assert out_lens.device.type == "cpu"

    cnn_out = cls._get_cnn_output_lengths(enc, torch.tensor([100]))
    assert cnn_out.device.type == "meta"

    # Idempotent on repeated calls.
    assert maybe_patch_qwen3_omni_thinker(THINKER) is False
    assert cls.forward is patched
