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
"""Registry gating and wiring of the Qwen3-Omni Thinker audio encoder patch.

The on-TPU behaviour of the patched encoder (exact window multiples, token
counts, CPU parity) is covered by test_qwen3_omni_audio_encoder_tpu.py; these
tests check the callback contract the patch registry relies on and the two
device moves the patch performs, on CPU with the encoder's methods stubbed.
"""

import importlib.util
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder as VllmQwen3OmniMoeAudioEncoder,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
)

from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_CHUNK_LENGTHS_PATCHED,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = pytest.mark.omni


def _model_config(*architectures: str) -> SimpleNamespace:
    """The only ModelConfig field the callback reads."""
    return SimpleNamespace(architectures=list(architectures))


OMNI_THINKER = _model_config("Qwen3OmniMoeForConditionalGeneration")


def _isolate_encoder_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore both vLLM and vLLM-Omni audio encoder classes after a test."""
    for cls in (VllmQwen3OmniMoeAudioEncoder, Qwen3OmniMoeAudioEncoder):
        monkeypatch.setattr(cls, "forward", cls.forward)
        monkeypatch.setattr(cls, "_get_cnn_output_lengths", cls._get_cnn_output_lengths)
        monkeypatch.setattr(
            cls,
            "_tpu_orig_get_cnn_output_lengths",
            getattr(cls, "_tpu_orig_get_cnn_output_lengths", None),
            raising=False,
        )


def test_non_omni_config_is_not_consumed():
    # ``False`` (not ``None``) tells the registry to retry at model_load. The
    # match is the literal architecture: a substring such as the thinker's own
    # class name does not count.
    before = Qwen3OmniMoeAudioEncoder.forward
    assert maybe_patch_qwen3_omni_thinker(None) is False
    assert maybe_patch_qwen3_omni_thinker(_model_config("LlamaForCausalLM")) is False
    assert (
        maybe_patch_qwen3_omni_thinker(
            _model_config("Qwen3OmniMoeThinkerForConditionalGeneration")
        )
        is False
    )
    assert Qwen3OmniMoeAudioEncoder.forward is before


def test_qwen3_omni_audio_chunk_lengths_patch(monkeypatch):
    _isolate_encoder_classes(monkeypatch)
    seen = {}

    def _forward(self, input_features, feature_lens, aftercnn_lens):
        seen["feature_lens_device"] = feature_lens.device.type
        return input_features, feature_lens, aftercnn_lens

    # Upstream forward runs the whole encoder. Both class attributes the patch
    # rebinds are restored after the test.
    for cls in (VllmQwen3OmniMoeAudioEncoder, Qwen3OmniMoeAudioEncoder):
        monkeypatch.setattr(cls, "forward", _forward)
        monkeypatch.setattr(cls, "_get_cnn_output_lengths", lambda self, x: x)
        monkeypatch.delattr(cls, "_tpu_orig_get_cnn_output_lengths", raising=False)

    assert maybe_patch_qwen3_omni_thinker(OMNI_THINKER) is True
    patched = Qwen3OmniMoeAudioEncoder.forward
    assert patched is not _forward
    assert getattr(patched, _AUDIO_CHUNK_LENGTHS_PATCHED, False)
    assert getattr(
        VllmQwen3OmniMoeAudioEncoder.forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False
    )

    class TrackedTensor(torch.Tensor):
        cpu_called: bool = False

        def cpu(self, *args, **kwargs):
            self.cpu_called = True
            return super().cpu(*args, **kwargs)

    # Put the stub weight on ``meta`` so the device-move check verifies that
    # ``_get_cnn_output_lengths`` actually moves a CPU tensor to the encoder's
    # weight device.
    enc = SimpleNamespace(conv2d1=SimpleNamespace(weight=torch.empty(0, device="meta")))
    lens = torch.tensor([800]).as_subclass(TrackedTensor)
    _, out_lens, _ = patched(enc, torch.randn(128, 800), lens, torch.tensor([104]))
    # chunk_num's ceil(feature_lens / window) runs on CPU, where it is exact.
    assert lens.cpu_called is True
    assert seen["feature_lens_device"] == "cpu"
    assert out_lens.device.type == "cpu"
    # ... and the CNN lengths come back on the encoder's device ("meta").
    out = Qwen3OmniMoeAudioEncoder._get_cnn_output_lengths(enc, torch.tensor([100]))
    assert out.device == enc.conv2d1.weight.device
    assert out.device.type == "meta"
    assert out.shape == (1,)

    # Idempotent: a second application keeps the same wrapper.
    assert maybe_patch_qwen3_omni_thinker(OMNI_THINKER) is True
    assert Qwen3OmniMoeAudioEncoder.forward is patched


def test_qwen3_omni_patch_without_vllm_omni_installed(monkeypatch):
    """Plain-vLLM Thinker-only startup without vllm-omni patches vLLM's encoder."""
    _isolate_encoder_classes(monkeypatch)

    def _vllm_forward(self, input_features, feature_lens, aftercnn_lens):
        return feature_lens

    monkeypatch.setattr(VllmQwen3OmniMoeAudioEncoder, "forward", _vllm_forward)
    monkeypatch.setattr(
        VllmQwen3OmniMoeAudioEncoder, "_get_cnn_output_lengths", lambda self, x: x
    )
    monkeypatch.delattr(
        VllmQwen3OmniMoeAudioEncoder, "_tpu_orig_get_cnn_output_lengths", raising=False
    )

    orig_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **kw: (
            None if name == "vllm_omni" else orig_find_spec(name, *a, **kw)
        ),
    )

    omni_before = Qwen3OmniMoeAudioEncoder.forward
    assert maybe_patch_qwen3_omni_thinker(OMNI_THINKER) is True
    assert getattr(
        VllmQwen3OmniMoeAudioEncoder.forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False
    )
    assert Qwen3OmniMoeAudioEncoder.forward is omni_before
