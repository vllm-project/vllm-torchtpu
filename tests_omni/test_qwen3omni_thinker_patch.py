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

from types import SimpleNamespace

import pytest
import torch
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
)

from vllm_torchtpu.omni.qwen3omni_thinker_patch import (
    _AUDIO_CHUNK_LENGTHS_PATCHED,
    _is_qwen3_omni_model,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = pytest.mark.omni


def _model_config(
    model_type: str, architectures: list[str], model: str
) -> SimpleNamespace:
    return SimpleNamespace(
        hf_config=SimpleNamespace(model_type=model_type, architectures=architectures),
        model=model,
    )


OMNI_THINKER = _model_config(
    "qwen3_omni_moe_thinker",
    ["Qwen3OmniMoeThinkerForConditionalGeneration"],
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
)


@pytest.mark.parametrize(
    "model_config, expected",
    [
        (None, False),
        (
            _model_config(
                "qwen3_vl", ["Qwen3VLForConditionalGeneration"], "Qwen/Qwen3-VL"
            ),
            False,
        ),
        (_model_config("qwen3_omni_moe", [], "Qwen/Qwen3-Omni-30B-A3B-Instruct"), True),
        (OMNI_THINKER, True),
        # The talker stage config has an empty model_type.
        (_model_config("", ["Qwen3OmniMoeTalkerForConditionalGeneration"], "x"), True),
    ],
    ids=["no_config", "qwen3_vl", "full_model", "thinker_stage", "talker_stage"],
)
def test_is_qwen3_omni_model(model_config, expected):
    assert _is_qwen3_omni_model(model_config) is expected


def test_non_omni_config_is_not_consumed():
    # ``False`` (not ``None``) tells the registry to retry at model_load.
    before = Qwen3OmniMoeAudioEncoder.forward
    assert maybe_patch_qwen3_omni_thinker(None) is False
    assert (
        maybe_patch_qwen3_omni_thinker(_model_config("llama", [], "meta-llama/x"))
        is False
    )
    assert Qwen3OmniMoeAudioEncoder.forward is before


def test_qwen3_omni_audio_chunk_lengths_patch(monkeypatch):
    seen = {}

    def _forward(self, input_features, feature_lens, aftercnn_lens):
        seen["feature_lens_device"] = feature_lens.device.type
        return input_features, feature_lens, aftercnn_lens

    # Upstream forward runs the whole encoder. Both class attributes the patch
    # rebinds are restored after the test.
    monkeypatch.setattr(Qwen3OmniMoeAudioEncoder, "forward", _forward)
    monkeypatch.setattr(
        Qwen3OmniMoeAudioEncoder, "_get_cnn_output_lengths", lambda self, x: x
    )

    assert maybe_patch_qwen3_omni_thinker(OMNI_THINKER) is True
    patched = Qwen3OmniMoeAudioEncoder.forward
    assert patched is not _forward
    assert getattr(patched, _AUDIO_CHUNK_LENGTHS_PATCHED, False)

    class TrackedTensor(torch.Tensor):
        cpu_called: bool = False

        def cpu(self, *args, **kwargs):
            self.cpu_called = True
            return super().cpu(*args, **kwargs)

    enc = SimpleNamespace(conv2d1=SimpleNamespace(weight=torch.empty(0)))
    lens = torch.tensor([800]).as_subclass(TrackedTensor)
    _, out_lens, _ = patched(enc, torch.randn(128, 800), lens, torch.tensor([104]))
    # chunk_num's ceil(feature_lens / window) runs on CPU, where it is exact.
    assert lens.cpu_called is True
    assert seen["feature_lens_device"] == "cpu"
    assert out_lens.device.type == "cpu"
    # ... and the CNN lengths come back on the encoder's device.
    out = Qwen3OmniMoeAudioEncoder._get_cnn_output_lengths(enc, torch.tensor([100]))
    assert out.device == enc.conv2d1.weight.device
    assert torch.equal(out, torch.tensor([100]))

    # Idempotent: a second application keeps the same wrapper.
    assert maybe_patch_qwen3_omni_thinker(OMNI_THINKER) is True
    assert Qwen3OmniMoeAudioEncoder.forward is patched
