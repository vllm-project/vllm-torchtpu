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
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm_omni.model_executor.models.qwen3_omni import (
    qwen3_omni_moe_thinker as thinker_mod,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
)

from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_STATIC_BUCKETING_PATCHED,
    _select_bucket,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = pytest.mark.omni

THINKER = SimpleNamespace(architectures=["Qwen3OmniMoeForConditionalGeneration"])


class _DummyLinear(nn.Module):
    def forward(self, x):
        return x, None


class _DummyEncoderLayer(nn.Module):
    def __init__(self, seen_seq_lens: list[int] | None = None):
        super().__init__()
        self.seen_seq_lens = seen_seq_lens

    def forward(self, hs, cu_seqlens, max_seqlen=None):
        assert cu_seqlens.shape == (128,)
        assert max_seqlen is None
        if self.seen_seq_lens is not None:
            self.seen_seq_lens.append(int(hs.shape[0]))
        return hs


class _DummyAudioEncoder(nn.Module):
    num_mel_bins = 16
    n_window = 50
    n_window_infer = 400
    conv_chunksize = 16

    def __init__(self, seen_seq_lens: list[int] | None = None):
        super().__init__()
        self.conv2d1 = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)
        self.conv_out = _DummyLinear()
        self.positional_embedding = SimpleNamespace(
            positional_embedding=torch.zeros(64, 8)
        )
        self.layers = nn.ModuleList([_DummyEncoderLayer(seen_seq_lens)])
        self.ln_post = nn.Identity()
        self.proj1 = _DummyLinear()
        self.act = F.gelu
        self.proj2 = _DummyLinear()


def _isolate_audio_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    cls = Qwen3OmniMoeAudioEncoder
    for attr in (
        "forward",
        "_tpu_orig_forward",
        "prepare_encoder_metadata",
        "forward_with_encoder_metadata",
    ):
        monkeypatch.setattr(cls, attr, getattr(cls, attr, None), raising=False)


def test_non_omni_config_is_not_patched() -> None:
    before = Qwen3OmniMoeAudioEncoder.forward
    assert maybe_patch_qwen3_omni_thinker(None) is False
    for arch in ("LlamaForCausalLM", "Qwen3OmniMoeThinkerForConditionalGeneration"):
        assert (
            maybe_patch_qwen3_omni_thinker(SimpleNamespace(architectures=[arch]))
            is False
        )
    assert Qwen3OmniMoeAudioEncoder.forward is before


def test_patch_moves_lengths_to_cpu_and_supports_metadata_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_audio_encoder(monkeypatch)
    cls = Qwen3OmniMoeAudioEncoder

    def _orig_forward(self, x, lens, after=None):
        return x

    monkeypatch.setattr(cls, "forward", _orig_forward)

    assert maybe_patch_qwen3_omni_thinker(THINKER) is True
    patched = cls.forward
    assert getattr(patched, _AUDIO_STATIC_BUCKETING_PATCHED, False)
    assert cls._tpu_orig_forward is _orig_forward

    class TrackedTensor(torch.Tensor):
        cpu_called: bool = False

        def cpu(self, *args, **kwargs):
            self.cpu_called = True
            return super().cpu(*args, **kwargs)

    seen_seq_lens: list[int] = []
    enc = _DummyAudioEncoder(seen_seq_lens)
    feat_lens = torch.tensor([800]).as_subclass(TrackedTensor)
    cnn_lens = torch.tensor([104]).as_subclass(TrackedTensor)
    out = patched(enc, torch.zeros(16, 800), feat_lens, cnn_lens)
    assert feat_lens.cpu_called is True
    assert cnn_lens.cpu_called is True
    assert out.shape == (104, 8)

    # Also test the v0.33.0rc1 (#2910) prepare_encoder_metadata +
    # forward_with_encoder_metadata call pattern used by _process_audio_input.
    metadata = cls.prepare_encoder_metadata(enc, [220, 800])
    assert metadata == {"feature_lens": [220, 800], "aftercnn_lens": [29, 104]}
    out_meta = cls.forward_with_encoder_metadata(enc, torch.zeros(16, 1020), metadata)
    splits = out_meta.split(metadata["aftercnn_lens"])
    assert [s.shape for s in splits] == [(29, 8), (104, 8)]

    # Idempotent on repeated calls.
    assert maybe_patch_qwen3_omni_thinker(THINKER) is False
    assert cls.forward is patched
    assert cls._tpu_orig_forward is _orig_forward


def test_audio_encoder_static_bucketing(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_layer_seq_lens: list[int] = []

    class DummyAudioEncoder(_DummyAudioEncoder):
        def __init__(self):
            super().__init__(seen_layer_seq_lens)

    monkeypatch.setattr(thinker_mod, "Qwen3OmniMoeAudioEncoder", DummyAudioEncoder)
    assert maybe_patch_qwen3_omni_thinker(THINKER) is True

    with patch.object(
        nn.Module,
        "cpu",
        side_effect=AssertionError("Unexpected Module.cpu() host mutation"),
    ):
        enc = DummyAudioEncoder()
        for flen, alen in ((220, 29), (310, 41), (670, 87), (800, 104)):
            out = enc(
                torch.zeros(16, flen),
                torch.tensor([flen], dtype=torch.long),
                torch.tensor([alen], dtype=torch.long),
            )
            assert out.shape == (alen, 8)
        assert seen_layer_seq_lens == [512, 512, 512, 512]
        assert _select_bucket(5, (8, 16), align=8) == 8
        assert _select_bucket(19, (8, 16), align=8) == 24
