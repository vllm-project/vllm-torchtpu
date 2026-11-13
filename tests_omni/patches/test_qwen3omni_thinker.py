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
"""CPU unit tests for the Qwen3-Omni Thinker audio encoder patch."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)

from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_CHUNK_BUCKETS,
    _AUDIO_CHUNK_LENGTHS_PATCHED,
    _select_bucket,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = pytest.mark.omni

THINKER_ARCHS = ["Qwen3OmniMoeForConditionalGeneration"]


@pytest.fixture(autouse=True)
def _isolate_thinker_patch_targets() -> Iterator[None]:
    """Snapshot and restore Qwen3OmniMoeAudioEncoder attributes across each test."""
    cls = Qwen3OmniMoeAudioEncoder
    saved = {
        attr: cls.__dict__.get(attr)
        for attr in (
            "forward",
            "prepare_encoder_metadata",
            "forward_with_encoder_metadata",
            "_tpu_orig_forward",
        )
    }
    try:
        yield
    finally:
        for attr, val in saved.items():
            if val is None:
                if attr in cls.__dict__:
                    delattr(cls, attr)
            else:
                setattr(cls, attr, val)


class _DummyPosEmb(nn.Module):
    def __init__(self, max_len: int = 64, d_model: int = 8) -> None:
        super().__init__()
        self.register_buffer("positional_embedding", torch.zeros(max_len, d_model))


class _DummyReplicatedLinear(nn.Module):
    """Minimal ReplicatedLinear stand-in returning ``(output, None)``."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.linear(x), None


def _make_dummy_audio_encoder(
    seen_conv_batch_sizes: list[int] | None = None,
    seen_layer_seq_lens: list[int] | None = None,
    seen_cu_seqlens: list[list[int]] | None = None,
) -> nn.Module:
    class _Layer(nn.Module):
        def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            max_seqlen: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del max_seqlen
            if seen_layer_seq_lens is not None:
                seen_layer_seq_lens.append(int(hidden_states.shape[0]))
            if seen_cu_seqlens is not None:
                seen_cu_seqlens.append(cu_seqlens.tolist())
            return hidden_states

    class _Conv1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if seen_conv_batch_sizes is not None:
                seen_conv_batch_sizes.append(int(x.shape[0]))
            b, c, f, t = x.shape
            return x[:, :, : max(1, f // 2), :13]

    class _DummyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_window = 50
            self.n_window_infer = 400
            self.output_dim = 8
            self.conv2d1 = _Conv1()
            self.conv2d2 = nn.Identity()
            self.conv2d3 = nn.Identity()
            self.conv_out = _DummyReplicatedLinear(8, 8)
            self.positional_embedding = _DummyPosEmb(max_len=64, d_model=8)
            self.layers = nn.ModuleList([_Layer(), _Layer()])
            self.ln_post = nn.Identity()
            self.proj1 = _DummyReplicatedLinear(8, 8)
            self.act = nn.Identity()
            self.proj2 = _DummyReplicatedLinear(8, 8)

        def _get_cnn_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
            return _get_feat_extract_output_lengths(input_lengths)

    return _DummyEncoder()


def test_non_omni_config_is_not_patched() -> None:
    before = Qwen3OmniMoeAudioEncoder.forward
    assert maybe_patch_qwen3_omni_thinker(None) is False
    assert maybe_patch_qwen3_omni_thinker([]) is False
    for arch in ("LlamaForCausalLM", "Qwen3OmniMoeThinkerForConditionalGeneration"):
        assert maybe_patch_qwen3_omni_thinker([arch]) is False
    assert Qwen3OmniMoeAudioEncoder.forward is before


def test_patch_moves_lengths_to_cpu_and_supports_metadata_api() -> None:
    cls = Qwen3OmniMoeAudioEncoder

    assert maybe_patch_qwen3_omni_thinker(THINKER_ARCHS) is True
    patched_forward = cls.forward
    assert getattr(patched_forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False)
    assert hasattr(cls, "_tpu_orig_forward")

    class TrackedTensor(torch.Tensor):
        cpu_called: bool = False

        def cpu(self, *args, **kwargs):
            TrackedTensor.cpu_called = True
            return super().cpu(*args, **kwargs)

    enc = _make_dummy_audio_encoder()
    lens = torch.tensor([800]).as_subclass(TrackedTensor)
    out = patched_forward(enc, torch.zeros(16, 800), lens, torch.tensor([104]))
    assert TrackedTensor.cpu_called is True
    assert out.shape == (104, 8)

    # Also verify 0.33.0rc1 entrypoint: prepare_encoder_metadata +
    # forward_with_encoder_metadata.
    metadata = cls.prepare_encoder_metadata(enc, [800, 250])
    assert metadata["aftercnn_lens"] == [104, 33]
    out_meta = cls.forward_with_encoder_metadata(enc, torch.zeros(16, 1050), metadata)
    assert out_meta.shape == (137, 8)

    # Idempotent on repeated calls.
    assert maybe_patch_qwen3_omni_thinker(THINKER_ARCHS) is False
    assert cls.forward is patched_forward


def test_audio_encoder_static_bucketing() -> None:
    """Verify (8, 16, 32)-window bucketing, cu_seqlens padding, and >32 fallback."""
    assert maybe_patch_qwen3_omni_thinker(THINKER_ARCHS) is True
    cls = Qwen3OmniMoeAudioEncoder

    seen_conv_batch_sizes: list[int] = []
    seen_layer_seq_lens: list[int] = []
    seen_cu_seqlens: list[list[int]] = []
    enc = _make_dummy_audio_encoder(
        seen_conv_batch_sizes=seen_conv_batch_sizes,
        seen_layer_seq_lens=seen_layer_seq_lens,
        seen_cu_seqlens=seen_cu_seqlens,
    )

    with patch.object(
        nn.Module,
        "cpu",
        side_effect=AssertionError("Unexpected Module.cpu() host mutation"),
    ):
        # 3 clips in one call spanning all 3 static buckets (8, 16, 32):
        # - 250 frames (3 windows -> 33 tokens) -> 8-window bucket (104 tokens)
        # - 900 frames (9 windows -> 117 tokens) -> 16-window bucket (208 tokens)
        # - 2000 frames (20 windows -> 260 tokens) -> 32-window bucket (416 tokens)
        feats = torch.zeros(16, 3150)
        lens = torch.tensor([250, 900, 2000], dtype=torch.long)
        aftercnn = _get_feat_extract_output_lengths(lens)
        out = cls.forward(enc, feats, lens, aftercnn)

        assert out.shape == (33 + 117 + 260, 8)
        assert seen_conv_batch_sizes == [8, 16, 32]
        assert seen_layer_seq_lens == [104, 104, 208, 208, 416, 416]
        for cu in seen_cu_seqlens:
            assert len(cu) == 128
        # Clip 1 (33 valid tokens, 8-window bucket = 104): [0, 33, 104, 105, ...]
        assert seen_cu_seqlens[0][:4] == [0, 33, 104, 105]
        # Clip 2 (117 valid tokens across 4+4+1 windows, 16-window bucket = 208):
        # [0, 52, 104, 117, 208, 209, ...]
        assert seen_cu_seqlens[2][:6] == [0, 52, 104, 117, 208, 209]
        # Clip 3 (260 valid tokens across 5x4 windows, 32-window bucket = 416):
        # [0, 52, 104, 156, 208, 260, 416, 417, ...]
        assert seen_cu_seqlens[4][:8] == [0, 52, 104, 156, 208, 260, 416, 417]

        assert _AUDIO_CHUNK_BUCKETS == (8, 16, 32)
        assert _select_bucket(5, _AUDIO_CHUNK_BUCKETS, align=8) == 8
        assert _select_bucket(9, _AUDIO_CHUNK_BUCKETS, align=8) == 16
        assert _select_bucket(20, _AUDIO_CHUNK_BUCKETS, align=8) == 32
        assert _select_bucket(33, _AUDIO_CHUNK_BUCKETS, align=8) == 40
