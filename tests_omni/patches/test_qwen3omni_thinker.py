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

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoderConfig,
)
from vllm_omni.model_executor.models.qwen3_omni import (
    qwen3_omni_moe_thinker as thinker_mod,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3Omni_VisionTransformer,
    Qwen3OmniMoeAudioEncoder,
)

from tests_omni.omni_test_utils import init_test_weights, use_native_ops_on_cpu
from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_STATIC_BUCKETING_PATCHED,
    _VIT_STATIC_BUCKETING_PATCHED,
    _select_bucket,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = pytest.mark.omni

THINKER = SimpleNamespace(architectures=["Qwen3OmniMoeForConditionalGeneration"])

TINY_VISION_CONFIG = Qwen3OmniMoeVisionEncoderConfig(
    hidden_size=64,
    intermediate_size=128,
    out_hidden_size=32,
    depth=2,
    num_heads=4,
    deepstack_visual_indexes=[0],
    patch_size=2,
    temporal_patch_size=2,
    spatial_merge_size=2,
    in_channels=3,
    image_size=128,
    apply_vit_abs_pos_embed=True,
)


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


def _isolate_thinker_patch_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    for cls in (Qwen3OmniMoeAudioEncoder, Qwen3Omni_VisionTransformer):
        for attr in (
            "forward",
            "_tpu_orig_forward",
            "prepare_encoder_metadata",
            "forward_with_encoder_metadata",
        ):
            monkeypatch.setattr(cls, attr, getattr(cls, attr, None), raising=False)


def test_non_omni_config_is_not_patched() -> None:
    audio_before = Qwen3OmniMoeAudioEncoder.forward
    vit_before = Qwen3Omni_VisionTransformer.forward
    assert maybe_patch_qwen3_omni_thinker(None) is False
    for arch in ("LlamaForCausalLM", "Qwen3OmniMoeThinkerForConditionalGeneration"):
        assert (
            maybe_patch_qwen3_omni_thinker(SimpleNamespace(architectures=[arch]))
            is False
        )
    assert Qwen3OmniMoeAudioEncoder.forward is audio_before
    assert Qwen3Omni_VisionTransformer.forward is vit_before


def test_patch_moves_lengths_to_cpu_and_supports_metadata_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_thinker_patch_targets(monkeypatch)
    cls = Qwen3OmniMoeAudioEncoder

    def _orig_forward(self, x, lens, after=None):
        return x

    monkeypatch.setattr(cls, "forward", _orig_forward)

    assert maybe_patch_qwen3_omni_thinker(THINKER) is True
    patched = cls.forward
    assert getattr(patched, _AUDIO_STATIC_BUCKETING_PATCHED, False)
    assert cls._tpu_orig_forward is _orig_forward
    assert getattr(
        Qwen3Omni_VisionTransformer.forward, _VIT_STATIC_BUCKETING_PATCHED, False
    )

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
    _isolate_thinker_patch_targets(monkeypatch)
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


@pytest.mark.parametrize(
    ("grid", "expected_chunks"),
    [
        # Single image (< 16384 patches) padded to 16384.
        ([[1, 20, 30]], [16384]),
        # Short multi-frame video (<= 16384 patches) padded to 16384.
        ([[4, 20, 30]], [16384]),
        # Multi-frame video (20480 patches > 16384) split into 4 + 1 frames.
        ([[5, 64, 64]], [16384, 16384]),
    ],
)
def test_qwen3_omni_vit_frame_chunking_and_parity(
    omni_ctx: None,
    monkeypatch: pytest.MonkeyPatch,
    grid: list[list[int]],
    expected_chunks: list[int],
) -> None:
    """Verify ViT bucketing, frame chunking, and parity on a real VisionTransformer."""
    _isolate_thinker_patch_targets(monkeypatch)
    assert maybe_patch_qwen3_omni_thinker(THINKER) is True
    assert hasattr(Qwen3Omni_VisionTransformer, "_tpu_orig_forward")

    torch.manual_seed(0)
    vit_patched = Qwen3Omni_VisionTransformer(TINY_VISION_CONFIG, norm_eps=1e-6).eval()
    init_test_weights(vit_patched, seed=0)
    use_native_ops_on_cpu(vit_patched)

    vit_ref = Qwen3Omni_VisionTransformer(TINY_VISION_CONFIG, norm_eps=1e-6).eval()
    vit_ref.load_state_dict(vit_patched.state_dict())
    use_native_ops_on_cpu(vit_ref)
    vit_ref.forward = MethodType(Qwen3Omni_VisionTransformer._tpu_orig_forward, vit_ref)

    seen_shapes: list[int] = []
    hook = vit_patched.blocks[0].register_forward_hook(
        lambda _mod, args, _out: seen_shapes.append(int(args[0].shape[0]))
    )

    total_patches = sum(t * h * w for t, h, w in grid)
    patch_dim = (
        TINY_VISION_CONFIG.in_channels
        * TINY_VISION_CONFIG.temporal_patch_size
        * TINY_VISION_CONFIG.patch_size
        * TINY_VISION_CONFIG.patch_size
    )
    generator = torch.Generator().manual_seed(total_patches)
    x = torch.randn(total_patches, patch_dim, generator=generator)
    grid_tensor = torch.tensor(grid, dtype=torch.int32)

    try:
        with (
            torch.no_grad(),
            patch.object(
                nn.Module,
                "cpu",
                side_effect=AssertionError("Unexpected Module.cpu() host mutation"),
            ),
        ):
            out_patched = vit_patched(x, grid_tensor)
            out_ref = vit_ref(x, grid_tensor)
    finally:
        hook.remove()

    expected_out_dim = TINY_VISION_CONFIG.out_hidden_size * (
        1 + len(TINY_VISION_CONFIG.deepstack_visual_indexes)
    )
    assert out_patched.shape == (
        total_patches // TINY_VISION_CONFIG.spatial_merge_size**2,
        expected_out_dim,
    )
    assert seen_shapes == expected_chunks
    assert torch.allclose(out_patched, out_ref, atol=1e-5, rtol=1e-5)
