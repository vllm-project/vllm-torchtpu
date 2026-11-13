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
"""Component-wise TPU vs CPU tests for the Qwen3-Omni thinker audio encoder."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
import torch
from transformers import Qwen3OmniMoeAudioEncoderConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder as HfQwen3OmniMoeAudioEncoder,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)

from tests_omni.omni_test_utils import assert_all_loaded, assert_parity
from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_CHUNK_LENGTHS_PATCHED,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = [pytest.mark.omni]

# Production chunking geometry (num_mel_bins 128, n_window 50 -> 100-frame
# windows, n_window_infer 800, three stride-2 convolutions) on a tiny encoder.
AUDIO_CONFIG = Qwen3OmniMoeAudioEncoderConfig(
    d_model=256,
    encoder_layers=1,
    encoder_attention_heads=4,
    encoder_ffn_dim=512,
    output_dim=128,
    downsample_hidden_size=32,
)

# 800 frames (8 s) is the exact 100-frame multiple the patch fixes; 801 takes
# the partial-window branch across two attention segments.
PARITY_LENGTHS = [800, 801]

# (max relative L2, max abs) vs float32 HF CPU reference (~2.5x measured on TPU v6e).
ATTN_LIMITS = (1.0e-2, 3.0e-3)
LAYER_LIMITS = (8.0e-3, 5.5e-2)
END_TO_END_LIMITS = (1.5e-2, 3.5e-3)


def _make_mel_features(n_frames: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(AUDIO_CONFIG.num_mel_bins, n_frames, generator=generator)


@pytest.fixture(scope="module")
def encoders(
    omni_ctx: None,
) -> Iterator[
    tuple[
        Qwen3OmniMoeAudioEncoder,
        HfQwen3OmniMoeAudioEncoder,
        Callable[..., torch.Tensor],
    ]
]:
    if not torch.accelerator.is_available():
        pytest.fail("Needs a TPU: the chunk-length failure is in TPU float division.")

    cls = Qwen3OmniMoeAudioEncoder
    upstream_forward = cls.forward
    orig_get_cnn_lengths = cls._get_cnn_output_lengths
    assert not getattr(upstream_forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False)
    try:
        assert (
            maybe_patch_qwen3_omni_thinker(["Qwen3OmniMoeForConditionalGeneration"])
            is True
        )
        assert getattr(cls.forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False)

        torch.manual_seed(0)
        ref = HfQwen3OmniMoeAudioEncoder(AUDIO_CONFIG).eval()

        tpu = cls(AUDIO_CONFIG).eval()
        loaded = tpu.load_weights(ref.named_parameters())
        assert_all_loaded(loaded, tpu)
        tpu = tpu.to(device="tpu", dtype=torch.bfloat16)
        yield tpu, ref, upstream_forward
    finally:
        cls.forward = upstream_forward
        cls._get_cnn_output_lengths = orig_get_cnn_lengths


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="vLLM-Omni 0.32.0rc1 takes ceil(feature_lens / 100) on the device, "
    "which TPU rounds up to 9 windows for 800 frames; the split then fails",
)
@torch.no_grad()
def test_upstream_forward_miscounts_exact_window_multiples(encoders) -> None:
    tpu, _, upstream_forward = encoders
    features = _make_mel_features(800).to(device="tpu", dtype=torch.bfloat16)
    feature_lens = torch.tensor([800], dtype=torch.long, device="tpu")
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
    try:
        out = upstream_forward(tpu, features, feature_lens, aftercnn_lens)
    except RuntimeError as exc:
        assert "split_with_sizes expects split_sizes to sum exactly to 800" not in str(
            exc
        ), str(exc)
        raise
    assert out.shape[0] == int(_get_feat_extract_output_lengths(800))


@torch.no_grad()
def test_audio_encoder_layer_matches_cpu_reference(encoders) -> None:
    tpu, ref, _ = encoders
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(105, AUDIO_CONFIG.d_model, generator=generator)
    cu_seqlens_cpu = torch.tensor([0, 104, 105], dtype=torch.int32)
    cu_seqlens_tpu = cu_seqlens_cpu.to("tpu")
    max_seqlen = 104

    ref_attn = ref.layers[0].self_attn(hidden, cu_seqlens_cpu, max_seqlen=max_seqlen)
    if isinstance(ref_attn, tuple):
        ref_attn = ref_attn[0]
    tpu_attn = tpu.layers[0].self_attn(
        hidden.to(device="tpu", dtype=torch.bfloat16),
        cu_seqlens_tpu,
        max_seqlen=max_seqlen,
    )
    assert_parity(
        tpu_attn,
        ref_attn,
        name="layers.0.self_attn",
        max_rel_l2=ATTN_LIMITS[0],
        max_abs=ATTN_LIMITS[1],
    )

    ref_layer = ref.layers[0](hidden, cu_seqlens_cpu, max_seqlen=max_seqlen)[0]
    tpu_layer = tpu.layers[0](
        hidden.to(device="tpu", dtype=torch.bfloat16),
        cu_seqlens_tpu,
        max_seqlen=max_seqlen,
    )
    assert_parity(
        tpu_layer,
        ref_layer,
        name="layers.0",
        max_rel_l2=LAYER_LIMITS[0],
        max_abs=LAYER_LIMITS[1],
    )


@pytest.mark.parametrize("n_frames", PARITY_LENGTHS)
@torch.no_grad()
def test_audio_encoder_matches_cpu_reference(encoders, n_frames: int) -> None:
    tpu, ref, _ = encoders
    features = _make_mel_features(n_frames)
    ref_lens = torch.tensor([n_frames], dtype=torch.long)
    reference = ref(features, ref_lens).last_hidden_state

    feature_lens = torch.tensor([n_frames], dtype=torch.long, device="tpu")
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
    out = tpu(
        features.to(device="tpu", dtype=torch.bfloat16),
        feature_lens,
        aftercnn_lens,
    )
    assert feature_lens.device.type == "tpu"
    assert out.device.type == "tpu"
    assert out.shape == (
        int(_get_feat_extract_output_lengths(n_frames)),
        AUDIO_CONFIG.output_dim,
    )
    assert_parity(
        out,
        reference,
        name=f"audio encoder ({n_frames} frames)",
        max_rel_l2=END_TO_END_LIMITS[0],
        max_abs=END_TO_END_LIMITS[1],
    )
