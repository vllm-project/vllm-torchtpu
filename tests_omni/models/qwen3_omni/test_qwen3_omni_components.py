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
"""TPU vs. float32 CPU parity tests for the Qwen3-Omni Audio Encoder."""

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

from tests_omni.omni_test_utils import (
    assert_all_loaded,
    assert_parity,
    restore_class_attrs,
)
from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _padded_audio_forward,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = [pytest.mark.omni]

AUDIO_CONFIG = Qwen3OmniMoeAudioEncoderConfig(
    num_mel_bins=128,
    d_model=128,
    encoder_layers=2,
    encoder_attention_heads=4,
    encoder_ffn_dim=256,
    downsample_hidden_size=64,
    output_dim=128,
    n_window=50,
    n_window_infer=400,
)

# Exercises all 5 buckets (8, 16, 32, 64, 128), sub-second odd/even CNN widths
# ([50], [79]), multi-clip packing ([800, 801], [801, 800], [2000, 2000]), and
# >128 multi-pass splits with dead-segment isolation ([12801, 200], [25601, 200]).
PARITY_CASES = [
    pytest.param([50], {8}, id="50"),
    pytest.param([79], {8}, id="79"),
    pytest.param([800], {8}, id="800"),
    pytest.param([801], {16}, id="801"),
    pytest.param([800, 801], {32}, id="800_801"),
    pytest.param([801, 800], {32}, id="801_800"),
    pytest.param([2000, 2000], {64}, id="2000_2000"),
    pytest.param([12800], {128}, id="12800"),
    pytest.param([12801, 200], {128, 8}, id="12801_200"),
    pytest.param([25601, 200], {128, 8}, id="25601_200"),
]

# (max relative L2, max abs) vs float32 HF CPU reference on TPU v6e and v7x.
ATTN_LIMITS = (1.0e-2, 3.0e-3)
LAYER_LIMITS = (8.0e-3, 5.5e-2)
END_TO_END_LIMITS = (1.5e-2, 3.5e-3)

_PATCHED_ATTRS = ("forward", "_tpu_enforce_eager")


def _make_mel_features(n_frames: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
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
    assert upstream_forward is not _padded_audio_forward
    with restore_class_attrs(cls, _PATCHED_ATTRS):
        assert (
            maybe_patch_qwen3_omni_thinker(["Qwen3OmniMoeForConditionalGeneration"])
            is True
        )
        assert cls.forward is _padded_audio_forward

        torch.manual_seed(0)
        ref = HfQwen3OmniMoeAudioEncoder(AUDIO_CONFIG).eval()

        tpu = cls(AUDIO_CONFIG).eval()
        loaded = tpu.load_weights(ref.named_parameters())
        assert_all_loaded(loaded, tpu)
        tpu = tpu.to(device="tpu", dtype=torch.bfloat16)
        yield tpu, ref, upstream_forward


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
    assert out.shape[0] == int(
        _get_feat_extract_output_lengths(torch.tensor([800])).item()
    )


@torch.no_grad()
def test_audio_encoder_layer_matches_cpu_reference(encoders) -> None:
    tpu, ref, _ = encoders
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(105, AUDIO_CONFIG.d_model, generator=generator)
    cu_seqlens_cpu = torch.tensor([0, 104, 105], dtype=torch.int32)
    cu_seqlens_tpu = cu_seqlens_cpu.to("tpu")
    max_seqlen = 104

    ref_attn = ref.layers[0].self_attn(hidden, cu_seqlens_cpu, max_seqlen=max_seqlen)
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


@pytest.mark.parametrize(("clip_lengths", "expected_buckets"), PARITY_CASES)
@torch.no_grad()
def test_audio_encoder_matches_cpu_reference(
    encoders, clip_lengths: list[int], expected_buckets: set[int]
) -> None:
    tpu, ref, _ = encoders
    features = torch.cat(
        [_make_mel_features(n, seed=idx) for idx, n in enumerate(clip_lengths)],
        dim=1,
    )
    ref_lens = torch.tensor(clip_lengths, dtype=torch.long)
    reference = ref(features, ref_lens).last_hidden_state

    feats_tpu = features.to(device="tpu", dtype=torch.bfloat16)
    feature_lens = torch.tensor(clip_lengths, dtype=torch.long, device="tpu")
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
    out = tpu(feats_tpu, feature_lens, aftercnn_lens)

    expected_tokens = int(_get_feat_extract_output_lengths(ref_lens).sum())
    assert out.device.type == "tpu"
    assert out.shape == (expected_tokens, AUDIO_CONFIG.output_dim)
    assert expected_buckets.issubset(getattr(tpu, "_tpu_compiled_buckets", {}))
    assert_parity(
        out,
        reference,
        name=f"audio encoder ({clip_lengths} frames)",
        max_rel_l2=END_TO_END_LIMITS[0],
        max_abs=END_TO_END_LIMITS[1],
    )


@pytest.mark.parametrize("clip_lengths", [[], [0]])
@torch.no_grad()
def test_audio_encoder_empty_inputs_on_tpu(encoders, clip_lengths: list[int]) -> None:
    tpu, _, _ = encoders
    assert not hasattr(tpu, "output_dim")
    feats_tpu = torch.zeros(
        (AUDIO_CONFIG.num_mel_bins, 0), device="tpu", dtype=torch.bfloat16
    )
    feature_lens = torch.tensor(clip_lengths, dtype=torch.long, device="tpu")
    out = tpu(feats_tpu, feature_lens)
    assert out.device.type == "tpu"
    assert out.dtype == torch.bfloat16
    assert out.shape == (0, AUDIO_CONFIG.output_dim)
