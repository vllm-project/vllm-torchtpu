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
"""TPU vs. float32 CPU parity tests for Qwen3-Omni Audio and Vision Encoders."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import MethodType

import pytest
import torch
from transformers import Qwen3OmniMoeAudioEncoderConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder as HfQwen3OmniMoeAudioEncoder,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3Omni_VisionTransformer,
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)

from tests_omni.omni_test_utils import (
    assert_all_loaded,
    assert_parity,
    init_test_weights,
    isolate_thinker_patch_targets,
    use_native_ops_on_cpu,
)
from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_CHUNK_LENGTHS_PATCHED,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = pytest.mark.omni

# Qwen3-Omni-30B-A3B audio geometry with 2 transformer layers instead of 32 so
# the fixture stays small and fast; every patched code path (conv stack,
# positional slice, _flash_attention with padded cu_seqlens, output projection)
# still runs.
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

VISION_CONFIG = Qwen3OmniMoeVisionEncoderConfig(
    depth=2,
    hidden_size=128,
    intermediate_size=256,
    num_heads=4,
    in_channels=3,
    image_size=256,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
    out_hidden_size=128,
    num_position_embeddings=256,
    apply_vit_abs_pos_embed=True,
    deepstack_visual_indexes=[0, 1],
)

# 800 frames = 8 full 100-frame windows (1 attention segment); 801 takes the
# partial-window branch across two attention segments; 3200 hits the exact
# 32-window bucket boundary; 3201 (33 windows) exercises the 8-window aligned
# fallback (40 windows) beyond _AUDIO_CHUNK_BUCKETS.
PARITY_LENGTHS = [800, 801, 3200, 3201]

# Single image (600 patches -> 4096 bucket), medium video (4800 patches -> 8192
# bucket), and multi-chunk video (20480 patches > 16384 split into 4 + 1 frames
# hitting the 16384 and 4096 buckets).
VISION_PARITY_GRIDS = [
    [[1, 20, 30]],
    [[8, 20, 30]],
    [[5, 64, 64]],
]

# (max relative L2, max abs) vs float32 HF CPU reference (~2.5x measured on TPU
# v6e across 800, 801, 3200, 3201 frames and multi-clip [800, 801] batches).
ATTN_LIMITS = (1.0e-2, 3.0e-3)
LAYER_LIMITS = (8.0e-3, 5.5e-2)
END_TO_END_LIMITS = (1.5e-2, 4.5e-3)

# (max relative L2, max abs) vs float32 CPU reference (~2.5x measured on TPU
# v6e across single-image, short-video, and multi-chunk video grids:
# measured rel_l2<=8e-3, max_abs<=6e-3).
VISION_LIMITS = (2.0e-2, 1.5e-2)


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
    assert not getattr(upstream_forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False)
    with isolate_thinker_patch_targets():
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


@torch.no_grad()
def test_audio_encoder_multi_clip_batch_matches_cpu_reference(encoders) -> None:
    """Verify multi-clip batching and 0.33.0rc1 metadata API parity against CPU."""
    tpu, ref, _ = encoders
    f_800 = _make_mel_features(800)
    f_801 = _make_mel_features(801)
    ref_800 = ref(f_800, torch.tensor([800], dtype=torch.long)).last_hidden_state
    ref_801 = ref(f_801, torch.tensor([801], dtype=torch.long)).last_hidden_state
    reference = torch.cat([ref_800, ref_801], dim=0)

    feats_tpu = torch.cat([f_800, f_801], dim=-1).to(device="tpu", dtype=torch.bfloat16)
    lens_tpu = torch.tensor([800, 801], dtype=torch.long, device="tpu")
    aftercnn_tpu = _get_feat_extract_output_lengths(lens_tpu)

    out_fwd = tpu(feats_tpu, lens_tpu, aftercnn_tpu)
    metadata = tpu.prepare_encoder_metadata([800, 801])
    out_meta = tpu.forward_with_encoder_metadata(feats_tpu, metadata)

    assert torch.equal(out_fwd, out_meta)
    assert_parity(
        out_fwd,
        reference,
        name="audio encoder multi-clip [800, 801]",
        max_rel_l2=END_TO_END_LIMITS[0],
        max_abs=END_TO_END_LIMITS[1],
    )


@pytest.mark.parametrize("grid", VISION_PARITY_GRIDS)
@torch.no_grad()
def test_vision_encoder_matches_cpu_reference(encoders, grid: list[list[int]]) -> None:
    """Verify Qwen3Omni_VisionTransformer TPU bucketing and chunking vs CPU."""
    del encoders  # ensures patch is active and TPU is available
    torch.manual_seed(0)
    vit_cpu = Qwen3Omni_VisionTransformer(VISION_CONFIG, norm_eps=1e-6).eval()
    init_test_weights(vit_cpu, seed=0)
    use_native_ops_on_cpu(vit_cpu)
    vit_cpu.forward = MethodType(Qwen3Omni_VisionTransformer._tpu_orig_forward, vit_cpu)

    vit_tpu = Qwen3Omni_VisionTransformer(VISION_CONFIG, norm_eps=1e-6).eval()
    vit_tpu.load_state_dict(vit_cpu.state_dict())
    vit_tpu = vit_tpu.to(device="tpu", dtype=torch.bfloat16)

    total_patches = sum(t * h * w for t, h, w in grid)
    patch_dim = (
        VISION_CONFIG.in_channels
        * VISION_CONFIG.temporal_patch_size
        * VISION_CONFIG.patch_size
        * VISION_CONFIG.patch_size
    )
    generator = torch.Generator().manual_seed(total_patches)
    pixels = (
        torch.randn(total_patches, patch_dim, generator=generator)
        .to(torch.bfloat16)
        .float()
    )
    grid_tensor = torch.tensor(grid, dtype=torch.int32)

    ref_out = vit_cpu(pixels, grid_tensor)
    tpu_out = vit_tpu(pixels.to(device="tpu", dtype=torch.bfloat16), grid_tensor)

    assert tpu_out.device.type == "tpu"
    assert_parity(
        tpu_out,
        ref_out,
        name=f"vision encoder grid={grid}",
        max_rel_l2=VISION_LIMITS[0],
        max_abs=VISION_LIMITS[1],
    )
