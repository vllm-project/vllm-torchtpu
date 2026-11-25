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
from types import MethodType, SimpleNamespace

import pytest
import torch
from transformers import Qwen3OmniMoeAudioEncoderConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder as HfQwen3OmniMoeAudioEncoder,
)
from vllm.multimodal.utils import set_mm_embedding_modality
from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorWrapper,
    CodePredictorWrapperConfig,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3Omni_VisionTransformer,
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)

from tests_omni.models.qwen3_omni.utils import (
    TINY_CP_CONFIG,
    bind_native_custom_ops_on_cpu,
    init_linear_weights,
    isolate_talker_patch_targets,
)
from tests_omni.omni_test_utils import (
    assert_all_loaded,
    assert_parity,
    build_qwen3_omni_thinker_stub,
    init_test_weights,
    isolate_thinker_patch_targets,
    native_ops_on_cpu,
)
from vllm_torchtpu.omni.patches.qwen3omni_talker import (
    maybe_patch_qwen3_omni_talker,
)
from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _padded_audio_forward,
    maybe_patch_qwen3_omni_thinker,
)
from vllm_torchtpu.omni.worker.tpu_ar_model_runner import TPUARModelRunner

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

# Small image (600 patches -> 2048 bucket), medium image (3000 patches -> 4096
# bucket), medium video (4800 patches -> 8192 bucket), and multi-chunk video
# (20480 patches > 16384 split into 4 + 1 frames hitting 16384 and 4096).
VISION_PARITY_GRIDS = [
    [[1, 20, 30]],
    [[1, 50, 60]],
    [[8, 20, 30]],
    [[5, 64, 64]],
]

# (max relative L2, max abs) vs float32 HF CPU reference on TPU v6e and v7x.
ATTN_LIMITS = (1.0e-2, 3.0e-3)
LAYER_LIMITS = (8.0e-3, 5.5e-2)
END_TO_END_LIMITS = (1.5e-2, 3.5e-3)

# (max relative L2, max abs) vs float32 CPU reference (~2.5x measured on TPU
# v6e across single-image, short-video, and multi-chunk video grids:
# measured rel_l2<=8e-3, max_abs<=6e-3).
VISION_LIMITS = (2.0e-2, 1.5e-2)


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
    with isolate_thinker_patch_targets():
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


@pytest.mark.parametrize("grid", VISION_PARITY_GRIDS)
@torch.no_grad()
def test_vision_encoder_matches_cpu_reference(encoders, grid: list[list[int]]) -> None:
    """Verify Qwen3Omni_VisionTransformer TPU bucketing and chunking vs CPU."""
    del encoders  # ensures patch is active and TPU is available
    torch.manual_seed(0)
    vit_cpu = Qwen3Omni_VisionTransformer(VISION_CONFIG, norm_eps=1e-6).eval()
    init_test_weights(vit_cpu, seed=0)
    vit_cpu.forward = MethodType(Qwen3Omni_VisionTransformer._tpu_orig_forward, vit_cpu)

    vit_tpu = Qwen3Omni_VisionTransformer(VISION_CONFIG, norm_eps=1e-6).eval()
    vit_tpu.load_state_dict(vit_cpu.state_dict())
    vit_tpu = vit_tpu.to(device="tpu", dtype=torch.bfloat16)

    total_patches = sum(t * h * w for t, h, w in grid)
    patch_dim = (
        VISION_CONFIG.in_channels
        * VISION_CONFIG.temporal_patch_size
        * VISION_CONFIG.patch_size**2
    )
    generator = torch.Generator().manual_seed(total_patches)
    pixels = (
        torch.randn(total_patches, patch_dim, generator=generator)
        .to(torch.bfloat16)
        .float()
    )
    grid_tensor = torch.tensor(grid, dtype=torch.int32)

    with native_ops_on_cpu(vit_cpu):
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


@pytest.mark.parametrize(
    ("input_ids_list", "vis_positions", "aud_positions", "vis_modality"),
    [
        pytest.param(
            [1, 10, 10, 11, 10, 11, 2, 3],
            [1, 2, 4],
            [3, 5],
            "video",
            id="interleaved_video_audio",
        ),
        pytest.param(
            [1, 12, 12, 2, 11, 11, 3, 4],
            [1, 2],
            [4, 5],
            "image",
            id="non_interleaved_image_audio",
        ),
    ],
)
@torch.no_grad()
def test_multimodal_merge_and_embed_input_ids_on_tpu(
    encoders,
    input_ids_list: list[int],
    vis_positions: list[int],
    aud_positions: list[int],
    vis_modality: str,
) -> None:
    """Verify runner mixed-dim slicing, embed_input_ids, and DeepStack clear on TPU."""
    del encoders
    tpu_dev = torch.device("tpu")
    hidden_size, deepstack_layers = 16, 2
    num_tokens = len(input_ids_list)
    vis_width = hidden_size * (1 + deepstack_layers)

    input_ids_cpu = torch.tensor(input_ids_list, dtype=torch.long)
    is_mm_cpu = torch.zeros(num_tokens, dtype=torch.bool)
    is_mm_cpu[vis_positions + aud_positions] = True
    vis_cpu = torch.arange(len(vis_positions) * vis_width, dtype=torch.float32).reshape(
        len(vis_positions), vis_width
    )
    aud_cpu = torch.arange(
        100, 100 + len(aud_positions) * hidden_size, dtype=torch.float32
    ).reshape(len(aud_positions), hidden_size)

    thinker = build_qwen3_omni_thinker_stub(
        hidden_size=hidden_size,
        deepstack_layers=deepstack_layers,
        max_tokens=num_tokens,
        device=tpu_dev,
    )
    for buf in thinker.deepstack_input_embeds:
        buf.fill_(99.0)
    thinker.deepstack_input_embeds_num_tokens = num_tokens

    runner = TPUARModelRunner.__new__(TPUARModelRunner)
    runner.supports_mm_inputs = True
    runner.model = thinker
    runner._omni_mm_embeds_list = [
        set_mm_embedding_modality(vis_cpu.to(tpu_dev), vis_modality),
        set_mm_embedding_modality(aud_cpu.to(tpu_dev), "audio"),
    ]
    row_indices = torch.arange(len(vis_positions) + len(aud_positions))
    out_ids, merged_tpu = runner._get_model_inputs(
        input_ids_cpu.to(tpu_dev), ([row_indices], is_mm_cpu[:-2])
    )

    expected_main = input_ids_cpu.float().unsqueeze(-1).expand(-1, hidden_size) * 0.1
    expected_main = expected_main.clone()
    expected_main[vis_positions] = vis_cpu[:, :hidden_size]
    expected_main[aud_positions] = aud_cpu
    assert (
        out_ids is None and merged_tpu is not None and merged_tpu.device.type == "tpu"
    )
    assert torch.allclose(merged_tpu.cpu(), expected_main)
    assert thinker.deepstack_input_embeds_num_tokens == num_tokens
    for layer_idx in range(deepstack_layers):
        expected_ds = torch.zeros((num_tokens, hidden_size))
        c = (layer_idx + 1) * hidden_size
        expected_ds[vis_positions] = vis_cpu[:, c : c + hidden_size]
        assert torch.allclose(
            thinker.deepstack_input_embeds[layer_idx][:num_tokens].cpu(), expected_ds
        )

    # Follow-up text-only step clears stale DeepStack buffers.
    runner._omni_mm_embeds_list = None
    text_ids = torch.arange(1, num_tokens + 1, dtype=torch.long)
    _, text_out = runner._get_model_inputs(
        text_ids.to(tpu_dev), ([], torch.zeros(num_tokens, dtype=torch.bool))
    )
    assert text_out is not None
    assert torch.allclose(
        text_out.cpu(), text_ids.float().unsqueeze(-1).expand(-1, hidden_size) * 0.1
    )
    assert thinker.deepstack_input_embeds_num_tokens == 0
    for buf in thinker.deepstack_input_embeds:
        assert torch.equal(
            buf[:num_tokens].cpu(), torch.zeros((num_tokens, hidden_size))
        )


def test_code_predictor_wrapper_on_tpu_matches_cpu_reference(omni_ctx: None) -> None:
    """Patched CodePredictorWrapper runs residual AR on CPU and returns TPU tensors."""
    if not torch.accelerator.is_available():
        pytest.fail("Needs a TPU to verify CodePredictorWrapper TPU offload.")

    with isolate_talker_patch_targets():
        assert (
            maybe_patch_qwen3_omni_talker(["Qwen3OmniMoeForConditionalGeneration"])
            is True
        )

        torch.manual_seed(0)
        wrapper = CodePredictorWrapper(
            vllm_config=SimpleNamespace(model_config=SimpleNamespace()),
            cp_config=TINY_CP_CONFIG,
            wrapper_config=CodePredictorWrapperConfig(
                use_parallel_embedding=False,
                use_projection=True,
                use_cuda_graphs=False,
                return_proj_buf=True,
            ),
            talker_hidden_size=48,
        ).eval()
        init_linear_weights(wrapper, seed=0)
        bind_native_custom_ops_on_cpu(wrapper)
        wrapper.set_sampling_params(do_sample=False)

        layer0_code = torch.tensor([[3]], dtype=torch.long)
        layer0_embed = torch.randn(1, 1, 48)
        last_talker_hidden = torch.randn(1, 1, 48)

        with torch.no_grad():
            ref_codes, ref_proj = wrapper(
                layer0_code=layer0_code,
                layer0_embed=layer0_embed,
                last_talker_hidden=last_talker_hidden,
                do_sample=False,
            )

        tpu_dev = torch.device("tpu")
        tpu_wrapper = CodePredictorWrapper(
            vllm_config=SimpleNamespace(model_config=SimpleNamespace()),
            cp_config=TINY_CP_CONFIG,
            wrapper_config=CodePredictorWrapperConfig(
                use_parallel_embedding=False,
                use_projection=True,
                use_cuda_graphs=False,
                return_proj_buf=True,
            ),
            talker_hidden_size=48,
        ).eval()
        tpu_wrapper.load_state_dict(wrapper.state_dict())
        bind_native_custom_ops_on_cpu(tpu_wrapper)
        tpu_wrapper.set_sampling_params(do_sample=False)
        tpu_wrapper = tpu_wrapper.to(tpu_dev)

        with torch.no_grad():
            tpu_codes, tpu_proj = tpu_wrapper(
                layer0_code=layer0_code.to(tpu_dev),
                layer0_embed=layer0_embed.to(tpu_dev),
                last_talker_hidden=last_talker_hidden.to(tpu_dev),
                do_sample=False,
            )
        assert next(tpu_wrapper.parameters()).device.type == "cpu"
        assert tpu_codes.device.type == "tpu"
        assert tpu_proj.device.type == "tpu"
        assert torch.equal(tpu_codes.cpu(), ref_codes)
        assert torch.allclose(tpu_proj.cpu(), ref_proj, atol=1e-5)
