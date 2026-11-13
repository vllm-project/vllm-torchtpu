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
"""TPU vs. float32 CPU parity tests for Qwen3-Omni Thinker components."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import MethodType, SimpleNamespace

import pytest
import torch
from transformers import Qwen3OmniMoeAudioEncoderConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder as HfQwen3OmniMoeAudioEncoder,
)
from vllm.multimodal.utils import set_mm_embedding_modality
from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorBaseModel,
    CodePredictorWrapper,
    CodePredictorWrapperConfig,
    _RMSNorm,
    _RotaryEmbedding,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3Omni_VisionTransformer,
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    _get_feat_extract_output_lengths,
)

from tests_omni.omni_test_utils import (
    assert_all_loaded,
    assert_parity,
    init_test_weights,
    isolate_thinker_patch_targets,
    use_native_ops_on_cpu,
)
from vllm_torchtpu.omni.patches.qwen3omni_talker import (
    maybe_patch_qwen3_omni_talker,
)
from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_CHUNK_LENGTHS_PATCHED,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = pytest.mark.omni

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

PARITY_LENGTHS = [800, 801, 3200, 3201]

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


@torch.no_grad()
def test_multimodal_merge_and_embed_input_ids_on_tpu(encoders) -> None:
    """Verify interleaved video+audio+image merge and DeepStack scatter on TPU."""
    del encoders
    tpu_dev = torch.device("tpu")
    hidden_size = 16
    deepstack_layers = 2
    vis_width = hidden_size * (1 + deepstack_layers)

    # Prompt layout (8 tokens): [text, video, video, audio, video, image, text, text]
    # Video ID=10, Audio ID=11, Image ID=12
    input_ids_cpu = torch.tensor([1, 10, 10, 11, 10, 12, 2, 3], dtype=torch.long)
    is_mm_cpu = torch.tensor([False, True, True, True, True, True, False, False])

    vid_cpu = torch.arange(3 * vis_width, dtype=torch.float32).reshape(3, vis_width)
    aud_cpu = torch.full((1, hidden_size), 100.0, dtype=torch.float32)
    img_cpu = torch.full((1, vis_width), 200.0, dtype=torch.float32)

    class _FakeThinker(Qwen3OmniMoeThinkerForConditionalGeneration):
        def __init__(self) -> None:
            self.use_deepstack = True
            self.deepstack_input_embeds_num_tokens = 4
            self.deepstack_input_embeds = [
                torch.ones((8, hidden_size), device=tpu_dev, dtype=torch.float32)
                for _ in range(deepstack_layers)
            ]
            self.config = SimpleNamespace(
                video_token_id=10,
                audio_token_id=11,
                text_config=SimpleNamespace(hidden_size=hidden_size),
            )
            self.visual = SimpleNamespace(deepstack_visual_indexes=[0, 1])
            self.language_model = SimpleNamespace(
                embed_input_ids=lambda ids: torch.zeros(
                    (ids.shape[0], hidden_size), device=ids.device, dtype=torch.float32
                )
            )

        def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
            for buf in self.deepstack_input_embeds:
                buf[:num_tokens].zero_()
            self.deepstack_input_embeds_num_tokens = 0

        def _set_deepstack_input_embeds(
            self, deepstack_input_embeds: torch.Tensor
        ) -> None:
            n = deepstack_input_embeds.shape[1]
            for i in range(deepstack_layers):
                self.deepstack_input_embeds[i][:n].copy_(deepstack_input_embeds[i])
            self.deepstack_input_embeds_num_tokens = n

    thinker = _FakeThinker()
    vid_tpu = set_mm_embedding_modality(vid_cpu.to(tpu_dev), "video")
    aud_tpu = set_mm_embedding_modality(aud_cpu.to(tpu_dev), "audio")
    img_tpu = set_mm_embedding_modality(img_cpu.to(tpu_dev), "image")

    merged_tpu = thinker.embed_input_ids(
        input_ids_cpu.to(tpu_dev),
        multimodal_embeddings=[vid_tpu, aud_tpu, img_tpu],
        is_multimodal=is_mm_cpu.to(tpu_dev),
    )

    expected_main = torch.zeros((8, hidden_size), dtype=torch.float32)
    expected_main[[1, 2, 4]] = vid_cpu[:, :hidden_size]
    expected_main[3] = aud_cpu[0]
    expected_main[5] = img_cpu[0, :hidden_size]

    assert torch.allclose(merged_tpu.cpu(), expected_main)
    assert thinker.deepstack_input_embeds_num_tokens == 8
    # Interleaved path scatters video deepstack features into video positions [1, 2, 4]
    assert torch.allclose(
        thinker.deepstack_input_embeds[0].cpu()[[1, 2, 4]],
        vid_cpu[:, hidden_size : 2 * hidden_size],
    )


def test_code_predictor_wrapper_on_tpu_matches_cpu_reference(encoders) -> None:
    """Patched CodePredictorWrapper runs residual AR on CPU and returns TPU tensors."""
    del encoders
    cp_config = Qwen3OmniMoeTalkerCodePredictorConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        num_code_groups=4,
    )
    rms_own = _RMSNorm.__dict__.get("forward_tpu")
    rotary_own = _RotaryEmbedding.__dict__.get("forward_tpu")
    base_fwd = CodePredictorBaseModel.__dict__.get("forward")
    wrapper_fwd = CodePredictorWrapper.__dict__.get("forward")
    talker_pre = Qwen3OmniMoeForConditionalGeneration.__dict__.get("talker_preprocess")
    try:
        assert (
            maybe_patch_qwen3_omni_talker(["Qwen3OmniMoeForConditionalGeneration"])
            is True
        )

        torch.manual_seed(0)
        wrapper = CodePredictorWrapper(
            vllm_config=SimpleNamespace(model_config=SimpleNamespace()),
            cp_config=cp_config,
            wrapper_config=CodePredictorWrapperConfig(
                use_parallel_embedding=False,
                use_projection=True,
                use_cuda_graphs=False,
                return_proj_buf=True,
            ),
            talker_hidden_size=48,
        ).eval()
        init_test_weights(wrapper, seed=0)
        use_native_ops_on_cpu(wrapper)
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
            cp_config=cp_config,
            wrapper_config=CodePredictorWrapperConfig(
                use_parallel_embedding=False,
                use_projection=True,
                use_cuda_graphs=False,
                return_proj_buf=True,
            ),
            talker_hidden_size=48,
        ).eval()
        tpu_wrapper.load_state_dict(wrapper.state_dict())
        use_native_ops_on_cpu(tpu_wrapper)
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
    finally:
        for cls, attr, val in (
            (_RMSNorm, "forward_tpu", rms_own),
            (_RotaryEmbedding, "forward_tpu", rotary_own),
            (CodePredictorBaseModel, "forward", base_fwd),
            (CodePredictorWrapper, "forward", wrapper_fwd),
            (Qwen3OmniMoeForConditionalGeneration, "talker_preprocess", talker_pre),
        ):
            if val is None:
                if attr in cls.__dict__:
                    delattr(cls, attr)
            else:
                setattr(cls, attr, val)
