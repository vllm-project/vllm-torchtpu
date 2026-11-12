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
"""Component-wise TPU vs CPU tests for Qwen3-Omni thinker audio and vision encoders."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import MethodType, SimpleNamespace

import pytest
import torch
import vllm.model_executor.models.utils as vllm_utils
from transformers import Qwen3OmniMoeAudioEncoderConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)
from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorBaseModel,
    CodePredictorWrapper,
    CodePredictorWrapperConfig,
    _RMSNorm,
    _RotaryEmbedding,
)
from vllm_omni.model_executor.models.qwen3_omni import (
    qwen3_omni_moe_thinker as thinker_mod,
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
    assert_parity,
    init_test_weights,
    use_native_ops_on_cpu,
)
from vllm_torchtpu.omni.patches.qwen3omni_talker import (
    maybe_patch_qwen3_omni_talker,
)
from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_STATIC_BUCKETING_PATCHED,
    _MM_MERGE_STATIC_PATCHED,
    _VIT_STATIC_BUCKETING_PATCHED,
    _pad_rows,
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

VISION_CONFIG = Qwen3OmniMoeVisionEncoderConfig(
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

# 800 frames (8 s) is the exact 100-frame multiple the patch fixes; 801 takes
# the partial-window branch across two attention segments; 3200 hits the upper
# edge of the 32-window bucket (512 padded tokens); 3201 crosses into the
# 40-window fallback (640 padded tokens).
PARITY_LENGTHS = [800, 801, 3200, 3201]

# (max relative L2, max abs) vs float32 CPU reference (~2.5x measured on TPU v6e).
END_TO_END_LIMITS = (1.5e-2, 4.5e-3)
COMPONENT_LIMITS = {
    "conv stack + positional embedding": (5e-3, 1.5e-2),
    "layers.0.self_attn": (9e-3, 3.5e-3),
    "layers.0": (8e-3, 3e-2),
    "ln_post + proj": (1.2e-2, 4.5e-3),
}


def _sample_features(n_frames: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(n_frames)
    return (
        torch.randn(AUDIO_CONFIG.num_mel_bins, n_frames, generator=generator)
        .to(torch.bfloat16)
        .float()
    )


def _encode(
    encoder: Qwen3OmniMoeAudioEncoder,
    n_frames: int | list[int],
    forward: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    lengths = [n_frames] if isinstance(n_frames, int) else list(n_frames)
    features = torch.cat([_sample_features(n) for n in lengths], dim=1)
    param = next(encoder.parameters())
    device, dtype = param.device, param.dtype
    feature_lens = torch.tensor(lengths, dtype=torch.long, device=device)
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
    run = encoder if forward is None else forward.__get__(encoder)
    with torch.no_grad():
        out = run(features.to(device=device, dtype=dtype), feature_lens, aftercnn_lens)
    assert feature_lens.device == device
    return out


def _check(actual: torch.Tensor, reference: torch.Tensor, name: str, limits) -> None:
    diff = actual.detach().cpu().double() - reference.detach().cpu().double()
    rel_l2 = float(diff.norm() / reference.detach().cpu().double().norm())
    print(
        f"parity {name:<40} rel_l2={rel_l2:.3e} max_abs={float(diff.abs().max()):.3e}"
    )
    max_rel_l2, max_abs = limits
    assert_parity(actual, reference, name=name, max_rel_l2=max_rel_l2, max_abs=max_abs)


@pytest.fixture(scope="module")
def encoders(
    omni_ctx: None,
) -> Iterator[
    tuple[
        Qwen3OmniMoeAudioEncoder, Qwen3OmniMoeAudioEncoder, Callable[..., torch.Tensor]
    ]
]:
    if not torch.accelerator.is_available():
        pytest.fail("Needs a TPU: the chunk-length failure is in TPU float division.")

    audio_cls = Qwen3OmniMoeAudioEncoder
    vit_cls = Qwen3Omni_VisionTransformer
    thinker_cls = Qwen3OmniMoeThinkerForConditionalGeneration
    upstream_forward = audio_cls.forward
    assert not getattr(upstream_forward, _AUDIO_STATIC_BUCKETING_PATCHED, False)
    audio_own = {
        n: audio_cls.__dict__.get(n)
        for n in (
            "forward",
            "_tpu_orig_forward",
            "prepare_encoder_metadata",
            "forward_with_encoder_metadata",
        )
    }
    vit_own = {n: vit_cls.__dict__.get(n) for n in ("forward", "_tpu_orig_forward")}
    thinker_cls_own = {n: thinker_cls.__dict__.get(n) for n in ("embed_input_ids",)}
    orig_vllm_merge = vllm_utils._merge_multimodal_embeddings
    orig_thinker_merge = thinker_mod._merge_multimodal_embeddings
    orig_check_interleaved = thinker_mod.check_interleaved_audio_video
    orig_merge_interleaved = thinker_mod.merge_interleaved_embeddings
    try:
        config = SimpleNamespace(architectures=["Qwen3OmniMoeForConditionalGeneration"])
        assert maybe_patch_qwen3_omni_thinker(config) is True
        assert getattr(audio_cls.forward, _AUDIO_STATIC_BUCKETING_PATCHED, False)
        assert getattr(vit_cls.forward, _VIT_STATIC_BUCKETING_PATCHED, False)
        assert getattr(thinker_cls.embed_input_ids, _MM_MERGE_STATIC_PATCHED, False)
        assert audio_cls._tpu_orig_forward is upstream_forward

        torch.manual_seed(0)
        cpu = audio_cls(AUDIO_CONFIG).eval()
        # Bind the unpatched upstream forward on the CPU reference so parity
        # checks compare the patched bucketed TPU forward against upstream.
        cpu.forward = MethodType(upstream_forward, cpu)
        init_test_weights(cpu, seed=0)
        assert all(torch.isfinite(p).all() for p in cpu.parameters())
        use_native_ops_on_cpu(cpu)

        tpu = audio_cls(AUDIO_CONFIG).eval()
        tpu.load_state_dict(cpu.state_dict())
        tpu = tpu.to(torch.device("tpu")).to(torch.bfloat16)
        yield tpu, cpu, upstream_forward
    finally:
        for cls, own in (
            (audio_cls, audio_own),
            (vit_cls, vit_own),
            (thinker_cls, thinker_cls_own),
        ):
            for name, value in own.items():
                if value is None:
                    if name in cls.__dict__:
                        delattr(cls, name)
                else:
                    setattr(cls, name, value)
        vllm_utils._merge_multimodal_embeddings = orig_vllm_merge
        thinker_mod._merge_multimodal_embeddings = orig_thinker_merge
        thinker_mod.check_interleaved_audio_video = orig_check_interleaved
        thinker_mod.merge_interleaved_embeddings = orig_merge_interleaved


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="vLLM-Omni 0.32.0rc1 takes ceil(feature_lens / 100) on the device, "
    "which TPU rounds up to 9 windows for 800 frames; the split then fails",
)
def test_upstream_forward_miscounts_exact_window_multiples(encoders) -> None:
    tpu, _, upstream_forward = encoders
    try:
        out = _encode(tpu, 800, forward=upstream_forward)
    except RuntimeError as exc:
        assert "split_with_sizes expects split_sizes to sum exactly to 800" not in str(
            exc
        ), str(exc)
        raise
    assert out.shape[0] == int(_get_feat_extract_output_lengths(800))


@pytest.mark.parametrize("n_frames", PARITY_LENGTHS)
def test_audio_encoder_matches_cpu_reference(encoders, n_frames: int) -> None:
    tpu, cpu, _ = encoders
    captured: dict[str, tuple[tuple, dict, torch.Tensor]] = {}

    def _capture(name):
        def hook(module, args, kwargs, output):
            captured[name] = (args, kwargs, output)

        return hook

    hooks = [
        cpu.layers[0].register_forward_hook(_capture("layer"), with_kwargs=True),
        cpu.layers[0].self_attn.register_forward_hook(
            _capture("attn"), with_kwargs=True
        ),
        tpu.layers[0].register_forward_hook(_capture("tpu_layer"), with_kwargs=True),
    ]
    try:
        reference = _encode(cpu, n_frames)
        out = _encode(tpu, n_frames)
    finally:
        for hook in hooks:
            hook.remove()
    expected_tokens = int(_get_feat_extract_output_lengths(n_frames))
    assert out.device.type == "tpu"
    assert out.shape == (expected_tokens, AUDIO_CONFIG.output_dim)
    assert out.shape == reference.shape

    def _on_tpu(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.device("tpu"), torch.bfloat16)

    (layer_in, _, _), _, layer_out = captured["layer"]
    _, attn_kwargs, attn_out = captured["attn"]
    (tpu_layer_in, tpu_cu_seqlens), tpu_layer_kwargs, _ = captured["tpu_layer"]
    padded_tokens = 512 if n_frames <= 3200 else 640
    assert tpu_layer_in.shape == (padded_tokens, AUDIO_CONFIG.d_model)
    assert tpu_cu_seqlens.shape == (128,)
    assert tpu_layer_kwargs.get("max_seqlen") is None

    with torch.no_grad():
        tpu_attn_out = tpu.layers[0].self_attn(
            _pad_rows(_on_tpu(attn_kwargs["hidden_states"]), padded_tokens),
            tpu_cu_seqlens,
            max_seqlen=None,
        )[:expected_tokens]
        tpu_layer_out = tpu.layers[0](
            _pad_rows(_on_tpu(layer_in), padded_tokens),
            tpu_cu_seqlens,
            max_seqlen=None,
        )[:expected_tokens]
        h = tpu.ln_post(_on_tpu(layer_out))
        tpu_proj_out = tpu.proj2(tpu.act(tpu.proj1(h)[0]))[0]

    tag = f" ({n_frames} frames)"
    for name, actual, expected in [
        (
            "conv stack + positional embedding",
            tpu_layer_in[:expected_tokens],
            layer_in,
        ),
        ("layers.0.self_attn", tpu_attn_out, attn_out),
        ("layers.0", tpu_layer_out, layer_out),
        ("ln_post + proj", tpu_proj_out, reference),
    ]:
        _check(actual, expected, name + tag, COMPONENT_LIMITS[name])
    _check(out, reference, "audio encoder end to end" + tag, END_TO_END_LIMITS)


def test_audio_encoder_multi_clip_batch_matches_cpu_reference(encoders) -> None:
    """Two-clip packed batch [800, 801] via both forward and metadata APIs."""
    tpu, cpu, _ = encoders
    lengths = [800, 801]
    reference = _encode(cpu, lengths)
    out = _encode(tpu, lengths)
    _check(out, reference, "audio encoder batch [800, 801]", END_TO_END_LIMITS)

    features = torch.cat([_sample_features(n) for n in lengths], dim=1).to(
        device=torch.device("tpu"), dtype=torch.bfloat16
    )
    metadata = tpu.prepare_encoder_metadata(lengths)
    with torch.no_grad():
        out_meta = tpu.forward_with_encoder_metadata(features, metadata)
    assert metadata["aftercnn_lens"] == [
        int(_get_feat_extract_output_lengths(n)) for n in lengths
    ]
    assert torch.equal(out_meta, out)


@pytest.mark.parametrize(
    ("grid", "expected_chunks"),
    [
        ([[1, 20, 30]], [16384]),
        ([[5, 64, 64]], [16384, 16384]),
    ],
)
def test_vision_encoder_matches_cpu_reference(
    encoders, grid: list[list[int]], expected_chunks: list[int]
) -> None:
    """Patched TPU vision encoder matches unpatched float32 CPU reference."""
    vit_cls = Qwen3Omni_VisionTransformer
    torch.manual_seed(0)
    cpu_vit = vit_cls(VISION_CONFIG, norm_eps=1e-6).eval()
    cpu_vit.forward = MethodType(vit_cls._tpu_orig_forward, cpu_vit)
    init_test_weights(cpu_vit, seed=0)
    assert all(torch.isfinite(p).all() for p in cpu_vit.parameters())
    use_native_ops_on_cpu(cpu_vit)

    tpu_vit = vit_cls(VISION_CONFIG, norm_eps=1e-6).eval()
    tpu_vit.load_state_dict(cpu_vit.state_dict())
    tpu_vit = tpu_vit.to(torch.device("tpu")).to(torch.bfloat16)

    seen_shapes: list[int] = []
    hook = tpu_vit.blocks[0].register_forward_hook(
        lambda _mod, args, _out: seen_shapes.append(int(args[0].shape[0]))
    )
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
    try:
        with torch.no_grad():
            reference = cpu_vit(pixels, grid_tensor)
            out = tpu_vit(
                pixels.to(torch.device("tpu"), torch.bfloat16),
                grid_tensor,
            )
    finally:
        hook.remove()

    assert out.device.type == "tpu"
    assert seen_shapes == expected_chunks
    assert out.shape == (
        total_patches // VISION_CONFIG.spatial_merge_size**2,
        VISION_CONFIG.out_hidden_size
        * (1 + len(VISION_CONFIG.deepstack_visual_indexes)),
    )
    _check(out, reference, f"vision encoder {grid}", (2e-2, 1.5e-2))


def test_multimodal_merge_on_tpu_matches_cpu_reference(encoders) -> None:
    """Patched static multimodal merge on TPU matches index-masked CPU reference."""
    from vllm.multimodal.utils import set_mm_embedding_modality

    tpu_dev = torch.device("tpu")
    is_video = torch.tensor([False, True, True, False, False, False], dtype=torch.bool)
    is_audio = torch.tensor([False, False, False, False, True, False], dtype=torch.bool)
    is_mm = is_video | is_audio
    base_embeds = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16)
    visual_embeds = torch.tensor(
        [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]], dtype=torch.bfloat16
    )
    audio_embeds = torch.tensor([[90.0, 100.0, 110.0, 120.0]], dtype=torch.bfloat16)

    expected = base_embeds.clone()
    expected[1:3] = visual_embeds
    expected[4:5] = audio_embeds

    merged_tpu = thinker_mod.merge_interleaved_embeddings(
        inputs_embeds=base_embeds.to(tpu_dev),
        multimodal_embeddings=[
            set_mm_embedding_modality(visual_embeds.to(tpu_dev), "video"),
            set_mm_embedding_modality(audio_embeds.to(tpu_dev), "audio"),
        ],
        is_video=is_video.to(tpu_dev),
        is_audio=is_audio.to(tpu_dev),
        is_multimodal=is_mm.to(tpu_dev),
    )
    assert merged_tpu.device.type == "tpu"
    assert torch.equal(merged_tpu.cpu(), expected)


def test_code_predictor_wrapper_on_tpu_matches_cpu_reference(encoders) -> None:
    """Patched CodePredictorWrapper runs residual AR on CPU and returns TPU tensors."""
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
        config = SimpleNamespace(architectures=["Qwen3OmniMoeForConditionalGeneration"])
        assert maybe_patch_qwen3_omni_talker(config) is True

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
