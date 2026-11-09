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
"""CPU tests for Muse Spark Open multimodal components on TorchTPU."""

from __future__ import annotations

import math
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_torchtpu.models.vllm.muse_spark.audio import (
    MuseSparkAudioCausalConv1d,
    MuseSparkAudioRMSNorm,
    MuseSparkAudioTower,
)
from vllm_torchtpu.models.vllm.muse_spark.configuration import (
    MUSE_SPARK_CONFIGS,
    MuseSparkAudioConfig,
    MuseSparkConfig,
    MuseSparkTextConfig,
    MuseSparkVisionConfig,
    register_muse_spark_configs,
)
from vllm_torchtpu.models.vllm.muse_spark.processing import (
    AUDIO_END,
    AUDIO_PLACEHOLDER,
    AUDIO_START,
    DEFAULT_IMAGE_PROCESSOR_CONFIG,
    DEFAULT_VIDEO_PROCESSOR_CONFIG,
    IMAGE_END,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
    VIDEO_END,
    VIDEO_PLACEHOLDER,
    VIDEO_SEPARATOR,
    VIDEO_START,
    MuseSparkDummyInputsBuilder,
    MuseSparkImageProcessor,
    MuseSparkMultiModalProcessor,
    MuseSparkProcessingInfo,
    MuseSparkVideoProcessor,
    smart_resize,
    video_prompt_text,
)
from vllm_torchtpu.models.vllm.muse_spark.vision import (
    MuseSparkPerception,
    compute_vision_geometry,
    weightless_rms_norm,
)


@pytest.fixture(autouse=True)
def _cpu_model_tests(monkeypatch):
    tpu = getattr(torch, "tpu", None)
    if tpu is not None:
        monkeypatch.setattr(tpu, "is_available", lambda: False)
        monkeypatch.setattr(tpu, "current_device", lambda: 0)
        monkeypatch.setattr(tpu, "device_count", lambda: 0)


@pytest.fixture
def _single_rank_dist(monkeypatch):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_name", "cpu", raising=False)
    monkeypatch.setattr(current_platform, "device_type", "cpu", raising=False)

    from vllm_torchtpu.layers.adapter.vision_attention import TpuMMEncoderAttention

    def _cpu_varlen_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del max_seqlen, sequence_lengths
        if cu_seqlens is None:
            return torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                scale=self.scale,
            ).transpose(1, 2)
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        chunks = []
        for q, k, v in zip(
            query.split(lens, 1), key.split(lens, 1), value.split(lens, 1)
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                scale=self.scale,
            )
            chunks.append(out.transpose(1, 2))
        return torch.cat(chunks, dim=1)

    monkeypatch.setattr(TpuMMEncoderAttention, "forward_oot", _cpu_varlen_sdpa)

    if not torch.distributed.is_initialized():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(1, 1)


def _find_ckpt_path(filename: str) -> Path | None:
    ckpt_dir = os.environ.get("MUSE_SPARK_CKPT_DIR", "")
    if not ckpt_dir:
        return None
    p = Path(ckpt_dir) / filename
    return p if p.exists() else None


def test_register_muse_spark_configs() -> None:
    register_muse_spark_configs()
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    for k, cls in MUSE_SPARK_CONFIGS.items():
        assert CONFIG_MAPPING._extra_content.get(k) is cls


def test_config_from_json_dict() -> None:
    cfg = MuseSparkConfig()
    assert cfg.model_type == "muse_spark"
    assert isinstance(cfg.text_config, MuseSparkTextConfig)
    assert isinstance(cfg.vision_config, MuseSparkVisionConfig)
    assert isinstance(cfg.audio_config, MuseSparkAudioConfig)
    assert cfg.get_text_config() is cfg.text_config
    assert cfg.image_token_id == 200092
    assert cfg.video_token_id == 200091
    assert cfg.audio_token_id == 200038
    assert cfg.audio_config.samples_per_token == 1920


def test_config_loads_released_checkpoint() -> None:
    cfg_path = _find_ckpt_path("config.json")
    if cfg_path is None:
        pytest.skip("config.json not found in test environment")
    register_muse_spark_configs()
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(cfg_path.parent))
    assert isinstance(cfg, MuseSparkConfig)
    text = cfg.get_text_config()
    assert text.hidden_size == 8192
    assert text.num_hidden_layers == 62
    assert text.num_attention_heads == 128
    assert text.num_key_value_heads == 16
    assert text.head_dim == 64
    assert text.moe_intermediate_size == 4096
    assert text.num_local_experts == 256
    assert text.num_experts_per_tok == 8
    assert text.moe_hidden_size == 4096
    assert math.isclose(text.residual_gate_temperature, 0.3)
    assert math.isclose(text.qk_scale_factor, 5.4730064863875)
    assert math.isclose(text.output_multiplier, 0.1767766952966369)
    assert math.isclose(text.post_norm_eps, 1e-8)
    assert math.isclose(text.final_logit_softcapping, 20.0)
    assert len(text.layer_types) == 62
    assert len(text.mlp_layer_types) == 62
    assert len(text.layer_rope_theta) == 62

    vision = cfg.vision_config
    assert vision.hidden_size == 1536
    assert vision.num_hidden_layers == 50
    assert vision.patch_size == 14
    assert vision.merge_size == 2
    assert vision.patch_temporal == 2
    assert len(vision.layer_types) == 50

    audio = cfg.audio_config
    assert audio.patch_sizes == [16, 6, 5, 4]
    assert audio.samples_per_token == 1920
    assert audio.out_hidden_size == 8192


def test_config_dimension_validation() -> None:
    with pytest.raises(ValueError, match="audio_config.out_hidden_size"):
        MuseSparkConfig(
            text_config=dict(hidden_size=4096),
            audio_config=dict(out_hidden_size=8192),
        )

    with pytest.raises(ValueError, match="out_hidden_size"):
        MuseSparkConfig(
            vision_config=dict(hidden_size=1024, merge_size=2),
            out_hidden_size=2048,
        )


def test_weightless_rms_norm() -> None:
    x_f32 = torch.randn(3, 8, dtype=torch.float32)
    eps = 1e-5
    out_f32 = weightless_rms_norm(x_f32, eps)
    assert out_f32.dtype == torch.float32
    expected = x_f32 * torch.pow(x_f32.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    torch.testing.assert_close(out_f32, expected, rtol=1e-5, atol=1e-6)

    x_bf16 = x_f32.to(torch.bfloat16)
    out_bf16 = weightless_rms_norm(x_bf16, eps)
    assert out_bf16.dtype == torch.bfloat16
    xb = x_bf16.float()
    expected_bf16 = xb * torch.pow(xb.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    torch.testing.assert_close(out_bf16, expected_bf16.to(torch.bfloat16))


def test_smart_resize() -> None:
    h, w = smart_resize(300, 500, patch_size=28, max_tokens=4096)
    assert h % 28 == 0
    assert w % 28 == 0
    assert (h // 28) * (w // 28) <= 4096
    # Ratio should be close to 500 / 300.
    ratio = (w / 28) / (h / 28)
    assert abs(ratio - 500 / 300) < 0.2


def test_vision_geometry_computation() -> None:
    grid_thw = [(1, 4, 6), (2, 2, 4)]
    geo = compute_vision_geometry(
        grid_thw, pos_emb_side=4, window_patches=4, merge_size=2
    )
    assert geo.num_patches == 1 * 4 * 6 + 2 * 2 * 4
    assert geo.window_index.shape[0] == geo.num_patches
    assert sum(geo.window_seqlens) == geo.num_patches
    assert sum(geo.full_seqlens) == geo.num_patches
    assert geo.rope_positions.shape == (geo.num_patches, 2)
    assert geo.interp_indices.shape == (geo.num_patches, 4)
    assert geo.interp_weights.shape == (geo.num_patches, 4)
    assert (geo.interp_weights >= 0).all()
    assert (geo.interp_weights.sum(-1) <= 1.0001).all()
    # Taps that fall outside the table are zero weighted (grid_sample padding).
    assert (geo.interp_weights.sum(-1) > 0.5).all()


def test_audio_tower_shape_and_padding() -> None:
    cfg = MuseSparkAudioConfig(
        hidden_sizes=[8, 16],
        patch_sizes=[4, 3],
        depths=[1, 1],
        num_hidden_layers=2,
        out_hidden_size=32,
        conv_kernel_size=3,
    )
    assert cfg.samples_per_token == 12
    tower = MuseSparkAudioTower(cfg).eval()

    # Unaligned length raises error.
    with pytest.raises(ValueError, match="not a multiple"):
        tower(torch.randn(1, 25))

    x = torch.randn(2, 36)
    out = tower(x)
    assert out.shape == (2, 3, 32)

    # encode_clips pads arbitrary lengths to token boundary.
    clips = [torch.randn(25), torch.randn(12)]
    encoded = tower.encode_clips(clips)
    assert len(encoded) == 2
    assert encoded[0].shape == (3, 32)
    assert encoded[1].shape == (1, 32)


def test_audio_conv1d_causality() -> None:
    conv = MuseSparkAudioCausalConv1d(
        in_channels=4, out_channels=4, kernel_size=5, stride=1, dilation=1
    )
    assert conv.left_pad == 4
    x = torch.randn(1, 4, 20)
    out = conv(x)
    assert out.shape == (1, 4, 20)

    # Modifying future values must not affect earlier steps.
    x_mod = x.clone()
    x_mod[:, :, 10:] = torch.randn(1, 4, 10)
    out_mod = conv(x_mod)
    torch.testing.assert_close(out[:, :, :10], out_mod[:, :, :10])


def test_audio_rms_norm() -> None:
    norm = MuseSparkAudioRMSNorm(16, eps=1e-6)
    x = torch.randn(2, 5, 16)
    out = norm(x)
    assert out.shape == x.shape
    xf = x.float()
    expected = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)
    torch.testing.assert_close(out, expected * norm.weight, rtol=1e-4, atol=1e-5)


def test_perception_forward_and_load_weights(_single_rank_dist) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config

    cfg = MuseSparkConfig(
        text_config=dict(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            num_key_value_heads=2,
            head_dim=16,
        ),
        vision_config=dict(
            hidden_size=32,
            num_attention_heads=2,
            num_hidden_layers=3,
            intermediate_size=48,
            pos_emb_height=4,
            pos_emb_width=4,
            patch_size=2,
        ),
        audio_config=dict(
            hidden_sizes=[8, 16],
            patch_sizes=[4, 3],
            depths=[1, 1],
            num_hidden_layers=2,
            out_hidden_size=64,
        ),
        out_hidden_size=128,
        projector_hidden_size=40,
    )
    with set_current_vllm_config(VllmConfig()):
        perception = MuseSparkPerception(cfg, prefix="model")
        assert perception.vision_tower is not None
        assert perception.audio_tower is not None

        grid = torch.tensor([[1, 4, 6], [2, 2, 4]])
        num_patches = int(grid.prod(-1).sum())
        px = torch.randn(num_patches, 2 * 3 * 4)
        clips = [torch.randn(25), torch.randn(12)]
        out = perception.embed_multimodal(
            pixel_values=px,
            image_grid_thw=grid,
            input_values=clips,
        )
        assert len(out) == 4
        assert out[0].shape == (6, 64)
        assert out[1].shape == (4, 64)
        assert out[2].shape == (3, 64)
        assert out[3].shape == (1, 64)

        # Test weight loading.
        dummy_weights = [
            (
                "model.vision_tower.patch_embedder.patch_embedding.weight",
                torch.randn_like(
                    perception.vision_tower.patch_embedder.patch_embedding.weight
                ),
            ),
            ("model.vision_tower.layers.0.attn.q_proj.weight", torch.randn(32, 32)),
            ("model.vision_tower.layers.0.attn.k_proj.weight", torch.randn(32, 32)),
            ("model.vision_tower.layers.0.attn.v_proj.weight", torch.randn(32, 32)),
            ("model.vision_tower.layers.0.attn.q_proj.bias", torch.randn(32)),
            ("model.vision_tower.layers.0.attn.k_proj.bias", torch.randn(32)),
            ("model.vision_tower.layers.0.attn.v_proj.bias", torch.randn(32)),
            (
                "model.vision_projection.weight",
                torch.randn_like(perception.vision_projection.weight),
            ),
            (
                "model.audio_tower.proj.weight",
                torch.randn_like(perception.audio_tower.proj.weight),
            ),
            ("model.language_model.embed_tokens.weight", torch.randn(128, 64)),
        ]
        loaded = perception.load_weights(dummy_weights)
        assert "vision_tower.patch_embedder.patch_embedding.weight" in loaded
        assert "vision_tower.layers.0.attn.qkv_proj.weight" in loaded
        assert "vision_tower.layers.0.attn.qkv_proj.bias" in loaded
        assert "vision_projection.weight" in loaded
        assert "audio_tower.proj.weight" in loaded
        # Language model weight must not be claimed.
        assert not any("embed_tokens" in name for name in loaded)


def test_image_processor_methods() -> None:
    proc = MuseSparkImageProcessor(**DEFAULT_IMAGE_PROCESSOR_CONFIG)
    img = Image.fromarray((np.random.rand(120, 160, 3) * 255).astype(np.uint8))
    out = proc(images=[img], return_tensors="pt")
    assert "pixel_values" in out
    assert "image_grid_thw" in out
    grid = out["image_grid_thw"][0].tolist()
    assert grid[0] == 1
    num_patches = proc.get_number_of_image_patches(120, 160)
    assert num_patches == grid[1] * grid[2]
    assert out["pixel_values"].shape[0] == num_patches


def test_video_processor_methods() -> None:
    proc = MuseSparkVideoProcessor(**DEFAULT_VIDEO_PROCESSOR_CONFIG)
    frames = (np.random.rand(6, 80, 120, 3) * 255).astype(np.uint8)
    from transformers.video_utils import VideoMetadata

    meta = VideoMetadata(total_num_frames=6, fps=2.0, frames_indices=list(range(6)))
    out = proc(
        videos=[frames],
        video_metadata=[meta],
        do_sample_frames=False,
        return_metadata=True,
        return_tensors="pt",
    )
    assert "pixel_values_videos" in out
    assert "video_grid_thw" in out
    grid = out["video_grid_thw"][0].tolist()
    assert grid[0] == 3
    assert out["pixel_values_videos"].shape[0] == grid[0] * grid[1] * grid[2]


def test_video_prompt_text_formatting() -> None:
    text = video_prompt_text(
        grid_thw=(3, 4, 4),
        timestamps=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        merge_size=2,
        temporal_patch_size=2,
        timestamp_format="Time: {timestamp:.1f}s",
    )
    assert text.startswith(VIDEO_START)
    assert text.endswith(VIDEO_END)
    assert "Time: 0.0s" in text
    assert "Time: 1.0s" in text
    assert "Time: 2.0s" in text
    assert VIDEO_SEPARATOR in text
    assert text.count(VIDEO_PLACEHOLDER) == 3 * (4 * 4 // 4)


_STUB_SPECIAL_TOKENS = [
    IMAGE_PLACEHOLDER,
    VIDEO_PLACEHOLDER,
    AUDIO_PLACEHOLDER,
    IMAGE_START,
    IMAGE_END,
    VIDEO_START,
    VIDEO_END,
    VIDEO_SEPARATOR,
    AUDIO_START,
    AUDIO_END,
]


def _stub_tokenizer(tmp_dir: Path):
    """Word-level tokenizer holding the Muse Spark multimodal special tokens."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0}
    for word in ("Hello", "and", "done", "a", "b", "c", "d", "<", ">", "."):
        vocab[word] = len(vocab)
    for i in range(10):
        vocab[str(i)] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(_STUB_SPECIAL_TOKENS)}
    )
    tokenizer.save_pretrained(str(tmp_dir))
    return tokenizer


def _make_processor(tmp_path: Path | None = None):
    """Processor over the released checkpoint, else a stub tokenizer and the
    default config (processor_config.json absent, so built-in defaults)."""
    tok_path = _find_ckpt_path("tokenizer.json")
    cfg_path = _find_ckpt_path("config.json")
    register_muse_spark_configs()
    if tok_path is not None and cfg_path is not None:
        from transformers import AutoConfig, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tok_path.parent))
        hf_config = AutoConfig.from_pretrained(str(cfg_path.parent))
        model_dir = str(cfg_path.parent)
    elif tmp_path is not None:
        tokenizer = _stub_tokenizer(tmp_path)
        hf_config = MuseSparkConfig()
        model_dir = str(tmp_path)
    else:
        pytest.skip("Checkpoint files not found for processor test")

    from vllm.config.multimodal import MultiModalConfig
    from vllm.multimodal.processing.context import InputProcessingContext

    mm_config = MultiModalConfig()
    model_config = SimpleNamespace(
        hf_config=hf_config,
        model=model_dir,
        revision=None,
        max_model_len=100000,
        encoder_config=None,
        multimodal_config=mm_config,
        get_multimodal_config=lambda: mm_config,
    )
    ctx = InputProcessingContext(model_config=model_config, tokenizer=tokenizer)
    info = MuseSparkProcessingInfo(ctx)
    processor = MuseSparkMultiModalProcessor(info, MuseSparkDummyInputsBuilder(info))
    return tokenizer, info, processor


def test_processor_matches_hf_reference_if_available() -> None:
    """Token ids and tensors equal transformers main MuseSparkProcessor."""
    ref_dir = os.environ.get("MUSE_SPARK_REF_DIR", "")
    ref_path = Path(ref_dir) / "proc_ref.pt" if ref_dir else None
    if ref_path is None or not ref_path.exists():
        pytest.skip("No precomputed HF processor reference found")
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    tokenizer, info, processor = _make_processor()
    ref = torch.load(ref_path, weights_only=False)
    prompt = tokenizer.encode(
        "a<|patch|>b<|video|>c<|audio|>d<|patch|>", add_special_tokens=False
    )
    video_meta = {
        "fps": 3.0,
        "total_num_frames": 7,
        "frames_indices": list(range(7)),
        "do_sample_frames": False,
    }
    mm_data = {
        "image": [Image.fromarray(ref["img1"]), Image.fromarray(ref["img2"])],
        "video": [(ref["vid"], video_meta)],
        "audio": [(ref["wav"], 24000)],
    }
    items = info.parse_mm_data(mm_data)
    result = processor.apply(
        ProcessorInputs(prompt=prompt, mm_data_items=items), TimingContext()
    )
    assert result["prompt_token_ids"] == ref["ids"]
    data = result["mm_kwargs"].get_data()
    for key in (
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    ):
        torch.testing.assert_close(data[key], ref[key], rtol=0, atol=0)
    np.testing.assert_array_equal(data["input_values"][0].numpy(), ref["wav"])


def test_multimodal_processor_apply(tmp_path: Path) -> None:
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    tokenizer, info, processor = _make_processor(tmp_path)

    prompt = tokenizer.encode(
        f"Hello {IMAGE_PLACEHOLDER} and {VIDEO_PLACEHOLDER} and "
        f"{AUDIO_PLACEHOLDER} done",
        add_special_tokens=False,
    )
    img = Image.fromarray((np.random.rand(60, 80, 3) * 255).astype(np.uint8))
    vid = (np.random.rand(4, 56, 56, 3) * 255).astype(np.uint8)
    wav = np.random.randn(3840).astype(np.float32)
    mm_data = {
        "image": [img],
        "video": [
            (vid, {"fps": 2.0, "total_num_frames": 4, "do_sample_frames": False})
        ],
        "audio": [(wav, 24000)],
    }
    items = info.parse_mm_data(mm_data)
    inputs = ProcessorInputs(prompt=prompt, mm_data_items=items)
    result = processor.apply(inputs, TimingContext())

    assert "prompt_token_ids" in result
    token_ids = result["prompt_token_ids"]
    assert tokenizer.convert_tokens_to_ids(IMAGE_START) in token_ids
    assert tokenizer.convert_tokens_to_ids(IMAGE_END) in token_ids
    assert tokenizer.convert_tokens_to_ids(VIDEO_START) in token_ids
    assert tokenizer.convert_tokens_to_ids(VIDEO_END) in token_ids
    assert tokenizer.convert_tokens_to_ids(AUDIO_START) in token_ids
    assert tokenizer.convert_tokens_to_ids(AUDIO_END) in token_ids

    placeholders = result["mm_placeholders"]
    image_proc = info.get_image_processor()
    grid_h, grid_w = image_proc.get_grid(60, 80)
    n_image = grid_h * grid_w // 4
    # Placeholder ranges cover start + tokens + end; embeds only the tokens.
    assert placeholders["image"][0].length == n_image + 2
    assert placeholders["audio"][0].length == 2 + 2
    image_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    audio_id = tokenizer.convert_tokens_to_ids(AUDIO_PLACEHOLDER)
    assert token_ids.count(image_id) == n_image
    assert token_ids.count(audio_id) == 2
    assert "video" in placeholders

    # The encoder needs rescaled and normalized pixels even though vLLM
    # defaults mm_device_do_normalize to True.
    data = result["mm_kwargs"].get_data()
    assert data["pixel_values"].dtype == torch.float32
    assert data["pixel_values"].abs().max() <= 1.0
    assert data["pixel_values_videos"].abs().max() <= 1.0


def test_embed_input_ids_merges_multimodal_rows() -> None:
    from vllm_torchtpu.models.vllm.muse_spark import MuseSparkForConditionalGeneration

    model = MuseSparkForConditionalGeneration.__new__(MuseSparkForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model._has_oov_mm_tokens = False
    model.model = SimpleNamespace(
        embed_input_ids=lambda ids: torch.stack((ids.float(), ids.float() + 10), -1)
    )
    input_ids = torch.tensor([1, 2, 3, 4])
    is_mm = torch.tensor([False, True, True, False])
    media = [torch.tensor([[20.0, 21.0]]), torch.tensor([[30.0, 31.0]])]
    torch.testing.assert_close(
        model.embed_input_ids(input_ids, media, is_multimodal=is_mm),
        torch.tensor([[1.0, 11.0], [20.0, 21.0], [30.0, 31.0], [4.0, 14.0]]),
    )
    torch.testing.assert_close(
        model.embed_input_ids(input_ids, None, is_multimodal=is_mm),
        torch.tensor([[1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [4.0, 14.0]]),
    )
    with pytest.raises(ValueError, match="1 multimodal tokens to 2"):
        model.embed_input_ids(input_ids, media[:1], is_multimodal=is_mm)


def test_reference_parity_if_available(_single_rank_dist) -> None:
    """Test numeric equality against precomputed HF reference if present."""
    ref_dir = os.environ.get("MUSE_SPARK_REF_DIR", "")
    ref_paths = (
        [Path(ref_dir) / "ref_float32.pt", Path(ref_dir) / "ref_bfloat16.pt"]
        if ref_dir
        else []
    )
    available = [p for p in ref_paths if p.exists()]
    if not available:
        pytest.skip("No precomputed HF reference files found")

    from vllm.config import VllmConfig, set_current_vllm_config

    for ref_path in available:
        ref = torch.load(ref_path, weights_only=False)
        dtype = ref["px"].dtype
        d = {k: v for k, v in ref["cfg"].items() if k != "model_type"}
        cfg = MuseSparkConfig(**d)
        with set_current_vllm_config(VllmConfig()):
            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                perception = MuseSparkPerception(cfg, prefix="model")
            finally:
                torch.set_default_dtype(default_dtype)
            perception.load_weights(("model." + k, v) for k, v in ref["sd"].items())
            perception = perception.to(dtype).eval()
            with torch.no_grad():
                out = perception.embed_multimodal(
                    pixel_values=ref["px"],
                    image_grid_thw=ref["grid"],
                    pixel_values_videos=ref["vpx"],
                    video_grid_thw=ref["vgrid"],
                    input_values=[ref["wav"][0], ref["wav"][1, :25]],
                )
            expected = (
                list(ref["img"]) + list(ref["vid"]) + [ref["aud"][:3], ref["aud"][3:]]
            )
            tol = 1e-4 if dtype == torch.float32 else 1e-2
            for actual, exp in zip(out, expected):
                assert actual.shape == exp.shape
                diff = (actual.float() - exp.float()).abs().max().item()
                assert diff < tol, f"{ref_path.name} diff {diff} exceeded {tol}"
