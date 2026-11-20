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
"""Tests for the Qwen3-VL processor patches."""

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from vllm.config import ModelConfig
from vllm.model_executor.models import qwen3_vl
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing.context import InputProcessingContext

from vllm_torchtpu.models.vllm import qwen3_vl_patch


def test_maybe_patch_qwen3_vl_video_prompt_expands_trained_format(
    monkeypatch: pytest.MonkeyPatch,
):
    # The patch rebinds a class attribute; restore it after the test.
    processor_cls = qwen3_vl.Qwen3VLMultiModalProcessor
    monkeypatch.setattr(
        processor_cls,
        "_expands_only_video_token",
        processor_cls._expands_only_video_token,
    )
    monkeypatch.setattr(
        processor_cls, "_tpu_trained_video_prompt", False, raising=False
    )
    # Only the configs and tokenizer are needed, not the weights.
    path = snapshot_download(
        "Qwen/Qwen3-VL-8B-Instruct", allow_patterns=["*.json", "*.txt"]
    )
    model_config = ModelConfig(model=path, limit_mm_per_prompt={"image": 0, "video": 1})
    assert qwen3_vl_patch.maybe_patch_qwen3_vl_video_prompt(model_config) is None

    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    # 4 frames at 2 fps, no resampling: two temporal patches of 2 frames.
    video = np.zeros((4, 64, 64, 3), dtype=np.uint8)
    metadata = {
        "fps": 2.0,
        "duration": 2.0,
        "total_num_frames": 4,
        "frames_indices": [0, 1, 2, 3],
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    prompt = "<|vision_start|><|video_pad|><|vision_end|>Hi"
    out = processor(
        prompt,
        mm_items=processor.info.parse_mm_data({"video": [(video, metadata)]}),
        hf_processor_mm_kwargs={},
    )
    tokenizer = processor.info.get_tokenizer()
    video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    text = tokenizer.decode([t for t in out["prompt_token_ids"] if t != video_token_id])
    # Per frame group: timestamp, then its own vision markers; no outer pair.
    assert text == (
        "<0.2 seconds><|vision_start|><|vision_end|>"
        "<1.2 seconds><|vision_start|><|vision_end|>Hi"
    )


def test_maybe_patch_qwen3_vl_image_hf_text_keeps_outputs(
    monkeypatch: pytest.MonkeyPatch,
):
    """Images skip the HF dummy prompt; the prompt ids and image fields stay."""
    processor_cls = qwen3_vl.Qwen3VLMultiModalProcessor
    monkeypatch.setattr(processor_cls, "_get_hf_mm_text", processor_cls._get_hf_mm_text)
    monkeypatch.setattr(
        processor_cls, "_tpu_image_hf_text_skipped", False, raising=False
    )
    path = snapshot_download(
        "Qwen/Qwen3-VL-8B-Instruct", allow_patterns=["*.json", "*.txt"]
    )
    model_config = ModelConfig(model=path, limit_mm_per_prompt={"image": 2, "video": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    rng = np.random.default_rng(0)
    images = [
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        for h, w in ((64, 96), (128, 64))
    ]
    prompt = "<|vision_start|><|image_pad|><|vision_end|>" * 2 + "Hi"

    def run():
        mm_items = processor.info.parse_mm_data({"image": images})
        return processor(prompt, mm_items=mm_items, hf_processor_mm_kwargs={})

    want = run()
    assert qwen3_vl_patch.maybe_patch_qwen3_vl_image_hf_text(model_config) is None
    assert processor._get_hf_mm_text({"image": 2}) is None
    assert processor._get_hf_mm_text({"image": 1, "video": 1})
    got = run()
    assert got["prompt_token_ids"] == want["prompt_token_ids"]
    for got_item, want_item in zip(
        got["mm_kwargs"]["image"], want["mm_kwargs"]["image"], strict=True
    ):
        assert got_item.keys() == want_item.keys()
        for key, elem in want_item.items():
            assert torch.equal(got_item[key].data, elem.data)


def test_maybe_patch_qwen3_vl_video_hf_text_keeps_outputs(
    monkeypatch: pytest.MonkeyPatch,
):
    """Videos skip the HF text vLLM drops; the prompt ids and video fields stay."""
    processor_cls = qwen3_vl.Qwen3VLMultiModalProcessor
    monkeypatch.setattr(
        processor_cls,
        "_apply_hf_processor_main",
        processor_cls._apply_hf_processor_main,
    )
    monkeypatch.setattr(
        processor_cls, "_tpu_video_hf_text_skipped", False, raising=False
    )
    monkeypatch.setattr(
        InputProcessingContext,
        "call_hf_processor",
        InputProcessingContext.call_hf_processor,
    )
    path = snapshot_download(
        "Qwen/Qwen3-VL-8B-Instruct", allow_patterns=["*.json", "*.txt"]
    )
    model_config = ModelConfig(model=path, limit_mm_per_prompt={"image": 1, "video": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    rng = np.random.default_rng(0)
    video = rng.integers(0, 256, (4, 64, 96, 3), dtype=np.uint8)
    metadata = {
        "fps": 2.0,
        "duration": 2.0,
        "total_num_frames": 4,
        "frames_indices": [0, 1, 2, 3],
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    image = Image.fromarray(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8))
    prompt = (
        "<|vision_start|><|image_pad|><|vision_end|>"
        "<|vision_start|><|video_pad|><|vision_end|>Hi"
    )

    def run():
        mm_items = processor.info.parse_mm_data(
            {"image": [image], "video": [(video, metadata)]}
        )
        return processor(prompt, mm_items=mm_items, hf_processor_mm_kwargs={})

    want = run()
    assert qwen3_vl_patch.maybe_patch_qwen3_vl_video_hf_text(model_config) is None
    got = run()
    assert got["prompt_token_ids"] == want["prompt_token_ids"]
    for modality in ("image", "video"):
        for got_item, want_item in zip(
            got["mm_kwargs"][modality], want["mm_kwargs"][modality], strict=True
        ):
            assert got_item.keys() == want_item.keys()
            for key, elem in want_item.items():
                got_data = got_item[key].data
                if isinstance(elem.data, torch.Tensor):
                    assert torch.equal(got_data, elem.data)
                else:
                    assert got_data == elem.data


def test_budget_inputs_match_upstream_metadata_padded():
    """The budget-inputs hook gives upstream's encoder metadata in its first rows
    and zeros after them (on the CPU, where both build the tables)."""
    from types import MethodType, SimpleNamespace

    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    tower_cls = qwen3_vl.Qwen3_VisionTransformer
    grid_side, merge, hidden, rope = 8, 2, 32, 16
    torch.manual_seed(0)
    cache = torch.randn(64, rope)
    visual = SimpleNamespace(
        spatial_merge_size=merge,
        num_grid_per_side=grid_side,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        pos_embed=torch.nn.Embedding(grid_side**2, hidden).to(torch.bfloat16),
        rotary_pos_emb=SimpleNamespace(
            cos_sin_cache=cache, get_cos_sin=lambda n: cache[:n].chunk(2, dim=-1)
        ),
        _rot_pos_ids_cache={},
        attn_backend=AttentionBackendEnum.TORCH_SDPA,
        hidden_size=hidden,
        tp_size=1,
        fp8_padded_hidden_size=None,
    )
    visual.fast_pos_embed_interpolate = lambda grid: torch.cat(
        [
            qwen3_vl.pos_embed_interpolate_native(
                visual.pos_embed.weight, t, h, w, grid_side, merge, visual.dtype
            )
            for t, h, w in grid
        ]
    )
    for name in ("rot_pos_ids", "rot_pos_emb", "prepare_encoder_metadata"):
        setattr(visual, name, MethodType(getattr(tower_cls, name), visual))
    model_cls = qwen3_vl.Qwen3VLForConditionalGeneration
    model = SimpleNamespace(visual=visual)
    for name in (
        "get_input_modality",
        "_get_grid_thw_by_modality",
        "_get_pixel_values_by_modality",
    ):
        setattr(model, name, MethodType(getattr(model_cls, name), model))

    grid = [[1, 4, 6], [2, 6, 2]]
    rows = 64
    n = sum(t * h * w for t, h, w in grid)
    pixels = torch.randint(0, 256, (n, 12), dtype=torch.uint8)
    template = {"pos_embeds": torch.zeros(rows, hidden, dtype=torch.bfloat16)}
    got = qwen3_vl_patch._prepare_encoder_budget_inputs(
        model,
        {"pixel_values": pixels, "image_grid_thw": torch.tensor(grid)},
        template,
        max_batch_size=4,
        max_frames_per_batch=8,
    )
    want = visual.prepare_encoder_metadata(grid, max_batch_size=4)
    assert got["pixel_values"] is pixels
    for key in ("pos_embeds", "rotary_pos_emb_cos", "rotary_pos_emb_sin"):
        assert got[key].shape[0] == rows
        assert torch.equal(got[key][:n], want[key].to(got[key].dtype)), key
        assert not got[key][n:].any(), key
    for key in ("cu_seqlens", "max_seqlen"):
        assert torch.equal(torch.as_tensor(got[key]), torch.as_tensor(want[key])), key
    assert got["sequence_lengths"] is None and want["sequence_lengths"] is None
