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
"""Tests for the Qwen3-VL video prompt patch."""

import functools
from types import SimpleNamespace

import numpy as np
import pytest
from huggingface_hub import snapshot_download
from transformers import (
    Cosmos3OmniConfig,
    Qwen3_5Config,
    Qwen3VLConfig,
    Qwen3VLMoeConfig,
)
from vllm.config import ModelConfig
from vllm.model_executor.models import qwen3_vl
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_torchtpu.models.vllm import qwen3_vl_patch

QWEN3_VL = "Qwen/Qwen3-VL-8B-Instruct"
FULL_VIDEO = "<|vision_start|><|video_pad|><|vision_end|>"
# Two frame groups, each its timestamp and its own vision markers.
FRAME_GROUPS = (
    "<0.2 seconds><|vision_start|><|vision_end|>"
    "<1.2 seconds><|vision_start|><|vision_end|>"
)


@pytest.fixture(autouse=True)
def restore_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    # The patch rebinds a class attribute; restore it after each test.
    processor_cls = qwen3_vl.Qwen3VLMultiModalProcessor
    monkeypatch.setattr(
        processor_cls, "_postprocess_prompt", processor_cls._postprocess_prompt
    )


@functools.cache
def _model_config(model: str) -> ModelConfig:
    # Only the configs and tokenizer are needed, not the weights.
    path = snapshot_download(model, allow_patterns=["*.json", "*.txt"])
    return ModelConfig(model=path, limit_mm_per_prompt={"image": 0, "video": 1})


def _expanded_text(model: str, prompt: str) -> str:
    processor = MULTIMODAL_REGISTRY.create_processor(_model_config(model))
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
    out = processor(
        prompt,
        mm_items=processor.info.parse_mm_data({"video": [(video, metadata)]}),
        hf_processor_mm_kwargs={},
    )
    tokenizer = processor.info.get_tokenizer()
    video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    return tokenizer.decode([t for t in out["prompt_token_ids"] if t != video_token_id])


@pytest.mark.parametrize(
    ("hf_config", "expected"),
    [
        (Qwen3VLConfig(), None),
        (Qwen3VLMoeConfig(), None),
        (Cosmos3OmniConfig(), None),
        (Qwen3_5Config(), False),
    ],
    ids=lambda v: type(v).__name__,
)
def test_maybe_patch_qwen3_vl_video_prompt_selects_models(hf_config, expected):
    processor_cls = qwen3_vl.Qwen3VLMultiModalProcessor
    stock = processor_cls._postprocess_prompt
    model_config = SimpleNamespace(hf_config=hf_config)
    assert qwen3_vl_patch.maybe_patch_qwen3_vl_video_prompt(model_config) is expected
    assert (processor_cls._postprocess_prompt is stock) == (expected is False)


@pytest.mark.parametrize(
    ("model", "prompt", "expected"),
    [
        pytest.param(
            QWEN3_VL, f"{FULL_VIDEO}Hi", f"{FRAME_GROUPS}Hi", id="full-qwen3-vl"
        ),
        pytest.param(
            QWEN3_VL, "<|video_pad|>Hi", f"{FRAME_GROUPS}Hi", id="bare-qwen3-vl"
        ),
        # The patch changes a class Qwen3.5 shares; Qwen3.5 keeps the stock layout.
        pytest.param(
            "Qwen/Qwen3.5-0.8B",
            f"{FULL_VIDEO}Hi",
            f"<|vision_start|>{FRAME_GROUPS}<|vision_end|>Hi",
            id="full-qwen3.5",
        ),
    ],
)
def test_video_prompt_layout(model, prompt, expected):
    qwen3_vl_patch.maybe_patch_qwen3_vl_video_prompt(_model_config(QWEN3_VL))
    assert _expanded_text(model, prompt) == expected


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="vLLM 0.33.0 keeps an outer vision marker pair per video "
    "(https://github.com/vllm-project/vllm/issues/69473)",
)
def test_upstream_expands_trained_video_layout():
    """Upstream canary: passes once vLLM expands the trained layout itself; then
    delete the whole patch."""
    assert _expanded_text(QWEN3_VL, f"{FULL_VIDEO}Hi") == f"{FRAME_GROUPS}Hi"
