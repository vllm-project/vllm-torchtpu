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

import numpy as np
import pytest
from huggingface_hub import snapshot_download
from vllm.config import ModelConfig
from vllm.model_executor.models import qwen3_vl
from vllm.multimodal import MULTIMODAL_REGISTRY

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
