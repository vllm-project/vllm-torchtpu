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
"""TPU Omni multimodal processing guards and audio budget for Muse Spark Open.

Extends upstream ``vllm.models.muse_spark.common.processor`` on two points:
1. Replaces upstream's 1024-token (~82 s) audio fallback with a 600 s cap
   (7,500 tokens at 24 kHz / 1920 samples per token) and rejects clips above it.
2. Rejects empty audio inputs and zero-dimension images/video frames cleanly
   with ``ValueError`` before resizing divides by zero.

The TPU KV-cache memory profiler (``TPUModelRunner.profile_run``) does not run
the audio encoder, so its activation peak is not reserved out of the KV-cache
budget. Long audio clips need HBM headroom left by a lower
``--gpu-memory-utilization``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from transformers.feature_extraction_utils import BatchFeature
from vllm.models.muse_spark.common.processor import (
    MuseSpark1xOmniMultiModalProcessor,
    MuseSpark1xOmniProcessingInfo,
)
from vllm.multimodal.parse import MultiModalDataItems

MAX_AUDIO_SECONDS = 600


def _validate_non_empty_extent(width: int, height: int, *, kind: str) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Muse Spark: empty {kind} (got width={width}, height={height})."
        )


def _validate_image_like(item: Any, *, kind: str) -> None:
    size = getattr(item, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        _validate_non_empty_extent(int(size[0]), int(size[1]), kind=kind)
        return
    if hasattr(item, "shape") and len(item.shape) >= 2:
        # Image/frame arrays may be [H, W, C], [C, H, W], or [H, W].
        if len(item.shape) >= 3 and int(item.shape[-1]) in (1, 3, 4):
            h, w = int(item.shape[-3]), int(item.shape[-2])
        else:
            h, w = int(item.shape[-2]), int(item.shape[-1])
        _validate_non_empty_extent(w, h, kind=kind)


def _check_media_dimensions(mm_items: MultiModalDataItems) -> None:
    valid_items = mm_items.select(
        {k for k, c in mm_items.get_all_counts().items() if c > 0}
    )
    proc_data: dict[str, object] = {}
    for items in valid_items.values():
        if items:
            proc_data.update(items.get_processor_data())

    images = proc_data.get("images")
    if images:
        for img in cast(Sequence[Any], images):
            _validate_image_like(img, kind="image")

    videos = proc_data.get("videos")
    if videos:
        for video_item in cast(Sequence[Any], videos):
            frames = (
                video_item[0]
                if isinstance(video_item, tuple) and len(video_item) >= 1
                else video_item
            )
            if hasattr(frames, "shape") and len(frames.shape) >= 3:
                if int(frames.shape[0]) <= 0:
                    raise ValueError("Muse Spark: empty video input.")
                # Video arrays are [T, H, W, C] or [T, C, H, W].
                if len(frames.shape) == 4 and int(frames.shape[-1]) in (1, 3, 4):
                    h, w = int(frames.shape[1]), int(frames.shape[2])
                else:
                    h, w = int(frames.shape[-2]), int(frames.shape[-1])
                _validate_non_empty_extent(w, h, kind="video frame")
            elif isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)):
                if len(frames) == 0:
                    raise ValueError("Muse Spark: empty video input.")
                for frame in frames:
                    _validate_image_like(frame, kind="video frame")


class TPUMuseSparkOmniProcessingInfo(MuseSpark1xOmniProcessingInfo):
    """Processing info with explicit 600 s audio token budget.

    Note: ``get_mm_max_tokens_per_item`` counts only encoder-embedded positions
    (``is_embed=True``); upstream marks wrapper and timestamp text tokens
    ``is_embed=False``.
    """

    def max_audio_tokens(self) -> int:
        samples_per_token = math.prod(self.get_hf_config().audio_config.patch_sizes)
        return -(-MAX_AUDIO_SECONDS * self.audio_sample_rate // samples_per_token)

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        modality_max = dict(super().get_mm_max_tokens_per_item(seq_len, mm_counts))
        if "audio" in modality_max:
            modality_max["audio"] = self.max_audio_tokens()
        return modality_max


class TPUMuseSparkOmniMultiModalProcessor(MuseSpark1xOmniMultiModalProcessor):
    """Omni processor enforcing non-empty media and the 600 s audio token cap."""

    info: TPUMuseSparkOmniProcessingInfo  # type: ignore[assignment]

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        _check_media_dimensions(mm_items)
        out = super()._apply_hf_processor_main(mm_items, hf_kwargs)
        cap = self.info.max_audio_tokens()
        features = out.get("input_features") or []
        counts = out.get("audio_token_counts") or []
        for wav, token_count in zip(features, counts, strict=False):
            if wav.numel() == 0 or int(token_count) <= 0:
                raise ValueError("Muse Spark: empty audio input.")
            if int(token_count) > cap:
                raise ValueError(
                    f"Muse Spark audio input requires {token_count} tokens, "
                    f"exceeding the {cap}-token ({MAX_AUDIO_SECONDS} s) cap."
                )
        return out


__all__ = [
    "MAX_AUDIO_SECONDS",
    "TPUMuseSparkOmniMultiModalProcessor",
    "TPUMuseSparkOmniProcessingInfo",
]
