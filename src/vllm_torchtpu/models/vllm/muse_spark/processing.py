# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""vLLM multimodal processing for Muse Spark Open (image, video, audio).

The checkpoint's ``processor_config.json`` names ``MuseGlimmerImageProcessor``
and ``MuseGlimmerVideoProcessor`` (transformers main) plus
``MuseSparkFeatureExtractor``. transformers 5.14 ships none of them, so the
image and video processors are vendored here from transformers main
(Apache 2.0) and the feature extractor (a pad to 1920 samples) is inlined.
Token expansion mirrors ``MuseSparkProcessor.replace_*_token``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_backends import TorchvisionBackend
from transformers.image_processing_base import BatchFeature as ImageBatchFeature
from transformers.image_transforms import group_images_by_shape, reorder_images
from transformers.image_utils import PILImageResampling, SizeDict
from transformers.processing_utils import ImagesKwargs, VideosKwargs
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import (
    VideoMetadata,
    group_videos_by_shape,
    reorder_videos,
)
from vllm.config.multimodal import (
    AudioDummyOptions,
    BaseDummyOptions,
    ImageDummyOptions,
    VideoDummyOptions,
)
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.renderers import TokenizeParams

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

IMAGE_PLACEHOLDER = "<|patch|>"
VIDEO_PLACEHOLDER = "<|video|>"
AUDIO_PLACEHOLDER = "<|audio|>"
IMAGE_START = "<|image_start|>"
IMAGE_END = "<|image_end|>"
VIDEO_START = "<|vid_start|>"
VIDEO_END = "<|vid_end|>"
VIDEO_SEPARATOR = "<|vid_frame_separator|>"
AUDIO_START = "<|soft_audio_start|>"
AUDIO_END = "<|soft_audio_end|>"

# Defaults from the released processor_config.json.
DEFAULT_IMAGE_PROCESSOR_CONFIG: dict[str, Any] = {
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "max_image_tokens": 4096,
    "merge_size": 2,
    "patch_size": 14,
    "resample": 1,
    "rescale_factor": 1 / 255,
    "temporal_patch_size": 2,
}
DEFAULT_VIDEO_PROCESSOR_CONFIG: dict[str, Any] = {
    **DEFAULT_IMAGE_PROCESSOR_CONFIG,
    "do_sample_frames": True,
    "fps": 2.0,
    "frame_timestamp_format": "Time: {timestamp:.2f}s",
    "max_video_frame_tokens": 256,
    "num_frames": 600,
    "return_metadata": True,
}
del DEFAULT_VIDEO_PROCESSOR_CONFIG["max_image_tokens"]
DEFAULT_FEATURE_EXTRACTOR_CONFIG: dict[str, Any] = {
    "sampling_rate": 24000,
    "samples_per_token": 1920,
    "padding_value": 0.0,
}
# Profiling bound for one audio item; HF imposes no cap.
DEFAULT_MAX_AUDIO_SECONDS = 600


def smart_resize(
    height: int, width: int, patch_size: int, max_tokens: int
) -> tuple[int, int]:
    """Integer cell grid closest to the input aspect ratio under the token cap.

    ``patch_size`` is the merged cell side (patch * merge). Returns pixels.
    """
    ideal_h = height / patch_size
    ideal_w = width / patch_size
    ratio = ideal_w / ideal_h if ideal_h > 0 else 1.0
    if ideal_h * ideal_w > max_tokens:
        ideal_h = (max_tokens / ratio) ** 0.5
        ideal_w = ideal_h * ratio
    candidates = list(
        set(
            itertools.product(
                [math.floor(ideal_h), math.ceil(ideal_h)],
                [math.floor(ideal_w), math.ceil(ideal_w)],
            )
        )
    )
    candidates = [
        (gh, gw)
        for gh, gw in candidates
        if gh >= 1 and gw >= 1 and gh * gw <= max_tokens
    ]
    if not candidates:
        candidates = [(max(1, round(ideal_h)), max(1, round(ideal_w)))]
    gh, gw = min(candidates, key=lambda g: abs(g[0] / g[1] - height / width))
    return gh * patch_size, gw * patch_size


class MuseSparkImageProcessorKwargs(ImagesKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    max_image_tokens: int


class MuseSparkVideoProcessorKwargs(VideosKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    max_video_frame_tokens: int
    frame_timestamp_format: str


class MuseSparkImageProcessor(TorchvisionBackend):
    """Vendored ``MuseGlimmerImageProcessor`` (transformers main)."""

    do_resize = True
    resample = PILImageResampling.LANCZOS
    size = None
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    max_image_tokens = 4096
    valid_kwargs = MuseSparkImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def _resize_smart(
        self,
        images: torch.Tensor,
        patch_size: int,
        merge_size: int,
        max_tokens: int,
        resample: Any,
    ) -> torch.Tensor:
        height, width = images.shape[-2:]
        new_h, new_w = smart_resize(height, width, patch_size * merge_size, max_tokens)
        return super().resize(
            image=images,
            size=SizeDict(height=new_h, width=new_w),
            resample=resample,
            antialias=True,
        )

    @staticmethod
    def patchify(images: torch.Tensor, patch_size: int, temporal_patch_size: int):
        # Each flattened patch is laid out (temporal, channel, py, px).
        batch, channel, height, width = images.shape
        grid_h, grid_w = height // patch_size, width // patch_size
        patches = images.view(batch, channel, grid_h, patch_size, grid_w, patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        flat = (
            patches.unsqueeze(3)
            .expand(-1, -1, -1, temporal_patch_size, -1, -1, -1)
            .reshape(
                batch,
                grid_h * grid_w,
                temporal_patch_size * channel * patch_size * patch_size,
            )
        )
        return flat, grid_h, grid_w

    def _preprocess(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ImageBatchFeature:
        images: list[torch.Tensor] = args[0] if args else kwargs.pop("images")
        do_resize = bool(args[1] if len(args) > 1 else kwargs["do_resize"])
        resample = kwargs["resample"]
        do_rescale = bool(kwargs["do_rescale"])
        rescale_factor = float(kwargs["rescale_factor"])
        do_normalize = bool(kwargs["do_normalize"])
        image_mean = kwargs["image_mean"]
        image_std = kwargs["image_std"]
        return_tensors = kwargs.get("return_tensors")
        patch_size = int(kwargs["patch_size"])
        temporal_patch_size = int(kwargs["temporal_patch_size"])
        max_image_tokens = int(kwargs["max_image_tokens"])
        merge_size = int(kwargs["merge_size"])
        disable_grouping = bool(kwargs.get("disable_grouping", False))

        grouped, index = group_images_by_shape(
            images, disable_grouping=disable_grouping
        )
        resized = {}
        for shape, stacked in grouped.items():
            if do_resize:
                stacked = self._resize_smart(
                    stacked,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    max_tokens=max_image_tokens,
                    resample=resample,
                )
            resized[shape] = stacked
        resized_images = reorder_images(resized, index)

        grouped, index = group_images_by_shape(
            resized_images, disable_grouping=disable_grouping
        )
        processed, grids = {}, {}
        for shape, stacked in grouped.items():
            stacked = self.rescale_and_normalize(
                stacked, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches, grid_h, grid_w = self.patchify(
                stacked, patch_size, temporal_patch_size
            )
            processed[shape] = patches
            grids[shape] = [[1, grid_h, grid_w]] * len(stacked)
        reordered = reorder_images(processed, index)
        pixel_values = (
            torch.cat(reordered, dim=0)
            if isinstance(reordered, (list, tuple))
            else reordered
        )
        image_grid_thw = torch.tensor(reorder_images(grids, index))
        return ImageBatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    def get_grid(self, height: int, width: int) -> tuple[int, int]:
        cell = self.patch_size * self.merge_size
        new_h, new_w = smart_resize(height, width, cell, self.max_image_tokens)
        return new_h // self.patch_size, new_w // self.patch_size

    def get_number_of_image_patches(self, height, width, images_kwargs=None) -> int:
        grid_h, grid_w = self.get_grid(height, width)
        return grid_h * grid_w

    def _validate_preprocess_kwargs(self, *args: Any, **kwargs: Any) -> None:
        # Resizing is driven by the token cap, not ``size``.
        kwargs["do_resize"] = False
        super()._validate_preprocess_kwargs(*args, **kwargs)


class MuseSparkVideoProcessor(BaseVideoProcessor):
    """Vendored ``MuseGlimmerVideoProcessor`` (transformers main)."""

    resample = PILImageResampling.LANCZOS
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    default_to_square = True
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    max_video_frame_tokens = 256
    num_frames = 600
    fps = 2.0
    do_sample_frames = True
    frame_timestamp_format = "Time: {timestamp:.2f}s"
    valid_kwargs = MuseSparkVideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def _validate_preprocess_kwargs(self, *args: Any, **kwargs: Any) -> None:
        kwargs["do_resize"] = False
        super()._validate_preprocess_kwargs(*args, **kwargs)

    def _resize_smart(
        self,
        videos: torch.Tensor,
        resample: Any,
        patch_size: int,
        merge_size: int,
        max_tokens: int,
    ) -> torch.Tensor:
        height, width = videos.shape[-2:]
        new_h, new_w = smart_resize(height, width, patch_size * merge_size, max_tokens)
        return super().resize(
            videos,
            size=SizeDict(height=new_h, width=new_w),
            resample=resample,
            antialias=True,
        )

    @staticmethod
    def patchify(videos: torch.Tensor, patch_size: int, temporal_patch_size: int):
        batch, num_frames, channel, height, width = videos.shape
        # Pad the frame count to a multiple of temporal_patch_size with the
        # last frame.
        if pad := -num_frames % temporal_patch_size:
            videos = torch.cat(
                (videos, videos[:, -1:].expand(-1, pad, -1, -1, -1)), dim=1
            )
            num_frames += pad
        grid_t = num_frames // temporal_patch_size
        grid_h, grid_w = height // patch_size, width // patch_size
        patches = videos.view(
            batch,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h,
            patch_size,
            grid_w,
            patch_size,
        )
        patches = patches.permute(0, 1, 4, 6, 2, 3, 5, 7)
        flat = patches.reshape(
            batch,
            grid_t * grid_h * grid_w,
            temporal_patch_size * channel * patch_size * patch_size,
        )
        return flat, grid_t, grid_h, grid_w

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: int | float | None = None,
        temporal_patch_size: int | None = None,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        fps = fps if fps is not None else self.fps
        num_frames = num_frames if num_frames is not None else self.num_frames
        temporal_patch_size = temporal_patch_size or self.temporal_patch_size
        if metadata.fps is None:
            metadata.fps = 24
        total = metadata.total_num_frames
        num_frames = min(int(total * fps / metadata.fps), num_frames, total)
        num_frames = max(
            temporal_patch_size,
            (num_frames // temporal_patch_size) * temporal_patch_size,
        )
        num_frames = min(num_frames, total)
        return torch.linspace(0, total - 1, num_frames).long()

    def _preprocess(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ImageBatchFeature:
        videos: list[torch.Tensor] = args[0] if args else kwargs.pop("videos")
        do_resize = bool(args[1] if len(args) > 1 else kwargs["do_resize"])
        resample = kwargs["resample"]
        do_rescale = bool(kwargs["do_rescale"])
        rescale_factor = float(kwargs["rescale_factor"])
        do_normalize = bool(kwargs["do_normalize"])
        image_mean = kwargs["image_mean"]
        image_std = kwargs["image_std"]
        return_tensors = kwargs.get("return_tensors")
        patch_size = int(kwargs["patch_size"])
        temporal_patch_size = int(kwargs["temporal_patch_size"])
        max_video_frame_tokens = int(kwargs["max_video_frame_tokens"])
        merge_size = int(kwargs["merge_size"])

        grouped, index = group_videos_by_shape(videos)
        resized = {}
        for shape, stacked in grouped.items():
            if do_resize:
                stacked = self._resize_smart(
                    stacked,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    max_tokens=max_video_frame_tokens,
                    resample=resample,
                )
            resized[shape] = stacked
        resized_videos = reorder_videos(resized, index)

        grouped, index = group_videos_by_shape(resized_videos)
        processed, grids = {}, {}
        for shape, stacked in grouped.items():
            stacked = self.rescale_and_normalize(
                stacked, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches, grid_t, grid_h, grid_w = self.patchify(
                stacked, patch_size, temporal_patch_size
            )
            processed[shape] = patches
            grids[shape] = [[grid_t, grid_h, grid_w]] * len(stacked)
        reordered_vid = reorder_videos(processed, index)
        pixel_values_videos = (
            torch.cat(reordered_vid, dim=0)
            if isinstance(reordered_vid, (list, tuple))
            else reordered_vid
        )
        video_grid_thw = torch.tensor(reorder_videos(grids, index))
        return ImageBatchFeature(
            data={
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
            },
            tensor_type=return_tensors,
        )

    def get_frame_grid(self, height: int, width: int) -> tuple[int, int]:
        cell = self.patch_size * self.merge_size
        new_h, new_w = smart_resize(height, width, cell, self.max_video_frame_tokens)
        return new_h // self.patch_size, new_w // self.patch_size


def video_prompt_text(
    grid_thw: Sequence[int],
    timestamps: Sequence[float],
    *,
    merge_size: int,
    temporal_patch_size: int,
    timestamp_format: str,
) -> str:
    """``MuseSparkProcessor.replace_video_token`` for one video.

    ``timestamps`` are per sampled frame (seconds); one is used per temporal
    group, padded by repeating the last.
    """
    n_groups = int(grid_thw[0])
    tokens_per_group = int(grid_thw[1]) * int(grid_thw[2]) // merge_size**2
    stamps = list(timestamps[::temporal_patch_size])[:n_groups]
    while len(stamps) < n_groups:
        stamps.append(stamps[-1] if stamps else 0.0)
    out = VIDEO_START
    for g, ts in enumerate(stamps):
        out += (
            timestamp_format.format(timestamp=ts) + VIDEO_PLACEHOLDER * tokens_per_group
        )
        out += VIDEO_SEPARATOR if g < n_groups - 1 else VIDEO_END
    return out


def _subconfig(full: Mapping[str, Any], key: str, defaults: Mapping[str, Any]):
    cfg = dict(defaults)
    cfg.update(full.get(key) or {})
    for k in ("image_processor_type", "video_processor_type", "feature_extractor_type"):
        cfg.pop(k, None)
    return cfg


class MuseSparkProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None, "audio": None}

    def get_default_tok_params(self) -> TokenizeParams:
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def _processor_config(self) -> Mapping[str, Any]:
        cached = getattr(self, "_muse_processor_config", None)
        if cached is None:
            from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

            try:
                cached = (
                    get_hf_file_to_dict(
                        "processor_config.json",
                        self.ctx.model_config.model,
                        self.ctx.model_config.revision,
                    )
                    or {}
                )
            except (OSError, ValueError) as e:
                # A missing file returns None above; this absorbs an unreadable
                # or malformed processor_config.json.
                logger.warning_once(
                    "Muse Spark: cannot read processor_config.json (%s); using "
                    "the built-in processor defaults.",
                    e,
                )
                cached = {}
            self._muse_processor_config = cached
        return cached

    def get_supported_mm_processor_kwargs(self) -> dict[str, set[str]]:
        return {
            "text_kwargs": set(),
            "images_kwargs": set(DEFAULT_IMAGE_PROCESSOR_CONFIG),
            "videos_kwargs": set(DEFAULT_VIDEO_PROCESSOR_CONFIG),
            "audio_kwargs": set(DEFAULT_FEATURE_EXTRACTOR_CONFIG),
        }

    def _overrides(self, kwargs: Mapping[str, object], modality: str):
        merged = self.ctx.get_merged_mm_kwargs(
            kwargs,
            supported_mm_processor_kwargs=self.supported_mm_processor_kwargs,
        )
        scope_key = {"image": "images_kwargs", "video": "videos_kwargs"}.get(
            modality, f"{modality}_kwargs"
        )
        scoped = merged.get(scope_key)
        effective = dict(merged)
        if isinstance(scoped, Mapping):
            effective.update(scoped)
        # The encoder expects normalized pixels and does not support
        # mm_device_do_normalize, so never let it strip rescale/normalize.
        return {
            k: v
            for k, v in effective.items()
            if v is not None and k not in ("do_rescale", "do_normalize")
        }

    def get_image_processor(self, **kwargs: object) -> MuseSparkImageProcessor:
        cfg = _subconfig(
            self._processor_config(), "image_processor", DEFAULT_IMAGE_PROCESSOR_CONFIG
        )
        allowed = set(DEFAULT_IMAGE_PROCESSOR_CONFIG)
        cfg.update(
            {k: v for k, v in self._overrides(kwargs, "image").items() if k in allowed}
        )
        return MuseSparkImageProcessor(**cfg)

    def get_video_processor(self, **kwargs: object) -> MuseSparkVideoProcessor:
        cfg = _subconfig(
            self._processor_config(), "video_processor", DEFAULT_VIDEO_PROCESSOR_CONFIG
        )
        allowed = set(DEFAULT_VIDEO_PROCESSOR_CONFIG)
        cfg.update(
            {k: v for k, v in self._overrides(kwargs, "video").items() if k in allowed}
        )
        return MuseSparkVideoProcessor(**cfg)

    def get_feature_extractor_config(self) -> Mapping[str, Any]:
        return _subconfig(
            self._processor_config(),
            "feature_extractor",
            DEFAULT_FEATURE_EXTRACTOR_CONFIG,
        )

    @property
    def sampling_rate(self) -> int:
        return int(self.get_feature_extractor_config()["sampling_rate"])

    @property
    def samples_per_token(self) -> int:
        return int(self.get_feature_extractor_config()["samples_per_token"])

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=self.sampling_rate,
            target_channels=1,
            video_needs_metadata=True,
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int:
        processor = self.get_image_processor()
        grid_h, grid_w = processor.get_grid(image_height, image_width)
        return grid_h * grid_w // processor.merge_size**2

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_image_processor()
        side = math.isqrt(int(processor.max_image_tokens))
        cell = processor.patch_size * processor.merge_size
        return ImageSize(width=side * cell, height=side * cell)

    def get_num_frames_with_most_features(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int:
        processor = self.get_video_processor()
        temporal = int(processor.temporal_patch_size)
        num_videos = max(mm_counts.get("video", 0), 1)
        groups = max(1, seq_len // num_videos // int(processor.max_video_frame_tokens))
        max_groups = max(1, int(processor.num_frames) // temporal)
        return min(groups, max_groups) * temporal

    def get_max_audio_samples(self) -> int:
        return DEFAULT_MAX_AUDIO_SECONDS * self.sampling_rate

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        image = self.get_image_processor()
        video = self.get_video_processor()
        num_frames = self.get_num_frames_with_most_features(seq_len, mm_counts)
        return {
            "image": int(image.max_image_tokens),
            "video": num_frames
            // int(video.temporal_patch_size)
            * int(video.max_video_frame_tokens),
            "audio": -(-self.get_max_audio_samples() // self.samples_per_token),
        }


class MuseSparkDummyInputsBuilder(BaseDummyInputsBuilder[MuseSparkProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return (
            IMAGE_PLACEHOLDER * mm_counts.get("image", 0)
            + VIDEO_PLACEHOLDER * mm_counts.get("video", 0)
            + AUDIO_PLACEHOLDER * mm_counts.get("audio", 0)
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        width, height = self.info.get_image_size_with_most_features()
        video = self.info.get_video_processor()
        side = math.isqrt(int(video.max_video_frame_tokens))
        video_size = side * video.patch_size * video.merge_size
        num_frames = self.info.get_num_frames_with_most_features(seq_len, mm_counts)
        num_videos = mm_counts.get("video", 0)
        videos = self._get_dummy_videos(
            width=video_size,
            height=video_size,
            num_frames=num_frames,
            num_videos=num_videos,
            overrides=cast(VideoDummyOptions | None, mm_options.get("video")),
        )
        fps = float(video.fps)
        video_items = [
            (
                v,
                {
                    "fps": fps,
                    "duration": len(v) / fps,
                    "total_num_frames": len(v),
                    "frames_indices": list(range(len(v))),
                    "video_backend": "opencv",
                    "do_sample_frames": False,
                },
            )
            for v in videos
        ]
        return {
            "image": self._get_dummy_images(
                width=width,
                height=height,
                num_images=mm_counts.get("image", 0),
                overrides=cast(ImageDummyOptions | None, mm_options.get("image")),
            ),
            "video": video_items,
            "audio": self._get_dummy_audios(
                length=self.info.get_max_audio_samples(),
                num_audios=mm_counts.get("audio", 0),
                overrides=cast(AudioDummyOptions | None, mm_options.get("audio")),
            ),
        }


_VIDEO_METADATA_KEYS = (
    "total_num_frames",
    "fps",
    "width",
    "height",
    "duration",
    "video_backend",
    "frames_indices",
)


def _video_metadata(
    frames: np.ndarray, meta: Mapping[str, Any] | None
) -> VideoMetadata:
    meta = dict(meta or {})
    kwargs = {k: meta[k] for k in _VIDEO_METADATA_KEYS if meta.get(k) is not None}
    kwargs.setdefault("total_num_frames", len(frames))
    if "frames_indices" in kwargs:
        kwargs["frames_indices"] = list(kwargs["frames_indices"])
    return VideoMetadata(**kwargs)


class MuseSparkMultiModalProcessor(BaseMultiModalProcessor[MuseSparkProcessingInfo]):
    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object] | None = None,
        *,
        hf_processor_mm_kwargs: Mapping[str, object] | None = None,
    ) -> BatchFeature:
        if hf_processor_mm_kwargs is None:
            hf_processor_mm_kwargs = hf_kwargs or {}
        valid = mm_items.select(
            {k for k, c in mm_items.get_all_counts().items() if c > 0}
        )
        mm_data: dict[str, object] = {}
        passthrough: dict[str, object] = {}
        for items in valid.values():
            if not items:
                continue
            mm_data.update(items.get_processor_data())
            passthrough.update(items.get_passthrough_data())
        data: dict[str, object] = {}

        images = list(cast(Iterable[Any], mm_data.get("images", ())))
        if images:
            out = self.info.get_image_processor(**hf_processor_mm_kwargs)(
                images=images, return_tensors="pt"
            )
            data["pixel_values"] = out["pixel_values"]
            data["image_grid_thw"] = out["image_grid_thw"]

        videos = list(cast(Iterable[Any], mm_data.get("videos", ())))
        if videos:
            processor = self.info.get_video_processor(**hf_processor_mm_kwargs)
            pixels, grids, stamps = [], [], []
            for item in videos:
                frames, meta = item if isinstance(item, tuple) else (item, None)
                frames = np.asarray(frames)
                metadata = _video_metadata(frames, meta)
                do_sample = bool((meta or {}).get("do_sample_frames", False))
                if metadata.frames_indices is None:
                    metadata.frames_indices = list(range(len(frames)))
                out = processor(
                    videos=[frames],
                    video_metadata=[metadata],
                    do_sample_frames=do_sample,
                    return_metadata=True,
                    return_tensors="pt",
                )
                grid = out["video_grid_thw"][0]
                used = out["video_metadata"][0]
                if used.fps is None:
                    used.fps = 24
                group_stamps = list(used.timestamps[:: processor.temporal_patch_size])
                group_stamps = group_stamps[: int(grid[0])]
                while len(group_stamps) < int(grid[0]):
                    group_stamps.append(group_stamps[-1] if group_stamps else 0.0)
                pixels.append(out["pixel_values_videos"])
                grids.append(grid)
                stamps.append(torch.tensor(group_stamps, dtype=torch.float64))
            data["pixel_values_videos"] = torch.cat(pixels)
            data["video_grid_thw"] = torch.stack(grids)
            data["video_timestamps"] = stamps

        audios = list(cast(Iterable[Any], mm_data.get("audios", ())))
        if audios:
            clips = []
            for audio in audios:
                arr = np.asarray(audio, dtype=np.float32)
                if arr.ndim != 1:
                    raise ValueError(
                        f"Muse Spark audio must be mono 1D, got shape {arr.shape}"
                    )
                clips.append(torch.from_numpy(arr.copy()))
            data["input_values"] = clips

        processed = BatchFeature(data=data, tensor_type=None)
        processed.update(passthrough)
        return processed

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields: dict[str, MultiModalFieldConfig] = {}
        if "image_grid_thw" in hf_inputs:
            sizes = hf_inputs["image_grid_thw"].prod(-1)
            fields.update(
                pixel_values=MultiModalFieldConfig.flat_from_sizes("image", sizes),
                image_grid_thw=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            )
        if "video_grid_thw" in hf_inputs:
            sizes = hf_inputs["video_grid_thw"].prod(-1)
            fields.update(
                pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                    "video", sizes
                ),
                video_grid_thw=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
                video_timestamps=MultiModalFieldConfig.batched(
                    "video", keep_on_cpu=True
                ),
            )
        if "input_values" in hf_inputs:
            fields["input_values"] = MultiModalFieldConfig.batched("audio")
        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        image_id = vocab[IMAGE_PLACEHOLDER]
        video_id = vocab[VIDEO_PLACEHOLDER]
        audio_id = vocab[AUDIO_PLACEHOLDER]
        image_start, image_end = vocab[IMAGE_START], vocab[IMAGE_END]
        video_start, video_end = vocab[VIDEO_START], vocab[VIDEO_END]
        video_sep = vocab[VIDEO_SEPARATOR]
        audio_start, audio_end = vocab[AUDIO_START], vocab[AUDIO_END]
        image_merge = self.info.get_image_processor(**hf_processor_mm_kwargs).merge_size
        video_processor = self.info.get_video_processor(**hf_processor_mm_kwargs)
        timestamp_format = video_processor.frame_timestamp_format
        samples_per_token = self.info.samples_per_token

        def image_replacement(item_idx: int) -> PromptUpdateDetails:
            grid = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
            assert isinstance(grid, torch.Tensor)
            n = int(grid.prod()) // image_merge**2
            return PromptUpdateDetails.select_token_id(
                [image_start] + [image_id] * n + [image_end], image_id
            )

        def video_replacement(item_idx: int) -> PromptUpdateDetails:
            item = out_mm_kwargs["video"][item_idx]
            v_grid = item["video_grid_thw"].data
            v_stamps = item["video_timestamps"].data
            assert isinstance(v_grid, torch.Tensor)
            assert isinstance(v_stamps, torch.Tensor)
            t, h, w = (int(x) for x in v_grid.tolist())
            per_group = h * w // video_processor.merge_size**2
            stamps = v_stamps.tolist()
            ids = [video_start]
            for g in range(t):
                ids.extend(
                    tokenizer.encode(
                        timestamp_format.format(timestamp=stamps[g]),
                        add_special_tokens=False,
                    )
                )
                ids.extend([video_id] * per_group)
                ids.append(video_sep if g < t - 1 else video_end)
            return PromptUpdateDetails.select_token_id(ids, video_id)

        def audio_replacement(item_idx: int) -> PromptUpdateDetails:
            clip = out_mm_kwargs["audio"][item_idx]["input_values"].data
            assert isinstance(clip, torch.Tensor)
            n = -(-int(clip.shape[-1]) // samples_per_token)
            return PromptUpdateDetails.select_token_id(
                [audio_start] + [audio_id] * n + [audio_end], audio_id
            )

        return [
            PromptReplacement("image", [image_id], image_replacement),
            PromptReplacement("video", [video_id], video_replacement),
            PromptReplacement("audio", [audio_id], audio_replacement),
        ]


__all__ = [
    "AUDIO_PLACEHOLDER",
    "IMAGE_PLACEHOLDER",
    "VIDEO_PLACEHOLDER",
    "MuseSparkDummyInputsBuilder",
    "MuseSparkImageProcessor",
    "MuseSparkMultiModalProcessor",
    "MuseSparkProcessingInfo",
    "MuseSparkVideoProcessor",
    "smart_resize",
    "video_prompt_text",
]
