# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse image, video, and waveform preprocessing with aligned placeholders."""

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Annotated, Any, Literal, cast

import numpy as np
import torch
from PIL import Image
from transformers import BatchFeature
from transformers.models.auto.video_processing_auto import VIDEO_PROCESSOR_MAPPING_NAMES
from vllm.config.multimodal import (
    AudioDummyOptions,
    BaseDummyOptions,
    ImageDummyOptions,
    VideoDummyOptions,
)
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.video import (
    VIDEO_LOADER_REGISTRY,
    VideoBackend,
    VideoSourceMetadata,
    VideoTargetMetadata,
)
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict
from vllm.utils.tensor_schema import TensorSchema, TensorShape

WAVEFORM_PER_TOKEN = 1920  # 80ms @ 24kHz per latent token
AUDIO_TARGET_SAMPLE_RATE = 24000

DEFAULT_IMAGE_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
DEFAULT_IMAGE_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)

# Match ImageMediaIO's alpha-compositing background.
RGBA_BACKGROUND_COLOR: tuple[int, int, int] = (255, 255, 255)


class MuseSparkImageInputs(TensorSchema):
    """Preprocessed images, one [C, H, W] fp32 tensor per image."""

    type: Literal["image_pixels"]
    images: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", 3, "h", "w", dynamic_dims={"h", "w"}),
    ]
    image_patch_counts: Annotated[torch.Tensor, TensorShape("bn")]


class MuseSparkVideoInputs(TensorSchema):
    """Preprocessed videos, one [T, C, H, W] fp32 tensor per video."""

    type: Literal["video_pixels"]
    videos: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "t", 3, "h", "w", dynamic_dims={"t", "h", "w"}),
    ]
    video_patch_counts: Annotated[torch.Tensor, TensorShape("bn")]


class MuseSparkAudioInputs(TensorSchema):
    """Raw audio waveforms, one [num_samples] fp32 tensor per audio."""

    type: Literal["audio_waveforms"]
    input_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "n", dynamic_dims={"n"}),
    ]
    audio_token_counts: Annotated[torch.Tensor, TensorShape("bn")]


@dataclass
class PreprocessConfig:
    """Configuration for image/video preprocessing."""

    patch_height: int = 14
    patch_width: int = 14
    patch_temporal: int = 2
    downsample_factor: int = 2
    min_num_tokens: int = 1
    max_num_image_tokens: int = 4096
    max_num_video_tokens: int = 4096
    use_per_video_token_budget: bool = False
    image_mean: tuple[float, float, float] = DEFAULT_IMAGE_MEAN
    image_std: tuple[float, float, float] = DEFAULT_IMAGE_STD
    video_downsample_factor: int = 2

    num_frames_per_video: int = 600
    sampling_fps: float | None = 2.0
    frame_factor: int | None = 2
    max_num_tokens_per_frame: int = 256
    video_frame_timestamp_format: str = "Time: {timestamp:.2f}s"

    @property
    def token_patch_height(self) -> int:
        """Spatial resolution of each token after downsampling."""
        return self.patch_height * self.downsample_factor

    @property
    def token_patch_width(self) -> int:
        """Spatial resolution of each token after downsampling."""
        return self.patch_width * self.downsample_factor


class NativeVariableSizeImageTransformError(Exception):
    """Raised when resolution fitting fails."""


def _to_rgb(image: Image.Image) -> Image.Image:
    """Convert to RGB, compositing transparency over the serving background."""
    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        composited = Image.new("RGB", image.size, RGBA_BACKGROUND_COLOR)
        composited.paste(image, mask=image.getchannel("A"))
        return composited
    return image.convert("RGB")


def _resize(image: Image.Image, target_h: int, target_w: int) -> torch.Tensor:
    """PIL LANCZOS resize to a uint8 [C, H, W] tensor (training parity)."""
    resized = image.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1)


def _resize_frames(
    frames: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Resize a uint8 [T, C, H, W] tensor frame by frame with PIL LANCZOS."""
    resized = [
        _resize(Image.fromarray(frame.permute(1, 2, 0).numpy()), target_h, target_w)
        for frame in frames
    ]
    return torch.stack(resized)


def _normalize_pixels(
    pixels: torch.Tensor,
    config: PreprocessConfig,
) -> torch.Tensor:
    """Scale uint8 pixels to [0, 1] and apply channel mean/std normalization."""
    assert pixels.dtype == torch.uint8, (
        f"Expected uint8 pixels for normalization, got {pixels.dtype}"
    )
    assert pixels.shape[-3] == len(config.image_mean), (
        f"Expected {len(config.image_mean)}-channel pixels for normalization, "
        f"got {pixels.shape[-3]} (shape {tuple(pixels.shape)})"
    )
    scaled = pixels.float().div_(255.0)
    if len(set(config.image_mean)) == 1 and len(set(config.image_std)) == 1:
        return scaled.sub_(config.image_mean[0]).div_(config.image_std[0])
    mean = torch.tensor(config.image_mean, dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(config.image_std, dtype=torch.float32).view(-1, 1, 1)
    return scaled.sub_(mean).div_(std)


def _least_distortion_fit(
    i_nph: float,
    i_npw: float,
    img_h: int,
    img_w: int,
    patch_h: int,
    patch_w: int,
    min_num_tokens: int,
    max_num_tokens: int,
) -> tuple[int, int]:
    """Find the patch grid (nph, npw) minimizing aspect-ratio distortion."""
    candidates = list(
        set(
            itertools.product(
                [math.floor(i_nph), math.ceil(i_nph)],
                [math.floor(i_npw), math.ceil(i_npw)],
            )
        )
    )

    def _filter(nph: int, npw: int) -> bool:
        return (nph > 0 and npw > 0) and (min_num_tokens <= nph * npw <= max_num_tokens)

    candidates = [c for c in candidates if _filter(*c)]

    if len(candidates) == 0:
        raise NativeVariableSizeImageTransformError(
            f"least distortion fit failed on {i_nph=}, {i_npw=}, "
            f"{img_h=}, {img_w=}, {max_num_tokens=}"
        )

    def _distortion(nph: int, npw: int) -> float:
        return abs((nph * patch_h) / (npw * patch_w) - (img_h / img_w))

    return min(candidates, key=lambda x: _distortion(*x))


def _least_distance_fit(
    i_nph: float,
    i_npw: float,
    min_num_tokens: int,
    max_num_tokens: int,
    patch_h: int,
    patch_w: int,
) -> tuple[int, int]:
    """Fallback: closest grid point between the min/max token hyperbolas."""
    nph = np.arange(1, max_num_tokens + 1)
    npw_upper = np.floor(max_num_tokens / nph)
    npw_lower = np.ceil(min_num_tokens / nph)
    valid_msk = npw_upper >= npw_lower

    if not any(valid_msk):
        raise NativeVariableSizeImageTransformError(
            "least distance fit failed: no valid grid point between "
            f"{min_num_tokens} and {max_num_tokens}"
        )

    # Round before clamping to match the reference resize.
    npw = np.clip(round(i_npw), a_min=npw_lower, a_max=npw_upper)
    nph = nph[valid_msk]
    npw = npw[valid_msk]

    l2 = ((nph - i_nph) * patch_h) ** 2 + ((npw - i_npw) * patch_w) ** 2
    l2_min_idx = np.argmin(l2)

    return int(nph[l2_min_idx].item()), int(npw[l2_min_idx].item())


def find_resize_target(
    img_h: int,
    img_w: int,
    patch_h: int,
    patch_w: int,
    min_num_tokens: int,
    max_num_tokens: int,
    resize_to_max_canvas: bool = False,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Choose a resolution preserving aspect ratio within the token budget."""
    i_nph = img_h / patch_h
    i_npw = img_w / patch_w
    ideal_ratio = (img_w / patch_w) / (img_h / patch_h)

    if resize_to_max_canvas or i_nph * i_npw > max_num_tokens:
        i_nph = (max_num_tokens / ideal_ratio) ** 0.5
        i_npw = i_nph * ideal_ratio
    elif i_nph * i_npw < min_num_tokens:
        i_nph = (min_num_tokens / ideal_ratio) ** 0.5
        i_npw = i_nph * ideal_ratio

    try:
        nph, npw = _least_distortion_fit(
            i_nph, i_npw, img_h, img_w, patch_h, patch_w, min_num_tokens, max_num_tokens
        )
    except NativeVariableSizeImageTransformError:
        nph, npw = _least_distance_fit(
            i_nph, i_npw, min_num_tokens, max_num_tokens, patch_h, patch_w
        )

    assert min_num_tokens <= nph * npw <= max_num_tokens, (
        f"Token count {nph * npw} outside bounds [{min_num_tokens}, {max_num_tokens}]"
    )

    return (nph * patch_h, npw * patch_w), (nph, npw)


def preprocess_image(
    image: Image.Image,
    config: PreprocessConfig,
    resize_to_max_canvas: bool = False,
) -> tuple[torch.Tensor, int]:
    """Resize and normalize a NaViT image."""
    # Composite before LANCZOS to prevent transparent-pixel color bleeding.
    image = _to_rgb(image)

    w, h = image.size

    (target_h, target_w), (nph, npw) = find_resize_target(
        img_h=h,
        img_w=w,
        patch_h=config.token_patch_height,
        patch_w=config.token_patch_width,
        min_num_tokens=config.min_num_tokens,
        max_num_tokens=config.max_num_image_tokens,
        resize_to_max_canvas=resize_to_max_canvas,
    )

    img_tensor = _resize(image, target_h, target_w)
    img_tensor = _normalize_pixels(img_tensor, config)

    num_tokens = nph * npw
    return img_tensor, num_tokens


def preprocess_video(
    frames: list[np.ndarray] | np.ndarray,
    timestamps: list[float],
    config: PreprocessConfig,
    resize_to_max_canvas: bool = False,
) -> tuple[torch.Tensor, list[int], list[float]]:
    """Return normalized video, per-frame token counts, and aligned timestamps."""
    assert len(frames) > 0, "Video must have at least one frame"
    vds = config.video_downsample_factor

    n_frames = len(frames)
    aligned_n = (n_frames // vds) * vds
    if aligned_n == 0:
        aligned_n = vds
        # Use list concatenation to repeat the last frame, not ndarray addition.
        padding_count = aligned_n - n_frames
        frames = list(frames) + [frames[-1]] * padding_count
        timestamps = list(timestamps) + [timestamps[-1]] * padding_count
    frames = frames[:aligned_n]
    timestamps = timestamps[:aligned_n]

    first_frame = frames[0]
    frame_h, frame_w = first_frame.shape[0], first_frame.shape[1]

    if config.max_num_tokens_per_frame > 0:
        max_tokens_per_frame = config.max_num_tokens_per_frame
    elif config.use_per_video_token_budget:
        num_temporal_groups = aligned_n // config.patch_temporal
        if num_temporal_groups > 0:
            max_tokens_per_frame = config.max_num_video_tokens // num_temporal_groups
        else:
            max_tokens_per_frame = config.max_num_video_tokens
    else:
        max_tokens_per_frame = config.max_num_video_tokens

    (target_h, target_w), (nph, npw) = find_resize_target(
        img_h=frame_h,
        img_w=frame_w,
        patch_h=config.token_patch_height,
        patch_w=config.token_patch_width,
        min_num_tokens=config.min_num_tokens,
        max_num_tokens=max_tokens_per_frame,
        resize_to_max_canvas=resize_to_max_canvas,
    )

    tokens_per_frame = nph * npw

    # asarray avoids copying a complete decoded video.
    stacked = np.asarray(frames)  # (T, H, W, C) uint8
    if stacked.ndim == 4 and stacked.shape[-1] == 4:
        raise ValueError(
            "Video frames must be 3-channel RGB, got 4 channels (alpha). "
            "Convert the frames to RGB before preprocessing: whether the "
            "alpha is straight or premultiplied is not recoverable from a "
            "bare frame array, so it cannot be composited correctly here."
        )
    # torch.from_numpy requires writable storage without negative strides.
    if not stacked.flags.writeable or any(s < 0 for s in stacked.strides):
        stacked = stacked.copy()
    frames_tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2)  # uint8 TCHW

    frames_tensor = _resize_frames(frames_tensor, target_h, target_w)

    video_tensor = _normalize_pixels(frames_tensor, config)

    per_frame_token_counts = [tokens_per_frame] * len(frames)
    factored_timestamps = timestamps

    return video_tensor, per_frame_token_counts, factored_timestamps


@dataclass
class VideoClipMetadata:
    """Metadata for temporal clips in a preprocessed video."""

    num_clips: int
    patches_per_clip: int
    clip_timestamps: list[float]


def compute_video_clip_metadata(
    video_tensor: torch.Tensor,
    factored_timestamps: list[float],
    config: PreprocessConfig,
) -> VideoClipMetadata:
    """Split a preprocessed video into temporal clips of `patch_temporal`
    frames and compute the patch count and representative (first-frame)
    timestamp of each clip."""
    t = video_tensor.shape[0]
    h = video_tensor.shape[2]
    w = video_tensor.shape[3]

    num_clips = t // config.patch_temporal
    patches_per_clip = (
        (h // config.patch_height)
        * (w // config.patch_width)
        // (config.downsample_factor**2)
    )

    clip_timestamps = [
        factored_timestamps[i * config.patch_temporal] for i in range(num_clips)
    ]

    return VideoClipMetadata(
        num_clips=num_clips,
        patches_per_clip=patches_per_clip,
        clip_timestamps=clip_timestamps,
    )


def _video_patch_count(
    video_tensor: torch.Tensor,
    config: PreprocessConfig,
) -> int:
    """Number of output patch tokens for a preprocessed [T, C, H, W] video."""
    t, _, h, w = video_tensor.shape
    patches_per_frame = (
        (h // config.patch_height)
        * (w // config.patch_width)
        // (config.downsample_factor**2)
    )
    return patches_per_frame * (t // config.patch_temporal)


def _preprocess_single_video(
    video_item: object,
    config: PreprocessConfig,
) -> tuple[torch.Tensor, int, list[float]]:
    """Sample using original frame-count metadata, then preprocess the video."""
    if isinstance(video_item, tuple) and len(video_item) == 2:
        frames, metadata = video_item
    else:
        frames = video_item
        metadata = {}

    frames = cast(list[np.ndarray] | np.ndarray, frames)
    fps = float(metadata.get("fps", 1.0))
    frames_indices = metadata.get("frames_indices", list(range(len(frames))))
    total_video_frames = metadata.get("total_num_frames", len(frames))
    # Respect both frame-factor and temporal-patch divisibility.
    ff = max(config.frame_factor or 2, config.patch_temporal)
    if config.sampling_fps is not None and config.sampling_fps > 0 and fps > 0:
        target_num_frames = int(total_video_frames * config.sampling_fps / fps)
    else:
        target_num_frames = len(frames)
    # Short high-fps clips still downsample to the frame factor.
    target_num_frames = min(
        max(target_num_frames, ff), config.num_frames_per_video, len(frames)
    )
    target_num_frames = max(ff, (target_num_frames // ff) * ff)

    local_frames = frames
    local_frames_indices = frames_indices
    if len(frames) > target_num_frames:
        indices = np.linspace(0, len(frames) - 1, target_num_frames, dtype=int)
        local_frames = [frames[i] for i in indices]
        local_frames_indices = [frames_indices[i] for i in indices]

    timestamps = [
        float(local_frames_indices[i]) / fps for i in range(len(local_frames))
    ]

    video_tensor, _, factored_ts = preprocess_video(local_frames, timestamps, config)

    return video_tensor, _video_patch_count(video_tensor, config), factored_ts


cast(dict[str, str], VIDEO_PROCESSOR_MAPPING_NAMES).setdefault(
    "muse_spark", "MuseGlimmerVideoProcessor"
)


@VIDEO_LOADER_REGISTRY.register(
    "muse_spark",
    video_processor="MuseGlimmerVideoProcessor",
)
class MuseSparkVideoBackend(VideoBackend):
    """Sample Muse video frames at load time, preserving metadata."""

    @classmethod
    def compute_frames_index_to_sample(
        cls,
        source: "VideoSourceMetadata",
        target: "VideoTargetMetadata",
        **kwargs: Any,
    ) -> list[int]:
        # Match preprocessing sampling before the loader decodes frames.
        defaults = PreprocessConfig()
        sampling_fps = target.fps if target.fps > 0 else defaults.sampling_fps
        num_frames_per_video = defaults.num_frames_per_video
        ff = max(defaults.frame_factor or 2, defaults.patch_temporal)

        total = source.total_frames_num
        video_fps = source.original_fps
        if total <= 0:
            return []
        if video_fps <= 0 or not sampling_fps or sampling_fps <= 0:
            n = min(num_frames_per_video, total)
        else:
            n = int(total * sampling_fps / video_fps)
            n = min(max(n, ff), num_frames_per_video, total)
        n = max(ff, (n // ff) * ff)
        if n >= total:
            return list(range(total))
        return np.linspace(0, total - 1, n, dtype=int).tolist()


class MuseSparkProcessingInfo(BaseProcessingInfo):
    """Processing info for the omni perception model."""

    def get_placeholder(self, modality: str) -> str:
        if modality == "image":
            return "<|patch|>"
        if modality in ("audio", "video"):
            return f"<|{modality}|>"
        raise ValueError(f"Unsupported Muse Spark modality: {modality}")

    def _has_audio_encoder(self) -> bool:
        return self.get_hf_config().audio_config is not None

    @cached_property
    def processor_config(self) -> dict[str, Any]:
        model_config = self.ctx.model_config
        return (
            get_hf_file_to_dict(
                "processor_config.json",
                model_config.model,
                revision=model_config.revision,
            )
            or {}
        )

    @property
    def audio_sample_rate(self) -> int:
        return self.processor_config.get("feature_extractor", {}).get(
            "sampling_rate", AUDIO_TARGET_SAMPLE_RATE
        )

    @cached_property
    def preprocess_config(self) -> PreprocessConfig:
        config = self.get_hf_config()
        vision = config.vision_config
        processor = self.processor_config
        image = processor.get("image_processor", {})
        video = processor.get("video_processor", {})
        return PreprocessConfig(
            patch_height=vision.patch_size,
            patch_width=vision.patch_size,
            patch_temporal=vision.patch_temporal,
            downsample_factor=vision.merge_size,
            max_num_image_tokens=image.get("max_image_tokens", 4096),
            image_mean=tuple(image.get("image_mean", DEFAULT_IMAGE_MEAN)),
            image_std=tuple(image.get("image_std", DEFAULT_IMAGE_STD)),
            video_downsample_factor=vision.patch_temporal,
            num_frames_per_video=video.get("num_frames", 600),
            sampling_fps=video.get("fps", 2.0),
            frame_factor=vision.patch_temporal,
            max_num_tokens_per_frame=video.get("max_video_frame_tokens", 256),
            video_frame_timestamp_format=video.get(
                "frame_timestamp_format", "Time: {timestamp:.2f}s"
            ),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        limits: dict[str, int | None] = {"image": None, "video": None}
        if self._has_audio_encoder():
            limits["audio"] = None
        return limits

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        config = self.preprocess_config
        max_image_tokens = config.max_num_image_tokens
        max_video_tokens = (
            config.num_frames_per_video // config.video_downsample_factor
        ) * config.max_num_tokens_per_frame
        fallback = 1024

        modality_max: dict[str, int] = {}
        for modality in mm_counts:
            if modality == "video":
                modality_max[modality] = max_video_tokens
            elif modality == "image":
                modality_max[modality] = max_image_tokens
            else:
                modality_max[modality] = fallback
        return modality_max

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=self.audio_sample_rate,
            video_needs_metadata=True,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class MuseSparkDummyInputsBuilder(BaseDummyInputsBuilder[MuseSparkProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        parts: list[str] = ["Question:"]
        parts += ["<|patch|>"] * mm_counts.get("image", 0)
        parts += ["<|video|>"] * mm_counts.get("video", 0)
        parts += ["<|audio|>"] * mm_counts.get("audio", 0)
        parts.append("what's in the media? Answer:")
        return " ".join(parts)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        data: dict[str, Any] = {}
        num_images = mm_counts.get("image", 0)
        if num_images:
            data["image"] = self._get_dummy_images(
                width=448,
                height=448,
                num_images=num_images,
                overrides=cast(ImageDummyOptions | None, mm_options.get("image")),
            )
        num_videos = mm_counts.get("video", 0)
        if num_videos:
            videos = self._get_dummy_videos(
                width=448,
                height=448,
                num_frames=4,
                num_videos=num_videos,
                overrides=cast(VideoDummyOptions | None, mm_options.get("video")),
            )
            data["video"] = [
                (
                    video,
                    {
                        "fps": 1.0,
                        "frames_indices": list(range(len(video))),
                        "total_num_frames": len(video),
                    },
                )
                for video in videos
            ]
        num_audios = mm_counts.get("audio", 0)
        if num_audios:
            data["audio"] = self._get_dummy_audios(
                length=self.info.audio_sample_rate,
                num_audios=num_audios,
                overrides=cast(AudioDummyOptions | None, mm_options.get("audio")),
            )
        return data


class MuseSparkMultiModalProcessor(BaseMultiModalProcessor[MuseSparkProcessingInfo]):
    """Omni processor using native preprocessing instead of an HF processor."""

    def _get_preprocess_config(self) -> PreprocessConfig:
        return self.info.preprocess_config

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Run native image/video/audio preprocessing."""
        valid_mm_items = mm_items.select(
            {k for k, c in mm_items.get_all_counts().items() if c > 0}
        )
        mm_data, _, passthrough_data = self._get_hf_mm_inputs(valid_mm_items, hf_kwargs)

        config = self._get_preprocess_config()
        result: dict[str, Any] = {}

        images = mm_data.get("images")
        if images:
            preprocessed: list[torch.Tensor] = []
            patch_counts: list[int] = []
            for image in cast(Sequence[Image.Image], images):
                img_tensor, num_tokens = preprocess_image(image, config)
                preprocessed.append(img_tensor)
                patch_counts.append(num_tokens)
            result["images"] = preprocessed
            result["image_patch_counts"] = patch_counts

        videos = mm_data.get("videos")
        if videos:
            preprocessed_videos: list[torch.Tensor] = []
            video_patch_counts: list[int] = []
            clip_num_clips: list[int] = []
            clip_patches_per_clip: list[int] = []
            clip_timestamps: list[list[float]] = []
            for video_item in cast(Sequence[object], videos):
                video_tensor, count, factored_ts = _preprocess_single_video(
                    video_item, config
                )
                clip_meta = compute_video_clip_metadata(
                    video_tensor, factored_ts, config
                )
                preprocessed_videos.append(video_tensor)
                video_patch_counts.append(count)
                clip_num_clips.append(clip_meta.num_clips)
                clip_patches_per_clip.append(clip_meta.patches_per_clip)
                clip_timestamps.append(clip_meta.clip_timestamps)
            result["videos"] = preprocessed_videos
            result["video_patch_counts"] = video_patch_counts
            result["video_clip_num_clips"] = clip_num_clips
            result["video_clip_patches_per_clip"] = clip_patches_per_clip
            result["video_clip_timestamps"] = clip_timestamps

        audios = mm_data.get("audio")
        if audios:
            if not self.info._has_audio_encoder():
                raise ValueError("Audio input requires a configured audio encoder.")
            input_features: list[torch.Tensor] = []
            audio_token_counts: list[int] = []
            for audio_item in cast(Sequence[Any], audios):
                if isinstance(audio_item, np.ndarray):
                    wav = torch.from_numpy(audio_item).float()
                elif isinstance(audio_item, torch.Tensor):
                    wav = audio_item.float()
                else:
                    wav = torch.tensor(audio_item, dtype=torch.float32)
                if wav.dim() > 1:
                    wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav.squeeze(0)
                input_features.append(wav)
                audio_token_counts.append(math.ceil(len(wav) / WAVEFORM_PER_TOKEN))
            result["input_features"] = input_features
            result["audio_token_counts"] = audio_token_counts

        processed_data = BatchFeature(data=result)
        processed_data.update(passthrough_data)
        return processed_data

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        config: dict[str, MultiModalFieldConfig] = {}
        if hf_inputs.get("images") is not None:
            config["images"] = MultiModalFieldConfig.batched("image")
            config["image_patch_counts"] = MultiModalFieldConfig.batched(
                "image", keep_on_cpu=True
            )
        if hf_inputs.get("videos") is not None:
            config["videos"] = MultiModalFieldConfig.batched("video")
            config["video_patch_counts"] = MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            )
            config["video_clip_num_clips"] = MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            )
            config["video_clip_patches_per_clip"] = MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            )
            config["video_clip_timestamps"] = MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            )
        if hf_inputs.get("input_features") is not None:
            config["input_features"] = MultiModalFieldConfig.batched("audio")
            config["audio_token_counts"] = MultiModalFieldConfig.batched(
                "audio", keep_on_cpu=True
            )
        return config

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        preprocess_config = self._get_preprocess_config()

        image_start_id = vocab.get("<|image_start|>")
        image_end_id = vocab.get("<|image_end|>")
        patch_id = vocab.get("<|patch|>")

        video_token_id = vocab.get("<|video|>")
        vid_start_id = vocab.get("<|vid_start|>")
        vid_end_id = vocab.get("<|vid_end|>")
        vid_frame_separator_id = vocab.get("<|vid_frame_separator|>")

        audio_token_id = vocab.get("<|audio|>")
        soft_audio_start_id = vocab.get("<|soft_audio_start|>")
        soft_audio_end_id = vocab.get("<|soft_audio_end|>")

        updates: list[PromptUpdate] = []

        if (
            image_start_id is not None
            and image_end_id is not None
            and patch_id is not None
        ):
            img_start, img_end, patch = image_start_id, image_end_id, patch_id

            def get_image_replacement(item_idx: int) -> PromptUpdateDetails:
                n = out_mm_kwargs["image"][item_idx]["image_patch_counts"].data
                if isinstance(n, torch.Tensor):
                    n = n.item()
                count = int(cast(Any, n))
                full = [img_start] + [patch] * count + [img_end]

                def is_embed(_full: list[int], _n: int = count) -> torch.Tensor:
                    return torch.tensor(
                        [False] + [True] * _n + [False], dtype=torch.bool
                    )

                return PromptUpdateDetails(full=full, is_embed=is_embed)

            updates.append(
                PromptReplacement(
                    modality="image",
                    target=tokenizer.encode("<|patch|>", add_special_tokens=False),
                    replacement=get_image_replacement,
                )
            )

        if (
            video_token_id is not None
            and vid_start_id is not None
            and vid_end_id is not None
            and vid_frame_separator_id is not None
        ):
            vid_token = video_token_id
            vid_start, vid_end = vid_start_id, vid_end_id
            vid_frame_separator = vid_frame_separator_id

            def get_video_replacement(item_idx: int) -> PromptUpdateDetails:
                item = out_mm_kwargs["video"][item_idx]
                num_clips = int(cast(Any, item["video_clip_num_clips"].data))
                patches_per_clip = int(
                    cast(Any, item["video_clip_patches_per_clip"].data)
                )
                clip_timestamps = item["video_clip_timestamps"].data
                if isinstance(clip_timestamps, torch.Tensor):
                    clip_timestamps = clip_timestamps.tolist()

                # Interleave timestamp text with video-token runs and frame separators.
                full: list[int] = [vid_start]
                is_embed_flags: list[bool] = [False]

                for clip_idx in range(num_clips):
                    if clip_idx > 0:
                        full.append(vid_frame_separator)
                        is_embed_flags.append(False)

                    # No trailing newline, matching training.
                    ts_text = preprocess_config.video_frame_timestamp_format.format(
                        timestamp=clip_timestamps[clip_idx]
                    )
                    ts_token_ids = tokenizer.encode(ts_text, add_special_tokens=False)
                    full.extend(ts_token_ids)
                    is_embed_flags.extend([False] * len(ts_token_ids))

                    full.extend([vid_token] * patches_per_clip)
                    is_embed_flags.extend([True] * patches_per_clip)

                full.append(vid_end)
                is_embed_flags.append(False)

                def is_embed(
                    _full: list[int], _flags: list[bool] = is_embed_flags
                ) -> torch.Tensor:
                    return torch.tensor(_flags, dtype=torch.bool)

                return PromptUpdateDetails(full=full, is_embed=is_embed)

            updates.append(
                PromptReplacement(
                    modality="video",
                    target=tokenizer.encode("<|video|>", add_special_tokens=False),
                    replacement=get_video_replacement,
                )
            )

        if (
            audio_token_id is not None
            and soft_audio_start_id is not None
            and soft_audio_end_id is not None
        ):
            soft_audio_start, soft_audio_end = soft_audio_start_id, soft_audio_end_id

            def get_audio_replacement(item_idx: int) -> PromptUpdateDetails:
                n = out_mm_kwargs["audio"][item_idx]["audio_token_counts"].data
                if isinstance(n, torch.Tensor):
                    n = n.item()
                count = int(cast(Any, n))
                full = [soft_audio_start] + [audio_token_id] * count + [soft_audio_end]

                def is_embed(_full: list[int], _n: int = count) -> torch.Tensor:
                    return torch.tensor(
                        [False] + [True] * _n + [False], dtype=torch.bool
                    )

                return PromptUpdateDetails(full=full, is_embed=is_embed)

            updates.append(
                PromptReplacement(
                    modality="audio",
                    target=tokenizer.encode("<|audio|>", add_special_tokens=False),
                    replacement=get_audio_replacement,
                )
            )

        return updates
