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
"""One-pass OpenCV decode for vLLM's frame-sampling video loaders.

vLLM's `decode_opencv` reads the source metadata with `get_video_metadata`,
which opens the stream a second time and seeks to its end and back to check
that the frames the header claims are presentable. That seek decodes: on a
61-frame 480x832 H.264 clip it costs 79 ms of a 157 ms load. `decode_opencv`
here trusts the header count and reads the sampled frames on one stream. When
the sampled indices end at the header's last frame, decoding that frame proves
the stream does not end early, which is the only case in which the check
lowers the count, so the frames and metadata equal upstream's. Otherwise (a
header that over-counts, no header count, other indices, frame recovery) it
takes upstream's path.

The function is a drop-in for `vllm.multimodal.video_decoders.opencv.
decode_opencv`, written to move there unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy.typing as npt
from vllm.multimodal.video_decoders import opencv
from vllm.multimodal.video_decoders.base import (
    VideoSourceMetadata,
    VideoTargetMetadata,
    check_frame_pixel_limit,
)

# The upstream functions this module reproduces or relies on; the tests pin
# their sources.
UPSTREAM_DECODE_OPENCV = opencv.decode_opencv
UPSTREAM_GET_VIDEO_METADATA = opencv.OpenCVVideoBackendMixin.get_video_metadata


def decode_opencv(
    loader_cls: Any,
    data: bytes,
    target: VideoTargetMetadata,
    sampling_kwargs: dict,
    *,
    frame_recovery: bool = False,
) -> tuple[npt.NDArray, VideoSourceMetadata, list[int], list[int]]:
    cv2 = opencv.cv2
    mixin = opencv.OpenCVVideoBackendMixin
    cap = mixin.open_video_capture(data)
    check_frame_pixel_limit(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames_num > 0 and not frame_recovery:
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        source = loader_cls._prepare_source(
            VideoSourceMetadata(
                total_frames_num=total_frames_num,
                original_fps=original_fps,
                duration=total_frames_num / original_fps if original_fps > 0 else 0,
            )
        )
        frame_idx = loader_cls.compute_frames_index_to_sample(
            source=source, target=target, **sampling_kwargs
        )
        if frame_idx and max(frame_idx) == total_frames_num - 1:
            frames, valid = loader_cls.read_frames(
                cap, frame_idx, total_frames_num=total_frames_num
            )
            if valid == sorted(set(frame_idx)):
                return frames, source, frame_idx, valid
            opencv.logger.info(
                "Video header claims %d frames; re-reading with the "
                "presentable frame count.",
                total_frames_num,
            )
        # The stream has moved on; upstream reads from the first frame.
        cap = mixin.open_video_capture(data)
    source = loader_cls._prepare_source(mixin.get_video_metadata(data))
    frame_idx = loader_cls.compute_frames_index_to_sample(
        source=source, target=target, **sampling_kwargs
    )
    frames, valid = loader_cls.read_frames(
        cap,
        frame_idx,
        total_frames_num=source.total_frames_num,
        frame_recovery=frame_recovery,
    )
    return frames, source, frame_idx, valid


def patch_opencv_one_pass_decode() -> None:
    """Serve vLLM's OpenCV video decodes with the one-pass `decode_opencv`.

    `decode_video` looks the decoder up on the module at each call, so
    replacing the module attribute is enough.
    """
    opencv.decode_opencv = decode_opencv
