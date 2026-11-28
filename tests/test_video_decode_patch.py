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
"""The one-pass OpenCV decode returns upstream's frames and metadata."""

import hashlib
import inspect
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import vllm
from vllm.multimodal.video import (
    Qwen2VLVideoBackend,
    Qwen3VLVideoBackend,
    VideoBackend,
)
from vllm.multimodal.video_decoders import opencv
from vllm.multimodal.video_decoders.base import VideoTargetMetadata

from vllm_torchtpu import video_decode_patch

PINNED_VLLM_VERSION = "0.33.0"
PINNED_SOURCE_HASHES = {
    "decode_opencv": (
        "a8725e87614ddcbe2742be11cefaecd8b8bce68fd038f88e62fc5f05b978aed8"
    ),
    "get_video_metadata": (
        "955354bc80417a7a11752afdf03c80acd06f516298f9ba25191b6e72c37ede0a"
    ),
    "open_video_capture": (
        "5b14dd5a610ba31df88091fe507ee3fe191e07384b6baa10f8d8816737bdd0d9"
    ),
}


@pytest.mark.skipif(
    vllm.__version__ != PINNED_VLLM_VERSION,
    reason=f"decode_opencv reproduces vLLM {PINNED_VLLM_VERSION}",
)
@pytest.mark.parametrize(
    ("name", "fn"),
    [
        ("decode_opencv", video_decode_patch.UPSTREAM_DECODE_OPENCV),
        ("get_video_metadata", video_decode_patch.UPSTREAM_GET_VIDEO_METADATA),
        ("open_video_capture", video_decode_patch.UPSTREAM_OPEN_VIDEO_CAPTURE),
    ],
)
def test_upstream_sources_unchanged(name, fn):
    """Fail when upstream changes what the decode patches reproduce."""
    digest = hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()
    assert digest == PINNED_SOURCE_HASHES[name], (
        f"Upstream {name} has drifted; re-sync src/vllm_torchtpu/video_decode_patch.py."
    )


def _clip(tmp_path, fourcc: str, ext: str, keep: float = 1.0) -> bytes:
    """30 random 64x48 frames at 10 fps; `keep` < 1 truncates the file, which
    leaves an AVI header that still claims all 30 frames."""
    path = tmp_path / f"clip.{ext}"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 10.0, (64, 48))
    rng = np.random.default_rng(0)
    for _ in range(30):
        writer.write(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
    writer.release()
    data = path.read_bytes()
    return data[: int(len(data) * keep)]


@pytest.mark.parametrize(
    ("fourcc", "ext", "keep", "one_pass"),
    [
        ("mp4v", "mp4", 1.0, True),  # inter-coded frames, honest header
        ("MJPG", "avi", 0.5, False),  # header over-counts: 30 claimed, 14 left
    ],
)
@pytest.mark.parametrize(
    ("loader", "target", "ends_at_last"),
    [
        (Qwen3VLVideoBackend, VideoTargetMetadata(-1, 2, 300), True),
        (VideoBackend, VideoTargetMetadata(8, -1, 300), True),
        # Samples 0, 5, ..., 25 of 30: nothing is read before upstream's path.
        (Qwen2VLVideoBackend, VideoTargetMetadata(-1, 2, 300), False),
    ],
)
def test_one_pass_decode_matches_upstream(
    tmp_path, fourcc, ext, keep, one_pass, loader, target, ends_at_last
):
    data = _clip(tmp_path, fourcc, ext, keep)
    want_frames, want_source, want_idx, want_valid = (
        video_decode_patch.UPSTREAM_DECODE_OPENCV(loader, data, target, {})
    )
    with patch.object(
        opencv.OpenCVVideoBackendMixin,
        "get_video_metadata",
        wraps=opencv.OpenCVVideoBackendMixin.get_video_metadata,
    ) as metadata:
        frames, source, idx, valid = video_decode_patch.decode_opencv(
            loader, data, target, {}
        )
    assert np.array_equal(frames, want_frames)
    assert (source, idx, valid) == (want_source, want_idx, want_valid)
    # The honest header skips upstream's seek-to-end check when the samples
    # end at its last frame; otherwise the decode takes upstream's path.
    assert metadata.called is not (one_pass and ends_at_last)


def test_decode_threads_capture_matches_upstream(tmp_path):
    """The thread-bounded capture decodes upstream's frames with its threads."""
    data = _clip(tmp_path, "mp4v", "mp4")
    target = VideoTargetMetadata(-1, 2, 300)
    want = video_decode_patch.UPSTREAM_DECODE_OPENCV(
        Qwen3VLVideoBackend, data, target, {}
    )
    with patch.object(
        opencv.OpenCVVideoBackendMixin,
        "open_video_capture",
        classmethod(video_decode_patch.open_video_capture),
    ):
        cap = opencv.OpenCVVideoBackendMixin.open_video_capture(data)
        got = video_decode_patch.UPSTREAM_DECODE_OPENCV(
            Qwen3VLVideoBackend, data, target, {}
        )
    assert cap.get(cv2.CAP_PROP_N_THREADS) == video_decode_patch.decode_threads()
    assert np.array_equal(got[0], want[0])
    assert got[1:] == want[1:]
