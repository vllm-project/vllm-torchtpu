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
"""Nightly end-to-end T2V smoke and seed determinism tests for LTX-2 on TPU v7x.

Runs the real ``Lightricks/LTX-2`` one-stage audio+video diffusion pipeline on a
single TPU v7x chip using the canonical 2-step 512x768x25 request (24 fps,
4-pass default guidance with CFG, STG, and modality isolation):
1. ``test_ltx2_t2v_smoke_tpu``: verifies output container types, shapes, dtypes,
   finite values, non-degenerate video spatial variance and temporal motion, and
   non-silent stereo 24 kHz audio (24,240 samples per channel).
2. ``test_ltx2_seed_determinism_tpu``: verifies bitwise-identical video and
   audio across repeated runs with seed 42 and divergence against seed 43.

Measured on TPU v7x (``j-50558160-e001-496d-a3cf``, 158.69s total for module,
well within the 1800s CI Nightly per-test timeout):
- Engine initialization: 58.81s; first request (with XLA compile): 89.32s
  (``test_ltx2_t2v_smoke_tpu`` total: 148.13s).
- Warm requests (``test_ltx2_seed_determinism_tpu``): 2.15s / 2.10s / 2.09s
  (7.53s total including teardown).
- Smoke metrics (seed 42): ``min_frame_std = 30.275`` (floor 5.0),
  ``mean_frame_std = 31.020``, ``temporal_diff_mean = 4.089`` (floor 0.5),
  ``audio_rms = 0.020038`` (floor 1e-3), ``audio_max_abs = 0.135742``.
- All three seed=42 requests (including the cold first request) match bitwise
  (``video_sha256 = 0e9bc093...``, ``audio_sha256 = 382ac716...``).
"""

import hashlib
import time
from collections.abc import Callable

import numpy as np
import pytest

pytest.importorskip("vllm_omni", reason="vllm_omni is required for LTX-2 tests")

from tests_omni.conftest import OmniRunner
from tests_omni.models.ltx2.utils import (
    LTX2_PROMPT,
    build_ltx2_request,
    extract_ltx2_outputs,
    get_ltx2_model,
    record,
)

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.spawns_tpu_workers,
]

# Plausibility floors for 2-step 512x768x25 T2V output (uint8 video in [0, 255],
# float32 stereo waveform at 24 kHz).
VIDEO_MIN_FRAME_STD_FLOOR = 5.0
VIDEO_TEMPORAL_DIFF_FLOOR = 0.5
AUDIO_RMS_FLOOR = 1e-3


@pytest.mark.parametrize(
    "omni_runner",
    [get_ltx2_model()],
    indirect=True,
)
def test_ltx2_t2v_smoke_tpu(
    omni_runner: OmniRunner,
    record_property: Callable[[str, object], None],
) -> None:
    """Verify LTX-2 T2V output shapes, dtypes, and non-degeneracy on TPU v7x."""
    t0 = time.perf_counter()
    raw_outputs = list(omni_runner.generate(LTX2_PROMPT, build_ltx2_request(seed=42)))
    req_elapsed_s = time.perf_counter() - t0

    out = extract_ltx2_outputs(raw_outputs)

    video_f32 = out.video.astype(np.float32)
    frame_stds = video_f32.std(axis=(1, 2, 3))
    min_frame_std = float(frame_stds.min())
    mean_frame_std = float(frame_stds.mean())
    temporal_diff_mean = float(np.abs(np.diff(video_f32, axis=0)).mean())

    audio_rms = float(np.sqrt(np.mean(out.audio**2)))
    audio_max_abs = float(np.max(np.abs(out.audio)))
    channel_diff_max = float(np.max(np.abs(out.audio[0] - out.audio[1])))

    video_sha256 = hashlib.sha256(out.video.tobytes()).hexdigest()
    audio_sha256 = hashlib.sha256(out.audio.tobytes()).hexdigest()

    record(
        record_property,
        "t2v-smoke",
        req_elapsed_s=round(req_elapsed_s, 3),
        video_shape=list(out.video.shape),
        audio_shape=list(out.audio.shape),
        sample_rate=out.sample_rate,
        fps=out.fps,
        min_frame_std=min_frame_std,
        mean_frame_std=mean_frame_std,
        temporal_diff_mean=temporal_diff_mean,
        audio_rms=audio_rms,
        audio_max_abs=audio_max_abs,
        channel_diff_max=channel_diff_max,
        video_sha256=video_sha256,
        audio_sha256=audio_sha256,
    )

    assert min_frame_std >= VIDEO_MIN_FRAME_STD_FLOOR, (
        f"Video min frame std {min_frame_std:.3f} < {VIDEO_MIN_FRAME_STD_FLOOR}"
    )
    assert temporal_diff_mean >= VIDEO_TEMPORAL_DIFF_FLOOR, (
        f"Video temporal diff {temporal_diff_mean:.3f} < {VIDEO_TEMPORAL_DIFF_FLOOR}"
    )
    assert audio_rms >= AUDIO_RMS_FLOOR, (
        f"Audio RMS {audio_rms:.6f} below floor {AUDIO_RMS_FLOOR}"
    )


@pytest.mark.parametrize(
    "omni_runner",
    [get_ltx2_model()],
    indirect=True,
)
def test_ltx2_seed_determinism_tpu(
    omni_runner: OmniRunner,
    record_property: Callable[[str, object], None],
) -> None:
    """Verify identical seeds produce bitwise-equal video and audio on TPU v7x."""
    t0 = time.perf_counter()
    out_42_a = extract_ltx2_outputs(
        list(omni_runner.generate(LTX2_PROMPT, build_ltx2_request(seed=42)))
    )
    t1 = time.perf_counter()
    out_42_b = extract_ltx2_outputs(
        list(omni_runner.generate(LTX2_PROMPT, build_ltx2_request(seed=42)))
    )
    t2 = time.perf_counter()
    out_43 = extract_ltx2_outputs(
        list(omni_runner.generate(LTX2_PROMPT, build_ltx2_request(seed=43)))
    )
    t3 = time.perf_counter()

    video_42_a_sha = hashlib.sha256(out_42_a.video.tobytes()).hexdigest()
    video_42_b_sha = hashlib.sha256(out_42_b.video.tobytes()).hexdigest()
    video_43_sha = hashlib.sha256(out_43.video.tobytes()).hexdigest()
    audio_42_a_sha = hashlib.sha256(out_42_a.audio.tobytes()).hexdigest()
    audio_42_b_sha = hashlib.sha256(out_42_b.audio.tobytes()).hexdigest()
    audio_43_sha = hashlib.sha256(out_43.audio.tobytes()).hexdigest()

    record(
        record_property,
        "seed-determinism",
        run1_s=round(t1 - t0, 3),
        run2_s=round(t2 - t1, 3),
        run3_seed43_s=round(t3 - t2, 3),
        video_42_a_sha256=video_42_a_sha,
        video_42_b_sha256=video_42_b_sha,
        video_43_sha256=video_43_sha,
        audio_42_a_sha256=audio_42_a_sha,
        audio_42_b_sha256=audio_42_b_sha,
        audio_43_sha256=audio_43_sha,
    )

    np.testing.assert_array_equal(
        out_42_a.video,
        out_42_b.video,
        err_msg="Identical seed 42 on TPU v7x must produce bitwise-identical video",
    )
    np.testing.assert_array_equal(
        out_42_a.audio,
        out_42_b.audio,
        err_msg="Identical seed 42 on TPU v7x must produce bitwise-identical audio",
    )
    assert not np.array_equal(out_42_a.video, out_43.video), (
        "Different seeds (42 vs 43) must produce different video outputs"
    )
    assert not np.array_equal(out_42_a.audio, out_43.audio), (
        "Different seeds (42 vs 43) must produce different audio outputs"
    )
