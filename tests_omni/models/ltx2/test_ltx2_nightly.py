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
3. ``test_ltx2_golden_answer_tpu``: compares video and audio with a CPU fp32
   diffusers golden (``ltx2_t2v_full_default_cpu_golden.npz`` in
   ``$LTX2_GOLDEN_DIR``, default ``$HF_HOME/golden/ltx2``) using the limits in
   ``LTX2_GOLDEN_LIMITS``. The golden reproduces the TPU trajectory: the same
   bf16 initial noise drawn by vllm-omni, vllm-omni's sigmas, the same 4-pass
   guidance (STG at block 29) and negative prompt. Audio is checked by
   log-spectrogram L1 and RMS ratio only: bf16 alone moves the 2-step waveform
   by rel-L2 ~1.0 (CPU bf16 vs CPU fp32), as much as TPU does.
   Regenerate the golden with the out-of-repo generate_ltx2_golden.py:
   ``python generate_ltx2_golden.py --model-dir <Lightricks/LTX-2 dir>
   --out-dir <dir> --runs ltx2_t2v_full_default`` (CPU, ~170 GB RAM, ~5 min on
   64 cores).

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

from collections.abc import Callable

import numpy as np
import pytest

pytest.importorskip("vllm_omni", reason="vllm_omni is required for LTX-2 tests")

from tests_omni.conftest import OmniRunner
from tests_omni.models.ltx2.utils import (
    LTX2_GOLDEN_EXTRA_ARGS,
    LTX2_PROMPT,
    assert_ltx2_golden_parity,
    build_ltx2_request,
    extract_ltx2_outputs,
    get_ltx2_model,
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
    raw_outputs = list(omni_runner.generate(LTX2_PROMPT, build_ltx2_request(seed=42)))

    out = extract_ltx2_outputs(raw_outputs)

    video_f32 = out.video.astype(np.float32)
    frame_stds = video_f32.std(axis=(1, 2, 3))
    min_frame_std = float(frame_stds.min())
    temporal_diff_mean = float(np.abs(np.diff(video_f32, axis=0)).mean())

    audio_rms = float(np.sqrt(np.mean(out.audio**2)))

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
    out_42_a = extract_ltx2_outputs(
        list(omni_runner.generate(LTX2_PROMPT, build_ltx2_request(seed=42)))
    )
    out_42_b = extract_ltx2_outputs(
        list(omni_runner.generate(LTX2_PROMPT, build_ltx2_request(seed=42)))
    )
    out_43 = extract_ltx2_outputs(
        list(omni_runner.generate(LTX2_PROMPT, build_ltx2_request(seed=43)))
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


@pytest.mark.parametrize(
    "omni_runner",
    [get_ltx2_model()],
    indirect=True,
)
def test_ltx2_golden_answer_tpu(omni_runner: OmniRunner) -> None:
    """Compare TPU v7x video and audio with the CPU fp32 diffusers golden."""
    out = extract_ltx2_outputs(
        list(
            omni_runner.generate(
                LTX2_PROMPT, build_ltx2_request(extra_args=LTX2_GOLDEN_EXTRA_ARGS)
            )
        )
    )
    assert_ltx2_golden_parity(out)
