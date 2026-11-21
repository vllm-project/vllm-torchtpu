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
"""Nightly end-to-end T2V smoke, seed determinism and golden tests for LTX-2.

Runs the real ``Lightricks/LTX-2`` one-stage audio+video diffusion pipeline on a
single TPU v7x chip with the canonical 512x768x25 request (24 fps, 4-pass
default guidance with CFG, STG, and modality isolation):
1. ``test_ltx2_t2v_smoke_tpu``: at 2 steps, verifies output container types,
   shapes, dtypes, finite values, non-degenerate video spatial variance and
   temporal motion, and non-silent stereo 24 kHz audio (24,240 samples per
   channel).
2. ``test_ltx2_seed_determinism_tpu``: at 2 steps, verifies bitwise-identical
   video and audio across repeated runs with seed 42 and divergence against
   seed 43.
3. ``test_ltx2_golden_answer_tpu``: runs the request at 40 steps and checks it
   against a TPU fp32 golden within ``LTX2_GOLDEN_LIMITS``. The golden
   (``ltx2_t2v_full_default_40step_tpu_fp32_golden.npz`` in
   ``$LTX2_GOLDEN_DIR``, default ``$HF_HOME/golden/ltx2``) is the same request
   with fp32 weights and tensor_parallel_size=4; it matches CPU fp32 diffusers
   to video MAE 0.0036.
4. ``test_ltx2_golden_answer_multichip_ulysses_tpu``: the same golden request
   and limits on CFG-parallel 2 x Ulysses 4 (8 TensorCores). Ulysses pads the
   26 audio latent tokens to 28 and sends a key-padding mask to audio
   self-attention and audio-to-video attention, a path the single-chip tests
   never take. It is marked ``multichip``, so only the Omni Multichip
   (Nightly) lane (tpu7x 2x2x1) runs it.

The first three share one single-chip engine and run in about 270s on TPU v7x,
well within the 1800s CI Nightly per-test timeout.
"""

import numpy as np
import pytest

pytest.importorskip("vllm_omni", reason="vllm_omni is required for LTX-2 tests")

from torch_tpu._internal.utils import hardware
from vllm_omni.diffusion.data import DiffusionParallelConfig

from tests_omni.conftest import OmniRunner
from tests_omni.models.ltx2.utils import (
    LTX2_GOLDEN_EXTRA_ARGS,
    LTX2_GOLDEN_STEPS,
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
def test_ltx2_t2v_smoke_tpu(omni_runner: OmniRunner) -> None:
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
def test_ltx2_seed_determinism_tpu(omni_runner: OmniRunner) -> None:
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


# NOTE: How the golden was generated (outside pytest, on 4 chips).
# 1. Omni model launch, with the tests_omni/conftest.py env defaults:
#        omni = Omni(model=get_ltx2_model(), dtype=LTX2_GOLDEN_DTYPE,
#                    tensor_parallel_size=LTX2_GOLDEN_TENSOR_PARALLEL_SIZE)
# 2. Request, the same as this test's:
#        request = build_ltx2_request(num_inference_steps=LTX2_GOLDEN_STEPS,
#                                     extra_args=LTX2_GOLDEN_EXTRA_ARGS)
#        out = extract_ltx2_outputs(list(omni.generate(LTX2_PROMPT, request)))
# 3. Output saving, with the checked meta plus the checkpoint's HF revision:
#        meta = {**ltx2_golden_meta(), "revision": <HF commit>}
#        np.savez_compressed(path, video=out.video, audio=out.audio,
#                            meta=json.dumps(meta))
# LTX2_GOLDEN_LIMITS are calibrated on the same request with the default
# (bf16, 1 chip) Omni.
@pytest.mark.parametrize(
    "omni_runner",
    [get_ltx2_model()],
    indirect=True,
)
def test_ltx2_golden_answer_tpu(omni_runner: OmniRunner) -> None:
    """Compare TPU v7x bf16 video and audio with the TPU fp32 golden."""
    out = extract_ltx2_outputs(
        list(
            omni_runner.generate(
                LTX2_PROMPT,
                build_ltx2_request(
                    num_inference_steps=LTX2_GOLDEN_STEPS,
                    extra_args=LTX2_GOLDEN_EXTRA_ARGS,
                ),
            )
        )
    )
    assert_ltx2_golden_parity(out)


# CFG-parallel 2 x Ulysses 4, passed as flat OmniRunner kwargs as in
# tests_omni/models/wan/test_wan_multirank.py. Ulysses 4 pads the 26 audio latent
# tokens of the golden request to 28 and masks the 2 padded keys; Ulysses 2
# would not pad them (26 divides by 2). Before #2878 this masked path returned
# wrong attention on torch_tpu 0.2.0 (google-pytorch/torch_tpu#5677), and this
# test failed audio_logspec_l1 (1.366), video_ssim (0.640), video_mae (0.067)
# and video_mean_shift (0.042); with #2878 it measured 0.330, 0.931, 0.022 and
# 0.004.
LTX2_MULTICHIP_DEGREES = {"cfg_parallel_size": 2, "ulysses_degree": 4}


@pytest.mark.multichip
@pytest.mark.parametrize(
    "omni_runner",
    [(get_ltx2_model(), None, LTX2_MULTICHIP_DEGREES)],
    indirect=True,
)
def test_ltx2_golden_answer_multichip_ulysses_tpu(omni_runner: OmniRunner) -> None:
    """Compare CFG-parallel 2 x Ulysses 4 video and audio with the TPU fp32 golden."""
    world_size = DiffusionParallelConfig(**LTX2_MULTICHIP_DEGREES).world_size
    # PCI scan: counts TPU devices without starting the TPU runtime here.
    device_count = hardware.get_tpu_device_count()
    assert device_count >= world_size, (
        f"Test requires {world_size} TPU devices, found {device_count}"
    )
    # Fail rather than silently run unpadded if the degrees were dropped.
    stage_parallel = omni_runner.stage_configs[0].parallel_config
    expected = {
        "cfg_parallel_size": 1,
        "ulysses_degree": 1,
        "ring_degree": 1,
        "tensor_parallel_size": 1,
        **LTX2_MULTICHIP_DEGREES,
    }
    actual = {name: getattr(stage_parallel, name) for name in expected}
    assert actual == expected, (
        f"Engine parallel degrees {actual} != expected {expected}"
    )

    out = extract_ltx2_outputs(
        list(
            omni_runner.generate(
                LTX2_PROMPT,
                build_ltx2_request(
                    num_inference_steps=LTX2_GOLDEN_STEPS,
                    extra_args=LTX2_GOLDEN_EXTRA_ARGS,
                ),
            )
        )
    )
    assert_ltx2_golden_parity(out)
