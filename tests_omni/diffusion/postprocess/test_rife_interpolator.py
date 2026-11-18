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
"""TPU tests for post-generation RIFE video frame interpolation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    FloatVideoConsumer,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoTransportConstraints,
    VideoValueRange,
)
from vllm_omni.diffusion.postprocess.media import finalize_diffusion_media
from vllm_omni.diffusion.postprocess.rife_interpolator import (
    FrameInterpolator,
    IFNet,
    interpolate_video_tensor,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

ANCHOR_FRAME_MAX_REL_L2 = 1e-5
RIFE_INTERP_MAX_REL_L2 = 1e-2


@pytest.mark.parametrize(
    ("exp", "scale"),
    [(1, 1.0), (2, 0.5)],
    ids=["exp1_scale1.0", "exp2_scale0.5"],
)
def test_post_generation_frame_interpolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exp: int,
    scale: float,
) -> None:
    """Post-Generation Frame Interpolation

    - Runs ``finalize_diffusion_media`` with RIFE interpolation on TPU tensors.
    - Setup check: IFNet loads onto ``"tpu"``, FPS multiplier equals ``2**exp``,
      and output frame count equals ``2**exp * (3 - 1) + 1``.
    - Outcome check: anchor frames match input frames and all interpolated frames
      match a CPU fp32 ``FrameInterpolator`` reference via relative L2.
    """
    torch.manual_seed(0)
    ifnet = IFNet()
    torch.save(ifnet.state_dict(), tmp_path / "flownet.pkl")

    video_cpu = torch.rand((1, 3, 3, 64, 64), dtype=torch.float32)
    video_tpu = video_cpu.to("tpu")

    interpolator = FrameInterpolator(model_path=str(tmp_path))
    loaded_model = interpolator._ensure_model_loaded()
    assert loaded_model.device().type == "tpu", (
        f"Expected RIFE IFNet on tpu, got {loaded_model.device()}"
    )

    media_tpu = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=video_tpu,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BCTHW,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.ZERO_TO_ONE,
            ),
            constraints=VideoTransportConstraints(
                pending_float_consumers=frozenset(
                    {FloatVideoConsumer.FRAME_INTERPOLATION}
                )
            ),
        ),
        prepared_for_transport=True,
    )
    sampling_params = OmniDiffusionSamplingParams(
        enable_frame_interpolation=True,
        frame_interpolation_exp=exp,
        frame_interpolation_scale=scale,
        frame_interpolation_model_path=str(tmp_path),
        output_type="np",
    )
    finalized = finalize_diffusion_media(media_tpu, sampling_params=sampling_params)

    expected_multiplier = 2**exp
    expected_frames = expected_multiplier * (3 - 1) + 1
    actual_multiplier = finalized["metadata"]["video"]["video_fps_multiplier"]
    assert actual_multiplier == expected_multiplier, (
        f"exp={exp} video_fps_multiplier {actual_multiplier} != {expected_multiplier}"
    )

    tpu_frames = torch.from_numpy(finalized["payload"]["video"])
    assert tpu_frames.shape == (1, expected_frames, 64, 64, 3), (
        f"exp={exp} frame shape {tuple(tpu_frames.shape)} != "
        f"(1, {expected_frames}, 64, 64, 3)"
    )

    anchor_ref = video_cpu.permute(0, 2, 3, 4, 1)
    assert_parity(
        tpu_frames[:, ::expected_multiplier],
        anchor_ref,
        name=f"rife anchor frames exp={exp} scale={scale}",
        max_rel_l2=ANCHOR_FRAME_MAX_REL_L2,
    )

    monkeypatch.setattr(
        "vllm_omni.diffusion.postprocess.rife_interpolator._select_torch_device",
        lambda: torch.device("cpu"),
    )
    cpu_interp_bcthw, cpu_mult = interpolate_video_tensor(
        video_cpu,
        exp=exp,
        scale=scale,
        model_path=str(tmp_path),
    )
    assert cpu_mult == expected_multiplier, (
        f"CPU reference multiplier {cpu_mult} != {expected_multiplier}"
    )
    cpu_frames = cpu_interp_bcthw.permute(0, 2, 3, 4, 1)
    assert_parity(
        tpu_frames,
        cpu_frames,
        name=f"rife interpolated frames exp={exp} scale={scale}",
        max_rel_l2=RIFE_INTERP_MAX_REL_L2,
    )
