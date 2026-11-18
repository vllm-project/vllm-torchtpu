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
"""TPU tests for device-side video uint8 postprocessing and SHM transport."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from vllm_omni.diffusion.data import (
    DiffusionOutput,
    OmniDiffusionConfig,
    VideoOutputTransportConfig,
)
from vllm_omni.diffusion.ipc import (
    pack_diffusion_output_shm,
    unpack_diffusion_output_shm,
)
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoValueRange,
)
from vllm_omni.diffusion.postprocess.device_reduction import (
    prepare_diffusion_media_for_transport,
)
from vllm_omni.diffusion.postprocess.media import finalize_diffusion_media
from vllm_omni.entrypoints.openai.video_api_utils import _coerce_video_to_uint8_frames
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.omni_test_utils import PR_TIMEOUT_S

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

BF16_UINT8_MAX_DIFF = 1


@pytest.mark.parametrize(
    "enable_borrowed_frames", [False, True], ids=["copy", "borrow"]
)
@pytest.mark.parametrize("enable_registered_shm", [False, True], ids=["shm", "reg_shm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_device_side_video_postprocessing_and_transport(
    dtype: torch.dtype,
    enable_registered_shm: bool,
    enable_borrowed_frames: bool,
) -> None:
    """Device-Side Video Postprocessing & Transport

    - Runs ``prepare_diffusion_media_for_transport`` on a >1 MB TPU video tensor
      with ``enable_device_postprocess=True``, then round-trips through SHM
      packing/unpacking and ``finalize_diffusion_media``.
    - Setup check: ``prepared.video.spec.encoding`` is ``UINT8_FRAMES``, the
      tensor stays contiguous uint8 ``[B, F, H, W, C]`` on ``"tpu"``, and the
      SHM payload is 4x smaller than float32.
    - Outcome check: finalized uint8 frames and ``_coerce_video_to_uint8_frames``
      match the CPU float32 host-postprocessed reference (exact for fp32, <= 1
      uint8 level for bf16).
    """
    torch.manual_seed(0)
    video_fp32_cpu = torch.rand((1, 3, 8, 256, 256), dtype=torch.float32) * 2.4 - 1.2
    video_tpu = video_fp32_cpu.to(dtype=dtype).to("tpu")

    spec = VideoTensorSpec(
        layout=VideoTensorLayout.BCTHW,
        encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
        value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
    )
    media_tpu = DiffusionMediaOutput(
        video=VideoMediaOutput(tensor=video_tpu, spec=spec)
    )
    od_config = OmniDiffusionConfig(
        video_output_transport=VideoOutputTransportConfig(
            enable_device_postprocess=True,
            enable_registered_shm=enable_registered_shm,
            enable_borrowed_frames=enable_borrowed_frames,
        )
    )
    prepared = prepare_diffusion_media_for_transport(media_tpu, od_config=od_config)

    assert prepared.video.spec.encoding is VideoTensorEncoding.UINT8_FRAMES, (
        f"Expected UINT8_FRAMES encoding, got {prepared.video.spec.encoding}"
    )
    assert prepared.video.tensor.device.type == "tpu", (
        f"Expected reduced tensor on tpu, got {prepared.video.tensor.device}"
    )
    assert prepared.video.tensor.dtype is torch.uint8, (
        f"Expected uint8 tensor, got {prepared.video.tensor.dtype}"
    )
    assert prepared.video.tensor.is_contiguous(), "Expected contiguous uint8 tensor"
    assert prepared.video.tensor.shape == (1, 8, 256, 256, 3), (
        f"Expected [1, 8, 256, 256, 3], got {tuple(prepared.video.tensor.shape)}"
    )

    packed = pack_diffusion_output_shm(
        DiffusionOutput(media=prepared),
        enable_registered_shm=enable_registered_shm,
    )
    shm_nbytes = packed.media["video"]["tensor"]["nbytes"]
    expected_fp32_nbytes = video_fp32_cpu.numel() * 4
    assert shm_nbytes * 4 == expected_fp32_nbytes, (
        f"SHM payload {shm_nbytes} bytes is not 4x smaller than fp32 "
        f"{expected_fp32_nbytes} bytes"
    )

    unpacked = unpack_diffusion_output_shm(packed)
    sampling_params = OmniDiffusionSamplingParams(output_type="np")
    finalized = finalize_diffusion_media(
        unpacked.media, sampling_params=sampling_params
    )
    tpu_frames = finalized["payload"]["video"][0]
    coerced_tpu_frames = _coerce_video_to_uint8_frames(tpu_frames)

    ref_media = DiffusionMediaOutput(
        video=VideoMediaOutput(tensor=video_fp32_cpu, spec=spec)
    )
    ref_config = OmniDiffusionConfig(
        video_output_transport=VideoOutputTransportConfig(
            enable_device_postprocess=False
        )
    )
    ref_prepared = prepare_diffusion_media_for_transport(
        ref_media, od_config=ref_config
    )
    ref_finalized = finalize_diffusion_media(
        ref_prepared, sampling_params=sampling_params
    )
    ref_frames = _coerce_video_to_uint8_frames(ref_finalized["payload"]["video"][0])

    max_diff = int(
        np.max(np.abs(tpu_frames.astype(np.int16) - ref_frames.astype(np.int16)))
    )
    max_coerced_diff = int(
        np.max(
            np.abs(coerced_tpu_frames.astype(np.int16) - ref_frames.astype(np.int16))
        )
    )
    limit = 0 if dtype == torch.float32 else BF16_UINT8_MAX_DIFF
    assert max_diff <= limit, (
        f"dtype={dtype} finalized uint8 max diff {max_diff} exceeds limit {limit}"
    )
    assert max_coerced_diff <= limit, (
        f"dtype={dtype} coerced uint8 max diff {max_coerced_diff} exceeds limit {limit}"
    )
