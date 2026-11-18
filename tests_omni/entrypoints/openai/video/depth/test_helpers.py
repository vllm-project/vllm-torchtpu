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
"""TPU test for the Video Depth Estimation API (/v1/videos/depth)."""

from __future__ import annotations

import base64
import io
from argparse import Namespace
from types import SimpleNamespace
from typing import Any

import av
import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from torch import nn
from vllm.entrypoints.openai.models.protocol import BaseModelPath

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

NUM_FRAMES = 3
VIDEO_HEIGHT = 28
VIDEO_WIDTH = 28
INPUT_SIZE = 28
FPS = 24
DEPTH_MAX_REL_L2 = 5e-2
DEPTH_MAX_ABS = 5e-2


def _encode_test_mp4() -> bytes:
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = VIDEO_WIDTH
        stream.height = VIDEO_HEIGHT
        stream.pix_fmt = "yuv420p"
        for idx in range(NUM_FRAMES):
            ramp = np.linspace(20, 220, VIDEO_WIDTH, dtype=np.uint8)
            frame_rgb = np.broadcast_to(
                ramp[None, :, None], (VIDEO_HEIGHT, VIDEO_WIDTH, 3)
            ).copy()
            frame_rgb[..., 1] = np.uint8(40 * idx)
            frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buf.getvalue()


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason=(
        "Video depth estimation API and VideoDepthAnythingPipeline are not in "
        "vllm-omni 0.32.0rc1"
    ),
)
def test_video_depth_estimation_api() -> None:
    """Video Depth Estimation API on TPU with random-weight VideoDepthAnythingPipeline.

    - TPU path: ``POST /v1/videos/depth`` backed by ``VideoDepthAnythingPipeline``
      (``vits`` backbone) running on ``"tpu"``.
    - Setup check: ``pipeline._device.type == "tpu"`` and ``supports_video_depth``
      resolves True for ``VideoDepthAnythingPipeline``.
    - Outcome check: HTTP 200 rendered ``inferno`` MP4 and returned ``.npz`` depth
      maps match the CPU fp32 ``VideoDepthAnythingPipeline.estimate`` reference.
    """
    from vllm_omni.diffusion.models.video_depth_anything.pipeline_video_depth_anything import (  # noqa: E501
        INFER_LEN,
        VideoDepthAnythingPipeline,
    )
    from vllm_omni.diffusion.models.video_depth_anything.vda_model import (
        VideoDepthAnything,
        VideoDepthAnythingConfig,
    )
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.entrypoints.openai.api_server import router
    from vllm_omni.entrypoints.openai.models.serving import _DiffusionServingModels
    from vllm_omni.entrypoints.openai.video.depth.helpers import supports_video_depth
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    torch.manual_seed(0)
    vda_cfg = VideoDepthAnythingConfig("vits", metric=False)
    cpu_model = VideoDepthAnything(vda_cfg, num_frames=INFER_LEN).float().eval()
    state = {k: v.detach().clone() for k, v in cpu_model.state_dict().items()}

    def build_pipeline(device: str) -> VideoDepthAnythingPipeline:
        model = VideoDepthAnything(vda_cfg, num_frames=INFER_LEN).float()
        model.load_state_dict(state, strict=True)
        pipe = object.__new__(VideoDepthAnythingPipeline)
        nn.Module.__init__(pipe)
        pipe.config = vda_cfg
        pipe.model = model.to(device).eval()
        pipe.od_config = SimpleNamespace(
            dtype=torch.float32,
            model_class_name="VideoDepthAnythingPipeline",
            supports_multimodal_inputs=False,
        )
        return pipe

    cpu_pipe = build_pipeline("cpu")
    tpu_pipe = build_pipeline("tpu")

    class _VideoDepthEngine:
        def __init__(self, pipeline: VideoDepthAnythingPipeline) -> None:
            self.pipeline = pipeline
            self.od_config = pipeline.od_config

        def get_diffusion_od_config(self) -> Any:
            return self.od_config

        async def generate(
            self,
            prompt: Any,
            request_id: str,
            sampling_params_list: list[OmniDiffusionSamplingParams] | None = None,
            **kwargs: Any,
        ):
            del kwargs
            params = (
                sampling_params_list[0]
                if sampling_params_list
                else OmniDiffusionSamplingParams()
            )
            req = OmniDiffusionRequest(
                prompt=prompt, sampling_params=params, request_id=request_id
            )
            out = self.pipeline.forward(DiffusionRequestBatch(requests=[req]))
            assert out.error is None, (
                f"VideoDepthAnythingPipeline forward failed: {out.error}"
            )
            yield SimpleNamespace(multimodal_output=out.output)

    engine = _VideoDepthEngine(tpu_pipe)
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = engine
    app.state.diffusion_engine = engine
    app.state.stage_configs = [SimpleNamespace(stage_type="diffusion")]
    app.state.openai_serving_models = _DiffusionServingModels(
        [
            BaseModelPath(
                name="depth-anything/Video-Depth-Anything-Small", model_path="vda"
            )
        ]
    )
    app.state.args = Namespace(
        default_sampling_params=None, max_generated_image_size=None
    )

    probe_req = Request({"type": "http", "app": app, "headers": []})
    assert tpu_pipe._device.type == "tpu", (
        f"Expected VideoDepthAnythingPipeline on tpu, got {tpu_pipe._device}"
    )
    assert supports_video_depth(probe_req, engine) is True, (
        "supports_video_depth returned False for VideoDepthAnythingPipeline"
    )

    mp4_bytes = _encode_test_mp4()
    with TestClient(app) as client:
        response = client.post(
            "/v1/videos/depth",
            files={"video": ("clip.mp4", mp4_bytes, "video/mp4")},
            data={
                "input_size": str(INPUT_SIZE),
                "colormap": "inferno",
                "return_depth": "true",
            },
        )

    assert response.status_code == 200, (
        f"POST /v1/videos/depth failed ({response.status_code}): {response.text}"
    )
    body = response.json()
    assert body["num_frames"] == NUM_FRAMES, (
        f"Expected {NUM_FRAMES} depth frames, got {body['num_frames']}"
    )
    assert body["width"] == VIDEO_WIDTH, (
        f"Expected width {VIDEO_WIDTH}, got {body['width']}"
    )
    assert body["height"] == VIDEO_HEIGHT, (
        f"Expected height {VIDEO_HEIGHT}, got {body['height']}"
    )
    assert body["metric"] is False, f"Expected metric=False, got {body['metric']}"
    assert body["depth"] is not None, (
        "Expected base64 depth archive when return_depth=true"
    )

    with np.load(io.BytesIO(base64.b64decode(body["depth"]))) as archive:
        tpu_depth = archive["depth"].astype(np.float32)
    with av.open(io.BytesIO(base64.b64decode(body["video"]))) as vid_container:
        rendered_frames = [
            f.to_ndarray(format="rgb24") for f in vid_container.decode(video=0)
        ]
    assert len(rendered_frames) == NUM_FRAMES, (
        f"Expected {NUM_FRAMES} rendered depth video frames, got {len(rendered_frames)}"
    )

    with av.open(io.BytesIO(mp4_bytes)) as src_container:
        src_frames = np.stack(
            [f.to_ndarray(format="rgb24") for f in src_container.decode(video=0)]
        )
    cpu_depth = cpu_pipe.estimate(src_frames, input_size=INPUT_SIZE)["depth"].astype(
        np.float32
    )

    assert_parity(
        torch.from_numpy(tpu_depth),
        torch.from_numpy(cpu_depth),
        name="video_depth_anything_tpu_vs_cpu",
        max_rel_l2=DEPTH_MAX_REL_L2,
        max_abs=DEPTH_MAX_ABS,
    )
