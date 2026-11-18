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
"""TPU test for the Video Segmentation API (/v1/videos/segmentations)."""

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

from tests_omni.omni_test_utils import PR_TIMEOUT_S

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

IMAGE_SIZE = 112
VIDEO_WIDTH = 32
VIDEO_HEIGHT = 32
NUM_FRAMES = 3
FPS = 24
TRACKER_HIDDEN = 128
SCORE_ATOL = 5e-2
BOX_ATOL = 1.0

TINY_DETECTOR_CONFIG: dict[str, Any] = {
    "vision_config": {
        "backbone_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": IMAGE_SIZE,
            "patch_size": 14,
            "window_size": 4,
            "global_attn_indexes": [1],
            "pretrain_image_size": 56,
        },
        "fpn_hidden_size": TRACKER_HIDDEN,
        "backbone_feature_sizes": [[32, 32], [16, 16], [8, 8]],
    },
    "text_config": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "projection_dim": TRACKER_HIDDEN,
        "vocab_size": 128,
        "max_position_embeddings": 32,
    },
    "geometry_encoder_config": {
        "hidden_size": TRACKER_HIDDEN,
        "num_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 64,
    },
    "detr_encoder_config": {
        "hidden_size": TRACKER_HIDDEN,
        "num_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 64,
    },
    "detr_decoder_config": {
        "hidden_size": TRACKER_HIDDEN,
        "num_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 64,
        "num_queries": 4,
    },
    "mask_decoder_config": {"hidden_size": TRACKER_HIDDEN, "num_attention_heads": 2},
}

TINY_TRACKER_CONFIG: dict[str, Any] = {
    "image_size": IMAGE_SIZE,
    "vision_config": {
        "fpn_hidden_size": TRACKER_HIDDEN,
        "backbone_feature_sizes": [[32, 32], [16, 16], [8, 8]],
    },
    "prompt_encoder_config": {
        "hidden_size": TRACKER_HIDDEN,
        "image_size": IMAGE_SIZE,
        "patch_size": 14,
        "mask_input_channels": 8,
    },
    "mask_decoder_config": {
        "hidden_size": TRACKER_HIDDEN,
        "mlp_dim": 64,
        "num_attention_heads": 2,
        "iou_head_hidden_dim": 32,
    },
    "memory_attention_hidden_size": TRACKER_HIDDEN,
    "memory_attention_num_layers": 2,
    "memory_attention_feed_forward_hidden_size": 64,
    "memory_attention_rope_feat_sizes": [8, 8],
    "memory_encoder_hidden_size": TRACKER_HIDDEN,
    "memory_encoder_output_channels": 64,
    "mask_downsampler_embed_dim": TRACKER_HIDDEN,
    "memory_fuser_embed_dim": TRACKER_HIDDEN,
    "memory_fuser_intermediate_dim": 64,
}


class _FixedTokenizer:
    def __call__(self, text: str, **kwargs: Any) -> dict[str, torch.Tensor]:
        del kwargs
        ids = torch.zeros(1, 32, dtype=torch.long)
        tokens = [1] + [min(ord(c), 120) for c in text[:6]] + [2]
        ids[0, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        mask = (ids != 0).long()
        return {"input_ids": ids, "attention_mask": mask}


def _encode_test_mp4() -> bytes:
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = VIDEO_WIDTH
        stream.height = VIDEO_HEIGHT
        stream.pix_fmt = "yuv420p"
        for idx in range(NUM_FRAMES):
            arr = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
            arr[4:20, 4 + idx * 2 : 20 + idx * 2, 0] = 220
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buf.getvalue()


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="Video segmentation API and Sam3VideoModel are not in vllm-omni 0.32.0rc1",
)
def test_video_segmentation_api() -> None:
    """Video Segmentation API on TPU with a tiny random-weight Sam3Pipeline.

    - TPU path: ``POST /v1/videos/segmentations`` backed by ``Sam3Pipeline``
      (detector + memory tracker) running on ``"tpu"``.
    - Setup check: ``pipeline._device.type == "tpu"``, ``pipeline.supports_video``
      is True, and ``supports_video_segmentation`` resolves True.
    - Outcome check: HTTP 200 response frames, tracked objects, RLE masks, and
      decoded ``mask_video`` match the CPU fp32 ``Sam3Pipeline`` reference.
    """
    from vllm_omni.diffusion.models.sam3.pipeline_sam3 import Sam3Pipeline
    from vllm_omni.diffusion.models.sam3.sam3_model import (
        Sam3Detector,
        Sam3DetectorConfig,
        Sam3VisionNeck,
    )
    from vllm_omni.diffusion.models.sam3.sam3_tracker import (
        Sam3Tracker,
        Sam3TrackerConfig,
    )
    from vllm_omni.diffusion.models.sam3.sam3_video import Sam3VideoConfig
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.entrypoints.openai.api_server import router
    from vllm_omni.entrypoints.openai.models.serving import _DiffusionServingModels
    from vllm_omni.entrypoints.openai.video.segmentation.helpers import (
        supports_video_segmentation,
    )
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    torch.manual_seed(0)
    det_cfg = Sam3DetectorConfig.from_dict(TINY_DETECTOR_CONFIG)
    trk_cfg = Sam3TrackerConfig.from_dict(TINY_TRACKER_CONFIG)
    vid_cfg = Sam3VideoConfig(
        score_threshold_detection=0.0,
        new_det_thresh=0.0,
        hotstart_delay=0,
    )

    def build_pipeline(device: str) -> Sam3Pipeline:
        pipe = object.__new__(Sam3Pipeline)
        nn.Module.__init__(pipe)
        pipe.config = det_cfg
        pipe.image_size = IMAGE_SIZE
        pipe.detector = Sam3Detector(det_cfg).to(device).eval()
        pipe.tracker = Sam3Tracker(trk_cfg).to(device).eval()
        pipe.tracker_neck = Sam3VisionNeck(det_cfg.vision_config).to(device).eval()
        pipe.video_config = vid_cfg
        pipe.tokenizer = _FixedTokenizer()
        return pipe

    cpu_pipe = build_pipeline("cpu")
    with torch.no_grad():
        for param in cpu_pipe.tracker.parameters():
            param.normal_(0.0, 0.1)
    state = {k: v.detach().clone() for k, v in cpu_pipe.state_dict().items()}

    tpu_pipe = build_pipeline("tpu")
    tpu_pipe.load_state_dict(state, strict=True)

    class _Sam3VideoEngine:
        def __init__(self, pipeline: Sam3Pipeline) -> None:
            self.pipeline = pipeline
            self.od_config = SimpleNamespace(
                model_class_name="Sam3Pipeline", supports_multimodal_inputs=True
            )

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
            assert out.error is None, f"Sam3Pipeline video forward failed: {out.error}"
            yield SimpleNamespace(multimodal_output=out.output)

    engine = _Sam3VideoEngine(tpu_pipe)
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = engine
    app.state.diffusion_engine = engine
    app.state.stage_configs = [SimpleNamespace(stage_type="diffusion")]
    app.state.openai_serving_models = _DiffusionServingModels(
        [BaseModelPath(name="facebook/sam3", model_path="sam3")]
    )
    app.state.args = Namespace(
        default_sampling_params=None, max_generated_image_size=None
    )

    probe_req = Request({"type": "http", "app": app, "headers": []})
    assert tpu_pipe._device.type == "tpu", (
        f"Expected Sam3Pipeline on tpu, got {tpu_pipe._device}"
    )
    assert tpu_pipe.supports_video is True, "Sam3Pipeline tracker was not initialized"
    assert supports_video_segmentation(probe_req, engine) is True, (
        "supports_video_segmentation returned False for Sam3Pipeline"
    )

    mp4_bytes = _encode_test_mp4()
    with TestClient(app) as client:
        response = client.post(
            "/v1/videos/segmentations",
            files={"video": ("clip.mp4", mp4_bytes, "video/mp4")},
            data={"prompt": "person", "mask_video": "true"},
        )

    assert response.status_code == 200, (
        f"POST /v1/videos/segmentations failed ({response.status_code}): "
        f"{response.text}"
    )
    body = response.json()
    assert body["width"] == VIDEO_WIDTH, (
        f"Expected width {VIDEO_WIDTH}, got {body['width']}"
    )
    assert body["height"] == VIDEO_HEIGHT, (
        f"Expected height {VIDEO_HEIGHT}, got {body['height']}"
    )
    assert body["num_frames"] == NUM_FRAMES, (
        f"Expected {NUM_FRAMES} frames, got {body['num_frames']}"
    )
    assert len(body["frames"]) == NUM_FRAMES, (
        f"Expected {NUM_FRAMES} frame entries, got {len(body['frames'])}"
    )

    with av.open(io.BytesIO(mp4_bytes)) as src_container:
        rgb_frames = [
            f.to_ndarray(format="rgb24") for f in src_container.decode(video=0)
        ]
    cpu_ref = cpu_pipe.segment_video(rgb_frames, ["person"])

    assert len(body["objects"]) == len(cpu_ref["objects"]), (
        f"Object count {len(body['objects'])} != CPU ref {len(cpu_ref['objects'])}"
    )
    for obj in body["objects"]:
        assert obj["mask_video"] is not None, f"Object {obj['id']} missing mask_video"
        with av.open(io.BytesIO(base64.b64decode(obj["mask_video"]))) as mask_container:
            decoded_masks = [
                f.to_ndarray(format="gray") for f in mask_container.decode(video=0)
            ]
        assert len(decoded_masks) == NUM_FRAMES, (
            f"mask_video frame count {len(decoded_masks)} != {NUM_FRAMES}"
        )

    for frame_idx, (tpu_frame, ref_frame) in enumerate(
        zip(body["frames"], cpu_ref["frames"])
    ):
        tpu_ids = [inst["id"] for inst in tpu_frame["instances"]]
        assert tpu_ids == ref_frame["object_ids"], (
            f"Frame {frame_idx} object_ids {tpu_ids} != "
            f"CPU ref {ref_frame['object_ids']}"
        )
        tpu_scores = np.asarray(
            [inst["score"] for inst in tpu_frame["instances"]], dtype=np.float32
        )
        ref_scores = np.asarray(ref_frame["scores"], dtype=np.float32)
        score_err = (
            float(np.max(np.abs(tpu_scores - ref_scores))) if len(ref_scores) else 0.0
        )
        assert score_err <= SCORE_ATOL, (
            f"Frame {frame_idx} max score diff {score_err:.4e} exceeds {SCORE_ATOL}"
        )
        tpu_boxes = np.asarray(
            [inst["box"] for inst in tpu_frame["instances"]], dtype=np.float32
        )
        ref_boxes = np.asarray(ref_frame["boxes"], dtype=np.float32)
        box_err = (
            float(np.max(np.abs(tpu_boxes - ref_boxes))) if len(ref_boxes) else 0.0
        )
        assert box_err <= BOX_ATOL, (
            f"Frame {frame_idx} max box diff {box_err:.4e} exceeds {BOX_ATOL}"
        )
