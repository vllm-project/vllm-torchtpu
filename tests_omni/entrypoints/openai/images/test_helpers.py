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
"""TPU tests for vLLM-Omni image editing and image segmentation HTTP routes."""

from __future__ import annotations

import argparse
import base64
import contextlib
import errno
import importlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import diffusers
import filelock
import numpy as np
import psutil
import pytest
import requests
import torch
import transformers
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.entrypoints.openai.models.serving import _DiffusionServingModels
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests.tpu_release import wait_for_tpu_release
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni]

QWEN_EDIT_MODEL_ID = "Qwen/Qwen-Image-Edit"
SERVED_EDIT_NAME = "qwen-edit-tiny"
EDIT_PROMPT = "turn the blue square into a warm orange square"
EDIT_SIZE = 256
EDIT_STEPS = 2
EDIT_CFG_SCALE = 1.0
EDIT_SEED = 42
TEXT_HIDDEN_SIZE = 256
SERVER_STARTUP_TIMEOUT_S = 240.0
REQUEST_TIMEOUT_S = 180.0

SAM3_IMAGE_SIZE = 112
SAM3_UPLOAD_WIDTH = 96
SAM3_UPLOAD_HEIGHT = 80
SAM3_BOXES_MAX_REL_L2 = 1e-4
SAM3_BOXES_MAX_ABS = 5e-2
SAM3_SCORES_MAX_REL_L2 = 1e-4
SAM3_SCORES_MAX_ABS = 1e-4
SAM3_TINY_CONFIG: dict[str, Any] = {
    "vision_config": {
        "backbone_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": SAM3_IMAGE_SIZE,
            "patch_size": 14,
            "window_size": 4,
            "global_attn_indexes": [1],
            "pretrain_image_size": 56,
        },
        "fpn_hidden_size": 32,
        "backbone_feature_sizes": [[32, 32], [16, 16], [8, 8]],
    },
    "text_config": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "projection_dim": 32,
        "vocab_size": 128,
        "max_position_embeddings": 32,
    },
    "geometry_encoder_config": {
        "hidden_size": 32,
        "num_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 64,
    },
    "detr_encoder_config": {
        "hidden_size": 32,
        "num_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 64,
    },
    "detr_decoder_config": {
        "hidden_size": 32,
        "num_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 64,
        "num_queries": 8,
    },
    "mask_decoder_config": {
        "hidden_size": 32,
        "num_attention_heads": 2,
    },
}


@dataclass(frozen=True)
class _QwenEditServerBundle:
    """Running Qwen-Image-Edit server with its input and control input images."""

    base_url: str
    model_dir: str
    proc: subprocess.Popen[bytes]
    input_png_bytes: bytes
    input_tensor: torch.Tensor
    control_png_bytes: bytes


def _shrink_qwen_dit(config: dict[str, Any]) -> dict[str, Any]:
    """Shrink Qwen-Image-Edit DiT to 2 layers of 4 heads (head_dim 128)."""
    return {
        **config,
        "num_layers": 2,
        "num_attention_heads": 4,
        "joint_attention_dim": TEXT_HIDDEN_SIZE,
    }


def _shrink_qwen2_5_vl_for_edit(config: dict[str, Any]) -> dict[str, Any]:
    """Shrink Qwen2.5-VL text and vision towers while preserving mrope head dim.

    Adapted from ``_shrink_qwen2_5_vl`` in ``tests_omni/tiny_models.py``.
    """
    text_config = config["text_config"]
    head_dim = text_config["hidden_size"] // text_config["num_attention_heads"]
    text_dims = {
        "num_hidden_layers": 2,
        "hidden_size": TEXT_HIDDEN_SIZE,
        "num_attention_heads": TEXT_HIDDEN_SIZE // head_dim,
        "num_key_value_heads": 1,
        "intermediate_size": 2 * TEXT_HIDDEN_SIZE,
    }
    vision_dims = {
        "depth": 2,
        "hidden_size": TEXT_HIDDEN_SIZE,
        "num_heads": 4,
        "intermediate_size": 2 * TEXT_HIDDEN_SIZE,
        "out_hidden_size": TEXT_HIDDEN_SIZE,
        "fullatt_block_indexes": [0, 1],
    }
    return {
        **config,
        **text_dims,
        "text_config": {
            **text_config,
            **text_dims,
            "layer_types": text_config["layer_types"][:2],
        },
        "vision_config": {**config["vision_config"], **vision_dims},
    }


def _build_tiny_qwen_image_edit() -> str:
    """Build a deterministic bf16 ``QwenImageEditPipeline`` directory once."""
    root = os.path.join(tempfile.gettempdir(), "vllm-torchtpu-tiny-models")
    model_dir = os.path.join(root, QWEN_EDIT_MODEL_ID.replace("/", "--"))
    if os.path.isdir(model_dir):
        return model_dir

    shrinks = {
        "text_encoder": _shrink_qwen2_5_vl_for_edit,
        "transformer": _shrink_qwen_dit,
    }
    index = diffusers.DiffusionPipeline.load_config(QWEN_EDIT_MODEL_ID)
    pipeline_cls = getattr(diffusers, index["_class_name"])
    init_dict, _, _ = pipeline_cls.extract_init_dict(index)
    _, optional_kwargs = diffusers.DiffusionPipeline._get_signature_keys(pipeline_cls)
    init_kwargs: dict[str, Any] = {
        k: init_dict.pop(k)
        for k in optional_kwargs
        if k in init_dict and k not in pipeline_cls._optional_components
    }
    rng_state = torch.get_rng_state()
    try:
        # CPU generator only: torch.manual_seed() also seeds, and so opens, the TPU.
        torch.default_generator.manual_seed(0)
        for name, (library, class_name) in init_dict.items():
            if library is None:
                continue
            cls = getattr(importlib.import_module(library), class_name)
            shrink = shrinks.get(name)
            if issubclass(cls, diffusers.ModelMixin):
                cfg = cls.load_config(QWEN_EDIT_MODEL_ID, subfolder=name)
                init_kwargs[name] = cls.from_config(shrink(cfg) if shrink else cfg)
            elif issubclass(cls, transformers.PreTrainedModel):
                cfg, _ = transformers.PreTrainedConfig.get_config_dict(
                    QWEN_EDIT_MODEL_ID, subfolder=name
                )
                init_kwargs[name] = cls(
                    cls.config_class.from_dict(shrink(cfg) if shrink else cfg)
                )
            else:
                init_kwargs[name] = cls.from_pretrained(
                    QWEN_EDIT_MODEL_ID, subfolder=name
                )
    finally:
        torch.set_rng_state(rng_state)
    pipeline = pipeline_cls(**init_kwargs)
    os.makedirs(root, exist_ok=True)
    staging_dir = tempfile.mkdtemp(dir=root)
    pipeline.to(torch.bfloat16).save_pretrained(staging_dir)
    os.rename(staging_dir, model_dir)
    return model_dir


def _make_edit_input_png(size: int, square_blue: int) -> tuple[bytes, torch.Tensor]:
    """Create a deterministic patterned RGB PNG for edit conditioning."""
    grid_y = np.linspace(16, 224, size, dtype=np.uint8)[:, None]
    grid_x = np.linspace(32, 200, size, dtype=np.uint8)[None, :]
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[..., 0] = grid_y
    arr[..., 1] = grid_x
    arr[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4, 2] = square_blue
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    raw = buf.getvalue()
    decoded = Image.open(io.BytesIO(raw)).convert("RGB")
    tensor = torch.from_numpy(np.asarray(decoded, dtype=np.uint8)).to(torch.float32)
    return raw, tensor


def _reserve_local_port(host: str = "127.0.0.1") -> tuple[int, filelock.FileLock]:
    """Reserve a localhost TCP port with an advisory file lock."""
    for _ in range(128):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            port = int(sock.getsockname()[1])
        lock_path = (
            Path(tempfile.gettempdir())
            / f"vllm-omni-test-port-{os.getuid()}-{port}.lock"
        )
        lock = filelock.FileLock(str(lock_path))
        try:
            lock.acquire(timeout=0)
        except filelock.Timeout:
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind((host, port))
        except OSError as exc:
            lock.release()
            if exc.errno != errno.EADDRINUSE:
                raise
            continue
        return port, lock
    raise RuntimeError("Could not reserve a free localhost port after 128 attempts")


def _stop_server_process(
    proc: subprocess.Popen[bytes], lock: filelock.FileLock
) -> None:
    """Terminate the ``vllm serve`` process tree and release its port lock."""
    try:
        try:
            parent = psutil.Process(proc.pid)
            children = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        for child in children:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                child.terminate()
        _, alive = psutil.wait_procs(children, timeout=5)
        for child in alive:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                child.kill()
        with contextlib.suppress(ChildProcessError):
            os.waitpid(proc.pid, os.WNOHANG)
    finally:
        if lock.is_locked:
            lock.release()
        wait_for_tpu_release()


def _start_omni_server(
    model_dir: str, serve_args: list[str]
) -> tuple[str, subprocess.Popen[bytes], filelock.FileLock]:
    """Start ``vllm serve --omni`` in a child process and poll ``/health``."""
    wait_for_tpu_release()
    host = "127.0.0.1"
    port, lock = _reserve_local_port(host)
    base_url = f"http://{host}:{port}"
    cmd = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        model_dir,
        "--host",
        host,
        "--port",
        str(port),
        "--omni",
        *serve_args,
    ]
    proc = subprocess.Popen(cmd, env=os.environ.copy())
    try:
        deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            ret = proc.poll()
            if ret is not None:
                raise RuntimeError(f"OmniServer exited early with code {ret} for {cmd}")
            try:
                resp = requests.get(f"{base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    return base_url, proc, lock
            except requests.RequestException:
                pass
            time.sleep(1.0)
        raise TimeoutError(
            f"OmniServer did not pass /health within {SERVER_STARTUP_TIMEOUT_S}s"
        )
    except BaseException:
        _stop_server_process(proc, lock)
        raise


def _edit_form(png_bytes: bytes) -> dict[str, Any]:
    """``files`` and ``data`` of one ``POST /v1/images/edits`` request."""
    return {
        "files": {"image": ("input.png", png_bytes, "image/png")},
        "data": {
            "model": SERVED_EDIT_NAME,
            "prompt": EDIT_PROMPT,
            "size": f"{EDIT_SIZE}x{EDIT_SIZE}",
            "num_inference_steps": str(EDIT_STEPS),
            "true_cfg_scale": str(EDIT_CFG_SCALE),
            "seed": str(EDIT_SEED),
            "output_format": "png",
            "return_stage_metrics": "true",
        },
    }


def _decode_edit_response(resp: requests.Response, label: str) -> torch.Tensor:
    """Check one edit response and decode its PNG to a float32 ``(H, W, 3)`` tensor."""
    assert resp.status_code == 200, (
        f"{label}: POST /v1/images/edits returned {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert body["output_format"] == "png", (
        f"{label}: expected output_format 'png', got {body['output_format']!r}"
    )
    assert body["size"] == f"{EDIT_SIZE}x{EDIT_SIZE}", (
        f"{label}: expected size '{EDIT_SIZE}x{EDIT_SIZE}', got {body['size']!r}"
    )
    assert len(body["data"]) == 1, (
        f"{label}: expected 1 edited image item, got {len(body['data'])}"
    )
    img = _decode_png_bytes(base64.b64decode(body["data"][0]["b64_json"]))
    assert img.shape == (EDIT_SIZE, EDIT_SIZE, 3), (
        f"{label}: decoded image shape {tuple(img.shape)} != "
        f"({EDIT_SIZE}, {EDIT_SIZE}, 3)"
    )
    return img


def _decode_png_bytes(raw_bytes: bytes) -> torch.Tensor:
    """Decode raw PNG bytes into a float32 ``(H, W, 3)`` RGB tensor in ``[0, 255]``."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return torch.from_numpy(np.asarray(img, dtype=np.uint8)).to(torch.float32)


@pytest.fixture(scope="module")
def _qwen_edit_bundle() -> Generator[_QwenEditServerBundle, None, None]:
    model_dir = _build_tiny_qwen_image_edit()
    input_png_bytes, input_tensor = _make_edit_input_png(EDIT_SIZE, square_blue=220)
    control_png_bytes, _ = _make_edit_input_png(EDIT_SIZE, square_blue=0)
    base_url, proc, lock = _start_omni_server(
        model_dir,
        ["--served-model-name", SERVED_EDIT_NAME],
    )
    try:
        yield _QwenEditServerBundle(
            base_url=base_url,
            model_dir=model_dir,
            proc=proc,
            input_png_bytes=input_png_bytes,
            input_tensor=input_tensor,
            control_png_bytes=control_png_bytes,
        )
    finally:
        _stop_server_process(proc, lock)


@pytest.mark.nightly
@pytest.mark.multichip
@pytest.mark.spawns_tpu_workers
@pytest.mark.timeout(NIGHTLY_TIMEOUT_S)
def test_images_edits_route(_qwen_edit_bundle: _QwenEditServerBundle) -> None:
    """Image Editing API.

    - TPU path: Uploads a PNG image and edit prompt via ``POST /v1/images/edits``
      to ``OmniServer`` running a 2-layer bf16 ``QwenImageEditPipeline`` on TPU.
    - Setup check: Asserts ``model_index.json`` declares ``QwenImageEditPipeline``,
      HTTP 200 and a ``(256, 256, 3)`` PNG output for every request.
    - Outcome check: the edit differs from its input image, a repeated request is
      bitwise identical, and a control input image yields a different edit.
    """
    with open(os.path.join(_qwen_edit_bundle.model_dir, "model_index.json")) as f:
        model_index = json.load(f)
    assert model_index["_class_name"] == "QwenImageEditPipeline", (
        f"Expected QwenImageEditPipeline, got {model_index['_class_name']!r}"
    )
    assert _qwen_edit_bundle.proc.poll() is None, (
        f"OmniServer exited unexpectedly with code {_qwen_edit_bundle.proc.poll()}"
    )

    edits_url = f"{_qwen_edit_bundle.base_url}/v1/images/edits"
    resp_a = requests.post(
        edits_url,
        **_edit_form(_qwen_edit_bundle.input_png_bytes),
        timeout=REQUEST_TIMEOUT_S,
    )
    resp_b = requests.post(
        edits_url,
        **_edit_form(_qwen_edit_bundle.input_png_bytes),
        timeout=REQUEST_TIMEOUT_S,
    )
    resp_ctrl = requests.post(
        edits_url,
        **_edit_form(_qwen_edit_bundle.control_png_bytes),
        timeout=REQUEST_TIMEOUT_S,
    )

    edited_a = _decode_edit_response(resp_a, "edit_a")
    edited_b = _decode_edit_response(resp_b, "edit_b")
    edited_ctrl = _decode_edit_response(resp_ctrl, "edit_control")

    assert not torch.equal(edited_a, _qwen_edit_bundle.input_tensor), (
        "Edited output image was identical to the uploaded input image"
    )
    assert torch.equal(edited_b, edited_a), (
        "Repeated edit request diverged: "
        f"max_abs={float((edited_b - edited_a).abs().max())}"
    )
    assert not torch.equal(edited_ctrl, edited_a), (
        "Control input image produced the same edit as the input image"
    )


class _Sam3TestTokenizer:
    """Deterministic fixed-length tokenizer for tiny random-weight ``Sam3Pipeline``."""

    def __call__(self, text: str, **kwargs: Any) -> dict[str, torch.Tensor]:
        del text, kwargs
        input_ids = torch.zeros((1, 32), dtype=torch.long)
        input_ids[:, :5] = torch.tensor([1, 17, 42, 9, 2])
        attention_mask = (input_ids != 0).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _Sam3InlineEngine:
    """Thin async adapter routing ``api_server.router`` calls into ``Sam3Pipeline``."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.od_config = SimpleNamespace(
            model_class_name="Sam3Model",
            supports_multimodal_inputs=True,
        )
        self.default_sampling_params_list = [OmniDiffusionSamplingParams()]

    def get_diffusion_od_config(self) -> Any:
        return self.od_config

    async def generate(
        self,
        prompt: Any,
        request_id: str,
        sampling_params_list: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        params = (
            sampling_params_list[0]
            if sampling_params_list
            else OmniDiffusionSamplingParams()
        )
        req = OmniDiffusionRequest(
            prompt=prompt,
            request_id=request_id,
            sampling_params=params,
        )
        out = self.pipeline(DiffusionRequestBatch(requests=[req]))
        yield SimpleNamespace(multimodal_output=out.output)


def _build_sam3_pipeline(sam3_pipeline_cls: type[Any], detector: Any) -> Any:
    """Instantiate ``Sam3Pipeline`` via ``__new__`` with a pre-built detector."""
    pipeline = object.__new__(sam3_pipeline_cls)
    torch.nn.Module.__init__(pipeline)
    pipeline.detector = detector
    pipeline.tokenizer = _Sam3TestTokenizer()
    pipeline.image_size = SAM3_IMAGE_SIZE
    return pipeline


def _decode_sam3_response_masks(
    items: list[dict[str, Any]], height: int, width: int
) -> torch.Tensor:
    """Decode base64 grayscale PNG instance masks into a bool ``(N, H, W)`` tensor."""
    if not items:
        return torch.zeros((0, height, width), dtype=torch.bool)
    masks = []
    for item in items:
        raw = base64.b64decode(item["mask"])
        arr = np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.uint8)
        masks.append(arr > 0)
    return torch.from_numpy(np.stack(masks, axis=0))


def _unpack_reference_sam3(
    ref_payload: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Order ``Sam3Pipeline.segment`` outputs by descending score to match the route."""
    scores_np = ref_payload["scores"]
    order = np.argsort(-scores_np, kind="stable")
    count, height, width = ref_payload["mask_shape"].tolist()
    unpacked = (
        np.unpackbits(ref_payload["masks"], axis=1, count=height * width)
        .reshape(count, height, width)
        .astype(bool)
    )
    masks = torch.from_numpy(unpacked[order])
    boxes = torch.from_numpy(ref_payload["boxes"][order]).to(torch.float32)
    scores = torch.from_numpy(scores_np[order]).to(torch.float32)
    return masks, boxes, scores


@pytest.mark.timeout(PR_TIMEOUT_S)
@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="not in vllm-omni 0.32.0rc1",
)
def test_image_segmentation_api() -> None:
    """Image Segmentation API.

    - TPU path: Builds a tiny random-weight ``Sam3Pipeline`` on ``"tpu"`` mounted
      behind ``POST /v1/images/segmentations`` (strict xfail while SAM 3 is absent
      in vllm-omni 0.32.0rc1).
    - Setup check: Verifies ``tpu_pipeline._device.type == "tpu"`` and
      ``_supports_segmentation(request_stub, engine)`` returns ``True``.
    - Outcome check: Sends a multipart PNG image, prompt, exemplar boxes, and
      ``score_threshold="0.0"``, then compares returned masks, boxes, and scores
      against an identical-weight CPU fp32 ``Sam3Pipeline``.
    """
    from vllm_omni.diffusion.models.sam3.pipeline_sam3 import Sam3Pipeline
    from vllm_omni.diffusion.models.sam3.sam3_model import (
        Sam3Detector,
        Sam3DetectorConfig,
    )
    from vllm_omni.entrypoints.openai import api_server
    from vllm_omni.entrypoints.openai.images.helpers import _supports_segmentation

    torch.default_generator.manual_seed(0)
    cfg = Sam3DetectorConfig.from_dict(SAM3_TINY_CONFIG)
    cpu_detector = Sam3Detector(cfg).eval().float()
    tpu_detector = Sam3Detector(cfg).eval().float()
    tpu_detector.load_state_dict(cpu_detector.state_dict())
    tpu_detector = tpu_detector.to("tpu")

    cpu_pipeline = _build_sam3_pipeline(Sam3Pipeline, cpu_detector)
    tpu_pipeline = _build_sam3_pipeline(Sam3Pipeline, tpu_detector)

    engine = _Sam3InlineEngine(tpu_pipeline)
    app = FastAPI()
    app.include_router(api_server.router)
    app.state.engine_client = engine
    app.state.diffusion_engine = engine
    app.state.stage_configs = [SimpleNamespace(stage_type="diffusion")]
    app.state.openai_serving_models = _DiffusionServingModels(
        [BaseModelPath(name="facebook/sam3", model_path="sam3")]
    )
    app.state.args = argparse.Namespace(
        default_sampling_params=None,
        max_generated_image_size=None,
    )
    request_stub = SimpleNamespace(app=app)

    assert tpu_pipeline._device.type == "tpu", (
        f"Expected Sam3Pipeline on 'tpu', got {tpu_pipeline._device}"
    )
    assert _supports_segmentation(request_stub, engine) is True, (
        "Expected _supports_segmentation to return True for Sam3Model"
    )

    rng = np.random.default_rng(7)
    rgb = rng.integers(
        0, 255, size=(SAM3_UPLOAD_HEIGHT, SAM3_UPLOAD_WIDTH, 3), dtype=np.uint8
    )
    pil_img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    decoded_pil = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    exemplar_boxes = [[8.0, 8.0, 48.0, 44.0]]
    exemplar_labels = [1]

    with TestClient(app) as client:
        resp = client.post(
            "/v1/images/segmentations",
            files={"image": ("scene.png", png_bytes, "image/png")},
            data={
                "prompt": "segmented object",
                "boxes": json.dumps(exemplar_boxes),
                "box_labels": json.dumps(exemplar_labels),
                "score_threshold": "0.0",
                "mask_format": "png",
            },
        )
    ref_raw = cpu_pipeline.segment(
        decoded_pil,
        "segmented object",
        exemplar_boxes,
        exemplar_labels,
        score_threshold=0.0,
        mask_threshold=0.5,
    )

    assert resp.status_code == 200, (
        f"POST /v1/images/segmentations returned {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert body["width"] == SAM3_UPLOAD_WIDTH, (
        f"Segmentation width {body['width']} != {SAM3_UPLOAD_WIDTH}"
    )
    assert body["height"] == SAM3_UPLOAD_HEIGHT, (
        f"Segmentation height {body['height']} != {SAM3_UPLOAD_HEIGHT}"
    )
    assert body["mask_format"] == "png", (
        f"Segmentation mask_format {body['mask_format']!r} != 'png'"
    )

    ref_masks, ref_boxes, ref_scores = _unpack_reference_sam3(ref_raw)
    actual_masks = _decode_sam3_response_masks(
        body["data"], SAM3_UPLOAD_HEIGHT, SAM3_UPLOAD_WIDTH
    )
    actual_boxes = torch.tensor(
        [item["box"] for item in body["data"]], dtype=torch.float32
    ).reshape(-1, 4)
    actual_scores = torch.tensor(
        [item["score"] for item in body["data"]], dtype=torch.float32
    )

    assert actual_masks.shape == ref_masks.shape, (
        f"Mask shape {tuple(actual_masks.shape)} != {tuple(ref_masks.shape)}"
    )
    assert torch.equal(actual_masks, ref_masks), (
        "TPU Sam3Pipeline binary masks differed from CPU fp32 reference"
    )
    assert_parity(
        actual_boxes,
        ref_boxes,
        name="Sam3Pipeline boxes TPU vs CPU fp32",
        max_rel_l2=SAM3_BOXES_MAX_REL_L2,
        max_abs=SAM3_BOXES_MAX_ABS,
    )
    assert_parity(
        actual_scores,
        ref_scores,
        name="Sam3Pipeline scores TPU vs CPU fp32",
        max_rel_l2=SAM3_SCORES_MAX_REL_L2,
        max_abs=SAM3_SCORES_MAX_ABS,
    )
