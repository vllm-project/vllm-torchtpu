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
"""TPU test for the OpenPI Robot Policy WebSocket endpoint."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from torch import nn
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaForConditionalGeneration,
)
from vllm_omni.diffusion.models.pi0.config import Pi0Config
from vllm_omni.diffusion.models.pi0.modeling_pi0 import (
    PaliGemmaWithActionExpert,
    Pi0ForActionPrediction,
)
from vllm_omni.diffusion.models.pi0.pipeline_pi0 import Pi0Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.entrypoints.openai.api_server import router
from vllm_omni.entrypoints.openpi.connection import _pack, _unpack
from vllm_omni.entrypoints.openpi.serving import ServingRealtimeRobotOpenPI
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

IMAGE_SIZE = 32
PATCH_SIZE = 16
VOCAB_SIZE = 128
HIDDEN_SIZE = 64
MLP_DIM = 128
NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = 32
TOKEN_LEN = 8
ACTION_HORIZON = 4
ACTION_DIM = 8
NUM_STEPS = 2

# TPU float32 matmuls execute via bf16 TensorCore passes; scale budget from CPU bf16.
BF16_RATIO = 4.0
REL_L2_FLOOR = 2e-2
ABS_FLOOR = 2e-2

POLICY_SERVER_CONFIG: dict[str, Any] = {
    "action_space": "joint_position",
    "action_horizon": ACTION_HORIZON,
    "action_dim": ACTION_DIM,
    "image_resolution": [IMAGE_SIZE, IMAGE_SIZE],
    "max_cameras": 1,
}


class _DeterministicTokenizer:
    """Minimal deterministic tokenizer matching ``Pi0Pipeline``'s call contract."""

    def __call__(
        self,
        text: str,
        *,
        padding: str,
        max_length: int,
        truncation: bool,
        add_special_tokens: bool,
        return_tensors: str | None,
    ) -> dict[str, list[int]]:
        del padding, truncation, add_special_tokens, return_tensors
        encoded = [min(ord(ch), VOCAB_SIZE - 2) for ch in text[: max_length - 1]]
        ids = (encoded + [0] * max_length)[:max_length]
        mask = [1] * min(len(encoded), max_length)
        mask += [0] * (max_length - len(mask))
        return {"input_ids": ids, "attention_mask": mask}


class _Pi0PipelineEngine:
    """EngineClient adapter driving a real ``Pi0Pipeline`` for OpenPI serving."""

    def __init__(self, pipeline: Pi0Pipeline) -> None:
        self.pipeline = pipeline
        self.default_sampling_params_list = [
            OmniDiffusionSamplingParams(num_inference_steps=NUM_STEPS)
        ]
        self._od_config = SimpleNamespace(
            model_config={"policy_server_config": POLICY_SERVER_CONFIG}
        )

    def get_diffusion_od_config(self) -> Any:
        return self._od_config

    async def generate(
        self,
        *,
        prompt: Any,
        request_id: str,
        sampling_params_list: list[OmniDiffusionSamplingParams],
    ):
        req = OmniDiffusionRequest(
            prompt=prompt,
            sampling_params=sampling_params_list[0],
            request_id=request_id,
        )
        out = self.pipeline.forward(DiffusionRequestBatch(requests=[req]))
        assert out.error is None, f"Pi0Pipeline failed: {out.error}"
        yield SimpleNamespace(multimodal_output=out.output)


def _tiny_pi0_config() -> Pi0Config:
    return Pi0Config(
        chunk_size=ACTION_HORIZON,
        max_action_dim=ACTION_DIM,
        max_state_dim=ACTION_DIM,
        num_inference_steps=NUM_STEPS,
        image_resolution=(IMAGE_SIZE, IMAGE_SIZE),
        tokenizer_max_length=TOKEN_LEN,
        max_cameras=1,
        dtype="float32",
        image_feature_keys=["observation.images.cam0"],
    )


def _build_tiny_pi0_model(config: Pi0Config) -> Pi0ForActionPrediction:
    vlm_cfg = CONFIG_MAPPING["paligemma"]()
    vlm_cfg._vocab_size = VOCAB_SIZE
    vlm_cfg.vocab_size = VOCAB_SIZE
    vlm_cfg.image_token_index = VOCAB_SIZE - 1
    vlm_cfg.projection_dim = HIDDEN_SIZE
    vlm_cfg.text_config.hidden_size = HIDDEN_SIZE
    vlm_cfg.text_config.intermediate_size = MLP_DIM
    vlm_cfg.text_config.num_attention_heads = NUM_HEADS
    vlm_cfg.text_config.head_dim = HEAD_DIM
    vlm_cfg.text_config.num_hidden_layers = NUM_LAYERS
    vlm_cfg.text_config.num_key_value_heads = 1
    vlm_cfg.text_config.hidden_activation = "gelu_pytorch_tanh"
    vlm_cfg.text_config.vocab_size = VOCAB_SIZE
    vlm_cfg.text_config.dtype = "float32"
    vlm_cfg.vision_config.hidden_size = HIDDEN_SIZE
    vlm_cfg.vision_config.intermediate_size = MLP_DIM
    vlm_cfg.vision_config.num_hidden_layers = NUM_LAYERS
    vlm_cfg.vision_config.num_attention_heads = NUM_HEADS
    vlm_cfg.vision_config.image_size = IMAGE_SIZE
    vlm_cfg.vision_config.patch_size = PATCH_SIZE
    vlm_cfg.vision_config.projection_dim = HIDDEN_SIZE
    vlm_cfg.vision_config.projector_hidden_act = "gelu_fast"
    vlm_cfg.vision_config.num_image_tokens = (IMAGE_SIZE // PATCH_SIZE) ** 2
    vlm_cfg.vision_config.dtype = "float32"

    expert_cfg = CONFIG_MAPPING["gemma"](
        head_dim=HEAD_DIM,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=MLP_DIM,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=NUM_LAYERS,
        num_key_value_heads=1,
        vocab_size=VOCAB_SIZE,
        hidden_activation="gelu_pytorch_tanh",
        dtype="float32",
    )

    backbone = object.__new__(PaliGemmaWithActionExpert)
    nn.Module.__init__(backbone)
    backbone.paligemma = PaliGemmaForConditionalGeneration(config=vlm_cfg)
    backbone.gemma_expert = GemmaForCausalLM(config=expert_cfg)
    backbone.gemma_expert.model.embed_tokens = None

    model = object.__new__(Pi0ForActionPrediction)
    nn.Module.__init__(model)
    model.config = config
    model.action_dim = ACTION_DIM
    model.max_state_dim = ACTION_DIM
    model.action_horizon = ACTION_HORIZON
    model.num_inference_steps = NUM_STEPS
    model.vlm_width = HIDDEN_SIZE
    model.expert_width = HIDDEN_SIZE
    model.paligemma_with_expert = backbone
    model.action_in_proj = nn.Linear(ACTION_DIM, HIDDEN_SIZE)
    model.action_out_proj = nn.Linear(HIDDEN_SIZE, ACTION_DIM)
    model.state_proj = nn.Linear(ACTION_DIM, HIDDEN_SIZE)
    model.action_time_mlp_in = nn.Linear(2 * HIDDEN_SIZE, HIDDEN_SIZE)
    model.action_time_mlp_out = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
    model._state_norm = None
    model._action_norm = None
    return model.eval()


def _make_pi0_pipeline(
    config: Pi0Config,
    state_dict: dict[str, torch.Tensor],
    *,
    device: str,
    dtype: torch.dtype,
) -> Pi0Pipeline:
    model = _build_tiny_pi0_model(config)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=dtype).eval()

    pipeline = object.__new__(Pi0Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.config = config
    pipeline._torch_dtype = dtype
    pipeline._device = torch.device(device)
    pipeline.tokenizer = _DeterministicTokenizer()
    pipeline.model = model
    return pipeline


def _build_openpi_app(serving: ServingRealtimeRobotOpenPI) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.openai_serving_realtime_robot = serving
    app.state.args = Namespace(robot_openpi_idle_timeout=30.0)
    return app


def test_openpi_robot_policy_websocket() -> None:
    """OpenPI Robot Policy WebSocket on TPU with a tiny random-weight Pi0Pipeline.

    - TPU path: ``WS /v1/realtime/robot/openpi`` via ``RobotRealtimeConnection`` and
      ``ServingRealtimeRobotOpenPI`` backed by ``Pi0Pipeline`` on ``"tpu"``.
    - Setup check: TPU omni platform active, ``Pi0Pipeline._device.type == "tpu"``,
      and WebSocket handshake returns ``POLICY_SERVER_CONFIG``.
    - Outcome check: MessagePack reset and infer frames return a finite ``(4, 8)``
      float32 action chunk matching the CPU fp32 reference within bf16 budget.
    """
    torch.manual_seed(0)
    config = _tiny_pi0_config()
    base_state = {
        k: v.detach().clone()
        for k, v in _build_tiny_pi0_model(config).state_dict().items()
    }

    tpu_pipeline = _make_pi0_pipeline(
        config, base_state, device="tpu", dtype=torch.float32
    )
    cpu_fp32_pipeline = _make_pi0_pipeline(
        config, base_state, device="cpu", dtype=torch.float32
    )
    cpu_bf16_pipeline = _make_pi0_pipeline(
        config, base_state, device="cpu", dtype=torch.bfloat16
    )

    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU omni platform, got {current_omni_platform.device_type!r}"
    )
    assert tpu_pipeline._device.type == "tpu", (
        f"Expected Pi0Pipeline on tpu, got {tpu_pipeline._device}"
    )

    rng = np.random.default_rng(7)
    camera_rgb = rng.integers(0, 255, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    joint_state = rng.standard_normal(ACTION_DIM).astype(np.float32)
    fixed_noise = rng.standard_normal((1, ACTION_HORIZON, ACTION_DIM)).astype(
        np.float32
    )
    obs_payload = {
        "session_id": "rollout-0",
        "prompt": "pick block",
        "state": joint_state,
        "images": {"observation.images.cam0": camera_rgb},
        "noise": fixed_noise,
    }

    serving = ServingRealtimeRobotOpenPI(
        engine_client=_Pi0PipelineEngine(tpu_pipeline),
        model_name="tiny-pi0",
    )
    app = _build_openpi_app(serving)

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/realtime/robot/openpi") as ws,
    ):
        handshake = _unpack(ws.receive_bytes())
        ws.send_bytes(_pack({"endpoint": "reset"}))
        reset_reply = _unpack(ws.receive_bytes())
        ws.send_bytes(_pack(dict(obs_payload)))
        tpu_actions = _unpack(ws.receive_bytes())

    assert handshake == POLICY_SERVER_CONFIG, (
        f"Handshake metadata {handshake} != expected {POLICY_SERVER_CONFIG}"
    )
    assert reset_reply == {"status": "reset successful"}, (
        f"Unexpected OpenPI reset reply: {reset_reply}"
    )
    assert isinstance(tpu_actions, np.ndarray), (
        f"Expected unpacked ndarray actions, got {type(tpu_actions)}"
    )
    assert tpu_actions.shape == (ACTION_HORIZON, ACTION_DIM), (
        f"Expected action shape {(ACTION_HORIZON, ACTION_DIM)}, got {tpu_actions.shape}"
    )
    assert tpu_actions.dtype == np.float32, (
        f"Expected float32 actions, got {tpu_actions.dtype}"
    )
    assert bool(np.isfinite(tpu_actions).all()), "TPU OpenPI actions contain NaN or Inf"

    ref_req = OmniDiffusionRequest(
        prompt="pick block",
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=NUM_STEPS,
            extra_args={
                "robot_obs": dict(obs_payload),
                "reset": True,
                "session_id": "ref",
            },
        ),
        request_id="cpu-ref",
    )
    cpu_fp32_actions = torch.from_numpy(
        cpu_fp32_pipeline.forward(DiffusionRequestBatch(requests=[ref_req])).output[
            "actions"
        ]
    )
    bf16_obs = {
        **obs_payload,
        "noise": torch.from_numpy(fixed_noise).to(torch.bfloat16),
    }
    bf16_req = OmniDiffusionRequest(
        prompt="pick block",
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=NUM_STEPS,
            extra_args={"robot_obs": bf16_obs, "reset": True, "session_id": "bf16"},
        ),
        request_id="cpu-bf16",
    )
    cpu_bf16_actions = torch.from_numpy(
        cpu_bf16_pipeline.forward(DiffusionRequestBatch(requests=[bf16_req])).output[
            "actions"
        ]
    )

    bf16_diff = (cpu_bf16_actions.double() - cpu_fp32_actions.double()).flatten()
    cpu_bf16_rel_l2 = float(
        bf16_diff.norm() / cpu_fp32_actions.double().flatten().norm()
    )
    cpu_bf16_abs = float(bf16_diff.abs().max())

    assert_parity(
        torch.from_numpy(tpu_actions),
        cpu_fp32_actions,
        name="openpi_pi0_actions",
        max_rel_l2=max(BF16_RATIO * cpu_bf16_rel_l2, REL_L2_FLOOR),
        max_abs=max(BF16_RATIO * cpu_bf16_abs, ABS_FLOOR),
    )
