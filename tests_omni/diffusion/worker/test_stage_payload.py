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
"""TPU tests for DiffusionStagePayloadMixin stage payload transport."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
    SharedMemoryConnector,
)
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVTransferManager,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.omni_test_utils import PR_TIMEOUT_S

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]


def _make_stage_kv_manager(
    connector: SharedMemoryConnector,
    *,
    send_stages: tuple[str, str],
    recv_stages: tuple[str, str],
) -> OmniKVTransferManager:
    """Construct a minimal OmniKVTransferManager around a SharedMemoryConnector."""
    mgr = object.__new__(OmniKVTransferManager)
    mgr._connector = connector
    mgr.send_stages = send_stages
    mgr.recv_stages = recv_stages
    mgr.update_sender_info = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    return mgr


def _make_stage_runner(
    od_config: SimpleNamespace,
    kv_manager: OmniKVTransferManager,
) -> DiffusionModelRunner:
    """Construct a single-rank DiffusionModelRunner on TPU for payload transport."""
    tpu_device = torch.device("tpu")
    runner = object.__new__(DiffusionModelRunner)
    runner.od_config = od_config  # type: ignore[assignment]
    runner.device = tpu_device
    runner._local_rank = 0
    runner.pipeline = SimpleNamespace(device=tpu_device)
    runner.kv_transfer_manager = kv_manager
    runner.init_omni_connectors(od_config, kv_manager, synchronous=True)  # type: ignore[arg-type]
    return runner


def test_hidden_state_and_conditioning_stage_payload_transport() -> None:
    """Hidden-State and Conditioning Stage Payload Transport.

    - TPU path: Build single-rank sender and receiver ``DiffusionModelRunner``
      instances on ``torch.device("tpu")`` backed by ``SharedMemoryConnector``
      and transfer bf16 conditioning/latent tensors via
      ``_maybe_send_stage_payload`` and ``_maybe_recv_stage_payload``.
    - Setup check: Assert input tensors have ``device.type == "tpu"``,
      payload keys are configured, and ``_maybe_send_stage_payload`` replaces
      declared keys with ``_stage_payload_transfer`` while keeping undeclared
      keys.
    - Outcome check: Assert ``_maybe_recv_stage_payload`` strips the handle,
      populates ``additional_information`` on ``device.type == "tpu"``, and
      matches CPU references bitwise.
    """
    tpu_device = torch.device("tpu")
    cpu_embeds = torch.tensor([[1.0, -2.5, 0.25, 4.0]], dtype=torch.bfloat16)
    cpu_latents = torch.tensor([[[0.5, -1.5], [2.0, -0.5]]], dtype=torch.bfloat16)

    tpu_embeds = cpu_embeds.to(tpu_device)
    tpu_latents = cpu_latents.to(tpu_device)
    assert tpu_embeds.device.type == "tpu", (
        f"Expected tpu_embeds on tpu, got {tpu_embeds.device}"
    )
    assert tpu_latents.device.type == "tpu", (
        f"Expected tpu_latents on tpu, got {tpu_latents.device}"
    )

    connector = SharedMemoryConnector({})
    try:
        kv_mgr = _make_stage_kv_manager(
            connector, send_stages=("0", "1"), recv_stages=("0", "1")
        )
        sender_cfg = SimpleNamespace(
            stage_id=0,
            stage_output_payload_keys=("prompt_embeds", "latents"),
        )
        receiver_cfg = SimpleNamespace(
            stage_id=1,
            stage_input_payload_keys=("prompt_embeds", "latents"),
            stage_input_optional_payload_keys=(),
        )
        assert sender_cfg.stage_output_payload_keys == ("prompt_embeds", "latents"), (
            "Expected sender_cfg.stage_output_payload_keys == "
            "('prompt_embeds', 'latents'), "
            f"got {sender_cfg.stage_output_payload_keys!r}"
        )
        assert receiver_cfg.stage_input_payload_keys == ("prompt_embeds", "latents"), (
            "Expected receiver_cfg.stage_input_payload_keys == "
            "('prompt_embeds', 'latents'), "
            f"got {receiver_cfg.stage_input_payload_keys!r}"
        )

        sender = _make_stage_runner(sender_cfg, kv_mgr)
        receiver = _make_stage_runner(receiver_cfg, kv_mgr)

        send_req = OmniDiffusionRequest(
            prompt={"prompt": "tpu test"},
            sampling_params=OmniDiffusionSamplingParams(),
            request_id="req-stage-0",
        )
        send_out = DiffusionOutput(
            custom_output={
                "prompt_embeds": tpu_embeds,
                "latents": tpu_latents,
                "undeclared_meta": {"step": 1},
            }
        )

        sender._maybe_send_stage_payload([send_req], [send_out])
        handle_key = DiffusionModelRunner._STAGE_PAYLOAD_HANDLE_KEY
        assert handle_key in send_out.custom_output, (
            "Expected _stage_payload_transfer handle in custom_output"
        )
        assert "prompt_embeds" not in send_out.custom_output, (
            "Declared key prompt_embeds was not popped from custom_output"
        )
        assert "latents" not in send_out.custom_output, (
            "Declared key latents was not popped from custom_output"
        )
        assert send_out.custom_output["undeclared_meta"] == {"step": 1}, (
            "Undeclared custom_output key was not preserved"
        )

        recv_prompt = {
            "prompt": "tpu test",
            handle_key: send_out.custom_output[handle_key],
            "additional_information": {"cond_meta": "keep"},
        }
        recv_req = OmniDiffusionRequest(
            prompt=recv_prompt,
            sampling_params=OmniDiffusionSamplingParams(),
            request_id="req-stage-0",
        )
        receiver._maybe_recv_stage_payload(recv_req)

        assert handle_key not in recv_req.prompt, (
            "Transport handle was not stripped from recv_req.prompt"
        )
        additional = recv_req.prompt["additional_information"]
        assert additional["cond_meta"] == "keep", (
            f"Conditioning metadata lost: {additional}"
        )
        recv_embeds = additional["prompt_embeds"]
        recv_latents = additional["latents"]
        assert recv_embeds.device.type == "tpu", (
            f"Received prompt_embeds expected on tpu, got {recv_embeds.device}"
        )
        assert recv_latents.device.type == "tpu", (
            f"Received latents expected on tpu, got {recv_latents.device}"
        )
        assert torch.equal(recv_embeds.cpu(), cpu_embeds), (
            "Received prompt_embeds does not match CPU reference"
        )
        assert torch.equal(recv_latents.cpu(), cpu_latents), (
            "Received latents does not match CPU reference"
        )
    finally:
        connector.close()
