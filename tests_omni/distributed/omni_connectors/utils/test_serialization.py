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
"""TPU tests for OmniSerializer payload roundtrip serialization."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from vllm.outputs import RequestOutput
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.mm_outputs import MultimodalCompletionOutput

from tests_omni.omni_test_utils import PR_TIMEOUT_S

pytestmark = [
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]


def test_payload_serialization() -> None:
    """Payload Serialization (OmniSerializer).

    - TPU path: Allocate bf16 2D, non-contiguous strided float32 view, int64
      1D, and 0-D scalar tensors on ``torch.device("tpu")`` and embed them with
      ``np.ndarray``, ``PIL.Image``, ``RequestOutput``/``CompletionOutput``
      ``multimodal_output``, and ``OmniRequestOutput`` in
      ``OmniSerializer.serialize`` / ``deserialize``.
    - Setup check: Assert input tensors reside on ``device.type == "tpu"`` and
      the strided view is non-contiguous before serialization.
    - Outcome check: Assert non-empty bytes output, CPU placement, shapes,
      dtypes, preserved metadata, and bitwise equality against CPU references.
    """
    tpu_device = torch.device("tpu")

    cpu_bf16 = torch.tensor(
        [[1.25, -2.5, 0.5], [3.0, -0.125, 4.5]], dtype=torch.bfloat16
    )
    cpu_base_f32 = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    cpu_strided_f32 = cpu_base_f32[:, ::2]
    cpu_i64 = torch.tensor([10, -20, 30, 42], dtype=torch.int64)
    cpu_scalar = torch.tensor(7.5, dtype=torch.float32)

    tpu_bf16 = cpu_bf16.to(tpu_device)
    tpu_strided_f32 = cpu_base_f32.to(tpu_device)[:, ::2]
    tpu_i64 = cpu_i64.to(tpu_device)
    tpu_scalar = cpu_scalar.to(tpu_device)

    for name, tensor in (
        ("bf16_2d", tpu_bf16),
        ("strided_f32", tpu_strided_f32),
        ("int64_1d", tpu_i64),
        ("scalar_0d", tpu_scalar),
    ):
        assert tensor.device.type == "tpu", (
            f"{name} expected on tpu, got {tensor.device}"
        )
    assert not tpu_strided_f32.is_contiguous(), (
        "Strided float32 view unexpectedly contiguous before serialization"
    )

    ref_arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    ref_img_arr = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [128, 64, 32]]],
        dtype=np.uint8,
    )
    ref_img = Image.fromarray(ref_img_arr, mode="RGB")

    completion = MultimodalCompletionOutput(
        multimodal_output={"latent": tpu_bf16},
        index=0,
        text="hello tpu",
        token_ids=[1, 2, 3],
        cumulative_logprob=-0.5,
        logprobs=None,
    )

    req_output = RequestOutput(
        request_id="req-serde-0",
        prompt="prompt",
        prompt_token_ids=[10, 11],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
    )
    req_output.multimodal_output = {"audio_codes": tpu_i64}

    omni_output = OmniRequestOutput.from_stage_output(
        req_output,
        stage_id=0,
        final_output_type="image",
        _custom_output={"strided": tpu_strided_f32, "scalar": tpu_scalar},
    )

    payload = {
        "bf16": tpu_bf16,
        "strided": tpu_strided_f32,
        "int64": tpu_i64,
        "scalar": tpu_scalar,
        "ndarray": ref_arr,
        "image": ref_img,
        "request_output": req_output,
        "omni_output": omni_output,
    }

    encoded = OmniSerializer.serialize(payload)
    assert isinstance(encoded, bytes), (
        f"Serialized output expected bytes, got {type(encoded).__name__}"
    )
    assert len(encoded) > 0, (
        f"Serialized output expected non-empty bytes, got length {len(encoded)}"
    )

    decoded = OmniSerializer.deserialize(encoded)
    assert isinstance(decoded, dict), (
        f"Deserialized payload expected dict, got {type(decoded).__name__}"
    )

    tensor_checks = (
        ("bf16", decoded["bf16"], cpu_bf16),
        ("strided", decoded["strided"], cpu_strided_f32),
        ("int64", decoded["int64"], cpu_i64),
        ("scalar", decoded["scalar"], cpu_scalar),
    )
    for name, actual, ref in tensor_checks:
        assert isinstance(actual, torch.Tensor), (
            f"{name} expected torch.Tensor, got {type(actual).__name__}"
        )
        assert actual.device.type == "cpu", (
            f"{name} expected on cpu after deserialize, got {actual.device}"
        )
        assert actual.dtype == ref.dtype, (
            f"{name} dtype mismatch: {actual.dtype} != {ref.dtype}"
        )
        assert actual.shape == ref.shape, (
            f"{name} shape mismatch: {actual.shape} != {ref.shape}"
        )
        assert torch.equal(actual, ref), (
            f"{name} tensor values differ from CPU reference"
        )

    assert isinstance(decoded["ndarray"], np.ndarray), (
        f"ndarray expected np.ndarray, got {type(decoded['ndarray']).__name__}"
    )
    assert np.array_equal(decoded["ndarray"], ref_arr), (
        "Deserialized ndarray differs from reference"
    )
    assert isinstance(decoded["image"], Image.Image), (
        f"image expected PIL.Image.Image, got {type(decoded['image']).__name__}"
    )
    assert np.array_equal(np.asarray(decoded["image"]), ref_img_arr), (
        "Deserialized PIL.Image pixels differ from reference"
    )

    out_request = decoded["request_output"]
    assert isinstance(out_request, RequestOutput), (
        f"Expected RequestOutput, got {type(out_request).__name__}"
    )
    assert torch.equal(out_request.outputs[0].multimodal_output["latent"], cpu_bf16), (
        "CompletionOutput.multimodal_output['latent'] mismatch"
    )

    out_omni = decoded["omni_output"]
    assert isinstance(out_omni, OmniRequestOutput), (
        f"Expected OmniRequestOutput, got {type(out_omni).__name__}"
    )
    assert out_omni.stage_id == 0, f"stage_id mismatch: {out_omni.stage_id}"
    assert out_omni.final_output_type == "image", (
        f"final_output_type mismatch: {out_omni.final_output_type}"
    )
    assert out_omni.request_id == "req-serde-0", (
        f"request_id mismatch: {out_omni.request_id}"
    )
    assert torch.equal(out_omni._multimodal_output["audio_codes"], cpu_i64), (
        "OmniRequestOutput._multimodal_output['audio_codes'] mismatch"
    )
    assert torch.equal(out_omni.custom_output["strided"], cpu_strided_f32), (
        "OmniRequestOutput.custom_output['strided'] mismatch"
    )
    assert torch.equal(out_omni.custom_output["scalar"], cpu_scalar), (
        "OmniRequestOutput.custom_output['scalar'] mismatch"
    )
