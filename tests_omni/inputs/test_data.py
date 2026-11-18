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
"""TPU tests for vllm_omni.inputs.data."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import (
    OmniDiffusionSamplingParams,
    OmniEmbedsPrompt,
    OmniTextPrompt,
    OmniTokensPrompt,
    token_inputs_omni,
)

from tests_omni.conftest import OmniRunner
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S
from tests_omni.tiny_models import tiny_wan2_1_t2v

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

_HEIGHT = 256
_WIDTH = 256
_NUM_FRAMES = 5
_TEXT_HIDDEN_SIZE = 256
_PROMPT_SEQ_LEN = 8


@pytest.fixture(scope="module")
def wan_runner() -> Generator[OmniRunner, None, None]:
    """Share a single inline tiny Wan2.1 runner across tests in this module."""
    with OmniRunner(tiny_wan2_1_t2v()) as runner:
        yield runner


def _extract_batched_wan_video(
    outputs: list[Any],
    expected_shape: tuple[int, int, int, int, int],
    case_name: str,
) -> np.ndarray:
    assert len(outputs) == 1, f"{case_name}: expected 1 output, got {len(outputs)}"
    output = outputs[0]
    assert output.final_output_type == "video", (
        f"{case_name}: expected final_output_type 'video', "
        f"got {output.final_output_type!r}"
    )
    assert len(output.images) == 1, (
        f"{case_name}: expected 1 video array, got {len(output.images)}"
    )
    video: np.ndarray = output.images[0]
    assert video.shape == expected_shape, (
        f"{case_name}: expected video shape {expected_shape}, got {video.shape}"
    )
    assert np.all(np.isfinite(video)), f"{case_name}: video contains NaN or Inf"
    return video


def _make_prompt_embeds(seed: int) -> torch.Tensor:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return torch.randn(_PROMPT_SEQ_LEN, _TEXT_HIDDEN_SIZE, dtype=torch.bfloat16)


def _prompt_case_spec(
    case_id: str,
) -> tuple[Any, Any, Any, str, Any, float]:
    embeds_a = _make_prompt_embeds(seed=11)
    embeds_b = _make_prompt_embeds(seed=29)
    text_a = "A serene mountain lake at sunrise"
    neg_a = "blurry, low quality"
    text_b = "A busy neon city street at night"
    neg_b = "overexposed, static"
    specs: dict[str, tuple[Any, Any, Any, str, Any, float]] = {
        "omni_text_prompt": (
            OmniTextPrompt(
                prompt=text_a,
                negative_prompt=neg_a,
                modalities=["video"],
            ),
            {"prompt": text_a, "negative_prompt": neg_a},
            OmniTextPrompt(
                prompt=text_b,
                negative_prompt=neg_b,
                modalities=["video"],
            ),
            "prompt",
            text_a,
            3.0,
        ),
        "omni_embeds_prompt": (
            OmniEmbedsPrompt(prompt_embeds=embeds_a),
            {"prompt_embeds": embeds_a},
            OmniEmbedsPrompt(prompt_embeds=embeds_b),
            "prompt_embeds",
            embeds_a,
            1.0,
        ),
        "omni_tokens_prompt": (
            OmniTokensPrompt(prompt_token_ids=[1, 2, 3], prompt_embeds=embeds_a),
            OmniEmbedsPrompt(prompt_embeds=embeds_a),
            OmniTokensPrompt(prompt_token_ids=[1, 2, 3], prompt_embeds=embeds_b),
            "prompt_embeds",
            embeds_a,
            1.0,
        ),
        "omni_token_inputs": (
            token_inputs_omni(
                prompt_token_ids=[1, 2, 3],
                additional_information={"prompt_embeds": embeds_a},
            ),
            OmniEmbedsPrompt(prompt_embeds=embeds_a),
            token_inputs_omni(
                prompt_token_ids=[1, 2, 3],
                additional_information={"prompt_embeds": embeds_b},
            ),
            "prompt_embeds",
            embeds_a,
            1.0,
        ),
    }
    return specs[case_id]


def test_diffusion_sampling_params(wan_runner: OmniRunner) -> None:
    """Diffusion Sampling Parameters (OmniDiffusionSamplingParams).

    - Run an inline single-device OmniRunner on tiny_wan2_1_t2v() on TPU with
      explicit OmniDiffusionSamplingParams and vllm.SamplingParams coerced via
      OmniDiffusionSamplingParams.from_params().
    - Setup check: from_params() maps seed, height, width, num_frames,
      num_inference_steps=2, guidance_scale=3.0, and num_outputs_per_prompt=2
      (batch_size == 2) while preserving extra_args={"flow_shift": 3.0}.
    - Outcome check: coerced TPU video has shape (2, 5, 256, 256, 3), finite
      pixels, distinct batch samples, bitwise equality with the direct run, and
      diverges from a 1-step baseline with different seed/guidance/flow_shift.
    """
    prompt = "A serene mountain lake at sunrise"
    expected_shape = (2, _NUM_FRAMES, _WIDTH, _HEIGHT, 3)

    vllm_params = SamplingParams(
        seed=42,
        extra_args={
            "height": _HEIGHT,
            "width": _WIDTH,
            "num_frames": _NUM_FRAMES,
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "num_outputs_per_prompt": 2,
            "generator_device": "cpu",
            "flow_shift": 3.0,
        },
    )
    coerced = OmniDiffusionSamplingParams.from_params(vllm_params)

    assert coerced.seed == 42, f"Expected seed 42, got {coerced.seed}"
    assert coerced.height == _HEIGHT, f"Expected height {_HEIGHT}, got {coerced.height}"
    assert coerced.width == _WIDTH, f"Expected width {_WIDTH}, got {coerced.width}"
    assert coerced.num_frames == _NUM_FRAMES, (
        f"Expected num_frames {_NUM_FRAMES}, got {coerced.num_frames}"
    )
    assert coerced.num_inference_steps == 2, (
        f"Expected num_inference_steps 2, got {coerced.num_inference_steps}"
    )
    assert coerced.guidance_scale == 3.0, (
        f"Expected guidance_scale 3.0, got {coerced.guidance_scale}"
    )
    assert coerced.num_outputs_per_prompt == 2, (
        f"Expected num_outputs_per_prompt 2, got {coerced.num_outputs_per_prompt}"
    )
    assert coerced.batch_size == 2, f"Expected batch_size 2, got {coerced.batch_size}"
    assert coerced.extra_args == {"flow_shift": 3.0}, (
        f"Expected extra_args {{'flow_shift': 3.0}}, got {coerced.extra_args}"
    )

    direct = OmniDiffusionSamplingParams(
        seed=42,
        height=_HEIGHT,
        width=_WIDTH,
        num_frames=_NUM_FRAMES,
        num_inference_steps=2,
        guidance_scale=3.0,
        num_outputs_per_prompt=2,
        generator_device="cpu",
        extra_args={"flow_shift": 3.0},
    )
    control = OmniDiffusionSamplingParams(
        seed=99,
        height=_HEIGHT,
        width=_WIDTH,
        num_frames=_NUM_FRAMES,
        num_inference_steps=1,
        guidance_scale=1.0,
        num_outputs_per_prompt=2,
        generator_device="cpu",
        extra_args={"flow_shift": 1.0},
    )

    coerced_video = _extract_batched_wan_video(
        list(wan_runner.generate(prompt, coerced)),
        expected_shape,
        "coerced_from_params",
    )
    direct_video = _extract_batched_wan_video(
        list(wan_runner.generate(prompt, direct)),
        expected_shape,
        "direct_sampling_params",
    )
    control_video = _extract_batched_wan_video(
        list(wan_runner.generate(prompt, control)),
        expected_shape,
        "control_1step",
    )

    assert not np.array_equal(coerced_video[0], coerced_video[1]), (
        "num_outputs_per_prompt=2 produced identical batch samples"
    )
    np.testing.assert_array_equal(
        coerced_video,
        direct_video,
        err_msg="Coerced SamplingParams video diverged from direct params",
    )
    assert not np.array_equal(coerced_video, control_video), (
        "Coerced 2-step video unexpectedly matched 1-step control baseline"
    )


def _assert_matches_reference_not_control(
    out_video: np.ndarray,
    ref_video: np.ndarray,
    ctrl_video: np.ndarray,
    case_id: str,
) -> None:
    np.testing.assert_array_equal(
        out_video,
        ref_video,
        err_msg=f"{case_id}: output diverged from equivalent reference carrier",
    )
    assert not np.array_equal(out_video, ctrl_video), (
        f"{case_id}: output unexpectedly matched distinct control carrier"
    )


@pytest.mark.parametrize(
    "case_id",
    [
        "omni_embeds_prompt",
        "omni_tokens_prompt",
        "omni_token_inputs",
    ],
)
def test_multimodal_input_prompt_types(
    wan_runner: OmniRunner,
    case_id: str,
) -> None:
    """Multi-Modal Input Prompt Types: prompt-embedding carriers.

    - TPU path: Run OmniEmbedsPrompt, OmniTokensPrompt, and token_inputs_omni on
      the shared tiny_wan2_1_t2v() OmniRunner on TPU.
    - Setup check: DiffusionRequestBatch.get_prompt_field resolves the expected
      prompt embedding tensor on each carrier before engine submission.
    - Outcome check: output has shape (1, 5, 256, 256, 3) with finite pixels,
      matches its equivalent reference carrier bitwise, and diverges from a
      distinct control embedding.
    """
    carrier, ref_carrier, ctrl_carrier, field_name, expected_val, guidance = (
        _prompt_case_spec(case_id)
    )
    expected_shape = (1, _NUM_FRAMES, _WIDTH, _HEIGHT, 3)

    resolved = DiffusionRequestBatch.get_prompt_field(carrier, field_name)
    assert isinstance(resolved, torch.Tensor), (
        f"{case_id}: get_prompt_field({field_name!r}) expected torch.Tensor, "
        f"got {type(resolved).__name__}"
    )
    assert torch.equal(resolved, expected_val), (
        f"{case_id}: get_prompt_field({field_name!r}) did not match expected tensor"
    )

    params = OmniDiffusionSamplingParams(
        seed=42,
        height=_HEIGHT,
        width=_WIDTH,
        num_frames=_NUM_FRAMES,
        num_inference_steps=1,
        guidance_scale=guidance,
        generator_device="cpu",
    )

    out_video = _extract_batched_wan_video(
        list(wan_runner.generate(carrier, params)),
        expected_shape,
        case_id,
    )
    ref_video = _extract_batched_wan_video(
        list(wan_runner.generate(ref_carrier, params)),
        expected_shape,
        f"{case_id}_reference",
    )
    ctrl_video = _extract_batched_wan_video(
        list(wan_runner.generate(ctrl_carrier, params)),
        expected_shape,
        f"{case_id}_control",
    )

    _assert_matches_reference_not_control(out_video, ref_video, ctrl_video, case_id)


def test_multimodal_input_prompt_types_text(wan_runner: OmniRunner) -> None:
    """Multi-Modal Input Prompt Types: OmniTextPrompt.

    - TPU path: Run OmniTextPrompt on the shared tiny_wan2_1_t2v() OmniRunner on
      TPU.
    - Setup check: DiffusionRequestBatch.get_prompt_field resolves the expected
      prompt text before engine submission.
    - Outcome check: output has shape (1, 5, 256, 256, 3) with finite pixels,
      matches its equivalent reference carrier bitwise, and diverges from a
      distinct control prompt.
    """
    case_id = "omni_text_prompt"
    carrier, ref_carrier, ctrl_carrier, field_name, expected_val, guidance = (
        _prompt_case_spec(case_id)
    )
    expected_shape = (1, _NUM_FRAMES, _WIDTH, _HEIGHT, 3)

    resolved = DiffusionRequestBatch.get_prompt_field(carrier, field_name)
    assert resolved == expected_val, (
        f"{case_id}: get_prompt_field({field_name!r}) returned {resolved!r}, "
        f"expected {expected_val!r}"
    )

    params = OmniDiffusionSamplingParams(
        seed=42,
        height=_HEIGHT,
        width=_WIDTH,
        num_frames=_NUM_FRAMES,
        num_inference_steps=1,
        guidance_scale=guidance,
        generator_device="cpu",
    )

    out_video = _extract_batched_wan_video(
        list(wan_runner.generate(carrier, params)),
        expected_shape,
        case_id,
    )
    ref_video = _extract_batched_wan_video(
        list(wan_runner.generate(ref_carrier, params)),
        expected_shape,
        f"{case_id}_reference",
    )
    ctrl_video = _extract_batched_wan_video(
        list(wan_runner.generate(ctrl_carrier, params)),
        expected_shape,
        f"{case_id}_control",
    )

    _assert_matches_reference_not_control(out_video, ref_video, ctrl_video, case_id)
