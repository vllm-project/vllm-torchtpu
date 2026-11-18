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
"""TPU component tests for Cosmos3 safety guardrails bypass."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from diffusers.video_processor import VideoProcessor
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.cosmos3 import (
    guardrails,
    pipeline_cosmos3,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_diffusion_dir

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

POSTPROCESS_MAX_REL_L2 = 1e-2
POSTPROCESS_MAX_ABS = 1e-2


def test_cosmos3_safety_guardrails_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cosmos3 Safety Guardrails Bypass.

    - TPU path: parse ``--no-guardrails`` via ``OmniServeCommand`` into
      ``OmniDiffusionConfig``, then run ``check_video_safety``,
      ``get_cosmos3_pre_process_func``, and ``get_cosmos3_post_process_func``
      on a seeded TPU bfloat16 video tensor ``[1, 3, 2, 16, 16]``.
    - Setup check: ``current_omni_platform.device_type == "tpu"``,
      ``is_guardrails_enabled(od_config)`` is False (vs True by default), and
      ``ensure_initialized`` leaves ``_text_guardrail`` and ``_video_guardrail``
      as None.
    - Outcome check: ``check_video_safety`` returns the TPU tensor unchanged
      without D2H copy when uninitialized; pre/post hooks with
      ``--no-guardrails`` skip installed sentinel guardrails and match CPU
      ``VideoProcessor.postprocess_video`` parity.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU platform, got {current_omni_platform.device_type}"
    )

    model_dir = tiny_diffusion_dir()
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")
    OmniServeCommand().subparser_init(subparsers)
    args = parser.parse_args(["serve", model_dir, "--omni", "--no-guardrails"])
    args.headless = True
    monkeypatch.setenv("VLLM_DISABLE_LOG_LOGO", "1")
    monkeypatch.setattr(
        "vllm_omni.entrypoints.cli.serve.run_headless", lambda _args: None
    )
    OmniServeCommand.cmd(args)

    assert args.model_config["guardrails"] is False, (
        f"Expected guardrails=False from --no-guardrails, got {args.model_config}"
    )

    od_config_bypass = OmniDiffusionConfig(
        model=model_dir,
        model_config=args.model_config,
    )
    od_config_default = OmniDiffusionConfig(model=model_dir)

    assert guardrails.is_guardrails_enabled(od_config_bypass) is False, (
        "Expected is_guardrails_enabled to be False with --no-guardrails"
    )
    assert guardrails.is_guardrails_enabled(od_config_default) is True, (
        "Expected is_guardrails_enabled to be True by default"
    )

    monkeypatch.setattr(guardrails, "_text_guardrail", None)
    monkeypatch.setattr(guardrails, "_video_guardrail", None)
    guardrails.ensure_initialized(od_config_bypass)
    assert guardrails._text_guardrail is None, (
        "ensure_initialized populated _text_guardrail despite --no-guardrails"
    )
    assert guardrails._video_guardrail is None, (
        "ensure_initialized populated _video_guardrail despite --no-guardrails"
    )

    torch.manual_seed(0)
    video_cpu = torch.rand(1, 3, 2, 16, 16, dtype=torch.bfloat16) * 2.0 - 1.0
    video_tpu = video_cpu.to("tpu")

    uninit_out = guardrails.check_video_safety(video_tpu)
    assert uninit_out is video_tpu, (
        "Expected uninitialized check_video_safety to return tensor unchanged"
    )
    assert uninit_out.device.type == "tpu", (
        f"Expected tensor to stay on tpu, got {uninit_out.device.type}"
    )

    hook_calls: list[str] = []

    def _sentinel_text(_prompt: str) -> None:
        hook_calls.append("text")

    def _sentinel_video(frames: np.ndarray) -> np.ndarray:
        hook_calls.append("video")
        return frames

    monkeypatch.setattr(guardrails, "_text_guardrail", _sentinel_text)
    monkeypatch.setattr(guardrails, "_video_guardrail", _sentinel_video)

    pre_bypass = pipeline_cosmos3.get_cosmos3_pre_process_func(od_config_bypass)
    post_bypass = pipeline_cosmos3.get_cosmos3_post_process_func(od_config_bypass)
    pre_default = pipeline_cosmos3.get_cosmos3_pre_process_func(od_config_default)
    post_default = pipeline_cosmos3.get_cosmos3_post_process_func(od_config_default)

    sampling_params = OmniDiffusionSamplingParams(num_inference_steps=1)
    req = OmniDiffusionRequest(
        prompt="a quiet mountain lake",
        sampling_params=sampling_params,
        request_id="req-cosmos3-guardrails",
    )
    pre_bypass(req)
    out_bypass = post_bypass(
        video_tpu, output_type="np", sampling_params=sampling_params
    )
    assert hook_calls == [], (
        f"Expected --no-guardrails to bypass sentinel hooks, got {hook_calls}"
    )

    pre_default(req)
    post_default(video_tpu, output_type="np", sampling_params=sampling_params)
    assert hook_calls == ["text", "video"], (
        f"Expected default config to invoke sentinel hooks, got {hook_calls}"
    )

    ref_video = VideoProcessor(vae_scale_factor=16).postprocess_video(
        video_cpu.float(), output_type="np"
    )
    assert_parity(
        torch.from_numpy(out_bypass.copy()),
        torch.from_numpy(ref_video.copy()),
        name="cosmos3_postprocess_bypass_vs_cpu",
        max_rel_l2=POSTPROCESS_MAX_REL_L2,
        max_abs=POSTPROCESS_MAX_ABS,
    )
