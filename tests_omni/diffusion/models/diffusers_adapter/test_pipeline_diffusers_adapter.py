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
"""Tests for DiffusersAdapterPipeline load and call kwargs on TPU."""

import argparse

import diffusers
import numpy as np
import pytest
import torch
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import (
    DiffusersAdapterPipeline,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S
from tests_omni.tiny_models import tiny_flux_schnell

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

REL_L2_RATIO = 4.0
REL_L2_FLOOR = 5e-2


def _rel_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    a = actual.detach().cpu().double().flatten()
    r = reference.detach().cpu().double().flatten()
    return float((a - r).norm() / r.norm())


def test_diffusers_adapter_loading_kwargs() -> None:
    """Diffusers Adapter Loading Kwargs

    - TPU path: parse --diffusers-load-kwargs and --diffusers-call-kwargs JSON
      via serve CLI parser, run inline single-device Omni on tiny_flux_schnell()
      with diffusion_load_format="diffusers", and generate a 256x256 image on TPU.
    - Setup check: assert isinstance(pipeline, DiffusersAdapterPipeline),
      od_config stores parsed diffusers_load_kwargs and diffusers_call_kwargs,
      and pipeline._pipeline.transformer parameters reside on "tpu".
    - Outcome check: assert _build_call_kwargs applies diffusers_call_kwargs
      defaults (and sampling_params overrides them) and TPU image output matches
      CPU fp32 reference within max(RATIO * cpu_bf16_err, FLOOR).
    """
    model_dir = tiny_flux_schnell()
    load_kwargs = {"local_files_only": True, "use_safetensors": True}
    call_kwargs = {
        "num_inference_steps": 1,
        "guidance_scale": 0.0,
        "max_sequence_length": 64,
    }

    call_kwargs_json = (
        '{"num_inference_steps": 1, "guidance_scale": 0.0, "max_sequence_length": 64}'
    )

    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    OmniServeCommand().subparser_init(subparsers)
    parsed = parser.parse_args(
        [
            "serve",
            model_dir,
            "--omni",
            "--diffusion-load-format",
            "diffusers",
            "--diffusers-load-kwargs",
            '{"local_files_only": true, "use_safetensors": true}',
            "--diffusers-call-kwargs",
            call_kwargs_json,
        ]
    )
    assert isinstance(parsed, argparse.Namespace), (
        f"Expected argparse.Namespace from CLI parse, got {type(parsed).__name__}"
    )
    assert parsed.diffusers_load_kwargs == load_kwargs, (
        f"Parsed diffusers_load_kwargs {parsed.diffusers_load_kwargs} != {load_kwargs}"
    )
    assert parsed.diffusers_call_kwargs == call_kwargs, (
        f"Parsed diffusers_call_kwargs {parsed.diffusers_call_kwargs} != {call_kwargs}"
    )

    prompt = "a red cube on a white table"
    omni = Omni(
        model=model_dir,
        diffusion_load_format="diffusers",
        diffusers_load_kwargs=parsed.diffusers_load_kwargs,
        diffusers_call_kwargs=parsed.diffusers_call_kwargs,
    )
    try:
        stage_client = omni.engine.stage_pools[0].stage_client
        pipeline = (
            stage_client._engine.executor.driver_worker.worker.model_runner.pipeline
        )
        assert isinstance(pipeline, DiffusersAdapterPipeline), (
            f"Expected DiffusersAdapterPipeline, got {type(pipeline)}"
        )
        actual_load = pipeline.od_config.diffusers_load_kwargs
        assert actual_load == load_kwargs, (
            f"od_config.diffusers_load_kwargs {actual_load} != {load_kwargs}"
        )
        actual_call = pipeline.od_config.diffusers_call_kwargs
        assert actual_call == call_kwargs, (
            f"od_config.diffusers_call_kwargs {actual_call} != {call_kwargs}"
        )
        transformer_device = next(pipeline._pipeline.transformer.parameters()).device
        assert transformer_device.type == "tpu", (
            f"Expected transformer on 'tpu', got {transformer_device.type!r}"
        )

        default_req = DiffusionRequestBatch(
            requests=[
                OmniDiffusionRequest(
                    prompt={"prompt": prompt},
                    sampling_params=OmniDiffusionSamplingParams(height=256, width=256),
                    request_id="req-default",
                )
            ]
        )
        default_built = pipeline._build_call_kwargs(default_req)
        assert default_built["num_inference_steps"] == 1, (
            f"Expected default num_inference_steps=1, got "
            f"{default_built.get('num_inference_steps')}"
        )

        override_req = DiffusionRequestBatch(
            requests=[
                OmniDiffusionRequest(
                    prompt={"prompt": prompt},
                    sampling_params=OmniDiffusionSamplingParams(
                        height=256, width=256, num_inference_steps=2
                    ),
                    request_id="req-override",
                )
            ]
        )
        override_built = pipeline._build_call_kwargs(override_req)
        assert override_built["num_inference_steps"] == 2, (
            f"Expected overridden num_inference_steps=2, got "
            f"{override_built.get('num_inference_steps')}"
        )

        outputs = omni.generate(
            prompt,
            OmniDiffusionSamplingParams(
                height=256,
                width=256,
                seed=42,
                generator_device="cpu",
            ),
        )
        tpu_img = torch.from_numpy(np.asarray(outputs[0].images[0], dtype=np.float32))
    finally:
        omni.close()

    fp32_pipe = diffusers.DiffusionPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        **load_kwargs,
    )
    fp32_out = fp32_pipe(
        prompt=prompt,
        height=256,
        width=256,
        generator=torch.Generator(device="cpu").manual_seed(42),
        output_type="np",
        **call_kwargs,
    ).images[0]
    cpu_fp32_img = torch.from_numpy((fp32_out * 255.0).clip(0, 255).astype(np.float32))

    bf16_pipe = diffusers.DiffusionPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        **load_kwargs,
    )
    bf16_out = bf16_pipe(
        prompt=prompt,
        height=256,
        width=256,
        generator=torch.Generator(device="cpu").manual_seed(42),
        output_type="np",
        **call_kwargs,
    ).images[0]
    cpu_bf16_img = torch.from_numpy((bf16_out * 255.0).clip(0, 255).astype(np.float32))

    cpu_bf16_err = _rel_l2(cpu_bf16_img, cpu_fp32_img)
    tpu_err = _rel_l2(tpu_img, cpu_fp32_img)
    limit = max(REL_L2_RATIO * cpu_bf16_err, REL_L2_FLOOR)
    assert tpu_err <= limit, (
        f"DiffusersAdapterPipeline TPU image rel_l2 {tpu_err:.4e} "
        f"exceeds limit {limit:.4e}"
    )
