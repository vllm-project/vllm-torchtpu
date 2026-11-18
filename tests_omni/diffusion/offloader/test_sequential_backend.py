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
"""TPU tests for module-level CPU offload in vllm-omni diffusion."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.offloader import (
    ModelLevelOffloadBackend,
    enable_offload_backend,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_diffusion_dir

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

MAX_REL_L2 = 2.0e-2
MAX_ABS = 5.0e-2


class _TinyOffloadPipeline(nn.Module):
    def __init__(self, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.text_encoder = nn.Linear(4, 4, bias=False, dtype=dtype, device=device)
        self.transformer = nn.Linear(4, 4, bias=False, dtype=dtype, device=device)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "OmniTpuPlatform.supports_cpu_offload() returns False so "
        "enable_offload_backend returns None"
    ),
)
@pytest.mark.parametrize(
    "offload_kwargs",
    [
        {
            "diffusion_offload_config": {
                "mode": "module",
                "components": ["dit", "text_encoder"],
            }
        },
        {"enable_cpu_offload": True},
    ],
    ids=["compact_module", "legacy_enable_cpu_offload"],
)
def test_module_level_cpu_offload(offload_kwargs: dict[str, Any]) -> None:
    """Module-Level CPU Offload.

    - TPU path: Build a tiny bf16 pipeline on TPU and call enable_offload_backend
      with compact mode="module" or legacy enable_cpu_offload=True.
    - Setup check: Assert ModelLevelOffloadBackend is returned and enabled.
    - Outcome check: Run text_encoder then transformer on TPU, verifying inactive
      component weights swap to CPU, and compare output with CPU fp32 reference.
    """
    torch.manual_seed(42)
    tpu_device = torch.device("tpu")
    ref_pipeline = _TinyOffloadPipeline(dtype=torch.float32, device=torch.device("cpu"))
    x_cpu = torch.randn(2, 4, dtype=torch.float32)

    pipeline = _TinyOffloadPipeline(dtype=torch.bfloat16, device=tpu_device)
    with torch.no_grad():
        pipeline.text_encoder.weight.copy_(
            ref_pipeline.text_encoder.weight.to(dtype=torch.bfloat16, device=tpu_device)
        )
        pipeline.transformer.weight.copy_(
            ref_pipeline.transformer.weight.to(dtype=torch.bfloat16, device=tpu_device)
        )

    od_config = OmniDiffusionConfig(model=tiny_diffusion_dir(), **offload_kwargs)
    pipeline, backend = enable_offload_backend(od_config, pipeline, device=tpu_device)
    assert isinstance(backend, ModelLevelOffloadBackend), (
        f"Expected ModelLevelOffloadBackend, got {type(backend).__name__}"
    )
    assert backend.enabled, "ModelLevelOffloadBackend was not enabled"

    x_tpu = x_cpu.to(dtype=torch.bfloat16, device=tpu_device)
    hidden = pipeline.text_encoder(x_tpu)
    assert pipeline.text_encoder.weight.device.type == "tpu", (
        "Expected active text_encoder on tpu, got "
        f"{pipeline.text_encoder.weight.device}"
    )
    assert pipeline.transformer.weight.device.type == "cpu", (
        "Expected inactive transformer on cpu, got "
        f"{pipeline.transformer.weight.device}"
    )

    tpu_out = pipeline.transformer(hidden)
    assert pipeline.transformer.weight.device.type == "tpu", (
        f"Expected active transformer on tpu, got {pipeline.transformer.weight.device}"
    )
    assert pipeline.text_encoder.weight.device.type == "cpu", (
        "Expected inactive text_encoder on cpu, got "
        f"{pipeline.text_encoder.weight.device}"
    )

    ref_out = ref_pipeline.transformer(ref_pipeline.text_encoder(x_cpu))
    assert_parity(
        tpu_out,
        ref_out,
        name="module_level_cpu_offload",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
