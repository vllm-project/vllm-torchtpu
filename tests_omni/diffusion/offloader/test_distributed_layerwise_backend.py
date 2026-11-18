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
"""TPU tests for distributed layerwise offload in vllm-omni diffusion."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.host_weight_plan import build_checkpoint_mmap_plan
from vllm_omni.diffusion.offloader import (
    DistributedLayerwiseOffloadBackend,
    OffloadStartupState,
    enable_offload_backend,
)
from vllm_omni.diffusion.offloader.startup import attach_offload_startup_state

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

MAX_REL_L2 = 2.0e-2
MAX_ABS = 5.0e-2


class _TinyDLODiT(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Linear(4, 4, bias=False, dtype=dtype, device=device) for _ in range(3)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class _TinyDLOPipeline(nn.Module):
    def __init__(self, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.transformer = _TinyDLODiT(dtype=dtype, device=device)


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "OmniTpuPlatform.supports_cpu_offload() is False so "
        "enable_offload_backend raises RuntimeError when a DLO plan is active"
    ),
)
@pytest.mark.parametrize(
    "dit_layer_options",
    [
        {"weight_transfer": "rank-local", "resident_layers": 1},
        {"weight_transfer": "allgather", "resident_layers": 0},
    ],
    ids=["rank_local", "allgather"],
)
def test_distributed_layerwise_offload(
    tmp_path: Path,
    dit_layer_options: dict[str, Any],
) -> None:
    """Distributed Layerwise Offload.

    - TPU path: Save a tiny 3-block bf16 DiT to safetensors, build a
      HostWeightPlan via build_checkpoint_mmap_plan, attach OffloadStartupState,
      and call enable_offload_backend on TPU.
    - Setup check: Assert plan_result.plan is not None before
      enable_offload_backend, then assert DistributedLayerwiseOffloadBackend is
      enabled.
    - Outcome check: Run the DLO-hooked DiT forward pass on TPU and compare
      output with CPU fp32 reference.
    """
    torch.manual_seed(42)
    ref_pipeline = _TinyDLOPipeline(dtype=torch.float32, device=torch.device("cpu"))
    x_cpu = torch.randn(2, 4, dtype=torch.float32)

    ckpt_dir = tmp_path / "transformer"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    bf16_state = {
        name: tensor.to(dtype=torch.bfloat16)
        for name, tensor in ref_pipeline.transformer.state_dict().items()
    }
    save_file(bf16_state, str(ckpt_dir / "model.safetensors"))

    tpu_device = torch.device("tpu")
    pipeline = _TinyDLOPipeline(dtype=torch.bfloat16, device=torch.device("meta"))
    plan_result = build_checkpoint_mmap_plan(
        pipeline,
        dit_modules=(("transformer", pipeline.transformer),),
        sources=(SimpleNamespace(prefix="transformer.", model_or_path=str(ckpt_dir)),),
        model_path=str(tmp_path),
        tensor_parallel_size=1,
        use_hsdp=False,
        online_quantization=False,
    )
    assert plan_result.plan is not None, (
        f"Expected checkpoint mmap plan, got fallback: {plan_result.fallback_reason}"
    )
    attach_offload_startup_state(
        pipeline, OffloadStartupState(host_weight_plan=plan_result.plan)
    )

    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        diffusion_offload_config={
            "mode": "layer",
            "components": ["dit"],
            "layer_options": {"dit": dit_layer_options},
        },
    )
    pipeline, backend = enable_offload_backend(od_config, pipeline, device=tpu_device)
    assert isinstance(backend, DistributedLayerwiseOffloadBackend), (
        f"Expected DistributedLayerwiseOffloadBackend, got {type(backend).__name__}"
    )
    assert backend.enabled, "DistributedLayerwiseOffloadBackend was not enabled"

    tpu_out = pipeline.transformer(x_cpu.to(dtype=torch.bfloat16, device=tpu_device))
    ref_out = ref_pipeline.transformer(x_cpu)
    assert_parity(
        tpu_out,
        ref_out,
        name="distributed_layerwise_offload",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
