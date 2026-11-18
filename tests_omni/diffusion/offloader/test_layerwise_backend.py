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
"""TPU tests for layerwise CPU offload in vllm-omni diffusion."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.offloader import (
    LayerWiseOffloadBackend,
    enable_offload_backend,
    is_materialized_tensor,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_diffusion_dir

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

MAX_REL_L2 = 2.0e-2
MAX_ABS = 5.0e-2


class _TinyDiT(nn.Module):
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


class _TinyLayerwisePipeline(nn.Module):
    def __init__(self, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.transformer = _TinyDiT(dtype=dtype, device=device)


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
        {"diffusion_offload_config": {"mode": "layer", "components": ["dit"]}},
        {"enable_layerwise_offload": True},
    ],
    ids=["compact_layer", "legacy_enable_layerwise_offload"],
)
def test_layerwise_cpu_offload(offload_kwargs: dict[str, Any]) -> None:
    """Layerwise CPU Offload.

    - TPU path: Build a tiny 3-block bf16 DiT pipeline on TPU and call
      enable_offload_backend with compact mode="layer" or legacy
      enable_layerwise_offload=True.
    - Setup check: Assert LayerWiseOffloadBackend is enabled and non-initial DiT
      blocks have offloaded placeholder storage before forward.
    - Outcome check: Run the 3-block DiT forward pass on TPU and compare output
      with CPU fp32 reference without offloading.
    """
    torch.manual_seed(42)
    tpu_device = torch.device("tpu")
    ref_pipeline = _TinyLayerwisePipeline(
        dtype=torch.float32, device=torch.device("cpu")
    )
    x_cpu = torch.randn(2, 4, dtype=torch.float32)

    pipeline = _TinyLayerwisePipeline(dtype=torch.bfloat16, device=tpu_device)
    with torch.no_grad():
        for tpu_blk, ref_blk in zip(
            pipeline.transformer.blocks, ref_pipeline.transformer.blocks
        ):
            tpu_blk.weight.copy_(
                ref_blk.weight.to(dtype=torch.bfloat16, device=tpu_device)
            )

    od_config = OmniDiffusionConfig(model=tiny_diffusion_dir(), **offload_kwargs)
    pipeline, backend = enable_offload_backend(od_config, pipeline, device=tpu_device)
    assert isinstance(backend, LayerWiseOffloadBackend), (
        f"Expected LayerWiseOffloadBackend, got {type(backend).__name__}"
    )
    assert backend.enabled, "LayerWiseOffloadBackend was not enabled"
    for idx, block in enumerate(pipeline.transformer.blocks[1:], start=1):
        assert not is_materialized_tensor(block.weight), (
            f"Expected non-initial block {idx} weight to use offloaded placeholder"
        )

    tpu_out = pipeline.transformer(x_cpu.to(dtype=torch.bfloat16, device=tpu_device))
    ref_out = ref_pipeline.transformer(x_cpu)
    assert_parity(
        tpu_out,
        ref_out,
        name="layerwise_cpu_offload",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
