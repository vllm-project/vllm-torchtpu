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
"""TPU tests for diffusion denoise-step device graph capture and replay."""

from __future__ import annotations

import pytest
import torch
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_diffusion_dir

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

MAX_REL_L2 = 1.0e-2
MAX_ABS = 2.0e-2


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="not in vllm-omni 0.32.0rc1",
)
def test_denoise_step_device_graphs(omni_ctx: None) -> None:
    """Denoise Step Device Graphs.

    - TPU path: import ``build_denoise_graph_runner``, request device graph
      capture on ``torch.device("tpu")`` with ``diffusion_use_graph=True`` and
      ``enforce_eager=False``, and replay two denoise steps.
    - Setup check: ``current_omni_platform.device_type == "tpu"`` and
      ``build_denoise_graph_runner`` returns an enabled graph runner.
    - Outcome check: two captured/replayed step outputs match CPU FP32
      reference via ``assert_parity``, and graph release/invalidation drop
      resident graphs cleanly.
    """
    from vllm_omni.diffusion.graph import (
        build_denoise_graph_runner,
        invalidate_denoise_graphs,
        release_denoise_graphs_after_denoise,
    )

    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU platform, got {current_omni_platform.device_type}"
    )

    od_config = OmniDiffusionConfig(
        model=tiny_diffusion_dir(),
        diffusion_use_graph=True,
        diffusion_graph_release_after_denoise=True,
        enforce_eager=False,
    )
    tpu_device = torch.device("tpu")
    torch.manual_seed(0)
    weight_cpu = torch.randn(16, 16, dtype=torch.float32) * 0.1
    weight_tpu = weight_cpu.to(dtype=torch.bfloat16, device=tpu_device)

    def _step_fn(*, hidden_states: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (hidden_states @ weight_tpu.T) * scale

    runner = build_denoise_graph_runner(od_config, _step_fn, "tpu_denoise_step")
    assert runner is not None, (
        "build_denoise_graph_runner returned None; TPU has no denoise device graph"
    )

    x1_cpu = torch.randn(2, 16, dtype=torch.float32)
    x2_cpu = torch.randn(2, 16, dtype=torch.float32)
    scale_cpu = torch.tensor(0.5, dtype=torch.float32)

    out1 = runner(
        hidden_states=x1_cpu.to(dtype=torch.bfloat16, device=tpu_device),
        scale=scale_cpu.to(dtype=torch.bfloat16, device=tpu_device),
    )
    out2 = runner(
        hidden_states=x2_cpu.to(dtype=torch.bfloat16, device=tpu_device),
        scale=scale_cpu.to(dtype=torch.bfloat16, device=tpu_device),
    )
    assert_parity(
        out1,
        (x1_cpu @ weight_cpu.T) * scale_cpu,
        name="denoise_graph_step1",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
    assert_parity(
        out2,
        (x2_cpu @ weight_cpu.T) * scale_cpu,
        name="denoise_graph_step2",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )

    release_denoise_graphs_after_denoise(od_config, runner)
    assert runner.num_graphs == 0, (
        f"Expected 0 graphs after release, got {runner.num_graphs}"
    )
    invalidate_denoise_graphs("test_cleanup")
