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
"""Tests for optimization barrier custom op."""

import pytest
import torch

pytestmark = pytest.mark.tpu_test


class TestOptimizationBarrierOnTpu:
    """Verify pallas::optimization_barrier executes on TPU and preserves values."""

    def test_optimization_barrier_on_tpu(self) -> None:
        from vllm_torchtpu.layers.adapter.custom_ops.optimization_barrier_op import (
            optimization_barrier,
        )

        torch.manual_seed(42)
        x_tpu = torch.randn(16, 128, dtype=torch.bfloat16, device="tpu")
        out = optimization_barrier(x_tpu)
        assert out.device.type == "tpu"
        assert out.dtype == torch.bfloat16
        assert torch.equal(out.cpu(), x_tpu.cpu())
