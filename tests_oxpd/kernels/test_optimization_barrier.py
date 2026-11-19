# SPDX-License-Identifier: Apache-2.0
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
