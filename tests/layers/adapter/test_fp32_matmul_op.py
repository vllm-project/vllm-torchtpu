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
"""The scoped full-precision FP32 matmul op."""

import pytest
import torch

from vllm_torchtpu.layers.adapter.custom_ops import fp32_matmul_op
from vllm_torchtpu.layers.adapter.custom_ops.fp32_matmul_op import (
    fp32_matmul,
    prebuild_fp32_matmul_op,
)


def test_prebuild_builds_the_op_once():
    prebuild_fp32_matmul_op()
    op = fp32_matmul_op._ops["fp32_matmul"]
    prebuild_fp32_matmul_op()
    assert fp32_matmul_op._ops["fp32_matmul"] is op


def test_cpu_path_is_an_fp32_matmul():
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, 5)
    out = fp32_matmul(x, weight)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, x.float() @ weight)


@pytest.mark.tpu_test
@torch.inference_mode()
def test_compiled_router_matmul_keeps_full_fp32_precision():
    """The op traces inside a full-graph compile once prebuilt, and rounds
    like FP32 rather than like BF16 passes."""
    prebuild_fp32_matmul_op()
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(64, 4096, generator=generator).bfloat16()
    weight = torch.randn(4096, 256, generator=generator)
    expected = (x.double() @ weight.double()).float()

    compiled = torch.compile(
        lambda a, b: fp32_matmul(a, b) * 2.0, backend="tpu", fullgraph=True
    )
    actual = compiled(x.to("tpu"), weight.to("tpu")).cpu() / 2.0

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)
