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
"""Tests for `fp8_linear_op` custom ops on CPU and TPU."""

import pytest
import torch

from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.custom_ops import fp8_linear_op


def test_quantize_fp8_activations_rejects_non_2d_or_non_bf16() -> None:
    with pytest.raises(NotImplementedError, match="2-D bfloat16"):
        fp8_linear_op.quantize_fp8_activations(
            torch.zeros((2, 4, 8), dtype=torch.bfloat16)
        )
    with pytest.raises(NotImplementedError, match="2-D bfloat16"):
        fp8_linear_op.quantize_fp8_activations(torch.zeros((4, 8), dtype=torch.float32))


def test_prequantized_fp8_matmul_validates_inputs() -> None:
    x_q = torch.zeros((4, 8), dtype=torch.uint8)
    x_scale = torch.ones((4, 1), dtype=torch.float32)
    w = torch.zeros((8, 4), dtype=torch.float8_e4m3fn)
    ws = torch.ones((4,), dtype=torch.float32)

    with pytest.raises(NotImplementedError, match="2-D uint8 x_q"):
        fp8_linear_op.prequantized_fp8_matmul(
            torch.zeros((1, 4, 8), dtype=torch.uint8), x_scale, w, ws
        )
    with pytest.raises(NotImplementedError, match="2-D uint8 x_q"):
        fp8_linear_op.prequantized_fp8_matmul(
            torch.zeros((4, 8), dtype=torch.bfloat16), x_scale, w, ws
        )
    # 1-D x_scale of length m would otherwise broadcast across columns when m == n.
    with pytest.raises(NotImplementedError, match=r"float32 x_scale of shape \(4, 1\)"):
        fp8_linear_op.prequantized_fp8_matmul(
            x_q, torch.ones((4,), dtype=torch.float32), w, ws
        )
    with pytest.raises(NotImplementedError, match=r"float32 x_scale of shape \(4, 1\)"):
        fp8_linear_op.prequantized_fp8_matmul(
            x_q, torch.ones((4, 1), dtype=torch.bfloat16), w, ws
        )
    with pytest.raises(NotImplementedError, match=r"float32 x_scale of shape \(4, 1\)"):
        fp8_linear_op.prequantized_fp8_matmul(
            x_q, torch.ones((3, 1), dtype=torch.float32), w, ws
        )
    with pytest.raises(NotImplementedError, match=r"float32 x_scale of shape \(4, 1\)"):
        fp8_linear_op.prequantized_fp8_matmul(
            x_q, torch.ones((4, 2), dtype=torch.float32), w, ws
        )
    with pytest.raises(NotImplementedError, match=r"shape\[0\] == x_q\.shape\[1\]"):
        fp8_linear_op.prequantized_fp8_matmul(
            x_q, x_scale, torch.zeros((6, 4), dtype=torch.float8_e4m3fn), ws
        )
    with pytest.raises(NotImplementedError, match=r"2-D weight"):
        fp8_linear_op.prequantized_fp8_matmul(
            x_q, x_scale, torch.zeros((8,), dtype=torch.float8_e4m3fn), ws
        )


@pytest.mark.tpu_test
def test_fp8_linear_custom_ops_match_quantized_matmul() -> None:
    """Sharded FP8 quantize + pre-quantized matmul match the fused FP8 matmul."""
    fp8_linear_op.register_fp8_linear_ops()
    gen = torch.Generator().manual_seed(7)
    x_cpu = (torch.randn((32, 64), generator=gen) * 3.0).to(torch.bfloat16)
    x_cpu[5] = 0
    w_cpu = torch.randn((64, 48), generator=gen).to(torch.float8_e4m3fn)
    ws_cpu = torch.rand((48,), generator=gen, dtype=torch.float32) * 0.09 + 0.01

    x_tpu = x_cpu.to("tpu")
    w_tpu = w_cpu.to("tpu")
    ws_tpu = ws_cpu.to("tpu")

    shards = [
        fp8_linear_op.quantize_fp8_activations(s) for s in torch.chunk(x_tpu, 4, dim=0)
    ]
    x_q = torch.cat([q for q, _ in shards], dim=0)
    x_scale = torch.cat([s for _, s in shards], dim=0)
    assert x_q.dtype == torch.uint8 and x_q.shape == (32, 64)
    assert x_scale.dtype == torch.float32 and x_scale.shape == (32, 1)

    out = fp8_linear_op.prequantized_fp8_matmul(x_q, x_scale, w_tpu, ws_tpu)
    fused = linear_common.quantized_matmul(x_tpu, w_tpu, ws_tpu, w8a16_max_rows=0)
    assert torch.equal(out.cpu(), fused.cpu())
