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
"""`quantize_kv_fp4_per_token` packs K/V to e2m1 with one scale per token-head.

CLAUDE COMMENT: new test for the per-token FP4 KV cache port. Dequantizing
with `unpack_uint8_to_fp4` (the batched RPA kernel's nibble order) times the
returned scale must reproduce the input: exactly for values already on the
scaled e2m1 grid, and to FP4 precision otherwise.
"""

import pytest
import torch

from vllm_torchtpu.layers.core.quantization import (
    quantize_kv_fp4_per_token,
    unpack_uint8_to_fp4,
)

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _dequantize(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return unpack_uint8_to_fp4(packed) * scale.to(torch.float32)


@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, torch.float32])
def test_shapes_and_dtypes(scale_dtype: torch.dtype):
    key = torch.randn(5, 3, 128, dtype=torch.bfloat16)
    value = torch.randn(5, 3, 128, dtype=torch.bfloat16)

    k_packed, v_packed, k_scale, v_scale = quantize_kv_fp4_per_token(
        key, value, scale_dtype=scale_dtype
    )

    assert k_packed.shape == v_packed.shape == (5, 3, 64)
    assert k_packed.dtype == v_packed.dtype == torch.uint8
    assert k_scale.shape == v_scale.shape == (5, 3, 1)
    assert k_scale.dtype == v_scale.dtype == scale_dtype


@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, torch.float32])
def test_values_on_the_scaled_grid_round_trip_exactly(scale_dtype: torch.dtype):
    """The stored scale is the one used to quantize, so grid values are exact."""
    torch.manual_seed(0)
    idx = torch.randint(0, 8, (4, 2, 128))
    sign = torch.where(torch.rand(4, 2, 128) < 0.5, -1.0, 1.0)
    # One element per token-head at the max magnitude pins amax / 6.
    idx[..., 0] = 7
    scale = torch.tensor([0.37, 2.5]).view(1, 2, 1).to(scale_dtype).float()
    key = _E2M1[idx] * sign * scale

    k_packed, _, k_scale, _ = quantize_kv_fp4_per_token(
        key, key, scale_dtype=scale_dtype
    )

    torch.testing.assert_close(_dequantize(k_packed, k_scale), key, rtol=0, atol=0)


def test_gaussian_round_trip_error_is_fp4_sized():
    torch.manual_seed(0)
    key = torch.randn(64, 4, 128) * torch.rand(64, 4, 1) * 10

    k_packed, _, k_scale, _ = quantize_kv_fp4_per_token(key, key)

    rel = torch.linalg.vector_norm(_dequantize(k_packed, k_scale) - key) / (
        torch.linalg.vector_norm(key)
    )
    assert rel < 0.15, f"relative L2 error {rel:.3f}"


def test_zero_rows_stay_zero():
    key = torch.zeros(2, 1, 16)

    k_packed, _, k_scale, _ = quantize_kv_fp4_per_token(key, key)

    assert torch.count_nonzero(k_scale) == 0
    assert torch.count_nonzero(_dequantize(k_packed, k_scale)) == 0


def test_odd_head_dim_is_rejected():
    key = torch.randn(1, 1, 7)
    with pytest.raises(ValueError, match="even head_dim"):
        quantize_kv_fp4_per_token(key, key)
