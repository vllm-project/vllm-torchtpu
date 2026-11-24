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
"""Unit tests for static per-tensor KV quantization fast paths."""

from unittest.mock import patch

import pytest
import torch

import vllm_torchtpu.layers.core.quantization as quant_mod
from vllm_torchtpu.layers.core.quantization import (
    quantize_kv,
    static_per_tensor_quantize_tensor,
)


@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_unit_scale_fp8_matches_fp32_across_all_16bit_patterns(src_dtype, quant_dtype):
    """All 65,536 bf16/fp16 bit patterns match the fp32 clamp-and-cast path."""
    x = torch.arange(-32768, 32768, dtype=torch.int16).view(src_dtype)
    info = torch.finfo(quant_dtype)
    expected = torch.clamp(x.to(torch.float32), info.min, info.max).to(quant_dtype)
    got = static_per_tensor_quantize_tensor(x, quant_dtype, scale=1.0)

    assert got.dtype == quant_dtype
    assert torch.equal(got.isnan(), expected.isnan())
    non_nan = ~expected.isnan()
    assert torch.equal(
        got[non_nan].view(torch.uint8), expected[non_nan].view(torch.uint8)
    )


def test_unit_scale_non_fp8_float_targets_keep_fp32_clamp():
    """65504.0 rounds to 65536.0 in bf16; fp16<-bf16 and bf16<-fp16 need fp32 clamp."""
    bf16_vals = torch.tensor(
        [-float("inf"), -65536.0, 0.0, 65536.0, float("inf")], dtype=torch.bfloat16
    )
    to_fp16 = static_per_tensor_quantize_tensor(bf16_vals, torch.float16, scale=1.0)
    assert to_fp16.dtype == torch.float16
    assert torch.isfinite(to_fp16).all()
    assert torch.equal(
        to_fp16,
        torch.tensor([-65504.0, -65504.0, 0.0, 65504.0, 65504.0], dtype=torch.float16),
    )

    fp16_vals = torch.tensor([-65504.0, 0.0, 65504.0], dtype=torch.float16)
    to_bf16 = static_per_tensor_quantize_tensor(fp16_vals, torch.bfloat16, scale=1.0)
    assert to_bf16.dtype == torch.bfloat16
    assert torch.equal(to_bf16, fp16_vals.to(torch.float32).to(torch.bfloat16))


def test_zero_scale_raises_and_non_unit_scale_divides():
    x = torch.tensor([-900.0, -2.0, 0.0, 2.0, 900.0], dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="non-zero"):
        static_per_tensor_quantize_tensor(x, torch.float8_e4m3fn, scale=0.0)

    got = static_per_tensor_quantize_tensor(x, torch.float8_e4m3fn, scale=2.0)
    expected = torch.clamp(x.float() / 2.0, -448.0, 448.0).to(torch.float8_e4m3fn)
    assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))


def test_quantize_kv_deduplicates_identical_key_and_value():
    kv = torch.tensor([[1.0, -2.0, 500.0]], dtype=torch.bfloat16)
    with patch.object(
        quant_mod,
        "static_per_tensor_quantize_tensor",
        wraps=static_per_tensor_quantize_tensor,
    ) as spy:
        qk, qv = quantize_kv(torch.float8_e4m3fn, kv, kv, k_scale=1.0, v_scale=1.0)
        assert spy.call_count == 1
        assert qv is qk

        spy.reset_mock()
        qk2, qv2 = quantize_kv(torch.float8_e4m3fn, kv, kv, k_scale=1.0, v_scale=2.0)
        assert spy.call_count == 2
        assert qv2 is not qk2

        spy.reset_mock()
        qk3, qv3 = quantize_kv(torch.float8_e4m3fn, kv, None, k_scale=1.0, v_scale=1.0)
        assert spy.call_count == 1
        assert qk3.dtype == torch.float8_e4m3fn and qv3 is None
