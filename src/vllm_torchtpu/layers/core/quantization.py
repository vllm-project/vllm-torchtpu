# Copyright 2025 Google LLC
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
"""PyTorch MXFP4 and FP8 Quantization Utilities for TPU.

Re-exports core primitives from `llm_tpu_core.quantization`.
"""

from vllm_torchtpu.llm_tpu_core.quantization import (
    E8M0_MAX_EXP,
    E8M0_MIN_EXP,
    FP8_E4M3_MAX,
    MXFP4_BLOCK_SIZE,
    REQUANTIZED_BLOCK_SIZE,
    _safe_inverse_scale,
    dequantize_mxfp4_packed,
    dequantize_tensor,
    dtype_max,
    e8m0_to_fp32,
    fp4_indices_to_float,
    is_floating_dtype,
    pack_fp4_indices,
    quantize_kv,
    quantize_tensor,
    quantize_tensor_to_fp4,
    static_per_tensor_quantize_tensor,
    unpack_uint8_to_fp4,
)

__all__ = [
    "E8M0_MAX_EXP",
    "E8M0_MIN_EXP",
    "FP8_E4M3_MAX",
    "MXFP4_BLOCK_SIZE",
    "REQUANTIZED_BLOCK_SIZE",
    "_safe_inverse_scale",
    "dequantize_mxfp4_packed",
    "dequantize_tensor",
    "dtype_max",
    "e8m0_to_fp32",
    "fp4_indices_to_float",
    "is_floating_dtype",
    "pack_fp4_indices",
    "quantize_kv",
    "quantize_tensor",
    "quantize_tensor_to_fp4",
    "static_per_tensor_quantize_tensor",
    "unpack_uint8_to_fp4",
]
