# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""TPU Quantization and Dequantization operations.

Re-exports core primitives from `llm_tpu_core.quantization.ops`.
"""

from vllm_torchtpu.llm_tpu_core.quantization.ops import (
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
