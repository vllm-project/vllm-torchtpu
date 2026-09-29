# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.mla.v2.kernel`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.mla.v2.kernel import (
    MlaCase,
    align_to,
    get_dtype_bitwidth,
    get_dtype_packing,
    get_kv_cache_shape,
    logger,
    mla_ragged_paged_attention,
    prepare_kv_inputs,
    prepare_kv_inputs_for_transposed_kv_cache,
    prepare_outputs,
    prepare_q_inputs,
    prepare_q_nope_inputs,
    static_validate_inputs,
    unsigned_align_to,
    unsigned_cdiv,
    unsigned_floor_div,
    unsigned_mod,
)

__all__ = [
    "MlaCase",
    "align_to",
    "get_dtype_bitwidth",
    "get_dtype_packing",
    "get_kv_cache_shape",
    "logger",
    "mla_ragged_paged_attention",
    "prepare_kv_inputs",
    "prepare_kv_inputs_for_transposed_kv_cache",
    "prepare_outputs",
    "prepare_q_inputs",
    "prepare_q_nope_inputs",
    "static_validate_inputs",
    "unsigned_align_to",
    "unsigned_cdiv",
    "unsigned_floor_div",
    "unsigned_mod",
]
