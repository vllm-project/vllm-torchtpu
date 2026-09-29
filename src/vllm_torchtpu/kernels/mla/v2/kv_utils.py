# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.mla.v2.kv_utils`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.mla.v2.kv_utils import (
    align_to,
    get_dtype_bitwidth,
    get_dtype_packing,
    pack_new_kv,
    pack_new_kv_reference,
    unsigned_align_to,
    unsigned_cdiv,
    unsigned_floor_div,
    unsigned_mod,
)

__all__ = [
    "align_to",
    "get_dtype_bitwidth",
    "get_dtype_packing",
    "pack_new_kv",
    "pack_new_kv_reference",
    "unsigned_align_to",
    "unsigned_cdiv",
    "unsigned_floor_div",
    "unsigned_mod",
]
