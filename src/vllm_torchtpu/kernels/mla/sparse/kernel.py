# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.mla.sparse.kernel`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.mla.sparse.kernel import (
    DEFAULT_VMEM_LIMIT_BYTES,
    SPARSECORE_CACHE_LAYOUT,
    TENSORCORE_CACHE_LAYOUT,
    align_to,
    cdiv,
    gather_page_ids,
    get_dtype_bitwidth,
    get_dtype_packing,
    prepare_outputs,
    prepare_q_inputs,
    sparse_ragged_paged_attention,
)

__all__ = [
    "DEFAULT_VMEM_LIMIT_BYTES",
    "SPARSECORE_CACHE_LAYOUT",
    "TENSORCORE_CACHE_LAYOUT",
    "align_to",
    "cdiv",
    "gather_page_ids",
    "get_dtype_bitwidth",
    "get_dtype_packing",
    "prepare_outputs",
    "prepare_q_inputs",
    "sparse_ragged_paged_attention",
]
