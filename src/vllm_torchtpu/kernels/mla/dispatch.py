# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.mla.dispatch`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.mla.dispatch import (
    GLM52_INDEX_TOPK,
    GLM52_NOPE_DIM,
    GLM52_NUM_HEADS,
    GLM52_PAGE_SIZE,
    GLM52_ROPE_DIM,
    MASKED_DENSE_ANALYTIC_LIMIT,
    MASKED_DENSE_ANALYTIC_MAX_KV_LEN,
    MASKED_DENSE_KV_BLOCK,
    MASKED_DENSE_KV_PAGES_PER_BLOCK,
    MASKED_DENSE_LIMIT,
    MASKED_DENSE_MAX_KV_LEN,
    MASKED_DENSE_MIN_TOKEN_BUCKET,
    MASKED_DENSE_PREFILL_COST_MODEL,
    MASKED_DENSE_QUERIES_PER_BLOCK,
    masked_dense_prefill_mla_tier,
    matches_glm52_tpu7x_profile,
    ragged_paged_attention,
)

__all__ = [
    "GLM52_INDEX_TOPK",
    "GLM52_NOPE_DIM",
    "GLM52_NUM_HEADS",
    "GLM52_PAGE_SIZE",
    "GLM52_ROPE_DIM",
    "MASKED_DENSE_ANALYTIC_LIMIT",
    "MASKED_DENSE_ANALYTIC_MAX_KV_LEN",
    "MASKED_DENSE_KV_BLOCK",
    "MASKED_DENSE_KV_PAGES_PER_BLOCK",
    "MASKED_DENSE_LIMIT",
    "MASKED_DENSE_MAX_KV_LEN",
    "MASKED_DENSE_MIN_TOKEN_BUCKET",
    "MASKED_DENSE_PREFILL_COST_MODEL",
    "MASKED_DENSE_QUERIES_PER_BLOCK",
    "masked_dense_prefill_mla_tier",
    "matches_glm52_tpu7x_profile",
    "ragged_paged_attention",
]
