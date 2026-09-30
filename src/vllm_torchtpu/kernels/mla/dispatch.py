# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.mla.dispatch`.
"""

import sys
from typing import Any

from vllm_torchtpu.llm_tpu_core.kernels.mla import dispatch as _core_dispatch
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

_this_module = sys.modules[__name__]

for _k in dir(_core_dispatch):
    if not (_k.startswith("__") and _k.endswith("__")):
        setattr(_this_module, _k, getattr(_core_dispatch, _k))


class _MlaDispatchModuleWrapper:
    def __init__(self, wrapped_module):
        self.__dict__["_wrapped_module"] = wrapped_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value
        setattr(self._wrapped_module, name, value)
        if hasattr(_core_dispatch, name):
            setattr(_core_dispatch, name, value)


sys.modules[__name__] = _MlaDispatchModuleWrapper(_this_module)
