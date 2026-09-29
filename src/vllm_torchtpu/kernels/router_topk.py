# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.router_topk`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.router_topk import (
    MAX_BLOCK_ROWS,
    NEG,
    router_topk,
    select,
)

__all__ = [
    "MAX_BLOCK_ROWS",
    "NEG",
    "router_topk",
    "select",
]
