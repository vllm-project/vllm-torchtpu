# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Sort-free MoE router top-k Pallas kernel.

Re-exports core primitives from `llm_tpu_core.kernels.router_topk`.
"""

from __future__ import annotations

from vllm_torchtpu.llm_tpu_core.kernels.router_topk import (
    MAX_BLOCK_ROWS,
    NEG,
    _select_kernel,
    router_topk,
    select,
)

__all__ = [
    "MAX_BLOCK_ROWS",
    "NEG",
    "_select_kernel",
    "router_topk",
    "select",
]
