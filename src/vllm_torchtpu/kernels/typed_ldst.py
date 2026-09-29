# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Typed load/store helpers for VMEM/HBM.

Re-exports core primitives from `llm_tpu_core.kernels.typed_ldst`.
"""

from __future__ import annotations

from vllm_torchtpu.llm_tpu_core.kernels.typed_ldst import (
    _typed_view,
    load_typed,
    store_typed,
)

__all__ = [
    "_typed_view",
    "load_typed",
    "store_typed",
]
