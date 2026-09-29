# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.typed_ldst`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.typed_ldst import (
    load_typed,
    store_typed,
)

__all__ = [
    "load_typed",
    "store_typed",
]
