# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.mla.scatter_utils`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.mla.scatter_utils import (
    Plan,
    scatter_rows,
    split_plan,
)

__all__ = [
    "Plan",
    "scatter_rows",
    "split_plan",
]
