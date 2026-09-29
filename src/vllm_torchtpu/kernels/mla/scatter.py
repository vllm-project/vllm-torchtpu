# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.mla.scatter`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.mla.scatter import (
    scatter,
)

__all__ = [
    "scatter",
]
