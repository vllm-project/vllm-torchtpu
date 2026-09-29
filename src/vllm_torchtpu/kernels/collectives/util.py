# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.collectives.util`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.collectives.util import (
    local_barrier,
)

__all__ = [
    "local_barrier",
]
