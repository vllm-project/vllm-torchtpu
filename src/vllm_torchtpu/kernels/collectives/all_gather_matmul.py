# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""All-gather matmul kernel.

Re-exports core primitives from `llm_tpu_core.kernels.collectives`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.collectives.all_gather_matmul import (
    all_gather_matmul,
    get_kernel_name,
)

__all__ = [
    "all_gather_matmul",
    "get_kernel_name",
]
