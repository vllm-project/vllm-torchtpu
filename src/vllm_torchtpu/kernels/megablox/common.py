# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.megablox.common`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.megablox.common import (
    assert_is_supported_dtype,
    is_tpu,
    tpu_generation,
    tpu_kind,
)

__all__ = [
    "assert_is_supported_dtype",
    "is_tpu",
    "tpu_generation",
    "tpu_kind",
]
