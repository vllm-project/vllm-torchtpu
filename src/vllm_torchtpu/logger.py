# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.logger`.
"""

from vllm_torchtpu.llm_tpu_core.logger import (
    init_logger,
)

__all__ = [
    "init_logger",
]
