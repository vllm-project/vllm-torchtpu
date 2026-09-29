# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.kimi_k3.decode_kda`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.kimi_k3.decode_kda import (
    decode_kda,
)

__all__ = [
    "decode_kda",
]
