# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.compute_kda`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.compute_kda import (
    activate_gate,
    chunked_kda,
    chunked_kda_per_seq,
    recurrent_kda,
    recurrent_kda_per_seq,
)

__all__ = [
    "activate_gate",
    "chunked_kda",
    "chunked_kda_per_seq",
    "recurrent_kda",
    "recurrent_kda_per_seq",
]
