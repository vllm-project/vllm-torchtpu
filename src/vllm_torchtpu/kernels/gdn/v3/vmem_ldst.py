# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.vmem_ldst`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.vmem_ldst import (
    load_activation_as_compact,
    load_activation_as_large,
    load_and_select_states,
    load_compact_heads,
    load_compact_to_large,
    load_state_region,
    load_strided_heads,
    store_state_region,
)

__all__ = [
    "load_activation_as_compact",
    "load_activation_as_large",
    "load_and_select_states",
    "load_compact_heads",
    "load_compact_to_large",
    "load_state_region",
    "load_strided_heads",
    "store_state_region",
]
