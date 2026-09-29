# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.config`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.config import (
    AttentionMode,
    Dtypes,
    GDNConfig,
    GDNMode,
    StateRegion,
    StateSourcePlan,
)

__all__ = [
    "AttentionMode",
    "Dtypes",
    "GDNConfig",
    "GDNMode",
    "StateRegion",
    "StateSourcePlan",
]
