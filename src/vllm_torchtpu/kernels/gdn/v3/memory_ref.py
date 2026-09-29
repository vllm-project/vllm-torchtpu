# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.memory_ref`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.memory_ref import (
    BaseBufferedRef,
    ConvWeightsRef,
    ExternalStateBufferedRef,
    GDNWeightsRef,
    InBufferedRef,
    MetadataRef,
    OutBufferedRef,
    SmemWrapper,
    StateBufferedRef,
    WeightRefs,
    create_allocs,
)

__all__ = [
    "BaseBufferedRef",
    "ConvWeightsRef",
    "ExternalStateBufferedRef",
    "GDNWeightsRef",
    "InBufferedRef",
    "MetadataRef",
    "OutBufferedRef",
    "SmemWrapper",
    "StateBufferedRef",
    "WeightRefs",
    "create_allocs",
]
