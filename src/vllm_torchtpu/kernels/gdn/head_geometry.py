# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.gdn.head_geometry`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.gdn.head_geometry import (
    GdnHeadGeometry,
    derive_gdn_head_geometry,
)

__all__ = [
    "GdnHeadGeometry",
    "derive_gdn_head_geometry",
]
