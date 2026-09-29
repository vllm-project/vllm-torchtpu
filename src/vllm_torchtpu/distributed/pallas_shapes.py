# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.distributed.pallas_shapes`.
"""

from vllm_torchtpu.llm_tpu_core.distributed.pallas_shapes import (
    get_global_shape,
    get_local_shape,
    install_pallas_shape_conversion,
)

__all__ = [
    "get_global_shape",
    "get_local_shape",
    "install_pallas_shape_conversion",
]
