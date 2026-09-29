# SPDX-License-Identifier: Apache-2.0
"""Shape conversion for native Pallas ops with multi-axis PartitionSpecs.

Re-exports core primitives from `llm_tpu_core.distributed.pallas_shapes`.
"""

from __future__ import annotations

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
