# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.pool_adapters`.
"""

import sys
from typing import Any

from vllm_torchtpu.llm_tpu_core.kernels import pool_adapters as _core_pool_adapters
from vllm_torchtpu.llm_tpu_core.kernels.pool_adapters import (
    copy_blocks,
    gather_region,
    is_seq_along_lane_pool,
    scatter_region,
    v3_state_source,
)

__all__ = [
    "copy_blocks",
    "gather_region",
    "is_seq_along_lane_pool",
    "scatter_region",
    "v3_state_source",
]

_this_module = sys.modules[__name__]

for _k in dir(_core_pool_adapters):
    if not (_k.startswith("__") and _k.endswith("__")):
        setattr(_this_module, _k, getattr(_core_pool_adapters, _k))


class _PoolAdaptersModuleWrapper:
    def __init__(self, wrapped_module):
        self.__dict__["_wrapped_module"] = wrapped_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value
        setattr(self._wrapped_module, name, value)
        if hasattr(_core_pool_adapters, name):
            setattr(_core_pool_adapters, name, value)


sys.modules[__name__] = _PoolAdaptersModuleWrapper(_this_module)
