# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper for PCP distributed collectives and mesh.

Re-exports core primitives from `vllm_torchtpu.llm_tpu_core.distributed.pcp`.
Synchronizes monkeypatched attributes with llm_tpu_core.
"""

import sys
from typing import Any

from vllm_torchtpu.llm_tpu_core.distributed import (
    pcp as _core_pcp,
)
from vllm_torchtpu.llm_tpu_core.distributed.pcp import (
    PcpGroupLayout,
    all_gather_equal_tokens,
    all_reduce_sum,
    get_or_create_pcp_mesh,
    get_pcp_group,
    get_pcp_group_layout,
    get_pcp_rank,
    get_pcp_world_size,
)


class _PcpModuleWrapper:
    def __init__(self, wrapped_module):
        self.__dict__["_wrapped_module"] = wrapped_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value
        if hasattr(_core_pcp, name):
            setattr(_core_pcp, name, value)


sys.modules[__name__] = _PcpModuleWrapper(sys.modules[__name__])

__all__ = [
    "PcpGroupLayout",
    "all_gather_equal_tokens",
    "all_reduce_sum",
    "get_or_create_pcp_mesh",
    "get_pcp_group",
    "get_pcp_group_layout",
    "get_pcp_rank",
    "get_pcp_world_size",
]
