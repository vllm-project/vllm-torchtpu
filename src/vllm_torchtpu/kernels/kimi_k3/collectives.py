# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from `llm_tpu_core.kernels.kimi_k3.collectives`.
Synchronizes monkeypatched attributes with llm_tpu_core.
"""

import sys
from typing import Any

from vllm_torchtpu.llm_tpu_core.kernels.kimi_k3 import collectives as _core_k3
from vllm_torchtpu.llm_tpu_core.kernels.kimi_k3.collectives import (
    CYCLES,
    _regions,
    gather_moe_inputs,
    gather_project,
    mla_chip_layout,
    mla_preprocess,
    pack_mla_gate,
    project_reduce_scatter,
    ring_collective,
    ring_tables,
)


class _CollectivesModuleWrapper:
    def __init__(self, wrapped_module):
        self.__dict__["_wrapped_module"] = wrapped_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value
        if hasattr(_core_k3, name):
            setattr(_core_k3, name, value)


sys.modules[__name__] = _CollectivesModuleWrapper(sys.modules[__name__])

__all__ = [
    "CYCLES",
    "_regions",
    "gather_moe_inputs",
    "gather_project",
    "mla_chip_layout",
    "mla_preprocess",
    "pack_mla_gate",
    "project_reduce_scatter",
    "ring_collective",
    "ring_tables",
]
