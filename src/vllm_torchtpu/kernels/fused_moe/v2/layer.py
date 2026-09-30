# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.fused_moe.v2.layer`.
"""

import sys
from typing import Any

from vllm_torchtpu.llm_tpu_core.kernels.fused_moe.v2 import (
    layer as _core_module,
)

_this_module = sys.modules[__name__]

for _k in dir(_core_module):
    if not (_k.startswith("__") and _k.endswith("__")):
        setattr(_this_module, _k, getattr(_core_module, _k))


class _FusedMoeV2LayerModuleWrapper:
    def __init__(self, wrapped_module):
        self.__dict__["_wrapped_module"] = wrapped_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value
        setattr(self._wrapped_module, name, value)
        if hasattr(_core_module, name):
            setattr(_core_module, name, value)


sys.modules[__name__] = _FusedMoeV2LayerModuleWrapper(_this_module)
