# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.experimental.adaptive_fused_moe.host`.
"""

import sys
from typing import Any

from vllm_torchtpu.llm_tpu_core.kernels.experimental.adaptive_fused_moe import (
    host as _core_host,
)

_this_module = sys.modules[__name__]

for _k in dir(_core_host):
    if not (_k.startswith("__") and _k.endswith("__")):
        setattr(_this_module, _k, getattr(_core_host, _k))


class _HostModuleWrapper:
    def __init__(self, wrapped_module):
        self.__dict__["_wrapped_module"] = wrapped_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value
        setattr(self._wrapped_module, name, value)
        if hasattr(_core_host, name):
            setattr(_core_host, name, value)


sys.modules[__name__] = _HostModuleWrapper(_this_module)
