# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Donating slot-to-slot copy for Mamba recurrent state arrays.

Re-exports core primitives from `llm_tpu_core.layers.mamba_state_copy_op`.
"""

import sys
from typing import Any

from vllm_torchtpu.llm_tpu_core.layers import (
    mamba_state_copy_op as _core_module,
)

_this_module = sys.modules[__name__]

for _k in dir(_core_module):
    if not (_k.startswith("__") and _k.endswith("__")):
        setattr(_this_module, _k, getattr(_core_module, _k))


class _MambaStateCopyOpModuleWrapper:
    def __init__(self, wrapped_module):
        self.__dict__["_wrapped_module"] = wrapped_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value
        setattr(self._wrapped_module, name, value)
        if hasattr(_core_module, name):
            setattr(_core_module, name, value)


sys.modules[__name__] = _MambaStateCopyOpModuleWrapper(_this_module)
