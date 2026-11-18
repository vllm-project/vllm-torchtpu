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
"""GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) for the TorchTPU vLLM plugin.

The model classes load lazily, so importing this package (for example to
reach `.config` during model registration) does not build the model module.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model import Glm5NextForCausalLM, Glm5NextForConditionalGeneration

__all__ = ["Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(importlib.import_module(".model", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
