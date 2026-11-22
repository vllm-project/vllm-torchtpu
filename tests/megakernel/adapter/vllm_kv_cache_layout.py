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
"""vLLM's `KVCacheLayout` for tests that run with or without vLLM installed."""

from __future__ import annotations

import enum
import importlib
import sys
import types
from typing import Any

import pytest

_MODULE = "vllm.v1.kv_cache_layout"


class _KVCacheLayout(enum.Enum):
    """The members of vLLM's `KVCacheLayout` that the megakernel names."""

    LBNHC = "LBNHC"
    LBHNC = "LBHNC"
    BLNHC = "BLNHC"
    BLHNC = "BLHNC"


def kv_cache_layout(monkeypatch: pytest.MonkeyPatch) -> Any:
    """vLLM's `KVCacheLayout`; without vLLM (CPU), an enum with its members,
    installed as `vllm.v1.kv_cache_layout` for the code under test."""
    try:
        return importlib.import_module(_MODULE).KVCacheLayout
    except ModuleNotFoundError:
        stub = types.ModuleType(_MODULE)
        stub.KVCacheLayout = _KVCacheLayout  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, _MODULE, stub)
        return _KVCacheLayout
