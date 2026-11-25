# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/common/__init__.py
"""Shared Kimi-K3 megakernel arena pool, collective, and aliasing helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .constants import GIB_BYTES, MIB_BYTES

if TYPE_CHECKING:
    from types import ModuleType

    from . import pool_alias


def __getattr__(name: str) -> ModuleType:
    """Lazily load ``pool_alias`` to skip its import-time lowering registration.

    Args:
        name: Attribute requested on ``common``.

    Returns:
        Lazily imported submodule.

    Raises:
        AttributeError: If ``name`` is not a lazily exposed submodule.
    """
    if name == "pool_alias":
        return importlib.import_module(".pool_alias", __name__)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "GIB_BYTES",
    "MIB_BYTES",
    "pool_alias",
]
