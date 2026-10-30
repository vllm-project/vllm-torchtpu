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
"""Load the built-in registrations on first registry lookup.

The package `__init__` resolves names lazily, so importing `megakernel` no
longer imports `megakernel.modules` (module tunables / debug options,
registered by `Module.__init_subclass__`), `megakernel.models` or
`megakernel.profiles` (built-in models and profiles), or
`megakernel.experimental` (the fused pool I/O implementation). Registry
lookups call `ensure_builtins()` first. A missing package (a partial tree, or one that
moved out) is skipped.
"""

from __future__ import annotations

import contextlib
import importlib

_LOADED = False


def ensure_builtins() -> None:
    """Import the built-in modules, models, profiles and fused pool I/O once."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for name in ("modules", "models", "profiles", "experimental"):
        with contextlib.suppress(ImportError):
            importlib.import_module(f"{__package__}.{name}")
