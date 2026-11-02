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
registered by `Module.__init_subclass__`) or `megakernel.experimental` (the
fused pool I/O implementation). Registry lookups call `ensure_builtins(...)`
first, for the packages their registry needs. A missing package (a partial
tree, or one that moved out) is skipped. Models and their profiles live in
model folders (`vllm_torchtpu.models.<model>.megakernel`), which register
them explicitly.
"""

from __future__ import annotations

import importlib

BUILTINS = ("modules", "experimental")
# The packages a registry's lookups need.
TUNABLES = ("modules",)
FUSED_IO = ("experimental",)

_LOADED: set[str] = set()


def ensure_builtins(*names: str) -> None:
    """Import the named built-in packages (all of `BUILTINS` by default) once.

    A lookup only loads what its registry needs: a module-tunables lookup
    runs while `profile` itself is being imported (building
    `DEFAULT_PROFILE`).
    """
    for name in names or BUILTINS:
        if name in _LOADED:
            continue
        _LOADED.add(name)
        full = f"{__package__}.{name}"
        try:
            importlib.import_module(full)
        except ModuleNotFoundError as exc:
            if exc.name != full:
                _LOADED.discard(name)
                raise
        except BaseException:
            _LOADED.discard(name)
            raise
