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
"""Hook through which modules reach the optional fused pool I/O helpers.

Mixer modules never import the fused pool I/O implementation directly; when
fused pool I/O is selected (`CacheExec.FUSED`) they call `fused_io()`, which
returns the implementation registered via `register_fused_io()` (loaded on the
first lookup, `_builtins.ensure_builtins`). Callers that resolve `"auto"` fused
pool I/O policy query `fused_io_available()` to fall back to split pool I/O when
no implementation is registered, whereas calling `fused_io()` directly without a
registered implementation raises `MegakernelConfigError`.

This module depends only on `errors` and `_builtins` (both stdlib at import), so
importing it does not load any mixer module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from ._builtins import FUSED_IO, ensure_builtins
from .errors import MegakernelConfigError

if TYPE_CHECKING:
    from .exec_spec import ExecSpec
    from .spec import StateDecl


class FusedPoolIO(Protocol):
    """Fused pool I/O helper protocol (`CacheExec.FUSED`)."""

    def gqa_fused_states(self, ex: ExecSpec) -> list[StateDecl]: ...

    def gqa_fused_outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]: ...

    def gqa_fused_scratch(self, ex: ExecSpec, hd: int) -> dict[str, Any]: ...

    def gqa_fused_init_buffers(self, ctx: Any) -> None: ...

    def gqa_fused_kvrd_issue(self, ctx: Any, a_cur: Any) -> None: ...

    def gqa_fused_kv_rmw(self, ctx: Any, ap: Any, a_cur: Any) -> None: ...

    def gqa_fused_epilogue(self, ctx: Any) -> None: ...

    def gdn_fused_scratch(self, ex: ExecSpec, gp: Any, plan: Any) -> dict[str, Any]: ...

    def gdn_fused_init_buffers(self, ctx: Any, gp: Any) -> None: ...

    def gdn_fused_emit(self, ctx: Any, gp: Any, o: Any, slot: Any) -> Any: ...

    def gdn_fused_epilogue(self, ctx: Any, gp: Any) -> None: ...


_IMPL: FusedPoolIO | None = None


def register_fused_io(impl: FusedPoolIO) -> None:
    """Install the fused pool I/O helpers (`gqa_fused_*`, `gdn_fused_*`)."""
    global _IMPL
    _IMPL = impl


def fused_io_available() -> bool:
    """Whether a fused pool I/O implementation is registered."""
    ensure_builtins(*FUSED_IO)
    return _IMPL is not None


def fused_io() -> FusedPoolIO:
    """Return the registered fused pool I/O helpers (used under `CacheExec.FUSED`)."""
    ensure_builtins(*FUSED_IO)
    if _IMPL is None:
        raise MegakernelConfigError(
            "fused pool I/O was requested, but no fused pool I/O implementation "
            "is registered (the megakernel experimental package is not loaded)"
        )
    return _IMPL


__all__ = ["FusedPoolIO", "fused_io", "fused_io_available", "register_fused_io"]
