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
"""Hook through which blocks reach the optional fused pool I/O helpers.

`GQABlock` and `GDNBlock` never import `experimental.fused_pool_io`; when fused
pool I/O is selected (`CacheExec.FUSED`) they call `fused_io()`, which returns
the implementation that `experimental/fused_pool_io.py` registers on import
(the package `__init__` imports `experimental`). Without a registered
implementation, `"auto"` fused selection falls back to split pool I/O
(`fused_io_available()`), and an explicit fused build fails with
`MegakernelConfigError`.

This module depends only on `errors`, so importing the experimental helpers does
not load any block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from .errors import MegakernelConfigError

if TYPE_CHECKING:
    from .exec_spec import ExecSpec
    from .spec import StateDecl


class FusedPoolIO(Protocol):
    """Fused pool I/O helpers called by `GQABlock` / `GDNBlock` (`CacheExec.FUSED`)."""

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


_REGISTRY: dict[str, FusedPoolIO] = {}
_KEY = "fused_pool_io"


def register_fused_io(impl: FusedPoolIO) -> None:
    """Install the fused pool I/O helpers (`gqa_fused_*`, `gdn_fused_*`)."""
    _REGISTRY[_KEY] = impl


def fused_io_available() -> bool:
    """Whether a fused pool I/O implementation is registered."""
    return _KEY in _REGISTRY


def fused_io() -> FusedPoolIO:
    """Return the registered fused pool I/O helpers (used under `CacheExec.FUSED`)."""
    impl = _REGISTRY.get(_KEY)
    if impl is None:
        raise MegakernelConfigError(
            "fused pool I/O was requested, but no fused pool I/O implementation "
            "is registered (the megakernel experimental package is not loaded)"
        )
    return impl


__all__ = ["FusedPoolIO", "fused_io", "fused_io_available", "register_fused_io"]
