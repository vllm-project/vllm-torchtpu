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
"""Per-block tunables and debug-options registries and immutable option maps."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from typing import Any, Literal

NumericsClass = Literal["exact", "approx"]

# Block name -> its `Tunables` class (registered via `register_tunables`).
_TUNABLES: dict[str, type] = {}
# Block name -> its `Debug` class (registered via `register_debug`).
_DEBUG: dict[str, type] = {}


def register_tunables(name: str, cls: type) -> None:
    """Register `cls` as the `Tunables` of blocks named `name`.

    `cls` must be a frozen dataclass whose `NUMERICS` class attribute maps
    every field to `"exact"` or `"approx"`. Re-registering the same class is a
    no-op; a different class for a registered name is an error.
    """
    if not (dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen):  # type: ignore[attr-defined]
        raise TypeError(f"{cls.__name__} must be a frozen dataclass")
    fields = {f.name for f in dataclasses.fields(cls)}
    numerics = dict(getattr(cls, "NUMERICS", {}))
    if set(numerics) != fields:
        raise TypeError(
            f"{cls.__name__}.NUMERICS must cover exactly its fields: missing "
            f"{sorted(fields - set(numerics))}, "
            f"unknown {sorted(set(numerics) - fields)}"
        )
    bad = {k: v for k, v in numerics.items() if v not in ("exact", "approx")}
    if bad:
        raise TypeError(f"{cls.__name__}.NUMERICS has invalid classes {bad}")
    prev = _TUNABLES.get(name)
    if prev is not None and prev is not cls:
        raise ValueError(
            f"block {name!r} already registered tunables {prev.__name__}, "
            f"not {cls.__name__}"
        )
    _TUNABLES[name] = cls


def tunables_class(name: str) -> type:
    """The registered `Tunables` class of blocks named `name`."""
    try:
        return _TUNABLES[name]
    except KeyError:
        raise KeyError(
            f"no tunables registered for block {name!r}; known: {sorted(_TUNABLES)}"
        ) from None


def registered_tunables() -> dict[str, type]:
    """All registered `{block name: Tunables class}` (a copy)."""
    return dict(_TUNABLES)


def register_debug(name: str, cls: type) -> None:
    """Register `cls` as the `Debug` options class of blocks named `name`.

    `cls` must be a frozen dataclass. Re-registering the same class is a no-op;
    a different class for a registered name is an error.
    """
    if not (dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen):  # type: ignore[attr-defined]
        raise TypeError(f"{cls.__name__} must be a frozen dataclass")
    prev = _DEBUG.get(name)
    if prev is not None and prev is not cls:
        raise ValueError(
            f"block {name!r} already registered Debug class {prev.__name__}, "
            f"not {cls.__name__}"
        )
    _DEBUG[name] = cls


def debug_class(name: str) -> type:
    """The registered `Debug` class of blocks named `name`."""
    try:
        return _DEBUG[name]
    except KeyError:
        raise KeyError(
            f"no Debug class registered for block {name!r}; known: {sorted(_DEBUG)}"
        ) from None


def registered_debug() -> dict[str, type]:
    """All registered `{block name: Debug class}` (a copy)."""
    return dict(_DEBUG)


class BlockOptionsMap(Mapping[str, Any]):
    """Immutable, hashable `{block name: frozen dataclass}` mapping.

    Entries equal to their class defaults are dropped, so maps that leave a
    block at its defaults compare equal however they were built.
    """

    __slots__ = ("_map",)
    # Owner and fix hint shown in error messages.
    _WHAT = "DebugOptions.blocks"
    _HINT = ""

    def __init__(self, items: Mapping[str, Any] | None = None) -> None:
        src = dict(items or {})
        entries: dict[str, Any] = {}
        for name in sorted(src):
            val = src[name]
            if not (
                dataclasses.is_dataclass(val)
                and not isinstance(val, type)
                and val.__dataclass_params__.frozen  # type: ignore[attr-defined]
            ):
                raise TypeError(
                    f"{self._WHAT}[{name!r}] must be a frozen dataclass instance, "
                    f"got {type(val).__name__}{self._HINT}"
                )
            self._check(name, val)
            if val != type(val)():
                entries[name] = val
        # Sorted insertion order keeps iteration, repr and hash canonical.
        self._map = entries

    def _check(self, name: str, val: Any) -> None:
        """Hook: raise if `val` is not a valid entry for `name`."""

    def __getitem__(self, key: str) -> Any:
        return self._map[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._map)

    def __len__(self) -> int:
        return len(self._map)

    def __hash__(self) -> int:
        # Not cached: a hash pickled with the map would be stale in another
        # process (str hashes are salted per process).
        return hash(tuple(self._map.items()))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BlockOptionsMap):
            return self._map == other._map
        return isinstance(other, Mapping) and self._map == dict(other)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._map!r})"


class TunablesMap(BlockOptionsMap):
    """`{block name: Tunables}` (`Tiles.blocks`): a `BlockOptionsMap` whose
    entries must be instances of the block's registered `Tunables` class."""

    __slots__ = ()
    _WHAT = "Tiles.blocks"
    _HINT = " (use Tiles.with_blocks for field dicts)"

    def _check(self, name: str, val: Any) -> None:
        # An unregistered (e.g. misspelled) block name raises `KeyError`.
        cls = tunables_class(name)
        if not isinstance(val, cls):
            raise TypeError(
                f"Tiles.blocks[{name!r}] must be a {cls.__name__}, got "
                f"{type(val).__name__}{self._HINT}"
            )


class DebugBlocksMap(BlockOptionsMap):
    """`{block name: Debug}` (`DebugOptions.blocks`): a `BlockOptionsMap` whose
    entries must name a block with a registered `Debug` class and be an
    instance of that class."""

    __slots__ = ()
    _WHAT = "DebugOptions.blocks"

    def _check(self, name: str, val: Any) -> None:
        cls = debug_class(name)
        if not isinstance(val, cls):
            raise TypeError(
                f"DebugOptions.blocks[{name!r}] must be a {cls.__name__}, got "
                f"{type(val).__name__}"
            )


__all__ = [
    "BlockOptionsMap",
    "DebugBlocksMap",
    "NumericsClass",
    "TunablesMap",
    "debug_class",
    "register_debug",
    "register_tunables",
    "registered_debug",
    "registered_tunables",
    "tunables_class",
]
