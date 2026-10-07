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
"""Merge layers of scenario data, remembering where every value came from.

Rules: mappings merge per key, lists and scalars replace, null deletes (a
missing key stays missing). Inside `server.roles.<r>.args` and `.env` a null
is kept: it removes the inherited server-level key for that role when the
role's args are combined with server.args.
"""

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from scenario.paths import KeyPath, format_path, is_prefix


@dataclass(frozen=True)
class Origin:
    """Where a value was set: a layer, and the file and line (or --set text)."""

    layer: str
    source: str | None = None
    line: int | None = None

    def render(self) -> str:
        if self.source and self.line:
            return f"{self.source}:{self.line}"
        return self.source or self.layer


TOOL = Origin("tool")

OriginOf = Callable[[KeyPath], Origin]


@dataclass
class Provenance:
    """Origins of the merged data.

    `nodes` has the latest origin of every key path present, mappings and
    list contents included; `leaves` that of every scalar, list and empty
    mapping. `superseded` keeps the earlier origins of a leaf, oldest first,
    and `deleted` the keys a null removed.
    """

    nodes: dict[KeyPath, Origin] = field(default_factory=dict)
    leaves: dict[KeyPath, Origin] = field(default_factory=dict)
    superseded: dict[KeyPath, list[Origin]] = field(default_factory=dict)
    deleted: dict[KeyPath, Origin] = field(default_factory=dict)

    def origin(self, path: KeyPath) -> Origin:
        """The origin of a leaf; `tool` for a value only a model default sets."""
        return self.leaves.get(path, TOOL)

    def locate(self, path: KeyPath) -> Origin:
        """The origin of `path`, or of its nearest ancestor that is present."""
        for end in range(len(path), -1, -1):
            if path[:end] in self.nodes:
                return self.nodes[path[:end]]
        return TOOL

    def forget(self, path: KeyPath) -> None:
        """Drop every record at or under `path`."""
        for table in (self.nodes, self.leaves, self.superseded, self.deleted):
            for key in [key for key in table if is_prefix(path, key)]:
                del table[key]

    def copy(self) -> "Provenance":
        return Provenance(
            dict(self.nodes),
            dict(self.leaves),
            {key: list(value) for key, value in self.superseded.items()},
            dict(self.deleted),
        )


def keeps_null(path: KeyPath) -> bool:
    """Whether a null at `path` is a value rather than a deletion."""
    return (
        len(path) >= 5
        and path[:2] == ("server", "roles")
        and path[3] in ("args", "env")
    )


def _record_tree(
    prov: Provenance, path: KeyPath, value: Any, origin_of: OriginOf
) -> None:
    prov.nodes[path] = origin_of(path)
    if isinstance(value, dict):
        for key, item in value.items():
            _record_tree(prov, (*path, key), item, origin_of)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _record_tree(prov, (*path, index), item, origin_of)


def _replace(acc: dict, prov: Provenance, path: KeyPath) -> None:
    """Prepare `acc[path[-1]]` to be replaced: its origin becomes superseded."""
    history = prov.superseded.get(path, [])
    previous = prov.leaves.get(path)
    prov.forget(path)
    if previous is not None:
        prov.superseded[path] = [*history, previous]
    acc.pop(path[-1], None)


def merge_into(
    acc: dict,
    prov: Provenance,
    layer: dict,
    origin_of: OriginOf,
    base: KeyPath = (),
) -> None:
    """Merge `layer` over `acc` (both at key path `base`), in place."""
    for key, value in layer.items():
        path = (*base, key)
        origin = origin_of(path)
        if value is None and not keeps_null(path):
            if key in acc:
                prov.forget(path)
                del acc[key]
                prov.deleted[path] = origin
            continue
        prov.deleted.pop(path, None)
        lower = acc.get(key)
        if isinstance(value, dict):
            if not isinstance(lower, dict):
                if key in acc:
                    _replace(acc, prov, path)
                acc[key] = {}
            prov.nodes[path] = origin
            prov.leaves.pop(path, None)
            merge_into(acc[key], prov, value, origin_of, path)
            if not acc[key]:
                prov.leaves[path] = origin
        else:
            if key in acc:
                _replace(acc, prov, path)
            acc[key] = copy.deepcopy(value)
            prov.leaves[path] = origin
            _record_tree(prov, path, value, origin_of)


class AssignError(ValueError):
    """A path that cannot be assigned in the data."""


def assign(
    acc: dict, prov: Provenance, path: KeyPath, value: Any, origin: Origin
) -> None:
    """Set `path` to `value` as a one-key layer; missing mappings are created.

    A `[i]` segment indexes an existing list item. A list counts as one leaf,
    so assigning inside one makes the whole list's origin `origin`.

    Raises:
        AssignError: the path crosses a scalar, or indexes past a list's end.
    """
    node: Any = acc
    for depth, key in enumerate(path[:-1]):
        walked = path[: depth + 1]
        if isinstance(key, int):
            if not isinstance(node, list) or not 0 <= key < len(node):
                raise AssignError(f"{format_path(walked[:-1])} has no item [{key}]")
            node = node[key]
            continue
        if not isinstance(node, dict):
            raise AssignError(f"{format_path(walked[:-1])} is not a mapping")
        if not isinstance(node.get(key), dict | list):
            if key in node:
                _replace(node, prov, walked)
            node[key] = {}
            prov.nodes[walked] = origin
        node = node[key]
    last = path[-1]
    list_path = next(
        (path[:depth] for depth in range(len(path)) if isinstance(path[depth], int)),
        None,
    )
    if list_path is None:
        merge_into(node, prov, {last: value}, lambda _: origin, path[:-1])
        return
    if isinstance(last, int):
        if not isinstance(node, list) or not 0 <= last < len(node):
            raise AssignError(f"{format_path(path[:-1])} has no item [{last}]")
        if value is None:
            raise AssignError("null cannot delete a list item")
        node[last] = copy.deepcopy(value)
    elif not isinstance(node, dict):
        raise AssignError(f"{format_path(path[:-1])} is not a mapping")
    elif value is None:
        node.pop(last, None)
    else:
        node[last] = copy.deepcopy(value)
    previous = prov.leaves.get(list_path)
    if previous is not None:
        prov.superseded.setdefault(list_path, []).append(previous)
    prov.leaves[list_path] = origin
    if value is None:
        prov.forget(path)
    else:
        _record_tree(prov, path, value, lambda _: origin)
