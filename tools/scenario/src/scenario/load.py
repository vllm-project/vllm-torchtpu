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
"""Read scenario YAML into plain data plus the line of every key.

Round-trip loading keeps the line of each key and list item, which every
diagnostic needs. The YAML subset is deliberately small: no tags, no merge
keys (`extends` is the one way to share settings), no dates, string keys,
and every duplicate key is an error. Anchors and aliases are expanded.
"""

import pathlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import MarkedYAMLError
from ruamel.yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from scenario.errors import Diagnostic, DiagnosticError
from scenario.paths import KeyPath

_CORE = "tag:yaml.org,2002:"
_ALLOWED_TAGS = {
    f"{_CORE}{name}" for name in ("str", "int", "float", "bool", "null", "map", "seq")
}


@dataclass(frozen=True)
class LoadedFile:
    """A YAML file as plain data, with the 1-based line of every key path."""

    source: str
    data: Any
    lines: Mapping[KeyPath, int]

    def line(self, path: KeyPath) -> int:
        """The line of `path`, or of its nearest ancestor present in the file."""
        for end in range(len(path), -1, -1):
            if path[:end] in self.lines:
                return self.lines[path[:end]]
        return 1


def _tag(node: Node) -> str:
    tag = node.tag
    return tag if isinstance(tag, str) else str(tag.value)


def _node_problems(node: Node, source: str, active: set[int]) -> Iterator[Diagnostic]:
    if id(node) in active:
        return
    line = node.start_mark.line + 1
    tag = _tag(node)
    if tag == f"{_CORE}timestamp":
        yield Diagnostic(source, line, "", "dates are not supported; quote the value")
    elif tag not in _ALLOWED_TAGS:
        yield Diagnostic(source, line, "", f"YAML tags are not allowed ({tag})")
    active.add(id(node))
    if isinstance(node, MappingNode):
        seen: dict[str, int] = {}
        for key, value in node.value:
            key_line = key.start_mark.line + 1
            if _tag(key) == f"{_CORE}merge":
                yield Diagnostic(
                    source,
                    key_line,
                    "",
                    "YAML merge keys (<<) are not supported; share settings with "
                    "extends",
                )
            elif not isinstance(key, ScalarNode) or _tag(key) != f"{_CORE}str":
                yield Diagnostic(
                    source,
                    key_line,
                    "",
                    f"keys are strings; quote {getattr(key, 'value', key)!r}",
                )
            elif key.value in seen:
                yield Diagnostic(
                    source,
                    key_line,
                    "",
                    f"duplicate key {key.value} (first at line {seen[key.value]})",
                )
            else:
                seen[key.value] = key_line
            yield from _node_problems(value, source, active)
    elif isinstance(node, SequenceNode):
        for item in node.value:
            yield from _node_problems(item, source, active)
    active.discard(id(node))


def _plain(value: Any, path: KeyPath, lines: dict[KeyPath, int]) -> Any:
    if isinstance(value, CommentedMap):
        out = {}
        for key in value:
            lines[(*path, key)] = value.lc.key(key)[0] + 1
            out[str(key)] = _plain(value[key], (*path, key), lines)
        return out
    if isinstance(value, CommentedSeq):
        items = []
        for index, item in enumerate(value):
            lines[(*path, index)] = value.lc.item(index)[0] + 1
            items.append(_plain(item, (*path, index), lines))
        return items
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


def load_yaml(text: str, source: str) -> LoadedFile:
    """Parse one YAML document.

    Raises:
        DiagnosticError: every syntax, tag, key and duplicate problem found.
    """
    yaml = YAML(typ="rt")
    try:
        node = yaml.compose(text)
    except MarkedYAMLError as error:
        mark = error.problem_mark or error.context_mark
        line = mark.line + 1 if mark is not None else None
        problem = "; ".join(part for part in (error.context, error.problem) if part)
        raise DiagnosticError(
            [Diagnostic(source, line, "", f"YAML syntax: {problem}")]
        ) from None
    if node is None:
        raise DiagnosticError([Diagnostic(source, 1, "", "the file is empty")])
    problems = list(_node_problems(node, source, set()))
    if problems:
        raise DiagnosticError(problems)
    if not isinstance(node, MappingNode):
        raise DiagnosticError(
            [Diagnostic(source, 1, "", "the top level must be a mapping")]
        )
    lines: dict[KeyPath, int] = {(): 1}
    data = _plain(yaml.load(text), (), lines)
    return LoadedFile(source=source, data=data, lines=lines)


def load_file(path: pathlib.Path, source: str | None = None) -> LoadedFile:
    """Load `path`; `source` is how diagnostics name it (default: the path)."""
    return load_yaml(path.read_text(), source or str(path))
