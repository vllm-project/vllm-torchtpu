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
"""`--set path=value` overrides.

The path is a dotted key path (`server.args.max-model-len`, `[i]` for a list
item, `[key]` for a key with a dot). The value is one YAML value: `4096`,
`0.9`, `true`, `[1, 2]`, `{a: 1}`; `null` deletes; anything else is a string,
and quoting forces one (`'"8"'`). Every override changes server.* or
tasks.*, so a run with one is an experiment and skips its gates.
"""

from dataclasses import dataclass
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from scenario.paths import KeyPath, PathSyntaxError, parse_path

ROOTS = ("server", "tasks")


@dataclass(frozen=True)
class Override:
    index: int
    text: str
    path: KeyPath
    value: Any

    @property
    def source(self) -> str:
        """How diagnostics name this override."""
        return f"--set[{self.index}] {self.text}"


class OverrideError(ValueError):
    """An override that cannot be parsed or is not allowed (V-30)."""


_PLAIN = (str, int, float, bool, list, dict, type(None))


def parse_value(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise OverrideError(
            'the value is empty; write null to delete the key, or "" for an '
            "empty string"
        )
    flow = stripped[0] in "[{"
    try:
        value = YAML(typ="safe", pure=True).load(stripped)
    except YAMLError as error:
        if flow:
            raise OverrideError(f"not a YAML value: {error}") from None
        return stripped
    if isinstance(value, list | dict) and not flow:
        return stripped
    if not isinstance(value, _PLAIN):
        return stripped
    return value


def parse_override(text: str, index: int) -> Override:
    """Parse one `path=value`.

    Raises:
        OverrideError: bad syntax, a path outside server.* and tasks.*, or a
            path through a baseline.
    """
    path_text, sep, value_text = text.partition("=")
    if not sep:
        raise OverrideError("expected path=value, e.g. server.args.max-model-len=4096")
    try:
        path = parse_path(path_text.strip())
    except PathSyntaxError as error:
        raise OverrideError(str(error)) from None
    if len(path) < 2 or path[0] not in ROOTS:
        raise OverrideError("--set changes server.* or tasks.* only")
    if "baseline" in path or "baselines" in path:
        raise OverrideError("baselines change only through scenario calibrate --write")
    return Override(index=index, text=text, path=path, value=parse_value(value_text))


def sparse_layer(path: KeyPath, value: Any) -> Any:
    """The override as layer data, for the key check.

    A `[i]` segment becomes a list holding the rest at position i.
    """
    node: Any = value
    for key in reversed(path):
        node = [None] * key + [node] if isinstance(key, int) else {key: node}
    return node
