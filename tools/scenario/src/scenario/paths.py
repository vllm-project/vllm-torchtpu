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
"""Key paths into scenario data, and their dotted text form.

The text form is what `--set`, sweep axes and diagnostics use:

    server.args.max-model-len         mapping keys joined by dots
    tasks.custom[0].timeout           [i] indexes a list
    server.args[profiler-config.x]    [key] for a key holding a dot

Arg names keep their dashes, so a plain segment allows letters, digits, `_`
and `-`.
"""

import re

KeyPath = tuple[str | int, ...]

_NAME = re.compile(r"[A-Za-z0-9_-]+")
_PLAIN = re.compile(r"^[A-Za-z0-9_-]+$")


class PathSyntaxError(ValueError):
    """A dotted path that does not follow the grammar above."""


def parse_path(text: str) -> KeyPath:
    """Split dotted path text into keys and list indices."""
    parts: list[str | int] = []
    i = 0
    expect_name = True
    while i < len(text):
        char = text[i]
        if char == "[":
            end = text.find("]", i + 1)
            if end == -1:
                raise PathSyntaxError(f"unclosed '[' in {text!r}")
            inner = text[i + 1 : end]
            if not inner:
                raise PathSyntaxError(f"empty [] in {text!r}")
            parts.append(int(inner) if inner.isdigit() else inner)
            i = end + 1
            expect_name = False
        elif char == ".":
            if expect_name:
                raise PathSyntaxError(f"empty segment in {text!r}")
            i += 1
            expect_name = True
        else:
            match = _NAME.match(text, i)
            if match is None or not expect_name:
                raise PathSyntaxError(f"unexpected {char!r} in {text!r}")
            parts.append(match.group())
            i = match.end()
            expect_name = False
    if not parts or expect_name:
        raise PathSyntaxError(f"{text!r} is not a dotted path")
    return tuple(parts)


def format_path(path: KeyPath) -> str:
    """The text form of a key path; the inverse of `parse_path`."""
    out = ""
    for part in path:
        if isinstance(part, int):
            out += f"[{part}]"
        elif _PLAIN.match(part) and not part.isdigit():
            out += f".{part}" if out else part
        else:
            out += f"[{part}]"
    return out


def is_prefix(prefix: KeyPath, path: KeyPath) -> bool:
    """True when `prefix` equals `path` or is one of its ancestors."""
    return path[: len(prefix)] == prefix
