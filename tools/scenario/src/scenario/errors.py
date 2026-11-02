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
"""Problems found in scenario data, and how they are reported.

An `Issue` is a problem at a key path of some data. A `Diagnostic` is the
same problem placed in a file, rendered as

    <file>:<line>: <dotted path>: <message>
      hint: <hint>

Every check collects all of its problems, so one run reports them all.
"""

import difflib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from scenario.paths import KeyPath, format_path

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    """A problem at a key path of scenario data.

    `message` reads after the dotted path ("server.multihost: ..."). A
    warning is reported but does not fail validation.
    """

    path: KeyPath
    message: str
    hint: str | None = None
    rule: str | None = None
    severity: Severity = "error"


def errors_only(issues: Iterable[Issue]) -> list[Issue]:
    return [issue for issue in issues if issue.severity == "error"]


@dataclass(frozen=True)
class Diagnostic:
    """A problem placed in a file (or a --set argument)."""

    source: str | None
    line: int | None
    path: str
    message: str
    hint: str | None = None
    rule: str | None = None
    severity: Severity = "error"

    def render(self) -> str:
        where = self.source or ""
        if self.source and self.line:
            where = f"{self.source}:{self.line}"
        parts = [where] if where else []
        if self.severity == "warning":
            parts.append("warning")
        if self.path:
            parts.append(self.path)
        text = ": ".join([*parts, self.message])
        if self.hint:
            text += f"\n  hint: {self.hint}"
        return text


def sort_diagnostics(diagnostics: Iterable[Diagnostic]) -> list[Diagnostic]:
    return sorted(diagnostics, key=lambda d: (d.source or "", d.line or 0))


class DiagnosticError(Exception):
    """One or more problems that stop resolution."""

    def __init__(self, diagnostics: Iterable[Diagnostic]) -> None:
        self.diagnostics = tuple(diagnostics)
        super().__init__(render_all(self.diagnostics))


def render_all(diagnostics: Sequence[Diagnostic]) -> str:
    return "\n".join(diagnostic.render() for diagnostic in diagnostics)


# (source, line) for a key path of the data being checked.
Locator = Callable[[KeyPath], tuple[str | None, int | None]]


def place(issues: Iterable[Issue], locate: Locator) -> list[Diagnostic]:
    """Put issues into files."""
    diagnostics = []
    for issue in issues:
        source, line = locate(issue.path)
        diagnostics.append(
            Diagnostic(
                source=source,
                line=line,
                path=format_path(issue.path),
                message=issue.message,
                hint=issue.hint,
                rule=issue.rule,
                severity=issue.severity,
            )
        )
    return diagnostics


def suggest(name: str, candidates: Iterable[str]) -> str | None:
    """The candidate closest to `name`, if any is close."""
    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


# Pydantic error types, rewritten into short messages.
_EXPECTED = {
    "int_type": "an integer",
    "int_parsing": "an integer",
    "int_from_float": "an integer",
    "float_type": "a number",
    "float_parsing": "a number",
    "bool_type": "true or false",
    "bool_parsing": "true or false",
    "string_type": "a string",
    "dict_type": "a mapping",
    "model_type": "a mapping",
    "model_attributes_type": "a mapping",
    "list_type": "a list",
    "none_required": "null",
}
_BOUNDS = {
    "greater_than": ("more than", "gt"),
    "greater_than_equal": ("at least", "ge"),
    "less_than": ("less than", "lt"),
    "less_than_equal": ("at most", "le"),
}


def _shown(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _message(error: Any) -> str:
    kind = error["type"]
    ctx = error.get("ctx") or {}
    got = f"got {_shown(error.get('input'))}"
    if kind == "missing":
        return "required"
    if kind == "extra_forbidden":
        return "unknown key"
    if kind in _EXPECTED:
        return f"expected {_EXPECTED[kind]}; {got}"
    if kind == "literal_error":
        return f"expected {ctx.get('expected')}; {got}"
    if kind in _BOUNDS:
        words, key = _BOUNDS[kind]
        return f"must be {words} {ctx.get(key)}; {got}"
    if kind == "too_short":
        return f"needs at least {ctx.get('min_length')} items; {got}"
    if kind == "too_long":
        return f"takes at most {ctx.get('max_length')} items; {got}"
    return error["msg"]


def _data_path(loc: Sequence[Any], data: Any, last_is_key: bool) -> KeyPath:
    """The part of a pydantic loc that indexes the data.

    Pydantic adds union tags and validator labels to locs; they are skipped.
    A key absent from the data is kept only at the end (a missing field).
    """
    path: list[str | int] = []
    node = data
    for index, part in enumerate(loc):
        if part == "[key]":
            break
        in_dict = isinstance(node, dict) and part in node
        in_list = isinstance(node, list) and isinstance(part, int) and part < len(node)
        if in_dict or in_list:
            node = node[part]
            path.append(part)
        elif last_is_key and index == len(loc) - 1 and isinstance(node, dict):
            path.append(part)
    return tuple(path)


def issues_from_validation_error(
    error: ValidationError, data: Any, base: KeyPath = ()
) -> list[Issue]:
    """One issue per problem pydantic found, at its key path in `data`.

    Branches of a union that fail on the same value become one issue:
    "expected a string or a mapping; got 5".
    """
    merged: dict[KeyPath, list[tuple[str, str | None]]] = {}
    for item in error.errors(include_url=False):
        kind = item["type"]
        path = _data_path(item["loc"], data, kind in ("missing", "extra_forbidden"))
        rule = kind if kind.startswith("V-") else None
        merged.setdefault((*base, *path), []).append((_message(item), rule))
    issues = []
    for path, found in merged.items():
        messages = list(dict.fromkeys(message for message, _ in found))
        rule = next((rule for _, rule in found if rule), None)
        expected = [m for m in messages if m.startswith("expected ") and "; got " in m]
        if len(expected) > 1 and len(expected) == len(messages):
            wants = [m[len("expected ") :].split("; got ")[0] for m in expected]
            got = expected[0].split("; got ", 1)[1]
            messages = [f"expected {' or '.join(wants)}; got {got}"]
        issues.extend(Issue(path, message, rule=rule) for message in messages)
    return issues
