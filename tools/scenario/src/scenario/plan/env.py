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
"""Env the tool sets: which plumbing item owns a name, and ${NAME} values."""

import re
from collections.abc import Collection, Mapping

from scenario.plan.plumbing import ITEMS
from scenario.plan.types import Value, concat

_ENV_RULES: dict[str, list[str]] = {}
for _item in ITEMS.values():
    for _name in _item.env:
        _ENV_RULES.setdefault(_name, []).append(_item.id)
_HOST_PREFIX = "SCENARIO_HOST_"
_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def env_rule(name: str, within: Collection[str] | None = None) -> str | None:
    """The plumbing item that sets env `name`, or None for scenario env.

    A name that several items set (PYTHONPATH: P-18 and P-21) goes to the
    last of them, unless `within` names the items that apply where it is.
    """
    rules = _ENV_RULES.get(name, [])
    if name.startswith(_HOST_PREFIX):
        rules = [*rules, "P-17"]
    if within is not None:
        rules = [rule for rule in rules if rule in within]
    return rules[-1] if rules else None


def substitute(text: str, provided: Mapping[str, Value]) -> Value:
    """A custom-task env value with each ${NAME} replaced by its provided value.

    `$$` is a literal `$`. V-22 has already rejected any other use of `$`.

    Raises:
        ValueError: a lone `$` or a name that is not provided.
    """
    parts: list[Value] = []
    literal = ""
    i = 0
    while i < len(text):
        if text.startswith("$$", i):
            literal += "$"
            i += 2
            continue
        if text[i] != "$":
            literal += text[i]
            i += 1
            continue
        match = _REFERENCE.match(text, i)
        if match is None or match.group(1) not in provided:
            raise ValueError(f"cannot substitute {text[i:]!r}")
        parts += [literal, provided[match.group(1)]]
        literal = ""
        i = match.end()
    parts.append(literal)
    return concat(*parts)
