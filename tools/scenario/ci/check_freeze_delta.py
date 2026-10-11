#!/usr/bin/env python3
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
"""Fail an image build whose scenario install moved any other package.

    check_freeze_delta.py BEFORE AFTER

BEFORE and AFTER are `uv pip freeze` output from either side of installing
tools/scenario into an image. The install may add the tool and the one
dependency such images lack, ruamel.yaml (with ruamel.yaml.clib). It may not
remove a package or change any installed version, so vLLM's resolution
still decides every package the image shares with it. Standard library only:
it runs inside the image build.
"""

import re
import sys

TOOL = "vllm-torchtpu-scenario"
MAY_ADD = frozenset({TOOL, "ruamel-yaml", "ruamel-yaml-clib"})


def normalize(name: str) -> str:
    """A distribution name as PEP 503 compares it."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse(text: str) -> dict[str, str]:
    """name -> version (or the URL of a direct reference) per freeze line.

    Raises:
        ValueError: a line that is neither `name==version` nor `name @ url`.
    """
    found = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        name, sep, version = line.partition(" @ ")
        if not sep:
            name, sep, version = line.partition("==")
        if not sep or not name.strip():
            raise ValueError(f"not a freeze line: {line!r}")
        found[normalize(name.strip())] = version.strip()
    return found


def pin(name: str, version: str) -> str:
    return f"{name} @ {version}" if "://" in version else f"{name}=={version}"


def problems(before: dict[str, str], after: dict[str, str]) -> list[str]:
    found = []
    for name in sorted(before):
        if name not in after:
            found.append(f"removed {pin(name, before[name])}")
        elif after[name] != before[name]:
            found.append(f"changed {name} from {before[name]} to {after[name]}")
    for name in sorted(set(after) - set(before) - MAY_ADD):
        found.append(
            f"added {pin(name, after[name])}, which only vLLM's resolution may"
        )
    if TOOL not in after:
        found.append(f"{TOOL} is not installed")
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_freeze_delta.py BEFORE AFTER", file=sys.stderr)
        return 2
    with open(argv[0]) as before, open(argv[1]) as after:
        old, new = parse(before.read()), parse(after.read())
    found = problems(old, new)
    for problem in found:
        print(f"error: installing the scenario tool {problem}", file=sys.stderr)
    if found:
        return 1
    added = ", ".join(pin(name, new[name]) for name in sorted(set(new) - set(old)))
    print(f"scenario tool installed; added {added}; no other package changed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
