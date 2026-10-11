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
"""Checks on file text: banned strings.

Like plan_fixtures.py it is imported by name, so checks over a repository's
own scenarios can use it too.
"""

import pathlib


def fleet_strings(text: str, banned: tuple[str, ...]) -> list[str]:
    """`line N: <string>` for every banned string, case-insensitively."""
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        lowered = line.lower()
        found += [f"line {number}: {word}" for word in banned if word in lowered]
    return found


def text_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Text files under `root`, leaving out dot-directories and bytecode."""
    files = []
    for directory, subdirs, names in root.walk():
        subdirs[:] = [
            d for d in subdirs if not d.startswith(".") and d != "__pycache__"
        ]
        for name in names:
            path = directory / name
            try:
                path.read_text()
            except UnicodeDecodeError:
                continue
            files.append(path)
    return sorted(files)
