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
"""No fleet strings in the tool (R-TEST-5).

The tool stays fleet-neutral; the fleet's paths, CI system and launcher
belong to its target. The banned strings are built from parts so this file
does not match itself.
"""

import pathlib

import pytest
from text_checks import fleet_strings, text_files

TOOL = pathlib.Path(__file__).resolve().parents[1]

CI = "build" + "kite"
LAUNCHER = "/opt/" + "launcher"
FLEET_DOMAIN = "tpu-ci" + ".google.com"

TOOL_BANNED = (CI, LAUNCHER, FLEET_DOMAIN)
# The rule keeps fleet logic out of the tool's code and data; its README may
# say where the fleet's own pieces live.
TOOL_DOCS = {"README.md"}


def test_tool_has_no_fleet_strings() -> None:
    files = text_files(TOOL)
    assert any(path.name == "cli.py" for path in files)
    found = {
        str(path.relative_to(TOOL)): hits
        for path in files
        if path.relative_to(TOOL).as_posix() not in TOOL_DOCS
        and (hits := fleet_strings(path.read_text(), TOOL_BANNED))
    }
    assert found == {}


@pytest.mark.parametrize(
    ("text", "hit"),
    [
        (f"cmd: [{CI}-agent, artifact, upload]", CI),
        (f"exec {LAUNCHER}/launch --manifest m.yaml", LAUNCHER),
        (f"{FLEET_DOMAIN}/defaults: standard", FLEET_DOMAIN),
    ],
)
def test_seeded_violations_are_caught(text: str, hit: str) -> None:
    assert fleet_strings(f"ok\n{text}\n", TOOL_BANNED) == [f"line 2: {hit}"]
