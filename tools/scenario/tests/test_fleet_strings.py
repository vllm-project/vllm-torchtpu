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
"""No fleet strings where they do not belong (R-TEST-5).

Scenarios state needs and the tool stays fleet-neutral; the fleet's paths,
CI system and launcher belong to its target. The banned strings are built
from parts so this file does not match itself.
"""

import pathlib

import pytest

TOOL = pathlib.Path(__file__).resolve().parents[1]
SCENARIOS = TOOL.parents[1] / "scenarios"

CI = "build" + "kite"
LAUNCHER = "/opt/" + "launcher"
FLEET_DOMAIN = "tpu-ci" + ".google.com"
FLEET_CACHE = "/" + "cache/"

TOOL_BANNED = (CI, LAUNCHER, FLEET_DOMAIN)
SCENARIO_BANNED = (FLEET_CACHE, CI, FLEET_DOMAIN)


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


def test_tool_has_no_fleet_strings() -> None:
    files = text_files(TOOL)
    assert any(path.name == "cli.py" for path in files)
    found = {
        str(path.relative_to(TOOL)): hits
        for path in files
        if (hits := fleet_strings(path.read_text(), TOOL_BANNED))
    }
    assert found == {}


def test_scenarios_have_no_fleet_strings() -> None:
    if not SCENARIOS.is_dir():
        pytest.skip("no scenarios/ yet")
    found = {
        str(path.relative_to(SCENARIOS)): hits
        for path in text_files(SCENARIOS)
        if (hits := fleet_strings(path.read_text(), SCENARIO_BANNED))
    }
    assert found == {}


@pytest.mark.parametrize(
    ("text", "banned", "hit"),
    [
        (f"cmd: [{CI}-agent, artifact, upload]", TOOL_BANNED, CI),
        (f"exec {LAUNCHER}/launch --manifest m.yaml", TOOL_BANNED, LAUNCHER),
        (f"{FLEET_DOMAIN}/defaults: standard", TOOL_BANNED, FLEET_DOMAIN),
        (f"VLLM_CACHE_ROOT: {FLEET_CACHE}jax/vllm", SCENARIO_BANNED, FLEET_CACHE),
        ("description: from Build" + "kite build 1003", SCENARIO_BANNED, CI),
    ],
)
def test_seeded_violations_are_caught(
    text: str, banned: tuple[str, ...], hit: str
) -> None:
    assert fleet_strings(f"ok\n{text}\n", banned) == [f"line 2: {hit}"]


def test_clean_text_passes() -> None:
    assert (
        fleet_strings("load-format: runai_streamer\nscratch: 64Gi\n", SCENARIO_BANNED)
        == []
    )
