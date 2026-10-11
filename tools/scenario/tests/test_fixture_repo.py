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
"""The fixture repository's scenario files are block-style YAML, as real ones are."""

import pathlib

import pytest
from text_checks import flow_mapping_lines

REPO = pathlib.Path(__file__).resolve().parent / "data" / "repo"
SCENARIO_FILES = sorted((REPO / "scenarios").rglob("*.yaml"))


def test_flow_mapping_lines() -> None:
    text = 'a: {b: 1}\nc: [1, [2]]\nd: "{x}"\ne:\n  - {f: {g: 2}}\n  - h: 1\n'
    assert flow_mapping_lines(text) == [1, 5, 5]


@pytest.mark.parametrize(
    "path", SCENARIO_FILES, ids=[str(p.relative_to(REPO)) for p in SCENARIO_FILES]
)
def test_scenario_files_use_block_mappings(path: pathlib.Path) -> None:
    """Scenario files write mappings as blocks; short scalar lists may stay inline."""
    lines = flow_mapping_lines(path.read_text())
    assert not lines, f"flow mappings at lines {lines}; write them as block mappings"
