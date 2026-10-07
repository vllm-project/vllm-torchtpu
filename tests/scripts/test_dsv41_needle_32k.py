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
"""CPU tests for scripts/dsv41_bench/needle_32k.py prompt construction."""

import importlib.util
import pathlib
import random
import sys

import pytest

pytestmark = pytest.mark.cpu_test

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "dsv41_bench"
    / "needle_32k.py"
)
_spec = importlib.util.spec_from_file_location("needle_32k", _SCRIPT)
needle = importlib.util.module_from_spec(_spec)
sys.modules["needle_32k"] = needle
_spec.loader.exec_module(needle)


@pytest.mark.parametrize("depth", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_needle_inserted_once_at_depth(depth):
    sentences = needle._filler(random.Random(3), 200)
    marker = "The special magic number for Porto is 1234567."
    prompt = needle.build_prompt(sentences, depth, marker, "Q?")
    assert prompt.count(marker) == 1
    assert prompt.endswith("Q?")
    body = prompt.split("<document>\n", 1)[1].split("\n</document>", 1)[0]
    before = body.split(marker, 1)[0]
    # Every filler sentence ends with "."; count them ahead of the needle.
    assert before.count(".") == round(depth * len(sentences))


def test_filler_is_deterministic_and_key_free():
    a = needle._filler(random.Random(7), 50)
    assert a == needle._filler(random.Random(7), 50)
    assert all("magic number" not in s for s in a)


def test_bad_depth_rejected():
    with pytest.raises(ValueError):
        needle.build_prompt(["x."], 1.5, "n", "q")
