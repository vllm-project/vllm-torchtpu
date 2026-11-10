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
"""Tests for the registry of --additional-config sections
(src/vllm_torchtpu/additional_config_sections.py)."""

from __future__ import annotations

import pathlib

import pytest

import vllm_torchtpu
from vllm_torchtpu import additional_config_sections as sections
from vllm_torchtpu.models.qwen3_5 import settings

PACKAGE = pathlib.Path(vllm_torchtpu.__file__).parent


def test_the_qwen3_5_section_is_known():
    assert settings.SECTION in sections.KNOWN_SECTIONS


@pytest.mark.parametrize("section", sorted(sections.KNOWN_SECTIONS))
def test_every_known_section_has_a_reader(section):
    """A listed section must be read somewhere in the package, or the list
    would accept a key that changes nothing."""
    needle = f'"{section}"'
    readers = [
        p
        for p in PACKAGE.rglob("*.py")
        if p.name != "additional_config_sections.py" and needle in p.read_text()
    ]
    assert readers, f"no module of the package reads {needle}"


@pytest.mark.parametrize(
    "key,owned",
    [
        ("qwen3_5", True),
        ("tpu_scheduler", True),
        ("tpu_", True),
        ("qwen3_5_host", False),
        ("megakernel", False),
        ("abort_on_preemption", False),
        ("TPU_SCHEDULER", False),
    ],
)
def test_namespaces(key, owned):
    assert sections.is_owned(key) is owned


def test_unknown_sections_lists_only_unread_owned_keys():
    config = {"tpu_b": 1, "qwen3_5": {}, "tpu_a": 2, "megakernel": {}, 3: "x"}
    assert sections.unknown_sections(config) == ["tpu_a", "tpu_b"]
    assert sections.unknown_sections(None) == []
    assert sections.unknown_sections("not a mapping") == []


def test_check_names_the_keys_and_the_known_sections():
    with pytest.raises(ValueError) as e:
        sections.check({"tpu_zz": {}})
    assert "['tpu_zz']" in str(e.value)
    assert str(sorted(sections.KNOWN_SECTIONS)) in str(e.value)
    sections.check({"qwen3_5": {}, "other": 1})
