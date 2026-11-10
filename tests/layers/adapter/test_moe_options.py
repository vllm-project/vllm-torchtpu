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
"""Parsing of ``additional_config["tpu_moe"]``."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.adapter import moe_options as mo

pytestmark = pytest.mark.cpu_test


def _config(additional_config):
    return SimpleNamespace(additional_config=additional_config)


@pytest.mark.parametrize(
    "additional_config",
    [None, {}, {"tpu_moe": None}, {"tpu_moe": {}}, {"other": 1}],
    ids=["none", "empty", "null_section", "empty_section", "other_section"],
)
def test_defaults(additional_config):
    assert mo.get(_config(additional_config)) == mo.MoeOptions()
    assert mo.MoeOptions().lowtok_fused_fp8_dispatch is True


def test_the_option_is_read():
    options = mo.get(_config({"tpu_moe": {"lowtok_fused_fp8_dispatch": False}}))
    assert options.lowtok_fused_fp8_dispatch is False


@pytest.mark.parametrize(
    "section,match",
    [
        (True, "must be a JSON object"),
        ({"lowtok_fused_fp8_dispatch": 0}, "must be a bool"),
        ({"lowtok_fused_fp8_dispatch": "false"}, "must be a bool"),
        ({"lowtok_fused_dispatch_form": "flat"}, "unknown keys"),
    ],
    ids=["not_an_object", "int_for_bool", "string_for_bool", "unknown_key"],
)
def test_malformed_sections_are_rejected(section, match):
    with pytest.raises(ValueError, match=match):
        mo.get(_config({"tpu_moe": section}))


def test_options_are_frozen():
    with pytest.raises(AttributeError):
        mo.MoeOptions().lowtok_fused_fp8_dispatch = False  # type: ignore[misc]


def test_the_section_is_registered_and_hashed():
    """A tpu_* section the tree does not list is refused at engine start, and
    the option changes compiled programs, so the section is in the
    compile-cache key."""
    from vllm_torchtpu import additional_config_sections as sections
    from vllm_torchtpu.runner.utils import HASH_IGNORED_ADDITIONAL_CONFIG_KEYS

    assert mo.SECTION in sections.KNOWN_SECTIONS
    sections.check({mo.SECTION: {"lowtok_fused_fp8_dispatch": False}})
    assert mo.SECTION not in HASH_IGNORED_ADDITIONAL_CONFIG_KEYS
