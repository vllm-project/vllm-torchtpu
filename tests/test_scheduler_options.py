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
"""The ``tpu_scheduler`` additional_config section."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu import additional_config_sections
from vllm_torchtpu.core import scheduler_options
from vllm_torchtpu.runner.utils import (
    HASH_IGNORED_ADDITIONAL_CONFIG_KEYS,
    SCHEDULER_OPTIONS_SECTION,
)


def test_defaults_are_mains_behaviour():
    assert scheduler_options.parse(None) == scheduler_options.SchedulerOptions()
    assert scheduler_options.parse({}).promote_warmest is True
    assert scheduler_options.get(
        SimpleNamespace(additional_config=None)
    ).promote_warmest


def test_parse_reads_the_option():
    cfg = SimpleNamespace(
        additional_config={"tpu_scheduler": {"promote_warmest": False}}
    )
    assert scheduler_options.get(cfg).promote_warmest is False


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"promote": False}, "unknown keys"),
        ({"promote_warmest": 0}, "must be a bool"),
        ({"promote_warmest": "false"}, "must be a bool"),
        ([], "must be a JSON object"),
    ],
)
def test_parse_is_strict(raw, match):
    with pytest.raises(ValueError, match=match):
        scheduler_options.parse(raw)


def test_section_is_registered_and_kept_out_of_the_compile_cache_key():
    assert scheduler_options.SECTION == SCHEDULER_OPTIONS_SECTION
    assert scheduler_options.SECTION in additional_config_sections.KNOWN_SECTIONS
    assert scheduler_options.SECTION in HASH_IGNORED_ADDITIONAL_CONFIG_KEYS
