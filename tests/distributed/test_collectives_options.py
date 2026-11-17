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
"""Parsing of ``additional_config["tpu_collectives"]``."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.distributed import collectives_options as co

pytestmark = pytest.mark.cpu_test


def _config(additional_config, expert_parallel=True):
    return SimpleNamespace(
        additional_config=additional_config,
        parallel_config=SimpleNamespace(enable_expert_parallel=expert_parallel),
    )


@pytest.mark.parametrize(
    "additional_config",
    [None, {}, {"tpu_collectives": None}, {"tpu_collectives": {}}, {"other": 1}],
    ids=["none", "empty", "null_section", "empty_section", "other_section"],
)
def test_defaults_are_mains_behaviour(additional_config):
    assert co.get(_config(additional_config)) == co.CollectivesOptions()
    assert co.CollectivesOptions().small_all_reduce is False


def test_the_option_is_read():
    options = co.get(_config({"tpu_collectives": {"small_all_reduce": True}}))
    assert options.small_all_reduce is True


@pytest.mark.parametrize(
    "section,match",
    [
        (True, "must be a JSON object"),
        ({"small_all_reduce": 1}, "must be a bool"),
        ({"small_all_reduce": "true"}, "must be a bool"),
        ({"small_all_reduce_max_rows": 64}, "unknown keys"),
    ],
    ids=["not_an_object", "int_for_bool", "string_for_bool", "unknown_key"],
)
def test_malformed_sections_are_rejected(section, match):
    with pytest.raises(ValueError, match=match):
        co.get(_config({"tpu_collectives": section}))


def test_options_are_frozen():
    with pytest.raises(AttributeError):
        co.CollectivesOptions().small_all_reduce = True  # type: ignore[misc]


@pytest.mark.parametrize(
    "additional_config,expert_parallel",
    [
        ({}, False),
        ({"tpu_collectives": {"small_all_reduce": False}}, False),
        ({"tpu_collectives": {"small_all_reduce": True}}, True),
    ],
    ids=["no_section", "off_without_ep", "on_with_ep"],
)
def test_validate_accepts_what_the_kernels_can_serve(
    additional_config, expert_parallel
):
    co.validate(_config(additional_config, expert_parallel))


def test_validate_refuses_small_all_reduce_without_expert_parallelism():
    config = _config({"tpu_collectives": {"small_all_reduce": True}}, False)
    with pytest.raises(NotImplementedError, match="--enable-expert-parallel"):
        co.validate(config)


def test_validate_rejects_a_malformed_section():
    with pytest.raises(ValueError, match="unknown keys"):
        co.validate(_config({"tpu_collectives": {"small_all_reduce_max_rows": 64}}))
