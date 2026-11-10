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


def _config(additional_config, moe=True, dp=1, pcp=1, tp=8):
    return SimpleNamespace(
        additional_config=additional_config,
        model_config=None if moe is None else SimpleNamespace(is_moe=moe),
        parallel_config=SimpleNamespace(
            data_parallel_size=dp,
            prefill_context_parallel_size=pcp,
            tensor_parallel_size=tp,
        ),
    )


@pytest.mark.parametrize(
    "additional_config",
    [None, {}, {"tpu_collectives": None}, {"tpu_collectives": {}}, {"other": 1}],
    ids=["none", "empty", "null_section", "empty_section", "other_section"],
)
def test_defaults_are_mains_behaviour(additional_config):
    assert co.get(_config(additional_config)) == co.CollectivesOptions()


def test_small_all_reduce_is_off_by_default():
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


ON = {"tpu_collectives": {"small_all_reduce": True}}


@pytest.mark.parametrize(
    "additional_config,kwargs",
    [
        ({}, {"moe": False, "tp": 1}),
        ({"tpu_collectives": {"small_all_reduce": False}}, {"moe": False}),
        # vLLM builds the EP group for every MoE model, so the option needs no
        # --enable-expert-parallel: a MoE model at TP8 without it still serves.
        (ON, {"moe": True, "tp": 8}),
        (ON, {"moe": True, "dp": 8, "tp": 1}),
        (ON, {"moe": True, "pcp": 2, "tp": 1}),
        (ON, {"moe": None, "tp": 8}),  # no model config: vLLM builds the group
    ],
    ids=[
        "no_section",
        "off_on_a_dense_model",
        "moe_tp8",
        "moe_dp8",
        "moe_pcp2",
        "no_model",
    ],
)
def test_validate_accepts_what_the_kernels_can_serve(additional_config, kwargs):
    co.validate(_config(additional_config, **kwargs))


@pytest.mark.parametrize(
    "kwargs,match",
    [({"moe": False, "tp": 8}, "a dense model"), ({"moe": True, "tp": 1}, "one rank")],
    ids=["dense_model", "one_rank"],
)
def test_validate_refuses_small_all_reduce_without_an_expert_parallel_group(
    kwargs, match
):
    with pytest.raises(NotImplementedError, match=match):
        co.validate(_config(ON, **kwargs))


def test_validate_rejects_a_malformed_section():
    with pytest.raises(ValueError, match="unknown keys"):
        co.validate(_config({"tpu_collectives": {"small_all_reduce_max_rows": 64}}))


def test_the_section_is_registered():
    """A tpu_* section the tree does not list is refused at engine start."""
    from vllm_torchtpu import additional_config_sections as sections

    assert co.SECTION in sections.KNOWN_SECTIONS
    sections.check({co.SECTION: {"small_all_reduce": True}})
