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
"""Which decode paths a launch line turns on: environment dict in, resolved
settings out, and the refusal for an unknown path."""

import pytest

from vllm_torchtpu import decode_switches as ds

pytestmark = pytest.mark.cpu_test

NEW_LINE = {
    "TPU_BATCHED_RPA_DECODE_BLOCK": "8192,4,2",
    "TPU_BATCHED_RPA_VERIFY_BLOCK": "window",
    "TPU_BATCHED_RPA_KV_PAGE_RUNS": "1",
    "TPU_DECODE_FAST_PATHS": "1",
    "TPU_ROPE_CACHE_ROW_MAJOR": "1",
}
ALL_PATHS = {p.name for p in ds.PATHS}


def test_the_path_table_names_are_short_lowercase_and_unique():
    names = [p.name for p in ds.PATHS]
    assert len(names) == len(set(names)) == 2
    assert all(n == n.lower() and " " not in n and len(n) <= 20 for n in names)


def test_the_launch_line_turns_every_path_on():
    sw = ds.resolve(NEW_LINE)
    assert sw.paths == ALL_PATHS


@pytest.mark.parametrize(
    "env,paths",
    [
        ({}, set()),
        ({"TPU_BATCHED_RPA_VERIFY_BLOCK": "window"}, {"draft_one_query"}),
        ({"TPU_BATCHED_RPA_VERIFY_BLOCK": "4096,4,2"}, {"draft_one_query"}),
        ({"TPU_BATCHED_RPA_DECODE_BLOCK": "8192,4,2"}, set()),
        ({"TPU_DECODE_FAST_PATHS": "1"}, {"xla_schedule"}),
        ({"TPU_DECODE_FAST_PATHS": "0"}, set()),
    ],
)
def test_each_parent_turns_on_its_own_paths(env, paths):
    assert ds.resolve(env).paths == paths


@pytest.mark.parametrize("path", sorted(ALL_PATHS))
def test_the_off_list_leaves_one_path_off(path):
    env = {**NEW_LINE, "TPU_DECODE_FAST_PATHS_OFF": path}
    assert ds.resolve(env).paths == ALL_PATHS - {path}


def test_the_off_list_takes_several_names_with_spaces():
    env = {**NEW_LINE, "TPU_DECODE_FAST_PATHS_OFF": " xla_schedule , draft_one_query,"}
    assert ds.resolve(env).paths == ALL_PATHS - {"xla_schedule", "draft_one_query"}


def test_an_unknown_name_in_the_off_list_refuses_with_the_valid_names():
    env = {**NEW_LINE, "TPU_DECODE_FAST_PATHS_OFF": "xla_schedule,gdn_fast"}
    with pytest.raises(ValueError, match="unknown path.*gdn_fast") as err:
        ds.resolve(env)
    for name in ALL_PATHS:
        assert name in str(err.value)
