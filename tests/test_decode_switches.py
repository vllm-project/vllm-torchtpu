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
settings out, and the refusals for unknown paths and combinations that cannot run."""

import pytest

from vllm_torchtpu import decode_switches as ds
from vllm_torchtpu import envs

pytestmark = pytest.mark.cpu_test

NEW_LINE = {
    "TPU_MOE_RAGGED_A2A": "1",
    "TPU_MOE_EXPERT_PLACEMENT": "/x/placement.npz",
    "TPU_BATCHED_RPA_DECODE_BLOCK": "8192,4,2",
    "TPU_BATCHED_RPA_VERIFY_BLOCK": "window",
    "TPU_BATCHED_RPA_KV_PAGE_RUNS": "1",
    "TPU_DECODE_FAST_PATHS": "1",
    "TPU_LM_HEAD_HIDDEN_MAJOR": "1",
    "TPU_ROPE_CACHE_ROW_MAJOR": "1",
    "TPU_EMBED_EP_SHARD": "1",
    "TPU_TP4_HIER_ALL_REDUCE": "1",
}
ALL_PATHS = {p.name for p in ds.PATHS}
EXCHANGE_OF_THE_NEW_LINE = {
    "capacity_row_tile": 32,
    "row_gather": True,
    "row_scatter": True,
    "streamed_combine": True,
    "owner_combine": True,
    "sorted_plan": True,
    "owner_tail": True,
    "routing_dma": True,
    "send_index_rows_minor": True,
}


def test_the_path_table_names_are_short_lowercase_and_unique():
    names = [p.name for p in ds.PATHS]
    assert len(names) == len(set(names)) == 12
    assert all(n == n.lower() and " " not in n and len(n) <= 20 for n in names)


def test_the_launch_line_turns_every_path_on():
    sw = ds.resolve(NEW_LINE)
    assert sw.paths == ALL_PATHS
    assert sw.ragged_a2a and sw.owner_combine
    assert ds.exchange_keywords(sw, owner_combine=True) == EXCHANGE_OF_THE_NEW_LINE


@pytest.mark.parametrize(
    "env,paths",
    [
        ({}, set()),
        # The exchange alone: its own paths, none of the others.
        (
            {"TPU_MOE_RAGGED_A2A": "1"},
            {
                "a2a_streamed_combine",
                "a2a_sorted_plan",
                "a2a_row_gather",
                "a2a_row_scatter",
                "a2a_send_rows_minor",
                "a2a_routing_dma",
                "a2a_unquantized",
                "a2a_owner_combine",
                "a2a_owner_tail",
            },
        ),
        ({"TPU_BATCHED_RPA_VERIFY_BLOCK": "window"}, {"draft_one_query"}),
        ({"TPU_BATCHED_RPA_VERIFY_BLOCK": "4096,4,2"}, {"draft_one_query"}),
        ({"TPU_BATCHED_RPA_DECODE_BLOCK": "8192,4,2"}, set()),
        ({"TPU_EMBED_EP_SHARD": "1"}, {"embed_rows"}),
        ({"TPU_DECODE_FAST_PATHS": "1"}, {"xla_schedule"}),
        ({"TPU_DECODE_FAST_PATHS": "0"}, set()),
    ],
)
def test_each_parent_turns_on_its_own_paths(env, paths):
    assert ds.resolve(env).paths == paths


@pytest.mark.parametrize("path", sorted(ALL_PATHS - {"a2a_sorted_plan"}))
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


@pytest.mark.parametrize("path", ["a2a_row_gather", "a2a_row_scatter"])
def test_row_copies_refuse_without_the_sorted_plan(path):
    other = {"a2a_row_gather": "a2a_row_scatter", "a2a_row_scatter": "a2a_row_gather"}
    env = {
        "TPU_MOE_RAGGED_A2A": "1",
        "TPU_DECODE_FAST_PATHS_OFF": f"a2a_sorted_plan,{other[path]}",
    }
    with pytest.raises(ValueError, match=f"{path} needs a2a_sorted_plan"):
        ds.resolve(env)
    # Leaving the copies off too is fine.
    env["TPU_DECODE_FAST_PATHS_OFF"] = "a2a_sorted_plan,a2a_row_gather,a2a_row_scatter"
    assert not ds.resolve(env).path("a2a_sorted_plan")


def test_the_send_index_layout_follows_the_row_scatter():
    env = {"TPU_MOE_RAGGED_A2A": "1", "TPU_DECODE_FAST_PATHS_OFF": "a2a_row_scatter"}
    kw = ds.exchange_keywords(ds.resolve(env), owner_combine=True)
    assert kw["row_scatter"] is False and kw["send_index_rows_minor"] is False
    env["TPU_DECODE_FAST_PATHS_OFF"] = "a2a_send_rows_minor"
    kw = ds.exchange_keywords(ds.resolve(env), owner_combine=True)
    assert kw["row_scatter"] is True and kw["send_index_rows_minor"] is False


def test_the_exchange_keywords_follow_the_layer_not_the_switch():
    sw = ds.resolve(NEW_LINE)
    plain = ds.exchange_keywords(sw, owner_combine=False)
    assert plain["owner_combine"] is False and plain["owner_tail"] is False
    assert plain["streamed_combine"] is True
    off = ds.resolve({**NEW_LINE, "TPU_DECODE_FAST_PATHS_OFF": "a2a_owner_tail"})
    kw = ds.exchange_keywords(off, owner_combine=True)
    assert kw["owner_combine"] is True and kw["owner_tail"] is False


def test_without_the_owner_combine_path_the_streamed_combine_serves_the_layer():
    env = {**NEW_LINE, "TPU_DECODE_FAST_PATHS_OFF": "a2a_owner_combine"}
    sw = ds.resolve(env)
    assert not sw.owner_combine and not sw.path("a2a_owner_combine")
    # The tail is on as a path but acts only on a layer with the owner combine.
    assert sw.path("a2a_owner_tail")
    assert not ds.owner_combine_layer(sw, _facts())
    kw = ds.exchange_keywords(sw, owner_combine=False)
    assert kw["owner_combine"] is False and kw["owner_tail"] is False
    assert kw["streamed_combine"] is True


def test_the_owner_combine_needs_the_exchange():
    env = {k: v for k, v in NEW_LINE.items() if k != "TPU_MOE_RAGGED_A2A"}
    sw = ds.resolve(env)
    assert not sw.owner_combine and not sw.path("a2a_owner_combine")
    assert ds.resolve({"TPU_MOE_RAGGED_A2A": "1"}).owner_combine


@pytest.mark.parametrize(
    "env,skip",
    [
        ({}, False),
        ({"TPU_MOE_RAGGED_A2A": "1"}, True),
        ({"TPU_MOE_SKIP_PADDED_TOKENS": "1"}, True),
        ({"TPU_MOE_RAGGED_A2A": "1", "TPU_MOE_SKIP_PADDED_TOKENS": "0"}, False),
        ({"TPU_MOE_RAGGED_A2A": "1", "TPU_MOE_SKIP_PADDED_TOKENS": "1"}, True),
        ({"TPU_MOE_RAGGED_A2A": "1", "TPU_MOE_SKIP_PADDED_TOKENS": ""}, True),
        ({"TPU_MOE_SKIP_PADDED_TOKENS": "0"}, False),
    ],
)
def test_padded_tokens_are_skipped_by_default_with_the_exchange(monkeypatch, env, skip):
    for name in ("TPU_MOE_RAGGED_A2A", "TPU_MOE_SKIP_PADDED_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert envs.environment_variables["TPU_MOE_SKIP_PADDED_TOKENS"]() is skip


# The owner combine of a layer.


def _facts(**overrides):
    base = dict(
        sequence_parallel=False,
        pcp_size=1,
        skip_final_all_reduce=False,
        has_output_transform=False,
        scaling_factor=1.0,
        zero_expert_router=False,
    )
    return ds.OwnerCombineFacts(**{**base, **overrides})


@pytest.mark.parametrize(
    "overrides,armed",
    [
        ({}, True),
        ({"sequence_parallel": True}, False),
        ({"pcp_size": 2}, False),
        ({"skip_final_all_reduce": True}, False),
        ({"has_output_transform": True}, False),
        ({"scaling_factor": 2.5}, False),
        ({"zero_expert_router": True}, False),
    ],
)
def test_the_owner_combine_of_a_layer_follows_the_runner_facts(overrides, armed):
    sw = ds.resolve(NEW_LINE)
    assert ds.owner_combine_layer(sw, _facts(**overrides)) is armed


def test_the_owner_combine_of_a_layer_needs_the_switches_and_the_facts():
    assert not ds.owner_combine_layer(ds.resolve(NEW_LINE), None)
    no_owner = ds.resolve(
        {"TPU_MOE_RAGGED_A2A": "1", "TPU_DECODE_FAST_PATHS_OFF": "a2a_owner_combine"}
    )
    assert not ds.owner_combine_layer(no_owner, _facts())
    assert not ds.owner_combine_layer(ds.resolve({}), _facts())


@pytest.mark.parametrize("scaling", [1.0, 2.5])
def test_the_op_and_the_runner_read_one_decision(scaling):
    """The op takes ``owner_combine`` and the runner arms its post-processing
    from the same attribute, so a layer the runner cannot arm is not summed
    over TP by the op as well."""

    class Layer:
        pass

    layer = Layer()
    setattr(layer, ds.OWNER_FACTS_ATTR, _facts(scaling_factor=scaling))
    armed = ds.arm_owner_combine(layer, ds.resolve(NEW_LINE))
    assert armed is (scaling == 1.0)
    op = ds.exchange_keywords(ds.resolve(NEW_LINE), owner_combine=armed)
    assert op["owner_combine"] is getattr(layer, ds.OWNER_COMBINE_ATTR) is armed
    assert op["owner_tail"] is armed
