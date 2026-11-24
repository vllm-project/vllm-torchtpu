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
"""A pipeline stage as the world of the parallelism inside it."""

from types import SimpleNamespace

import pytest
import torch.distributed as dist
from vllm.distributed import parallel_state

from vllm_torchtpu.distributed import pipeline_stage
from vllm_torchtpu.distributed.pipeline_stage import PipelineStage


@pytest.fixture(autouse=True)
def no_stage(monkeypatch):
    monkeypatch.setattr(pipeline_stage, "_STAGE", None)


def _distributed(monkeypatch, world_size, rank):
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: world_size)
    monkeypatch.setattr(dist, "get_rank", lambda *a, **k: rank)


def _pipeline(monkeypatch, stage_of_rank, rank, group_ranks=None):
    """A world whose rank r runs stage `stage_of_rank[r]`, seen from `rank`.
    Returns the rank lists the stage groups are formed from."""
    stages = max(stage_of_rank) + 1
    _distributed(monkeypatch, len(stage_of_rank), rank)
    if group_ranks is None:
        by_stage = [
            [r for r, s in enumerate(stage_of_rank) if s == stage]
            for stage in range(stages)
        ]
        group_ranks = [list(column) for column in zip(*by_stage)]
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(
            world_size=stages,
            rank_in_group=stage_of_rank[rank],
            group_ranks=group_ranks,
        ),
    )
    world = SimpleNamespace(
        world_size=len(stage_of_rank),
        local_rank=rank,
        cpu_group=object(),
        device_group=object(),
    )
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: world)
    monkeypatch.setattr(dist, "get_backend", lambda group: "tpu_dist")

    def unexpected_all_gather(*_args, **_kwargs):
        raise AssertionError("init_pipeline_stage must not run all_gather_object")

    monkeypatch.setattr(dist, "all_gather_object", unexpected_all_gather)
    formed = []

    def init_model_parallel_group(group_ranks, local_rank, backend, group_name):
        assert (local_rank, backend) == (rank, "tpu_dist")
        assert group_name == pipeline_stage.STAGE_GROUP_NAME
        formed.append(group_ranks)
        mine = next(ranks for ranks in group_ranks if rank in ranks)
        return SimpleNamespace(ranks=mine, rank_in_group=mine.index(rank))

    monkeypatch.setattr(
        parallel_state, "init_model_parallel_group", init_model_parallel_group
    )
    return formed


def test_a_process_without_torch_distributed_is_its_own_stage(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    assert pipeline_stage.get_pipeline_stage() == PipelineStage(0, 1, (0,), 0, None)


def test_the_stage_is_the_world_without_a_pipeline(monkeypatch):
    _distributed(monkeypatch, world_size=8, rank=5)
    monkeypatch.setattr(
        parallel_state, "get_pp_group", lambda: SimpleNamespace(world_size=1)
    )
    stage = pipeline_stage.get_pipeline_stage()
    assert stage == PipelineStage(0, 1, tuple(range(8)), 5, None)
    assert stage.size == 8


def test_the_stage_is_the_world_before_the_parallel_groups_exist(monkeypatch):
    _distributed(monkeypatch, world_size=2, rank=1)
    monkeypatch.setattr(parallel_state, "_PP", None)
    assert pipeline_stage.get_pipeline_stage() == PipelineStage(0, 1, (0, 1), 1, None)


def test_a_pipeline_forms_nothing_without_stages(monkeypatch):
    formed = _pipeline(monkeypatch, [0, 0, 0, 0], rank=2)
    pipeline_stage.init_pipeline_stage()
    assert formed == []
    assert pipeline_stage.get_pipeline_stage().ranks == (0, 1, 2, 3)


@pytest.mark.parametrize(
    "stage_of_rank,rank,index,ranks,position",
    [
        # Four stages of two workers each.
        ([0, 0, 1, 1, 2, 2, 3, 3], 5, 2, (4, 5), 1),
        # Two stages of four workers, whatever axes split them.
        ([0, 0, 0, 0, 1, 1, 1, 1], 6, 1, (4, 5, 6, 7), 2),
        # Two engines joined by expert parallelism: a stage holds the
        # workers of both, which are not neighbours in rank order.
        ([0, 0, 1, 1, 0, 0, 1, 1], 4, 0, (0, 1, 4, 5), 2),
    ],
)
def test_a_stage_holds_the_workers_of_its_layers(
    monkeypatch, stage_of_rank, rank, index, ranks, position
):
    formed = _pipeline(monkeypatch, stage_of_rank, rank)
    pipeline_stage.init_pipeline_stage()
    stage = pipeline_stage.get_pipeline_stage()
    assert (stage.index, stage.count) == (index, max(stage_of_rank) + 1)
    assert (stage.ranks, stage.position, stage.size) == (ranks, position, len(ranks))
    assert stage.group.ranks == list(ranks)
    # Every worker forms the groups of all stages, in stage order.
    assert formed == [
        [
            [r for r, s in enumerate(stage_of_rank) if s == stage]
            for stage in range(max(stage_of_rank) + 1)
        ]
    ]


def test_stages_of_different_sizes_are_refused(monkeypatch):
    _pipeline(monkeypatch, [0, 0, 0, 1], rank=0, group_ranks=[[0, 3], [1], [2]])
    with pytest.raises(RuntimeError, match="different numbers of stages"):
        pipeline_stage.init_pipeline_stage()


def test_a_pipeline_stage_must_be_formed_before_it_is_read(monkeypatch):
    _pipeline(monkeypatch, [0, 1], rank=0)
    with pytest.raises(RuntimeError, match="init_pipeline_stage"):
        pipeline_stage.get_pipeline_stage()


@pytest.mark.parametrize("world_size,pp,expected", [(8, 1, 8), (8, 4, 2), (32, 4, 8)])
def test_stage_size_is_the_engines_workers_per_stage(world_size, pp, expected):
    config = SimpleNamespace(world_size=world_size, pipeline_parallel_size=pp)
    assert pipeline_stage.stage_size(config) == expected


def test_engine_rank_orders_workers_stage_then_pcp_then_tp():
    seen = [
        pipeline_stage.engine_rank(
            pp_rank=s, pcp_rank=p, pcp_size=2, tp_rank=t, tp_size=4
        )
        for s in range(3)
        for p in range(2)
        for t in range(4)
    ]
    assert seen == list(range(24))
    assert pipeline_stage.engine_rank(pcp_rank=3, tp_rank=1, tp_size=2) == 7
