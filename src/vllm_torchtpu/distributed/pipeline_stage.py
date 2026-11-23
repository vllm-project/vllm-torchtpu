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
"""A pipeline stage: the workers that run one slice of the layers.

Pipeline parallelism splits the layers over stages and is the only axis that
does. Every other axis shards the work of the layers a worker holds, so it
lives inside a stage, and a stage is to those axes what the whole world is
without a pipeline: the ranks their meshes cover and their collectives span.

The two sides meet here. Code of an axis inside a stage asks for the stage
where it would have asked for the world, and never whether a pipeline is on
(custom kernels tuned to a fixed whole-host topology still check the world and
fall back to c10d inside a pipeline). Pipeline code sees a stage as its `size`
workers, each at a `position`, and never which axes put them there; the worker
at one position of every stage forms a pipeline group, the hand-off's column.

Without a pipeline the one stage is the world.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

STAGE_GROUP_NAME = "pp_stage"


@dataclass(frozen=True)
class PipelineStage:
    """This worker's pipeline stage.

    Attributes:
      index: the stage, counted from the first layers.
      count: the number of stages.
      ranks: the global ranks of the stage's workers, ascending.
      position: this worker's index in `ranks`.
      group: the stage's own process group, over `ranks` in that order; None
        without a pipeline, where the stage is the world.
    """

    index: int
    count: int
    ranks: tuple[int, ...]
    position: int
    group: Any | None

    @property
    def size(self) -> int:
        return len(self.ranks)


_STAGE: PipelineStage | None = None


def stage_size(parallel_config: Any) -> int:
    """How many workers of one engine a pipeline stage holds."""
    return parallel_config.world_size // parallel_config.pipeline_parallel_size


def _stage_ranks(group_ranks: list[list[int]], stages: int) -> list[list[int]]:
    """The ranks of each stage, from the pipeline group's columns."""
    if not group_ranks or any(len(column) != stages for column in group_ranks):
        raise RuntimeError(
            f"Pipeline columns hold different numbers of stages: {group_ranks}."
        )
    return [
        sorted(int(column[stage]) for column in group_ranks) for stage in range(stages)
    ]


def init_pipeline_stage() -> None:
    """Forms the process group of every pipeline stage.

    Collective over the world: every worker calls it once, after the model
    parallel groups exist. Without a pipeline the stage is the world and
    nothing is formed.
    """
    global _STAGE
    import torch.distributed as dist
    from vllm.distributed.parallel_state import (
        get_pp_group,
        get_world_group,
        init_model_parallel_group,
    )

    pipeline = get_pp_group()
    stages = int(pipeline.world_size)
    if stages == 1:
        return
    world = get_world_group()
    group = init_model_parallel_group(
        _stage_ranks(pipeline.group_ranks, stages),
        world.local_rank,
        dist.get_backend(world.device_group),
        group_name=STAGE_GROUP_NAME,
    )
    _STAGE = PipelineStage(
        index=int(pipeline.rank_in_group),
        count=stages,
        ranks=tuple(int(rank) for rank in group.ranks),
        position=int(group.rank_in_group),
        group=group,
    )
    logger.info(
        "Pipeline stage %d/%d | ranks=%s | position=%d",
        _STAGE.index,
        _STAGE.count,
        _STAGE.ranks,
        _STAGE.position,
    )


def get_pipeline_stage() -> PipelineStage:
    """This worker's pipeline stage; the world without a pipeline."""
    if _STAGE is not None:
        return _STAGE
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return PipelineStage(index=0, count=1, ranks=(0,), position=0, group=None)
    from vllm.distributed.parallel_state import get_pp_group

    try:
        stages = int(get_pp_group().world_size)
    except AssertionError:
        # The model parallel groups do not exist yet.
        stages = 1
    if stages > 1:
        raise RuntimeError(
            "init_pipeline_stage() has not formed this worker's pipeline stage."
        )
    return PipelineStage(
        index=0,
        count=1,
        ranks=tuple(range(int(dist.get_world_size()))),
        position=int(dist.get_rank()),
        group=None,
    )
