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
"""API-server-side emitter: report the engine's parallel configuration.

Written once to the log as a ``rank_placement`` JSON line, from the server's
lifespan startup, so that how a deployment was parallelised is recorded
next to its other startup output.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from vllm.config import ParallelConfig

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.observability.placement_types import ParallelSummary

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def parallel_summary(parallel_config: ParallelConfig) -> ParallelSummary:
    return ParallelSummary(
        tensor_parallel_size=parallel_config.tensor_parallel_size,
        pipeline_parallel_size=parallel_config.pipeline_parallel_size,
        data_parallel_size=parallel_config.data_parallel_size,
        enable_expert_parallel=parallel_config.enable_expert_parallel,
        prefill_context_parallel_size=parallel_config.prefill_context_parallel_size,
        decode_context_parallel_size=parallel_config.decode_context_parallel_size,
        world_size=parallel_config.world_size,
        nnodes=parallel_config.nnodes,
    )


def emit_log(summary: ParallelSummary) -> None:
    logger.info(
        "rank_placement %s", json.dumps({"parallel": summary.to_dict()}, sort_keys=True)
    )


def publish_placement(vllm_config: VllmConfig) -> None:
    """Publish once, from the API server's lifespan startup.

    Reads typed ``ParallelConfig`` fields only, so on a build whose tests pass
    it cannot fail; a field renamed at a pin bump fails the unit tests instead
    of being hidden at runtime. ``VLLM_TPU_EMIT_PLACEMENT=0`` disables the
    whole path.
    """
    emit_log(parallel_summary(vllm_config.parallel_config))
