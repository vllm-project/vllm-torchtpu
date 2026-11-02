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
"""Expert parallelism inside the tensor-parallel group (TPU_MOE_TP_EXPERT_SPLIT).

Without ``--enable-expert-parallel``, vLLM shards every expert's intermediate
dimension across the TP ranks, so each rank runs every routed row of the batch
through a slice of its expert. With this patch each TP rank instead owns whole
experts (``ep_size = tp_size``, ``ep_rank = tp_rank``) and runs only the rows
routed to them. The MoE output is a partial sum over the rank's experts either
way, and ``MoERunner`` reduces it with the same TP all-reduce
(``tp_size > 1 or ep_size > 1``). Nothing crosses the TP group: data-parallel
replicas stay independent, which is the point of running without vLLM's EP.
``--enable-expert-parallel`` cannot express this: its EP group spans the
data-parallel replicas and couples them.

The arithmetic changes with the layout: each expert's down projection now runs
over its whole intermediate on one rank (quantized in 512-column blocks)
instead of over per-rank slices summed by the all-reduce.
"""

from __future__ import annotations

import dataclasses

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def split_experts_over_tp(config):
    """The EP-over-TP form of a flat MoE parallel config, or ``config`` itself.

    Only a config that shards experts over the TP group alone qualifies: EP
    configs, data/pipeline-context/sequence-parallel MoE, and single-rank
    layers keep vLLM's layout.
    """
    if (
        config.use_ep
        or config.tp_size <= 1
        or config.dp_size > 1
        or config.pcp_size > 1
        or config.sp_size > 1
    ):
        return config
    return dataclasses.replace(
        config,
        ep_size=config.tp_size,
        ep_rank=config.tp_rank,
        tp_size=1,
        tp_rank=0,
        use_ep=True,
    )


def patch_tp_expert_split() -> None:
    """Install ``split_experts_over_tp`` on ``FusedMoEParallelConfig.make``."""
    if not envs.TPU_MOE_TP_EXPERT_SPLIT:
        return
    if envs.TPU_MOE_HIERARCHICAL_EP:
        # Hierarchical EP rewrites EP configs; on top of this split it would
        # re-split layers it expects to find flat.
        raise ValueError(
            "TPU_MOE_TP_EXPERT_SPLIT and TPU_MOE_HIERARCHICAL_EP cannot be combined."
        )
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    if getattr(FusedMoEParallelConfig, "_tpu_tp_expert_split_patch", False):
        return
    original_make = FusedMoEParallelConfig.make

    def make(*args, **kwargs):
        config = original_make(*args, **kwargs)
        split = split_experts_over_tp(config)
        if split is not config:
            logger.info_once(
                "TPU_MOE_TP_EXPERT_SPLIT: MoE experts split over the TP group "
                "(ep_size=%d) instead of sharding their intermediate dimension.",
                split.ep_size,
            )
        return split

    FusedMoEParallelConfig.make = staticmethod(make)
    FusedMoEParallelConfig._tpu_tp_expert_split_patch = True
    logger.info("Applied TPU patch: split MoE experts over the TP group.")
