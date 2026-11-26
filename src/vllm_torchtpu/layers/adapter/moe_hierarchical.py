# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Hierarchical MoE parallelism: EP between chips, TP within a chip."""

from __future__ import annotations

import contextlib
import dataclasses

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.chip_topology import hierarchical_moe_split
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.glm_dsa_dispatch import resolve_ep_tp_size

logger = init_logger(__name__)


def ep_mesh_is_rank_ordered(parallel_config) -> bool:
    """True when ``build_ep_mesh`` orders EP shards by rank rather than device ID."""
    return (
        parallel_config.decode_context_parallel_size > 1
        or parallel_config.prefill_context_parallel_size > 1
        or bool(envs.MOE_FUSED_EP_SORTED_GATHER)
    )


def ep_tp_size(vllm_config=None) -> int:
    """Intra-group MoE TP size for hierarchical EPxTP (``2`` on GLM-5 DSA v7x)."""
    if envs.TPU_MOE_HIERARCHICAL_EP:
        return 1
    if vllm_config is None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    if (
        vllm_config is None
        or vllm_config.model_config is None
        or vllm_config.parallel_config is None
        or getattr(vllm_config, "speculative_config", None) is not None
    ):
        return 1
    pc = vllm_config.parallel_config
    if (
        not pc.enable_expert_parallel
        or pc.data_parallel_size != 1
        or pc.prefill_context_parallel_size != 1
        or pc.enable_eplb
        or pc.use_sequence_parallel_moe
    ):
        return 1
    tp_world = int(pc.tensor_parallel_size)
    factor = resolve_ep_tp_size(tp_world, vllm_config.model_config)
    if factor <= 1 or tp_world % factor != 0:
        return 1
    if not ep_mesh_is_rank_ordered(pc):
        logger.info_once(
            "Skipping hierarchical MoE EPxTP split: EP mesh is device-id ordered "
            "(decode_context_parallel_size=1, MOE_FUSED_EP_SORTED_GATHER=0); "
            "keeping flat expert parallelism."
        )
        return 1
    return factor


def ep_tp_split(
    world_size: int, rank: int, tp_size: int
) -> tuple[int, int, int, int] | None:
    """``(ep_size, ep_rank, tp_size, tp_rank)`` over groups of ``tp_size`` ranks."""
    if tp_size <= 1 or world_size % tp_size != 0:
        logger.warning_once(
            "Hierarchical EPxTP tp_size=%d does not divide TP world size %d; "
            "falling back to flat expert parallelism.",
            tp_size,
            world_size,
        )
        return None
    return (world_size // tp_size, rank // tp_size, tp_size, rank % tp_size)


def hierarchical_split_or_none() -> tuple[int, int, int, int] | None:
    """``(ep_size, ep_rank, tp_size, tp_rank)``, or ``None`` to stay flat.

    ``None`` is the safe answer everywhere: the caller keeps vLLM's own
    parallel config and the model runs exactly as it does today.
    """
    use_chip_ep = bool(envs.TPU_MOE_HIERARCHICAL_EP)
    ep_tp = 1 if use_chip_ep else ep_tp_size()
    if not use_chip_ep and ep_tp <= 1:
        return None

    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    world_size = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()
    if world_size <= 1:
        return None

    if ep_tp > 1:
        return ep_tp_split(world_size, rank, ep_tp)

    split = hierarchical_moe_split(world_size, rank)
    if split is None:
        logger.warning_once(
            "TPU_MOE_HIERARCHICAL_EP is set but the chip topology could not be "
            "used; falling back to flat expert parallelism."
        )
        return None

    return split


def _split_parallel_config(config, split: tuple[int, int, int, int]):
    if not config.use_ep:
        logger.warning_once(
            "A hierarchical MoE split is active but expert parallelism is off; "
            "leaving the MoE parallel config alone."
        )
        return config
    ep_size, ep_rank, tp_size, tp_rank = split
    return dataclasses.replace(
        config, ep_size=ep_size, ep_rank=ep_rank, tp_size=tp_size, tp_rank=tp_rank
    )


_filter_patched = False


def align_ep_weight_filter() -> None:
    """Teach ``--enable-ep-weight-filter`` the chip-block expert ownership."""
    global _filter_patched
    if _filter_patched:
        return

    from vllm.model_executor.model_loader import default_loader

    original = default_loader.compute_local_expert_ids

    def group_aware(
        num_experts: int, ep_size: int, ep_rank: int, placement: str = "linear"
    ) -> set[int] | None:
        split = hierarchical_split_or_none()
        if split is not None:
            ep_size, ep_rank = split[0], split[1]
        return original(num_experts, ep_size, ep_rank, placement=placement)

    default_loader.compute_local_expert_ids = group_aware
    _filter_patched = True
    logger.info("EP weight filter realigned to the hierarchical MoE expert split.")


@contextlib.contextmanager
def hierarchical_moe_parallel_config():
    """Make ``FusedMoE`` built inside this context manager use the chip-aware split."""
    split = hierarchical_split_or_none()
    if split is None:
        yield None
        return

    ep_size, ep_rank, tp_size, tp_rank = split

    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    original_make = FusedMoEParallelConfig.make

    def patched_make(*args, **kwargs):
        return _split_parallel_config(original_make(*args, **kwargs), split)

    # Safe here: the loader calls _init_ep_weight_filter from load_weights,
    # which runs after the model is constructed, so this lands in time.
    align_ep_weight_filter()

    logger.info_once(
        "Hierarchical MoE parallelism: EP between chips (ep_size=%d, "
        "ep_rank=%d), TP within a chip (tp_size=%d, tp_rank=%d)",
        ep_size,
        ep_rank,
        tp_size,
        tp_rank,
    )
    FusedMoEParallelConfig.make = staticmethod(patched_make)
    try:
        yield split
    finally:
        FusedMoEParallelConfig.make = original_make


def check_ep_tp_config(config, split: tuple[int, int, int, int]) -> None:
    """Validate that EPxTP partial sums meet in the TP group's all-reduce."""
    if config.dp_size != 1 or config.pcp_size != 1 or config.is_sequence_parallel:
        raise ValueError(
            "Hierarchical EPxTP MoE supports DP=1, PCP=1, and no sequence-parallel "
            f"MoE only; got dp={config.dp_size}, pcp={config.pcp_size}, "
            f"sp={config.sp_size}."
        )
    if config.enable_eplb:
        raise ValueError("Hierarchical EPxTP MoE does not support EPLB.")
    if config.tp_size != 1 or config.ep_size != split[0] * split[2]:
        raise ValueError(
            "Hierarchical EPxTP MoE expects flat EP over the whole TP group "
            f"(tp_size=1, ep_size={split[0] * split[2]}); got "
            f"tp_size={config.tp_size}, ep_size={config.ep_size}."
        )


def install_ep_tp_split_patch() -> None:
    """Install the ``FusedMoEParallelConfig.make`` wrapper for GLM-5 DSA EPxTP."""
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    if getattr(FusedMoEParallelConfig, "_tpu_ep_tp_split_patch", False):
        return
    original_make = FusedMoEParallelConfig.make

    def ep_tp_make(*args, **kwargs):
        config = original_make(*args, **kwargs)
        if envs.TPU_MOE_HIERARCHICAL_EP or ep_tp_size() <= 1:
            return config
        split = hierarchical_split_or_none()
        if split is None:
            return config
        if not config.use_ep:
            return _split_parallel_config(config, split)
        check_ep_tp_config(config, split)
        logger.info_once(
            "MoE experts split EP %d x TP %d: ep_rank=%d, tp_rank=%d.",
            split[0],
            split[2],
            split[1],
            split[3],
        )
        return _split_parallel_config(config, split)

    FusedMoEParallelConfig.make = staticmethod(ep_tp_make)
    FusedMoEParallelConfig._tpu_ep_tp_split_patch = True  # type: ignore[attr-defined]
    align_ep_weight_filter()
    logger.info("Applied TPU patch: hierarchical MoE EP x TP expert split.")
