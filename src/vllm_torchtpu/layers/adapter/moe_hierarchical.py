# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hierarchical MoE parallelism: EP between chips, TP within a chip."""

from __future__ import annotations

import contextlib
import dataclasses

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.chip_topology import hierarchical_moe_split
from vllm_torchtpu.distributed.mesh_utils import CpGroupLayout, get_cp_group_layout
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_SPLIT_CACHE: dict[CpGroupLayout, tuple[int, int, int, int]] = {}


def _agreed_split(group, layout: CpGroupLayout) -> tuple[int, int, int, int]:
    """This rank's split, once every rank has resolved a consistent one."""
    split = hierarchical_moe_split(layout.device_ids, layout.rank_in_group)
    gathered: list[tuple[int, int, int, int] | None] = [None] * layout.world_size
    torch.distributed.all_gather_object(gathered, split, group=group.cpu_group)
    shapes = {(s[0], s[2]) for s in gathered if s is not None}
    slots = {(s[1], s[3]) for s in gathered if s is not None}
    if split is None or len(shapes) != 1 or len(slots) != layout.world_size:
        raise ValueError(
            "TPU_MOE_HIERARCHICAL_EP requires the EP group to cover complete "
            f"multi-core chips on every rank; got {gathered}"
        )
    return split


def hierarchical_split_or_none() -> tuple[int, int, int, int] | None:
    """Return chip-level EP and intra-chip TP, or None when disabled."""
    if not envs.TPU_MOE_HIERARCHICAL_EP:
        return None

    from vllm.config import get_current_vllm_config_or_none

    from vllm_torchtpu.distributed.ep_mesh import get_ep_group

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None or not vllm_config.parallel_config.enable_expert_parallel:
        return None
    if vllm_config.parallel_config.enable_eplb or envs.USE_MOE_FUSED_EP_KERNEL:
        raise ValueError(
            "TPU_MOE_HIERARCHICAL_EP requires the GMM MoE backend with EPLB disabled"
        )
    group = get_ep_group()
    if group is None:
        return None
    layout = get_cp_group_layout(group)
    if layout not in _SPLIT_CACHE:
        _SPLIT_CACHE[layout] = _agreed_split(group, layout)
    return _SPLIT_CACHE[layout]


_filter_patched = False


def align_ep_weight_filter() -> None:
    """Teach ``--enable-ep-weight-filter`` the chip-block expert ownership."""
    global _filter_patched
    if _filter_patched:
        return

    split = hierarchical_split_or_none()
    if split is None:
        return
    ep_size, ep_rank, _, _ = split

    from vllm.model_executor.model_loader import default_loader

    original = default_loader.compute_local_expert_ids

    def chip_aware(num_experts, _ep_size, _ep_rank, placement="linear"):
        # Ignore the caller's rank-derived ownership; the experts this rank
        # needs are its chip's block, shared with the other chiplet.
        return original(num_experts, ep_size, ep_rank, placement=placement)

    default_loader.compute_local_expert_ids = chip_aware
    _filter_patched = True
    logger.info(
        "EP weight filter realigned to chip-block ownership: ep_size=%d, "
        "ep_rank=%d (was rank-derived).",
        ep_size,
        ep_rank,
    )


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
        config = original_make(*args, **kwargs)
        assert config.use_ep
        if config.is_sequence_parallel:
            raise ValueError("Hierarchical MoE does not support sequence parallelism")
        return dataclasses.replace(
            config, ep_size=ep_size, ep_rank=ep_rank, tp_size=tp_size, tp_rank=tp_rank
        )

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
