# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hierarchical MoE parallelism: experts split EP between groups, TP within one.

Two ways to ask for it:

- ``TPU_MOE_HIERARCHICAL_EP``: EP between chips, TP between the cores of a
  chip. Model code opts in by building its experts inside
  ``hierarchical_moe_parallel_config()`` (Kimi K3).
- ``TPU_MOE_EP_TP_SIZE=N``: EP between groups of N consecutive ranks, TP
  inside a group. Applied to every ``FusedMoE`` the process builds by
  ``install_ep_tp_split_patch`` (a registry patch), so upstream models such as
  DeepSeek-V2/V3 and GLM-5 take it without model code.
"""

from __future__ import annotations

import contextlib
import dataclasses

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.chip_topology import hierarchical_moe_split
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


# Architectures the default TPU_MOE_EP_TP_SIZE was measured on (#1510).
_DEFAULT_EP_TP_ARCHS = ("GlmMoeDsaForCausalLM",)
_default_ep_tp: int | None = None


def ep_tp_size() -> int:
    """``TPU_MOE_EP_TP_SIZE`` as it applies to this deployment.

    An explicit value is returned as is (and refuses an unsupported deployment
    downstream). The default applies only where it was measured -- the
    architectures above with EP over the whole TP group -- and stays 1 (flat)
    elsewhere. Decided
    once, from the vLLM config current when the first FusedMoE is built, so
    the MoE split and the EP weight filter at load time always agree.
    """
    if envs.is_set("TPU_MOE_EP_TP_SIZE"):
        return envs.TPU_MOE_EP_TP_SIZE
    global _default_ep_tp
    if _default_ep_tp is None:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if config is None or config.model_config is None:
            return 1  # no model yet; decide when there is one
        _default_ep_tp = _default_ep_tp_for(config)
    return _default_ep_tp


def _default_ep_tp_for(config) -> int:
    default = envs.TPU_MOE_EP_TP_SIZE
    pc = config.parallel_config
    archs = getattr(config.model_config.hf_config, "architectures", None) or []
    if (
        envs.TPU_MOE_HIERARCHICAL_EP
        or not any(arch in _DEFAULT_EP_TP_ARCHS for arch in archs)
        or not pc.enable_expert_parallel
        or pc.data_parallel_size != 1
        or pc.prefill_context_parallel_size != 1
        or pc.enable_eplb
        or pc.use_sequence_parallel_moe
        or pc.tensor_parallel_size % default != 0
    ):
        return 1
    return default


def hierarchical_split_or_none() -> tuple[int, int, int, int] | None:
    """``(ep_size, ep_rank, tp_size, tp_rank)``, or ``None`` to stay flat.

    ``None`` is the safe answer everywhere: the caller keeps vLLM's own
    parallel config and the model runs exactly as it does today.
    """
    ep_tp = ep_tp_size()
    if ep_tp > 1 and envs.TPU_MOE_HIERARCHICAL_EP:
        raise ValueError(
            "TPU_MOE_EP_TP_SIZE and TPU_MOE_HIERARCHICAL_EP both pick the MoE "
            "expert split; set one."
        )
    if ep_tp <= 1 and not envs.TPU_MOE_HIERARCHICAL_EP:
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


def ep_tp_split(
    world_size: int, rank: int, tp_size: int
) -> tuple[int, int, int, int] | None:
    """``(ep_size, ep_rank, tp_size, tp_rank)`` for EP between groups of
    ``tp_size`` consecutive ranks and TP inside a group, or ``None`` when
    ``tp_size`` does not divide ``world_size`` (stay flat)."""
    if tp_size <= 1 or world_size % tp_size != 0:
        logger.warning_once(
            "TPU_MOE_EP_TP_SIZE=%d does not divide the TP world size %d; "
            "falling back to flat expert parallelism.",
            tp_size,
            world_size,
        )
        return None
    return (world_size // tp_size, rank // tp_size, tp_size, rank % tp_size)


def _split_parallel_config(config, split):
    """Rewrite one ``FusedMoEParallelConfig`` to the split; flat stays flat."""
    if not config.use_ep:
        # Expert parallelism is off entirely (no --enable-expert-parallel);
        # there is no expert split to make hierarchical, and forcing one
        # would silently change what the flat TP config means.
        logger.warning_once(
            "A hierarchical MoE split is set but expert parallelism is off; "
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

    def group_aware(num_experts, ep_size, ep_rank, placement="linear"):
        # Read at load time, not patch time: the registry installs this before
        # the TP group exists. Ignore the caller's rank-derived ownership; the
        # experts this rank needs are its group's block, shared with the other
        # ranks of the group.
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


def check_ep_tp_config(config, split) -> None:
    """Refuse a deployment where the EP x TP partial sums would not meet.

    The split is computed over the TP group, and the only reduction of the
    routed output is the all-reduce over that group. With DP, PCP or
    sequence-parallel MoE the flat EP group spans ranks that all-reduce does
    not; with EPLB the expert map is not the linear block the split assumes.
    """
    if config.dp_size != 1 or config.pcp_size != 1 or config.is_sequence_parallel:
        raise ValueError(
            "TPU_MOE_EP_TP_SIZE supports DP=1, PCP=1 and no sequence-parallel "
            f"MoE only; got dp={config.dp_size}, pcp={config.pcp_size}, "
            f"sp={config.sp_size}."
        )
    if config.enable_eplb:
        raise ValueError("TPU_MOE_EP_TP_SIZE does not support EPLB.")
    if config.tp_size != 1 or config.ep_size != split[0] * split[2]:
        raise ValueError(
            "TPU_MOE_EP_TP_SIZE expects vLLM's flat EP config over the whole TP "
            f"group (tp_size=1, ep_size={split[0] * split[2]}); got "
            f"tp_size={config.tp_size}, ep_size={config.ep_size}."
        )


def install_ep_tp_split_patch() -> None:
    """Apply ``TPU_MOE_EP_TP_SIZE`` to every ``FusedMoE`` this process builds.

    A registry patch (worker init). Inert unless the env is set. The split is
    computed at each ``make`` call, after the TP group exists; a world size the
    env does not divide stays flat.
    """
    if envs.TPU_MOE_EP_TP_SIZE <= 1:
        return

    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    if getattr(FusedMoEParallelConfig, "_tpu_ep_tp_split_patch", False):
        return
    original_make = FusedMoEParallelConfig.make

    def ep_tp_make(*args, **kwargs):
        config = original_make(*args, **kwargs)
        split = hierarchical_split_or_none()
        if split is None:
            return config
        if not config.use_ep:
            return _split_parallel_config(config, split)  # warns, stays flat
        check_ep_tp_config(config, split)
        logger.info_once(
            "MoE experts split EP %d x TP %d (TPU_MOE_EP_TP_SIZE): this rank is "
            "ep_rank=%d, tp_rank=%d.",
            split[0],
            split[2],
            split[1],
            split[3],
        )
        return _split_parallel_config(config, split)

    FusedMoEParallelConfig.make = staticmethod(ep_tp_make)
    FusedMoEParallelConfig._tpu_ep_tp_split_patch = True
    align_ep_weight_filter()
    logger.info("Applied TPU patch: MoE EP x TP expert split (TPU_MOE_EP_TP_SIZE).")
