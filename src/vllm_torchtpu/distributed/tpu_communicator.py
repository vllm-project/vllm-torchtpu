# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TPU device communicator for EP across data-parallel engines.

Dispatch gathers tokens across DP ranks that share the same TP rank. Combine
sums those matching-TP partials across DP and returns this DP rank's token
slice. vLLM's existing ``reduce_results`` TP all-reduce remains responsible for
the final TP-side reduction. Callers must pass equal token counts on each DP
rank.
"""

import torch
from vllm.compilation import backends as vllm_compilation_backends
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import get_dp_group, get_ep_group

from vllm_torchtpu.distributed.ep_combine_layout import (
    COMBINE_BLOCKS,
    combine_piece_rows,
)


def combine_splits() -> bool:
    """Whether ``combine`` may split into pieces.

    Needs at least two row blocks per rank, so every piece spans both halves
    of every rank's slice. Fixed at trace time. Only the backbone is traced
    once per bucket group; the MTP drafter has one graph for every bucket, so
    it must not compare its token count and keeps one reduce-scatter. The
    fused MoE asks the same question to lay out the combine's input, so the
    DP and EP groups must agree in size: combine reduce-scatters over either.
    It lays it out only for expert-parallel layers, so the split needs EP.
    """
    dp_group = get_dp_group()
    dp_size = dp_group.world_size
    return (
        vllm_compilation_backends.model_tag == "backbone"
        and dp_size > 1
        and dp_group.device_communicator.expert_parallel
        and get_ep_group().world_size == dp_size
        and COMBINE_BLOCKS % (2 * dp_size) == 0
    )


def _to_words(tensor: torch.Tensor) -> torch.Tensor:
    """The tensor's bits as flat int32, one word per element below 4 bytes."""
    flat = tensor.reshape(-1)
    if flat.element_size() >= 4:
        return flat.view(torch.int32)
    narrow = {1: torch.int8, 2: torch.int16}[flat.element_size()]
    return flat.view(narrow).to(torch.int32)


def _from_words(words: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_to_words``: rows shaped like ``like``'s trailing dims."""
    if like.element_size() >= 4:
        flat = words.view(like.dtype)
    else:
        narrow = {1: torch.int8, 2: torch.int16}[like.element_size()]
        flat = words.to(narrow).view(like.dtype)
    return flat.view(-1, *like.shape[1:])


def _all_gather_packed(group, *tensors: torch.Tensor) -> list[torch.Tensor]:
    """All-gather row-aligned tensors as one flat int32 buffer.

    Narrow ``[T, k]`` routing tensors would pad ``k`` to a full lane row on
    TPU and take a collective each; flattened side by side they travel
    lane-dense in one. Bit-exact: only the bits move.
    """
    words = [_to_words(t) for t in tensors]
    gathered = group.all_gather(torch.cat(words), dim=0)
    per_rank = gathered.view(group.world_size, -1).split(
        [w.shape[0] for w in words], dim=1
    )
    return [_from_words(part.reshape(-1), t) for part, t in zip(per_rank, tensors)]


class TpuNullAll2AllManager(All2AllManagerBase):
    """Probe-only stub; TPU EP dispatch/combine never goes through it."""


class TpuDeviceCommunicator(DeviceCommunicatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TPU performs EP dispatch/combine directly (see dispatch/combine
        # below), not through an all2all manager. The stub only satisfies
        # vLLM's DP>1 + MoE fault-tolerance probe
        # (support_fault_tolerance=False); it is never used for compute.
        if self.use_all2all:
            self.all2all_manager = TpuNullAll2AllManager(self.cpu_group)
        # Read here, under the config the groups are built with, not at trace.
        config = get_current_vllm_config_or_none()
        self.expert_parallel = bool(
            config is not None and config.parallel_config.enable_expert_parallel
        )

    def _moe_group(self, is_sequence_parallel: bool):
        group = get_ep_group() if is_sequence_parallel else get_dp_group()
        return group if group.world_size > 1 else None

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors=None,
    ):
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported by TpuDeviceCommunicator"
            )
        group = self._moe_group(is_sequence_parallel)
        if group is None:
            return hidden_states, router_logits
        hidden_states = group.all_gather(hidden_states, dim=0)
        router_logits = group.all_gather(router_logits, dim=0)
        return hidden_states, router_logits

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors=None,
    ):
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported by TpuDeviceCommunicator"
            )
        group = self._moe_group(is_sequence_parallel)
        if group is None:
            return hidden_states, topk_weights, topk_ids
        hidden_states = group.all_gather(hidden_states, dim=0)
        topk_weights, topk_ids = _all_gather_packed(group, topk_weights, topk_ids)
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        group = self._moe_group(is_sequence_parallel)
        if group is None:
            return hidden_states
        piece_rows = combine_piece_rows(hidden_states.shape[0])
        if not piece_rows or not combine_splits():
            return group.reduce_scatter(hidden_states, dim=0)
        # The input is piece-major (ep_combine_layout): each piece is a slice,
        # and its reduce-scatter can start once the unpermute has written it.
        scattered = [
            group.reduce_scatter(part, dim=0)
            for part in hidden_states.split([COMBINE_BLOCKS * r for r in piece_rows])
        ]
        # This rank's rows are its blocks, each one sub-block from every piece.
        blocks = COMBINE_BLOCKS // group.world_size
        return torch.cat(
            [
                part.unflatten(0, (blocks, -1))[b]
                for b in range(blocks)
                for part in scattered
            ]
        )
