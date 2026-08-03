# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TPU device communicator for EP across data-parallel engines.

Ordinary DP dispatch/combine communicates across the DP group. Sequence-
parallel MoE instead communicates across the full EP group so that TP token
shards are routed exactly once. Callers must pass equal token counts per rank.
"""

import torch
import torch.distributed._functional_collectives as funcol
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase, DeviceCommunicatorBase)
from vllm.distributed.parallel_state import get_dp_group, get_ep_group


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

    def custom_all_gather(self,
                          input_: torch.Tensor,
                          dim: int = 0) -> torch.Tensor:
        """SP all-gather without allocating a symbolically sized output.

        The base communicator creates the destination with ``torch.empty``.
        TorchTPU cannot export that allocation when the token dimension is a
        SymInt. Functional collectives carry the symbolic result shape through
        the graph and are lowered by the distributed backend instead.
        """
        if self.world_size == 1:
            return input_
        if dim < 0:
            dim += input_.dim()
        input_shape = input_.shape
        gathered = funcol.wait_tensor(
            funcol.all_gather_tensor(input_.contiguous(), 0,
                                     self.device_group))
        gathered = gathered.reshape((self.world_size, ) + input_shape)
        gathered = gathered.movedim(0, dim)
        return gathered.reshape(input_shape[:dim] +
                                (self.world_size * input_shape[dim], ) +
                                input_shape[dim + 1:])

    def custom_reduce_scatter(self,
                              input_: torch.Tensor,
                              dim: int = 0) -> torch.Tensor:
        """SP reduce-scatter without a symbolically sized allocation."""
        if self.world_size == 1:
            return input_
        if dim < 0:
            dim += input_.dim()
        input_ = input_.movedim(dim, 0).contiguous()
        output = funcol.wait_tensor(
            funcol.reduce_scatter_tensor(input_, "sum", 0, self.device_group))
        return output.movedim(0, dim).contiguous()

    @staticmethod
    def _dispatch_group(is_sequence_parallel: bool):
        return get_ep_group() if is_sequence_parallel else get_dp_group()

    @staticmethod
    def _all_gather(group, input_: torch.Tensor,
                    is_sequence_parallel: bool) -> torch.Tensor:
        if is_sequence_parallel:
            return group.device_communicator.custom_all_gather(input_)
        return group.all_gather(input_, dim=0)

    @staticmethod
    def _reduce_scatter(group, input_: torch.Tensor,
                        is_sequence_parallel: bool) -> torch.Tensor:
        if is_sequence_parallel:
            return group.device_communicator.custom_reduce_scatter(input_)
        return group.reduce_scatter(input_, dim=0)

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors=None,
    ):
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported by TpuDeviceCommunicator")
        group = self._dispatch_group(is_sequence_parallel)
        if group.world_size == 1:
            return hidden_states, router_logits
        hidden_states = self._all_gather(group, hidden_states,
                                         is_sequence_parallel)
        router_logits = self._all_gather(group, router_logits,
                                         is_sequence_parallel)
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
                "extra_tensors is not supported by TpuDeviceCommunicator")
        group = self._dispatch_group(is_sequence_parallel)
        if group.world_size == 1:
            return hidden_states, topk_weights, topk_ids
        hidden_states = self._all_gather(group, hidden_states,
                                         is_sequence_parallel)
        topk_weights = self._all_gather(group, topk_weights,
                                        is_sequence_parallel)
        topk_ids = self._all_gather(group, topk_ids, is_sequence_parallel)
        return hidden_states, topk_weights, topk_ids

    def combine(self,
                hidden_states: torch.Tensor,
                is_sequence_parallel: bool = False) -> torch.Tensor:
        group = self._dispatch_group(is_sequence_parallel)
        if group.world_size == 1:
            return hidden_states
        return self._reduce_scatter(group, hidden_states, is_sequence_parallel)
