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

try:
    from vllm.distributed.device_communicators.base_device_communicator import (
        All2AllManagerBase,
        DeviceCommunicatorBase,
    )
    from vllm.distributed.parallel_state import get_dp_group
except ImportError:  # pragma: no cover - CPU test environment without vLLM
    class All2AllManagerBase:
        pass

    class DeviceCommunicatorBase:
        def __init__(self, *args, **kwargs):
            self.use_all2all = False
            self.cpu_group = None
            self.ranks = []

        def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
            return input_

    def get_dp_group():
        return None

from vllm_torchtpu.layers.adapter import small_all_reduce_op


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

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # Every TP all-reduce (row-parallel output projections, GDN out-proj,
        # vocab embedding, MoE reduce_results) comes through here.
        return small_all_reduce_op.all_reduce_or(input_, self.ranks, super().all_reduce)

    def _dp_gather(self, x: torch.Tensor) -> torch.Tensor:
        return get_dp_group().all_gather(x, dim=0)

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
        dp = get_dp_group()
        if dp.world_size == 1:
            return hidden_states, router_logits
        hidden_states = self._dp_gather(hidden_states)
        router_logits = self._dp_gather(router_logits)
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
        dp = get_dp_group()
        if dp.world_size == 1:
            return hidden_states, topk_weights, topk_ids
        hidden_states = self._dp_gather(hidden_states)
        topk_weights = self._dp_gather(topk_weights)
        topk_ids = self._dp_gather(topk_ids)
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        dp = get_dp_group()
        if dp.world_size == 1:
            return hidden_states
        return small_all_reduce_op.dp_combine_or(
            hidden_states, lambda x: dp.reduce_scatter(x, dim=0)
        )
