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

Sequence-parallel MoE layers gather and scatter over the EP group instead. A
step below the sequence-parallel row threshold (``sequence_parallel.min_rows``)
keeps its rows replicated over TP. With one DP engine that is every row on
every rank, so its dispatch is a no-op and its combine is an EP all-reduce.
Under data parallelism each engine holds only its own rows (replicated over its
TP ranks), so such a step gathers over DP like a non-sequence-parallel layer,
and its combine is the DP reduce-scatter followed by a TP all-reduce (the
sequence-parallel layer replicates its shared expert and skips the layer's own
TP reduction).
"""

import torch
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import get_dp_group, get_ep_group, get_tp_group

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import sequence_parallel, small_all_reduce_op


class TpuNullAll2AllManager(All2AllManagerBase):
    """Probe-only stub; TPU EP dispatch/combine never goes through it."""


def _packs_router_gather(
    hidden_states: torch.Tensor, router_logits: torch.Tensor
) -> bool:
    """Whether a replicated DP dispatch gathers hidden rows and router logits
    as one packed all-gather (TPU_DP_PACKED_ROUTER_GATHER, same dtype, 2D,
    same rows)."""
    return (
        envs.TPU_DP_PACKED_ROUTER_GATHER
        and hidden_states.dtype == router_logits.dtype
        and hidden_states.dim() == 2
        and router_logits.dim() == 2
        and hidden_states.shape[0] == router_logits.shape[0]
    )


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

    def _moe_group(self, is_sequence_parallel: bool):
        group = get_ep_group() if is_sequence_parallel else get_dp_group()
        return group if group.world_size > 1 else None

    def _replicated_dp_group(self, is_sequence_parallel: bool):
        """The DP group a replicated sequence-parallel step gathers over, or
        None when the step is not one or there is a single DP engine."""
        if not sequence_parallel.rows_are_replicated(is_sequence_parallel):
            return None
        group = get_dp_group()
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
        if sequence_parallel.rows_are_replicated(is_sequence_parallel):
            dp_group = self._replicated_dp_group(is_sequence_parallel)
            if dp_group is None:
                return hidden_states, router_logits

            def _gather_replicated_dp(h: torch.Tensor, lg: torch.Tensor):
                if _packs_router_gather(h, lg):
                    # One DP all-gather instead of two (each is latency-bound at
                    # decode sizes): pack the logits after the hidden columns
                    # and split the gathered rows. The values are unchanged.
                    width = h.shape[-1]
                    packed = dp_group.all_gather(
                        torch.cat([h, lg], dim=-1), dim=0
                    )
                    return packed[:, :width], packed[:, width:]
                return dp_group.all_gather(h, dim=0), dp_group.all_gather(
                    lg, dim=0
                )

            return small_all_reduce_op.dp_dispatch_or(
                hidden_states, router_logits, _gather_replicated_dp
            )
        group = self._moe_group(is_sequence_parallel)
        if group is None:
            return hidden_states, router_logits
        if is_sequence_parallel:
            hidden_states = group.all_gather(hidden_states, dim=0)
            router_logits = group.all_gather(router_logits, dim=0)
            return hidden_states, router_logits
        # DP-only dispatch: the fused Pallas all-gather when armed for the DP mesh.
        return small_all_reduce_op.dp_dispatch_or(
            hidden_states,
            router_logits,
            lambda h, lg: (
                group.all_gather(h, dim=0),
                group.all_gather(lg, dim=0),
            ),
        )

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
        if sequence_parallel.rows_are_replicated(is_sequence_parallel):
            dp_group = self._replicated_dp_group(is_sequence_parallel)
            if dp_group is None:
                return hidden_states, topk_weights, topk_ids
            hidden_states = dp_group.all_gather(hidden_states, dim=0)
            topk_weights = dp_group.all_gather(topk_weights, dim=0)
            topk_ids = dp_group.all_gather(topk_ids, dim=0)
            return hidden_states, topk_weights, topk_ids
        group = self._moe_group(is_sequence_parallel)
        if group is None:
            return hidden_states, topk_weights, topk_ids
        hidden_states = group.all_gather(hidden_states, dim=0)
        topk_weights = group.all_gather(topk_weights, dim=0)
        topk_ids = group.all_gather(topk_ids, dim=0)
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        group = self._moe_group(is_sequence_parallel)
        if group is None:
            return hidden_states
        if sequence_parallel.rows_are_replicated(is_sequence_parallel):
            dp_group = self._replicated_dp_group(is_sequence_parallel)
            if dp_group is None:
                # Every rank holds every row: sum the expert partials in place.
                return group.all_reduce(hidden_states)
            # This DP engine's rows, summed over the DP ranks of this TP rank,
            # then over TP: the EP-wide sum of the expert partials.
            out = small_all_reduce_op.dp_combine_or(
                hidden_states, lambda x: dp_group.reduce_scatter(x, dim=0)
            )
            return get_tp_group().all_reduce(out)
        if is_sequence_parallel:
            return group.reduce_scatter(hidden_states, dim=0)
        # DP-only combine: the Pallas reduce-scatter when armed for the DP mesh.
        return small_all_reduce_op.dp_combine_or(
            hidden_states, lambda x: group.reduce_scatter(x, dim=0)
        )
