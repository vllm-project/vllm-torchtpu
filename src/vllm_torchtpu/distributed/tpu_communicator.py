# Copyright 2025 Google LLC
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
"""TPU device communicator for EP across data-parallel engines.

Dispatch gathers tokens across DP ranks that share the same TP rank. Combine
sums those matching-TP partials across DP and returns this DP rank's token
slice. vLLM's existing ``reduce_results`` TP all-reduce remains responsible for
the final TP-side reduction. Callers must pass equal token counts on each DP
rank.
"""

import torch
from torch_tpu._internal import pallas
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_dp_group,
    get_ep_group,
)

from vllm_torchtpu.kernels.compact_tiling import pin_compact_tiling

_pin_compact_tiling_op = pallas.jax_op("pallas::pin_compact_tiling", pin_compact_tiling)
_pin_compact_tiling_op.register_fake(lambda x: torch.empty_like(x))


def _gather_hidden_states(
    group: GroupCoordinator, hidden_states: torch.Tensor
) -> torch.Tensor:
    """All-gather the token rows in the tiling the Pallas MoE kernels read.

    Pinned on this rank's rows, any relayout XLA needs for the kernels runs
    before the gather, on 1/world_size of the bytes it would move after it.
    """
    return group.all_gather(_pin_compact_tiling_op(hidden_states), dim=0)


def gather_topk(
    group: GroupCoordinator,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *row_values: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """All-gather the top-k weights and ids, and any other per-row
    `row_values`, as one int32 collective; returns them in that order.

    Each is a few bytes per row, so each gather costs about its fixed
    latency; one gather pays it once. Floats ride as f32 bits, which hold
    any float of at most 32 bits exactly.
    """
    tensors = (topk_weights, topk_ids, *row_values)
    if topk_ids.dtype != torch.int32 or not all(
        t.shape[:-1] == topk_ids.shape[:-1]
        and (
            t.dtype == torch.int32 or (t.is_floating_point() and t.element_size() <= 4)
        )
        for t in tensors
    ):
        return tuple(group.all_gather(t, dim=0) for t in tensors)
    words = [
        t if t.dtype == torch.int32 else t.float().view(torch.int32) for t in tensors
    ]
    packed = group.all_gather(torch.cat(words, dim=-1), dim=0)
    as_float = packed.view(torch.float32)
    gathered = []
    start = 0
    for t in tensors:
        end = start + t.shape[-1]
        if t.dtype == torch.int32:
            gathered.append(packed[..., start:end].contiguous())
        else:
            gathered.append(as_float[..., start:end].to(t.dtype))
        start = end
    return tuple(gathered)


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
        hidden_states = _gather_hidden_states(group, hidden_states)
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
        hidden_states = _gather_hidden_states(group, hidden_states)
        topk_weights, topk_ids = gather_topk(group, topk_weights, topk_ids)
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        group = self._moe_group(is_sequence_parallel)
        if group is None:
            return hidden_states
        return group.reduce_scatter(hidden_states, dim=0)
