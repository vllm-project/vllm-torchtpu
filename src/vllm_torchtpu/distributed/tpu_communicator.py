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

Dispatch gathers all of its per-token tensors (hidden states plus router logits,
or plus top-k weights and ids) in one all-gather. At decode batch sizes these
collectives are latency-bound, so the number of collectives per MoE layer, not
their size, sets the cost.
"""

import math

import torch
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import get_dp_group, get_ep_group

from vllm_torchtpu.distributed.hier_reduce_scatter import moe_combine_reduce_scatter

# Pack only decode-sized steps. There the gathers are latency-bound and one
# collective instead of two or three is a net win; at prefill sizes the
# concatenate / split copies cost more than the saved collective (measured on
# v7x: 32K-context prefill took 1.4x as long with packing always on).
_PACK_MAX_TOKENS = 512


def _packed_all_gather(group, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """All-gather per-token tensors along dim 0 with a single collective.

    The rows of every tensor are bit-cast to the first tensor's dtype (the
    hidden states) and concatenated along the last dim, gathered once, then
    split and cast back. Using the hidden-state dtype as the carrier keeps the
    bulk of the payload in its native layout; a byte carrier forces TPU to
    repack every element. An all-gather only copies, so the result is
    bit-identical to gathering each tensor separately. Tensors whose element
    size is not a multiple of the carrier's, and steps above
    `_PACK_MAX_TOKENS` tokens, are gathered tensor by tensor.
    """
    num_tokens = tensors[0].shape[0]
    if any(t.shape[0] != num_tokens for t in tensors):
        raise ValueError("packed all-gather needs the same token count per tensor")
    carrier = tensors[0].dtype
    if num_tokens > _PACK_MAX_TOKENS or any(
        t.dtype.itemsize % carrier.itemsize for t in tensors
    ):
        return [group.all_gather(t, dim=0) for t in tensors]
    rows = [
        t.contiguous().reshape(num_tokens, math.prod(t.shape[1:])).view(carrier)
        for t in tensors
    ]
    gathered = group.all_gather(torch.cat(rows, dim=1), dim=0)
    parts = gathered.split([r.shape[1] for r in rows], dim=1)
    return [
        part.contiguous().view(t.dtype).reshape(gathered.shape[0], *t.shape[1:])
        for t, part in zip(tensors, parts)
    ]


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
        hidden_states, router_logits = _packed_all_gather(
            group, [hidden_states, router_logits]
        )
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
        hidden_states, topk_weights, topk_ids = _packed_all_gather(
            group, [hidden_states, topk_weights, topk_ids]
        )
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        group = self._moe_group(is_sequence_parallel)
        if group is None:
            return hidden_states
        return moe_combine_reduce_scatter(group, hidden_states)
