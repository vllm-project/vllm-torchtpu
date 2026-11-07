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
from vllm.compilation import backends as vllm_compilation_backends
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import GroupCoordinator, get_dp_group, get_ep_group

# Dispatch gathers its tensors as one collective up to this many tokens per
# rank. At 256 tokens of hidden size 6144 the pack and split copies move about
# 20 MB of HBM (~5 us), under the ~10 us fixed latency of the saved all-gather.
PACKED_DISPATCH_MAX_TOKENS = 256

_INT_DTYPE_BY_SIZE = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _all_gather_packed(
    group: GroupCoordinator, tensors: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, ...]:
    """All-gather row-aligned tensors along dim 0 in one collective.

    Rows are reinterpreted as integers of the first tensor's element size and
    concatenated on the last dim, then gathered in the first tensor's dtype.
    The concatenation is integer: XLA lowers a float one as pad(-inf) + max,
    which need not keep NaN or denormal bit patterns of the reinterpreted
    words. Every output is bit-identical to gathering its tensor alone.
    """
    int_dtype = _INT_DTYPE_BY_SIZE[tensors[0].element_size()]
    rows = [t.contiguous().view(int_dtype).reshape(t.shape[0], -1) for t in tensors]
    packed = torch.cat(rows, dim=-1).view(tensors[0].dtype)
    gathered = group.all_gather(packed, dim=0).view(int_dtype)
    parts = torch.split(gathered, [r.shape[-1] for r in rows], dim=-1)
    return tuple(
        part.contiguous().view(t.dtype).reshape(-1, *t.shape[1:])
        for part, t in zip(parts, tensors)
    )


def _gathers_packed(tensors: tuple[torch.Tensor, ...]) -> bool:
    # Decided while tracing. Only the backbone is traced per token bucket
    # group (vllm_torchtpu.compilation.shape_variants), so a drafter graph must
    # not compare its token count; the tag is checked first.
    first = tensors[0]
    return (
        vllm_compilation_backends.model_tag == "backbone"
        and 0 < first.shape[0] <= PACKED_DISPATCH_MAX_TOKENS
        and all(t.element_size() % first.element_size() == 0 for t in tensors)
    )


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
        if _gathers_packed((hidden_states, router_logits)):
            return _all_gather_packed(group, (hidden_states, router_logits))
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
        if _gathers_packed((hidden_states, topk_weights, topk_ids)):
            return _all_gather_packed(group, (hidden_states, topk_weights, topk_ids))
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
        return group.reduce_scatter(hidden_states, dim=0)
