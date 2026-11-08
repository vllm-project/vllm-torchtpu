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

from vllm_torchtpu.distributed.dp_exchange import (
    dp_pair_all_gather,
    dp_pair_exchange_ready,
)

# Dispatch gathers its tensors as one collective up to this many tokens per
# rank. At 256 tokens of hidden size 6144 the pack and split copies move about
# 20 MB of HBM (~5 us), under the ~10 us fixed latency of the saved all-gather.
PACKED_DISPATCH_MAX_TOKENS = 256


def _to_int16_columns(t: torch.Tensor) -> torch.Tensor:
    """[N, ...] 16- or 32-bit tensor as [N, k] int16 holding its bits.

    A 32-bit word is split into its high and low halves with shifts rather
    than a size-changing ``view(dtype)``, whose element order on the device
    need not match PyTorch's.
    """
    rows = t.contiguous().reshape(t.shape[0], -1)
    if t.element_size() == 2:
        return rows.view(torch.int16)
    words = rows.view(torch.int32)
    hi = (words >> 16).to(torch.int16)
    lo = ((words << 16) >> 16).to(torch.int16)
    return torch.stack((hi, lo), dim=-1).reshape(t.shape[0], -1)


def _from_int16_columns(columns: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_to_int16_columns`` for rows shaped like ``like``'s."""
    if like.element_size() == 2:
        rows = columns.contiguous().view(like.dtype)
    else:
        halves = columns.reshape(columns.shape[0], -1, 2).to(torch.int32)
        words = (halves[..., 0] << 16) | (halves[..., 1] & 0xFFFF)
        rows = words.view(like.dtype)
    return rows.reshape(-1, *like.shape[1:])


def _all_gather_packed(
    group: GroupCoordinator, tensors: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, ...]:
    """All-gather row-aligned tensors along dim 0 in one int16 collective.

    Every output is bit-identical to gathering its tensor alone: only integer
    ops touch the bits. A float concatenation (lowered by XLA as pad(-inf) +
    max) or a float collective need not keep the NaN, denormal or signed-zero
    patterns that reinterpreted words take.
    """
    columns = [_to_int16_columns(t) for t in tensors]
    gathered = group.all_gather(torch.cat(columns, dim=-1), dim=0)
    parts = torch.split(gathered, [c.shape[-1] for c in columns], dim=-1)
    return tuple(_from_int16_columns(p, t) for p, t in zip(parts, tensors))


def _gathers_packed(tensors: tuple[torch.Tensor, ...]) -> bool:
    # Decided while tracing. Only the backbone is traced per token bucket
    # group (vllm_torchtpu.compilation.shape_variants), so a drafter graph must
    # not compare its token count; the tag is checked first.
    return (
        vllm_compilation_backends.model_tag == "backbone"
        and 0 < tensors[0].shape[0] <= PACKED_DISPATCH_MAX_TOKENS
        and all(t.element_size() in (2, 4) for t in tensors)
    )


def _exchanges_dp_pair(
    tensors: tuple[torch.Tensor, ...], is_sequence_parallel: bool
) -> bool:
    # The op is built only when the DP group is a pair (see
    # prepare_dp_pair_exchange). It takes any nonempty batch, so the choice
    # holds for every token bucket and for drafter graphs alike.
    return (
        not is_sequence_parallel
        and dp_pair_exchange_ready()
        and tensors[0].shape[0] > 0
        and all(t.element_size() in (2, 4) for t in tensors)
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
        if _exchanges_dp_pair((hidden_states, router_logits), is_sequence_parallel):
            return dp_pair_all_gather(hidden_states), dp_pair_all_gather(router_logits)
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
        tensors = (hidden_states, topk_weights, topk_ids)
        if _exchanges_dp_pair(tensors, is_sequence_parallel):
            return tuple(dp_pair_all_gather(t) for t in tensors)
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
