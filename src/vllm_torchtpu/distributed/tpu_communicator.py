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

from typing import Any, Protocol, runtime_checkable

import torch
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import get_dp_group, get_ep_group

from vllm_torchtpu.distributed.hier_reduce_scatter import moe_combine_reduce_scatter


class ModelAllReduce(Protocol):
    """A model's own all-reduce for some of its tensors.

    Off by default: a model opts in by setting the attribute
    ``model_all_reduce`` (`MODEL_ALL_REDUCE_ATTR`) on its top-level module to
    an object with these two methods. Every other all-reduce, and every model
    that does not set it, keeps vLLM's all-reduce.
    """

    def setup(self, model: torch.nn.Module, vllm_config: Any, device: Any) -> bool:
        """Called once on every rank after the model loads (it may run
        collectives); returns whether `all_reduce` is used."""
        ...

    def all_reduce(self, ranks: Any, x: torch.Tensor) -> torch.Tensor | None:
        """`x` all-reduced over the group of `ranks`, or None for vLLM's
        all-reduce."""
        ...


MODEL_ALL_REDUCE_ATTR = "model_all_reduce"


@runtime_checkable
class SupportsModelAllReduce(Protocol):
    """Capability of a top-level model module that supplies its own
    all-reduce: the attribute ``model_all_reduce`` (`MODEL_ALL_REDUCE_ATTR`),
    a `ModelAllReduce` or None for vLLM's all-reduce."""

    model_all_reduce: ModelAllReduce | None


_model_all_reduce: ModelAllReduce | None = None


def install_model_all_reduce(
    model: torch.nn.Module, vllm_config: Any, device: Any
) -> bool:
    """Install the loaded model's `ModelAllReduce`, if it sets one; returns
    whether one is in use. Every rank loads the same model, so every rank
    installs or none does."""
    global _model_all_reduce
    _model_all_reduce = None
    # As a plain object: `nn.Module.__getattr__` would type every attribute.
    candidate: object = model
    if not isinstance(candidate, SupportsModelAllReduce):
        return False
    impl = candidate.model_all_reduce
    if impl is None or not impl.setup(model, vllm_config, device):
        return False
    _model_all_reduce = impl
    return True


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
        impl = _model_all_reduce
        out = None if impl is None else impl.all_reduce(self.ranks, input_)
        return super().all_reduce(input_) if out is None else out

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
        topk_weights = group.all_gather(topk_weights, dim=0)
        topk_ids = group.all_gather(topk_ids, dim=0)
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        group = self._moe_group(is_sequence_parallel)
        if group is None:
            return hidden_states
        return moe_combine_reduce_scatter(group, hidden_states)
