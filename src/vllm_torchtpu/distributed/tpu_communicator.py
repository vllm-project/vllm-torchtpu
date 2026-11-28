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

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import get_dp_group, get_ep_group

from vllm_torchtpu.distributed.hier_reduce_scatter import (
    _agreed_reason,
    moe_combine_reduce_scatter,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class ModelAllReduce(Protocol):
    """A model's own all-reduce for the tensors of one process group.

    Off by default: a model opts in through `SupportsModelAllReduce`. Every
    other all-reduce, and every model that does not opt in, keeps vLLM's
    all-reduce.
    """

    def setup(
        self, model: torch.nn.Module, vllm_config: VllmConfig, device: torch.device
    ) -> Sequence[int] | None:
        """Called once per `load_model` on every rank, after the model loads
        and outside the traced forward pass (it may run collectives).

        Returns the ranks of the one group whose all-reduces `all_reduce`
        serves, or None to decline. If any rank declines, no rank installs
        the implementation.
        """
        ...

    def all_reduce(self, ranks: Sequence[int], x: torch.Tensor) -> torch.Tensor | None:
        """`x` all-reduced over the group of `ranks`, or None for vLLM's
        all-reduce. Only called for the group `setup` returned.

        Runs while Dynamo traces the compiled forward pass. vLLM traces it
        once and drops the Dynamo guards (`compilation/shape_variants.py`),
        so the choice between a result and None stays in the compiled graph:
        base it only on what is fixed for the traced program (shape, dtype,
        group), never on tensor values or on state that changes later. It
        also receives the all-reduces of a TP-sharded draft model and of the
        runner itself (`tpu_runner.py`) on that group.
        """
        ...


@runtime_checkable
class SupportsModelAllReduce(Protocol):
    """Capability of a top-level model class that supplies its own
    all-reduce."""

    def get_model_all_reduce(self) -> ModelAllReduce | None:
        """The model's `ModelAllReduce`, or None for vLLM's all-reduce."""
        ...


class _Installed:
    def __init__(self, impl: ModelAllReduce, ranks: Sequence[int]) -> None:
        self.impl = impl
        self.ranks = tuple(int(r) for r in ranks)


_model_all_reduce: _Installed | None = None


def install_model_all_reduce(
    model: torch.nn.Module, vllm_config: VllmConfig, device: torch.device
) -> bool:
    """Install the loaded model's `ModelAllReduce`, if it supplies one;
    returns whether one is in use. Every rank loads the same model; the
    ranks then agree, so every rank installs or none does."""
    global _model_all_reduce
    _model_all_reduce = None
    if not isinstance(model, SupportsModelAllReduce):
        return False
    impl = model.get_model_all_reduce()
    if impl is None:
        return False
    ranks = impl.setup(model, vllm_config, device)
    reason = _agreed_reason(None if ranks is not None else "setup() declined")
    if reason is not None:
        logger.warning_once(
            "The model's own all-reduce is not installed; vLLM's all-reduce "
            "is used: %s.",
            reason,
        )
        return False
    assert ranks is not None
    _model_all_reduce = _Installed(impl, ranks)
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
        installed = _model_all_reduce
        out = None
        if installed is not None and tuple(self.ranks) == installed.ranks:
            out = installed.impl.all_reduce(self.ranks, input_)
        return super().all_reduce(input_) if out is None else out

    def destroy(self) -> None:
        # The installed implementation holds the old engine's setup() state.
        global _model_all_reduce
        _model_all_reduce = None
        super().destroy()

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
