# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Gated (SwiGLU / SITU) MLP with optional token-sharded collectives."""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, Protocol
from unittest.mock import patch

import torch
from torch import nn
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


class TokenCollectives(Protocol):
    """Gathers and reduce-scatters token shards along dim 0."""

    def all_gather(self, x: torch.Tensor, /, dim: int) -> torch.Tensor: ...

    def reduce_scatter(self, x: torch.Tensor, /, dim: int) -> torch.Tensor: ...


class SituAndMul(nn.Module):
    """The SITU gated activation.

    Keep this implementation local because released vLLM versions do not
    expose ``SituAndMul`` from their generic activation module.
    """

    def __init__(self, beta: float, linear_beta: float | None) -> None:
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return gate * up


class GatedMLP(nn.Module):
    """The standard vLLM merged-projection gated MLP.

    Args:
        tp_group: Shards the MLP over this group instead of the TP group
            (for example the cores of one chip); the output is reduced over
            it.
        sp_group: Collectives for sequence-parallel calls, which gather the
            token shards before the MLP and reduce-scatter its output.
            ``None`` uses ``tp_group`` or the TP group.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        reduce_results: bool = True,
        situ_beta: float | None = None,
        situ_linear_beta: float | None = None,
        tp_group: GroupCoordinator | None = None,
        sp_group: TokenCollectives | None = None,
    ) -> None:
        super().__init__()
        self.tp_group = tp_group
        self.sp_group = sp_group
        # Reuse the existing merged/row weight loaders with chip-local rank
        # metadata. Scope the override to construction of these two layers.
        with ExitStack() as stack:
            if tp_group is not None:
                module = "vllm.model_executor.layers.linear"
                stack.enter_context(
                    patch(
                        f"{module}.get_tensor_model_parallel_rank",
                        return_value=tp_group.rank_in_group,
                    )
                )
                stack.enter_context(
                    patch(
                        f"{module}.get_tensor_model_parallel_world_size",
                        return_value=tp_group.world_size,
                    )
                )
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size,
                [intermediate_size, intermediate_size],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_up_proj",
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                reduce_results=reduce_results,
                prefix=f"{prefix}.down_proj",
            )
        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        elif hidden_act == "situ":
            self.act_fn = SituAndMul(
                beta=situ_beta or 1.0,
                linear_beta=situ_linear_beta,
            )
        else:
            raise ValueError(f"Unsupported gated MLP activation {hidden_act!r}")

    def forward(
        self, hidden_states: torch.Tensor, sequence_parallel: bool = False
    ) -> torch.Tensor:
        group: TokenCollectives | GroupCoordinator | None = self.tp_group
        if sequence_parallel:
            group = self.sp_group or group or get_tp_group()
            hidden_states = group.all_gather(hidden_states, dim=0)
        hidden_states, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        if sequence_parallel or self.tp_group is not None:
            # Input features already match this rank's row-parallel shard.
            hidden_states = self.down_proj.quant_method.apply(
                self.down_proj, hidden_states, None
            )
            if sequence_parallel:
                assert group is not None
                return group.reduce_scatter(hidden_states, dim=0)
            assert self.tp_group is not None
            return self.tp_group.all_reduce(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states
