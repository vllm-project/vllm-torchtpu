# Copyright 2026 Google LLC
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
"""LoRA layers whose forward has to differ from vLLM's on TPU."""

from collections.abc import Iterable
from typing import cast

import torch
from vllm.lora.layers import BaseLayerWithLoRA, RowParallelLinearWithLoRA
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_torchtpu.layers.adapter.linear import (
    defers_bias,
    forward_with_deferred_bias,
)


class TpuRowParallelLinearWithLoRA(RowParallelLinearWithLoRA):
    """RowParallelLinearWithLoRA that adds the bias after the TP all-reduce.

    vLLM's forward repeats RowParallelLinear's and passes the bias into the
    GEMM on rank 0 only, so with tensor parallelism the ranks trace different
    graphs and the all-reduce breaks, as TpuRowParallelLinear describes. The
    bias is replicated on every rank, so adding it after the all-reduce sums
    the same terms and keeps the graphs identical.

    Only the deferred case is overridden; everything else delegates to
    ``super().forward()``.
    """

    # TODO: drop this override once vLLM adds the bias after the all-reduce:
    # https://github.com/vllm-project/vllm/issues/70000
    def forward(
        self, input_: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        # vLLM types base_layer as LinearBase; this class wraps RowParallelLinear.
        base_layer = cast(RowParallelLinear, self.base_layer)
        if not defers_bias(base_layer):
            return super().forward(input_)
        return forward_with_deferred_bias(
            base_layer, input_, lambda x: self.apply(x, None)
        )


def use_tpu_lora_layers(modules: Iterable[BaseLayerWithLoRA]) -> None:
    """Gives vLLM's LoRA layers their TPU forward.

    vLLM picks the LoRA layer for each module from a fixed list, so the layers
    are retyped once vLLM has created them. The TPU classes add no state.
    """
    # TODO: register the TPU layer through vLLM once it has a hook for
    # out-of-tree LoRA layers, and stop retyping:
    # https://github.com/vllm-project/vllm/issues/70000
    for module in modules:
        if type(module) is RowParallelLinearWithLoRA:
            module.__class__ = TpuRowParallelLinearWithLoRA
