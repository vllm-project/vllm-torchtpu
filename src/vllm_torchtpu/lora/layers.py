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

import torch
from vllm.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
from vllm.lora.layers import BaseLayerWithLoRA, RowParallelLinearWithLoRA


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

    def forward(
        self, input_: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        base_layer = self.base_layer
        defer_bias = (
            base_layer.reduce_results
            and self.tp_size > 1
            and not base_layer.skip_bias_add
            and base_layer.bias is not None
        )
        if not defer_bias:
            return super().forward(input_)

        if base_layer.input_is_parallel:
            input_parallel = input_
        else:
            split_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = split_input[self.tp_rank].contiguous()

        output_parallel = self.apply(input_parallel, None)
        output = tensor_model_parallel_all_reduce(output_parallel)
        output = output + base_layer.bias

        # skip_bias_add is False here, so there is no bias left to return.
        if not base_layer.return_bias:
            return output
        return output, None


def use_tpu_lora_layers(modules: Iterable[BaseLayerWithLoRA]) -> None:
    """Gives vLLM's LoRA layers their TPU forward.

    vLLM picks the LoRA layer for each module from a fixed list, so the layers
    are retyped once vLLM has created them. The TPU classes add no state.
    """
    for module in modules:
        if type(module) is RowParallelLinearWithLoRA:
            module.__class__ = TpuRowParallelLinearWithLoRA
