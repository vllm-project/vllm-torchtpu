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

from collections.abc import Callable

import torch
from torch.nn import Parameter
from vllm.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import RowParallelLinear


def defers_bias(layer: RowParallelLinear) -> bool:
    """Whether the layer reduces over ranks and has a bias to add to the sum."""
    return (
        layer.reduce_results
        and layer.tp_size > 1
        and not layer.skip_bias_add
        and layer.bias is not None
    )


def forward_with_deferred_bias(
    layer: RowParallelLinear,
    input_: torch.Tensor,
    gemm: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
    """RowParallelLinear's forward with the bias added after the all-reduce.

    ``gemm`` computes this rank's product without a bias. Call this only when
    ``defers_bias(layer)`` holds.
    """
    if layer.input_is_parallel:
        input_parallel = input_
    else:
        split_input = split_tensor_along_last_dim(input_, num_partitions=layer.tp_size)
        input_parallel = split_input[layer.tp_rank].contiguous()

    output = tensor_model_parallel_all_reduce(gemm(input_parallel))
    output = output + layer.bias

    # skip_bias_add is False here, so there is no bias left to return.
    if not layer.return_bias:
        return output
    return output, None


@RowParallelLinear.register_oot
class TpuRowParallelLinear(RowParallelLinear):
    """RowParallelLinear that adds the bias after the TP all-reduce.

    Upstream fuses the bias into the GEMM on rank 0 only, so the per-rank FX
    graphs differ and the torch_tpu XLA backend hands the all_reduce mismatched
    channel IDs, breaking the collective. The bias is replicated on every rank,
    so adding it afterwards sums the same terms in a different order and keeps
    the graphs identical.

    Only the deferred case is overridden; everything else delegates to
    ``super().forward()``.
    """

    # TODO: drop this override once vLLM adds the bias after the all-reduce:
    # https://github.com/vllm-project/vllm/issues/70000
    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if not defers_bias(self):
            return super().forward(input_)
        return forward_with_deferred_bias(
            self, input_, lambda x: self.quant_method.apply(self, x, None)
        )
