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

import torch
from torch.nn import Parameter
from vllm.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_torchtpu.layers.adapter.tp_all_reduce import (
    tp4_hier_all_reduce,
    tp4_hier_all_reduce_op,
)


@RowParallelLinear.register_oot
class TpuRowParallelLinear(RowParallelLinear):
    """RowParallelLinear that adds the bias after the TP all-reduce.

    Upstream fuses the bias into the GEMM on rank 0 only, so the per-rank FX
    graphs differ and the torch_tpu XLA backend hands the all_reduce mismatched
    channel IDs, breaking the collective. The bias is replicated on every rank,
    so adding it afterwards sums the same terms in a different order and keeps
    the graphs identical.

    With ``TPU_TP4_HIER_ALL_REDUCE`` on a four-core TP group on two chips,
    the reduction runs through the hierarchical all-reduce kernel instead.

    Only those cases are overridden; everything else delegates to
    ``super().forward()``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tp4_all_reduce = (
            tp4_hier_all_reduce_op()
            if self.reduce_results and self.tp_size == 4
            else None
        )

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        defer_bias = (
            self.reduce_results
            and self.tp_size > 1
            and not self.skip_bias_add
            and self.bias is not None
        )
        hier_op = self._tp4_all_reduce
        if not defer_bias and hier_op is None:
            return super().forward(input_)

        if self.input_is_parallel:
            input_parallel = input_
        else:
            split_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = split_input[self.tp_rank].contiguous()

        output_parallel = self.quant_method.apply(self, input_parallel, None)
        if hier_op is not None:
            output = tp4_hier_all_reduce(hier_op, output_parallel)
        else:
            output = tensor_model_parallel_all_reduce(output_parallel)
        if defer_bias:
            output = output + self.bias
            output_bias = None
        else:
            output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias
