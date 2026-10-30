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

from vllm_torchtpu import transforms


@RowParallelLinear.register_oot
class TpuRowParallelLinear(RowParallelLinear):
    """RowParallelLinear that adds the bias after the TP all-reduce.

    Upstream fuses the bias into the GEMM on rank 0 only, so the per-rank FX
    graphs differ and the torch_tpu XLA backend hands the all_reduce mismatched
    channel IDs, breaking the collective. The bias is replicated on every rank,
    so adding it afterwards sums the same terms in a different order and keeps
    the graphs identical.

    Only the deferred case (or an active ``transforms.microbatch`` region) is
    overridden; everything else delegates to ``super().forward()``.
    """

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
        if not defer_bias and not transforms.is_microbatch_active():
            return super().forward(input_)

        with transforms.resource("TC"):
            if self.input_is_parallel:
                input_parallel = input_
            else:
                split_input = split_tensor_along_last_dim(
                    input_, num_partitions=self.tp_size
                )
                input_parallel = split_input[self.tp_rank].contiguous()

            bias_ = (
                None
                if (defer_bias or self.tp_rank > 0 or self.skip_bias_add)
                else self.bias
            )
            output_parallel = self.quant_method.apply(
                self, input_parallel, bias_
            )

        if self.reduce_results and self.tp_size > 1:
            with transforms.resource("SC"):
                output = tensor_model_parallel_all_reduce(output_parallel)
                if defer_bias:
                    output = output + self.bias
        else:
            output = output_parallel

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
