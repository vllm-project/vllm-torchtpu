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
"""``GemmaRMSNorm`` whose normalization can move into the layers it feeds.

A norm whose every consumer is an FP8 linear that normalizes its own input
(``INPUT_RMS_NORM_ATTR``) is told so with ``defer_to_consumers``. From then on
it only adds the residual and returns that sum in place of the normalized
activation, so the normalized activation is computed in VMEM by the consumer
and never written to HBM.
"""

import torch
from vllm import ir
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

from vllm_torchtpu.layers.adapter.ir_ops import TPU_IR_PROVIDER, residual_sum


@CustomOp.register_oot(name="GemmaRMSNorm")
class TpuGemmaRMSNorm(GemmaRMSNorm):
    """``GemmaRMSNorm`` that can defer its normalization to its consumers."""

    deferred = False

    @staticmethod
    def can_defer() -> bool:
        """Whether consumers compute what this norm would: their
        normalization (``quantized_matmul.util.rms_norm``) is the TPU IR provider's."""
        return all(
            op.get_priority()[:1] == [TPU_IR_PROVIDER]
            for op in (ir.ops.rms_norm, ir.ops.fused_add_rms_norm)
        )

    def defer_to_consumers(self) -> None:
        """Stop normalizing: every consumer now normalizes its own input."""
        if not self.can_defer():
            raise RuntimeError("consumers would not compute this norm's output")
        self.deferred = True

    def consumer_weight(self) -> torch.Tensor:
        """The f32 weight a consumer normalizes with (Gemma's ``1 + w``)."""
        return self.weight.float() + 1.0

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not self.deferred:
            return super().forward(x, residual)
        if residual is None:
            return x
        residual = residual_sum(x, residual)
        return residual, residual
