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
"""Out-of-tree (OOT) custom wrappers for GemmaRMSNorm."""

import torch
import torch.nn as nn
from vllm import ir
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedGemmaNormWeightFold,
)


@GemmaRMSNorm.register_oot
class VllmTPUGemmaRMSNorm(GemmaRMSNorm):
    """Holds ``1 + w`` in fp32 so forward skips the per-step upcast-and-add.

    Gemma normalizes with ``x * (1 + w)``, and upstream's ``forward_native``
    rebuilds ``self.weight.float() + 1.0`` on every call. The result is a
    function of the checkpoint alone, so it is folded into the parameter once,
    after loading, by `VllmUnquantizedGemmaNormWeightFold` -- see that class
    for why the fold is carried by a `quant_method` and what it costs.

    Note that ``self.weight`` therefore no longer holds the checkpoint value.
    Nothing reads it raw today besides the forward below, but the same hazard
    is documented at length in ``layers/adapter/linear_common.py`` for flipped
    linear layouts.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(hidden_size, eps)
        # fp32 explicitly: __init__ runs under
        # `set_default_torch_dtype(model_config.dtype)`, so the upstream
        # parameter would be bf16, which cannot hold `1 + w` at the precision
        # the per-step `.float()` upcast was providing.
        self.weight = nn.Parameter(
            torch.zeros(hidden_size, dtype=torch.float32), requires_grad=False
        )
        # The sole reason a norm carries a quantization method: it is the only
        # hook in vLLM's post-load sweep that a plain norm layer can reach.
        self.quant_method = VllmUnquantizedGemmaNormWeightFold()

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        if residual is None:
            return ir.ops.rms_norm(x, self.weight, self.variance_epsilon)
        return ir.ops.fused_add_rms_norm(
            x, residual, self.weight, self.variance_epsilon
        )
