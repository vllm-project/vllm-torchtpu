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
"""TPU-optimized RMSNorm layer adapter.

Fuses residual accumulation and RMSNorm into a single high-efficiency path:
- Eliminates redundant float32 type conversions and buffer allocations on inputs,
  residuals, and weights.
- In-place residual accumulation  without intermediate float32 bouncing.
- Numerically stable float32 variance reduction:  with .
- Fused scaling preserving original activation dtype ().
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from vllm.model_executor.custom_op import CustomOp
    from vllm.model_executor.layers.layernorm import RMSNorm

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    CustomOp = object
    RMSNorm = nn.Module

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def tpu_rmsnorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Fused residual addition + RMSNorm execution on TPU.

    Eliminates redundant float32 conversions and intermediate buffer allocations:
    1. Residual accumulation is fused directly:  without
       double-casting both operands to FP32 and downcasting back to original dtype.
    2. Variance reduction is computed via float32 accumulation for numerical stability:
       
    3. Normalization and weight scaling are computed in a single fused expression:
       
    """
    orig_dtype = x.dtype
    if residual is not None:
        x = x + residual
        residual = x

    # Numerically stable variance computation in float32
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    inv_std = torch.rsqrt(variance + variance_epsilon)
    out = (x_f32 * inv_std * weight.float()).to(orig_dtype)

    if residual is None:
        return out
    return out, residual


class TpuRMSNorm(RMSNorm):
    """TPU-optimized RMSNorm with fused residual addition.

    Layers built with a partial-variance width (`var_hidden_size !=
    hidden_size`, i.e. `variance_size_override`) keep upstream's
    `RMSNorm.forward_native`; the fused path always normalizes over the full
    hidden size.
    """

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if getattr(self, "variance_size_override", None) is not None:
            return super().forward_native(x, residual)
        return tpu_rmsnorm_forward(x, self.weight, self.variance_epsilon, residual)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if getattr(self, "variance_size_override", None) is not None:
            return super().forward_native(x, residual)
        return tpu_rmsnorm_forward(x, self.weight, self.variance_epsilon, residual)

    forward = forward_oot


def register_tpu_rmsnorm():
    """Register TpuRMSNorm with vLLM's CustomOp / RMSNorm registry."""
    if not HAS_VLLM:
        return
    registered = False
    if hasattr(RMSNorm, "register_oot"):
        try:
            RMSNorm.register_oot(TpuRMSNorm)
            registered = True
            logger.info("Registered TpuRMSNorm via RMSNorm.register_oot")
        except Exception as e:
            logger.debug("RMSNorm.register_oot failed: %s", e)
    if not registered and hasattr(CustomOp, "register_oot"):
        try:
            CustomOp.register_oot(name="RMSNorm")(TpuRMSNorm)
            registered = True
            logger.info("Registered TpuRMSNorm via CustomOp.register_oot(RMSNorm)")
        except Exception as e:
            logger.debug("CustomOp.register_oot failed: %s", e)


register_tpu_rmsnorm()
