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
"""Idempotent TPU compatibility helpers for upstream ``vllm.models.muse_spark``."""

from __future__ import annotations

import torch
from torch import nn

_APPLIED = False


def apply() -> None:
    """Apply idempotent TPU compatibility patches for ``MuseSpark1xRMSNorm``.

    1. Upstream ``MuseSpark1xRMSNorm.supports_fused_kernel`` dispatches to its
       Triton ``fused_rmsnorm_fwd`` kernel without checking ``x.is_cuda``;
       gating on ``x.is_cuda`` routes TPU/CPU tensors to ``forward_native``.
    2. Upstream ``MuseSpark1xRMSNormWithInputScale.forward_native`` calls
       ``F.rms_norm(x, (hidden_size,), None, eps)``, whose ``torch_tpu`` C++
       autograd wrapper calls ``numel()`` and fails under Dynamo symbolic batch
       shapes. Replacing that call with the explicit FP32 ``rsqrt(mean(x^2) +
       eps)`` reduction (matching ``MuseSpark1xRMSNorm.forward_native``) is
       numerically identical and symbolic-shape safe.
    """
    global _APPLIED
    if _APPLIED:
        return

    from vllm.models.muse_spark.common.layers import (
        MuseSpark1xRMSNorm,
        MuseSpark1xRMSNormWithInputScale,
    )

    orig_supports = MuseSpark1xRMSNorm.supports_fused_kernel

    def _supports_fused_kernel_cuda_only(
        self: MuseSpark1xRMSNorm, x: torch.Tensor
    ) -> bool:
        return bool(x.is_cuda and orig_supports(self, x))

    def _input_scale_forward_native(
        self: MuseSpark1xRMSNormWithInputScale,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        out_dtype = out_dtype or orig_dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)
        if self.weight is not None:
            weight = self.weight.float()
            if self.zero_centered_gamma:
                weight = weight + self.gain_center
            x = x * weight
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = (x * torch.rsqrt(variance + self.variance_epsilon)).to(out_dtype)
        if residual is None:
            return x
        return x, residual

    MuseSpark1xRMSNorm.supports_fused_kernel = _supports_fused_kernel_cuda_only  # type: ignore[method-assign]
    MuseSpark1xRMSNormWithInputScale.forward_native = _input_scale_forward_native  # type: ignore[method-assign]
    _APPLIED = True


def sync_residual_constants(model: nn.Module) -> None:
    """Synchronize newly registered residual-gate ``alpha``/``beta`` buffers on TPU."""
    from vllm.models.muse_spark.common.layers import (
        MuseSpark1xNormPreservingResidualConnection,
    )

    from vllm_torchtpu.utils import synchronize_tensors

    tpu_buffers: list[torch.Tensor] = []
    for module in model.modules():
        if isinstance(module, MuseSpark1xNormPreservingResidualConnection):
            for name in ("alpha", "beta"):
                buf = getattr(module, name, None)
                if (
                    buf is not None
                    and isinstance(buf, torch.Tensor)
                    and buf.device.type == "tpu"
                ):
                    tpu_buffers.append(buf)
    if tpu_buffers:
        synchronize_tensors(tpu_buffers)
