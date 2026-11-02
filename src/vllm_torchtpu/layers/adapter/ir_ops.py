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
"""TPU implementations of vLLM IR ops with CUDA-matching numerics.

vLLM routes ``RMSNorm``, ``GemmaRMSNorm`` and the fused residual-add variants
through ``vllm.ir.ops.rms_norm`` / ``vllm.ir.ops.fused_add_rms_norm``. Each IR
op dispatches to a provider chosen by ``KernelConfig.ir_op_priority``; on CUDA
that is the ``vllm_c`` kernel, on TPU it falls through to ``native``.

The ``native`` reference rounds at different points than the CUDA kernels
(``csrc/layernorm_kernels.cu``):

* ``rms_norm``: native casts the normalized activation to ``weight.dtype``
  (BF16) *before* the weight multiply. CUDA keeps ``x * s_variance * w`` in
  FP32 and casts once on store.
* ``fused_add_rms_norm``: native keeps the residual sum in FP32 for the
  variance and normalization while storing a BF16 copy as the new residual.
  CUDA adds in the activation dtype (``_f16Vec::operator+=``), so the stored
  residual *is* the value that gets normalized, and multiplies the weight in
  FP32.

For RL rollouts the trainer recomputes log-probs from the same weights, so
these rounding differences surface directly as trainer/sampler KL mismatch.
The ``tpu`` providers below reproduce the CUDA rounding points
with plain PyTorch ops. ``TpuPlatform.import_ir_kernels`` imports this module
and ``TpuPlatform.get_default_ir_op_priority`` puts ``tpu`` ahead of
``native``; ``--kernel-config '{"ir_op_priority": {"rms_norm": "native",
"fused_add_rms_norm": "native"}}'`` restores the upstream numerics for A/B
comparisons.
"""

import torch
from torch import Tensor
from vllm import ir

TPU_IR_PROVIDER = "tpu"


def _normalize_fp32(
    x_f32: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None
) -> Tensor:
    """``x * rsqrt(mean(x^2) + eps) * w`` entirely in FP32 (CUDA order)."""
    x_var = x_f32 if variance_size is None else x_f32[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x_f32 = x_f32 * torch.rsqrt(variance + epsilon)
    if weight is not None:
        x_f32 = x_f32 * weight.to(torch.float32)
    return x_f32


@ir.ops.rms_norm.register_impl(TPU_IR_PROVIDER)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    """RMSNorm with the weight multiply in FP32, matching ``rms_norm_kernel``."""
    orig_dtype = x.dtype
    out = _normalize_fp32(x.to(torch.float32), weight, epsilon, variance_size)
    return out.to(orig_dtype)


@ir.ops.fused_add_rms_norm.register_impl(TPU_IR_PROVIDER)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Residual add rounded to the activation dtype, then FP32 RMSNorm.

    Matches ``fused_add_rms_norm_kernel``: the new residual is the rounded
    sum, and that rounded sum (not the FP32 sum) is what gets normalized.
    """
    orig_dtype = x.dtype
    x_residual = (x.to(torch.float32) + x_residual.to(torch.float32)).to(orig_dtype)
    out = _normalize_fp32(x_residual.to(torch.float32), weight, epsilon, variance_size)
    return out.to(orig_dtype), x_residual
