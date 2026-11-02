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
"""TPU IR-op providers for RMSNorm reproduce the CUDA kernel rounding points."""

import pytest
import torch
from vllm import ir

from vllm_torchtpu.layers.adapter import ir_ops
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

EPS = 1e-6


def _cuda_rms_norm_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``rms_norm_kernel``: FP32 throughout, one cast on store."""
    xf = x.float()
    s = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (xf * s * weight.float()).to(x.dtype)


def _cuda_fused_add_reference(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``fused_add_rms_norm_kernel``: add in the activation dtype, then FP32."""
    z = x + residual  # bf16 + bf16 -> rounded to bf16, like _f16Vec +=
    zf = z.float()
    s = torch.rsqrt(zf.pow(2).mean(-1, keepdim=True) + EPS)
    return (zf * s * weight.float()).to(x.dtype), z


@pytest.fixture
def bf16_inputs():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 256, generator=g).to(torch.bfloat16)
    residual = torch.randn(64, 256, generator=g).to(torch.bfloat16)
    # Weights away from 1.0 so the BF16-vs-FP32 multiply order is visible.
    weight = (1.0 + 0.3 * torch.randn(256, generator=g)).to(torch.bfloat16)
    return x, residual, weight


def test_tpu_provider_registered_by_platform():
    TpuPlatform.import_ir_kernels()
    assert ir_ops.TPU_IR_PROVIDER in ir.ops.rms_norm.impls
    assert ir_ops.TPU_IR_PROVIDER in ir.ops.fused_add_rms_norm.impls
    assert ir.ops.rms_norm.impls[ir_ops.TPU_IR_PROVIDER].supported
    assert ir.ops.fused_add_rms_norm.impls[ir_ops.TPU_IR_PROVIDER].supported


def test_platform_default_priority_prefers_tpu_then_native():
    priority = TpuPlatform.get_default_ir_op_priority(vllm_config=None)
    assert priority.rms_norm == [ir_ops.TPU_IR_PROVIDER, "native"]
    assert priority.fused_add_rms_norm == [ir_ops.TPU_IR_PROVIDER, "native"]
    # Ops we do not override keep upstream's native default.
    assert priority.gelu_and_mul_sparse == ["native"]


def test_rms_norm_matches_cuda_reference(bf16_inputs):
    x, _, weight = bf16_inputs
    out = ir_ops.rms_norm.impl_fn(x, weight, EPS)
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, _cuda_rms_norm_reference(x, weight), rtol=0, atol=0)


def test_rms_norm_differs_from_native_rounding(bf16_inputs):
    """Guards that the provider is not a no-op: native rounds before the
    weight multiply, so the two disagree on some elements."""
    x, _, weight = bf16_inputs
    native = ir.ops.rms_norm.impls["native"].impl_fn(x, weight, EPS)
    tpu = ir_ops.rms_norm.impl_fn(x, weight, EPS)
    assert not torch.equal(native, tpu)
    # ...but both are the same function up to BF16 rounding.
    torch.testing.assert_close(native.float(), tpu.float(), rtol=2e-2, atol=2e-2)


def test_rms_norm_variance_size_and_no_weight():
    x = torch.randn(4, 16).to(torch.bfloat16)
    out = ir_ops.rms_norm.impl_fn(x, None, EPS, variance_size=8)
    xf = x.float()
    s = torch.rsqrt(xf[..., :8].pow(2).mean(-1, keepdim=True) + EPS)
    torch.testing.assert_close(out, (xf * s).to(x.dtype), rtol=0, atol=0)


def test_fused_add_rms_norm_matches_cuda_reference(bf16_inputs):
    x, residual, weight = bf16_inputs
    out, new_residual = ir_ops.fused_add_rms_norm.impl_fn(x, residual, weight, EPS)
    ref_out, ref_residual = _cuda_fused_add_reference(x, residual, weight)
    assert out.dtype == x.dtype and new_residual.dtype == x.dtype
    torch.testing.assert_close(new_residual, ref_residual, rtol=0, atol=0)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)


def test_fused_add_rms_norm_normalizes_the_rounded_residual(bf16_inputs):
    """Native normalizes the FP32 sum; CUDA (and we) normalize the stored
    BF16 residual. Feeding the returned residual back through plain rms_norm
    must therefore reproduce the fused output exactly."""
    x, residual, weight = bf16_inputs
    out, new_residual = ir_ops.fused_add_rms_norm.impl_fn(x, residual, weight, EPS)
    torch.testing.assert_close(
        out, ir_ops.rms_norm.impl_fn(new_residual, weight, EPS), rtol=0, atol=0
    )
    native_out, _ = ir.ops.fused_add_rms_norm.impls["native"].impl_fn(
        x, residual, weight, EPS
    )
    assert not torch.equal(native_out, out)


def test_dispatch_selects_tpu_provider_under_platform_priority(bf16_inputs):
    x, residual, weight = bf16_inputs
    TpuPlatform.import_ir_kernels()
    priority = TpuPlatform.get_default_ir_op_priority(vllm_config=None)
    with priority.set_priority():
        assert ir.ops.rms_norm.dispatch(x, weight, EPS).provider == "tpu"
        assert (
            ir.ops.fused_add_rms_norm.dispatch(x, residual, weight, EPS).provider
            == "tpu"
        )
        torch.testing.assert_close(
            ir.ops.rms_norm(x, weight, EPS),
            _cuda_rms_norm_reference(x, weight),
            rtol=0,
            atol=0,
        )
