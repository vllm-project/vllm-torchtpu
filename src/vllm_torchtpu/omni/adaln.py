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
"""TPU custom-op wrappers and fallbacks for vLLM-Omni indexed AdaLN modulation."""

from __future__ import annotations

from functools import cache as run_once
from typing import Any

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.diffusion_adaln.kernel import (
    indexed_gate_pallas,
    indexed_gate_rms_norm_scale_shift_pallas,
    indexed_scale_shift_pallas,
    rms_norm_indexed_scale_shift_pallas,
)

# Below ~16k local tokens (e.g. Ulysses-8 where S_local ~= 4,720), XLA fuses
# ATen index_select + RMSNorm + scale_shift directly into adjacent matmuls
# faster than crossing a custom_op barrier; at large local token counts
# (e.g. TP-8 where S = 37,760), the fused Pallas kernel wins and avoids
# materializing [S, D] FP32 intermediates in HBM.
MIN_PALLAS_ADALN_TOKENS: int = 16384


def _can_use_pallas_adaln(x: torch.Tensor, indices: torch.Tensor) -> bool:
    return (
        x.device.type == "tpu"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[0] >= max(1, MIN_PALLAS_ADALN_TOKENS)
        and x.shape[1] > 0
        and x.shape[1] % 128 == 0
        and indices.ndim == 1
        and indices.shape[0] == x.shape[0]
    )


@run_once
def _rms_norm_ss_op() -> Any:
    """Build the torch custom op for ``rms_norm_indexed_scale_shift`` (once)."""

    def _fn(
        x: jax.Array,
        weight: jax.Array,
        shift: jax.Array,
        scale: jax.Array,
        indices: jax.Array,
        eps: float,
    ) -> jax.Array:
        return rms_norm_indexed_scale_shift_pallas(
            x, weight, shift, scale, indices, eps=eps
        )

    op = pallas.jax_op("pallas::diffusion_rms_norm_indexed_scale_shift", _fn)

    def _fake(x, weight, shift, scale, indices, eps):
        del weight, shift, scale, indices, eps
        return torch.empty_like(x)

    op.register_fake(_fake)
    return op


@run_once
def _gate_rms_norm_ss_op() -> Any:
    """Build the torch custom op for ``indexed_gate_rms_norm_scale_shift`` (once)."""

    def _fn(
        residual: jax.Array,
        gate: jax.Array,
        branch: jax.Array,
        weight: jax.Array,
        shift: jax.Array,
        scale: jax.Array,
        indices: jax.Array,
        eps: float,
    ) -> tuple[jax.Array, jax.Array]:
        return indexed_gate_rms_norm_scale_shift_pallas(
            residual, gate, branch, weight, shift, scale, indices, eps=eps
        )

    op = pallas.jax_op("pallas::diffusion_indexed_gate_rms_norm_scale_shift", _fn)

    def _fake(residual, gate, branch, weight, shift, scale, indices, eps):
        del gate, branch, weight, shift, scale, indices, eps
        return torch.empty_like(residual), torch.empty_like(residual)

    op.register_fake(_fake)
    return op


@run_once
def _gate_op() -> Any:
    """Build the torch custom op for ``indexed_gate`` (once)."""

    def _fn(
        x: jax.Array,
        gate: jax.Array,
        other: jax.Array,
        indices: jax.Array,
    ) -> jax.Array:
        return indexed_gate_pallas(x, gate, other, indices)

    op = pallas.jax_op("pallas::diffusion_indexed_gate", _fn)

    def _fake(x, gate, other, indices):
        del gate, other, indices
        return torch.empty_like(x)

    op.register_fake(_fake)
    return op


@run_once
def _scale_shift_op() -> Any:
    """Build the torch custom op for ``indexed_scale_shift_`` (once)."""

    def _fn(
        x: jax.Array,
        shift: jax.Array,
        scale: jax.Array,
        indices: jax.Array,
    ) -> jax.Array:
        return indexed_scale_shift_pallas(x, shift, scale, indices)

    op = pallas.jax_op("pallas::diffusion_indexed_scale_shift", _fn)

    def _fake(x, shift, scale, indices):
        del shift, scale, indices
        return torch.empty_like(x)

    op.register_fake(_fake)
    return op


_RMS_NORM_SS_OP: Any = None
_GATE_RMS_NORM_SS_OP: Any = None
_GATE_OP: Any = None
_SCALE_SHIFT_OP: Any = None


def register_adaln_pallas_ops() -> None:
    """Pre-register Pallas AdaLN custom ops outside any ``torch.compile`` region."""
    global _RMS_NORM_SS_OP, _GATE_RMS_NORM_SS_OP, _GATE_OP, _SCALE_SHIFT_OP
    try:
        import torch_tpu  # noqa: F401
    except ImportError:
        return
    if _RMS_NORM_SS_OP is None:
        _RMS_NORM_SS_OP = _rms_norm_ss_op()
    if _GATE_RMS_NORM_SS_OP is None:
        _GATE_RMS_NORM_SS_OP = _gate_rms_norm_ss_op()
    if _GATE_OP is None:
        _GATE_OP = _gate_op()
    if _SCALE_SHIFT_OP is None:
        _SCALE_SHIFT_OP = _scale_shift_op()


def tpu_indexed_scale_shift_(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    if x.shape[0] == 0:
        return x
    if (
        _can_use_pallas_adaln(x, indices)
        and shift.dtype == torch.bfloat16
        and scale.dtype == torch.bfloat16
    ):
        op = _SCALE_SHIFT_OP
        if op is None:
            register_adaln_pallas_ops()
            op = _SCALE_SHIFT_OP
        assert op is not None
        out = op(
            x.contiguous(),
            shift.contiguous(),
            scale.contiguous(),
            indices.to(torch.int32).contiguous(),
        )
    else:
        out = (
            x.float() * (1.0 + scale.index_select(0, indices).float())
            + shift.index_select(0, indices).float()
        ).to(x.dtype)
    x.copy_(out)
    return x


def tpu_indexed_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    if x.shape[0] == 0:
        return torch.empty_like(x)
    if (
        _can_use_pallas_adaln(x, indices)
        and other.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
    ):
        op = _GATE_OP
        if op is None:
            register_adaln_pallas_ops()
            op = _GATE_OP
        assert op is not None
        return op(
            x.contiguous(),
            gate.contiguous(),
            other.contiguous(),
            indices.to(torch.int32).contiguous(),
        )
    return (x.float() + gate.index_select(0, indices).float() * other.float()).to(
        x.dtype
    )


def tpu_rms_norm_indexed_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if x.shape[0] == 0:
        return torch.empty_like(x)
    if (
        _can_use_pallas_adaln(x, indices)
        and weight.dtype == torch.bfloat16
        and shift.dtype == torch.bfloat16
        and scale.dtype == torch.bfloat16
    ):
        op = _RMS_NORM_SS_OP
        if op is None:
            register_adaln_pallas_ops()
            op = _RMS_NORM_SS_OP
        assert op is not None
        return op(
            x.contiguous(),
            weight.contiguous(),
            shift.contiguous(),
            scale.contiguous(),
            indices.to(torch.int32).contiguous(),
            float(eps),
        )
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    normalized = x_f32 * torch.rsqrt(variance + eps) * weight.float()
    return (
        normalized * (1.0 + scale.index_select(0, indices).float())
        + shift.index_select(0, indices).float()
    ).to(x.dtype)


def tpu_indexed_gate_rms_norm_scale_shift(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual.shape[0] == 0:
        return torch.empty_like(residual), torch.empty_like(residual)
    if (
        _can_use_pallas_adaln(residual, indices)
        and branch.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and shift.dtype == torch.bfloat16
        and scale.dtype == torch.bfloat16
    ):
        op = _GATE_RMS_NORM_SS_OP
        if op is None:
            register_adaln_pallas_ops()
            op = _GATE_RMS_NORM_SS_OP
        assert op is not None
        return op(
            residual.contiguous(),
            gate.contiguous(),
            branch.contiguous(),
            weight.contiguous(),
            shift.contiguous(),
            scale.contiguous(),
            indices.to(torch.int32).contiguous(),
            float(eps),
        )
    residual_out = (
        residual.float() + gate.index_select(0, indices).float() * branch.float()
    ).to(residual.dtype)
    updated = residual_out.float()
    variance = updated.pow(2).mean(-1, keepdim=True)
    normalized = updated * torch.rsqrt(variance + eps) * weight.float()
    modulated_out = (
        normalized * (1.0 + scale.index_select(0, indices).float())
        + shift.index_select(0, indices).float()
    ).to(residual.dtype)
    return residual_out, modulated_out
