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
"""Pure PyTorch versions of the upstream Beam Triton kernels.

Each function is named after the upstream wrapper it replaces (in
vllm/models/beam/nvidia/) and reproduces the kernel's FP32 math and
activation-dtype rounding points. XLA fuses the elementwise chains on TPU.
"""

import torch


def _r(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Round to ``dtype`` and continue in FP32."""
    return x.to(dtype).float()


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 RMS normalization over the last dim (no affine)."""
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def beam_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in FP32, rounding only after the affine multiply."""
    return (_rms(x.float(), eps) * weight.float()).to(x.dtype)


def beam_norm_residual_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    post_eps: float,
    next_weight: torch.Tensor,
    next_eps: float,
    scales: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Post-norm, depth-scaled residual add and the next pre-norm.

    Rounds after the post-norm affine, after each scale and after the add,
    like ``_norm_residual_norm_kernel``.
    """
    dtype = x.dtype
    x_scale, r_scale = scales
    h = _r(_rms(x.float(), post_eps) * post_weight.float(), dtype)
    h = _r(h * x_scale, dtype)
    r = _r(_r(residual, dtype) * r_scale, dtype)
    hidden = (h + r).to(dtype)
    return hidden, beam_rms_norm(hidden, next_weight, next_eps)


def beam_silu_mul(x: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` on the ``[gate | up]`` halves in FP32, rounded once."""
    gate, up = x.float().chunk(2, dim=-1)
    return (gate / (1.0 + torch.exp(-gate)) * up).to(x.dtype)


def beam_softcap_logits(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """``cap * tanh(logits * (1/cap))`` in FP32, written back into ``logits``.

    Multiplies by ``1/cap`` like upstream's ``_softcap_logits_kernel``; ``/ cap``
    differs in FP32 (and occasionally by one BF16 step after rounding).
    """
    capped = torch.tanh(logits.float() * (1.0 / cap)) * cap
    return logits.copy_(capped)


def _rotate(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int, neox: bool
) -> torch.Tensor:
    """FP32 RoPE on the first ``rotary_dim`` channels of ``[T, H, D]``."""
    rot, rest = x[..., :rotary_dim], x[..., rotary_dim:]
    cos, sin = cos[:, None, :], sin[:, None, :]
    if neox:
        x1, x2 = rot.chunk(2, dim=-1)
        rot = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    else:
        x1, x2 = rot[..., 0::2], rot[..., 1::2]
        rot = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        rot = rot.flatten(-2)
    return torch.cat((rot, rest), dim=-1)


def _head_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, head_dim: int
) -> torch.Tensor:
    """Per-head FP32 RMSNorm of ``[T, H * D]`` into ``[T, H, D]``, rounded."""
    y = _rms(x.unflatten(-1, (-1, head_dim)).float(), eps) * weight.float()
    return _r(y, x.dtype)


def beam_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    head_dim: int,
    rotary_dim: int = 0,
    is_neox_style: bool = True,
    *,
    ssmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head Q/K RMSNorm, rounded, then RoPE or SSMax in FP32.

    RoPE (sliding layers) rotates Q and K with ``cos_sin_cache[positions]``.
    SSMax (NoPE full-attention layers) scales Q only by
    ``(1 + (position + 1) / 2048) ** 0.1``. Upstream's kernel can also write
    K/V into the cache; here the TPU attention backend does that.
    """
    dtype = q.dtype
    qf = _head_norm(q, q_weight, eps, head_dim)
    kf = _head_norm(k, k_weight, eps, head_dim)
    if rotary_dim:
        assert cos_sin_cache is not None
        cos_sin = cos_sin_cache.index_select(0, positions.long()).float()
        cos, sin = cos_sin.chunk(2, dim=-1)
        qf = _rotate(qf, cos, sin, rotary_dim, is_neox_style)
        kf = _rotate(kf, cos, sin, rotary_dim, is_neox_style)
    elif ssmax:
        scale = torch.exp(0.1 * torch.log1p((positions.float() + 1.0) / 2048.0))
        qf = qf * scale[:, None, None]
    return qf.to(dtype).flatten(-2), kf.to(dtype).flatten(-2)


def beam_output_gate(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    gate_bias: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """``attn_output * sigmoid(gate + bias)`` rounding after the add and sigmoid."""
    dtype = gate.dtype
    g = gate.float().unflatten(-1, (num_heads, head_dim)) + gate_bias.float()[:, None]
    g = _r(g, dtype)
    s = _r(1.0 / (1.0 + torch.exp(-g)), dtype)
    out = attn_output.float().unflatten(-1, (num_heads, head_dim)) * s
    return out.flatten(-2).to(attn_output.dtype)
