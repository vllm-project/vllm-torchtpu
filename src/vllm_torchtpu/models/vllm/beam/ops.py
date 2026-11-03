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

Each public function is named after the upstream wrapper it replaces in
vllm/models/beam/nvidia/ (``layernorm.py``, ``activation.py``,
``attention_ops.py``; vLLM v0.33.0) and reproduces the kernel's FP32 math and
activation-dtype rounding points.
"""

from typing import NamedTuple

import torch
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

_SSMAX_POSITION_SCALE = 2048.0
_SSMAX_EXPONENT = 0.1


class ResidualScales(NamedTuple):
    """Residual-add coefficients of the sublayer output and the residual."""

    branch: float
    residual: float


def _round_to(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Rounds ``x`` to ``dtype`` and continues in FP32."""
    return x.to(dtype).float()


def _rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 RMS normalization over the last dim, without the affine weight."""
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def _head_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Rounded per-head RMSNorm of ``[T, H * D]``, as FP32 ``[T, H, D]``."""
    heads = x.unflatten(-1, (-1, weight.shape[-1])).float()
    return _round_to(_rms_normalize(heads, eps) * weight.float(), x.dtype)


def _ssmax_scale(positions: torch.Tensor) -> torch.Tensor:
    """FP32 SSMax Q scale per token, in upstream's ``exp(log1p(.))`` form.

    ``(1 + (position + 1) / _SSMAX_POSITION_SCALE) ** _SSMAX_EXPONENT``.
    """
    normalized = (positions.float() + 1.0) / _SSMAX_POSITION_SCALE
    return torch.exp(_SSMAX_EXPONENT * torch.log1p(normalized))


def beam_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 RMSNorm, rounded to ``x.dtype`` only after the affine multiply."""
    return (_rms_normalize(x.float(), eps) * weight.float()).to(x.dtype)


def beam_norm_residual_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    *,
    post_weight: torch.Tensor,
    next_weight: torch.Tensor,
    eps: float,
    scales: ResidualScales,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Post-norm, depth-scaled residual add and the next pre-norm.

    Rounds after the post-norm affine, after each scale and after the add,
    like ``_norm_residual_norm_kernel``.

    Args:
        x: Sublayer output, ``[T, hidden]``.
        residual: Residual stream, ``[T, hidden]``, in ``x.dtype``.
        post_weight: Post-norm weight applied to ``x``.
        next_weight: Pre-norm weight of the next sublayer.
        eps: Epsilon of both norms.
        scales: Coefficients of ``x`` and ``residual`` in the add.

    Returns:
        The new residual stream and its normalization by ``next_weight``.
    """
    assert residual.dtype == x.dtype, (residual.dtype, x.dtype)
    dtype = x.dtype
    post_normed = _round_to(_rms_normalize(x.float(), eps) * post_weight.float(), dtype)
    scaled_branch = _round_to(post_normed * scales.branch, dtype)
    scaled_residual = _round_to(residual.float() * scales.residual, dtype)
    hidden_states = (scaled_branch + scaled_residual).to(dtype)
    return hidden_states, beam_rms_norm(hidden_states, next_weight, eps)


def beam_silu_mul(x: torch.Tensor) -> torch.Tensor:
    """FP32 ``silu(gate) * up`` on the ``[gate | up]`` halves, rounded once."""
    gate, up = x.float().chunk(2, dim=-1)
    # Spelled like upstream's kernel; F.silu's TPU lowering has not been
    # checked against this form.
    return (gate / (1.0 + torch.exp(-gate)) * up).to(x.dtype)


def beam_softcap_logits(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """``cap * tanh(logits * (1/cap))`` in FP32, written back into ``logits``.

    Multiplies by ``1/cap`` like upstream's ``_softcap_logits_kernel``; ``/ cap``
    differs in FP32 (and occasionally by one BF16 step after rounding).
    """
    capped = torch.tanh(logits.float() * (1.0 / cap)) * cap
    return logits.copy_(capped)


def beam_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    rotary_emb: RotaryEmbedding | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head Q/K RMSNorm, rounded, then RoPE or SSMax in FP32.

    Sliding-window layers pass ``rotary_emb`` and rotate Q and K. The NoPE
    full-attention layers pass None and apply SSMax, which scales Q only.
    Upstream's kernel can also write K/V into the cache; here the TPU attention
    backend does that.

    Args:
        q: ``[T, num_heads * head_dim]``.
        k: ``[T, num_kv_heads * head_dim]``.
        q_weight: Per-head Q norm weight, ``[head_dim]``.
        k_weight: Per-head K norm weight, ``[head_dim]``.
        eps: Epsilon of both norms.
        positions: ``[T]`` token positions.
        rotary_emb: The FP32 RoPE of a sliding-window layer, or None for SSMax.

    Returns:
        Q and K in their input shapes and dtype.
    """
    dtype = q.dtype
    q_normed = _head_rms_norm(q, q_weight, eps)
    k_normed = _head_rms_norm(k, k_weight, eps)
    if rotary_emb is not None:
        q_normed, k_normed = rotary_emb.forward_native(positions, q_normed, k_normed)
        assert k_normed is not None
    else:
        q_normed = q_normed * _ssmax_scale(positions)[:, None, None]
    return q_normed.to(dtype).flatten(-2), k_normed.to(dtype).flatten(-2)


def beam_output_gate(
    attn_output: torch.Tensor, gate: torch.Tensor, gate_bias: torch.Tensor
) -> torch.Tensor:
    """``attn_output * sigmoid(gate + gate_bias)``, rounding after each step.

    Args:
        attn_output: ``[T, num_heads * head_dim]``.
        gate: Output-gate projection, shaped like ``attn_output``.
        gate_bias: Per-head bias, ``[num_heads]``.

    Returns:
        The gated attention output, in ``attn_output.dtype``.
    """
    dtype = gate.dtype
    num_heads = gate_bias.shape[0]
    gate_logits = (
        gate.float().unflatten(-1, (num_heads, -1)) + gate_bias.float()[:, None]
    )
    gate_logits = _round_to(gate_logits, dtype)
    # Spelled like upstream's kernel, as in beam_silu_mul.
    gate_sigmoid = _round_to(1.0 / (1.0 + torch.exp(-gate_logits)), dtype)
    gated = attn_output.float().unflatten(-1, (num_heads, -1)) * gate_sigmoid
    return gated.flatten(-2).to(attn_output.dtype)
