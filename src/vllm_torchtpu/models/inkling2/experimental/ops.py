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
"""TPU-portable pure-PyTorch primitives for Inkling 2 / Inkling 2 Loop.

These reference ops run under ``torch.compile(backend="tpu")`` with zero CUDA
or Triton dependencies and serve as the golden numerical target for specialized
Pallas TPU kernels.
"""

from __future__ import annotations

import torch
from torch.nn import functional


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float = 1e-6,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """FP32-accumulated RMSNorm; ``weight=None`` applies weightless RMSNorm."""
    x_fp32 = x.to(torch.float32)
    rstd = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    y = x_fp32 * rstd
    if weight is not None:
        y = y * weight.to(torch.float32)
    return y.to(out_dtype or x.dtype)


def value_norm_no_gamma(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-head weightless RMSNorm applied to V after short convolution."""
    return rms_norm(v, None, eps=eps, out_dtype=v.dtype)


def l2_clip(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    threshold: float,
    eps: float = 1e-6,
    tau: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-head L2 norm clipping with optional scale and log-length ``tau``."""
    x_fp32 = x.to(torch.float32)
    if scale is not None:
        x_fp32 = x_fp32 * scale.to(torch.float32)
    norm = torch.linalg.vector_norm(x_fp32, ord=2, dim=-1, keepdim=True)
    y = x_fp32 * torch.clamp(threshold / (norm + eps), max=1.0)
    if tau is not None:
        y = y.to(x.dtype).to(torch.float32) * tau.to(torch.float32)
    return y.to(x.dtype)


def compute_log_scaling_tau(
    positions: torch.Tensor,
    n_floor: int = 8192,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Compute ``1 + alpha * log(max((pos + 1) / n_floor, 1))`` for full layers."""
    ratio = torch.clamp((positions.to(torch.float32) + 1.0) / float(n_floor), min=1.0)
    return 1.0 + alpha * torch.log(ratio)


def attn_res(
    stack: torch.Tensor,
    last: torch.Tensor,
    addend: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float = 1e-6,
    normalize_values: bool = True,
    rms_eps: float = 1e-6,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Block Attention Residual: fp32 softmax-weighted mix of residual blocks."""
    last_fp32 = last.to(torch.float32)
    if addend is not None:
        last_fp32 = last_fp32 + addend.to(torch.float32)
    sources = (
        torch.cat((stack.to(torch.float32), last_fp32.unsqueeze(0)), dim=0)
        if stack.shape[0] > 0
        else last_fp32.unsqueeze(0)
    )
    inv_rms = torch.rsqrt(sources.square().mean(dim=-1, keepdim=True) + rms_eps)
    dots = torch.einsum("qd,ntd->qnt", queries.to(torch.float32), sources)
    probs = torch.softmax(dots * inv_rms.squeeze(-1).unsqueeze(0), dim=1)
    values = sources * inv_rms if normalize_values else sources
    mixed = torch.einsum("qnt,ntd->qtd", probs, values)
    if norm_weight is None:
        return mixed
    weight = norm_weight.unsqueeze(0) if norm_weight.ndim == 1 else norm_weight
    if weight.shape[0] == 1 and mixed.shape[0] > 1:
        weight = weight.expand(mixed.shape[0], -1)
    rstd = torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + norm_eps)
    return (mixed * rstd * weight.unsqueeze(1).to(torch.float32)).to(out_dtype)


def message_controller_blend(
    dest: torch.Tensor,
    message: torch.Tensor,
    down_weight: torch.Tensor,
    gain_weight: torch.Tensor,
    forget_weight: torch.Tensor,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Rank-r cross-pass loop message controller batched over streams."""
    num_streams, num_tokens, hidden = message.shape
    dest_fp32 = dest.expand(num_streams, num_tokens, hidden).to(torch.float32)
    msg_fp32 = message.to(torch.float32)
    pair = torch.cat(
        (
            rms_norm(dest_fp32, None, eps=1e-6, out_dtype=torch.bfloat16),
            rms_norm(msg_fp32, None, eps=1e-6, out_dtype=torch.bfloat16),
        ),
        dim=-1,
    )
    hidden_proj = torch.bmm(pair, down_weight.to(torch.bfloat16).transpose(1, 2))
    hidden_fp32 = hidden_proj.to(torch.float32)
    gain = torch.bmm(hidden_fp32, gain_weight.to(torch.float32).transpose(1, 2))
    forget = torch.bmm(hidden_fp32, forget_weight.to(torch.float32).transpose(1, 2))
    blended = (1.0 - torch.tanh(forget)) * dest_fp32 + torch.tanh(gain) * msg_fp32
    if norm_weight is None:
        return blended
    weight = norm_weight.unsqueeze(0) if norm_weight.ndim == 1 else norm_weight
    if weight.shape[0] == 1 and num_streams > 1:
        weight = weight.expand(num_streams, -1)
    rstd = torch.rsqrt(blended.square().mean(dim=-1, keepdim=True) + norm_eps)
    return (blended * rstd * weight.unsqueeze(1).to(torch.float32)).to(out_dtype)


def short_conv_static(
    x: torch.Tensor,
    weight: torch.Tensor,
    history: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal static depthwise short conv (oldest tap first, ``(W+1)**-0.5``)."""
    taps = weight.squeeze(1) if weight.ndim == 3 else weight
    channels, kernel_size = taps.shape
    num_tokens = x.shape[0]
    hist = (
        torch.zeros((kernel_size - 1, channels), dtype=x.dtype, device=x.device)
        if history is None
        else history
    )
    padded = torch.cat((hist, x), dim=0).to(torch.float32)
    taps_fp32 = taps.to(torch.float32)
    acc = x.to(torch.float32).clone()
    for tap_idx in range(kernel_size):
        acc = acc + padded[tap_idx : tap_idx + num_tokens] * taps_fp32[
            :, tap_idx
        ].unsqueeze(0)
    out = (acc * ((kernel_size + 1) ** -0.5)).to(x.dtype)
    new_hist = padded[-(kernel_size - 1) :].to(x.dtype) if kernel_size > 1 else hist
    return out, new_hist


def short_conv_dynamic(
    x: torch.Tensor,
    static_w: torch.Tensor,
    dyn_taps: torch.Tensor,
    num_heads: int,
    history: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal dynamic depthwise short conv (newest tap first, ``(2W+1)**-0.5``)."""
    kernel_size, channels = static_w.shape
    num_tokens = x.shape[0]
    if channels % num_heads != 0:
        raise ValueError(
            f"Channels {channels} must be divisible by num_heads {num_heads}"
        )
    if dyn_taps.shape != (num_tokens, num_heads * kernel_size):
        raise ValueError(
            f"Expected dyn_taps shape {(num_tokens, num_heads * kernel_size)}, "
            f"got {tuple(dyn_taps.shape)}"
        )
    head_dim = channels // num_heads
    hist = (
        torch.zeros((kernel_size - 1, channels), dtype=x.dtype, device=x.device)
        if history is None
        else history
    )
    padded = torch.cat((hist, x), dim=0).to(torch.float32)
    static_fp32 = static_w.to(torch.float32)
    dyn_fp32 = dyn_taps.to(torch.float32).reshape(num_tokens, num_heads, kernel_size)
    acc = x.to(torch.float32).clone()
    for lag in range(kernel_size):
        start = (kernel_size - 1) - lag
        x_lag = padded[start : start + num_tokens]
        dyn_lag = (
            dyn_fp32[:, :, lag]
            .unsqueeze(-1)
            .expand(num_tokens, num_heads, head_dim)
            .reshape(num_tokens, channels)
        )
        acc = acc + (static_fp32[lag].unsqueeze(0) + dyn_lag) * x_lag
    out = (acc * ((2 * kernel_size + 1) ** -0.5)).to(x.dtype)
    new_hist = padded[-(kernel_size - 1) :].to(x.dtype) if kernel_size > 1 else hist
    return out, new_hist


def _expand_kv_heads(kv: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    seq_len, num_kv_heads, head_dim = kv.shape
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads {num_q_heads} must be divisible by "
            f"num_kv_heads {num_kv_heads}"
        )
    group_size = num_q_heads // num_kv_heads
    if group_size == 1:
        return kv
    return (
        kv.unsqueeze(2)
        .expand(seq_len, num_kv_heads, group_size, head_dim)
        .reshape(seq_len, num_q_heads, head_dim)
    )


def _rel_bias_matrix(
    r: torch.Tensor,
    rel_logits_proj: torch.Tensor,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
) -> torch.Tensor:
    extent = rel_logits_proj.shape[1]
    rel_scores = torch.einsum(
        "thd,de->the", r.to(torch.float32), rel_logits_proj.to(torch.float32)
    )
    dist = q_positions[:, None] - k_positions[None, :]
    in_extent = (dist >= 0) & (dist < extent)
    idx = dist.clamp(0, extent - 1).unsqueeze(1).expand(-1, r.shape[1], -1)
    bias = torch.where(
        in_extent.unsqueeze(1),
        torch.gather(rel_scores, dim=2, index=idx),
        torch.zeros_like(idx, dtype=torch.float32),
    )
    return bias.permute(1, 0, 2)


def _apply_gate_and_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    gated = out.to(torch.float32) * torch.sigmoid(gate.to(torch.float32))
    sink_scale = torch.sigmoid(lse.to(torch.float32) - sinks.to(torch.float32))
    return (gated * sink_scale.unsqueeze(-1)).to(out_dtype)


def _masked_attention_with_lse(
    scores: torch.Tensor,
    v_exp: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    has_key = mask.any(dim=-1)
    masked = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    lse = torch.where(
        has_key.unsqueeze(0),
        torch.logsumexp(masked, dim=-1),
        torch.full(
            (scores.shape[0], scores.shape[1]),
            float("-inf"),
            dtype=torch.float32,
            device=scores.device,
        ),
    )
    safe_lse = torch.where(has_key.unsqueeze(0), lse, torch.zeros_like(lse))
    probs = torch.where(
        mask.unsqueeze(0),
        torch.exp(masked - safe_lse.unsqueeze(-1)),
        torch.zeros_like(masked),
    )
    return torch.einsum("hqk,khd->qhd", probs, v_exp), lse


def relative_bias_attention_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
    rel_logits_proj: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    *,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Reference Inkling 2 relative-bias attention with post-softmax gate and sink."""
    _, num_heads, head_dim = q.shape
    k_exp = _expand_kv_heads(k, num_heads).to(torch.float32)
    v_exp = _expand_kv_heads(v, num_heads).to(torch.float32)
    scores = torch.einsum("qhd,khd->hqk", q.to(torch.float32), k_exp) * (head_dim**-0.5)
    scores = scores + _rel_bias_matrix(r, rel_logits_proj, q_positions, k_positions)
    dist = q_positions[:, None] - k_positions[None, :]
    visible = dist >= 0
    if sliding_window is not None:
        visible = visible & (dist < sliding_window)
    out, lse = _masked_attention_with_lse(scores, v_exp, visible)
    return _apply_gate_and_sink(out, lse.transpose(0, 1), gate, sinks, q.dtype)


def relative_bias_attention_split_tpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
    rel_logits_proj: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    *,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Split-attention formulation separating zero-bias far keys (``dist >= E``)
    for stock Pallas RPA from local relative-bias keys (``dist < E``), merged
    via exact FlashDecoding LSE before applying post-softmax gate and sink.
    """
    _, num_heads, head_dim = q.shape
    extent = rel_logits_proj.shape[1]
    k_exp = _expand_kv_heads(k, num_heads).to(torch.float32)
    v_exp = _expand_kv_heads(v, num_heads).to(torch.float32)
    base_scores = torch.einsum("qhd,khd->hqk", q.to(torch.float32), k_exp) * (
        head_dim**-0.5
    )
    dist = q_positions[:, None] - k_positions[None, :]
    visible = dist >= 0
    if sliding_window is not None:
        visible = visible & (dist < sliding_window)
    near_mask = visible & (dist < extent)
    far_mask = visible & (dist >= extent)

    near_scores = base_scores + _rel_bias_matrix(
        r, rel_logits_proj, q_positions, k_positions
    )
    out_near, lse_near = _masked_attention_with_lse(near_scores, v_exp, near_mask)
    if not far_mask.any():
        return _apply_gate_and_sink(
            out_near, lse_near.transpose(0, 1), gate, sinks, q.dtype
        )

    out_far, lse_far = _masked_attention_with_lse(base_scores, v_exp, far_mask)
    lse_max = torch.maximum(lse_near, lse_far)
    lse_safe = torch.where(torch.isinf(lse_max), torch.zeros_like(lse_max), lse_max)
    w_near = torch.exp(lse_near - lse_safe)
    w_far = torch.exp(lse_far - lse_safe)
    denom = (w_near + w_far).clamp_min(1e-30)
    lse_total = lse_safe + torch.log(denom)
    out_total = out_near * (w_near / denom).transpose(0, 1).unsqueeze(-1) + out_far * (
        w_far / denom
    ).transpose(0, 1).unsqueeze(-1)
    return _apply_gate_and_sink(
        out_total, lse_total.transpose(0, 1), gate, sinks, q.dtype
    )


def inkling2_moe_forward(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    topk: int = 8,
    route_scale: float = 8.0,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
) -> torch.Tensor:
    """Routed MoE reference with sigmoid+bias routing and interleaved softcapped SiTU.

    ``w13_weight`` stores interleaved ``(gate, up)`` rows; ``sigmoid(gate)`` is
    computed on the uncapped gate before applying ``gate_cap`` and ``up_cap``.
    """
    num_tokens, hidden = x.shape
    num_experts = gate_weight.shape[0]
    x_fp32 = x.to(torch.float32)
    logits = functional.linear(x_fp32, gate_weight.to(torch.float32))
    selection = torch.sigmoid(logits) + gate_bias.to(torch.float32)
    _, topk_ids = torch.topk(selection, k=topk, dim=-1, largest=True, sorted=True)
    gammas = (
        torch.softmax(functional.logsigmoid(torch.gather(logits, -1, topk_ids)), dim=-1)
        * route_scale
    )

    out = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=x.device)
    w13_fp32 = w13_weight.to(torch.float32)
    w2_fp32 = w2_weight.to(torch.float32)
    for expert_id in range(num_experts):
        tok_idx, slot_idx = torch.where(topk_ids == expert_id)
        if tok_idx.numel() == 0:
            continue
        gate_up = functional.linear(x_fp32[tok_idx], w13_fp32[expert_id])
        gate = gate_up[:, 0::2]
        up = gate_up[:, 1::2]
        sig = torch.sigmoid(gate)
        if gate_cap > 0:
            gate = gate_cap * torch.tanh(gate / gate_cap)
        if up_cap > 0:
            up = up_cap * torch.tanh(up / up_cap)
        gamma_e = gammas[tok_idx, slot_idx].unsqueeze(-1)
        act = (((gate * sig) * up) * gamma_e).to(torch.bfloat16).to(torch.float32)
        out.index_add_(0, tok_idx, functional.linear(act, w2_fp32[expert_id]))
    return out.to(x.dtype)
