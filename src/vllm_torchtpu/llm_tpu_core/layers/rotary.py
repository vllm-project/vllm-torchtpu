# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Hardware-adapted Rotary Position Embedding (RoPE) for TPU.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input (NeoX style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    """Rotates interleaved pairs of the hidden dimensions (GPT-J style)."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1)
    return x_rot.flatten(-2)


rotate_half = rotate_neox


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to queries and keys."""
    # cos, sin: [seq_len, rotary_dim] or [batch, seq_len, rotary_dim]
    if position_ids is not None:
        cos = cos[position_ids]  # [batch, seq_len, rotary_dim]
        sin = sin[position_ids]
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, rotary_dim]
        sin = sin.unsqueeze(0).unsqueeze(2)
    elif cos.ndim == 3:
        cos = cos.unsqueeze(2)  # [batch, seq_len, 1, rotary_dim]
        sin = sin.unsqueeze(2)

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]
    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    else:
        q_rot, q_pass = q, None
        k_rot, k_pass = k, None

    rotate_fn = rotate_neox if is_neox_style else rotate_gptj
    q_embed = (q_rot * cos) + (rotate_fn(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_fn(k_rot) * sin)

    if q_pass is not None:
        q_embed = torch.cat((q_embed, q_pass), dim=-1)
        k_embed = torch.cat((k_embed, k_pass), dim=-1)

    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding adapted for TPU acceleration."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int | None = None,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        is_neox_style: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim or head_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        if self.is_neox_style:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.repeat_interleave(freqs, 2, dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(self.dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(self.dtype), persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors."""
        max_pos = int(positions.max().item()) + 1 if positions.numel() > 0 else 0
        if max_pos > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(seq_len=max(max_pos, self.cos_cached.shape[0] * 2))

        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]

        if self.rotary_dim < self.head_size:
            query_rot = query[..., : self.rotary_dim]
            query_pass = query[..., self.rotary_dim :]
            key_rot = key[..., : self.rotary_dim]
            key_pass = key[..., self.rotary_dim :]
        else:
            query_rot = query
            query_pass = None
            key_rot = key
            key_pass = None

        if query.ndim == 4:
            if positions.ndim == 1:
                # query is [batch, seq_len, num_heads, head_size]
                # positions is [seq_len], cos is [seq_len, rotary_dim]
                # -> [1, seq_len, 1, rotary_dim]
                cos = cos.unsqueeze(0).unsqueeze(2)
                sin = sin.unsqueeze(0).unsqueeze(2)
            elif positions.ndim == 2:
                # positions is [batch, seq_len]
                # cos is [batch, seq_len, rotary_dim] -> [batch, seq_len, 1, rotary_dim]
                cos = cos.unsqueeze(2)
                sin = sin.unsqueeze(2)
        elif query.ndim == 3:
            # query is [tokens, num_heads, head_size]
            # cos is [tokens, rotary_dim] -> [tokens, 1, rotary_dim]
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
        q_embed = (query_rot * cos) + (rotate_fn(query_rot) * sin)
        k_embed = (key_rot * cos) + (rotate_fn(key_rot) * sin)

        if query_pass is not None:
            q_embed = torch.cat((q_embed, query_pass), dim=-1)
            k_embed = torch.cat((k_embed, key_pass), dim=-1)

        return q_embed, k_embed
