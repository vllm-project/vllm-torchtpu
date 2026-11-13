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
"""TPU-friendly rotary embeddings for vLLM-Omni diffusion models.

``RotaryEmbedding.forward_native`` and ``RotaryEmbeddingWan.forward_native``
rotate pairs through minor-dim strided slices / unbinds. On TPU the size-2
minor dimension forces sequence-minor relayouts plus strided copies around
every q/k rotation (the XLA "data formatting" and copy ops around each
attention layer in a device profile).

The same rotation is written here on the natural ``(..., head_dim)``
layout via ``tpu_apply_rotary`` (supporting both interleaved and NeoX
styles)::

    out = x * cos2 + (x @ P) * sin2

where ``cos2``/``sin2`` repeat each half-dim frequency for its pair and
``P`` is the signed pair-swap or half-swap permutation. The permutation runs
on the otherwise idle MXU and is exact (one +-1 term per output), so the
result matches the native formula term for term. On a v7x Wan2.2-A14B block
at 480p (32760 tokens) this cuts ~7 ms per transformer layer (80.8 -> 73.9 ms).
"""

import functools
from functools import cache as run_once

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.diffusion_attention.qk_norm_rope import qk_norm_rope


def _repeat_pairs(t: torch.Tensor) -> torch.Tensor:
    """``(..., D/2) -> (..., D)`` repeating each entry for its (even, odd) pair."""
    return t.unsqueeze(-1).expand(*t.shape, 2).flatten(-2)


def apply_rotary_pair_swap(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Interleaved RoPE of ``x`` (..., D) with half-dim ``cos``/``sin`` (..., D/2)."""
    return tpu_apply_rotary(x, cos, sin, interleaved=True)


@functools.cache
def _rope_matrices(
    head_dim: int, interleaved: bool, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constant matrices for ``rotate_half`` and half-dim cos/sin expansion."""
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    half = head_dim // 2
    r = torch.zeros(head_dim, head_dim)
    e = torch.zeros(half, head_dim)
    idx = torch.arange(half)
    if interleaved:
        r[2 * idx + 1, 2 * idx] = -1.0
        r[2 * idx, 2 * idx + 1] = 1.0
        e[idx, 2 * idx] = 1.0
        e[idx, 2 * idx + 1] = 1.0
    else:
        r[idx + half, idx] = -1.0
        r[idx, idx + half] = 1.0
        e[idx, idx] = 1.0
        e[idx, idx + half] = 1.0
    return r.to(dtype=dtype, device=device), e.to(dtype=dtype, device=device)


@torch._dynamo.assume_constant_result
def _rope_matrices_const(
    head_dim: int, interleaved: bool, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``_rope_matrices`` resolved eagerly while dynamo traces."""
    return _rope_matrices(head_dim, interleaved, dtype, device)


def tpu_apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = True,
) -> torch.Tensor:
    """``x * cos + rotate_half(x) * sin`` via constant 0/+-1 MXU matmuls."""
    head_dim = x.shape[-1]
    r, e = _rope_matrices_const(head_dim, interleaved, x.dtype, x.device)
    cos_full = cos.to(x.dtype) @ e
    sin_full = sin.to(x.dtype) @ e
    if cos_full.dim() == 2 and x.dim() >= 3:
        cos_full = cos_full.unsqueeze(-2)
        sin_full = sin_full.unsqueeze(-2)
    return (x * cos_full + (x @ r) * sin_full).to(x.dtype)


def rotary_embedding_wan_forward(
    self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Drop-in ``RotaryEmbeddingWan.forward`` for bf16 interleaved half-dim tables."""
    if not (self.interleaved and self.half_head_dim) or x.dtype != torch.bfloat16:
        return self._forward_method(x, cos, sin)
    # vLLM-Omni 0.32 (vllm-omni#8112) lets callers such as LingBot-World pass
    # (tokens, D/2) tables; expand them to (1, S, 1, D/2) as forward_native does.
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    return apply_rotary_pair_swap(x, cos, sin)


@functools.cache
def _first_half_lanes(width: int, head_dim: int, device: torch.device) -> torch.Tensor:
    return (torch.arange(width) % head_dim < head_dim // 2).to(device)


def split_rope_natural_tables(
    cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """LTX-2 split-RoPE tables ``(B, H, T, D/2) -> (B, 1, T, H * D)``.

    Lays the tables out like q/k (head-major channels): entries ``j`` and
    ``j + D/2`` of a head share half-dim frequency ``j``, and the sine table
    carries the sign of each half (``-sin`` then ``sin``). T stays at dim 2,
    where LTX-2's sequence-parallel plan splits the tables.
    """
    b, heads, tokens, half = cos.shape

    def natural(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return (
            torch.cat([first, second], dim=-1)
            .transpose(1, 2)
            .reshape(b, 1, tokens, heads * 2 * half)
        )

    return natural(cos, cos), natural(-sin, sin)


def apply_split_rotary_half_swap(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, head_dim: int
) -> torch.Tensor:
    """Split RoPE of ``x`` (B, T, H * D) with ``split_rope_natural_tables`` tables.

    Rolling the channels by -D/2 and +D/2 and selecting per lane puts each
    head's other half under every entry, so with the signed sine table
    ``out[j] = x[j] cos[j] - x[j + D/2] sin[j]`` and
    ``out[j + D/2] = x[j + D/2] cos[j] + x[j] sin[j]`` stay elementwise on
    q/k's own (B, T, H * D) layout, with no per-head reshape. As in
    vLLM-Omni's ``apply_split_rotary_emb``, the products and the sum are
    taken in the promoted dtype of ``x`` and the tables (``addcmul_``); the
    result is rounded to ``x.dtype`` once, where vLLM-Omni's attention
    processor casts it.
    """
    half = head_dim // 2
    swapped = torch.where(
        _first_half_lanes(x.shape[-1], head_dim, x.device),
        x.roll(-half, dims=-1),
        x.roll(half, dims=-1),
    )
    out = x * cos.squeeze(1)
    out.addcmul_(swapped, sin.squeeze(1))
    return out.to(x.dtype)


@run_once
def _qk_norm_rope_op():
    """Builds the torch custom op wrapping the fused Pallas kernel (once)."""

    def wan_qk_norm_rope(
        qkv: jax.Array,
        wq: jax.Array,
        wk: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        eps: float,
        head_dim: int,
    ) -> tuple[jax.Array, jax.Array]:
        return qk_norm_rope(qkv, wq, wk, cos, sin, eps=eps, head_dim=head_dim)

    op = pallas.jax_op("pallas::wan_qk_norm_rope", wan_qk_norm_rope)

    def fake(qkv, wq, wk, cos, sin, eps, head_dim):
        shape = (qkv.shape[0], qkv.shape[1] // 3)
        return qkv.new_empty(shape), qkv.new_empty(shape)

    op.register_fake(fake)
    return op


def can_fuse_qk_norm_rope(
    attn, hidden_states: torch.Tensor, rotary_emb: tuple | None
) -> bool:
    """Whether a WanSelfAttention call fits the fused q/k norm + RoPE kernel."""
    # kernel supports bf16 MHA with per-row RMSNorm, no gate compression,
    # head_dim % 128 == 0, half-dim RoPE tables.
    from vllm_omni.diffusion.layers.norm import RMSNorm

    return (
        hidden_states.dtype == torch.bfloat16
        and rotary_emb is not None
        and attn.to_gate_compress is None
        and attn.num_kv_heads == attn.num_heads
        and attn.head_dim % 128 == 0
        and isinstance(attn.norm_q, RMSNorm)
        and isinstance(attn.norm_k, RMSNorm)
    )


def wan_self_attention_fused(
    attn, hidden_states: torch.Tensor, rotary_emb: tuple, attn_metadata=None
) -> torch.Tensor:
    """``WanSelfAttention.forward`` with the fused q/k norm + RoPE kernel.

    Same computation as vLLM-Omni's forward (QKV projection, q/k RMSNorm over
    all heads, interleaved half-dim RoPE, attention, output projection).
    """
    qkv, _ = attn.to_qkv(hidden_states)
    batch, seq, width = qkv.shape
    cos, sin = rotary_emb  # (1 or batch, seq, 1, head_dim / 2)

    def table(t: torch.Tensor) -> torch.Tensor:
        t = _repeat_pairs(t.reshape(t.shape[0], seq, -1))
        return t.expand(batch, seq, attn.head_dim).reshape(batch * seq, -1)

    query, key = _qk_norm_rope_op()(
        qkv.reshape(batch * seq, width),
        attn.norm_q.weight,
        attn.norm_k.weight,
        table(cos),
        table(sin),
        attn.norm_q.variance_epsilon,
        attn.head_dim,
    )
    heads = (attn.num_heads, attn.head_dim)
    query = query.reshape(batch, seq, *heads)
    key = key.reshape(batch, seq, *heads)
    value = qkv[..., 2 * attn.num_heads * attn.head_dim :].unflatten(2, heads)
    out = attn.attn(query, key, value, attn_metadata)
    out = out.flatten(2, 3).type_as(query)
    return attn.dropout(attn.to_out(out))
