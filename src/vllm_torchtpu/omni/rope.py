# SPDX-License-Identifier: Apache-2.0
"""TPU-friendly interleaved rotary embedding for vLLM-Omni Wan models.

``RotaryEmbeddingWan.forward_native`` rotates (even, odd) pairs through
``x.unflatten(-1, (-1, 2)).unbind(-1)`` and ``torch.stack(..., dim=-1)``.
On TPU the size-2 minor dimension forces sequence-minor relayouts plus
strided copies around every q/k rotation (the XLA "data formatting" and
copy ops around each attention layer in a device profile).

The same rotation is written here on the natural ``(..., head_dim)``
layout::

    out = x * cos2 + (x @ P) * sin2

where ``cos2``/``sin2`` repeat each half-dim frequency for its pair and
``P`` is the signed pair-swap permutation (``(x @ P)[2i] = -x[2i+1]``,
``(x @ P)[2i+1] = x[2i]``). The permutation runs on the otherwise idle
MXU and is exact (one +-1 term per output), so the result matches the
native formula term for term. On a v7x Wan2.2-A14B block at 480p (32760
tokens) this cuts ~7 ms per transformer layer (80.8 -> 73.9 ms).
"""

import functools

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.diffusion_attention.qk_norm_rope import qk_norm_rope


@functools.cache
def _pair_swap_matrix(
    dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    p = torch.zeros(dim, dim, dtype=torch.float32)
    even = torch.arange(0, dim, 2)
    p[even + 1, even] = -1.0
    p[even, even + 1] = 1.0
    return p.to(device=device, dtype=dtype)


def _repeat_pairs(t: torch.Tensor) -> torch.Tensor:
    """``(..., D/2) -> (..., D)`` repeating each entry for its (even, odd) pair."""
    return t.unsqueeze(-1).expand(*t.shape, 2).flatten(-2)


def apply_rotary_pair_swap(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Interleaved RoPE of ``x`` (..., D) with half-dim ``cos``/``sin`` (..., D/2).

    In ``torch.bfloat16`` (the dtype Wan and LingBot-World pass after
    ``to(x.dtype)``), ``x @ P`` is bit-exact because each column of ``P``
    has a single ``+-1`` entry. In ``torch.float32``, TPU matmuls default
    to bf16 input rounding unless high precision is configured.
    """
    swapped = torch.matmul(x, _pair_swap_matrix(x.shape[-1], x.dtype, x.device))
    return (x * _repeat_pairs(cos) + swapped * _repeat_pairs(sin)).to(x.dtype)


def rotary_embedding_wan_forward(
    self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Drop-in ``RotaryEmbeddingWan.forward`` for bf16 interleaved half-dim tables."""
    if not (self.interleaved and self.half_head_dim) or x.dtype != torch.bfloat16:
        return self._forward_method(x, cos, sin)
    return apply_rotary_pair_swap(x, cos, sin)


@functools.cache
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

    op = pallas.jax_op("vllm_torchtpu::wan_qk_norm_rope", wan_qk_norm_rope)

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
