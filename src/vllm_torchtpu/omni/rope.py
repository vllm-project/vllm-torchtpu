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

import torch


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
    """Interleaved RoPE of ``x`` (..., D) with half-dim ``cos``/``sin`` (..., D/2)."""
    swapped = torch.matmul(x, _pair_swap_matrix(x.shape[-1], x.dtype, x.device))
    return (x * _repeat_pairs(cos) + swapped * _repeat_pairs(sin)).to(x.dtype)


def rotary_embedding_wan_forward(
    self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Drop-in ``RotaryEmbeddingWan.forward`` for interleaved half-dim tables."""
    if not (self.interleaved and self.half_head_dim):
        return self._forward_method(x, cos, sin)
    return apply_rotary_pair_swap(x, cos, sin)
