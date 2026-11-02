# SPDX-License-Identifier: Apache-2.0
"""Torch boundary for the TP1 packed Engram kernel (PR #1853 adaptation)."""

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.engram.kernel import fused_engram


def _engram_jax(
    packed: jax.Array,
    hidden: jax.Array,
    weight: jax.Array,
    scale: jax.Array,
    query: jax.Array,
    key: jax.Array,
    mask: jax.Array,
    eps: float,
    clamp: float,
) -> jax.Array:
    return fused_engram(packed, hidden, weight, scale, query, key, mask, eps, clamp)


_op = pallas.jax_op("pallas::engram_offload_fused", _engram_jax)


@_op.register_fake
def _fake_engram(packed, hidden, weight, scale, query, key, mask, eps, clamp):
    return torch.empty_like(hidden)


def fused_engram_torch(
    packed: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
    clamp: float,
) -> torch.Tensor:
    """Consume the existing flat raw-then-scale staging allocation."""
    return _op(packed, hidden, weight, scale, query, key, mask, eps, clamp)
