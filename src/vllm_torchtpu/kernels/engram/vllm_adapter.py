# SPDX-License-Identifier: Apache-2.0
"""Torch bridge for the packed offload Engram computation."""

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.engram.kernel import fused_engram

_op = None


def _engram_jax(
    raw: jax.Array,
    scales: jax.Array,
    hidden: jax.Array,
    weight: jax.Array,
    weight_scale: jax.Array,
    query_weight: jax.Array,
    key_weight: jax.Array,
    token_mask: jax.Array,
    tp_size: int,
    rank: int,
    eps: float,
    clamp_value: float,
) -> jax.Array:
    return fused_engram(
        raw.reshape(tp_size, -1, raw.shape[-1]),
        scales.reshape(tp_size, -1, scales.shape[-1]),
        hidden,
        weight,
        weight_scale,
        query_weight,
        key_weight,
        token_mask,
        rank,
        eps,
        clamp_value,
    )


def initialize_fused_engram() -> None:
    global _op
    if _op is not None:
        return
    # The Torch caller completes TP communication before this local kernel.
    _op = pallas.jax_op("pallas::engram_offload_fused", _engram_jax)

    def fake(
        raw, scales, hidden, weight, weight_scale, q, k, mask, tp, rank, eps, clamp
    ):
        return torch.empty_like(hidden)

    _op.register_fake(fake)


def fused_engram_torch(
    raw: torch.Tensor,
    scales: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    token_mask: torch.Tensor,
    tp_size: int,
    rank: int,
    eps: float,
    clamp_value: float,
) -> torch.Tensor:
    assert _op is not None, "initialize_fused_engram must run before model tracing"
    return _op(
        raw,
        scales,
        hidden,
        weight,
        weight_scale,
        query_weight,
        key_weight,
        token_mask,
        tp_size,
        rank,
        eps,
        clamp_value,
    )
