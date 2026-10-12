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
    """Restore the TP axis before calling the local Pallas kernel.

    raw is int32 [P*T,C/4] and scales is int32 [P*T,C/(4*B)], where
    P=tp_size, T is full padded tokens, C is width per head shard, and B is
    scale block size. They become [P,T,C/4] and [P,T,C/(4*B)]. Other arguments
    and the BF16 [T/P,S,D] result follow fused_engram's shape contract.
    """
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
        """Return BF16 [T/P,S,D]; inputs follow fused_engram_torch's contract."""
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
    """Run the fused device computation after the caller's TP all-gather.

    P=tp_size, T is full padded tokens, t=T/P is local tokens, C is embedding
    width per head shard, B is embedding scale block size, S is stream count,
    and D is hidden width. K=P*C is the projection input width.

    Args:
        raw: int32 [P*T, C/4], packed FP8 E4M3FN rows from every TP rank.
        scales: int32 [P*T, C/(4*B)], packed E8M0 embedding scales.
        hidden: BF16 [t, S, D], this rank's residual inputs.
        weight: FP8 [K, (S+1)*D], processed WKV weights.
        weight_scale: FP32 [(S+1)*D], per-output-channel scales.
        query_weight: BF16 [S, D], query normalization weights.
        key_weight: BF16 [S, D], key normalization weights.
        token_mask: [t], boolean or numeric 0/1 for this rank's tokens.
        tp_size: Scalar number of equally sized TP head shards.
        rank: Scalar TP rank identifying this rank's SP token interval.
        eps: Scalar RMS normalization epsilon.
        clamp_value: Scalar lower bound before the gate's square root.

    Returns:
        BF16 [t, S, D], matching hidden's layout.
    """
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
