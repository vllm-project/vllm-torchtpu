# SPDX-License-Identifier: Apache-2.0
"""Torch bridge for the measured packed Engram specialization.

The staging interface follows vllm-project/vllm-torchtpu PR #2668;
the implementation retains FP32 keys while reducing the RMS gate.
"""

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.engram.kernel import fused_engram
from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import get_tpu_version


@torch.compiler.assume_constant_result
def _is_tpu_v7() -> bool:
    """Resolve immutable hardware outside the captured Torch tensor graph."""
    return get_tpu_version() == 7


def is_supported_by_fused_engram(
    packed: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> bool:
    """Whether tensors match the T288, K6144, four-stream specialization."""
    return (
        hidden.device.type == "tpu"
        and hidden.shape == (288, 4, 5120)
        and hidden.dtype == torch.bfloat16
        and packed.numel() == 288 * (6144 // 4 + 6144 // 128)
        and packed.dtype == torch.int32
        and weight.shape == (6144, 25600)
        and weight.dtype == torch.float8_e4m3fn
        and scale.shape == (25600,)
        and scale.dtype == torch.float32
        and query.shape == key.shape == (4, 5120)
        and query.dtype == key.dtype == torch.bfloat16
        and (mask is None or (mask.shape == (288,) and mask.dtype == torch.bool))
        and _is_tpu_v7()
    )


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


_engram_op = pallas.jax_op("pallas::engram_packed_fused", _engram_jax)


@_engram_op.register_fake
def _fake_engram(
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
    """Consume flat int32 staging with all raw words before all scale words."""
    return _engram_op(packed, hidden, weight, scale, query, key, mask, eps, clamp)
