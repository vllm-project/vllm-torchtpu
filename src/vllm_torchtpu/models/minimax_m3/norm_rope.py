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
"""Torch bridge for the one-pass q / k / index norm + RoPE kernel
(`kernels/norm_rope.py`), used by `attention.norm_rope_heads` on steps of at
least ``TPU_MINIMAX_M3_FUSED_NORM_ROPE_MIN_TOKENS`` tokens."""

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3.kernels.norm_rope import minimax_m3_norm_rope

logger = init_logger(__name__)

MIN_TOKENS = envs.TPU_MINIMAX_M3_FUSED_NORM_ROPE_MIN_TOKENS


def _fake(qkv, r, w, cos_sin, num_heads, num_kv_heads, head_dim, rot):
    heads = num_heads + 2 * num_kv_heads + 1
    return torch.empty(
        (qkv.shape[0], heads * head_dim), dtype=qkv.dtype, device=qkv.device
    )


# Registered at import: called from inside the attention forward.
_op = pallas.jax_op("pallas::minimax_m3_norm_rope", minimax_m3_norm_rope)
_op.register_fake(_fake)


def applies(num_tokens: int, qkv: torch.Tensor, rotary_emb) -> bool:
    """Whether this step takes the fused kernel: the flag is on, the step has
    at least ``MIN_TOKENS`` tokens, and the rope is the plain NeoX partial
    rotation the kernel reproduces."""
    return (
        MIN_TOKENS > 0
        and num_tokens >= MIN_TOKENS
        and qkv.device.type == "tpu"
        and qkv.dtype == torch.bfloat16
        and getattr(rotary_emb, "is_neox_style", False)
        and type(rotary_emb).__name__ == "TpuRotaryEmbedding"
    )


def fused_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    r: torch.Tensor,
    weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_emb,
) -> torch.Tensor:
    """``[T, heads, head_dim]`` normed + rotated q / k / index_q / index_k
    heads of ``qkv``; ``r [T, heads]`` f32 is the norms' ``rsqrt`` factor."""
    logger.info_once(
        "TPU_MINIMAX_M3_FUSED_NORM_ROPE_MIN_TOKENS=%d: one-pass q/k/index norm + "
        "rope kernel armed (first step %d tokens)",
        MIN_TOKENS,
        qkv.shape[0],
    )
    # Same cache rows as `rotary_embedding._rotary_forward_fp32`.
    cache = rotary_emb._match_cos_sin_cache_dtype(qkv)
    cos_sin = cache.index_select(0, positions.flatten())
    out = _op(
        qkv.contiguous(),
        r.contiguous(),
        weight.contiguous(),
        cos_sin.contiguous(),
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_emb.rotary_dim,
    )
    return out.view(qkv.shape[0], -1, head_dim)
