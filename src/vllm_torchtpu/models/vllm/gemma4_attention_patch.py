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
"""Gemma4 attention from its fused qkv rows (TPU_GEMMA4_FUSED_ATTN_PROLOGUE).

Between its qkv and o projections, upstream ``Gemma4Attention.forward`` runs
per-head RMSNorms of q, k and v, neox rotary on q and k, and the attention
layer, whose TPU backend then quantizes K/V to FP8 and lays q and K/V out for
the batched RPA kernel. XLA runs that as a dozen passes with f32
intermediates. With the flag, ``attention_from_qkv`` hands the projection's
rows to ``PallasBatchedRPAAttentionBackendImpl.forward_from_qkv``, which does
all of it in one Pallas pass (``kernels/experimental/batched_rpa/prologue``).
It is a model patch because upstream exposes nothing between the qkv
projection and ``self.attn``: a backend or custom-op hook only ever sees the
normed, rotated q/k/v, never the projection's rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.models.gemma4 import Gemma4Attention

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_orig_attention_forward = Gemma4Attention.forward
_orig_attention_init = Gemma4Attention.__init__


def _upstream_attention(
    attn: Any, positions: torch.Tensor, qkv: torch.Tensor
) -> torch.Tensor:
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q = attn.q_norm(q.unflatten(-1, (attn.num_heads, attn.head_dim))).flatten(-2, -1)
    k = attn.k_norm(k.unflatten(-1, (attn.num_kv_heads, attn.head_dim)))
    q, k = attn.rotary_emb(positions, q, k.flatten(-2, -1))
    v = attn.v_norm(v.unflatten(-1, (attn.num_kv_heads, attn.head_dim)))
    return attn.attn(q, k, v.flatten(-2, -1))


def _fused_attention(
    attn: Any, positions: torch.Tensor, qkv: torch.Tensor
) -> torch.Tensor:
    rope = attn.rotary_emb
    if not rope.is_neox_style:
        raise NotImplementedError("TPU_GEMMA4_FUSED_ATTN_PROLOGUE needs neox rotary")
    # The rotary cache rows of the tokens, in the activation dtype as the
    # upstream rotary reads them.
    cos_sin = rope._match_cos_sin_cache_dtype(qkv).index_select(0, positions.flatten())
    # Contiguous: the op's operands are dense buffers, not strided views.
    cos, sin = (half.contiguous() for half in cos_sin.chunk(2, dim=-1))
    attn_metadata, layer, kv_cache, _ = get_attention_context(attn.attn.layer_name)
    return layer.impl.forward_from_qkv(
        layer,
        qkv,
        cos,
        sin,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.q_norm.variance_epsilon,
        kv_cache,
        attn_metadata,
    )


def attention_from_qkv(
    attn: Any, positions: torch.Tensor, qkv: torch.Tensor
) -> torch.Tensor:
    """``Gemma4Attention.forward`` between the qkv and the o projection."""
    if envs.TPU_GEMMA4_FUSED_ATTN_PROLOGUE:
        return _fused_attention(attn, positions, qkv)
    return _upstream_attention(attn, positions, qkv)


def _patched_attention_forward(
    self: Gemma4Attention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    if self.is_kv_shared_layer:  # only q is projected; K/V come from the cache
        return _orig_attention_forward(self, positions, hidden_states, **kwargs)
    qkv, _ = self.qkv_proj(hidden_states)
    output, _ = self.o_proj(_fused_attention(self, positions, qkv))
    return output


def _patched_attention_init(self: Gemma4Attention, *args: Any, **kwargs: Any) -> None:
    """Upstream's init, then a refusal of what forward_from_qkv cannot run,
    so it surfaces at model build rather than at the first trace."""
    from vllm_torchtpu.layers.adapter.attention import (
        PallasBatchedRPAAttentionBackendImpl,
    )

    _orig_attention_init(self, *args, **kwargs)
    if self.is_kv_shared_layer:
        return
    impl = self.attn.impl
    if not isinstance(impl, PallasBatchedRPAAttentionBackendImpl):
        raise NotImplementedError(
            "TPU_GEMMA4_FUSED_ATTN_PROLOGUE needs the batched RPA backend "
            f"(--attention-backend CUSTOM); {self.attn.layer_name} uses "
            f"{type(impl).__name__}"
        )
    unsupported = impl.qkv_prologue_unsupported()
    if not self.rotary_emb.is_neox_style:
        unsupported.append("non-neox rotary")
    if unsupported:
        raise NotImplementedError(
            f"TPU_GEMMA4_FUSED_ATTN_PROLOGUE does not support {', '.join(unsupported)}"
            f" ({self.attn.layer_name}); unset it for this configuration"
        )
    # The backend builds forward_from_qkv's op in initialize_kernel, before
    # the forward is traced.
    impl.qkv_prologue_eps = self.q_norm.variance_epsilon


def maybe_patch_gemma4_attention(model_config: ModelConfig | None) -> bool | None:
    """Returns False until a Gemma4 text model config is available."""
    if not envs.TPU_GEMMA4_FUSED_ATTN_PROLOGUE:
        return None
    if model_config is None or model_config.hf_text_config.model_type != "gemma4_text":
        return False
    Gemma4Attention.__init__ = _patched_attention_init
    Gemma4Attention.forward = _patched_attention_forward
    logger.info(
        "Applied TPU patch: Gemma4 attention takes its qkv rows straight to the "
        "batched RPA kernel (q/k/v norms, rotary and FP8 KV in one Pallas pass)."
    )
    return None
