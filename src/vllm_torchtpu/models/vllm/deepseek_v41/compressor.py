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
"""DeepSeek-V4.1 KV/score compressor and ring state cache for TPU."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from vllm.config import VllmConfig
from vllm.ir.ops import rms_norm
from vllm.models.deepseek_v41 import compressor as compressor_upstream
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_torchtpu.models.vllm.deepseek_v4 import attention as v4_attn

from .cache import NativeCacheView, PagedCacheRows, write_cache
from .metadata import (
    active_cache,
    cache_slots,
    layer_meta,
    not_replayed,
    query_token_requests,
    rotate_rope,
)

_COMPRESS_PAIR_SIZE: int = 2
_DEFAULT_PAGE_SIZE: int = 16


class AttentionBackend(compressor_upstream.CompressorBackend):
    """TPU sparse MLA attention backend supporting arbitrary kernel block sizes."""

    @classmethod
    def get_supported_kernel_block_sizes(
        cls, kv_cache_spec: KVCacheSpec | None = None
    ) -> list[int | MultipleOf]:
        """Return supported kernel block-size multiples for TPU sparse MLA."""
        del kv_cache_spec
        return [MultipleOf(1)]

    @classmethod
    def get_page_size(cls, vllm_config: VllmConfig) -> int:
        """Return the configured KV cache page block size (or ``16`` when unset)."""
        block_size = vllm_config.cache_config.block_size
        return block_size if block_size is not None else _DEFAULT_PAGE_SIZE

    @classmethod
    def get_min_page_size(cls, vllm_config: VllmConfig) -> int:
        """Return the minimum supported page block size (``1`` token)."""
        del vllm_config
        return 1


# pyrefly: ignore[inconsistent-inheritance]
class CompressorStateCache(NativeCacheView, compressor_upstream.CompressorStateCache):
    """Per-request ring cache of unpaired ``[kv, score]`` rows."""

    def __init__(self, state_dim: int, dtype: torch.dtype, prefix: str) -> None:
        compressor_upstream.CompressorStateCache.__init__(
            self, state_dim, dtype, prefix
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Return the TPU :class:`AttentionBackend` class."""
        return AttentionBackend


def save_compress_norm(
    kv_score: Tensor,
    positions: Tensor,
    state_cache: PagedCacheRows | Tensor,
    slot_mapping: Tensor | None,
    norm_weight: Tensor,
    eps: float,
    compress_ratio: int,
    query_start_loc: Tensor | None = None,
) -> Tensor:
    """Save unpaired compressor state rows and RMSNorm compressed latents.

    Data lifecycle:
        Used: ``kv_score`` ``[tokens, (2 if ratio > 1 else 1) * head_dim]``,
            ``positions``, ``slot_mapping``, ``norm_weight``, ``query_start_loc``.
        Changed: ``state_cache`` updated via ``write_cache(..., after_read=True)``
            for even-position rows in each request's trailing ``ring_capacity``.
            Relies on ``NativeCacheView.zero_blocks`` for newly claimed blocks
            (genuine NaNs fail fast on odd cross-step reads without silent
            element-wise filtering, while ``torch.where`` avoids ``0 * NaN`` on
            even or in-chunk rows).
        Created: Returns ``bfloat16[tokens, head_dim]`` normalized latents.
    """
    head_dim = norm_weight.shape[0]
    kv_f32 = kv_score[:, :head_dim].float()
    weight_f32 = norm_weight.float()
    if compress_ratio == 1 or slot_mapping is None or not state_cache.numel():
        return rms_norm(kv_f32, weight_f32, eps).to(torch.bfloat16)

    pos_i64 = positions.to(torch.int64)
    valid_slots = slot_mapping >= 0
    ring_capacity = state_cache.shape[1]
    num_tokens = positions.shape[0]
    token_rows = torch.arange(num_tokens, device=positions.device)
    if query_start_loc is not None and query_start_loc.numel() > 1:
        query_ends = query_start_loc[1:].to(positions.device)
        req_indices = query_token_requests(query_start_loc, positions).long()
        clamped_reqs = req_indices.clamp(max=query_ends.shape[0] - 1)
        req_ends = query_ends[clamped_reqs].to(torch.int64)
    else:
        req_indices = torch.zeros_like(token_rows)
        req_ends = torch.full_like(pos_i64, num_tokens)

    prev_row_idx = (token_rows - 1).clamp_min(0)
    in_chunk = (
        (token_rows > 0)
        & (positions[prev_row_idx] == positions - 1)
        & (req_indices[prev_row_idx] == req_indices)
    )
    is_second = ((pos_i64 % _COMPRESS_PAIR_SIZE == 1) & valid_slots).unsqueeze(-1)
    prev_blocks = (slot_mapping.clamp_min(0) // ring_capacity).long()
    prev_slot = (pos_i64 - 1).clamp_min(0) % ring_capacity
    raw_prev = state_cache[prev_blocks, prev_slot].to(kv_score.dtype)
    use_prev = is_second & ~in_chunk.unsqueeze(-1)
    prev_rows = torch.where(use_prev, raw_prev, 0)
    paired = torch.where(in_chunk.unsqueeze(-1), kv_score[prev_row_idx], prev_rows)

    pair_kv = torch.stack((paired[:, :head_dim].float(), kv_f32), dim=1)
    score_prev = paired[:, head_dim:].float()
    score_curr = kv_score[:, head_dim:].float()
    gate_weights = torch.softmax(torch.stack((score_prev, score_curr), dim=1), dim=1)
    pooled = (pair_kv * gate_weights).sum(dim=1)

    publish_mask = (
        (pos_i64 % _COMPRESS_PAIR_SIZE == 0)
        & valid_slots
        & (token_rows >= req_ends - ring_capacity)
    )
    write_cache(state_cache, slot_mapping, kv_score, publish_mask, after_read=True)
    normed = rms_norm(torch.where(is_second, pooled, kv_f32), weight_f32, eps)
    return normed.to(torch.bfloat16)


class DeepseekCompressor(compressor_upstream.DeepseekCompressor):
    """TPU compressor layer using ``PagedCacheRows`` for state and KV writes.

    Reuses ``VllmDeepseekV4MLAAttention.kv_rope_op`` (which caches the lazy
    ``pallas::deepseek_v4_kv_rope`` custom op in ``self.__dict__``) so model
    modules under ``deepseek_v41/`` remain free of direct ``jax`` imports.
    """

    kv_rope_op = cast(
        Callable[[Tensor, Tensor, Tensor], Tensor],
        v4_attn.VllmDeepseekV4MLAAttention.kv_rope_op,
    )

    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix: str = "",
    ) -> None:
        # TODO(upstream): vllm/models/deepseek_v41/compressor.py:235 hardcodes
        # CompressorStateCache instead of exposing a platform state_cache_cls hook.
        super().__init__(
            vllm_config=vllm_config,
            compress_ratio=compress_ratio,
            hidden_size=hidden_size,
            head_dim=head_dim,
            rotate=rotate,
            prefix=prefix,
            k_cache_prefix=k_cache_prefix,
        )
        self.window_size: int = vllm_config.model_config.hf_config.sliding_window
        if self.compress_ratio > 1:
            state_prefix = f"{prefix}.state_cache"
            state_dim = _COMPRESS_PAIR_SIZE * self.head_dim
            # Evict base CompressorStateCache before registering NativeCacheView.
            self._static_forward_context.pop(state_prefix, None)
            self.state_cache = CompressorStateCache(
                state_dim=state_dim,
                dtype=torch.float32,
                prefix=state_prefix,
            )

    def project(self, hidden_states: Tensor) -> Tensor:
        """Project ``hidden_states`` into float32 ``[kv, score]`` rows.

        Args:
            hidden_states: ``[num_tokens, hidden_size]`` activations (Used).

        Returns:
            ``float32[num_tokens, state_dim]`` projected ``[kv, score]`` tensor
            (Created).
        """
        weight = cast(Tensor, self.fused_wkv_wgate.weight)
        return torch.mm(hidden_states.float(), weight.float().T)

    def forward(self, kv_score: Tensor, positions: Tensor) -> Tensor:
        """Compress ``kv_score`` across token groups and return BF16 latents.

        Args:
            kv_score: ``float32[num_tokens, state_dim]`` projected tensor (Used).
            positions: ``int32[num_tokens]`` token positions (Used).

        Returns:
            ``bfloat16[num_tokens, head_dim]`` RMS-normalized latents (Created).
        """
        if self.compress_ratio > 1 and self.state_cache is not None:
            meta = layer_meta(self.state_cache.prefix)
            state_kv: PagedCacheRows | Tensor = active_cache(self.state_cache)
            has_state = True
        else:
            meta = layer_meta(self.k_cache_prefix)
            state_kv = torch.tensor([])
            has_state = False
        if has_state and meta is not None and bool(state_kv.numel()):
            slots: Tensor | None = cache_slots(state_kv, meta, positions, ring=True)
        else:
            slots = None
        query_start_loc = (
            cast(Tensor | None, meta.query_start_loc) if meta is not None else None
        )
        return save_compress_norm(
            kv_score,
            positions,
            state_kv,
            slots,
            cast(Tensor, self.norm.weight),
            self.rms_norm_eps,
            self.compress_ratio,
            query_start_loc,
        )

    def insert_cache(
        self,
        latent: Tensor | None,
        positions: Tensor,
        rotary_emb: nn.Module,
    ) -> None:
        """Rotate ``latent`` at group positions and publish non-replayed KV rows."""
        if latent is None:
            return
        k_cache = active_cache(self._static_forward_context[self.k_cache_prefix])
        meta = layer_meta(self.k_cache_prefix)
        ratio = self.compress_ratio
        if meta is not None and k_cache.numel():
            rotated = rotate_rope(
                latent, positions // ratio, rotary_emb, self.kv_rope_op
            )
            slots = cache_slots(k_cache, meta, positions, compress_ratio=ratio)
            keep = ((positions + 1) % ratio == 0) & not_replayed(
                meta, positions, self.window_size
            )
            write_cache(k_cache, slots, rotated, keep)
