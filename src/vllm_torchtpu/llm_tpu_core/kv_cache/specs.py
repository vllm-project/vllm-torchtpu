# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""KV cache spec and config dataclasses for llm_tpu_core.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class KVCacheSpec:
    """Base specification for a KV cache layer."""

    block_size: int = 16
    dtype: torch.dtype = torch.bfloat16


@dataclass
class AttentionSpec(KVCacheSpec):
    """Specification for standard multi-head / grouped-query attention KV cache."""

    num_kv_heads: int = 1
    head_size: int = 128
    cache_layout: str = "NHD"
    page_size: int | None = None

    @property
    def page_shape(self) -> tuple[int, ...]:
        if self.cache_layout == "HND":
            return (self.num_kv_heads, self.block_size, self.head_size)
        return (self.block_size, self.num_kv_heads, self.head_size)


@dataclass
class EncoderOnlyAttentionSpec(AttentionSpec):
    """Specification for encoder-only attention KV cache."""

    pass


@dataclass
class MLAAttentionSpec(KVCacheSpec):
    """Specification for Multi-Head Latent Attention (MLA) compressed KV cache."""

    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    cache_layout: str = "NHD"
    page_size: int | None = None

    @property
    def head_size(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def num_kv_heads(self) -> int:
        return 1

    @property
    def page_shape(self) -> tuple[int, ...]:
        return (self.block_size, self.head_size)


@dataclass
class MambaSpec(KVCacheSpec):
    """Specification for Mamba recurrent SSM / Conv state cache."""

    d_state: int = 16
    d_conv: int = 4
    num_heads: int = 1
    head_dim: int = 64
    max_num_blocks_per_req: int = 1


class UniformTypeKVCacheSpecs:
    """Container for uniform type KV cache specs."""

    def __init__(self, kv_cache_specs: dict[str, KVCacheSpec] | None = None) -> None:
        self.kv_cache_specs = kv_cache_specs or {}


@dataclass
class AttentionGroup:
    """Group of attention layers sharing cache geometry."""

    spec: KVCacheSpec
    layer_names: list[str] = field(default_factory=list)
    stride_bytes: int = 0
    offset_bytes: int = 0


@dataclass
class KVCacheConfig:
    """Configuration describing the overall KV cache allocation."""

    num_blocks: int
    block_size: int
    kv_cache_groups: list[AttentionGroup] = field(default_factory=list)
    target_num_blocks: int | None = None
