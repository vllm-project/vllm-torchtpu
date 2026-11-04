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

"""Configuration for Vision Encoder fused QKV projection and 2D mRoPE.

Specifically tailored for vision transformers (e.g., Gemma-4 ViT) feeding
directly into FlashAttention (TpuMMEncoderAttention) without KV cache, paged memory,
or batched RPA layout indirection.
"""

from __future__ import annotations

import dataclasses
import enum


class Norm(enum.Enum):
    """Normalization mode."""
    NONE = "none"
    RMS = "rms"
    LAYER_NORM = "layer_norm"


@dataclasses.dataclass(frozen=True)
class VisionMropeSpec:
    """2D Multimodal Rotary Embedding specification.

    Attributes:
        rotary_dim: Total rotary dimension (e.g., 64). Must equal 2 * n_pairs.
        n_pairs: Total rotation pairs (rotary_dim // 2, e.g., 32).
        section_h: Rotation pairs assigned to spatial height dimension (e.g., 16).
        section_w: Rotation pairs assigned to spatial width dimension (e.g., 16).
        theta: Frequency base (typically 10000.0).
    """
    rotary_dim: int
    section_h: int
    section_w: int
    theta: float = 10000.0

    @property
    def n_pairs(self) -> int:
        return self.rotary_dim // 2

    def __post_init__(self):
        if self.section_h + self.section_w != self.n_pairs:
            raise ValueError(
                f"section_h ({self.section_h}) + section_w ({self.section_w}) "
                f"must equal n_pairs ({self.n_pairs})"
            )
        # Power of 2 requirement for bitwise in-register XOR lane swap (l ^ n)
        if (self.n_pairs & (self.n_pairs - 1)) != 0:
            raise ValueError(f"n_pairs ({self.n_pairs}) must be a power of 2 for XOR lane-swap.")


@dataclasses.dataclass(frozen=True)
class VisionAttentionConfig:
    """Attention configuration for vision encoder blocks.

    Attributes:
        hidden_size: Input feature dimension (e.g., 1152 for Gemma-4 vision).
        num_heads: Number of attention heads (e.g., 16 for Gemma-4 vision).
        head_dim: Dimension per attention head (e.g., 72 for Gemma-4 vision).
        mrope: 2D mRoPE specification. If None, no rotary embedding is applied.
        input_norm: Normalization before projection.
        bias: Whether linear projections include bias terms.
        dtype: Data type string (e.g., 'bfloat16', 'float32').
    """
    hidden_size: int = 1152
    num_heads: int = 16
    head_dim: int = 72
    mrope: VisionMropeSpec | None = dataclasses.field(
        default_factory=lambda: VisionMropeSpec(rotary_dim=64, section_h=16, section_w=16)
    )
    input_norm: Norm = Norm.NONE
    bias: bool = False
    dtype: str = "bfloat16"

    @property
    def qkv_dim(self) -> int:
        """Total projected dimension for Q, K, and V."""
        return 3 * self.num_heads * self.head_dim
