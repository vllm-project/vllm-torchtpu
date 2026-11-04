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

"""2D Multimodal Rotary Position Embedding (2D mRoPE) via in-register Lane-Swap FMA.

Adapted from the Lane-swap technique (go/fused_qkv_proj, cl/989795783)
for Gemma-4 2D vision grids (b/581011672).

Replaces tensor slicing (torch.chunk / jnp.split) and concatenation (torch.cat)
with an in-register bitwise XOR lane permutation (l ^ n) and a single FMA:
    Y[l] = X[l] * C[l] + X[l ^ n] * S[l]
"""

from __future__ import annotations

import math
from typing import Sequence

from vllm_torchtpu.kernels.vision_fused_qkv.configs import VisionMropeSpec


def compute_inv_freq(rotary_dim: int, theta: float = 10000.0) -> list[float]:
    """Compute base inverse frequencies for rotary embedding."""
    n_pairs = rotary_dim // 2
    return [1.0 / (theta ** (2 * i / rotary_dim)) for i in range(n_pairs)]


def compute_2d_mrope_angles(
    pos_h: int,
    pos_w: int,
    inv_freq: Sequence[float],
    section_h: int,
    section_w: int,
) -> tuple[list[float], list[float]]:
    """Compute cos and sin factors for spatial patch coordinates (pos_h, pos_w).

    Args:
        pos_h: Height index on the image patch grid.
        pos_w: Width index on the image patch grid.
        inv_freq: Precomputed inverse frequencies of length n_pairs.
        section_h: Number of rotation pairs allocated to height.
        section_w: Number of rotation pairs allocated to width.

    Returns:
        tuple of (cos_vals, sin_vals) each of length n_pairs.
    """
    n_pairs = len(inv_freq)
    assert section_h + section_w == n_pairs
    cos_vals: list[float] = []
    sin_vals: list[float] = []
    for i in range(n_pairs):
        pos = pos_h if i < section_h else pos_w
        angle = pos * inv_freq[i]
        cos_vals.append(math.cos(angle))
        sin_vals.append(math.sin(angle))
    return cos_vals, sin_vals


class LaneRope2D:
    """In-register Lane-Swap 2D mRoPE operator.

    Transforms per-head features without slicing or memory round-trips.
    Pre-expands cos/sin into full head_dim arrays:
      C[l]: cos factor (or 1.0 for unrotated lanes)
      S[l]: signed sin factor (-sin for l < n, +sin for n <= l < 2n, 0.0 for unrotated)
      partner[l]: l ^ n for rotated lanes, l for unrotated lanes
    """

    def __init__(
        self,
        spec: VisionMropeSpec,
        head_dim: int,
        cos_vals: Sequence[float],
        sin_vals: Sequence[float],
    ):
        self.spec = spec
        self.head_dim = head_dim
        n = spec.n_pairs

        self.C = [1.0] * head_dim
        self.S = [0.0] * head_dim
        self.partner = [i for i in range(head_dim)]

        for l in range(head_dim):
            if l < n:
                self.C[l] = cos_vals[l]
                self.S[l] = -sin_vals[l]
                self.partner[l] = l ^ n
            elif l < 2 * n:
                angle_idx = l - n
                self.C[l] = cos_vals[angle_idx]
                self.S[l] = sin_vals[angle_idx]
                self.partner[l] = l ^ n
            else:
                self.C[l] = 1.0
                self.S[l] = 0.0
                self.partner[l] = l

    def rotate_head_list(self, x: Sequence[float]) -> list[float]:
        """Apply in-register lane swap FMA to a single head vector."""
        head_dim = self.head_dim
        partner = self.partner
        C = self.C
        S = self.S
        # Y[l] = X[l] * C[l] + X[partner[l]] * S[l]
        return [x[l] * C[l] + x[partner[l]] * S[l] for l in range(head_dim)]
