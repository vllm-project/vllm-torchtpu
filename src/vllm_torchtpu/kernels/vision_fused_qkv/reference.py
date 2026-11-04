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

"""Reference implementations for Vision QKV projection and 2D mRoPE.

Used as golden truth to prove bitwise numerical equivalence against the
in-register Lane-Swap FMA kernel.
"""

from __future__ import annotations

from typing import Sequence
from vllm_torchtpu.kernels.vision_fused_qkv.configs import VisionMropeSpec


def reference_neox_2d_mrope(
    x: Sequence[float],
    cos_vals: Sequence[float],
    sin_vals: Sequence[float],
    spec: VisionMropeSpec,
) -> list[float]:
    """Standard NeoX RoPE via slicing and concatenation.

    Steps:
      x1 = x[:n], x2 = x[n:2n], unrotated = x[2n:]
      y1 = x1 * cos - x2 * sin
      y2 = x2 * cos + x1 * sin
      return y1 + y2 + unrotated
    """
    n = spec.n_pairs
    x1 = x[:n]
    x2 = x[n : 2 * n]
    unrotated = x[2 * n :]

    y1 = [x1[i] * cos_vals[i] - x2[i] * sin_vals[i] for i in range(n)]
    y2 = [x2[i] * cos_vals[i] + x1[i] * sin_vals[i] for i in range(n)]

    return y1 + y2 + list(unrotated)
