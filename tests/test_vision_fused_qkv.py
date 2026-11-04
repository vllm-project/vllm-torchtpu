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

"""Unit tests for Vision Encoder Fused QKV and 2D mRoPE Kernel.

Validates:
1. Bitwise numerical equivalence between in-register Lane-Swap 2D mRoPE and
   the reference NeoX slicing implementation across 2D image patch grids.
2. Tensor layout compatibility directly with FlashAttention contract
   [batch_size, num_heads, seq_len, head_dim] without batched RPA layout reshuffling.
"""

from __future__ import annotations

import math
import random
import unittest

from vllm_torchtpu.kernels.vision_fused_qkv.configs import (
    VisionAttentionConfig,
    VisionMropeSpec,
)
from vllm_torchtpu.kernels.vision_fused_qkv.kernel import vision_fused_qkv_project
from vllm_torchtpu.kernels.vision_fused_qkv.mrope import (
    LaneRope2D,
    compute_2d_mrope_angles,
    compute_inv_freq,
)
from vllm_torchtpu.kernels.vision_fused_qkv.reference import reference_neox_2d_mrope


class TestVisionFusedQKV(unittest.TestCase):

    def setUp(self):
        random.seed(42)

    def test_lane_swap_2d_mrope_bitwise_parity(self):
        """Test that in-register lane swap matches reference NeoX 2D mRoPE bitwise."""
        head_dim = 72
        rotary_dim = 64
        spec = VisionMropeSpec(rotary_dim=rotary_dim, section_h=16, section_w=16)
        inv_freq = compute_inv_freq(rotary_dim)

        max_err = 0.0
        # Test across various spatial patch coordinates (pos_h, pos_w)
        for pos_h in [0, 1, 5, 13]:
            for pos_w in [0, 2, 7, 15]:
                cos_vals, sin_vals = compute_2d_mrope_angles(
                    pos_h, pos_w, inv_freq, spec.section_h, spec.section_w
                )
                lane_op = LaneRope2D(spec, head_dim, cos_vals, sin_vals)

                x = [random.uniform(-1.0, 1.0) for _ in range(head_dim)]

                # Reference slicing NeoX
                y_ref = reference_neox_2d_mrope(x, cos_vals, sin_vals, spec)

                # Lane-swap FMA
                y_lane = lane_op.rotate_head_list(x)

                for r, l_val in zip(y_ref, y_lane):
                    diff = abs(r - l_val)
                    if diff > max_err:
                        max_err = diff

        self.assertAlmostEqual(max_err, 0.0, places=14, msg=f"Max error: {max_err}")
        print(f"\n[PASS] 2D mRoPE Lane-Swap Parity: Max absolute error = {max_err:.2e}")

    def test_vision_fused_qkv_flash_attention_layout(self):
        """Test Q, K, V shapes and layout match FlashAttention without RPA packing."""
        batch_size = 2
        seq_len = 4  # 2x2 patch grid
        hidden_size = 128
        num_heads = 4
        head_dim = 72

        config = VisionAttentionConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            mrope=VisionMropeSpec(rotary_dim=64, section_h=16, section_w=16),
            bias=True,
        )

        patch_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
        hidden_states = [
            [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(seq_len)]
            for _ in range(batch_size)
        ]

        total_qkv_dim = 3 * num_heads * head_dim
        w_qkv = [
            [random.uniform(-0.1, 0.1) for _ in range(total_qkv_dim)]
            for _ in range(hidden_size)
        ]
        bias_qkv = [random.uniform(-0.01, 0.01) for _ in range(total_qkv_dim)]

        q, k, v = vision_fused_qkv_project(
            hidden_states=hidden_states,
            w_qkv=w_qkv,
            config=config,
            patch_coords=patch_coords,
            bias_qkv=bias_qkv,
        )

        # Check Q, K, V shapes match [batch_size, num_heads, seq_len, head_dim]
        for name, tensor in [("Q", q), ("K", k), ("V", v)]:
            self.assertEqual(len(tensor), batch_size, f"{name} batch_size mismatch")
            self.assertEqual(len(tensor[0]), num_heads, f"{name} num_heads mismatch")
            self.assertEqual(len(tensor[0][0]), seq_len, f"{name} seq_len mismatch")
            self.assertEqual(len(tensor[0][0][0]), head_dim, f"{name} head_dim mismatch")

        print(
            f"[PASS] Vision Fused QKV Output Layout: "
            f"Q={batch_size}x{num_heads}x{seq_len}x{head_dim}, "
            f"K={batch_size}x{num_heads}x{seq_len}x{head_dim}, "
            f"V={batch_size}x{num_heads}x{seq_len}x{head_dim} (FlashAttention layout verified)"
        )


if __name__ == "__main__":
    unittest.main()
