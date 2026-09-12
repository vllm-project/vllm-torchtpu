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
"""Unit tests for GDN v3 decode tile size clamping against VMEM capacity limits."""

import unittest

from vllm_torchtpu.kernels.gdn.v3.wrapper import compute_decode_tile_size


class GDNV3TileClampingTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.vmem_64mib = 64 * 1024 * 1024  # Cloud TPU v7x (64 MiB per TensorCore)
        self.vmem_32mib = 32 * 1024 * 1024  # Cloud TPU v5p (32 MiB per TensorCore)

    def test_low_tp_nv64_clamped_to_two(self):
        """At TP=1 for Qwen 3.5 397B (nv=64), cand=4 overflows 64 MiB VMEM (66.14 MB).

        Clamping must reduce decode_tile_size to 2.
        """
        tile_size = compute_decode_tile_size(
            batch_size=4,
            n_v=64,
            d_k=128,
            d_v=128,
            dim=12288,
            aligned_num_v_heads=128,
            kernel_size=4,
            num_spec_tokens=0,
            requested_tile_size=4,
            vmem_capacity_bytes=self.vmem_64mib,
        )
        self.assertEqual(tile_size, 2)

    def test_tp8_nv8_retains_four(self):
        """At TP=8 for Qwen 3.5 397B (nv=8), tile footprint easily fits in VMEM.

        Requested decode_tile_size=4 should remain 4.
        """
        tile_size = compute_decode_tile_size(
            batch_size=4,
            n_v=8,
            d_k=128,
            d_v=128,
            dim=1536,
            aligned_num_v_heads=128,
            kernel_size=4,
            num_spec_tokens=0,
            requested_tile_size=4,
            vmem_capacity_bytes=self.vmem_64mib,
        )
        self.assertEqual(tile_size, 4)

    def test_batch_size_upper_bounds(self):
        """Batch size strictly bounds decode_tile_size even if VMEM allows more."""
        tile_size_b1 = compute_decode_tile_size(
            batch_size=1,
            n_v=8,
            d_k=128,
            d_v=128,
            dim=1536,
            aligned_num_v_heads=128,
            kernel_size=4,
            num_spec_tokens=0,
            requested_tile_size=4,
            vmem_capacity_bytes=self.vmem_64mib,
        )
        self.assertEqual(tile_size_b1, 1)

        tile_size_b2 = compute_decode_tile_size(
            batch_size=2,
            n_v=8,
            d_k=128,
            d_v=128,
            dim=1536,
            aligned_num_v_heads=128,
            kernel_size=4,
            num_spec_tokens=0,
            requested_tile_size=4,
            vmem_capacity_bytes=self.vmem_64mib,
        )
        self.assertEqual(tile_size_b2, 2)

    def test_speculative_window_clamping(self):
        """Speculative verify window (num_spec_tokens=4, window=5) scales tile footprint."""
        # At TP=1 (nv=64) with 4 speculative tokens, window=5 forces tile size to 1
        tile_size_spec = compute_decode_tile_size(
            batch_size=4,
            n_v=64,
            d_k=128,
            d_v=128,
            dim=12288,
            aligned_num_v_heads=128,
            kernel_size=4,
            num_spec_tokens=4,
            requested_tile_size=4,
            vmem_capacity_bytes=self.vmem_64mib,
        )
        self.assertEqual(tile_size_spec, 1)

    def test_low_vmem_capacity_floor(self):
        """Clamped tile size never drops below minimum valid tile size of 1."""
        tile_size_floor = compute_decode_tile_size(
            batch_size=4,
            n_v=64,
            d_k=128,
            d_v=128,
            dim=12288,
            aligned_num_v_heads=128,
            kernel_size=4,
            num_spec_tokens=8,
            requested_tile_size=4,
            vmem_capacity_bytes=4 * 1024 * 1024,  # Very small VMEM
        )
        self.assertEqual(tile_size_floor, 1)


if __name__ == "__main__":
    unittest.main()
