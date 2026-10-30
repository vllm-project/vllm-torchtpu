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
"""Host-only tests for the native SparseCore cache byte contract."""

import numpy as np
from absl.testing import absltest

from vllm_torchtpu.kernels.mla.sparse import native_sc_cache_layout


class NativeScCacheLayoutTest(absltest.TestCase):
    def test_nope_word_bytes_and_inverse_are_exact(self):
        nope = (
            np.arange(2 * 8 * 4 * 128, dtype=np.int32)
            .astype(np.uint8)
            .reshape(2, 8, 4, 128)
        )

        packed = native_sc_cache_layout.pack_nope(nope)

        self.assertEqual(packed.shape, (2, 8, 128))
        words_as_bytes = packed.view(np.uint8).reshape(2, 8, 128, 4)
        np.testing.assert_array_equal(words_as_bytes, np.swapaxes(nope, -2, -1))
        np.testing.assert_array_equal(native_sc_cache_layout.unpack_nope(packed), nope)

    def test_rope_banded_word_bytes_and_inverse_are_exact(self):
        rope = (
            np.arange(3 * 2 * 4 * 128, dtype=np.int32)
            .astype(np.uint8)
            .reshape(3, 2, 4, 128)
        )

        packed = native_sc_cache_layout.pack_rope_banded(rope)

        self.assertEqual(packed.shape, (3, 2, 128))
        words_as_bytes = (
            packed.reshape(3, 2, 4, 32).view(np.uint8).reshape(3, 2, 4, 32, 4)
        )
        expected = rope.reshape(3, 2, 4, 4, 32).swapaxes(-2, -1)
        np.testing.assert_array_equal(words_as_bytes, expected)
        np.testing.assert_array_equal(
            native_sc_cache_layout.unpack_rope_banded(packed), rope
        )

    def test_native_allocations_have_no_per_token_padding(self):
        nope = np.zeros((3, 256, 4, 128), np.uint8)
        rope = np.zeros((3, 64, 4, 128), np.uint8)

        packed_nope = native_sc_cache_layout.pack_nope(nope)
        packed_rope = native_sc_cache_layout.pack_rope_banded(rope)

        self.assertEqual(
            native_sc_cache_layout.source_bytes_per_token(packed_nope, packed_rope),
            (512, 128),
        )
        self.assertEqual(
            native_sc_cache_layout.flatten_nope_rows(packed_nope).shape, (768, 128)
        )
        self.assertEqual(
            native_sc_cache_layout.flatten_rope_rows(packed_rope).shape, (768, 32)
        )

    def test_selected_rope_oracle_groups_four_tokens_for_tensorcore(self):
        rope = np.arange(8 * 128, dtype=np.int32).astype(np.uint8).reshape(8, 128)

        packed = native_sc_cache_layout.pack_selected_rope_for_tc(rope)

        self.assertEqual(packed.shape, (2, 128))
        expected = rope.reshape(2, 4, 128).swapaxes(1, 2)
        np.testing.assert_array_equal(
            packed.view(np.uint8).reshape(2, 128, 4), expected
        )

    def test_pack_rope_packed_supports_aligned_widths_and_rejects_invalid_shapes(
        self,
    ):
        rng = np.random.default_rng(19)
        nope = np.zeros((3, 256, 4, 128), np.uint8)
        packed_nope = native_sc_cache_layout.pack_nope(nope)

        rope_64 = rng.integers(0, 256, (3, 256, 64), np.uint8)
        packed_64 = native_sc_cache_layout.pack_rope_packed(rope_64)
        self.assertEqual(packed_64.shape, (3, 32, 128))
        self.assertEqual(
            native_sc_cache_layout.source_bytes_per_token(packed_nope, packed_64),
            (512, 64),
        )
        self.assertEqual(
            native_sc_cache_layout.flatten_rope_rows(packed_64, rope_words=16).shape,
            (768, 16),
        )

        rope_128 = rng.integers(0, 256, (3, 256, 128), np.uint8)
        packed_128 = native_sc_cache_layout.pack_rope_packed(rope_128)
        self.assertEqual(packed_128.shape, (3, 64, 128))
        np.testing.assert_array_equal(
            packed_128,
            native_sc_cache_layout.pack_rope_banded(rope_128.reshape(3, 64, 4, 128)),
        )

        for bad_shape in ((3, 256, 0), (3, 256, 30), (3, 256, 96), (3, 5, 64)):
            with self.subTest(bad_shape=bad_shape), self.assertRaises(ValueError):
                native_sc_cache_layout.pack_rope_packed(np.zeros(bad_shape, np.uint8))

        with self.assertRaises(ValueError):
            native_sc_cache_layout.source_bytes_per_token(
                packed_nope, np.zeros((2, 32, 128), np.uint32)
            )


if __name__ == "__main__":
    absltest.main()
