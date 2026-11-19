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


if __name__ == "__main__":
    absltest.main()
