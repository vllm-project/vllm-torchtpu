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
"""The sort program's size estimate against the compiler's counts
(measured offline, compiled without the device: 8192 tokens
of top-10 over 512 experts is an 86,016-row table, 2688 rows a subcore;
4096 tokens 45,056 rows, 1408 a subcore; 2048 tokens 24,576 rows, 768 a
subcore). No device needed."""

import jax.numpy as jnp
import pytest
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.fused_moe.v3 import row_gather_kernel as gather

pytestmark = pytest.mark.cpu_test


class SortProgramTest(parameterized.TestCase):
    @parameterized.parameters(
        # (rows a subcore, lane blocks, rows a block, compiles, the
        # compiler's count): every case compiled, whole and aborted.
        (2688, 8, 16, True, 3872),
        (2688, 16, 16, True, 3872),
        (2688, 32, 16, True, 3872),
        (2688, 64, 16, True, None),
        (1408, 32, 16, True, 2032),
        (768, 20, 16, True, 4658),
        (1408, 20, 32, True, 5146),
        (2688, 12, 16, False, 18797),
        (2688, 20, 16, False, 16238),
        (2688, 24, 16, False, 16239),
        (2688, 28, 16, False, 18797),
        (2688, 40, 16, False, 16930),
        (2688, 48, 16, False, 16930),
        (2688, 56, 16, False, 16930),
        (1408, 20, 16, False, 8578),
    )
    def test_against_the_compiler(
        self, rows, lane_blocks, block_rows, compiles, counted
    ):
        dtype = jnp.float8_e4m3fn
        self.assertEqual(
            gather.sort_program_fits(rows, lane_blocks, dtype, block_rows), compiles
        )
        if counted is not None:
            # The estimate stands on the largest costs seen, so it is never
            # under the count and within a sixth over it.
            estimate = gather.sort_program_bundles(rows, lane_blocks, dtype, block_rows)
            self.assertGreaterEqual(estimate, counted)
            self.assertLess(estimate, counted * 7 / 6)

    def test_one_copy_per_block(self):
        # A row of a power of two words moves as one copy per block: 1024,
        # 2048, 4096 and 8192 fp8 values; bf16 rows of half the blocks.
        for lane_blocks in (8, 16, 32, 64):
            self.assertTrue(gather.row_words_whole_copy(lane_blocks))
            self.assertTrue(gather.row_words_whole_copy(lane_blocks // 2, jnp.bfloat16))
        for lane_blocks in (12, 20, 24, 28, 40, 48, 56):
            self.assertFalse(gather.row_words_whole_copy(lane_blocks))
        self.assertFalse(gather.row_words_whole_copy(20, jnp.bfloat16))

    def test_the_sizes(self):
        # 23 bundles a block against 92 a block and 1.25 a row, on 8: the
        # 397B table at 8192 tokens (168 blocks of 16 a subcore) is 3872
        # bundles as whole copies and 18,824 at a width that is not; 84
        # blocks of 32 rows are 1940 and 11,096.
        self.assertEqual(gather.sort_program_bundles(2688, 32), 3872)
        self.assertEqual(gather.sort_program_bundles(2688, 20), 18824)
        self.assertEqual(gather.sort_program_bundles(2688, 32, block_rows=32), 1940)
        self.assertEqual(gather.sort_program_bundles(2688, 20, block_rows=32), 11096)


if __name__ == "__main__":
    absltest.main()
