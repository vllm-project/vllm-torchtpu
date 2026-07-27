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
"""Byte-exact tests for the SparseCore csa_gather kernel.

The gather is pure transport (plus the byte de-interleave documented by
`process_nope_reference` in csa_gather.py), so the host-side oracle is
direct indexing into the source arrays and every comparison is exact:
gathered nope bytes must equal the token's 512 cache bytes, and gathered
rope bf16 must be bit-identical to the values encoded in the high/low
byte planes.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from vllm_torchtpu.kernels.mla.sparse import csa_gather

LKV_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 32


class CsaGatherTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        if jax.devices()[0].platform != "tpu":
            self.skipTest("csa_gather requires a TPU (SparseCore).")
        self.rng = np.random.default_rng(42)

    def _make_caches(self, pages):
        """Random caches plus the plain arrays the oracle indexes into."""
        num_tokens = pages * PAGE_SIZE
        nope_bytes = self.rng.integers(0, 256, (num_tokens, LKV_DIM), np.uint8)
        rope_vals = jnp.asarray(self.rng.standard_normal(
            (num_tokens, ROPE_DIM)).astype(np.float32),
                                dtype=jnp.bfloat16)
        rope_u16 = np.asarray(
            jax.lax.bitcast_convert_type(rope_vals, jnp.uint16))
        rope_planes = np.concatenate([(rope_u16 >> 8).astype(np.uint8),
                                      (rope_u16 & 0xFF).astype(np.uint8)],
                                     -1)  # [T, 128]

        nope_cache = jnp.asarray(nope_bytes).reshape(
            pages, PAGE_SIZE, csa_gather.TILE_SUBROWS,
            csa_gather.TILE_LANE_BYTES)
        rope_cache = jnp.asarray(rope_planes).reshape(
            pages, PAGE_SIZE // csa_gather.TILE_SUBROWS,
            csa_gather.TILE_SUBROWS, csa_gather.TILE_LANE_BYTES)
        return nope_cache, rope_cache, nope_bytes, rope_u16

    def _check(self, nope_cache, rope_cache, nope_bytes, rope_u16, indices):
        nope_out, rope_out = csa_gather.csa_gather(nope_cache, rope_cache,
                                                   jnp.asarray(indices))
        np.testing.assert_array_equal(np.asarray(nope_out),
                                      nope_bytes[indices])
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(rope_out, jnp.uint16)),
            rope_u16[indices])

    @parameterized.named_parameters(
        # Odd count exercises the internal padding to the SC block size;
        # sampling with replacement exercises duplicate indices.
        dict(testcase_name="random_with_duplicates", pages=8, n=1000),
        dict(testcase_name="small_odd", pages=2, n=17),
        dict(testcase_name="large", pages=16, n=8192),
    )
    def test_gather_matches_direct_indexing(self, pages, n):
        nope_cache, rope_cache, nope_bytes, rope_u16 = self._make_caches(pages)
        indices = self.rng.integers(0, pages * PAGE_SIZE, n).astype(np.int32)
        self._check(nope_cache, rope_cache, nope_bytes, rope_u16, indices)

    def test_all_byte_lanes(self):
        """Every position of every page: covers all four rope byte lanes and
        every tile row, in permuted (non-contiguous) order."""
        pages = 4
        nope_cache, rope_cache, nope_bytes, rope_u16 = self._make_caches(pages)
        indices = self.rng.permutation(pages * PAGE_SIZE).astype(np.int32)
        self._check(nope_cache, rope_cache, nope_bytes, rope_u16, indices)

    def test_single_repeated_index(self):
        """Maximal duplication: the gather-hotspot access pattern."""
        pages = 4
        nope_cache, rope_cache, nope_bytes, rope_u16 = self._make_caches(pages)
        indices = np.full(512, 77, np.int32)
        self._check(nope_cache, rope_cache, nope_bytes, rope_u16, indices)
