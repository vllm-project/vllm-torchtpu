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
"""Native cache DMA correctness across donated, sharded serving steps."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout, KVCacheType, SparseMLAKVCacheSpec,
    update_sparse_mla_kv_cache)
from vllm_torchtpu.kernels.mla.sparse import native_sc_cache_layout as layout


class NativeScCacheUpdateTest(parameterized.TestCase):

    @parameterized.parameters((13, 5, 3), (64, 5, 3), (1024, 513, 257),
                              (64, 0, 0))
    def test_retained_caches_match_byte_oracle(self, bucket, q0, q1):
        if jax.devices()[0].platform != "tpu":
            self.skipTest("Native cache DMA requires TPU SparseCore")
        pages, page_size = 16, 256
        specs = tuple(
            SparseMLAKVCacheSpec.create(kind, KVCacheLayout.SPARSECORE, pages,
                                        page_size, dim, 4)
            for kind, dim in ((KVCacheType.NOPE, 512), (KVCacheType.ROPE, 64)))
        rng = np.random.default_rng(44)
        # Nonzero sentinels expose destruction of untouched RoPE quarters.
        expected_nope = rng.integers(0, 256, (pages, page_size, 512), np.uint8)
        expected_rope = rng.integers(0, 256, (pages, page_size, 128), np.uint8)
        caches = (jnp.asarray(
            layout.pack_nope(expected_nope.reshape(pages, page_size, 4, 128))),
                  jnp.asarray(
                      layout.pack_rope_banded(
                          expected_rope.reshape(pages, page_size // 4, 4,
                                                128))))
        tables = rng.permutation(pages).astype(np.int32).reshape(4, 4)
        # Empty sequences before and between live ones exercise searchsorted.
        cu = np.array([0, 0, q0, q0, q0 + q1], np.int32)
        mesh = Mesh(np.array(jax.local_devices()[:1]), ("x", ))

        @functools.partial(jax.jit, donate_argnums=(0, 1))
        @functools.partial(jax.shard_map,
                           mesh=mesh,
                           in_specs=(P(), ) * 7,
                           out_specs=(P(), P()),
                           check_vma=False)
        def run(nope, rope, kv, pe, lengths, table, starts):
            return update_sparse_mla_kv_cache(nope,
                                              rope,
                                              kv,
                                              pe,
                                              lengths,
                                              table,
                                              starts,
                                              nope_spec=specs[0],
                                              rope_spec=specs[1])

        for step in range(3):
            # Begin at quarter 3 and cross a page edge; successive steps retain
            # and overwrite some prior writes with fresh data.
            starts = (253 + step, 3 + step)
            lengths = np.array([0, starts[0] + q0, 0, starts[1] + q1],
                               np.int32)
            kv = jnp.asarray(rng.normal(size=(bucket, 512)), jnp.float8_e4m3fn)
            pe = jnp.asarray(rng.normal(size=(bucket, 64)), jnp.float8_e4m3fn)
            kv_bytes = np.asarray(jax.lax.bitcast_convert_type(kv, jnp.uint8))
            pe_bytes = np.asarray(jax.lax.bitcast_convert_type(pe, jnp.uint8))
            for seq, count, offset, start in ((1, q0, 0, starts[0]),
                                              (3, q1, q0, starts[1])):
                for i in range(count):
                    pos = start + i
                    page, slot = tables[seq, pos // page_size], pos % page_size
                    expected_nope[page, slot] = kv_bytes[offset + i]
                    expected_rope[page, slot, :64] = pe_bytes[offset + i]
                    expected_rope[page, slot, 64:] = 0
            caches = run(*caches, kv, pe, jnp.asarray(lengths),
                         jnp.asarray(tables.reshape(-1)), jnp.asarray(cu))
            np.testing.assert_array_equal(
                np.asarray(caches[0]),
                layout.pack_nope(
                    expected_nope.reshape(pages, page_size, 4, 128)))
            np.testing.assert_array_equal(
                np.asarray(caches[1]),
                layout.pack_rope_banded(
                    expected_rope.reshape(pages, page_size // 4, 4, 128)))
