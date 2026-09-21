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
import functools

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.deepseek_v4.core_attention.csa_gather import \
    csa_gather


def gather_nope_ref_impl(cache, indices):
    page_size = cache.shape[1]
    return cache[indices // page_size, indices % page_size, :, :]


def gather_rope_ref_impl(cache, indices, rope_period=1024):
    gathered = cache.reshape(-1, 128)[indices]
    hi = gathered[:, 0:64].astype(jnp.uint16)
    lo = gathered[:, 64:128].astype(jnp.uint16)
    rope = ((hi << 8) | lo).view(jnp.bfloat16)
    rope = rope.reshape(-1, 2, rope_period // 2, 64)
    return jnp.concatenate([rope[:, 0], rope[:, 1]], axis=-1).reshape(-1, 128)


@functools.partial(jax.jit, static_argnames=("rope_period", ))
def gather_ref_impl(nope_cache, rope_cache, indices, rope_period=1024):
    nope_out = gather_nope_ref_impl(nope_cache, indices)
    rope_out = gather_rope_ref_impl(rope_cache,
                                    indices,
                                    rope_period=rope_period)
    return nope_out, rope_out


@jax.jit
def create_nope_cache():
    key = jax.random.key(0)
    key, perm_key, cache_key = jax.random.split(key, 3)
    num_pages = 1000
    page_size = 256
    cache = jax.random.randint(cache_key,
                               shape=(num_pages, page_size, 4, 128),
                               minval=0,
                               maxval=256)
    cache = cache.astype(jnp.uint8)
    return cache, perm_key


@jax.jit
def create_rope_cache():
    key = jax.random.key(41)
    key, perm_key, cache_key = jax.random.split(key, 3)
    num_pages = 1000
    page_size = 256
    cache = jax.random.randint(cache_key,
                               shape=(num_pages, page_size // 4, 4, 128),
                               minval=0,
                               maxval=256)
    cache = cache.astype(jnp.uint8)
    return cache, perm_key


class GatherTest(parameterized.TestCase):

    @parameterized.parameters(
        (256, 256),
        (4096, 256),
        (4096, 512),
        (128 * 1024, 1024),
        (128 * 1024, 2048),
    )
    def test_correctness(self, n, rope_period):
        nope_cache, perm_key = create_nope_cache()
        rope_cache, _ = create_rope_cache()

        max_index = nope_cache.shape[0] * nope_cache.shape[1]
        indices = jax.random.randint(perm_key, (n, ),
                                     0,
                                     max_index,
                                     dtype=jnp.int32)

        nope_ref, rope_ref = gather_ref_impl(nope_cache,
                                             rope_cache,
                                             indices,
                                             rope_period=rope_period)
        nope_sc, rope_sc = csa_gather(nope_cache,
                                      rope_cache,
                                      indices,
                                      rope_period=rope_period)

        np.testing.assert_array_equal(nope_ref.view(jnp.uint8),
                                      nope_sc.view(jnp.uint8))
        np.testing.assert_array_equal(rope_ref.view(jnp.uint16),
                                      rope_sc.view(jnp.uint16))

    @parameterized.parameters(
        (4096, 1024),
        (128 * 1024, 16 * 1024),
        (128 * 1024, 1024),
    )
    def test_num_valid_indices(self, n, num_valid):
        nope_cache, perm_key = create_nope_cache()
        rope_cache, _ = create_rope_cache()

        max_index = nope_cache.shape[0] * nope_cache.shape[1]
        indices = jax.random.randint(perm_key, (n, ),
                                     0,
                                     max_index,
                                     dtype=jnp.int32)

        nope_ref, rope_ref = gather_ref_impl(nope_cache, rope_cache,
                                             indices[:num_valid])
        nope_sc, rope_sc = csa_gather(nope_cache,
                                      rope_cache,
                                      indices,
                                      num_valid_indices=num_valid)

        np.testing.assert_array_equal(nope_ref.view(jnp.uint8),
                                      nope_sc[:num_valid].view(jnp.uint8))
        np.testing.assert_array_equal(
            rope_ref.view(jnp.uint16),
            rope_sc[:num_valid // 2].view(jnp.uint16))


if __name__ == "__main__":
    absltest.main()
