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
"""The one-sort MSA prefill page-cache table equals the three-sort one."""

import jax
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse as lse


def _pages(tiles, cols, live, distinct_pool, seed):
    rng = np.random.default_rng(seed)
    p = rng.integers(0, distinct_pool, (tiles, cols)).astype(np.int32)
    for t in range(tiles):
        p[t, rng.integers(0, live + 1):] = -1
    return p.reshape(-1)


@pytest.mark.parametrize(
    "tiles,cols,live,pool,cache",
    [
        (128, 64, 64, 300, 512),  # fits
        (128, 64, 64, 30000, 512),  # overflows the cache
        (16, 96, 80, 200, 512),  # union wider than the cached head
        (8, 32, 0, 10, 64),  # all padding
        (64, 64, 40, 64, 64),  # exactly cache_size distinct possible
    ],
)
def test_same_table(tiles, cols, live, pool, cache):
    for seed in range(4):
        pages = jax.numpy.asarray(_pages(tiles, cols, live, pool, seed))
        a = lse._page_cache(pages, num_tiles=tiles, cache_size=cache)
        b = lse._page_cache_one_sort(pages, num_tiles=tiles, cache_size=cache)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
