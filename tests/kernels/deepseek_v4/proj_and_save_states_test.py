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
"""One `proj_and_save_states` call writes what separate calls write."""

import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    proj_and_save_state as proj_kernel,
)

pytestmark = pytest.mark.tpu_test

HIDDEN = 512
NUM_PAGES = 300


def _projection(rng, positions, requests, width, cache_shape, dtype, rows_per_token):
    """A projection whose states fill whole pages per request, the last
    three tokens padding (slot -1)."""
    page_rows = cache_shape[0]
    tokens_per_page = page_rows // rows_per_token
    pages = 3 + requests * 140 + positions // tokens_per_page
    slots = pages * page_rows + (positions % tokens_per_page) * rows_per_token
    slots[-3:] = -1
    return proj_kernel.StateProjection(
        wkv_wgate=jnp.asarray(rng.standard_normal((HIDDEN, 2 * width)), jnp.bfloat16),
        ape=jnp.asarray(rng.standard_normal((4, width)), jnp.float32),
        slot_mapping=jnp.asarray(slots, jnp.int32),
        cache=jnp.zeros((NUM_PAGES, *cache_shape), dtype),
    )


class ProjAndSaveStatesTest(jtu.JaxTestCase):
    # 40 tokens: one tile; 600: one group of 3 tiles; 2100: 3 groups of 3.
    @parameterized.parameters(40, 600, 2100)
    def test_paired_call_matches_separate_calls(self, num_tokens):
        rng = np.random.default_rng(0)
        half = num_tokens // 2
        token = np.arange(num_tokens)
        requests = (token >= half).astype(np.int32)
        positions = np.where(requests == 0, token, token - half + 5).astype(np.int32)
        hidden = jnp.asarray(rng.standard_normal((num_tokens, HIDDEN)), jnp.bfloat16)
        # The lightning indexer's and CSA's state layouts.
        indexer = _projection(
            rng, positions, requests, 256, (64, 4, 256), jnp.uint8, rows_per_token=2
        )
        csa = _projection(
            rng, positions, requests, 1024, (256, 128), jnp.int32, rows_per_token=16
        )
        positions = jnp.asarray(positions)

        paired = proj_kernel.proj_and_save_states(
            hidden, positions, (indexer, csa), compress_ratios=(4, 4)
        )
        for proj, got in zip((indexer, csa), paired, strict=True):
            want = proj_kernel.proj_and_save_state(
                hidden,
                proj.wkv_wgate,
                proj.ape,
                positions,
                proj.slot_mapping,
                proj.cache,
                compress_ratio=4,
            )
            self.assertGreater(int(jnp.count_nonzero(want)), 0)
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
