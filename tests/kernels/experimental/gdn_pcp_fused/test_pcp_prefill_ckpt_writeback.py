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
"""CPU tests for the PCP fused kernel's prefill checkpoint request masking and
compact-update scatter."""

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.gdn_pcp_fused import pcp_metadata, pcp_wrapper


def test_prefill_ckpt_requests_mask_rows_without_a_checkpoint_to_sentinels():
    offsets = jnp.asarray([0, 128, 64, 192, 64, 3072], dtype=jnp.int32)
    indices = jnp.asarray([5, 6, -1, 7, 8, 9], dtype=jnp.int32)
    # Query lengths 100, 300, 300, 192, 0, 0: the last two rows are empty.
    query_start_loc = jnp.asarray([0, 100, 400, 700, 892, 892, 892], jnp.int32)

    masked_offsets, masked_indices = pcp_metadata.mask_prefill_ckpt_requests(
        offsets, indices, query_start_loc[1:] - query_start_loc[:-1]
    )

    # Only row 1 has 0 < offset < query_len and a block id.
    np.testing.assert_array_equal(np.asarray(masked_offsets), [0, 128, 0, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(masked_indices), [-1, 6, -1, -1, -1, -1])


def test_prefill_ckpt_scatter_skips_sentinel_rows():
    pool = jnp.arange(6 * 4 * 2, dtype=jnp.float32).reshape(6, 4, 2)
    # One compact update per sequence covers the first two slices of a block.
    updates = jnp.stack([jnp.full((2, 2), 100 + row, pool.dtype) for row in range(3)])
    query_start_loc = jnp.asarray([0, 200, 500, 800], dtype=jnp.int32)
    _, ckpt_dst = pcp_metadata.mask_prefill_ckpt_requests(
        jnp.asarray([0, 128, 192], dtype=jnp.int32),
        jnp.asarray([0, 2, -1], dtype=jnp.int32),
        query_start_loc[1:] - query_start_loc[:-1],
    )

    actual = pcp_wrapper._scatter_state_updates(
        pool, updates, ckpt_dst, jnp.asarray(3, dtype=jnp.int32)
    )

    # Row 0 has no checkpoint, so the null block 0 stays untouched, and row 2
    # names no block. Only row 1 writes the first two slices of block 2.
    expected = pool.at[2, :2].set(101)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
