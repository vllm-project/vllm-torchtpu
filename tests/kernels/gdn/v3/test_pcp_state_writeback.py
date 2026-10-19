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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gdn.v3 import config, metadata, pcp_wrapper


def _region(*, kb0: int, nblocks: int, row0: int, nrows: int) -> config.StateRegion:
    return config.StateRegion(
        kb0=kb0,
        nblocks=nblocks,
        row0=row0,
        nrows=nrows,
        view_dtype=jnp.dtype(jnp.float32),
        lane_split=1,
        rows_used=nrows,
    )


def test_compact_state_update_shape_matches_pool_layout():
    nhd_pool = jax.ShapeDtypeStruct((8, 256, 1, 4, 128), jnp.float8_e4m3fn)
    hnd_pool = jax.ShapeDtypeStruct((8, 4, 32, 4, 256), jnp.float8_e4m3fn)
    region = _region(kb0=0, nblocks=2, row0=0, nrows=16)

    nhd_shape = pcp_wrapper._compact_state_updates_shape(
        nhd_pool, 3, region, whole_block_dma=False
    )
    hnd_shape = pcp_wrapper._compact_state_updates_shape(
        hnd_pool, 3, region, whole_block_dma=True
    )

    assert nhd_shape.shape == (3, 2, 16, 1, 4, 128)
    assert hnd_shape.shape == (3, 2, 4, 32, 4, 256)


def test_nhd_compact_state_scatter_updates_only_selected_rows():
    pool = jnp.arange(4 * 8 * 2, dtype=jnp.float32).reshape(4, 8, 2)
    updates = jnp.full((1, 1, 2, 2), 99, dtype=pool.dtype)
    region = _region(kb0=0, nblocks=1, row0=3, nrows=2)

    actual = pcp_wrapper._scatter_compact_state_updates(
        pool,
        updates,
        jnp.asarray([1], dtype=jnp.int32),
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        state_stride=2,
        region=region,
        whole_block_dma=False,
    )

    expected = pool.at[2, 3:5].set(99)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_hnd_compact_state_scatter_replaces_complete_page():
    pool = jnp.arange(4 * 2 * 3 * 4 * 8, dtype=jnp.float32).reshape(4, 2, 3, 4, 8)
    updates = jnp.full((1, 1, 2, 3, 4, 8), 77, dtype=pool.dtype)
    region = _region(kb0=1, nblocks=1, row0=0, nrows=4)

    actual = pcp_wrapper._scatter_compact_state_updates(
        pool,
        updates,
        jnp.asarray([1], dtype=jnp.int32),
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        state_stride=2,
        region=region,
        whole_block_dma=True,
    )

    expected = pool.at[3].set(77)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_hnd_compact_state_writeback_accepts_disjoint_aligned_pages():
    plan = config.StateSourcePlan(
        stride=3,
        conv=_region(kb0=2, nblocks=1, row0=0, nrows=16),
        recurrent=_region(kb0=0, nblocks=2, row0=0, nrows=256),
        whole_block_dma=True,
    )

    pcp_wrapper._validate_compact_state_writeback_plan(plan)


@pytest.mark.parametrize(
    ("conv", "recurrent", "message"),
    [
        pytest.param(
            _region(kb0=0, nblocks=1, row0=128, nrows=16),
            _region(kb0=1, nblocks=1, row0=0, nrows=256),
            "page-aligned",
            id="unaligned-conv-region",
        ),
        pytest.param(
            _region(kb0=1, nblocks=1, row0=0, nrows=16),
            _region(kb0=0, nblocks=2, row0=0, nrows=256),
            "non-overlapping",
            id="shared-conv-recurrent-page",
        ),
    ],
)
def test_hnd_compact_state_writeback_rejects_unsafe_page_plans(
    conv: config.StateRegion, recurrent: config.StateRegion, message: str
):
    plan = config.StateSourcePlan(
        stride=3,
        conv=conv,
        recurrent=recurrent,
        whole_block_dma=True,
    )

    with pytest.raises(NotImplementedError, match=message):
        pcp_wrapper._validate_compact_state_writeback_plan(plan)


def test_prefill_ckpt_requests_mask_rows_without_a_checkpoint_to_sentinels():
    offsets = jnp.asarray([0, 128, 64, 192, 64, 3072], dtype=jnp.int32)
    indices = jnp.asarray([5, 6, -1, 7, 8, 9], dtype=jnp.int32)
    # Query lengths 100, 300, 300, 192, 0, 0: the last two rows are empty.
    query_start_loc = jnp.asarray([0, 100, 400, 700, 892, 892, 892], jnp.int32)

    masked_offsets, masked_indices = metadata.mask_prefill_ckpt_requests(
        offsets, indices, query_start_loc[1:] - query_start_loc[:-1]
    )

    # Only row 1 has 0 < offset < query_len and a block id.
    np.testing.assert_array_equal(np.asarray(masked_offsets), [0, 128, 0, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(masked_indices), [-1, 6, -1, -1, -1, -1])


@pytest.mark.parametrize("whole_block_dma", [False, True])
def test_prefill_ckpt_scatter_skips_sentinel_rows(whole_block_dma):
    pool = jnp.arange(6 * 4 * 2, dtype=jnp.float32).reshape(6, 4, 2)
    region = _region(kb0=1, nblocks=1, row0=0 if whole_block_dma else 2, nrows=2)
    if whole_block_dma:
        updates = jnp.stack(
            [jnp.full((1, 4, 2), 100 + row, pool.dtype) for row in range(3)]
        )
    else:
        updates = jnp.stack(
            [jnp.full((1, 2, 2), 100 + row, pool.dtype) for row in range(3)]
        )
    query_start_loc = jnp.asarray([0, 200, 500, 800], dtype=jnp.int32)
    _, indices = metadata.mask_prefill_ckpt_requests(
        jnp.asarray([0, 128, 192], dtype=jnp.int32),
        jnp.asarray([0, 2, -1], dtype=jnp.int32),
        query_start_loc[1:] - query_start_loc[:-1],
    )

    actual = pcp_wrapper._scatter_compact_state_updates(
        pool,
        updates,
        indices,
        query_start_loc,
        jnp.asarray(3, dtype=jnp.int32),
        state_stride=2,
        region=region,
        whole_block_dma=whole_block_dma,
    )

    # Row 0 has no checkpoint, so the null block 0 stays untouched, and row 2
    # names no block. Only row 1 writes block 2, at source block 2 * 2 + 1.
    rows = slice(None) if whole_block_dma else slice(2, 4)
    expected = pool.at[5, rows].set(101)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
