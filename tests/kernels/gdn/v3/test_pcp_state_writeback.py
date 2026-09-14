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

from vllm_torchtpu.kernels.gdn.v3 import config, pcp_wrapper


def _region(*, kb0: int, nblocks: int, row0: int,
            nrows: int) -> config.StateRegion:
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

    nhd_shape = pcp_wrapper._compact_state_updates_shape(nhd_pool,
                                                         3,
                                                         region,
                                                         whole_block_dma=False)
    hnd_shape = pcp_wrapper._compact_state_updates_shape(hnd_pool,
                                                         3,
                                                         region,
                                                         whole_block_dma=True)

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
    pool = jnp.arange(4 * 2 * 3 * 4 * 8,
                      dtype=jnp.float32).reshape(4, 2, 3, 4, 8)
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
        conv: config.StateRegion, recurrent: config.StateRegion, message: str):
    plan = config.StateSourcePlan(
        stride=3,
        conv=conv,
        recurrent=recurrent,
        whole_block_dma=True,
    )

    with pytest.raises(NotImplementedError, match=message):
        pcp_wrapper._validate_compact_state_writeback_plan(plan)
