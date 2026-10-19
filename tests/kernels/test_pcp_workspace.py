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
"""PCP manager indexer workspace layout and page table (`pcp_workspace.py`)."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla import pcp_workspace as ws

NUM_SEGMENTS = 3


def _workspace(dcp, num_ws_pages, page_size, width=3):
    shape = (dcp, num_ws_pages, page_size // 4, 4, width)
    return np.arange(np.prod(shape), dtype=np.int32).reshape(shape)


def _token(kv, table, seg, pos):
    """The row the top-k kernel reads for position `pos` of segment `seg`:
    page size from the cache shape, row width from the table length."""
    page_size = kv.shape[1] * kv.shape[2]
    pages_per_seq = len(table) // NUM_SEGMENTS
    page = table[seg * pages_per_seq + pos // page_size]
    return np.asarray(kv).reshape(kv.shape[0], page_size, -1)[page, pos % page_size]


def _reference_token(w, offset, pos, interleave):
    """Position `pos` of the request whose first virtual page is `offset`:
    owned by DCP rank `(pos // interleave) % dcp` at its local slot."""
    dcp, _, rows, packing, _ = w.shape
    page_size = rows * packing
    cycle = dcp * interleave
    owner = (pos // interleave) % dcp
    local = (pos // cycle) * interleave + pos % interleave
    flat = w.reshape(dcp, -1, page_size, w.shape[-1])
    return flat[owner, offset + local // page_size, local % page_size]


@pytest.mark.parametrize(
    "dcp, interleave, page_size, num_ws_pages",
    [(8, 4, 64, 6), (4, 4, 64, 5), (8, 1, 16, 3), (2, 16, 32, 4)],
)
def test_position_ordered_pages_are_cache_sized(
    dcp, interleave, page_size, num_ws_pages
):
    w = _workspace(dcp, num_ws_pages, page_size)
    kv = np.asarray(ws.position_ordered(jnp.asarray(w), interleave))
    assert kv.shape == (num_ws_pages * dcp, page_size // 4, 4, w.shape[-1])

    offsets = [0, 1, num_ws_pages - 2]
    table = np.asarray(
        ws.workspace_index_page_table(
            jnp.asarray(offsets, jnp.int32), num_ws_pages, dcp
        )
    )
    assert len(table) == NUM_SEGMENTS * num_ws_pages * dcp
    for seg, offset in enumerate(offsets):
        for pos in range((num_ws_pages - offset) * dcp * page_size):
            np.testing.assert_array_equal(
                _token(kv, table, seg, pos),
                _reference_token(w, offset, pos, interleave),
            )


@pytest.mark.parametrize("max_pages_per_seq", [1, 5, 7, 1000])
def test_page_table_row_width_capped_by_max_pages(max_pages_per_seq):
    dcp, interleave, page_size, num_ws_pages = 4, 4, 32, 6
    w = _workspace(dcp, num_ws_pages, page_size)
    kv = np.asarray(ws.position_ordered(jnp.asarray(w), interleave))
    offsets = [0, 2, num_ws_pages - 1]
    table = np.asarray(
        ws.workspace_index_page_table(
            jnp.asarray(offsets, jnp.int32), num_ws_pages, dcp, max_pages_per_seq
        )
    )
    width = min(num_ws_pages * dcp, max_pages_per_seq)
    assert len(table) == NUM_SEGMENTS * width
    # Every position a request of at most `width` pages can hold reads the
    # same token as with a full-workspace row.
    for seg, offset in enumerate(offsets):
        limit = min(width, (num_ws_pages - offset) * dcp) * page_size
        for pos in range(limit):
            np.testing.assert_array_equal(
                _token(kv, table, seg, pos),
                _reference_token(w, offset, pos, interleave),
            )
