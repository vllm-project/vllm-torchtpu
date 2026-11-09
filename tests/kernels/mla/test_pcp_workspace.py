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
"""Position-ordered indexer K workspace (`kernels/mla/pcp_workspace.py`)."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla import pcp_workspace as ws


def _rank_major_workspace(dcp, num_ws_pages, rows, packing, interleave):
    """Gathered `[dcp, Pw, rows, packing, 1]` workspace whose value at every
    slot is that token's position in the workspace (virtual page major)."""
    page_size = rows * packing
    virtual_page = dcp * page_size
    out = np.zeros((dcp, num_ws_pages, page_size), np.int32)
    for d in range(dcp):
        for v in range(num_ws_pages):
            for slot in range(page_size):
                t = (slot // interleave) * dcp * interleave + d * interleave
                t += slot % interleave
                out[d, v, slot] = v * virtual_page + t
    return jnp.asarray(out.reshape(dcp, num_ws_pages, rows, packing, 1))


@pytest.mark.parametrize("dcp,interleave", [(2, 1), (4, 2), (8, 4)])
def test_page_table_reads_every_position_in_order(dcp, interleave):
    num_ws_pages, rows, packing = 3, 2, 4
    page_size = rows * packing
    virtual_page = dcp * page_size
    workspace = ws.position_ordered(
        _rank_major_workspace(dcp, num_ws_pages, rows, packing, interleave),
        interleave,
    )
    # One page per virtual page.
    assert workspace.shape == (num_ws_pages, dcp * rows, packing, 1)
    tokens = np.asarray(workspace).reshape(num_ws_pages, virtual_page)

    ws_page_offset = jnp.asarray([0, 1, 2], jnp.int32)
    table = np.asarray(
        ws.workspace_index_page_table(ws_page_offset, num_ws_pages)
    ).reshape(len(ws_page_offset), num_ws_pages)
    for seg, offset in enumerate(np.asarray(ws_page_offset)):
        for pos in range((num_ws_pages - offset) * virtual_page):
            page = table[seg, pos // virtual_page]
            assert tokens[page, pos % virtual_page] == offset * virtual_page + pos
        # Entries past the workspace clamp to its last page.
        assert table[seg].max() == num_ws_pages - 1


def test_page_table_width_capped_at_max_pages_per_seq():
    ws_page_offset = jnp.asarray([0, 5], jnp.int32)
    table = np.asarray(
        ws.workspace_index_page_table(ws_page_offset, 8, max_pages_per_seq=3)
    ).reshape(2, 3)
    np.testing.assert_array_equal(table, [[0, 1, 2], [5, 6, 7]])
    # A cap past the workspace keeps the workspace width.
    assert ws.workspace_index_page_table(ws_page_offset, 8, 100).shape == (16,)


@pytest.mark.parametrize(
    "virtual_page,expected",
    # Cache page 64 / 256 / 512 / 1024 at DCP 8.
    [(512, 8), (2048, 2), (4096, 1), (8192, 1)],
)
def test_index_pages_per_block(virtual_page, expected):
    assert ws.index_pages_per_block(virtual_page) == expected


def test_topk_rows_past_token_end_are_masked():
    # Two split rows hold global positions; the decode row after them holds
    # a rank-local cache index far past the workspace, the padding row junk.
    page_size, dcp, interleave, chunk_pages = 4, 2, 1, 2
    topk = jnp.asarray(
        [[0, 9, -1], [3, 1, 2], [1 << 20, 7, -1], [-5, 1 << 30, 4]], jnp.int32
    )
    qsl = jnp.asarray([0, 2, 3, 4], jnp.int32)
    ws_page_offset = jnp.asarray([0, 0, 0], jnp.int32)
    coords = ws.topk_workspace_coords(
        topk,
        qsl,
        ws_page_offset,
        jnp.int32(2),
        page_size=page_size,
        dcp_size=dcp,
        interleave=interleave,
    )
    valid = np.asarray(coords[0])
    np.testing.assert_array_equal(valid[:2], np.asarray(topk[:2]) >= 0)
    assert not valid[2:].any()
    idx = np.asarray(
        ws.remap_topk_chunked(
            coords, chunk_pages=chunk_pages, page_size=page_size, dcp_size=dcp
        )
    )
    # Masked rows read one in-bounds dummy key and nothing else.
    np.testing.assert_array_equal(idx[2:], [[0, -1, -1], [0, -1, -1]])
    assert idx[:2].max() < 2 * chunk_pages * dcp * page_size
