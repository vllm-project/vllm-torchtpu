# SPDX-License-Identifier: Apache-2.0
"""Staged repack rows for a PCP producer (main KV split over its PCP ranks).

A PCP producer stores token t of a P-page entry on rank (t // I) % P at row
``qsa_rpa.dcp_owner_row(t)`` of that rank's page of the entry. The decode stages
each rank's pages separately; ``main_rows_np(src_pcp=(P, I))`` must point every
token at the staged row holding it, for any decode DCP slice and prefix start.
"""

import numpy as np
import pytest

from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import RankSlice, main_rows_np


def _owner_row(t, page, world, interleave):
    offset = t % (world * page)
    return (offset // interleave) % world, offset // (interleave * world) * interleave + offset % interleave


@pytest.mark.parametrize("main_dcp,dcp_rank", [(1, 0), (4, 0), (4, 3)])
@pytest.mark.parametrize("start", [0, 6400, 12800])
def test_pcp_source_rows_hold_their_tokens(main_dcp, dcp_rank, start):
    src_block, p, interleave, dst_block = 3200, 2, 128, 512
    num_tokens = 3 * p * src_block - 77
    entries = -(-num_tokens // (p * src_block))
    first = start // (p * src_block)
    # Staging carrier: every staged row records the token it holds (-1 = empty).
    staged_pages = np.arange(1, 1 + p * entries).reshape(p, entries)
    carrier = np.full(((1 + p * entries) * src_block,), -1, np.int64)
    t = np.arange(num_tokens)
    owner, row = _owner_row(t, src_block, p, interleave)
    carrier[staged_pages[owner, t // (p * src_block)] * src_block + row] = t
    dst_pages = np.arange(100, 100 + num_tokens // (main_dcp * dst_block) + 2)
    rank = RankSlice(tp_size=8, tp_rank=dcp_rank, num_kv_heads=2, main_dcp=main_dcp)
    src, dst = main_rows_np(
        num_tokens, staged_pages[:, first:], src_block, dst_pages, dst_block, rank,
        start=start, src_first=first, src_pcp=(p, interleave),
    )
    want = np.arange(start, num_tokens)
    if main_dcp > 1:
        want = want[(want // dst_block) % main_dcp == rank.dcp_rank]
    np.testing.assert_array_equal(carrier[src], want)
    assert len(dst) == len(want)
