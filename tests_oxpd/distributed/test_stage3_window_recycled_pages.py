# SPDX-License-Identifier: Apache-2.0
"""Stage 3 must not transfer a sliding-window page the allocator has taken back.

Both peers count the dead leading pages of a windowed group from
``num_tokens``, the transfer prefix, while the allocator freed pages against
wherever the producer actually stopped. The two agree only while the producer
halts at the handoff token, which ``_stage3_truncates_prompt`` arranges. Let
them drift and the send lists a page pointed at the null block, which Raiden
rejects as a duplicate destination by raising out of ``execute_model``.

The tables below are verbatim ``group_block_ids`` from a live DP4/DP4
DeepSeek-V4-Flash pair.
"""

from __future__ import annotations

import pytest

from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPURaidenConnectorScheduler,
    _stage3_recycled_transfer_pages,
    _Stage3TransferGroup,
)

# DeepSeek-V4-Flash compressor-state geometry: a 16-token page under an
# 8-token window, the tightest ratio the model has and so the first to drift.
_CSA_PAGE = 16
_CSA_WINDOW = 8


@pytest.mark.parametrize(
    "ids,num_tokens,page_tokens,window_tokens,expected",
    [
        # The normal P/D case, computed=114 against num_tokens=113: the
        # allocator freed six pages and the connector skips six.
        ([0] * 6 + [286, 58], 113, _CSA_PAGE, _CSA_WINDOW, 0),
        # computed=375 against num_tokens=374, so the allocator freed 23 pages
        # and the connector skips 22. Without this check the drift is silent:
        # one dead page in an 8-token window survives an accuracy sweep while
        # still corrupting the decode replica's state.
        ([0] * 23 + [13], 374, _CSA_PAGE, _CSA_WINDOW, 1),
        # The engine-killing case: a 114-token prompt run out to 161 tokens,
        # which is what a request sent straight at the producer does.
        # Normalization keeps [0] * 8 and skipping six leaves two null blocks,
        # the duplicate destination Raiden refuses.
        ([0] * 8, 113, _CSA_PAGE, _CSA_WINDOW, 2),
        # Truncating that 375-token prompt to 374 puts both formulas on the
        # same input.
        ([0] * 22 + [13, 41], 374, _CSA_PAGE, _CSA_WINDOW, 0),
        # dsv4.state.hca.g5, computed=287: a wider window drifts the same way,
        # just less often.
        ([0, 0, 0, 0, 0, 7, 5, 4, 2], 286, 32, 128, 1),
        # dsv4.state.idx.g3, computed=327.
        ([0] * 10 + [280], 326, 32, 8, 1),
        # dsv4.swa.g1, computed=2175, the least exposed geometry.
        ([0] * 16 + [270], 2174, 128, 128, 1),
        # A full-attention group has no window, so nothing is ever skipped.
        ([84, 85, 242, 279, 109], 4371, 1024, None, 0),
    ],
)
def test_captured_group_tables(ids, num_tokens, page_tokens, window_tokens, expected):
    assert (
        _stage3_recycled_transfer_pages(
            ids,
            num_tokens=num_tokens,
            page_tokens=page_tokens,
            window_tokens=window_tokens,
        )
        == expected
    )


@pytest.mark.parametrize(
    "mamba_indices,windows,expected",
    [
        # Qwen3.5 and GLM: one full-attention group, so P may run to the end
        # of the prompt.
        ([], [None], False),
        # A windowed group recycles pages against how far P ran.
        ([], [None, 128, 8], True),
        # The original Mamba reason, unchanged.
        ([2], [None], True),
        # Both at once still truncates exactly once.
        ([2], [None, 8], True),
    ],
)
def test_which_models_truncate_the_prompt(mamba_indices, windows, expected):
    scheduler = TPURaidenConnectorScheduler.__new__(TPURaidenConnectorScheduler)
    scheduler._stage3_mamba_group_indices = list(mamba_indices)
    scheduler._stage3_transfer_groups = tuple(
        _Stage3TransferGroup(
            cache_group_index=index, page_tokens=16, window_tokens=window
        )
        for index, window in enumerate(windows)
    )
    assert scheduler._stage3_truncates_prompt() is expected
