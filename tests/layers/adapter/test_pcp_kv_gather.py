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
"""CPU tests for the PCP decode-window row helpers."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter.pcp_kv_gather import (
    merge_decode_rows,
    take_decode_rows,
)


def _md(*, rep_row_start, window, num_rep, local_padded):
    # Same window fields as `HeadTailPcpShardingPlan.prepare_real`.
    rep_row_idx = np.minimum(np.arange(window) + rep_row_start, local_padded - 1)
    return SimpleNamespace(
        pcp_metadata=SimpleNamespace(
            rep_row_idx=torch.tensor(rep_row_idx, dtype=torch.int32),
            rep_request_distribution=torch.tensor([num_rep] * 3, dtype=torch.int32),
        )
    )


def _rows(num_rows, *trailing):
    return torch.arange(num_rows * int(np.prod(trailing or (1,)))).reshape(
        num_rows, *trailing
    )


def test_take_decode_rows_clamps_the_window_to_the_padded_rows():
    x = _rows(8, 3)
    md = _md(rep_row_start=5, window=4, num_rep=2, local_padded=8)
    torch.testing.assert_close(take_decode_rows(x, md), x[[5, 6, 7, 7]])


def test_merge_decode_rows_without_decodes_keeps_the_prefill_rows():
    prefill = _rows(8, 3)
    md = _md(rep_row_start=0, window=4, num_rep=0, local_padded=8)
    decode = torch.full((4, 3), -1)
    torch.testing.assert_close(merge_decode_rows(prefill, decode, md), prefill)


@pytest.mark.parametrize("trailing", [(3,), (2, 3)])
def test_merge_decode_rows_writes_only_the_valid_window_rows(trailing):
    # Decodes at rows 3 and 4; the window's padding slots 2 and 3 map past
    # them and must not overwrite rows 5 and 6.
    prefill = _rows(8, *trailing)
    md = _md(rep_row_start=3, window=4, num_rep=2, local_padded=8)
    decode = -1 - _rows(4, *trailing)
    out = merge_decode_rows(prefill, decode, md)
    expected = prefill.clone()
    expected[3:5] = decode[:2]
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("rep_row_start, num_rep", [(0, 4), (2, 3), (6, 1)])
def test_merge_of_taken_rows_round_trips(rep_row_start, num_rep):
    x = _rows(8, 3)
    md = _md(rep_row_start=rep_row_start, window=4, num_rep=num_rep, local_padded=8)
    torch.testing.assert_close(merge_decode_rows(x, take_decode_rows(x, md), md), x)
