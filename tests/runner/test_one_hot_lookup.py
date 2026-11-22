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
"""`small_lookup` (traced: a one-hot sum; eager or large: a gather) equals
indexing, out-of-range indices giving 0, and the async-spec glue built on it."""

import pytest
import torch

from vllm_torchtpu.runner import tpu_runner_async_output as ao
from vllm_torchtpu.utils import small_lookup


@pytest.mark.parametrize("table_len, index_len", [(8, 96), (512, 8192), (64, 7)])
@pytest.mark.parametrize("traced", [False, True])
def test_lookup_matches_indexing(table_len, index_len, traced):
    gen = torch.Generator().manual_seed(table_len + index_len)
    table = torch.randint(-5, 200_000, (table_len,), generator=gen, dtype=torch.int32)
    index = torch.randint(-1, table_len + 1, (index_len,), generator=gen)
    index = index.to(torch.int32)
    fn = torch.compile(small_lookup, backend="eager") if traced else small_lookup
    got = fn(table, index)
    in_range = (index >= 0) & (index < table_len)
    want = torch.where(in_range, table[index.clamp(0, table_len - 1)], 0)
    assert got.dtype == table.dtype and torch.equal(got, want)


def test_subtract_leaves_unmapped_slots():
    """-1 indices subtract nothing (the one-hot never matches them)."""
    num_rejected = torch.tensor([3, 1, 0, 2], dtype=torch.int32)
    seq_lens = torch.full((6,), 100, dtype=torch.int32)
    positions = torch.arange(10, 20, dtype=torch.int32)
    seq_idx = torch.tensor([0, -1, 3, 1, -1, 2], dtype=torch.int32)
    pos_idx = torch.tensor([0, 0, -1, 3, 3, 3, 1, -1, 2, 2], dtype=torch.int32)
    new_seq, new_pos = ao.subtract_num_rejected_tokens(
        seq_lens, positions, num_rejected, seq_idx, pos_idx
    )
    assert new_seq.tolist() == [97, 100, 98, 99, 100, 100]
    assert new_pos.tolist() == [7, 8, 12, 11, 12, 13, 15, 17, 18, 19]


@pytest.mark.parametrize("traced", [False, True])
def test_in_range_lookup_matches_indexing(traced):
    gen = torch.Generator().manual_seed(5)
    table = torch.randint(0, 200_000, (512,), generator=gen, dtype=torch.int32)
    index = torch.randint(0, 512, (96,), generator=gen, dtype=torch.int32)

    def fn(t, i):
        return small_lookup(t, i, in_range=True)

    run = torch.compile(fn, backend="eager") if traced else fn
    assert torch.equal(run(table, index), table[index])
