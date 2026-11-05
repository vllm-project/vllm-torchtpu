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
"""The per-shard argmax plus pair exchange equals torch.argmax over the
gathered scores, ties and NaNs included."""

import pytest
import torch

from vllm_torchtpu.spec_decode.vocab_parallel_argmax import (
    reduce_shard_argmax_pairs,
    shard_argmax_pair,
)


def sharded_argmax(full: torch.Tensor, tp_size: int, shard: int) -> torch.Tensor:
    """Simulate tp_size ranks of ``shard`` padded columns each (vLLM layout:
    the vocab fills the padded space from the front, pads sit at the tail and
    are -inf), then all-gather the pairs rank-major."""
    rows, vocab = full.shape
    pairs = []
    for rank in range(tp_size):
        start = min(rank * shard, vocab)
        end = min(start + shard, vocab)
        local = torch.full((rows, shard), float("-inf"), dtype=full.dtype)
        local[:, : end - start] = full[:, start:end]
        pairs.append(shard_argmax_pair(local, torch.tensor(start)))
    return reduce_shard_argmax_pairs(torch.cat(pairs, dim=-1), tp_size)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tp_size,vocab,shard", [(2, 64, 32), (4, 96, 24), (4, 90, 24)])
def test_matches_gathered_argmax_with_ties(dtype, tp_size, vocab, shard):
    gen = torch.Generator().manual_seed(0)
    # Few distinct values: most rows have ties inside and across shards.
    full = torch.randint(-3, 4, (256, vocab), generator=gen).to(dtype)
    # Ties exactly at every shard boundary, and the max both sides of it.
    for b in range(1, tp_size):
        full[b, :] = -5
        full[b, b * shard - 1] = 7
        full[b, b * shard] = 7
        full[tp_size + b, :] = -5
        full[tp_size + b, b * shard] = 7
        full[tp_size + b, vocab - 1] = 7
    # The max only in the last real column, beside the pad columns.
    full[2 * tp_size, :] = -5
    full[2 * tp_size, vocab - 1] = 1
    full[2 * tp_size + 1, :] = float("-inf")  # all -inf: torch gives id 0
    full[2 * tp_size + 2, :] = 2  # one value everywhere: id 0
    expected = full.argmax(dim=-1)
    assert sharded_argmax(full, tp_size, shard).tolist() == expected.tolist()


def test_nan_wins_like_torch_argmax():
    full = torch.zeros(3, 64)
    full[0, 40] = float("nan")
    full[1, 10] = float("nan")
    full[1, 50] = float("nan")
    full[2, 33] = float("nan")
    full[2, 5] = 9.0
    expected = full.argmax(dim=-1)
    assert expected.tolist() == [40, 10, 33]
    assert sharded_argmax(full, 4, 16).tolist() == expected.tolist()


def test_ids_beyond_bf16_precision_are_exact():
    # MiniMax M3.1 vocab, 4 shards; bf16 would round these ids.
    vocab, tp_size = 200_064, 4
    full = torch.zeros(4, vocab, dtype=torch.bfloat16)
    ids = [vocab - 1, 150_049, 100_033, 50_017]
    for row, token in enumerate(ids):
        full[row, token] = 1
    assert sharded_argmax(full, tp_size, vocab // tp_size).tolist() == ids
