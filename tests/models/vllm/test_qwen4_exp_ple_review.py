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

"""Owned PLE rows, packed collective bytes and public checkpoint contracts."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from torch import nn
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)

from vllm_torchtpu.models.vllm.qwen4_exp.parallel import PLEParallel
from vllm_torchtpu.models.vllm.qwen4_exp.ple import PLEInputs, Qwen4ExpNGramEmbedding
from vllm_torchtpu.models.vllm.qwen4_exp.ple_storage import HostPLETable
from vllm_torchtpu.models.vllm.qwen4_exp.pooled_ple import Qwen4ExpPooledPLE


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("ids", [[-1, 19, 20, 22, 25, 24], [-1, 19, 25], []])
def test_host_gathers_only_owned_rows(monkeypatch, fp8, ids):
    table = HostPLETable.__new__(HostPLETable)
    nn.Module.__init__(table)
    table._finalized = True
    table.storage_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    table.shard_indices = VocabParallelEmbeddingShardIndices(20, 25, 0, 0, 20, 25, 0, 0)
    table.embedding_dim = 7
    table._shared_table = None
    table._gather_workers = 1
    table._gather_pool = None
    bits = torch.arange(35).reshape(5, 7).to(torch.uint8 if fp8 else torch.int16)
    table.weight = nn.Parameter(bits.view(table.storage_dtype), requires_grad=False)
    observed = []
    take = table._take_rows

    def gather(source, rows, output):
        observed.append(rows.copy())
        assert np.all((rows >= 0) & (rows < 5))
        take(source, rows, output)

    monkeypatch.setattr(table, "_take_rows", gather)
    query = torch.tensor(ids, dtype=torch.int32).reshape(-1, 1)
    actual = table.lookup_host(query)
    expected = torch.zeros(len(ids), 7, dtype=actual.dtype)
    for row, index in enumerate(ids):
        if 20 <= index < 25:
            expected[row] = bits[index - 20].view(actual.dtype)
    assert torch.equal(actual, expected)
    owned = [i - 20 for i in ids if 20 <= i < 25]
    assert len(observed) == bool(owned)
    if owned:
        np.testing.assert_array_equal(observed[0], owned)
    table.close()


@pytest.mark.parametrize("carrier", [torch.int8, torch.int16])
@pytest.mark.parametrize(
    "tp,dp,across", [(1, 8, True), (2, 1, False), (2, 4, True), (1, 1, False)]
)
@pytest.mark.parametrize("tokens,width", [(0, 3), (1, 3), (7, 5), (8, 16)])
def test_combine_uses_word_carriers_for_each_multi_rank_group(
    carrier, tp, dp, across, tokens, width
):
    calls = []

    class Group:
        def __init__(self, name, size):
            self.name, self.world_size, self.rank_in_group = name, size, 0

        def all_reduce(self, value):
            calls.append((self.name, value.dtype, value.nbytes))
            if self.world_size > 1:
                assert value.dtype == torch.int32
            return value.clone()

    parallel = PLEParallel(Group("tp", tp), Group("dp", dp), embedding_across_dp=across)
    count = tokens * (dp if parallel.across_dp else 1)
    bits = (torch.arange(count * width) * 73 + 159).to(carrier).reshape(count, width)
    before = bits.clone()
    actual = parallel.combine(bits, tokens)
    assert actual.shape == (tokens, width) and actual.dtype == carrier
    assert torch.equal(actual, before[:tokens]) and torch.equal(bits, before)
    assert [name for name, _, _ in calls] == (
        ["dp", "tp"] if parallel.across_dp else ["tp"]
    )


def test_pooled_forward_masks_null_padding_and_tail(monkeypatch):
    layer = Qwen4ExpPooledPLE.__new__(Qwen4ExpPooledPLE)
    nn.Module.__init__(layer)
    layer._finalized, layer.cpu_offload = True, True
    layer.cache_owners = (torch.full((4, 2, 8), 19, dtype=torch.bfloat16),)
    before = layer.cache_owners[0].clone()
    table = HostPLETable.__new__(HostPLETable)
    nn.Module.__init__(table)
    monkeypatch.setattr(table, "decode", lambda raw: raw)
    layer.ple_embedding = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(layer.ple_embedding)
    layer.ple_embedding.ngram_embedding = table
    layer.ple_embedding.ngram_size = 2
    layer.ple_embedding.heads_per_ngram = 1
    table.embedding_dim = 8
    layer.parallel = PLEParallel(
        NS(world_size=1, rank_in_group=0, all_reduce=lambda value: value),
        NS(world_size=1, rank_in_group=0, all_reduce=lambda value: value),
        embedding_across_dp=False,
    )
    monkeypatch.setattr(layer, "norm_conv", nn.Identity(), raising=False)
    layer.conv1d = nn.Conv1d(8, 8, 1, bias=False)
    layer._conv = lambda *args: torch.ones_like(args[0])
    monkeypatch.setattr(
        layer, "gated_values", lambda hidden, decoded: torch.full_like(hidden, 3)
    )
    hidden = torch.zeros((7, 8), dtype=torch.bfloat16)
    inputs = PLEInputs(
        torch.arange(7),
        torch.tensor([0, 2, 4, 6]),
        torch.zeros((3, 1), dtype=torch.int64),
    )
    actual = layer.forward_with_history(
        hidden,
        inputs,
        torch.tensor([0, 2, -1]),
        torch.zeros(3, dtype=torch.bool),
        torch.tensor(6),
        hidden.view(torch.int16),
    )
    expected = torch.zeros_like(hidden)
    expected[2:4] = 4
    assert torch.equal(actual, expected)
    assert torch.equal(layer.cache_owners[0], before)
