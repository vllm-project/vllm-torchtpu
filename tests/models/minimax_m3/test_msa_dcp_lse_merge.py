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
"""The MSA DCP decode-step collectives (`ops.msa`): the exchange merge gives
the reduce-scatter merge's result, and the joint query + candidate gather the
two separate gathers', bit for bit on every rank."""

import concurrent.futures
import threading

import numpy as np
import pytest
import torch

from vllm_torchtpu.models.minimax_m3.ops import msa


class _Hub:
    def __init__(self, world):
        self.slots = [None] * world
        self.barrier = threading.Barrier(world)

    def exchange(self, rank, x):
        self.slots[rank] = x
        self.barrier.wait()
        got = list(self.slots)
        self.barrier.wait()
        return got


class _Group:
    """One rank of a DCP group whose ranks are threads in lockstep. The
    reduce-scatter adds the ranks' terms in f32 and rounds once."""

    def __init__(self, hub, rank, world):
        self._hub = hub
        self.rank_in_group = rank
        self.world_size = world
        self.device_group = None

    def all_gather(self, x, dim=0):
        return torch.cat(self._hub.exchange(self.rank_in_group, x), dim=dim)

    def reduce_scatter(self, x, dim=0):
        parts = self._hub.exchange(self.rank_in_group, x)
        total = sum(p.to(torch.float32) for p in parts).to(x.dtype)
        width = x.shape[dim] // self.world_size
        return total.narrow(dim, self.rank_in_group * width, width)

    def all_to_all(self, x):
        parts = self._hub.exchange(self.rank_in_group, x)
        width = x.shape[0] // self.world_size
        mine = slice(self.rank_in_group * width, (self.rank_in_group + 1) * width)
        return torch.cat([p[mine] for p in parts])


def _partials(rng, world, tokens, local_heads=8, head_dim=128):
    """Head-major bf16 partials and f32 LSEs hitting every merge rule."""
    heads = world * local_heads
    out = rng.standard_normal((world, heads, tokens, head_dim)).astype(np.float32)
    out = torch.from_numpy(out).to(torch.bfloat16)
    lse = torch.from_numpy(
        (rng.standard_normal((world, heads, tokens)) * 4).astype(np.float32)
    )
    lse[:, :, 0] += 200.0  # exp overflows without the logsumexp shift
    lse[:, :, 1] -= 200.0  # exp underflows without it
    lse[1:, :, 2] = float("-inf")  # rank 0 owns token 2 alone ...
    out[1:, :, 2] = float("nan")  # ... and the others' partials are garbage
    lse[:, :, 3] = float("-inf")  # nobody owns token 3 (a padding row)
    lse[world - 1] = float("-inf")  # the last rank owns nothing
    lse[0, :local_heads, 4] = float("-inf")  # a row mixed per head
    return out, lse


def _merge_every_rank(out, lse, monkeypatch, max_tokens):
    world = out.shape[0]
    monkeypatch.setattr(msa, "EXCHANGE_MERGE_MAX_TOKENS", max_tokens)
    monkeypatch.setattr(msa, "_all_to_all", lambda x, group: group.all_to_all(x))
    hub = _Hub(world)

    def run(rank):
        group = _Group(hub, rank, world)
        return msa.merge_head_partials(out[rank], lse[rank], group)

    with concurrent.futures.ThreadPoolExecutor(world) as pool:
        return list(pool.map(run, range(world)))


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("tokens", [8, 13, 24])
def test_exchange_merge_is_the_reduce_scatter_merge(world, tokens, monkeypatch):
    out, lse = _partials(np.random.default_rng(world * 100 + tokens), world, tokens)
    scattered = _merge_every_rank(out, lse, monkeypatch, max_tokens=0)
    exchanged = _merge_every_rank(out, lse, monkeypatch, max_tokens=tokens)
    for rank in range(world):
        assert exchanged[rank].shape == (tokens, 8, 128)
        assert exchanged[rank].dtype == torch.bfloat16
        torch.testing.assert_close(
            exchanged[rank], scattered[rank], rtol=0, atol=0, equal_nan=True
        )
        assert torch.isfinite(exchanged[rank]).all()
        # A lone owner's partial comes back verbatim; an unowned row is zero.
        own = slice(rank * 8, (rank + 1) * 8)
        torch.testing.assert_close(exchanged[rank][2], out[0, own, 2], rtol=0, atol=0)
        assert (exchanged[rank][3] == 0).all()


def test_positive_control_a_misrouted_partial_is_caught(monkeypatch):
    """The comparison fails when a rank merges another rank's heads."""
    out, lse = _partials(np.random.default_rng(5), 2, 8)
    scattered = _merge_every_rank(out, lse, monkeypatch, max_tokens=0)
    monkeypatch.setattr(msa, "EXCHANGE_MERGE_MAX_TOKENS", 8)
    hub = _Hub(2)

    def swapped(x, group):  # every rank receives the other rank's heads
        got = group.all_to_all(torch.roll(x, x.shape[0] // 2, dims=0))
        return got

    monkeypatch.setattr(msa, "_all_to_all", swapped)

    def run(rank):
        return msa.merge_head_partials(out[rank], lse[rank], _Group(hub, rank, 2))

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        exchanged = list(pool.map(run, range(2)))
    assert not torch.equal(exchanged[0], scattered[0])


@pytest.mark.parametrize("world", [2, 8])
def test_queries_cross_with_the_candidates_unchanged(world):
    """One gather of queries + candidates gives both separate gathers'
    results bit for bit, including -inf scores and -1 block ids."""
    rng = np.random.default_rng(world)
    tokens, heads, head_dim, width = 8, 8, 128, 32
    q = torch.from_numpy(
        rng.standard_normal((world, tokens, heads, head_dim)).astype(np.float32)
    ).to(torch.bfloat16)
    candidates = torch.from_numpy(
        rng.standard_normal((world, tokens, width)).astype(np.float32)
    )
    candidates[:, :, 10:16] = float("-inf")
    candidates[:, :, 26:] = -1.0
    hub = _Hub(world)

    def run(rank):
        group = _Group(hub, rank, world)
        together = msa.gather_queries_with_candidates(q[rank], candidates[rank], group)
        apart = (
            msa.gather_query_heads(q[rank], group),
            group.all_gather(candidates[rank].unsqueeze(0), dim=0),
        )
        return together, apart

    with concurrent.futures.ThreadPoolExecutor(world) as pool:
        for (q_all, cands), (q_ref, cands_ref) in pool.map(run, range(world)):
            assert q_all.dtype == torch.bfloat16
            torch.testing.assert_close(q_all, q_ref, rtol=0, atol=0)
            torch.testing.assert_close(cands, cands_ref, rtol=0, atol=0)
