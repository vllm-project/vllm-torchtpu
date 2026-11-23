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
"""The packed DCP exchanges of `ops.msa` for decode steps, every rank run
on its own thread against an in-process group: the queries and candidates
arrive bit for bit, and the all-to-all merge is the exact log-sum-exp merge
rounded once."""

import threading

import pytest
import torch

from vllm_torchtpu.models.minimax_m3.ops import msa


class _Group:
    """`all_gather` / all-to-all over threads, one per rank."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self._barrier = threading.Barrier(world_size)
        self._slots: list = [None] * world_size
        self._rank = threading.local()

    def exchange(self, x: torch.Tensor) -> list[torch.Tensor]:
        self._slots[self._rank.value] = x
        self._barrier.wait()
        got = list(self._slots)
        self._barrier.wait()
        return got

    def all_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.cat(self.exchange(x), dim=dim)

    def all_to_all(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0] // self.world_size
        r = self._rank.value
        return torch.cat([g[r * n : (r + 1) * n] for g in self.exchange(x)], dim=0)

    def run(self, fn):
        results: list = [None] * self.world_size

        def body(rank):
            self._rank.value = rank
            results[rank] = fn(rank)

        threads = [
            threading.Thread(target=body, args=(r,)) for r in range(self.world_size)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results


@pytest.mark.parametrize("dcp,num_tokens", [(2, 8), (2, 24), (4, 5)])
def test_queries_and_candidates_arrive_bit_for_bit(dcp, num_tokens):
    torch.manual_seed(dcp * 100 + num_tokens)
    group = _Group(dcp)
    qs = [torch.randn(num_tokens, 8, 128).to(torch.bfloat16) for _ in range(dcp)]
    qs[0][0, 0, :4] = 0  # the high half of a word: a subnormal f32
    cands = []
    for _ in range(dcp):
        scores = torch.randn(num_tokens, 16)
        scores[0, -1] = float("-inf")
        blocks = torch.randint(-1, 2048, (num_tokens, 16)).float()
        cands.append(torch.cat([scores, blocks], dim=1))
    got = group.run(
        lambda r: msa.exchange_queries_and_candidates(qs[r], cands[r], group)
    )
    want_q, want_c = torch.cat(qs, dim=1), torch.stack(cands)
    for q_all, c in got:
        assert torch.equal(q_all.view(torch.int16), want_q.view(torch.int16))
        assert torch.equal(c.view(torch.int32), want_c.view(torch.int32))


@pytest.mark.parametrize("dcp,num_tokens", [(2, 8), (2, 24), (4, 5)])
def test_all_to_all_merge_is_the_exact_merge_rounded_once(dcp, num_tokens, monkeypatch):
    torch.manual_seed(dcp * 10 + num_tokens)
    group = _Group(dcp)
    monkeypatch.setattr(msa, "_all_to_all", lambda x, g: g.all_to_all(x))
    heads = 8 * dcp
    outs = [torch.randn(heads, num_tokens, 128).to(torch.bfloat16) for _ in range(dcp)]
    lses = [3 * torch.randn(heads, num_tokens) for _ in range(dcp)]
    lses[0][:, 0], outs[0][:, 0] = float("-inf"), float("nan")  # rank 0 saw nothing
    for lse in lses:
        lse[:, 1] = float("-inf")  # a padding token
    got = group.run(lambda r: msa.merge_head_partials_a2a(outs[r], lses[r], group))

    out = torch.stack([o.double() for o in outs])
    lse = torch.stack([x.double() for x in lses])
    weight = torch.exp(lse - torch.logsumexp(lse, 0, keepdim=True)).unsqueeze(-1)
    exact = torch.where(weight > 0, out * weight, torch.zeros_like(out)).sum(0)
    local = heads // dcp
    for r, merged in enumerate(got):
        want = exact[r * local : (r + 1) * local].transpose(0, 1)
        assert merged.dtype == torch.bfloat16
        torch.testing.assert_close(merged.double(), want, rtol=2**-8, atol=1e-6)
        assert (merged[1] == 0).all()
