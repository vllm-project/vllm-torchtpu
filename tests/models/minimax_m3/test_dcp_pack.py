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
"""`models.minimax_m3.dcp_pack` against the four-collective path, on CPU,
with the DCP ranks simulated in one process."""

import importlib.util
import pathlib

import pytest
import torch

_PATH = (
    pathlib.Path(__file__).parents[3]
    / "src/vllm_torchtpu/models/minimax_m3/dcp_pack.py"
)
_spec = importlib.util.spec_from_file_location("dcp_pack", _PATH)
dcp_pack = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dcp_pack)


class _Group:
    """One rank's view of a simulated DCP group: collectives read every
    rank's operand from `bank` (filled by the caller before the call)."""

    def __init__(self, rank, world, bank):
        self.rank, self.world_size, self.bank = rank, world, bank

    def all_gather(self, x, dim):
        return torch.cat([self.bank[r] for r in range(self.world_size)], dim=dim)


def _all_to_all(x, group):
    world = group.world_size
    return torch.cat(
        [group.bank[r].chunk(world, dim=0)[group.rank] for r in range(world)], dim=0
    )


def _reference_merge(outs, lses, rank, round_operands):
    """`ops.msa.merge_head_partials` (no hadamard, before the transpose):
    LSE all-gather, per-rank weight, reduce-scatter of the weighted partials."""
    world = len(outs)
    lse_t = [lse.to(torch.float32).transpose(0, 1).contiguous() for lse in lses]
    lses_all = torch.stack(lse_t)  # [world, T, H]
    total = torch.logsumexp(lses_all, dim=0)
    weighted = []
    for r in range(world):
        w = torch.exp(lse_t[r] - total).transpose(0, 1).unsqueeze(-1)
        x = outs[r].to(torch.float32) * w
        x = torch.where(w > 0, x, torch.zeros_like(x))
        if round_operands:
            x = x.to(outs[r].dtype)
        weighted.append(x)
    local = outs[0].shape[0] // world
    parts = [x[rank * local : (rank + 1) * local] for x in weighted]
    acc = parts[0]
    for p in parts[1:]:
        acc = acc + p
    return acc.to(outs[0].dtype)


def _partials(world, heads, tokens, dim, seed, dead_rows=()):
    g = torch.Generator().manual_seed(seed)
    outs, lses = [], []
    for _ in range(world):
        outs.append(torch.randn(heads, tokens, dim, generator=g).to(torch.bfloat16))
        lse = torch.randn(heads, tokens, generator=g) * 4
        for t in dead_rows:
            lse[:, t] = float("-inf")
        lses.append(lse)
    # A rank that holds none of a token's selected blocks: -inf LSE, NaN out.
    if tokens > 1:
        lses[0][:, 1] = float("-inf")
        outs[0][:, 1] = float("nan")
    return outs, lses


@pytest.mark.parametrize("round_operands", [True, False])
@pytest.mark.parametrize("tokens", [1, 8, 24, 48, 384])
def test_merge_matches_reduce_scatter_dcp2(tokens, round_operands):
    world, heads, dim = 2, 16, 128
    outs, lses = _partials(world, heads, tokens, dim, seed=tokens, dead_rows=(0,))
    bank = [dcp_pack.pack_partials(o, l) for o, l in zip(outs, lses)]
    for rank in range(world):
        got = dcp_pack.merge_head_partials_all_to_all(
            outs[rank], lses[rank], _Group(rank, world, bank), _all_to_all,
            round_operands,
        )
        want = _reference_merge(outs, lses, rank, round_operands)
        assert got.dtype == want.dtype and got.shape == want.shape
        assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_merge_dcp4_close():
    world, heads, tokens, dim = 4, 32, 8, 128
    outs, lses = _partials(world, heads, tokens, dim, seed=4)
    bank = [dcp_pack.pack_partials(o, l) for o, l in zip(outs, lses)]
    for rank in range(world):
        got = dcp_pack.merge_head_partials_all_to_all(
            outs[rank], lses[rank], _Group(rank, world, bank), _all_to_all, True
        )
        want = _reference_merge(outs, lses, rank, True)
        torch.testing.assert_close(got.float(), want.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("tokens", [1, 8, 48, 384])
def test_gather_queries_and_candidates(tokens):
    world, heads, dim, cols = 2, 8, 128, 32
    g = torch.Generator().manual_seed(tokens)
    qs = [torch.randn(tokens, heads, dim, generator=g).to(torch.bfloat16) for _ in range(world)]
    cands = []
    for _ in range(world):
        c = torch.randn(tokens, cols, generator=g)
        c[:, cols // 2 :] = torch.randint(-1, 4000, (tokens, cols // 2), generator=g).float()
        c[0, 0] = float("-inf")
        c[-1, 1] = 1e-40  # a subnormal travels as bits
        cands.append(c)
    bank = [
        torch.cat([q.view(torch.int16), dcp_pack._rows(c, dim)], 1).unsqueeze(0)
        for q, c in zip(qs, cands)
    ]
    want_q = torch.cat(qs, dim=1)
    want_c = torch.stack(cands)
    for rank in range(world):
        q_all, c_all = dcp_pack.gather_queries_and_candidates(
            qs[rank], cands[rank], _Group(rank, world, bank)
        )
        assert torch.equal(q_all.view(torch.int16), want_q.view(torch.int16))
        assert torch.equal(c_all.view(torch.int32), want_c.view(torch.int32))
