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
"""dcp_local_kv: a prepared qkv_proj keeps the unprepared weight as its
column prefix, and `group_heads` turns its projection into this rank's q, the
group's K / V / index_q heads in DCP rank order (the order ops/msa.py
`gather_group_kv` all-gathers them in) and its index_k."""

import pytest
import torch

from vllm_torchtpu.models.minimax_m3.dcp_local_kv import (
    group_columns,
    group_heads,
    group_order,
)

TP, NQ, D, K = 8, 4, 16, 32


def _ranks(replicas: int):
    """Per TP rank: [K, (NQ + 4) D] weight and [(NQ + 4) D] scale of
    ``[q | k | v | index_q | index_k]``; rank r holds KV head r // replicas."""
    gen = torch.Generator().manual_seed(0)
    heads = TP // replicas
    kvi = [torch.randn(K, 3 * D, generator=gen) for _ in range(heads)]
    kvi_s = [torch.rand(3 * D, generator=gen) for _ in range(heads)]
    ik, ik_s = torch.randn(K, D, generator=gen), torch.rand(D, generator=gen)
    out = []
    for r in range(TP):
        q = torch.randn(K, NQ * D, generator=gen)
        q_s = torch.rand(NQ * D, generator=gen)
        h = r // replicas
        out.append((torch.cat([q, kvi[h], ik], 1), torch.cat([q_s, kvi_s[h], ik_s])))
    return out


def _project(w, s, x):
    return x @ (w * s)


@pytest.mark.parametrize("dcp, replicas", [(4, 2), (8, 2), (4, 1), (8, 4)])
def test_group_heads_in_rank_order(dcp, replicas):
    ranks = _ranks(replicas)
    x = torch.randn(5, K, generator=torch.Generator().manual_seed(1))
    kvi = slice(NQ * D, (NQ + 3) * D)
    heads = dcp // replicas
    width = (NQ + 4) * D
    for first in range(0, TP, dcp):
        group = ranks[first : first + dcp]
        own = [_project(w, s, x) for w, s in group]
        # The gather path: rank g * replicas brings group head g.
        gathered = torch.stack(own)[::replicas, :, kvi].reshape(heads, -1, 3, D)
        for rank, (w, s) in enumerate(group):
            new_w, new_s = group_columns(
                w,
                s,
                torch.stack([gw[:, kvi] for gw, _ in group]),
                torch.stack([gs[kvi] for _, gs in group]),
                rank,
                replicas,
            )
            assert new_w.shape == (K, width + 3 * (heads - 1) * D)
            assert torch.equal(new_w[:, :width], w)
            assert torch.equal(new_s[:width], s)
            order = group_order(
                torch.tensor([rank], dtype=torch.int32), heads, replicas
            )
            got = group_heads(_project(new_w, new_s, x), NQ, D, heads, order)
            want = _project(w, s, x)
            assert got.shape == (5, (NQ + 3 * heads + 1) * D)
            torch.testing.assert_close(got[:, : NQ * D], want[:, : NQ * D])
            torch.testing.assert_close(got[:, -D:], want[:, -D:])
            mid = got[:, NQ * D : -D].reshape(-1, 3, heads, D).permute(2, 0, 1, 3)
            torch.testing.assert_close(mid, gathered)


def test_group_columns_move_bytes():
    """e4m3 weights travel as their bytes: uint8 in, the same bytes out."""
    ranks = _ranks(2)
    to_bytes = lambda t: t.to(torch.float8_e4m3fn).view(torch.uint8)  # noqa: E731
    kvi = slice(NQ * D, (NQ + 3) * D)
    group = ranks[:4]
    w8 = [to_bytes(w) for w, _ in group]
    new_w, _ = group_columns(
        w8[3],
        group[3][1],
        torch.stack([w[:, kvi] for w in w8]),
        torch.stack([s[kvi] for _, s in group]),
        3,
        2,
    )
    assert new_w.dtype == torch.uint8
    assert torch.equal(new_w[:, : w8[3].shape[1]], w8[3])
    assert torch.equal(new_w[:, w8[3].shape[1] :], w8[0][:, kvi])
