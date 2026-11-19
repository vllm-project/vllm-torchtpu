# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_DCP_ONE_GATHER: the pair's partial outputs and LSEs gathered in
one all-gather must reach the DCP combine with the same values, bit for bit,
as the two separate all-gathers, on every rank of the pair.

CPU only. The pair group is emulated: each rank's all-gather returns the
concatenation of every rank's input to the same (ordered) call.
"""

import pytest
import torch

from vllm_torchtpu.layers.adapter.custom_ops import qsa_op


class _Record:
    """Pass 1: record each rank's inputs, return a correctly shaped dummy."""

    def __init__(self, size):
        self.size, self.calls = size, []

    def all_gather(self, x, dim=0):
        self.calls.append(x.clone())
        return torch.cat([x] * self.size, dim=dim)


class _Replay:
    """Pass 2: call i returns the concatenation of every rank's call-i input."""

    def __init__(self, recorded):
        self.recorded, self.i = recorded, 0

    def all_gather(self, x, dim=0):
        parts = [calls[self.i] for calls in self.recorded]
        self.i += 1
        return torch.cat(parts, dim=dim)


def _run(monkeypatch, one_gather, outs, lses, qgates, head_dim):
    monkeypatch.setattr(qsa_op, "_dcp_gather_partials", _ORIG, raising=True)
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "VLLM_TPU_QSA_DCP_ONE_GATHER", one_gather, raising=False)
    size = len(outs)
    seen = []

    def combine(out_pair, lse_pair, gate, rank):
        seen.append((out_pair.clone(), lse_pair.clone(), gate.clone(), int(rank)))
        tokens, heads, d = gate.shape
        return torch.zeros(tokens, heads * d)

    recs = [_Record(size) for _ in range(size)]
    for r in range(size):
        qsa_op._dcp_finish(outs[r], lses[r], qgates[r], head_dim, r, recs[r], combine)
    recorded = [rec.calls for rec in recs]
    seen.clear()
    for r in range(size):
        qsa_op._dcp_finish(outs[r], lses[r], qgates[r], head_dim, r, _Replay(recorded), combine)
    return seen, [len(c) for c in recorded]


_ORIG = qsa_op._dcp_gather_partials


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("heads_pair,head_dim", [(16, 128), (8, 64), (200, 128), (3, 128)])
@pytest.mark.parametrize("tokens", [4, 16])
def test_one_gather_same_values(monkeypatch, size, heads_pair, head_dim, tokens):
    g = torch.Generator().manual_seed(heads_pair * 31 + tokens + size)
    outs = [torch.randn(tokens, heads_pair, head_dim, generator=g) for _ in range(size)]
    lses = [torch.randn(tokens, heads_pair, generator=g) * 5 for _ in range(size)]
    # Dead rows carry -inf LSE (qsa_rpa: jnp.where(live, lse, -inf)): must survive.
    lses[0][0] = -float("inf")
    local_heads = max(1, heads_pair // size)
    qgates = [torch.randn(tokens, local_heads * 2 * head_dim, generator=g) for _ in range(size)]
    ref, n_ref = _run(monkeypatch, False, outs, lses, qgates, head_dim)
    got, n_got = _run(monkeypatch, True, outs, lses, qgates, head_dim)
    assert n_ref == [2] * size and n_got == [1] * size  # two collectives -> one
    assert len(ref) == len(got) == size
    for (o1, l1, g1, r1), (o2, l2, g2, r2) in zip(ref, got):
        assert r1 == r2
        assert o1.shape == o2.shape == (size, tokens, heads_pair, head_dim)
        assert l1.shape == l2.shape == (size, tokens, heads_pair)
        assert torch.equal(o1, o2) and torch.equal(l1, l2) and torch.equal(g1, g2)
    # And they are the real pair data, rank-ordered.
    assert torch.equal(got[1][0], torch.stack(outs))
    assert torch.equal(got[1][1], torch.stack(lses))


def test_flag_default_off():
    from vllm_torchtpu import envs

    assert envs.environment_variables["VLLM_TPU_QSA_DCP_ONE_GATHER"]() is False
