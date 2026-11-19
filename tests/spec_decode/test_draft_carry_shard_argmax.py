# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_DRAFT_CARRY_SHARD_ARGMAX: draft step 0's first token in
`_carry_and_token_body` (served by _draft_rows_carry_and_token and its
ROWS_FRESH variant) takes the vocab-shard argmax and picks bit-identically the
token the full-logits argmax picks; the position carry is unchanged (CPU)."""

import types

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.spec_decode import shard_argmax
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer
from vllm_torchtpu.spec_decode.shard_argmax import local_candidates

TP, ORG, VPAD, ROWS = 4, 1000, 1024, 11


def _logits(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-4, 4, (ROWS, VPAD), generator=g).to(torch.bfloat16)
    x[:, ORG:] = 1e4  # vocab padding: must never win
    x[0] = 3  # whole row tied: lowest id
    x[1] = -5
    x[1, VPAD // TP - 1] = 7  # tie across a shard boundary
    x[1, VPAD // TP] = 7
    return x


class _Head:
    def __init__(self, rank):
        self.tp_size, self.tp_rank = TP, rank
        self.num_embeddings_padded = self.org_vocab_size_padded = VPAD


class _LP:
    def __init__(self, full, rank):
        self.full, self.rank = full, rank
        self.org_vocab_size, self.soft_cap, self.scale = ORG, None, 1.0
        self.logits_as_input, self.use_all_gather = False, True

    def _get_logits(self, hidden, head, bias, skip_gather=False):
        assert skip_gather
        s = VPAD // TP
        return self.full[:, self.rank * s : (self.rank + 1) * s]


def _proposer(full, rank, monkeypatch):
    s = VPAD // TP
    cands = [local_candidates(full[:, r * s : (r + 1) * s], r, ORG) for r in range(TP)]

    def fake_all_gather(cand, dim):
        assert dim == -1
        torch.testing.assert_close(cand, cands[rank], rtol=0, atol=0)
        return torch.cat(cands, dim=-1)

    monkeypatch.setattr(
        shard_argmax, "tensor_model_parallel_all_gather", fake_all_gather
    )
    monkeypatch.setattr(
        shard_argmax, "get_tensor_model_parallel_world_size", lambda: TP
    )
    draft = types.SimpleNamespace(
        logits_processor=_LP(full, rank),
        lm_head=_Head(rank),
        compute_logits=lambda h: full[:, :ORG],  # gathered, vocab-trimmed
    )
    return types.SimpleNamespace(draft_model=draft)


@pytest.mark.parametrize("rank", range(TP))
@pytest.mark.parametrize("positions_2d", [False, True])
def test_carry_token_bitwise_equal(monkeypatch, rank, positions_2d):
    full = _logits(seed=rank + 5)
    me = _proposer(full, rank, monkeypatch)
    positions = torch.arange(40, dtype=torch.int64) * 3
    if positions_2d:  # mrope-style [3, T]
        positions = torch.stack([positions, positions + 1, positions + 2])
    indices = torch.tensor([5, 0, 39, -1, 7, 2, 2, 11, -1, 30, 1])
    hidden = torch.zeros(ROWS, 8)
    monkeypatch.setattr(envs, "VLLM_TPU_DRAFT_SHARD_ARGMAX", True)
    monkeypatch.setattr(envs, "VLLM_TPU_DRAFT_CARRY_SHARD_ARGMAX", False)
    pos_off, tok_off = Eagle3Proposer._carry_and_token_body(
        me, positions, indices, hidden
    )
    monkeypatch.setattr(envs, "VLLM_TPU_DRAFT_CARRY_SHARD_ARGMAX", True)
    pos_on, tok_on = Eagle3Proposer._carry_and_token_body(
        me, positions, indices, hidden
    )
    assert tok_off.dtype == tok_on.dtype == torch.int32
    torch.testing.assert_close(tok_on, tok_off, rtol=0, atol=0)
    torch.testing.assert_close(pos_on, pos_off, rtol=0, atol=0)
    assert int(tok_on[0]) == 0 and int(tok_on[1]) == VPAD // TP - 1


def test_flag_on_takes_the_shard_path(monkeypatch):
    full = _logits(seed=1)
    me = _proposer(full, 0, monkeypatch)
    calls = []
    real = shard_argmax.tensor_model_parallel_all_gather
    monkeypatch.setattr(
        shard_argmax,
        "tensor_model_parallel_all_gather",
        lambda c, dim: calls.append(c.shape) or real(c, dim),
    )
    monkeypatch.setattr(envs, "VLLM_TPU_DRAFT_SHARD_ARGMAX", True)
    monkeypatch.setattr(envs, "VLLM_TPU_DRAFT_CARRY_SHARD_ARGMAX", True)
    Eagle3Proposer._carry_and_token_body(
        me, torch.arange(20), torch.arange(ROWS), torch.zeros(ROWS, 8)
    )
    assert calls == [torch.Size([ROWS, 2])]  # one (value, id) pair per row


def test_off_by_default():
    assert envs.environment_variables["VLLM_TPU_DRAFT_CARRY_SHARD_ARGMAX"]() is False
