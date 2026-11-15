# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_DRAFT_SHARD_ARGMAX: the per-shard argmax + candidate all-gather picks
exactly the token the full-logits argmax picks (CPU)."""

import types

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.spec_decode import shard_argmax
from vllm_torchtpu.spec_decode.shard_argmax import (
    combine_candidates,
    greedy_draft_token,
    local_candidates,
)


def _padded(org, tp):
    return -(-org // 64) * 64 + (0 if (-(-org // 64) * 64) % tp == 0 else 64)


def _logits(rows, org, tp, dtype, seed):
    """Small-integer logits (many exact ties, also across shard boundaries),
    huge values in the vocab padding (must never win), some -inf."""
    g = torch.Generator().manual_seed(seed)
    vpad = _padded(org, tp)
    x = torch.randint(-4, 4, (rows, vpad), generator=g).to(dtype)
    x[:, org:] = 1e4  # padding beats everything unless masked
    x[0] = 3  # whole row tied: lowest id wins
    shard = vpad // tp
    x[1] = -5
    x[1, shard - 1] = 7  # tie right at a shard boundary
    x[1, shard] = 7
    x[2, : org // 2] = float("-inf")
    return x


def _sharded_pick(full, org, tp):
    shard = full.shape[-1] // tp
    cands = [
        local_candidates(full[:, r * shard : (r + 1) * shard], r, org)
        for r in range(tp)
    ]
    return combine_candidates(torch.cat(cands, dim=-1))  # all_gather(dim=-1) order


@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("org", [1000, 151936, 64 * 8 + 3])
def test_shard_pick_equals_full_argmax(tp, dtype, org):
    rows = 37 if org < 10000 else 5
    full = _logits(rows, org, tp, dtype, seed=tp * 7 + org)
    want = full[:, :org].argmax(dim=-1).to(torch.int32)
    got = _sharded_pick(full, org, tp)
    assert got.dtype == torch.int32
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert int(got[0]) == 0 and int(got[1]) == full.shape[-1] // tp - 1


def test_positive_control_unmasked_padding_would_differ():
    tp, org = 4, 1000
    full = _logits(9, org, tp, torch.float32, seed=1)
    shard = full.shape[-1] // tp
    cands = [
        local_candidates(full[:, r * shard : (r + 1) * shard], r, full.shape[-1])
        for r in range(tp)
    ]  # no padding mask
    assert (combine_candidates(torch.cat(cands, -1))[3:] >= org).all()  # rows 0-2 overwrite the padding


class _Head:
    def __init__(self, tp, rank, vpad):
        self.tp_size, self.tp_rank = tp, rank
        self.num_embeddings_padded = self.org_vocab_size_padded = vpad


class _LP:
    def __init__(self, full, tp, rank, org, soft_cap, scale):
        self.full, self.tp, self.rank = full, tp, rank
        self.org_vocab_size, self.soft_cap, self.scale = org, soft_cap, scale
        self.logits_as_input, self.use_all_gather = False, True

    def _get_logits(self, hidden, head, bias, skip_gather=False):
        assert skip_gather
        shard = self.full.shape[-1] // self.tp
        return self.full[:, self.rank * shard : (self.rank + 1) * shard]

    def finish(self, x):  # LogitsProcessor.forward's elementwise tail
        if self.soft_cap is not None:
            x = torch.tanh(x / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            x = x * self.scale
        return x


@pytest.mark.parametrize("soft_cap,scale", [(None, 1.0), (30.0, 0.5)])
def test_greedy_draft_token_wiring(monkeypatch, soft_cap, scale):
    tp, org = 4, 1000
    full = _logits(11, org, tp, torch.bfloat16, seed=3) * 3
    vpad = full.shape[-1]
    shard = vpad // tp
    monkeypatch.setattr(envs, "VLLM_TPU_DRAFT_SHARD_ARGMAX", True)
    monkeypatch.setattr(shard_argmax, "get_tensor_model_parallel_world_size", lambda: tp)
    for rank in range(tp):
        lp = _LP(full, tp, rank, org, soft_cap, scale)
        others = [
            local_candidates(lp.finish(full[:, r * shard : (r + 1) * shard]), r, org)
            for r in range(tp)
        ]

        def fake_all_gather(cand, dim, rank=rank, others=others):
            assert dim == -1
            torch.testing.assert_close(cand, others[rank], rtol=0, atol=0)
            return torch.cat(others, dim=-1)

        monkeypatch.setattr(shard_argmax, "tensor_model_parallel_all_gather", fake_all_gather)
        model = types.SimpleNamespace(logits_processor=lp, lm_head=_Head(tp, rank, vpad))
        got = greedy_draft_token(model, torch.zeros(11, 8))
        want = lp.finish(full)[:, :org].argmax(dim=-1).to(torch.int32)
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_flag_off_uses_compute_logits(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_DRAFT_SHARD_ARGMAX", False)
    logits = torch.tensor([[0.0, 2.0, 2.0, 1.0]])
    model = types.SimpleNamespace(compute_logits=lambda h: logits)
    assert greedy_draft_token(model, torch.zeros(1, 8)).tolist() == [1]
