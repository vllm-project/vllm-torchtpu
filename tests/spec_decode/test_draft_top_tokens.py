# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_DRAFT_TOP_TOKENS: the greedy draft's tokens from the
vocabulary-parallel argmax match compute_logits(...).argmax(-1), on fake
tensor-parallel shards whose all-gather is played back from every rank."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.layers import logits_processor as lp_module
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm_torchtpu import envs
from vllm_torchtpu.runner import greedy_top_tokens
from vllm_torchtpu.spec_decode import mtp_fused_loop
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer


class _HeadShard(torch.nn.Module):
    """One rank's slice of a vocab-parallel LM head (rows [start, start+n))."""

    def __init__(self, weight, start, tp_size):
        super().__init__()
        self.weight = weight
        self.tp_size = tp_size
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=start, num_org_vocab_padding=0
        )
        self.quant_method = SimpleNamespace(
            apply=lambda layer, x, bias=None: x @ layer.weight.T
        )


def _processor(vocab):
    with patch.object(
        lp_module,
        "get_current_vllm_config",
        return_value=SimpleNamespace(model_config=None),
    ):
        return LogitsProcessor(vocab)


def _shards(weight, tp_size):
    vocab = weight.shape[0]
    assert vocab % tp_size == 0
    per_rank = vocab // tp_size
    return [
        _HeadShard(weight[r * per_rank : (r + 1) * per_rank], r * per_rank, tp_size)
        for r in range(tp_size)
    ]


class _FakeMtp(torch.nn.Module):
    """An MTP draft whose hidden state moves with the step and ignores the
    input ids, so every rank's forward is the same whatever ids it was fed."""

    plain_compute_logits = True

    def __init__(self, head, vocab, mix):
        super().__init__()
        self.lm_head = head
        self.logits_processor = _processor(vocab)
        self.mix = mix

    def forward(
        self,
        input_ids,
        positions,
        hidden_states,
        intermediate_tensors=None,
        inputs_embeds=None,
        spec_step_idx=0,
    ):
        return torch.tanh(hidden_states @ self.mix + spec_step_idx).to(
            hidden_states.dtype
        )

    def compute_logits(self, hidden_states, spec_step_idx=0):
        return self.logits_processor(self.lm_head, hidden_states)


def _play_back(run_rank, tp_size, monkeypatch):
    """Runs `run_rank(rank)` on every rank twice: once recording the pair each
    all-gather call contributes, then with every call answered by the
    concatenation of all ranks' pairs. Returns rank 0's second result."""
    recorded: dict[int, list[torch.Tensor]] = {}
    calls = [0]

    def record(pair, dim):
        recorded.setdefault(calls[0], []).append(pair)
        calls[0] += 1
        return pair.repeat(1, tp_size)

    monkeypatch.setattr(lp_module, "tensor_model_parallel_all_gather", record)
    for rank in range(tp_size):
        calls[0] = 0
        run_rank(rank)
    calls[0] = 0

    def play(pair, dim):
        out = torch.cat(recorded[calls[0]], dim=-1)
        calls[0] += 1
        return out

    monkeypatch.setattr(lp_module, "tensor_model_parallel_all_gather", play)
    return run_rank(0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tp_size", [1, 8])
def test_draft_top_token_ids_match_logits_argmax(monkeypatch, dtype, tp_size):
    gen = torch.Generator().manual_seed(7 + tp_size)
    vocab, hidden, rows = 8 * 97, 64, 64
    weight = torch.randn(vocab, hidden, generator=gen).to(dtype)
    hidden_states = torch.randn(rows, hidden, generator=gen).to(dtype)
    shards = _shards(weight, tp_size)

    def run_rank(rank):
        owner = SimpleNamespace(
            lm_head=shards[rank], logits_processor=_processor(vocab)
        )
        return greedy_top_tokens.top_token_ids(owner, hidden_states)

    got = _play_back(run_rank, tp_size, monkeypatch)
    want = torch.argmax(hidden_states @ weight.T, dim=-1).to(torch.int32)
    assert got.dtype == torch.int32 and got.shape == (rows,)
    assert torch.equal(got, want)


def test_draft_top_token_ids_break_ties_on_the_lowest_id(monkeypatch):
    # 16 ids, 2 per rank on 8 ranks. Row 0 ties ids 3 and 12 (ranks 1 and 6);
    # row 1 ties ids 4 and 5 (rank 2); row 2 ties every id.
    weight = torch.zeros(16, 3)
    weight[[3, 12], 0] = 2.0
    weight[[4, 5], 1] = 2.0
    hidden_states = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    shards = _shards(weight, 8)

    def run_rank(rank):
        owner = SimpleNamespace(lm_head=shards[rank], logits_processor=_processor(16))
        return greedy_top_tokens.top_token_ids(owner, hidden_states)

    got = _play_back(run_rank, 8, monkeypatch)
    assert got.tolist() == [3, 4, 0]
    assert torch.equal(got, torch.argmax(hidden_states @ weight.T, -1).to(torch.int32))


def test_fused_loop_top_tokens_match_compute_logits_argmax(monkeypatch):
    gen = torch.Generator().manual_seed(3)
    vocab, hidden, rows, tp_size = 8 * 61, 32, 16, 8
    weight = torch.randn(vocab, hidden, generator=gen).to(torch.bfloat16)
    mix = torch.randn(hidden, hidden, generator=gen).to(torch.bfloat16) / 4
    start_hidden = torch.randn(rows, hidden, generator=gen).to(torch.bfloat16)
    input_ids = torch.zeros(rows, dtype=torch.int32)
    positions = torch.arange(rows)
    shards = _shards(weight, tp_size)

    def loop(mtp, top_tokens):
        return mtp_fused_loop._MtpFusedLoop(
            vllm_config=None, prefix="p", mtp=mtp, num_steps=2, top_tokens=top_tokens
        )(input_ids, positions, start_hidden, torch.zeros(rows, dtype=torch.int32))

    def run_rank(rank):
        return loop(_FakeMtp(shards[rank], vocab, mix), top_tokens=True)

    got = _play_back(run_rank, tp_size, monkeypatch)
    want = loop(_FakeMtp(_HeadShard(weight, 0, 1), vocab, mix), top_tokens=False)
    assert len(got) == len(want) == 2
    for g, w in zip(got, want):
        assert g.dtype == torch.int32
        assert torch.equal(g, w)


def _proposer(draft_model):
    proposer = object.__new__(Eagle3Proposer)
    proposer.draft_model = draft_model
    proposer.runner = SimpleNamespace(_dp_lockstep_enabled=lambda: False)
    return proposer


def test_resolve_draft_head_owner(monkeypatch):
    mtp = _FakeMtp(_HeadShard(torch.zeros(8, 2), 0, 1), 8, torch.eye(2))
    monkeypatch.setattr(envs, "VLLM_TPU_DRAFT_TOP_TOKENS", False, raising=False)
    assert _proposer(mtp)._resolve_draft_head_owner() is None

    monkeypatch.setattr(envs, "VLLM_TPU_DRAFT_TOP_TOKENS", True, raising=False)
    assert _proposer(mtp)._resolve_draft_head_owner() is mtp

    mtp.draft_id_to_target_id = torch.zeros(8, dtype=torch.int64)
    with pytest.raises(NotImplementedError, match="draft_id_to_target_id"):
        _proposer(mtp)._resolve_draft_head_owner()
