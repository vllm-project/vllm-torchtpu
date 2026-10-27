# SPDX-License-Identifier: Apache-2.0
"""TPU_DRAFT_VOCAB_PARALLEL: the draft's BF16 greedy head split over all
DP x TP ranks returns, for every DP engine's rows, the full-logits argmax
(ties to the lowest id), the same tokens as the per-TP-group head, and every
rank issues the same collectives."""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import pytest
import torch
from vllm.distributed import parallel_state

from vllm_torchtpu.runner import greedy_top_tokens as gtt


class _Method:
    """vLLM's UnquantizedEmbeddingMethod.apply: F.linear on the weight."""

    def apply(self, layer, x, bias=None):
        out = torch.nn.functional.linear(x, layer.weight)
        return out if bias is None else out + bias


class _Shard(torch.nn.Module):
    """One TP rank's BF16 head shard (the draft's twin)."""

    def __init__(self, weight, start, tp_size, num_pad):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.tp_size = tp_size
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=start, num_org_vocab_padding=num_pad
        )
        self.quant_method = _Method()
        self._quant_mode = None


class _Processor:
    scale = 1.0
    soft_cap = None

    def get_top_tokens(self, lm_head, hidden):
        return gtt.tpu_get_top_tokens(self, lm_head, hidden)

    def _apply_head(self, lm_head, hidden, bias=None):
        return lm_head.quant_method.apply(lm_head, hidden, bias)


class _Owner(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.lm_head = head
        self.logits_processor = _Processor()


def _tp_shards(weight, tp_size, dp_size):
    """vLLM-style padded TP shards; the padding rows would win the argmax if
    the mask were missing. Shard size is a multiple of dp_size."""
    vocab, hidden = weight.shape
    per_rank = math.ceil(vocab / (tp_size * dp_size)) * dp_size
    pad_rows = torch.full((per_rank * tp_size - vocab, hidden), 50.0, dtype=weight.dtype)
    padded = torch.cat([weight, pad_rows])
    shards = []
    for t in range(tp_size):
        start = t * per_rank
        num_pad = min(per_rank, max(0, start + per_rank - vocab))
        shards.append(_Shard(padded[start : start + per_rank].clone(), start, tp_size, num_pad))
    return shards


class _Group:
    def __init__(self, world_size, rank_in_group, all_gather=None):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self._all_gather = all_gather

    def all_gather(self, t, dim=0):
        return self._all_gather(t, dim)


def _patch_groups(monkeypatch, dp, tp, d, t, dp_gather, ep_gather, ep_rank=None):
    ep_rank = d * tp + t if ep_rank is None else ep_rank
    monkeypatch.setattr(parallel_state, "get_dp_group", lambda: _Group(dp, d, dp_gather))
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: _Group(tp, t))
    monkeypatch.setattr(
        parallel_state, "get_ep_group", lambda: _Group(dp * tp, ep_rank, ep_gather)
    )


def _vp_heads(monkeypatch, shards, dp, tp):
    heads = {}
    for d in range(dp):
        for t in range(tp):
            _patch_groups(monkeypatch, dp, tp, d, t, None, None)
            owner = gtt.vocab_parallel_draft_owner(_Owner(shards[t]))
            assert isinstance(owner.lm_head, gtt._VocabParallelDraftHead)
            heads[(d, t)] = owner
    return heads


def _run_vp(monkeypatch, weight, hiddens, dp, tp):
    """Every rank's pass with the collectives played back: phase 1 records each
    rank's (value, index) pairs, phase 2 hands each rank the EP-order gather.
    Returns {(d, t): tokens} and the per-rank collective log."""
    shards = _tp_shards(weight, tp, dp)
    owners = _vp_heads(monkeypatch, shards, dp, tp)
    all_rows = torch.cat(hiddens, dim=0)
    pairs = {}
    log = {}

    def run(d, t, ep_gather):
        calls = []

        def dp_gather(x, dim):
            calls.append(("dp", tuple(x.shape), x.dtype, dim))
            assert torch.equal(x, hiddens[d])
            return all_rows

        def ep_wrap(x, dim):
            calls.append(("ep", tuple(x.shape), x.dtype, dim))
            return ep_gather(x, dim)

        _patch_groups(monkeypatch, dp, tp, d, t, dp_gather, ep_wrap)
        owner = owners[(d, t)]
        out = owner.logits_processor.get_top_tokens(owner.lm_head, hiddens[d])
        log[(d, t)] = calls
        return out

    for d in range(dp):
        for t in range(tp):

            def record(x, dim, key=(d, t)):
                pairs[key] = x
                return x.repeat(dp * tp, 1)

            run(d, t, record)
    order = [pairs[(d, t)] for d in range(dp) for t in range(tp)]  # EP rank order
    tokens = {}
    for d in range(dp):
        for t in range(tp):
            tokens[(d, t)] = run(d, t, lambda x, dim: torch.cat(order, dim=0))
    return tokens, log


def _full_argmax(weight, hidden):
    return torch.argmax(torch.nn.functional.linear(hidden, weight), dim=-1)


def _pair_path(weight, hidden, tp, dp, monkeypatch):
    """Today's per-TP-group BF16 head, the TP all-gather played back."""
    from vllm.model_executor.layers import logits_processor as lp_module

    shards = _tp_shards(weight, tp, dp)
    recorded = []
    monkeypatch.setattr(
        lp_module,
        "tensor_model_parallel_all_gather",
        lambda x, dim=-1: (recorded.append(x), x.repeat(tp, 1))[1],
    )
    for s in shards:
        _Owner(s).logits_processor.get_top_tokens(s, hidden)
    monkeypatch.setattr(
        lp_module, "tensor_model_parallel_all_gather", lambda x, dim=-1: torch.cat(recorded, 0)
    )
    return _Owner(shards[0]).logits_processor.get_top_tokens(shards[0], hidden)


@pytest.mark.parametrize("dp,tp", [(4, 2), (2, 2), (4, 1)])
@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("vocab", [301, 256])
def test_matches_full_logits_argmax_and_pair_path(monkeypatch, dp, tp, rows, vocab):
    gen = torch.Generator().manual_seed(dp * 100 + tp * 10 + rows + vocab)
    weight = torch.randn(vocab, 32, generator=gen).to(torch.bfloat16)
    hiddens = [torch.randn(rows, 32, generator=gen).to(torch.bfloat16) for _ in range(dp)]

    tokens, _ = _run_vp(monkeypatch, weight, hiddens, dp, tp)

    for d in range(dp):
        ref = _full_argmax(weight, hiddens[d])
        for t in range(tp):
            got = tokens[(d, t)]
            assert got.dtype == torch.int64 and got.shape == (rows,)
            assert torch.equal(got, ref), (d, t)
        if tp > 1:
            assert torch.equal(_pair_path(weight, hiddens[d], tp, dp, monkeypatch), ref)


def test_ties_resolve_to_the_lowest_token_id_across_slices(monkeypatch):
    # dp 4, tp 2, 64 tokens -> 8 slices of 8. Slice k (vocab order) lives on
    # tp = k // 4, dp = k % 4, so EP order (dp-major) differs from vocab order.
    dp, tp = 4, 2
    weight = torch.zeros(64, 4, dtype=torch.bfloat16)
    weight[[9, 41], 0] = 2.0  # slice 1 (ep rank 2) vs slice 5 (ep rank 3)
    weight[[50, 17], 1] = 2.0  # slice 6 vs slice 2: lowest is 17
    weight[[33, 34], 2] = 2.0  # inside slice 4
    hiddens = [
        torch.eye(4, dtype=torch.bfloat16)[[0, 1]],
        torch.eye(4, dtype=torch.bfloat16)[[2, 3]],  # row 3: every logit 0
        torch.eye(4, dtype=torch.bfloat16)[[1, 0]],
        torch.eye(4, dtype=torch.bfloat16)[[3, 2]],
    ]
    tokens, _ = _run_vp(monkeypatch, weight, hiddens, dp, tp)
    expect = {0: [9, 17], 1: [33, 0], 2: [17, 9], 3: [0, 33]}
    for (d, _t), got in tokens.items():
        assert got.tolist() == expect[d]
        assert torch.equal(got, _full_argmax(weight, hiddens[d]))


def test_negative_and_positive_zero_tie(monkeypatch):
    dp, tp = 2, 2
    weight = torch.full((16, 2), -1.0, dtype=torch.bfloat16)
    weight[:, 1] = -2.0
    weight[9, 1] = torch.tensor(-0.0, dtype=torch.bfloat16)  # slice 2 (tp 1)
    weight[13, 1] = torch.tensor(0.0, dtype=torch.bfloat16)  # slice 3
    hiddens = [torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16)] * dp
    tokens, _ = _run_vp(monkeypatch, weight, hiddens, dp, tp)
    assert all(v.tolist() == [9] for v in tokens.values())


def test_padding_rows_are_masked(monkeypatch):
    # vocab 301 at dp 4 x tp 2: 304 rows, the last 3 are padding with logits
    # far above every real token.
    dp, tp = 4, 2
    weight = torch.zeros(301, 8, dtype=torch.bfloat16)
    weight[300, 0] = 1.0
    hiddens = [torch.ones(2, 8, dtype=torch.bfloat16)] * dp
    tokens, _ = _run_vp(monkeypatch, weight, hiddens, dp, tp)
    assert all(v.tolist() == [300, 300] for v in tokens.values())


def test_slices_cover_the_vocab_once_from_the_own_tp_shard(monkeypatch):
    dp, tp = 4, 2
    weight = torch.randn(301, 8).to(torch.bfloat16)
    shards = _tp_shards(weight, tp, dp)
    heads = _vp_heads(monkeypatch, shards, dp, tp)
    starts = sorted(int(o.lm_head.org_vocab_start_index) for o in heads.values())
    n = shards[0].weight.shape[0] // dp
    assert starts == [k * n for k in range(dp * tp)]
    for (d, t), o in heads.items():
        h = o.lm_head
        assert int(h.org_vocab_start_index) == t * shards[0].weight.shape[0] + d * n
        assert torch.equal(h.weight, shards[t].weight[d * n : (d + 1) * n])
        assert h.weight.dtype == torch.bfloat16 and h._quant_desc == "bf16"
        # Slice k of the EP-order gather is the rank that owns vocab slice k.
        order = h.vp_slice_order.tolist()
        assert order == [(k % dp) * tp + k // dp for k in range(dp * tp)]
    pads = {k: int(o.lm_head.num_org_vocab_padding) for k, o in heads.items()}
    assert sum(pads.values()) == dp * tp * n - 301
    assert pads[(dp - 1, tp - 1)] == 3


@pytest.mark.parametrize("busy_rows", [[1, 0, 0, 0], [3, 3, 3, 3]])
def test_every_rank_issues_the_same_collectives(monkeypatch, busy_rows):
    # Under lockstep idle engines run the same program on zero (dummy) rows at
    # the same loop bucket; the collective sequence must not depend on the
    # rows' contents or on the rank.
    dp, tp, rows = 4, 2, 4
    weight = torch.randn(64, 8).to(torch.bfloat16)
    hiddens = []
    for d in range(dp):
        h = torch.zeros(rows, 8, dtype=torch.bfloat16)
        h[: busy_rows[d]] = torch.randn(busy_rows[d], 8).to(torch.bfloat16)
        hiddens.append(h)
    _, log = _run_vp(monkeypatch, weight, hiddens, dp, tp)
    seqs = {tuple(v) for v in log.values()}
    assert len(seqs) == 1
    (seq,) = seqs
    assert seq == (
        ("dp", (rows, 8), torch.bfloat16, 0),
        ("ep", (dp * rows, 2), torch.float32, 0),
    )


def test_owner_gating(monkeypatch, caplog):
    weight = torch.randn(64, 8).to(torch.bfloat16)
    shards = _tp_shards(weight, 2, 4)
    owner = _Owner(shards[0])

    _patch_groups(monkeypatch, 1, 2, 0, 0, None, None)  # one DP engine
    assert gtt.vocab_parallel_draft_owner(owner) is owner

    _patch_groups(monkeypatch, 4, 2, 1, 0, None, None, ep_rank=1)  # wrong EP layout
    with caplog.at_level(logging.WARNING):
        assert gtt.vocab_parallel_draft_owner(owner) is owner

    fp8 = _Owner(_tp_shards(weight, 2, 4)[0])
    fp8.lm_head._quant_mode = "fp8"
    _patch_groups(monkeypatch, 4, 2, 1, 0, None, None)
    assert gtt.vocab_parallel_draft_owner(fp8) is fp8

    vp = gtt.vocab_parallel_draft_owner(owner)
    assert isinstance(vp.lm_head, gtt._VocabParallelDraftHead)
    assert vp.logits_processor is owner.logits_processor


@pytest.mark.parametrize("value,expect", [(None, True), ("1", True), ("0", False), ("off", False)])
def test_env_default_on_and_opt_out(monkeypatch, value, expect):
    if value is None:
        monkeypatch.delenv("TPU_DRAFT_VOCAB_PARALLEL", raising=False)
    else:
        monkeypatch.setenv("TPU_DRAFT_VOCAB_PARALLEL", value)
    assert gtt.draft_vocab_parallel_enabled() is expect


def _proposer(draft, lockstep, sharded=True, replicated=False):
    return SimpleNamespace(
        draft_model=draft,
        runner=SimpleNamespace(_dp_lockstep_enabled=lambda: lockstep),
        _dp_lockstep_sharded=lambda: lockstep and sharded,
        _draft_replicated=replicated,
    )


class _Draft(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.lm_head = head
        self.logits_processor = _Processor()


@pytest.mark.parametrize(
    "lockstep,sharded,replicated,env,expect_vp",
    [
        (True, True, False, None, True),
        (True, True, False, "0", False),
        (False, True, False, None, False),
        (True, False, False, None, False),
        (True, True, True, None, False),
    ],
)
def test_eagle3_uses_the_vocab_parallel_head_only_under_lockstep(
    monkeypatch, lockstep, sharded, replicated, env, expect_vp
):
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    monkeypatch.setenv("TPU_GREEDY_TOP_TOKENS", "1")
    monkeypatch.delenv("TPU_DRAFT_LM_HEAD_QUANT", raising=False)
    if env is None:
        monkeypatch.delenv("TPU_DRAFT_VOCAB_PARALLEL", raising=False)
    else:
        monkeypatch.setenv("TPU_DRAFT_VOCAB_PARALLEL", env)
    shards = _tp_shards(torch.randn(64, 8).to(torch.bfloat16), 2, 4)
    _patch_groups(monkeypatch, 4, 2, 2, 1, None, None)
    draft = _Draft(shards[1])
    proposer = _proposer(draft, lockstep, sharded=sharded, replicated=replicated)
    Eagle3Proposer._init_draft_greedy_head(proposer)
    owner = proposer._draft_greedy_head_owner
    assert isinstance(owner.lm_head, gtt._VocabParallelDraftHead) is expect_vp
    if not expect_vp:
        assert owner.lm_head is draft.lm_head
