# SPDX-License-Identifier: Apache-2.0
"""q38 L4 v5 composition (q38-l4-v5c with AGREE): DRAFT-2 (c) / EMB-REP-TGT's load-time agreement runs through
runner/l4_agree.py `gather` (the composition fix: AGREE removed draft_asm._gather, which draftemb imported).
CDK CPU pod only. V5COMP-BUILD (helper of L4-E).

Driven with draftemb's own harness (tests/q38_draftemb/test_draft_embed_repl.py: 8 simulated TP ranks, each its own
TPU-patched VocabParallelEmbedding shard; the CPU-group all_gather and the weight all-reduce emulated at
torch.distributed / parallel_state, which l4_agree.gather calls). Every load is counted at l4_agree.gather.

E1  VLLM_TPU_DRAFT_EMBED_REPL=1 on all 8 ranks, with and without VLLM_TPU_TARGET_EMBED_REPL=1: each rank's load
    agrees through exactly one l4_agree.gather call and binds the full table (and the target's, with the second
    flag), equal to the unsharded weight.
E2  one perturbed rank (its flag off, or its target flag different): every rank raises (fail-stop) after its one
    l4_agree.gather call, nothing bound.
E3  draft_asm has no `_gather` on this tree, and mtp's agreement names l4_agree.gather (the old import would fail).
Run: python -m pytest -q -p no:cacheprovider -p no:logging -s tests/q38_l4_v5c/test_v5c_emb_agree.py
"""

import os
import sys
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "q38_draftemb"))
import test_draft_embed_repl as de  # noqa: E402  (draftemb's harness: _weight, _shard, _emulate_*, TP, V, H)
from vllm_torchtpu.runner import l4_agree  # noqa: E402
from vllm_torchtpu.spec_decode import draft_asm  # noqa: E402

COUNTS: dict = {}


def _holder(shard, repl, tgt):
    m = nn.Module()
    m.embed_tokens = shard
    m.embed_repl = repl
    m.embed_repl_tgt = tgt
    return m


def _bind(holder):
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    return mtp.Qwen4ExpMTP.bind_replicated_embedding(NS(model=holder))


def _count_l4_gather(monkeypatch):
    calls = []
    real = l4_agree.gather

    def counted(vec):
        calls.append(list(vec))
        return real(vec)

    monkeypatch.setattr(l4_agree, "gather", counted)
    return calls


@pytest.mark.parametrize("tgt", [False, True])
def test_e1_eight_ranks_agree_through_l4_agree_gather(monkeypatch, tgt):
    weight = de._weight(11)
    shards = [de._shard(r, weight) for r in range(de.TP)]
    for rank in range(de.TP):
        g = _count_l4_gather(monkeypatch)
        calls = de._emulate_agreement(monkeypatch, [(1, int(tgt))] * de.TP, rank)
        seen = de._emulate_gather(monkeypatch, shards, rank)
        holder = _holder(shards[rank], True, tgt)
        assert _bind(holder) is True
        assert g == [[1, int(tgt)]] and calls == [rank] and seen == [rank], (rank, g, calls, seen)
        assert torch.equal(holder.embed_full, weight) and holder.embed_full.shape == (de.V, de.H)
        bound_tgt = getattr(shards[rank], "_q38_repl_table", None)
        assert (bound_tgt is holder.embed_full) if tgt else bound_tgt is None, rank
        if tgt:
            del shards[rank]._q38_repl_table  # the next case binds afresh
    COUNTS[f"e1_tgt{int(tgt)}"] = de.TP
    print(f"\nV5C_EMBAGREE E1 TP={de.TP} DRAFT_EMBED_REPL=1 TARGET_EMBED_REPL={int(tgt)}: every rank agreed through "
          f"one l4_agree.gather call and bound the table (target table bound: {tgt})")


@pytest.mark.parametrize("votes", [[(1, 0)] * 3 + [(0, 0)] + [(1, 0)] * 4, [(1, 1)] * 7 + [(1, 0)]])
def test_e2_a_perturbed_rank_stops_every_rank(monkeypatch, votes):
    weight = de._weight(12)
    shards = [de._shard(r, weight) for r in range(de.TP)]
    for rank in range(de.TP):
        g = _count_l4_gather(monkeypatch)
        calls = de._emulate_agreement(monkeypatch, votes, rank)
        seen = de._emulate_gather(monkeypatch, shards, rank)
        repl, tgt = (bool(x) for x in votes[rank])
        holder = _holder(shards[rank], repl, tgt)
        with pytest.raises(RuntimeError, match="across the TP ranks"):
            _bind(holder)
        assert len(g) == 1 and calls == [rank] and seen == [] and not hasattr(holder, "embed_full"), rank
    COUNTS["e2_cases"] = COUNTS.get("e2_cases", 0) + 1
    print(f"\nV5C_EMBAGREE E2 votes={votes}: all {de.TP} ranks fail-stop after one l4_agree.gather call, nothing bound")


def test_e3_the_agreement_names_l4_agree_gather():
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    assert not hasattr(draft_asm, "_gather")
    src = open(mtp.__file__).read()
    assert src.count("from vllm_torchtpu.runner.l4_agree import gather as _gather") == 1
    assert "draft_asm import _gather" not in src
    COUNTS["e3"] = 1


def test_zz_counts():
    print("\nV5C_EMBAGREE_COUNTS " + " ".join(f"{k}={v}" for k, v in sorted(COUNTS.items())))
