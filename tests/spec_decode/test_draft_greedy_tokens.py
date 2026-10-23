# SPDX-License-Identifier: Apache-2.0
"""Draft greedy tokens with ``VLLM_TPU_MTP_DRAFT_TOP_TOKENS``.

The flag routes a vocabulary-parallel draft head through
``LogitsProcessor.get_top_tokens`` (per-rank argmax, pairs exchanged); vLLM's
own tests cover that reduction against the full-logits argmax. This checks
the routing: flag off, replicated drafts (overridden ``_gather_logits``) and
non-positive scales keep the full-logits argmax.
"""

import torch
from torch import nn

from vllm_torchtpu.spec_decode.utils import draft_greedy_tokens

FLAG = "VLLM_TPU_MTP_DRAFT_TOP_TOKENS"


class _Processor:
    scale = 1.0

    def __init__(self):
        self.calls = 0

    def get_top_tokens(self, head, hidden):
        self.calls += 1
        return (hidden @ head.weight.T).argmax(dim=-1)


class _Head(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.shard_indices = object()


class _Draft(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.lm_head = _Head(weight)
        self.logits_processor = _Processor()
        self.full = 0

    def compute_logits(self, hidden):
        self.full += 1
        return hidden @ self.lm_head.weight.T


def _draft_and_hidden():
    torch.manual_seed(0)
    weight = torch.randn(50, 16)
    hidden = torch.randn(6, 16)
    return _Draft(weight), hidden, (hidden @ weight.T).argmax(dim=-1)


def test_flag_routes_to_top_tokens(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    draft, hidden, expected = _draft_and_hidden()
    tokens = draft_greedy_tokens(draft, hidden)
    assert draft.logits_processor.calls == 1 and draft.full == 0
    assert tokens.dtype == torch.int32
    torch.testing.assert_close(tokens, expected.to(torch.int32))


def test_flag_off_keeps_full_logits(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    draft, hidden, expected = _draft_and_hidden()
    tokens = draft_greedy_tokens(draft, hidden)
    assert draft.logits_processor.calls == 0 and draft.full == 1
    torch.testing.assert_close(tokens, expected.to(torch.int32))


def test_replicated_or_negative_scale_keep_full_logits(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    draft, hidden, _ = _draft_and_hidden()
    draft.logits_processor._gather_logits = lambda logits: logits
    draft_greedy_tokens(draft, hidden)
    assert draft.logits_processor.calls == 0 and draft.full == 1
    draft, hidden, _ = _draft_and_hidden()
    draft.logits_processor.scale = -1.0
    draft_greedy_tokens(draft, hidden)
    assert draft.logits_processor.calls == 0 and draft.full == 1
