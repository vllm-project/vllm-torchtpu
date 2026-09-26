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

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.spec_decode import utils


def test_own_vocab_weights_are_kept_when_tp_layout_matches():
    target_embed = object()
    target_head = object()
    draft_embed = object()
    draft_head = object()
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=target_embed),
        lm_head=target_head,
    )
    draft = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=draft_embed),
        lm_head=draft_head,
        has_own_embed_tokens=True,
        has_own_lm_head=True,
    )

    utils.maybe_share_embeddings(draft, target, tp_layout_matches=True)
    utils.maybe_share_lm_head(
        draft, target, draft_replicated=False, tp_layout_matches=True
    )

    assert draft.model.embed_tokens is draft_embed
    assert draft.lm_head is draft_head


class _SharedHeadLayer(torch.nn.Module):
    def __init__(self, hidden: int, vocab: int):
        super().__init__()
        self.shared_head = torch.nn.Module()
        self.shared_head.head = torch.nn.Linear(hidden, vocab, bias=False)


def _make_mtp_draft(num_layers: int = 1, hidden: int = 2, vocab: int = 3):
    """A DeepSeek/GLM-5.2-shaped MTP draft: no top-level lm_head, one head per
    MTP layer at model.layers.<N>.shared_head.head, LogitsProcessor on the
    inner model."""
    layers = torch.nn.ModuleDict(
        {str(78 + i): _SharedHeadLayer(hidden, vocab) for i in range(num_layers)}
    )
    inner = torch.nn.Module()
    inner.layers = layers
    inner.logits_processor = SimpleNamespace(
        _gather_logits=lambda logits: logits.clone()
    )
    draft = torch.nn.Module()
    draft.model = inner
    return draft


def _make_target(hidden: int = 2, vocab: int = 3):
    return SimpleNamespace(lm_head=torch.nn.Linear(hidden, vocab, bias=False))


def test_mtp_shared_head_is_bound_to_target_lm_head():
    """G1: DeepSeekMTP keeps its head at model.layers.<N>.shared_head.head, so
    the plain lm_head attribute walk never reaches it."""
    draft = _make_mtp_draft(num_layers=2)
    target = _make_target()

    utils.maybe_share_lm_head(
        draft, target, draft_replicated=False, tp_layout_matches=True
    )

    heads = [sh.head for sh in utils.iter_mtp_shared_heads(draft)]
    assert len(heads) == 2
    assert all(h is target.lm_head for h in heads)
    # No dead top-level attribute is invented when shared heads were bound.
    assert not hasattr(draft, "lm_head")


def test_mtp_replicated_draft_overrides_inner_logits_processor():
    """G2: DeepSeekMTP's LogitsProcessor lives on the inner model, so the
    top-level getattr used to return None and skip the TP-gather override."""
    draft = _make_mtp_draft()
    target = _make_target()

    utils.maybe_share_lm_head(
        draft, target, draft_replicated=True, tp_layout_matches=True
    )

    logits = torch.randn(2, 3)
    assert draft.model.logits_processor._gather_logits(logits) is logits


def test_replicated_draft_without_logits_processor_raises():
    draft = _make_mtp_draft()
    del draft.model.logits_processor
    target = _make_target()

    with pytest.raises(RuntimeError, match="LogitsProcessor"):
        utils.maybe_share_lm_head(
            draft, target, draft_replicated=True, tp_layout_matches=True
        )


def test_top_level_lm_head_draft_is_unchanged():
    """Regression: eagle3 / Qwen3.5-MTP drafts have a top-level lm_head and no
    shared_head, so the shared-head walk must be a no-op for them."""
    draft = SimpleNamespace(
        model=SimpleNamespace(),
        lm_head=torch.nn.Linear(2, 3, bias=False),
        logits_processor=SimpleNamespace(_gather_logits=lambda logits: logits.clone()),
    )
    target = _make_target()

    utils.maybe_share_lm_head(
        draft, target, draft_replicated=True, tp_layout_matches=True
    )

    assert draft.lm_head is target.lm_head
    assert list(utils.iter_mtp_shared_heads(draft)) == []
    logits = torch.randn(2, 3)
    assert draft.logits_processor._gather_logits(logits) is logits
