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
from vllm.model_executor.models.utils import PPMissingLayer

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


@pytest.mark.parametrize("has_own_embed_tokens", [None, True])
def test_draft_keeps_own_embed_when_target_embed_is_on_another_pp_stage(
    has_own_embed_tokens,
):
    # Only the first PP stage builds a DeepseekV2 target's embed_tokens.
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=PPMissingLayer()))
    draft_embed = object()
    draft = SimpleNamespace(model=SimpleNamespace(embed_tokens=draft_embed))
    if has_own_embed_tokens is not None:
        draft.has_own_embed_tokens = has_own_embed_tokens

    utils.maybe_share_embeddings(draft, target, tp_layout_matches=True)

    assert draft.model.embed_tokens is draft_embed


@pytest.mark.parametrize(
    ("draft_embed", "has_own_embed_tokens", "force_share"),
    [(None, None, False), (object(), None, True), (object(), False, False)],
    ids=["no-draft-embed", "force-share", "declares-no-own-embed"],
)
def test_draft_needing_target_embed_raises_when_it_is_on_another_pp_stage(
    draft_embed, has_own_embed_tokens, force_share
):
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=PPMissingLayer()))
    draft = SimpleNamespace(model=SimpleNamespace(embed_tokens=draft_embed))
    if has_own_embed_tokens is not None:
        draft.has_own_embed_tokens = has_own_embed_tokens

    with pytest.raises(RuntimeError, match="not on this pipeline stage"):
        utils.maybe_share_embeddings(
            draft, target, tp_layout_matches=True, force_share=force_share
        )


def test_handoff_seed_tokens_replaces_only_rows_with_a_published_send():
    from types import SimpleNamespace

    seed = torch.tensor([11, 22, 33], dtype=torch.int32)
    metadata = SimpleNamespace(
        reqs_to_send={
            "a": SimpleNamespace(next_prompt_token_id=7),
            "c": SimpleNamespace(next_prompt_token_id=None),
        }
    )
    out = utils.handoff_seed_tokens(seed, ["a", "b", "c"], metadata)
    assert out.tolist() == [7, 22, 33]
    assert seed.tolist() == [11, 22, 33]
    # No metadata, or no send with a token: the seed is returned as is.
    assert utils.handoff_seed_tokens(seed, ["a"], None) is seed
    assert (
        utils.handoff_seed_tokens(
            seed, ["b"], SimpleNamespace(reqs_to_send={"a": object()})
        )
        is seed
    )


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


def test_fit_rows_returns_slices_or_pads_the_leading_dim():
    t = torch.arange(8, dtype=torch.int32).view(4, 2)
    assert utils.fit_rows(t, 4) is t
    assert torch.equal(utils.fit_rows(t, 2), t[:2])
    out = utils.fit_rows(t, 6, value=-1)
    assert out.dtype == torch.int32 and out.shape == (6, 2)
    assert torch.equal(out[:4], t) and out[4:].eq(-1).all()


def _qwen35_mtp_config(arch="Qwen3_5MoeForConditionalGeneration", method="mtp"):
    from types import SimpleNamespace as NS

    return NS(
        model_config=NS(architectures=[arch]),
        speculative_config=None if method is None else NS(method=method),
    )


@pytest.mark.parametrize(
    ("arch", "method", "expected"),
    [
        ("Qwen3_5MoeForConditionalGeneration", "mtp", True),
        ("Qwen3_5MoeForConditionalGeneration", "eagle3", False),
        ("Qwen3_5MoeForConditionalGeneration", None, False),
        ("Qwen3NextForCausalLM", "mtp", False),
        ("DeepseekV3ForCausalLM", "mtp", False),
    ],
)
def test_qwen3_5_mtp_rules_follow_the_model_and_draft(
    monkeypatch, arch, method, expected
):
    """Greedy top tokens, fast token substitution and the fused draft loop
    run by rule only for a Qwen3.5 target with an MTP draft; every other
    configuration keeps main's opt-in env keys (default off)."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.spec_decode import utils

    for key in (
        "TPU_GREEDY_TOP_TOKENS",
        "VLLM_TPU_FAST_TOKEN_SUBSTITUTION",
        "VLLM_TPU_FUSED_DRAFT_LOOP",
    ):
        monkeypatch.setattr(envs, key, False)
    config = _qwen35_mtp_config(arch, method)
    assert utils.qwen3_5_mtp(config) is expected
    assert utils.greedy_top_tokens_enabled(config) is expected
    assert utils.fast_token_substitution_enabled(config) is expected
    assert (
        utils.fused_draft_loop_enabled(config, draft_kv_shared=False, dp_lockstep=False)
        is expected
    )


def test_qwen3_5_fused_draft_loop_rule_exceptions(monkeypatch):
    """The fused loop rule stays off for a KV-sharing draft and under EP-DP
    lockstep; main's key still turns it on anywhere."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.spec_decode import utils

    monkeypatch.setattr(envs, "VLLM_TPU_FUSED_DRAFT_LOOP", False)
    config = _qwen35_mtp_config()
    on = utils.fused_draft_loop_enabled
    assert on(config, draft_kv_shared=True, dp_lockstep=False) is False
    assert on(config, draft_kv_shared=False, dp_lockstep=True) is False
    monkeypatch.setattr(envs, "VLLM_TPU_FUSED_DRAFT_LOOP", True)
    other = _qwen35_mtp_config("Qwen3NextForCausalLM")
    assert on(other, draft_kv_shared=False, dp_lockstep=False) is True
    monkeypatch.setattr(envs, "TPU_GREEDY_TOP_TOKENS", True)
    assert utils.greedy_top_tokens_enabled(other) is True
