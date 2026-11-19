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
"""The MTP draft's LM head runs at the checkpoint's BF16 precision.

Qwen3.5-397B-A17B-FP8 ships ``lm_head`` in BF16, and its MTP draft has no head
of its own: it borrows the target's. The Qwen3.5 ``quantize_target_lm_head``
setting quantizes the target's head to W8A8 FP8 in place, which would leave
the draft sharing an FP8 head. These tests pin the behaviour with the
setting on: the target's head is FP8, the draft gets a BF16 twin holding the
checkpoint weight, and the draft's greedy head uses that twin; no quantized
draft head exists.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.runner import greedy_top_tokens as gtt
from vllm_torchtpu.spec_decode import utils

VOCAB, HIDDEN = 256, 128


class _UnquantizedMethod:
    """Stands in for vLLM's UnquantizedEmbeddingMethod: F.linear on the weight."""

    def apply(self, layer, x, bias=None):
        out = x @ layer.weight.t().to(x.dtype)
        return out if bias is None else out + bias


class ParallelLMHead(torch.nn.Module):
    """Stands in for vLLM's ParallelLMHead; the fixture below makes the
    post-load hook's exact-type check match this class."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.weight = torch.nn.Parameter(
            torch.randn(VOCAB, HIDDEN, generator=generator).to(torch.bfloat16),
            requires_grad=False,
        )
        self.tp_size = 1
        self.org_vocab_size = VOCAB
        self.embedding_dim = HIDDEN
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=0, num_org_vocab_padding=0
        )
        self.quant_method = _UnquantizedMethod()


class _Processor:
    """A LogitsProcessor stand-in that resolve_head_owner accepts."""

    scale = 1.0
    soft_cap = None

    def get_top_tokens(self, lm_head, hidden):
        return gtt.tpu_get_top_tokens(self, lm_head, hidden)

    def _apply_head(self, lm_head, hidden, bias=None):
        return lm_head.quant_method.apply(lm_head, hidden, bias)


class _Draft(torch.nn.Module):
    """A Qwen3.5-MTP-shaped draft: top-level lm_head + logits_processor."""

    def __init__(self) -> None:
        super().__init__()
        self.lm_head = ParallelLMHead()  # replaced by sharing
        self.logits_processor = _Processor()


def _target_with_head() -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(), lm_head=ParallelLMHead())


def _quantize_target_head_like_the_hook(head: torch.nn.Module) -> None:
    gtt.keep_bf16_head_for_draft(head)
    gtt.quantize_lm_head(head, mode="fp8", in_place=True)


def test_twin_keeps_the_bf16_weight_after_in_place_fp8():
    head = ParallelLMHead()
    w = head.weight
    method = head.quant_method

    _quantize_target_head_like_the_hook(head)
    twin = gtt.draft_bf16_head_of(head)

    # Target: FP8 in place, BF16 weight gone from the module.
    assert head._quant_mode == "fp8"
    assert "weight" not in head._parameters
    assert isinstance(head.quant_method, gtt._W8A8EmbeddingQuantMethod)
    # Draft twin: the very same BF16 tensor, the original method, no quant.
    assert twin is not None and twin is not head
    assert twin.weight is w and twin.weight.dtype == torch.bfloat16
    assert twin.quant_method is method
    assert twin._quant_mode is None and twin._quant_desc == "bf16"
    assert not hasattr(twin, "_w8a8_weight_kn")
    assert gtt.head_precision(twin) == "bf16"
    assert gtt.head_precision(head) == "w8a8"
    # The twin is not a submodule of the target head.
    assert twin not in list(head.modules())


def test_maybe_share_lm_head_binds_the_bf16_twin_not_the_fp8_head():
    target = _target_with_head()
    _quantize_target_head_like_the_hook(target.lm_head)
    draft = _Draft()

    utils.maybe_share_lm_head(
        draft, target, draft_replicated=False, tp_layout_matches=True
    )

    assert draft.lm_head is gtt.draft_bf16_head_of(target.lm_head)
    assert draft.lm_head is not target.lm_head
    assert draft.lm_head.weight.dtype == torch.bfloat16
    assert getattr(draft.lm_head, "_quant_mode", None) is None
    assert target.lm_head._quant_mode == "fp8"


def test_without_a_twin_the_draft_still_shares_the_target_head():
    # An unquantized target (quantize_target_lm_head off): nothing changes.
    target = _target_with_head()
    draft = _Draft()
    utils.maybe_share_lm_head(
        draft, target, draft_replicated=False, tp_layout_matches=True
    )
    assert draft.lm_head is target.lm_head


@pytest.mark.parametrize("materialize", [False, True])
def test_a_draft_on_another_tp_layout_releases_the_twin(monkeypatch, materialize):
    """A draft on another TP layout (a replicated MTP draft, or DFlash, which
    gathers a replicated copy from the twin) never binds the twin, so it is
    released."""
    target = _target_with_head()
    _quantize_target_head_like_the_hook(target.lm_head)
    twin = gtt.draft_bf16_head_of(target.lm_head)
    draft = _Draft()
    own = draft.lm_head
    copied = []
    monkeypatch.setattr(
        utils,
        "populate_draft_lm_head_from_target",
        lambda _draft, head: copied.append(head),
    )

    utils.maybe_share_lm_head(
        draft,
        target,
        draft_replicated=False,
        tp_layout_matches=False,
        materialize_if_mismatched=materialize,
    )

    assert draft.lm_head is own
    assert copied == ([twin] if materialize else [])
    assert gtt.draft_bf16_head_of(target.lm_head) is None


def test_a_draft_with_its_own_head_releases_the_twin():
    target = _target_with_head()
    _quantize_target_head_like_the_hook(target.lm_head)
    draft = _Draft()
    own = draft.lm_head
    draft.has_own_lm_head = True

    utils.maybe_share_lm_head(
        draft, target, draft_replicated=False, tp_layout_matches=True
    )

    assert draft.lm_head is own
    assert gtt.draft_bf16_head_of(target.lm_head) is None


def test_draft_head_logits_are_bf16_exact_target_logits_are_fp8():
    target = _target_with_head()
    w = target.lm_head.weight.detach().clone()
    _quantize_target_head_like_the_hook(target.lm_head)
    draft = _Draft()
    utils.maybe_share_lm_head(
        draft, target, draft_replicated=False, tp_layout_matches=True
    )
    x = torch.randn(4, HIDDEN).to(torch.bfloat16)
    ref = x @ w.t()

    draft_logits = draft.lm_head.quant_method.apply(draft.lm_head, x)
    target_logits = target.lm_head.quant_method.apply(target.lm_head, x)

    assert torch.equal(draft_logits, ref)
    assert not torch.equal(target_logits, ref)


@pytest.fixture(autouse=True)
def _stand_in_lm_head_class(monkeypatch):
    from vllm_torchtpu.models.vllm import qwen3_next_patch

    monkeypatch.setattr(qwen3_next_patch, "ParallelLMHead", ParallelLMHead)


def _fake_qwen3_next_model(head: torch.nn.Module) -> torch.nn.Module:
    from vllm.model_executor.models import qwen3_next

    marker = qwen3_next.Qwen3NextModel.__new__(qwen3_next.Qwen3NextModel)
    torch.nn.Module.__init__(marker)
    model = torch.nn.Module()
    model.model = marker
    model.lm_head = head
    return model


def test_post_load_hook_keeps_a_bf16_twin_by_default(monkeypatch):
    from vllm_torchtpu.models.vllm import qwen3_next_patch

    monkeypatch.setattr(qwen3_next_patch, "_quantize_target_lm_head", lambda: True)
    monkeypatch.setattr(qwen3_next_patch, "_draft_borrows_target_head", lambda: True)
    head = ParallelLMHead()
    w = head.weight

    qwen3_next_patch.setup_qwen3_next_v6_after_loading(_fake_qwen3_next_model(head))

    assert head._quant_mode == "fp8"  # target stays FP8
    twin = gtt.draft_bf16_head_of(head)
    assert twin is not None and twin.weight is w

    # End to end: the draft that borrows the head gets the BF16 twin.
    draft = _Draft()
    utils.maybe_share_lm_head(
        draft,
        SimpleNamespace(model=SimpleNamespace(), lm_head=head),
        draft_replicated=False,
        tp_layout_matches=True,
    )
    assert draft.lm_head is twin
    assert gtt.head_precision(draft.lm_head) == "bf16"


def test_post_load_hook_keeps_no_twin_without_a_draft(monkeypatch):
    from vllm_torchtpu.models.vllm import qwen3_next_patch

    monkeypatch.setattr(qwen3_next_patch, "_quantize_target_lm_head", lambda: True)
    # No speculative decoding: no twin, so no extra memory.
    monkeypatch.setattr(qwen3_next_patch, "_draft_borrows_target_head", lambda: False)
    head = ParallelLMHead()
    qwen3_next_patch.setup_qwen3_next_v6_after_loading(_fake_qwen3_next_model(head))
    assert head._quant_mode == "fp8" and gtt.draft_bf16_head_of(head) is None


@pytest.mark.parametrize(
    ("method", "borrows"),
    [
        ("mtp", True),
        ("eagle3", True),
        ("dflash", True),
        ("dspark", True),
        ("ngram", False),
        (None, False),
    ],
)
def test_only_a_draft_that_borrows_the_target_head_keeps_a_twin(
    monkeypatch, method, borrows
):
    """The BF16 twin is kept only for an EAGLE/MTP, DFlash or DSpark draft,
    the drafts that bind the target's head; an n-gram drafter or no drafter
    keeps none."""
    import vllm.config

    from vllm_torchtpu.models.vllm import qwen3_next_patch

    spec = (
        None
        if method is None
        else SimpleNamespace(
            method=method, use_eagle=lambda: method in ("eagle", "eagle3", "mtp")
        )
    )
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(speculative_config=spec),
    )
    assert qwen3_next_patch._draft_borrows_target_head() is borrows


def _init_greedy_head(draft: torch.nn.Module):
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    proposer = SimpleNamespace(
        draft_model=draft,
        vllm_config=None,
        runner=SimpleNamespace(_dp_lockstep_enabled=lambda: False),
        _dp_lockstep_sharded=lambda: False,
    )
    Eagle3Proposer._init_draft_greedy_head(proposer)
    return proposer._draft_greedy_head_owner


def _shared_draft() -> tuple[_Draft, SimpleNamespace]:
    target = _target_with_head()
    _quantize_target_head_like_the_hook(target.lm_head)
    draft = _Draft()
    utils.maybe_share_lm_head(
        draft, target, draft_replicated=False, tp_layout_matches=True
    )
    return draft, target


def test_draft_greedy_head_defaults_to_bf16_with_no_proxy(monkeypatch):
    monkeypatch.setenv("TPU_GREEDY_TOP_TOKENS", "1")
    draft, target = _shared_draft()

    owner = _init_greedy_head(draft)

    assert owner is not None
    assert owner.lm_head is draft.lm_head  # the twin itself, no proxy
    assert getattr(owner.lm_head, "_quant_desc", "bf16") == "bf16"
    assert owner.lm_head.weight.dtype == torch.bfloat16

    # TPU_GREEDY_TOP_TOKENS still works: the argmax equals the BF16 logits'.
    x = torch.randn(5, HIDDEN).to(torch.bfloat16)
    tokens = owner.logits_processor.get_top_tokens(owner.lm_head, x)
    ref = (x @ draft.lm_head.weight.t()).argmax(dim=-1)
    assert torch.equal(tokens, ref)


def _vocab_parallel_owner(base_head: torch.nn.Module):
    """The owner vocab_parallel_draft_owner builds at DP2 x TP1 on DP rank 1:
    a proxy holding a _VocabParallelDraftHead over the upper half of
    ``base_head``'s rows."""
    head = gtt._VocabParallelDraftHead(
        base_head, dp_rank=1, dp_size=2, tp_rank=0, tp_size=1
    )
    return gtt._DraftHeadOwnerProxy(_Processor(), head)


def test_a_vocab_parallel_draft_head_releases_the_bf16_twin():
    """Under EP-DP lockstep the vocab-parallel head clones its slice of the
    BF16 twin, so the twin is released: the target head drops it, and the
    draft's own reference now holds the vocab-parallel head. The head the
    draft's tokens come from keeps its own copy of the same rows."""
    draft, target = _shared_draft()
    twin = gtt.draft_bf16_head_of(target.lm_head)
    assert twin is not None and draft.lm_head is twin
    upper_rows = twin.weight.detach()[VOCAB // 2 :].clone()
    owner = _vocab_parallel_owner(twin)

    assert gtt.release_twin_behind_vocab_parallel_head(owner, draft, target.lm_head)

    assert gtt.draft_bf16_head_of(target.lm_head) is None
    assert draft.lm_head is owner.lm_head
    assert all(module is not twin for module in draft.modules())
    assert torch.equal(owner.lm_head.weight, upper_rows)
    assert owner.lm_head.weight.data_ptr() != twin.weight.data_ptr()


@pytest.mark.parametrize("owner_kind", ["fallback", "per_tp_group"])
def test_the_bf16_twin_stays_while_the_draft_reads_it(owner_kind):
    """Without a greedy head (compute_logits + argmax reads the draft's head)
    or with the per-TP-group head (the twin itself), the twin is kept."""
    draft, target = _shared_draft()
    twin = gtt.draft_bf16_head_of(target.lm_head)
    owner = None if owner_kind == "fallback" else gtt.resolve_head_owner(draft, False)

    assert not gtt.release_twin_behind_vocab_parallel_head(owner, draft, target.lm_head)

    assert gtt.draft_bf16_head_of(target.lm_head) is twin
    assert draft.lm_head is twin


@pytest.mark.parametrize("vocab_parallel", [True, False])
def test_the_proposer_releases_the_twin_behind_a_vocab_parallel_head(vocab_parallel):
    """load_model releases the twin right after resolving the greedy head, and
    only when that head is the vocab-parallel one."""
    import ast
    import inspect
    import textwrap

    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    draft, target = _shared_draft()
    twin = gtt.draft_bf16_head_of(target.lm_head)
    proposer = SimpleNamespace(
        draft_model=draft,
        _draft_greedy_head_owner=_vocab_parallel_owner(twin)
        if vocab_parallel
        else None,
    )

    Eagle3Proposer._release_unread_draft_bf16_head(proposer, target)

    assert (gtt.draft_bf16_head_of(target.lm_head) is None) is vocab_parallel
    tree = ast.parse(textwrap.dedent(inspect.getsource(Eagle3Proposer.load_model)))
    lines = {
        node.func.attr: node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert (
        lines["_release_unread_draft_bf16_head"] == lines["_init_draft_greedy_head"] + 1
    )


def test_target_lm_head_stays_bf16_by_default(monkeypatch):
    """quantize_target_lm_head is off unless set: the target head keeps its
    BF16 weight and no twin is made, whatever the draft."""
    from vllm_torchtpu.models.vllm import qwen3_next_patch

    monkeypatch.setattr(qwen3_next_patch, "_draft_borrows_target_head", lambda: True)
    monkeypatch.setattr(qwen3_next_patch, "_quantize_target_lm_head", lambda: False)
    head = ParallelLMHead()
    w = head.weight
    qwen3_next_patch.setup_qwen3_next_v6_after_loading(_fake_qwen3_next_model(head))
    assert getattr(head, "_quant_mode", None) is None
    assert head.weight is w
    assert gtt.draft_bf16_head_of(head) is None


def _engine_config(architectures, section=None):
    from types import SimpleNamespace as NS

    return NS(
        model_config=NS(architectures=architectures),
        additional_config={} if section is None else {"qwen3_5": section},
    )


def test_quantize_target_lm_head_follows_the_qwen3_5_setting(monkeypatch):
    import vllm.config

    from vllm_torchtpu.models.vllm import qwen3_next_patch

    qwen35 = ["Qwen3_5MoeForConditionalGeneration"]
    cases = [
        (_engine_config(qwen35), False),
        (_engine_config(qwen35, {"quantize_target_lm_head": True}), True),
        (_engine_config(qwen35, {"quantize_target_lm_head": False}), False),
        # Qwen3-Next (and an MTP draft's own config) keep the BF16 head.
        (_engine_config(["Qwen3NextForCausalLM"]), False),
    ]
    for config, expected in cases:
        monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda c=config: c)
        assert qwen3_next_patch._quantize_target_lm_head() is expected
