# SPDX-License-Identifier: Apache-2.0
"""The MTP draft's LM head runs at the checkpoint's BF16 precision.

Qwen3.5-397B-A17B-FP8 ships ``lm_head`` in BF16, and its MTP draft has no head
of its own: it borrows the target's. ``TPU_QUANTIZE_SHARED_LM_HEAD`` (default
on) quantizes the target's head to W8A8 FP8 in place, which used to leave the
draft sharing an FP8 head, and the draft's greedy head defaulted to an FP8
proxy on top. These tests pin the compliant behaviour: the target's head is
FP8, the draft gets a BF16 twin holding the checkpoint weight, and the draft's
greedy head uses that twin with no quantized proxy unless explicitly asked.
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
    """Same class name as vLLM's, which is what the post-load hook matches."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.weight = torch.nn.Parameter(
            torch.randn(VOCAB, HIDDEN).to(torch.bfloat16), requires_grad=False
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
    # An unquantized target (TPU_QUANTIZE_SHARED_LM_HEAD=0): nothing changes.
    target = _target_with_head()
    draft = _Draft()
    utils.maybe_share_lm_head(
        draft, target, draft_replicated=False, tp_layout_matches=True
    )
    assert draft.lm_head is target.lm_head


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


def _fake_qwen3_next_model(head: torch.nn.Module) -> torch.nn.Module:
    from vllm.model_executor.models import qwen3_next

    marker = qwen3_next.Qwen3NextModel.__new__(qwen3_next.Qwen3NextModel)
    torch.nn.Module.__init__(marker)
    model = torch.nn.Module()
    model.model = marker
    model.lm_head = head
    return model


@pytest.mark.parametrize("env", [None, "1"])
def test_post_load_hook_keeps_a_bf16_twin_by_default(monkeypatch, env):
    from vllm_torchtpu.models.vllm import qwen3_next_patch

    if env is None:
        monkeypatch.delenv("TPU_DRAFT_LM_HEAD_BF16", raising=False)
    else:
        monkeypatch.setenv("TPU_DRAFT_LM_HEAD_BF16", env)
    monkeypatch.delenv("TPU_QUANTIZE_SHARED_LM_HEAD", raising=False)
    monkeypatch.setattr(
        qwen3_next_patch, "_speculative_decoding_configured", lambda: True
    )
    head = ParallelLMHead()
    w = head.weight

    qwen3_next_patch.setup_qwen3_next_v6_after_loading(_fake_qwen3_next_model(head))

    assert head._quant_mode == "fp8"  # target stays FP8
    twin = gtt.draft_bf16_head_of(head)
    assert twin is not None and twin.weight is w

    # End to end: the draft that borrows the head gets the BF16 twin.
    draft = _Draft()
    utils.maybe_share_lm_head(
        draft, SimpleNamespace(model=SimpleNamespace(), lm_head=head),
        draft_replicated=False, tp_layout_matches=True,
    )
    assert draft.lm_head is twin
    assert gtt.head_precision(draft.lm_head) == "bf16"


def test_post_load_hook_opt_out_and_no_draft(monkeypatch):
    from vllm_torchtpu.models.vllm import qwen3_next_patch

    monkeypatch.delenv("TPU_QUANTIZE_SHARED_LM_HEAD", raising=False)
    # Opt-out: the draft would share the FP8 head (the old behaviour).
    monkeypatch.setenv("TPU_DRAFT_LM_HEAD_BF16", "0")
    monkeypatch.setattr(
        qwen3_next_patch, "_speculative_decoding_configured", lambda: True
    )
    head = ParallelLMHead()
    qwen3_next_patch.setup_qwen3_next_v6_after_loading(_fake_qwen3_next_model(head))
    assert head._quant_mode == "fp8" and gtt.draft_bf16_head_of(head) is None

    # No speculative decoding: no twin, so no extra memory.
    monkeypatch.delenv("TPU_DRAFT_LM_HEAD_BF16", raising=False)
    monkeypatch.setattr(
        qwen3_next_patch, "_speculative_decoding_configured", lambda: False
    )
    head = ParallelLMHead()
    qwen3_next_patch.setup_qwen3_next_v6_after_loading(_fake_qwen3_next_model(head))
    assert head._quant_mode == "fp8" and gtt.draft_bf16_head_of(head) is None


def test_speculative_decoding_configured_reads_the_vllm_config(monkeypatch):
    import vllm.config

    from vllm_torchtpu.models.vllm import qwen3_next_patch

    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(speculative_config=object()),
    )
    assert qwen3_next_patch._speculative_decoding_configured() is True
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(speculative_config=None),
    )
    assert qwen3_next_patch._speculative_decoding_configured() is False


def _init_greedy_head(draft: torch.nn.Module):
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    proposer = SimpleNamespace(
        draft_model=draft,
        runner=SimpleNamespace(_dp_lockstep_enabled=lambda: False),
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
    monkeypatch.delenv("TPU_DRAFT_LM_HEAD_QUANT", raising=False)
    draft, target = _shared_draft()

    owner = _init_greedy_head(draft)

    assert owner is not None
    assert owner.lm_head is draft.lm_head  # the twin itself, no proxy
    assert not isinstance(owner.lm_head, gtt._QuantizedLMHeadProxy)
    assert getattr(owner.lm_head, "_quant_desc", "bf16") == "bf16"
    assert owner.lm_head.weight.dtype == torch.bfloat16

    # TPU_GREEDY_TOP_TOKENS still works: the argmax equals the BF16 logits'.
    x = torch.randn(5, HIDDEN).to(torch.bfloat16)
    tokens = owner.logits_processor.get_top_tokens(owner.lm_head, x)
    ref = (x @ draft.lm_head.weight.t()).argmax(dim=-1)
    assert torch.equal(tokens, ref)


@pytest.mark.parametrize("value", ["bf16", "none"])
def test_draft_greedy_head_explicit_bf16_has_no_proxy(monkeypatch, value):
    monkeypatch.setenv("TPU_GREEDY_TOP_TOKENS", "1")
    monkeypatch.setenv("TPU_DRAFT_LM_HEAD_QUANT", value)
    draft, _ = _shared_draft()
    owner = _init_greedy_head(draft)
    assert owner.lm_head is draft.lm_head


def test_draft_greedy_head_quant_is_an_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("TPU_GREEDY_TOP_TOKENS", "1")
    monkeypatch.setenv("TPU_DRAFT_LM_HEAD_QUANT", "fp8")
    draft, _ = _shared_draft()
    owner = _init_greedy_head(draft)
    assert isinstance(owner.lm_head, gtt._QuantizedLMHeadProxy)
    assert owner.lm_head._quant_desc == "w8a8"
