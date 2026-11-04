# SPDX-License-Identifier: Apache-2.0
"""An LM head stored [hidden, vocab shard] projects like vLLM's layout."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import vllm.model_executor.layers.vocab_parallel_embedding as vpe

from vllm_torchtpu.runner import utils as runner_utils


def _head(weight):
    head = vpe.ParallelLMHead.__new__(vpe.ParallelLMHead)
    torch.nn.Module.__init__(head)
    head.weight = torch.nn.Parameter(weight, requires_grad=False)
    head.quant_method = vpe.UnquantizedEmbeddingMethod()
    return head


def _model(head, embed_weight=None, head_dtype=None):
    embed = SimpleNamespace(weight=embed_weight)
    return SimpleNamespace(
        lm_head=head,
        logits_processor=SimpleNamespace(head_dtype=head_dtype),
        model=SimpleNamespace(embed_tokens=embed),
    )


def _relayout(model):
    with (
        patch.object(runner_utils, "_to_row_major", lambda t: t.clone()),
        patch.object(runner_utils, "_install_entry_layout_pin") as pin,
    ):
        changed = runner_utils.relayout_lm_head(model)
    return changed, pin


@pytest.mark.parametrize("bias", [False, True])
def test_hidden_major_head_projects_like_the_vocab_major_head(bias):
    # Small integers: every sum is exact, so any accumulation order agrees.
    weight = torch.randint(-3, 4, (40, 16)).float()
    x = torch.randint(-3, 4, (2, 5, 16)).float()
    b = torch.randint(-3, 4, (40,)).float() if bias else None
    head = _head(weight.clone())
    want = head.quant_method.apply(head, x, b)
    changed, pin = _relayout(_model(head))
    assert changed
    pin.assert_called_once_with()
    assert head.weight.shape == (16, 40)
    assert torch.equal(head.quant_method.apply(head, x, b), want)
    with pytest.raises(RuntimeError):
        head.quant_method.embedding(head, torch.tensor([1]))


@pytest.mark.parametrize("case", ["tied", "fp32_head", "quantized"])
def test_head_kept_when_its_weight_is_read_another_way(case):
    weight = torch.randn(40, 16)
    head = _head(weight)
    if case == "quantized":
        head.quant_method = object()
    model = _model(
        head,
        embed_weight=head.weight if case == "tied" else None,
        head_dtype=torch.float64 if case == "fp32_head" else None,
    )
    changed, pin = _relayout(model)
    assert not changed
    pin.assert_not_called()
    assert head.weight.shape == (40, 16)
