# SPDX-License-Identifier: Apache-2.0
"""Rejection precompile inputs must have the runtime tensor aliasing pattern."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_torchtpu.runner import tpu_runner


@pytest.mark.parametrize("synthetic_mode", [False, True])
def test_head_ids_warmup_keeps_draft_and_head_inputs_independent(
    monkeypatch, synthetic_mode
):
    rows, requests, drafts = 288, 48, 5
    output = torch.zeros(requests, dtype=torch.int32)
    sampler = Mock(return_value=output, synthetic_mode=synthetic_mode)
    sampler.greedy_from_head_ids.return_value = output
    sampler.greedy_from_token_ids.return_value = output
    runner = SimpleNamespace(
        _is_async_drafter=True,
        speculative_config=SimpleNamespace(num_speculative_tokens=drafts),
        rejection_sampler=sampler,
        num_tokens_paddings=[rows],
        num_reqs_paddings=[requests],
        max_num_reqs=requests,
        _fused_greedy_head=SimpleNamespace(rows=rows),
        _hidden_states_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        vocab_size=16,
        model_config=SimpleNamespace(get_hidden_size=lambda: 8),
        _precompile_timed=lambda _name: nullcontext(),
        _spec_verify_bucket_reachable=lambda _rows, _reqs: True,
        compute_selected_greedy_ids=Mock(return_value=output),
        compute_greedy_ids=Mock(return_value=output),
        spec_bonus_and_target_token_ids=Mock(return_value=(output, output)),
        spec_gather_bonus_and_target_logits=Mock(return_value=(output, output)),
        _greedy_sample=Mock(return_value=output),
    )
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda _value: None)

    tpu_runner.TPUModelRunner._precompile_rejection_sampler(runner)

    sampler.greedy_from_head_ids.assert_called_once()
    inputs = sampler.greedy_from_head_ids.call_args.kwargs
    head_ids, draft_ids = inputs["head_token_ids"], inputs["draft_token_ids"]
    # Dynamo guards both object aliasing and storage relationships.
    assert head_ids is not draft_ids
    assert head_ids.data_ptr() != draft_ids.data_ptr()
    assert head_ids.shape == draft_ids.shape == (rows,)
    assert head_ids.dtype == draft_ids.dtype == torch.int32
    assert head_ids.device == draft_ids.device
