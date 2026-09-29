# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Unit tests verifying standalone TPU runner device ops and padding."""

import torch

from vllm_torchtpu.llm_tpu_core.runner.device_ops import (
    compute_logits_from_hidden_states,
    compute_selected_logits,
    gather_logprobs,
    sample_from_logits,
    substitute_placeholder_token,
)
from vllm_torchtpu.llm_tpu_core.runner.padding import (
    get_num_slices_per_kv_cache_update_block,
    get_padded_num_kv_cache_update_slices,
    get_padded_num_reqs_with_upper_limit,
    get_padded_token_len,
    get_req_paddings,
    prev_power_of_2,
)


def test_runner_padding_math():
    paddings = get_req_paddings(min_req_size=8, max_req_size=64)
    assert paddings == [8, 16, 32, 64]
    assert get_padded_token_len(paddings, 12) == 16
    assert get_padded_num_reqs_with_upper_limit(10, 64) == 16
    assert prev_power_of_2(35) == 32
    assert get_padded_num_kv_cache_update_slices(100, 4, 16) == 14
    assert get_num_slices_per_kv_cache_update_block(1024 * 1024) == 16


def test_runner_device_ops_cpu():
    # Substitute placeholder
    input_ids = torch.tensor([10, 20, -1, -1], dtype=torch.int32)
    cur_indices = torch.tensor([2, 3, 0, 1], dtype=torch.int32)
    next_indices = torch.tensor([0, -1, -1, -1], dtype=torch.int32)
    next_tokens = torch.tensor([99, 100], dtype=torch.int32)
    updated = substitute_placeholder_token(
        input_ids, cur_indices, next_indices, next_tokens
    )
    assert updated[2].item() == 99

    # Logits and sampling
    h = torch.randn(2, 64)
    w = torch.randn(100, 64)
    logits = compute_logits_from_hidden_states(h, w)
    assert logits.shape == (2, 100)

    selected = compute_selected_logits(h, torch.tensor([1], dtype=torch.long), w)
    assert selected.shape == (1, 100)

    sampled = sample_from_logits(
        logits,
        torch.tensor([0.0, 0.0]),
        torch.zeros(2, 100),
        all_greedy=True,
    )
    assert sampled.shape == (2, 1)

    logprobs = gather_logprobs(logits, sampled, max_logprobs=3)
    assert logprobs.logprobs.shape == (2, 4)
