# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def test_env_flag_parsing(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", raising=False)
    assert envs.VLLM_TPU_FAST_TOKEN_SUBSTITUTION is False

    monkeypatch.setenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", "1")
    assert envs.environment_variables["VLLM_TPU_FAST_TOKEN_SUBSTITUTION"]() is True

    monkeypatch.setenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", "true")
    assert envs.environment_variables["VLLM_TPU_FAST_TOKEN_SUBSTITUTION"]() is True

    monkeypatch.setenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", "0")
    assert envs.environment_variables["VLLM_TPU_FAST_TOKEN_SUBSTITUTION"]() is False

    monkeypatch.setenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", "false")
    assert envs.environment_variables["VLLM_TPU_FAST_TOKEN_SUBSTITUTION"]() is False


def test_substitution_disabled_by_default():
    """When _fast_token_substitution is False, calls _substitute_placeholder_token."""
    runner = SimpleNamespace(
        _fast_token_substitution=False,
        device=torch.device("cpu"),
        _pre_async_results=SimpleNamespace(
            next_tokens_tpu=torch.tensor([101, 102, 103, 104], dtype=torch.int64)
        ),
    )
    input_ids = torch.tensor([-1, -1, -1, -1], dtype=torch.int32)
    cur_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    pre_indices = np.array([0, 1, 2, 3], dtype=np.int32)

    with patch(
        "vllm_torchtpu.runner.tpu_runner._substitute_placeholder_token"
    ) as mock_kernel:
        mock_kernel.return_value = torch.tensor([101, 102, 103, 104], dtype=torch.int32)
        out, drafts = TPUModelRunner._apply_async_token_substitution(
            runner, input_ids, cur_indices, pre_indices
        )
        assert drafts is None
        assert mock_kernel.called
        assert torch.equal(out, torch.tensor([101, 102, 103, 104], dtype=torch.int32))


def test_substitution_enabled_fast_path():
    """When _fast_token_substitution is True, direct slicing bypasses kernel."""
    runner = SimpleNamespace(
        _fast_token_substitution=True,
        device=torch.device("cpu"),
        _pre_async_results=SimpleNamespace(
            next_tokens_tpu=torch.tensor([201, 202, 203, 204], dtype=torch.int64)
        ),
    )
    input_ids = torch.tensor([-1, -1, -1, -1], dtype=torch.int32)
    cur_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    pre_indices = np.array([0, 1, 2, 3], dtype=np.int32)

    with patch(
        "vllm_torchtpu.runner.tpu_runner._substitute_placeholder_token"
    ) as mock_kernel:
        out, drafts = TPUModelRunner._apply_async_token_substitution(
            runner, input_ids, cur_indices, pre_indices
        )
        assert drafts is None
        # Direct slice returns next_tokens_tpu[:4] without calling kernel
        assert not mock_kernel.called
        assert torch.equal(out, torch.tensor([201, 202, 203, 204], dtype=torch.int32))


def test_substitution_gathers_drafts_in_the_same_program():
    """With draft rows as long as the input bucket, the compiled path returns
    the substituted ids and the drafts gathered from them, from one program."""
    from vllm_torchtpu.runner.tpu_runner import _substitute_placeholder
    from vllm_torchtpu.runner.tpu_runner_async_output import extract_draft_token_ids

    next_tokens = torch.tensor([201, 202, 203, 204, 205, 206], dtype=torch.int64)
    logits_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    target_indices = torch.tensor([0, 1, 2, 0], dtype=torch.int32)
    runner = SimpleNamespace(
        _fast_token_substitution=False,
        device=torch.device("cpu"),
        _pre_async_results=SimpleNamespace(next_tokens_tpu=next_tokens),
        _unfenced_h2d_sources=[],
        _registered_h2d_sources=set(),
    )
    cur_indices = np.array([0, 1, 2], dtype=np.int32)
    pre_indices = np.array([3, 4, 5], dtype=np.int32)

    def program(ids, cur, pre, nxt, li, ti):
        ids = _substitute_placeholder(ids, cur, pre, nxt)
        return ids, extract_draft_token_ids(ids, li, ti)

    with patch(
        "vllm_torchtpu.runner.tpu_runner._substitute_placeholder_token_and_drafts",
        side_effect=program,
    ) as mock_kernel:
        out, drafts = TPUModelRunner._apply_async_token_substitution(
            runner,
            torch.full((4,), -1, dtype=torch.int32),
            cur_indices,
            pre_indices,
            (logits_indices, target_indices),
        )
    assert mock_kernel.called
    assert out.tolist() == [204, 205, 206, -1]
    assert drafts.tolist() == [205, 206, -1, 205]


def test_substitution_with_a_shorter_logits_bucket_gathers_separately():
    """A logits bucket shorter than the input keeps the substitution program
    and the separate draft gather (the combined program is not warmed there)."""
    runner = SimpleNamespace(
        _fast_token_substitution=False,
        device=torch.device("cpu"),
        _pre_async_results=SimpleNamespace(
            next_tokens_tpu=torch.tensor([201, 202, 203, 204], dtype=torch.int64)
        ),
        _unfenced_h2d_sources=[],
        _registered_h2d_sources=set(),
    )
    rows = (
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
    )
    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner._substitute_placeholder_token",
            return_value=torch.tensor([101, 102, 103, 104], dtype=torch.int32),
        ) as single,
        patch(
            "vllm_torchtpu.runner.tpu_runner._substitute_placeholder_token_and_drafts"
        ) as combined,
    ):
        out, drafts = TPUModelRunner._apply_async_token_substitution(
            runner,
            torch.full((4,), -1, dtype=torch.int32),
            np.array([0, 1], dtype=np.int32),
            np.array([2, 3], dtype=np.int32),
            rows,
        )
    assert single.called and not combined.called
    assert out.tolist() == [101, 102, 103, 104]
    assert drafts.tolist() == [102, 102]
