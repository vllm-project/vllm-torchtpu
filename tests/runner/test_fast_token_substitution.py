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
    assert envs.environment_variables["VLLM_TPU_FAST_TOKEN_SUBSTITUTION"](
    ) is True

    monkeypatch.setenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", "true")
    assert envs.environment_variables["VLLM_TPU_FAST_TOKEN_SUBSTITUTION"](
    ) is True

    monkeypatch.setenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", "0")
    assert envs.environment_variables["VLLM_TPU_FAST_TOKEN_SUBSTITUTION"](
    ) is False

    monkeypatch.setenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", "false")
    assert envs.environment_variables["VLLM_TPU_FAST_TOKEN_SUBSTITUTION"](
    ) is False


def test_substitution_disabled_by_default():
    """When _fast_token_substitution is False, calls _substitute_placeholder_token."""
    runner = SimpleNamespace(
        _fast_token_substitution=False,
        device=torch.device("cpu"),
        _pre_async_results=SimpleNamespace(next_tokens_tpu=torch.tensor(
            [101, 102, 103, 104], dtype=torch.int64)),
    )
    input_ids = torch.tensor([-1, -1, -1, -1], dtype=torch.int32)
    cur_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    pre_indices = np.array([0, 1, 2, 3], dtype=np.int32)

    with patch("vllm_torchtpu.runner.tpu_runner._substitute_placeholder_token"
               ) as mock_kernel:
        mock_kernel.return_value = torch.tensor([101, 102, 103, 104],
                                                dtype=torch.int32)
        out = TPUModelRunner._apply_async_token_substitution(
            runner, input_ids, cur_indices, pre_indices)
        assert mock_kernel.called
        assert torch.equal(
            out, torch.tensor([101, 102, 103, 104], dtype=torch.int32))


def test_substitution_enabled_fast_path():
    """When _fast_token_substitution is True, direct slicing bypasses kernel."""
    runner = SimpleNamespace(
        _fast_token_substitution=True,
        device=torch.device("cpu"),
        _pre_async_results=SimpleNamespace(next_tokens_tpu=torch.tensor(
            [201, 202, 203, 204], dtype=torch.int64)),
    )
    input_ids = torch.tensor([-1, -1, -1, -1], dtype=torch.int32)
    cur_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    pre_indices = np.array([0, 1, 2, 3], dtype=np.int32)

    with patch("vllm_torchtpu.runner.tpu_runner._substitute_placeholder_token"
               ) as mock_kernel:
        out = TPUModelRunner._apply_async_token_substitution(
            runner, input_ids, cur_indices, pre_indices)
        # Direct slice returns next_tokens_tpu[:4] without calling kernel
        assert not mock_kernel.called
        assert torch.equal(
            out, torch.tensor([201, 202, 203, 204], dtype=torch.int32))
