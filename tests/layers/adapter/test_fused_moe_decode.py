# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Token-cap dispatch of the fused-MoE-decode op's per-rank body: up to
FUSED_MOE_DECODE_KERNEL_MAX_TOKENS the kernel serves it; above, the GMM MoE
fallback. No TPU needed: both paths are stubs."""

import numpy as np
import pytest

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import fused_moe_decode as bridge

H = 8


def test_max_tokens_env(monkeypatch):
    monkeypatch.delenv("FUSED_MOE_DECODE_KERNEL", raising=False)
    monkeypatch.delenv("FUSED_MOE_DECODE_KERNEL_MAX_TOKENS", raising=False)
    assert envs.FUSED_MOE_DECODE_KERNEL is False
    assert envs.FUSED_MOE_DECODE_KERNEL_MAX_TOKENS == 256
    monkeypatch.setenv("FUSED_MOE_DECODE_KERNEL_MAX_TOKENS", "64")
    assert envs.FUSED_MOE_DECODE_KERNEL_MAX_TOKENS == 64


@pytest.mark.parametrize("expert_parallel", [False, True])
@pytest.mark.parametrize("tokens", [1, 256, 257, 1024])
def test_token_cap_dispatch(monkeypatch, tokens, expert_parallel):
    calls = []

    def kernel(rank, x, *args, expert_start, **kwargs):
        calls.append(("kernel", expert_start))
        return x

    def gmm(*args, topk):
        calls.append(("gmm", args[12]))
        return args[0]

    monkeypatch.setattr(bridge, "_kernel_body", kernel)
    monkeypatch.setattr(bridge, "_gmm_layer", gmm)
    rank = np.array([[3, 192]], np.int32)  # [[mesh index, first expert]]
    w = np.zeros((1, 1), np.float32)
    bridge._layer(
        rank,
        np.zeros((tokens, H), np.float32),
        *([w] * 9),
        np.zeros((tokens, 2), np.float32),
        np.zeros((tokens, 2), np.int32),
        n_ranks=8,
        topk=2,
        expert_parallel=expert_parallel,
        max_tokens=256,
    )
    path, start = calls[0]
    assert len(calls) == 1
    assert path == ("kernel" if tokens <= 256 else "gmm")
    assert start == (192 if expert_parallel else None)
