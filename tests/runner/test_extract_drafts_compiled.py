# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""VLLM_TPU_EXTRACT_DRAFTS_COMPILED and VLLM_TPU_SUBTRACT_REJECTED_WARM_NO_GRAD:
the verified-draft gather runs as a static-shape compiled program warmed for
every shape serving reaches, and the rejected-count warmup matches serving's
grad mode."""

import contextlib
import random
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _extract_drafts_shapes,
    _get_padded_token_len,
)
from vllm_torchtpu.runner.tpu_runner_async_output import extract_draft_token_ids

PADDINGS = [4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 384, 512, 1024, 2048, 3072,
            4096, 8192]  # fmt: skip


def _max_logits_len(max_num_reqs, k, paddings):
    return _get_padded_token_len(paddings, min(max_num_reqs * (k + 1), paddings[-1]))


def test_shapes_cover_every_verify_step():
    """Every (token bucket, padded logits length) a verify step produces is
    warmed: random mixes of decode rows (1 + drafts tokens, 1 + drafts logits
    rows) and prefill chunks (any token count, one logits row), as
    get_spec_decode_metadata pads them."""
    max_num_reqs, k = 32, 3
    max_tokens = PADDINGS[-1]
    shapes = set(_extract_drafts_shapes(PADDINGS, _max_logits_len(32, 3, PADDINGS)))
    rng = random.Random(0)
    seen = set()
    for _ in range(20000):
        num_reqs = rng.randint(1, max_num_reqs)
        tokens = logits = 0
        for _ in range(num_reqs):
            if rng.random() < 0.2:
                chunk = rng.randint(1, max(1, max_tokens // num_reqs))
                tokens, logits = tokens + chunk, logits + 1
            else:
                drafts = rng.randint(0, k)
                tokens, logits = tokens + 1 + drafts, logits + 1 + drafts
        if tokens > max_tokens:
            continue
        pair = (
            _get_padded_token_len(PADDINGS, tokens),
            _get_padded_token_len(PADDINGS, logits),
        )
        assert pair in shapes, pair
        seen.add(pair)
    # Nothing outside the reachable set: logits rows never exceed the tokens
    # nor max_num_reqs * (1 + K).
    assert all(m <= n and m <= 128 for n, m in shapes)
    assert len(shapes) == 117
    assert len(seen) > 60  # the sweep reaches most of the set


def test_compiled_extract_matches_eager_and_strips_dynamic_stamp(monkeypatch):
    """Same values as the eager gather, and a forward-stamped (mark_dynamic)
    input reaches the static region as a stamp-free alias."""
    calls = []

    def fake_compile(fn):
        def run(*args):
            calls.append([getattr(a, "_dynamo_dynamic_indices", None) for a in args])
            return fn(*args)

        return run

    monkeypatch.setattr(tpu_runner, "_tpu_compile", fake_compile)
    monkeypatch.setattr(tpu_runner, "_extract_drafts_fn", None)
    gen = torch.Generator().manual_seed(0)
    for n, m in ((8, 8), (8192, 128), (384, 24), (16, 4)):
        ids = torch.randint(0, 1 << 20, (n,), generator=gen, dtype=torch.int32)
        torch._dynamo.mark_dynamic(ids, 0)
        logits_idx = torch.randint(0, n, (m,), generator=gen, dtype=torch.int32)
        target_idx = torch.randint(0, m - 1, (m,), generator=gen, dtype=torch.int32)
        want = extract_draft_token_ids(ids, logits_idx, target_idx)
        got = tpu_runner._extract_drafts_compiled(ids, logits_idx, target_idx)
        assert got.dtype == want.dtype and torch.equal(got, want)
        assert calls[-1][0] is None
    monkeypatch.setattr(tpu_runner, "_extract_drafts_fn", None)


def _fake_runner(k=3, max_num_reqs=32):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=k),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        max_num_reqs=max_num_reqs,
        num_tokens_paddings=list(PADDINGS),
        num_reqs_max_model_len=max_num_reqs,
        num_reqs_most_model_len=None,
        device=torch.device("cpu"),
        _precompile_timed=lambda name: contextlib.nullcontext(),
    )


@pytest.mark.parametrize("flag", ["0", "1"])
def test_precompile_extract_drafts_warms_all_shapes_under_no_grad(monkeypatch, flag):
    monkeypatch.setenv("VLLM_TPU_EXTRACT_DRAFTS_COMPILED", flag)
    warmed = []

    def record(ids, logits_idx, target_idx):
        warmed.append(
            (ids.shape[0], logits_idx.shape[0], target_idx.shape[0],
             ids.dtype, logits_idx.dtype, torch.is_grad_enabled())
        )  # fmt: skip
        return ids[:1]

    monkeypatch.setattr(tpu_runner, "_extract_drafts_compiled", record)
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a: None)
    TPUModelRunner._precompile_extract_drafts(_fake_runner())
    if flag == "0":
        assert warmed == []
        return
    want = _extract_drafts_shapes(PADDINGS, 128)
    assert [(n, m) for n, m, *_ in warmed] == want
    for n, m, t, ids_dtype, idx_dtype, grad in warmed:
        assert t == m and ids_dtype == torch.int32 and idx_dtype == torch.int32
        assert grad is False


@pytest.mark.parametrize("flag", ["0", "1"])
def test_subtract_rejected_warm_grad_mode(monkeypatch, flag):
    """With the flag, the rejected-count correction and its row pad are
    warmed under no_grad (execute_model's mode); without it, unchanged."""
    monkeypatch.setenv("VLLM_TPU_SUBTRACT_REJECTED_COMPILED", "1")
    monkeypatch.setenv("VLLM_TPU_SUBTRACT_REJECTED_PAD_ROWS", "1")
    monkeypatch.setenv("VLLM_TPU_SUBTRACT_REJECTED_WARM_NO_GRAD", flag)
    modes = []

    def pad(x, rows):
        modes.append(("pad", torch.is_grad_enabled()))
        return x

    def subtract(*args):
        modes.append(("sub", torch.is_grad_enabled()))
        return args[0]

    monkeypatch.setattr(tpu_runner, "_pad_rejected_rows_compiled", pad)
    monkeypatch.setattr(tpu_runner, "_subtract_num_rejected_tokens_compiled", subtract)
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a: None)
    with torch.enable_grad():
        TPUModelRunner._precompile_subtract_rejected(_fake_runner())
    assert {kind for kind, _ in modes} == {"pad", "sub"}
    assert all(grad is (flag == "0") for _, grad in modes)
