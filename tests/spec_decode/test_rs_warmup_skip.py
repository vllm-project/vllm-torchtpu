# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_RS_WARMUP_SKIP_SEPARATE_CANDIDATES: the rejection_sampler warmup
drops the separate candidates verify only when no step can run it, keeps
every other graph, and the served dispatch never requests a skipped graph."""

import types

import pytest
import torch

import vllm_torchtpu.runner.tpu_runner as runner
from vllm_torchtpu import envs
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    separate_candidates_verify_reachable,
)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("check", [0, 2000])
@pytest.mark.parametrize("dev", ["tpu", "cpu"])
def test_predicate(fused, check, dev):
    want_unreachable = fused and check == 0 and dev == "tpu"
    assert separate_candidates_verify_reachable(fused, check, dev) == (not want_unreachable)


class _RS:
    def __init__(self, calls, synthetic=True, with_tokens=True):
        self.calls = calls
        self.synthetic_mode = synthetic
        self.synthetic_conditional_rates = None
        self.with_tokens = with_tokens

    def __call__(self, **kw):
        key = ("rs", bool(kw.get("do_sampling", False)), int(kw.get("topk_candidates", 0) or 0))
        if self.with_tokens:
            key += (kw["draft_token_ids"].shape[0],)
        self.calls.append(key)
        return kw["draft_token_ids"]


def _fake(calls, dev="tpu"):
    rejection = _RS(calls)
    f = types.SimpleNamespace(
        _is_async_drafter=True,
        speculative_config=types.SimpleNamespace(num_speculative_tokens=3),
        rejection_sampler=rejection,
        num_tokens_paddings=[4, 8, 16, 32],
        num_reqs_paddings=[8],
        max_num_reqs=8,
        vocab_size=256,
        _hidden_states_dtype=torch.bfloat16,
        device="cpu",
        _greedy_head_owner=None,
        _candidate_bounds=None,
        _spec_verify_bucket_reachable=lambda t, r: True,
        spec_bonus_and_target_logits=lambda l, b, t: (b, l),
        spec_gather_bonus_and_target_logits=lambda l, b, t: (l, l),
        _topk_candidate_blocks=lambda: 2,
        _sampler_candidates_only=lambda c: c > 0,
        _verify_device_type=lambda: dev,
        _precompile_timed=lambda name: __import__("contextlib").nullcontext(),
    )
    f._separate_candidates_verify_reachable = types.MethodType(
        TPUModelRunner._separate_candidates_verify_reachable, f
    )
    f._separate_candidates_verify_warm_needed = types.MethodType(
        TPUModelRunner._separate_candidates_verify_warm_needed, f
    )
    return f


def _warm_keys(monkeypatch, skip, fused, check, dev="tpu"):
    monkeypatch.setattr(envs, "VLLM_TPU_RS_WARMUP_SKIP_SEPARATE_CANDIDATES", skip, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_VERIFY_FUSED", fused, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_VERIFY_FUSED_CHECK", check, raising=False)
    fused_calls = []
    monkeypatch.setattr(
        TPUModelRunner, "_precompile_verify_fused", lambda self, *a: fused_calls.append(a[0].shape[0]),
        raising=False,
    )
    monkeypatch.setattr(runner, "synchronize_tensors", lambda *a, **k: None)
    calls = []
    f = _fake(calls, dev)
    f._precompile_verify_fused = lambda *a: fused_calls.append(a[0].shape[0])
    TPUModelRunner._precompile_rejection_sampler(f)
    return set(calls), fused_calls


# Served combos: c32/c16 render (fused on, check off) and the validation arm
# (fused on, check 2000), plus fused off.
@pytest.mark.parametrize("fused,check", [(True, 0), (True, 2000), (False, 0)])
def test_warm_key_sets(monkeypatch, fused, check):
    off, fused_off = _warm_keys(monkeypatch, False, fused, check)
    on, fused_on = _warm_keys(monkeypatch, True, fused, check)
    assert fused_off == fused_on  # fused verify warmup untouched
    sep_cand = {k for k in off if k[1] and k[2] > 0}
    assert sep_cand  # flag off warms the separate candidates path
    if fused and check == 0:
        assert on == off - sep_cand  # only those keys are dropped
    else:
        assert on == off  # reachable -> still compiled


def test_cpu_device_never_skips(monkeypatch):
    off, _ = _warm_keys(monkeypatch, False, True, 0, dev="cpu")
    on, _ = _warm_keys(monkeypatch, True, True, 0, dev="cpu")
    assert on == off


def _dispatch_fake(calls, all_greedy, grammar, topk_blocks):
    f = types.SimpleNamespace(
        rejection_sampler=_RS(calls, synthetic=False, with_tokens=False),
        speculative_config=types.SimpleNamespace(num_speculative_tokens=3),
        structured_decoding_manager=types.SimpleNamespace(
            mask_spec_logits=lambda t, b, *a: (t, b)
        ),
        spec_bonus_and_target_logits=lambda l, b, t: (b, l.as_subclass(torch.Tensor)),
        spec_gather_bonus_and_target_logits=lambda l, b, t: (
            l.as_subclass(torch.Tensor), l.as_subclass(torch.Tensor)
        ),
        _greedy_sample=lambda l: torch.zeros(l.shape[0], dtype=torch.int64),
        _build_padded_sampling_params=lambda s, e, l: (
            torch.ones(l.shape[0], 1), torch.ones(l.shape[0], 1, dtype=torch.int32),
            torch.ones(l.shape[0], 1),
        ),
        _sampler_topk_candidates=lambda s, e, g: 0 if g is not None else topk_blocks,
        _sampler_candidates_only=lambda c: c > 0,
        sample_from_logits_func=lambda *a, **k: torch.zeros(a[0].shape[0], dtype=torch.int64),
        _sample_spec_verify_fused=lambda *a: calls.append(("fused",)) or torch.zeros(1),
        _verify_device_type=lambda: "tpu",
    )
    f._separate_candidates_verify_reachable = types.MethodType(
        TPUModelRunner._separate_candidates_verify_reachable, f
    )
    f._separate_candidates_verify_warm_needed = types.MethodType(
        TPUModelRunner._separate_candidates_verify_warm_needed, f
    )
    return f


class _TpuLogits(torch.Tensor):
    """CPU tensor that reports a TPU device to the dispatch's device check."""

    @property
    def device(self):  # noqa: D401
        return types.SimpleNamespace(type="tpu")


@pytest.mark.parametrize("all_greedy", [False, True])
@pytest.mark.parametrize("grammar", [None, object()])
def test_served_dispatch_never_requests_skipped_graph(monkeypatch, all_greedy, grammar):
    """With fused on and check off, the step never reaches the separate
    rejection sampler with candidates, whatever greedy/grammar mix it has."""
    monkeypatch.setattr(envs, "VLLM_TPU_RS_WARMUP_SKIP_SEPARATE_CANDIDATES", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_VERIFY_FUSED", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_VERIFY_FUSED_CHECK", 0, raising=False)
    calls = []
    f = _dispatch_fake(calls, all_greedy, grammar, 2)
    md = types.SimpleNamespace(
        draft_token_ids=torch.zeros(4, dtype=torch.int32),
        draft_lengths=torch.zeros(1, dtype=torch.int32),
        draft_lengths_cpu=None,
        bonus_logits_indices=torch.zeros(1, dtype=torch.int32),
        target_logits_indices=torch.zeros(4, dtype=torch.int32),
        segment_ids=torch.zeros(4, dtype=torch.int64),
        group_indices=torch.zeros(4, dtype=torch.int32),
    )
    logits = torch.zeros(5, 256).as_subclass(_TpuLogits)
    try:
        TPUModelRunner._sample_spec_verify_chunk(
            f, logits, md, grammar, types.SimpleNamespace(scheduled_spec_decode_tokens={}),
            0, 1, all_greedy, None,
        )
    except Exception as e:  # downstream plumbing of the fake; routing matters
        calls.append(("exc", repr(e)[:200]))
    assert calls and not any(c[0] == "exc" for c in calls), calls
    if not all_greedy and grammar is None:
        assert calls == [("fused",)]
    assert not any(c[0] == "rs" and c[1] and c[2] > 0 for c in calls), calls
