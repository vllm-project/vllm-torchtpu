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
"""MiniMax M3's fused non-greedy verify (spec_verify.FUSED_SPEC_VERIFY, bound
as the model's spec_verify_sampler) returns the tokens of the runner's
unfused verify, draw for draw."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    RejectionSampler,
)
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import apply_top_k_top_p
from vllm_torchtpu.models.minimax_m3.experimental import spec_verify
from vllm_torchtpu.runner.spec_verify_sampler import (
    SpecVerifyInputs,
    SpecVerifySampler,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

K = 7
SAMPLING_EPS = 1e-5


def _reference_sample_from_logits(
    logits,
    temperatures,
    u,
    top_k,
    top_p,
    all_greedy=False,
    topk_candidates=0,
    apply_topk_topp=True,
):
    """TPUModelRunner.sample_from_logits (the runner's unfused bonus sampler),
    frozen."""
    if all_greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)
    is_greedy = temperatures <= SAMPLING_EPS
    safe_temperatures = torch.where(temperatures == 0.0, 1.0, temperatures)
    scaled_logits = logits / safe_temperatures
    if apply_topk_topp:
        masked_logits = apply_top_k_top_p(scaled_logits, top_k, top_p, topk_candidates)
    else:
        masked_logits = scaled_logits
    u_clamped = torch.clamp(
        u, min=torch.finfo(u.dtype).tiny, max=1.0 - torch.finfo(u.dtype).eps
    )
    noisy_logits = masked_logits + (-torch.log(-torch.log(u_clamped)))
    final_logits = torch.where(is_greedy, logits, noisy_logits)
    return torch.argmax(final_logits, dim=-1, keepdim=True)


def _spec_config(synthetic: bool):
    if not synthetic:
        return None
    return SimpleNamespace(
        rejection_sample_method="synthetic",
        # Mean acceptance length ~4 of 7, so both bonus and recovered tokens
        # occur.
        synthetic_acceptance_rates=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
    )


def _case(seed, num_reqs, padded_num_reqs, vocab, dtype, top_k_rows):
    """Logits, metadata and per-request sampling parameters of one verify
    chunk laid out as get_spec_decode_metadata does."""
    rng = np.random.default_rng(seed)
    draft_lens = rng.integers(0, K + 1, size=num_reqs)
    draft_lens[0] = K
    num_sampled = draft_lens + 1
    cu = np.cumsum(num_sampled)
    num_rows = int(cu[-1])
    padded_len = 1 << (num_rows - 1).bit_length()
    starts = cu - num_sampled
    bonus = np.zeros(padded_num_reqs, np.int32)
    bonus[:num_reqs] = cu - 1
    target = np.zeros(padded_len, np.int32)
    t = np.concatenate([s + np.arange(n) for s, n in zip(starts, draft_lens)])
    target[: t.size] = t
    seg = np.full(padded_len, num_reqs, np.int64)
    seg[: t.size] = np.repeat(np.arange(num_reqs), draft_lens)
    grp = np.zeros(padded_len, np.int32)
    grp[: t.size] = np.concatenate([np.arange(n) for n in draft_lens])
    nd = np.zeros(padded_num_reqs, np.int32)
    nd[:num_reqs] = draft_lens

    g = torch.Generator().manual_seed(seed)
    # Peaked logits so that real (non-synthetic) acceptance accepts some
    # drafts: the draft token is often the row's argmax.
    logits = (torch.randn((padded_len, vocab), generator=g) * 3).to(dtype)
    draft = np.zeros(padded_len, np.int32)
    argmax = logits.float().argmax(-1).numpy()
    for i, row in enumerate(t):
        draft[i] = argmax[row] if rng.random() < 0.7 else rng.integers(vocab)
    md = SimpleNamespace(
        draft_token_ids=torch.from_numpy(draft),
        draft_lengths=torch.from_numpy(nd),
        draft_lengths_cpu=draft_lens,
        target_logits_indices=torch.from_numpy(target),
        bonus_logits_indices=torch.from_numpy(bonus),
        segment_ids=torch.from_numpy(seg),
        group_indices=torch.from_numpy(grp),
    )
    temperature = rng.choice([0.0, 0.01, 0.6, 1.0, 1.3], size=num_reqs)
    temperature[:2] = 1.0
    top_p = rng.choice([0.5, 0.9, 0.95, 1.0], size=num_reqs)
    top_k = np.full(num_reqs, vocab)  # vLLM's "unset"
    top_k[:top_k_rows] = rng.integers(1, 40, size=top_k_rows)
    input_batch = SimpleNamespace(
        temperature_cpu_tensor=torch.tensor(temperature, dtype=torch.float32),
        top_k_cpu_tensor=torch.tensor(top_k, dtype=torch.int32),
        top_p_cpu_tensor=torch.tensor(top_p, dtype=torch.float32),
        no_top_k=top_k_rows == 0,
    )
    return logits, md, input_batch


def _runner(fused, synthetic, vocab, input_batch):
    runner = SimpleNamespace(
        _spec_verify_sampler=spec_verify.FUSED_SPEC_VERIFY if fused else None,
        vocab_size=vocab,
        input_batch=input_batch,
        rejection_sampler=RejectionSampler(_spec_config(synthetic)),
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        sample_from_logits_func=_reference_sample_from_logits,
        _sampler_topk_candidates=lambda start, end, grammar: 0,
        _sampler_applies_topk_topp=lambda: True,
    )
    for name in (
        "_sample_spec_verify_model",
        "_sample_chunk_logits",
        "_build_padded_sampling_params",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    runner.spec_gather_bonus_and_target_logits = (
        TPUModelRunner.spec_gather_bonus_and_target_logits.__wrapped__.__get__(runner)
    )
    return runner


def _verify(runner, logits, md, num_reqs, seed):
    with torch.compiler.set_stance("force_eager"):
        return TPUModelRunner._sample_spec_verify_chunk(
            runner,
            logits,
            md,
            grammar_output=None,
            scheduler_output=SimpleNamespace(),
            cur_start_idx=0,
            cur_end_idx=num_reqs,
            all_greedy=False,
            sampling_generator=torch.Generator().manual_seed(seed + 1000),
        )


def test_fused_spec_verify_is_a_spec_verify_sampler():
    assert isinstance(spec_verify.FUSED_SPEC_VERIFY, SpecVerifySampler)


@pytest.mark.parametrize("synthetic", [True, False])
@pytest.mark.parametrize(
    "num_reqs,padded_num_reqs,dtype,top_k_rows",
    [
        (13, 16, torch.bfloat16, 0),
        (13, 16, torch.float32, 4),
        (60, 64, torch.bfloat16, 0),
        (60, 64, torch.bfloat16, 9),
    ],
)
def test_fused_verify_matches_unfused(
    synthetic, num_reqs, padded_num_reqs, dtype, top_k_rows
):
    vocab = 4096
    for seed in range(3):
        logits, md, input_batch = _case(
            seed, num_reqs, padded_num_reqs, vocab, dtype, top_k_rows
        )
        args = (synthetic, vocab, input_batch)
        want = _verify(_runner(False, *args), logits, md, num_reqs, seed)
        got = _verify(_runner(True, *args), logits, md, num_reqs, seed)
        assert torch.equal(got, want), (seed, got, want)
        # The case exercises both outcomes: some requests take the bonus token
        # (all drafts accepted), some a recovered token.
        accepted = (want[:num_reqs] != -1).sum(-1) - 1
        lens = torch.from_numpy(md.draft_lengths_cpu)
        assert bool((accepted == lens).any()) and bool((accepted < lens).any())


def test_fused_verify_large_vocab_matches_unfused():
    """A 200,064-entry vocab at bf16, synthetic, top-p only."""
    vocab = 200_064
    logits, md, input_batch = _case(7, 6, 8, vocab, torch.bfloat16, 0)
    args = (True, vocab, input_batch)
    want = _verify(_runner(False, *args), logits, md, 6, 7)
    got = _verify(_runner(True, *args), logits, md, 6, 7)
    assert torch.equal(got, want)


@pytest.mark.parametrize("blocks", [0, 2])
def test_precompile_warms_every_fused_variant(monkeypatch, blocks):
    """precompile runs the fused graph at each variant sample can pick, on
    inputs of the served dtypes."""
    num_tokens, num_reqs, vocab = 16, 2, 256
    monkeypatch.setattr(spec_verify, "synchronize_tensors", lambda out: None)
    seen = []
    fused_fn = spec_verify.spec_verify_sample_fused

    def recording(*args, **kwargs):
        seen.append(
            (
                kwargs["apply_topk_topp"],
                kwargs["apply_top_k"],
                kwargs["topk_candidates"],
            )
        )
        out = fused_fn(*args, **kwargs)
        assert out.shape == (num_reqs, K + 1)
        return out

    monkeypatch.setattr(spec_verify, "spec_verify_sample_fused", recording)
    sampler = RejectionSampler(_spec_config(True))
    inputs = SpecVerifyInputs(
        logits=torch.zeros((num_tokens, vocab), dtype=torch.bfloat16),
        bonus_logits_indices=torch.zeros(num_reqs, dtype=torch.int32),
        target_logits_indices=torch.zeros(num_tokens, dtype=torch.int32),
        draft_token_ids=torch.zeros(num_tokens, dtype=torch.int32),
        num_draft_tokens=torch.zeros(num_reqs, dtype=torch.int32),
        segment_ids=torch.zeros(num_tokens, dtype=torch.int64),
        group_indices=torch.zeros(num_tokens, dtype=torch.int32),
        temperatures=torch.ones((num_reqs, 1), dtype=torch.bfloat16),
        top_k=torch.zeros((num_reqs, 1), dtype=torch.int32),
        top_p=torch.ones((num_reqs, 1), dtype=torch.float32),
        bonus_u=torch.zeros((num_reqs, vocab), dtype=torch.float32),
        accept_u=torch.zeros(num_tokens, dtype=torch.float32),
        recover_u=torch.zeros((num_tokens, vocab), dtype=torch.float32),
        synthetic_conditional_rates=sampler.synthetic_conditional_rates,
        synthetic_mode=sampler.synthetic_mode,
        max_draft_tokens=K,
        topk_candidates=0,
        apply_topk_topp=True,
        any_top_k=True,
    )
    variants = [(0, True), (0, False)] + ([(blocks, True)] if blocks else [])
    with torch.compiler.set_stance("force_eager"):
        spec_verify.FUSED_SPEC_VERIFY.precompile(inputs, variants)
    want = [(True, True, 0), (False, True, 0)]
    if blocks:
        want.append((True, True, blocks))
    want.append((True, False, 0))  # top-p only
    assert seen == want


def _segment_info(lengths):
    counts = np.asarray(lengths, dtype=np.int64)
    seg = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    starts = np.zeros(len(counts), dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    grp = np.arange(seg.shape[0], dtype=np.int64) - np.repeat(starts, counts)
    return torch.from_numpy(seg), torch.from_numpy(grp)


def _impl(sampler, kwargs, **overrides):
    kw = dict(kwargs, **overrides)
    return spec_verify._random_rejection_sample_with_segment_impl(
        kw["draft_token_ids"],
        kw["target_logits"],
        kw["num_draft_tokens"],
        kw["bonus_token_ids"],
        kw["segment_ids"],
        kw["group_indices"],
        kw["max_draft_tokens"],
        kw["temperatures"],
        kw["top_k"],
        kw["top_p"],
        kw["accept_u"],
        kw["recover_u"],
        sampler.synthetic_conditional_rates,
        sampler.synthetic_mode,
        0,
        kw.get("apply_topk_topp", True),
        kw.get("draft_probs"),
        kw.get("apply_top_k", True),
    )


@pytest.mark.parametrize("with_draft_probs", [False, True])
@pytest.mark.parametrize("apply_topk_topp", [True, False])
def test_synthetic_recovers_rejected_rows_only(apply_topk_topp, with_draft_probs):
    """Synthetic sampling recovers one row per request. The tokens must match
    recovering every draft row and keeping the first rejected slot (the
    shared rejection sampler), at the served bucket (64 requests x 7 drafts)
    with padding, short drafts, greedy rows and top-k rows."""
    num_reqs, k, vocab_size = 64, 7, 128 * 64
    rates = [0.9, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2]
    sampler = RejectionSampler(
        SimpleNamespace(
            rejection_sample_method="synthetic", synthetic_acceptance_rates=rates
        )
    )
    gen = torch.Generator().manual_seed(0)
    lengths = [k] * num_reqs
    for i in range(1, num_reqs, 5):
        lengths[i] = i % k
    rows = num_reqs * k
    used = sum(lengths)
    segment_ids, group_indices = _segment_info(lengths)
    # Rows past the last draft are bucket padding with a sentinel segment.
    segment_ids = torch.cat([segment_ids, torch.full((rows - used,), num_reqs)])
    group_indices = torch.cat(
        [group_indices, torch.zeros((rows - used,), dtype=torch.int64)]
    )
    logits = torch.randn(rows, vocab_size, generator=gen) * 4
    drafts = torch.where(
        torch.rand(rows, generator=gen) < 0.5,
        logits.argmax(dim=-1),
        torch.randint(0, vocab_size, (rows,), generator=gen),
    )
    drafts[used:] = -1
    temperatures = torch.ones((rows, 1))
    temperatures[::9] = 0.0
    top_k = torch.zeros((rows, 1), dtype=torch.int32)
    top_k[::4] = 40
    top_p = torch.full((rows, 1), 0.95)
    draft_probs = None
    if with_draft_probs:
        draft_probs = torch.softmax(torch.randn(rows, vocab_size, generator=gen), -1)
    kwargs = dict(
        draft_token_ids=drafts.to(torch.int32),
        num_draft_tokens=torch.tensor(lengths, dtype=torch.int32),
        target_logits=logits,
        bonus_token_ids=torch.arange(num_reqs, dtype=torch.int32),
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=k,
        temperatures=temperatures,
        top_k=top_k,
        top_p=top_p,
        accept_u=torch.rand(rows, generator=gen),
        recover_u=torch.rand(rows, vocab_size, generator=gen),
        apply_topk_topp=apply_topk_topp,
        draft_probs=draft_probs,
    )
    output = _impl(sampler, kwargs)
    with torch.compiler.set_stance("force_eager"):
        shared = sampler(**kwargs, do_sampling=True)
    assert torch.equal(output, shared)

    # Reference: recover every row, then take the first rejected slot.
    recovered = spec_verify._recover_tokens(
        kwargs["target_logits"],
        kwargs["draft_token_ids"].to(torch.int64),
        kwargs["temperatures"],
        kwargs["top_k"],
        kwargs["top_p"],
        kwargs["recover_u"],
        0,
        apply_topk_topp,
        draft_probs,
    )
    conditional = sampler.synthetic_conditional_rates
    accept_u = kwargs["accept_u"]
    expected = torch.full((num_reqs, k + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    row = 0
    for req, length in enumerate(lengths):
        for slot in range(length):
            token = int(drafts[row + slot])
            if token >= 0 and accept_u[row + slot] < conditional[slot]:
                expected[req, slot] = token
            else:
                expected[req, slot] = recovered[row + slot]
                break
        else:
            expected[req, length] = req
        row += length
    assert torch.equal(output, expected)


@pytest.mark.parametrize("synthetic", [False, True])
def test_skip_top_k_matches_full_path(synthetic):
    """apply_top_k=False (no request sets top-k) gives the same tokens as the
    full top-k + top-p path at the served bucket, synthetic or not."""
    num_reqs, k, vocab_size = 64, 7, 128 * 64
    rows = num_reqs * k
    gen = torch.Generator().manual_seed(1)
    logits = torch.randn(rows, vocab_size, generator=gen) * 4
    drafts = torch.where(
        torch.rand(rows, generator=gen) < 0.5,
        logits.argmax(dim=-1),
        torch.randint(0, vocab_size, (rows,), generator=gen),
    )
    temperatures = torch.ones((rows, 1))
    temperatures[::9] = 0.0
    segment_ids, group_indices = _segment_info([k] * num_reqs)
    kwargs = dict(
        draft_token_ids=drafts.to(torch.int32),
        num_draft_tokens=torch.full((num_reqs,), k, dtype=torch.int32),
        target_logits=logits,
        bonus_token_ids=torch.arange(num_reqs, dtype=torch.int32),
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=k,
        temperatures=temperatures,
        top_k=torch.zeros((rows, 1), dtype=torch.int32),
        top_p=torch.full((rows, 1), 0.95),
        accept_u=torch.rand(rows, generator=gen),
        recover_u=torch.rand(rows, vocab_size, generator=gen),
    )
    sampler = RejectionSampler(
        _spec_config(True) if synthetic else None,
    )
    expected = _impl(sampler, kwargs)
    result = _impl(sampler, kwargs, apply_top_k=False)
    assert torch.equal(result, expected)
    # Positive control: the inputs reject and recover somewhere.
    assert bool((expected[:, :k] != kwargs["draft_token_ids"].view(num_reqs, k)).any())


def test_apply_top_k_top_p_skip_top_k_is_exact_without_top_k():
    gen = torch.Generator().manual_seed(2)
    logits = torch.randn(16, 4096, generator=gen) * 4
    top_k = torch.zeros((16, 1), dtype=torch.int32)
    top_p = torch.rand((16, 1), generator=gen)
    top_p[::3] = 1.0
    expected = apply_top_k_top_p(logits, top_k, top_p)
    assert torch.equal(spec_verify.apply_top_k_top_p(logits, top_k, top_p), expected)
    result = spec_verify.apply_top_k_top_p(logits, top_k, top_p, apply_top_k=False)
    assert torch.equal(result, expected)
    # With top-k set on a row the skip is not exact, so the caller must not
    # take it then.
    top_k[0] = 5
    assert not torch.equal(
        spec_verify.apply_top_k_top_p(logits, top_k, top_p),
        spec_verify.apply_top_k_top_p(logits, top_k, top_p, apply_top_k=False),
    )
