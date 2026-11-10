# Copyright 2025 Google LLC
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
"""The fused non-greedy verify (spec_verify_fused_sampler) returns the tokens
of the unfused verify, draw for draw."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter.sample import fused_spec_verify
from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import apply_top_k_top_p
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
    apply_top_k=True,
):
    """TPUModelRunner.sample_from_logits as of cdf89e7fd4, frozen, so the
    refactor onto fused_spec_verify.sample_rows is checked too."""
    if all_greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)
    is_greedy = temperatures <= SAMPLING_EPS
    safe_temperatures = torch.where(temperatures == 0.0, 1.0, temperatures)
    scaled_logits = logits / safe_temperatures
    if apply_topk_topp:
        masked_logits = apply_top_k_top_p(
            scaled_logits, top_k, top_p, topk_candidates, apply_top_k
        )
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
    )
    return logits, md, input_batch


def _runner(fused, synthetic, vocab, input_batch, apply_top_k):
    runner = SimpleNamespace(
        _fused_spec_sampler=fused,
        vocab_size=vocab,
        input_batch=input_batch,
        rejection_sampler=RejectionSampler(_spec_config(synthetic)),
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        sample_from_logits_func=_reference_sample_from_logits,
        _sampler_topk_candidates=lambda start, end, grammar: 0,
        _sampler_applies_topk_topp=lambda: True,
        _sampler_applies_top_k=lambda: apply_top_k,
    )
    for name in (
        "_sample_spec_verify_fused",
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


@pytest.mark.parametrize("synthetic", [True, False])
@pytest.mark.parametrize(
    "num_reqs,padded_num_reqs,dtype,top_k_rows,apply_top_k",
    [
        (13, 16, torch.bfloat16, 0, False),
        (13, 16, torch.float32, 4, True),
        (60, 64, torch.bfloat16, 0, False),
        (60, 64, torch.bfloat16, 9, True),
    ],
)
def test_fused_verify_matches_unfused(
    synthetic, num_reqs, padded_num_reqs, dtype, top_k_rows, apply_top_k
):
    vocab = 4096
    for seed in range(3):
        logits, md, input_batch = _case(
            seed, num_reqs, padded_num_reqs, vocab, dtype, top_k_rows
        )
        args = (synthetic, vocab, input_batch, apply_top_k)
        want = _verify(_runner(False, *args), logits, md, num_reqs, seed)
        got = _verify(_runner(True, *args), logits, md, num_reqs, seed)
        assert torch.equal(got, want), (seed, got, want)
        # The case exercises both outcomes: some requests take the bonus token
        # (all drafts accepted), some a recovered token.
        accepted = (want[:num_reqs] != -1).sum(-1) - 1
        lens = torch.from_numpy(md.draft_lengths_cpu)
        assert bool((accepted == lens).any()) and bool((accepted < lens).any())


def test_fused_verify_large_vocab_matches_unfused():
    """MiniMax-M3 vocab (200,064) at bf16, synthetic, top-p only."""
    vocab = 200_064
    logits, md, input_batch = _case(7, 6, 8, vocab, torch.bfloat16, 0)
    args = (True, vocab, input_batch, False)
    want = _verify(_runner(False, *args), logits, md, 6, 7)
    got = _verify(_runner(True, *args), logits, md, 6, 7)
    assert torch.equal(got, want)


def test_fused_sampler_enabled():
    assert not fused_spec_verify.fused_sampler_enabled({})
    assert not fused_spec_verify.fused_sampler_enabled(None)
    assert fused_spec_verify.fused_sampler_enabled({"spec_verify_fused_sampler": True})
    with pytest.raises(ValueError):
        fused_spec_verify.fused_sampler_enabled({"spec_verify_fused_sampler": 1})


@pytest.mark.parametrize("blocks", [0, 2])
def test_precompile_warms_every_fused_variant(monkeypatch, blocks):
    """_precompile_fused_spec_verify runs the fused graph at each variant
    _sample_spec_verify_fused can pick, on inputs of the served dtypes."""
    from vllm_torchtpu.runner import tpu_runner

    num_tokens, num_reqs, vocab = 16, 2, 256
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda out: None)
    seen = []
    fused_fn = fused_spec_verify.spec_verify_sample_fused

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

    monkeypatch.setattr(fused_spec_verify, "spec_verify_sample_fused", recording)
    runner = SimpleNamespace(
        rejection_sampler=RejectionSampler(_spec_config(True)),
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        vocab_size=vocab,
        _topk_candidate_blocks=lambda: blocks,
    )
    with torch.compiler.set_stance("force_eager"):
        TPUModelRunner._precompile_fused_spec_verify(
            runner,
            torch.zeros((num_tokens, vocab), dtype=torch.bfloat16),
            torch.zeros(num_reqs, dtype=torch.int32),
            torch.zeros(num_tokens, dtype=torch.int32),
            torch.zeros(num_tokens, dtype=torch.int32),
            torch.zeros(num_reqs, dtype=torch.int32),
            torch.zeros(num_tokens, dtype=torch.int64),
            torch.zeros(num_tokens, dtype=torch.int32),
            torch.zeros(num_tokens, dtype=torch.float32),
            torch.zeros((num_tokens, vocab), dtype=torch.float32),
        )
    want = [(True, True, 0), (False, True, 0), (True, False, 0)]
    if blocks:
        want.append((True, True, blocks))
    assert seen == want
