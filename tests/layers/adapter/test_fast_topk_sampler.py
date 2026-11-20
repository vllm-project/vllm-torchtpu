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
"""Parity of the fast small-k sampler (VLLM_TPU_FAST_TOPK_TOPP) vs the stock one.

The stock path masks the full vocabulary with apply_top_k_top_p and takes a
Gumbel-max over it; the fast path keeps only the top `k_cap` logits (block
pre-filtered when the vocab allows) and does the same on those. For any row
with 1 <= top_k <= k_cap they must keep the same tokens with the same logits,
so they sample from the same distribution (only the RNG stream differs).

Known bound: the stock top-k mask keeps every logit tied with the k-th
largest, while the fast path sees at most `k_cap` candidates. Rows whose
top-k plus its boundary ties exceed `k_cap` keep fewer tokens on the fast
path. The bf16 cases below therefore stay at top_k <= 24 with k_cap = 32;
top_k == k_cap is checked on tie-free float32 logits.
"""

import pytest
import torch

from vllm_torchtpu.layers.adapter.sample import top_k_top_p as sampling

SAMPLING_EPS = 1e-5
K_CAP = 32


def _stock_sample(logits, temperatures, u, top_k, top_p):
    """TPUModelRunner.sample_from_logits without torch.compile / fast path."""
    is_greedy = temperatures <= SAMPLING_EPS
    safe_t = torch.where(temperatures == 0.0, 1.0, temperatures)
    masked = sampling.apply_top_k_top_p(logits / safe_t, top_k, top_p)
    u = torch.clamp(
        u, min=torch.finfo(u.dtype).tiny, max=1.0 - torch.finfo(u.dtype).eps
    )
    noisy = masked + (-torch.log(-torch.log(u)))
    return torch.argmax(torch.where(is_greedy, logits, noisy), dim=-1, keepdim=True)


def _stock_probs(logits, temperature, top_k, top_p):
    """Exact sampling distribution of the stock path for one row."""
    masked = sampling.apply_top_k_top_p(
        logits / temperature,
        torch.tensor([[top_k]], dtype=torch.int32),
        torch.tensor([[top_p]]),
    )
    return torch.softmax(masked.double(), dim=-1)[0]


def _logits(rows, vocab, seed, scale=3.0, bf16=True):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, vocab, generator=gen) * scale
    # bf16-rounded, like real model logits, so near-ties are common.
    return x.bfloat16().float() if bf16 else x


@pytest.mark.parametrize(
    "top_k, bf16", [(1, True), (5, True), (20, True), (24, True), (32, False)]
)
@pytest.mark.parametrize("top_p", [1.0, 0.95, 0.5])
def test_fast_mask_keeps_same_tokens_as_full_vocab(top_k, bf16, top_p):
    logits = _logits(8, 128 * 400, seed=top_k, bf16=bf16) / 0.6
    tk = torch.full((8, 1), top_k, dtype=torch.int32)
    tp = torch.full((8, 1), top_p)
    full = sampling.apply_top_k_top_p(logits, tk, tp)
    vals, idx = torch.topk(logits, K_CAP, dim=-1, sorted=True)
    fast = sampling.apply_top_k_top_p_sorted(vals, tk, tp)
    for r in range(8):
        keep_full = torch.nonzero(full[r] > sampling.MASKED_LOGIT_VALUE).flatten()
        keep_fast = idx[r][fast[r] > sampling.MASKED_LOGIT_VALUE]
        assert sorted(keep_full.tolist()) == sorted(keep_fast.tolist())
        torch.testing.assert_close(
            full[r, keep_fast], fast[r][fast[r] > sampling.MASKED_LOGIT_VALUE],
            rtol=0, atol=0,
        )


@pytest.mark.parametrize("vocab", [128 * 400, 128 * 20, 1000])
def test_fast_sampler_only_draws_kept_tokens(vocab):
    # 128*400: block pre-filter; 128*20 (< k_cap blocks) and 1000 (not a
    # multiple of 128): plain top-k.
    rows = 256
    logits = _logits(rows, vocab, seed=vocab)
    temps = torch.full((rows, 1), 0.7)
    tk = torch.randint(1, K_CAP + 1, (rows, 1), generator=torch.Generator().manual_seed(1))
    tp = torch.full((rows, 1), 0.9)
    u = torch.rand(rows, K_CAP, generator=torch.Generator().manual_seed(2))
    tokens = sampling.sample_from_logits_fast_topk(
        logits, temps, u, tk.to(torch.int32), tp, k_cap=K_CAP
    )
    full = sampling.apply_top_k_top_p(logits / temps, tk.to(torch.int32), tp)
    picked = full.gather(-1, tokens)
    assert bool((picked > sampling.MASKED_LOGIT_VALUE).all())


def test_fast_sampler_greedy_rows_match_stock():
    rows, vocab = 64, 128 * 400
    logits = _logits(rows, vocab, seed=7)
    temps = torch.where(torch.arange(rows) % 2 == 0, 0.0, 0.8).reshape(rows, 1)
    tk = torch.full((rows, 1), 20, dtype=torch.int32)
    tp = torch.full((rows, 1), 0.95)
    gen = torch.Generator().manual_seed(3)
    fast = sampling.sample_from_logits_fast_topk(
        logits, temps, torch.rand(rows, K_CAP, generator=gen), tk, tp, k_cap=K_CAP
    )
    stock = _stock_sample(logits, temps, torch.rand(rows, vocab, generator=gen), tk, tp)
    greedy = temps.flatten() <= SAMPLING_EPS
    torch.testing.assert_close(fast[greedy], stock[greedy])
    torch.testing.assert_close(fast[greedy].flatten(), logits[greedy].argmax(-1))


@pytest.mark.parametrize(
    "top_k, top_p, temperature",
    [(20, 0.95, 1.0), (32, 1.0, 0.7), (5, 0.8, 1.3)],
)
def test_fast_sampler_histogram_matches_stock_distribution(top_k, top_p, temperature):
    # Qwen generation_config is (top_k=20, top_p=0.95, temperature=1.0).
    vocab, draws, chunk = 128 * 40, 120_000, 10_000
    row = _logits(1, vocab, seed=11, scale=1.5)
    p_ref = _stock_probs(row, temperature, top_k, top_p)
    support = torch.nonzero(p_ref > 0).flatten()
    assert 1 <= support.numel() <= K_CAP

    gen = torch.Generator().manual_seed(top_k * 1000 + int(top_p * 100))
    counts_fast = torch.zeros(vocab, dtype=torch.float64)
    counts_stock = torch.zeros(vocab, dtype=torch.float64)
    logits = row.expand(chunk, vocab).contiguous()
    temps = torch.full((chunk, 1), temperature)
    tk = torch.full((chunk, 1), top_k, dtype=torch.int32)
    tp = torch.full((chunk, 1), top_p)
    for _ in range(draws // chunk):
        t_fast = sampling.sample_from_logits_fast_topk(
            logits, temps, torch.rand(chunk, K_CAP, generator=gen), tk, tp, k_cap=K_CAP
        )
        t_stock = _stock_sample(
            logits, temps, torch.rand(chunk, vocab, generator=gen), tk, tp
        )
        counts_fast += torch.bincount(t_fast.flatten(), minlength=vocab).double()
        counts_stock += torch.bincount(t_stock.flatten(), minlength=vocab).double()

    for counts in (counts_fast, counts_stock):
        # Never outside the stock support.
        assert counts.sum() == counts[support].sum() == draws
        emp = counts[support] / draws
        # Total variation vs the exact distribution; with 120k draws over <= 32
        # tokens the sampling noise is ~0.005.
        tv = 0.5 * (emp - p_ref[support]).abs().sum().item()
        assert tv < 0.015, tv
        # Pearson chi-square on tokens with >= 5 expected draws. The
        # 99.99th percentile of chi2(df) is < df + 9 * sqrt(2 * df) + 20.
        exp = p_ref[support] * draws
        ok = exp >= 5
        chi2 = (((counts[support] - exp) ** 2) / exp)[ok].sum().item()
        df = int(ok.sum().item()) - 1
        assert chi2 < df + 9 * (2 * max(df, 1)) ** 0.5 + 20, (chi2, df)
