# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_SAMPLER_BONUS_F32: the non-greedy sampler (the bonus token of a
spec verify) scales and top-k/top-p masks float32 logits.

In bf16 the softmax and the top-p mass sums round to ~0.4%, so a row whose
top-p cut falls within that rounding keeps the wrong set. HEAD is a top_k 8
row whose top-3 mass is 0.94960 < top_p = 0.95: the exact sampler keeps 4
tokens, the bf16 one keeps 3 on CPU (the dropped token has p = 0.023).
HEAD_TPU is the measured TPU case (top-3 mass 0.94856, TV 0.032 on TPU); the
CPU's bf16 sum rounds once at the end and happens to keep 4 there, so only
the flag-on side is checked on it. Flag off must stay as before (bf16), flag
on must keep the exact set on every sampler path: gathered, candidate
blocks, logits blocks (kmax, top-p narrow); the candidate-sampling op is in
test_candidate_sampling.py. Greedy rows are bitwise the same either way.
"""

import functools

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    CANDIDATE_BLOCK,
    apply_top_k_top_p,
    apply_top_k_top_p_candidates_pre,
    row_block_gather,
)
from vllm_torchtpu.runner import tpu_runner as tr

# Top 8 logits (bf16-exact), top_k 8, top_p 0.95.
HEAD = [10.0, 9.5, 8.625, 6.90625, 6.34375, 5.59375, 5.5, 4.53125]
HEAD_TPU = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.3125, 4.28125]
TOP_K, TOP_P = 8, 0.95
VOCAB = 16 * CANDIDATE_BLOCK
NUM_BLOCKS = 16  # candidate blocks; the gate keeps top_k <= NUM_BLOCKS // 2


def _row(head=HEAD, perm_seed=0):
    """[VOCAB] bf16: `head` at scattered ids over a tail far below it."""
    g = torch.Generator().manual_seed(perm_seed)
    row = torch.randn(VOCAB, generator=g) - 4.0
    ids = torch.randperm(VOCAB, generator=g)[: len(head)]
    row[ids] = torch.tensor(head)
    return row.to(torch.bfloat16), ids


def _reference_probs(row, top_k, top_p, temp=1.0):
    """float64 vLLM top-k (ties kept) then top-p over sorted probabilities."""
    x = row.double() / temp
    kth = x.sort(descending=True).values[top_k - 1]
    x = torch.where(x >= kth, x, torch.tensor(float("-inf"), dtype=x.dtype))
    p = torch.softmax(x, -1)
    order = p.argsort(descending=True, stable=True)
    cum = p[order].cumsum(0)
    n_keep = int((cum < top_p).sum()) + 1  # smallest prefix with mass >= p
    keep = torch.zeros_like(p, dtype=torch.bool)
    keep[order[:n_keep]] = True
    p = torch.where(keep, p, 0.0)
    return p / p.sum()


def _blocks_sampler(kmax, topp_narrow):
    """`_sample_from_logits_pre` (VLLM_TPU_SAMPLER_LOGITS_BLOCKS) on [R, V]."""

    def sample(logits, temps, u, top_k, top_p):
        rows = logits.shape[0]
        bpr = logits.shape[1] // CANDIDATE_BLOCK
        flat = logits.reshape(rows * bpr, CANDIDATE_BLOCK)
        block_max = logits.reshape(rows, bpr, CANDIDATE_BLOCK).amax(-1)
        idx = torch.arange(rows)
        return tr._sample_from_logits_pre(
            logits,
            temps,
            u,
            top_k,
            top_p,
            NUM_BLOCKS,
            block_max,
            lambda b: row_block_gather(flat, bpr, idx, b),
            kmax,
            topp_narrow,
        )

    return sample


SAMPLERS = {
    "gathered": tr._sample_from_logits_impl,
    "candidates": functools.partial(
        tr._sample_from_logits_impl, topk_candidates=NUM_BLOCKS
    ),
    "logits_blocks": _blocks_sampler(0, False),
    "logits_blocks_kmax": _blocks_sampler(NUM_BLOCKS // 2, False),
    "logits_blocks_narrow": _blocks_sampler(NUM_BLOCKS // 2, True),
}


def _params(rows, temp=1.0, top_k=TOP_K, top_p=TOP_P):
    return (
        torch.full((rows, 1), temp, dtype=torch.bfloat16),
        torch.full((rows, 1), top_k, dtype=torch.int32),
        torch.full((rows, 1), top_p, dtype=torch.float32),
    )


def _kept_ids(sampler, row):
    """The sampler's support, read through the sampler itself: for each head
    token, uniforms that put its Gumbel noise ~21 above every other token's
    (more than the head's logit spread), so it is drawn iff it is unmasked."""
    rows = VOCAB
    logits = row.expand(rows, VOCAB).contiguous()
    u = torch.full((rows, VOCAB), torch.finfo(torch.float32).tiny)
    u.fill_diagonal_(1.0)  # clamped to 1 - eps
    temps, top_k, top_p = _params(rows)
    out = sampler(logits, temps, u, top_k, top_p).view(-1)
    return set((out == torch.arange(rows)).nonzero().view(-1).tolist())


@pytest.mark.cpu_test
@pytest.mark.parametrize("path", list(SAMPLERS))
@pytest.mark.parametrize("head,off_keeps", [(HEAD, 3), (HEAD_TPU, None)])
def test_flag_on_keeps_the_exact_top_p_set(monkeypatch, path, head, off_keeps):
    row, ids = _row(head)
    exact = set(_reference_probs(row, TOP_K, TOP_P).nonzero().view(-1).tolist())
    assert exact == set(ids[:4].tolist())  # 4 tokens: top-3 mass < 0.95
    monkeypatch.setattr(tr, "_SAMPLER_BONUS_F32", False)
    off = _kept_ids(SAMPLERS[path], row)
    monkeypatch.setattr(tr, "_SAMPLER_BONUS_F32", True)
    on = _kept_ids(SAMPLERS[path], row)
    if off_keeps is not None:
        assert off == set(ids[:off_keeps].tolist()), path  # bf16 drops the 4th
    assert on == exact, path


def _chi2(counts, probs):
    """Pearson chi-square p-value of counts against probs (cells with
    expected < 5 pooled)."""
    from scipy.stats import chisquare

    n = counts.sum()
    exp = probs * n
    big = exp >= 5
    obs = np.append(counts[big], counts[~big].sum())
    exp = np.append(exp[big], exp[~big].sum())
    if exp[-1] == 0:
        obs, exp = obs[:-1], exp[:-1]
    return float(chisquare(obs, exp).pvalue)


@pytest.mark.cpu_test
@pytest.mark.parametrize("path", ["gathered", "logits_blocks_narrow"])
def test_flag_on_samples_the_exact_distribution(monkeypatch, path):
    """20k draws from HEAD: flag off fails a chi-square GOF against the
    float64 distribution (the dropped token has p = 0.023),
    flag on passes it."""
    row, _ = _row()
    probs = _reference_probs(row, TOP_K, TOP_P).numpy()
    n, batch = 20000, 2000
    temps, top_k, top_p = _params(batch)
    logits = row.expand(batch, VOCAB).contiguous()
    pvals, tvs = {}, {}
    for flag in (False, True):
        monkeypatch.setattr(tr, "_SAMPLER_BONUS_F32", flag)
        g = torch.Generator().manual_seed(7)
        draws = []
        for _ in range(n // batch):
            u = torch.rand((batch, VOCAB), generator=g)
            draws.append(SAMPLERS[path](logits, temps, u, top_k, top_p).view(-1))
        counts = np.bincount(torch.cat(draws).numpy(), minlength=VOCAB)
        pvals[flag] = _chi2(counts, probs)
        tvs[flag] = 0.5 * np.abs(counts / n - probs).sum()
    print(path, "p-values", pvals, "TV", tvs)
    assert pvals[False] < 1e-6 and tvs[False] > 0.02
    assert pvals[True] > 1e-3 and tvs[True] < 0.01


def _old_sample(logits, temps, u, top_k, top_p, topk_candidates=0):
    """`_sample_from_logits_impl` before the flag, verbatim."""
    is_greedy = temps <= tr.SAMPLING_EPS
    scaled = tr._apply_temperature_impl(logits, temps)
    masked = apply_top_k_top_p(scaled, top_k, top_p, topk_candidates)
    u = torch.clamp(
        u, min=torch.finfo(u.dtype).tiny, max=1.0 - torch.finfo(u.dtype).eps
    )
    noisy = masked + -torch.log(-torch.log(u))
    return torch.argmax(torch.where(is_greedy, logits, noisy), dim=-1, keepdim=True)


def _old_sample_pre(kmax, topp_narrow):
    """`_sample_from_logits_pre` before the flag, verbatim (full-row inputs)."""

    def sample(logits, temps, u, top_k, top_p):
        rows = logits.shape[0]
        bpr = logits.shape[1] // CANDIDATE_BLOCK
        flat = logits.reshape(rows * bpr, CANDIDATE_BLOCK)
        block_max = logits.reshape(rows, bpr, CANDIDATE_BLOCK).amax(-1)
        idx = torch.arange(rows)
        scale = functools.partial(tr._apply_temperature_impl, temperatures=temps)
        masked = apply_top_k_top_p_candidates_pre(
            scale(logits),
            top_k,
            top_p,
            NUM_BLOCKS,
            scale(block_max),
            lambda b: scale(row_block_gather(flat, bpr, idx, b)),
            kmax,
            topp_narrow,
        )
        is_greedy = temps <= tr.SAMPLING_EPS
        u = torch.clamp(
            u, min=torch.finfo(u.dtype).tiny, max=1.0 - torch.finfo(u.dtype).eps
        )
        noisy = masked + -torch.log(-torch.log(u))
        return torch.argmax(torch.where(is_greedy, logits, noisy), dim=-1, keepdim=True)

    return sample


OLD = {
    "gathered": _old_sample,
    "candidates": functools.partial(_old_sample, topk_candidates=NUM_BLOCKS),
    "logits_blocks": _old_sample_pre(0, False),
    "logits_blocks_kmax": _old_sample_pre(NUM_BLOCKS // 2, False),
    "logits_blocks_narrow": _old_sample_pre(NUM_BLOCKS // 2, True),
}


def _mixed_batch(seed, rows=64):
    """Random bf16 rows (normal, coarse-grid ties, peaked) with mixed greedy
    and sampled params, top_k in [1, NUM_BLOCKS // 2]."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, VOCAB, generator=g) * 3
    logits[: rows // 4] = (
        torch.randint(-6, 7, (rows // 4, VOCAB), generator=g).float() * 0.5
    )
    hot = torch.randint(0, VOCAB, (rows, 6), generator=g)
    logits[rows // 2 :].scatter_(1, hot[rows // 2 :], 12.0)
    temps = torch.tensor([0.0, 0.6, 1.0, 1.3])[
        torch.randint(0, 4, (rows,), generator=g)
    ]
    top_k = torch.randint(
        1, NUM_BLOCKS // 2 + 1, (rows, 1), generator=g, dtype=torch.int32
    )
    top_p = torch.tensor([1.0, 0.95, 0.8, 0.5])[
        torch.randint(0, 4, (rows,), generator=g)
    ]
    u = torch.rand(rows, VOCAB, generator=g)
    return (
        logits.to(torch.bfloat16),
        temps[:, None].to(torch.bfloat16),
        u,
        top_k,
        top_p[:, None],
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("path", list(SAMPLERS))
def test_flag_off_is_the_previous_sampler(monkeypatch, path):
    monkeypatch.setattr(tr, "_SAMPLER_BONUS_F32", False)
    for seed in range(4):
        args = _mixed_batch(seed)
        assert torch.equal(SAMPLERS[path](*args), OLD[path](*args)), (path, seed)


@pytest.mark.cpu_test
@pytest.mark.parametrize("path", list(SAMPLERS))
def test_greedy_rows_are_bitwise_unchanged(monkeypatch, path):
    for seed in range(4):
        args = _mixed_batch(seed)
        greedy = (args[1][:, 0] == 0).nonzero().view(-1)
        assert greedy.numel() > 0
        monkeypatch.setattr(tr, "_SAMPLER_BONUS_F32", False)
        off = SAMPLERS[path](*args)
        monkeypatch.setattr(tr, "_SAMPLER_BONUS_F32", True)
        on = SAMPLERS[path](*args)
        assert torch.equal(on[greedy], off[greedy]), (path, seed)
        assert torch.equal(on[greedy].view(-1), args[0][greedy].float().argmax(-1))


@pytest.mark.cpu_test
def test_flag_is_registered_and_default_off(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.delenv("VLLM_TPU_SAMPLER_BONUS_F32", raising=False)
    assert envs.environment_variables["VLLM_TPU_SAMPLER_BONUS_F32"]() is False
    monkeypatch.setenv("VLLM_TPU_SAMPLER_BONUS_F32", "1")
    assert envs.environment_variables["VLLM_TPU_SAMPLER_BONUS_F32"]() is True
