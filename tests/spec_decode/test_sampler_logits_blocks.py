# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_SAMPLER_LOGITS_BLOCKS: the fused one-draw verify with the logits
block path (modes 1 and 2) returns bit-identical tokens and read offsets to
the default candidates-only path, on the same flat uniforms."""

import pytest
import torch

from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    CANDIDATE_BLOCK,
    apply_top_k_top_p,
    apply_top_k_top_p_kmax,
    topk_cutoff_mask,
    topk_mask,
)
from vllm_torchtpu.runner.tpu_runner import (
    _verify_fused_one_draw_impl,
    _verify_uniforms_count,
)

K = 3


def _case(seed, num_reqs, pad_reqs, blocks, family, num_cand):
    g = torch.Generator().manual_seed(seed)
    vocab = blocks * CANDIDATE_BLOCK
    rows = pad_reqs * (K + 1)
    if family == "normal":
        logits = torch.randn(rows, vocab, generator=g) * 4
    elif family == "grid":
        # Coarse bf16 grid: many exact ties, including at block maxima.
        logits = torch.randint(-6, 7, (rows, vocab), generator=g).float() * 0.5
    elif family == "peaked":
        logits = torch.randn(rows, vocab, generator=g)
        hot = torch.randint(0, vocab, (rows, 4), generator=g)
        logits.scatter_(1, hot, 12.0)
    else:  # zeros: signed zeros and ties everywhere
        logits = torch.randint(-1, 1, (rows, vocab), generator=g).float() * 0.0
        logits[:, ::7] = -0.0
    logits = logits.to(torch.bfloat16)
    target_idx = torch.zeros(pad_reqs * K, dtype=torch.int32)
    seg = torch.full((pad_reqs * K,), num_reqs, dtype=torch.int64)
    grp = torch.zeros(pad_reqs * K, dtype=torch.int32)
    draft = torch.full((pad_reqs * K,), -1, dtype=torch.int32)
    d = torch.randint(0, K + 1, (num_reqs,), generator=g)
    pos = 0
    for r in range(num_reqs):
        for j in range(int(d[r])):
            row = r * (K + 1) + j
            target_idx[pos] = row
            seg[pos] = r
            grp[pos] = j
            if torch.rand(1, generator=g).item() < 0.6:
                draft[pos] = int(logits[row].float().argmax())
            else:
                draft[pos] = int(torch.randint(0, vocab, (1,), generator=g))
            pos += 1
    bonus_idx = torch.zeros(pad_reqs, dtype=torch.int32)
    bonus_idx[:num_reqs] = torch.arange(num_reqs, dtype=torch.int32) * (K + 1) + K
    draft_lengths = torch.zeros(pad_reqs, dtype=torch.int32)
    draft_lengths[:num_reqs] = d.to(torch.int32)
    kmax = num_cand // 2
    temps = torch.ones(pad_reqs, 1, dtype=torch.bfloat16)
    top_k = torch.zeros(pad_reqs, 1, dtype=torch.int32)
    top_p = torch.ones(pad_reqs, 1, dtype=torch.float32)
    temps[:num_reqs, 0] = torch.tensor(
        [[0.0, 0.6, 1.0, 1.3][int(x)] for x in torch.randint(0, 4, (num_reqs,), generator=g)]
    ).to(torch.bfloat16)
    top_k[:num_reqs, 0] = torch.randint(1, kmax + 1, (num_reqs,), generator=g).to(torch.int32)
    top_p[:num_reqs, 0] = torch.tensor(
        [[1.0, 0.95, 0.8, 0.5][int(x)] for x in torch.randint(0, 4, (num_reqs,), generator=g)]
    )
    n = _verify_uniforms_count(pad_reqs, vocab, tuple(draft.shape), target_idx.shape[0])
    u = torch.rand(n, generator=g)
    return (
        logits,
        bonus_idx,
        target_idx,
        temps,
        top_k,
        top_p,
        u,
        draft,
        draft_lengths,
        seg,
        grp,
        None,
        K,
        False,
        num_cand,
        True,
        torch.tensor(num_reqs),
    )


@pytest.mark.parametrize("family", ["normal", "grid", "peaked", "zeros"])
@pytest.mark.parametrize("mode", [1, 2, 3])
def test_logits_blocks_bitwise(family, mode):
    for seed in range(12):
        num_reqs = 1 + seed % 3
        args = _case(seed, num_reqs, 4, 40, family, 16)
        ref_tokens, ref_off = _verify_fused_one_draw_impl(*args)
        tokens, off = _verify_fused_one_draw_impl(*args, mode)
        assert torch.equal(ref_tokens, tokens), (family, mode, seed)
        assert torch.equal(ref_off, off), (family, mode, seed)


def test_logits_blocks_b8_shape():
    # 8 padded requests, 32-wide candidates, more blocks.
    for seed in range(4):
        args = _case(100 + seed, 6, 8, 80, "normal", 32)
        ref = _verify_fused_one_draw_impl(*args)
        for mode in (1, 2, 3):
            out = _verify_fused_one_draw_impl(*args, mode)
            assert all(torch.equal(a, b) for a, b in zip(ref, out))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_topk_cutoff_mask_matches_bisection(dtype):
    g = torch.Generator().manual_seed(0)
    for trial in range(20):
        x = (torch.randint(-8, 8, (6, 512), generator=g).float() * 0.25).to(dtype)
        if trial % 2:
            x = torch.randn(6, 512, generator=g).to(dtype)
        k = torch.randint(1, 33, (6, 1), generator=g)
        ref = topk_mask(x, k)
        out = topk_cutoff_mask(x, k, 32)
        assert torch.equal(ref, out)
        assert ref.dtype == out.dtype


def test_mutation_is_caught(monkeypatch):
    # Positive control: gathering the candidate blocks of the wrong source row
    # must change some tokens.
    import vllm_torchtpu.runner.tpu_runner as runner

    real = runner.row_block_gather

    def wrong_row(blocks_flat, blocks_per_row, rows, block_ids):
        return real(blocks_flat, blocks_per_row, (rows + 1) % 4, block_ids)

    refs = [
        _verify_fused_one_draw_impl(*_case(s, 3, 4, 40, "peaked", 16))[0]
        for s in range(12)
    ]
    monkeypatch.setattr(runner, "row_block_gather", wrong_row)
    diffs = sum(
        int(not torch.equal(ref, _verify_fused_one_draw_impl(*_case(s, 3, 4, 40, "peaked", 16), 1)[0]))
        for s, ref in enumerate(refs)
    )
    assert diffs > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_topp_narrow_matches_bisection(dtype):
    """mode 3's top-p search returns the bisection's mask, including p close
    to 1 (the whole kept mass below p) and tie-heavy rows."""
    g = torch.Generator().manual_seed(1)
    ps = [1e-6, 0.3, 0.5, 0.8, 0.95, 0.999, 0.9999999, 1.0]
    for trial in range(40):
        rows = 8
        if trial % 3 == 0:
            x = torch.randint(-6, 7, (rows, 1024), generator=g).float() * 0.5
        elif trial % 3 == 1:
            x = torch.randn(rows, 1024, generator=g) * (1 + trial % 5)
        else:
            x = torch.randn(rows, 1024, generator=g)
            x[:, :3] = 20.0
        x = x.to(dtype)
        k = torch.randint(1, 33, (rows, 1), generator=g)
        p = torch.tensor([[ps[int(i)]] for i in torch.randint(0, len(ps), (rows,), generator=g)])
        ref = apply_top_k_top_p(x, k, p)
        out = apply_top_k_top_p_kmax(x, k, p, 32, topp_narrow=True)
        assert torch.equal(ref, out), trial


def test_topp_narrow_keep_all_rows():
    # bf16 probabilities whose kept mass rounds below p: the bisection keeps
    # every top-k entry; the narrowed search must too.
    g = torch.Generator().manual_seed(3)
    for _ in range(50):
        x = (torch.randn(8, 1024, generator=g) * 2).to(torch.bfloat16)
        k = torch.full((8, 1), 32)
        p = torch.full((8, 1), 0.9999999)
        assert torch.equal(
            apply_top_k_top_p(x, k, p), apply_top_k_top_p_kmax(x, k, p, 32, True)
        )


@pytest.mark.parametrize(
    "lb,narrow,want", [(0, False, 0), (0, True, 0), (1, True, 1), (2, False, 2), (2, True, 3)]
)
def test_runner_mode(monkeypatch, lb, narrow, want):
    from vllm_torchtpu import envs
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    monkeypatch.setattr(envs, "VLLM_TPU_SAMPLER_LOGITS_BLOCKS", lb, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_SAMPLER_TOPP_NARROW", narrow, raising=False)
    assert TPUModelRunner._sampler_logits_blocks() == want


def test_ro_lb_impl_matches_ro_impl():
    from vllm_torchtpu.runner.tpu_runner import (
        _verify_one_draw_ro_impl,
        _verify_one_draw_ro_lb_impl,
    )

    args = _case(7, 3, 4, 40, "peaked", 16)[:-1]
    slots = torch.arange(4, dtype=torch.int32)
    for mode in (1, 2, 3):
        ro_a = torch.zeros(8, dtype=torch.int32)
        ro_b = torch.zeros(8, dtype=torch.int32)
        n = torch.tensor(3)
        out_a = _verify_one_draw_ro_impl(ro_a, [slots], n, *args)
        out_b = _verify_one_draw_ro_lb_impl(ro_b, [slots], n, mode, *args)
        assert torch.equal(out_a, out_b) and torch.equal(ro_a, ro_b)
