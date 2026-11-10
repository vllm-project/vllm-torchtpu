# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter.sample import top_k_top_p as sampling
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def test_int32_search_boundaries():
    thresholds = torch.tensor(
        [[-2147483648, -37, -1], [0, 128, 2147483647]], dtype=torch.int32
    )
    result = sampling._int32_bsearch(
        thresholds.shape, lambda x: x > thresholds, thresholds.device
    )
    torch.testing.assert_close(result, thresholds)


def test_float32_search_boundaries():
    maximum = torch.finfo(torch.float32).max
    thresholds = torch.tensor([-maximum, -10.5, -1.0, 0.0, 3.25, maximum])
    result = sampling._float32_bsearch(
        thresholds.shape, lambda x: x > thresholds, thresholds.device
    )
    torch.testing.assert_close(
        result, thresholds, rtol=0, atol=torch.finfo(torch.float32).tiny
    )


@pytest.mark.parametrize("shape", [(5,), (2, 5), (2, 3, 5)])
@pytest.mark.parametrize("k", [1, 2, 5])
def test_topk_matches_sorted_cutoff(shape, k):
    logits = torch.tensor([-4.0, 3.0, 1.0, 3.0, -2.0]).expand(shape)
    cutoff = logits.sort(dim=-1).values[..., -k, None]
    expected = torch.where(logits >= cutoff, logits, -999.0)
    result = sampling.topk_mask(logits, torch.full((*shape[:-1], 1), k), -999.0)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "p, keep",
    [
        (0.1, [True, False, False, False, False]),
        (0.65, [True, True, True, False, False]),
        (1.0, [True, True, True, True, True]),
    ],
)
def test_topp_includes_cutoff_ties(p, keep):
    logits = torch.log(torch.tensor([0.5, 0.2, 0.2, 0.08, 0.02]))
    expected = torch.where(torch.tensor(keep), logits, -999.0)
    result = sampling.topp_mask(logits, torch.tensor([p]), -999.0)
    torch.testing.assert_close(result, expected)


def test_apply_top_k_top_p_per_request():
    probabilities = np.array(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.40, 0.30, 0.20, 0.10],
            [0.40, 0.30, 0.20, 0.10],
            [0.40, 0.30, 0.20, 0.10],
        ],
        dtype=np.float32,
    )
    logits = torch.from_numpy(np.log(probabilities))
    top_k = torch.tensor([[0], [2], [-1], [3]])
    top_p = torch.tensor([[1.0], [1.0], [0.65], [0.5]])
    # In row 3, top-k renormalizes .4/.3/.2; top-p needs two tokens to reach .5.
    keep = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [True, True, False, False],
            [True, True, False, False],
        ]
    )
    expected = torch.where(keep, logits, sampling.MASKED_LOGIT_VALUE)
    original = logits.clone()
    result = sampling.apply_top_k_top_p(logits, top_k, top_p)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(logits, original)


def test_candidate_blocks_match_full_vocab_for_top_k_sampling():
    gen = torch.Generator().manual_seed(0)
    logits = (torch.randn(6, 128 * 400, generator=gen) * 3).bfloat16().float() / 0.6
    top_k = torch.full((6, 1), 20, dtype=torch.int32)
    top_p = torch.full((6, 1), 0.95)
    expected = sampling.apply_top_k_top_p(logits, top_k, top_p)
    result = sampling.apply_top_k_top_p_candidates(logits, top_k, top_p, 64)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_candidate_blocks_keep_ties_within_the_documented_bound():
    # 19 distinct large logits plus 13 ties at the 20th value, one per block:
    # 32 logits >= the 20th largest, exactly num_blocks.
    num_blocks, k = 32, 20
    logits = torch.full((1, 128 * 100), -5.0)
    for i in range(k - 1):
        logits[0, 128 * i] = 10.0 + i
    for j in range(num_blocks - k + 1):
        logits[0, 128 * (k - 1 + j) + 7] = 9.0
    top_k = torch.tensor([[k]], dtype=torch.int32)
    for p in (1.0, 0.9):
        top_p = torch.tensor([[p]])
        expected = sampling.apply_top_k_top_p(logits, top_k, top_p)
        result = sampling.apply_top_k_top_p_candidates(logits, top_k, top_p, num_blocks)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_candidate_blocks_leave_rows_without_top_k_unmasked():
    gen = torch.Generator().manual_seed(1)
    logits = torch.randn(3, 128 * 50, generator=gen)
    top_k = torch.tensor([[0], [-1], [5]], dtype=torch.int32)
    top_p = torch.tensor([[1.0], [1.0], [0.9]])
    expected = sampling.apply_top_k_top_p(logits, top_k, top_p)
    result = sampling.apply_top_k_top_p_candidates(logits, top_k, top_p, 16)
    torch.testing.assert_close(result[:2], logits[:2], rtol=0, atol=0)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_topk_candidate_blocks_gating(monkeypatch):
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.vocab_size = 248320

    monkeypatch.setattr(envs, "TPU_SAMPLER_TOPK_CANDIDATES", 0)
    assert runner._topk_candidate_blocks() == 0

    monkeypatch.setattr(envs, "TPU_SAMPLER_TOPK_CANDIDATES", 64)
    runner.vocab_size = 248321
    assert runner._topk_candidate_blocks() == 0
    runner.vocab_size = 248320
    assert runner._topk_candidate_blocks() == 64


def test_sampler_topk_candidates_gating(monkeypatch):
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.vocab_size = 248320
    monkeypatch.setattr(envs, "TPU_SAMPLER_TOPK_CANDIDATES", 64)

    input_batch = SimpleNamespace(
        top_k_cpu_tensor=torch.tensor([20, 20, 20, 20], dtype=torch.int64)
    )
    runner.input_batch = input_batch

    assert runner._sampler_topk_candidates(0, 4, None) == 64
    assert runner._sampler_topk_candidates(0, 4, object()) == 0

    input_batch.top_k_cpu_tensor = torch.tensor([40, 20, 20, 20], dtype=torch.int64)
    assert runner._sampler_topk_candidates(0, 4, None) == 0

    input_batch.top_k_cpu_tensor = torch.tensor([0, 20, 20, 20], dtype=torch.int64)
    assert runner._sampler_topk_candidates(0, 4, None) == 0


def test_rejection_sampler_topk_candidates_cpu_parity():
    gen = torch.Generator().manual_seed(42)
    vocab_size = 128 * 400
    rows = 4
    logits = (torch.randn(rows, vocab_size, generator=gen) * 3).float() / 0.6
    top_k = torch.full((rows, 1), 20, dtype=torch.int32)
    top_p = torch.full((rows, 1), 0.95)

    expected = sampling.apply_top_k_top_p(logits, top_k, top_p, topk_candidates=0)
    result = sampling.apply_top_k_top_p(logits, top_k, top_p, topk_candidates=64)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


# --- TPU_SAMPLER_VOCAB_PARALLEL: radix bisection + static mask skips ---


@pytest.mark.parametrize("radix_bits", [2, 3, 4, 8])
def test_int32_search_radix_matches_bitwise(radix_bits):
    thresholds = torch.tensor(
        [[-2147483648, -37, -1, 5], [0, 128, 2147483647, -(1 << 30)]],
        dtype=torch.int32,
    )
    result = sampling._int32_bsearch(
        thresholds.shape, lambda x: x > thresholds, thresholds.device, radix_bits
    )
    torch.testing.assert_close(result, thresholds)


@pytest.mark.parametrize("radix_bits", [2, 4])
def test_float32_search_radix_boundaries(radix_bits):
    maximum = torch.finfo(torch.float32).max
    thresholds = torch.tensor([-maximum, -10.5, -1.0, 0.0, 3.25, maximum])
    result = sampling._float32_bsearch(
        thresholds.shape, lambda x: x > thresholds, thresholds.device, radix_bits
    )
    torch.testing.assert_close(
        result, thresholds, rtol=0, atol=torch.finfo(torch.float32).tiny
    )


def _random_rows(rows=6, vocab=4099, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, vocab, generator=g) * 3.0
    logits[1, :50] = logits[1, 50]  # ties across the cutoff
    top_k = torch.tensor([[0], [7], [-1], [100], [1], [vocab - 1]])[:rows]
    top_p = torch.tensor([[1.0], [0.95], [0.5], [0.9], [1.0], [0.3]])[:rows]
    return logits, top_k, top_p


@pytest.mark.parametrize("radix_bits", [2, 3, 4])
def test_apply_top_k_top_p_radix_is_bit_identical(radix_bits):
    logits, top_k, top_p = _random_rows()
    expected = sampling.apply_top_k_top_p(logits, top_k, top_p)
    result = sampling.apply_top_k_top_p(logits, top_k, top_p, radix_bits=radix_bits)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("radix_bits", [1, 2])
def test_apply_top_k_top_p_skips_are_identity(radix_bits):
    logits, _, _ = _random_rows()
    rows = logits.shape[0]
    no_k = torch.zeros((rows, 1), dtype=torch.int64)
    no_p = torch.ones((rows, 1))
    some_k = torch.tensor([[0], [7], [0], [100], [1], [0]])
    some_p = torch.tensor([[1.0], [0.95], [1.0], [0.9], [1.0], [0.3]])
    for k, p, need_k, need_p in [
        (no_k, some_p, False, True),
        (some_k, no_p, True, False),
        (no_k, no_p, False, False),
    ]:
        expected = sampling.apply_top_k_top_p(logits, k, p)
        result = sampling.apply_top_k_top_p(logits, k, p, need_k, need_p, radix_bits)
        assert torch.equal(result, expected)


def _fast_top_p_cases(rows, vocab, seed):
    """Random and adversarial [rows, vocab] logits for the fast top-p gate."""
    g = torch.Generator().manual_seed(seed)
    yield "randn2", torch.randn(rows, vocab, generator=g) * 2.0
    yield "randn6", torch.randn(rows, vocab, generator=g) * 6.0
    hot = torch.randn(rows, vocab, generator=g) * 2.0
    idx = torch.randint(0, vocab, (rows, 8), generator=g)
    yield "spiky", hot.scatter_add(1, idx, torch.rand(rows, 8, generator=g) * 12.0)
    x = torch.full((rows, vocab), -5.0)
    x[:, :10] = 5.0
    yield "ties10", x
    x = torch.full((rows, vocab), -5.0)
    x[:, :2000] = 3.0
    yield "wide_ties", x
    yield "flat", torch.zeros(rows, vocab)
    yield "rounded", torch.round(torch.randn(rows, vocab, generator=g) * 4.0)
    # Top-n mass landing on p: n leaders at logit a over a flat tail.
    n = torch.arange(1, rows + 1).view(-1, 1)
    a = torch.log(0.95 / 0.05 * (vocab - n) / n)
    x = torch.zeros(rows, vocab)
    x[:, : rows + 1] = torch.where(torch.arange(rows + 1) < n, a, 0.0)
    yield "mass_on_p", x


_FAST_P = [0.95, 0.9, 0.5, 0.99, 0.1, 1.0, 0.95, 0.7, 1e-6, 0.999999] * 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fast_top_p_is_bit_identical(dtype):
    """fast_top_p skips only bisection steps proven idle: same masked logits."""
    rows, vocab = 20, 4096
    top_p = torch.tensor(_FAST_P).view(-1, 1)
    for name, x in _fast_top_p_cases(rows, vocab, seed=5):
        logits = x.to(dtype)
        want = sampling.topp_mask(logits, top_p)
        got = sampling.topp_mask(logits, top_p, fast_top_p=True)
        assert torch.equal(got, want), (str(dtype), name)
        k = torch.zeros((rows, 1), dtype=torch.int32)
        want = sampling.apply_top_k_top_p(logits, k, top_p, False, True)
        got = sampling.apply_top_k_top_p(logits, k, top_p, False, True, 1, True)
        assert torch.equal(got, want), (str(dtype), name)


def test_skip_bits_matches_full_search_when_idle():
    # A threshold whose skipped bits are zero is found in fewer rounds.
    thresholds = torch.tensor([0, 1 << 16, 7 << 20, (1 << 30) - (1 << 16)])
    full = sampling._int32_bsearch(
        thresholds.shape, lambda x: x > thresholds, thresholds.device
    )
    fast = sampling._int32_bsearch(
        thresholds.shape,
        lambda x: x > thresholds,
        thresholds.device,
        skip_bits=(1 << 30) | 0xFFFF,
    )
    torch.testing.assert_close(fast, full)


@pytest.mark.tpu_test
# Dynamo's own patch restores the limit; monkeypatch's undo hides the key
# for later tests. cache_size_limit is an alias of recompile_limit.
@torch._dynamo.config.patch(recompile_limit=64)
def test_fast_top_p_matches_bisection(device):
    """On TPU, compiled like the sampler graphs: f32 rows like the verify
    program and bf16 rows like the bonus program."""
    if device.type != "tpu":
        pytest.skip("compiled TPU check")

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def mask(logits, top_k, top_p, fast):
        return sampling.apply_top_k_top_p(logits, top_k, top_p, False, True, 1, fast)

    rows, vocab = 20, 154880
    top_k = torch.zeros((rows, 1), dtype=torch.int32, device=device)
    top_p = torch.tensor(_FAST_P, device=device).view(-1, 1)
    for dtype in (torch.float32, torch.bfloat16):
        for name, x in _fast_top_p_cases(rows, vocab, seed=3):
            logits = x.to(dtype).to(device)
            want = mask(logits, top_k, top_p, False).cpu()
            got = mask(logits, top_k, top_p, True).cpu()
            assert torch.equal(got, want), (str(dtype), name)
