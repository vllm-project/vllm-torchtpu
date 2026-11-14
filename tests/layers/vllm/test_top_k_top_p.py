# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.vllm.sample import top_k_top_p as sampling


def test_int32_search_boundaries():
    thresholds = torch.tensor([[-2147483648, -37, -1], [0, 128, 2147483647]],
                              dtype=torch.int32)
    result = sampling._int32_bsearch(thresholds.shape,
                                     lambda x: x > thresholds,
                                     thresholds.device)
    torch.testing.assert_close(result, thresholds)


def test_float32_search_boundaries():
    maximum = torch.finfo(torch.float32).max
    thresholds = torch.tensor([-maximum, -10.5, -1.0, 0.0, 3.25, maximum])
    result = sampling._float32_bsearch(thresholds.shape,
                                       lambda x: x > thresholds,
                                       thresholds.device)
    torch.testing.assert_close(result,
                               thresholds,
                               rtol=0,
                               atol=torch.finfo(torch.float32).tiny)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_unscaled", [False, True])
@pytest.mark.parametrize("shape", [(5, ), (2, 5), (2, 3, 5)])
@pytest.mark.parametrize("k", [1, 2, 5])
def test_topk_matches_sorted_cutoff(dtype, with_unscaled, shape, k):
    base = torch.tensor([-4., 3., 1., 3., -2.], dtype=dtype).expand(shape)
    if with_unscaled and dtype == torch.bfloat16:
        unscaled = (base * 1.5).to(torch.bfloat16)
        logits = (unscaled / 1.5).to(torch.bfloat16)
    else:
        unscaled = None
        logits = base
    cutoff = logits.sort(dim=-1).values[..., -k, None]
    expected = torch.where(logits >= cutoff, logits,
                           torch.tensor(-999., dtype=dtype))
    result = sampling.topk_mask(logits,
                                torch.full((*shape[:-1], 1), k),
                                -999.,
                                unscaled_logits=unscaled)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("p, keep", [(0.1, [True, False, False, False, False]),
                                     (0.65, [True, True, True, False, False]),
                                     (1.0, [True, True, True, True, True])])
def test_topp_includes_cutoff_ties(dtype, p, keep):
    logits = torch.log(torch.tensor([.5, .2, .2, .08, .02])).to(dtype)
    expected = torch.where(torch.tensor(keep), logits,
                           torch.tensor(-999., dtype=dtype))
    result = sampling.topp_mask(logits, torch.tensor([p]), -999.)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_unscaled", [False, True])
def test_apply_top_k_top_p_per_request(dtype, with_unscaled):
    probabilities = np.array([[.40, .30, .20, .10], [.40, .30, .20, .10],
                              [.40, .30, .20, .10], [.40, .30, .20, .10]],
                             dtype=np.float32)
    base_logits = torch.from_numpy(np.log(probabilities)).to(dtype)
    if with_unscaled and dtype == torch.bfloat16:
        unscaled = (base_logits * 0.8).to(torch.bfloat16)
        logits = (unscaled / 0.8).to(torch.bfloat16)
    else:
        unscaled = None
        logits = base_logits
    top_k = torch.tensor([[0], [2], [-1], [3]])
    top_p = torch.tensor([[1.0], [1.0], [.65], [.5]])
    # In row 3, top-k renormalizes .4/.3/.2; top-p needs 2 tokens to reach .5.
    keep = torch.tensor([[True, True, True, True], [True, True, False, False],
                         [True, True, False, False],
                         [True, True, False, False]])
    expected = torch.where(
        keep, logits, torch.tensor(sampling.MASKED_LOGIT_VALUE, dtype=dtype))
    original = logits.clone()
    result = sampling.apply_top_k_top_p(logits,
                                        top_k,
                                        top_p,
                                        unscaled_logits=unscaled)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(logits, original)


@pytest.mark.parametrize("with_unscaled", [False, True])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.5])
def test_bf16_topk_edge_cases_match_fp32_mask(with_unscaled, temperature):
    """Covers ties, -inf/NaN rows, and k > vocab on the 16-pass bf16 top-k
    path (with and without unscaled_logits)."""
    raw = torch.tensor(
        [
            [2.0, 2.0, 2.0, -1.0, -3.0, 0.5],  # ties at cutoff
            [1.0, -float("inf"), -float("inf"), 3.0, -2.0, 0.0
             ],  # -inf entries
            [-float("inf"), -float("inf"), 0.25, -float("inf"), -1.0, 0.25],
            [float("nan"), 4.0, 1.0, -1.0, 2.0, 0.0],  # NaN entry
            [0.5, -0.5, 1.5, 2.5, -1.5, 3.5],  # k > vocab row
        ],
        dtype=torch.bfloat16,
    )
    unscaled = raw if with_unscaled else None
    logits = (raw / temperature).to(torch.bfloat16)
    k = torch.tensor([[2], [3], [4], [2], [10]], dtype=torch.int32)

    got = sampling.topk_mask(logits, k, -999.0, unscaled_logits=unscaled)
    ref = sampling.topk_mask(logits.to(torch.float32), k, -999.0)
    got_keep = got != torch.tensor(-999.0, dtype=torch.bfloat16)
    ref_keep = ref != -999.0
    assert torch.equal(got_keep, ref_keep)


@pytest.mark.parametrize("p_val", [0.1, 0.5, 0.95, 0.9921875, 0.999, 1.0])
def test_bf16_topp_matches_fp32_mask_on_bf16_probs(p_val):
    """Covers ties, -inf entries, and p near 1 on the 14-pass bf16 top-p
    path."""
    logits = torch.tensor(
        [
            [3.0, 3.0, 1.0, 0.0, -2.0, -5.0],
            [2.0, 1.5, 1.0, -float("inf"), -float("inf"), 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    p = torch.full((logits.shape[0], 1), p_val, dtype=torch.float32)
    got = sampling.topp_mask(logits, p, -999.0)
    probs = torch.softmax(logits, dim=-1)
    zero_val = torch.zeros_like(probs)

    def predicate(threshold: torch.Tensor) -> torch.Tensor:
        mass = torch.sum(
            torch.where(probs >= threshold.unsqueeze(-1), probs, zero_val),
            dim=-1,
        )
        return mass < p.squeeze(-1)

    ref_thresh = sampling._float32_bsearch(logits.shape[:-1], predicate,
                                           logits.device).unsqueeze(-1)
    ref_keep = probs >= ref_thresh
    got_keep = got != torch.tensor(-999.0, dtype=torch.bfloat16)
    assert torch.equal(got_keep, ref_keep)
