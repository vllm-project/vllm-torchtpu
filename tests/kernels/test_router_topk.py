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
"""Pallas router top-k kernel vs its torch reference.

``interpret=True`` runs the kernel body on the host, so block splitting, the
clamp and the tie rule are gated without a TPU. It does not gate Mosaic
lowering.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax import lax

from vllm_torchtpu.kernels.router_topk import (
    MAX_BLOCK_ROWS,
    select,
    select_with_bias,
    select_with_raw,
)
from vllm_torchtpu.layers.core.rowmax_topk import rowmax_topk

EXPERTS = 512
TOPK = 10


def _both(scores_np):
    kw, ki = select(jnp.asarray(scores_np), TOPK, interpret=True)
    rw, ri = rowmax_topk(torch.from_numpy(scores_np), TOPK)
    return (np.asarray(kw), np.asarray(ki)), (rw.numpy(), ri.numpy())


@pytest.mark.parametrize("rows", [8, 64, 256, 512, 1024, 1536])
def test_kernel_matches_torch_reference(rows):
    rng = np.random.default_rng(rows)
    scores = rng.random((rows, EXPERTS), dtype=np.float32)
    (kw, ki), (rw, ri) = _both(scores)
    np.testing.assert_array_equal(kw, rw)
    np.testing.assert_array_equal(ki, ri)


@pytest.mark.parametrize("rows", [17, 33, 250, 1000, 1040, 4384])
def test_partial_last_block_does_not_leak(rows):
    """Row counts that are not a multiple of the block stay exact."""
    rng = np.random.default_rng(rows)
    scores = rng.random((rows, EXPERTS), dtype=np.float32)
    (kw, ki), (rw, ri) = _both(scores)
    assert kw.shape == ki.shape == (rows, TOPK)
    np.testing.assert_array_equal(kw, rw)
    np.testing.assert_array_equal(ki, ri)
    # A leak from the surplus region shows up as an impossible expert or a
    # sentinel weight first.
    assert ((ki >= 0) & (ki < EXPERTS)).all()
    assert np.isfinite(kw).all()


def test_kernel_matches_torch_topk_on_real_routing_distribution():
    rng = np.random.default_rng(99)
    logits = torch.from_numpy(
        rng.standard_normal((1024, EXPERTS), dtype=np.float32) * 2.0
    ).to(torch.bfloat16)
    scores = logits.float().softmax(dim=-1)
    kw, ki = select(jnp.asarray(scores.numpy()), TOPK, interpret=True)
    ref_v, _ = torch.topk(scores, TOPK, dim=-1)
    np.testing.assert_array_equal(np.asarray(kw), ref_v.numpy())
    gathered = torch.gather(scores, 1, torch.from_numpy(np.asarray(ki)).long())
    np.testing.assert_array_equal(gathered.numpy(), ref_v.numpy())


def test_all_nan_row_selects_k_distinct_experts():
    """The kernel half of the same property (see the torch reference test).

    An all-NaN row is the flush sentinel in every column; masking with that
    same sentinel keeps every column tied with the row max, so every pass
    returns column 0 and the row selects one expert ``TOPK`` times. Measured
    against the sort path at 512 experts / top-k 10, that is 1 distinct expert
    where ``lax.sort`` takes 10.
    """
    scores = np.full((3, EXPERTS), np.nan, dtype=np.float32)
    kw, ki = select(jnp.asarray(scores), TOPK, interpret=True)
    ki = np.asarray(ki)
    assert np.isnan(np.asarray(kw)).all()
    for row in ki:
        assert sorted(row.tolist()) == list(range(TOPK))
    iota = lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    _, sort_i = lax.sort(
        (-jnp.asarray(scores), iota), dimension=1, is_stable=False, num_keys=1
    )
    for got, ref in zip(ki, np.asarray(sort_i)[:, :TOPK]):
        assert len(set(got.tolist())) == len(set(ref.tolist())) == TOPK


def test_partial_nan_row_selects_k_distinct_finite_experts():
    rng = np.random.default_rng(5)
    scores = rng.random((2, EXPERTS), dtype=np.float32)
    scores[:, :5] = np.nan
    kw, ki = select(jnp.asarray(scores), TOPK, interpret=True)
    assert np.isfinite(np.asarray(kw)).all()
    for row in np.asarray(ki):
        assert len(set(row.tolist())) == TOPK
        assert min(row.tolist()) >= 5


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(np.nan, id="nan"),
        pytest.param(-np.inf, id="neg_inf"),
        pytest.param(np.finfo(np.float32).min, id="neg_flt_max"),
    ],
)
def test_row_at_or_below_the_sentinel_selects_k_distinct_experts(bad):
    """Kernel half of the same property, for all three spellings.

    ``NEG`` is above ``-inf`` and above ``-FLT_MAX``, so before the clamp a
    masked column outranked a genuine one and the row collapsed onto expert 0
    ``TOPK`` times -- measured at 1 distinct expert where ``lax.sort`` takes
    10. The torch twin is asserted alongside, because fixing one and not the
    other leaves the two agreeing with each other and wrong together.
    """
    scores = np.full((3, EXPERTS), bad, dtype=np.float32)
    kw, ki = select(jnp.asarray(scores), TOPK, interpret=True)
    assert np.isnan(np.asarray(kw)).all()
    for row in np.asarray(ki):
        assert sorted(row.tolist()) == list(range(TOPK))
    tw, ti = rowmax_topk(torch.from_numpy(scores), TOPK)
    assert np.array_equal(np.asarray(ki), ti.numpy())
    assert np.isnan(tw.numpy()).all()


def test_kernel_ties_and_nan_match_the_reference():
    scores = np.zeros((4, EXPERTS), dtype=np.float32)
    scores[0] = 1.0 / EXPERTS  # padded token: all tied
    scores[1, 200:212] = 4.0  # tie across the boundary
    scores[2] = np.nan  # fully poisoned row
    scores[3, :3] = np.nan  # partially poisoned
    scores[3, 300:310] = np.linspace(1.0, 0.1, 10)
    (kw, ki), (rw, ri) = _both(scores)
    np.testing.assert_array_equal(ki, ri)
    assert np.isnan(kw[2]).all() and np.isnan(rw[2]).all()
    keep = [0, 1, 3]
    np.testing.assert_array_equal(kw[keep], rw[keep])


@pytest.mark.parametrize(
    "rows,grid",
    [
        (4, 1),
        (250, 1),
        (512, 1),
        (1000, 2),
        (4384, 9),
        (8191, 16),
        (16384, 32),
    ],
)
def test_grid_rounds_up_over_the_tuned_block(rows, grid):
    block = min(MAX_BLOCK_ROWS, rows)
    assert -(-rows // block) == grid


@pytest.mark.parametrize("rows", [8, 250, 512, 1000, 1024, 2048, 4096])
def test_select_with_raw_and_bias_match_torch_reference(rows):
    rng = np.random.default_rng(rows + 123)
    raw = rng.random((rows, 384), dtype=np.float32)
    bias = (rng.random((1, 384), dtype=np.float32) - 0.5) * 0.2
    biased = raw + bias

    kw_raw, ki_raw = select_with_raw(
        jnp.asarray(biased), jnp.asarray(raw), 6, interpret=True
    )
    kw_bias, ki_bias = select_with_bias(
        jnp.asarray(raw), jnp.asarray(bias), 6, interpret=True
    )
    _, ri = rowmax_topk(torch.from_numpy(biased), 6)
    rw = np.take_along_axis(raw, ri.numpy(), axis=-1)

    np.testing.assert_array_equal(np.asarray(ki_raw), ri.numpy())
    np.testing.assert_array_equal(np.asarray(kw_raw), rw)
    np.testing.assert_array_equal(np.asarray(ki_bias), ri.numpy())
    np.testing.assert_array_equal(np.asarray(kw_bias), rw)


@pytest.mark.parametrize("rows", [8, 256, 512, 1024, 2048])
@pytest.mark.parametrize("scoring_fn", ["sqrtsoftplus", "sigmoid", "softmax"])
def test_fused_router_with_bias_matches_torch_reference(rows, scoring_fn):
    import torch.nn.functional as F

    rng = np.random.default_rng(rows + 456)
    logits = rng.standard_normal((rows, 384), dtype=np.float32) * 1.5
    bias = (rng.random((1, 384), dtype=np.float32) - 0.5) * 0.2
    scale = 2.5

    logits_t = torch.from_numpy(logits)
    if scoring_fn == "sqrtsoftplus":
        scores_t = torch.sqrt(F.softplus(logits_t))
    elif scoring_fn == "sigmoid":
        scores_t = logits_t.sigmoid()
    else:
        scores_t = logits_t.softmax(dim=-1)
    bias_t = torch.from_numpy(bias)
    rw_t, ri_t = rowmax_topk(scores_t + bias_t, 6, raw_scores=scores_t)
    rw_t = (rw_t / torch.clamp(rw_t.sum(dim=-1, keepdim=True), min=1e-20)) * scale

    kw, ki = select_with_bias(
        jnp.asarray(logits),
        jnp.asarray(bias),
        6,
        interpret=True,
        scoring_fn=scoring_fn,
        renormalize=True,
        routed_scaling_factor=scale,
    )
    np.testing.assert_array_equal(np.asarray(ki), ri_t.numpy())
    np.testing.assert_allclose(np.asarray(kw), rw_t.numpy(), rtol=1e-5, atol=1e-6)
