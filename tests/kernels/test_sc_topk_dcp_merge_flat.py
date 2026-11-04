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
"""TPU_SC_TOPK_DCP_MERGE_FLAT: the DCP top-k merge over flat, rank-major
candidate scores (no 32-row pad, no 2-D relayout) is bitwise identical to the
padded `[b, dcp * k]` merge. Through `streamindex_topk_dcp`: see
test_dsv4_streamindex_topk_dcp.py::test_flat_merge_*."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
    rank_major_merge_fits,
    sparsecore_topk_dcp_merge,
)

pytestmark = pytest.mark.tpu_test


def _require_sparsecore():
    try:
        if pltpu.get_tpu_info().sparse_core is not None:
            return
    except Exception:  # noqa: S110
        pass
    pytest.skip("needs SparseCore")


def _merge_case(b, dcp, k, seed):
    """Per-rank candidate lists as `streamindex_topk_dcp` hands them over:
    f32 score bits with -inf tails (unfilled slots), heavy ties (a coarse value
    grid, so the threshold bucket is ranked), and local slots with -1 holes."""
    rng = np.random.default_rng(seed)
    vals = rng.integers(-64, 64, size=(dcp, b, k)).astype(np.float32) / 8.0
    fill = rng.integers(k // 4, k + 1, size=(dcp, b))
    vals[np.arange(k)[None, None, :] >= fill[..., None]] = -np.inf
    if b > 1:
        vals[:, 1, :] = -np.inf  # a row with no candidates anywhere
    bits = vals.view(np.int32)
    idxs = rng.integers(0, 1 << 20, size=(b, k)).astype(np.int32)
    idxs[rng.random((b, k)) < 0.1] = -1
    scores_2d = np.concatenate(list(bits), axis=1)  # [b, dcp * k], rank-major cols
    scores_flat = bits.reshape(-1)  # [dcp * b * k], what the flat all-gather makes
    return scores_2d, scores_flat, idxs


def test_rank_major_gate_is_one_wave():
    assert rank_major_merge_fits(1) and rank_major_merge_fits(32)
    assert not rank_major_merge_fits(33) and not rank_major_merge_fits(64)


@pytest.mark.parametrize("b", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("dcp,k", [(8, 2048), (16, 2048), (2, 256), (8, 512)])
def test_flat_merge_is_bitwise_identical(b, dcp, k):
    _require_sparsecore()
    scores_2d, scores_flat, idxs = _merge_case(b, dcp, k, seed=b * 131 + dcp + k)
    for rank in sorted({0, dcp // 2, dcp - 1}):
        r = jnp.int32(rank)
        want = sparsecore_topk_dcp_merge(
            jnp.asarray(scores_2d), jnp.asarray(idxs), k, dcp_rank=r, dcp_size=dcp
        )
        got = sparsecore_topk_dcp_merge(
            jnp.asarray(scores_flat),
            jnp.asarray(idxs),
            k,
            dcp_rank=r,
            dcp_size=dcp,
            rank_major=True,
        )
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        assert (np.asarray(want) >= 0).any(), "case selects nothing: vacuous"


def test_flat_merge_comparison_detects_a_wrong_row_order():
    """Positive control: the row-major flatten of `[b, dcp * k]` (the order the
    kernel must NOT assume) gives a different answer, so the test can fail."""
    _require_sparsecore()
    b, dcp, k = 4, 8, 2048
    scores_2d, _, idxs = _merge_case(b, dcp, k, seed=7)
    want = sparsecore_topk_dcp_merge(
        jnp.asarray(scores_2d),
        jnp.asarray(idxs),
        k,
        dcp_rank=jnp.int32(3),
        dcp_size=dcp,
    )
    bad = sparsecore_topk_dcp_merge(
        jnp.asarray(scores_2d.reshape(-1)),
        jnp.asarray(idxs),
        k,
        dcp_rank=jnp.int32(3),
        dcp_size=dcp,
        rank_major=True,
    )
    assert (np.asarray(bad) != np.asarray(want)).any()


def test_flat_merge_rejects_a_two_wave_batch():
    with pytest.raises(ValueError):
        sparsecore_topk_dcp_merge(
            jnp.zeros((8 * 64 * 256,), jnp.int32),
            jnp.zeros((64, 256), jnp.int32),
            256,
            dcp_rank=jnp.int32(0),
            dcp_size=8,
            rank_major=True,
        )


def test_flat_merge_hlo_has_no_padded_operand():
    """The point of the change: no 32-row operand reaches the merge call."""
    _require_sparsecore()
    # b = 2 keeps the flat operand (dcp * b * k) apart from 32 * k.
    b, dcp, k = 2, 8, 2048
    _, scores_flat, idxs = _merge_case(b, dcp, k, seed=11)
    fn = jax.jit(
        lambda s, i: sparsecore_topk_dcp_merge(
            s, i, k, dcp_rank=jnp.int32(0), dcp_size=dcp, rank_major=True
        )
    )
    text = fn.lower(jnp.asarray(scores_flat), jnp.asarray(idxs)).compile().as_text()
    assert f"s32[{32 * dcp * k}]" not in text
    assert f"s32[{32 * k}]" not in text
    assert re.search(rf"s32\[{dcp * b * k}\]", text)
