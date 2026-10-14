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
"""topk_cap composed with is_dcp_mode: the seam neither PR's tests cover."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.kernels.test_mla_native_sc_attention import NOPE_DIM, _build_problem
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla

pytestmark = pytest.mark.tpu_test


def _run(problem, topk_idx, real_tokens, chunk, **extra):
    return sparse_mla.sparse_ragged_paged_attention(
        problem["q"],
        problem["sparsecore_nope"],
        problem["sparsecore_rope"],
        topk_idx,
        problem["page_table"],
        jnp.array([0, real_tokens], jnp.int32),
        jnp.array([0, 1, 1], jnp.int32),
        sm_scale=0.125,
        k_scale=0.75,
        return_lse=True,
        cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
        gather_and_attention_chunk_size=chunk,
        attention_kernel_batch_size=16,
        **extra,
    )


# counts chosen so distinct batch groups land on distinct rungs of #1477's
# ladder (eighth/quarter/half/full of bkv_sz=topk_cap) while others overflow
# the cap and must hit #1525's fix-up.
_COUNTS = {
    "all_eighth": [1, 17, 33, 63],
    "all_quarter": [65, 90, 110, 128],
    "all_half": [129, 200, 240, 256],
    "mixed_rungs": [1, 63, 129, 256, 300, 511],
    "overflow_only": [513, 700, 1024, 2048],
    "mixed_all": [1, 63, 255, 256, 257, 511, 512, 513, 700, 1024, 1025, 2048, 130],
}


@pytest.mark.parametrize("name", sorted(_COUNTS))
@pytest.mark.parametrize("topk_cap", (256, 512))
def test_topk_cap_with_dcp_mode_matches_uncapped(name, topk_cap):
    if jax.devices()[0].platform != "tpu":
        pytest.skip("SparseCore cache-layout attention requires a TPU")
    topk, query_tokens, chunk = 2048, 128, 64
    real_tokens = 100
    rng = np.random.default_rng(20261005)
    problem = _build_problem(rng, k_scale=0.75, query_tokens=query_tokens, topk=topk)
    counts = np.resize(np.array(_COUNTS[name]), query_tokens)
    rng.shuffle(counts)
    live = np.arange(topk)[None, :] < counts[:, None]
    topk_idx = jnp.where(jnp.asarray(live), problem["topk"], -1)

    # reference: uncapped, no DCP — the path both PRs agree on
    ref, ref_lse = jax.block_until_ready(_run(problem, topk_idx, real_tokens, chunk))
    out, lse = jax.block_until_ready(
        _run(problem, topk_idx, real_tokens, chunk, topk_cap=topk_cap, is_dcp_mode=True)
    )
    real = np.asarray(out, np.float32)[:real_tokens]
    np.testing.assert_allclose(
        real, np.asarray(ref, np.float32)[:real_tokens], rtol=2e-2, atol=2e-2
    )
    np.testing.assert_allclose(
        np.asarray(lse)[:real_tokens],
        np.asarray(ref_lse)[:real_tokens],
        rtol=1e-4,
        atol=1e-4,
    )
    # positive control: the comparison must be able to fail
    assert (
        np.abs(real - np.asarray(problem["q"], np.float32)[:real_tokens]).max() > 2e-2
    )


def test_topk_cap_dcp_split_q_zero_batch_matches_uncapped():
    """Split q + lane-dense lse through the DCP ladder, incl. an all-empty batch."""
    if jax.devices()[0].platform != "tpu":
        pytest.skip("SparseCore cache-layout attention requires a TPU")
    topk, query_tokens, chunk, real_tokens = 2048, 128, 64, 100
    rng = np.random.default_rng(20261013)
    problem = _build_problem(rng, k_scale=0.75, query_tokens=query_tokens, topk=topk)
    counts = np.resize(np.array(_COUNTS["mixed_all"]), query_tokens)
    rng.shuffle(counts)
    counts[:16] = 0  # one whole kernel batch owns nothing: the zero-batch bypass
    live = np.arange(topk)[None, :] < counts[:, None]
    topk_idx = jnp.where(jnp.asarray(live), problem["topk"], -1)
    q_nope = problem["q"][..., :NOPE_DIM]
    q_pe = problem["q"][..., NOPE_DIM:]

    ref, ref_lse = jax.block_until_ready(_run(problem, topk_idx, real_tokens, chunk))
    out, lse = jax.block_until_ready(
        _run(
            dict(problem, q=q_nope),
            topk_idx,
            real_tokens,
            chunk,
            topk_cap=512,
            is_dcp_mode=True,
            q_pe=q_pe,
            lse_lane_dense=True,
        )
    )
    owned = counts[:real_tokens] > 0
    out = np.asarray(out, np.float32)[:real_tokens]
    lse = np.asarray(lse)[:real_tokens]
    np.testing.assert_allclose(
        out[owned],
        np.asarray(ref, np.float32)[:real_tokens, :, :NOPE_DIM][owned],
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        lse[owned], np.asarray(ref_lse)[:real_tokens][owned], rtol=1e-4, atol=1e-4
    )
    # Empty rows: zero output and -inf lse, so a DCP merge weights them 0.
    np.testing.assert_array_equal(out[~owned], 0.0)
    assert np.all(np.isneginf(lse[~owned]))
