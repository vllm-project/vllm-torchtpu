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
"""TPU_MINIMAX_M3_DCP_MERGE_KERNEL_MIN_TOKENS: the merge kernel gives the
dense ranking's DCP 2 selection bit for bit (interpret)."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp

ENV = "TPU_MINIMAX_M3_DCP_MERGE_KERNEL_MIN_TOKENS"


def _candidates(rng, dcp, tokens, heads, topk, num_blocks):
    """Each rank's packed candidates: distinct blocks of its own (block %
    dcp == rank), coarse scores (ties), every third row short (-inf, block -1)."""
    out = np.zeros((dcp, tokens, heads * 2 * topk), np.float32)
    for r in range(dcp):
        for t in range(tokens):
            for g in range(heads):
                n = int(rng.integers(0, topk // 2)) if t % 3 == 0 else topk
                own = np.arange(r, num_blocks, dcp)
                blocks = np.full(topk, -1, np.int64)
                scores = np.full(topk, -np.inf, np.float32)
                blocks[:n] = rng.choice(own, n, replace=False)
                scores[:n] = np.round(rng.standard_normal(n), 1)
                base = g * 2 * topk
                out[r, t, base : base + topk] = scores
                out[r, t, base + topk : base + 2 * topk] = blocks
    return jnp.asarray(out)


@pytest.mark.parametrize("tokens,heads", [(256, 1), (40, 1), (512, 2)])
def test_kernel_matches_dense(monkeypatch, tokens, heads):
    rng = np.random.default_rng(tokens + heads)
    topk, dcp = 16, 2
    c = _candidates(rng, dcp, tokens, heads, topk, 400)
    monkeypatch.setenv(ENV, "0")
    ref = np.asarray(msa_dcp.merge_head_candidates(c, topk, interpret=True))
    monkeypatch.setenv(ENV, "1")
    got = np.asarray(msa_dcp.merge_head_candidates(c, topk, interpret=True))
    np.testing.assert_array_equal(got, ref)
    assert (ref >= 0).any() and (ref < 0).any()


def test_below_min_tokens_keeps_dense(monkeypatch):
    rng = np.random.default_rng(0)
    c = _candidates(rng, 2, 40, 1, 16, 400)
    monkeypatch.setenv(ENV, "4096")
    got = np.asarray(msa_dcp.merge_head_candidates(c, 16, interpret=True))
    rows = c.reshape(2, 40, 32)
    np.testing.assert_array_equal(got[:, 0], np.asarray(msa_dcp.merge_candidates(rows, 16)))
