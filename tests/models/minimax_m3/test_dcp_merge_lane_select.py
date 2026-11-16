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
"""TPU_MINIMAX_M3_DCP_MERGE_LANE_SELECT: the lane-axis slot select of the DCP
candidate merge returns exactly the default merge's selection."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp

ENV = "TPU_MINIMAX_M3_DCP_MERGE_LANE_SELECT"


def _candidates(rng, dcp, rows, topk, ties, missing):
    """[dcp, rows, 2 * topk]: rank r holds blocks = r mod dcp; scores with
    ties and -inf (missing) candidates, blocks -1 where missing."""
    scores = rng.standard_normal((dcp, rows, topk)).astype(np.float32)
    if ties:
        scores = np.round(scores * 2) / 2
    blocks = np.empty((dcp, rows, topk), np.int32)
    for r in range(dcp):
        for t in range(rows):
            blocks[r, t] = rng.choice(4096, topk, replace=False) * dcp + r
    miss = rng.random((dcp, rows, topk)) < missing
    scores[miss] = -np.inf
    blocks[miss] = -1
    return jnp.asarray(msa_dcp.pack_candidates(jnp.asarray(scores), jnp.asarray(blocks)))


@pytest.mark.parametrize("rows", [1, 8, 24, 32, 40, 48, 96, 384])
@pytest.mark.parametrize("ties,missing", [(False, 0.0), (True, 0.0), (True, 0.3), (False, 1.0)])
def test_lane_select_matches_default(monkeypatch, rows, ties, missing):
    dcp, topk = 2, 16
    c = _candidates(np.random.default_rng(rows * 7 + int(ties) + int(missing * 10)), dcp, rows, topk, ties, missing)
    monkeypatch.delenv(ENV, raising=False)
    ref = np.asarray(jax.jit(msa_dcp.merge_candidates, static_argnums=1)(c, topk))
    monkeypatch.setenv(ENV, "1")
    got = np.asarray(jax.jit(msa_dcp.merge_candidates, static_argnums=1)(c, topk))
    np.testing.assert_array_equal(got, ref)
    assert ref.shape == (rows, topk)


def test_flag_default_off(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.delenv(ENV, raising=False)
    assert envs.TPU_MINIMAX_M3_DCP_MERGE_LANE_SELECT is False
    monkeypatch.setenv(ENV, "1")
    assert envs.TPU_MINIMAX_M3_DCP_MERGE_LANE_SELECT is True
