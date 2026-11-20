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
"""MSA DCP candidates: picking from the token-major score rows equals picking
from the gathered scores.

``interpret=True`` runs the kernel body on the host; it does not gate Mosaic
lowering.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp

_TOPK = 16
_BLOCK = 128


def _pick(scores, pos, rank, **kwargs):
    return msa_dcp.local_candidates(
        scores,
        pos,
        block_size=_BLOCK,
        topk=_TOPK,
        init_blocks=1,
        local_blocks=2,
        rank=rank,
        dcp=2,
        interpret=True,
        **kwargs,
    )


# Row-major blocks (8, 24 tokens) and a transposed one (128).
@pytest.mark.parametrize("tokens", [8, 24, 128])
@pytest.mark.parametrize("rank", [0, 1])
def test_token_rows_match_gathered_scores(tokens: int, rank: int) -> None:
    rng = np.random.default_rng(tokens + rank)
    num_rows, cols = 3 * tokens + 8, 256
    # Few distinct values, so most picks break ties.
    scores = jnp.asarray(rng.integers(-3, 4, (num_rows, cols)), jnp.float32)
    rows = jnp.asarray(rng.permutation(num_rows)[:tokens], jnp.int32)
    pos = jnp.asarray(rng.integers(0, 2 * _BLOCK * cols, tokens), jnp.int32)
    want = _pick(scores[rows][:, None, :], pos, rank)
    got = _pick(scores, pos, rank, rows=rows)
    for w, g in zip(want, got):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
