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
"""DCP local top-k candidates (interpret mode) against a numpy reference."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_select
from vllm_torchtpu.models.minimax_m3.kernels.msa_dcp import local_candidates

BLOCK = 128
TOPK = 16


def _reference(scores, pos, rank, dcp, init_blocks, local_blocks):
    """Per row: this rank's top-k (forced blocks first, then score, ties to
    the lower block) as (scores, blocks), (-inf, -1) padded."""
    num_tokens, num_cols = scores.shape
    out_s = np.full((num_tokens, TOPK), -np.inf, np.float32)
    out_b = np.full((num_tokens, TOPK), -1, np.int32)
    for t in range(num_tokens):
        visible = pos[t] // BLOCK + 1
        cand = []
        for c in range(num_cols):
            b = c * dcp + rank
            if b >= visible:
                continue
            s = scores[t, c]
            if b >= visible - local_blocks:
                s = msa_select._FORCED_LOCAL_SCORE
            if b < init_blocks:
                s = msa_select._FORCED_INIT_SCORE
            cand.append((-s, b, s))
        cand.sort()
        for i, (_, b, s) in enumerate(cand[:TOPK]):
            out_s[t, i], out_b[t, i] = s, b
    return out_s, out_b


@pytest.mark.parametrize("pick_widths", [True, False])
@pytest.mark.parametrize(
    "num_tokens,num_cols,dcp,rank,max_pos,init_blocks,local_blocks",
    [
        (300, 512, 2, 1, 40_000, 0, 1),  # several row blocks, few chunks visible
        (300, 512, 2, 0, 130_000, 1, 1),  # every chunk visible in some block
        (5, 256, 8, 3, 9_000, 0, 1),  # decode-sized: one 8-row block
        (40, 96, 4, 2, 20_000, 1, 2),  # a width below one 128-column step
        (300, 384, 2, 0, 60_000, 0, 1),  # widths 128, 256 and the full 384
    ],
)
def test_candidates_match_reference(
    num_tokens,
    num_cols,
    dcp,
    rank,
    max_pos,
    init_blocks,
    local_blocks,
    pick_widths,
    monkeypatch,
):
    if pick_widths:  # per-row-block widths, as for a prefill batch
        monkeypatch.setattr(msa_dcp, "_MIN_CANDIDATE_TOKENS", 0)
    rng = np.random.default_rng(num_tokens + num_cols)
    # Few distinct values so ties are common.
    scores = rng.integers(0, 6, (num_tokens, num_cols)).astype(np.float32)
    pos = np.sort(rng.integers(0, max_pos, num_tokens)).astype(np.int32)
    got_s, got_b = local_candidates(
        jnp.asarray(scores)[:, None, :],
        jnp.asarray(pos),
        block_size=BLOCK,
        topk=TOPK,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        rank=rank,
        dcp=dcp,
        interpret=True,
    )
    want_s, want_b = _reference(scores, pos, rank, dcp, init_blocks, local_blocks)
    np.testing.assert_array_equal(np.asarray(got_b), want_b)
    np.testing.assert_array_equal(np.asarray(got_s), want_s)
