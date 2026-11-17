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
"""`metadata.lane_slice_q_metadata`: a slice of a bucket's query tokens keeps
every kept token's causal row length (TPU_DCP_INDEXER_LANE_SPLIT)."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import metadata
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import (
    _effective_row_lengths,
)


def _batch(rng, num_tokens, max_num_seqs):
    num_seqs = int(rng.integers(1, max_num_seqs + 1))
    real = int(rng.integers(num_seqs, num_tokens + 1))
    cuts = np.sort(rng.choice(np.arange(1, real), num_seqs - 1, replace=False))
    cu = np.concatenate([[0], cuts, [real], np.full(max_num_seqs - num_seqs, real)])
    q_lens = np.diff(cu)
    seq_lens = q_lens + rng.integers(0, 5000, max_num_seqs)
    return seq_lens.astype(np.int32), cu.astype(np.int32), num_seqs


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("num_lanes", [2, 4])
def test_lane_slices_keep_every_tokens_row_length(seed, num_lanes):
    rng = np.random.default_rng(seed)
    num_tokens, max_num_seqs = 256, 6
    seq_lens, cu, num_seqs = _batch(rng, num_tokens, max_num_seqs)
    dist = jnp.array([0, 0, num_seqs], jnp.int32)
    kw = dict(
        num_positions=1 << 20, compression_ratio=1, cp_rank=1, cp_size=4, interleave_c=4
    )
    full = _effective_row_lengths(
        jnp.asarray(seq_lens), jnp.asarray(cu), dist, num_tokens, **kw
    )
    size = num_tokens // num_lanes
    for lane in range(num_lanes):
        start = lane * size
        s, c = metadata.lane_slice_q_metadata(
            jnp.asarray(seq_lens), jnp.asarray(cu), jnp.int32(start), size
        )
        got = _effective_row_lengths(s, c, dist, size, **kw)
        np.testing.assert_array_equal(
            np.asarray(got), np.asarray(full[start : start + size])
        )


def test_request_cut_by_the_slice_end_loses_only_its_tail():
    # One request of 10 tokens on a 100-token context, cut at token 6.
    s, c = metadata.lane_slice_q_metadata(
        jnp.array([100], jnp.int32), jnp.array([0, 10], jnp.int32), jnp.int32(0), 6
    )
    assert s.tolist() == [96] and c.tolist() == [0, 6]
    s, c = metadata.lane_slice_q_metadata(
        jnp.array([100], jnp.int32), jnp.array([0, 10], jnp.int32), jnp.int32(6), 6
    )
    assert s.tolist() == [100] and c.tolist() == [0, 4]
