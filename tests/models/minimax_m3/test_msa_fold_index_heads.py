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
"""`msa.fold_index_heads` (a gather by the inverse of its rows) against a
scatter of the index queries into those rows."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa


@pytest.mark.parametrize("seed", range(12))
def test_fold_matches_scatter_into_rows(seed):
    rng = np.random.default_rng(seed)
    num_seqs = int(rng.integers(1, 9))
    num_tokens = int(rng.choice([16, 48, 128]))
    num_heads = int(rng.choice([1, 2, 4]))
    q_lens = rng.integers(0, 20, num_seqs)
    while q_lens.sum() > num_tokens:
        q_lens //= 2
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
    seq_lens = jnp.asarray(q_lens + rng.integers(0, 100, num_seqs), jnp.int32)
    index_q = jnp.asarray(
        rng.standard_normal((num_tokens, num_heads, 8)), jnp.bfloat16
    )
    folded, rows = msa.fold_index_heads(index_q, seq_lens, cu)
    want = (
        jnp.zeros_like(folded)
        .at[rows.reshape(-1)]
        .set(index_q.transpose(1, 0, 2).reshape(-1, index_q.shape[2]))
    )
    np.testing.assert_array_equal(np.asarray(folded), np.asarray(want))
