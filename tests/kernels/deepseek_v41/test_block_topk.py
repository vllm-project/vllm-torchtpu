# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Block-maximum top-k: ties, masked keys, short rows, forced and published blocks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.block_topk import (
    block_topk,
    is_supported,
)

K, BLOCK, PUBLISH = 512, 8, 2048
# Empty, at most k, fewer than the scanned blocks' keys (4224), and full rows.
LENGTHS = [0, 1, 7, 511, 512, 513, 4223, 4224, 4225, 5001, 8191, 20000, 49407, 49408]
CASES = [(32, 49408, d) for d in ("random", "relu", "signed_zero", "masked")]
# 1M-token contexts (ratio 2 and 1): rows stream from HBM; the widest is
# unsupported and must fall back.
CASES += [(32, 524288, "random"), (32, 1048576, "masked")]


def _scores(distribution, rows, width, rng):
    scores = rng.normal(size=(rows, width)).astype(np.float32)
    if distribution == "relu":
        scores = np.maximum(scores - 1, 0)  # Most keys tie at zero.
    elif distribution == "signed_zero":
        scores[:] = 0
        scores[:, ::3] = np.float32(-0.0)
    elif distribution == "masked":
        scores[:, (np.arange(width) // 256) % 3 == 1] = -np.inf
    return scores


@pytest.mark.parametrize("publish", [0, PUBLISH])
@pytest.mark.parametrize("rows,width,distribution", [*CASES, (2048, 8192, "random")])
def test_block_topk(rows, width, distribution, publish):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore top-k requires TPU")
    if publish and width // BLOCK < publish:
        pytest.skip("Publication needs more blocks than it publishes")
    if not is_supported(rows, width, K, BLOCK):
        pytest.skip("Block maxima of this width do not fit SparseCore VMEM")
    rng = np.random.default_rng(23)
    scores = _scores(distribution, rows, width, rng)
    fixed = rows == 32 and width == LENGTHS[-1]
    lengths = np.resize(LENGTHS, rows) if fixed else rng.integers(0, width + 1, rows)
    lengths = lengths.astype(np.int32)
    scores[np.arange(width)[None, :] >= lengths[:, None]] = -np.inf
    result = block_topk(
        jnp.asarray(scores),
        jnp.asarray(lengths),
        k=K,
        block_size=BLOCK,
        publish=publish,
    )
    selected = np.asarray(result[0] if publish else result)
    for row, length in enumerate(lengths):
        ids, visible = selected[row][selected[row] >= 0], scores[row, :length]
        rest = visible > -np.inf
        assert np.unique(ids).size == ids.size == min(K, rest.sum())
        assert (visible[ids] > -np.inf).all()
        rest[ids] = False
        if ids.size and rest.any():
            assert visible[ids].min() >= visible[rest].max()
        if publish:
            # The highest block maxima in radix order (+0 above -0), the newest
            # visible block forced first, lowest IDs on ties, ascending.
            blocks = scores[row].reshape(-1, BLOCK)
            maxima = blocks.max(-1)
            positive_zero = ((blocks == 0) & ~np.signbit(blocks)).any(-1)
            maxima[(maxima == 0) & positive_zero] = np.float32(0.0)
            maxima[(length - 1) // BLOCK] = np.inf
            blocks = np.flatnonzero(maxima[: -(-length // BLOCK)] > -np.inf)
            values = maxima[blocks]
            order = np.lexsort((blocks, np.signbit(values), -values))[:publish]
            expected = np.full(publish, -1, np.int32)
            expected[: order.size] = np.sort(blocks[order])
            np.testing.assert_array_equal(np.asarray(result[1])[row], expected)
