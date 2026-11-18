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
"""The vision flash kernel on packed rows matches the reference attention (TPU).

Packed rows walk only the kv columns whose segment ids can match a q block's
row groups.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.flash_attention.kernel import (
    BlockSizes,
    SegmentIds,
    flash_attention,
    mha_reference,
)


@pytest.fixture(autouse=True)
def _tpu_only():
    # Not a module-level skipif: initializing JAX at collection claims the TPU.
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel")


def _segment_ids(lengths: list[int], total: int) -> jax.Array:
    """vt's vision segment ids: image i -> i + 1, the padding after them -> the
    next id."""
    ids = np.repeat(np.arange(1, len(lengths) + 1), lengths)
    ids = np.concatenate([ids, np.full(total - ids.size, len(lengths) + 1)])
    return jnp.asarray(ids.astype(np.int32)[None])


@pytest.mark.parametrize(
    ("lengths", "total", "head_dim", "block_sizes"),
    [
        ([1024], 1024, 128, None),  # one segment: nothing to skip
        ([300, 1500], 2048, 72, BlockSizes(128, 128, 128, 1)),  # unaligned
        ([128, 256, 128], 1024, 72, BlockSizes(128, 512, 128, 1)),  # aligned
        ([700, 3000], 4096, 72, BlockSizes(256, 1024, 256, 1)),  # padding blocks
        # Two row groups, the second straddling a boundary; 512-column tails.
        ([1500, 2000], 4096, 72, BlockSizes(2048, 4096, 1024, 1)),
        # Uneven row groups, a partial last q block, windows over two kv blocks.
        ([3000, 900], 4096, 72, BlockSizes(1536, 2048, 1024, 1)),
    ],
)
def test_packed_rows_match_reference(lengths, total, head_dim, block_sizes):
    keys = jax.random.split(jax.random.key(0), 3)
    q, k, v = (
        jax.random.normal(key, (1, 2, total, head_dim), jnp.float32).astype(
            jnp.bfloat16
        )
        for key in keys
    )
    seg = _segment_ids(lengths, total)
    segment_ids = SegmentIds(q=seg, kv=seg)
    out = flash_attention(
        q,
        k,
        v,
        segment_ids=segment_ids,
        sm_scale=head_dim**-0.5,
        block_sizes=block_sizes,
        vmem_limit_bytes=64 * 1024 * 1024,
    )
    want = mha_reference(q, k, v, None, segment_ids, sm_scale=head_dim**-0.5)
    np.testing.assert_allclose(
        np.asarray(out, np.float32), np.asarray(want, np.float32), atol=2e-2, rtol=2e-2
    )
