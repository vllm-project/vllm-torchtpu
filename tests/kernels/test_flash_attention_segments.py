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
"""The flash kernel matches the reference attention (TPU).

Packed rows walk only the kv columns whose segment ids can match a q block's
row groups; the table also covers the other multi-step paths (no segment ids,
causal, attention bias, batch > 1) the same kernel body serves.
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

pytestmark = pytest.mark.tpu_test


@pytest.fixture(autouse=True)
def _tpu_only():
    # Not a module-level skipif: initializing JAX at collection claims the TPU.
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel")


def _segment_ids(lengths: list[int], total: int) -> jax.Array:
    """TpuMMEncoderAttention's segment ids: image i -> i + 1, then one more id
    per 128 rows of padding."""
    ids = np.repeat(np.arange(1, len(lengths) + 1), lengths)
    pad = len(lengths) + 1 + np.arange(total - ids.size) // 128
    ids = np.concatenate([ids, pad])
    return jnp.asarray(ids.astype(np.int32)[None])


def _windows(
    block_q: int, block_k_major: int, block_k: int, block_b: int = 1, **kwargs
) -> BlockSizes:
    return BlockSizes(
        block_q, block_k_major, block_k, block_b, kv_windows=True, **kwargs
    )


@pytest.mark.parametrize(
    ("lengths", "total", "head_dim", "block_sizes", "causal", "bias", "batch"),
    [
        ([1024], 1024, 128, None, False, False, 1),  # one segment: nothing to skip
        ([300, 1500], 2048, 72, _windows(128, 128, 128), False, False, 1),
        ([128, 256, 128], 1024, 72, _windows(128, 512, 128), False, False, 1),
        ([700, 3000], 4096, 72, _windows(256, 1024, 256), False, False, 1),
        # Two row groups, the second straddling a boundary; 512-column tails.
        ([1500, 2000], 4096, 72, _windows(2048, 4096, 1024), False, False, 1),
        # Uneven row groups, a partial last q block, windows over two kv blocks.
        ([3000, 900], 4096, 72, _windows(1536, 2048, 1024), False, False, 1),
        # Several 512-column chunks per kv window at block_k 2048.
        ([2500, 1400], 4096, 72, _windows(2048, 4096, 2048), False, False, 1),
        # A small kv block keeps the f32 bias tile within VMEM.
        ([1500, 2000], 4096, 72, _windows(2048, 1024, 512), False, True, 1),
        ([1500, 2000], 4096, 72, _windows(2048, 4096, 1024), False, False, 2),
        # A partial last batch tile.
        ([1500, 2000], 4096, 72, _windows(2048, 4096, 1024, 2), False, False, 3),
        # Row groups and tail chunks other than the defaults.
        (
            [1500, 2000],
            4096,
            72,
            _windows(2048, 4096, 1024, block_q_group=512, block_k_tail=256),
            False,
            False,
            1,
        ),
        # Without kv windows: every kv block, one tile per q block.
        ([1500, 2000], 4096, 72, BlockSizes(2048, 4096, 1024, 1), False, False, 1),
        (None, 4096, 128, BlockSizes(2048, 4096, 1024, 1), False, False, 1),
        (None, 4096, 128, BlockSizes(2048, 4096, 1024, 1), True, False, 1),
    ],
)
def test_matches_reference(lengths, total, head_dim, block_sizes, causal, bias, batch):
    keys = jax.random.split(jax.random.key(0), 4)
    q, k, v = (
        jax.random.normal(key, (batch, 2, total, head_dim), jnp.float32).astype(
            jnp.bfloat16
        )
        for key in keys[:3]
    )
    ab = (
        jax.random.normal(keys[3], (batch, 2, total, total), jnp.float32)
        if bias
        else None
    )
    segment_ids = None
    if lengths is not None:
        seg = jnp.tile(_segment_ids(lengths, total), (batch, 1))
        segment_ids = SegmentIds(q=seg, kv=seg)
    out = flash_attention(
        q,
        k,
        v,
        ab,
        segment_ids=segment_ids,
        causal=causal,
        sm_scale=head_dim**-0.5,
        block_sizes=block_sizes,
        vmem_limit_bytes=64 * 1024 * 1024,
    )
    want = mha_reference(
        q, k, v, ab, segment_ids, causal=causal, sm_scale=head_dim**-0.5
    )
    np.testing.assert_allclose(
        np.asarray(out, np.float32), np.asarray(want, np.float32), atol=2e-2, rtol=2e-2
    )
