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
"""Numerics of the single-step flash-attention path against a float64 reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.flash_attention.kernel import (
    BlockSizes,
    SegmentIds,
    _flash_attention,
)

pytestmark = pytest.mark.tpu_test

_VMEM_LIMIT_BYTES = 64 * 1024 * 1024
# bf16 rounding of the output alone gives ~0.17%; P @ V in bf16 adds ~0.06%.
_MAX_REL_RMS = 3e-3
# With P rounded before normalization this case measured 0.229%; after, 0.178%.
_MAX_REL_RMS_PEAKED = 2e-3


def _inputs(num_heads, seq_len, head_dim, q_scale, seed=0):
    kq, kk, kv = jax.random.split(jax.random.PRNGKey(seed), 3)
    shape = (1, num_heads, seq_len, head_dim)
    q = jax.random.normal(kq, shape) * q_scale / head_dim**0.5
    k = jax.random.normal(kk, shape)
    v = jax.random.normal(kv, shape)
    return tuple(x.astype(jnp.bfloat16) for x in (q, k, v))


def _two_segments(seq_len, split):
    ids = jnp.where(jnp.arange(seq_len) < split, 1, 2).astype(jnp.int32)[None]
    return SegmentIds(q=ids, kv=ids)


def _reference(q, k, v, segment_ids, causal, sm_scale):
    q, k, v = (np.asarray(x.astype(jnp.float32), np.float64)[0] for x in (q, k, v))
    s = np.einsum("hqd,hkd->hqk", q, k) * sm_scale
    seq_len = q.shape[1]
    allowed = np.ones((seq_len, seq_len), bool)
    if segment_ids is not None:
        ids = np.asarray(segment_ids.q[0])
        allowed &= ids[:, None] == ids[None, :]
    if causal:
        allowed &= np.tril(allowed)
    s = np.where(allowed, s, -np.inf)
    m = s.max(-1)
    e = np.exp(s - m[..., None])
    row_sum = e.sum(-1)
    return np.einsum("hqk,hkd->hqd", e / row_sum[..., None], v), row_sum, m


def _rel_rms(actual, expected):
    actual = np.asarray(jnp.asarray(actual).astype(jnp.float32), np.float64)
    return float(np.sqrt(((actual - expected) ** 2).mean() / (expected**2).mean()))


def _single_step(q, k, v, segment_ids, causal, sm_scale):
    seq_len = k.shape[2]
    block_sizes = BlockSizes(
        block_q=128, block_k_major=seq_len, block_k=seq_len, block_b=1
    )
    o, row_sum, row_max = _flash_attention(
        q,
        k,
        v,
        None,
        segment_ids,
        True,  # save_residuals
        causal,
        sm_scale,
        block_sizes,
        _VMEM_LIMIT_BYTES,
        False,
    )
    return o[0], row_sum[0], row_max[0]


@pytest.mark.parametrize(
    ("num_heads", "seq_len", "head_dim", "segments", "causal", "sm_scale"),
    (
        (4, 512, 64, False, False, 1.0),
        (4, 512, 64, True, False, 1.0),
        (4, 512, 64, False, True, 1.0),
        (4, 512, 64, True, True, 1.0),
        (2, 1024, 128, True, False, 0.5),
        (2, 640, 80, True, False, 1.0),
    ),
)
def test_single_step_matches_reference(
    num_heads, seq_len, head_dim, segments, causal, sm_scale
):
    q, k, v = _inputs(num_heads, seq_len, head_dim, q_scale=1.0)
    segment_ids = _two_segments(seq_len, seq_len // 3) if segments else None
    o, row_sum, row_max = _single_step(q, k, v, segment_ids, causal, sm_scale)
    ref_o, ref_sum, ref_max = _reference(q, k, v, segment_ids, causal, sm_scale)

    assert _rel_rms(o, ref_o) < _MAX_REL_RMS
    np.testing.assert_allclose(np.asarray(row_max), ref_max, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(row_sum), ref_sum, rtol=1e-4)


def test_single_step_peaked_logits():
    q, k, v = _inputs(4, 1536, 64, q_scale=4.0)
    segment_ids = _two_segments(1536, 1536)
    o, _, _ = _single_step(q, k, v, segment_ids, False, 1.0)
    ref_o, _, _ = _reference(q, k, v, segment_ids, False, 1.0)

    assert _rel_rms(o, ref_o) < _MAX_REL_RMS_PEAKED


def test_tolerance_rejects_wrong_normalization():
    q, k, v = _inputs(4, 512, 64, q_scale=1.0)
    segment_ids = _two_segments(512, 200)
    o, row_sum, _ = _single_step(q, k, v, segment_ids, False, 1.0)
    ref_o, _, _ = _reference(q, k, v, segment_ids, False, 1.0)
    row_sum = np.asarray(row_sum)[..., None]
    o = np.asarray(o.astype(jnp.float32))

    assert _rel_rms(o * row_sum, ref_o) > 100 * _MAX_REL_RMS  # missing `/ l`
    assert _rel_rms(o / row_sum, ref_o) > 100 * _MAX_REL_RMS  # divided twice
