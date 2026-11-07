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
"""MSA sparse attention on 4-bit caches (interpret mode) against the dense
reference over the dequantized cache (`msa_fp4`)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from tests.models.minimax_m3.test_msa_attention import _reference
from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4
from vllm_torchtpu.models.minimax_m3.kernels.msa_attention import msa_sparse_attention
from vllm_torchtpu.models.minimax_m3.kernels.msa_attention_lse import (
    msa_sparse_attention_lse,
)

HEADS = 16
HEAD_DIM = 128
PAGE = 128
PAGES = 10
TOPK = 4


def _fp4_cache(rng, k_row=0, v_row=1):
    """Random K/V quantized per (token, head): the uint8 cache, its scale
    cache, and the dequantized rows ``[pages, S, 2D]`` the kernel must match.
    Scale rows other than K/V and the last page's last slots hold NaN (as an
    unwritten slot may), which the kernel must never let through."""
    k = rng.standard_normal((PAGES, PAGE, HEAD_DIM)).astype(np.float32)
    v = rng.standard_normal((PAGES, PAGE, HEAD_DIM)).astype(np.float32)
    kq, ks = msa_fp4.quantize(jnp.asarray(k))
    vq, vs = msa_fp4.quantize(jnp.asarray(v))
    cache = msa_fp4.pack_rows(jnp.concatenate([kq, vq], axis=-1))
    scales = np.full((PAGES, msa_fp4.SCALE_ROWS, PAGE), np.nan, np.float32)
    scales[:, k_row] = np.asarray(ks)
    scales[:, v_row] = np.asarray(vs)
    rows = np.concatenate(
        [
            np.asarray(msa_fp4.dequantize(kq, ks)),
            np.asarray(msa_fp4.dequantize(vq, vs)),
        ],
        axis=-1,
    )
    return cache, scales, rows


def _batch(rng, num_tokens, num_real, poison_page):
    selected = np.full((num_tokens, TOPK), -1, np.int32)
    local_page = np.full(num_tokens, -1, np.int32)
    local_limit = np.zeros(num_tokens, np.int32)
    others = np.array([p for p in range(PAGES) if p != poison_page])
    for t in range(num_real):
        pages = rng.choice(others, rng.integers(1, TOPK + 1), replace=False)
        selected[t, : len(pages)] = pages
        local_page[t] = pages[0]
        local_limit[t] = rng.integers(1, PAGE + 1)
    # Only token 0 reads the poisoned page, as its local page, up to slot 40.
    selected[0, 0], local_page[0], local_limit[0] = poison_page, poison_page, 40
    return selected, local_page, local_limit


@pytest.mark.parametrize("block_q", [16, 64])
def test_matches_dequantized_reference(block_q):
    rng = np.random.default_rng(block_q)
    cache, scales, rows = _fp4_cache(rng)
    poison = 3
    scales[poison, :, 64:] = np.nan  # unwritten tail of a local page
    num_tokens, num_real = 37, 33
    selected, local_page, local_limit = _batch(rng, num_tokens, num_real, poison)
    q = jnp.asarray(rng.standard_normal((num_tokens, HEADS, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    out = msa_sparse_attention(
        q,
        cache,
        jnp.asarray(selected),
        jnp.asarray(local_page),
        jnp.asarray(local_limit),
        sm_scale=HEAD_DIM**-0.5,
        block_q=block_q,
        scale_cache=jnp.asarray(scales),
        interpret=True,
    )
    expected = _reference(
        np.asarray(q.astype(jnp.float32)), rows, selected, local_page, local_limit
    )
    got = np.asarray(out.astype(jnp.float32))
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, atol=3e-2)


def test_lse_matches_dense_and_plain_kernel():
    """The DCP partial kernel: same output as the plain kernel, and the LSE of
    the dequantized scores."""
    rng = np.random.default_rng(5)
    cache, scales, rows = _fp4_cache(rng, k_row=0, v_row=1)
    num_tokens, num_real = 40, 36
    selected, local_page, local_limit = _batch(rng, num_tokens, num_real, 2)
    q = jnp.asarray(rng.standard_normal((num_tokens, HEADS, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    table = jnp.arange(PAGES, dtype=jnp.int32)  # keys are physical pages
    common = dict(sm_scale=HEAD_DIM**-0.5, block_q=16, interpret=True)
    out, lse = msa_sparse_attention_lse(
        q,
        cache,
        jnp.asarray(selected),
        jnp.asarray(local_page),
        jnp.asarray(local_limit),
        table,
        scale_cache=jnp.asarray(scales),
        **common,
    )
    plain = msa_sparse_attention(
        q,
        cache,
        jnp.asarray(selected),
        jnp.asarray(local_page),
        jnp.asarray(local_limit),
        scale_cache=jnp.asarray(scales),
        **common,
    )
    np.testing.assert_allclose(
        np.asarray(out.astype(jnp.float32)), np.asarray(plain.astype(jnp.float32))
    )
    qf = np.asarray(q.astype(jnp.float32))
    for t in range(num_real):
        keys = []
        for p in selected[t][selected[t] >= 0]:
            n = local_limit[t] if p == local_page[t] else PAGE
            keys.append(rows[p, :n, :HEAD_DIM])
        s = qf[t] @ np.concatenate(keys).T * HEAD_DIM**-0.5  # [H, keys]
        m = s.max(-1)
        want = m + np.log(np.exp(s - m[:, None]).sum(-1))
        np.testing.assert_allclose(np.asarray(lse[t]), want, atol=2e-2)
    assert np.all(np.isneginf(np.asarray(lse[num_real:])))


def test_lowers_for_tpu():
    """Pallas -> Mosaic lowering of the 4-bit path runs without a TPU."""
    tokens = 256
    args = (
        jax.ShapeDtypeStruct((tokens, HEADS, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((64, PAGE // 8, 4, 2 * HEAD_DIM), jnp.uint8),
        jax.ShapeDtypeStruct((tokens, 16), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
    )
    scales = jax.ShapeDtypeStruct((64, msa_fp4.SCALE_ROWS, PAGE), jnp.float32)
    attention = functools.partial(msa_sparse_attention, sm_scale=HEAD_DIM**-0.5)
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = (
            jax.jit(lambda *a: attention(*a[:5], scale_cache=a[5]))
            .trace(*args, scales)
            .lower(lowering_platforms=("tpu",))
        )
    assert "tpu_custom_call" in lowered.as_text()
