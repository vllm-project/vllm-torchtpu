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
"""MSA sparse attention kernel (interpret mode) against a dense reference."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels.msa_attention import (
    msa_sparse_attention,
)

HEADS = 16
HEAD_DIM = 128
PAGE = 128
PAGES = 12
TOPK = 4


def _reference(q, kv, selected, local_page, local_limit):
    out = np.zeros_like(q)
    for t in range(q.shape[0]):
        keys, values = [], []
        for page in selected[t]:
            if page < 0:
                continue
            visible = local_limit[t] if page == local_page[t] else PAGE
            keys.append(kv[page, :visible, 0, 0])
            values.append(kv[page, :visible, 0, 1])
        if not keys:
            continue
        s = q[t] @ np.concatenate(keys).T * HEAD_DIM**-0.5
        w = np.exp(s - s.max(-1, keepdims=True))
        out[t] = (w / w.sum(-1, keepdims=True)) @ np.concatenate(values)
    return out


@pytest.mark.parametrize(
    "num_tokens,num_real",
    [
        (37, 35),  # last tile part padding, token count not a tile multiple
        (48, 20),  # one tile entirely padding
    ],
)
def test_matches_dense_reference(num_tokens, num_real):
    rng = np.random.default_rng(num_tokens)
    q = jnp.asarray(rng.standard_normal((num_tokens, HEADS, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    kv = jnp.asarray(rng.standard_normal((PAGES, PAGE, 1, 2, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    selected = np.full((num_tokens, TOPK), -1, np.int32)
    local_page = np.full(num_tokens, -1, np.int32)
    local_limit = np.zeros(num_tokens, np.int32)
    for t in range(num_real):
        pages = rng.choice(PAGES, rng.integers(1, TOPK + 1), replace=False)
        selected[t, : len(pages)] = pages
        local_page[t] = pages[0]
        local_limit[t] = rng.integers(1, PAGE + 1)

    out = msa_sparse_attention(
        q,
        kv,
        jnp.asarray(selected),
        jnp.asarray(local_page),
        jnp.asarray(local_limit),
        sm_scale=HEAD_DIM**-0.5,
        interpret=True,
    )

    expected = _reference(
        np.asarray(q.astype(jnp.float32)),
        np.asarray(kv.astype(jnp.float32)),
        selected,
        local_page,
        local_limit,
    )
    np.testing.assert_allclose(np.asarray(out.astype(jnp.float32)), expected, atol=2e-2)


def test_rejects_unsupported_cache():
    kv = jnp.zeros((2, PAGE, 1, 4, HEAD_DIM), jnp.float8_e4m3fn)
    with pytest.raises(NotImplementedError):
        msa_sparse_attention(
            jnp.zeros((16, HEADS, HEAD_DIM), jnp.bfloat16),
            kv,
            jnp.zeros((16, TOPK), jnp.int32),
            jnp.zeros(16, jnp.int32),
            jnp.ones(16, jnp.int32),
            sm_scale=1.0,
            interpret=True,
        )


def test_lowers_for_tpu():
    """Pallas -> Mosaic lowering and verification run without a TPU."""
    tokens, topk = 256, 16
    args = (
        jax.ShapeDtypeStruct((tokens, HEADS, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((64, PAGE, 1, 2, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, topk), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
    )
    attention = functools.partial(msa_sparse_attention, sm_scale=HEAD_DIM**-0.5)
    lowered = jax.jit(attention).trace(*args).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()
