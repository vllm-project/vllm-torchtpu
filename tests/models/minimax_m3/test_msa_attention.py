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
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from tests.models.minimax_m3.packing import CACHE_DTYPES, PACKING, pack, rounded
from vllm_torchtpu.models.minimax_m3.kernels import msa_attention
from vllm_torchtpu.models.minimax_m3.kernels.msa_attention import (
    msa_sparse_attention,
)

HEADS = 16
HEAD_DIM = 128
PAGE = 128
PAGES = 12
TOPK = 4


def _reference(q, kv_rows, selected, local_page, local_limit):
    out = np.zeros_like(q)
    for t in range(q.shape[0]):
        keys, values = [], []
        for page in selected[t]:
            if page < 0:
                continue
            visible = local_limit[t] if page == local_page[t] else PAGE
            keys.append(kv_rows[page, :visible, :HEAD_DIM])
            values.append(kv_rows[page, :visible, HEAD_DIM:])
        if not keys:
            continue
        s = q[t] @ np.concatenate(keys).T * HEAD_DIM**-0.5
        w = np.exp(s - s.max(-1, keepdims=True))
        out[t] = (w / w.sum(-1, keepdims=True)) @ np.concatenate(values)
    return out


@pytest.mark.parametrize("kernel,block_q", [("v1", 64), ("v2", 64), ("v2", 16)])
@pytest.mark.parametrize("dtype", CACHE_DTYPES)
@pytest.mark.parametrize(
    "num_tokens,num_real,tokens_per_call",
    [
        (37, 35, None),  # last tile part padding, token count not a tile multiple
        (48, 20, None),  # one tile entirely padding
        (150, 140, 64),  # several kernel calls (a distinct shape: no jit reuse)
    ],
)
def test_matches_dense_reference(
    num_tokens, num_real, tokens_per_call, dtype, kernel, block_q, monkeypatch
):
    if tokens_per_call is not None:
        monkeypatch.setattr(msa_attention, "MAX_TOKENS_PER_CALL", tokens_per_call)
    rng = np.random.default_rng(num_tokens)
    q = jnp.asarray(rng.standard_normal((num_tokens, HEADS, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    kv_rows = rounded(rng.standard_normal((PAGES, PAGE, 2 * HEAD_DIM)), dtype)
    kv = pack(kv_rows, dtype)
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
        block_q=block_q,
        kernel=kernel,
        interpret=True,
    )

    expected = _reference(
        np.asarray(q.astype(jnp.float32)),
        kv_rows,
        selected,
        local_page,
        local_limit,
    )
    np.testing.assert_allclose(np.asarray(out.astype(jnp.float32)), expected, atol=2e-2)


@pytest.mark.parametrize("kernel,block_q", [("v1", 64), ("v2", 64), ("v2", 16)])
@pytest.mark.parametrize("dtype", CACHE_DTYPES)
def test_unseen_cache_bytes_do_not_leak(dtype, kernel, block_q):
    """NaN in slots no token sees leaves the output finite and unchanged.

    A tile's page union is cut into groups of `pages_per_step`; the slots past
    the union read the null page 0, and a local page's slots past the newest
    token are masked. Masked slots get p = 0, but 0 * NaN is NaN, so a
    never-written page (j-30e45a15: page 0 held NaN rows on the dummy-weight
    twin) turned whole prefill tiles NaN.
    """
    num_tokens = 64
    rng = np.random.default_rng(7)
    q = jnp.asarray(rng.standard_normal((num_tokens, HEADS, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    kv_rows = rounded(rng.standard_normal((PAGES, PAGE, 2 * HEAD_DIM)), dtype)
    # One prefill run on pages 1..3 (local page 3, 20 + t slots visible):
    # a union of 3 pages, so 5 slots of the 8-page group read page 0.
    selected = np.full((num_tokens, TOPK), -1, np.int32)
    selected[:, :3] = [1, 2, 3]
    local_page = np.full(num_tokens, 3, np.int32)
    local_limit = 20 + np.arange(num_tokens, dtype=np.int32)
    poisoned = kv_rows.copy()
    poisoned[0] = np.nan
    poisoned[3, 100:] = np.nan  # past every token's limit (<= 83)

    def run(rows):
        return msa_sparse_attention(
            q,
            pack(rows, dtype),
            jnp.asarray(selected),
            jnp.asarray(local_page),
            jnp.asarray(local_limit),
            sm_scale=HEAD_DIM**-0.5,
            block_q=block_q,
            kernel=kernel,
            interpret=True,
        )

    clean, dirty = run(kv_rows), run(poisoned)
    assert np.isfinite(np.asarray(dirty.astype(jnp.float32))).all()
    np.testing.assert_array_equal(
        np.asarray(dirty.astype(jnp.float32)), np.asarray(clean.astype(jnp.float32))
    )


def test_rejects_unsupported_cache():
    kv = jnp.zeros((2, PAGE, 1, 2 * HEAD_DIM), jnp.float32)
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


@pytest.mark.parametrize("kernel,block_q", [("v1", 64), ("v2", 64), ("v2", 16)])
@pytest.mark.parametrize("dtype", CACHE_DTYPES)
def test_lowers_for_tpu(dtype, kernel, block_q):
    """Pallas -> Mosaic lowering and verification run without a TPU."""
    tokens, topk = 256, 16
    args = (
        jax.ShapeDtypeStruct((tokens, HEADS, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct(
            (64, PAGE // PACKING[dtype], PACKING[dtype], 2 * HEAD_DIM), dtype
        ),
        jax.ShapeDtypeStruct((tokens, topk), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
    )
    attention = functools.partial(
        msa_sparse_attention, sm_scale=HEAD_DIM**-0.5, block_q=block_q, kernel=kernel
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = jax.jit(attention).trace(*args).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()
