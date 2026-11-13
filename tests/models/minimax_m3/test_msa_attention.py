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
from vllm_torchtpu.models.minimax_m3.kernels import (
    msa_attention,
    msa_attention_lse,
    msa_fp4,
)
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


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
def test_page_table_keys_match_physical_pages(dtype):
    """Page keys translated by `page_table` in the page fetch give the output
    of physical pages (the XLA block-table gather they replace)."""
    num_tokens = 70
    rng = np.random.default_rng(3)
    q = jnp.asarray(rng.standard_normal((num_tokens, HEADS, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    kv = pack(rounded(rng.standard_normal((PAGES, PAGE, 2 * HEAD_DIM)), dtype), dtype)
    table = rng.permutation(PAGES).astype(np.int32)  # key -> page, one-to-one
    keys = np.full((num_tokens, TOPK), -1, np.int32)
    local_key = np.full(num_tokens, -1, np.int32)
    local_limit = np.zeros(num_tokens, np.int32)
    for t in range(num_tokens - 4):  # the last tokens are padding
        chosen = rng.choice(PAGES, rng.integers(1, TOPK + 1), replace=False)
        keys[t, : len(chosen)] = chosen
        local_key[t] = chosen[0]
        local_limit[t] = rng.integers(1, PAGE + 1)

    def physical(k):
        return np.where(k >= 0, table[np.maximum(k, 0)], -1)

    common = dict(sm_scale=HEAD_DIM**-0.5, kernel="v2", interpret=True)
    by_key = msa_sparse_attention(
        q,
        kv,
        jnp.asarray(keys),
        jnp.asarray(local_key),
        jnp.asarray(local_limit),
        page_table=jnp.asarray(table),
        **common,
    )
    by_page = msa_sparse_attention(
        q,
        kv,
        jnp.asarray(physical(keys)),
        jnp.asarray(physical(local_key)),
        jnp.asarray(local_limit),
        **common,
    )
    # The union is sorted by key instead of by page, so pages accumulate in a
    # different order: equal up to bf16 rounding.
    np.testing.assert_allclose(
        np.asarray(by_key.astype(jnp.float32)),
        np.asarray(by_page.astype(jnp.float32)),
        atol=2e-3,
    )


@pytest.mark.parametrize(
    "num_tokens,block_q,topk,pages_per_step",
    [
        (16, 16, 16, 16),
        (16, 8, 16, 16),
        (64, 16, 16, 16),
        (48, 16, 4, 8),
        (48, 16, 3, 5),  # groups per tile (10) padded to a power of two
    ],
)
def test_dense_work_list_matches_unique(
    monkeypatch, num_tokens, block_q, topk, pages_per_step
):
    """The dense small-batch work list equals the `jnp.unique` one."""
    rng = np.random.default_rng(num_tokens * block_q + topk)
    # Few distinct pages so tiles share and repeat them; some tokens pick none.
    selected = rng.integers(-1, 40, size=(num_tokens, topk)).astype(np.int32)
    selected[rng.random(num_tokens) < 0.2] = -1
    dense = msa_attention._work_list(jnp.asarray(selected), block_q, pages_per_step)
    monkeypatch.setattr(msa_attention, "_DENSE_WORK_LIST_ELEMS", 0)
    unique = msa_attention._work_list(jnp.asarray(selected), block_q, pages_per_step)
    for got, want in zip(dense, unique, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def _reference_work_list(selected, block_q, pages_per_step):
    """`msa_attention._work_list` by loops: each tile's sorted distinct pages,
    its groups packed after the earlier tiles' (an empty tile gets one), and
    the groups past them as the packed form leaves them (last tile, no bits,
    no pages)."""
    num_tokens, topk = selected.shape
    tiles = num_tokens // block_q
    per_tile = msa_attention._groups_per_tile(block_q, topk, pages_per_step)
    unions = np.full((tiles, per_tile * pages_per_step), -1, np.int32)
    meta = np.zeros(tiles * per_tile, np.int32)
    count = np.zeros(tiles * per_tile, np.int32)
    w = 0
    for t in range(tiles):
        pages = sorted(set(selected[t * block_q : (t + 1) * block_q].ravel()) - {-1})
        unions[t, : len(pages)] = pages
        groups = max(-(-len(pages) // pages_per_step), 1)
        for g in range(groups):
            meta[w] = (
                ((t * per_tile + g) << msa_attention._ROW_SHIFT)
                | (msa_attention._FIRST_BIT if g == 0 else 0)
                | (msa_attention._LAST_BIT if g == groups - 1 else 0)
            )
            count[w] = min(max(len(pages) - g * pages_per_step, 0), pages_per_step)
            w += 1
        last_start = w - groups
    num_groups = w
    for dead in range(num_groups, tiles * per_tile):
        group = min(dead - last_start, per_tile - 1)
        meta[dead] = ((tiles - 1) * per_tile + group) << msa_attention._ROW_SHIFT
    return unions.reshape(-1), meta, num_groups, count


@pytest.mark.parametrize(
    "num_tokens,block_q,pages_per_step",
    [(8, 8, 16), (16, 16, 16), (8, 8, 8), (24, 8, 16), (32, 16, 8)],
)
@pytest.mark.parametrize("fill", ["none", "few", "distinct", "sparse"])
def test_work_list_matches_the_reference(num_tokens, block_q, pages_per_step, fill):
    """One tile (a decode batch, its own scalar-free form) and several tiles
    give the reference work list, value for value."""
    rng = np.random.default_rng(num_tokens + block_q + pages_per_step)
    topk = 16
    selected = {
        "none": np.full((num_tokens, topk), -1),
        "few": rng.integers(0, 40, size=(num_tokens, topk)),
        "distinct": np.arange(num_tokens * topk).reshape(num_tokens, topk)[:, ::-1],
        "sparse": np.where(
            rng.random((num_tokens, topk)) < 0.3,
            -1,
            rng.integers(0, 3000, size=(num_tokens, topk)),
        ),
    }[fill].astype(np.int32)
    got = msa_attention._work_list(jnp.asarray(selected), block_q, pages_per_step)
    want = _reference_work_list(selected, block_q, pages_per_step)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
    # `msa_attention_lse._packed_work` tells the live groups this way.
    _, meta, num_groups, count = want
    live = (count > 0) | ((meta & msa_attention._FIRST_BIT) != 0)
    np.testing.assert_array_equal(live, np.arange(meta.size) < num_groups)


def test_fp4_lse_small_tiles_match_default_tiles():
    """The DCP attention's 8-token tiles (small batches) on a 4-bit cache give
    the 16-token tiles' output and lse."""
    num_tokens = 24
    rng = np.random.default_rng(5)
    q = jnp.asarray(rng.standard_normal((num_tokens, HEADS, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    nib, _ = msa_fp4.quantize(
        jnp.asarray(rng.standard_normal((PAGES, PAGE, 2 * HEAD_DIM)), jnp.float32)
    )
    kv = msa_fp4.pack_rows(nib)
    scale_cache = jnp.asarray(rng.random((PAGES, 8, PAGE)) * 0.3, jnp.float32)
    keys = np.full((num_tokens, TOPK), -1, np.int32)
    local_key = np.full(num_tokens, -1, np.int32)
    local_limit = np.zeros(num_tokens, np.int32)
    for t in range(num_tokens - 3):  # the last tokens are padding
        chosen = rng.choice(PAGES, rng.integers(1, TOPK + 1), replace=False)
        keys[t, : len(chosen)] = chosen
        local_key[t] = chosen[0]
        local_limit[t] = rng.integers(1, PAGE + 1)
    args = (q, kv, jnp.asarray(keys), jnp.asarray(local_key), jnp.asarray(local_limit))
    common = dict(
        page_table=jnp.arange(PAGES, dtype=jnp.int32),
        sm_scale=HEAD_DIM**-0.5,
        scale_cache=scale_cache,
        interpret=True,
    )
    small = msa_attention_lse.msa_sparse_attention_lse(
        *args, block_q=msa_attention.SMALL_BATCH_BLOCK_Q, **common
    )
    default = msa_attention_lse.msa_sparse_attention_lse(
        *args, block_q=msa_attention.DEFAULT_BLOCK_Q, **common
    )
    for a, b in zip(small, default):
        np.testing.assert_allclose(
            np.asarray(a.astype(jnp.float32)),
            np.asarray(b.astype(jnp.float32)),
            atol=2e-3,
        )


def test_fp4_lse_small_tiles_lower_for_tpu():
    tokens, topk = 8, 16
    args = (
        jax.ShapeDtypeStruct((tokens, HEADS, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((4096, PAGE // 8, 4, 2 * HEAD_DIM), jnp.uint8),
        jax.ShapeDtypeStruct((tokens, topk), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((4096,), jnp.int32),
        jax.ShapeDtypeStruct((4096, 8, PAGE), jnp.float32),
    )

    def attention(q, kv, sel, page, limit, table, scales):
        return msa_attention_lse.msa_sparse_attention_lse(
            q,
            kv,
            sel,
            page,
            limit,
            table,
            sm_scale=0.1,
            scale_cache=scales,
            block_q=msa_attention.SMALL_BATCH_BLOCK_Q,
        )

    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = jax.jit(attention).trace(*args).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()
