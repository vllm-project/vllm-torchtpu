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
"""In-place MSA cache row writes (interpret mode) against the XLA scatter."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from tests.models.minimax_m3.packing import (
    CACHE_DTYPES,
    PACKING,
    pack,
    pack_index,
    pack_index_fp4,
    unpack,
)
from vllm_torchtpu.models.minimax_m3.kernels import (
    msa,
    msa_fp4,
    msa_quantize,
    msa_write,
)

PAGE = 128
NUM_PAGES = 12
DROPPED = 2**30


def _batch(q_lens, ctx_lens, rng, pad_tokens):
    """Pages and positions of every token of a ragged batch, as msa_forward
    computes them; padding tokens are dropped."""
    perm = rng.permutation(np.arange(1, NUM_PAGES))
    pages, pos, used = [], [], 0
    for q, c in zip(q_lens, ctx_lens):
        n = -(-(q + c) // PAGE)
        table = perm[used : used + n]
        used += n
        p = np.arange(c, c + q)
        pages.append(table[p // PAGE])
        pos.append(p)
    pages = np.concatenate(pages + [np.full(pad_tokens, DROPPED)]).astype(np.int32)
    pos = np.concatenate(pos + [np.zeros(pad_tokens, np.int64)]).astype(np.int32)
    return jnp.asarray(pages), jnp.asarray(pos)


def _zero_past_runs(rows, pages, pos):
    """``rows [pages, S, ...]`` with each written page's slots past its last
    written one zeroed, as the kernel writes them."""
    rows = np.array(rows)
    page_size, pages, pos = rows.shape[1], np.asarray(pages), np.asarray(pos)
    for p in np.unique(pages[pages < rows.shape[0]]):
        rows[p, pos[pages == p].max() % page_size + 1 :] = 0
    return rows


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
@pytest.mark.parametrize("width", [128, 256])
@pytest.mark.parametrize(
    "q_lens,ctx_lens,pad_tokens",
    [
        ([1, 1, 1], [5, 130, 255], 2),  # decode: mid-page, page start, last slot
        ([8, 8, 8], [124, 40, 250], 8),  # verify windows, two crossing a page edge
        ([300], [70], 0),  # prefill from mid-page across three pages
        ([1, 200, 3], [127, 0, 125], 4),  # mixed, a run ending at a page edge
        ([], [], 3),  # nothing to write
    ],
)
def test_matches_xla_scatter(q_lens, ctx_lens, pad_tokens, width, dtype):
    rng = np.random.default_rng(sum(q_lens) + width)
    num_tokens = sum(q_lens) + pad_tokens
    init = rng.standard_normal((NUM_PAGES, PAGE, width)).astype(np.float32)
    cache = pack(init, dtype)
    rows = jnp.asarray(rng.standard_normal((num_tokens, width)) * 300).astype(
        jnp.bfloat16
    )
    pages, pos = _batch(q_lens, ctx_lens, rng, pad_tokens)

    expected = msa.write_rows_xla(
        pack(_zero_past_runs(init, pages, pos), dtype), rows, pages, pos
    )
    got = msa.write_rows(
        cache, rows, pages, pos, max_num_seqs=max(len(q_lens), 1), interpret=True
    )
    np.testing.assert_array_equal(
        np.asarray(got.astype(jnp.float32)), np.asarray(expected.astype(jnp.float32))
    )


@pytest.mark.parametrize("dtype", CACHE_DTYPES + ["int4"])
@pytest.mark.parametrize(
    "q_lens,ctx_lens,pad_tokens",
    [
        ([1, 1, 1], [5, 130, 255], 2),
        ([8, 8, 8], [124, 40, 250], 8),
        ([300], [70], 0),
        ([1, 200, 3], [127, 0, 125], 4),
    ],
)
def test_dim_major_matches_token_major(q_lens, ctx_lens, pad_tokens, dtype):
    """A dim-major write (the index cache's, `Rows.dim_major`) leaves the
    pages a token-major write leaves, transposed: written slots, zeroed
    slots past each run, untouched pages."""
    rng = np.random.default_rng(sum(q_lens) + 7)
    num_tokens = sum(q_lens) + pad_tokens
    pages, pos = _batch(q_lens, ctx_lens, rng, pad_tokens)
    write = functools.partial(
        msa.write_rows, pages=pages, pos=pos, max_num_seqs=3, interpret=True
    )
    if dtype == "int4":
        init = rng.integers(0, 16, (NUM_PAGES, PAGE, PAGE), dtype=np.uint8)
        init_scales = rng.random((NUM_PAGES, PAGE)).astype(np.float32)
        rows = jnp.asarray(rng.integers(0, 16, (num_tokens, PAGE)), jnp.uint8)
        scales = jnp.asarray(rng.random(num_tokens), jnp.float32)
        token_major = write(msa_fp4.pack_rows(init, init_scales), rows, scales=scales)
        got = write(
            pack_index_fp4(init, init_scales), rows, scales=scales, dim_major=True
        )
        want = pack_index_fp4(
            msa_fp4.unpack_rows(token_major), msa_fp4.folded_scales(token_major)
        )
    else:
        init = rng.standard_normal((NUM_PAGES, PAGE, PAGE)).astype(np.float32)
        rows = jnp.asarray(rng.standard_normal((num_tokens, PAGE))).astype(dtype)
        token_major = write(pack(init, dtype), rows)
        got = write(pack_index(init, dtype), rows, dim_major=True)
        want = pack_index(unpack(token_major), dtype)
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint8), np.asarray(want).view(np.uint8)
    )


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
def test_lowers_for_tpu(dtype):
    """Pallas -> Mosaic lowering and verification run without a TPU."""
    tokens, width = 256, 256
    args = (
        jax.ShapeDtypeStruct(
            (4096, PAGE // PACKING[dtype], PACKING[dtype], width), dtype
        ),
        jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = (
            jax.jit(functools.partial(msa.write_rows, max_num_seqs=16))
            .trace(*args)
            .lower(lowering_platforms=("tpu",))
        )
    text = lowered.as_text()
    assert "tpu_custom_call" in text
    assert "minimax_m3_msa_write_runs" in text


@pytest.mark.parametrize("folded", [False, True])
@pytest.mark.parametrize(
    "q_lens,ctx_lens,pad_tokens",
    [
        ([1, 1, 1], [5, 130, 255], 2),
        ([8, 8, 8], [124, 40, 250], 8),
        ([300], [70], 0),
    ],
)
def test_fp4_matches_scatter(q_lens, ctx_lens, pad_tokens, folded):
    """4-bit writes land the nibbles and folded scales of the logical scatter."""
    rng = np.random.default_rng(sum(q_lens))
    num_tokens = sum(q_lens) + pad_tokens
    nib = jnp.asarray(rng.integers(0, 16, (NUM_PAGES, PAGE, PAGE)), jnp.uint8)
    page_scales = jnp.asarray(rng.random((NUM_PAGES, PAGE)), jnp.float32)
    cache = msa_fp4.pack_rows(nib, page_scales if folded else None)
    rows = jnp.asarray(rng.integers(0, 16, (num_tokens, PAGE)), jnp.uint8)
    scales = jnp.asarray(rng.random(num_tokens), jnp.float32) if folded else None
    pages, pos = _batch(q_lens, ctx_lens, rng, pad_tokens)

    def write():
        return msa.write_rows(
            cache,
            rows,
            pages,
            pos,
            max_num_seqs=len(q_lens),
            scales=scales,
            interpret=True,
        )

    got = write()
    want = _zero_past_runs(nib, pages, pos)
    keep = np.asarray(pages) < NUM_PAGES
    p, s = np.asarray(pages)[keep], np.asarray(pos)[keep] % PAGE
    want[p, s] = np.asarray(rows)[keep]
    np.testing.assert_array_equal(np.asarray(msa_fp4.unpack_rows(got)), want)
    if folded:  # slots past each run take scale 0, as their keys take 0
        want_scales = _zero_past_runs(page_scales, pages, pos)
        want_scales[p, s] = np.asarray(scales)[keep]
        np.testing.assert_array_equal(
            np.asarray(msa_fp4.folded_scales(got)), want_scales
        )


@pytest.mark.parametrize("tokens", [8, 512])
@pytest.mark.parametrize("folded", [False, True])
def test_fp4_lowers_for_tpu(folded, tokens):
    args = (
        jax.ShapeDtypeStruct((4096, PAGE // 8 + folded, 4, PAGE), jnp.uint8),
        jax.ShapeDtypeStruct((tokens, PAGE), jnp.uint8),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
    )
    if folded:
        args += (jax.ShapeDtypeStruct((tokens,), jnp.float32),)

    def write(cache, rows, pages, pos, scales=None):
        return msa.write_rows(cache, rows, pages, pos, max_num_seqs=64, scales=scales)

    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = jax.jit(write).trace(*args).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()


@pytest.mark.parametrize("num_values", [2, 8])
@pytest.mark.parametrize(
    "q_lens,ctx_lens,pad_tokens",
    [
        ([1, 1, 1], [5, 130, 255], 2),
        ([8, 8, 8], [124, 40, 250], 8),
        ([300], [70], 0),
    ],
)
def test_slot_scales_match_xla_scatter(q_lens, ctx_lens, pad_tokens, num_values):
    """Per-slot scale writes equal the XLA scatter they replace."""
    rng = np.random.default_rng(sum(q_lens) + num_values)
    num_tokens = sum(q_lens) + pad_tokens
    cache = jnp.asarray(rng.random((NUM_PAGES, 8, PAGE)), jnp.float32)
    values = jnp.asarray(rng.random((num_tokens, num_values)), jnp.float32)
    pages, pos = _batch(q_lens, ctx_lens, rng, pad_tokens)
    max_runs = msa_write.max_write_runs(num_tokens, len(q_lens), PAGE)

    def write():
        return msa_write.write_slot_scales(
            cache, values, pages, pos % PAGE, max_runs=max_runs, interpret=True
        )

    got = write()
    keep = np.asarray(pages) < NUM_PAGES
    p, s = np.asarray(pages)[keep], np.asarray(pos)[keep] % PAGE
    want = np.asarray(cache).copy()
    want[p, :num_values, s] = np.asarray(values)[keep]
    np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("rows,num_values", [(8, 2), (16, 16)])
def test_slot_scales_lower_for_tpu(rows, num_values):
    """Up to the largest batch the kernel takes, with 8 KV heads per rank."""
    tokens = 8192
    args = (
        jax.ShapeDtypeStruct((4096, rows, PAGE), jnp.float32),
        jax.ShapeDtypeStruct((tokens, num_values), jnp.float32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
    )
    write = functools.partial(msa_write.write_slot_scales, max_runs=130)
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = jax.jit(write).trace(*args).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()


@pytest.mark.parametrize(
    "q_lens,ctx_lens,pad_tokens",
    [
        ([1, 1, 1], [5, 130, 255], 2),
        ([8, 8, 8], [124, 40, 250], 8),
        ([300, 2], [70, 126], 1),
        ([8] * 11, [10 * i + 3 for i in range(11)], 0),  # more runs than in flight
    ],
)
def test_write_runs_targets_match_single_writes(q_lens, ctx_lens, pad_tokens):
    """One call writing a layer's 4-bit K/V rows, folded index rows and slot
    scales equals one call per cache."""
    rng = np.random.default_rng(sum(q_lens) + 7)
    num_tokens = sum(q_lens) + pad_tokens
    kv = msa_fp4.pack_rows(
        jnp.asarray(rng.integers(0, 16, (NUM_PAGES, PAGE, 2 * PAGE)), jnp.uint8)
    )
    index = msa_fp4.pack_rows(
        jnp.asarray(rng.integers(0, 16, (NUM_PAGES, PAGE, PAGE)), jnp.uint8),
        jnp.asarray(rng.random((NUM_PAGES, PAGE)), jnp.float32),
    )
    scale_cache = jnp.asarray(rng.random((NUM_PAGES, 8, PAGE)), jnp.float32)
    kv_rows = jnp.asarray(rng.integers(0, 16, (num_tokens, 2 * PAGE)), jnp.uint8)
    index_rows = jnp.asarray(rng.integers(0, 16, (num_tokens, PAGE)), jnp.uint8)
    index_scales = jnp.asarray(rng.random(num_tokens), jnp.float32)
    values = jnp.asarray(rng.random((num_tokens, 2)), jnp.float32)
    pages, pos = _batch(q_lens, ctx_lens, rng, pad_tokens)
    slots = pos % PAGE
    max_runs = msa_write.max_write_runs(num_tokens, len(q_lens), PAGE)
    write = functools.partial(msa_write.write_rows, max_runs=max_runs, interpret=True)

    got = msa_write.write_runs(
        [
            msa_write.Rows(kv, kv_rows, 4),
            msa_write.Rows(index, index_rows, 4, index_scales),
            msa_write.SlotValues(scale_cache, values),
        ],
        pages,
        slots,
        max_runs=max_runs,
        interpret=True,
    )
    expected = [
        write(kv, kv_rows, pages, slots, bits=4),
        write(index, index_rows, pages, slots, bits=4, scales=index_scales),
        msa_write.write_slot_scales(
            scale_cache, values, pages, slots, max_runs=max_runs, interpret=True
        ),
    ]
    for g, e in zip(got, expected):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(e))


def test_small_pages_match_xla_scatter():
    """16-slot pages (the TP index shards'): run windows round up to the
    row alignment and clamp at the batch's end."""
    page, num_pages = 16, 40
    rng = np.random.default_rng(16)
    q_lens, ctx_lens = [70, 3, 29], [5, 30, 0]
    perm = rng.permutation(np.arange(1, num_pages))
    pages, pos, used = [], [], 0
    for q, c in zip(q_lens, ctx_lens):
        n = -(-(q + c) // page)
        p = np.arange(c, c + q)
        pages.append(perm[used : used + n][p // page])
        pos.append(p)
        used += n
    pages = jnp.asarray(np.concatenate(pages + [[DROPPED] * 2]), jnp.int32)
    pos = jnp.asarray(np.concatenate(pos + [[0, 0]]), jnp.int32)
    dtype = jnp.float8_e4m3fn
    init = rng.standard_normal((num_pages, page, 128)).astype(np.float32)
    cache = pack(init, dtype)
    rows = jnp.asarray(rng.standard_normal((pages.shape[0], 128))).astype(jnp.bfloat16)
    expected = msa.write_rows_xla(
        pack(_zero_past_runs(init, pages, pos), dtype), rows, pages, pos
    )
    got = msa.write_rows(cache, rows, pages, pos, max_num_seqs=3, interpret=True)
    np.testing.assert_array_equal(
        np.asarray(got.astype(jnp.float32)), np.asarray(expected.astype(jnp.float32))
    )


@pytest.mark.parametrize(
    "num_tokens,heads,index",
    [
        (512, 4, False),
        (1000, 4, False),
        (512, 2, True),
        (1000, 1, True),
        (2500, 1, True),  # a partial last token block
    ],
)
def test_quantize_kv_rows_matches_xla(num_tokens, heads, index):
    """`msa_quantize.quantize_kv_rows` equals the XLA rotate + quantize +
    interleave of `msa.write_fp4`, bit for bit (interpret mode), all-zero
    rows included; so do the index keys' nibbles and scales it makes in the
    same pass."""
    rng = np.random.default_rng(num_tokens)
    shape = (num_tokens, heads, 128)
    k = (
        jnp.asarray(
            rng.standard_normal(shape) * rng.uniform(0, 3, (num_tokens, heads, 1)),
            jnp.bfloat16,
        )
        .at[3]
        .set(0)
    )
    v = jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)
    index_k = jnp.asarray(rng.standard_normal((num_tokens, 128)), jnp.bfloat16)
    index_k = index_k.at[5].set(0)
    got = msa_quantize.quantize_kv_rows(
        k, v, index_k if index else None, interpret=True
    )
    k_nib, k_scale = msa_fp4.quantize(msa_fp4.rotate_keys(k))
    v_nib, v_scale = msa_fp4.quantize(msa_fp4.rotate_keys(v))
    want = [
        jnp.concatenate([k_nib, v_nib], -1).reshape(num_tokens, -1),
        jnp.stack([k_scale, v_scale], -1).reshape(num_tokens, -1),
    ]
    if index:
        want += list(msa_fp4.quantize(msa_fp4.rotate_keys(index_k)))
        lanes = np.asarray(got[-1]).view(np.float32)
        got = got[:-1]
        middle = lanes[:, 2 * heads : -1]
        np.testing.assert_array_equal(middle, np.zeros_like(middle))
        np.testing.assert_array_equal(lanes[:, : 2 * heads], np.asarray(want[1]))
        np.testing.assert_array_equal(lanes[:, -1], np.asarray(want[3]))
    assert len(got) == len(want)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


@pytest.mark.parametrize("heads", [1, 2])
def test_fp4_targets_quantize_kernel_matches_xla(heads):
    """A long step's write targets (`msa.fp4_targets`, quantized by
    `msa_quantize.quantize_kv_rows`, index keys included) equal the XLA
    rotate + quantize of every row, bit for bit."""
    rng = np.random.default_rng(heads)
    num_tokens, pages = msa_quantize.MIN_TOKENS, 4
    k, v = (
        jnp.asarray(rng.standard_normal((num_tokens, heads, 128)), jnp.bfloat16)
        for _ in range(2)
    )
    index_k = jnp.asarray(rng.standard_normal((num_tokens, 128)), jnp.bfloat16)
    caches = (
        jnp.zeros((pages, 16, 4, heads * 2 * 128), jnp.uint8),
        jnp.zeros((pages, 17, 4, 128), jnp.uint8),
        jnp.zeros((pages, 8, 128), jnp.float32),
    )
    got = msa.fp4_targets(*caches, k, v, index_k, interpret=True)
    k_nib, k_scale = msa_fp4.quantize(msa_fp4.rotate_keys(k))
    v_nib, v_scale = msa_fp4.quantize(msa_fp4.rotate_keys(v))
    i_nib, i_scale = msa_fp4.quantize(msa_fp4.rotate_keys(index_k))
    want = [
        msa_write.Rows(
            caches[0], jnp.concatenate([k_nib, v_nib], -1).reshape(num_tokens, -1), 4
        ),
        msa_write.Rows(caches[1], i_nib, 4, i_scale, dim_major=True),
        msa_write.SlotValues(
            caches[2], jnp.stack([k_scale, v_scale], -1).reshape(num_tokens, -1)
        ),
    ]
    assert got[2].lanes is not None
    fields = [(0, "rows"), (1, "rows"), (1, "scales"), (2, "values")]
    for i, name in fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(got[i], name)), np.asarray(getattr(want[i], name))
        )
    # The kernel's lanes are the ones the write would build from the scales.
    pages = jnp.arange(num_tokens, dtype=jnp.int32) // 128 % pages
    slots = jnp.arange(num_tokens, dtype=jnp.int32) % 128
    operands = [
        msa_write.write_operands(t, pages, slots, max_runs=64).inputs
        for t in (got, want)
    ]
    assert len(operands[0]) == len(operands[1]) == 3
    for g, w in zip(*operands):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
