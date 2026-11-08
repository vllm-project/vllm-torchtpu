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

from tests.models.minimax_m3.packing import CACHE_DTYPES, PACKING, pack
from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_fp4, msa_write

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
    cache = pack(
        rng.standard_normal((NUM_PAGES, PAGE, width)).astype(np.float32), dtype
    )
    rows = jnp.asarray(rng.standard_normal((num_tokens, width)) * 300).astype(
        jnp.bfloat16
    )
    pages, pos = _batch(q_lens, ctx_lens, rng, pad_tokens)

    expected = msa.write_rows_xla(cache, rows, pages, pos)
    got = msa.write_rows(
        cache, rows, pages, pos, max_num_seqs=max(len(q_lens), 1), interpret=True
    )
    np.testing.assert_array_equal(
        np.asarray(got.astype(jnp.float32)), np.asarray(expected.astype(jnp.float32))
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
    assert "minimax_m3_msa_write_rows" in text


@pytest.mark.parametrize("folded", [False, True])
@pytest.mark.parametrize(
    "q_lens,ctx_lens,pad_tokens",
    [
        ([1, 1, 1], [5, 130, 255], 2),
        ([8, 8, 8], [124, 40, 250], 8),
        ([300], [70], 0),
    ],
)
def test_fp4_matches_page_writes(
    q_lens, ctx_lens, pad_tokens, folded, monkeypatch, request
):
    """4-bit writes, token by token (small batches) and page by page, land the
    same nibbles and folded scales."""
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
    monkeypatch.setattr(msa_write, "_BY_TOKEN_MAX_TOKENS", 0)
    jax.clear_caches()  # the writes read the threshold when traced
    request.addfinalizer(jax.clear_caches)  # no page-path trace outlives the test
    expected = write()
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
    # The page writes are v24's; check them against the logical scatter too.
    want = np.asarray(nib).copy()
    keep = np.asarray(pages) < NUM_PAGES
    p, s = np.asarray(pages)[keep], np.asarray(pos)[keep] % PAGE
    want[p, s] = np.asarray(rows)[keep]
    np.testing.assert_array_equal(np.asarray(msa_fp4.unpack_rows(got)), want)
    if folded:
        want_scales = np.asarray(page_scales).copy()
        want_scales[p, s] = np.asarray(scales)[keep]
        np.testing.assert_array_equal(
            np.asarray(msa_fp4.folded_scales(got)), want_scales
        )


@pytest.mark.parametrize("tokens", [8, 512])
@pytest.mark.parametrize("folded", [False, True])
def test_fp4_by_token_lowers_for_tpu(folded, tokens):
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
def test_slot_scales_match_xla_scatter(
    q_lens, ctx_lens, pad_tokens, num_values, monkeypatch, request
):
    """Per-slot scale writes token by token equal the XLA scatter they replace."""
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
    monkeypatch.setattr(msa_write, "_BY_TOKEN_MAX_TOKENS", 0)
    jax.clear_caches()  # the writes read the threshold when traced
    request.addfinalizer(jax.clear_caches)  # no page-path trace outlives the test
    expected = write()
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("rows,num_values", [(8, 2), (16, 16)])
def test_slot_scales_lower_for_tpu(rows, num_values):
    """Up to the largest batch the kernel takes, with 8 KV heads per rank."""
    tokens = msa_write._BY_TOKEN_MAX_TOKENS
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
