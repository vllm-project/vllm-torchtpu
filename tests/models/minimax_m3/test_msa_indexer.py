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
"""MSA indexer block scores (interpret mode) against a numpy reference."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from tests.models.minimax_m3.packing import CACHE_DTYPES, PACKING, pack, rounded
from vllm_torchtpu.models.minimax_m3.kernels.msa_indexer import block_scores

PAGE = 128
HEAD_DIM = 128
PAGES_PER_SEQ = 8
MAX_SEQS = 4
NUM_PAGES = 40


@pytest.mark.parametrize("kernel", ["v1", "v2", "v3"])
@pytest.mark.parametrize("dtype", CACHE_DTYPES)
@pytest.mark.parametrize(
    "q_lens,ctx_lens,block_q,large_block_q,pages_per_step",
    [
        # small (decode, 3 tokens) and large (130 tokens: 3 large tiles)
        ([1, 130, 3], [300, 200, 0], 64, 48, 4),
        ([1, 1], [1000, 5], 128, 512, 2),  # small class only, page groups
    ],
)
def test_matches_reference(
    q_lens, ctx_lens, block_q, large_block_q, pages_per_step, dtype, kernel
):
    rng = np.random.default_rng(len(q_lens))
    num_seqs = len(q_lens)
    num_real = sum(q_lens)
    num_tokens = num_real + 5
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(MAX_SEQS + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    tables = np.zeros((MAX_SEQS, PAGES_PER_SEQ), np.int32)
    tables[:num_seqs] = rng.permutation(np.arange(1, NUM_PAGES))[
        : num_seqs * PAGES_PER_SEQ
    ].reshape(num_seqs, PAGES_PER_SEQ)
    keys = rounded(rng.standard_normal((NUM_PAGES, PAGE, HEAD_DIM)), dtype)
    cache = pack(keys, dtype)
    index_q = jnp.asarray(rng.standard_normal((num_tokens, 1, HEAD_DIM))).astype(
        jnp.bfloat16
    )

    scores = block_scores(
        index_q,
        cache,
        jnp.asarray(seq_lens),
        jnp.asarray(tables.reshape(-1)),
        jnp.asarray(cu_q),
        jnp.int32(num_seqs),
        block_q=block_q,
        large_block_q=large_block_q,
        pages_per_step=pages_per_step,
        small_pages_per_step=pages_per_step,
        kernel=kernel,
        large_kernel=kernel,
        interpret=True,
    )

    queries = np.asarray(index_q.astype(jnp.float32))[:, 0]
    scores = np.asarray(scores)[:, 0]
    for r in range(num_seqs):
        num_pages = -(-seq_lens[r] // PAGE)
        for t in range(cu_q[r], cu_q[r + 1]):
            for b in range(num_pages):
                expected = (keys[tables[r, b]] @ queries[t]).max()
                np.testing.assert_allclose(scores[t, b], expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("kernel", ["v1", "v2", "v3"])
@pytest.mark.parametrize("dtype", CACHE_DTYPES)
def test_lowers_for_tpu(dtype, kernel):
    """Pallas -> Mosaic lowering and verification run without a TPU."""
    tokens, max_seqs, pages_per_seq = 256, 16, 2048
    args = (
        jax.ShapeDtypeStruct((tokens, 1, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct(
            (4096, PAGE // PACKING[dtype], PACKING[dtype], HEAD_DIM), dtype
        ),
        jax.ShapeDtypeStruct((max_seqs,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs * pages_per_seq,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs + 1,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = (
            jax.jit(functools.partial(block_scores, kernel=kernel, large_kernel=kernel))
            .trace(*args)
            .lower(lowering_platforms=("tpu",))
        )
    assert "tpu_custom_call" in lowered.as_text()


@pytest.mark.parametrize("layout", ["folded", "scale_page"])
@pytest.mark.parametrize("kernel", ["v2", "v3"])
@pytest.mark.parametrize(
    "q_lens,ctx_lens,block_q,large_block_q,pages_per_step",
    [([1, 130, 3], [300, 200, 0], 64, 48, 4), ([1, 1], [1000, 5], 128, 512, 2)],
)
def test_fp4_matches_dequantized_reference(
    q_lens, ctx_lens, block_q, large_block_q, pages_per_step, kernel, layout
):
    """4-bit index cache: scores equal rotated queries against the dequantized
    rotated keys. "folded": the scales sit in each page's last word row (the
    served layout); "scale_page": a separate scale page row (NaN elsewhere)."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

    rng = np.random.default_rng(7 + len(q_lens))
    num_seqs = len(q_lens)
    num_real = sum(q_lens)
    num_tokens = num_real + 5
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(MAX_SEQS + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    tables = np.zeros((MAX_SEQS, PAGES_PER_SEQ), np.int32)
    tables[:num_seqs] = rng.permutation(np.arange(1, NUM_PAGES))[
        : num_seqs * PAGES_PER_SEQ
    ].reshape(num_seqs, PAGES_PER_SEQ)
    raw = rng.standard_normal((NUM_PAGES, PAGE, HEAD_DIM)) * rng.uniform(
        0.1, 5, (1, 1, HEAD_DIM)
    )
    nib, scale = msa_fp4.quantize(msa_fp4.rotate_keys(jnp.asarray(raw, jnp.float32)))
    row = 2
    scale_cache = np.full((NUM_PAGES, 8, msa_fp4.SCALE_LANES), np.nan, np.float32)
    scale_cache[:, row] = np.asarray(scale)
    if layout == "folded":
        cache = msa_fp4.pack_rows(nib, scales=scale)
        scale_kwargs = {}
    else:
        cache = msa_fp4.pack_rows(nib)
        scale_kwargs = dict(scale_cache=jnp.asarray(scale_cache), scale_row=row)
    keys = np.asarray(msa_fp4.dequantize(nib, scale))
    index_q = jnp.asarray(rng.standard_normal((num_tokens, 1, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    q_rot = msa_fp4.rotate_queries(index_q)

    scores = block_scores(
        q_rot,
        cache,
        jnp.asarray(seq_lens),
        jnp.asarray(tables.reshape(-1)),
        jnp.asarray(cu_q),
        jnp.int32(num_seqs),
        block_q=block_q,
        large_block_q=large_block_q,
        pages_per_step=pages_per_step,
        small_pages_per_step=pages_per_step,
        kernel=kernel,
        large_kernel=kernel,
        interpret=True,
        **scale_kwargs,
    )

    queries = np.asarray(q_rot.astype(jnp.float32))[:, 0]
    scores = np.asarray(scores)[:, 0]
    for r in range(num_seqs):
        num_pages = -(-seq_lens[r] // PAGE)
        for t in range(cu_q[r], cu_q[r + 1]):
            for b in range(num_pages):
                expected = (keys[tables[r, b]] @ queries[t]).max()
                # Folded pages scale the keys in bf16 (msa_indexer.SCALE_KEYS):
                # the scale and the scaled key are each rounded to bf16.
                rtol = 1e-2 if layout == "folded" else 1e-4
                np.testing.assert_allclose(scores[t, b], expected, rtol=rtol, atol=1e-3)


@pytest.mark.parametrize("kernel", ["v2", "v3"])
def test_fp4_folded_lowers_for_tpu(kernel):
    tokens, max_seqs, pages_per_seq = 256, 16, 2048
    args = (
        jax.ShapeDtypeStruct((tokens, 1, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((4096, PAGE // 8 + 1, 4, HEAD_DIM), jnp.uint8),
        jax.ShapeDtypeStruct((max_seqs,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs * pages_per_seq,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs + 1,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    fn = functools.partial(block_scores, kernel=kernel, large_kernel=kernel)
    with use_abstract_mesh(tpu7x):
        lowered = jax.jit(fn).trace(*args).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()


@pytest.mark.parametrize("kernel", ["v2", "v3"])
def test_fp4_lowers_for_tpu(kernel):
    tokens, max_seqs, pages_per_seq = 256, 16, 2048
    args = (
        jax.ShapeDtypeStruct((tokens, 1, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((4096, PAGE // 8, 4, HEAD_DIM), jnp.uint8),
        jax.ShapeDtypeStruct((max_seqs,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs * pages_per_seq,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs + 1,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((4096, 8, 128), jnp.float32),
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))

    def f(q, cache, seq_lens, tables, cu_q, n, scales):
        return block_scores(
            q,
            cache,
            seq_lens,
            tables,
            cu_q,
            n,
            kernel=kernel,
            large_kernel=kernel,
            scale_cache=scales,
            scale_row=2,
        )

    with use_abstract_mesh(tpu7x):
        lowered = jax.jit(f).trace(*args).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()


@pytest.mark.parametrize("folded_tile_q", [8, 16])
def test_int4_fp8_keys_match_bf16_keys(monkeypatch, folded_tile_q):
    """`set_int4_fp8_keys`: transposed folded 4-bit scoring with the int4 keys
    fed as fp8 gives the same scores as feeding them as bf16."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4, msa_indexer

    rng = np.random.default_rng(11)
    q_lens, ctx_lens = [4, 4, 2], [600, 250, 900]
    num_seqs, num_real = len(q_lens), sum(q_lens)
    num_tokens = num_real + 2
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(MAX_SEQS + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    tables = np.zeros((MAX_SEQS, PAGES_PER_SEQ), np.int32)
    tables[:num_seqs] = rng.permutation(np.arange(1, NUM_PAGES))[
        : num_seqs * PAGES_PER_SEQ
    ].reshape(num_seqs, PAGES_PER_SEQ)
    raw = rng.standard_normal((NUM_PAGES, PAGE, HEAD_DIM))
    nib, scale = msa_fp4.quantize(msa_fp4.rotate_keys(jnp.asarray(raw, jnp.float32)))
    cache = msa_fp4.pack_rows(nib, scales=scale)
    q = msa_fp4.rotate_queries(
        jnp.asarray(rng.standard_normal((num_tokens, 1, HEAD_DIM))).astype(jnp.bfloat16)
    )
    args = (q, cache, jnp.asarray(seq_lens), jnp.asarray(tables.reshape(-1)), jnp.asarray(cu_q), jnp.int32(num_seqs))
    kw = dict(folded_tile_q=folded_tile_q, pages_per_step=4, small_pages_per_step=4, interpret=True)
    out = {}
    for enabled in (False, True):
        monkeypatch.setattr(msa_indexer, "_int4_fp8_keys", enabled)
        out[enabled] = np.asarray(block_scores.__wrapped__(*args, **kw))[:, 0]
    for r in range(num_seqs):
        live = -(-seq_lens[r] // PAGE)
        rows = slice(cu_q[r], cu_q[r + 1])
        np.testing.assert_array_equal(out[True][rows, :live], out[False][rows, :live])


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
def test_large_class_pages_per_step_same_scores(dtype):
    """Key minimax_indexer_large_pages_per_step widens the large (prefill)
    class's grid step only: the scores stay bit-identical."""
    rng = np.random.default_rng(7)
    q_lens, ctx_lens = [1, 130, 3], [300, 200, 0]
    num_seqs = len(q_lens)
    num_real = sum(q_lens)
    num_tokens = num_real + 5
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(MAX_SEQS + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    tables = np.zeros((MAX_SEQS, PAGES_PER_SEQ), np.int32)
    tables[:num_seqs] = rng.permutation(np.arange(1, NUM_PAGES))[
        : num_seqs * PAGES_PER_SEQ
    ].reshape(num_seqs, PAGES_PER_SEQ)
    keys = rounded(rng.standard_normal((NUM_PAGES, PAGE, HEAD_DIM)), dtype)
    cache = pack(keys, dtype)
    index_q = jnp.asarray(rng.standard_normal((num_tokens, 1, HEAD_DIM))).astype(
        jnp.bfloat16
    )

    def scores(pages_per_step):
        return np.asarray(
            block_scores(
                index_q,
                cache,
                jnp.asarray(seq_lens),
                jnp.asarray(tables.reshape(-1)),
                jnp.asarray(cu_q),
                jnp.int32(num_seqs),
                block_q=64,
                large_block_q=48,
                pages_per_step=pages_per_step,
                small_pages_per_step=2,
                interpret=True,
            )
        )

    def live(s):
        # Only the rows' own pages are defined (as in `test_matches_reference`).
        return [
            s[t, 0, : -(-seq_lens[r] // PAGE)].view(np.int32)
            for r in range(num_seqs)
            for t in range(cu_q[r], cu_q[r + 1])
        ]

    want = live(scores(2))
    for pages_per_step in (4, 8):
        for a, b in zip(want, live(scores(pages_per_step)), strict=True):
            np.testing.assert_array_equal(a, b)
