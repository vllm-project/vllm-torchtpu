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
"""Index-cache sharding (interpret mode): S shard caches and the max over
ranks of their partial block scores against the unsharded cache and scores."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3.packing import CACHE_DTYPES, pack, rounded, unpack
from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_write

PAGE = 128
HEAD_DIM = 128
PAGES_PER_SEQ = 8
MAX_SEQS = 4
NUM_PAGES = 40


def _xla_write_rows(cache, rows, pages, slots, *, max_num_seqs, interpret=False):
    """`msa_write.write_rows` as an XLA scatter (same contract).

    The Pallas write is covered by test_msa_write; this keeps the test on the
    sharding logic and runnable where interpret mode lacks bitcasts.
    """
    del max_num_seqs, interpret
    packing = cache.shape[2]
    return cache.at[pages, slots // packing, slots % packing].set(rows, mode="drop")


@pytest.fixture(autouse=True)
def _scatter_writes(monkeypatch):
    monkeypatch.setattr(msa_write, "write_rows", _xla_write_rows)


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
@pytest.mark.parametrize("shards", [4, 2])
@pytest.mark.parametrize(
    "q_lens,ctx_lens",
    [
        ([1, 130, 3], [300, 200, 0]),  # decode, long chunk (large tiles), short
        ([4, 4, 4, 4], [1000, 5, 127, 128]),  # spec-decode rows across page ends
    ],
)
def test_sharded_scores_match_unsharded(q_lens, ctx_lens, shards, dtype):
    rng = np.random.default_rng(len(q_lens) + shards)
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
    index_k = jnp.asarray(rng.standard_normal((num_tokens, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    index_q = jnp.asarray(rng.standard_normal((num_tokens, shards, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    meta = (
        jnp.asarray(seq_lens),
        jnp.asarray(tables.reshape(-1)),
        jnp.asarray(cu_q),
        jnp.int32(num_seqs),
    )

    # Unsharded reference: write all rows, score each head.
    seq_id, pos, valid = msa.token_positions(num_tokens, meta[0], meta[2])
    pages = msa._token_pages(jnp.asarray(tables), seq_id, pos, valid, PAGE)
    full = msa.write_rows(
        pack(keys, dtype), index_k, pages, pos, max_num_seqs=MAX_SEQS, interpret=True
    )
    expected = np.stack(
        [
            np.asarray(msa._block_scores(index_q[:, h : h + 1], full, *meta, True))[
                :, 0
            ]
            for h in range(shards)
        ]
    )

    combined = None
    full_rows = unpack(full)
    for rank in range(shards):
        shard = pack(keys[:, rank::shards], dtype)
        shard = msa.write_index_shard(
            shard,
            index_k,
            pages,
            pos,
            valid,
            rank=rank,
            shards=shards,
            max_num_seqs=MAX_SEQS,
            interpret=True,
        )
        np.testing.assert_array_equal(unpack(shard), full_rows[:, rank::shards])
        partial = np.asarray(
            msa.partial_block_scores(
                index_q, shard, *meta, shards=shards, interpret=True
            )
        )
        combined = partial if combined is None else np.maximum(combined, partial)

    for s in range(num_seqs):
        num_pages = -(-seq_lens[s] // PAGE)
        rows = slice(cu_q[s], cu_q[s + 1])
        np.testing.assert_array_equal(
            combined[:, rows, :num_pages], expected[:, rows, :num_pages]
        )


@pytest.mark.parametrize("shards", [4, 2])
def test_shard_scores_count_logical_pages(monkeypatch, shards):
    """block_scores must see each request's logical page count, not the shard
    page count (4x more at S = 4): extra page groups write past the scores'
    padded pages, a TPU bounds-check halt at long contexts."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_indexer

    seen = {}
    real = msa_indexer.block_scores

    def spy(index_q, index_cache, seq_lens, *args, **kwargs):
        seen["pages"] = -(-np.asarray(seq_lens) // msa.page_size(index_cache))
        return real(index_q, index_cache, seq_lens, *args, **kwargs)

    monkeypatch.setattr(msa_indexer, "block_scores", spy)
    seq_lens = np.array([PAGES_PER_SEQ * PAGE, 1, PAGE + 1, 0], np.int32)
    num_tokens = 4
    cu_q = np.array([0, 1, 2, 3, 4], np.int32)
    tables = np.arange(MAX_SEQS * PAGES_PER_SEQ, dtype=np.int32) % NUM_PAGES
    shard = pack(
        np.zeros((NUM_PAGES, PAGE // shards, HEAD_DIM), np.float32), jnp.bfloat16
    )
    index_q = jnp.zeros((num_tokens, shards, HEAD_DIM), jnp.bfloat16)
    msa.partial_block_scores(
        index_q,
        shard,
        jnp.asarray(seq_lens),
        jnp.asarray(tables),
        jnp.asarray(cu_q),
        jnp.int32(3),
        shards=shards,
        interpret=True,
    )
    np.testing.assert_array_equal(seen["pages"], -(-seq_lens // PAGE))


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
def test_sharded_forward_lowers_for_tpu(dtype):
    """Both halves of the sharded forward lower to Mosaic without a TPU."""
    import functools

    import jax
    from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

    from tests.models.minimax_m3.packing import PACKING

    tokens, max_seqs, pages_per_seq, shards, heads = 256, 16, 2048, 4, 16
    packing = PACKING[dtype]
    i32 = jnp.int32
    meta = (
        jax.ShapeDtypeStruct((max_seqs,), i32),
        jax.ShapeDtypeStruct((max_seqs * pages_per_seq,), i32),
        jax.ShapeDtypeStruct((max_seqs + 1,), i32),
    )
    kv = jax.ShapeDtypeStruct((4096, PAGE // packing, packing, 2 * HEAD_DIM), dtype)
    score_args = (
        jax.ShapeDtypeStruct((tokens, 1, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, 1, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, shards, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, HEAD_DIM), jnp.bfloat16),
        kv,
        jax.ShapeDtypeStruct(
            (4096, PAGE // shards // packing, packing, HEAD_DIM), dtype
        ),
        *meta,
        jax.ShapeDtypeStruct((3,), i32),
    )
    attend_args = (
        jax.ShapeDtypeStruct((tokens, heads, HEAD_DIM), jnp.bfloat16),
        kv,
        jax.ShapeDtypeStruct((tokens, 1, pages_per_seq), jnp.float32),
        *meta,
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        for fn, args in (
            (
                functools.partial(
                    msa.msa_write_and_score_sharded, rank=1, shards=shards
                ),
                score_args,
            ),
            (
                functools.partial(
                    msa.msa_attend_sharded,
                    sm_scale=HEAD_DIM**-0.5,
                    topk=16,
                    init_blocks=0,
                    local_blocks=1,
                ),
                attend_args,
            ),
        ):
            lowered = jax.jit(fn).trace(*args).lower(lowering_platforms=("tpu",))
            assert "tpu_custom_call" in lowered.as_text()
