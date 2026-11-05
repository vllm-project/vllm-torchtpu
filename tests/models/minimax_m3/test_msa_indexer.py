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
