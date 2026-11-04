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
"""MSA block selection and sparse attention against a brute-force reference."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    ref_ragged_paged_attention,
)
from vllm_torchtpu.models.minimax_m3.kernels import msa

BLOCK = 16
HEAD_DIM = 128
Q_PER_KV = 4
TOPK = 4


def _padded_ref_rpa(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, dist, **kw):
    out, cache = ref_ragged_paged_attention(
        q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, dist, **kw
    )
    return jnp.pad(out, ((0, q.shape[0] - out.shape[0]), (0, 0), (0, 0))), cache


def _jittable_single_token_rpa(
    q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, dist, *, sm_scale, **kw
):
    """Traceable stand-in for the kernel on one-token sequences."""
    del k, v, cu_q_lens, dist, kw
    rows, num_heads, head_dim = q.shape
    pages = page_indices.reshape(rows, -1)
    kv = kv_cache[pages].reshape(rows, -1, 2, head_dim)
    logits = jnp.einsum("rhd,rkd->rhk", q, kv[:, :, 0]) * sm_scale
    visible = jnp.arange(kv.shape[1])[None, None] < kv_lens[:, None, None]
    w = jax.nn.softmax(jnp.where(visible, logits, -jnp.inf), axis=-1)
    out = jnp.einsum("rhk,rkd->rhd", jnp.nan_to_num(w), kv[:, :, 1])
    return out, kv_cache


def _reference(q, k, v, iq, ik, ctx_k, ctx_v, ctx_ik, q_lens, ctx_lens, sm_scale):
    """Dense numpy MSA over each request's full context + new tokens."""
    outs = []
    start = 0
    for r, (q_len, ctx_len) in enumerate(zip(q_lens, ctx_lens)):
        keys = np.concatenate([ctx_k[r][:ctx_len], k[start : start + q_len, 0]])
        vals = np.concatenate([ctx_v[r][:ctx_len], v[start : start + q_len, 0]])
        ikeys = np.concatenate([ctx_ik[r][:ctx_len], ik[start : start + q_len]])
        for i in range(q_len):
            p = ctx_len + i
            s = ikeys[: p + 1] @ iq[start + i, 0]
            nvis = p // BLOCK + 1
            bscore = np.full(nvis, -np.inf)
            for b in range(nvis):
                bscore[b] = s[b * BLOCK : (b + 1) * BLOCK].max()
            bscore[nvis - 1] = np.inf
            chosen = np.argsort(-bscore)[: min(TOPK, nvis)]
            mask = np.zeros(p + 1, bool)
            for b in chosen:
                mask[b * BLOCK : (b + 1) * BLOCK] = True
            mask = mask[: p + 1]
            logits = (q[start + i] @ keys[: p + 1].T) * sm_scale
            logits = np.where(mask[None], logits, -np.inf)
            w = np.exp(logits - logits.max(-1, keepdims=True))
            w /= w.sum(-1, keepdims=True)
            outs.append(w @ vals[: p + 1])
        start += q_len
    return np.stack(outs)


@pytest.mark.parametrize("small_chunks", [False, True])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize(
    "q_lens,ctx_lens",
    [
        ([1, 1, 1], [70, 5, 200]),  # decode
        ([40], [0]),  # first prefill chunk
        ([1, 37, 3], [129, 60, 17]),  # mixed batch
    ],
)
def test_msa_forward_matches_dense_reference(
    q_lens, ctx_lens, jit, small_chunks, monkeypatch
):
    if small_chunks:
        # Several scoring iterations per request, and padding past the table.
        monkeypatch.setattr(msa, "_PAGES_PER_CHUNK", 2)
        monkeypatch.setattr(msa, "_SMALL_WINDOW_PAGES_PER_CHUNK", 4)
    rng = np.random.default_rng(0)
    num_seqs = len(q_lens)
    max_num_seqs = num_seqs + 1
    pages_per_seq = 16
    num_pages = max_num_seqs * pages_per_seq + 1
    num_real = sum(q_lens)
    num_tokens = num_real + 3
    num_heads = Q_PER_KV
    f32 = np.float32

    q = rng.standard_normal((num_tokens, num_heads, HEAD_DIM)).astype(f32)
    k = rng.standard_normal((num_tokens, 1, HEAD_DIM)).astype(f32)
    v = rng.standard_normal((num_tokens, 1, HEAD_DIM)).astype(f32)
    iq = rng.standard_normal((num_tokens, 1, HEAD_DIM)).astype(f32)
    ik = rng.standard_normal((num_tokens, HEAD_DIM)).astype(f32)
    ctx_k = rng.standard_normal((num_seqs, 256, HEAD_DIM)).astype(f32)
    ctx_v = rng.standard_normal((num_seqs, 256, HEAD_DIM)).astype(f32)
    ctx_ik = rng.standard_normal((num_seqs, 256, HEAD_DIM)).astype(f32)

    # Shuffled physical pages; page 0 stays unused.
    perm = rng.permutation(np.arange(1, num_pages))
    block_tables = np.zeros((max_num_seqs, pages_per_seq), np.int32)
    block_tables[:num_seqs] = perm[: num_seqs * pages_per_seq].reshape(num_seqs, -1)

    # f32 packs one value per word: K and V are separate rows of the head axis.
    kv_cache = np.zeros((num_pages, BLOCK, 2, 1, HEAD_DIM), f32)
    index_cache = np.zeros((num_pages, BLOCK, 1, HEAD_DIM), f32)
    for r in range(num_seqs):
        for j in range(ctx_lens[r]):
            page = block_tables[r, j // BLOCK]
            kv_cache[page, j % BLOCK, 0, 0] = ctx_k[r, j]
            kv_cache[page, j % BLOCK, 1, 0] = ctx_v[r, j]
            index_cache[page, j % BLOCK, 0] = ctx_ik[r, j]

    seq_lens = np.zeros(max_num_seqs, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(max_num_seqs + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    dist = np.array([num_seqs] * 3, np.int32)
    sm_scale = HEAD_DIM**-0.5

    forward = functools.partial(
        msa.msa_forward,
        sm_scale=sm_scale,
        topk=TOPK,
        init_blocks=0,
        local_blocks=1,
        rpa_func=_jittable_single_token_rpa if jit else _padded_ref_rpa,
    )
    if jit:
        forward = jax.jit(forward)
    new_kv, new_idx, out = forward(
        *(jnp.asarray(x) for x in (q, k, v, iq, ik, kv_cache, index_cache)),
        *(jnp.asarray(x) for x in (seq_lens, block_tables.reshape(-1), cu_q, dist)),
    )

    expected = _reference(
        q, k, v, iq, ik, ctx_k, ctx_v, ctx_ik, q_lens, ctx_lens, sm_scale
    )
    np.testing.assert_allclose(np.asarray(out[:num_real]), expected, atol=2e-4)
    np.testing.assert_array_equal(np.asarray(out[num_real:]), 0)
    start = 0
    for r in range(num_seqs):
        for i in range(q_lens[r]):
            p = ctx_lens[r] + i
            page = block_tables[r, p // BLOCK]
            np.testing.assert_array_equal(
                new_kv[page, p % BLOCK, 0, 0], k[start + i, 0]
            )
            np.testing.assert_array_equal(
                new_kv[page, p % BLOCK, 1, 0], v[start + i, 0]
            )
            np.testing.assert_array_equal(new_idx[page, p % BLOCK, 0], ik[start + i])
        start += q_lens[r]
    np.testing.assert_array_equal(np.asarray(new_kv[0]), 0)


def test_select_blocks_forces_local_and_pads():
    scores = jnp.array([[[5.0, 1.0, 9.0, 2.0, 7.0]]])
    pos = jnp.array([2 * BLOCK + 3])  # local block 2; blocks 3, 4 not visible
    picked = msa.select_blocks(
        scores, pos, block_size=BLOCK, topk=4, init_blocks=0, local_blocks=1
    )
    np.testing.assert_array_equal(np.asarray(picked), [[[0, 1, 2, -1]]])
    picked = msa.select_blocks(
        scores, pos, block_size=BLOCK, topk=2, init_blocks=0, local_blocks=1
    )
    np.testing.assert_array_equal(np.asarray(picked), [[[0, 2]]])
    picked = msa.select_blocks(
        scores, pos, block_size=BLOCK, topk=2, init_blocks=1, local_blocks=1
    )
    np.testing.assert_array_equal(np.asarray(picked), [[[0, 2]]])
