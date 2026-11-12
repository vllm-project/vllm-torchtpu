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
"""The drafter's paged window-attention kernel against its XLA reference,
and the sharded write + attention against dense attention."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.adapter.cp_draft_attention import (
    attend_block,
    attend_cached,
    attend_cached_reference,
    write_kv,
)


@pytest.mark.parametrize(
    "dcp, head_ranks, dtype, scale",
    [
        (2, 2, jnp.float8_e4m3fn, None),
        (2, 2, jnp.float8_e4m3fn, 0.5),
        (8, 2, jnp.float8_e4m3fn, None),
        (8, 2, jnp.float8_e4m3fn, 0.5),
        (8, 1, jnp.float8_e4m3fn, None),
        (8, 2, jnp.bfloat16, None),
    ],
)
def test_attend_cached_matches_reference(dcp, head_ranks, dtype, scale):
    rng = np.random.default_rng(0)
    packing = 32 // jax.dtypes.itemsize_bits(dtype)
    words = -(-2 * (dcp // head_ranks) // packing)
    cache = jnp.asarray(
        rng.standard_normal((64, 128, words, packing, 128)), jnp.float32
    ).astype(dtype)
    # Long, short, partial-block and empty (padding) requests.
    seq_lens = jnp.asarray([3000, 9, 700, 130, 0], jnp.int32)
    q_lens = jnp.asarray([8, 8, 5, 8, 0], jnp.int32)
    cu_q_lens = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(q_lens)])
    tables = jnp.asarray(rng.integers(1, 64, (5, 32)), jnp.int32)
    q = jnp.asarray(rng.standard_normal((40, 16, 128)), jnp.float32).astype(
        jnp.bfloat16
    )
    kwargs = dict(
        head_ranks=head_ranks,
        max_query_len=8,
        sliding_window=512,
        sm_scale=0.088,
        dcp=dcp,
        k_scale=scale,
        v_scale=scale,
    )
    for rank in range(dcp):
        args = (q, cache, seq_lens, tables, cu_q_lens)
        want_out, want_lse = attend_cached_reference(
            *args, rank=jnp.int32(rank), **kwargs
        )
        got_out, got_lse = attend_cached(
            *args, rank=jnp.int32(rank), interpret=True, **kwargs
        )
        np.testing.assert_allclose(
            np.asarray(got_out, np.float32),
            np.asarray(want_out, np.float32),
            rtol=2e-2,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            np.asarray(got_lse), np.asarray(want_lse), rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize("dcp, head_ranks", [(2, 2), (8, 2)])
def test_uniform_blocks_match_the_general_layout(dcp, head_ranks):
    """A draft forward's rows (`uniform_blocks`: 7 per request, padding rows
    after them): the reshaped paths equal the general ones, a request without
    cached tokens (the draft's padding requests) included."""
    rng = np.random.default_rng(1)
    block, num_reqs, num_tokens = 7, 4, 32
    words = -(-2 * (dcp // head_ranks) // 4)
    cache = jnp.asarray(
        rng.standard_normal((64, 128, words, 4, 128)), jnp.float32
    ).astype(jnp.float8_e4m3fn)
    seq_lens = jnp.asarray([3000, 7, 700, 135], jnp.int32)
    cu_q_lens = jnp.arange(num_reqs + 1, dtype=jnp.int32) * block
    tables = jnp.asarray(rng.integers(1, 64, (num_reqs, 32)), jnp.int32)
    q = jnp.asarray(rng.standard_normal((num_tokens, 16, 128)), jnp.bfloat16)
    key = jnp.asarray(rng.standard_normal((num_tokens, 1, 128)), jnp.bfloat16)
    value = jnp.asarray(rng.standard_normal((num_tokens, 1, 128)), jnp.bfloat16)
    for rank in range(dcp):
        outs = [
            attend_cached(
                q, cache, seq_lens, tables, cu_q_lens, head_ranks=head_ranks,
                max_query_len=block, sliding_window=512, sm_scale=0.088, dcp=dcp,
                rank=jnp.int32(rank), interpret=True, uniform_blocks=uniform,
            )
            for uniform in (False, True)
        ]
        for got, want in zip(outs[1], outs[0]):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    blocks = [
        attend_block(
            q, key, value, seq_lens, cu_q_lens, max_query_len=block,
            sm_scale=0.088, causal=False, uniform_blocks=uniform,
        )
        for uniform in (False, True)
    ]
    for got, want in zip(blocks[1], blocks[0]):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("dcp, head_ranks", [(8, 2), (8, 1), (2, 2)])
def test_sharded_window_attention_matches_dense(dcp, head_ranks):
    """Every rank writes its KV head's context in a prefill and two decode
    writes (odd starts, block and page crossings), then attends a 7-token
    query block per request; merging the head's ranks' partials equals dense
    sliding-window attention over the quantized context."""
    rng = np.random.default_rng(2)
    dtype, block, window, num_pages = jnp.float8_e4m3fn, 7, 512, 96
    kv_heads, q_per_rank = dcp // head_ranks, 2
    words = -(-2 * kv_heads // 4)
    ctx_lens = [1501, 300, 1033]
    num_reqs, head_dim = len(ctx_lens), 128
    tables = jnp.asarray(
        rng.permutation(np.arange(1, num_pages))[: num_reqs * 8].reshape(num_reqs, 8),
        jnp.int32,
    )
    keys = rng.standard_normal((kv_heads, num_reqs, 1600, head_dim)).astype(np.float32)
    values = rng.standard_normal((kv_heads, num_reqs, 1600, head_dim)).astype(
        np.float32
    )
    caches = [
        jnp.zeros((num_pages, 128, words, 4, head_dim), dtype) for _ in range(dcp)
    ]
    # Context written in three steps: [0, a), [a, b), [b, ctx).
    cuts = [[0, n - 13, n - 6, n] for n in ctx_lens]
    for step in range(3):
        starts = [c[step] for c in cuts]
        ends = [c[step + 1] for c in cuts]
        q_lens = np.asarray(ends) - np.asarray(starts)
        cu = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
        for rank in range(dcp):
            head = rank // head_ranks
            spans = list(enumerate(zip(starts, ends)))
            k = np.concatenate([keys[head, r, s:e] for r, (s, e) in spans])
            v = np.concatenate([values[head, r, s:e] for r, (s, e) in spans])
            caches[rank] = write_kv(
                caches[rank],
                jnp.asarray(k[:, None], jnp.bfloat16),
                jnp.asarray(v[:, None], jnp.bfloat16),
                jnp.asarray(ends, jnp.int32),
                tables,
                cu,
                dcp=dcp,
                head_ranks=head_ranks,
                rank=jnp.int32(rank),
                interpret=True,
            )
    seq_lens = jnp.asarray(ctx_lens, jnp.int32) + block
    cu = jnp.arange(num_reqs + 1, dtype=jnp.int32) * block
    q_shape = (num_reqs * block, kv_heads, head_ranks * q_per_rank, head_dim)
    q = jnp.asarray(rng.standard_normal(q_shape), jnp.bfloat16)
    for head in range(kv_heads):
        partials = [
            attend_cached(
                q[:, head],
                caches[head * head_ranks + i],
                seq_lens,
                tables,
                cu,
                head_ranks=head_ranks,
                max_query_len=block,
                sliding_window=window,
                sm_scale=0.088,
                dcp=dcp,
                rank=jnp.int32(head * head_ranks + i),
                interpret=True,
            )
            for i in range(head_ranks)
        ]
        lses = jnp.stack([lse for _, lse in partials])
        merged = jax.nn.logsumexp(lses, axis=0)
        got = sum(
            jnp.exp(lse - merged)[..., None] * out.astype(jnp.float32)
            for out, lse in partials
        )
        for r, n in enumerate(ctx_lens):
            # Written as bf16, then quantized.
            k = jnp.asarray(keys[head, r, :n], jnp.bfloat16).astype(dtype)
            v = jnp.asarray(values[head, r, :n], jnp.bfloat16).astype(dtype)
            k, v = k.astype(jnp.float32), v.astype(jnp.float32)
            qr = q[r * block : (r + 1) * block, head].astype(jnp.float32)
            s = jnp.einsum("qhd,kd->qhk", qr, k) * 0.088
            k_pos, q_pos = jnp.arange(n), n + jnp.arange(block)
            s = jnp.where((q_pos[:, None] - k_pos < window)[:, None, :], s, -jnp.inf)
            want = jnp.einsum("qhk,kd->qhd", jax.nn.softmax(s, axis=-1), v)
            np.testing.assert_allclose(
                np.asarray(got[r * block : (r + 1) * block]),
                np.asarray(want),
                rtol=2e-2,
                atol=2e-2,
            )
