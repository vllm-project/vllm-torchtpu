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
"""`batched_rpa_new` 2-pass PCP RPA vs a dense causal-attention reference.

Each case runs two steps on the same KV cache: a prefill of every request's
prefix (`cache_len = 0`), then the rest (`cache_len > 0`, chunked prefill),
with a 1-token decode in the second step. Q/K/V are packed rank-major in the
sharding mode's token layout; outputs are unpacked back to request-major.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa_new import schedule_cp, wrapper
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order,
)
from vllm_torchtpu.layers.adapter import pcp_attention

pytestmark = pytest.mark.multichip

PCP = 4
PAGE_SIZE = 128
INTERLEAVE = 64
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
HEAD_DIM = 128
DTYPE = jnp.bfloat16


def _head_tail_token_order(q_lens, pcp, padded_num_tokens):
    """Rank-major source rows for DualChunkSwap: rank `r` holds chunks `r`
    and `2P-1-r` of every request, -1 for padding."""
    local = [[] for _ in range(pcp)]
    start = 0
    for q_len in q_lens:
        chunk = -(-q_len // (2 * pcp))
        for r in range(pcp):
            for c in (r, 2 * pcp - 1 - r):
                lo, hi = min(c * chunk, q_len), min((c + 1) * chunk, q_len)
                local[r].extend(range(start + lo, start + hi))
        start += q_len
    width = padded_num_tokens // pcp
    order = np.full(padded_num_tokens, -1, dtype=np.int64)
    for r, rows in enumerate(local):
        assert len(rows) <= width
        order[r * width : r * width + len(rows)] = rows
    return order


def _token_order(sharding_mode, q_lens, padded_num_tokens):
    if sharding_mode == "head_tail":
        return _head_tail_token_order(q_lens, PCP, padded_num_tokens)
    owner_starts = np.concatenate([[0], np.cumsum(q_lens)[:-1]])
    order, _ = build_pcp_rank_major_token_order(
        q_lens,
        PCP,
        INTERLEAVE,
        padded_num_tokens,
        token_owner_start_offsets_per_req=owner_starts,
    )
    return order


def _reference(q, k, v, q_positions, kv_lens, q_lens, kv_offsets):
    """Causal GQA attention of each request's new queries over its first
    `kv_len` tokens of the full K/V."""
    out = np.zeros_like(q)
    group = NUM_Q_HEADS // NUM_KV_HEADS
    row = 0
    for s, q_len in enumerate(q_lens):
        ks = k[kv_offsets[s] : kv_offsets[s] + kv_lens[s]]
        vs = v[kv_offsets[s] : kv_offsets[s] + kv_lens[s]]
        for i in range(q_len):
            pos = q_positions[s] + i
            for h in range(NUM_Q_HEADS):
                logits = ks[: pos + 1, h // group] @ q[row + i, h] / np.sqrt(HEAD_DIM)
                p = np.exp(logits - logits.max())
                out[row + i, h] = (p / p.sum()) @ vs[: pos + 1, h // group]
        row += q_len
    return out


def _run_step(mesh, kv_cache, sharding_mode, q, k, v, q_lens, kv_lens, page_indices):
    num_tokens = int(sum(q_lens))
    local_padded = -(-num_tokens // PCP) + 2 * PCP * len(q_lens)
    local_padded = -(-local_padded // 128) * 128
    order = _token_order(sharding_mode, q_lens, local_padded * PCP)
    valid = order >= 0
    src = np.where(valid, order, 0)

    def pack(x):
        return jnp.asarray(np.where(valid[:, None, None], x[src], 0), dtype=DTYPE)

    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    args = (
        mesh,
        pack(q),
        pack(k),
        pack(v),
        kv_cache,
        jnp.asarray(kv_lens, dtype=jnp.int32),
        jnp.asarray(page_indices, dtype=jnp.int32),
        jnp.asarray(cu_q_lens),
        jnp.asarray([0, 0, len(q_lens)], dtype=jnp.int32),
    )
    kwargs = dict(
        sm_scale=HEAD_DIM**-0.5,
        k_scale=None,
        v_scale=None,
        sharding_mode=sharding_mode,
        cp_kv_cache_interleave_size=INTERLEAVE if sharding_mode == "interleaved" else 1,
        kv_layout=wrapper.configs.KVLayout.HEAD_ALONG_SUBLANE,
    )
    # The cache pass reads the KV cache before the new-tokens pass writes it.
    cache_out, cache_lse = pcp_attention.pcp_cache_attention(*args, **kwargs)
    new_out, kv_cache, new_lse = pcp_attention.pcp_new_tokens_attention(*args, **kwargs)
    out_packed, _ = schedule_cp.merge_attn_results(
        cache_out, cache_lse, new_out, new_lse
    )
    out_packed = np.asarray(out_packed.astype(jnp.float32))
    out = np.zeros((num_tokens, NUM_Q_HEADS, HEAD_DIM), dtype=np.float32)
    out[order[valid]] = out_packed[valid][..., :HEAD_DIM]
    return out, kv_cache


@pytest.mark.parametrize("sharding_mode", ["interleaved", "head_tail"])
def test_pcp_rpa_matches_dense_reference(sharding_mode):
    if len(jax.devices()) < PCP:
        pytest.skip(f"needs {PCP} devices")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:PCP]), ("pcp",))
    # Prefix and suffix lengths per request: a multiple of `P * I`, one that
    # is not (K1's partial last chunk), a short one, and a decode.
    prefix_lens = [1024, 1025, 300, 777]
    suffix_lens = [512, 1025, 17, 1]
    total_lens = [p + s for p, s in zip(prefix_lens, suffix_lens)]
    num_seqs = len(total_lens)

    rng = np.random.default_rng(0)
    q_full = rng.standard_normal((sum(total_lens), NUM_Q_HEADS, HEAD_DIM))
    k_full = rng.standard_normal((sum(total_lens), NUM_KV_HEADS, HEAD_DIM))
    v_full = rng.standard_normal((sum(total_lens), NUM_KV_HEADS, HEAD_DIM))
    # bf16-round the inputs so the reference sees what the kernel sees.
    q_full, k_full, v_full = (
        np.asarray(jnp.asarray(x, dtype=DTYPE).astype(jnp.float32))
        for x in (q_full, k_full, v_full)
    )
    offsets = np.concatenate([[0], np.cumsum(total_lens)[:-1]])

    # One logical block is one page on each PCP rank.
    blocks_per_seq = -(-max(total_lens) // (PAGE_SIZE * PCP))
    page_indices = np.arange(num_seqs * blocks_per_seq).reshape(num_seqs, -1)
    kv_cache = jnp.zeros(
        wrapper.get_kv_cache_shape(
            PCP * num_seqs * blocks_per_seq, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, DTYPE
        ),
        dtype=DTYPE,
    )

    for cache_lens, q_lens in (
        (
            [0] * num_seqs,
            prefix_lens,
        ),
        (prefix_lens, suffix_lens),
    ):
        rows = np.concatenate(
            [
                np.arange(o + c, o + c + n)
                for o, c, n in zip(offsets, cache_lens, q_lens)
            ]
        )
        kv_lens = [c + n for c, n in zip(cache_lens, q_lens)]
        out, kv_cache = _run_step(
            mesh,
            kv_cache,
            sharding_mode,
            q_full[rows],
            k_full[rows],
            v_full[rows],
            q_lens,
            kv_lens,
            page_indices,
        )
        expected = _reference(
            q_full[rows], k_full, v_full, cache_lens, kv_lens, q_lens, offsets
        )
        np.testing.assert_allclose(out, expected, atol=3e-2, rtol=3e-2)


def test_unknown_sharding_mode_is_rejected():
    with pytest.raises(ValueError, match="sharding_mode"):
        pcp_attention.build_pcp_kernels(1.0, None, None, "rows", 256)
