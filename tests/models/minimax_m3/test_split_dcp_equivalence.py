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
"""The MSA DCP forward of a step equals its two token halves run in order
(`models.minimax_m3.split`), in Pallas interpret mode on CPU.

Two DCP ranks are emulated: each writes its tokens and scores its blocks, the
candidates are stacked as the all-gather would, each attends, and the partials
are merged by log-sum-exp as `ops.msa.merge_head_partials` does.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp
from vllm_torchtpu.models.minimax_m3.split import split_query_layout

D = 128
BLOCK = 128
DCP = 2
TOPK = 16
Q_HEADS = 16  # the DCP group's query heads of its one KV head
# Compiled Pallas on a TPU (real step sizes) instead of interpret mode on CPU.
ON_TPU = os.environ.get("MSA_SPLIT_TEST_TPU") == "1"
INTERPRET = not ON_TPU


def _forward(caches, step, *, num_reqs, dtype):
    """Merged attention ``[T, H, D]`` and the ranks' updated caches."""
    q, k, v, iq, ik, seq_lens, block_tables, qsl = step
    dist = jnp.array([0, 0, num_reqs], jnp.int32)
    new_caches, cands = [], []
    for r in range(DCP):
        kv, idx = caches[r]
        kv, idx, cand = msa_dcp.msa_write_and_candidates_dcp(
            k, v, iq, ik, kv, idx, seq_lens, block_tables, qsl, dist,
            topk=TOPK, init_blocks=0, local_blocks=1, rank=r, dcp=DCP,
            interpret=INTERPRET,
        )
        new_caches.append((kv, idx))
        cands.append(cand)
    cands = jnp.stack(cands)
    outs, lses = [], []
    for r in range(DCP):
        out, lse = msa_dcp.msa_attend_dcp(
            q, new_caches[r][0], cands, seq_lens, block_tables, qsl,
            sm_scale=D**-0.5, topk=TOPK, rank=r, dcp=DCP, interpret=INTERPRET,
        )
        outs.append(out.astype(jnp.float32))
        lses.append(lse.astype(jnp.float32))
    lses = jnp.stack(lses)  # [dcp, H, T]
    w = jnp.exp(lses - jax.scipy.special.logsumexp(lses, axis=0))
    merged = sum(jnp.where(w[r][..., None] > 0, outs[r] * w[r][..., None], 0)
                 for r in range(DCP))
    return merged.transpose(1, 0, 2), new_caches


def _case(dtype, q_lens, ctx_lens, max_seqs=8, seed=0, pad=0):
    rng = np.random.default_rng(seed)
    t_real = sum(q_lens)
    t = t_real + pad  # padded bucket rows, as the runner sends them
    blocks = [-(-(c + q) // (DCP * BLOCK)) for c, q in zip(ctx_lens, q_lens)]
    pages_per_seq = max(16, -(-max(blocks) // 16) * 16)
    num_pages = sum(blocks) + 8
    seq_lens = np.zeros(max_seqs, np.int32)
    qsl = np.full(max_seqs + 1, t_real, np.int32)  # tail filled with the last value
    qsl[0] = 0
    for i, (ql, cl) in enumerate(zip(q_lens, ctx_lens)):
        seq_lens[i] = cl + ql
        qsl[i + 1] = qsl[i] + ql
    # Every request its own pages; unused entries point at a spare page.
    perm = rng.permutation(num_pages)
    block_tables = np.full((max_seqs, pages_per_seq), perm[-1], np.int32)
    used = 0
    for i, b in enumerate(blocks):
        block_tables[i, :b] = perm[used : used + b]
        used += b
    block_tables = block_tables.reshape(-1)
    packing = 32 // (8 * jnp.dtype(dtype).itemsize)

    def cache(width):
        x = rng.standard_normal((num_pages, BLOCK // packing, packing, width), np.float32)
        return jnp.asarray(x * 0.5, dtype)

    caches = [(cache(2 * D), cache(D)) for _ in range(DCP)]
    f = lambda *s: jnp.asarray(rng.standard_normal(s, np.float32) * 0.5, jnp.bfloat16)  # noqa: E731
    step = (f(t, Q_HEADS, D), f(t, 1, D), f(t, 1, D), f(t, 1, D), f(t, D),
            jnp.asarray(seq_lens), jnp.asarray(block_tables), jnp.asarray(qsl))
    return caches, step, len(q_lens)


# (q_lens, ctx_lens, max_seqs[, padding rows])
CASES = [
    # decode-sized rows, a chunk straddling the split, a fresh prompt
    ([8, 760, 512], [3000, 2500, 0], 8),
    # one long chunk over a long context (the usual 8K split step)
    ([1024], [5000], 8),
    # padded bucket: verify windows + a chunk, 324 padding rows after the real tokens
    ([8, 8, 684], [3000, 2600, 4000], 8, 324),
]
if ON_TPU:
    CASES += [
        # served shapes (max_num_seqs 32): 8192-token step, 30 verify windows of
        # 8 rows beside a 7952-token chunk over a long context
        ([8] * 30 + [7952], [20000 + 1000 * i for i in range(30)] + [100000], 32),
        # 4096-token step: halves of 2048 take the decode-sized scoring path
        ([8] * 8 + [4032], [30000] * 8 + [60000], 32),
        # one 8192-token chunk over 120K
        ([8192], [120000], 32),
        # two prompts side by side, the split inside the second
        ([3000, 5192], [0, 40000], 32),
    ]


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize("case", CASES)
def test_halves_equal_whole(dtype, case):
    q_lens, ctx_lens, max_seqs, *pad = case
    caches, step, n = _case(dtype, q_lens, ctx_lens, max_seqs, pad=pad[0] if pad else 0)
    t_real = sum(q_lens)
    whole, whole_caches = _forward(caches, step, num_reqs=n, dtype=dtype)

    q, k, v, iq, ik, seq_lens, block_tables, qsl = step
    t = q.shape[0]
    split = t // 2
    q1, l1, q2, l2 = split_query_layout(qsl, seq_lens, split)
    s1 = (q[:split], k[:split], v[:split], iq[:split], ik[:split], l1, block_tables, q1)
    s2 = (q[split:], k[split:], v[split:], iq[split:], ik[split:], l2, block_tables, q2)
    a, mid = _forward(caches, s1, num_reqs=n, dtype=dtype)
    b, half_caches = _forward(mid, s2, num_reqs=n, dtype=dtype)
    halves = jnp.concatenate([a, b])

    for (wk, wi), (hk, hi) in zip(whole_caches, half_caches):
        np.testing.assert_array_equal(np.asarray(wk, np.float32), np.asarray(hk, np.float32))
        np.testing.assert_array_equal(np.asarray(wi, np.float32), np.asarray(hi, np.float32))
    err = np.abs(np.asarray(halves)[:t_real] - np.asarray(whole)[:t_real])
    rows = np.where(err.max(axis=(1, 2)) > 2e-2)[0]
    assert rows.size == 0, (
        f"{rows.size}/{t} rows differ (first {rows[:8]}, max {err.max():.3g}); "
        f"split at {split}, qsl {np.asarray(qsl)[: n + 1]}"
    )
