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
"""DSv4-Pro sliding-window attention block sweep at the served per-rank prefill
shape (TP8: 16 heads x 512, window 128, 128-token SWA pages overlaid on the
256-row CSA page, bucket T tokens = one chunk at a deep prefix + a few verify
windows). Times each (bq_mixed, q_compute_block, kv_block_tokens) and checks
the outputs against the production config (1,32,32)/4/256.

    python3 tests/kernels/bench_dsv4_swa_blocks.py [--T 2048] [--prefix 100000]
"""

import argparse
import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.core_attention import mla_swa

H, D, W, PAGE = 16, 512, 128, 128
VQ = 7


def build(T, prefix, n_verify, seed=0):
    rng = np.random.default_rng(seed)
    chunk = T - n_verify * VQ - 7  # leave bucket padding
    q_lens = [VQ] * n_verify + ([chunk] if chunk > 0 else [])
    ctx = [prefix + 1000 * i for i in range(n_verify)] + ([prefix] if chunk > 0 else [])
    kv_lens = [c + q for c, q in zip(ctx, q_lens)]
    n = len(q_lens)
    max_seqs = 8
    pages_per_seq = max(-(-kl // PAGE) for kl in kv_lens) + 1
    total_pages = n * pages_per_seq + 1
    table = np.zeros((max_seqs, pages_per_seq), np.int32)
    perm = rng.permutation(total_pages - 1).astype(np.int32) + 1
    for i in range(n):
        table[i] = perm[i * pages_per_seq : (i + 1) * pages_per_seq]
    cache = jnp.zeros((total_pages, PAGE * 2, 4, 128), jnp.uint8)
    cache = csa_cache_layout.slabs_to_words(cache)
    q = jax.random.normal(jax.random.PRNGKey(seed), (T, H, D), jnp.float32).astype(
        jnp.bfloat16
    )
    new_kv = jax.random.normal(jax.random.PRNGKey(seed + 1), (T, D), jnp.float32).astype(
        jnp.bfloat16
    )
    cu = np.zeros(max_seqs + 1, np.int32)
    cu[1 : n + 1] = np.cumsum(q_lens)
    cu[n + 1 :] = cu[n]
    kl = np.zeros(max_seqs, np.int32)
    kl[:n] = kv_lens
    dist = np.array([0, n_verify, n], np.int32)
    sinks = jnp.asarray(rng.random(H, dtype=np.float32))
    return dict(
        q=q,
        new_kv=new_kv,
        cache_kv=cache,
        kv_lens=jnp.asarray(kl),
        page_indices=jnp.asarray(table.reshape(-1)),
        cu_q_lens=jnp.asarray(cu),
        distribution=jnp.asarray(dist),
        attention_sinks=sinks,
    ), int(cu[n])


def run(inp, bq, qcb, kvtok):
    return mla_swa.mla_sliding_window_ragged_paged_attention(
        **inp,
        sm_scale=0.04,
        sliding_window=W,
        logical_page_size=PAGE,
        num_kv_pages_per_block=(max(1, kvtok // PAGE),) * 3,
        num_queries_per_block=(1, bq, bq),
        q_compute_block_size=qcb,
        unnormalized_output=True,
        chunk_prefill_size=VQ,
    )


def timeit(f, iters=20):
    jax.block_until_ready(f())
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        outs = [f() for _ in range(iters)]
        jax.block_until_ready(outs[-1])
        best = min(best, (time.perf_counter() - t0) * 1e3 / iters)
    return round(best, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--prefix", type=int, default=100000)
    ap.add_argument("--verify", type=int, default=2)
    ap.add_argument("--combos", default="")
    a = ap.parse_args()
    inp, rows = build(a.T, a.prefix, a.verify)
    fn = jax.jit(run, static_argnums=(1, 2, 3))
    base = [np.asarray(x) for x in fn(inp, 32, 4, 256)]
    combos = [(32, 4, 256)] + [
        c
        for c in itertools.product([32, 64, 128, 256], [4, 8, 16], [256, 512])
        if c != (32, 4, 256)
    ]
    if a.combos:
        combos = [tuple(int(x) for x in c.split(":")) for c in a.combos.split(",")]
    for bq, qcb, kvtok in combos:
        try:
            out = [np.asarray(x) for x in fn(inp, bq, qcb, kvtok)]
            o, ob = out[0][:rows].astype(np.float32), base[0][:rows].astype(np.float32)
            diff = float(np.max(np.abs(o - ob)))
            exact = bool(np.array_equal(out[0][:rows], base[0][:rows]))
            ms = timeit(lambda: fn(inp, bq, qcb, kvtok))
            print({"T": a.T, "prefix": a.prefix, "bq": bq, "qcb": qcb,
                   "kv_tokens": kvtok, "ms": ms, "max_abs_diff": diff,
                   "bitexact": exact}, flush=True)
        except Exception as e:  # noqa
            print({"bq": bq, "qcb": qcb, "kv_tokens": kvtok,
                   "error": repr(e)[:300]}, flush=True)


if __name__ == "__main__":
    main()
