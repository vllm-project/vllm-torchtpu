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
"""Microbenchmark for the MiniMax-M3 MSA DCP candidate merge.

Times `msa_dcp.merge_candidates_serial` (``topk`` rounds of max / min /
mask) against `msa_dcp.merge_candidates` (two row sorts) and
`msa_dcp.merge_candidates_pairwise` (all-pairs ranks) per layer, as the
slope ``(t(layers) - t(1)) / (layers - 1)`` of a jitted chain of dependent
merges, at the DCP shapes of MiniMax-M3.1 on TP 8: ``[dcp, T * G, 2 * topk]``
candidates with ``G = 4`` KV heads per rank at DCP 8 (1 at DCP 2).

Before timing, each shape checks that all merges return the same blocks,
so a CPU run is a correctness check (its times are not meaningful).

Usage:
  python scripts/benchmark_msa_dcp_merge_candidates.py
  python scripts/benchmark_msa_dcp_merge_candidates.py --dcp=2 --layers=8
  python scripts/benchmark_msa_dcp_merge_candidates.py --shapes=decode:512
"""

from __future__ import annotations

import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp

_IMPLS = {
    "serial": msa_dcp.merge_candidates_serial,
    "sort": msa_dcp.merge_candidates,
    "pairwise": msa_dcp.merge_candidates_pairwise,
}


def make_candidates(
    key: jax.Array, *, dcp: int, rows: int, topk: int, num_cols: int
) -> jax.Array:
    """Packed candidates as the ranks emit them: rank ``r``'s distinct blocks
    ``c * dcp + r`` by descending score, a quarter of the rows short of
    ``topk`` (``(-inf, -1)`` tails)."""
    k_cols, k_scores, k_len = jax.random.split(key, 3)
    cols = jnp.argsort(jax.random.uniform(k_cols, (dcp, rows, num_cols)), axis=-1)[
        ..., :topk
    ]
    blocks = cols * dcp + jnp.arange(dcp, dtype=jnp.int32)[:, None, None]
    # Few distinct values, so ties are common.
    scores = (
        jax.random.randint(k_scores, (dcp, rows, topk), -64, 64).astype(jnp.float32) / 8
    )
    scores = -jnp.sort(-scores, axis=-1)
    n = jnp.where(
        jnp.arange(rows) % 4 == 0,
        jax.random.randint(k_len, (dcp, rows), 0, topk + 1),
        topk,
    )
    missing = jnp.arange(topk) >= n[..., None]
    scores = jnp.where(missing, -jnp.inf, scores)
    blocks = jnp.where(missing, -1, blocks)
    return jax.vmap(msa_dcp.pack_candidates)(scores, blocks.astype(jnp.int32))


@functools.partial(jax.jit, static_argnames=("impl", "topk", "layers"))
def chain(candidates: jax.Array, *, impl: str, topk: int, layers: int):
    """``layers`` dependent merges: each layer's input depends on the
    previous layer's selection, so XLA can neither batch nor drop them."""
    merge = _IMPLS[impl]
    out = merge(candidates, topk)
    for _ in range(layers - 1):
        bump = jnp.where(out[0, 0] == -2, 1.0, 0.0).astype(candidates.dtype)
        out = merge(candidates.at[0, 0, 0].add(bump), topk)
    return out


def time_chain(candidates, *, impl, topk, layers, iters):
    fn = functools.partial(chain, impl=impl, topk=topk, layers=layers)
    jax.block_until_ready(fn(candidates))
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(fn(candidates))
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dcp", type=int, default=8)
    parser.add_argument(
        "--kv-heads",
        type=int,
        default=None,
        help="KV heads per rank (default: 4 at DCP 8, 1 otherwise)",
    )
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument(
        "--num-cols",
        type=int,
        default=256,
        help="sparse blocks per rank to pick candidates from",
    )
    parser.add_argument(
        "--shapes",
        default="decode:512,prefill:2048,prefill:8192",
        help="comma-separated name:tokens (decode c64 x 8 tokens = 512)",
    )
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    kv_heads = args.kv_heads or (4 if args.dcp == 8 else 1)

    print(f"backend={jax.default_backend()} devices={jax.devices()[:1]}")
    print(
        f"dcp={args.dcp} kv_heads={kv_heads} topk={args.topk} "
        f"layers={args.layers} (per-layer slope over 1..{args.layers})"
    )
    header = (
        f"{'shape':>14} {'rows':>7} {'serial us':>10} {'sort us':>10} "
        f"{'pairwise us':>12} {'best speedup':>13}"
    )
    print(header)
    for i, spec in enumerate(args.shapes.split(",")):
        _, tokens = spec.split(":")
        rows = int(tokens) * kv_heads
        cand = make_candidates(
            jax.random.key(i),
            dcp=args.dcp,
            rows=rows,
            topk=args.topk,
            num_cols=args.num_cols,
        )
        want = np.asarray(msa_dcp.merge_candidates_serial(cand, args.topk))
        for impl in ("sort", "pairwise"):
            got = np.asarray(jax.jit(_IMPLS[impl], static_argnums=1)(cand, args.topk))
            if not np.array_equal(got, want):
                bad = int(np.sum(np.any(got != want, axis=1)))
                raise SystemExit(f"{spec}: {impl} differs on {bad}/{rows} rows")
        per_layer = {}
        for impl in _IMPLS:
            t1 = time_chain(cand, impl=impl, topk=args.topk, layers=1, iters=args.iters)
            tn = time_chain(
                cand, impl=impl, topk=args.topk, layers=args.layers, iters=args.iters
            )
            per_layer[impl] = (tn - t1) / (args.layers - 1) * 1e6
        best = min(per_layer["sort"], per_layer["pairwise"])
        print(
            f"{spec:>14} {rows:>7} {per_layer['serial']:>10.1f} "
            f"{per_layer['sort']:>10.1f} {per_layer['pairwise']:>12.1f} "
            f"{per_layer['serial'] / best:>12.2f}x"
        )
    print("parity: bit-exact on every shape")


if __name__ == "__main__":
    main()
