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
"""Microbenchmark for the MiniMax-M3 MSA DCP candidate merge, as served.

Times the region of `msa_dcp.msa_attend_dcp` in front of the attention
kernel: the DCP all-gather of every rank's packed candidates, the merge
(`msa_dcp.merge_head_candidates`) and the page keys built from it
(`msa_dcp.selected_page_keys`), one per layer, for every merge impl:

  legacy  the original rounds (the served default)
  serial  the same rounds on lane-dense rows
  sort    two row sorts
  pallas  the rounds in one kernel
  none    the gather and the page keys of a fixed selection, no merge: the
          floor the others are measured against

The served program (j-adc17a5e) runs this inside a whole-model graph on
each of the 8 cores, after an all-gather over the DCP group. Here a
`shard_map` over the cores (axis ``dcp``) does the same: each core starts
from its own ``[T, G * 2 * topk]`` f32 candidates and all-gathers them.

Each layer reads its own inputs (``--distinct`` sets, picked by an index
that depends on the previous layer's result, in a `fori_loop`), so XLA can
neither hoist nor CSE a layer. A chain that only perturbs one element of
one input is not enough: XLA then merges the rows once and redoes only the
perturbed row (the first version of this script measured ~12 us per layer
that way, against ~1 ms served).

Per-layer time is the slope ``(t(2L) - t(L)) / L`` of the loop. Before
timing, every impl is checked bit-exact against "legacy" at each shape.
``--dump-dir`` writes each compiled module and prints its fusion count.

Usage (TPU, 8 cores):
  python scripts/benchmark_msa_dcp_merge_candidates.py
  python scripts/benchmark_msa_dcp_merge_candidates.py --shapes=prefill:8192
  python scripts/benchmark_msa_dcp_merge_candidates.py --dump-dir=/tmp/hlo
CPU (correctness only): add --single-device --interpret --layers=2 --iters=1
"""

from __future__ import annotations

import argparse
import functools
import os
import re
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_dcp

try:
    from jax import shard_map
except ImportError:  # older jax
    from jax.experimental.shard_map import shard_map

_IMPLS = ("none",) + msa_dcp.MERGE_IMPLS
_BLOCK_SIZE = 128
_MAX_SEQS = 64
_PAGES_PER_SEQ = 2048


def make_candidates(
    key: jax.Array, *, dcp: int, tokens: int, heads: int, topk: int, num_cols: int
) -> jax.Array:
    """``[dcp, T, G * 2 * topk]``: each rank's `pack_candidates` of its ``G``
    heads, as `msa_write_and_candidates_dcp` emits them: distinct blocks
    ``c * dcp + rank`` by descending score (few distinct values, so ties are
    common), a quarter of the rows short of ``topk`` (``(-inf, -1)``)."""
    k_cols, k_scores, k_len = jax.random.split(key, 3)
    shape = (dcp, tokens, heads)
    cols = jnp.argsort(jax.random.uniform(k_cols, shape + (num_cols,)), axis=-1)
    blocks = (
        cols[..., :topk] * dcp + jnp.arange(dcp, dtype=jnp.int32)[:, None, None, None]
    )
    scores = jax.random.randint(k_scores, shape + (topk,), -64, 64) / 8
    scores = -jnp.sort(-scores.astype(jnp.float32), axis=-1)
    n = jnp.where(
        (jnp.arange(tokens) % 4 == 0)[None, :, None],
        jax.random.randint(k_len, shape, 0, topk + 1),
        topk,
    )
    missing = jnp.arange(topk) >= n[..., None]
    scores = jnp.where(missing, -jnp.inf, scores)
    blocks = jnp.where(missing, -1, blocks).astype(jnp.float32)
    packed = jnp.concatenate([scores, blocks], axis=-1)  # [dcp, T, G, 2k]
    return packed.reshape(dcp, tokens, heads * 2 * topk)


def make_metadata(tokens: int):
    """``tokens`` split over up to `_MAX_SEQS` requests with ~100K context
    each: ``(seq_lens, cu_q_lens)``. Only the page-key arithmetic reads
    them, not the merge."""
    num_seqs = min(_MAX_SEQS, tokens)
    q_lens = np.full(num_seqs, tokens // num_seqs)
    q_lens[: tokens % num_seqs] += 1
    seq_lens = np.zeros(_MAX_SEQS, np.int32)
    seq_lens[:num_seqs] = q_lens + 100_000
    cu = np.zeros(_MAX_SEQS + 1, np.int32)
    cu[1 : num_seqs + 1] = np.cumsum(q_lens)
    cu[num_seqs + 1 :] = cu[num_seqs]
    return jnp.asarray(seq_lens), jnp.asarray(cu)


def layer(candidates, seq_lens, cu_q_lens, rank, *, impl, topk, dcp, interpret):
    """One layer's region: gathered candidates -> page keys, reduced to a
    scalar so the next layer can depend on it."""
    num_tokens = candidates.shape[1]
    seq_id, pos, valid = msa.token_positions(num_tokens, seq_lens, cu_q_lens)
    if impl == "none":
        heads = candidates.shape[2] // (2 * topk)
        fixed = candidates[0, :, : heads * topk].astype(jnp.int32)
        fixed = fixed.reshape(num_tokens, heads, topk)
        own = pos // _BLOCK_SIZE
        local_key = jnp.where(valid & (own % dcp == rank), seq_id + own, -1)
        mine = (fixed >= 0) & (fixed % dcp == rank) & valid[:, None, None]
        keys = jnp.where(mine, seq_id[:, None, None] * _PAGES_PER_SEQ + fixed, -1)
    else:
        local_key, keys = msa_dcp.selected_page_keys(
            candidates,
            seq_id,
            pos,
            valid,
            topk=topk,
            block_size=_BLOCK_SIZE,
            pages_per_seq=_PAGES_PER_SEQ,
            rank=rank,
            dcp=dcp,
            merge_impl=impl,
            interpret=interpret,
        )
    return jnp.sum(keys, dtype=jnp.int32) + jnp.sum(local_key, dtype=jnp.int32)


def build(args, impl, layers, mesh):
    """A jitted loop of ``layers`` dependent layers over the distinct
    per-layer inputs ``[S, dcp, T, G * 2 * topk]``."""
    kw = dict(impl=impl, topk=args.topk, dcp=args.dcp, interpret=args.interpret)

    def loop(inputs, seq_lens, cu_q_lens, rank, gather):
        num_sets = inputs.shape[0]

        def body(i, acc):
            # The set depends on the previous layer: no hoisting, no CSE.
            idx = (i + (acc == 123456789).astype(jnp.int32)) % num_sets
            x = jax.lax.dynamic_index_in_dim(inputs, idx, keepdims=False)
            return acc + layer(gather(x), seq_lens, cu_q_lens, rank, **kw)

        return jax.lax.fori_loop(0, layers, body, jnp.int32(0))

    if mesh is None:

        def single(inputs, seq_lens, cu_q_lens):
            return loop(inputs, seq_lens, cu_q_lens, jnp.int32(0), lambda x: x)

        return jax.jit(single)

    def shard(inputs, seq_lens, cu_q_lens):
        rank = jax.lax.axis_index("dcp").astype(jnp.int32)

        def gather(x):  # this core's [T, W] -> every rank's [dcp, T, W]
            return jax.lax.all_gather(x, "dcp", axis=0)

        acc = loop(inputs[:, 0], seq_lens, cu_q_lens, rank, gather)
        return jax.lax.psum(acc, "dcp")

    return jax.jit(
        shard_map(
            shard,
            mesh=mesh,
            in_specs=(P(None, "dcp"), P(), P()),
            out_specs=P(),
            check_vma=False,
        )
    )


def time_fn(fn, fn_args, iters):
    jax.block_until_ready(fn(*fn_args))
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(fn(*fn_args))
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def check_parity(args, gathered):
    """Every impl's selection bit-exact against "legacy" on one layer."""
    merge = functools.partial(
        msa_dcp.merge_head_candidates, topk=args.topk, interpret=args.interpret
    )
    want = np.asarray(jax.jit(functools.partial(merge, impl="legacy"))(gathered))
    for impl in msa_dcp.MERGE_IMPLS[1:]:
        got = np.asarray(jax.jit(functools.partial(merge, impl=impl))(gathered))
        if not np.array_equal(got, want):
            bad = int(np.sum(np.any(got != want, axis=(1, 2))))
            raise SystemExit(f"{impl} differs from legacy on {bad} tokens")


def dump(fn, fn_args, path):
    text = fn.lower(*fn_args).compile().as_text()
    with open(path, "w") as f:
        f.write(text)
    return len(re.findall(r" fusion\(", text)), len(re.findall(r" copy\(", text))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dcp", type=int, default=8)
    parser.add_argument(
        "--kv-heads",
        type=int,
        default=None,
        help="KV heads per rank, G (default: 4 at DCP 8, 1 otherwise; TP 8)",
    )
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--num-cols", type=int, default=256)
    parser.add_argument(
        "--shapes",
        default="decode:512,prefill:2048,prefill:8192",
        help="comma-separated name:tokens per rank (decode: c64 x 8 tokens)",
    )
    parser.add_argument("--impls", default=",".join(_IMPLS))
    parser.add_argument("--layers", type=int, default=16, help="L of the slope")
    parser.add_argument("--distinct", type=int, default=4, help="input sets")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--single-device",
        action="store_true",
        help="no shard_map: one device holds all ranks' candidates",
    )
    parser.add_argument("--interpret", action="store_true", help="Pallas interpret")
    parser.add_argument("--dump-dir", default=None)
    args = parser.parse_args()
    heads = args.kv_heads or (4 if args.dcp == 8 else 1)
    impls = args.impls.split(",")

    devices = jax.devices()
    mesh = None
    if not args.single_device:
        if len(devices) < args.dcp:
            raise SystemExit(
                f"{len(devices)} devices < dcp={args.dcp}; pass --single-device"
            )
        mesh = Mesh(np.array(devices[: args.dcp]), ("dcp",))
    print(f"backend={jax.default_backend()} devices={len(devices)} mesh={mesh}")
    print(
        f"dcp={args.dcp} kv_heads={heads} topk={args.topk} layers={args.layers} "
        f"distinct={args.distinct}; per-layer us = (t(2L) - t(L)) / L"
    )
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
    print(f"{'shape':>14} {'rows':>7} " + " ".join(f"{i + ' us':>11}" for i in impls))
    for n, spec in enumerate(args.shapes.split(",")):
        name, tokens = spec.split(":")
        tokens = int(tokens)
        make = functools.partial(
            make_candidates,
            dcp=args.dcp,
            tokens=tokens,
            heads=heads,
            topk=args.topk,
            num_cols=args.num_cols,
        )
        keys = jax.random.split(jax.random.key(n), args.distinct)
        inputs = jnp.stack([make(k) for k in keys])  # [S, dcp, T, W]
        check_parity(args, inputs[0])
        seq_lens, cu_q_lens = make_metadata(tokens)
        if mesh is not None:
            inputs = jax.device_put(inputs, NamedSharding(mesh, P(None, "dcp")))
        fn_args = (inputs, seq_lens, cu_q_lens)
        row = []
        for impl in impls:
            short = build(args, impl, args.layers, mesh)
            long = build(args, impl, 2 * args.layers, mesh)
            t_short = time_fn(short, fn_args, args.iters)
            t_long = time_fn(long, fn_args, args.iters)
            row.append((t_long - t_short) / args.layers * 1e6)
            if args.dump_dir:
                path = os.path.join(args.dump_dir, f"{name}{tokens}_{impl}.hlo.txt")
                fusions, copies = dump(short, fn_args, path)
                print(f"  {impl}: {fusions} fusions, {copies} copies -> {path}")
        print(f"{spec:>14} {tokens * heads:>7} " + " ".join(f"{t:>11.1f}" for t in row))
    print("parity: every impl bit-exact against legacy on every shape")


if __name__ == "__main__":
    main()
