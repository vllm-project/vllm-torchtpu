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
"""Microbenchmark: the MiniMax-M3 indexer call of a c48 DCP4 step
(`msa.partial_block_scores`, two KV heads per rank, folded 4-bit index
pages, 48 request slots) with keys minimax_indexer_hoist_metadata /
minimax_indexer_row_gather off and on.

One jitted program chains `--layers` indexer calls on the same step metadata
(each layer its own queries), as the 60 layers of a forward do, so XLA can
CSE the hoisted per-step metadata across them. Reported per variant and
step size T: the wall time of 1 and of L layers, and the per-layer slope
``(t(L) - t(1)) / (L - 1)``. Each variant's scores are checked bit for bit
against the keys-off run on every request's live pages.

Steps of T tokens hold `--decode-reqs` decode requests of 8 rows (1 + 7 MTP)
and one prefill chunk with the rest. With two heads per rank a step of
T <= 1536 tokens scores its prefill chunk under the indexer's cond (the
path the keys change); at T = 2048 it takes the unconditional prefill path
(only minimax_indexer_row_gather applies), and served c48 steps of >= 1024
tokens take `msa_dcp._tile_candidates` instead (key minimax_msa_tile_select),
which neither key touches.

Usage (on a TPU host):
  python scripts/bench_indexer_hoist.py
  python scripts/bench_indexer_hoist.py --tokens 384,512,768,2048 --layers 8
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_dcp, msa_fp4, msa_indexer

PAGE = 128
HEAD_DIM = 128
VARIANTS = {
    "off": (False, False),
    "hoist": (True, False),
    "row_gather": (False, True),
    "hoist+row_gather": (True, True),
}


def _step(args, num_tokens, rng):
    """Ragged metadata, a folded 4-bit index cache and per-layer queries."""
    rows = 8
    num_decode = min(args.decode_reqs, args.max_seqs - 1)
    chunk = num_tokens - rows * num_decode
    if chunk <= msa_indexer.DEFAULT_BLOCK_Q:
        raise SystemExit(f"T={num_tokens}: prefill chunk of {chunk} rows is not large")
    q_lens = [rows] * num_decode + [chunk]
    ctx = [args.decode_ctx] * num_decode + [args.prefill_ctx]
    num_seqs = len(q_lens)
    seq_lens = np.zeros(args.max_seqs, np.int32)
    seq_lens[:num_seqs] = np.add(ctx, q_lens)
    cu_q = np.full(args.max_seqs + 1, num_tokens, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    local_lens = msa_dcp.local_seq_lens(
        jnp.asarray(seq_lens), block_size=PAGE, rank=args.rank, dcp=args.dcp
    )
    pages_per_seq = -(-args.max_model_len // (PAGE * args.dcp))
    need = -(-np.asarray(local_lens) // PAGE)
    num_pages = int(need.sum()) + 1
    tables = np.zeros((args.max_seqs, pages_per_seq), np.int32)
    perm = rng.permutation(np.arange(1, num_pages))
    at = 0
    for s in range(num_seqs):
        tables[s, : need[s]] = perm[at : at + need[s]]
        at += need[s]
    nib, scale = msa_fp4.quantize(
        msa_fp4.rotate_keys(
            jnp.asarray(rng.standard_normal((num_pages, PAGE, HEAD_DIM)), jnp.float32)
        )
    )
    cache = msa_fp4.pack_rows(nib, scales=scale)
    index_q = msa_fp4.rotate_queries(
        jnp.asarray(
            rng.standard_normal((args.layers * num_tokens, args.heads, HEAD_DIM)),
            jnp.bfloat16,
        )
    ).reshape(args.layers, num_tokens, args.heads, HEAD_DIM)
    meta = (
        local_lens,
        jnp.asarray(tables.reshape(-1)),
        jnp.asarray(cu_q),
        jnp.int32(num_seqs),
    )
    live = (
        np.arange(pages_per_seq)[None, :] < need[:, None],
        np.repeat(np.arange(args.max_seqs), np.diff(cu_q))[:num_tokens],
    )
    return index_q, cache, meta, live


def _program(num_layers, interpret=False):
    def fn(index_q, cache, local_lens, tables, cu_q, num_seqs):
        return [
            msa.partial_block_scores(
                index_q[layer],
                cache,
                local_lens,
                tables,
                cu_q,
                num_seqs,
                shards=1,
                interpret=interpret,
            )
            for layer in range(num_layers)
        ]

    return jax.jit(fn)


def _time(fn, inputs, iters):
    jax.block_until_ready(fn(*inputs))  # compile + warm
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*inputs))
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def _live_bits(scores, live):
    """Each token's scores on its request's own pages, as int32 bits."""
    pages, seq_of_row = live
    s = np.asarray(scores)  # [G, T, P]
    mask = pages[seq_of_row][None, :, : s.shape[2]]
    mask = np.pad(mask, ((0, 0), (0, 0), (0, s.shape[2] - mask.shape[2])))
    return np.where(mask, s.view(np.int32), 0)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tokens", default="512,2048")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--heads", type=int, default=2, help="KV heads per rank")
    p.add_argument("--max-seqs", type=int, default=48)
    p.add_argument("--dcp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--decode-reqs", type=int, default=40)
    p.add_argument("--decode-ctx", type=int, default=60000)
    p.add_argument("--prefill-ctx", type=int, default=30000)
    p.add_argument("--max-model-len", type=int, default=196608)
    p.add_argument(
        "--large-pages-per-step",
        type=int,
        default=64,
        help="minimax_indexer_large_pages_per_step (c48: 64)",
    )
    p.add_argument("--dma-ring3", type=int, default=1, help="c48: on")
    p.add_argument("--interpret", action="store_true", help="CPU smoke test")
    args = p.parse_args()
    msa_indexer.LARGE_PAGES_PER_STEP = args.large_pages_per_step
    msa_indexer.DMA_RING3 = bool(args.dma_ring3)
    print(f"devices: {jax.devices()}")
    print(
        f"heads/rank {args.heads}, max_seqs {args.max_seqs}, DCP {args.dcp} rank "
        f"{args.rank}, {args.decode_reqs} decode reqs x 8 rows @ {args.decode_ctx}, "
        f"prefill ctx {args.prefill_ctx}, layers {args.layers}"
    )
    print(
        f"{'T':>5} {'variant':<17} {'t(1) ms':>9} {'t(L) ms':>9} "
        f"{'us/layer':>9} {'vs off':>8}  bits"
    )
    for num_tokens in (int(t) for t in args.tokens.split(",")):
        rng = np.random.default_rng(num_tokens)
        index_q, cache, meta, live = _step(args, num_tokens, rng)
        cond = not msa_indexer.is_prefill_step(args.heads * num_tokens, args.max_seqs)
        base = None
        for name, (hoist, row_gather) in VARIANTS.items():
            msa_indexer.HOIST_METADATA, msa_indexer.ROW_GATHER = hoist, row_gather
            jax.clear_caches()  # the keys are read at trace time
            t1 = _time(_program(1, args.interpret), (index_q, cache, *meta), args.iters)
            prog = _program(args.layers, args.interpret)
            t_l = _time(prog, (index_q, cache, *meta), args.iters)
            slope = (t_l - t1) / (args.layers - 1) * 1e3
            outs = prog(index_q, cache, *meta)
            bits = [_live_bits(o, live) for o in outs]
            if base is None:
                base, base_slope, same = bits, slope, "ref"
            else:
                same = (
                    "same"
                    if all(np.array_equal(a, b) for a, b in zip(bits, base))
                    else "DIFF"
                )
            print(
                f"{num_tokens:>5} {name:<17} {t1:>9.3f} {t_l:>9.3f} {slope:>9.1f} "
                f"{slope - base_slope:>+8.1f}  {same}"
                + ("" if cond else "  (prefill path: no cond)")
            )
    msa_indexer.HOIST_METADATA = msa_indexer.ROW_GATHER = False


if __name__ == "__main__":
    main()
