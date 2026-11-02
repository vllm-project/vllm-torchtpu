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
"""Decode Indexer query-split A/B across TP ranks.

`_select_paged_indices` splits the batch's queries over TP ranks in
contiguous parts of ``part`` rows. With the 16-row alignment
(``part = (count // (16 * tp) + 1) * 16``) a decode batch of ``C <= 16``
lands entirely on rank 0 (``part = 16``) while ranks 1..7 score only
zero-length padding rows, then everyone waits in the all-gather.

``--split pad16`` replays that policy; ``--split balanced`` uses
``part = count // tp + 1`` (every rank gets at most ``ceil(C / tp) + 1``
rows). Each rank runs ``paged_indexer_jax`` on its part for
``--subop candidate_write`` (L20 full_context source), ``candidate_filter``
(L24/28/32/36) or ``full_context`` (L2/8/14; ``compress_ratio = 2``, so pass
``--kv-len`` = context // 2), ``--reps`` times inside one compiled loop; the
step is the slowest rank. ``--verify`` checks that every real token's
selection is identical to the ``--legacy`` baseline configuration.
"""

import argparse
import pathlib
import statistics
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import common  # noqa: E402

SUBOPS = ("candidate_write", "candidate_filter", "full_context")
SPLITS = ("pad16", "balanced", "ceil")
LEGACY_TILE = (16384, 2048)  # baseline (_INDEX_KEYS_PER_ITER, _INDEX_MAX_TILE)


def split_part(count: int, tp: int, policy: str) -> int:
    """Rows per rank for ``count`` queries (mirrors `_select_paged_indices`)."""
    if policy == "pad16":
        rows = 16 * tp
        return ((count // rows + 1) * rows) // tp
    if policy == "ceil":
        return -(-count // tp)
    return count // tp + 1


def build_inputs(
    *,
    seed: int,
    count: int,
    kv_len: int,
    pool_rows: int,
    heads: int,
    head_dim: int,
    candidate_blocks: int,
    block_size: int,
    page_records: int,
) -> dict[str, np.ndarray]:
    """Full-batch decode inputs: request ``i`` owns token ``i``."""
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout

    rng = np.random.default_rng(seed + 4049)
    keys = jnp.asarray(rng.standard_normal((pool_rows, 128)), jnp.bfloat16)
    encoded = index_key_layout.encode_records(keys)
    pool = np.asarray((encoded[::2] | (encoded[1::2] << 16)).astype(jnp.int32)).copy()
    if page_records:
        # As `_request_row_slots`: each request owns random distinct pages and
        # logical row r is record ``page[r // page_records] * page + r % page``.
        pages = -(-kv_len // page_records)
        table = np.stack(
            [rng.permutation(pool_rows // page_records)[:pages] for _ in range(count)]
        )
        offsets = np.arange(kv_len) % page_records
        records = (
            table[:, np.arange(kv_len) // page_records] * page_records + offsets
        ).astype(np.int32)
    else:
        records = rng.integers(0, pool_rows, size=(count, kv_len), dtype=np.int32)
    q = rng.standard_normal((count, heads, head_dim)).astype(np.float32) * 0.25
    weights = rng.standard_normal((count, heads)).astype(np.float32) * (
        head_dim**-0.5 * heads**-0.5
    )
    lengths = (kv_len - rng.integers(0, 64, size=(count,))).astype(np.int32)
    blocks = -(-kv_len // block_size)
    candidates = rng.integers(0, blocks, size=(count, candidate_blocks), dtype=np.int32)
    return {
        "pool": pool,
        "records": records,
        "q": q,
        "weights": weights,
        "lengths": lengths,
        "candidates": candidates,
        "requests": np.arange(count, dtype=np.int32),
    }


def expand_tokens(full: dict[str, np.ndarray], tokens: int) -> dict:
    """Prefill shape: ``tokens`` causal query tokens per request row."""
    seqs = full["q"].shape[0]
    rep = np.repeat(np.arange(seqs), tokens)
    back = np.tile(np.arange(tokens - 1, -1, -1), seqs)
    out = dict(full)
    for key in ("q", "weights", "candidates"):
        out[key] = np.ascontiguousarray(full[key][rep])
    out["lengths"] = np.maximum(full["lengths"][rep] - back, 1).astype(np.int32)
    out["requests"] = rep.astype(np.int32)
    return out


def rank_slice(full: dict[str, np.ndarray], part: int, rank: int, tp: int) -> dict:
    """Pad the per-token arrays to ``part * tp`` rows and take ``rank``'s part."""
    count = full["q"].shape[0]
    padding = part * tp - count
    fills = {"q": 0.0, "weights": 0.0, "lengths": 0, "candidates": -1, "requests": 0}
    out = {}
    for key, fill in fills.items():
        arr = full[key]
        padded = np.concatenate(
            [arr, np.full((padding, *arr.shape[1:]), fill, dtype=arr.dtype)]
        )
        out[key] = np.ascontiguousarray(padded[rank * part : (rank + 1) * part])
    return out


def worker(ctx: common.WorkerContext, args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import torch
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas
    from vllm.distributed import get_tp_group

    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
        indexer_impl as indexer_ops,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_impl import paged_indexer_jax
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    if args.keys_per_iter is not None:
        indexer_ops._INDEX_KEYS_PER_ITER = args.keys_per_iter
    if args.max_tile is not None:
        indexer_ops._INDEX_MAX_TILE = args.max_tile
    current = (
        indexer_ops._INDEX_KEYS_PER_ITER,
        indexer_ops._INDEX_MAX_TILE,
        indexer_ops._visible_top_k,
    )
    src_visible = indexer_ops._visible_top_k

    def legacy_visible(scores, k, bound, lengths=None, unordered=False):
        # Baseline: fewer than 16 rows skipped SparseCore (bucketed XLA top_k).
        if scores.shape[0] < 16:
            lengths = None
        return src_visible(scores, k, bound, lengths, unordered)

    def use(config: str) -> None:
        """Set module state read at trace time (ops trace on first call)."""
        state = current if config == "src" else (*LEGACY_TILE, legacy_visible)
        (
            indexer_ops._INDEX_KEYS_PER_ITER,
            indexer_ops._INDEX_MAX_TILE,
            indexer_ops._visible_top_k,
        ) = state

    use("legacy" if args.legacy else "src")
    geo = common.Dsv41Geometry()
    tp = ctx.tp_size
    rank = get_tp_group().rank_in_group
    full = build_inputs(
        seed=args.seed,
        count=args.num_seqs,
        kv_len=args.kv_len,
        pool_rows=args.pool_rows,
        heads=geo.index_n_heads,
        head_dim=geo.index_head_dim,
        candidate_blocks=geo.candidate_topk_blocks,
        block_size=geo.candidate_block_size,
        page_records=args.layout_page,
    )
    if args.tokens_per_seq > 1:
        full = expand_tokens(full, args.tokens_per_seq)
    count = full["q"].shape[0]
    cand_write = args.subop == "candidate_write"
    cand_filter = args.subop == "candidate_filter"
    batch_max = torch.tensor([int(full["lengths"].max())], dtype=torch.int32)
    pool = torch.from_numpy(full["pool"]).to("tpu")
    records = torch.from_numpy(full["records"]).to("tpu")
    batch_max = batch_max.to("tpu")

    def device_part(policy: str) -> tuple[int, list[Any]]:
        part = split_part(count, tp, policy)
        loc = rank_slice(full, part, rank, tp)
        q = torch.from_numpy(loc["q"]).to(torch.bfloat16).to("tpu")
        rest = [
            torch.from_numpy(loc[k]).to("tpu")
            for k in ("weights", "lengths", "candidates", "requests")
        ]
        return part, [q, *rest]

    with common.model_wrapper_context():
        mesh = get_vllm_model_wrapper_context().mesh

        def make_op(name: str, reps: int, page_records: int | None = None):
            def fn(
                q: jax.Array,
                p: jax.Array,
                r: jax.Array,
                w: jax.Array,
                lengths: jax.Array,
                cand: jax.Array,
                req: jax.Array,
                bmax: jax.Array,
            ) -> jax.Array:
                def body(_, carry):
                    q_i, acc = jax.lax.optimization_barrier((q, carry))
                    if args.stage == "scores":
                        scores = indexer_ops.paged_index_scores_jax(
                            q_i, p, r, w, lengths, req, page_records=args.page_records
                        )
                        hit = (scores.max(axis=1, keepdims=True) > 0).astype(acc.dtype)
                        return acc + hit
                    if args.stage == "select":
                        # Scores are a loop input: time only the selection.
                        s_i = jax.lax.optimization_barrier(
                            q_i[:, :1, 0].astype(jax.numpy.float32) * 0 + base
                        )
                        sel, pub = indexer_ops.select_indices_jax(
                            s_i,
                            lengths,
                            cand,
                            geo.index_topk,
                            geo.candidate_block_size,
                            cand_write,
                            cand_filter,
                            True,
                            bmax,
                        )
                        return acc + sel + pub[:, :1] * 0
                    sel, pub = paged_indexer_jax(
                        q_i,
                        p,
                        r,
                        w,
                        lengths,
                        cand,
                        req,
                        topk=geo.index_topk,
                        block_size=geo.candidate_block_size,
                        candidate_write=cand_write,
                        candidate_filter=cand_filter,
                        unordered=True,
                        batch_max=bmax,
                        page_records=page_records,
                    )
                    return acc + sel + pub[:, :1] * 0

                base = None
                if args.stage == "select":
                    base = indexer_ops.paged_index_scores_jax(q, p, r, w, lengths, req)
                    base = jax.numpy.where(
                        jax.numpy.take(r >= 0, req, axis=0), base, -jax.numpy.inf
                    )

                sel0, _ = jax.eval_shape(
                    lambda: paged_indexer_jax(
                        q,
                        p,
                        r,
                        w,
                        lengths,
                        cand,
                        req,
                        topk=geo.index_topk,
                        block_size=geo.candidate_block_size,
                        candidate_write=cand_write,
                        candidate_filter=cand_filter,
                        unordered=True,
                        batch_max=bmax,
                    )
                )
                init = jax.numpy.zeros(sel0.shape, sel0.dtype)
                return jax.lax.fori_loop(0, reps, body, init)

            # No outer shard_map: `_sparsecore_top_k` shard_maps itself over
            # the model mesh (as bench_indexer.py passes paged_indexer_jax).
            return pallas.jax_op(
                f"pallas::dsv41_bench_dec_idx_{name}",
                fn,
                mesh=mesh,
                input_partition_specs=(P(),) * 8,
            )

        split = "pad16" if args.legacy else args.split
        config = "legacy" if args.legacy else "src"
        part, (q, w, lengths, cand, req) = device_part(split)
        pages = (args.page_records or None) if config == "src" else None
        op = make_op(f"{args.subop}_{split}_{config}_p{pages}", args.reps, pages)
        common.synchronize((pool, records, q, w, lengths, cand, req))

        def step():
            return {"sel": op(q, pool, records, w, lengths, cand, req, batch_max)}

        loop_samples, _ = common.time_steps(
            ctx, step, warmup=args.warmup, iters=args.iters
        )
        verify: dict[str, Any] = {}
        if args.verify and args.stage == "scores":

            def both(
                q: jax.Array,
                p: jax.Array,
                r: jax.Array,
                w: jax.Array,
                lengths: jax.Array,
                req: jax.Array,
            ) -> jax.Array:
                return jax.numpy.stack(
                    [
                        indexer_ops.paged_index_scores_jax(q, p, r, w, lengths, req),
                        indexer_ops.paged_index_scores_jax(
                            q, p, r, w, lengths, req, page_records=args.page_records
                        ),
                    ]
                )

            pair = (
                pallas.jax_op(
                    "pallas::dsv41_bench_dec_idx_scores_pair",
                    both,
                    mesh=mesh,
                    input_partition_specs=(P(),) * 6,
                )(q, pool, records, w, lengths, req)
                .cpu()
                .numpy()
            )
            gathered = common.gather_object(ctx, bool(np.array_equal(pair[0], pair[1])))
            if ctx.rank == 0:
                verify = {"page_vs_row_scores_bitwise": all(gathered[:tp])}
                if not verify["page_vs_row_scores_bitwise"]:
                    raise common.VerificationError("page gather scores differ")
        elif args.verify:
            # Reference: baseline (pad16 split, legacy tile and top-k path).
            cases = {"legacy": ("pad16", "legacy"), "test": (args.split, "src")}
            one = {}
            for name, (policy, cfg) in cases.items():
                use(cfg)
                p_rows, tensors = device_part(policy)
                pages = (args.page_records or None) if name == "test" else None
                sel = make_op(f"{args.subop}_{policy}_{cfg}_p{pages}_v", 1, pages)(
                    tensors[0], pool, records, *tensors[1:], batch_max
                )
                one[name] = (p_rows, sel.cpu().numpy())
            gathered = common.gather_object(ctx, one)
            if ctx.rank == 0:
                rows = {}
                for name in cases:
                    stacked = np.concatenate([g[name][1] for g in gathered[:tp]])
                    rows[name] = stacked[:count]
                equal = bool(np.array_equal(rows["legacy"], rows["test"]))
                verify = {
                    f"legacy_vs_{args.split}_src_exact": equal,
                    "valid_per_row_min": int((rows["test"] >= 0).sum(1).min()),
                }
                if not equal:
                    raise common.VerificationError("selection differs from legacy")

    per_rep = [s / args.reps for s in loop_samples]
    return {
        "bench": "decode_indexer_split",
        "subop": args.subop,
        "split": split,
        "config": config,
        "num_seqs": count,
        "part": part,
        "real_rows_this_rank": int(max(0, min(count - rank * part, part))),
        "kv_len": args.kv_len,
        "keys_per_iter": current[0] if config == "src" else LEGACY_TILE[0],
        "max_tile": current[1] if config == "src" else LEGACY_TILE[1],
        "reps": args.reps,
        "stage": args.stage,
        "wall_samples_ms": per_rep,
        "per_rep_median_ms": statistics.median(per_rep),
        "verify": verify,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser, SUBOPS)
    parser.add_argument("--split", choices=list(SPLITS), default="pad16")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--pool-rows", type=int, default=1 << 20)
    parser.add_argument(
        "--stage",
        choices=["all", "scores", "select"],
        default="all",
        help="full_context path only: time the score loop or the selection alone",
    )
    parser.add_argument(
        "--layout-page",
        type=int,
        default=16,
        help="records per cache page in the synthetic records (0: random rows)",
    )
    parser.add_argument(
        "--tokens-per-seq",
        type=int,
        default=1,
        help="query tokens per request row (>1: prefill-shaped, shared rows)",
    )
    parser.add_argument(
        "--page-records",
        type=int,
        default=None,
        help="gather whole pages of this many records (src config only)",
    )
    parser.add_argument("--keys-per-iter", type=int, default=None)
    parser.add_argument("--max-tile", type=int, default=None)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="baseline reference: pad16 split, 16384/2048 tile, XLA top-k < 16 rows",
    )
    parser.set_defaults(num_seqs=16, kv_len=100000)
    common.run_main("decode_indexer_split", parser, worker)


if __name__ == "__main__":
    main()
