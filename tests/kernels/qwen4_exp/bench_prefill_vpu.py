# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark + accuracy of VLLM_TPU_QSA_PREFILL_VPU at served prefill shapes.

TP8 per-rank QSA prefill (3 query heads, 1 KV head x 256, FP8 paired pages of
512 tokens, top-512 compressed groups + tail, ratio 4) at 256/512/3072-row
steps over 108K and 233K contexts, plus the c48 DCP2 form (6 gathered heads,
one owned page per KV tile, query block 1024). Each row selects 64 recent
groups and 448 groups from a per-query-block hot set of 1024-token tiles
covering ~25% of the context (the ~57 tiles per block seen at 233K).

Prints one BENCH line per (shape, variant): device us per call of the whole
op, of the attention kernel alone (whole op minus a run with the attention
pallas_call stubbed out), us per visited KV tile, and the error against the
default loop and against an f64 reference on sampled rows.

Usage (one TPU chip): python bench_prefill_vpu.py [--variants ...] [--shapes ...]
"""

import argparse
import functools
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

import vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill as bsp

PAGE, RATIO, TOPK, DIM = 512, 4, 512, 256
ORIGINAL_TILES = bsp.PAIRED_KV_TILES
SHAPES = {
    # name: (rows, context, query heads, query block, kv tile, query tile, dcp world)
    "r256_c108k": (256, 108_000, 3, 256, 1024, 256, 1),
    "r512_c108k": (512, 108_000, 3, 512, 1024, 256, 1),
    "r3072_c108k": (3072, 108_000, 3, 512, 1024, 256, 1),
    "r256_c233k": (256, 233_000, 3, 256, 1024, 256, 1),
    "r512_c233k": (512, 233_000, 3, 512, 1024, 256, 1),
    "r3072_c233k": (3072, 233_000, 3, 512, 1024, 256, 1),
    "dcp2_r3072_c233k": (3072, 233_000, 6, 1024, PAGE, 256, 2),
}
VARIANTS = {
    "base": "",
    "mask": "mask",
    "mask_pv16": "mask,pv16",
    "mask_qk16": "mask,qk16",
    "vpu1": "1",
    # Same features, one 512-row query subtile (variant name suffix _qt512).
    "vpu1_qt512": "1",
    # Suffix _kv2048: 2048-key tiles (16 selection words per tile).
    "base_kv2048": "",
    "vpu1_kv2048": "1",
    # Suffix _pc: VLLM_TPU_QSA_PREFILL_PAGE_COPIES=1.
    "base_pc": "",
    "vpu1_pc": "1",
}


def make_inputs(rows, context, query_heads, query_block, dcp_world, seed=0):
    rng = np.random.default_rng(seed)
    lengths = (context - rows + np.arange(rows) + 1).astype(np.int32)
    groups_total = context // RATIO
    tiles_total = context // 1024
    hot = max(8, round(0.25 * tiles_total))
    blocks = np.empty((rows, TOPK), np.int32)
    for start in range(0, rows, query_block):
        first = int(lengths[start])
        # Hot tiles fully inside every row's context, away from the recent window.
        avail = (first - 2048) // 1024
        hot_tiles = rng.choice(avail, size=min(hot, avail), replace=False)
        pool = (hot_tiles[:, None] * 256 + np.arange(256)[None, :]).reshape(-1)
        for row in range(start, min(start + query_block, rows)):
            complete = int(lengths[row]) // RATIO
            recent = np.arange(complete - 64, complete)
            far = rng.choice(pool, size=TOPK - 64, replace=False)
            blocks[row] = np.sort(np.concatenate((far, recent)))
    assert blocks.max() < groups_total
    entry = PAGE * dcp_world
    entries = -(-context // entry) + 1
    physical = entries + 3
    table_row = rng.permutation(physical)[:entries].astype(np.int32)
    table = np.broadcast_to(table_row, (rows, entries)).copy()
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    query = (jax.random.normal(k1, (rows, query_heads, DIM), jnp.float32)).astype(
        jnp.bfloat16
    )
    # Paired rows [K0, V0, K1, V1]; keys std 1.5 give peaked softmax rows.
    raw = jax.random.normal(k2, (physical, PAGE, 1, 4, DIM), jnp.float32)
    raw = raw * jnp.asarray([1.5, 0.5, 1.5, 0.5], jnp.float32)[None, None, None, :, None]
    cache = raw.astype(jnp.float8_e4m3fn)
    return (
        query,
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(table),
        cache,
    ), (blocks, lengths, table_row)


def reference_rows(inputs, host, sample, *, query_heads, dcp_world, rank, k_scale):
    """f64 attention of sampled rows over this rank's selected tokens."""
    query, _, _, _, cache = inputs
    blocks, lengths, table_row = host
    q = np.asarray(query[sample].astype(jnp.float32), np.float64)
    cache_np = np.asarray(cache.astype(jnp.float32), np.float64)
    out = np.zeros((len(sample), query_heads, DIM))
    for i, row in enumerate(sample):
        length = int(lengths[row])
        groups = set(int(g) for g in blocks[row] if 0 <= g < length // RATIO)
        tail = length // RATIO
        tokens = [t for g in sorted(groups) for t in range(g * RATIO, g * RATIO + RATIO)]
        tokens += [t for t in range(tail * RATIO, length)]
        tokens = np.asarray(tokens)
        if dcp_world > 1:
            tokens = tokens[(tokens // PAGE) % dcp_world == rank]
        entry = PAGE * dcp_world
        phys = table_row[tokens // entry]
        off = tokens % PAGE
        keys = cache_np[phys, off, 0, 0] * k_scale
        values = cache_np[phys, off, 0, 1]
        for h in range(query_heads):
            s = keys @ q[i, h] / np.sqrt(DIM)
            p = np.exp(s - s.max())
            out[i, h] = (p / p.sum()) @ values
    return out


def device_time(fn, args, iters):
    out = fn(*args)
    jax.block_until_ready(out)
    out = fn(*args)
    jax.block_until_ready(out)
    best = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(*args)
        jax.block_until_ready(out)
        best.append((time.perf_counter() - t0) / iters * 1e6)
    return min(best), out


def build(shape, k_scale, stub=False, query_tile=None, kv_tile_override=None):
    rows, context, qh, qb, kv_tile, qt, dcp_world = shape
    kv_tile = kv_tile_override or kv_tile
    qt = min(query_tile or qt, qb)
    jax.clear_caches()
    original = pl.pallas_call

    def stubbed(kernel, *a, **kw):
        if not kw.get("name", "").startswith("general_qsub256"):
            return original(kernel, *a, **kw)
        out_shape = kw["out_shape"]

        def run(*operands):
            # Keep the metadata live: depend on the active counts.
            dep = (operands[2][0] * 0).astype(jnp.float32)
            shapes = out_shape if isinstance(out_shape, (list, tuple)) else [out_shape]
            outs = [jnp.zeros(o.shape, o.dtype) + dep.astype(o.dtype) for o in shapes]
            return outs if isinstance(out_shape, (list, tuple)) else outs[0]

        return run

    pl.pallas_call = stubbed if stub else original
    extra = {}
    if dcp_world > 1:
        # Served DCP returns the LSE for the cross-rank combine.
        extra = {
            "dcp_rank": jnp.zeros((1,), jnp.int32),
            "dcp_world": dcp_world,
            "return_lse": True,
        }
    fn = jax.jit(
        functools.partial(
            bsp.block_sparse_gqa_prefill,
            ratio=RATIO,
            kv_heads=1,
            query_block=qb,
            kv_tile=kv_tile,
            k_scale=k_scale,
            v_scale=1.0,
            bf16_io=True,
            query_tile=qt,
            **extra,
        )
    )
    return fn, lambda: setattr(pl, "pallas_call", original)


def tile_visits(inputs, shape):
    rows, context, qh, qb, kv_tile, qt, dcp_world = shape
    blocks, lengths = np.asarray(inputs[1]), np.asarray(inputs[2])
    visits = 0
    for start in range(0, rows, qb):
        tiles = set()
        for row in range(start, start + qb):
            length = int(lengths[row])
            toks = [g * RATIO for g in blocks[row] if 0 <= g < length // RATIO]
            toks.append((length // RATIO) * RATIO)
            for t in toks:
                if dcp_world > 1:
                    if (t // PAGE) % dcp_world:
                        continue
                    tiles.add(t // (PAGE * dcp_world))
                else:
                    tiles.add(t // kv_tile)
        visits += len(tiles)
    return visits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--k-scale", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--sample", type=int, default=48)
    args = parser.parse_args()
    os.environ["VLLM_TPU_QSA_PREFILL_LEAN"] = "1"
    print("devices", jax.devices(), flush=True)
    for shape_name in args.shapes.split(","):
        shape = SHAPES[shape_name]
        rows, context, qh, qb, kv_tile, qt, dcp_world = shape
        inputs, host = make_inputs(rows, context, qh, qb, dcp_world)
        visits = tile_visits(inputs, shape)
        rng = np.random.default_rng(1)
        sample = np.sort(rng.choice(rows, size=min(args.sample, rows), replace=False))
        ref = reference_rows(
            inputs, host, sample, query_heads=qh, dcp_world=dcp_world, rank=0,
            k_scale=args.k_scale,
        )
        os.environ["VLLM_TPU_QSA_PREFILL_VPU"] = ""
        fn, restore = build(shape, args.k_scale, stub=True)
        meta_us, _ = device_time(fn, inputs, args.iters)
        restore()
        print(f"META shape={shape_name} meta_us={meta_us:.1f} tile_visits={visits}", flush=True)
        base_out = None
        base_lse = None
        for variant in args.variants.split(","):
            spec = VARIANTS[variant]
            kv_override = 2048 if variant.endswith("_kv2048") else None
            if kv_override and dcp_world > 1:
                continue
            bsp.PAIRED_KV_TILES = ORIGINAL_TILES + ((2048,) if kv_override else ())
            os.environ["VLLM_TPU_QSA_PREFILL_VPU"] = spec
            os.environ["VLLM_TPU_QSA_PREFILL_PAGE_COPIES"] = (
                "1" if variant.endswith("_pc") else "0"
            )
            try:
                fn, restore = build(
                    shape, args.k_scale,
                    query_tile=512 if variant.endswith("_qt512") else None,
                    kv_tile_override=kv_override,
                )
                total_us, out = device_time(fn, inputs, args.iters)
                restore()
            except Exception as exc:  # noqa: BLE001 - report and continue
                restore()
                print(f"BENCH shape={shape_name} variant={variant} FAILED {type(exc).__name__}: {str(exc)[:600]}", flush=True)
                continue
            lse = None
            if isinstance(out, (tuple, list)):
                out, lse = out
                lse = np.asarray(lse, np.float64)
                if variant == "base":
                    base_lse = lse
                elif base_lse is not None:
                    print(f"LSE shape={shape_name} variant={variant} bitexact="
                          f"{bool(np.array_equal(lse, base_lse))} max_abs="
                          f"{float(np.nanmax(np.abs(np.where(np.isfinite(base_lse), lse - base_lse, 0)))):.3e}",
                          flush=True)
            out = np.asarray(out.astype(jnp.float32), np.float64)
            if variant == "base":
                base_out = out
            attn_us = total_us - meta_us
            sub = out[sample]
            err_ref = np.abs(sub - ref)
            rel_ref = np.linalg.norm((sub - ref).reshape(len(sample), -1), axis=1) / np.maximum(
                np.linalg.norm(ref.reshape(len(sample), -1), axis=1), 1e-30
            )
            line = (
                f"BENCH shape={shape_name} variant={variant} total_us={total_us:.1f} "
                f"attn_us={attn_us:.1f} us_per_tile={attn_us / max(visits, 1):.3f} "
                f"ref_max_abs={err_ref.max():.3e} ref_row_rel_max={rel_ref.max():.3e} "
                f"ref_row_rel_mean={rel_ref.mean():.3e} finite={bool(np.isfinite(out).all())}"
            )
            if base_out is not None:
                diff = np.abs(out - base_out)
                rel = np.linalg.norm((out - base_out).reshape(rows, -1), axis=1) / np.maximum(
                    np.linalg.norm(base_out.reshape(rows, -1), axis=1), 1e-30
                )
                line += (
                    f" base_max_abs={diff.max():.3e} base_row_rel_max={rel.max():.3e} "
                    f"base_bitexact_frac={float(np.mean(diff == 0)):.6f}"
                )
            print(line, flush=True)


if __name__ == "__main__":
    main()
