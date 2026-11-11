# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark of the QSA prefill union walk against the block-sparse kernel at
served TP8-rank shapes, including the paired page's second layer
(VLLM_TPU_QSA_PREFILL_UNION_OFFSET, kv_head_offset=1) and main-KV DCP-2
(VLLM_TPU_QSA_PREFILL_UNION_DCP, 6 gathered heads, LSE out).

Both arms go through ``qsa_rpa._prefill_attention`` with the served flags, so
each time includes its metadata (block-sparse bitset / union plan) and the walk
includes its fallback test. Block-sparse arm: VPU=2, PAGE_COPIES=1,
BITSET_RANGED=1, LEAN=1 (the r2 render). Union arm: the same plus UNION_WALK=128,
KV_TILE=1024, CAP=12288, MIN_CONTEXT=32768, QK_AHEAD=1 (+ OFFSET / DCP).
``forced`` arms call ``union_walk_prefill`` directly (no 1024-row minimum).

Selections: ``syn`` = 64 recent groups + 448 from a per-512-row hot set of 25% of
the 1024-token tiles (bench_prefill_vpu's model); ``real`` = a served chunk's
indexer selections (seldump npz: rows at 87.5K-93K context), ``realx`` = the same
rows stretched to a longer context (distances beyond the recent 1K tokens scaled).

Prints one BENCH line per (shape, arm): device us per call (whole op), us of the
attention kernel alone (whole op minus a run with that pallas_call stubbed), the
union size per 128-row group (mean / max; ``fits`` = every group within the cap,
otherwise the walk falls back), and errors vs an f64 reference on sampled rows and
vs the block-sparse arm (LSE too under DCP).

Usage (one chip): python bench_union_ext.py --shapes ... [--arms ...] [--dump path.npz]
"""

import argparse
import contextlib
import functools
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

PAGE, RATIO, TOPK, DIM = 512, 4, 512, 256
SERVED_ENV = {
    "VLLM_TPU_QSA_PREFILL_VPU": "2",
    "VLLM_TPU_QSA_PREFILL_PAGE_COPIES": "1",
    "VLLM_TPU_QSA_PREFILL_BITSET_RANGED": "1",
    "VLLM_TPU_QSA_PREFILL_LEAN": "1",
    "VLLM_TPU_QSA_PREFILL_UNION_WALK": "0",
}
UNION_ENV = {
    "VLLM_TPU_QSA_PREFILL_UNION_WALK": "128",
    "VLLM_TPU_QSA_PREFILL_UNION_KV_TILE": "1024",
    "VLLM_TPU_QSA_PREFILL_UNION_CAP": "12288",
    "VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT": "32768",
    "VLLM_TPU_QSA_PREFILL_UNION_QK_AHEAD": "1",
    "VLLM_TPU_QSA_PREFILL_UNION_OFFSET": "1",
    "VLLM_TPU_QSA_PREFILL_UNION_DCP": "1",
}
# name: (rows, context, selections, dcp world)
SHAPES = {
    "r3072_c108k_syn": (3072, 108_000, "syn", 1),
    "r3072_c233k_syn": (3072, 233_000, "syn", 1),
    "r512_c233k_syn": (512, 233_000, "syn", 1),
    "dcp2_r3072_c233k_syn": (3072, 233_000, "syn", 2),
    "dcp2_r8192_c110k_syn": (8192, 110_000, "syn", 2),
    "r3072_c93k_real": (3072, 0, "real", 1),
    "r5632_c93k_real": (5632, 0, "real", 1),
    "dcp2_r5632_c93k_real": (5632, 0, "real", 2),
    "r3072_c233k_realx": (3072, 233_000, "realx", 1),
    "dcp2_r5632_c130k_realx": (5632, 130_000, "realx", 2),
}
# arm: (union env on, kv_head_offset, forced union_walk_prefill call)
ARMS = {
    "bsp": (False, 0, False),
    "union": (True, 0, False),
    "union_noahead": (True, 0, False),
    "bsp_h1": (False, 1, False),
    "union_h1": (True, 1, False),
    "union_forced": (True, 0, True),
}


@contextlib.contextmanager
def env(values):
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def syn_selections(rows, context, query_block, rng):
    lengths = (context - rows + np.arange(rows) + 1).astype(np.int32)
    tiles_total = context // 1024
    hot = max(8, round(0.25 * tiles_total))
    blocks = np.empty((rows, TOPK), np.int32)
    for start in range(0, rows, query_block):
        first = int(lengths[start])
        avail = (first - 2048) // 1024
        hot_tiles = rng.choice(avail, size=min(hot, avail), replace=False)
        pool = (hot_tiles[:, None] * 256 + np.arange(256)[None, :]).reshape(-1)
        for row in range(start, min(start + query_block, rows)):
            complete = int(lengths[row]) // RATIO
            recent = np.arange(complete - 64, complete)
            far = rng.choice(pool, size=TOPK - 64, replace=False)
            blocks[row] = np.sort(np.concatenate((far, recent)))
    return blocks, lengths


def real_selections(dump, rows, stretch_to=0):
    d = np.load(dump)
    n = int(d["meta"][3])
    if rows > n:
        raise ValueError(f"dump has {n} rows < {rows}")
    sel = d["selected"][:rows]  # token starts of the selected groups (-1 pad)
    lengths = (d["positions"][:rows] + 1).astype(np.int32)
    groups = np.where(sel >= 0, sel // RATIO, -1)
    width = min(TOPK, groups.shape[1])
    if stretch_to:
        shift = stretch_to - int(lengths.max())
        new_lengths = lengths + shift
        complete = lengths[:, None] // RATIO
        new_complete = new_lengths[:, None] // RATIO
        dist = complete - groups
        scale = (new_lengths.max() / lengths.max())
        new_dist = np.where(dist <= 256, dist, np.round(256 + (dist - 256) * scale).astype(np.int64))
        g = np.where(groups >= 0, new_complete - new_dist, -1)
        g = np.where((g >= 0) & (g < new_complete), g, -1)
        out = np.full((rows, groups.shape[1]), -1, np.int32)
        for r in range(rows):
            u = np.unique(g[r][g[r] >= 0])
            out[r, : len(u)] = u
        groups, lengths = out, new_lengths.astype(np.int32)
    blocks = np.full((rows, TOPK), -1, np.int32)
    for r in range(rows):
        v = groups[r][(groups[r] >= 0) & (groups[r] < lengths[r] // RATIO)]
        v = np.unique(v)[-TOPK:]
        blocks[r, : len(v)] = v
    return blocks, lengths


def make_inputs(shape, dump, seed=0):
    rows, context, kind, world = shape
    rng = np.random.default_rng(seed)
    if kind == "syn":
        blocks, lengths = syn_selections(rows, context, 512, rng)
    else:
        blocks, lengths = real_selections(dump, rows, stretch_to=context if kind == "realx" else 0)
    heads = 3 * world
    entry = PAGE * world
    entries = -(-int(lengths.max()) // entry) + 1
    physical = entries + 3
    table_row = rng.permutation(physical)[:entries].astype(np.int32)
    table = np.broadcast_to(table_row, (rows, entries)).copy()
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    query = jax.random.normal(k1, (rows, heads, DIM), jnp.float32).astype(jnp.bfloat16)
    raw = jax.random.normal(k2, (physical, PAGE, 1, 4, DIM), jnp.float32)
    raw = raw * jnp.asarray([1.5, 0.5, 1.2, 0.7], jnp.float32)[None, None, None, :, None]
    cache = raw.astype(jnp.float8_e4m3fn)
    # gate 30: sigmoid rounds to 1.0 in bf16, so the gated output is bf16(out).
    gate = jnp.full((rows, heads, DIM), 30.0, jnp.bfloat16)
    starts = jnp.asarray([0, rows], jnp.int32)
    inputs = (query, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), cache, starts)
    return inputs, (blocks, lengths, table_row)


def reference_rows(inputs, host, sample, *, world, rank, offset):
    query, _, _, _, _, cache, _ = inputs
    blocks, lengths, table_row = host
    q = np.asarray(query[sample].astype(jnp.float32), np.float64)
    heads = q.shape[1]
    cache_np = np.asarray(cache.astype(jnp.float32), np.float64)
    out = np.zeros((len(sample), heads, DIM))
    lse = np.full((len(sample), heads), -np.inf)
    for i, row in enumerate(sample):
        length = int(lengths[row])
        groups = set(int(g) for g in blocks[row] if 0 <= g < length // RATIO)
        tail = length // RATIO
        tokens = [t for g in sorted(groups) for t in range(g * RATIO, g * RATIO + RATIO)]
        tokens += list(range(tail * RATIO, length))
        tokens = np.asarray(tokens)
        if world > 1:
            tokens = tokens[(tokens // PAGE) % world == rank]
        if not len(tokens):
            continue
        phys = table_row[tokens // (PAGE * world)]
        off = tokens % PAGE
        keys = cache_np[phys, off, 0, 2 * offset]
        values = cache_np[phys, off, 0, 2 * offset + 1]
        for h in range(heads):
            s = keys @ q[i, h] / np.sqrt(DIM)
            p = np.exp(s - s.max())
            out[i, h] = (p / p.sum()) @ values
            lse[i, h] = s.max() + np.log(p.sum())
    return out, lse


def union_stats(inputs, world, rank):
    from vllm_torchtpu.kernels.qwen4_exp.qsa_union_dense import dcp_local_selection
    from vllm_torchtpu.kernels.qwen4_exp.qsa_union_plan import union_bits_plan

    _, _, blocks, lengths, table, _, _ = inputs
    if world > 1:
        blocks, lengths = dcp_local_selection(
            blocks, lengths, (jnp.asarray([rank], jnp.int32), world), page_size=PAGE, ratio=RATIO
        )
    _, _, usize, _ = union_bits_plan(
        blocks, lengths, group=128, union_cap=12288, groups_cap=table.shape[1] * PAGE // RATIO, ratio=RATIO
    )
    usize = np.asarray(usize)
    return float(usize.mean()), int(usize.max())


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


STUB_PREFIXES = ("general_qsub256", "qsa_union_walk")


def build(world, offset, forced, stub=False):
    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa
    from vllm_torchtpu.kernels.qwen4_exp import qsa_union_dense as ud

    jax.clear_caches()
    original = pl.pallas_call

    def stubbed(kernel, *a, **kw):
        if not kw.get("name", "").startswith(STUB_PREFIXES):
            return original(kernel, *a, **kw)
        out_shape = kw["out_shape"]

        def run(*operands):
            dep = (jnp.asarray(operands[0]).reshape(-1)[0] * 0).astype(jnp.float32)
            shapes = out_shape if isinstance(out_shape, (list, tuple)) else [out_shape]
            outs = [jnp.zeros(o.shape, o.dtype) + dep.astype(o.dtype) for o in shapes]
            return outs if isinstance(out_shape, (list, tuple)) else outs[0]

        return run

    pl.pallas_call = stubbed if stub else original
    dcp = (jnp.zeros((1,), jnp.int32), world) if world > 1 else None

    def call(query, gate, blocks, lengths, table, cache, starts):
        options = dict(kv_heads=1, ratio=RATIO)
        if offset:
            options["kv_head_offset"] = offset
        if dcp is not None:
            options.update(dcp=dcp, return_lse=True)
        if forced:
            served = functools.partial(
                qsa_rpa._prefill_attention, query, gate, blocks, lengths, table, cache, starts, **options
            )
            result = ud.union_walk_prefill(
                query, gate, blocks, lengths, table, cache, starts, kv_heads=1, ratio=RATIO,
                group=128, kv_tile=1024, union_cap=12288, min_context=0, lean=True,
                kv_head_offset=offset, dcp=dcp, return_lse=dcp is not None, served=served,
            )
        else:
            result = qsa_rpa._prefill_attention(
                query, gate, blocks, lengths, table, cache, starts, **options
            )
        return result[:-1]

    return jax.jit(call), lambda: setattr(pl, "pallas_call", original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="bsp,union")
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--sample", type=int, default=32)
    parser.add_argument("--dump", default="")
    args = parser.parse_args()
    print("devices", jax.devices(), flush=True)
    for shape_name in args.shapes.split(","):
        shape = SHAPES[shape_name]
        rows, _, _, world = shape
        try:
            inputs, host = make_inputs(shape, args.dump)
        except Exception as exc:  # noqa: BLE001
            print(f"BENCH shape={shape_name} SKIP {type(exc).__name__}: {exc}", flush=True)
            continue
        lengths = host[1]
        mean_u, max_u = union_stats(inputs, world, 0)
        print(f"SHAPE {shape_name} rows={rows} ctx={int(lengths.min())}-{int(lengths.max())} world={world} "
              f"union128_mean={mean_u:.0f} union128_max={max_u} fits={max_u <= 12288}", flush=True)
        rng = np.random.default_rng(1)
        sample = np.sort(rng.choice(rows, size=min(args.sample, rows), replace=False))
        refs = {}
        base = {}
        for arm in args.arms.split(","):
            union_on, offset, forced = ARMS[arm]
            if offset not in refs:
                refs[offset] = reference_rows(inputs, host, sample, world=world, rank=0, offset=offset)
            ref_out, ref_lse = refs[offset]
            values = dict(SERVED_ENV)
            if union_on:
                values.update(UNION_ENV)
                if arm.endswith("noahead"):
                    values["VLLM_TPU_QSA_PREFILL_UNION_QK_AHEAD"] = "0"
            restore = lambda: None  # noqa: E731
            try:
                with env(values):
                    fn, restore = build(world, offset, forced, stub=True)
                    meta_us, _ = device_time(fn, inputs, args.iters)
                    restore()
                    fn, restore = build(world, offset, forced)
                    total_us, out = device_time(fn, inputs, args.iters)
                    restore()
            except Exception as exc:  # noqa: BLE001
                restore()
                print(f"BENCH shape={shape_name} arm={arm} FAILED {type(exc).__name__}: {str(exc)[:1500]}",
                      flush=True)
                continue
            lse = None
            if world > 1:
                out, lse = out
                lse = np.asarray(lse, np.float64)
            else:
                (out,) = out
            out = np.asarray(jnp.asarray(out).astype(jnp.float32), np.float64)
            sub = out[sample]
            live = np.isfinite(ref_lse[:, 0])
            rel = np.linalg.norm((sub - ref_out).reshape(len(sample), -1), axis=1) / np.maximum(
                np.linalg.norm(ref_out.reshape(len(sample), -1), axis=1), 1e-30)
            line = (f"BENCH shape={shape_name} arm={arm} total_us={total_us:.1f} meta_us={meta_us:.1f} "
                    f"kernel_us={total_us - meta_us:.1f} ref_max_abs={np.abs(sub - ref_out).max():.3e} "
                    f"ref_row_rel_max={rel.max():.3e} ref_row_rel_mean={rel.mean():.3e} "
                    f"finite={bool(np.isfinite(out).all())}")
            if lse is not None:
                lse_err = np.abs(lse[sample][live] - ref_lse[live]).max() if live.any() else 0.0
                line += f" ref_lse_max_abs={lse_err:.3e}"
            key = offset
            if key in base:
                b_out, b_lse = base[key]
                d = np.abs(out - b_out)
                line += f" bsp_max_abs={d.max():.3e} bsp_bitexact_frac={float(np.mean(d == 0)):.4f}"
                if lse is not None:
                    fin = np.isfinite(b_lse)
                    line += (f" bsp_lse_max_abs={np.abs(np.where(fin, lse - b_lse, 0)).max():.3e}"
                             f" lse_inf_match={bool(np.array_equal(fin, np.isfinite(lse)))}")
            elif not union_on:
                base[key] = (out, lse)
            print(line, flush=True)


if __name__ == "__main__":
    main()
