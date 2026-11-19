# SPDX-License-Identifier: Apache-2.0
"""On-TPU time of the PLE checkpoint writer alone (q38 L1 PLE-WRDEFER script bench; no engine).

``pool.write_region`` stock (one grid step and one waited DMA per row) against
``pool.write_region_ring`` (one grid step, K DMAs in flight) at the served region: 9 x
10240 BF16 values = 180 pool tokens per row, pool f8e4m3fn[pages, 512, 1, 4, 256], rows
8 / 16 / 32 / 64 / 128 / 256 (all live: unique positive ids; a second set with a quarter
of the rows null). Each variant is chained on device with the pool as the carry; the time
per call is the slope over chain lengths (no dispatch or host time in it). Before timing,
every variant's pool is compared byte for byte with the stock writer's on device.

Card rule (reports/decode_levers_v6_delta.md "#### PLE-WRDEFER"): KEEP if the ring writer
costs at most 0.25 us per token (per written row). The stock marginal is about 0.72 us
per token (M-fit from the E4 optable's pool_in rows).

  python3 tests/kernels/tpu_bench_ple_wrdefer.py            # TPU: RESULT json lines + SUMMARY
  python3 tests/kernels/tpu_bench_ple_wrdefer.py --selftest # CPU: tiny geometry, classic interpret, ring K vs ring K=1
"""

import argparse
import json
import time

import numpy as np

ELEMS = 9 * 10240
BLOCK = 512
ROWS = (8, 16, 32, 64, 128, 256)
DEPTHS = (2, 4, 8)


def case(rng, rows, pages, null_quarter, elements=ELEMS, block=BLOCK):
    import jax.numpy as jnp

    ids = rng.permutation(np.arange(1, pages))[:rows].astype(np.int32)
    if null_quarter:
        ids[rng.choice(rows, size=max(1, rows // 4), replace=False)] = 0
    values = jnp.asarray(rng.normal(size=(rows, elements)).astype(np.float32), jnp.bfloat16)
    raw = rng.integers(0, 256, size=(pages, block, 1, 4, 256), dtype=np.uint8)
    return jnp.asarray(raw.view(jnp.float8_e4m3fn)), values, jnp.asarray(ids)


def writers(depths):
    from vllm_torchtpu.kernels.qwen4_exp import pool as P

    out = {"stock": P._write_region_stock}
    for k in depths:
        out[f"ring{k}"] = lambda p, v, i, k=k: P.write_region_ring(p, v, i, depth=k)
    return out


def slope_us(jax, f, pool, values, ids, chains=(4, 16), reps=5):
    best = {}
    for n in chains:
        g = jax.jit(lambda p, n=n: jax.lax.fori_loop(0, n, lambda _, q: f(q, values, ids), p), donate_argnums=(0,))
        pool = jax.block_until_ready(g(pool))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            pool = jax.block_until_ready(g(pool))
            ts.append(time.perf_counter() - t0)
        best[n] = min(ts)
    k, _ = np.polyfit(np.array(list(best), float), np.array(list(best.values()), float), 1)
    return round(k * 1e6, 3), pool


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pages", type=int, default=1024)
    a = ap.parse_args(argv)
    import jax

    if a.selftest:
        from jax.experimental import pallas as pl

        real = pl.pallas_call
        pl.pallas_call = lambda *x, **k: real(*x, **{**k, "interpret": True})
        rng = np.random.default_rng(0)
        pool, values, ids = case(rng, 13, 20, True, elements=9 * 512, block=18)
        from vllm_torchtpu.kernels.qwen4_exp import pool as P

        ref = np.asarray(P.write_region_ring(pool, values, ids, depth=1)).view(np.uint8)
        ok = all(np.array_equal(np.asarray(P.write_region_ring(pool, values, ids, depth=k)).view(np.uint8), ref)
                 for k in DEPTHS)
        print("SELFTEST", json.dumps({"ring_k_equal_k1": ok, "wrote": bool(np.any(ref != np.asarray(pool).view(np.uint8)))}))
        return 0 if ok else 1
    print("DEVICES", jax.devices(), flush=True)
    rng = np.random.default_rng(0)
    keep = True
    fits = {}
    for null_quarter in (False, True):
        per = {}
        for rows in ROWS:
            pool, values, ids = case(rng, rows, a.pages, null_quarter)
            live = int(np.sum(np.asarray(ids) > 0))
            ws = writers(DEPTHS)
            ref = np.asarray(jax.jit(ws["stock"])(pool, values, ids)).view(np.uint8)
            out = {"rows": rows, "live": live, "null_quarter": null_quarter,
                   "stock_wrote": bool(np.any(ref != np.asarray(pool).view(np.uint8)))}
            for name, f in ws.items():  # bytes first, on the untouched pool (timing donates and rewrites it)
                got = np.asarray(jax.jit(f)(pool, values, ids)).view(np.uint8)
                out[f"{name}_bytes_differ"] = int((got != ref).sum())
            for name, f in ws.items():
                us, pool = slope_us(jax, f, pool, values, ids)
                out[f"{name}_us"] = us
                out[f"{name}_us_per_live_row"] = round(us / max(live, 1), 4)
                per.setdefault(name, []).append((live, us))
            print("RESULT", json.dumps(out), flush=True)
        for name, pts in per.items():
            x, y = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
            k, b = np.polyfit(x, y, 1)
            fits[f"{name}{'_nullq' if null_quarter else ''}"] = {"marginal_us_per_row": round(k, 4), "fixed_us": round(b, 3)}
    for name, fit in fits.items():
        print("FIT", name, json.dumps(fit), flush=True)
    best = min((f["marginal_us_per_row"], n) for n, f in fits.items() if n.startswith("ring") and "_nullq" not in n)
    keep = best[0] <= 0.25
    print("SUMMARY", json.dumps({"best_ring": best[1], "marginal_us_per_row": best[0],
                                 "stock_marginal_us_per_row": fits["stock"]["marginal_us_per_row"],
                                 "rule": "KEEP if <= 0.25 us per token", "verdict": "KEEP" if keep else "NO-KEEP"}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
