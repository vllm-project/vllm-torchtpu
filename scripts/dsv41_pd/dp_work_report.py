#!/usr/bin/env python3
"""汇总 TPU_DP_ATTN_WORK_LOG 写的每步 DP 工作量（rank 0 的 dp_attn_work-*.jsonl）。

每行：t（墙钟）、s（步号）、b（DP 共同 bucket）、tok[r]（rank r 本步 token 数）、
w[r]（rank r 本步 Σ 新 token × 上下文末端）。

输出：
  - 忙碌步（至少一个 rank 有 token）的比例、平均在算的 rank 数；
  - attention 工作量 W 的不均衡：Σ_step max_r W / Σ_step mean_r W（lockstep 下每步等最重的 rank，
    完全均衡时这个比值是 1）；
  - 相邻步时间差 dt 对 (bucket, max W) 做最小二乘，估计每单位 W 的耗时，
    进而估计"如果每步 W 都等于均值"能省下的墙钟比例。
用法：python3 dp_work_report.py <jsonl 或目录> [--t0 秒] [--t1 秒]（相对第一行的时间窗口）
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def load(path):
    files = sorted(glob.glob(os.path.join(path, "*.jsonl"))) if os.path.isdir(path) else [path]
    rows = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # last line may be partially flushed
    rows.sort(key=lambda r: r["t"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=1e18)
    ap.add_argument("--max-dt", type=float, default=2.0, help="丢掉间隔大于此值的步（空闲）")
    a = ap.parse_args()
    rows = load(a.path)
    if not rows:
        sys.exit("no rows")
    T0 = rows[0]["t"]
    rows = [r for r in rows if a.t0 <= r["t"] - T0 <= a.t1]
    t = np.array([r["t"] for r in rows])
    b = np.array([r["b"] for r in rows], dtype=float)
    tok = np.array([r["tok"] for r in rows], dtype=float)
    w = np.array([r["w"] for r in rows], dtype=float)
    n = tok.shape[1]
    busy = tok.sum(1) > 0
    print(f"steps={len(rows)} span={t[-1]-t[0]:.0f}s ranks={n} busy={busy.mean():.1%}")
    tb, wb = tok[busy], w[busy]
    active = (tb > 0).sum(1)
    print(f"busy steps: active ranks mean={active.mean():.2f} hist="
          + " ".join(f"{k}:{(active==k).mean():.0%}" for k in range(1, n + 1)))
    print(f"tokens/step/rank (busy): mean={tb.mean():.0f} max-rank mean={tb.max(1).mean():.0f}  "
          f"Σmax/Σmean tokens={tb.max(1).sum()/tb.mean(1).sum():.2f}")
    wmax, wmean = wb.max(1), wb.mean(1)
    print(f"attn work W (Σ new×ctx): Σmax/Σmean={wmax.sum()/max(wmean.sum(),1):.2f}  "
          f"median per-step max/mean={np.median(wmax/np.maximum(wmean,1)):.2f}")
    # 按"最重 rank 的上下文规模"分档：W_max / tok_of_that_rank ≈ 该 rank 的平均上下文末端
    arg = wb.argmax(1)
    ctx_heavy = wmax / np.maximum(tb[np.arange(len(tb)), arg], 1)
    for lo, hi in [(0, 32e3), (32e3, 64e3), (64e3, 128e3), (128e3, 256e3), (256e3, 1e9)]:
        m = (ctx_heavy >= lo) & (ctx_heavy < hi)
        if m.any():
            print(f"  heaviest-rank ctx {lo/1e3:>4.0f}k-{hi/1e3:<4.0f}k: {m.mean():5.1%} of busy steps, "
                  f"active ranks {active[m].mean():.2f}, W max/mean {wmax[m].sum()/wmean[m].sum():.2f}")
    # dt 回归：第 i 步的耗时 ≈ t[i+1]-t[i]（async 下近似）
    dt = np.diff(t)
    idx = np.where(busy[:-1] & (dt < a.max_dt))[0]
    if len(idx) > 50:
        X = np.column_stack([np.ones(len(idx)), b[idx], w[idx].max(1)])
        coef, *_ = np.linalg.lstsq(X, dt[idx], rcond=None)
        pred = X @ coef
        r2 = 1 - ((dt[idx] - pred) ** 2).sum() / ((dt[idx] - dt[idx].mean()) ** 2).sum()
        print(f"dt fit: dt = {coef[0]*1e3:.1f}ms + {coef[1]*1e3:.4f}ms×bucket + {coef[2]*1e12:.3f}ms per 1e9 W   R²={r2:.2f}")
        Xb = X.copy(); Xb[:, 2] = w[idx].mean(1)
        save = 1 - (Xb @ coef).sum() / pred.sum()
        print(f"  measured busy time {dt[idx].sum():.0f}s; mean step {dt[idx].mean()*1e3:.1f}ms; "
              f"W-term share {(coef[2]*w[idx].max(1)).sum()/pred.sum():.1%}; "
              f"perfect-balance saving (W_max→W_mean) ≈ {save:.1%}")


if __name__ == "__main__":
    main()
