#!/usr/bin/env python3
"""dp_attn_work JSONL 按墙钟窗口汇总：python3 dp_work_window.py <jsonl/目录> <HH:MM> <HH:MM>（UTC，当天）。"""
import datetime, glob, json, os, sys
import numpy as np
p, a, b = sys.argv[1:4]
files = sorted(glob.glob(os.path.join(p, "*.jsonl"))) if os.path.isdir(p) else [p]
R = [json.loads(l) for f in files for l in open(f) if l.strip().endswith("}")]
day = datetime.datetime.fromtimestamp(R[0]["t"], datetime.timezone.utc).date()
ts = lambda s: datetime.datetime.combine(day, datetime.time(*map(int, s.split(":"))), datetime.timezone.utc).timestamp()
R = [r for r in R if ts(a) <= r["t"] < ts(b)]
t = np.array([r["t"] for r in R]); w = np.array([r["w"] for r in R], float); tok = np.array([r["tok"] for r in R], float)
busy = tok.sum(1) > 0
print(f"steps={len(R)} span={t[-1]-t[0]:.0f}s steps/s={len(R)/(t[-1]-t[0]):.2f} tokens/s={tok.sum()/(t[-1]-t[0]):.0f}")
print(f"busy={busy.mean():.1%} tokens/rank-step={tok[busy].mean():.0f}  W Σmax/Σmean={w[busy].max(1).sum()/w[busy].mean(1).sum():.2f}  "
      f"mean maxW={w[busy].max(1).mean()/1e9:.3f}e9 mean meanW={w[busy].mean(1).mean()/1e9:.3f}e9  ΣW/s={w.sum()/(t[-1]-t[0])/1e9:.1f}e9")
print("maxW pct (1e9):", (np.percentile(w[busy].max(1), [50, 90, 99]) / 1e9).round(3))
