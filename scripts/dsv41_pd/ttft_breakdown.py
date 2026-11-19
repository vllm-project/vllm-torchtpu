#!/usr/bin/env python3
"""从 proxy 日志拆 TTFT：P 段（prefill_send→prefill_recv）、D 段（decode_send→first_chunk）。
    python3 ttft_breakdown.py <proxy.log> [跳过前 N 个请求]"""
import re, sys, statistics as st
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 0
pre, dfirst, ttft = [], [], []
for l in open(sys.argv[1], errors="replace"):
    if "PERF PROXY prefill_recv" in l: pre.append(float(re.search(r"dur_ms=([\d.]+)", l).group(1)))
    elif "PERF PROXY first_chunk" in l:
        dfirst.append(float(re.search(r"dur_from_decode_send_ms=([\d.]+)", l).group(1)))
        ttft.append(float(re.search(r"ttft_ms=([\d.]+)", l).group(1)))
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] / 1e3 if v else float("nan")
for name, v in (("P 段", pre[skip:]), ("D 段", dfirst[skip:]), ("proxy TTFT", ttft[skip:])):
    print(f"{name:10s} n={len(v):5d}  P50={q(v,.5):6.2f}s  P90={q(v,.9):6.2f}s  P99={q(v,.99):6.2f}s  mean={st.mean(v)/1e3 if v else 0:6.2f}s")
