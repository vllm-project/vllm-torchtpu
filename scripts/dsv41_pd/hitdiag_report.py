#!/usr/bin/env python3
"""汇总 proxy 的 PERF PROXY hit 行：按时间段、按缺口类型统计。用法: hitdiag_report.py <proxy.log> [skip_first_n]"""
import re, sys, collections
rows = []
for l in open(sys.argv[1], errors="replace"):
    m = re.search(r"PERF PROXY hit .* rank=(-?\d+) isl=(\d+) p_cached=(\w+) sess_lcp=(\d+) rank_sim=(\d+) global_sim=(\d+)", l)
    if m and m.group(3) != "None":
        rows.append(tuple(int(x) for x in (m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6))))
rows = rows[1:]  # drop smoke
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 0
print(f"{len(rows)} requests")
def summ(rs, tag):
    I = sum(r[1] for r in rs) or 1
    P, S, R, G = (sum(r[i] for r in rs) for i in (2, 3, 4, 5))
    print(f"{tag:14s} n={len(rs):5d} isl_avg={I/len(rs):8.0f}  P实际 {P/I*100:5.1f}%  同会话 {S/I*100:5.1f}%  本rank模拟 {R/I*100:5.1f}%  全局模拟 {G/I*100:5.1f}%")
n = len(rows); k = 6
for i in range(k):
    summ(rows[i*n//k:(i+1)*n//k], f"段{i+1}/{k}")
st = rows[skip:]
summ(st, "稳态(跳过前%d)" % skip)
# gap classification: tokens P missed relative to global ideal
c = collections.Counter(); ct = collections.Counter()
for rank, isl, p, s, r, g in st:
    gap = g - p
    if gap <= 256: cls = "ok(<=256)"
    elif r >= g - 256 and p < r - 256: cls = "本rank该有却没命中(淘汰/对齐)"
    elif r < g - 256: cls = "前缀在别的rank(路由)"
    else: cls = "其他"
    c[cls] += 1; ct[cls] += max(gap, 0)
I = sum(r[1] for r in st)
for cls, v in ct.most_common():
    print(f"  {cls:28s} 请求 {c[cls]:5d}  缺口 token {v/1e6:8.2f}M = {v/I*100:5.2f}% of ISL")
