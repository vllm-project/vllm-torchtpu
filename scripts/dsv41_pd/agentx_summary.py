#!/usr/bin/env python3
"""一行汇总 AgentX 结果目录：aiperf 指标 + D 各 rank 负载 + DSpark 接受长度。
    python3 agentx_summary.py <results/agentx-...> [<dir> ...]   （跳过前 SKIP 个 metrics 抓取，默认 6 = 3 分钟）
"""
import csv, glob, os, re, sys
SKIP = int(os.environ.get("SKIP", "6"))
def aiperf(d):
    m = {}
    for row in csv.reader(open(f"{d}/artifacts/profile_export_aiperf.csv")):
        if len(row) > 1: m[row[0]] = row[1:]
    g = lambda k, i: float(m[k][i]) if k in m and m[k][i] else float("nan")
    # columns: avg,min,max,sum,p1,p5,p10,p25,p50,p75,p90,p95,p99,std
    return dict(rps=g("Request Throughput (requests/sec)", 0), otps=g("Output Token Throughput (tokens/sec)", 0),
                ttft50=g("Time to First Token (ms)", 8) / 1e3, ttft90=g("Time to First Token (ms)", 10) / 1e3,
                itl50=g("Inter Token Latency (ms)", 8), itl90=g("Inter Token Latency (ms)", 10), osl=g("Output Sequence Length (tokens)", 0))
def prom(f, name):
    out = {}
    for l in open(f):
        if l.startswith(name):
            e = re.search(r'engine="(\d+)"', l); out[e.group(1) if e else "0"] = float(l.split()[-1])
    return out
def dstats(d):
    fs = sorted(glob.glob(f"{d}/metrics/d-*.prom"))[SKIP:]
    fs = [f for f in fs if os.path.getsize(f) > 0]
    if len(fs) < 2: return {}
    mx = mn = mean = 0
    for f in fs:
        r = list(prom(f, "vllm:num_requests_running").values()) or [0]
        mx += max(r); mn += min(r); mean += sum(r) / len(r)
    n = len(fs)
    acc = lambda f: (sum(prom(f, "vllm:spec_decode_num_accepted_tokens_total").values()), sum(prom(f, "vllm:spec_decode_num_drafts_total").values()))
    (a0, d0), (a1, d1) = acc(fs[0]), acc(fs[-1])
    return dict(dmax=mx / n, dmin=mn / n, dmean=mean / n, tok_per_step=1 + (a1 - a0) / max(d1 - d0, 1))
print(f"{'run':44s} {'req/s':>6s} {'out/s':>6s} {'TTFT50':>6s} {'TTFT90':>6s} {'ITL50':>6s} {'ITL90':>6s} {'OSL':>5s} | {'Dmax':>5s} {'Dmin':>5s} {'Dmean':>5s} {'tok/st':>6s}")
for d in sys.argv[1:]:
    d = d.rstrip("/"); a = aiperf(d); s = dstats(d)
    print(f"{os.path.basename(d)[:44]:44s} {a['rps']:6.2f} {a['otps']:6.0f} {a['ttft50']:6.2f} {a['ttft90']:6.2f} {a['itl50']:6.1f} {a['itl90']:6.1f} {a['osl']:5.0f} | "
          f"{s.get('dmax', float('nan')):5.1f} {s.get('dmin', float('nan')):5.1f} {s.get('dmean', float('nan')):5.1f} {s.get('tok_per_step', float('nan')):6.2f}")
