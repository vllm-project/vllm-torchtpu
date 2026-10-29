#!/usr/bin/env python3
"""单个 PD 请求的毫秒级时间线（需要 P、D 用 VLLM_LOGGING_PREFIX='%(created).3f ' 启动）。
    python3 pd_timeline.py <proxy.log> <prefill.log> <decode.log> [req_id 前缀]
默认取 proxy 日志里最后一个完成的请求。"""
import re, sys
proxy, plog, dlog = sys.argv[1:4]
want = sys.argv[4] if len(sys.argv) > 4 else None
ev = {}
for line in open(proxy, errors="replace"):
    m = re.search(r"PERF PROXY (\w+) req_id=(\S+)(.*)", line)
    if m:
        ev.setdefault(m.group(2), []).append((m.group(1), m.group(3)))
rid = want or [r for r, e in ev.items() if any(k == "first_chunk" for k, _ in e)][-1]
rid = next(r for r in ev if r.startswith(rid))
rows = []
for name, rest in ev[rid]:
    ts = re.search(r"ts=([\d.]+)", rest)
    dur = re.search(r"dur_ms=([\d.]+)", rest)
    if ts:
        rows.append((float(ts.group(1)), "proxy", name + (f" dur={dur.group(1)}" if dur else "")))
for role, path in (("P", plog), ("D", dlog)):
    for line in open(path, errors="replace"):
        if rid not in line:
            continue
        m = re.match(r"([\d]{10}\.\d{3}) \((\S+) pid=\d+\) \w+ \S+ \S+ \[(\S+)\] (.*)", line)
        if not m:
            continue
        msg = m.group(4)
        e = re.search(r'"event": "(\w+)"', msg)
        msg = e.group(1) if e else msg[:110]
        rows.append((float(m.group(1)), f"{role}:{m.group(2)[:14]}", f"[{m.group(3)}] {msg}"))
rows.sort()
t0 = rows[0][0]
for t, who, what in rows:
    print(f"{(t - t0) * 1000:8.1f}  {who:16s} {what}")
