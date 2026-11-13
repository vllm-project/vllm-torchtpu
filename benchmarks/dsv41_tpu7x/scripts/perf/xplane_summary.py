#!/usr/bin/env python3
"""Summarize a TPU xplane (one rank) into a per-op / per-module decode-step breakdown.

  python3 xplane_summary.py <file.xplane.pb> [--out summary.json] [--window-ms 400]

Uses xprof's trace_viewer converter (JSON trace events), then:
  * lists processes/threads (planes/lines),
  * per device line: total busy time, top events by self time, counts,
  * a sequential timeline of the XLA Modules line (module name, start, dur, gap before),
  * host python/threads: top events by total time.
"""
import argparse
import collections
import gzip
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("xplane")
ap.add_argument("--out", default=None)
ap.add_argument("--top", type=int, default=60)
ap.add_argument("--tool", default="trace_viewer")
a = ap.parse_args()

from xprof.convert import raw_to_tool_data as rtd  # noqa: E402

data, ctype = rtd.xspace_to_tool_data([a.xplane], a.tool, {})
if isinstance(data, bytes):
    try:
        data = gzip.decompress(data)
    except Exception:
        pass
    data = data.decode()
tr = json.loads(data)
evs = tr["traceEvents"] if isinstance(tr, dict) else tr
pname, tname = {}, {}
for e in evs:
    if e.get("ph") == "M":
        if e.get("name") == "process_name":
            pname[e["pid"]] = e["args"]["name"]
        elif e.get("name") == "thread_name":
            tname[(e["pid"], e["tid"])] = e["args"]["name"]
X = [e for e in evs if e.get("ph") == "X"]
print(f"events={len(X)} processes={len(pname)}")
by_line = collections.defaultdict(list)
for e in X:
    by_line[(e["pid"], e.get("tid"))].append(e)
summary = {"lines": []}
for key, es in sorted(by_line.items(), key=lambda kv: -sum(x.get("dur", 0) for x in kv[1])):
    p, t = key
    es.sort(key=lambda x: x["ts"])
    t0 = es[0]["ts"]
    t1 = max(x["ts"] + x.get("dur", 0) for x in es)
    agg = collections.defaultdict(lambda: [0.0, 0])
    for x in es:
        agg[x["name"]][0] += x.get("dur", 0)
        agg[x["name"]][1] += 1
    top = sorted(agg.items(), key=lambda kv: -kv[1][0])[: a.top]
    line = {"process": pname.get(p, str(p)), "thread": tname.get(key, str(t)), "n": len(es),
            "span_us": t1 - t0, "sum_dur_us": sum(x.get("dur", 0) for x in es),
            "top": [(n, round(d, 1), c) for n, (d, c) in top]}
    if "Module" in line["thread"] or "Step" in line["thread"]:
        seq, prev_end = [], None
        for x in es:
            gap = None if prev_end is None else x["ts"] - prev_end
            seq.append((x["name"][:90], round(x["ts"] - t0, 1), round(x.get("dur", 0), 1), None if gap is None else round(gap, 1)))
            prev_end = x["ts"] + x.get("dur", 0)
        line["sequence"] = seq
    summary["lines"].append(line)
for l in summary["lines"][:40]:
    print(f"\n=== {l['process']} | {l['thread']} | n={l['n']} span={l['span_us']/1e3:.2f}ms busy={l['sum_dur_us']/1e3:.2f}ms")
    for n, d, c in l["top"][:25]:
        print(f"   {d/1e3:9.3f} ms  x{c:<6} {n[:140]}")
if a.out:
    json.dump(summary, open(a.out, "w"))
    print(f"wrote {a.out}")
