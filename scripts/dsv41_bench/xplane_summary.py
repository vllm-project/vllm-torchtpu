"""Summarize a vLLM TPU .xplane.pb (CPU-only): module/step timing and top ops.

Usage: python xplane_summary.py <rank0.xplane.pb> [plane=/device:TPU:0]
"""

import collections
import sys

from jax._src.profiler import ProfileData

path = sys.argv[1]
plane_name = sys.argv[2] if len(sys.argv) > 2 else "/device:TPU:0"
pd = ProfileData.from_file(path)
print("planes:", [p.name for p in pd.planes][:20])
plane = next(p for p in pd.planes if p.name == plane_name)
lines = {ln.name: ln for ln in plane.lines}
print("lines:", list(lines))

mods = lines.get("XLA Modules")
if mods is not None:
    evs = sorted(mods.events, key=lambda e: e.start_ns)
    span = (evs[-1].start_ns + evs[-1].duration_ns - evs[0].start_ns) / 1e6
    busy = sum(e.duration_ns for e in evs) / 1e6
    by = collections.defaultdict(list)
    for e in evs:
        by[e.name].append(e.duration_ns / 1e6)
    print(f"modules: n={len(evs)} span={span:.1f}ms busy={busy:.1f}ms")
    for n, d in sorted(by.items(), key=lambda kv: -sum(kv[1]))[:12]:
        d.sort()
        print(
            f"  {n[:70]:70s} n={len(d):4d} tot={sum(d):9.1f}ms "
            f"med={d[len(d) // 2]:8.3f}ms max={d[-1]:8.3f}ms"
        )
    big = [e for e in evs if e.duration_ns > 5e6]
    gaps = [
        (b.start_ns - (a.start_ns + a.duration_ns)) / 1e6
        for a, b in zip(big, big[1:], strict=False)
    ]
    if gaps:
        gs = sorted(gaps)
        print(
            f"big-module (>5ms) n={len(big)} gap med={gs[len(gs) // 2]:.2f}ms "
            f"mean={sum(gs) / len(gs):.2f}ms max={gs[-1]:.2f}ms"
        )

ops = lines.get("XLA Ops")
if ops is not None:
    by = collections.defaultdict(float)
    cnt = collections.Counter()
    for e in ops.events:
        key = e.name.split(".")[0].split("=")[0][:60]
        by[key] += e.duration_ns / 1e6
        cnt[key] += 1
    tot = sum(by.values())
    print(f"ops total={tot:.1f}ms")
    for n, t in sorted(by.items(), key=lambda kv: -kv[1])[:30]:
        print(f"  {n:60s} n={cnt[n]:6d} tot={t:9.1f}ms ({100 * t / tot:5.1f}%)")
