"""Per-scope device time of the whole-step call from a jax.profiler xplane (VLLM_TPU_MK_SCOPES).

usage: analyze_scopes.py <profile dir> <calls profiled>
Sums event durations by name on /device:TPU:0, per line, divided by the call count.
"""
import collections
import glob
import sys

from jax.profiler import ProfileData

f = sorted(glob.glob(sys.argv[1] + "/**/*.xplane.pb", recursive=True))[-1]
n = int(sys.argv[2])
pd = ProfileData.from_file(f)
for plane in pd.planes:
    if plane.name != "/device:TPU:0":
        continue
    for line in plane.lines:
        tot, cnt = collections.Counter(), collections.Counter()
        for ev in line.events:
            tot[ev.name] += ev.duration_ns
            cnt[ev.name] += 1
        if not tot:
            continue
        print(f"LINE {line.name} events {sum(cnt.values())}")
        for name, d in sorted(tot.items(), key=lambda x: -x[1])[:45]:
            print(f"SCOPE {line.name[:24]:24s} {d / n / 1e3:9.1f} us x{cnt[name] / n:6.1f} {name[:80]}")
