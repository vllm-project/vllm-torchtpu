"""Summarize a jax.profiler perfetto trace: per-op device time on the first TPU, last call."""
import collections, glob, gzip, json, sys

f = sorted(glob.glob(sys.argv[1] + "/**/perfetto_trace.json.gz", recursive=True))[-1]
ev = json.load(gzip.open(f))
ev = ev["traceEvents"] if isinstance(ev, dict) else ev
pids = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
tnames = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
tpu0 = [p for p, n in pids.items() if "TPU:0" in n]
print("tpu0", [(p, pids[p]) for p in tpu0])
xs = [e for e in ev if e.get("ph") == "X" and e.get("pid") in tpu0 and "dur" in e]
by = collections.defaultdict(list)
for e in xs:
    by[(e["pid"], e["tid"])].append(e)
for k, v in by.items():
    print("thread", k, tnames.get(k), len(v), [x["name"][:40] for x in v[:3]])
mods = [e for e in xs if e["name"].startswith("jit_call")]
m = max(mods[-4:], key=lambda e: e["dur"])
print("module", m["name"][:60], "dur %.1f" % m["dur"])
w0, w1 = m["ts"], m["ts"] + m["dur"]
cand = {k: [e for e in v if w0 <= e["ts"] <= w1 and not e["name"].startswith("jit_")] for k, v in by.items()}
k = max(cand, key=lambda k: len(cand[k]))
ops = sorted(cand[k], key=lambda e: e["ts"])
print("ops thread", k, tnames.get(k), len(ops))
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 3
for e in ops:
    if e["dur"] >= thr:
        print("%8.1f %7.1f %s" % (e["ts"] - w0, e["dur"], e["name"][:90]))
print("span %.1f" % (ops[-1]["ts"] + ops[-1]["dur"] - w0))
