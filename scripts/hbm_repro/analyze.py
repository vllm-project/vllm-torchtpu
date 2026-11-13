# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gzip
import json
import re
import statistics
import sys
from pathlib import Path

r = Path(sys.argv[1])
run = json.loads((r / "run.json").read_text())
results = {c["tag"]: dict(**c, ranks={}) for c in run["configs"]}
for p in sorted(r.glob("profile_*/**/*.trace.json.gz")):
    dirname = next(s for s in p.parts if s.startswith("profile_"))
    off = int(dirname.split("_")[1])
    count = int(dirname.split("_")[2][1:]) if len(dirname.split("_")) > 2 else 16
    batch = run["configs"][off : off + count]
    if "--only" in sys.argv and off != int(sys.argv[sys.argv.index("--only") + 1]):
        continue
    with gzip.open(p) as trace_file:
        es = json.load(trace_file)["traceEvents"]
    pids = {
        e["pid"]: e["args"]["name"]
        for e in es
        if e.get("ph") == "M"
        and e.get("name") == "process_name"
        and re.fullmatch(r"/device:TPU:\d+", e.get("args", {}).get("name", ""))
    }
    tids = {
        (e["pid"], e["args"]["name"]): e["tid"]
        for e in es
        if e.get("ph") == "M" and e.get("name") == "thread_name"
    }
    for pid, rank in pids.items():
        modules = sorted(
            [
                e
                for e in es
                if e.get("pid") == pid
                and e.get("tid") == tids.get((pid, "XLA Modules"))
                and e.get("ph") == "X"
                and e["name"].startswith("jit_")
            ],
            key=lambda e: e["ts"],
        )
        assert len(modules) == len(batch) * 10, (p, rank, len(modules), len(batch) * 10)
        scopes = sorted(
            [
                e
                for e in es
                if e.get("pid") == pid
                and e.get("tid") == tids.get((pid, "XLA TraceMe"))
                and e.get("ph") == "X"
                and e["name"].startswith("DMA_")
            ],
            key=lambda e: e["ts"],
        )
        for j, cfg in enumerate(batch):
            ms = modules[j * 10 : (j + 1) * 10]
            calls = []
            eventnames = set()
            expected = 64 if cfg["mode"] == "latency" else 1
            for m in ms:
                ev = [
                    e
                    for e in scopes
                    if m["ts"] - 0.01 <= e["ts"]
                    and e["ts"] + e["dur"] <= m["ts"] + m["dur"] + 0.01
                ]
                assert len(ev) == expected, (
                    cfg["tag"],
                    rank,
                    m["ts"],
                    len(ev),
                    expected,
                )
                calls.append(statistics.mean(e["dur"] for e in ev))
                eventnames.update(e["name"] for e in ev)
            us = statistics.mean(calls[2:])
            res = dict(
                scope_us=us,
                call_scope_us=calls,
                std_us=statistics.pstdev(calls[2:]),
                events=expected * 10,
                device_module_us=[m["dur"] for m in ms],
                observed_scope_names=sorted(eventnames),
            )
            if cfg["mode"] != "latency":
                res.update(
                    bandwidth_TBps=run["total_bytes"] / us / 1e6,
                    MBU=run["total_bytes"] / us / 1e-6 / run["peak_bytes_per_second"],
                )
            results[cfg["tag"]]["ranks"][rank] = res
    print("ANALYZED_BATCH", off, flush=True)
out = (
    [o for o in results.values() if o["ranks"]]
    if "--only" in sys.argv
    else list(results.values())
)
for o in out:
    assert len(o["ranks"]) == 8, (o["tag"], len(o["ranks"]))
    o["mean_scope_us"] = statistics.mean(x["scope_us"] for x in o["ranks"].values())
(
    r
    / (
        f"batch_{sys.argv[sys.argv.index('--only') + 1]}.json"
        if "--only" in sys.argv
        else "analysis.json"
    )
).write_text(json.dumps(out, indent=2))
print("VERIFIED", len(out), "configurations x8ranks")
for o in out:
    d = o["ranks"]["/device:TPU:0"]
    print(o["tag"], round(d["scope_us"], 6), round(d.get("MBU", 0) * 100, 2))
