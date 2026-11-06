#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device step per TensorCore from a server log with libtpu pjrt_client lines
(TPU_VMODULE=tpu_pjrt_client=1).

    pjrt_step.py SERVER_LOG[.gz] REQUEST_DIR --tz +0800 [--year 2026] [--pad 2.0]

REQUEST_DIR holds bench_sweep.py's request-*.json of one run (client clock = this
host's clock). Window = bench_sweep's steady window: [last first token + pad, first
last token - pad] (all requests decoding). Step = mean spacing of the once-per-step
program (spec verify prelude / target sampler at K>0; the logits row gather on the
candidate verify path) on each core. Busy/idle per core from the in-flight count.
Prints one JSON object.
"""

import argparse
import collections
import glob
import gzip
import json
import re
import statistics
from datetime import datetime, timedelta, timezone

DEV = re.compile(
    r"^I(\d\d)(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{6})\s+\d+ tpu_pjrt_client\.cc:\d+\]\s+"
    r"\(([^)]*)\)Executables(Start|Complete)(?:\(([^)]*)\))?: "
    r"[+-]+in_flight_computations_count_: (\d+)"
)
MARKERS = ("spec_bonus_and_target_tokens", "sample_from_logits_argmax")
FALLBACK = "spec_gather_bonus_and_target_logits"


def parse(path, tz, year):
    sign = 1 if tz[0] == "+" else -1
    zone = timezone(timedelta(hours=int(tz[1:3]), minutes=int(tz[3:5])) * sign)
    cores = collections.defaultdict(list)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="ignore") as f:
        for line in f:
            if "tpu_pjrt_client" not in line:
                continue
            m = DEV.match(line)
            if not m:
                continue
            mo, dd, hh, mi, ss, us, core, kind, name, count = m.groups()
            ts = datetime(
                year, int(mo), int(dd), int(hh), int(mi), int(ss), int(us), tzinfo=zone
            ).timestamp()
            cores[core].append((ts, kind, name or "", int(count)))
    return cores


def executions(events):
    queue, out, last = collections.deque(), [], None
    for ts, kind, name, _ in events:
        if kind == "Start":
            queue.append(ts)
        elif queue:
            enq = queue.popleft()
            start = enq if last is None else max(enq, last)
            out.append((enq, start, ts, name))
            last = ts
    return out


def busy_idle(events, w0, w1):
    busy = idle = 0.0
    prev_t, prev_c = None, 0
    for ts, _, _, c in events:
        if prev_t is not None:
            a, b = max(prev_t, w0), min(ts, w1)
            if b > a:
                if prev_c > 0:
                    busy += b - a
                else:
                    idle += b - a
        prev_t, prev_c = ts, c
    return busy, idle


def window(req_dir, pad):
    rows = []
    for f in glob.glob(f"{req_dir}/request-*.json"):
        with open(f) as fh:
            rows.append(json.load(fh)["summary"])
    if not rows or not all("first" in r for r in rows):
        return None
    lo = max(r["first"] for r in rows) + pad
    hi = min(r["last"] for r in rows) - pad
    off = statistics.median(r["wall_start"] - r["start"] for r in rows)
    return lo + off, hi + off, len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("req_dir")
    p.add_argument("--tz", default="+0800")
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--pad", type=float, default=2.0)
    a = p.parse_args()
    w = window(a.req_dir, a.pad)
    out = {"log": a.log}
    if not w or w[1] <= w[0]:
        out["error"] = "no steady window"
        print(json.dumps(out))
        return
    w0, w1, n = w
    out.update(window_unix=[w0, w1], window_s=w1 - w0, requests=n)
    cores = parse(a.log, a.tz, a.year)
    res = {}
    for core in sorted(cores):
        ev = cores[core]
        ex = executions(ev)
        marks = [
            done
            for _, _, done, nm in ex
            if any(m in nm for m in MARKERS) and w0 <= done <= w1
        ]
        if (
            len(marks) < 3
        ):  # candidate verify path: its logits row gather runs once per step
            marks = [
                done for _, _, done, nm in ex if FALLBACK in nm and w0 <= done <= w1
            ]
        busy, idle = busy_idle(ev, w0, w1)
        r = {
            "busy_frac": round(busy / (w1 - w0), 4),
            "idle_frac": round(idle / (w1 - w0), 4),
        }
        if len(marks) >= 3:
            r["step_ms"] = round(1000 * (marks[-1] - marks[0]) / (len(marks) - 1), 3)
            r["steps"] = len(marks) - 1
        cnt = collections.Counter(nm for _, _, done, nm in ex if w0 <= done <= w1)
        r["programs_per_step"] = round(sum(cnt.values()) / max(r.get("steps", 1), 1), 2)
        res[core] = r
    out["cores"] = res
    steps = [r["step_ms"] for r in res.values() if "step_ms" in r]
    if steps:
        out["step_ms_median"] = round(statistics.median(steps), 3)
        out["step_ms_min"] = min(steps)
        out["step_ms_max"] = max(steps)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
