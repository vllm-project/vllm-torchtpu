# SPDX-License-Identifier: Apache-2.0
"""Verify fine TraceMe and report overlay spans per core and invocation."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from vllm_torchtpu.tools.xplane_pb2 import XSpace


def analyze(path: Path) -> dict:
    space = XSpace()
    space.ParseFromString(path.read_bytes())
    report = {}
    for plane in space.planes:
        if not (
            plane.name.startswith("/device:TPU:")
            and plane.name.rsplit(":", 1)[-1].isdigit()
        ):
            continue

        def events(line_name, plane=plane):
            return sorted(
                set(
                    (
                        line.timestamp_ns * 1000 + event.offset_ps,
                        event.duration_ps,
                        plane.event_metadata[event.metadata_id].display_name
                        or plane.event_metadata[event.metadata_id].name,
                    )
                    for line in plane.lines
                    if line.name == line_name
                    for event in line.events
                )
            )

        traces = events("XLA TraceMe")
        counts = Counter(name for _, _, name in traces)
        modules = defaultdict(list)
        for start, duration, name in events("XLA Modules"):
            modules[name].append((start, duration))
        runs = [entries for entries in modules.values() if len(entries) == 5]
        samples = []
        for index in range(2, 5) if runs else []:
            start = min(entries[index][0] for entries in runs)
            end = max(sum(entries[index]) for entries in runs)
            overlays = [
                (t, d)
                for t, d, name in traces
                if name == "overlayer-overhead" and start <= t and t + d <= end + 100000
            ]
            # Union intervals: nested TraceMe must not be double-counted.
            merged = []
            for t, duration in overlays:
                if merged and t <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], t + duration)
                else:
                    merged.append([t, t + duration])
            samples.append(
                {
                    "call_index": index,
                    "full_us": (end - start) / 1e6,
                    "overlay_union_us": sum(b - a for a, b in merged) / 1e6,
                    "overlay_spans_us": [d / 1e6 for _, d in overlays],
                }
            )
        required = [
            "ep_copy_in",
            "ep_wait_in",
            "ep_run_kernel",
            "bm64_first_last",
            "ep_finalize",
        ]
        report[plane.name] = {
            "missing_traces": [name for name in required if not counts[name]],
            "trace_counts": {name: counts[name] for name in required},
            "samples": samples,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = list(args.profile.rglob("*.xplane.pb"))
    if not paths:
        raise SystemExit("No xplane.pb files found")
    reports = {str(path): analyze(path) for path in paths}
    args.output.write_text(json.dumps(reports, indent=2))
    valid = all(reports.values()) and all(
        core["samples"] and not core["missing_traces"]
        for report in reports.values()
        for core in report.values()
    )
    if not valid:
        raise SystemExit("Missing five-call module windows or required fine TraceMe")
    print(f"Verified TraceMe; per-core measurements: {args.output}")


if __name__ == "__main__":
    main()
