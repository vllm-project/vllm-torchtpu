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
"""Render all recorded device measurements as Markdown; no TPU dependencies."""

import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("# HBM DMA measurements\n")
print(
    "Times are microseconds. MBU uses 3.69 TB/s/device. "
    "Means cover eight devices, with the first two of ten calls discarded.\n"
)
for path in sorted(root.glob("*/analysis.json")):
    suite = path.parent.name
    records = json.loads(path.read_text())
    print(f"## {suite} ({len(records)} configurations)\n")
    print("| Mode | KiB | Q | Pattern | Fragments | Time (us) | MBU (%) |")
    print("| --- | ---: | ---: | --- | ---: | ---: | ---: |")
    for item in records:
        assert len(item["ranks"]) == 8
        for rank in item["ranks"].values():
            assert len(rank["call_scope_us"]) == 10
            assert rank["events"] == (640 if item["mode"] == "latency" else 10)
        mbu = statistics.mean(rank.get("MBU", 0) for rank in item["ranks"].values())
        utilization = "—" if item["mode"] == "latency" else f"{mbu * 100:.2f}"
        print(
            f"| {item['mode']} | {item['block_bytes'] / 1024:g} | {item['q']} | "
            f"{item.get('pattern', 'direct')} | {item.get('fragments', 1)} | "
            f"{item['mean_scope_us']:.3f} | {utilization} |"
        )
    if path != sorted(root.glob("*/analysis.json"))[-1] or suite == "contiguous":
        print()
    if suite == "contiguous":
        latency = [item for item in records if item["mode"] == "latency"]
        for minimum in [0, 1024**2]:
            selected = [item for item in latency if item["block_bytes"] >= minimum]
            if len(selected) < 3:
                continue
            xs = [item["block_bytes"] / 1024**2 for item in selected]
            ys = [item["mean_scope_us"] for item in selected]
            slope, intercept = statistics.linear_regression(xs, ys)
            error = max(abs((intercept + slope * x - y) / y) for x, y in zip(xs, ys))
            print(
                f"Fit, minimum {minimum} bytes: a={intercept:.6f} us; "
                f"B={1024**2 / slope / 1e6:.6f} TB/s; "
                f"max relative error={error * 100:.2f}%.\n"
            )
