# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize device spans from the per-capture XProf traces.

Reads <root>/profiling/xprof/<backend>-b<b>/*.xplane.pb and writes
<root>/trace_summary.json with, per capture: Host gather kernel spans; each
XLA module's per-step time, taking the slowest device at every step; and an
inventory of plane and line names for checking what the capture contains.
"""

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from vllm_torchtpu.tools import xplane_pb2
except ImportError:  # Copied next to this script when run outside the repo.
    import xplane_pb2

KERNELS = (
    "host_gather_request",
    "host_gather_receive",
    "host_gather_publish",
    "engram_ffi_request",
    "engram_ffi_h2d_start",
    "engram_ffi_h2d_wait",
    "host_embedding_dma_start",
    "host_embedding_dma_wait",
)


def load(path):
    data = path.read_bytes()
    space = xplane_pb2.XSpace()
    space.ParseFromString(gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data)
    return space


def stats(values):
    return {
        "count": len(values),
        "p50": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def main(root):
    root = Path(root)
    summary = {}
    for capture in sorted((root / "profiling/xprof").iterdir()):
        kernels, modules, inventory = defaultdict(list), defaultdict(list), Counter()
        ops = defaultdict(list)
        for path in sorted(capture.glob("*.xplane.pb")):
            for plane in load(path).planes:
                names = {k: v.name for k, v in plane.event_metadata.items()}
                executions = defaultdict(list)
                for line in plane.lines:
                    inventory[f"{plane.name} | {line.name}"] += len(line.events)
                    for event in sorted(line.events, key=lambda e: e.offset_ps):
                        name = names.get(event.metadata_id, "")
                        span_us = event.duration_ps / 1e6
                        if line.name == "XLA Modules":
                            executions[name].append(span_us)
                        # TensorCore ops, to compare single matmuls across graphs.
                        elif line.name == "XLA Ops" and "SparseCore" not in plane.name:
                            ops[name].append(span_us)
                        kernel = next((k for k in KERNELS if k in name), None)
                        if kernel:
                            kernels[f"{kernel} | {line.name}"].append(span_us)
                for name, spans in executions.items():
                    modules[name].append(spans)

        # The k-th execution of a module on every device is the same step.
        per_step = {}
        for name, devices in modules.items():
            steps = min(len(spans) for spans in devices)
            slowest = np.max([spans[:steps] for spans in devices], axis=0)
            per_step[name] = {**stats(slowest), "devices": len(devices)}
        busiest = sorted(per_step, key=lambda n: -per_step[n]["p50"])[:6]
        heaviest = sorted(ops, key=lambda n: -sum(ops[n]))[:8]
        summary[capture.name] = {
            "kernels_us": {key: stats(v) for key, v in sorted(kernels.items())},
            "modules_us": {name: per_step[name] for name in busiest},
            "ops_us": {name: stats(ops[name]) for name in heaviest},
            "inventory": dict(inventory.most_common(40)),
        }

    (root / "trace_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    brief = {
        name: {
            "kernels": v["kernels_us"],
            "modules": v["modules_us"],
            "ops": v["ops_us"],
        }
        for name, v in summary.items()
    }
    print("XPROF_SUMMARY", json.dumps(brief))
    first = next(iter(summary.values()), {})
    print("XPROF_INVENTORY", json.dumps(first.get("inventory", {})))


if __name__ == "__main__":
    main(sys.argv[1])
