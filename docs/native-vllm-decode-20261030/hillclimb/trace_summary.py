# SPDX-License-Identifier: Apache-2.0
"""Per-step op time of one decode trace (one TensorCore's XPlane), by category.

  python3 trace_summary.py RANK.xplane.pb [--mtp-k 3] [--layers 92]

Prints one JSON line: steps in the window (executions of the busiest program
divided by MTP_K + 1), and per step the device time of the ops on the core's
"XLA Ops" line (container ops excluded) split into the MoE exchange
(ragged all-to-all and the home-rank sum), the fused expert kernel (which also
sends the direct-combine rows), other collectives, attention, GDN and the
rest. With the ragged all-to-all exchange, a MoE layer on the capacity path
with the direct combine runs one ragged all-to-all and one fused expert kernel;
a layer that overflowed the capacity runs the full-size exchange, two ragged
all-to-alls and no fused kernel. ``capacity_fallbacks_per_step`` estimates the
latter from those counts.
"""

import argparse
import collections
import importlib.util
import json
import os
import re


def _xplane_pb2():
    """vllm_torchtpu/tools/xplane_pb2.py, loaded without importing the package."""
    root = os.path.dirname(importlib.util.find_spec("vllm_torchtpu").origin)
    spec = importlib.util.spec_from_file_location(
        "xplane_pb2", os.path.join(root, "tools", "xplane_pb2.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTAINERS = ("conditional", "while", "call")
CATEGORIES = (
    ("exchange_a2a", r"^ragged_all_to_all|^ragged-all-to-all"),
    ("exchange_home_sum", r"^ragged_a2a_home_sum"),
    ("expert_kernel", r"^gmm_v2"),
    ("sparsecore_gather", r"^sc_ragged_gather"),
    ("all_gather", r"^all-gather"),
    ("all_reduce", r"^all-reduce"),
    ("reduce_scatter", r"^reduce-scatter"),
    ("attention", r"^RPA|^rpa_"),
    ("gdn", r"gdn"),
    ("router", r"^router_topk"),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xplane")
    ap.add_argument("--mtp-k", type=int, default=3)
    ap.add_argument("--layers", type=int, default=92)
    a = ap.parse_args()
    xs = _xplane_pb2().XSpace()
    with open(a.xplane, "rb") as f:
        xs.ParseFromString(f.read())
    plane = next(p for p in xs.planes if p.name == "/device:TPU:0")
    names = {k: v.name for k, v in plane.event_metadata.items()}
    lines = {line.name: line for line in plane.lines}
    modules = collections.Counter(
        re.sub(r"\(\d+\)$", "", names.get(e.metadata_id, ""))
        for e in lines["XLA Modules"].events
    )
    module_ps = collections.Counter()
    for e in lines["XLA Modules"].events:
        module_ps[re.sub(r"\(\d+\)$", "", names.get(e.metadata_id, ""))] += (
            e.duration_ps
        )
    main_module = module_ps.most_common(1)[0][0]
    steps = modules[main_module] / (a.mtp_k + 1)
    ps = collections.Counter()
    count = collections.Counter()
    gmm_rows = collections.Counter()
    for e in lines["XLA Ops"].events:
        n = names.get(e.metadata_id, "?").split(" = ")[0].lstrip("%")
        base = re.sub(r"\.\d+$", "", n)
        if base in CONTAINERS:
            continue
        cat = next((c for c, pat in CATEGORIES if re.search(pat, base)), "other")
        ps[cat] += e.duration_ps
        count[cat] += 1
        if base.startswith("gmm_v2_mlp_scatter"):
            count["fused_expert_kernel"] += 1
        m = re.search(r"gmm_v2-.*-m_(\d+)-", base)
        if m:
            gmm_rows[int(m.group(1))] += 1
    total = sum(ps.values())
    per_step = {c: round(v / 1e9 / steps, 3) for c, v in sorted(ps.items())}
    counts = {c: round(v / steps, 2) for c, v in sorted(count.items())}
    a2a, fused = count["exchange_a2a"], count["fused_expert_kernel"]
    fallbacks = (a2a - fused) / 2 / steps if fused else None
    print(
        json.dumps(
            {
                "trace_summary": a.xplane.rsplit("/", 1)[-1],
                "steps": round(steps, 2),
                "op_ms_per_step": round(total / 1e9 / steps, 3),
                "exchange_share": round(
                    (ps["exchange_a2a"] + ps["exchange_home_sum"]) / total, 4
                )
                if total
                else None,
                "ms_per_step": per_step,
                "events_per_step": counts,
                "gmm_rows_per_step": {
                    str(k): round(v / steps, 2) for k, v in sorted(gmm_rows.items())
                },
                "capacity_fallbacks_per_step": None
                if fallbacks is None
                else round(fallbacks, 2),
                "moe_layers": a.layers,
            }
        )
    )


if __name__ == "__main__":
    main()
