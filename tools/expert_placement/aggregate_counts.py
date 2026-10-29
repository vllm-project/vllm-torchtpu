#!/usr/bin/env python3
# Copyright 2026 Google LLC
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
"""Sum the routing counter's per-process dumps into one per-layer counts file.

Reads every ``ep_routing_counts_rank<r>_pid<p>.npz`` under the given
directories (numpy only, no torch), optionally subtracts an earlier copy of
the same dumps (``--subtract``: drops warm-up and anything before the window
you want), and writes ``counts.npz`` with, per routed decoder layer (draft
layers are left out):

- ``decode`` int64 [L, E]: hits of valid rows in decode steps
- ``prefill`` int64 [L, E]: hits of valid rows in prefill steps
- ``steps`` int64 [L, S], ``step_names``: the counter's step counters
- ``layers`` (layers with traffic), ``num_experts``, ``topk``, ``workers``,
  ``ranks_identical``, ``sources``

Ranks that route the same tokens (attention replicated over the TP group)
write identical counts; summing them scales every layer equally, which the
solver does not see (it uses each layer's shares and per-step ratios).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

DUMP_RE = re.compile(r"ep_routing_counts_rank(\d+)_pid(\d+)\.npz$")


def find_dumps(dirs: list[str]) -> list[str]:
    out = []
    for d in dirs:
        out += glob.glob(
            os.path.join(d, "**", "ep_routing_counts_rank*_pid*.npz"), recursive=True
        )
    return sorted(p for p in out if DUMP_RE.search(os.path.basename(p)))


def load_dump(path: str) -> dict:
    """One dump as target-layer arrays keyed by layer index."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if int(d.get("format_version", 0)) != 1:
        raise ValueError(f"{path}: not an ep_routing_counter dump (format 1)")
    target = ~d["slot_is_draft"].astype(bool)
    layers = d["slot_layer"][target]
    if len(set(layers.tolist())) != len(layers):
        raise ValueError(f"{path}: two target MoE slots share a layer index")
    n = int(layers.max()) + 1 if len(layers) else 0
    by_class = np.zeros((n,) + d["counts_by_class"].shape[1:], dtype=np.int64)
    steps = np.zeros((n, d["step_counts"].shape[1]), dtype=np.int64)
    by_class[layers] = d["counts_by_class"][target]
    steps[layers] = d["step_counts"][target]
    return {
        "key": (int(d["rank"]), int(d["pid"])),
        "by_class": by_class,
        "steps": steps,
        "class_names": [str(x) for x in d["class_names"]],
        "step_names": [str(x) for x in d["step_names"]],
        "num_experts": int(d["num_experts"]),
        "topk": int(d["topk"]),
        "draft_layers": sorted(int(x) for x in d["slot_layer"][~target]),
    }


def _by_key(paths: list[str]) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for p in paths:
        d = load_dump(p)
        if d["key"] in out:
            raise ValueError(f"two dumps for rank/pid {d['key']}: {p}")
        d["path"] = p
        out[d["key"]] = d
    return out


def subtract(later: dict, earlier: dict) -> dict:
    """later - earlier per worker; refuses a changed worker set (a restarted
    worker counts from zero again) or a count that went down."""
    if set(later) != set(earlier):
        raise ValueError(
            "--subtract needs dumps of the same processes: "
            f"only later {sorted(set(later) - set(earlier))}, "
            f"only earlier {sorted(set(earlier) - set(later))}"
        )
    out = {}
    for key, d in later.items():
        e = earlier[key]
        if d["by_class"].shape != e["by_class"].shape:
            raise ValueError(f"worker {key}: dump shapes differ")
        dc = d["by_class"] - e["by_class"]
        ds = d["steps"] - e["steps"]
        if (dc < 0).any() or (ds < 0).any():
            raise ValueError(f"worker {key}: a count decreased; earlier is not earlier")
        out[key] = dict(d, by_class=dc, steps=ds)
    return out


def aggregate(workers: dict) -> dict:
    ds = [workers[k] for k in sorted(workers)]
    first = ds[0]
    for d in ds[1:]:
        for f in ("class_names", "step_names", "num_experts", "topk"):
            if d[f] != first[f]:
                raise ValueError(f"dumps disagree on {f}")
        if d["by_class"].shape != first["by_class"].shape:
            raise ValueError("dumps disagree on the layer set")
    by_class = sum(d["by_class"] for d in ds)
    steps = sum(d["steps"] for d in ds)
    cls = {n: i for i, n in enumerate(first["class_names"])}
    stp = {n: i for i, n in enumerate(first["step_names"])}
    decode = by_class[:, cls["decode"]]
    prefill = by_class[:, cls["prefill"]] + by_class[:, cls["prefill_decode_rows"]]
    layers = np.flatnonzero(by_class.sum(axis=(1, 2)) > 0)
    identical = all(np.array_equal(d["by_class"], first["by_class"]) for d in ds[1:])
    topk = first["topk"]
    unknown = int(
        by_class[:, cls["unknown"]].sum() + by_class[:, cls["unknown_padded"]].sum()
    )
    checks = {
        "unknown_hits": unknown,
        "pad_mismatch_rows": int(
            steps[layers, stp["pad_mismatch_rows"]].max(initial=0)
        ),
        "step_counters_equal_across_layers": all(
            int(np.ptp(steps[layers, stp[n]])) == 0
            for n in ("decode_steps", "prefill_steps", "unknown_steps")
        )
        if len(layers)
        else True,
        "hits_multiple_of_topk": bool(
            topk > 0 and not np.any(by_class[layers].sum(axis=2) % topk)
        ),
        "ranks_identical": bool(identical),
    }
    return {
        "decode": decode,
        "prefill": prefill,
        "steps": steps,
        "step_names": first["step_names"],
        "layers": layers,
        "num_experts": first["num_experts"],
        "topk": topk,
        "workers": len(ds),
        "draft_layers": first["draft_layers"],
        "checks": checks,
        "sources": [d["path"] for d in ds],
    }


def save(path: str, agg: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        decode=agg["decode"],
        prefill=agg["prefill"],
        steps=agg["steps"],
        step_names=np.array(agg["step_names"]),
        layers=agg["layers"].astype(np.int64),
        num_experts=np.int64(agg["num_experts"]),
        topk=np.int64(agg["topk"]),
        workers=np.int64(agg["workers"]),
        ranks_identical=np.bool_(agg["checks"]["ranks_identical"]),
        checks_json=np.array(json.dumps(agg["checks"])),
        sources=np.array(agg["sources"]),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", help="directories searched for dumps")
    ap.add_argument("-o", "--output", required=True, help="counts .npz to write")
    ap.add_argument("--subtract", default="", help="directory of earlier dumps")
    args = ap.parse_args(argv)
    paths = find_dumps(args.dirs)
    if not paths:
        print(f"no ep_routing_counts_rank*_pid*.npz under {args.dirs}", file=sys.stderr)
        return 1
    workers = _by_key(paths)
    if args.subtract:
        workers = subtract(workers, _by_key(find_dumps([args.subtract])))
    agg = aggregate(workers)
    save(args.output, agg)
    lay = agg["layers"]
    stp = {n: i for i, n in enumerate(agg["step_names"])}
    ref = int(lay[0]) if len(lay) else 0
    print(
        f"dumps            : {agg['workers']} (ranks identical: "
        f"{agg['checks']['ranks_identical']})"
    )
    print(
        f"layers           : {len(lay)}"
        + (f" ({lay.min()}-{lay.max()})" if len(lay) else "")
        + f"; draft layers left out: {agg['draft_layers']}"
    )
    print(
        f"decode hits      : {int(agg['decode'].sum()):,} over "
        f"{int(agg['steps'][ref, stp['decode_steps']]):,} decode steps (layer {ref})"
    )
    print(
        f"prefill hits     : {int(agg['prefill'].sum()):,} over "
        f"{int(agg['steps'][ref, stp['prefill_steps']]):,} prefill steps"
    )
    print(f"checks           : {json.dumps(agg['checks'])}")
    if agg["checks"]["unknown_hits"]:
        print("WARNING: rows were counted without attention metadata (unknown)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
