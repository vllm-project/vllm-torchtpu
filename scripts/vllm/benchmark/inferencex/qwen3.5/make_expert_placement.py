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
"""Build a TPU_EXPERT_PLACEMENT table (expert_placement_*_copies<N>.json) from
a MoE routing recording (TPU_MOE_ROUTING_RECORD).

Inputs: one or more recordings, summed -- the recorder's
``routing_counts_rank<r>.json`` files, or a server log holding its
``MOE_ROUTING_COUNTS`` lines (the last dump of every layer per rank is
taken). Every rank sees every token under TP/EP, so one rank's file is
enough; passing all of them only scales the counts.

Per layer the load of expert e is ``decode[e] + prefill_weight * prefill[e]``
(default: decode-sized steps only, where the few-token kernel and the copies
act). The placement:

  * primaries: the balanced permutation -- vLLM's EPLB policy
    (``DefaultEplbPolicy.rebalance_experts``, one group, one node,
    ``num_replicas = experts``) when vllm imports, else a greedy longest-
    processing-time partition (experts by load, descending, each to the core
    with the least load that still has a free primary slot). Every expert
    exactly once across the cores;
  * copies: the ``--copies`` experts with the highest load, in that order,
    each placed on the core with the least expected load that has a free
    extra slot and does not already hold the expert. Expected load counts a
    copied expert as half on each holder.

Output: the format `src/vllm_torchtpu/expert_placement.py` loads --
``shards``, ``experts``, ``owned_per_core``, ``extra_slots``,
``slots_per_core``, ``copies``, then ``layers[<n>].cores[r]`` (the
``slots_per_core`` global expert ids of rank r: primaries first, then its
copy slots) and ``layers[<n>].copies`` (expert -> holder of its copy).

    make_expert_placement.py routing_counts_rank0.json \\
        --shards 8 --experts 512 --copies 128 -o expert_placement_agentx_copies128.json
    make_expert_placement.py --check expert_placement_dp8_ep8_copies128.json
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import zlib
from typing import Any

LOG_TAG = "MOE_ROUTING_COUNTS"
_LOG_RE = re.compile(
    rf"{LOG_TAG} rank=(\d+) step=(\d+) layer=(\S+) ([A-Za-z0-9+/=]+)\s*$"
)


# --- reading recordings -----------------------------------------------------


def _decode(text: str) -> Any:
    return json.loads(zlib.decompress(base64.b64decode(text)).decode())


def read_log(path: str) -> dict[str, Any]:
    """The last ``MOE_ROUTING_COUNTS`` dump of every layer, summed over the
    ranks that logged, as a recording document."""
    last: dict[tuple[int, str], tuple[int, dict]] = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = _LOG_RE.search(line)
            if not m:
                continue
            rank, step, layer, payload = m.groups()
            key = (int(rank), layer)
            if key not in last or last[key][0] <= int(step):
                last[key] = (int(step), _decode(payload))
    if not last:
        raise SystemExit(f"{path}: no {LOG_TAG} lines")
    layers: dict[str, dict] = {}
    for (_, layer), (_, entry) in sorted(last.items()):
        acc = layers.setdefault(layer, {"decode": None, "prefill": None})
        for k in ("decode", "prefill"):
            acc[k] = (
                list(entry[k])
                if acc[k] is None
                else [a + b for a, b in zip(acc[k], entry[k], strict=True)]
            )
    return {"layers": layers, "source": path}


def read_recording(path: str) -> dict[str, Any]:
    if path.endswith(".json"):
        with open(path) as f:
            doc = json.load(f)
        if "layers" not in doc:
            raise SystemExit(f"{path}: not a routing recording (no 'layers')")
        return doc
    return read_log(path)


def layer_loads(
    docs: list[dict[str, Any]], prefill_weight: float
) -> dict[int, list[float]]:
    """layer index -> per-expert load, summed over the recordings."""
    loads: dict[int, list[float]] = {}
    for doc in docs:
        for key, entry in doc["layers"].items():
            m = re.search(r"(\d+)", key)
            if m is None:
                raise SystemExit(f"layer key {key!r} carries no index")
            idx = int(m.group(1))
            dec = entry["decode"]
            pre = entry.get("prefill") or [0] * len(dec)
            load = [d + prefill_weight * p for d, p in zip(dec, pre, strict=True)]
            if idx in loads:
                loads[idx] = [a + b for a, b in zip(loads[idx], load, strict=True)]
            else:
                loads[idx] = load
    return loads


# --- the placement ------------------------------------------------------------


def balanced_primaries_lpt(load: list[float], shards: int) -> list[list[int]]:
    """Greedy longest-processing-time partition: experts by load, descending,
    each to the core with the least load that still has a free primary slot."""
    experts = len(load)
    owned = experts // shards
    cores: list[list[int]] = [[] for _ in range(shards)]
    core_load = [0.0] * shards
    for e in sorted(range(experts), key=lambda e: (-load[e], e)):
        r = min(
            (r for r in range(shards) if len(cores[r]) < owned),
            key=lambda r: (core_load[r], r),
        )
        cores[r].append(e)
        core_load[r] += load[e]
    return cores


def balanced_primaries_eplb(loads: dict[int, list[float]], shards: int):
    """vLLM's EPLB policy on every layer at once (its signature takes the
    [layers, experts] matrix). None when vllm is not importable."""
    try:
        import torch
        from vllm.distributed.eplb.policy.default import DefaultEplbPolicy
    except Exception:  # noqa: BLE001 - any import failure: fall back
        return None
    idxs = sorted(loads)
    experts = len(loads[idxs[0]])
    weight = torch.tensor([loads[i] for i in idxs], dtype=torch.float32)
    phy2log = DefaultEplbPolicy.rebalance_experts(
        weight, num_replicas=experts, num_groups=1, num_nodes=1, num_ranks=shards
    )
    owned = experts // shards
    out = {}
    for row, i in enumerate(idxs):
        flat = [int(v) for v in phy2log[row].tolist()]
        out[i] = [flat[r * owned : (r + 1) * owned] for r in range(shards)]
    return out


def place_copies(
    load: list[float], cores: list[list[int]], extra_slots: int, copies: int
) -> dict[int, int]:
    """expert -> holder of its copy, hottest first: the core with the least
    expected load that has a free extra slot and does not own the expert.
    Expected load counts a copied expert as half on each holder."""
    shards = len(cores)
    owner = {e: r for r, c in enumerate(cores) for e in c}
    expected = [sum(load[e] for e in c) for c in cores]
    free = [extra_slots] * shards
    placed: dict[int, int] = {}
    for e in sorted(range(len(load)), key=lambda e: (-load[e], e))[:copies]:
        candidates = [r for r in range(shards) if free[r] > 0 and r != owner[e]]
        if not candidates:
            # the free slots left are all on e's owner: move an earlier copy
            # (one whose expert the owner does not own) there, and take its
            # slot
            o = owner[e]
            moved = next(
                (
                    (e2, r2)
                    for e2, r2 in reversed(list(placed.items()))
                    if owner[e2] != o and r2 != o
                ),
                None,
            )
            if moved is None:
                raise SystemExit(f"no core can take a copy of expert {e}")
            e2, r2 = moved
            placed[e2] = o
            free[o] -= 1
            free[r2] += 1
            expected[r2] -= load[e2] / 2
            expected[o] += load[e2] / 2
            candidates = [r2]
        r = min(candidates, key=lambda r: (expected[r], r))
        placed[e] = r
        free[r] -= 1
        expected[owner[e]] -= load[e] / 2
        expected[r] += load[e] / 2
    return placed


def build_layer(
    load: list[float], primaries: list[list[int]], extra_slots: int, copies: int
) -> dict[str, Any]:
    placed = place_copies(load, primaries, extra_slots, copies)
    cores = [list(c) for c in primaries]
    for e, r in placed.items():
        cores[r].append(e)
    for c in cores:  # pad unused extra slots? none: slots_per_core is fixed
        if len(c) != len(primaries[0]) + extra_slots:
            raise SystemExit(
                "copies must fill every extra slot: "
                f"{copies} copies over {len(cores)} cores x {extra_slots} slots"
            )
    return {"cores": cores, "copies": {str(e): r for e, r in placed.items()}}


def build(
    loads: dict[int, list[float]],
    shards: int,
    experts: int,
    copies: int,
    algorithm: str = "auto",
) -> dict[str, Any]:
    if experts % shards:
        raise SystemExit(f"{experts} experts do not split over {shards} shards")
    if copies % shards:
        raise SystemExit(f"{copies} copies do not split over {shards} shards")
    owned = experts // shards
    extra = copies // shards
    for i, load in loads.items():
        if len(load) != experts:
            raise SystemExit(
                f"layer {i}: {len(load)} experts recorded, {experts} asked"
            )
    eplb = (
        balanced_primaries_eplb(loads, shards)
        if algorithm in ("auto", "eplb")
        else None
    )
    if algorithm == "eplb" and eplb is None:
        raise SystemExit("--algorithm eplb: vllm's EPLB policy is not importable")
    used = (
        "vllm.distributed.eplb.policy.default.DefaultEplbPolicy.rebalance_experts "
        "(global: one group, one node)"
        if eplb is not None
        else "greedy longest-processing-time partition (experts by load, each to "
        "the least-loaded core with a free primary slot)"
    )
    layers = {}
    for i in sorted(loads):
        prim = eplb[i] if eplb is not None else balanced_primaries_lpt(loads[i], shards)
        layers[str(i)] = (
            build_layer(loads[i], prim, extra, copies) if copies else {"cores": prim}
        )
    table: dict[str, Any] = {
        "shards": shards,
        "experts": experts,
        "owned_per_core": owned,
        "extra_slots": extra,
        "slots_per_core": owned + extra,
        "copies": bool(copies),
        "recording": "Per-layer expert activation statistics.",
        "algorithm": (
            f"the model-level balanced permutation as the primaries ({used}), then "
            f"the {copies} experts with the highest recorded activation each given "
            "one copy, placed on the core with the least expected load that still "
            "has a free slot"
        ),
        "rule": (
            "the few-token decode kernel picks, for each copied expert that a step "
            "hits, whichever of its two holders has fewer active experts on that step"
        ),
        "layers": layers,
    }
    validate(table)
    return table


# --- validation (the loader's checks) ---------------------------------------


def validate(table: dict[str, Any]) -> None:
    experts = int(table["experts"])
    ranks = int(table["shards"])
    per_rank = int(table["slots_per_core"])
    copies = bool(table.get("copies", False))
    owned_n = int(table.get("owned_per_core", per_rank)) if copies else per_rank
    if ranks * owned_n != experts or owned_n > per_rank:
        raise ValueError(
            f"{ranks} ranks x {owned_n} owned slots must equal {experts} experts "
            f"(slots per rank {per_rank})"
        )
    if (
        copies
        and int(table.get("extra_slots", per_rank - owned_n)) != per_rank - owned_n
    ):
        raise ValueError("extra_slots must be slots_per_core - owned_per_core")
    for n, layer in table["layers"].items():
        cores = layer["cores"]
        if len(cores) != ranks or any(len(c) != per_rank for c in cores):
            raise ValueError(f"layer {n} must list {ranks} ranks x {per_rank} slots")
        if sorted(e for c in cores for e in c[:owned_n]) != list(range(experts)):
            raise ValueError(
                f"layer {n} must hold every expert exactly once among the owned slots"
            )
        if copies:
            if any(e in c[:owned_n] for c in cores for e in c[owned_n:]):
                raise ValueError(f"layer {n} holds an expert as a copy on its owner")
            copy_slots = [e for c in cores for e in c[owned_n:]]
            if any(e < 0 or e >= experts for e in copy_slots):
                raise ValueError(f"layer {n} has an invalid copy id")
            if len(copy_slots) != len(set(copy_slots)):
                raise ValueError(f"layer {n} allows at most one extra copy per expert")
            declared = layer.get("copies")
            if isinstance(declared, dict):
                actual = {str(e): r for r, c in enumerate(cores) for e in c[owned_n:]}
                if declared != actual:
                    raise ValueError(f"layer {n} copy order must match the copy slots")


# --- writing ---------------------------------------------------------------------


def dumps(table: dict[str, Any]) -> str:
    """The reference files' layout: one line per core, eight copies per line."""
    out = ["{"]
    for k, v in table.items():
        if k != "layers":
            out.append(f"  {json.dumps(k)}: {json.dumps(v)},")
    out.append('  "layers": {')
    layer_items = list(table["layers"].items())
    for li, (n, layer) in enumerate(layer_items):
        out.append(f"    {json.dumps(n)}: {{")
        out.append('      "cores": [')
        for ci, c in enumerate(layer["cores"]):
            sep = "," if ci + 1 < len(layer["cores"]) else ""
            out.append(f"        [{', '.join(str(e) for e in c)}]{sep}")
        copies = layer.get("copies")
        if copies is None:
            out.append("      ]")
        else:
            out.append("      ],")
            out.append('      "copies": {')
            items = list(copies.items())
            for i in range(0, len(items), 8):
                chunk = items[i : i + 8]
                sep = "," if i + 8 < len(items) else ""
                out.append(
                    "        "
                    + ", ".join(f"{json.dumps(e)}: {r}" for e, r in chunk)
                    + sep
                )
            out.append("      }")
        out.append("    }" + ("," if li + 1 < len(layer_items) else ""))
    out.append("  }")
    out.append("}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "inputs", nargs="*", help="routing_counts_rank<r>.json or server logs"
    )
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--copies", type=int, default=128)
    p.add_argument(
        "--prefill-weight",
        type=float,
        default=0.0,
        help="weight of the prefill-sized steps' counts (default 0: decode only)",
    )
    p.add_argument("--algorithm", choices=["auto", "eplb", "lpt"], default="auto")
    p.add_argument("-o", "--output", default="-")
    p.add_argument(
        "--check", metavar="PLACEMENT_JSON", help="validate an existing table and exit"
    )
    args = p.parse_args(argv)
    if args.check:
        with open(args.check) as f:
            table = json.load(f)
        validate(table)
        print(
            f"{args.check}: ok -- {table['shards']} shards x {table['slots_per_core']} "
            f"slots, {table['experts']} experts, {len(table['layers'])} layers, "
            f"copies={bool(table.get('copies', False))}"
        )
        return 0
    if not args.inputs:
        p.error("a recording is required (or --check)")
    docs = [read_recording(path) for path in args.inputs]
    loads = layer_loads(docs, args.prefill_weight)
    table = build(loads, args.shards, args.experts, args.copies, args.algorithm)
    text = dumps(table)
    if args.output == "-":
        sys.stdout.write(text)
    else:
        with open(args.output, "w") as f:
            f.write(text)
        print(
            f"wrote {args.output}: {len(table['layers'])} layers, "
            f"{args.shards} x {table['slots_per_core']} slots "
            f"({table['owned_per_core']} owned + {table['extra_slots']} copies); "
            "primaries by "
            + ("EPLB" if "DefaultEplbPolicy" in table["algorithm"] else "LPT"),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
