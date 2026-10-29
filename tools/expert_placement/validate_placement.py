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
"""Check a placement table with the loader's own checks, before serving.

Runs the loader's table checks (``expert_placement.parse`` and
``check_layers``: a permutation per layer, only routed decoder layers), the
deployment's experts / ranks / slots and,
with ``--hf-config``, ``check_routing`` (no grouped top-k, no hash-routed
layers). The model bounds come from the model's ``config.json``
(``n_routed_experts``, ``first_k_dense_replace``, ``num_hidden_layers``) or
from flags. Prints the sha256 the loader logs on every EP rank:
``[eplb] placement installed ep_rank=<r> layers=<n> table_sha256=<16 hex>``.

Exit 0: valid. Exit 1: refused (the reason is printed).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from types import SimpleNamespace


def model_bounds(hf: dict) -> dict:
    """experts, first MoE layer and decoder layer count from a config.json."""
    text = hf.get("text_config") or hf
    if not text.get("n_routed_experts") or "num_hidden_layers" not in text:
        raise ValueError("config has no n_routed_experts or num_hidden_layers")
    return {
        "experts": int(text["n_routed_experts"]),
        "first_moe_layer": int(text.get("first_k_dense_replace") or 0),
        "num_layers": int(text["num_hidden_layers"]),
        "hf": SimpleNamespace(**text),
    }


def validate(
    raw: bytes,
    source: str,
    *,
    experts: int,
    ranks: int,
    slots: int,
    first_moe_layer: int,
    num_layers: int,
):
    """The loader's checks for one deployment, without a model: parse, the
    shape the deployment needs (``slots``: experts per rank, which a table
    with copies holds as its primaries), and the routed layer range."""
    from vllm_torchtpu import expert_placement as ep

    table = ep.parse(json.loads(raw), source)
    if (table.experts, table.ranks, table.owned) != (experts, ranks, slots):
        held = "slots" if table.owned == table.slots else "primaries"
        raise ValueError(
            f"expert placement {source}: the table is {table.ranks} ranks x "
            f"{table.owned} {held} over {table.experts} experts; this "
            f"deployment needs {ranks} x {slots} over {experts}"
        )
    ep.check_layers(table, first_moe_layer, num_layers, source)
    return dataclasses.replace(table, sha256=hashlib.sha256(raw).hexdigest())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("table", help="placement table (local path)")
    ap.add_argument("--ranks", type=int, required=True, help="EP ranks")
    ap.add_argument("--hf-config", default="", help="the model's config.json")
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--first-moe-layer", type=int, default=-1)
    ap.add_argument("--num-layers", type=int, default=0)
    args = ap.parse_args(argv)
    if args.ranks <= 0:
        ap.error("--ranks must be positive")

    from vllm_torchtpu import expert_placement as ep

    hf = None
    if args.hf_config:
        with open(args.hf_config) as f:
            b = model_bounds(json.load(f))
        hf = b.pop("hf")
    else:
        b = {
            "experts": args.experts,
            "first_moe_layer": args.first_moe_layer,
            "num_layers": args.num_layers,
        }
        if b["experts"] <= 0 or b["first_moe_layer"] < 0 or b["num_layers"] <= 0:
            ap.error(
                "pass --hf-config, or --experts, --first-moe-layer and --num-layers"
            )
    with open(args.table, "rb") as f:
        raw = f.read()
    try:
        table = validate(
            raw, args.table, ranks=args.ranks, slots=b["experts"] // args.ranks, **b
        )
        if hf is not None:
            ep.check_routing(table, hf, args.table)
    except (ValueError, KeyError, TypeError) as e:
        print(f"REFUSED {args.table}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    layers = sorted(table.cores)
    if not layers:
        print(f"REFUSED {args.table}: places no layer", file=sys.stderr)
        return 1
    print(
        f"OK {args.table}: {len(layers)} layers ({layers[0]}..{layers[-1]}), "
        f"{table.ranks} ranks x {table.slots} slots over {table.experts} experts; "
        f"sha256 {table.sha256}; expect table_sha256={table.sha256[:16]} in the "
        "[eplb] placement installed lines"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
