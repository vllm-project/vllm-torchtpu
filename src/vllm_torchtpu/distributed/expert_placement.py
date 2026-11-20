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
"""Static expert placement: which EP rank holds which routed expert, per layer.

A TABLE_FORMAT table gives every MoE layer `L` the physical index `slots[L][e]`
of each logical routed expert `e`. Tables come from the served traffic's
routing (`moe_prefill_sp`'s `moe_expert_load_stats` counts, dumped by
`TPUWorker.dump_moe_expert_load`) through `lpt_slots`; `main` turns a dump into
a table offline.
"""

import argparse
import json
import os
from typing import Any

import numpy as np

TABLE_FORMAT = "vllm_torchtpu.moe_expert_placement.v1"
STATS_FORMAT = "vllm_torchtpu.moe_expert_load.v1"


def lpt_slots(rows: np.ndarray, ep_size: int, ep_tp_size: int) -> np.ndarray:
    """slot[e], the physical index of logical expert e, from every expert's
    routed rows. Longest first, each expert goes to the EP group (chip) with the
    fewest rows among those with room left; then, longest first within each
    group, to the group member with the fewest rows. Physical expert p sits on
    mesh rank p // (E / ep_size), as the linear expert map lays them out; a rank
    holds its experts in index order."""
    rows = np.asarray(rows, np.int64)
    e = rows.size
    if ep_size % ep_tp_size or e % ep_size:
        raise ValueError(
            f"{e} experts do not split over ep {ep_size} / tp {ep_tp_size}"
        )
    groups, per_rank = ep_size // ep_tp_size, e // ep_size
    room = np.full(groups, per_rank * ep_tp_size)
    load = np.zeros(groups, np.int64)
    group_of = np.empty(e, np.int64)
    order = np.argsort(-rows, kind="stable")
    for x in order:
        free = room > 0
        g = np.flatnonzero(free)[np.argmin(load[free])]
        group_of[x], load[g], room[g] = g, load[g] + rows[x], room[g] - 1
    rank_of = np.empty(e, np.int64)
    rank_load = np.zeros(ep_size, np.int64)
    rank_room = np.full(ep_size, per_rank)
    for x in order:
        ranks = group_of[x] * ep_tp_size + np.arange(ep_tp_size)
        ranks = ranks[rank_room[ranks] > 0]
        rank_of[x] = ranks[np.argmin(rank_load[ranks])]
        rank_load[rank_of[x]] += rows[x]
        rank_room[rank_of[x]] -= 1
    slot = np.empty(e, np.int64)
    for r in range(ep_size):
        mine = np.flatnonzero(rank_of == r)
        slot[mine] = r * per_rank + np.arange(per_rank)
    return slot


def table_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """A TABLE_FORMAT placement from a STATS_FORMAT dump: LPT per layer."""
    if stats.get("format") != STATS_FORMAT:
        raise ValueError(f"not a {STATS_FORMAT} dump: {stats.get('format')!r}")
    ep, tp = int(stats["ep_size"]), int(stats["ep_tp_size"])
    return dict(
        format=TABLE_FORMAT,
        num_experts=int(stats["num_experts"]),
        ep_size=ep,
        ep_tp_size=tp,
        layers={
            layer: lpt_slots(np.asarray(rows), ep, tp).tolist()
            for layer, rows in stats["layers"].items()
        },
    )


def write_json(path: str, payload: dict[str, Any]) -> None:
    """Atomic write: concurrent writers of the same payload leave one copy."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="LPT placement table from a load dump")
    p.add_argument("stats", help=f"a {STATS_FORMAT} file")
    p.add_argument("table", help=f"the {TABLE_FORMAT} file to write")
    a = p.parse_args(argv)
    with open(a.stats) as f:
        write_json(a.table, table_from_stats(json.load(f)))


if __name__ == "__main__":
    main()
