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
"""Static expert placement for expert parallelism, balanced on routing counts.

Linear placement puts expert ``e`` on EP rank ``e // local_num_experts``.
Real routing sends many more routes to some experts than to others, and the
EP rank holding the busiest experts sets the pace of every MoE layer: the
others wait for it in the exchange and in the TP all-reduce.

A placement maps each logical expert to a physical slot; slot ``s`` lives on
EP rank ``s // local_num_experts``, so the expert parallel layout in slot
space is still linear. ``balance`` fills the ranks largest count first, each
expert onto the rank with the smallest count so far that has a free slot.

A single busiest expert still sets the floor when a rank holds few experts.
``replicate`` adds slots to every rank and gives them copies of the busiest
experts (one copy at a time to the expert with the largest count per copy),
then places every copy like ``balance``, never two copies of an expert on one
rank. A route to an expert with ``c`` copies goes to one copy chosen by a
hash of the route, so the copies split the expert's rows.

A placement file (.npz) holds ``slots`` [layers, experts] (int32, each row a
permutation of the expert ids) or, with copies, [layers, experts, copies]
(copy ``j`` of each expert, -1 past its last copy) plus ``slots_per_rank``;
``layer_names`` [layers] (the MoE layers the rows apply to) and ``ep_size``
(the EP size the rows were balanced for). Build one from routing counts with
(numpy only):

    python3 src/vllm_torchtpu/layers/core/expert_permutation.py ROUTES OUT \
        --ep-size 32 [--extra-slots 1]

where ROUTES is an .npz with ``counts`` [layers, experts] or captured ``ids``
[layers, ..., topk] (negative ids are ignored).
"""

import argparse

import numpy as np

DEFAULT_LAYER_NAME = "model.layers.{}.mlp.experts"


def balance(counts: np.ndarray, ep_size: int) -> np.ndarray:
    """Slot of each logical expert [E] (int32) for one layer's ``counts`` [E].

    Each rank gets ``E // ep_size`` experts, kept in logical order within the
    rank. Ties go to the lower expert id and the lower rank.
    """
    counts = np.asarray(counts, dtype=np.float64)
    num_experts = counts.shape[0]
    if num_experts % ep_size:
        raise ValueError(f"{num_experts} experts do not split over {ep_size} ranks")
    local = num_experts // ep_size
    load = np.zeros(ep_size)
    free = np.full(ep_size, local)
    rank = np.empty(num_experts, dtype=np.int64)
    for e in np.argsort(-counts, kind="stable"):
        open_ranks = np.flatnonzero(free)
        r = open_ranks[np.argmin(load[open_ranks])]
        rank[e] = r
        load[r] += counts[e]
        free[r] -= 1
    slots = np.empty(num_experts, dtype=np.int32)
    for r in range(ep_size):
        experts = np.flatnonzero(rank == r)
        slots[experts] = r * local + np.arange(local)
    return slots


def replicate(
    counts: np.ndarray, ep_size: int, extra_slots: int, copies: int | None = None
) -> np.ndarray:
    """Slots of each logical expert's copies [E, C] (int32, -1 past the last
    copy) for one layer's ``counts`` [E], with ``extra_slots`` more slots per
    rank than ``E // ep_size``.

    ``copies`` extra copies (default: one per added slot) go one at a time to
    the expert with the largest count per copy, at most one copy per rank.
    Copies are placed largest count per copy first onto the rank with the
    smallest count so far that has a free slot and no copy of that expert.
    Each rank keeps its experts in logical order. Ties go to the lower expert
    id and the lower rank. With no extra slot this is ``balance``.
    """
    counts = np.asarray(counts, dtype=np.float64)
    num_experts = counts.shape[0]
    if num_experts % ep_size:
        raise ValueError(f"{num_experts} experts do not split over {ep_size} ranks")
    local = num_experts // ep_size + extra_slots
    spare = ep_size * extra_slots
    spare = spare if copies is None else min(copies, spare)
    num_copies = np.ones(num_experts, dtype=np.int64)
    for _ in range(spare):
        per_copy = np.where(num_copies < ep_size, counts / num_copies, -1.0)
        e = int(np.argmax(per_copy))
        if per_copy[e] <= 0:
            break
        num_copies[e] += 1
    items = sorted(
        (
            (counts[e] / num_copies[e], e)
            for e in range(num_experts)
            for _ in range(num_copies[e])
        ),
        key=lambda item: (-item[0], item[1]),
    )
    load = np.zeros(ep_size)
    free = np.full(ep_size, local)
    holds = np.zeros((ep_size, num_experts), dtype=bool)
    for weight, e in items:
        open_ranks = np.flatnonzero((free > 0) & ~holds[:, e])
        r = open_ranks[np.argmin(load[open_ranks])]
        holds[r, e] = True
        load[r] += weight
        free[r] -= 1
    slots = np.full((num_experts, int(num_copies.max())), -1, dtype=np.int32)
    filled = np.zeros(num_experts, dtype=np.int64)
    for r in range(ep_size):
        for i, e in enumerate(np.flatnonzero(holds[r])):
            slots[e, filled[e]] = r * local + i
            filled[e] += 1
    return slots


def _as_copies(slots: np.ndarray) -> np.ndarray:
    """[layers, experts] -> [layers, experts, 1]; [layers, experts, C] as is."""
    slots = np.asarray(slots)
    return slots[..., None] if slots.ndim == 2 else slots


def validate(
    slots: np.ndarray, layer_names, ep_size: int, slots_per_rank: int | None = None
) -> int:
    """Check a placement; returns the slots per rank."""
    slots = np.asarray(slots)
    if slots.ndim not in (2, 3) or len(layer_names) != slots.shape[0]:
        raise ValueError(
            "slots must be [layers, experts(, copies)] with one name per layer"
        )
    num_experts = slots.shape[1]
    if ep_size <= 0 or num_experts % ep_size:
        raise ValueError(f"{num_experts} experts do not split over {ep_size} ranks")
    local = slots_per_rank or num_experts // ep_size
    if local < num_experts // ep_size:
        raise ValueError(f"{local} slots per rank cannot hold {num_experts} experts")
    if slots.ndim == 2 and local != num_experts // ep_size:
        raise ValueError("slots without copies take E // ep_size slots per rank")
    for name, row in zip(layer_names, _as_copies(slots)):
        present = row >= 0
        used = row[present]
        if (
            not present[:, 0].all()
            or (np.diff(present.astype(np.int8), axis=1) > 0).any()
        ):
            raise ValueError(f"{name}: every expert needs its copies first, then -1")
        if (used >= ep_size * local).any() or len(np.unique(used)) != used.size:
            raise ValueError(f"{name}: slots are not a permutation of the experts")
        ranks = np.where(present, row // local, -1 - np.arange(row.shape[1]))
        if (np.sort(ranks, axis=1)[:, 1:] == np.sort(ranks, axis=1)[:, :-1]).any():
            raise ValueError(f"{name}: two copies of an expert on one rank")
    if len(set(layer_names)) != len(layer_names):
        raise ValueError("layer names repeat")
    return local


def save(
    path: str, slots: np.ndarray, layer_names, ep_size: int, slots_per_rank=None
) -> None:
    local = validate(slots, layer_names, ep_size, slots_per_rank)
    extra = {} if np.asarray(slots).ndim == 2 else {"slots_per_rank": np.int64(local)}
    np.savez(
        path,
        slots=np.asarray(slots, dtype=np.int32),
        layer_names=np.asarray(list(layer_names)),
        ep_size=np.int64(ep_size),
        **extra,
    )


def load(path: str) -> tuple[dict[str, np.ndarray], int, int]:
    """``({layer name: slots [E, C]}, ep_size, slots per rank)`` from a
    placement file (C = 1 without copies)."""
    with np.load(path) as data:
        slots = data["slots"].astype(np.int32)
        names = [str(n) for n in data["layer_names"]]
        ep_size = int(data["ep_size"])
        local = int(data["slots_per_rank"]) if "slots_per_rank" in data else None
    local = validate(slots, names, ep_size, local)
    return dict(zip(names, _as_copies(slots))), ep_size, local


def counts_from_routes(path: str, num_experts: int | None = None) -> np.ndarray:
    """Per-layer routing counts [layers, experts] from ``counts`` or ``ids``."""
    with np.load(path) as data:
        if "counts" in data:
            return np.asarray(data["counts"], dtype=np.int64)
        ids = np.asarray(data["ids"], dtype=np.int64)
    num_experts = num_experts or int(ids.max()) + 1
    flat = ids.reshape(ids.shape[0], -1)
    return np.stack([np.bincount(row[row >= 0], minlength=num_experts) for row in flat])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("routes", help=".npz with counts [L, E] or ids [L, ..., K]")
    parser.add_argument("out", help="placement file to write (.npz)")
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument(
        "--layer-name",
        default=DEFAULT_LAYER_NAME,
        help="MoE layer name with {} for the layer index (default %(default)s)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="write this many layers, layer i balanced on routes layer i %% L",
    )
    parser.add_argument(
        "--extra-slots",
        type=int,
        default=0,
        help="slots per rank beyond E / ep-size, filled with copies of the busiest "
        "experts (default %(default)s)",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=None,
        help="extra copies per layer (default: one per added slot)",
    )
    args = parser.parse_args(argv)
    counts = counts_from_routes(args.routes, args.num_experts)
    num_layers = args.num_layers or counts.shape[0]
    num_experts = counts.shape[1]
    if args.extra_slots:
        rows = [
            replicate(
                counts[i % counts.shape[0]], args.ep_size, args.extra_slots, args.copies
            )
            for i in range(num_layers)
        ]
        width = max(r.shape[1] for r in rows)
        slots = np.full((num_layers, num_experts, width), -1, dtype=np.int32)
        for i, r in enumerate(rows):
            slots[i, :, : r.shape[1]] = r
        local = num_experts // args.ep_size + args.extra_slots
    else:
        slots = np.stack(
            [
                balance(counts[i % counts.shape[0]], args.ep_size)
                for i in range(num_layers)
            ]
        )
        local = num_experts // args.ep_size
    names = [args.layer_name.format(i) for i in range(num_layers)]
    save(args.out, slots, names, args.ep_size, local)
    for i in range(min(num_layers, counts.shape[0])):
        row = _as_copies(slots)[i]
        num_copies = (row >= 0).sum(axis=1)
        per_rank = np.zeros(args.ep_size)
        for e, copies in enumerate(row):
            for slot in copies[copies >= 0]:
                per_rank[slot // local] += counts[i, e] / num_copies[e]
        linear = counts[i].reshape(args.ep_size, -1).sum(1)
        mean = per_rank.mean() or 1.0
        print(
            f"layer {i}: busiest rank / mean routes {linear.max() / mean:.3f} "
            f"linear -> {per_rank.max() / mean:.3f} placed "
            f"({int(num_copies.sum()) - num_experts} copies, at most "
            f"{num_copies.max()} per expert)"
        )
    print(
        f"wrote {args.out}: {num_layers} layers, EP {args.ep_size}, "
        f"{local} slots per rank"
    )


if __name__ == "__main__":
    main()
