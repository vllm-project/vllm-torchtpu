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

A placement file (.npz) holds ``slots`` [layers, experts] (int32, each row a
permutation of the expert ids), ``layer_names`` [layers] (the MoE layers the
rows apply to) and ``ep_size`` (the EP size the rows were balanced for).
Build one from routing counts with (numpy only):

    python3 src/vllm_torchtpu/layers/core/expert_placement.py ROUTES OUT \
        --ep-size 32

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


def validate(slots: np.ndarray, layer_names, ep_size: int) -> None:
    slots = np.asarray(slots)
    if slots.ndim != 2 or len(layer_names) != slots.shape[0]:
        raise ValueError("slots must be [layers, experts] with one name per layer")
    num_experts = slots.shape[1]
    if ep_size <= 0 or num_experts % ep_size:
        raise ValueError(f"{num_experts} experts do not split over {ep_size} ranks")
    expected = np.arange(num_experts)
    for name, row in zip(layer_names, slots):
        if not np.array_equal(np.sort(row), expected):
            raise ValueError(f"{name}: slots are not a permutation of the experts")
    if len(set(layer_names)) != len(layer_names):
        raise ValueError("layer names repeat")


def save(path: str, slots: np.ndarray, layer_names, ep_size: int) -> None:
    validate(slots, layer_names, ep_size)
    np.savez(
        path,
        slots=np.asarray(slots, dtype=np.int32),
        layer_names=np.asarray(list(layer_names)),
        ep_size=np.int64(ep_size),
    )


def load(path: str) -> tuple[dict[str, np.ndarray], int]:
    """``({layer name: slots [E]}, ep_size)`` from a placement file."""
    with np.load(path) as data:
        slots = data["slots"].astype(np.int32)
        names = [str(n) for n in data["layer_names"]]
        ep_size = int(data["ep_size"])
    validate(slots, names, ep_size)
    return dict(zip(names, slots)), ep_size


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
    args = parser.parse_args(argv)
    counts = counts_from_routes(args.routes, args.num_experts)
    num_layers = args.num_layers or counts.shape[0]
    slots = np.stack(
        [balance(counts[i % counts.shape[0]], args.ep_size) for i in range(num_layers)]
    )
    names = [args.layer_name.format(i) for i in range(num_layers)]
    save(args.out, slots, names, args.ep_size)
    local = counts.shape[1] // args.ep_size
    for i in range(min(num_layers, counts.shape[0])):
        per_rank = np.bincount(
            slots[i] // local, weights=counts[i], minlength=args.ep_size
        )
        linear = counts[i].reshape(args.ep_size, local).sum(1)
        mean = per_rank.mean() or 1.0
        print(
            f"layer {i}: busiest rank / mean routes {linear.max() / mean:.3f} "
            f"linear -> {per_rank.max() / mean:.3f} balanced"
        )
    print(f"wrote {args.out}: {num_layers} layers, EP {args.ep_size}")


if __name__ == "__main__":
    main()
