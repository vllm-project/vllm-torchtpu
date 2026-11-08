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
expert onto the rank with the smallest count so far that has a free slot,
the same greedy packing as vLLM EPLB's ``balanced_packing``.

A placement file (.npz) holds ``slots`` [layers, experts] (int32, each row a
permutation of the expert ids), ``layer_names`` [layers] (the MoE layers the
rows apply to) and ``ep_size`` (the EP size the rows were balanced for).
Build one from your own traffic:

1. Serve representative traffic for a while with
   ``TPU_MOE_ROUTING_COUNTS_DIR=DIR`` on local disk (setting or unsetting it
   restarts the server). Each DP engine and pipeline stage rewrites a
   ``DIR/routing-counts-*.npz`` file with ``counts`` [layers, experts] (its
   real tokens' routes to each logical expert since start) and
   ``layer_names`` [layers], nothing else. Use one DIR per model version
   and, with disaggregated prefill and decode, one per role; not with PCP.
2. With every host's files copied into one DIR, balance them for the EP
   size (DP x TP) that will serve the placement (numpy only):

       python3 src/vllm_torchtpu/layers/core/expert_placement.py DIR OUT \
           --ep-size 32

3. Serve with ``TPU_MOE_EXPERT_PLACEMENT=OUT`` on every host of that EP
   size. The ragged all-to-all capacities then default to x2
   (``TPU_MOE_A2A_CAPACITY``, ``TPU_MOE_A2A_ROUTE_CAPACITY``).
"""

import argparse
import glob
import os

import numpy as np

# Routing counts files: <TPU_MOE_ROUTING_COUNTS_DIR>/routing-counts-*.npz.
COUNTS_PREFIX = "routing-counts-"


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


def save_counts(path: str, counts: np.ndarray, layer_names) -> None:
    """Write a routing counts file in one step: a reader, or a restart after
    a host crash, finds the previous file or this one, never a partial one."""
    tmp = path + ".tmp"
    # A file object: np.savez would append ".npz" to the temporary name.
    with open(tmp, "wb") as f:
        np.savez(
            f,
            counts=np.asarray(counts, dtype=np.int64),
            layer_names=np.asarray(list(layer_names)),
        )
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_counts(paths) -> tuple[np.ndarray, list[str]]:
    """Routing counts [layers, experts] and their layer names, summed by name
    over routing counts files and directories of them."""
    files = []
    for path in paths:
        if not os.path.isdir(path):
            files.append(path)
            continue
        found = sorted(glob.glob(os.path.join(path, COUNTS_PREFIX + "*.npz")))
        if not found:
            raise ValueError(f"{path}: no {COUNTS_PREFIX}*.npz files")
        files += found
    totals: dict[str, np.ndarray] = {}
    for file in files:
        try:
            with np.load(file) as data:
                names, rows = list(data["layer_names"]), data["counts"]
        except Exception as e:
            raise ValueError(f"{file}: not a routing counts file ({e})") from e
        for name, row in zip(names, rows):
            row = np.asarray(row, dtype=np.int64)
            total = totals.setdefault(str(name), np.zeros_like(row))
            if total.shape != row.shape:
                raise ValueError(
                    f"{name}: {row.size} experts in {file}, {total.size} before"
                )
            total += row
    names = list(totals)
    return np.stack([totals[name] for name in names]), names


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "counts",
        nargs="+",
        help="TPU_MOE_ROUTING_COUNTS_DIR directories or routing counts files",
    )
    parser.add_argument("out", help="placement file to write (.npz)")
    parser.add_argument("--ep-size", type=int, required=True)
    args = parser.parse_args(argv)
    counts, names = load_counts(args.counts)
    # Every token routes through every MoE layer: a layer without counts was
    # not captured.
    empty = [name for name, row in zip(names, counts) if not row.any()]
    if empty:
        raise ValueError(
            f"no routes counted for {len(empty)} of {len(names)} layers "
            f"({empty[0]} first)"
        )
    slots = np.stack([balance(row, args.ep_size) for row in counts])
    save(args.out, slots, names, args.ep_size)
    local = counts.shape[1] // args.ep_size
    for name, row, placed in zip(names, counts, slots):
        per_rank = np.bincount(placed // local, weights=row, minlength=args.ep_size)
        linear = row.reshape(args.ep_size, local).sum(1)
        mean = per_rank.mean() or 1.0
        print(
            f"{name}: busiest rank / mean routes {linear.max() / mean:.3f} "
            f"linear -> {per_rank.max() / mean:.3f} balanced"
        )
    print(f"wrote {args.out}: {len(names)} layers, EP {args.ep_size}")


if __name__ == "__main__":
    main()
