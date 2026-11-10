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
   size. The ragged all-to-all capacities then default to x2 up to 8 DP
   members, x4 up to 16 and x7 above
   (``TPU_MOE_A2A_CAPACITY``, ``TPU_MOE_A2A_ROUTE_CAPACITY``).

Redundant experts. A balanced placement still leaves the busiest experts on
their ranks: one expert can carry several times the mean count. With
``--redundant-per-rank R`` every rank gets ``R`` redundant slots per layer
that hold replicas of the busiest experts (``replicate``, ``place_replicas``:
the replication and largest-first packing of vLLM EPLB's policy, adapted from
DeepSeek EPLB, around the fixed primaries). The file then also holds
``redundant`` [layers, ep_size * R] (int32): the logical expert in redundant
slot ``q`` (on EP rank ``q // R``), -1 where empty. The routes of a
replicated expert are split over its replicas; outputs are those of the same
file without redundant slots. ``--slots-from FILE`` keeps that file's slots
and plans replicas from new counts. Redundant slots cost HBM (R expert slots
per layer and rank) and are served by the ragged all-to-all exchange with
its owner combine and sorted plan (decode_switches paths a2a_owner_combine and
a2a_sorted_plan).
"""

import argparse
import glob
import os

import numpy as np

# Routing counts files: <TPU_MOE_ROUTING_COUNTS_DIR>/routing-counts-*.npz.
COUNTS_PREFIX = "routing-counts-"
# A replica must carry at least this share of the mean expert count.
REPLICA_GUARD = 0.7
# With replicas, the routing all-gather packs a route's slot and owner order
# (these low bits, topk <= 16) into the 16 bits above its bf16 weight.
ROUTE_ORDER_BITS = 4


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


def replicate(counts: np.ndarray, extra: int, max_replicas: int = 0) -> np.ndarray:
    """Replicas of each logical expert [E] (int64, >= 1) for ``extra`` more
    slots of one layer: each goes to the expert with the largest count per
    replica (ties to the lower id), and none to a replica that would carry
    less than ``REPLICA_GUARD`` times the mean count, so near-uniform counts
    get no replicas. ``max_replicas`` > 0: no expert gets more than that many
    besides its primary."""
    counts = np.asarray(counts, dtype=np.float64)
    replicas = np.ones(counts.shape[0], dtype=np.int64)
    floor = REPLICA_GUARD * counts.mean()
    for _ in range(extra):
        share = counts / replicas
        if max_replicas > 0:
            share = np.where(replicas > max_replicas, -np.inf, share)
        e = int(np.argmax(share))
        if share[e] == -np.inf or counts[e] / (replicas[e] + 1) < floor:
            break
        replicas[e] += 1
    return replicas


def place_replicas(
    counts: np.ndarray,
    slots: np.ndarray,
    ep_size: int,
    redundant_per_rank: int,
    replicas: np.ndarray,
) -> np.ndarray:
    """Redundant slots [ep_size * redundant_per_rank] (int32, the logical
    expert each holds, -1 = empty) for one layer's fixed primary ``slots``
    [E] and ``replicas`` [E] per expert.

    Each rank starts with its primaries' loads (count per replica); replicas
    go heaviest first (ties to the lower id) onto the least loaded rank (ties
    to the lower rank) with a free redundant slot and no replica of that
    expert. A replica no rank can take is dropped.
    """
    counts = np.asarray(counts, dtype=np.float64)
    num_experts = counts.shape[0]
    rank = np.asarray(slots, dtype=np.int64) // (num_experts // ep_size)
    share = counts / replicas
    load = np.bincount(rank, weights=share, minlength=ep_size)
    holds = np.zeros((ep_size, num_experts), dtype=bool)
    holds[rank, np.arange(num_experts)] = True
    filled = np.zeros(ep_size, dtype=np.int64)
    redundant = np.full(ep_size * redundant_per_rank, -1, dtype=np.int32)
    items = [(e, share[e]) for e in range(num_experts) for _ in range(replicas[e] - 1)]
    for e, w in sorted(items, key=lambda item: (-item[1], item[0])):
        open_ranks = np.flatnonzero((filled < redundant_per_rank) & ~holds[:, e])
        if not open_ranks.size:
            continue
        r = open_ranks[np.argmin(load[open_ranks])]
        redundant[r * redundant_per_rank + filled[r]] = e
        filled[r] += 1
        load[r] += w
        holds[r, e] = True
    return redundant


def physical_slots(
    slots: np.ndarray, ep_size: int, redundant_per_rank: int
) -> np.ndarray:
    """File slots -> physical slots with ``redundant_per_rank`` (R)
    redundant slots after each rank's ``L`` primaries: rank ``r`` holds
    slots ``r * (L + R)`` to ``(r + 1) * (L + R) - 1``, redundant ones last."""
    slots = np.asarray(slots)
    local = slots.shape[-1] // ep_size
    return (slots // local) * (local + redundant_per_rank) + slots % local


def replica_table(slots: np.ndarray, redundant: np.ndarray, ep_size: int) -> np.ndarray:
    """One layer's redundant slots [3, ep_size * R] (int32) in physical slots:
    per redundant slot, its own slot, its replica's primary slot (-1 where
    empty) and its replica index (1, 2, ... in slot order among the expert's
    replicas; 0 where empty)."""
    redundant = np.asarray(redundant, dtype=np.int64)
    redundant_per_rank = redundant.shape[0] // ep_size
    local = np.asarray(slots).shape[0] // ep_size
    q = np.arange(redundant.shape[0])
    own = (q // redundant_per_rank) * (local + redundant_per_rank) + local
    own += q % redundant_per_rank
    primary = physical_slots(slots, ep_size, redundant_per_rank)
    filled = redundant >= 0
    replica = np.zeros_like(q)
    seen: dict[int, int] = {}
    for i in np.flatnonzero(filled):
        seen[redundant[i]] = seen.get(redundant[i], 0) + 1
        replica[i] = seen[redundant[i]]
    table = np.stack(
        [own, np.where(filled, primary[np.maximum(redundant, 0)], -1), replica]
    )
    return table.astype(np.int32)


def slot_checksums(*arrays):
    """[slots, 2 * len(arrays)] int32, in JAX: per slot (leading axis) of each
    array, the wrapping sum of its words, plain and weighted by their (odd)
    position, so equal sums mean equal bytes with near certainty. 4-byte
    values are read as int32 words, 1-byte values (fp4 two per byte) as
    bytes. JAX is imported here, so the builder runs on numpy alone."""
    import jax
    import jax.numpy as jnp

    sums = []
    for a in arrays:
        num_slots = a.shape[0]
        if jnp.dtype(a.dtype).itemsize == 4:
            words = jax.lax.bitcast_convert_type(a, jnp.int32)
        else:
            if a.dtype == jnp.float4_e2m1fn:
                a = a.reshape(*a.shape[:-1], a.shape[-1] // 2, 2)
            words = jax.lax.bitcast_convert_type(a, jnp.uint8).astype(jnp.int32)
        words = words.reshape(num_slots, -1)
        weight = jnp.arange(words.shape[1], dtype=jnp.int32) * 2 + 1
        sums += [words.sum(axis=1), (words * weight).sum(axis=1)]
    return jnp.stack(sums, axis=1)


def validate(slots: np.ndarray, layer_names, ep_size: int, redundant=None) -> None:
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
    if redundant is None:
        return
    redundant = np.asarray(redundant)
    if (
        redundant.ndim != 2
        or redundant.shape[0] != slots.shape[0]
        or not redundant.shape[1]
        or redundant.shape[1] % ep_size
    ):
        raise ValueError(
            f"redundant must be [layers, ep_size * R] with R >= 1, got "
            f"{redundant.shape} for {slots.shape[0]} layers over EP {ep_size}"
        )
    redundant_per_rank = redundant.shape[1] // ep_size
    rank_slots = ep_size * (num_experts // ep_size + redundant_per_rank)
    if rank_slots << ROUTE_ORDER_BITS >= 1 << 15:
        raise ValueError(
            f"{rank_slots} expert slots over EP {ep_size} do not fit the packed "
            f"routing (below {1 << (15 - ROUTE_ORDER_BITS)})"
        )
    if redundant.min() < -1 or redundant.max() >= num_experts:
        raise ValueError(f"redundant must hold expert ids below {num_experts} or -1")
    rank_of_slot = np.arange(redundant.shape[1]) // redundant_per_rank
    for name, row, replicated in zip(layer_names, slots, redundant):
        filled = replicated >= 0
        held = np.concatenate([row // (num_experts // ep_size), rank_of_slot[filled]])
        experts = np.concatenate([np.arange(num_experts), replicated[filled]])
        if np.unique(held * num_experts + experts).size != held.size:
            raise ValueError(f"{name}: an EP rank holds an expert twice")


def save(
    path: str, slots: np.ndarray, layer_names, ep_size: int, redundant=None
) -> None:
    validate(slots, layer_names, ep_size, redundant)
    arrays = {}
    if redundant is not None:
        arrays["redundant"] = np.asarray(redundant, dtype=np.int32)
    np.savez(
        path,
        slots=np.asarray(slots, dtype=np.int32),
        layer_names=np.asarray(list(layer_names)),
        ep_size=np.int64(ep_size),
        **arrays,
    )


def load(path: str) -> tuple[dict[str, np.ndarray], int]:
    """``({layer name: slots [E]}, ep_size)`` from a placement file."""
    with np.load(path) as data:
        slots = data["slots"].astype(np.int32)
        names = [str(n) for n in data["layer_names"]]
        ep_size = int(data["ep_size"])
    validate(slots, names, ep_size)
    return dict(zip(names, slots)), ep_size


def load_redundant(path: str) -> tuple[dict[str, np.ndarray], int]:
    """``({layer name: redundant [ep_size * R]}, R)`` from a placement file;
    ``({}, 0)`` for a file without redundant slots."""
    with np.load(path) as data:
        if "redundant" not in data.files:
            return {}, 0
        redundant = data["redundant"].astype(np.int32)
        slots = data["slots"].astype(np.int32)
        names = [str(n) for n in data["layer_names"]]
        ep_size = int(data["ep_size"])
    validate(slots, names, ep_size, redundant)
    return dict(zip(names, redundant)), redundant.shape[1] // ep_size


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
    parser.add_argument(
        "--redundant-per-rank",
        type=int,
        default=0,
        help="redundant expert slots per EP rank and layer, for replicas of the "
        "busiest experts (0: none)",
    )
    parser.add_argument(
        "--max-replicas",
        type=int,
        default=0,
        help="replicas per expert at most, besides its primary (0: no limit; "
        "the plan kernel's balanced replica choice takes at most 7)",
    )
    parser.add_argument(
        "--slots-from",
        help="placement file whose slots to keep (replicas come from the counts)",
    )
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
    if args.redundant_per_rank < 0 or args.max_replicas < 0:
        raise ValueError("--redundant-per-rank and --max-replicas must be >= 0")
    if args.slots_from:
        kept, kept_ep_size = load(args.slots_from)
        missing = [name for name in names if name not in kept]
        if kept_ep_size != args.ep_size or missing:
            raise ValueError(
                f"{args.slots_from}: EP {kept_ep_size}, "
                f"{len(missing)} of {len(names)} counted layers missing"
            )
        slots = np.stack([kept[name] for name in names])
    else:
        slots = np.stack([balance(row, args.ep_size) for row in counts])
    redundant = None
    if args.redundant_per_rank:
        extra = args.ep_size * args.redundant_per_rank
        redundant = np.stack(
            [
                place_replicas(
                    row,
                    placed,
                    args.ep_size,
                    args.redundant_per_rank,
                    replicate(row, extra, args.max_replicas),
                )
                for row, placed in zip(counts, slots)
            ]
        )
    save(args.out, slots, names, args.ep_size, redundant)
    local = counts.shape[1] // args.ep_size
    for i, (name, row, placed) in enumerate(zip(names, counts, slots)):
        per_rank = np.bincount(placed // local, weights=row, minlength=args.ep_size)
        linear = row.reshape(args.ep_size, local).sum(1)
        mean = per_rank.mean() or 1.0
        line = (
            f"{name}: busiest rank / mean routes {linear.max() / mean:.3f} "
            f"linear -> {per_rank.max() / mean:.3f} balanced"
        )
        if redundant is not None:
            busiest = _busiest_with_replicas(row, placed, redundant[i], args.ep_size)
            line += f" -> {busiest:.3f} with {int((redundant[i] >= 0).sum())} replicas"
        print(line)
    if redundant is not None:
        used = (redundant >= 0).sum(axis=1)
        print(
            f"replicas: {int(used.sum())} of {redundant.size} redundant slots filled "
            f"(per layer {int(used.min())}-{int(used.max())})"
        )
    print(f"wrote {args.out}: {len(names)} layers, EP {args.ep_size}")


def _busiest_with_replicas(counts, slots, redundant, ep_size: int) -> float:
    """Busiest rank / mean routes when each expert's routes split evenly over
    its replicas."""
    num_experts = counts.shape[0]
    redundant_per_rank = redundant.shape[0] // ep_size
    filled = redundant >= 0
    experts = np.concatenate([np.arange(num_experts), redundant[filled]])
    ranks = np.concatenate(
        [
            np.asarray(slots) // (num_experts // ep_size),
            np.flatnonzero(filled) // redundant_per_rank,
        ]
    )
    replicas = np.bincount(experts, minlength=num_experts)
    share = np.asarray(counts, dtype=np.float64)[experts] / replicas[experts]
    per_rank = np.bincount(ranks, weights=share, minlength=ep_size)
    return per_rank.max() / (per_rank.mean() or 1.0)


if __name__ == "__main__":
    main()
