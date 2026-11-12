# SPDX-License-Identifier: Apache-2.0
"""Build an EP16 expert placement plan (``KernelConfig.expert_plan``)
from the prefill's MoE route counts.

    python -m tools.megakernel.expert_plan --counts DIR[:WEIGHT] ... --out PLAN.npz

Each ``DIR`` holds the snapshots of one run with ``prefill_route_counts`` on
(``prefill-rNN-wW.npy``, saved by each chip's even rank: chip ``NN // 2``'s
cumulative ``[92, 56]`` route counts of its local experts); the latest
snapshot of every chip is used, and ``--counted-plan`` names the placement the
run counted under (default: the pre-shard's, chip c holding experts
56c..56c+55).

Each run is one token distribution (e.g. random token ids, sequential id
sweeps, real text): their hot experts differ, so a placement balanced for one
can leave another's busiest chip with half again the mean load. Per MoE
layer, each run's loads are normalized to a mean chip load of one and scaled
by its ``WEIGHT`` (default 1), and the placement minimizes the largest chip
load over all runs: experts are placed in decreasing order of their largest
load, each on the open chip (56 slots) where the resulting largest load is
smallest, then pairs of experts on the busiest chip and another are swapped
while a swap lowers it. Each chip's experts are ordered hottest first (summed
over runs), so its slot 0 holds its hottest expert. The plan's
``order[layer, 56 c + j]`` is the logical expert in chip c's slot j.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

LAYERS, EXPERTS, CHIPS = 92, 896, 16
LOCAL = EXPERTS // CHIPS
SNAPSHOT = re.compile(r"prefill-r(\d+)-w(\d+)\.npy$")


def load_counts(directory: Path, counted_plan: np.ndarray) -> np.ndarray:
    """``[LAYERS, EXPERTS]`` route counts per logical expert of one run."""
    latest: dict[int, tuple[int, Path]] = {}
    for path in directory.iterdir():
        match = SNAPSHOT.search(path.name)
        if not match:
            continue
        chip, window = int(match.group(1)) // 2, int(match.group(2))
        if chip not in latest or window > latest[chip][0]:
            latest[chip] = (window, path)
    if sorted(latest) != list(range(CHIPS)):
        found = sorted(latest)
        raise SystemExit(f"{directory}: snapshots of chips {found}, need 0..15")
    windows = {window for window, _ in latest.values()}
    if len(windows) != 1:
        raise SystemExit(f"{directory}: chips' latest snapshots differ ({windows})")
    counts = np.zeros((LAYERS, EXPERTS), np.float64)
    for chip, (_, path) in latest.items():
        local = np.load(path)
        if local.shape != (LAYERS, LOCAL):
            raise SystemExit(f"{path}: shape {local.shape}, expected {(LAYERS, LOCAL)}")
        slots = counted_plan[:, chip * LOCAL : (chip + 1) * LOCAL]
        np.add.at(counts, (np.arange(LAYERS)[:, None], slots), local)
    return counts


def _place_layer(loads: np.ndarray, max_swaps: int) -> list[np.ndarray]:
    """Each chip's experts for one layer's normalized loads ``[EXPERTS, runs]``."""
    members: list[list[int]] = [[] for _ in range(CHIPS)]
    total = np.zeros((CHIPS, loads.shape[1]))
    # Largest load first; ties by expert id, so the plan is reproducible.
    for expert in np.lexsort((np.arange(EXPERTS), -loads.max(axis=1))):
        open_chips = [c for c in range(CHIPS) if len(members[c]) < LOCAL]
        chip = min(open_chips, key=lambda c: ((total[c] + loads[expert]).max(), c))
        members[chip].append(int(expert))
        total[chip] += loads[expert]
    chips = [np.array(experts) for experts in members]
    for _ in range(max_swaps):
        peak = total.max(axis=1)
        busiest = int(np.argmax(peak))
        best, swap = peak[busiest], None
        ours = loads[chips[busiest]]  # [LOCAL, runs]
        for other in range(CHIPS):
            if other == busiest:
                continue
            theirs = loads[chips[other]]
            delta = theirs[None, :, :] - ours[:, None, :]  # [ours, theirs, runs]
            after = np.maximum(
                (total[busiest] + delta).max(axis=2), (total[other] - delta).max(axis=2)
            )
            rest = np.delete(peak, [busiest, other]).max()
            i, j = np.unravel_index(np.argmin(after), after.shape)
            if max(after[i, j], rest) < best - 1e-12:
                best, swap = max(after[i, j], rest), (other, i, j)
        if swap is None:
            break
        other, i, j = swap
        a, b = chips[busiest][i], chips[other][j]
        chips[busiest][i], chips[other][j] = b, a
        total[busiest] += loads[b] - loads[a]
        total[other] += loads[a] - loads[b]
    return chips


def place(runs: list[np.ndarray], max_swaps: int = 400) -> np.ndarray:
    """The plan ``order`` [LAYERS, EXPERTS] for the runs' per-expert loads
    (each ``[LAYERS, EXPERTS]``, already weighted)."""
    order = np.empty((LAYERS, EXPERTS), np.int32)
    for layer in range(LAYERS):
        loads = np.stack(
            [CHIPS * run[layer] / max(run[layer].sum(), 1e-30) for run in runs],
            axis=1,
        )
        hot = loads.sum(axis=1)
        for chip, experts in enumerate(_place_layer(loads, max_swaps)):
            ranked = experts[np.lexsort((experts, -hot[experts]))]
            order[layer, chip * LOCAL : (chip + 1) * LOCAL] = ranked
    return order


def imbalance(counts: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Per layer, the most loaded chip's routes over the mean chip's."""
    loads = np.take_along_axis(counts, order, axis=1).reshape(LAYERS, CHIPS, LOCAL)
    per_chip = loads.sum(axis=2)
    return per_chip.max(axis=1) / np.maximum(per_chip.mean(axis=1), 1e-30)


def identity() -> np.ndarray:
    return np.broadcast_to(np.arange(EXPERTS, dtype=np.int32), (LAYERS, EXPERTS)).copy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--counts", nargs="+", required=True, metavar="DIR[:WEIGHT]")
    parser.add_argument("--counted-plan", default="", help="npz the runs counted under")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    counted = identity()
    if args.counted_plan:
        counted = np.load(args.counted_plan)["order"].astype(np.int64)
    runs = []
    for spec in args.counts:
        directory, _, weight = spec.partition(":")
        counts = load_counts(Path(directory), counted)
        runs.append((directory, float(weight or 1) * counts))
    order = place([counts for _, counts in runs])
    for layer in range(LAYERS):
        assert sorted(order[layer]) == list(range(EXPERTS))
    np.savez(args.out, order=order)
    for directory, counts in runs:
        before, after = imbalance(counts, identity()), imbalance(counts, order)
        print(
            f"{directory}: max/mean chip load, pre-shard placement mean "
            f"{before.mean():.4f} worst {before.max():.4f}; plan mean "
            f"{after.mean():.4f} worst {after.max():.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
