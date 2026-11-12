# SPDX-License-Identifier: Apache-2.0
"""Build an EP16 expert placement plan (``KernelConfig.prefill_expert_plan``)
from the prefill's MoE route counts.

    python -m tools.megakernel.expert_plan --counts DIR[:WEIGHT] ... --out PLAN.npz

Each ``DIR`` holds the snapshots of one run with ``prefill_route_counts`` on
(``prefill-rNN-wW.npy``, saved by each chip's even rank: chip ``NN // 2``'s
cumulative ``[92, 56]`` route counts of its local experts); the latest
snapshot of every chip is used, ``--counted-plan`` names the placement the run
counted under (default: the pre-shard's, chip c holding experts 56c..56c+55),
and the runs' counts add, each normalized to unit total and scaled by its
``WEIGHT`` (default 1).

Per MoE layer, longest-processing-time places the experts, hottest first, on
the least loaded of the 16 chips with a free slot (56 each); each chip's
experts are then ordered hottest first, so its slot 0 holds its hottest
expert. The plan's ``order[layer, 56 c + j]`` is the logical expert in chip
c's slot j.
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


def place(counts: np.ndarray) -> np.ndarray:
    """The plan ``order`` [LAYERS, EXPERTS] for per-expert loads ``counts``."""
    order = np.empty((LAYERS, EXPERTS), np.int32)
    for layer in range(LAYERS):
        load = np.zeros(CHIPS)
        members: list[list[int]] = [[] for _ in range(CHIPS)]
        # Hottest first; ties by expert id, so the plan is reproducible.
        for expert in np.lexsort((np.arange(EXPERTS), -counts[layer])):
            open_chips = [c for c in range(CHIPS) if len(members[c]) < LOCAL]
            chip = min(open_chips, key=lambda c: (load[c], c))
            members[chip].append(int(expert))
            load[chip] += counts[layer, expert]
        for chip, experts in enumerate(members):
            # Placed hottest first already: slot 0 is the chip's hottest.
            order[layer, chip * LOCAL : (chip + 1) * LOCAL] = experts
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
    total = np.zeros((LAYERS, EXPERTS))
    for spec in args.counts:
        directory, _, weight = spec.partition(":")
        counts = load_counts(Path(directory), counted)
        total += float(weight or 1) * counts / counts.sum()
    order = place(total)
    for layer in range(LAYERS):
        assert sorted(order[layer]) == list(range(EXPERTS))
    np.savez(args.out, order=order)
    before, after = imbalance(total, identity()), imbalance(total, order)
    print(
        f"{args.out}: max/mean chip load, pre-shard placement mean "
        f"{before.mean():.4f} worst {before.max():.4f}; plan mean "
        f"{after.mean():.4f} worst {after.max():.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
