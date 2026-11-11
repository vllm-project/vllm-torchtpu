# SPDX-License-Identifier: Apache-2.0
"""Per-device tile-cost model of the fused-EP v2 MoE kernel, for sizing a
static expert layout from tapped routing counts.

    kernel_us = K + max_d [ v*visits_d + tau*tiles_d + d64*n64_d + d96*n96_d
                            + d128*n128_d + rho*rows_d ]

An expert's rows are aligned up to ROWBLK (2) per (expert, source rank) run;
tiles = ceil(rows / 128), full tiles run at height 128 and the tail tile at the
smallest rung in (32, 64, 96, 128) that covers it. PARAMS were fitted to 21
kernel-mode slopes of the isolated harness at the DSv4.1 C384 decode shape
(8 EP devices, 48 local experts, 2304 global rows, top-6; RMS error 4.4 us at
00fae88d6). The per-device cost is a sum over the experts the device holds, so
the same per-expert law prices any EP width and slot count; only `K` and the
rung extras are shape-specific, and they cancel in a comparison of layouts.
"""

import numpy as np

TILE, ROWBLK = 128, 2
RUNGS = (32, 64, 96, 128)
# K, v, tau, d64, d96, d128, rho
PARAMS = np.array([75.348, -0.670, 6.432, 0.867, 2.369, 2.546, 0.003])


def rows_from_counts(counts):
    """[experts] ROWBLK-aligned rows from counts [sources, experts]."""
    counts = np.asarray(counts)
    return ((counts + ROWBLK - 1) // ROWBLK * ROWBLK).sum(0)


def tiles(n):
    """The tile heights an expert with n rows runs."""
    full, tail = divmod(int(n), TILE)
    return [TILE] * full + ([next(r for r in RUNGS if r >= tail)] if tail else [])


def expert_cost(n, p=PARAMS):
    """One slot's cost (us, without K) at n rows; zero rows are never visited."""
    if n <= 0:
        return 0.0
    rungs = tiles(n)
    extra = {32: 0.0, 64: p[3], 96: p[4], 128: p[5]}
    return p[1] + len(rungs) * p[2] + sum(extra[r] for r in rungs) + p[6] * n


def device_costs(rows_e, ranks, replicas=None, p=PARAMS):
    """Per-device cost (us, without K) and tile passes under a layout.

    rows_e [experts]: this step's rows per logical expert. ranks: per rank,
    the logical experts it holds. replicas: per expert, how many copies the
    layout holds (rows split evenly across them); None means one each.
    """
    rows_e = np.asarray(rows_e, dtype=np.float64)
    experts = len(rows_e)
    if replicas is None:
        replicas = np.ones(experts, dtype=np.int64)
    cost = np.zeros(len(ranks))
    passes = np.zeros(len(ranks), dtype=np.int64)
    for d, held in enumerate(ranks):
        for e in held:
            n = rows_e[e] / replicas[e]
            # Rows split across copies are still aligned per copy.
            n = int(np.ceil(n / ROWBLK) * ROWBLK)
            cost[d] += expert_cost(n, p)
            passes[d] += len(tiles(n))
    return cost, passes


def contiguous_ranks(experts, shards):
    """The checkpoint's linear blocks: expert e on rank e // (experts/shards)."""
    g = experts // shards
    return [list(range(r * g, (r + 1) * g)) for r in range(shards)]


def replica_counts(ranks, experts):
    counts = np.zeros(experts, dtype=np.int64)
    for held in ranks:
        for e in held:
            counts[e] += 1
    return counts


def predict(rows_e, ranks, replicas=None, p=PARAMS):
    """(slowest device us, mean device us, slowest passes, mean passes), K included."""
    cost, passes = device_costs(rows_e, ranks, replicas, p)
    return (
        p[0] + cost.max(),
        p[0] + cost.mean(),
        int(passes.max()),
        float(passes.mean()),
    )
