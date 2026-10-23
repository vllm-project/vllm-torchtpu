# SPDX-License-Identifier: Apache-2.0
"""Debug-only QSA top-k statistics from a host callback (`VLLM_TPU_QSA_TOPK_STATS`).

Per selection: the mean run length of each row's selected compressed groups
(consecutive group ids form one run), and for requests with exactly four rows
(MTP K=3 verify) the overlap of adjacent rows and the union size relative to the
sum of the rows' selections (what fetching a request's union once would save).
"""

import jax
import numpy as np

_acc = {"calls": 0, "rows": 0, "runs": 0, "picks": 0, "pairs": 0, "shared": 0, "union": 0, "sum4": 0}


def _record(blocks, lengths, starts, every):
    blocks, lengths, starts = np.asarray(blocks), np.asarray(lengths), np.asarray(starts)
    a = _acc
    a["calls"] += 1
    sets = []
    for row, length in zip(blocks, lengths):
        g = np.unique(row[row >= 0]) if length > 0 else np.empty(0, row.dtype)
        sets.append(g)
        if len(g):
            a["rows"] += 1
            a["picks"] += len(g)
            a["runs"] += 1 + int(np.count_nonzero(np.diff(g) != 1))
    for r in range(len(starts) - 1):
        lo, hi = int(starts[r]), int(starts[r + 1])
        if hi - lo != 4 or hi > len(sets) or not all(len(sets[i]) for i in range(lo, hi)):
            continue
        for i in range(lo, hi - 1):
            a["pairs"] += len(sets[i])
            a["shared"] += len(np.intersect1d(sets[i], sets[i + 1]))
        a["union"] += len(np.unique(np.concatenate(sets[lo:hi])))
        a["sum4"] += sum(len(sets[i]) for i in range(lo, hi))
    if a["calls"] % every == 0 and a["rows"]:
        print(
            "qsa topk stats calls=%d rows=%d mean_run=%.2f adj_overlap=%.3f union_over_sum=%.3f"
            % (
                a["calls"],
                a["rows"],
                a["picks"] / max(a["runs"], 1),
                a["shared"] / max(a["pairs"], 1),
                a["union"] / max(a["sum4"], 1),
            ),
            flush=True,
        )


def record(blocks, lengths, starts, every: int) -> None:
    jax.debug.callback(lambda b, l, s: _record(b, l, s, every), blocks, lengths, starts)
