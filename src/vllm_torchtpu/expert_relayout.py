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
"""Dynamic expert re-layout (EPLB round 2): solve from the live load window,
stage the moved experts into spare slots, install on every rank in the same
step.

Every rank runs one `Controller`. Its inputs are state every rank shares:
the load tap's accumulators (replicated sums, identical after the same step)
and the coordinated step counter (every rank performs the DP step all-reduce
once per step). So the trigger, the solve and the install are the same
function of the same data on every rank, and the protocol needs no
collective of its own; three int64 fields ride the existing per-step MAX
all-reduce to confirm that and to learn when every rank has staged:

    fields()  -> (plan_hash, -plan_hash, -staged_generation)
    reduced   -> (max hash, -min hash, -min staged)

A generation: trigger at step T0 (every `window_steps`): snapshot the tap,
form the window, solve (policy + rank matching + move budget + hysteresis)
-> a Plan with a 62-bit hash. Step T0 + 1: the hashes agree on every rank
(else the generation is dropped, identically everywhere); the mover starts
copying the moved experts' slabs into the destinations' staging slots in the
background. When the reduced staged generation reaches the plan's on every
rank -- the same step for all of them -- each rank writes its layers' new
slot tables between that step and the next. Rank B's rows for rank A's old
slot were processed inside A's own step, so A overwriting the slot after its
step is safe, and B routes with the new table from the next step because it
installs between the same two steps.

Load windows are taken only over clean steps (`step_phase`): a padded or
mixed step's routing is not the serving mix, so it pauses the window, and a
change between prefill and decode starts it over.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from vllm_torchtpu import eplb_policy
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

HASH_BITS = 62
FREE = -1
# A step is "decode" when its padded token bucket is at most this.
DECODE_MAX_TOKENS = 512
# A clean step: every rank's tokens (and, in decode, requests) fill at least
# this fraction of the bucket (of the request slots).
CLEAN_FILL = 0.8
# The first decode window after a phase change is this many steps (or the
# window, if shorter), so a layout for the decode mix lands early.
DECODE_FIRST_WINDOW = 16
# How long the ranks drain a timed-out generation's host exchange together.
DRAIN_S = 30.0


# ---------------------------------------------------------------------------
# Layouts, plans, moves


@dataclass
class LayerLayout:
    """The live physical layout of one layer: per rank, per physical slot, the
    expert held or FREE (a staging slot)."""

    held: list[list[int]]

    @property
    def shards(self) -> int:
        return len(self.held)

    @property
    def slots(self) -> int:
        return len(self.held[0])

    def ranks(self) -> list[list[int]]:
        """Per rank, the experts held, in slot order."""
        return [[e for e in row if e != FREE] for row in self.held]

    def holders(self, expert: int) -> list[int]:
        return [r for r, row in enumerate(self.held) if expert in row]

    def slot_of(self, rank: int, expert: int) -> int:
        return self.held[rank].index(expert)

    def free_slots(self, rank: int) -> list[int]:
        return [j for j, e in enumerate(self.held[rank]) if e == FREE]

    def copies(self, experts: int) -> np.ndarray:
        counts = np.zeros(experts, dtype=np.int64)
        for row in self.held:
            for e in row:
                if e != FREE:
                    counts[e] += 1
        return counts

    def slot_table(self, experts: int, cap: int) -> list[list[int]]:
        """Row e: the physical slots holding e, padded with -1 to `cap`."""
        rows: list[list[int]] = [[] for _ in range(experts)]
        for r, row in enumerate(self.held):
            for j, e in enumerate(row):
                if e != FREE:
                    rows[e].append(r * self.slots + j)
        if any(len(h) > cap for h in rows):
            raise ValueError(f"an expert has more than {cap} copies")
        return [h + [FREE] * (cap - len(h)) for h in rows]

    def expert_of_slot(self) -> np.ndarray:
        """[shards * slots] -> expert or FREE."""
        return np.asarray([e for row in self.held for e in row], dtype=np.int64)

    def copy(self) -> LayerLayout:
        return LayerLayout([list(row) for row in self.held])

    @classmethod
    def from_group(cls, group, layer: int) -> LayerLayout:
        ranks = group.ranks(layer)
        return cls([list(held) + [FREE] * group.staging for held in ranks])


@dataclass(frozen=True)
class Move:
    group: str
    layer: int
    expert: int
    src_rank: int
    src_slot: int
    dst_rank: int
    dst_slot: int


@dataclass
class Plan:
    generation: int
    trigger_step: int
    layouts: dict[tuple[str, int], LayerLayout] = field(default_factory=dict)
    moves: list[Move] = field(default_factory=list)
    predicted: dict[tuple[str, int], tuple[float, float]] = field(default_factory=dict)

    @property
    def hash62(self) -> int:
        """A positive 62-bit digest of what every rank must agree on."""
        doc = {
            "generation": self.generation,
            "layouts": [
                [g, lid, self.layouts[(g, lid)].held]
                for (g, lid) in sorted(self.layouts)
            ],
        }
        digest = hashlib.sha256(json.dumps(doc, separators=(",", ":")).encode())
        value = int.from_bytes(digest.digest()[:8], "big") & ((1 << HASH_BITS) - 1)
        return max(value, 1)

    def moves_for(self, rank: int) -> tuple[list[Move], list[Move]]:
        """(incoming, outgoing) moves of one rank."""
        inc = [m for m in self.moves if m.dst_rank == rank]
        out = [m for m in self.moves if m.src_rank == rank]
        return inc, out

    def summary(self) -> str:
        per_rank: dict[int, int] = {}
        for m in self.moves:
            per_rank[m.dst_rank] = per_rank.get(m.dst_rank, 0) + 1
        gains = [
            f"{g}/{lid}: {b:.0f}->{a:.0f}"
            for (g, lid), (b, a) in sorted(self.predicted.items())
        ]
        return (
            f"generation {self.generation}: {len(self.layouts)} layer(s), "
            f"{len(self.moves)} slab move(s), per destination rank "
            f"{dict(sorted(per_rank.items()))}; slowest-rank rows per step "
            f"{', '.join(gains[:6])}{' ...' if len(gains) > 6 else ''}"
        )


# ---------------------------------------------------------------------------
# Window and solve


def window_rows(delta: np.ndarray, layout: LayerLayout, experts: int) -> np.ndarray:
    """Rows per logical expert from a window's per-slot counts."""
    rows = np.zeros(experts, dtype=np.float64)
    owner = layout.expert_of_slot()
    n = min(len(owner), len(delta))
    for s in range(n):
        if owner[s] != FREE:
            rows[owner[s]] += float(delta[s])
    return rows


def slowest_rank_load(rows: np.ndarray, layout: LayerLayout, experts: int) -> float:
    """The most rows any rank processes per step under `layout`, an expert's
    rows split evenly across its copies: under DP lockstep the slowest rank
    is the layer's cost."""
    copies = np.maximum(layout.copies(experts), 1)
    per_copy = rows / copies
    return float(max(per_copy[held].sum() if held else 0.0 for held in layout.ranks()))


def _match_ranks(new_lists: list[list[int]], old: LayerLayout) -> list[list[int]]:
    """Relabel the policy's ranks so each lands on the old rank it overlaps
    most (greedy on the overlap matrix): the policy's rank labels are
    arbitrary, so this removes every move that is only a renaming."""
    shards = len(new_lists)
    old_sets = [set(r) for r in old.ranks()]
    overlap = np.zeros((shards, shards), dtype=np.int64)
    for i, lst in enumerate(new_lists):
        s = set(lst)
        for j in range(shards):
            overlap[i, j] = len(s & old_sets[j])
    assign = [-1] * shards  # new index -> old rank
    taken_new, taken_old = set(), set()
    # Largest overlap first; ties by (new, old) index for determinism.
    order = sorted(
        ((int(overlap[i, j]), -i, -j) for i in range(shards) for j in range(shards)),
        reverse=True,
    )
    for _, ni, nj in order:
        i, j = -ni, -nj
        if i in taken_new or j in taken_old:
            continue
        assign[i] = j
        taken_new.add(i)
        taken_old.add(j)
    matched: list[list[int]] = [[] for _ in range(shards)]
    for i, j in enumerate(assign):
        matched[j] = list(new_lists[i])
    return matched


def _budgeted_next(
    old: LayerLayout,
    targets: list[list[int]],
    load: np.ndarray,
    staging: int,
    cap: int | None = None,
) -> LayerLayout:
    """The next layout toward `targets` within the move budget.

    Each rank may gain at most `staging` experts (its free slots) and must
    end the generation holding its active count again, so it removes as many
    departing experts as it gained; an expert is removed only where another
    rank still holds it afterwards, so every intermediate layout is complete.
    Adds are kept by descending load, removes chosen by ascending load.

    `cap` bounds the copies of one expert in the layout this returns, which
    the slot table's width requires. The policy's target honours it, but the
    step toward the target need not: a rank that departs two experts and adds
    one removes the lighter departure and keeps the hot one, while other ranks
    add that hot expert, so a replicated hot expert gains copies net. An
    expert over the cap gives back its adds on the highest-numbered ranks
    (those ranks take their next wanted expert instead), and the loop runs to
    a fixed point; the choice is deterministic, as every rank solves alone.
    """
    shards = old.shards
    old_sets = [set(r) for r in old.ranks()]
    target_sets = [set(t) for t in targets]
    adds = {
        r: sorted(target_sets[r] - old_sets[r], key=lambda e: (-load[e], e))
        for r in range(shards)
    }
    departing = {
        r: sorted(old_sets[r] - target_sets[r], key=lambda e: (load[e], e))
        for r in range(shards)
    }
    budget = {r: min(staging, len(old.free_slots(r))) for r in range(shards)}
    blocked: dict[int, set[int]] = {r: set() for r in range(shards)}
    while True:
        kept = {
            r: [e for e in adds[r] if e not in blocked[r]][: budget[r]]
            for r in range(shards)
        }
        # Holders after this generation's adds, before its removes.
        holders: dict[int, set[int]] = {}
        for r in range(shards):
            for e in old_sets[r] | set(kept[r]):
                holders.setdefault(e, set()).add(r)
        removes: dict[int, list[int]] = {}
        trimmed = False
        for r in range(shards):
            chosen: list[int] = []
            for e in departing[r]:
                if len(chosen) == len(kept[r]):
                    break
                if len(holders[e]) > 1:
                    chosen.append(e)
                    holders[e].discard(r)
            if len(chosen) < len(kept[r]):
                # Not enough removable departures: give back the least
                # valuable adds and try again with everyone's holders.
                budget[r] = len(chosen)
                trimmed = True
            removes[r] = chosen
        if cap is not None:
            for e in sorted(holders):
                excess = len(holders[e]) - cap
                if excess <= 0:
                    continue
                adders = sorted(
                    (r for r in range(shards) if e in kept[r]), reverse=True
                )
                for r in adders[:excess]:
                    blocked[r].add(e)
                    trimmed = True
        if not trimmed:
            break
    nxt = old.copy()
    for r in range(shards):
        for e in removes[r]:
            nxt.held[r][nxt.held[r].index(e)] = FREE
        free = nxt.free_slots(r)
        for e, j in zip(kept[r], free, strict=False):
            nxt.held[r][j] = e
    return nxt


def _direct_staged_next(
    old: LayerLayout,
    load: np.ndarray,
    staging: int = 2,
    cap: int | None = None,
    start_from: LayerLayout | None = None,
) -> LayerLayout:
    """Directly optimize the next layout within `staging` moves per rank from `old`
    by combining replica retargeting with pairwise single-copy/multi-copy swaps
    between the heaviest and lightest ranks."""
    shards = old.shards
    experts = len(load)
    max_cap = cap if cap is not None else shards
    old_sets = [set(r) for r in old.ranks()]
    base_lay = start_from if start_from is not None else old
    cur_sets = [set(r) for r in base_lay.ranks()]
    rem_budget = [
        min(staging, len(old.free_slots(r))) - len(cur_sets[r] - old_sets[r])
        for r in range(shards)
    ]
    if max(rem_budget, default=0) <= 0:
        return base_lay.copy()

    copies = np.zeros(experts, dtype=np.int64)
    holders = [set() for _ in range(experts)]
    for r in range(shards):
        for e in cur_sets[r]:
            copies[e] += 1
            holders[e].add(r)

    rep_set = {int(e) for e in np.where(copies > 1)[0]}
    w = (load / np.maximum(copies, 1)).astype(np.float64)
    rl = np.array(
        [sum(float(w[e]) for e in cur_sets[r]) for r in range(shards)],
        dtype=np.float64,
    )
    removable = [set(cur_sets[r] & old_sets[r]) for r in range(shards)]
    hot_order = sorted(range(experts), key=lambda e: (-float(load[e]), e))[:24]

    while True:
        max_l = float(rl.max())
        best_action = None
        best_new_max = max_l

        # Action A: Replace a removable redundant expert `e_rep` (`copies > 1`)
        # on `r_lo` with a replica of a hot expert `e_hot`.
        if rep_set:
            rep_ranks = []
            for r_lo in range(shards):
                if rem_budget[r_lo] > 0:
                    reps = removable[r_lo] & rep_set
                    if reps:
                        e_rep = min(
                            reps,
                            key=lambda e: (float(load[e]) / (copies[e] - 1), e),
                        )
                        rep_ranks.append((r_lo, e_rep))
            if rep_ranks:
                hot_candidates = sorted(hot_order, key=lambda e: (-float(w[e]), e))[:8]
                for r_lo, e_rep in rep_ranks:
                    old_w_rep = float(w[e_rep])
                    new_w_rep = float(load[e_rep]) / (copies[e_rep] - 1)
                    delta_rep = new_w_rep - old_w_rep
                    for e_hot in hot_candidates:
                        if (
                            copies[e_hot] >= max_cap
                            or e_hot in cur_sets[r_lo]
                            or e_hot == e_rep
                        ):
                            continue
                        old_w_hot = float(w[e_hot])
                        new_w_hot = float(load[e_hot]) / (copies[e_hot] + 1)
                        delta_hot = new_w_hot - old_w_hot
                        trial_rl = rl.copy()
                        trial_rl[r_lo] += new_w_hot - old_w_rep
                        for r_h in holders[e_rep]:
                            if r_h != r_lo:
                                trial_rl[r_h] += delta_rep
                        for r_h in holders[e_hot]:
                            trial_rl[r_h] += delta_hot
                        new_max = float(trial_rl.max())
                        if new_max < best_new_max - 1e-6:
                            best_new_max = new_max
                            best_action = ("rep", r_lo, e_rep, e_hot, trial_rl)

        # Action B: Pairwise swap between heaviest available rank `r_hi` and
        # lightest available rank `r_lo` (breaks single-copy permutation cycles).
        avail = [r for r in range(shards) if rem_budget[r] > 0 and removable[r]]
        if len(avail) >= 2:
            r_hi = max(avail, key=lambda r: (float(rl[r]), -r))
            r_lo = min(avail, key=lambda r: (float(rl[r]), r))
            if r_hi != r_lo and rl[r_hi] - rl[r_lo] > 1e-6:
                gap = float(rl[r_hi] - rl[r_lo])
                other_max = max(
                    (float(rl[r]) for r in range(shards) if r != r_hi and r != r_lo),
                    default=0.0,
                )
                mov_hi = sorted(
                    (e for e in removable[r_hi] if e not in cur_sets[r_lo]),
                    key=lambda e: (-float(w[e]), e),
                )[:10]
                mov_lo = sorted(
                    (e for e in removable[r_lo] if e not in cur_sets[r_hi]),
                    key=lambda e: (float(w[e]), e),
                )[:10]
                for eh in mov_hi:
                    wh = float(w[eh])
                    for ec in mov_lo:
                        d = wh - float(w[ec])
                        if d <= 0:
                            continue
                        new_hi = float(rl[r_hi]) - d
                        new_lo = float(rl[r_lo]) + d
                        new_pair_max = max(new_hi, new_lo)
                        if new_pair_max < float(rl[r_hi]) - 1e-6:
                            gm = max(other_max, new_pair_max)
                            if gm < best_new_max - 1e-6 or (
                                best_action is None
                                and abs(new_hi - new_lo) < gap - 1e-6
                            ):
                                best_new_max = gm
                                gap = abs(new_hi - new_lo)
                                best_action = (
                                    "swap",
                                    r_hi,
                                    eh,
                                    r_lo,
                                    ec,
                                    new_hi,
                                    new_lo,
                                )

        if best_action is None:
            break
        if best_action[0] == "rep":
            _, r_lo, e_rep, e_hot, rl = best_action
            cur_sets[r_lo].remove(e_rep)
            cur_sets[r_lo].add(e_hot)
            removable[r_lo].remove(e_rep)
            holders[e_rep].remove(r_lo)
            holders[e_hot].add(r_lo)
            copies[e_rep] -= 1
            if copies[e_rep] <= 1:
                rep_set.discard(e_rep)
            copies[e_hot] += 1
            if copies[e_hot] > 1:
                rep_set.add(e_hot)
            w[e_rep] = load[e_rep] / copies[e_rep]
            w[e_hot] = load[e_hot] / copies[e_hot]
            rem_budget[r_lo] -= 1
        else:
            _, r1, eh, r2, ec, new_hi, new_lo = best_action
            cur_sets[r1].remove(eh)
            cur_sets[r1].add(ec)
            cur_sets[r2].remove(ec)
            cur_sets[r2].add(eh)
            removable[r1].remove(eh)
            removable[r2].remove(ec)
            holders[eh].remove(r1)
            holders[eh].add(r2)
            holders[ec].remove(r2)
            holders[ec].add(r1)
            rl[r1] = new_hi
            rl[r2] = new_lo
            rem_budget[r1] -= 1
            rem_budget[r2] -= 1

    res = old.copy()
    for r in range(shards):
        removed = old_sets[r] - cur_sets[r]
        added = sorted(cur_sets[r] - old_sets[r])
        for e in removed:
            res.held[r][res.held[r].index(e)] = FREE
        free = res.free_slots(r)
        for e, j in zip(added, free, strict=False):
            res.held[r][j] = e
    return res


def _moves_between(
    group: str, layer: int, old: LayerLayout, nxt: LayerLayout
) -> list[Move]:
    moves = []
    for r in range(old.shards):
        for j, e in enumerate(nxt.held[r]):
            if e == FREE or e in old.held[r]:
                continue
            src = min(old.holders(e))  # deterministic: the lowest holder sends
            moves.append(Move(group, layer, int(e), src, old.slot_of(src, e), r, j))
    return moves


def solve_group(
    group: str,
    layouts: dict[int, LayerLayout],
    windows: dict[int, np.ndarray],
    *,
    experts: int,
    active: int,
    staging: int,
    replica_cap: int,
    min_gain: float,
) -> tuple[dict[int, LayerLayout], list[Move], dict[int, tuple[float, float]]]:
    """Next layouts for one group's layers from their windows [K, experts]."""
    layer_ids = sorted(windows)
    if not layer_ids:
        return {}, [], {}
    shards = next(iter(layouts.values())).shards
    rows = np.stack([windows[lid] for lid in layer_ids], axis=1)  # [K, L, E]

    def policy(rows_kle):
        return eplb_policy.rebalance_experts(
            rows_kle, shards * active, active, max_copies=replica_cap
        )

    try:
        lists = policy(rows)
        targets_of = {lid: lists[i] for i, lid in enumerate(layer_ids)}
    except ValueError:
        # The greedy packer refuses a layer whose last slots could only take
        # an expert the rank already holds, and the batch fails as a whole.
        # Pack the layers one at a time and skip the refused ones; every rank
        # sees the same windows, so every rank skips the same layers.
        targets_of = {}
        for i, lid in enumerate(layer_ids):
            try:
                targets_of[lid] = policy(rows[:, i : i + 1, :])[0]
            except ValueError:
                continue
        logger.warning(
            "expert relayout | %s: the packer refused %d of %d layers this generation",
            group,
            len(layer_ids) - len(targets_of),
            len(layer_ids),
        )
    out_layouts, moves, predicted = {}, [], {}
    for i, lid in enumerate(layer_ids):
        if lid not in targets_of:
            continue
        old = layouts[lid]
        mean_rows = rows[:, i, :].mean(axis=0)
        targets = _match_ranks(targets_of[lid], old)
        cand1 = _budgeted_next(old, targets, mean_rows, staging, cap=replica_cap)
        cand1_sw = _direct_staged_next(
            old, mean_rows, staging=staging, cap=replica_cap, start_from=cand1
        )
        cand2 = _direct_staged_next(
            old, mean_rows, staging=staging, cap=replica_cap, start_from=None
        )

        def _valid(lay: LayerLayout) -> bool:
            c = lay.copies(experts)
            return (
                int(c.min()) >= 1
                and int(c.max()) <= replica_cap
                and all(len(r) == active for r in lay.ranks())
            )

        p1 = (
            slowest_rank_load(mean_rows, cand1_sw, experts)
            if _valid(cand1_sw)
            else float("inf")
        )
        p2 = (
            slowest_rank_load(mean_rows, cand2, experts)
            if _valid(cand2)
            else float("inf")
        )
        nxt = cand1_sw if p1 <= p2 else cand2
        if nxt.held == old.held:
            continue
        copies = nxt.copies(experts)
        if (
            copies.min() < 1
            or copies.max() > replica_cap
            or any(len(r) != active for r in nxt.ranks())
        ):
            # Never hand install a layout the slot table cannot hold.
            logger.warning(
                "expert relayout | %s layer %d: budgeted layout invalid "
                "(copies %d..%d, cap %d), layer skipped this generation",
                group,
                lid,
                int(copies.min()),
                int(copies.max()),
                replica_cap,
            )
            continue
        # The gain gate: the slowest rank's mean rows per step must drop by
        # at least `min_gain` (relative) for the moves to be worth making.
        before = slowest_rank_load(mean_rows, old, experts)
        after = slowest_rank_load(mean_rows, nxt, experts)
        if before <= 0 or (before - after) / before < min_gain:
            continue
        out_layouts[lid] = nxt
        predicted[lid] = (float(before), float(after))
        moves.extend(_moves_between(group, lid, old, nxt))
    return out_layouts, moves, predicted


def step_phase(
    bucket: int,
    *,
    max_tokens: int,
    min_tokens: int,
    num_chunks: int,
    max_reqs: int,
    min_reqs: int,
    max_num_reqs: int,
) -> str:
    """The coordinated step's phase from the DP all-reduce's fields (the same
    on every rank): "prefill" or "decode" for a clean step, "padded" for one
    whose load does not represent the serving mix (an idle or partly filled
    rank, a multi-chunk step, a request batch far from full in decode)."""
    is_decode = bucket <= DECODE_MAX_TOKENS
    req_target = max(int(max_num_reqs), max_reqs)
    clean = (
        max_tokens > 0
        and bucket > 0
        and num_chunks == 1
        and min_tokens >= int(CLEAN_FILL * bucket)
        and (
            not is_decode or (max_reqs > 0 and min_reqs >= int(CLEAN_FILL * req_target))
        )
    )
    if not clean:
        return "padded"
    return "decode" if is_decode else "prefill"


# ---------------------------------------------------------------------------
# Movers and the controller


class Mover(Protocol):
    """Copies the moved experts' slabs into the destinations' staging slots.

    `start` returns at once and works in the background; `staged_generation`
    is the highest generation whose incoming slabs are all in place on this
    rank (the active generation when nothing is in flight); `failed` says the
    current generation cannot complete.
    """

    def start(self, plan: Plan, modules: dict[tuple[str, int], object]) -> None: ...

    def pump(self) -> None:
        """One bounded slice of this rank's device work, from the step hook."""

    def staged_generation(self) -> int: ...

    def failed(self) -> bool: ...

    def cancel(self) -> None: ...

    def settle(self, generation: int) -> None: ...


IDLE, SOLVING, PENDING, STAGING = "idle", "solving", "pending", "staging"


_SOLVE_WORKER_MAIN = """
import logging, os, pickle, struct, sys, types
out_fd = os.dup(1)
os.dup2(2, 1)
sys.stdout = sys.stderr
pipe_out = os.fdopen(out_fd, "wb", buffering=0)
pkg_dir = sys.argv[1]
parent_dir = os.path.dirname(pkg_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
pkg = types.ModuleType("vllm_torchtpu")
pkg.__path__ = [pkg_dir]
sys.modules["vllm_torchtpu"] = pkg
log_mod = types.ModuleType("vllm_torchtpu.logger")
log_mod.init_logger = logging.getLogger
sys.modules["vllm_torchtpu.logger"] = log_mod
sys.modules["torch"] = None
from vllm_torchtpu import expert_relayout as er

stdin = sys.stdin.buffer
while True:
    hdr = stdin.read(4)
    if not hdr or len(hdr) < 4:
        break
    (n,) = struct.unpack("<I", hdr)
    data = stdin.read(n)
    if len(data) < n:
        break
    try:
        payload = pickle.loads(data)
        res = (True, er._solve_from_payload(payload))
    except BaseException as e:
        res = (False, repr(e))
    out = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
    pipe_out.write(struct.pack("<I", len(out)) + out)
    pipe_out.flush()
"""


def _solve_from_payload(payload: dict) -> tuple[Plan, int]:
    ctrl = object.__new__(Controller)
    ctrl.groups = payload["groups"]
    ctrl.min_gain = payload["min_gain"]
    ctrl.move_bytes = payload["move_bytes"]
    ctrl.rank_bytes_cap = payload["rank_bytes_cap"]
    ctrl.stats = {"floored": 0}
    plan = ctrl._solve(
        payload["history"],
        payload["layouts"],
        payload["generation"],
        payload["step"],
        logical_windows=payload.get("logical_windows"),
    )
    return plan, int(ctrl.stats.get("floored", 0))


class _SolveWorker:
    """Persistent lightweight subprocess that runs the 40-layer NumPy solver
    outside the server process (no torch import), so the server's GIL is
    never held during the ~300 ms solve."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _ensure_proc(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            pkg_dir = os.path.dirname(os.path.abspath(__file__))
            env = {
                **os.environ,
                "PJRT_DEVICE": "CPU",
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            self._proc = subprocess.Popen(
                [sys.executable, "-u", "-c", _SOLVE_WORKER_MAIN, pkg_dir],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env=env,
            )
        return self._proc

    def solve(self, payload: dict) -> tuple[Plan, int]:
        with self._lock:
            proc = self._ensure_proc()
            assert proc.stdin is not None and proc.stdout is not None
            raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            proc.stdin.write(struct.pack("<I", len(raw)) + raw)
            proc.stdin.flush()
            hdr = proc.stdout.read(4)
            if not hdr or len(hdr) < 4:
                raise RuntimeError("expert relayout solve worker exited unexpectedly")
            (n,) = struct.unpack("<I", hdr)
            buf = bytearray()
            while len(buf) < n:
                chunk = proc.stdout.read(n - len(buf))
                if not chunk:
                    raise RuntimeError("expert relayout solve worker truncated reply")
                buf.extend(chunk)
            ok, val = pickle.loads(bytes(buf))
            if not ok:
                raise RuntimeError(f"solve worker error: {val}")
            return val


_SOLVE_WORKER: _SolveWorker | None = None
_SOLVE_WORKER_INIT_LOCK = threading.Lock()


def _get_solve_worker() -> _SolveWorker:
    global _SOLVE_WORKER
    with _SOLVE_WORKER_INIT_LOCK:
        if _SOLVE_WORKER is None:
            _SOLVE_WORKER = _SolveWorker()
        return _SOLVE_WORKER


class Controller:
    """One rank's view of the generation protocol (module docstring)."""

    def __init__(
        self,
        *,
        rank: int,
        groups: dict[str, dict],
        layouts: dict[tuple[str, int], LayerLayout],
        modules: dict[tuple[str, int], object],
        snapshot,
        install,
        mover: Mover,
        window_steps: int,
        sub_windows: int = 4,
        min_gain: float = 0.03,
        warmup_steps: int | None = None,
        timeout_windows: int = 2,
        synchronous_solve: bool = False,
        move_bytes: dict[str, int] | None = None,
        step_budget_bytes: int = 0,
    ) -> None:
        """groups: name -> dict(experts, active, staging, replica_cap).
        snapshot(): (group, layer) -> cumulative counts [P]. install(module,
        table): writes a layer's table. layouts: the live layouts at arm."""
        self.rank = rank
        self.groups = groups
        self.layouts = layouts
        self.modules = modules
        self.snapshot = snapshot
        self.install = install
        self.mover = mover
        self.window_steps = max(int(window_steps), 1)
        self.sub_windows = max(int(sub_windows), 1)
        self.min_gain = float(min_gain)
        self.move_bytes = dict(move_bytes or {})  # group -> bytes of one slab move
        self._pumped_since_step = False
        self.warmup_steps = (
            self.window_steps if warmup_steps is None else int(warmup_steps)
        )
        self.timeout_steps = max(int(timeout_windows), 1) * self.window_steps
        # One rank's slabs in plus out pass through the mover's per-step byte
        # budget (one pump a step): a generation larger than half of what the
        # timeout lets through cannot stage in time, so it is capped there
        # (0 = an unbounded budget, no cap).
        budget = max(int(step_budget_bytes), 0)
        self.rank_bytes_cap = budget * self.timeout_steps // 2
        self.generation = 0
        self.state = IDLE
        self.pending: Plan | None = None
        self.history: list[dict[tuple[str, int], np.ndarray]] = []
        self.stats = {
            "triggers": 0,
            "skipped": 0,
            "aborted": 0,
            "installed": 0,
            "moves": 0,
        }
        self._stage_started: float | None = None
        # The solve (policy and matching over every layer) takes hundreds of
        # milliseconds on the host; a thread hands it to the solve subprocess
        # so neither the step hook nor the GIL carries it. Tests may run it
        # inline.
        self.synchronous_solve = bool(synchronous_solve)
        self._solve_thread: threading.Thread | None = None
        self._solve_result: Plan | None = None
        self._solve_error: BaseException | None = None
        self._solve_step = 0
        self._phase: str = "init"
        self._steps_since_reset: int = 0
        self._first_decode_window: bool = False
        self._need_read_issue: bool = False
        self._last_step: int = 0
        self.decode_first_window: int = max(DECODE_FIRST_WINDOW, 8)
        self._decode_only: bool = True
        self._chain_decode: bool = True
        self._freeze_converged: bool = True
        self._decode_logical_windows: dict[tuple[str, int], np.ndarray] | None = None
        self._decode_converged: bool = False

    # -- the three int64 fields of the per-step all-reduce

    def fields(self) -> tuple[int, int, int]:
        h = self.pending.hash62 if (self.state == PENDING and self.pending) else 0
        return h, -h, -int(self.mover.staged_generation())

    # -- per step, after the all-reduce, before the forward

    def on_step(
        self,
        step: int,
        reduced: tuple[int, int, int],
        *,
        phase: str | None = None,
    ) -> None:
        """`phase` is the step's `step_phase`; None (tests) triggers every
        `window_steps` steps without gating."""
        from vllm_torchtpu.layers.adapter import expert_load

        expert_load.on_step(step)
        self._last_step = step
        hash_max, neg_hash_max, neg_staged_max = (int(v) for v in reduced)
        min_hash = -neg_hash_max
        min_staged = -neg_staged_max
        if self.state == SOLVING:
            self._collect_solve(step)
        if self.state == STAGING:
            assert self.pending is not None
            if not self._pumped_since_step:
                # No forward ran since the last hook (an idle step on this
                # rank): the mover still gets its slice of work.
                self.mover.pump()
            self._pumped_since_step = False
            if self.mover.failed():
                self._abort(step, "the mover failed")
            elif min_staged >= self.pending.generation:
                self._install(step)
            elif step - self.pending.trigger_step > self.timeout_steps:
                # Every rank reaches this at the same step: drain together.
                self._abort(
                    step, f"not staged within {self.timeout_steps} steps", drain=True
                )
        elif self.state == PENDING:
            assert self.pending is not None
            if hash_max == min_hash == self.pending.hash62:
                self.pending.trigger_step = step - 1
                self.state = STAGING
                self._stage_started = time.monotonic()
                self._pumped_since_step = True
                self.mover.start(self.pending, self.modules)
                logger.info("expert relayout | confirmed %s", self.pending.summary())
            elif min_hash == 0:
                # A rank is still solving (it publishes 0 until it has a
                # plan); wait, within the generation's timeout.
                if step - self.pending.trigger_step > self.timeout_steps:
                    self._abort(step, "not every rank published a plan in time")
            else:
                self._abort(
                    step,
                    f"plan hash disagrees across ranks (max {hash_max:#x}, "
                    f"min {min_hash:#x}, mine {self.pending.hash62:#x})",
                )
        if phase is None:
            if (
                self.state == IDLE
                and step >= self.warmup_steps
                and step % self.window_steps == 0
            ):
                self._trigger(step)
            return

        if phase == "padded":
            if self._phase != "padded":
                logger.info(
                    "expert relayout | step %d: entering padded/drain phase (was %s); "
                    "pausing load window",
                    step,
                    self._phase,
                )
                self._phase = "padded"
                self._steps_since_reset = 0
                self.history = []
                self._need_read_issue = False
            return

        if phase != self._phase:
            logger.info(
                "expert relayout | step %d: entering %s phase (was %s); "
                "resetting load window",
                step,
                phase,
                self._phase,
            )
            self._phase = phase
            self._steps_since_reset = 0
            self.history = []
            self._first_decode_window = phase == "decode"
            self._need_read_issue = False

        if self.state != IDLE:
            self._steps_since_reset = 0
            self.history = []
            return

        if self._phase == "prefill" and self._decode_only:
            return

        if self._phase == "decode" and self._decode_converged:
            return

        self._steps_since_reset += 1
        target_win = (
            min(self.decode_first_window, self.window_steps)
            if (self._phase == "decode" and self._first_decode_window)
            else self.window_steps
        )
        target_win = max(target_win, 8)
        if not self.history:
            if self._steps_since_reset == 1:
                self._need_read_issue = True
            elif self._steps_since_reset >= 3:
                expert_load.flush()
                snap = self.snapshot()
                if snap:
                    self.history = [snap]
        if self._steps_since_reset == target_win - 4:
            self._need_read_issue = True
        if self._steps_since_reset >= target_win:
            if self._phase == "prefill" and step < self.warmup_steps:
                return
            self._first_decode_window = False
            self._steps_since_reset = 0
            self._trigger(step)
            if self.state == IDLE and self.history:
                self.history = self.history[-1:]

    def after_dispatch(self) -> None:
        """Called once per step right after the forward's programs are
        dispatched: the tap's read and the mover's device legs are issued
        here, behind the step's launches in the device's in-order queue,
        never ahead of them."""
        from vllm_torchtpu.layers.adapter import expert_load

        if self._need_read_issue:
            self._need_read_issue = False
            expert_load.issue_now()
        if self.state == STAGING:
            self._pumped_since_step = True
            self.mover.pump()

    # -- pieces

    def _compute_logical_windows(
        self, history, layouts
    ) -> dict[tuple[str, int], np.ndarray]:
        out: dict[tuple[str, int], np.ndarray] = {}
        for name, spec in self.groups.items():
            layer_ids = sorted(lid for (g, lid) in layouts if g == name)
            for lid in layer_ids:
                key = (name, lid)
                deltas = []
                for a, b in zip(history[:-1], history[1:], strict=False):
                    if key in a and key in b:
                        delta = (b[key] - a[key]) & 0xFFFFFFFF
                        deltas.append(window_rows(delta, layouts[key], spec["experts"]))
                if deltas and sum(d.sum() for d in deltas) > 0:
                    out[key] = np.stack(deltas)
        return out

    def _launch_solve(
        self,
        step: int,
        history,
        layouts,
        logical_windows: dict[tuple[str, int], np.ndarray] | None = None,
    ) -> None:
        self.stats["triggers"] += 1
        generation = self.generation + 1
        self._solve_result = None
        self._solve_error = None
        self._solve_step = step
        self.state = SOLVING
        if self.synchronous_solve:
            try:
                self._solve_result = self._solve(
                    history, layouts, generation, step, logical_windows=logical_windows
                )
            except Exception as e:  # noqa: BLE001 - reported by _collect_solve
                self._solve_error = e
            self._collect_solve(step)
            return

        payload = {
            "groups": self.groups,
            "min_gain": self.min_gain,
            "move_bytes": self.move_bytes,
            "rank_bytes_cap": self.rank_bytes_cap,
            "history": history,
            "layouts": layouts,
            "generation": generation,
            "step": step,
            "logical_windows": logical_windows,
        }

        def run():
            try:
                plan, floored = _get_solve_worker().solve(payload)
                if floored:
                    self.stats["floored"] = self.stats.get("floored", 0) + floored
                self._solve_result = plan
            except BaseException as e:  # noqa: BLE001 - reported on the main thread
                self._solve_error = e

        self._solve_thread = threading.Thread(
            target=run, name="expert-relayout-solve", daemon=True
        )
        self._solve_thread.start()

    def _trigger(self, step: int) -> None:
        from vllm_torchtpu.layers.adapter import expert_load

        expert_load.flush()
        snap = self.snapshot()
        if not snap:
            return  # no counts have landed yet; the next boundary will
        self.history.append(snap)
        self.history = self.history[-(self.sub_windows + 1) :]
        if len(self.history) < 2:
            return  # the first snapshot is the baseline
        history = list(self.history)
        layouts = {k: v.copy() for k, v in self.layouts.items()}
        logical_windows = None
        if self._phase == "decode" and self._chain_decode:
            logical_windows = self._compute_logical_windows(history, layouts)
            self._decode_logical_windows = logical_windows
        self._launch_solve(step, history, layouts, logical_windows=logical_windows)

    def _collect_solve(self, step: int) -> None:
        """Adopt a finished solve: a plan goes PENDING, nothing goes IDLE."""
        if self._solve_thread is not None and self._solve_thread.is_alive():
            return
        self._solve_thread = None
        if self._solve_error is not None:
            logger.error(
                "expert relayout | step %d: the solve failed: %r",
                step,
                self._solve_error,
            )
            self._solve_error = None
            self.stats["skipped"] += 1
            self._decode_logical_windows = None
            self.state = IDLE
            if self._phase in ("prefill", "decode") and self.history:
                self.history = self.history[-1:]
                self._steps_since_reset = 0
            return
        plan = self._solve_result
        self._solve_result = None
        if plan is None or not plan.layouts:
            self.stats["skipped"] += 1
            was_chaining = self._decode_logical_windows is not None
            self._decode_logical_windows = None
            if (self._phase == "decode" or was_chaining) and self._freeze_converged:
                self._decode_converged = True
            logger.info(
                "expert relayout | step %d: no layer clears the %.0f%% gain gate%s",
                step,
                100 * self.min_gain,
                " (decode converged; freezing)" if self._decode_converged else "",
            )
            self.state = IDLE
            if self._phase in ("prefill", "decode") and self.history:
                self.history = self.history[-1:]
                self._steps_since_reset = 0
            return
        plan.trigger_step = self._solve_step
        self.pending = plan
        self.state = PENDING
        logger.info("expert relayout | step %d: solved %s", step, plan.summary())

    def _solve(
        self,
        history,
        layouts,
        generation: int,
        step: int,
        logical_windows: dict[tuple[str, int], np.ndarray] | None = None,
    ) -> Plan:
        """The whole solve over every group, from copies of the window and the
        layouts: safe to run off the main thread."""
        plan = Plan(generation=generation, trigger_step=step)
        for name, spec in self.groups.items():
            layer_ids = sorted(lid for (g, lid) in layouts if g == name)
            windows = {}
            for lid in layer_ids:
                key = (name, lid)
                if logical_windows is not None and key in logical_windows:
                    windows[lid] = logical_windows[key]
                    continue
                deltas = []
                for a, b in zip(history[:-1], history[1:], strict=False):
                    if key in a and key in b:
                        delta = (b[key] - a[key]) & 0xFFFFFFFF
                        deltas.append(window_rows(delta, layouts[key], spec["experts"]))
                if deltas and sum(d.sum() for d in deltas) > 0:
                    windows[lid] = np.stack(deltas)
            if not windows:
                continue
            new_layouts, moves, predicted = solve_group(
                name,
                {lid: layouts[(name, lid)] for lid in layer_ids},
                windows,
                experts=spec["experts"],
                active=spec["active"],
                staging=spec["staging"],
                replica_cap=spec["replica_cap"],
                min_gain=self.min_gain,
            )
            for lid, lay in new_layouts.items():
                plan.layouts[(name, lid)] = lay
                plan.predicted[(name, lid)] = predicted[lid]
            plan.moves.extend(moves)
        return self._cap(plan)

    def _cap(self, plan: Plan) -> Plan:
        """Keep layers by saving per byte while every rank's slabs in plus
        out fit `rank_bytes_cap`, what the step budget lets through before
        the staging timeout."""
        keys = list(plan.layouts)
        saving = {k: plan.predicted[k][0] - plan.predicted[k][1] for k in keys}
        drop = set()
        rank_cap = self.rank_bytes_cap
        if rank_cap > 0:
            cost = dict.fromkeys(keys, 0)
            per_rank: dict = {k: {} for k in keys}
            for m in plan.moves:
                k = (m.group, m.layer)
                if k not in cost:
                    continue
                b = self.move_bytes.get(m.group, 0)
                cost[k] += b
                for r in (m.src_rank, m.dst_rank):
                    per_rank[k][r] = per_rank[k].get(r, 0) + b
            spent_rank: dict[int, int] = {}
            for k in sorted(keys, key=lambda k: -(saving[k] / max(cost[k], 1))):
                if any(
                    spent_rank.get(r, 0) + b > rank_cap for r, b in per_rank[k].items()
                ):
                    drop.add(k)
                    continue
                for r, b in per_rank[k].items():
                    spent_rank[r] = spent_rank.get(r, 0) + b
        if drop:
            for k in drop:
                plan.layouts.pop(k, None)
                plan.predicted.pop(k, None)
            plan.moves = [m for m in plan.moves if (m.group, m.layer) not in drop]
            self.stats["floored"] = self.stats.get("floored", 0) + len(drop)
        return plan

    def _install(self, step: int) -> None:
        assert self.pending is not None
        plan = self.pending
        for key, layout in plan.layouts.items():
            spec = self.groups[key[0]]
            self.install(
                self.modules[key],
                layout.slot_table(spec["experts"], spec["replica_cap"]),
            )
            self.layouts[key] = layout
        self.generation = plan.generation
        self.mover.settle(self.generation)
        self.stats["installed"] += 1
        self.stats["moves"] += len(plan.moves)
        staged_s = (
            time.monotonic() - self._stage_started if self._stage_started else 0.0
        )
        logger.info(
            "expert relayout | step %d: installed generation %d (%d layers, %d moves, "
            "staged in %.1f s, %d steps after the trigger)",
            step,
            self.generation,
            len(plan.layouts),
            len(plan.moves),
            staged_s,
            step - plan.trigger_step,
        )
        # The window starts over from the counts the new layout begins with.
        from vllm_torchtpu.layers.adapter import expert_load

        if self._phase == "init":
            expert_load.flush()
            self.history = [self.snapshot()]
        else:
            self.history = []
            self._steps_since_reset = 0
        self.pending = None
        self.state = IDLE
        if self._chain_decode and self._decode_logical_windows:
            layouts = {k: v.copy() for k, v in self.layouts.items()}
            self._launch_solve(
                step,
                [],
                layouts,
                logical_windows=self._decode_logical_windows,
            )

    def _abort(self, step: int, why: str, drain: bool = False) -> None:
        """Drop the pending generation. A drop every rank takes at the same
        step (the staging timeout) drains the mover's exchange first, so no
        posted receive outlives its generation; a rank-local drop (a failed
        mover) cannot count on its peers and gives the generation up."""
        assert self.pending is not None
        self.stats["aborted"] += 1
        how = ""
        progress = getattr(self.mover, "progress", None)
        if progress is not None and self.state == STAGING:
            logger.warning(
                "expert relayout | step %d: generation %d at the drop: %s",
                step,
                self.pending.generation,
                progress(),
            )
        drainer = getattr(self.mover, "drain", None)
        if drain and drainer is not None:
            t0 = time.monotonic()
            ok = drainer(DRAIN_S)
            how = (
                f"; exchange drained in {time.monotonic() - t0:.2f} s"
                if ok
                else f"; exchange abandoned after {DRAIN_S:.0f} s"
            )
        logger.warning(
            "expert relayout | step %d: generation %d dropped: %s%s",
            step,
            self.pending.generation,
            why,
            how,
        )
        self.mover.cancel()
        self.mover.settle(self.generation)
        self.pending = None
        self.state = IDLE


# ---------------------------------------------------------------------------
# Arming in a server


def arm(
    *,
    ep_rank: int,
    ep_world: int,
    ep_global_ranks: list[int] | None = None,
    lockstep: bool = True,
    vllm_config=None,
    transport=None,
) -> Controller | None:
    """Build this rank's controller, or return None with the reason logged.

    Called once per rank at the same point of startup on every rank (after
    the model is loaded and warmed up): the mover's gloo group is created
    here, collectively. Needs the dynamic knob, a layout with staging slots
    applied to at least one layer, the load tap registered, and the DP
    lockstep (every rank performs the step all-reduce every step).
    `transport` defaults to tpu-sync's raw transfers (tests pass a CPU one).
    """
    from vllm_torchtpu import envs, expert_mover, expert_placement
    from vllm_torchtpu.layers.adapter import expert_load

    if not envs.TPU_MOE_EPLB_DYNAMIC:
        return None
    placement = expert_placement.current(vllm_config)
    modules = expert_placement.applied_layers()
    if placement is None or not modules:
        logger.warning(
            "expert relayout: TPU_MOE_EPLB_DYNAMIC is set but no layer is placed"
        )
        return None
    if not lockstep:
        logger.error("expert relayout: needs the DP/EP lockstep (DP > 1 with EP)")
        return None
    if not expert_load.registered_layers():
        logger.error("expert relayout: the load tap registered no layer")
        return None
    # Start the solve subprocess now, not at the first window.
    _get_solve_worker()._ensure_proc()
    groups: dict[str, dict] = {}
    layouts: dict[tuple[str, int], LayerLayout] = {}
    kept_modules: dict[tuple[str, int], object] = {}
    for (gname, lid), module in sorted(modules.items(), key=lambda kv: kv[0]):
        group = placement.group(gname)
        if group is None or group.staging < 1:
            continue
        table = expert_placement.expert_slot_table(module)
        if table is None:
            continue
        cap = int(table.shape[1])
        spec = groups.setdefault(
            gname,
            dict(
                experts=group.experts,
                active=group.active,
                staging=group.staging,
                replica_cap=cap,
            ),
        )
        if spec["replica_cap"] != cap:
            logger.error(
                "expert relayout: %s layers disagree on the table width", gname
            )
            return None
        layouts[(gname, lid)] = LayerLayout.from_group(group, lid)
        kept_modules[(gname, lid)] = module
    if not groups:
        logger.warning(
            "expert relayout: layout has no staging slots; nothing to re-lay"
        )
        return None
    if transport is None:
        transport = expert_mover.RawTransferTransport()
        if hasattr(transport, "prewarm"):
            try:
                transport.prewarm(list(kept_modules.values()))
            except Exception as e:  # noqa: BLE001
                logger.warning("expert relayout: transport prewarm skipped (%s)", e)
    ranks = (
        list(ep_global_ranks) if ep_global_ranks is not None else list(range(ep_world))
    )
    p2p = expert_mover.GlooP2P(expert_mover.make_gloo_group(ranks), rank_map=ranks)
    mover = expert_mover.SlabMover(
        ep_rank,
        transport,
        p2p,
        step_budget_bytes=int(envs.TPU_MOE_EPLB_MOVER_STEP_BUDGET_MB) << 20,
    )
    move_bytes = {}
    for (gname, _lid), module in kept_modules.items():
        if gname in move_bytes:
            continue
        try:
            move_bytes[gname] = expert_mover.move_bytes(module, transport)
        except Exception:  # noqa: BLE001 - a layer without expert tensors (tests' fakes)
            move_bytes[gname] = 0
    controller = Controller(
        rank=ep_rank,
        groups=groups,
        layouts=layouts,
        modules=kept_modules,
        snapshot=expert_load.snapshot,
        install=expert_placement.install_table,
        mover=mover,
        window_steps=envs.TPU_MOE_EPLB_WINDOW_STEPS,
        min_gain=envs.TPU_MOE_EPLB_MIN_GAIN,
        move_bytes=move_bytes,
        step_budget_bytes=mover.step_budget_bytes,
    )
    first = next(iter(kept_modules.values()))
    try:
        parts = expert_mover.describe_parts(first, transport)
    except Exception as e:  # noqa: BLE001 - a layer without expert tensors (tests' fakes)
        parts = str(e)
    logger.info("expert relayout | first placed layer's tensors: %s", parts)
    logger.info(
        "expert relayout armed | rank %d of %d | groups %s | %d layers | "
        "window %d steps "
        "x %d sub-windows | min gain %.0f%% | step budget %.0f MB | per-rank cap "
        "%.0f MB",
        ep_rank,
        ep_world,
        {g: (v["active"], v["staging"], v["replica_cap"]) for g, v in groups.items()},
        len(layouts),
        controller.window_steps,
        controller.sub_windows,
        100 * controller.min_gain,
        mover.step_budget_bytes / 1e6,
        controller.rank_bytes_cap / 1e6,
    )
    return controller


def arm_for_runner(runner) -> Controller | None:
    """The server's entry point: the EP group's rank and world. Raises only
    when tpu-sync lacks what the mover needs (the knob asked for a dynamic
    layout the server cannot provide); any other failure is logged and the
    static layout served."""
    try:
        from vllm.distributed.parallel_state import get_ep_group

        group = get_ep_group()
        return arm(
            ep_rank=int(group.rank_in_group),
            ep_world=int(group.world_size),
            ep_global_ranks=[int(r) for r in group.ranks],
            lockstep=bool(runner._dp_lockstep_enabled()),
            vllm_config=getattr(runner, "vllm_config", None),
        )
    except ImportError:
        raise
    except Exception:  # noqa: BLE001 - serving goes on with the static layout
        logger.exception("expert relayout: arming failed; serving the static layout")
        return None
