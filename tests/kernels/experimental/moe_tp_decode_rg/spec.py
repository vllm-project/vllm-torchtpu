# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Shared cell spec for the moe_tp_decode row-gather campaign: shapes, routing kinds, weights, sweep drivers.

One module feeds test_accuracy.py and bench.py so the two can never disagree on a cell. A cell is
(m, deployment, routing kind, n_valid). Routing is constructed explicitly: each row's 8 picks are drawn by the kind's
rule and the scores are built so that top-8 of (scores + bias) is exactly that set (picked logits 2 + N(0, 0.3), others
N(-1, 0.5); distinct values, so no ties), so the kernel's own routing reproduces the intended pattern.

Kinds: indep (each row draws 8 distinct experts uniformly); mtp<rho> (a request's 4 rows keep each of the request's 8
base picks with prob rho, else redraw uniformly); zipf<s> (picks weighted by popularity 1/rank^s over a random expert
permutation, rows independent); zipf<s>+mtp<rho> both.
"""

import dataclasses
import math

import numpy as np

H, E, TOPK, FS, Q = 6144, 256, 8, 128, 4  # hidden, router experts, top-k, shared cols/rank, MTP rows/request
N_RANKS = 16
ROUTED_SCALE = 2.5
I_CHUNK = 128
M_ALL = (16, 32, 64, 80, 96, 112, 128, 144, 160, 176, 192)
KINDS = ("indep", "mtp0.8", "zipf0.6")


@dataclasses.dataclass(frozen=True)
class Dep:
    """One rank's slice: E_l local experts of `E` at `i` columns; rank r holds experts [r * e_l, ...) if ep."""

    name: str
    e_l: int
    i: int
    ep: bool
    h: int = H
    e: int = E
    fs: int = FS

    @property
    def n_chunks(self):
        return self.i // min(self.i, I_CHUNK)


TP16 = Dep("tp16", e_l=E, i=128, ep=False)
EP16 = Dep("ep16", e_l=E // N_RANKS, i=2048, ep=True)
# CPU-tier shapes: same routing statistics (E, top-k), small hidden.
TP16_SMALL = Dep("tp16s", e_l=E, i=128, ep=False, h=1024)
EP16_SMALL = Dep("ep16s", e_l=E // N_RANKS, i=2048, ep=True, h=1024)
DEPS = {d.name: d for d in (TP16, EP16, TP16_SMALL, EP16_SMALL)}


def picks(m, kind, seed, e=E, topk=TOPK):
    """bool [m, e]: the experts each row routes to (topk per row)."""
    rng = np.random.default_rng(seed)
    w = None
    rho = None
    for part in kind.split("+"):
        if part.startswith("zipf"):
            s = float(part[4:])
            w = 1.0 / np.arange(1, e + 1) ** s
            w = w[rng.permutation(e)]
            w /= w.sum()
        elif part.startswith("mtp"):
            rho = float(part[3:])
        elif part != "indep":
            raise ValueError(kind)

    def draw():
        return rng.choice(e, topk, replace=False, p=w)

    out = np.zeros((m, e), bool)
    base = None
    for r in range(m):
        if rho is None:
            sel = draw()
        else:
            if r % Q == 0:
                base = draw()
            keep = list(base[rng.random(topk) < rho])
            sel = (keep + [x for x in draw() if x not in keep])[:topk]
        out[r, sel] = True
    return out


def scores_bias(m, kind, seed, e=E, topk=TOPK):
    """(scores [m, e] f32 = sigmoid(logits), bias [e] f32, picks [m, e] bool) with top-k(scores + bias) == picks."""
    rng = np.random.default_rng(seed + 7919)
    p = picks(m, kind, seed, e, topk)
    logits = np.where(p, 2.0 + 0.3 * rng.standard_normal((m, e)), -1.0 + 0.5 * rng.standard_normal((m, e)))
    scores = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    bias = rng.uniform(-0.02, 0.02, e).astype(np.float32)
    # Safety: picked min (sigmoid(2 - 4*0.3) ~ 0.69) stays above unpicked max (sigmoid(-1 + 4*0.5) ~ 0.73)? Not by
    # construction at 4 sigma, so verify and repair rows that would route differently.
    biased = scores + bias
    for r in range(m):
        top = np.argsort(-biased[r], kind="stable")[:topk]
        if set(top) != set(np.flatnonzero(p[r])):
            lo = biased[r, p[r]].min()
            hi = biased[r, ~p[r]].max()
            scores[r, ~p[r]] -= (hi - lo) + 0.01
            scores[r] = np.clip(scores[r], 1e-4, 1.0)
            biased[r] = scores[r] + bias
    for r in range(m):
        assert set(np.argsort(-biased[r])[:topk]) == set(np.flatnonzero(p[r]))
    return scores, bias, p


def drivers(p, n_valid, dep, r_slab=16):
    """Sweep drivers (M6) from the picks of valid rows, for this deployment (EP: per rank; reports max and mean)."""
    pv = p[:n_valid]
    n_e = pv.sum(0)  # rows per global expert
    hit = n_e > 0
    out = dict(sum_n=int(n_e.sum()), max_n=int(n_e.max()) if n_e.size else 0)
    m = p.shape[0]
    tiles = pv.reshape(-1, 8, p.shape[1]).any(1).sum(0) if n_valid % 8 == 0 else None
    if dep.ep:
        per_rank_items = hit.reshape(N_RANKS, -1).sum(1) * dep.n_chunks
        out.update(items_max=int(per_rank_items.max()), items_mean=float(per_rank_items.mean()))
    else:
        out.update(items_max=int(hit.sum()), items_mean=float(hit.sum()))
    out["passes_extra"] = int(np.maximum(0, np.ceil(n_e / r_slab) - 1).sum())
    out["mean_n_hit"] = float(n_e[hit].mean()) if hit.any() else 0.0
    if tiles is not None:
        out["tiles8_mean_hit"] = float(tiles[hit].mean()) if hit.any() else 0.0
    return out


def cells(ms=M_ALL, deps=("tp16", "ep16"), kinds=KINDS):
    return [(m, d, k) for d in deps for m in ms for k in kinds]


def seed_of(m, dep, kind, salt=0):
    return (m * 1009 + sum(map(ord, dep)) * 31 + sum(map(ord, kind)) * 7 + salt) % (2**31)


def fp8_columns_np(w):
    """Per-output-channel FP8 scale for [..., K, N] (host reference of the tests' _fp8_columns)."""
    s = np.max(np.abs(w), axis=-2) / 448.0
    return np.where(s == 0, 1.0, s)


def weight_stds(dep):
    return 1.0 / math.sqrt(dep.h), 1.0 / math.sqrt(16 * dep.i)
