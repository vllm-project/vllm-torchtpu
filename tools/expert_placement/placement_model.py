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
"""Cost model and per-layer search for static expert placement tables.

A placement of one MoE layer gives each of ``ranks`` EP ranks ``slots``
experts (a permutation of the ``experts`` ids, as ``TPU_EXPERT_PLACEMENT``
needs). Its cost is the expected MoE time of the busiest rank, weighted over
the steps a deployment runs:

    J(M) = w_dec * sum_T P(T) * E[max_r (decode_us + decode_us_per_active
                                         * active_r)]
         + sum_k w_k * E[max_r (prefill_us + prefill_ns_per_pair / 1000
                                * pairs_r)]          (prefill step type k)

- ``active_r``: experts of rank r with at least one routed row in a decode
  step of T tokens (a rank streams the weights of every active expert).
- ``pairs_r``: (token, expert) pairs routed to rank r in a prefill step.
- Routing is sampled per expert: n_e ~ Binomial(T, pi_e), pi_e =
  min(1, topk * p_e), p_e the expert's share of the profile's hits (decode
  steps for the decode term, prefill steps for the prefill terms).

The analytic J used by the search is exact for the decode term under this
sampler (Poisson-binomial per rank) and uses a normal approximation for the
prefill pairs. ``monte_carlo`` evaluates the same model by sampling, as a
check on the analytic terms.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
from scipy.special import log_ndtr


@dataclasses.dataclass(frozen=True)
class Shape:
    """The deployment: routed experts, EP ranks and experts per token."""

    experts: int
    ranks: int
    topk: int

    def __post_init__(self) -> None:
        if self.experts % self.ranks:
            raise ValueError(
                f"{self.experts} experts do not split evenly over {self.ranks} ranks"
            )
        if not 0 < self.topk <= self.experts:
            raise ValueError(f"topk {self.topk} out of range")

    @property
    def slots(self) -> int:
        return self.experts // self.ranks


@dataclasses.dataclass
class CostModel:
    """Per-layer, per-rank MoE cost of a step, and the step mix.

    The four cost constants are the kernel's time on one rank for one layer.
    The defaults were measured for GLM-5.2-FP8 on a 16-chip v7x slice with
    flat EP16 (decode: GMM at 128 tokens; prefill: 4096-token chunks); measure
    them again for another model, kernel or slice. Only their ratios matter
    for the search, so rough values still rank placements sensibly.
    """

    decode_us: float = 31.9
    decode_us_per_active_expert: float = 11.3
    prefill_us: float = 200.0
    prefill_ns_per_pair: float = 53.0
    # Decode steps by token count: {tokens: relative frequency}.
    decode_tokens: dict[int, float] | None = None
    # Decode steps per unit time, on the same scale as the prefill weights.
    decode_weight: float | None = None
    # Prefill step types: [[tokens, steps per unit time], ...].
    prefill_steps: list[list[float]] | None = None
    # Decode steps with fewer tokens are treated as placement-independent
    # (for kernels whose time does not depend on the active experts there).
    decode_min_tokens: int = 0
    # Numerics: equal-probability decode bins and the prefill grid size.
    decode_bins: int = 8
    grid: int = 64

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        if self.decode_tokens is not None:
            d["decode_tokens"] = {
                str(k): v for k, v in sorted(self.decode_tokens.items())
            }
        return d

    @classmethod
    def from_json(cls, d: dict) -> CostModel:
        d = dict(d)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(
                f"unknown cost model keys {unknown}; known: {sorted(known)}"
            )
        if d.get("decode_tokens") is not None:
            d["decode_tokens"] = {
                int(k): float(v) for k, v in d["decode_tokens"].items()
            }
        if d.get("prefill_steps") is not None:
            d["prefill_steps"] = [[int(t), float(w)] for t, w in d["prefill_steps"]]
        return cls(**d)

    @classmethod
    def load(cls, path: str | None) -> CostModel:
        if not path:
            return cls()
        with open(path) as f:
            return cls.from_json(json.load(f))

    def resolved(self, workload: dict) -> CostModel:
        """A copy with every None step-mix field taken from ``workload``.

        ``workload`` (from ``workload_from_counts``): decode_steps,
        decode_tokens_per_step, prefill_steps, prefill_tokens_per_step.
        The decode and prefill weights share one scale, so they come both
        from the cost model or both from the counts.
        """
        if (self.decode_weight is None) != (self.prefill_steps is None):
            raise ValueError(
                "give decode_weight and prefill_steps together (one scale), "
                "or neither (both from the counts)"
            )
        out = dataclasses.replace(self)
        dec_t = int(round(workload.get("decode_tokens_per_step", 0.0)))
        if out.decode_tokens is None:
            out.decode_tokens = {dec_t: 1.0} if dec_t > 0 else {}
        if out.decode_weight is None:
            out.decode_weight = float(workload.get("decode_steps", 0.0))
        if out.prefill_steps is None:
            pre_t = int(round(workload.get("prefill_tokens_per_step", 0.0)))
            n = float(workload.get("prefill_steps", 0.0))
            out.prefill_steps = [[pre_t, n]] if pre_t > 0 and n > 0 else []
        return out

    def decode_probs(self) -> tuple[np.ndarray, np.ndarray]:
        """(T values, P(T)) over all decode steps; P is zeroed below
        decode_min_tokens and not renormalized (those steps are treated as
        placement-independent)."""
        hist = self.decode_tokens or {}
        t = np.array(sorted(hist), dtype=np.int64)
        if len(t) == 0 or sum(hist.values()) <= 0:
            return np.array([1], dtype=np.int64), np.array([0.0])
        n = np.array([hist[int(x)] for x in t], dtype=np.float64)
        p = n / n.sum()
        return t, np.where(t >= self.decode_min_tokens, p, 0.0)

    def decode_bins_t(self) -> tuple[np.ndarray, np.ndarray]:
        """Compressed (T values, weights) for the analytic decode term:
        equal-probability bins, each at its probability-weighted mean T."""
        t, p = self.decode_probs()
        keep = p > 0
        if not keep.any():
            return np.array([1], dtype=np.int64), np.array([0.0])
        t, p = t[keep], p[keep]
        if len(t) <= self.decode_bins:
            return t, p
        cdf = np.cumsum(p) / p.sum()
        edges = np.linspace(0, 1, self.decode_bins + 1)[1:-1]
        bid = np.searchsorted(edges, cdf - 0.5 * p / p.sum())
        ts, ws = [], []
        for b in np.unique(bid):
            m = bid == b
            ws.append(p[m].sum())
            ts.append(int(round((t[m] * p[m]).sum() / p[m].sum())))
        return np.array(ts, dtype=np.int64), np.array(ws)


def workload_from_counts(
    decode: np.ndarray,
    prefill: np.ndarray,
    decode_steps: float,
    prefill_steps: float,
    topk: int,
) -> dict:
    """Mean tokens per decode and prefill step from one layer's counts.

    ``decode`` / ``prefill``: that layer's hits per expert (topk per token).
    """
    dec_tok = float(np.sum(decode)) / topk
    pre_tok = float(np.sum(prefill)) / topk
    return {
        "decode_steps": float(decode_steps),
        "prefill_steps": float(prefill_steps),
        "decode_tokens_per_step": dec_tok / decode_steps if decode_steps else 0.0,
        "prefill_tokens_per_step": pre_tok / prefill_steps if prefill_steps else 0.0,
    }


def norm_p(h) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    s = h.sum()
    return h / s if s > 0 else np.full_like(h, 1.0 / len(h))


def linear_cores(shape: Shape) -> list[list[int]]:
    s = shape.slots
    return [list(range(r * s, (r + 1) * s)) for r in range(shape.ranks)]


def _balanced(shape: Shape, order, load) -> list[list[int]]:
    """Each expert in ``order`` onto the open rank with the least load."""
    cores: list[list[int]] = [[] for _ in range(shape.ranks)]
    rl = np.zeros(shape.ranks)
    for e in order:
        open_r = [r for r in range(shape.ranks) if len(cores[r]) < shape.slots]
        r = min(open_r, key=lambda q: rl[q])
        cores[r].append(int(e))
        rl[r] += load[e]
    return cores


def lpt_permutation(shape: Shape, load) -> list[list[int]]:
    """Equal-cardinality longest-processing-time assignment of ``load``."""
    load = np.asarray(load, dtype=np.float64)
    return _balanced(shape, np.argsort(-load, kind="stable"), load)


def lpt_idle(shape: Shape, a) -> list[list[int]]:
    """Most-often-idle expert first onto the least expected-active rank."""
    a = np.asarray(a, dtype=np.float64)
    return _balanced(shape, np.argsort(-(1 - a), kind="stable"), a)


def snake(shape: Shape, a) -> list[list[int]]:
    order = np.argsort(-np.asarray(a), kind="stable")
    cores: list[list[int]] = [[] for _ in range(shape.ranks)]
    for i, e in enumerate(order):
        blk, pos = divmod(i, shape.ranks)
        cores[pos if blk % 2 == 0 else shape.ranks - 1 - pos].append(int(e))
    return cores


def is_permutation(shape: Shape, cores) -> bool:
    if len(cores) != shape.ranks or any(len(c) != shape.slots for c in cores):
        return False
    return sorted(int(e) for c in cores for e in c) == list(range(shape.experts))


def pb_add(pmf, a):
    """Adds one Bernoulli(a) to a count pmf (last axis = count 0..K)."""
    out = pmf * (1.0 - a)[..., None]
    out[..., 1:] += pmf[..., :-1] * a[..., None]
    return out


def pb_dist(a) -> np.ndarray:
    """Poisson-binomial pmf over the last axis of a: [..., k] -> [..., k+1]."""
    k = a.shape[-1]
    out = np.zeros(a.shape[:-1] + (k + 1,))
    out[..., 0] = 1.0
    for j in range(k):
        out = pb_add(out, a[..., j])
    return out


class LayerModel:
    """Per-expert quantities of one layer for the analytic J."""

    def __init__(self, p_dec, p_pre, shape: Shape, cost: CostModel, decode_only=False):
        self.shape = shape
        self.cost = cost
        self.b_us = cost.prefill_ns_per_pair / 1000.0
        self.w_dec = float(cost.decode_weight or 0.0)
        self.pi_dec = np.minimum(1.0, shape.topk * norm_p(p_dec))
        self.pi_pre = np.minimum(1.0, shape.topk * norm_p(p_pre))
        self.t_dec, self.p_dec = cost.decode_bins_t()
        if self.p_dec.sum() <= 0:
            self.w_dec = 0.0
        # a[e, s]: P(expert e active in a decode step of t_dec[s] tokens).
        self.a = 1.0 - (1.0 - self.pi_dec[:, None]) ** self.t_dec[None, :]
        self.reg = []
        for t, w in cost.prefill_steps or []:
            if decode_only or w <= 0:
                continue
            mu = t * self.pi_pre
            var = t * self.pi_pre * (1.0 - self.pi_pre)
            self.reg.append((int(t), float(w), mu, var))
        self.const = self.w_dec * self.p_dec.sum() * cost.decode_us + sum(
            w for _, w, _, _ in self.reg
        ) * (cost.prefill_us)

    def _dec_state(self, m):
        per = self.shape.slots
        am = np.moveaxis(self.a[m], 1, -1)  # [R, S, PER]
        pmf = pb_dist(am)  # [R, S, PER+1]
        logc = np.log(np.maximum(np.cumsum(pmf, -1), 1e-300))
        # Leave-one-out pmfs: [R, PER (excluded slot), S, PER+1].
        ex = np.repeat(am[:, None], per, axis=1)
        idx = np.arange(per)
        ex[:, idx, :, idx] = 0.0
        loo = pb_dist(ex)
        return logc, loo

    @staticmethod
    def _emax_from_logsum(tot):
        # tot [..., S, PER+1]: sum_r log P(A_r <= m); E[max] = sum_{m<PER} 1-prod.
        return np.sum(1.0 - np.exp(tot[..., :-1]), axis=-1)

    def dec_emax(self, m) -> np.ndarray:
        """E[max_r active_r] per decode bin, [S]."""
        logc, _ = self._dec_state(m)
        return self._emax_from_logsum(logc.sum(0))

    def _grid(self, mu_r, sd_r):
        """Integration grid valid for the current state and every single swap.

        A swap changes two ranks, so one of the three largest-mean ranks is
        unchanged and prod F ~ 0 below (3rd largest mean - 7 sd). Above: the
        current max + 7 sd + 5% of the mean rank; swaps whose new mean + 5 sd
        falls outside are rejected (they raise the expected max anyway).
        """
        sdm = sd_r.max()
        lo = np.sort(mu_r)[-min(3, len(mu_r))] - 7.0 * sdm
        hi = mu_r.max() + 7.0 * sdm + 0.05 * mu_r.mean()
        return np.linspace(max(lo, 0.0), max(hi, max(lo, 0.0) + 1.0), self.cost.grid)

    @staticmethod
    def _sd(var, x):
        """Rank sd floored at one grid step: a nearly deterministic rank has a
        step-like CDF that the trapezoid rule would miss by half a step."""
        return np.sqrt(np.maximum(var, (x[1] - x[0]) ** 2))

    @staticmethod
    def _emax_normal(x, log_f_tot):
        # E[max] = x0 + int_{x0}^{x1} (1 - prod F) dx, with prod F(x0) ~ 0.
        y = 1.0 - np.exp(log_f_tot)
        dx = x[1] - x[0]
        return x[0] + dx * (y[..., 1:] + y[..., :-1]).sum(-1) * 0.5

    def pre_emax(self, m) -> list[float]:
        """E[max_r pairs_r] per prefill step type."""
        out = []
        for _, _, mu, var in self.reg:
            mu_r = mu[m].sum(1)
            x = self._grid(mu_r, np.sqrt(np.maximum(var[m].sum(1), 1e-6)))
            sd_r = self._sd(var[m].sum(1), x)
            lf = log_ndtr((x[None, :] - mu_r[:, None]) / sd_r[:, None]).sum(0)
            out.append(float(self._emax_normal(x, lf)))
        return out

    def terms(self, m) -> dict:
        m = np.asarray(m)
        dec_act = float(self.dec_emax(m) @ self.p_dec)
        pre = self.pre_emax(m)
        c = self.cost
        j_dec = self.w_dec * (
            self.p_dec.sum() * c.decode_us + c.decode_us_per_active_expert * dec_act
        )
        j_pre = [
            w * (c.prefill_us + self.b_us * e) for (_, w, _, _), e in zip(self.reg, pre)
        ]
        return {
            "decode_emax_active": dec_act,
            "prefill_emax_pairs": pre,
            "J_decode": j_dec,
            "J_prefill": float(sum(j_pre)),
            "J": j_dec + float(sum(j_pre)),
        }

    def J(self, m) -> float:
        return self.terms(m)["J"]

    def _candidates(self, ranks_a, all_pairs: bool):
        rr, per = self.shape.ranks, self.shape.slots
        if all_pairs:
            rp = [(r1, r2) for r1 in range(rr) for r2 in range(r1 + 1, rr)]
        else:
            rp = sorted(
                {
                    (min(r1, r2), max(r1, r2))
                    for r1 in ranks_a
                    for r2 in range(rr)
                    if r1 != r2
                }
            )
        rp = np.array(rp, dtype=np.int64).reshape(-1, 2)
        i1, i2 = np.meshgrid(np.arange(per), np.arange(per), indexing="ij")
        i1, i2 = i1.ravel(), i2.ravel()
        r1 = np.repeat(rp[:, 0], len(i1))
        r2 = np.repeat(rp[:, 1], len(i1))
        return r1, np.tile(i1, len(rp)), r2, np.tile(i2, len(rp))

    def _eval_swaps(self, m, r1, i1, r2, i2, dstate):
        """Analytic J (without its constant terms) of every candidate swap
        m[r1, i1] <-> m[r2, i2], [N]."""
        e1, e2 = m[r1, i1], m[r2, i2]
        j = np.zeros(len(r1))
        if self.w_dec > 0:
            logc, loo = dstate
            base = logc.sum(0)
            n1 = pb_add(loo[r1, i1], self.a[e2])
            n2 = pb_add(loo[r2, i2], self.a[e1])
            l1 = np.log(np.maximum(np.cumsum(n1, -1), 1e-300))
            l2 = np.log(np.maximum(np.cumsum(n2, -1), 1e-300))
            tot = base[None] - logc[r1] - logc[r2] + l1 + l2
            emax = self._emax_from_logsum(tot)
            j += (
                self.w_dec * self.cost.decode_us_per_active_expert * (emax @ self.p_dec)
            )
        for _, w, mu, var in self.reg:
            mu_r = mu[m].sum(1)
            var_r = var[m].sum(1)
            x = self._grid(mu_r, np.sqrt(np.maximum(var_r, 1e-6)))
            sd_r = self._sd(var_r, x)
            lf = log_ndtr((x[None, :] - mu_r[:, None]) / sd_r[:, None])
            base = lf.sum(0)
            m1 = mu_r[r1] - mu[e1] + mu[e2]
            m2 = mu_r[r2] - mu[e2] + mu[e1]
            s1 = self._sd(var_r[r1] - var[e1] + var[e2], x)
            s2 = self._sd(var_r[r2] - var[e2] + var[e1], x)
            tot = (
                base[None]
                - lf[r1]
                - lf[r2]
                + log_ndtr((x[None] - m1[:, None]) / s1[:, None])
                + log_ndtr((x[None] - m2[:, None]) / s2[:, None])
            )
            em = self._emax_normal(x, tot)
            over = np.maximum(m1 + 5 * s1, m2 + 5 * s2) > x[-1]
            j += w * self.b_us * np.where(over, np.inf, em)
        return j

    def _top_ranks(self, m, k=3):
        out: set[int] = set()
        if self.w_dec > 0:
            am = self.a[m]
            score = (am.sum(1) + 2 * np.sqrt((am * (1 - am)).sum(1))) @ self.p_dec
            out |= set(np.argsort(-score)[:k].tolist())
        if self.reg:
            _, _, mu, var = self.reg[0]
            score = mu[m].sum(1) + 2 * np.sqrt(var[m].sum(1))
            out |= set(np.argsort(-score)[:k].tolist())
        return sorted(out)

    def local_search(self, cores, max_iter=200, rel_tol=3e-5, top=3):
        """Steepest descent over pairwise swaps on the analytic J.

        Candidates: swaps between the ``top`` worst ranks (decode and prefill
        scores) and every other rank; when none improves, one pass over all
        rank pairs. Stops when no swap lowers J by more than ``rel_tol`` of
        the starting J (relative, so the weights' scale does not matter).
        Returns (cores, J, iterations, swaps applied).
        """
        m = np.array([list(c) for c in cores], dtype=np.int64)
        cur = self.J(m)
        it = swaps = 0
        if self.w_dec <= 0 and not self.reg:  # J does not depend on m
            return [list(map(int, r)) for r in m], cur, it, swaps
        tol = rel_tol * abs(cur)
        all_pairs = False
        for it in range(max_iter):
            dstate = self._dec_state(m) if self.w_dec > 0 else None
            cand = self._candidates(self._top_ranks(m, top), all_pairs)
            if len(cand[0]) == 0:  # one rank, or no worst rank to move
                if all_pairs:
                    break
                all_pairs = True
                continue
            jc = self._eval_swaps(m, *cand, dstate)
            j = int(np.argmin(jc))
            if float(jc[j]) + self.const < cur - tol:
                r1, i1, r2, i2 = (int(c[j]) for c in cand)
                m[r1, i1], m[r2, i2] = m[r2, i2], m[r1, i1]
                new = self.J(m)
                if new > cur - tol * 0.5:  # analytic delta disagreed: undo
                    m[r1, i1], m[r2, i2] = m[r2, i2], m[r1, i1]
                    if all_pairs:
                        break
                    all_pairs = True
                    continue
                cur = new
                swaps += 1
                all_pairs = False
            elif not all_pairs:
                all_pairs = True
            else:
                break
        return [list(map(int, r)) for r in m], cur, it, swaps


def solve_layer(
    p_dec,
    p_pre,
    shape: Shape,
    cost: CostModel,
    max_iter=200,
) -> dict:
    """Linear, q-LPT (decode term only), LPT (prefill pairs) and the joint
    table: the best swap search on J from q-LPT and LPT, never
    worse than linear on J."""
    lm = LayerModel(p_dec, p_pre, shape, cost)
    lin = linear_cores(shape)
    lm_dec = LayerModel(p_dec, p_pre, shape, cost, decode_only=True)
    a_mean = lm.a @ (lm.p_dec / max(lm.p_dec.sum(), 1e-12))
    q0 = min((lpt_idle(shape, a_mean), snake(shape, a_mean)), key=lm_dec.J)
    if lm_dec.w_dec > 0:
        q, _, q_it, _ = lm_dec.local_search(q0, max_iter=max_iter)
    else:
        q, q_it = q0, 0
    lpt = lpt_permutation(shape, p_pre if np.sum(p_pre) > 0 else p_dec)
    res = {"linear": lin, "qlpt": q, "lpt": lpt}
    cands = [(n, res[n]) for n in ("qlpt", "lpt")]
    best = None
    for name, start in cands:
        c, jv, it, sw = lm.local_search(start, max_iter=max_iter)
        if best is None or jv < best[1]:
            best = (c, jv, it, sw, name)
    assert best is not None, "solve_layer needs at least one start"
    res["joint"] = best[0] if best[1] <= lm.J(np.array(lin)) else lin
    out = {
        "tables": res,
        "joint_start": best[4],
        "joint_iters": best[2],
        "joint_swaps": best[3],
        "qlpt_iters": q_it,
        "terms": {name: lm.terms(np.array(c)) for name, c in res.items()},
    }
    return out


def own_matrix(shape: Shape, cores) -> np.ndarray:
    own = np.zeros((shape.ranks, shape.experts), dtype=np.float32)
    for r, c in enumerate(cores):
        own[r, list(c)] = 1.0
    return own


def monte_carlo(
    tables: dict,
    p_dec,
    p_pre,
    shape: Shape,
    cost: CostModel,
    rng,
    s_dec=2000,
    s_pre=400,
) -> dict:
    """Sampled E[max active] (decode, T drawn from the decode mix, steps below
    decode_min_tokens counted as 0) and E[max pairs] per prefill step type,
    per table, with common random numbers across tables."""
    t_vals, p = cost.decode_probs()
    pi_d = np.minimum(1.0, shape.topk * norm_p(p_dec))
    pi_p = np.minimum(1.0, shape.topk * norm_p(p_pre))
    hist = cost.decode_tokens or {1: 1.0}
    p_all = np.array([hist.get(int(x), 0.0) for x in t_vals], dtype=np.float64)
    p_all = (
        p_all / p_all.sum() if p_all.sum() > 0 else np.ones(len(t_vals)) / len(t_vals)
    )
    t = rng.choice(t_vals, size=s_dec, p=p_all)
    act = (rng.binomial(t[:, None], pi_d[None, :]) > 0).astype(np.float32)
    incl = (t >= cost.decode_min_tokens).astype(np.float32)
    pre = [
        rng.binomial(int(tok), np.repeat(pi_p[None, :], s_pre, 0)).astype(np.float32)
        for tok, _ in cost.prefill_steps or []
    ]
    out = {}
    for name, cores in tables.items():
        own = own_matrix(shape, cores)
        out[name] = {
            "decode_emax_active": float(((act @ own.T).max(1) * incl).mean()),
            "prefill_emax_pairs": [float((n @ own.T).max(1).mean()) for n in pre],
        }
    return out


def pct_saved(j_base: float, j: float) -> float:
    return 0.0 if j_base <= 0 else (j_base - j) / j_base * 100
