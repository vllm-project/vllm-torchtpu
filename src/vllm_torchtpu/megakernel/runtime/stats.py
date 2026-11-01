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
"""Framework-free step statistics: batch-shape inspection from host metadata,
the periodic step-mix tracker and the A/B latency calibrator (stdlib + numpy;
no jax, no policy or profile imports)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BatchShape:
    """Framework-free description of the current forward step's batch shape."""

    eligible: bool
    num_reqs: int = 0
    tokens_per_seq: int = 0
    reason: str = "stock_early"


def check_qsl_uniform(qsl: Any, num_reqs: int) -> BatchShape:
    """Check whether `qsl[:num_reqs + 1]` has a uniform step size in `{1, 4}`."""
    if num_reqs <= 0 or len(qsl) <= num_reqs:
        return BatchShape(False, 0, 0, "stock_n0")
    s = qsl[: num_reqs + 1]
    vals = s.tolist() if hasattr(s, "tolist") else [int(x) for x in s]
    if int(vals[0]) != 0:
        return BatchShape(False, num_reqs, 0, "stock_mixed")
    q0 = int(vals[1]) - int(vals[0])
    for i in range(1, num_reqs):
        if int(vals[i + 1]) - int(vals[i]) != q0:
            return BatchShape(False, num_reqs, 0, "stock_mixed")
    if q0 in (1, 4):
        return BatchShape(True, num_reqs, q0, "ok")
    return BatchShape(False, num_reqs, 0, f"stock_q{q0}")


def check_mrope_uniform(mrope_3xN: Any, active_rows: int) -> bool:
    """Check that all 3 M-RoPE position rows agree across `:active_rows`."""
    arr = np.asarray(mrope_3xN)
    if arr.ndim != 2 or arr.shape[0] != 3 or arr.shape[1] < active_rows:
        return False
    r0 = arr[0, :active_rows]
    return bool(
        np.array_equal(r0, arr[1, :active_rows])
        and np.array_equal(r0, arr[2, :active_rows])
    )


def inspect_batch_shape(
    *,
    num_reqs: int | None = None,
    has_roff: bool = False,
    decode_and_total_reqs: tuple[int, int] | None = None,
    qsl: Any | None = None,
    fallback_decode_and_total_reqs: tuple[int, int] | None = None,
    fallback_num_seqs: int | None = None,
    fallback_qsl: Any | None = None,
) -> BatchShape:
    """Determine `(eligible, num_reqs, tokens_per_seq, reason)` from host metadata."""
    if has_roff and decode_and_total_reqs is not None:
        num_windowed, dist_reqs = map(int, decode_and_total_reqs)
        if num_reqs is None:
            num_reqs = dist_reqs
        if dist_reqs <= 0 or num_reqs <= 0:
            return BatchShape(False, 0, 0, "stock_n0")
        if num_windowed != dist_reqs:
            reason = "stock_prefill" if num_windowed == 0 else "stock_mixed"
            return BatchShape(False, num_reqs, 0, reason)

    if qsl is not None and num_reqs is not None:
        bs = check_qsl_uniform(qsl, num_reqs)
        if bs.eligible and not has_roff and bs.tokens_per_seq != 1:
            return BatchShape(False, bs.num_reqs, 0, "stock_prefill")
        return bs

    if fallback_decode_and_total_reqs is not None:
        d_end, total_reqs = map(int, fallback_decode_and_total_reqs)
        if total_reqs <= 0:
            return BatchShape(False, 0, 0, "stock_n0")
        if d_end != total_reqs:
            reason = "stock_prefill" if d_end == 0 else "stock_mixed"
            return BatchShape(False, total_reqs, 0, reason)
        if num_reqs is None:
            num_reqs = total_reqs

    if num_reqs is None and fallback_num_seqs is not None:
        num_reqs = int(fallback_num_seqs)
    if fallback_qsl is not None and num_reqs is not None:
        return check_qsl_uniform(fallback_qsl, num_reqs)
    return BatchShape(False, 0, 0, "stock_no_qsl")


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo, hi = int(pos), min(int(pos) + 1, n - 1)
    return sorted_vals[lo] * (1.0 - (pos - lo)) + sorted_vals[hi] * (pos - lo)


class _LatencySeries:
    """Bounded-memory ring-buffer sample accumulator (exact mean + recent quantiles)."""

    __slots__ = ("cap", "count", "samples", "sum_ms")

    def __init__(self, cap: int = 4000) -> None:
        self.cap, self.count, self.sum_ms = int(cap), 0, 0.0
        self.samples: list[float] = []

    def add(self, dt_ms: float) -> None:
        v, idx = float(dt_ms), self.count
        self.count, self.sum_ms = idx + 1, self.sum_ms + v
        if len(self.samples) < self.cap:
            self.samples.append(v)
        else:
            self.samples[idx % self.cap] = v

    def summary(self) -> list[int | float]:
        if self.count == 0:
            return [0, 0.0, 0.0, 0.0, 0.0]
        s = sorted(self.samples)
        qs = (round(_percentile(s, q), 3) for q in (0.10, 0.50, 0.90))
        return [self.count, round(self.sum_ms / self.count, 3), *qs]


class MKABCalibrator:
    """Deterministic per-key MK vs stock alternation and latency accumulator."""

    def __init__(
        self, enabled: bool = False, period_s: float = 60.0, cap: int = 4000
    ) -> None:
        self.enabled, self.period_s, self.cap = bool(enabled), float(period_s), int(cap)
        self.counts: dict[str, int] = {}
        self.last_decision: tuple[str, str] = ("stock_early", "stock")
        self._window: dict[str, dict[str, _LatencySeries]] = {}
        self._cumulative: dict[str, dict[str, _LatencySeries]] = {}
        self._t0 = time.monotonic()

    def reset(self) -> None:
        self.counts.clear()
        self.last_decision = ("stock_early", "stock")
        self._window.clear()
        self._cumulative.clear()
        self._t0 = time.monotonic()

    def choose(self, key: str, mk_eligible: bool) -> tuple[bool, str, str]:
        """Return `(actual_run_mk, mix_key, arm)` for the current step."""
        if not mk_eligible or not self.enabled:
            arm = "mk" if (mk_eligible and not self.enabled) else "stock"
            self.last_decision = (key, arm)
            return arm == "mk", key, arm
        c = self.counts.get(key, 0)
        self.counts[key] = c + 1
        arm = "mk" if (c & 1) == 0 else "stock"
        self.last_decision = (key, arm)
        return arm == "mk", f"{key}_ab_{arm}", arm

    def record(
        self, dt_ms: float, padded_tokens: int | None = None
    ) -> tuple[float, str, str] | None:
        """Record a timed step and return `(elapsed_s, window_json, cum_json)`."""
        if not self.enabled:
            return None
        key, arm = self.last_decision
        rec_key = f"{key}_p{int(padded_tokens)}" if padded_tokens is not None else key
        for store in (self._window, self._cumulative):
            b_map = store.setdefault(rec_key, {})
            if arm not in b_map:
                b_map[arm] = _LatencySeries(self.cap)
            b_map[arm].add(dt_ms)
        now = time.monotonic()
        if (elapsed := now - self._t0) < self.period_s:
            return None
        win_j, cum_j = (
            self.format_summary(self._window),
            self.format_summary(self._cumulative),
        )
        self._window.clear()
        self._t0 = now
        return elapsed, win_j, cum_j

    @staticmethod
    def format_summary(store: dict[str, dict[str, _LatencySeries]]) -> str:
        out: dict[str, dict[str, Any]] = {}
        for k in sorted(store):
            arms = store[k]
            entry: dict[str, Any] = {
                arm: arms[arm].summary() for arm in ("mk", "stock") if arm in arms
            }
            if "mk" in arms and "stock" in arms:
                mk_s, st_s = entry["mk"], entry["stock"]
                entry["d_mean_ms"] = round(float(mk_s[1]) - float(st_s[1]), 3)
                entry["d_p50_ms"] = round(float(mk_s[3]) - float(st_s[3]), 3)
            out[k] = entry
        return json.dumps(out, separators=(",", ":"))


class StepMixTracker:
    """Periodic step-mix counter and summary formatter."""

    def __init__(self, period_s: float = 60.0) -> None:
        self.period_s, self.counts, self.t0 = float(period_s), {}, time.monotonic()

    def reset(self) -> None:
        self.counts, self.t0 = {}, time.monotonic()

    def note_step(
        self, key: str, run: bool, ab_calibrator: MKABCalibrator | None = None
    ) -> tuple[bool, tuple[float, int, int, float, dict[str, int]] | None]:
        """Record `key` and return `(actual_run, window_summary_or_None)`."""
        actual_run, mix_key = (
            ab_calibrator.choose(key, run)[:2]
            if ab_calibrator is not None
            else (bool(run), key)
        )
        self.counts[mix_key] = self.counts.get(mix_key, 0) + 1
        now = time.monotonic()
        if (elapsed := now - self.t0) < self.period_s:
            return actual_run, None
        total = sum(self.counts.values())
        mk_cnt = sum(v for k, v in self.counts.items() if k.startswith("mk_"))
        pct, snap = 100.0 * mk_cnt / max(total, 1), dict(sorted(self.counts.items()))
        self.counts, self.t0 = {}, now
        return actual_run, (elapsed, mk_cnt, total, pct, snap)


__all__ = [
    "BatchShape",
    "MKABCalibrator",
    "StepMixTracker",
    "check_mrope_uniform",
    "check_qsl_uniform",
    "inspect_batch_shape",
]
