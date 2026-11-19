# SPDX-License-Identifier: Apache-2.0
"""q38 L1-C48 DECODE-TRIM: row-aware decode step trimming (VLLM_TPU_DECODE_TRIM, default off).

Why (T98g c48, STACK-D v2, measured): the whole-step program D14 serves at most 64 rows, and with MTP k3 every
decode request is 4 rows, so any step with more than 16 running decode requests runs the stock path at 80-128 rows.
That was 22.45 % of c48's steps and about 31 % of its time. Per request-step, D14 + S1's 64-row step is cheaper than
the stock 80 / 96-row steps, so serving 16 requests and letting the rest wait one step can give more tokens per ms
than serving them all in a padded stock bucket.

Policy: on every engine step with only decode requests running, count the requests the step would serve (n: the
running decode requests plus the waiting requests whose KV is ready, which are never held back) and pick

    m* = argmax over m in [must, n] of  (m * AL - (n - m) * (AL - 1) * reentry) / cost(bucket(m * rows_per_request))

from the cost table, taking m* < n only if its score beats serving all n, n * AL / cost(bucket(n * rows)), by the
margin. AL is the mean tokens per request-step. The (n - m) * (AL - 1) term charges the re-entry loss: a request held
back for a step has no async substitution source when it re-enters (tpu_runner.py
_prepare_async_token_substitution_indices skips request ids absent from the previous step's results, and vLLM's
AsyncScheduler ships -1 placeholder drafts), so it yields only its bonus token on that step. reentry = 1 charges it in
full; 0 is for a runner that keeps the drafts across the skip. ``must`` = the requests held back in the previous
step plus the ready waiting requests, so **no request is held back in two consecutive steps**: the starvation bound
is one step. Among the rest, the requests served least recently stay in; ties go to the requests held back most
often, then to the earlier admitted, so the held-back slot rotates round-robin; the others are held back for this
step. They stay RUNNING, keep their KV blocks and are not preempted; vLLM ships their
tokens again when they re-enter (``all_token_ids`` for requests not scheduled in the previous step).

Mechanism: vLLM's scheduler asks ``self._mamba_block_aligned_split(request, num_new_tokens, ...)`` for every request
it is about to schedule when ``self.need_mamba_block_aligned_split`` is true (pp_chunk_scheduler.py clips prefill
chunks at the same point), and a running request for which it returns 0 is skipped for the step. The trim installs
that hook as an instance attribute on the engine core's scheduler, returns 0 for the held-back requests and otherwise
defers to the class's own method exactly when the model needed it. Waiting requests are never held back.

Env (both host-only, compile-env-ignored):
  VLLM_TPU_DECODE_TRIM      '' / '0' = off (nothing is wrapped). Otherwise the per-bucket step cost table in ms,
                            'ROWS:MS/ROWS:MS/...', optionally '/margin:F' (default 0.02), '/al:F' (mean tokens per
                            request-step, default 2.3), '/reentry:F' (0..1, default 1) and '/above:ROWS' (trim only
                            steps whose untrimmed rows exceed ROWS, e.g. 64 = only steps that would leave D14;
                            default 0 = any step). Rows are the padded token
                            buckets the runner serves (16, 32, 48, 64, 80, 96, 112, 128, ...). A step whose rows fall
                            past the largest key is never trimmed. No ',' or '+' (served / bench env separators).
  VLLM_TPU_DECODE_TRIM_LOG  engine steps per 'q38 DECODE-TRIM' window line (default 2000; 0 = install line only).
"""

from __future__ import annotations

import math
import re
import time
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_PAIR = re.compile(r"^(\d+):(\d+(?:\.\d+)?)$")
DEFAULT_MARGIN = 0.02
DEFAULT_AL = 2.3
DEFAULT_REENTRY = 1.0
_PRUNE_EVERY = 1024
_GAP_CAP = 200_000  # gap samples kept per window (about 16 requests x 2000 steps x 6)


class TrimTable:
    """The per-bucket step cost table and the throughput rule."""

    def __init__(self, costs: dict[int, float], margin: float = DEFAULT_MARGIN, al: float = DEFAULT_AL,
                 reentry: float = DEFAULT_REENTRY, above: int = 0):
        if not costs:
            raise ValueError("DECODE-TRIM: empty cost table")
        for rows, ms in costs.items():
            if rows <= 0 or not (ms > 0 and math.isfinite(ms)):
                raise ValueError(f"DECODE-TRIM: bad table entry {rows}:{ms}")
        if not (0.0 <= margin < 1.0):
            raise ValueError(f"DECODE-TRIM: margin {margin} outside [0, 1)")
        if not (al >= 1.0 and math.isfinite(al)):
            raise ValueError(f"DECODE-TRIM: al {al} must be >= 1")
        if not (0.0 <= reentry <= 1.0):
            raise ValueError(f"DECODE-TRIM: reentry {reentry} outside [0, 1]")
        if above < 0 or above != int(above):
            raise ValueError(f"DECODE-TRIM: above {above} must be a non-negative integer")
        self.costs = dict(sorted(costs.items()))
        self.keys = list(self.costs)
        self.margin = margin
        self.al = al
        self.reentry = reentry
        self.above = int(above)

    @classmethod
    def parse(cls, raw: str) -> TrimTable:
        if "," in raw or "+" in raw or " " in raw:
            raise ValueError(f"DECODE-TRIM: ',', '+' and spaces are not allowed in {raw!r}")
        costs: dict[int, float] = {}
        opts = {"margin": DEFAULT_MARGIN, "al": DEFAULT_AL, "reentry": DEFAULT_REENTRY, "above": 0.0}
        for item in raw.split("/"):
            key = item.split(":", 1)[0]
            if key in opts:
                opts[key] = float(item[len(key) + 1 :])
                continue
            m = _PAIR.match(item)
            if not m:
                raise ValueError(f"DECODE-TRIM: bad item {item!r} in {raw!r} (want ROWS:MS)")
            rows = int(m.group(1))
            if rows in costs:
                raise ValueError(f"DECODE-TRIM: bucket {rows} given twice in {raw!r}")
            costs[rows] = float(m.group(2))
        return cls(costs, opts["margin"], opts["al"], opts["reentry"], opts["above"])

    def bucket(self, rows: int) -> int | None:
        for k in self.keys:
            if rows <= k:
                return k
        return None

    def choose(self, n: int, rows_per_req: int, must: int = 0) -> int:
        """How many of `n` requests to serve this step (n = no trim). Never fewer than `must`."""
        if n <= 1 or must >= n or n * rows_per_req <= self.above:
            return n
        b = self.bucket(n * rows_per_req)
        if b is None:
            return n
        al, loss = self.al, (self.al - 1.0) * self.reentry
        base = n * al / self.costs[b]
        best_m, best = n, base
        for m in range(n - 1, max(must, 1) - 1, -1):
            score = (m * al - (n - m) * loss) / self.costs[self.bucket(m * rows_per_req)]  # type: ignore[index]
            if score > best:
                best_m, best = m, score
        if best_m < n and best >= base * (1.0 + self.margin):
            return best_m
        return n

    def describe(self) -> str:
        return ("/".join(f"{k}:{v:g}" for k, v in self.costs.items())
                + f" margin {self.margin:g} al {self.al:g} reentry {self.reentry:g} above {self.above}")


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(math.ceil(q * len(s))) - 1 if q > 0 else 0)]


class DecodeTrim:
    """Per-scheduler state: the plan for the current step, rotation and the counters."""

    def __init__(self, sched: Any, table: TrimTable, rows_per_req: int, log_every: int,
                 orig_split: Any = None):
        self.sched = sched
        self.table = table
        self.rows_per_req = max(1, int(rows_per_req))
        self.log_every = log_every
        self.orig_split = orig_split  # the class's own hook, used only if the model needed it
        self.step = 0
        self.skip: frozenset[str] = frozenset()
        self.skipped_prev: set[str] = set()
        self.last_served: dict[str, int] = {}
        self.last_time: dict[str, int] = {}
        self.admit: dict[str, int] = {}
        self.consec: dict[str, int] = {}
        self.skips: dict[str, int] = {}
        self._admit_n = 0
        self.planned_m = 0
        self.planned_n = 0
        self.totals = {"steps": 0, "trimmed": 0, "skipped_req_steps": 0, "unhonored": 0, "max_consec_skip": 0}
        self._reset_window()

    # ---------------------------------------------------------------- window counters
    def _reset_window(self) -> None:
        self.w = {"steps": 0, "trimmed": 0, "skipped_req_steps": 0, "forced": 0, "unhonored": 0, "over": 0,
                  "mixed": 0, "hook_skips": 0, "max_consec_skip": 0}
        self.w_nm: dict[tuple[int, int], int] = {}
        self.w_gap: list[float] = []
        self.w_gap_after_skip: list[float] = []

    # ---------------------------------------------------------------- plan
    def _ready_waiting(self) -> int:
        """Waiting requests that this step will schedule (KV ready); never held back, so counted in n."""
        n = 0
        try:
            from vllm.v1.request import RequestStatus

            recv = getattr(self.sched, "finished_recving_kv_req_ids", None) or ()
            for r in self.sched.waiting:
                st = getattr(r, "status", None)
                if st == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    n += r.request_id in recv
                elif st in (RequestStatus.WAITING, RequestStatus.PREEMPTED) and r.num_computed_tokens > 0:
                    n += 1
        except Exception:  # a count, never a reason to fail the step
            return 0
        return n

    def begin(self) -> None:
        self.step += 1
        self.skip = frozenset()
        self.planned_m = self.planned_n = 0
        running = list(self.sched.running)
        for r in running:
            rid = r.request_id
            if rid not in self.admit:
                self._admit_n += 1
                self.admit[rid] = self._admit_n
                self.last_served[rid] = self.step - 1
        if any(r.num_computed_tokens < r.num_prompt_tokens for r in running):
            self.w["mixed"] += 1  # a prefill chunk in the step: never trimmed
            return
        reserve = self._ready_waiting()
        n = len(running) + reserve
        forced = [r for r in running if r.request_id in self.skipped_prev]
        m = self.table.choose(n, self.rows_per_req, must=len(forced) + reserve)
        self.planned_n, self.planned_m = n, m
        if m >= n:
            return
        keep_others = m - reserve - len(forced)
        others = [r for r in running if r.request_id not in self.skipped_prev]
        others.sort(key=lambda r: (self.last_served[r.request_id], -self.skips.get(r.request_id, 0),
                                   self.admit[r.request_id]))
        self.skip = frozenset(r.request_id for r in others[keep_others:])

    def hook(self, request: Any, num_new_tokens: int, *args: Any, **kwargs: Any) -> int:
        if num_new_tokens > 0 and request.request_id in self.skip:
            self.w["hook_skips"] += 1
            return 0
        if self.orig_split is not None:
            return self.orig_split(request, num_new_tokens, *args, **kwargs)
        return num_new_tokens

    # ---------------------------------------------------------------- bookkeeping
    def end(self, out: Any) -> None:
        now = time.monotonic_ns()
        sched_ids = set(getattr(out, "num_scheduled_tokens", {}) or {})
        honored = self.skip - sched_ids
        unhonored = len(self.skip & sched_ids)
        w = self.w
        w["steps"] += 1
        self.totals["steps"] += 1
        if self.skip:
            w["trimmed"] += 1
            self.totals["trimmed"] += 1
            key = (self.planned_n, self.planned_m)
            self.w_nm[key] = self.w_nm.get(key, 0) + 1
            b = self.table.bucket(self.planned_m * self.rows_per_req)
            if b is not None and int(getattr(out, "total_num_scheduled_tokens", 0) or 0) > b:
                w["over"] += 1
        w["skipped_req_steps"] += len(honored)
        self.totals["skipped_req_steps"] += len(honored)
        w["unhonored"] += unhonored
        self.totals["unhonored"] += unhonored
        w["forced"] += len(self.skipped_prev & sched_ids)
        for rid in sched_ids:
            t0 = self.last_time.get(rid)
            if t0 is not None and len(self.w_gap) < _GAP_CAP:
                gap = (now - t0) / 1e6
                self.w_gap.append(gap)
                if self.consec.get(rid, 0):
                    self.w_gap_after_skip.append(gap)
            self.last_time[rid] = now
            self.last_served[rid] = self.step
            self.consec[rid] = 0
        for rid in honored:
            self.skips[rid] = self.skips.get(rid, 0) + 1
            c = self.consec.get(rid, 0) + 1
            self.consec[rid] = c
            if c > w["max_consec_skip"]:
                w["max_consec_skip"] = c
            if c > self.totals["max_consec_skip"]:
                self.totals["max_consec_skip"] = c
        self.skipped_prev = set(honored)
        self.skip = frozenset()
        if self.step % _PRUNE_EVERY == 0:
            live = set(getattr(self.sched, "requests", {}) or {})
            for d in (self.last_served, self.last_time, self.admit, self.consec, self.skips):
                for rid in [k for k in d if k not in live]:
                    del d[rid]
        if self.log_every and w["steps"] >= self.log_every:
            self.log_window()

    def log_window(self) -> None:
        w = self.w
        g, ga = self.w_gap, self.w_gap_after_skip
        nm = " ".join(f"{n}>{m}:{c}" for (n, m), c in sorted(self.w_nm.items()))
        logger.info(
            "q38 DECODE-TRIM steps=%d trimmed=%d skipped_req_steps=%d forced=%d unhonored=%d over=%d mixed=%d "
            "hook_skips=%d max_consec_skip=%d gap_ms_all p50/p99/max=%.2f/%.2f/%.2f n=%d "
            "gap_ms_after_skip p50/p99/max=%.2f/%.2f/%.2f n=%d nm=%s totals=%s",
            w["steps"], w["trimmed"], w["skipped_req_steps"], w["forced"], w["unhonored"], w["over"], w["mixed"],
            w["hook_skips"], w["max_consec_skip"], _pct(g, 0.5), _pct(g, 0.99), max(g) if g else float("nan"),
            len(g), _pct(ga, 0.5), _pct(ga, 0.99), max(ga) if ga else float("nan"), len(ga), nm or "-",
            self.totals,
        )
        self._reset_window()


def _rows_per_request(sched: Any) -> int:
    k = getattr(sched, "num_spec_tokens", None)
    if k is None:
        spec = getattr(getattr(sched, "vllm_config", None), "speculative_config", None)
        k = getattr(spec, "num_speculative_tokens", 0) if spec is not None else 0
    return 1 + int(k or 0)


def install(sched: Any, table: TrimTable, log_every: int = 2000) -> DecodeTrim | None:
    """Wrap one scheduler instance (instance attributes only). Returns None, with a warning, when the scheduler's
    class overrides the hook (the pipeline chunk scheduler owns it) or the trim is already installed."""
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(sched, "_q38_decode_trim", None) is not None:
        return None
    cls_split = getattr(type(sched), "_mamba_block_aligned_split", None)
    if cls_split is not Scheduler._mamba_block_aligned_split:
        logger.warning(
            "q38 DECODE-TRIM refused: %s overrides _mamba_block_aligned_split", type(sched).__name__
        )
        return None
    need = bool(getattr(sched, "need_mamba_block_aligned_split", False))
    orig_split = sched._mamba_block_aligned_split if need else None
    trim = DecodeTrim(sched, table, _rows_per_request(sched), log_every, orig_split)
    orig_schedule = sched.schedule

    def schedule(*args: Any, **kwargs: Any) -> Any:
        trim.begin()
        try:
            out = orig_schedule(*args, **kwargs)
        except BaseException:
            trim.skip = frozenset()
            raise
        trim.end(out)
        return out

    schedule._q38_decode_trim = True  # type: ignore[attr-defined]
    schedule.__wrapped__ = orig_schedule  # type: ignore[attr-defined]
    sched.schedule = schedule
    sched._mamba_block_aligned_split = trim.hook
    sched.need_mamba_block_aligned_split = True
    sched._q38_decode_trim = trim
    logger.info(
        "q38 DECODE-TRIM on: scheduler %s, rows/request %d, table %s, model hook %s, window %d steps",
        type(sched).__name__, trim.rows_per_req, table.describe(), "kept" if need else "none", log_every,
    )
    return trim


def table_from_env() -> TrimTable | None:
    raw = str(envs.VLLM_TPU_DECODE_TRIM or "").strip()
    if not raw or raw.lower() in ("0", "off", "false"):
        return None
    return TrimTable.parse(raw)


def patch_engine_core_for_decode_trim() -> bool:
    """Install the trim on each engine core's scheduler after EngineCore.__init__. Called from the engine-core
    process entry point; idempotent; a no-op (nothing wrapped) unless VLLM_TPU_DECODE_TRIM is set. A malformed
    table raises here, at engine start."""
    table = table_from_env()
    if table is None:
        return False
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_q38_decode_trim_patch", False):
        return True
    orig_init = EngineCore.__init__
    every = int(envs.VLLM_TPU_DECODE_TRIM_LOG)

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        sched = getattr(self, "scheduler", None)
        if sched is not None:
            install(sched, table, every)

    EngineCore.__init__ = __init__
    EngineCore._q38_decode_trim_patch = True
    return True
