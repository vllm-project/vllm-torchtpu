# SPDX-License-Identifier: Apache-2.0
"""Per-load landing timeline of a staged Qwen PD decode (QWEN4_EXP_PD_LANDING_LOG=N).

q38 XFER-4. Log-only: host clocks, no device sync, no change to any compiled
program (the env is in tpu_compiler's compile-ignore list). With N=0 (the
default) every hook returns at once.

The landing leg today is one number in the readers, first_chunk - the last
``raiden_stage3_receiver_complete``. These lines split it.

Scheduler (engine core), QWEN4_EXP_PD_SYNC_REPACK only, one line per load at
the first step that schedules it after it landed::

    {"event": "qwen_pd_land_timeline", "req_id", "t_all", "t_named", "t_sched",
     "steps_named_to_sched", "batch", "num_prompt_tokens"}

* ``t_all``: update_connector_output saw the load done on every rank;
* ``t_named``: build_connector_meta named it for the repack;
* ``t_sched``: the first build_connector_meta whose scheduler output schedules
  it (equal to ``t_named`` when it is admitted in the naming step; later when
  the decode is full);
* ``batch``: requests in that step's scheduler output.

Times are ``time.time()`` (the host's wall clock, the same clock as the log
line stamps), so they join with receiver_complete and the proxy's first_chunk.

Worker (TP rank 0), one line per repacked load::

    {"event": "qwen_pd_landing", "req_id", "uuid", "num_tokens", "skip",
     "rows_main", "rows_copy", "gdn_pages", "chunks", "plan_ms", "run_ms"}

``plan_ms`` is the numpy index build, ``run_ms`` the host time to enqueue the
repack programs (including any wait those calls make), so the two sum to the
repack's share of HOST_PHASES' ``kv_connector`` phase. And every N steps::

    {"event": "qwen_pd_kvconn_steps", "rank", "steps", "repack_steps",
     "repack_ms_mean", "repack_ms_p90", "repack_ms_max",
     "load_ms_mean", "load_ms_p90", "load_ms_max"}

which splits ``kv_connector`` into repack_landed and the base start_load_kv
(new loads' submission). Means are per step over all N steps.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Callable, Iterable, Mapping

EVENT_TIMELINE = "qwen_pd_land_timeline"
EVENT_LANDING = "qwen_pd_landing"
EVENT_STEPS = "qwen_pd_kvconn_steps"


def every() -> int:
    """QWEN4_EXP_PD_LANDING_LOG (0 = off)."""
    from vllm_torchtpu import envs

    return int(getattr(envs, "QWEN4_EXP_PD_LANDING_LOG", 0) or 0)


def dumps(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True)


def _q(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))]


class LandingTimeline:
    """Scheduler-side stamps per load: landed on all ranks, named, scheduled.

    Bounded: at most ``cap`` loads are tracked; the oldest is dropped first, so
    a load that never gets scheduled (aborted, recomputed) cannot leak."""

    def __init__(self, cap: int = 4096):
        self.cap = cap
        self.step = 0
        self._loads: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def _entry(self, req_id: str) -> dict[str, Any]:
        entry = self._loads.get(req_id)
        if entry is None:
            entry = self._loads[req_id] = {}
            while len(self._loads) > self.cap:
                self._loads.popitem(last=False)
        return entry

    def landed(self, req_ids: Iterable[str], now: float) -> None:
        for req_id in req_ids:
            self._entry(req_id).setdefault("t_all", now)

    def named(self, req_ids: Iterable[str], now: float, prompt_tokens: Mapping[str, Any] | None = None) -> None:
        for req_id in req_ids:
            entry = self._entry(req_id)
            entry.setdefault("t_named", now)
            entry.setdefault("step_named", self.step)
            if prompt_tokens is not None and prompt_tokens.get(req_id) is not None:
                entry["num_prompt_tokens"] = int(prompt_tokens[req_id])

    def forget(self, req_ids: Iterable[str]) -> None:
        for req_id in req_ids:
            self._loads.pop(req_id, None)

    def scheduled(self, num_scheduled_tokens: Mapping[str, int] | None, now: float) -> list[dict[str, Any]]:
        """One record per named load this step schedules; advances the step."""
        out = []
        scheduled = num_scheduled_tokens or {}
        for req_id in [r for r, e in self._loads.items() if "t_named" in e and r in scheduled]:
            entry = self._loads.pop(req_id)
            out.append(
                {
                    "event": EVENT_TIMELINE,
                    "req_id": req_id,
                    "t_all": entry.get("t_all"),
                    "t_named": entry["t_named"],
                    "t_sched": now,
                    "steps_named_to_sched": self.step - entry["step_named"],
                    "batch": len(scheduled),
                    "num_prompt_tokens": entry.get("num_prompt_tokens"),
                }
            )
        self.step += 1
        return out

    def __len__(self) -> int:
        return len(self._loads)


class KvConnectorSteps:
    """Worker-side split of HOST_PHASES' kv_connector phase, reported every N steps."""

    def __init__(self, every_n: int, rank: int = 0):
        self.every_n = max(1, int(every_n))
        self.rank = rank
        self._repack: list[float] = []
        self._load: list[float] = []

    def add(self, repack_ms: float | None, load_ms: float) -> dict[str, Any] | None:
        """``repack_ms`` None means the step repacked nothing."""
        self._repack.append(-1.0 if repack_ms is None else float(repack_ms))
        self._load.append(float(load_ms))
        if len(self._load) < self.every_n:
            return None
        steps = len(self._load)
        rp = [x for x in self._repack if x >= 0.0]
        record = {
            "event": EVENT_STEPS,
            "rank": self.rank,
            "steps": steps,
            "repack_steps": len(rp),
            "repack_ms_mean": round(sum(rp) / steps, 4),
            "repack_ms_p90": round(_q(rp, 0.9), 3),
            "repack_ms_max": round(max(rp, default=0.0), 3),
            "load_ms_mean": round(sum(self._load) / steps, 4),
            "load_ms_p90": round(_q(self._load, 0.9), 3),
            "load_ms_max": round(max(self._load), 3),
        }
        self._repack, self._load = [], []
        return record


def landing_record(
    req_id: str,
    uuid: int,
    num_tokens: int,
    skip: int,
    rows: tuple[int, int, int],
    chunks: int,
    plan_ms: float,
    run_ms: float,
) -> dict[str, Any]:
    return {
        "event": EVENT_LANDING,
        "req_id": req_id,
        "uuid": int(uuid),
        "num_tokens": int(num_tokens),
        "skip": int(skip),
        "rows_main": int(rows[0]),
        "rows_copy": int(rows[1]),
        "gdn_pages": int(rows[2]),
        "chunks": int(chunks),
        "plan_ms": round(float(plan_ms), 3),
        "run_ms": round(float(run_ms), 3),
    }


def timed(fn: Callable[[], Any], clock: Callable[[], float]) -> tuple[Any, float]:
    """(fn(), elapsed ms) on ``clock`` (seconds)."""
    t0 = clock()
    out = fn()
    return out, (clock() - t0) * 1000.0
