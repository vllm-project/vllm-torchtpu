# SPDX-License-Identifier: Apache-2.0
"""QWEN4_EXP_PD_LANDING_LOG on CPU (q38 XFER-4): the staged decode's per-load
landing timeline is log-only.

Off (the default): no hook does anything and the repack call is exactly
today's ``staging.repack``. On: the scheduler stamps each load landed on all
ranks / named / first scheduled, TP rank 0 logs each repack's host halves,
and every N steps the split of HOST_PHASES' kv_connector phase. Fakes only:
no TPU, no Raiden.
"""

import json
from types import SimpleNamespace

import pytest

from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPUConnectorMetadata,
    TPURaidenConnector,
)
from vllm_torchtpu.models.vllm.qwen4_exp import pd_connector as pc
from vllm_torchtpu.models.vllm.qwen4_exp import pd_landing_log as ll
from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import (
    Qwen4ExpRaidenConnector,
    Qwen4ExpRaidenWorker,
)


class Lines:
    """Captures pd_connector's logger.info JSON lines."""

    def __init__(self):
        self.records = []

    def info(self, fmt, *args):
        text = fmt % args if args else fmt
        if text.startswith("{"):
            self.records.append(json.loads(text))

    def warning(self, *a, **k):
        pass

    def of(self, event):
        return [r for r in self.records if r.get("event") == event]


@pytest.fixture
def lines(monkeypatch):
    cap = Lines()
    monkeypatch.setattr(pc, "logger", cap)
    return cap


class FakeStaging:
    def __init__(self):
        self.reservations = {}
        self.skips = {}
        self.calls = []
        self.released = []

    def reserve(self, uuid, skip=0):
        self.reservations[uuid] = {"pages": [uuid]}
        self.skips[uuid] = (skip, None)

    def repack(self, uuid, tables, num_tokens):
        self.calls.append(("repack", uuid))

    def plan_repack(self, uuid, tables, num_tokens):
        assert uuid in self.reservations, "plan after release"
        self.calls.append(("plan", uuid))
        self.last_plan_rows = (num_tokens - self.skips[uuid][0], 7, 3)
        return SimpleNamespace(passes=[], chunks=[0, 1])

    def run_repack(self, plan):
        self.calls.append(("run", len(plan.chunks)))

    def release(self, uuid):
        self.reservations.pop(uuid, None)
        self.released.append(uuid)


def worker(rank=0):
    w = object.__new__(Qwen4ExpRaidenWorker)
    w._staging = FakeStaging()
    w._stage3_submitted_load_metas = {}
    w._pending_repacks = {}
    w._sync_repack = True
    w._sync_on = lambda: True
    w.tp_rank = rank
    return w


def land(w, req_id, uuid, num_tokens=9000, skip=2048):
    w._staging.reserve(uuid, skip)
    w._stage3_submitted_load_metas[req_id] = SimpleNamespace(
        uuid=uuid, group_block_ids=[[1, 2], [3]], num_tokens=num_tokens
    )
    w._stage3_finish_destination(req_id, True)


class FakeScheduler:
    def update_connector_output(self, out):
        pass

    def build_connector_meta(self, scheduler_output):
        return TPUConnectorMetadata()


def scheduler_connector():
    c = object.__new__(Qwen4ExpRaidenConnector)
    c.connector_scheduler = FakeScheduler()
    c.connector_worker = None
    c._sync_repack = True
    c._landed_on_all_ranks = []
    c._failed_on_some_rank = []
    return c


def out(finished=(), failed=()):
    return SimpleNamespace(
        finished_recving=set(finished), failed_recving=set(failed), finished_sending=None
    )


def sched(*req_ids):
    return SimpleNamespace(num_scheduled_tokens={r: 1 for r in req_ids})


# --- the helper ---------------------------------------------------------------


def test_timeline_admitted_in_the_naming_step():
    t = ll.LandingTimeline()
    t.landed(["a"], 1.0)
    t.named(["a"], 2.0, {"a": 100000})
    recs = t.scheduled({"a": 1, "z": 4}, 2.0)
    assert recs == [
        {
            "event": ll.EVENT_TIMELINE, "req_id": "a", "t_all": 1.0, "t_named": 2.0,
            "t_sched": 2.0, "steps_named_to_sched": 0, "batch": 2,
            "num_prompt_tokens": 100000,
        }
    ]
    assert len(t) == 0


def test_timeline_queued_after_naming_counts_steps():
    t = ll.LandingTimeline()
    assert t.scheduled({}, 0.1) == [] and t.scheduled({}, 0.2) == []  # steps before the load
    t.landed(["a"], 1.0)
    t.named(["a"], 2.0)
    assert t.scheduled({"z": 1}, 2.0) == []
    assert t.scheduled({}, 3.0) == []
    (rec,) = t.scheduled({"a": 1}, 4.0)
    assert (rec["t_named"], rec["t_sched"], rec["steps_named_to_sched"]) == (2.0, 4.0, 2)


def test_timeline_unnamed_loads_are_not_reported_and_restamps_are_ignored():
    t = ll.LandingTimeline()
    t.landed(["a"], 1.0)
    t.landed(["a"], 5.0)
    assert t.scheduled({"a": 1}, 2.0) == []  # landed, not yet named
    t.named(["a"], 6.0)
    (rec,) = t.scheduled({"a": 1}, 7.0)
    assert rec["t_all"] == 1.0


def test_timeline_is_bounded_and_forgets():
    t = ll.LandingTimeline(cap=2)
    t.landed(["a", "b", "c"], 1.0)
    assert len(t) == 2
    t.forget(["b", "nope"])
    assert len(t) == 1


def test_kvconn_steps_reports_every_n():
    s = ll.KvConnectorSteps(3, rank=0)
    assert s.add(None, 1.0) is None
    assert s.add(4.0, 2.0) is None
    rec = s.add(8.0, 3.0)
    assert rec["steps"] == 3 and rec["repack_steps"] == 2
    assert rec["repack_ms_mean"] == pytest.approx(4.0)  # (4 + 8) / 3 steps
    assert rec["repack_ms_max"] == 8.0
    assert rec["load_ms_mean"] == pytest.approx(2.0) and rec["load_ms_max"] == 3.0
    assert s.add(None, 1.0) is None  # reset after a report


def test_timed_uses_the_clock():
    ticks = iter([10.0, 10.25])
    out_, ms = ll.timed(lambda: "x", lambda: next(ticks))
    assert out_ == "x" and ms == pytest.approx(250.0)


# --- off: exactly today's behaviour --------------------------------------------


def test_off_by_default_no_lines_and_plain_repack(monkeypatch, lines):
    monkeypatch.delenv("QWEN4_EXP_PD_LANDING_LOG", raising=False)
    c = scheduler_connector()
    c.update_connector_output(out(finished={"a"}))
    c.build_connector_meta(sched("a"))
    assert c._landing_timeline() is None
    w = worker()
    land(w, "a", 11)
    w.repack_landed(["a"])
    assert w._staging.calls == [("repack", 11)]
    assert w._kvconn_steps() is None
    assert lines.records == []


# --- on -------------------------------------------------------------------------


def test_on_scheduler_stamps_landed_named_scheduled(monkeypatch, lines):
    monkeypatch.setenv("QWEN4_EXP_PD_LANDING_LOG", "5")
    clock = iter([100.0, 101.0, 102.0, 103.0])
    monkeypatch.setattr(pc.time, "time", lambda: next(clock))
    c = scheduler_connector()
    c.update_connector_output(out(finished={"a", "b"}))  # 100.0
    meta = c.build_connector_meta(sched("a", "x"))  # 101.0: both named, only a scheduled
    assert meta.qwen_repack_req_ids == ["a", "b"]
    meta = c.build_connector_meta(sched("b", "x"))  # 102.0
    recs = lines.of(ll.EVENT_TIMELINE)
    assert [(r["req_id"], r["t_all"], r["t_named"], r["t_sched"], r["steps_named_to_sched"], r["batch"]) for r in recs] == [
        ("a", 100.0, 101.0, 101.0, 0, 2),
        ("b", 100.0, 101.0, 102.0, 1, 2),
    ]


def test_on_failed_loads_are_never_reported(monkeypatch, lines):
    monkeypatch.setenv("QWEN4_EXP_PD_LANDING_LOG", "5")
    c = scheduler_connector()
    c.update_connector_output(out(finished={"a"}, failed={"a"}))
    c.build_connector_meta(sched("a"))
    assert lines.of(ll.EVENT_TIMELINE) == []


def test_on_rank0_repack_is_plan_then_run_with_one_line(monkeypatch, lines):
    monkeypatch.setenv("QWEN4_EXP_PD_LANDING_LOG", "5")
    w = worker(rank=0)
    land(w, "a", 11, num_tokens=9000, skip=2048)
    w.repack_landed(["a"])
    assert w._staging.calls == [("plan", 11), ("run", 2)]
    assert w._staging.released == [11]
    (rec,) = lines.of(ll.EVENT_LANDING)
    assert (rec["req_id"], rec["uuid"], rec["num_tokens"], rec["skip"]) == ("a", 11, 9000, 2048)
    assert (rec["rows_main"], rec["rows_copy"], rec["gdn_pages"], rec["chunks"]) == (6952, 7, 3, 2)
    assert rec["plan_ms"] >= 0 and rec["run_ms"] >= 0


def test_on_other_ranks_repack_plainly_and_log_nothing(monkeypatch, lines):
    monkeypatch.setenv("QWEN4_EXP_PD_LANDING_LOG", "5")
    w = worker(rank=3)
    land(w, "a", 11)
    w.repack_landed(["a"])
    assert w._staging.calls == [("repack", 11)]
    assert lines.records == []


def test_on_start_load_kv_splits_repack_and_base_every_n(monkeypatch, lines):
    monkeypatch.setenv("QWEN4_EXP_PD_LANDING_LOG", "2")
    base_calls = []
    monkeypatch.setattr(TPURaidenConnector, "start_load_kv", lambda self, *a, **k: base_calls.append(a))
    w = worker(rank=0)
    land(w, "a", 11)
    c = object.__new__(Qwen4ExpRaidenConnector)
    c.connector_worker = w
    c._sync_repack = True
    c._connector_metadata = SimpleNamespace(qwen_repack_req_ids=["a"], qwen_drop_req_ids=[])
    c.start_load_kv("ctx")
    c._connector_metadata = SimpleNamespace(qwen_repack_req_ids=[], qwen_drop_req_ids=[])
    c.start_load_kv("ctx")
    assert base_calls == [("ctx",), ("ctx",)]
    (rec,) = lines.of(ll.EVENT_STEPS)
    assert (rec["steps"], rec["repack_steps"]) == (2, 1)
    assert len(lines.of(ll.EVENT_LANDING)) == 1


def test_landing_log_stays_out_of_the_compile_key():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import (
        _TPU_COMPILE_ENV_IGNORED,
        _tpu_compile_env_factors,
    )

    assert "QWEN4_EXP_PD_LANDING_LOG" in envs.environment_variables
    assert "QWEN4_EXP_PD_LANDING_LOG" in _TPU_COMPILE_ENV_IGNORED
    assert "QWEN4_EXP_PD_LANDING_LOG" not in set(_tpu_compile_env_factors())
