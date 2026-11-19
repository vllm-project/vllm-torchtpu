# SPDX-License-Identifier: Apache-2.0
"""QWEN4_EXP_PD_SYNC_REPACK on CPU: every rank repacks a staged load in the
step the scheduler names it, after the load landed on all ranks.

The served decode repacked on each rank in the step its own bytes landed.
Ranks land one load up to ~0.7 s apart (c48: spread p50 247 ms), and each
rank's repack holds the whole TP group at the next collective, so one load
stalled 2.4-3.5 decode steps. These tests drive the connector's scheduler
and worker halves with fakes: no TPU, no Raiden.
"""

import pickle
from types import SimpleNamespace

from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPUConnectorMetadata
from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import (
    Qwen4ExpRaidenConnector,
    Qwen4ExpRaidenWorker,
    sync_repack_enabled,
    sync_repack_on,
)


class FakeStaging:
    """Records repacks and releases; ``step`` is set by the driver."""

    def __init__(self):
        self.reservations = {}
        self.repacks = []
        self.released = []
        self.step = None

    def reserve(self, uuid):
        self.reservations[uuid] = {"pages": [uuid]}

    def repack(self, uuid, tables, num_tokens):
        assert uuid in self.reservations, "repack after release"
        self.repacks.append((self.step, uuid, [list(t) for t in tables], num_tokens))

    def release(self, uuid):
        self.reservations.pop(uuid, None)
        self.released.append(uuid)


def worker(sync: bool, rank: int = 0) -> Qwen4ExpRaidenWorker:
    w = object.__new__(Qwen4ExpRaidenWorker)
    w._staging = FakeStaging()
    w._stage3_submitted_load_metas = {}
    w._pending_repacks = {}
    w._sync_repack = sync
    w._sync_on = lambda: sync
    w.tp_rank = rank
    return w


def submit(w, req_id, uuid, tables=((1, 2), (7,)), num_tokens=4096):
    w._staging.reserve(uuid)
    w._stage3_submitted_load_metas[req_id] = SimpleNamespace(
        uuid=uuid, group_block_ids=[list(t) for t in tables], num_tokens=num_tokens
    )


class FakeScheduler:
    def __init__(self):
        self.outputs = []

    def update_connector_output(self, out):
        self.outputs.append(out)

    def build_connector_meta(self, scheduler_output):
        return TPUConnectorMetadata()


def scheduler_connector(sync: bool) -> Qwen4ExpRaidenConnector:
    c = object.__new__(Qwen4ExpRaidenConnector)
    c.connector_scheduler = FakeScheduler()
    c.connector_worker = None
    c._sync_repack = sync
    c._landed_on_all_ranks = []
    c._failed_on_some_rank = []
    return c


def out(finished=(), failed=()):
    return SimpleNamespace(
        finished_recving=set(finished), failed_recving=set(failed), finished_sending=None
    )


# --- worker half -------------------------------------------------------------


def test_off_repacks_in_the_landing_step_as_served():
    w = worker(sync=False)
    submit(w, "a", 11)
    w._staging.step = 5
    w._stage3_finish_destination("a", True)
    assert [r[:2] for r in w._staging.repacks] == [(5, 11)]
    assert w._staging.released == [11]
    assert w._pending_repacks == {}


def test_on_defers_the_repack_until_named_and_keeps_the_pages():
    w = worker(sync=True)
    submit(w, "a", 11, tables=((3, 4, 5), (9,)), num_tokens=7000)
    w._staging.step = 5
    w._stage3_finish_destination("a", True)
    assert w._staging.repacks == [] and w._staging.released == []
    assert 11 in w._staging.reservations
    # The metas may be popped by later bookkeeping; the pending record is a
    # snapshot, so the repack still uses the submitted tables.
    w._stage3_submitted_load_metas.clear()
    w._staging.step = 7
    w.repack_landed(["a"])
    assert w._staging.repacks == [(7, 11, [[3, 4, 5], [9]], 7000)]
    assert w._staging.released == [11]
    assert w._pending_repacks == {}


def test_on_cleanup_before_naming_releases_without_repack():
    w = worker(sync=True)
    submit(w, "a", 11)
    w._stage3_finish_destination("a", True)
    w._stage3_finish_destination("a", False)  # aborted / finished
    assert w._staging.repacks == [] and w._staging.released == [11]
    w.repack_landed(["a"])  # a late name is a no-op, never a write
    assert w._staging.repacks == []


def test_on_drop_frees_a_load_that_failed_on_another_rank():
    w = worker(sync=True)
    submit(w, "a", 11)
    w._stage3_finish_destination("a", True)
    w.repack_landed([], drop=["a"])
    assert w._staging.repacks == [] and w._staging.released == [11]


def test_on_cleanup_after_the_repack_is_a_no_op():
    w = worker(sync=True)
    submit(w, "a", 11)
    w._stage3_finish_destination("a", True)
    w.repack_landed(["a"])
    w._stage3_finish_destination("a", False)
    assert len(w._staging.repacks) == 1 and w._staging.released == [11]


# --- scheduler half ----------------------------------------------------------


def test_scheduler_names_each_all_rank_landing_once_and_drops_failures():
    c = scheduler_connector(sync=True)
    c.update_connector_output(out(finished={"a", "b", "c"}, failed={"b"}))
    meta = c.build_connector_meta(None)
    assert meta.qwen_repack_req_ids == ["a", "c"]
    assert meta.qwen_drop_req_ids == ["b"]
    assert c.connector_scheduler.outputs, "the base bookkeeping still runs"
    meta2 = c.build_connector_meta(None)
    assert meta2.qwen_repack_req_ids == [] and meta2.qwen_drop_req_ids == []


def test_scheduler_off_leaves_the_metadata_unchanged():
    c = scheduler_connector(sync=False)
    c.update_connector_output(out(finished={"a"}))
    meta = c.build_connector_meta(None)
    assert not hasattr(meta, "qwen_repack_req_ids")


def test_names_survive_the_executor_pickle():
    c = scheduler_connector(sync=True)
    c.update_connector_output(out(finished={"a"}))
    meta = pickle.loads(pickle.dumps(c.build_connector_meta(None)))
    assert meta.qwen_repack_req_ids == ["a"]


# --- 8 ranks in lockstep -----------------------------------------------------


def run_lockstep(sync: bool, land_step: dict[int, int], steps: int = 12):
    """Rank r's bytes land at step land_step[r]. Each step: the scheduler
    builds the metadata, every worker runs start_load_kv (repacks named
    loads), then get_finished (finish_destination for loads landed by now),
    then the scheduler aggregates (a load is finished when all 8 report)."""
    ranks = range(8)
    workers = [worker(sync, r) for r in ranks]
    conns = []
    for w in workers:
        submit(w, "a", 11)
        w.process_send_load = lambda *a, **k: None  # the base send/load pass
        conn = object.__new__(Qwen4ExpRaidenConnector)
        conn.connector_worker = w
        conn.connector_scheduler = None
        conn._sync_repack = sync
        conns.append(conn)
    sched = scheduler_connector(sync)
    reported = set()
    for step in range(steps):
        meta = pickle.loads(pickle.dumps(sched.build_connector_meta(None)))
        for w, conn in zip(workers, conns):
            w._staging.step = step
            conn._connector_metadata = meta
            conn.start_load_kv(None)
        for r, w in zip(ranks, workers):
            if step == land_step[r]:
                w._stage3_finish_destination("a", True)
                reported.add(r)
        if reported == set(ranks):
            sched.update_connector_output(out(finished={"a"}))
            reported = {-1}
    return [[r[0] for r in w._staging.repacks] for w in workers]


SKEWED = {0: 1, 1: 1, 2: 3, 3: 2, 4: 5, 5: 2, 6: 7, 7: 4}


def test_lockstep_off_repacks_in_as_many_steps_as_landing_steps():
    steps = run_lockstep(False, SKEWED)
    assert sorted({s[0] for s in steps}) == [1, 2, 3, 4, 5, 7]


def test_lockstep_on_every_rank_repacks_once_in_the_same_step():
    steps = run_lockstep(True, SKEWED)
    assert all(len(s) == 1 for s in steps), steps
    assert {s[0] for s in steps} == {8}, steps  # the step after the last rank (7)


# --- wall-clock A/B (QWEN4_EXP_PD_SYNC_REPACK_AB_S) ---------------------------


def test_ab_mode_alternates_by_period(monkeypatch):
    monkeypatch.delenv("QWEN4_EXP_PD_SYNC_REPACK", raising=False)
    monkeypatch.setenv("QWEN4_EXP_PD_SYNC_REPACK_AB_S", "300")
    assert sync_repack_enabled()
    assert [sync_repack_on(t) for t in (0, 299.9, 300, 599.9, 600, 900)] == [
        False, False, True, True, False, True
    ]
    monkeypatch.setenv("QWEN4_EXP_PD_SYNC_REPACK_AB_S", "0")
    assert not sync_repack_enabled() and not sync_repack_on(300)
    monkeypatch.setenv("QWEN4_EXP_PD_SYNC_REPACK", "1")
    assert sync_repack_enabled() and sync_repack_on(0)


def test_ab_off_period_repacks_on_landing_and_flushes_deferred_loads():
    w = worker(sync=True)
    mode = {"on": True}
    w._sync_on = lambda: mode["on"]
    submit(w, "a", 11)
    submit(w, "b", 12)
    w._staging.step = 1
    w._stage3_finish_destination("a", True)  # on period: deferred
    assert w._staging.repacks == []
    mode["on"] = False
    w._staging.step = 2
    w._stage3_finish_destination("b", True)  # off period: repacked on landing
    assert [r[:2] for r in w._staging.repacks] == [(2, 12)]
    w._staging.step = 3
    w.repack_landed([])  # first step of the off period flushes "a" unnamed
    assert [r[:2] for r in w._staging.repacks] == [(2, 12), (3, 11)]
    assert w._pending_repacks == {}
    w.repack_landed(["a", "b"])  # late names are no-ops
    assert len(w._staging.repacks) == 2


def test_sync_repack_knobs_stay_out_of_the_compile_key():
    """Host-side scheduling of an unchanged program: toggling must not recompile."""
    from vllm_torchtpu.compilation.tpu_compiler import (
        _TPU_COMPILE_ENV_IGNORED,
        _tpu_compile_env_factors,
    )

    knobs = {"QWEN4_EXP_PD_SYNC_REPACK", "QWEN4_EXP_PD_SYNC_REPACK_AB_S"}
    assert knobs <= _TPU_COMPILE_ENV_IGNORED
    assert not knobs & set(_tpu_compile_env_factors())
