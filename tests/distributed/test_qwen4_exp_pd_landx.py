# SPDX-License-Identifier: Apache-2.0
"""q38 LAND-OVERLAP (QWEN4_EXP_PD_LANDX_DEFER_REPACK) on CPU.

Port of HCDEC-LANDX's tests (q38-hcdec-landx 9be68404d, tests/distributed/test_qwen4_exp_pd_landx.py) to this tree
by BLSTACK-LANDX. Here the key is a plain env bool read at connector build (``_landx``); there is no l6_ab lever, no
ab: window and no engine-core step hook, so those tests are not carried.

With QWEN4_EXP_PD_SYNC_REPACK a staged decode repacks a load that landed on every rank in start_load_kv, before the
forward of the step whose scheduler named it, because that step also admits the request. With the key on, the
scheduler holds the load out of vLLM's finished_recving for one step and names it in qwen_landx_repack_req_ids; every
rank repacks it in submit_deferred_loads (after the forward and sampling dispatch); the scheduler releases it at its
next update, so the request joins one step later and its first forward still follows its repack on every rank.

Checked here, with fakes (no TPU, no Raiden):
  - scheduler: hold, deferred naming, release at the next update; key off = today's metadata and finished_recving and
    no new state; an aborted held load is dropped and still released; the 'q38 LANDX sched' engagement line;
  - worker: deferred names repack in submit_deferred_loads (once), pre-forward names in start_load_kv; names left by a
    step whose post-forward point never ran repack before the next forward; key off = today's calls; the
    'q38 LANDX rank0' engagement line;
  - 8 ranks in lockstep against a model of vLLM's two scheduler rules this relies on: on every rank, every request's
    repack precedes its first forward, key on and off; with the key on the request joins one step later;
  - the real TPUModelRunner hooks (forward step, no-forward step) and the real sample_tokens gate: the post-forward
    submit point runs with DEFER_LOAD_SUBMIT or LANDX;
  - registration (envs registry + compile-key ignore list) and the build-time refusals.
"""

import ast
import inspect
import pickle
import textwrap
from types import SimpleNamespace

import pytest

from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPUConnectorMetadata
from vllm_torchtpu.models.vllm.qwen4_exp import pd_connector as pdc
from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import (
    Qwen4ExpRaidenConnector,
    Qwen4ExpRaidenWorker,
)

PHASE = {"pre": 0, "forward": 1, "post": 2, "poll": 3}


# --- fakes (as tests/distributed/test_qwen4_exp_pd_sync_repack.py) ----------------------------------------------------


class FakeStaging:
    def __init__(self, log=None, rank=0):
        self.reservations = {}
        self.repacks = []
        self.released = []
        self.step = None
        self.phase = None
        self.log = log if log is not None else []
        self.rank = rank

    def reserve(self, uuid):
        self.reservations[uuid] = {"pages": [uuid]}

    def repack(self, uuid, tables, num_tokens):
        assert uuid in self.reservations, "repack after release"
        self.repacks.append((self.step, self.phase, uuid))
        self.log.append(("repack", self.rank, uuid, self.step, self.phase))

    def release(self, uuid):
        self.reservations.pop(uuid, None)
        self.released.append(uuid)


def worker(rank=0, log=None):
    w = object.__new__(Qwen4ExpRaidenWorker)
    w._staging = FakeStaging(log, rank)
    w._stage3_submitted_load_metas = {}
    w._pending_repacks = {}
    w._sync_repack = True
    w._sync_on = lambda: True
    w.tp_rank = rank
    w.process_send_load = lambda *a, **k: None
    return w


def submit(w, req_id, uuid, tables=((1, 2), (7,)), num_tokens=4096):
    w._staging.reserve(uuid)
    w._stage3_submitted_load_metas[req_id] = SimpleNamespace(
        uuid=uuid, group_block_ids=[list(t) for t in tables], num_tokens=num_tokens
    )


def land(w, req_id):
    w._stage3_finish_destination(req_id, True)


def worker_conn(w, landx=True):
    c = object.__new__(Qwen4ExpRaidenConnector)
    c.connector_worker = w
    c.connector_scheduler = None
    c._sync_repack = True
    c._landx = landx
    return c


class FakeScheduler:
    def update_connector_output(self, out):
        pass

    def build_connector_meta(self, scheduler_output):
        return TPUConnectorMetadata()


def scheduler_connector(landx=True):
    c = object.__new__(Qwen4ExpRaidenConnector)
    c.connector_scheduler = FakeScheduler()
    c.connector_worker = None
    c._sync_repack = True
    c._landx = landx
    c._landed_on_all_ranks = []
    c._failed_on_some_rank = []
    return c


def out(finished=(), failed=()):
    return SimpleNamespace(finished_recving=set(finished) or None, failed_recving=set(failed), finished_sending=None)


def meta(repack=(), deferred=None, drop=()):
    m = TPUConnectorMetadata()
    m.qwen_repack_req_ids = list(repack)
    m.qwen_drop_req_ids = list(drop)
    if deferred is not None:
        m.qwen_landx_repack_req_ids = list(deferred)
    return m


def landx_state(c):
    return sorted(k for k in c.__dict__ if k.startswith("_landx") and k != "_landx")


# --- scheduler half --


def test_scheduler_holds_names_deferred_and_releases_at_the_next_update():
    c = scheduler_connector()
    o1 = out(finished={"a", "b", "c"}, failed={"b"})
    c.update_connector_output(o1)
    assert o1.finished_recving == {"b"}, "held loads leave finished_recving; the failed one stays (recomputes)"
    m1 = pickle.loads(pickle.dumps(c.build_connector_meta(None)))
    assert m1.qwen_repack_req_ids == [] and m1.qwen_landx_repack_req_ids == ["a", "c"]
    assert m1.qwen_drop_req_ids == ["b"]
    o2 = out(finished={"d"})
    c.update_connector_output(o2)
    assert o2.finished_recving == {"a", "c"}, "named loads come back; the new landing is held"
    m2 = c.build_connector_meta(None)
    assert m2.qwen_repack_req_ids == [] and m2.qwen_landx_repack_req_ids == ["d"]
    o3 = out()
    c.update_connector_output(o3)
    assert o3.finished_recving == {"d"}
    m3 = c.build_connector_meta(None)
    assert m3.qwen_repack_req_ids == [] and m3.qwen_landx_repack_req_ids == []
    o4 = out()
    c.update_connector_output(o4)
    assert o4.finished_recving is None
    assert c._landx_sched == {"held": 3, "released": 3, "sync": 0, "dropped": 0}


def test_scheduler_two_updates_before_one_schedule_hold_until_named():
    c = scheduler_connector()
    o1, o2 = out(finished={"a"}), out(finished={"b"})
    c.update_connector_output(o1)
    c.update_connector_output(o2)
    assert o1.finished_recving is None and o2.finished_recving is None, "nothing released before it is named"
    m = c.build_connector_meta(None)
    assert m.qwen_landx_repack_req_ids == ["a", "b"]
    o3 = out()
    c.update_connector_output(o3)
    assert o3.finished_recving == {"a", "b"}


def test_scheduler_key_off_is_todays_sync_repack():
    c = scheduler_connector(landx=False)
    o1 = out(finished={"a", "c"}, failed=set())
    c.update_connector_output(o1)
    assert o1.finished_recving == {"a", "c"}
    m = c.build_connector_meta(None)
    assert m.qwen_repack_req_ids == ["a", "c"]
    assert not hasattr(m, "qwen_landx_repack_req_ids"), "no new metadata field with the key off"
    assert landx_state(c) == [], "no LAND-OVERLAP state with the key off"


class Req:
    def __init__(self, finished=False):
        from vllm.v1.request import RequestStatus

        self.status = RequestStatus.FINISHED_ABORTED if finished else RequestStatus.WAITING_FOR_REMOTE_KVS
        self._finished = finished

    def is_finished(self):
        return self._finished


def test_scheduler_aborted_while_held_is_dropped_and_still_released():
    c = scheduler_connector()
    c._load_requests = {"a": Req(), "c": Req()}
    c.update_connector_output(out(finished={"a", "c"}))
    c._load_requests["a"] = Req(finished=True)  # aborted after landing, before the metadata went out
    m = c.build_connector_meta(None)
    assert m.qwen_landx_repack_req_ids == ["c"] and m.qwen_drop_req_ids == ["a"]
    o2 = out()
    c.update_connector_output(o2)
    # vLLM frees an aborted remote-KV request's blocks only when its id reaches finished_recving.
    assert o2.finished_recving == {"a", "c"}
    assert c._landx_sched["dropped"] == 1


def test_scheduler_engagement_line_first_hold_then_every_n(monkeypatch):
    lines = []
    monkeypatch.setattr(pdc.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    monkeypatch.setattr(pdc, "_LANDX_SCHED_EVERY", 3)
    c = scheduler_connector()
    for i in range(7):
        c.update_connector_output(out(finished={"r%d" % i}))
        c.build_connector_meta(None)
    got = [x for x in lines if x.startswith("q38 LANDX sched ")]
    assert got == [
        "q38 LANDX sched held=1 released=0 sync=0 dropped=0 holding=1",
        "q38 LANDX sched held=3 released=2 sync=0 dropped=0 holding=1",
        "q38 LANDX sched held=6 released=5 sync=0 dropped=0 holding=1",
    ], got


# --- worker half --


def test_worker_deferred_names_repack_after_the_forward_once():
    w = worker()
    c = worker_conn(w)
    for r, u in (("a", 11), ("b", 12)):
        submit(w, r, u)
        land(w, r)
    c._connector_metadata = meta(repack=["b"], deferred=["a"])
    w._staging.step, w._staging.phase = 3, "pre"
    c.start_load_kv(None)
    assert w._staging.repacks == [(3, "pre", 12)], "only the pre-forward name repacks before the forward"
    w._staging.phase = "post"
    assert c.submit_deferred_loads() is None
    assert w._staging.repacks == [(3, "pre", 12), (3, "post", 11)]
    w._staging.phase = "poll"
    c.submit_deferred_loads()
    assert w._staging.repacks == [(3, "pre", 12), (3, "post", 11)], "once per step"
    assert w._staging.released == [12, 11] and w._pending_repacks == {}


def test_worker_names_left_by_a_step_without_its_post_point_repack_before_the_next_forward():
    w = worker()
    c = worker_conn(w)
    submit(w, "a", 11)
    land(w, "a")
    c._connector_metadata = meta(deferred=["a"])
    w._staging.step, w._staging.phase = 3, "pre"
    c.start_load_kv(None)
    assert w._staging.repacks == []
    # step 3's post-forward point never ran; step 4 begins
    c._connector_metadata = meta(deferred=[])
    w._staging.step = 4
    c.start_load_kv(None)
    assert w._staging.repacks == [(4, "pre", 11)]
    assert c._landx_counts["fallback_loads"] == 1
    c.submit_deferred_loads()
    assert w._staging.repacks == [(4, "pre", 11)]


def test_worker_key_off_is_todays_calls():
    w = worker()
    c = worker_conn(w, landx=False)
    for r, u in (("a", 11), ("b", 12)):
        submit(w, r, u)
        land(w, r)
    c._connector_metadata = meta(repack=["a", "b"])
    w._staging.step, w._staging.phase = 3, "pre"
    c.start_load_kv(None)
    w._staging.phase = "post"
    c.submit_deferred_loads()
    assert w._staging.repacks == [(3, "pre", 11), (3, "pre", 12)]
    assert landx_state(c) == [], "no LAND-OVERLAP state with the key off"


def test_worker_engagement_line_reports_the_share_of_steps(monkeypatch):
    lines = []
    monkeypatch.setattr(pdc.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    monkeypatch.setattr(pdc, "_LANDX_STATS_EVERY", 4)
    w = worker()
    c = worker_conn(w)
    submit(w, "a", 11)
    land(w, "a")
    for step, m in enumerate((meta(deferred=["a"]), meta(deferred=[]), meta(deferred=[]), meta(deferred=[]))):
        w._staging.step = step
        c._connector_metadata = m
        c.start_load_kv(None)
        c.submit_deferred_loads()
    got = [x for x in lines if x.startswith("q38 LANDX rank0 ")]
    assert got == ["q38 LANDX rank0 steps=4 deferred_steps=1 share=0.2500 deferred_loads=1 post_loads=1 "
                   "fallback_loads=0 presync_loads=0 post_repack_ms_mean=%s" % got[0].rsplit("=", 1)[1]], got


# --- 8 ranks in lockstep against vLLM's scheduler rules --


class MiniVllm:
    """The two vLLM v0.31 scheduler facts this relies on (vllm/v1/core/sched/scheduler.py):
    _update_from_kv_xfer_finished calls connector.update_connector_output BEFORE it reads finished_recving, and
    _handle_blocked_waiting_request promotes a WAITING_FOR_REMOTE_KVS request only once its id is in
    finished_recving_kv_req_ids; schedule() promotes first and builds the connector metadata last."""

    def __init__(self, conn):
        self.conn = conn
        self.ready = set()
        self.running = set()

    def update(self, o):
        self.conn.update_connector_output(o)
        self.ready |= set(o.finished_recving or ())

    def schedule(self):
        new = sorted(self.ready - self.running)
        self.running |= set(new)
        return set(self.running), self.conn.build_connector_meta(SimpleNamespace(num_scheduled_tokens={}))


def run_lockstep(land_steps, steps=14, no_forward=(), landx=True):
    """land_steps: request -> {rank: step its bytes land on that rank}. Per step: the scheduler promotes and builds
    the metadata; every rank runs start_load_kv (pre), the forward (records its batch), submit_deferred_loads (post)
    and the poll; the aggregator reports a load once all 8 ranks have; the scheduler updates."""
    log = []
    ranks = range(8)
    ws = [worker(r, log) for r in ranks]
    conns = [worker_conn(w, landx) for w in ws]
    for i, req in enumerate(sorted(land_steps)):
        for w in ws:
            submit(w, req, 100 + i)
    sched = MiniVllm(scheduler_connector(landx))
    reported = {req: set() for req in land_steps}
    done = set()
    first_fwd = {}
    for step in range(steps):
        batch, m = sched.schedule()
        m = pickle.loads(pickle.dumps(m))
        for w, c in zip(ws, conns):
            w._staging.step = step
            c._connector_metadata = m
            w._staging.phase = "pre"
            c.start_load_kv(None)
            if step not in no_forward:
                log.append(("forward", w.tp_rank, sorted(batch), step, "forward"))
                for req in batch:
                    first_fwd.setdefault((req, w.tp_rank), step)
                w._staging.phase = "post"
                c.submit_deferred_loads()
            w._staging.phase = "poll"
            c.submit_deferred_loads()  # the poll's call (get_finished_kv_transfers)
            for req, at in land_steps.items():
                if at[w.tp_rank] == step:
                    land(w, req)
                    reported[req].add(w.tp_rank)
        fin = {req for req, rs in reported.items() if rs == set(ranks) and req not in done}
        done |= fin
        sched.update(out(finished=fin))
    return log, first_fwd


SKEWED = {"a": {0: 1, 1: 1, 2: 3, 3: 2, 4: 5, 5: 2, 6: 7, 7: 4}, "b": {r: 7 for r in range(8)},
          "c": {r: 2 + (r % 3) for r in range(8)}}


def _repack_before_first_forward(log, first_fwd):
    uuid = {req: 100 + i for i, req in enumerate(sorted(SKEWED))}
    for (req, rank), fstep in first_fwd.items():
        reps = [(e[3], PHASE[e[4]]) for e in log if e[0] == "repack" and e[1] == rank and e[2] == uuid[req]]
        assert len(reps) == 1, (req, rank, reps)
        assert reps[0] < (fstep, PHASE["forward"]), (req, rank, reps[0], fstep)


@pytest.mark.parametrize("on", [False, True])
def test_lockstep_every_request_reads_its_repack_on_every_rank(on):
    log, first = run_lockstep(SKEWED, landx=on)
    assert {req for req, _ in first} == set(SKEWED), "every request is scheduled"
    _repack_before_first_forward(log, first)
    uuid = {req: 100 + i for i, req in enumerate(sorted(SKEWED))}
    for req in SKEWED:
        reps = {(e[3], e[4]) for e in log if e[0] == "repack" and e[2] == uuid[req]}
        assert len(reps) == 1, "all 8 ranks repack a load in the same step and phase: %s" % reps
        (rstep, phase), = reps
        f = {first[(req, r)] for r in range(8)}
        assert len(f) == 1
        if on:
            assert phase == "post" and f == {rstep + 1}, (req, rstep, phase, f)
        else:
            assert phase == "pre" and f == {rstep}, (req, rstep, phase, f)


def test_lockstep_no_forward_steps_repack_in_the_poll():
    log, first = run_lockstep(SKEWED, no_forward={4, 5, 8})
    _repack_before_first_forward(log, first)
    uuid = {req: 100 + i for i, req in enumerate(sorted(SKEWED))}
    phases = {req: {e[4] for e in log if e[0] == "repack" and e[2] == uuid[req]} for req in SKEWED}
    assert phases["a"] == {"poll"} and phases["b"] == {"poll"}, phases  # named in no-forward step 8


# --- the real runner hooks --


def test_real_runner_hooks_place_the_deferred_repack_after_the_forward_dispatch(monkeypatch):
    from vllm_torchtpu.runner import tpu_runner
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    log = []
    w = worker(0, log)
    w.get_finished = lambda finished_req_ids: (set(), set())
    w.get_block_ids_with_load_errors = lambda: set()
    w.last_failed_recving = set()
    c = worker_conn(w)
    c._connector_metadata = None
    c.handle_preemptions = lambda m: None
    c.build_connector_worker_meta = lambda: None
    c.get_kv_connector_stats = lambda: None
    c.get_block_ids_with_load_errors_group_index = lambda: None
    monkeypatch.setattr(tpu_runner, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(tpu_runner, "get_kv_transfer_group", lambda: c)
    monkeypatch.setattr(tpu_runner.dist_utils, "get_raiden_inline_load", lambda: False)

    class Runner:
        maybe_setup_kv_connector = TPUModelRunner.maybe_setup_kv_connector
        _submit_deferred_kv_loads = TPUModelRunner._submit_deferred_kv_loads
        get_finished_kv_transfers = TPUModelRunner.get_finished_kv_transfers
        kv_connector_no_forward = TPUModelRunner.kv_connector_no_forward
        _stage_timing = SimpleNamespace(take_samples=lambda: (None, None))

    runner = Runner()
    for i, req in enumerate(("a", "b", "c", "d")):
        submit(w, req, 11 + i)
        land(w, req)
    steps = [("forward", meta(repack=["a"], deferred=["b"])), ("no_forward", meta(deferred=["c"])),
             ("forward", meta(deferred=["d"])), ("forward", meta(deferred=[]))]
    for step, (kind, m) in enumerate(steps):
        w._staging.step = step
        so = SimpleNamespace(kv_connector_metadata=m, finished_req_ids=set())
        w._staging.phase = "pre"
        if kind == "forward":
            runner.maybe_setup_kv_connector(so)
            log.append(("forward_dispatch", step))
            w._staging.phase = "post"
            runner._submit_deferred_kv_loads()  # sample_tokens' call, after the forward and sampling dispatch
            log.append(("prev_results_wait", step))
            w._staging.phase = "poll"
            runner.get_finished_kv_transfers(so)
        else:
            w._staging.phase = "poll"
            runner.kv_connector_no_forward(so, None)
        log.append(("poll_done", step))
    reps = [(e[2], e[3], e[4]) for e in log if e[0] == "repack"]
    assert reps == [(11, 0, "pre"), (12, 0, "post"), (13, 1, "poll"), (14, 2, "post")], reps
    i = log.index(("forward_dispatch", 0))
    j = log.index(("repack", 0, 12, 0, "post"))
    assert log.index(("repack", 0, 11, 0, "pre")) < i < j < log.index(("prev_results_wait", 0))


def _sample_tokens_submit_gate():
    """The test expression of the `if` in the real sample_tokens whose body calls self._submit_deferred_kv_loads()."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    tree = ast.parse(textwrap.dedent(inspect.getsource(TPUModelRunner.sample_tokens)))
    gates = []
    for node in ast.walk(tree):
        # The innermost `if`: the call is a statement of its own body.
        if isinstance(node, ast.If) and any(
            isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == "_submit_deferred_kv_loads"
            for stmt in node.body
        ):
            gates.append(node.test)
    assert len(gates) == 1, len(gates)
    return compile(ast.Expression(gates[0]), "<sample_tokens gate>", "eval")


@pytest.mark.parametrize("defer,landx,want", [(0, 0, False), (1, 0, True), (0, 1, True), (1, 1, True)])
def test_sample_tokens_post_forward_submit_gate_serves_landx_without_defer_load_submit(defer, landx, want):
    gate = _sample_tokens_submit_gate()
    envs = SimpleNamespace(QWEN4_EXP_PD_DEFER_LOAD_SUBMIT=bool(defer), QWEN4_EXP_PD_LANDX_DEFER_REPACK=bool(landx))
    assert bool(eval(gate, {"envs": envs})) is want


# --- registration, refusals --


def test_key_is_registered_and_compile_ignored(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED, _tpu_compile_env_factors

    assert "QWEN4_EXP_PD_LANDX_DEFER_REPACK" in envs.environment_variables
    assert "QWEN4_EXP_PD_LANDX_DEFER_REPACK" in _TPU_COMPILE_ENV_IGNORED
    monkeypatch.delenv("QWEN4_EXP_PD_LANDX_DEFER_REPACK", raising=False)
    off = _tpu_compile_env_factors()
    monkeypatch.setenv("QWEN4_EXP_PD_LANDX_DEFER_REPACK", "1")
    assert envs.QWEN4_EXP_PD_LANDX_DEFER_REPACK is True and pdc.landx_enabled()
    on = _tpu_compile_env_factors()
    assert "QWEN4_EXP_PD_LANDX_DEFER_REPACK" not in on and on == off


def test_refusals(monkeypatch):
    monkeypatch.delenv("QWEN4_EXP_PD_SYNC_REPACK_AB_S", raising=False)
    assert pdc.landx_refusals(True, False) == []
    assert any("QWEN4_EXP_PD_SYNC_REPACK=1" in x for x in pdc.landx_refusals(False, False))
    assert any("SHARED_COMPRESSED" in x for x in pdc.landx_refusals(True, True))
    monkeypatch.setenv("QWEN4_EXP_PD_SYNC_REPACK_AB_S", "300")
    assert any("SYNC_REPACK_AB_S" in x for x in pdc.landx_refusals(True, False))


def test_connector_build_raises_on_a_refused_setting():
    src = inspect.getsource(Qwen4ExpRaidenConnector.__init__)
    assert "landx_refusals(self._sync_repack, self._shared_compressed)" in src
    assert 'raise ValueError("QWEN4_EXP_PD_LANDX_DEFER_REPACK is refused: "' in src
    assert "self._landx = landx_enabled() and not (transfer is not None and transfer.is_kv_producer)" in src
