# SPDX-License-Identifier: Apache-2.0
"""QWEN4_EXP_PD_EARLY_POLL on CPU (q38 DEC-STEP-2): a PD consumer polls its KV
connector right after the step's deferred submit, before the previous-results
wait, and nothing else changes.

The real runner hooks (``TPUModelRunner.maybe_setup_kv_connector``,
``_submit_deferred_kv_loads``, ``_early_poll_kv_connector``,
``get_finished_kv_transfers``, ``kv_connector_no_forward``) drive the real
``Qwen4ExpRaidenConnector`` worker half (``start_load_kv`` /
``submit_deferred_loads``, LAND-OVERLAP included) over a fake connector worker
whose landings follow a fixed per-step schedule. With the key on and off: the same
per-step KV connector outputs, the same submits and repacks, one metadata clear
and one poll per step; with the key on the poll runs before the wait, with it off
after the drafter. The structural checks pin the call order in the real
``sample_tokens`` source and the unchanged stock poll block.
"""

import ast
import inspect
import logging
import textwrap
from types import SimpleNamespace

import pytest

KEY = "QWEN4_EXP_PD_EARLY_POLL"


# --- fakes -------------------------------------------------------------------


class FakeWorker:
    """The connector worker's submit / poll surface. ``schedule`` maps a step to
    the loads that land by that step's poll; ``clock`` is advanced by the test."""

    def __init__(self, log, schedule, failed=None):
        self.log = log
        self.schedule = schedule
        self.failed_by_step = failed or {}
        self.clock = 0
        self.reported = set()
        self._staging = None
        self.last_failed_recving = set()
        self._pending_repacks = {}

    def process_send_load(self, meta, **kw):
        self.log.append(("submit", self.clock))

    def repack_landed(self, names, drop=()):
        if names or drop:
            self.log.append(("repack", self.clock, tuple(names), tuple(drop)))

    def get_finished(self, finished_req_ids):
        landed = set()
        for step, reqs in self.schedule.items():
            if step <= self.clock:
                landed |= set(reqs)
        new = landed - self.reported
        self.reported |= new
        self.last_failed_recving = set(self.failed_by_step.get(self.clock, ()))
        self.log.append(("poll", self.clock, tuple(sorted(new))))
        return set(), new | self.last_failed_recving


def make_connector(log, schedule, failed=None):
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenConnector

    w = FakeWorker(log, schedule, failed)
    c = object.__new__(Qwen4ExpRaidenConnector)
    c.connector_worker, c.connector_scheduler = w, None
    c._sync_repack = True
    c._defer_load_submit = True
    c._deferred_load_submit = None
    c._landx = False
    c._connector_metadata = None
    c.handle_preemptions = lambda m: None

    def bind(m):
        c._connector_metadata = m

    def clear():
        log.append(("clear", w.clock))
        c._connector_metadata = None

    c.bind_connector_metadata = bind
    c.clear_connector_metadata = clear
    c.build_connector_worker_meta = lambda: None
    c.get_kv_connector_stats = lambda: None
    c.get_block_ids_with_load_errors = lambda: set()
    c.get_block_ids_with_load_errors_group_index = lambda: None
    return c, w


def consumer_config():
    return SimpleNamespace(is_kv_consumer=True, is_kv_producer=False)


def make_runner(monkeypatch, connector, key_on, transfer=None):
    from vllm_torchtpu.runner import tpu_runner
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    if key_on:
        monkeypatch.setenv(KEY, "1")
    else:
        monkeypatch.delenv(KEY, raising=False)
    monkeypatch.setattr(tpu_runner, "has_kv_transfer_group", lambda: connector is not None)
    monkeypatch.setattr(tpu_runner, "get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(tpu_runner.dist_utils, "get_raiden_inline_load", lambda: False)

    class Runner:
        maybe_setup_kv_connector = TPUModelRunner.maybe_setup_kv_connector
        _submit_deferred_kv_loads = TPUModelRunner._submit_deferred_kv_loads
        maybe_wait_for_kv_save = TPUModelRunner.maybe_wait_for_kv_save
        get_finished_kv_transfers = TPUModelRunner.get_finished_kv_transfers
        kv_connector_no_forward = TPUModelRunner.kv_connector_no_forward
        _kv_early_poll = TPUModelRunner._kv_early_poll
        _decide_kv_early_poll = TPUModelRunner._decide_kv_early_poll
        _early_poll_kv_connector = TPUModelRunner._early_poll_kv_connector
        _stage_timing = SimpleNamespace(take_samples=lambda: (None, None))

    r = Runner()
    r.vllm_config = SimpleNamespace(kv_transfer_config=transfer or consumer_config())
    return r


def meta():
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPUConnectorMetadata

    m = TPUConnectorMetadata()
    m.qwen_repack_req_ids = []
    m.qwen_drop_req_ids = []
    return m


def canon(out):
    fs, fr, wm, inv, grp, stats = out
    return (tuple(sorted(fs or ())), tuple(sorted(fr or ())), wm, tuple(sorted(inv or ())), grp, stats)


def run_steps(monkeypatch, key_on, kinds, schedule, advance_on_wait=False, failed=None):
    """The connector calls of execute_model + sample_tokens in served order
    (sample_tokens' order as pinned by test_sample_tokens_source_order)."""
    log = []
    c, w = make_connector(log, schedule, failed)
    r = make_runner(monkeypatch, c, key_on)
    outs = []
    for step, kind in enumerate(kinds):
        w.clock = step
        so = SimpleNamespace(kv_connector_metadata=meta(), finished_req_ids=set())
        if kind == "forward":
            r.maybe_setup_kv_connector(so)
            log.append(("forward_dispatch", step))
            r._submit_deferred_kv_loads()  # the DEFER_LOAD_SUBMIT / LANDX gate
            if r._kv_early_poll():
                r._early_poll_kv_connector(so)
            log.append(("prev_results_wait", step))
            if advance_on_wait:
                w.clock = step + 0.5  # a load may land while the host waits
            log.append(("draft_propose", step))
            r.maybe_wait_for_kv_save()
            outs.append(canon(r.get_finished_kv_transfers(so)))
        else:
            kco = getattr(r.kv_connector_no_forward(so, None), "kv_connector_output", None)
            outs.append(("no_forward", tuple(sorted((kco.finished_recving or ()) if kco is not None else ()))))
        log.append(("step_done", step))
    return log, outs, w


# --- registry / refusals / decision -------------------------------------------


def test_key_is_registered_and_compile_ignored(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert KEY in envs.environment_variables
    monkeypatch.delenv(KEY, raising=False)
    assert envs.QWEN4_EXP_PD_EARLY_POLL is False
    monkeypatch.setenv(KEY, "1")
    assert envs.QWEN4_EXP_PD_EARLY_POLL is True
    assert KEY in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def _connectors():
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPURaidenConnector
    from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import TPUMultiConnector
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenConnector

    class Saving(Qwen4ExpRaidenConnector):
        def wait_for_save(self):
            pass

    return {
        "qwen": object.__new__(Qwen4ExpRaidenConnector),
        "raiden": object.__new__(TPURaidenConnector),
        "multi": object.__new__(TPUMultiConnector),
        "saving": object.__new__(Saving),
        "other": object(),
    }


@pytest.mark.parametrize(
    "transfer,conn,want",
    [
        ((True, False), "qwen", []),
        ((True, False), "raiden", []),
        ((False, True), "qwen", ["producer"]),
        ((True, True), "qwen", ["producer"]),
        (None, "qwen", ["producer"]),
        ((True, False), "multi", ["Raiden PD consumer connector"]),
        ((True, False), "other", ["Raiden PD consumer connector"]),
        ((True, False), "saving", ["save path"]),
    ],
)
def test_refusals_accept_only_a_raiden_pd_consumer(transfer, conn, want):
    from vllm_torchtpu.runner.tpu_runner import kv_early_poll_refusals

    cfg = None if transfer is None else SimpleNamespace(is_kv_consumer=transfer[0], is_kv_producer=transfer[1])
    why = kv_early_poll_refusals(cfg, _connectors()[conn])
    assert len(why) == len(want), why
    for w, text in zip(why, want):
        assert text in w, why


def test_decision_off_is_cached_and_silent(monkeypatch, caplog):
    c, _ = make_connector([], {})
    r = make_runner(monkeypatch, c, key_on=False)
    calls = []
    real = type(r)._decide_kv_early_poll
    monkeypatch.setattr(type(r), "_decide_kv_early_poll", lambda self: calls.append(1) or real(self))
    with caplog.at_level(logging.INFO):
        assert r._kv_early_poll() is False
        assert r._kv_early_poll() is False
    assert calls == [1]
    assert "q38 EARLY_POLL" not in caplog.text


def test_decision_on_engages_once_with_its_line(monkeypatch, caplog):
    c, _ = make_connector([], {})
    r = make_runner(monkeypatch, c, key_on=True)
    calls = []
    real = type(r)._decide_kv_early_poll
    monkeypatch.setattr(type(r), "_decide_kv_early_poll", lambda self: calls.append(1) or real(self))
    caplog.set_level(logging.INFO)
    assert r._kv_early_poll() is True
    assert r._kv_early_poll() is True
    assert calls == [1]
    assert "q38 EARLY_POLL engaged=1 connector=Qwen4ExpRaidenConnector" in caplog.text


def test_decision_on_without_a_connector_is_not_engaged(monkeypatch, caplog):
    r = make_runner(monkeypatch, None, key_on=True)
    caplog.set_level(logging.INFO)
    assert r._kv_early_poll() is False
    assert "q38 EARLY_POLL engaged=0 reason=no KV connector" in caplog.text


def test_decision_on_a_producer_raises(monkeypatch):
    c, _ = make_connector([], {})
    r = make_runner(monkeypatch, c, key_on=True, transfer=SimpleNamespace(is_kv_consumer=False, is_kv_producer=True))
    with pytest.raises(ValueError, match="QWEN4_EXP_PD_EARLY_POLL is refused: only on a KV consumer"):
        r._kv_early_poll()


# --- hand-over -------------------------------------------------------------------


def test_early_poll_hands_its_result_to_the_stock_site_once(monkeypatch):
    log = []
    c, w = make_connector(log, {0: ["a"], 1: ["b"]})
    r = make_runner(monkeypatch, c, key_on=True)
    so = SimpleNamespace(kv_connector_metadata=meta(), finished_req_ids=set())
    r.maybe_setup_kv_connector(so)
    r._early_poll_kv_connector(so)
    first = r.get_finished_kv_transfers(so)
    assert [e[0] for e in log].count("poll") == 1
    assert [e[0] for e in log].count("clear") == 1
    assert canon(first)[1] == ("a",)
    # Handed over once: the next call (the next step) polls the connector again.
    w.clock = 1
    r.maybe_setup_kv_connector(SimpleNamespace(kv_connector_metadata=meta(), finished_req_ids=set()))
    second = r.get_finished_kv_transfers(so)
    assert [e[0] for e in log].count("poll") == 2
    assert canon(second)[1] == ("b",)
    assert "_kv_early_poll_result" not in r.__dict__


def test_key_off_never_parks_a_result(monkeypatch):
    log = []
    c, _ = make_connector(log, {0: ["a"]})
    r = make_runner(monkeypatch, c, key_on=False)
    so = SimpleNamespace(kv_connector_metadata=meta(), finished_req_ids=set())
    r.maybe_setup_kv_connector(so)
    assert r._kv_early_poll() is False
    r.get_finished_kv_transfers(so)
    assert "_kv_early_poll_result" not in r.__dict__
    assert [e[0] for e in log].count("poll") == 1


# --- served order, on vs off -------------------------------------------------------


KINDS = ["forward", "forward", "no_forward", "forward", "forward", "forward"]
SCHEDULE = {0: ["a"], 1: ["b", "c"], 2: ["d"], 4: ["e"]}
FAILED = {3: ["x"]}


def _step_events(log, step, names):
    return [e[0] for e in log if len(e) > 1 and e[1] == step and e[0] in names]


def test_outputs_submits_and_clears_are_the_same_on_and_off(monkeypatch):
    log_off, outs_off, _ = run_steps(monkeypatch, False, KINDS, SCHEDULE, failed=FAILED)
    log_on, outs_on, _ = run_steps(monkeypatch, True, KINDS, SCHEDULE, failed=FAILED)
    assert outs_on == outs_off
    for kind in ("submit", "repack", "clear", "poll"):
        assert [e for e in log_on if e[0] == kind] == [e for e in log_off if e[0] == kind], kind
    # One poll and one metadata clear per step.
    assert len([e for e in log_on if e[0] == "poll"]) == len(KINDS)
    assert len([e for e in log_on if e[0] == "clear"]) == len(KINDS)


def test_on_polls_before_the_wait_and_off_after_the_drafter(monkeypatch):
    log_off, _, _ = run_steps(monkeypatch, False, KINDS, SCHEDULE)
    log_on, _, _ = run_steps(monkeypatch, True, KINDS, SCHEDULE)
    names = ("forward_dispatch", "submit", "poll", "prev_results_wait", "draft_propose")
    for step, kind in enumerate(KINDS):
        if kind != "forward":
            continue
        on = [e for e in log_on if e[0] in names and e[1] == step]
        off = [e for e in log_off if e[0] in names and e[1] == step]
        assert [e[0] for e in on] == ["forward_dispatch", "submit", "poll", "prev_results_wait", "draft_propose"], on
        assert [e[0] for e in off] == ["forward_dispatch", "submit", "prev_results_wait", "draft_propose", "poll"], off


def test_a_load_landing_during_the_wait_is_reported_at_the_next_poll_not_lost(monkeypatch):
    sched = {0.5: ["late"], 2: ["z"]}
    _, outs_off, _ = run_steps(monkeypatch, False, ["forward"] * 3, sched, advance_on_wait=True)
    _, outs_on, _ = run_steps(monkeypatch, True, ["forward"] * 3, sched, advance_on_wait=True)
    got_off = [o[1] for o in outs_off]
    got_on = [o[1] for o in outs_on]
    assert got_off == [("late",), (), ("z",)]
    assert got_on == [(), ("late",), ("z",)]


# --- the real sample_tokens -------------------------------------------------------


def _sample_tokens_src():
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    return textwrap.dedent(inspect.getsource(TPUModelRunner.sample_tokens))


def test_sample_tokens_source_order():
    src = _sample_tokens_src()
    submit = src.index("self._submit_deferred_kv_loads()")
    early = src.index("self._early_poll_kv_connector(scheduler_output)")
    wait = src.index("self._modify_prev_results()", early)
    stock = src.index("self.get_finished_kv_transfers(scheduler_output)")
    assert submit < early < wait < stock
    assert src.count("self._early_poll_kv_connector(") == 1
    # The early call is the whole body of `if self._kv_early_poll():` (besides its host phase mark).
    tree = ast.parse(src)
    gates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Call)
            and isinstance(s.value.func, ast.Attribute)
            and s.value.func.attr == "_early_poll_kv_connector"
            for s in node.body
        )
    ]
    assert len(gates) == 1
    t = gates[0].test
    assert isinstance(t, ast.Call) and isinstance(t.func, ast.Attribute) and t.func.attr == "_kv_early_poll"


def test_stock_poll_block_is_unchanged():
    src = _sample_tokens_src()
    block = textwrap.dedent(
        """\
        self.maybe_wait_for_kv_save()
        (
            finished_sending,
            finished_recving,
            kv_worker_meta,
            invalid_block_ids,
            invalid_block_group_index,
            kv_connector_stats,
        ) = self.get_finished_kv_transfers(scheduler_output)
        """
    )
    flat = "\n".join(line.strip() for line in src.splitlines())
    want = "\n".join(line.strip() for line in block.splitlines())
    assert flat.count(want) == 1


def test_early_poll_dispatches_nothing_but_the_save_and_the_poll():
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    tree = ast.parse(textwrap.dedent(inspect.getsource(TPUModelRunner._early_poll_kv_connector)))
    calls = sorted(
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    )
    assert calls == ["get_finished_kv_transfers", "maybe_wait_for_kv_save"]
