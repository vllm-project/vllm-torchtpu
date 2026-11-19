# SPDX-License-Identifier: Apache-2.0
"""q38 L6-S3ASYNC S3-DEFER-AR through the real runner hooks on four CPU gloo ranks.

Adapted from Benliang Wang's tests/distributed/test_qwen4_exp_pd_defer_load_submit.py
(benliangw/q38fh-integ-pin-5b44702aea, commit 09b6990cb), credited: the same fake worker,
steps and checks, with two changes for this lever: the switch is lever s3defer (read in
start_load_kv), and the worker's s3d_issue ISSUES the batched admission all-reduce
asynchronously in start_load_kv (as tpu_connector.s3d_issue does), which the deferred
submit then waits for.

The real TPUModelRunner.maybe_setup_kv_connector / _submit_deferred_kv_loads /
get_finished_kv_transfers / kv_connector_no_forward and the real
Qwen4ExpRaidenConnector.start_load_kv / submit_deferred_loads drive a fake worker whose
submit mirrors _submit_stage3_loads (parked loads first, one CPU-group MIN all-reduce of
the ranks' free staging pages, a load that does not fit parks, fail_only fails;
repack_landed runs its own all-reduce). On every rank, lever on and off: the same submit
decisions, the same collectives with the same values in the same order, every failure in
the same step's poll; with the lever on, the admission collective is issued before the
forward dispatch and the submit runs after it (forward steps) or inside the poll
(no-forward steps), exactly once a step. A structural check pins the call order in the
real sample_tokens source.
"""

import inspect
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch.multiprocessing as mp

WORLD = 4


class FakeWorker:
    """The connector worker's issue / submit / poll surface with real CPU-group collectives."""

    is_producer = False

    def __init__(self, rank, group, log):
        self.rank = rank
        self.group = group
        self.log = log
        self._staging = None
        self.free = 10 + rank  # rank-local free staging pages; the all-reduce agrees on the minimum
        self.waits = {}
        self.done = set()
        self.failed = set()
        self.last_failed_recving = set()
        self.submits = 0
        self.issued = None

    def _loads(self, metadata):
        return list(self.waits.items()) + [(d, m) for d, m in metadata.reqs_to_load.items() if d not in self.waits]

    def _collective(self, name, value):
        import torch
        import torch.distributed as dist

        t = torch.tensor([value], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=self.group)
        self.log.append(("collective", name, int(t.item())))
        return int(t.item())

    def s3d_issue(self, metadata):
        import torch
        import torch.distributed as dist

        loads = [x for x in self._loads(metadata) if not x[1].fail_only]
        if loads:
            t = torch.tensor([self.free], dtype=torch.int64)
            self.issued = (t, dist.all_reduce(t, op=dist.ReduceOp.MIN, group=self.group, async_op=True))
            self.log.append(("issue",))
        return True

    def repack_landed(self, req_ids, drop=()):
        if req_ids:
            self._collective("agree_named_loads", len(req_ids))
            self.log.append(("repack", tuple(req_ids)))

    def process_send_load(self, metadata, wait_for_completion=False, report_completion=True):
        self.submits += 1
        loads = self._loads(metadata)
        if not loads:
            self.log.append(("submit", ()))
            return
        if any(not m.fail_only for _, m in loads):
            if self.issued is not None:
                t, work = self.issued
                self.issued = None
                work.wait()
                agreed = int(t.item())
                self.log.append(("collective", "batched_admission", agreed))
            else:
                agreed = self._collective("batched_admission", self.free)
        else:
            agreed = 0
        out = []
        for dest, meta in loads:
            if meta.fail_only:
                self.failed.add(dest)
                self.waits.pop(dest, None)
                out.append((dest, "failed"))
            elif meta.pages <= agreed:
                agreed -= meta.pages
                self.free -= meta.pages
                self.waits.pop(dest, None)
                self.done.add(dest)
                out.append((dest, "submitted"))
            else:
                self.waits[dest] = meta
                out.append((dest, "parked"))
        self.log.append(("submit", tuple(out)))

    def get_finished(self, finished_req_ids):
        done, self.done = self.done, set()
        self.last_failed_recving, self.failed = self.failed, set()
        return set(), done

    def get_block_ids_with_load_errors(self):
        return set()


def _meta(loads=(), repacks=(), park=(), fail=()):
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPUConnectorMetadata

    m = TPUConnectorMetadata()
    for d in loads:
        m.reqs_to_load[d] = SimpleNamespace(pages=100 if d in park else 1, fail_only=d in fail)
    m.qwen_repack_req_ids = list(repacks)
    return m


STEPS = [
    ("forward", dict(loads=("a", "b"))),
    ("forward", dict(repacks=("a", "b"))),  # lands only: no new loads
    ("no_forward", dict(loads=("c",))),
    ("forward", dict(loads=("d", "big"), park=("big",))),  # big parks
    ("forward", dict(loads=("e",))),  # big is retried first, parks again
    ("no_forward", dict(loads=("f",), fail=("f",))),  # submit-time failure
    ("forward", dict(loads=("g",), fail=("g",))),
    ("free", dict()),  # pages freed (not a step)
    ("forward", dict(loads=())),  # big retried with room: submitted
]


def _drive(rank, defer, group):
    from vllm_torchtpu.distributed.kv_transfer import stage3_async as s3a
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenConnector
    from vllm_torchtpu.runner import tpu_runner
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    log = []
    worker = FakeWorker(rank, group, log)
    conn = object.__new__(Qwen4ExpRaidenConnector)
    conn.connector_worker = worker
    conn._sync_repack = True
    conn._s3d_deferred = None
    conn._connector_metadata = None
    conn.handle_preemptions = lambda meta: None
    conn.build_connector_worker_meta = lambda: None
    conn.get_kv_connector_stats = lambda: None
    conn.get_block_ids_with_load_errors_group_index = lambda: None

    tpu_runner.has_kv_transfer_group = lambda: True
    tpu_runner.get_kv_transfer_group = lambda: conn
    tpu_runner.dist_utils.get_raiden_inline_load = lambda: False
    s3a.DEFER_CONFIGURED = True
    s3a.defer_on = lambda: defer

    class Runner:
        maybe_setup_kv_connector = TPUModelRunner.maybe_setup_kv_connector
        _submit_deferred_kv_loads = TPUModelRunner._submit_deferred_kv_loads
        get_finished_kv_transfers = TPUModelRunner.get_finished_kv_transfers
        kv_connector_no_forward = TPUModelRunner.kv_connector_no_forward
        _stage_timing = SimpleNamespace(take_samples=lambda: (None, None))

    runner = Runner()
    polls, step = [], 0
    for kind, kw in STEPS:
        if kind == "free":
            worker.free += 200
            continue
        step += 1
        log.append(("step", step))
        so = SimpleNamespace(kv_connector_metadata=_meta(**kw), finished_req_ids=set())
        before = worker.submits
        if kind == "forward":
            runner.maybe_setup_kv_connector(so)  # execute_model, before the chunk loop
            log.append(("forward_dispatch",))
            runner._submit_deferred_kv_loads()  # sample_tokens, as served (a no-op unless deferred)
            log.append(("prev_results_wait",))
            res = runner.get_finished_kv_transfers(so)
        else:
            out = runner.kv_connector_no_forward(so, None)
            res = (None, out.kv_connector_output.finished_recving if out.kv_connector_output else None)
        assert worker.submits == before + 1, (kind, step, worker.submits - before)
        assert conn._s3d_deferred is None and worker.issued is None
        polls.append((step, sorted(res[1] or ()), sorted(runner._kv_failed_recving)))
        log.append(("poll", polls[-1]))
    return log, polls


def _rank(rank, init_file, q):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
    try:
        out = {}
        for defer in (False, True):
            dist.barrier()
            out[defer] = _drive(rank, defer, dist.group.WORLD)
        every = [None] * WORLD
        dist.all_gather_object(every, out)
        q.put((rank, out, every))
    except Exception as exc:  # pragma: no cover - surfaced through the queue
        import traceback

        q.put((rank, None, traceback.format_exc() + repr(exc)))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def runs():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        init = os.path.join(d, "init")
        procs = [ctx.Process(target=_rank, args=(r, init, q)) for r in range(WORLD)]
        for p in procs:
            p.start()
        res = [q.get(timeout=300) for _ in procs]
        for p in procs:
            p.join(timeout=60)
    res = sorted(res, key=lambda r: r[0])
    for rank, out, info in res:
        assert out is not None, info
    return res


def _only(log, kinds):
    return [e for e in log if e[0] in kinds]


def test_every_rank_logs_the_same_sequence(runs):
    for defer in (False, True):
        ref = runs[0][1][defer][0]
        for rank, out, _ in runs:
            assert out[defer][0] == ref, (defer, rank)


def test_same_submits_collectives_and_polls_with_the_lever_on(runs):
    for _, out, _ in runs:
        off_log, off_polls = out[False]
        on_log, on_polls = out[True]
        keep = ("step", "submit", "collective", "repack", "poll")
        assert _only(on_log, keep) == _only(off_log, keep)
        assert on_polls == off_polls


def test_issue_before_the_forward_submit_after_it(runs):
    for _, out, _ in runs:
        for defer, log in ((False, out[False][0]), (True, out[True][0])):
            steps, cur = {}, None
            for e in log:
                if e[0] == "step":
                    cur = steps.setdefault(e[1], [])
                elif e[0] in ("submit", "forward_dispatch", "prev_results_wait", "poll", "repack", "issue"):
                    cur.append(e[0])
            for step, seq in steps.items():
                kind = [k for k, _ in STEPS if k != "free"][step - 1]
                assert seq.count("submit") == 1, (defer, step, seq)
                # Lever on: the admission is issued in every step whose pass has a load to admit
                # (all but step 2, which only lands); lever off: never.
                assert ("issue" in seq) == (defer and step != 2), (defer, step, seq)
                if kind == "no_forward":
                    assert seq.index("submit") < seq.index("poll")
                    continue
                if "repack" in seq:
                    assert seq.index("repack") < seq.index("forward_dispatch")
                if defer:
                    if "issue" in seq:
                        assert seq.index("issue") < seq.index("forward_dispatch"), seq
                    assert seq.index("forward_dispatch") < seq.index("submit") < seq.index("prev_results_wait"), seq
                else:
                    assert "issue" not in seq
                    assert seq.index("submit") < seq.index("forward_dispatch"), seq


def test_park_retry_and_failures_land_in_the_expected_steps(runs):
    log, polls = runs[0][1][True]
    submits = [e[1] for e in log if e[0] == "submit"]
    assert submits[3] == (("d", "submitted"), ("big", "parked"))
    assert submits[4] == (("big", "parked"), ("e", "submitted"))
    assert submits[7] == (("big", "submitted"),)
    assert polls[5] == (6, [], ["f"])
    assert polls[6] == (7, [], ["g"])
    assert polls[0] == (1, ["a", "b"], [])


def test_sample_tokens_submits_before_the_wait_and_the_poll():
    """The served call order in the real source: deferred submit, then the
    previous-results wait, then the poll that clears the metadata."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    src = inspect.getsource(TPUModelRunner.sample_tokens)
    submit = src.index("self._submit_deferred_kv_loads()")
    wait = src.index("self._modify_prev_results()", submit)
    poll = src.index("self.get_finished_kv_transfers(scheduler_output)", submit)
    assert submit < wait < poll
    src = inspect.getsource(TPUModelRunner.get_finished_kv_transfers)
    assert src.index("self._submit_deferred_kv_loads()") < src.index("kv_connector.clear_connector_metadata()")


def test_inline_mode_never_defers(monkeypatch):
    from vllm_torchtpu.distributed.kv_transfer import stage3_async as s3a
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPUConnectorMetadata
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenConnector

    calls = []
    conn = object.__new__(Qwen4ExpRaidenConnector)
    conn.connector_worker = SimpleNamespace(
        _staging=None,
        process_send_load=lambda meta, **kw: calls.append(kw),
        s3d_issue=lambda meta: calls.append("issue") or True,
    )
    conn._sync_repack = False
    conn._s3d_deferred = None
    conn._connector_metadata = TPUConnectorMetadata()
    monkeypatch.setattr(s3a, "DEFER_CONFIGURED", True)
    monkeypatch.setattr(s3a, "defer_on", lambda: True)
    conn.start_load_kv(None, wait_for_completion=True, report_completion=False)
    assert calls == [{"wait_for_completion": True, "report_completion": False}] and conn._s3d_deferred is None
    conn.start_load_kv(None)
    assert calls[-1] == "issue" and conn._s3d_deferred is not None
    assert conn.submit_deferred_loads() is True and conn.submit_deferred_loads() is False
