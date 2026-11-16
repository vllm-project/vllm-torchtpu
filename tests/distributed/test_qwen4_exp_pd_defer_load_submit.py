# SPDX-License-Identifier: Apache-2.0
"""QWEN4_EXP_PD_DEFER_LOAD_SUBMIT on CPU: the Stage-3 load submit moves behind the
step's forward dispatch and nothing else changes.

Four gloo ranks drive the real runner hooks
(``TPUModelRunner.maybe_setup_kv_connector``, ``_submit_deferred_kv_loads``,
``get_finished_kv_transfers``, ``kv_connector_no_forward``) and the real
``Qwen4ExpRaidenConnector.start_load_kv`` / ``submit_deferred_loads`` over a fake
connector worker. The worker's submit mirrors ``_submit_stage3_loads``: parked
loads first, then the step's new ones, one CPU-group all-reduce (MIN of the ranks'
free staging pages, as the batched admission agrees), a load that does not fit
parks, a ``fail_only`` load records a receive failure; ``repack_landed`` runs its own
all-reduce like ``agree_named_loads``. Steps: a forward step with loads, a forward
step that only lands (no loads), a step without a forward, a step whose admission
parks a load (resubmitted next step), and a submit-time failure.

On every rank, with the flag on and off: the same submit calls, the same admission
outcomes, the same collectives in the same order, every failure in the same step's
poll; with the flag on, the submit runs after the forward dispatch (forward steps)
or inside the poll (no-forward steps), exactly once a step. A structural check pins
the call order in the real ``sample_tokens`` source.
"""

import inspect
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch.multiprocessing as mp

WORLD = 4


class FakeWorker:
    """The connector worker's submit/poll surface with real CPU-group collectives."""

    is_producer = False

    def __init__(self, rank, group, log):
        self.rank = rank
        self.group = group
        self.log = log
        self._staging = None
        # Rank-local free staging pages; the all-reduce agrees on the minimum.
        self.free = 10 + rank
        self.waits = {}
        self.done = set()
        self.failed = set()
        self.last_failed_recving = set()
        self.submits = 0

    def _collective(self, name, value):
        import torch
        import torch.distributed as dist

        t = torch.tensor([value], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=self.group)
        self.log.append(("collective", name, int(t.item())))
        return int(t.item())

    def repack_landed(self, req_ids, drop=()):
        if req_ids:
            self._collective("agree_named_loads", len(req_ids))
            self.log.append(("repack", tuple(req_ids)))

    def process_send_load(
        self, metadata, wait_for_completion=False, report_completion=True
    ):
        self.submits += 1
        loads = list(self.waits.items()) + [
            (d, m) for d, m in metadata.reqs_to_load.items() if d not in self.waits
        ]
        if not loads:
            self.log.append(("submit", ()))
            return
        agreed = self._collective("batched_admission", self.free)
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
        m.reqs_to_load[d] = SimpleNamespace(
            pages=100 if d in park else 1, fail_only=d in fail
        )
    m.qwen_repack_req_ids = list(repacks)
    return m


# (kind, metadata kwargs, freed pages before the step). "park" loads need 100 pages;
# "free" before a step frees pages so the parked load fits on its retry.
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
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenConnector
    from vllm_torchtpu.runner import tpu_runner
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    log = []
    worker = FakeWorker(rank, group, log)
    conn = object.__new__(Qwen4ExpRaidenConnector)
    conn.connector_worker = worker
    conn._sync_repack = True
    conn._defer_load_submit = defer
    conn._deferred_load_submit = None
    conn._connector_metadata = None
    conn.handle_preemptions = lambda meta: None
    conn.build_connector_worker_meta = lambda: None
    conn.get_kv_connector_stats = lambda: None
    conn.get_block_ids_with_load_errors_group_index = lambda: None

    tpu_runner.has_kv_transfer_group = lambda: True
    tpu_runner.get_kv_transfer_group = lambda: conn
    tpu_runner.dist_utils.get_raiden_inline_load = lambda: False
    os.environ["QWEN4_EXP_PD_DEFER_LOAD_SUBMIT"] = "1" if defer else "0"

    class Runner:
        maybe_setup_kv_connector = TPUModelRunner.maybe_setup_kv_connector
        _submit_deferred_kv_loads = TPUModelRunner._submit_deferred_kv_loads
        get_finished_kv_transfers = TPUModelRunner.get_finished_kv_transfers
        kv_connector_no_forward = TPUModelRunner.kv_connector_no_forward
        _stage_timing = SimpleNamespace(take_samples=lambda: (None, None))

    runner = Runner()
    polls = []
    step = 0
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
            if envs.QWEN4_EXP_PD_DEFER_LOAD_SUBMIT:  # sample_tokens, as served
                runner._submit_deferred_kv_loads()
            log.append(("prev_results_wait",))
            res = runner.get_finished_kv_transfers(so)
        else:
            out = runner.kv_connector_no_forward(so, None)
            res = (
                None,
                out.kv_connector_output.finished_recving
                if out.kv_connector_output
                else None,
            )
        assert worker.submits == before + 1, (kind, step, worker.submits - before)
        assert conn._deferred_load_submit is None
        polls.append((step, sorted(res[1] or ()), sorted(runner._kv_failed_recving)))
        log.append(("poll", polls[-1]))
    return log, polls


def _rank(rank, init_file, q):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
    )
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


def test_same_submits_collectives_and_polls_with_the_flag_on(runs):
    for _, out, _ in runs:
        off_log, off_polls = out[False]
        on_log, on_polls = out[True]
        keep = ("step", "submit", "collective", "repack", "poll")
        assert _only(on_log, keep) == _only(off_log, keep)
        assert on_polls == off_polls


def test_the_submit_moves_behind_the_forward_dispatch(runs):
    for _, out, _ in runs:
        for defer, log in ((False, out[False][0]), (True, out[True][0])):
            steps, cur = {}, None
            for e in log:
                if e[0] == "step":
                    cur = steps.setdefault(e[1], [])
                elif e[0] in (
                    "submit",
                    "forward_dispatch",
                    "prev_results_wait",
                    "poll",
                    "repack",
                ):
                    cur.append(e[0])
            for step, seq in steps.items():
                kind = [k for k, _ in STEPS if k != "free"][step - 1]
                assert seq.count("submit") == 1, (defer, step, seq)
                if kind == "no_forward":
                    assert seq.index("submit") < seq.index("poll")
                    continue
                if "repack" in seq:
                    assert seq.index("repack") < seq.index("forward_dispatch")
                if defer:
                    assert (
                        seq.index("forward_dispatch")
                        < seq.index("submit")
                        < seq.index("prev_results_wait")
                    ), seq
                else:
                    assert seq.index("submit") < seq.index("forward_dispatch"), seq


def test_park_retry_and_failures_land_in_the_expected_steps(runs):
    log, polls = runs[0][1][True]
    submits = [e[1] for e in log if e[0] == "submit"]
    assert submits[3] == (("d", "submitted"), ("big", "parked"))
    assert submits[4] == (("big", "parked"), ("e", "submitted"))
    assert submits[7] == (("big", "submitted"),)
    # A submit-time failure reaches the poll of its own step, forward or not.
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
    assert src.index("self._submit_deferred_kv_loads()") < src.index(
        "kv_connector.clear_connector_metadata()"
    )


def test_inline_mode_and_producer_never_defer():
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenConnector

    calls = []
    conn = object.__new__(Qwen4ExpRaidenConnector)
    conn.connector_worker = SimpleNamespace(
        _staging=None,
        process_send_load=lambda meta, **kw: calls.append(kw),
    )
    conn._sync_repack = False
    conn._defer_load_submit = True
    conn._deferred_load_submit = None
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPUConnectorMetadata

    conn._connector_metadata = TPUConnectorMetadata()
    conn.start_load_kv(None, wait_for_completion=True, report_completion=False)
    assert calls == [{"wait_for_completion": True, "report_completion": False}]
    assert conn._deferred_load_submit is None
    conn.start_load_kv(None)
    assert len(calls) == 1 and conn._deferred_load_submit is not None
    conn.submit_deferred_loads()
    conn.submit_deferred_loads()
    assert len(calls) == 2 and conn._deferred_load_submit is None
