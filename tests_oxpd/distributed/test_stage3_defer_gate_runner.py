# SPDX-License-Identifier: Apache-2.0
"""q38 L1-S3DGATE: toggling lever s3defer between steps (as the S3D load gate does at window boundaries) leaves no
deferred submit outstanding at any step boundary, on four CPU gloo ranks.

Reuses test_stage3_defer_ar_runner.py's rig (L6-S3ASYNC, itself adapted from Benliang Wang's
test_qwen4_exp_pd_defer_load_submit.py, 09b6990cb, credited): the real TPUModelRunner.maybe_setup_kv_connector /
_submit_deferred_kv_loads / get_finished_kv_transfers / kv_connector_no_forward and the real
Qwen4ExpRaidenConnector.start_load_kv / submit_deferred_loads over the same fake worker and steps. Here the lever's
state flips between steps in several patterns (on-off-on..., runs, all on, all off): after every step nothing is
deferred and nothing is issued; every step submits exactly once; the submits, collectives and polls equal the
all-off run's; every rank logs the same. Structural checks pin why a window toggle cannot strand a deferred submit
in the served runner: execute_model refuses to start while a step's sample_tokens is pending (before begin_step,
where the gate's window states are fixed); after maybe_setup_kv_connector, execute_model returns only after
recording execute_model_state (or on a non-last PP stage); sample_tokens runs _submit_deferred_kv_loads before any
return that follows a recorded state.
"""

import inspect
import os
import re
import tempfile
from types import SimpleNamespace

import pytest
import torch.multiprocessing as mp

from . import test_stage3_defer_ar_runner as R

WORLD = R.WORLD
N_STEPS = sum(1 for k, _ in R.STEPS if k != "free")
PATTERNS = {
    "off": [False] * N_STEPS,
    "alt": [i % 2 == 0 for i in range(N_STEPS)],
    "alt2": [i % 2 == 1 for i in range(N_STEPS)],
    "runs": [True, True, False, False, True, False, True, True][:N_STEPS],
    "on": [True] * N_STEPS,
}


def _drive(rank, pattern, group):
    from vllm_torchtpu.distributed.kv_transfer import stage3_async as s3a
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenConnector
    from vllm_torchtpu.runner import tpu_runner
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    log = []
    worker = R.FakeWorker(rank, group, log)
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
    state = {"on": False}
    s3a.defer_on = lambda: state["on"]

    class Runner:
        maybe_setup_kv_connector = TPUModelRunner.maybe_setup_kv_connector
        _submit_deferred_kv_loads = TPUModelRunner._submit_deferred_kv_loads
        get_finished_kv_transfers = TPUModelRunner.get_finished_kv_transfers
        kv_connector_no_forward = TPUModelRunner.kv_connector_no_forward
        _stage_timing = SimpleNamespace(take_samples=lambda: (None, None))

    runner = Runner()
    polls, step, bad = [], 0, []
    for kind, kw in R.STEPS:
        if kind == "free":
            worker.free += 200
            continue
        state["on"] = pattern[step]  # the lever's state for this step (a window boundary may fall between any two)
        step += 1
        log.append(("step", step))
        so = SimpleNamespace(kv_connector_metadata=R._meta(**kw), finished_req_ids=set())
        before = worker.submits
        if kind == "forward":
            runner.maybe_setup_kv_connector(so)
            log.append(("forward_dispatch",))
            runner._submit_deferred_kv_loads()
            log.append(("prev_results_wait",))
            res = runner.get_finished_kv_transfers(so)
        else:
            out = runner.kv_connector_no_forward(so, None)
            res = (None, out.kv_connector_output.finished_recving if out.kv_connector_output else None)
        if worker.submits != before + 1:
            bad.append((step, "submits", worker.submits - before))
        if conn._s3d_deferred is not None or worker.issued is not None:
            bad.append((step, "outstanding at the step boundary"))
        polls.append((step, sorted(res[1] or ()), sorted(runner._kv_failed_recving)))
        log.append(("poll", polls[-1]))
    return log, polls, bad


def _rank(rank, init_file, q):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
    try:
        out = {}
        for name, pat in PATTERNS.items():
            dist.barrier()
            out[name] = _drive(rank, pat, dist.group.WORLD)
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
        res = [q.get(timeout=int(os.environ.get("Q38_S3DGATE_TIMEOUT", "300"))) for _ in procs]
        for p in procs:
            p.join(timeout=60)
    res = sorted(res, key=lambda r: r[0])
    for rank, out, info in res:
        assert out is not None, info
    return res


def test_toggled_lever_leaves_nothing_outstanding_and_decides_as_off(runs):
    keep = ("step", "submit", "collective", "repack", "poll")
    for rank, out, every in runs:
        off_log, off_polls, off_bad = out["off"]
        assert not off_bad, (rank, off_bad)
        for name in PATTERNS:
            log, polls, bad = out[name]
            assert not bad, (rank, name, bad)
            assert R._only(log, keep) == R._only(off_log, keep), (rank, name)
            assert polls == off_polls, (rank, name)
            assert ("issue",) in log or not any(PATTERNS[name]), (rank, name)
        assert all(e == every[0] for e in every), rank
    print("Q38S3DGATE_TOGGLE patterns=%d steps=%d ranks=%d" % (len(PATTERNS), N_STEPS, len(runs)))


def _returns(src):
    return [m.start() for m in re.finditer(r"\n[ \t]+return\b", src)]


def test_served_runner_cannot_strand_a_deferred_submit_across_steps():
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    em = inspect.getsource(TPUModelRunner.execute_model)
    guard = em.index("if self.execute_model_state is not None:")
    assert guard < em.index("_l6_ab.begin_step(") and "raise RuntimeError" in em[guard:guard + 200]
    setup = em.index("self.maybe_setup_kv_connector(")
    state = em.index("self.execute_model_state = ExecuteModelState(")
    for r in _returns(em):
        if setup < r < state:  # only the non-last PP stage's return, right under its test
            pp = em.rfind("if not self._pp_is_last:", setup, r)
            assert pp > 0 and em[pp:r].count("\n") <= 2, em[r:r + 80]
    st = inspect.getsource(TPUModelRunner.sample_tokens)
    sub = st.index("self._submit_deferred_kv_loads()")
    recorded = st.index("state = self.execute_model_state")
    for r in _returns(st):
        assert r < recorded or r > sub, st[r:r + 80]
    pi = inspect.getsource(TPUModelRunner._sample_tokens_pp_intermediate_stage)
    assert "self.get_finished_kv_transfers(scheduler_output)" in pi
    gf = inspect.getsource(TPUModelRunner.get_finished_kv_transfers)
    assert gf.index("self._submit_deferred_kv_loads()") < gf.index("kv_connector.clear_connector_metadata()")
