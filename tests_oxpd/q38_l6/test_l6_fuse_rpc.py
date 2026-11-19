# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever fuserpc (core/l6_fuse_rpc.py): one execute_and_sample RPC instead of execute_model +
sample_tokens. Real vLLM MultiprocExecutor RPC path (shm MessageQueues, FutureWrapper, KV output
aggregator) to forked processes running vLLM's real WorkerProc busy loop + async output thread over a
WorkerWrapperBase; engine side = vLLM 0.31.0's step_with_batch_queue vs the lever's copy on a fake
engine core with a scripted scheduler. CDK CPU pod only."""

from __future__ import annotations

import ast
import inspect
import os
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l6_mk_harness as H  # noqa: E402

from vllm.v1.engine.core import EngineCore  # noqa: E402
from vllm.v1.executor.multiproc_executor import MultiprocExecutor  # noqa: E402

from vllm_torchtpu.core import l6_ab  # noqa: E402
from vllm_torchtpu.core import l6_fuse_rpc as F  # noqa: E402

# A hung rig shows its stacks through pytest's own faulthandler_timeout (the harness passes -o
# faulthandler_timeout=240; a module-level dump_traceback_later is cancelled by pytest at the first failure).
os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "120")  # a lost reply fails, not hangs
# Resolve vLLM's platform once on this thread before any rig: WorkerProc's async output thread reads
# vllm.platforms.current_platform, and the rig's worker threads must not race to resolve it lazily.
from vllm.platforms import current_platform as _platform  # noqa: E402

_ = _platform.device_type

VLLM_STEP = EngineCore.__dict__["step_with_batch_queue"]
assert not getattr(VLLM_STEP, "_l6_fuse", False), "vLLM's method was already replaced at import"
COUNTS: dict = {}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    def wipe():
        for k in list(os.environ):
            if k.startswith("VLLM_TPU_L6_") or k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_SERVED_AB"):
                monkeypatch.delenv(k, raising=False)
        l6_ab._resolve()
        COUNTS.clear()
        F._KV.update(check=None, turn=0)

    wipe()
    orig_log = l6_ab._log_window

    def log_window():
        for k, v in l6_ab._state["counts"].items():
            COUNTS[k] = COUNTS.get(k, 0) + v
        orig_log()

    monkeypatch.setattr(l6_ab, "_log_window", log_window)
    # The executor method the copy calls (install_engine adds it to the class in serving).
    monkeypatch.setattr(MultiprocExecutor, F.METHOD, F.executor_execute_and_sample, raising=False)
    yield monkeypatch
    wipe()
    EngineCore.step_with_batch_queue = VLLM_STEP
    F._INSTALLED["engine"] = False


def _set(mp, **env):
    for k, v in env.items():
        mp.setenv(k, v)
    l6_ab._resolve()


def counts() -> dict:
    out = dict(COUNTS)
    for k, v in l6_ab._state["counts"].items():
        out[k] = out.get(k, 0) + v
    return out


def _run(tmp_path, tag, step_fn, plan, kv_agg=True, world=2, fail_exec_at=None, n_req=4, tpu_async=False, delay_ms=0.0,
         mode=None):
    d = tmp_path / tag
    d.mkdir()
    rig = H.RpcRig(str(d), world=world, kv_agg=kv_agg, fail_exec_at=fail_exec_at, n_req=n_req,
                   tpu_async=tpu_async, delay_ms=delay_ms, mode=mode)
    try:
        err = None
        try:
            tr = H.run_rpc_stream(rig, step_fn, plan, n_req=n_req)
        except Exception as e:  # noqa: BLE001 - compared below
            tr, err = None, (type(e).__name__, str(e))
            if "injected" not in str(e):
                print(f"RIG_ERROR {tag}: {type(e).__name__}: {e}\n{rig.diagnostics()}", flush=True)
        enq = list(rig.enqueued)
        if err is None:
            rig.sync()
        return tr, err, rig.worker_logs(), enq
    finally:
        rig.close()


def same_engine(c, a, plan, fused_on=True):
    """Engine-side equality: every step's engine outputs and every scheduler update equal vLLM's. The only
    difference allowed: on a fused step the copy does not call scheduler.get_grammar_bitmask, which returns
    None on its first line for a step without structured-output requests (scheduler.py:1948)."""
    ca, cc = c[0], a[0]
    if ca is None or cc is None:
        return False
    if ca["results"] != cc["results"] or ca["updates"] != cc["updates"] or ca["kvonly_off"] != cc["kvonly_off"]:
        return False
    got, want = ca["grammar"], cc["grammar"]
    skipped = [i for i in want if i not in got]
    if [i for i in want if i in got] != got:  # the copy's calls are a subsequence of vLLM's
        return False
    return all(not plan[i].get("structured") for i in skipped) and (fused_on or not skipped)


def _eligible(plan):
    return sum(1 for p in plan if p.get("tokens", 1) > 0 and not p.get("structured"))


@pytest.mark.parametrize("kv_agg", [True, False], ids=["kv-aggregator", "output-rank-only"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fused_matches_stock_over_random_streams(clean, tmp_path, kv_agg, seed):
    plan = H.rpc_plan(seed, 120, structured=0.1, pending=0.05, empty=0.05)
    a = _run(tmp_path, "vllm", VLLM_STEP, plan, kv_agg)  # vLLM's method, lever unset
    b = _run(tmp_path, "copy_off", F.step_with_batch_queue, plan, kv_agg)  # the copy, lever off
    _set(clean, VLLM_TPU_L6_FUSE_RPC="1")
    c = _run(tmp_path, "copy_on", F.step_with_batch_queue, plan, kv_agg)
    for r in (a, b, c):
        assert r[1] is None, r[1]
    assert b[0] == a[0], "the copy with the lever off differs from vLLM's method"
    assert same_engine(c, a, plan), "engine-side results / scheduler updates differ"
    assert b[2] == a[2] and c[2] == a[2], "worker call order differs"
    n = _eligible(plan)
    assert a[3] == b[3] and a[3].count("execute_model") == len(plan)
    assert c[3].count(F.METHOD) == n and c[3].count("execute_model") == len(plan) - n, c[3][:10]
    assert len(c[3]) == len(a[3]) - n
    assert counts().get("fuserpc.fused", 0) == n
    print(f"ENGAGEMENT fuserpc seed={seed} kv_agg={kv_agg} steps={len(plan)} fused={n} "
          f"rpcs {len(a[3])}->{len(c[3])}")


def test_fused_and_kvonly_match_stock_with_worker_processes(clean, tmp_path):
    """The same comparison with the rig workers as spawned processes (process isolation)."""
    plan = H.rpc_plan(11, 40, structured=0.1, pending=0.05, empty=0.05)
    a = _run(tmp_path, "vllm", VLLM_STEP, plan, world=2, tpu_async=True, mode="spawn")
    _set(clean, VLLM_TPU_L6_FUSE_RPC="1", VLLM_TPU_L6_OUT_KVONLY="1")
    c = _run(tmp_path, "copy_on", F.step_with_batch_queue, plan, world=2, tpu_async=True, mode="spawn")
    assert a[1] is None and c[1] is None, (a[1], c[1])
    assert same_engine(c, a, plan) and _strip_waits(c[2]) == _strip_waits(a[2])
    n = _eligible(plan)
    assert c[3].count(F.METHOD) == n and counts().get("fuserpc.fused", 0) == n
    print(f"ENGAGEMENT fuserpc+e2kvonly spawn steps={len(plan)} fused={n} rpcs {len(a[3])}->{len(c[3])}")


def test_fused_ab_windows_match_stock(clean, tmp_path):
    plan = H.rpc_plan(5, 90)
    a = _run(tmp_path, "vllm", VLLM_STEP, plan)
    _set(clean, VLLM_TPU_L6_FUSE_RPC="ab:1", VLLM_TPU_L6_AB_STEPS="4", VLLM_TPU_L6_AB_CONFIRM="1")
    c = _run(tmp_path, "copy_ab", F.step_with_batch_queue, plan)
    assert same_engine(c, a, plan) and c[2] == a[2] and c[1] is None
    n = counts().get("fuserpc.fused", 0)
    assert 0 < n < _eligible(plan), n


def test_grammar_and_deferred_steps_stay_split(clean, tmp_path):
    plan = [{"structured": True}] * 6 + [{"structured": True, "pending": True}] * 4 + [{}] * 4
    a = _run(tmp_path, "vllm", VLLM_STEP, plan)
    _set(clean, VLLM_TPU_L6_FUSE_RPC="1")
    c = _run(tmp_path, "copy_on", F.step_with_batch_queue, plan)
    assert same_engine(c, a, plan) and c[2] == a[2]
    assert c[0]["grammar"] == list(range(10)) and a[0]["grammar"] == list(range(14))
    assert c[3].count(F.METHOD) == 4 and c[3].count("sample_tokens") == 10 and a[3].count("sample_tokens") == 14
    # The grammar reached the workers on every structured step.
    assert sum("'sample', " in ln and ln.rstrip(")").split(", ")[-1] != "None" for ln in c[2][0]) == 10


@pytest.mark.parametrize("kv_agg", [True, False], ids=["kv-aggregator", "output-rank-only"])
def test_execute_model_failure_raises_the_same_error(clean, tmp_path, kv_agg):
    plan = [{}] * 8
    a = _run(tmp_path, "vllm", VLLM_STEP, plan, kv_agg, fail_exec_at=3)
    _set(clean, VLLM_TPU_L6_FUSE_RPC="1")
    c = _run(tmp_path, "copy_on", F.step_with_batch_queue, plan, kv_agg, fail_exec_at=3)
    assert a[1] is not None and a[1][0] == "RuntimeError" and "injected execute_model failure at step 3" in a[1][1]
    assert c[1] == a[1], (a[1], c[1])


def test_worker_method_order_and_failure_semantics():
    calls = []

    class W:
        def __init__(self, fail=False, sample_none=False):
            self.fail, self.sample_none = fail, sample_none

        def execute_model(self, so):
            calls.append(("exec", so))
            if self.fail:
                raise ValueError("boom")
            return None

        def sample_tokens(self, g):
            calls.append(("sample", g))
            return None if self.sample_none else ("out", g)

    assert F.execute_and_sample(W(), "so", None) == ("out", None)
    assert calls == [("exec", "so"), ("sample", None)]
    calls.clear()
    with pytest.raises(ValueError, match="boom"):
        F.execute_and_sample(W(fail=True, sample_none=True), "so", None)
    assert calls == [("exec", "so"), ("sample", None)]  # sample_tokens still ran, as its RPC would
    calls.clear()
    assert F.execute_and_sample(W(fail=True), "so", "g") == ("out", "g")


class _Strip(ast.NodeTransformer):
    """Remove the lever's additions from the copy: the rule, its branch, its counters, the local imports
    and the docstring. What remains must be vLLM's method."""

    def _drop(self, node):
        src = ast.unparse(node)
        return "_l6_ab" in src or "fused =" in src

    def visit_FunctionDef(self, node):
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
            body = body[1:]
        body = [b for b in body if not isinstance(b, (ast.Import, ast.ImportFrom))]
        node.body = body
        node.decorator_list = []
        node.returns = None
        node.name = "f"
        self.generic_visit(node)
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if isinstance(node.test, ast.Name) and node.test.id == "fused":
            return node.orelse or None
        if "_l6_ab" in ast.unparse(node.test):
            return None
        return node

    def visit_Assign(self, node):
        return None if any(isinstance(t, ast.Name) and t.id == "fused" for t in node.targets) else node

    def visit_Expr(self, node):
        return None if "_l6_ab" in ast.unparse(node) else node


def _norm(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    tree = _Strip().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)


def test_copy_is_vllms_method_plus_the_fused_branch():
    assert _norm(F.step_with_batch_queue) == _norm(VLLM_STEP)


def test_install_binds_the_copy_and_refuses_unverified_sources(clean, monkeypatch):
    from types import SimpleNamespace as NS

    assert F.install_engine(NS(parallel_config=NS(pipeline_parallel_size=1))) is False  # flag unset
    _set(clean, VLLM_TPU_L6_FUSE_RPC="1")
    assert F.install_engine(NS(parallel_config=NS(pipeline_parallel_size=2))) is False
    assert EngineCore.__dict__["step_with_batch_queue"] is VLLM_STEP
    monkeypatch.setitem(F.VLLM_SOURCES, "MultiprocExecutor.collective_rpc", "0000000000000000")
    assert F.install_engine(NS(parallel_config=NS(pipeline_parallel_size=1))) is False
    assert EngineCore.__dict__["step_with_batch_queue"] is VLLM_STEP
    monkeypatch.setitem(F.VLLM_SOURCES, "MultiprocExecutor.collective_rpc", F.vllm_source_hashes()["MultiprocExecutor.collective_rpc"])
    print(f"FUSERPC_SOURCES {F.vllm_source_hashes()}")
    assert F.vllm_source_hashes() == F.VLLM_SOURCES
    assert F.install_engine(NS(parallel_config=NS(pipeline_parallel_size=1))) is True
    assert EngineCore.__dict__["step_with_batch_queue"] is F.step_with_batch_queue


def test_worker_install_adds_the_method():
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    F.install_worker()
    assert WorkerWrapperBase.execute_and_sample is F.execute_and_sample


# ------------------------------------------------------------------------------------- e2kvonly
def _waits(logs):
    """Per rank: the steps whose token copy the worker waited for."""
    return [[int(ln.split(", ")[1].rstrip(")")) for ln in rank if ln.startswith("('wait'")] for rank in logs]


def _strip_waits(logs):
    return [[ln for ln in rank if not ln.startswith("('wait'")] for rank in logs]


@pytest.mark.parametrize("world", [2, 8])
def test_kvonly_matches_stock_and_skips_the_other_ranks_waits(clean, tmp_path, world):
    """8 ranks with staggered copy-ready times (TPU async outputs): with fuserpc + e2kvonly the engine's
    results equal vLLM's, the output rank waits for every step's copy, and on fused steps the other
    ranks reply without waiting."""
    plan = H.rpc_plan(world, 60, structured=0.1, pending=0.0)
    a = _run(tmp_path, "vllm", VLLM_STEP, plan, world=world, tpu_async=True, delay_ms=0.5)
    _set(clean, VLLM_TPU_L6_FUSE_RPC="1", VLLM_TPU_L6_OUT_KVONLY="1")
    c = _run(tmp_path, "kvonly", F.step_with_batch_queue, plan, world=world, tpu_async=True, delay_ms=0.5)
    assert a[1] is None and c[1] is None, (a[1], c[1])
    assert same_engine(c, a, plan), "engine-side results / scheduler updates differ"
    assert _strip_waits(c[2]) == _strip_waits(a[2]), "worker call order differs"
    wa, wc = _waits(a[2]), _waits(c[2])
    fused = {i for i, p in enumerate(plan) if not p.get("structured")}
    assert all(sorted(w) == list(range(len(plan))) for w in wa), "stock: every rank waits for every step"
    assert sorted(wc[0]) == list(range(len(plan))), "the output rank must wait for every step"
    for r in range(1, world):
        assert sorted(wc[r]) == sorted(set(range(len(plan))) - fused), (r, wc[r])
    assert counts().get("e2kvonly.sent", 0) == len(fused)
    # Finished-sending / receiving ids reached the scheduler (all ranks' KV outputs aggregated).
    assert any(u[3] and u[3][0] for u in c[0]["updates"]) and any(u[3] and u[3][1] for u in c[0]["updates"])
    print(f"ENGAGEMENT e2kvonly world={world} steps={len(plan)} kvonly={len(fused)} "
          f"non_output_waits {sum(len(w) for w in wa[1:])}->{sum(len(w) for w in wc[1:])}")


def test_fused_without_kvonly_keeps_full_replies(clean, tmp_path):
    plan = [{}] * 20
    _set(clean, VLLM_TPU_L6_FUSE_RPC="1")
    c = _run(tmp_path, "fused", F.step_with_batch_queue, plan, world=4, tpu_async=True)
    assert all(sorted(w) == list(range(20)) for w in _waits(c[2]))
    assert counts().get("e2kvonly.sent", 0) == 0


def test_kvonly_check_mode_finds_no_mismatch(clean, tmp_path):
    plan = H.rpc_plan(3, 40, structured=0.1, pending=0.0)
    a = _run(tmp_path, "vllm", VLLM_STEP, plan, world=4, tpu_async=True)
    _set(clean, VLLM_TPU_L6_FUSE_RPC="1", VLLM_TPU_L6_OUT_KVONLY="1", VLLM_TPU_L6_OUT_KVONLY_CHECK="1")
    c = _run(tmp_path, "check", F.step_with_batch_queue, plan, world=4, tpu_async=True)
    assert same_engine(c, a, plan) and c[1] is None
    n = sum(1 for p in plan if not p.get("structured"))
    got = counts()
    assert got.get("e2kvonly.check_ok", 0) == n and got.get("e2kvonly.mismatch", 0) == 0, got


def test_kvonly_check_catches_a_dropped_kv_output(clean, tmp_path, monkeypatch):
    """A kv-only form that drops a rank's KV output changes the aggregation: the CHECK sees it on the
    first fused step (step 0 reports a finished send), returns the real result and turns e2kvonly off."""
    plan = [{}] * 12
    a = _run(tmp_path, "vllm", VLLM_STEP, plan, world=4, tpu_async=True)
    orig = F.kv_only

    def dropping(out):
        o = orig(out)
        if o is not out:
            o.kv_connector_output = None
        return o

    monkeypatch.setattr(F, "kv_only", dropping)
    _set(clean, VLLM_TPU_L6_FUSE_RPC="1", VLLM_TPU_L6_OUT_KVONLY="1", VLLM_TPU_L6_OUT_KVONLY_CHECK="1")
    c = _run(tmp_path, "check", F.step_with_batch_queue, plan, world=4, tpu_async=True)
    assert c[1] is None and c[0]["updates"] == a[0]["updates"] and c[0]["results"] == a[0]["results"]
    # One mismatch per CHECK in flight when the first one resolved (the batch queue holds two steps).
    assert c[0]["kvonly_off"] is True and 1 <= counts().get("e2kvonly.mismatch", 0) <= 2
    assert counts().get("e2kvonly.sent", 0) == 0
