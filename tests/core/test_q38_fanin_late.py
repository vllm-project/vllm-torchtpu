# SPDX-License-Identifier: Apache-2.0
"""q38 SV-KNOB FANIN-L (core/fanin_late.py): the engine core reads the non-output ranks' step replies one call late.

Default off, nothing patched; the late path reads only the output rank in the call and the others first thing in the
next call (FIFO per rank); the aggregator counts every rank's report once, one step RPC later; the tokens are the output
rank's reply object; errors surface one call later; the in-process A/B windows; refusals; compile-ignore; wiring."""

import collections
import os
import pathlib
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.outputs import KVConnectorOutput

from vllm_torchtpu.core import fanin_late as fl

SRC = pathlib.Path(fl.__file__).resolve().parents[1]  # .../src/vllm_torchtpu
KEYS = ("VLLM_TPU_FANIN_LATE", "VLLM_TPU_FANIN_LATE_AB_MS")
OK, BAD = WorkerProc.ResponseStatus.SUCCESS, WorkerProc.ResponseStatus.FAILURE
DAY_NS = 86_400 * 10**9


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    fl._load()
    fl._state["ab_window"] = None
    saved = (fl._now_ns, dict(fl.stats))
    yield
    for k in KEYS:
        os.environ.pop(k, None)
    fl._now_ns = saved[0]
    fl.stats.clear()
    fl.stats.update(saved[1])
    fl._load()
    fl._state["ab_window"] = None


class MQ:
    def __init__(self, rank):
        self.rank, self.q, self.reads = rank, collections.deque(), []

    def put(self, tag, result=None, status=OK):
        self.q.append((status, result if result is not None else SimpleNamespace(tag=tag, rank=self.rank, kv_connector_output=None)))

    def dequeue(self, timeout=None):
        if not self.q:
            raise TimeoutError("empty")
        item = self.q.popleft()
        self.reads.append(item)
        return item


class Exec:
    def __init__(self, n=8, pp=1):
        self.rpc_broadcast_mq = SimpleNamespace(sent=[], enqueue=lambda x: self.rpc_broadcast_mq.sent.append(x))
        self.is_failed = False
        self.response_mqs = [MQ(r) for r in range(n)]
        self.futures_queue = collections.deque()
        self.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=pp))


def out(tag, rank, recv=None):
    return SimpleNamespace(tag=tag, rank=rank, kv_connector_output=KVConnectorOutput(finished_recving=set(recv) if recv else None))


def step(ex, tag, recv_by_rank=None):
    """Every rank replies to one step RPC; recv_by_rank: {rank: {req ids reported finished}}."""
    for r, mq in enumerate(ex.response_mqs):
        mq.put(tag, out(tag, r, (recv_by_rank or {}).get(r)))


def call(rpc, ex, agg, method="sample_tokens", non_block=False):
    return rpc(ex, method, timeout=None, args=(None,), unique_reply_rank=0, kv_output_aggregator=agg, non_block=non_block)


def _on(monkeypatch, ab=None):
    monkeypatch.setenv("VLLM_TPU_FANIN_LATE", "1")
    if ab is not None:
        monkeypatch.setenv("VLLM_TPU_FANIN_LATE_AB_MS", str(ab))
    fl._load()
    return fl.make_collective_rpc()


# ---- off


def test_default_off_patches_nothing():
    from vllm.v1.engine.core import EngineCoreProc

    orig, init = MultiprocExecutor.collective_rpc, EngineCoreProc.__init__
    assert fl.ENABLED is False and fl.AB_MS == 0
    assert fl.patch_fanin_late() is False
    assert MultiprocExecutor.collective_rpc is orig and not hasattr(orig, "__wrapped__")
    assert EngineCoreProc.__init__ is init


def test_off_in_a_fresh_interpreter_and_no_line():
    code = textwrap.dedent(
        """
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor
        from vllm.v1.engine.core import EngineCoreProc
        orig, init = MultiprocExecutor.collective_rpc, EngineCoreProc.__init__
        from vllm_torchtpu.core import fanin_late as fl
        r = fl.patch_fanin_late()
        print("OFFCHECK", r, MultiprocExecutor.collective_rpc is orig, EngineCoreProc.__init__ is init)
        """
    )
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    p = subprocess.run([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-3000:]
    assert "OFFCHECK False True True" in p.stdout and "FANIN_LATE" not in p.stdout


# ---- the late path


def test_late_call_reads_only_the_output_rank(monkeypatch):
    rpc = _on(monkeypatch)
    ex, agg = Exec(), KVOutputAggregator(8)
    step(ex, "s1")
    res = call(rpc, ex, agg)
    assert res.tag == "s1" and res.rank == 0
    assert len(ex.response_mqs[0].reads) == 1 and all(not mq.reads for mq in ex.response_mqs[1:])
    st = ex._tpu_fanin_late
    assert list(st["pending"]) == [([1, 2, 3, 4, 5, 6, 7], True)]
    assert ex.rpc_broadcast_mq.sent[-1][3] is None  # every worker still replies


def test_next_call_drains_first_in_fifo_order(monkeypatch):
    rpc = _on(monkeypatch)
    ex, agg = Exec(), KVOutputAggregator(8)
    for tag in ("e1", "s1", "e2", "s2"):
        step(ex, tag)
    for tag in ("e1", "s1", "e2", "s2"):
        assert call(rpc, ex, agg, method="execute_model" if tag[0] == "e" else "sample_tokens").tag == tag
    for r, mq in enumerate(ex.response_mqs):
        want = ["e1", "s1", "e2", "s2"] if r == 0 else ["e1", "s1", "e2"]
        assert [x[1].tag for x in mq.reads] == want, r
    assert len(ex._tpu_fanin_late["pending"]) == 1  # never more than the last call


def test_finished_one_step_late_counted_once_tokens_identical(monkeypatch):
    rpc = _on(monkeypatch)
    ex, agg = Exec(), KVOutputAggregator(8)
    step(ex, "s1", {r: {"X"} for r in range(8)})
    step(ex, "s2")
    step(ex, "s3")
    r0_s1 = ex.response_mqs[0].q[0][1]
    a = call(rpc, ex, agg)
    assert a is r0_s1  # the tokens object is the output rank's reply
    assert not a.kv_connector_output.finished_recving  # only rank 0's report counted so far
    b = call(rpc, ex, agg)
    assert b.kv_connector_output.finished_recving == {"X"}  # ranks 1-7 counted one step RPC later
    c = call(rpc, ex, agg)
    assert not c.kv_connector_output.finished_recving  # once
    assert agg._recv_remaining_count == {}


def test_stock_path_finishes_in_the_same_call():
    ex, agg = Exec(), KVOutputAggregator(8)
    step(ex, "s1", {r: {"X"} for r in range(8)})
    res = MultiprocExecutor.collective_rpc(ex, "sample_tokens", args=(None,), unique_reply_rank=0, kv_output_aggregator=agg)
    assert res.kv_connector_output.finished_recving == {"X"}


def test_error_from_a_later_rank_surfaces_one_call_later(monkeypatch):
    rpc = _on(monkeypatch)
    ex, agg = Exec(), KVOutputAggregator(8)
    for r, mq in enumerate(ex.response_mqs):
        mq.put("s1", "boom" if r == 3 else None, status=BAD if r == 3 else OK)
    step(ex, "s2")
    assert call(rpc, ex, agg).tag == "s1"
    with pytest.raises(RuntimeError, match="Worker failed with error 'boom'"):
        call(rpc, ex, agg)


def test_other_rpcs_drain_first_and_get_their_own_replies(monkeypatch):
    rpc = _on(monkeypatch)
    ex, agg = Exec(), KVOutputAggregator(8)
    step(ex, "s1", {r: {"Y"} for r in range(8)})
    assert call(rpc, ex, agg).tag == "s1"
    for r, mq in enumerate(ex.response_mqs):
        mq.put("health", SimpleNamespace(tag="health", rank=r))
    got = rpc(ex, "check_health", timeout=None)
    assert [g.tag for g in got] == ["health"] * 8 and [g.rank for g in got] == list(range(8))
    step(ex, "s2")
    assert call(rpc, ex, agg).kv_connector_output.finished_recving == {"Y"}  # the carry reached the next step RPC


def test_empty_output_rank_reply_drops_late_replies_as_stock(monkeypatch):
    rpc = _on(monkeypatch)
    ex, agg = Exec(), KVOutputAggregator(8)
    for r, mq in enumerate(ex.response_mqs):
        if r == 0:
            mq.q.append((OK, None))
        else:
            mq.put("e1", out("e1", r, {"Z"}))
    step(ex, "s1")
    assert call(rpc, ex, agg, method="execute_model") is None
    res = call(rpc, ex, agg)
    assert not res.kv_connector_output.finished_recving and agg._recv_remaining_count == {}
    assert fl.stats["dropped"] >= 7


def test_pipeline_stages_keep_stock_reads(monkeypatch):
    rpc = _on(monkeypatch)
    ex, agg = Exec(pp=2), KVOutputAggregator(8)
    step(ex, "s1")
    call(rpc, ex, agg)
    assert all(len(mq.reads) == 1 for mq in ex.response_mqs)


# ---- A/B windows


def test_ab_windows_late_in_even_full_in_odd(monkeypatch):
    rpc = _on(monkeypatch, ab=20000)
    lines = []
    monkeypatch.setattr(fl.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    t = [20_414 * DAY_NS]
    fl._now_ns = lambda: t[0]
    ex, agg = Exec(), KVOutputAggregator(8)
    step(ex, "s1")
    call(rpc, ex, agg)                      # window even: late
    assert all(not mq.reads for mq in ex.response_mqs[1:])
    t[0] += 20 * 10**9
    step(ex, "s2")
    call(rpc, ex, agg)                      # window odd: drains s1, then reads every rank of s2
    assert all(len(mq.reads) == 2 for mq in ex.response_mqs) and not ex._tpu_fanin_late["pending"]
    t[0] += 20 * 10**9
    step(ex, "s3")
    call(rpc, ex, agg)                      # even again: late
    assert all(len(mq.reads) == 2 for mq in ex.response_mqs[1:]) and len(ex.response_mqs[0].reads) == 3
    ab = [ln for ln in lines if ln.startswith("FANIN_LATE AB window=")]
    assert [(" on=1 " in ln) for ln in ab] == [True, False, True]


@pytest.mark.parametrize("bad", ["7000", "28800000", "86400000", "-1", "x"])
def test_bad_ab_period_refused(monkeypatch, bad):
    monkeypatch.setenv("VLLM_TPU_FANIN_LATE_AB_MS", bad)
    with pytest.raises(ValueError):
        fl._load()


# ---- install, refusal, wiring, compile key


def test_patch_installs_and_logs_once(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_FANIN_LATE", "1")
    fl._load()
    lines = []
    monkeypatch.setattr(fl.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    orig = MultiprocExecutor.collective_rpc
    monkeypatch.setattr(MultiprocExecutor, "collective_rpc", orig)
    from vllm.v1.engine.core import EngineCoreProc

    monkeypatch.setattr(EngineCoreProc, "__init__", EngineCoreProc.__init__)
    assert fl.patch_fanin_late() is True
    assert MultiprocExecutor.collective_rpc.__wrapped__ is orig
    assert fl.patch_fanin_late() is True  # idempotent
    assert lines == ["VLLM_TPU_FANIN_LATE=1: step replies of the non-output ranks are read one call late (AB 0 ms)"]


def _fake_engine(monkeypatch, aggregator=True, pp=1):
    from vllm.v1.engine.core import EngineCoreProc

    class TpuLikeExecutor(MultiprocExecutor):
        pass

    ex = object.__new__(TpuLikeExecutor)
    ex.response_mqs = [MQ(r) for r in range(8)]
    ex.output_rank = 0
    ex.kv_output_aggregator = KVOutputAggregator(8) if aggregator else None
    ex.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=pp))
    monkeypatch.setattr(EngineCoreProc, "__init__", lambda self, *a, **k: setattr(self, "model_executor", ex))
    monkeypatch.setattr(MultiprocExecutor, "collective_rpc", MultiprocExecutor.collective_rpc)
    return EngineCoreProc


def test_engine_core_logs_the_engagement_line_once_built(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_FANIN_LATE", "1")
    fl._load()
    lines, warns = [], []
    monkeypatch.setattr(fl.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    monkeypatch.setattr(fl.logger, "warning", lambda fmt, *a: warns.append(fmt % a))
    ECP = _fake_engine(monkeypatch)
    assert fl.patch_fanin_late() is True
    lines.clear()
    ECP.__init__(object.__new__(ECP))
    assert lines == ["FANIN_LATE engaged: non-output ranks' step replies read one call late (8 workers, output rank 0, TpuLikeExecutor; AB 0 ms)"]
    assert warns == []


@pytest.mark.parametrize("aggregator,pp", [(False, 1), (True, 2)])
def test_engine_core_says_not_engaged(monkeypatch, aggregator, pp):
    monkeypatch.setenv("VLLM_TPU_FANIN_LATE", "1")
    fl._load()
    lines, warns = [], []
    monkeypatch.setattr(fl.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    monkeypatch.setattr(fl.logger, "warning", lambda fmt, *a: warns.append(fmt % a))
    ECP = _fake_engine(monkeypatch, aggregator=aggregator, pp=pp)
    assert fl.patch_fanin_late() is True
    lines.clear()
    ECP.__init__(object.__new__(ECP))
    assert not any("engaged:" in ln for ln in lines)
    assert len(warns) == 1 and warns[0].startswith("FANIN_LATE not engaged: ")


def test_refused_when_wrapped_or_source_differs(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_FANIN_LATE", "1")
    fl._load()
    orig = MultiprocExecutor.collective_rpc

    def wrapped(self, *a, **k):
        return orig(self, *a, **k)

    wrapped.__wrapped__ = orig
    monkeypatch.setattr(MultiprocExecutor, "collective_rpc", wrapped)
    assert fl.patch_fanin_late() is False and MultiprocExecutor.collective_rpc is wrapped
    monkeypatch.setattr(MultiprocExecutor, "collective_rpc", orig)
    monkeypatch.setattr(fl, "SOURCE_SHA256", "0" * 64)
    assert fl.patch_fanin_late() is False and MultiprocExecutor.collective_rpc is orig


def test_vllm_source_is_the_copied_one():
    assert fl.source_sha256(MultiprocExecutor.collective_rpc) == fl.SOURCE_SHA256


def test_registered_engine_core_and_compile_ignored(monkeypatch):
    from vllm_torchtpu import envs, patch_registry
    from vllm_torchtpu.compilation import tpu_compiler

    p = [x for x in patch_registry.PATCHES if x.target == "vllm_torchtpu.core.fanin_late:patch_fanin_late"]
    assert len(p) == 1 and p[0].stages == ("engine_core",)
    for k in KEYS:
        assert k in envs.environment_variables and k in tpu_compiler._TPU_COMPILE_ENV_IGNORED, k
    base = tpu_compiler._tpu_compile_env_factors()
    monkeypatch.setenv("VLLM_TPU_FANIN_LATE", "1")
    monkeypatch.setenv("VLLM_TPU_FANIN_LATE_AB_MS", "20000")
    assert tpu_compiler._tpu_compile_env_factors() == base


def test_host_only():
    src = (SRC / "core/fanin_late.py").read_text().split('"""', 2)[2]
    for tok in ("import torch", "import jax", "import numpy", "scheduler_output", ".tolist(", "sampled_token_ids"):
        assert tok not in src, tok
