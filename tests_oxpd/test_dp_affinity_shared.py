# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_DP_AFFINITY_SHARED: one prefix table for all API-server processes of a DP decode server (q38 CACHE-3)."""

import importlib.util
import multiprocessing as mp
import os
import pathlib
import uuid

import pytest

np = pytest.importorskip("numpy")
_PATH = pathlib.Path(__file__).resolve().parents[1] / "src/vllm_torchtpu/core/dp_affinity.py"
_spec = importlib.util.spec_from_file_location("q38_dp_affinity_shared_t", _PATH)
DA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DA)


def _name():
    return "q38t_" + uuid.uuid4().hex[:12]


def _unlink(name):
    from multiprocessing import shared_memory
    try:
        s = shared_memory.SharedMemory(name=name, create=False)
        s.close(); s.unlink()
    except FileNotFoundError:
        pass


@pytest.fixture
def name():
    n = _name()
    yield n
    _unlink(n)


def test_roundtrip_overwrite_and_zero_key(name):
    t = DA.SharedPrefixTable(name, slots=1024)
    assert t.created and t.slots == 1024 and len(t) == 0
    assert t.get(12345) is None
    t.set(12345, 2); t.set(-7, 0); t.set(0, 3)
    assert (t.get(12345), t.get(-7), t.get(0)) == (2, 0, 3)
    t.set(12345, 1)
    assert t.get(12345) == 1 and len(t) == 3
    t.close()


def test_slots_round_up_to_a_power_of_two(name):
    t = DA.SharedPrefixTable(name, slots=3000)
    assert t.slots == 4096 and t.mask == 4095
    t.close()


def test_probe_collisions_and_full_neighbourhood(name):
    t = DA.SharedPrefixTable(name, slots=1024, probe=4)
    base = 5  # keys that share the home slot 5
    ks = [base + 1024 * j for j in range(6)]
    for e, k in enumerate(ks[:4]):
        t.set(k, e)
    assert [t.get(k) for k in ks[:4]] == [0, 1, 2, 3]
    assert t.get(ks[4]) is None          # probed 4 occupied slots, not found
    t.set(ks[4], 3)                      # neighbourhood full: overwrites the home slot
    assert t.get(ks[4]) == 3 and t.get(ks[0]) is None
    t.close()


def test_attach_sees_the_creators_writes_and_checks_size(name):
    a = DA.SharedPrefixTable(name, slots=2048)
    a.set(99, 1)
    b = DA.SharedPrefixTable(name, slots=2048)
    assert not b.created and b.get(99) == 1
    b.set(100, 3)
    assert a.get(100) == 3
    with pytest.raises(RuntimeError):
        DA.SharedPrefixTable(name, slots=4096)
    a.close(); b.close()


def _child(name, q):
    t = DA.SharedPrefixTable(name, slots=2048)
    q.put(t.get(7))
    t.set(8, 2)
    t.close()


def test_two_processes_share_one_table(name):
    a = DA.SharedPrefixTable(name, slots=2048)
    a.set(7, 3)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(name, q))
    p.start(); p.join(60)
    assert p.exitcode == 0 and q.get(timeout=5) == 3
    assert a.get(8) == 2
    # the child's exit did not unlink the segment (resource tracker untracked)
    c = DA.SharedPrefixTable(name, slots=2048)
    assert c.get(8) == 2
    a.close(); c.close()


def test_a_separate_interpreter_exiting_does_not_unlink_the_table(name):
    """A fresh interpreter has its OWN resource tracker (a multiprocessing child shares the parent's, so it cannot
    show this). Python < 3.13 registers an attacher with its tracker, which unlinks the segment when that process
    exits; SharedPrefixTable unregisters, so the table must survive the attacher's exit."""
    import subprocess
    import sys

    a = DA.SharedPrefixTable(name, slots=2048)
    a.set(11, 1)
    code = ("import importlib.util, sys\n"
            f"s = importlib.util.spec_from_file_location('m', {str(_PATH)!r})\n"
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
            f"t = m.SharedPrefixTable({name!r}, slots=2048)\n"
            "assert t.get(11) == 1; t.set(12, 2); t.close()\n")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    b = DA.SharedPrefixTable(name, slots=2048)   # attaches only if the segment still exists
    assert not b.created and b.get(12) == 2
    a.close(); b.close()


def test_epoch_name_changes_with_the_parent_and_override(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_DP_AFFINITY_SHARED_NAME", raising=False)
    me = DA.shared_table_name(os.getpid())
    assert me.startswith(f"q38dpaff_{os.getpid()}_")
    if os.path.exists("/proc/self/stat"):
        assert not me.endswith("_0")     # carries the start-time epoch on Linux
    assert DA.shared_table_name(os.getpid()) == me
    assert DA.shared_table_name(1) != me
    monkeypatch.setenv("VLLM_TPU_DP_AFFINITY_SHARED_NAME", "q38x")
    assert DA.shared_table_name() == "q38x"


def _hashes(router, conv, turns, chunk=4):
    toks = [hash((conv, i)) % 50000 for i in range(chunk * 3 * turns)]
    return router.chunk_hashes(toks[: chunk * 3 * turns])


def test_four_api_servers_share_affinity_only_with_the_shared_table(name):
    """Turn 1 routed by server 0, turn 2 by server 1: with per-process tables server 1 knows nothing ('new');
    with the shared table it routes to turn 1's engine ('affine')."""
    scores = [0.0, 0.0, 0.0, 0.0]
    local = [DA.PrefixAffinityRouter(4, chunk=4) for _ in range(4)]
    h1 = _hashes(local[0], "A", 1); h2 = _hashes(local[0], "A", 2)
    local[0].record(h1, 2)
    assert local[1].choose(h2, scores) == (None, "new")
    tables = [DA.SharedPrefixTable(name, slots=4096) for _ in range(4)]
    shared = [DA.PrefixAffinityRouter(4, chunk=4, table=t) for t in tables]
    shared[0].record(h1, 2)
    assert shared[1].choose(h2, scores) == (2, "affine")
    assert shared[3].lookup(h2) == (2, len(h1))
    assert shared[2].entries() == len(h1) and local[0].entries() == len(h1)
    for t in tables:
        t.close()


def test_shared_router_keeps_the_bound_and_ignores_out_of_range_engines(name):
    t = DA.SharedPrefixTable(name, slots=4096)
    r = DA.PrefixAffinityRouter(2, chunk=4, slack=0.25, table=t)
    h = _hashes(r, "B", 1)
    r.record(h, 1)
    assert r.choose(h, [0.0, 7.0]) == (None, "fallback")   # 7 > 1.25 x 3.5 + 1
    assert r.choose(h, [2.0, 5.0]) == (1, "affine")        # 5 <= 1.25 x 3.5 + 1
    r.record(h, 5)                                        # an engine id this server does not have
    assert r.lookup(h) == (None, 0)
    t.close()


def test_patch_uses_the_shared_table_and_falls_back(monkeypatch, name):
    """The patch body's table choice, through the env flags (no vLLM needed: the router factory is exercised directly)."""
    monkeypatch.setenv("VLLM_TPU_DP_AFFINITY_SHARED_NAME", name)
    t = DA.SharedPrefixTable(DA.shared_table_name(), 2048)
    assert t.name == name and t.created
    t2 = DA.SharedPrefixTable(DA.shared_table_name(), 2048)
    assert not t2.created
    with pytest.raises(RuntimeError):
        DA.SharedPrefixTable(DA.shared_table_name(), 1 << 20)   # size mismatch -> the patch logs and uses per-process
    t.close(); t2.close()


# ---- the patch itself against vLLM's real DPLBAsyncMPClient (skipped where vLLM is absent) ----

def _base_tests():
    spec = importlib.util.spec_from_file_location("q38_dp_affinity_base_t", pathlib.Path(__file__).with_name("test_dp_affinity.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _patched(monkeypatch, shared, name):
    core_client = pytest.importorskip("vllm.v1.engine.core_client")
    cls = core_client.DPLBAsyncMPClient
    monkeypatch.setattr(cls, "get_core_engine_for_request", cls.get_core_engine_for_request)
    monkeypatch.setenv("VLLM_TPU_DP_AFFINITY", "1")
    monkeypatch.setenv("VLLM_TPU_DP_AFFINITY_CHUNK", "4")
    monkeypatch.delenv("VLLM_TPU_DP_AFFINITY_SLOT_GUARD", raising=False)
    monkeypatch.setenv("VLLM_TPU_DP_AFFINITY_SHARED_NAME", name)
    monkeypatch.setenv("VLLM_TPU_DP_AFFINITY_SHARED_SLOTS", "4096")
    if shared:
        monkeypatch.setenv("VLLM_TPU_DP_AFFINITY_SHARED", "1")
    else:
        monkeypatch.delenv("VLLM_TPU_DP_AFFINITY_SHARED", raising=False)
    from vllm_torchtpu.core import dp_affinity

    dp_affinity.patch_dp_affinity()
    return cls.get_core_engine_for_request


@pytest.mark.parametrize("shared", [True, False])
def test_patch_two_api_servers(monkeypatch, name, shared):
    """Two API-server clients (separate processes in serving; separate client objects here). Turn 1 goes through
    server A to an idle engine; turn 2 goes through server B. Shared: B routes it to turn 1's engine. Per-process:
    B has no record and stock routing picks its own least-loaded engine."""
    B = _base_tests()
    wrapped = _patched(monkeypatch, shared, name)
    a, b = B.FakeClient(), B.FakeClient()
    for cl in (a, b):
        for e, n in ((0, 2), (1, 2), (2, 0), (3, 2)):
            cl.engine_inflight[cl.core_engines[e]] = n
    first = wrapped(a, B.Req("t1", B.toks(1, 2)))
    assert B.eng(a, first) == 2
    b.engine_inflight[b.core_engines[2]] = 3        # B's own view: engine 2 busier, engine 0 now the least loaded
    b.engine_inflight[b.core_engines[0]] = 1
    second = wrapped(b, B.Req("t2", B.toks(1, 2, 3)))
    router_b = b.__dict__["_tpu_dp_router"]
    if shared:
        assert B.eng(b, second) == 2 and router_b.table is not None and router_b.table.name == name
    else:
        assert B.eng(b, second) == 0 and router_b.table is None


def test_patch_falls_back_when_the_table_cannot_attach(monkeypatch, name):
    squat = DA.SharedPrefixTable(name, slots=1024)      # same name, wrong size
    B = _base_tests()
    wrapped = _patched(monkeypatch, True, name)
    cl = B.FakeClient()
    wrapped(cl, B.Req("f1", B.toks(1, 2)))
    assert cl.__dict__["_tpu_dp_router"].table is None   # per-process table, serving continues
    squat.close()
