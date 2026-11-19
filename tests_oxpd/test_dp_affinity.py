# SPDX-License-Identifier: Apache-2.0
"""Prefix-affine DP routing (src/vllm_torchtpu/core/dp_affinity.py).

The router and the wrapper body are loaded by path, so the pure tests run
without vLLM. ``FakeClient`` + ``stock_route`` mirror vLLM v0.31.0's
``DPLBAsyncMPClient`` fields and ``get_core_engine_for_request``
(v1/engine/core_client.py:1589-1639); the last group runs the wrapper against
the real vLLM function and is skipped where vLLM is not installed.
"""

import importlib.util
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "src/vllm_torchtpu/core/dp_affinity.py"


def _load():
    spec = importlib.util.spec_from_file_location("q38_dp_affinity_under_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


DA = _load()
C = 4  # chunk tokens in these tests


def toks(*chunk_ids, tail=0):
    """A prompt made of whole C-token chunks (chunk i = C copies of id i) plus a partial tail."""
    out = []
    for i in chunk_ids:
        out += [i] * C
    return out + [999] * tail


class Req:
    def __init__(self, rid, token_ids, rank=None, pooling=None):
        self.request_id = rid
        self.prompt_token_ids = token_ids
        self.data_parallel_rank = rank
        self.pooling_params = pooling


class FakeClient:
    def __init__(self, n=4, client_count=1):
        self.core_engines = [bytes([i, 0]) for i in range(n)]
        self.lb_engines = [[0, 0, 0.0] for _ in range(n)]
        self.engine_inflight = Counter()
        self.reqs_in_flight = {}
        self.client_count = client_count
        self.eng_start_index = 0
        self.seen_ranks = []

    def finish(self, rid):
        eng = self.reqs_in_flight.pop(rid)
        self.engine_inflight[eng] -= 1


def stock_route(self, request):
    """vLLM v0.31.0 DPLBAsyncMPClient.get_core_engine_for_request, minus late interaction."""
    self.seen_ranks.append(request.data_parallel_rank)
    if (eng_index := request.data_parallel_rank) is None:
        current_counts = self.lb_engines
        num_engines = len(current_counts)
        min_score = sys.maxsize
        eng_index = 0
        for i in range(num_engines):
            idx = (self.eng_start_index + i) % num_engines
            waiting, running, kv_cache_usage = current_counts[idx]
            inflight = self.engine_inflight[self.core_engines[idx]]
            score = max(self.client_count * inflight, waiting + running)
            if waiting:
                score += waiting * 6.0 * max(0.0, kv_cache_usage - 0.5)
            if score < min_score:
                min_score = score
                eng_index = idx
        current_counts[eng_index][0] += self.client_count
        self.eng_start_index = (self.eng_start_index + 1) % num_engines
    chosen_engine = self.core_engines[eng_index]
    self.reqs_in_flight[request.request_id] = chosen_engine
    self.engine_inflight[chosen_engine] += 1
    return chosen_engine


def fresh_stats():
    return {"affine": 0, "fallback": 0, "new": 0, "pinned": 0, "errors": 0, "logged_at": 0.0}


def eng(client, chosen):
    return client.core_engines.index(chosen)


# ---------------------------------------------------------------- pure router


def test_chunk_hashes_are_chained_and_skip_the_partial_tail():
    r = DA.PrefixAffinityRouter(4, chunk=C)
    a = r.chunk_hashes(toks(1, 2, 3, tail=C - 1))
    assert len(a) == 3
    b = r.chunk_hashes(toks(1, 2, 3, 4))
    assert b[:3] == a
    # Same chunks 2 and 3 after a different chunk 1: every hash differs (chained).
    c = r.chunk_hashes(toks(9, 2, 3))
    assert all(x != y for x, y in zip(a, c))
    assert r.chunk_hashes(toks(tail=C - 1)) == []


def test_lookup_returns_the_longest_recorded_prefix():
    r = DA.PrefixAffinityRouter(4, chunk=C)
    shared = r.chunk_hashes(toks(1))
    r.record(shared, 0)  # a system prompt first seen on engine 0
    r.record(r.chunk_hashes(toks(1, 5, 6)), 2)  # session B's turn 1 on engine 2
    r.record(r.chunk_hashes(toks(1, 7)), 1)  # session C's turn 1 on engine 1
    # B's turn 2 shares chunk 1 with everyone but chunks 1,5,6 only with B.
    assert r.lookup(r.chunk_hashes(toks(1, 5, 6, 8))) == (2, 3)
    assert r.lookup(r.chunk_hashes(toks(1, 7, 8))) == (1, 2)
    assert r.lookup(r.chunk_hashes(toks(3))) == (None, 0)


def test_bound_is_one_plus_slack_times_mean_plus_one():
    r = DA.PrefixAffinityRouter(4, chunk=C, slack=0.25)
    # mean 10 -> bound 13.5
    assert r.within_bound(0, [13, 9, 9, 9])  # mean 10, 13 <= 13.5
    assert not r.within_bound(0, [14, 10, 9, 7])  # mean 10, 14 > 13.5
    r0 = DA.PrefixAffinityRouter(4, chunk=C, slack=0.0)
    assert r0.within_bound(0, [1, 0, 0, 0])  # the +1 lets an idle system stay affine
    assert not r0.within_bound(0, [3, 1, 1, 1])  # mean 1.5, 3 > 2.5


def test_slot_guard_refuses_a_full_target_while_a_slot_is_free():
    r = DA.PrefixAffinityRouter(4, chunk=C, slack=1.0, cap=16)
    assert not r.within_bound(0, [16, 15, 16, 16])  # target full, engine 1 has a slot
    assert r.within_bound(0, [15, 12, 16, 16])  # target has a slot
    assert r.within_bound(0, [17, 16, 16, 16])  # everyone full: relative bound decides
    rn = DA.PrefixAffinityRouter(4, chunk=C, slack=1.0, cap=None)
    assert rn.within_bound(0, [16, 15, 16, 16])


def test_choose_kinds():
    r = DA.PrefixAffinityRouter(4, chunk=C, slack=0.25)
    h = r.chunk_hashes(toks(1, 2))
    assert r.choose(h, [0, 0, 0, 0]) == (None, "new")
    r.record(h, 3)
    assert r.choose(h, [0, 0, 0, 0]) == (3, "affine")
    assert r.choose(h, [0, 0, 0, 9]) == (None, "fallback")


def test_lru_evicts_oldest_and_refreshes_on_record():
    r = DA.PrefixAffinityRouter(4, chunk=C, max_entries=3)
    r.record(r.chunk_hashes(toks(1)), 0)
    r.record(r.chunk_hashes(toks(2)), 1)
    r.record(r.chunk_hashes(toks(3)), 2)
    r.record(r.chunk_hashes(toks(1)), 0)  # refresh 1
    r.record(r.chunk_hashes(toks(4)), 3)  # evicts 2, the oldest
    assert len(r._map) == 3
    assert r.lookup(r.chunk_hashes(toks(2))) == (None, 0)
    assert r.lookup(r.chunk_hashes(toks(1))) == (0, 1)
    assert r.lookup(r.chunk_hashes(toks(3))) == (2, 1)


def test_record_moves_a_prefix_to_its_newest_engine():
    r = DA.PrefixAffinityRouter(4, chunk=C)
    h = r.chunk_hashes(toks(1, 2))
    r.record(h, 0)
    r.record(h, 2)
    assert r.lookup(h) == (2, 2)


def test_bad_parameters_raise():
    for kw in ({"num_engines": 0}, {"chunk": 0}, {"slack": -0.1}, {"max_entries": 0}, {"cap": 0}):
        args = {"num_engines": 4, **kw}
        with pytest.raises(ValueError):
            DA.PrefixAffinityRouter(**args)


# ------------------------------------------------------- wrapper (fake client)


def test_new_session_takes_the_stock_choice_and_its_next_turn_returns():
    cl, st = FakeClient(), fresh_stats()
    r = DA.PrefixAffinityRouter(4, chunk=C, slack=0.25)
    for e, n in ((0, 3), (1, 1), (2, 3), (3, 3)):
        cl.engine_inflight[cl.core_engines[e]] = n
    first = DA.route(cl, Req("s1t1", toks(1, 2)), stock_route, r, st)
    assert eng(cl, first) == 1  # least loaded, exactly as stock
    assert st["new"] == 1
    cl.finish("s1t1")
    # Turn 2 extends turn 1: back to engine 1 even though engine 1 is now not least loaded.
    cl.engine_inflight[cl.core_engines[1]] = 4
    second = DA.route(cl, Req("s1t2", toks(1, 2, 3)), stock_route, r, st)
    assert eng(cl, second) == 1
    assert st["affine"] == 1


def test_affine_pick_restores_the_rank_and_counts_like_stock():
    cl, st = FakeClient(), fresh_stats()
    r = DA.PrefixAffinityRouter(4, chunk=C)
    r.record(r.chunk_hashes(toks(1)), 2)
    req = Req("x", toks(1, 4))
    before = [row[0] for row in cl.lb_engines]
    chosen = DA.route(cl, req, stock_route, r, st)
    assert eng(cl, chosen) == 2
    assert cl.seen_ranks == [2]  # the stock code was asked for engine 2
    assert req.data_parallel_rank is None  # and the request leaves unpinned
    assert cl.reqs_in_flight["x"] == cl.core_engines[2]
    assert cl.engine_inflight[cl.core_engines[2]] == 1
    after = [row[0] for row in cl.lb_engines]
    assert [a - b for a, b in zip(after, before)] == [0, 0, 1, 0]
    assert cl.eng_start_index == 0  # the pinned stock branch does not rotate


def test_overloaded_target_falls_back_and_the_session_moves():
    cl, st = FakeClient(), fresh_stats()
    r = DA.PrefixAffinityRouter(4, chunk=C, slack=0.25)
    r.record(r.chunk_hashes(toks(1, 2)), 0)
    for e, n in ((0, 9), (1, 2), (2, 2), (3, 2)):  # mean 3.75, bound 5.69
        cl.engine_inflight[cl.core_engines[e]] = n
    chosen = DA.route(cl, Req("t2", toks(1, 2, 3)), stock_route, r, st)
    assert eng(cl, chosen) != 0
    assert st["fallback"] == 1
    moved = eng(cl, chosen)
    assert r.lookup(r.chunk_hashes(toks(1, 2, 3))) == (moved, 3)


def test_preset_rank_tokenless_and_pooling_requests_pass_through():
    cl, st = FakeClient(), fresh_stats()
    r = DA.PrefixAffinityRouter(4, chunk=C)
    r.record(r.chunk_hashes(toks(1)), 2)
    assert eng(cl, DA.route(cl, Req("a", toks(1), rank=3), stock_route, r, st)) == 3
    DA.route(cl, Req("b", None), stock_route, r, st)
    DA.route(cl, Req("c", toks(1), pooling=object()), stock_route, r, st)
    assert st["pinned"] == 1 and st["affine"] == 0 and st["new"] == 0
    assert len(r._map) == 1  # nothing recorded for pass-through requests


def test_router_error_falls_back_to_stock():
    cl, st = FakeClient(), fresh_stats()

    class Broken(DA.PrefixAffinityRouter):
        def choose(self, hashes, scores):
            raise RuntimeError("boom")

    chosen = DA.route(cl, Req("e", toks(1)), stock_route, Broken(4, chunk=C), st)
    assert chosen == cl.core_engines[0]
    assert st["errors"] == 1


def test_rank_is_restored_even_if_the_stock_call_raises():
    cl, st = FakeClient(), fresh_stats()
    r = DA.PrefixAffinityRouter(4, chunk=C)
    r.record(r.chunk_hashes(toks(1)), 1)
    req = Req("f", toks(1))

    def raising(client, request):
        raise KeyError("x")

    with pytest.raises(KeyError):
        DA.route(cl, req, raising, r, st)
    assert req.data_parallel_rank is None


def test_stock_score_matches_vllm_formula():
    cl = FakeClient(client_count=2)
    cl.lb_engines[1] = [3, 5, 0.9]
    cl.engine_inflight[cl.core_engines[1]] = 6
    # max(2 * 6, 3 + 5) + 3 * 6 * (0.9 - 0.5)
    assert DA.stock_score(cl, 1) == pytest.approx(12 + 7.2)
    cl.lb_engines[2] = [0, 4, 0.99]
    assert DA.stock_score(cl, 2) == 4.0


def _stock_pick(cl):
    best, pick = sys.maxsize, 0
    for i in range(len(cl.core_engines)):
        idx = (cl.eng_start_index + i) % len(cl.core_engines)
        sc = DA.stock_score(cl, idx)
        if sc < best:
            best, pick = sc, idx
    return pick


@pytest.mark.parametrize("slack,min_affine", [(0.25, 0.8), (0.0, 0.5)])
def test_closed_loop_oracle_identity_and_bound(slack, min_affine):
    """64 sessions x ~30 turns against an independent oracle: a returning turn goes to
    its session's previous engine exactly when that engine passes the bound
    ((1 + s) x mean + 1, and not full while another engine has a free slot);
    otherwise, and for a new session, it goes where stock vLLM would send it."""
    import random

    rnd = random.Random(7)
    cl, st = FakeClient(), fresh_stats()
    r = DA.PrefixAffinityRouter(4, chunk=C, slack=slack, cap=16)
    hist = {s: [100 + s] for s in range(64)}
    where, live = {}, []
    stayed = moved = 0
    for step in range(64 * 30):
        s = rnd.randrange(64)
        hist[s].append(10_000 + step)
        scores = [DA.stock_score(cl, i) for i in range(4)]
        mean = sum(scores) / 4
        prev = where.get(s)
        ok = prev is not None and scores[prev] <= (1 + slack) * mean + 1 and (
            scores[prev] < 16 or min(scores) >= 16)
        expect = prev if ok else _stock_pick(cl)
        rid = f"{s}:{step}"
        e = eng(cl, DA.route(cl, Req(rid, toks(*hist[s])), stock_route, r, st))
        assert e == expect, (step, s, prev, scores, e, expect)
        if prev is not None:
            stayed += e == prev
            moved += e != prev
        where[s] = e
        live.append(rid)
        if len(live) > 48:
            cl.finish(live.pop(rnd.randrange(len(live))))
    assert st["new"] == 64
    assert st["affine"] + st["fallback"] == stayed + moved
    assert st["affine"] / (stayed + moved) > min_affine
    assert st["fallback"] >= 1  # both branches exercised


# ------------------------------------------------------ real vLLM stock code


def test_wrapper_against_the_real_vllm_router(monkeypatch):
    core_client = pytest.importorskip("vllm.v1.engine.core_client")
    real = core_client.DPLBAsyncMPClient.get_core_engine_for_request
    cl, st = FakeClient(), fresh_stats()
    r = DA.PrefixAffinityRouter(4, chunk=C)
    for e, n in ((0, 2), (1, 2), (2, 0), (3, 2)):
        cl.engine_inflight[cl.core_engines[e]] = n
    req = Req("v1", toks(1, 2))
    req.pooling_params = None
    first = DA.route(cl, req, real, r, st)
    assert eng(cl, first) == 2 and st["new"] == 1
    cl.engine_inflight[cl.core_engines[2]] = 3
    req2 = Req("v2", toks(1, 2, 3))
    second = DA.route(cl, req2, real, r, st)
    assert eng(cl, second) == 2 and st["affine"] == 1
    assert req2.data_parallel_rank is None
    assert cl.reqs_in_flight["v2"] == cl.core_engines[2]


def test_patch_installs_once_and_routes(monkeypatch):
    core_client = pytest.importorskip("vllm.v1.engine.core_client")
    cls = core_client.DPLBAsyncMPClient
    monkeypatch.setattr(cls, "get_core_engine_for_request", cls.get_core_engine_for_request)
    monkeypatch.setenv("VLLM_TPU_DP_AFFINITY", "1")
    monkeypatch.setenv("VLLM_TPU_DP_AFFINITY_CHUNK", str(C))
    monkeypatch.setenv("VLLM_TPU_DP_AFFINITY_SLOT_GUARD", "1")
    from vllm_torchtpu.core import dp_affinity

    dp_affinity.patch_dp_affinity()
    wrapped = cls.get_core_engine_for_request
    assert getattr(wrapped, "_tpu_dp_affinity", False)
    dp_affinity.patch_dp_affinity()
    assert cls.get_core_engine_for_request is wrapped
    cl = FakeClient()
    cl.vllm_config = types.SimpleNamespace(scheduler_config=types.SimpleNamespace(max_num_seqs=16))
    for x in cl.core_engines:
        cl.engine_inflight[x] = 2
    a = wrapped(cl, Req("p1", toks(1, 2)))  # all equal: stock picks engine 0
    assert eng(cl, a) == 0
    b = wrapped(cl, Req("p2", toks(1, 2, 3)))  # engine 0 now busiest (3 vs 2), within bound
    assert a == b
    assert cl.__dict__["_tpu_dp_router"].cap == 16


def test_patch_is_off_by_default(monkeypatch):
    core_client = pytest.importorskip("vllm.v1.engine.core_client")
    cls = core_client.DPLBAsyncMPClient
    orig = cls.get_core_engine_for_request
    monkeypatch.setattr(cls, "get_core_engine_for_request", orig)
    monkeypatch.delenv("VLLM_TPU_DP_AFFINITY", raising=False)
    from vllm_torchtpu.core import dp_affinity

    dp_affinity.patch_dp_affinity()
    assert cls.get_core_engine_for_request is orig


def test_patch_default_has_no_slot_guard(monkeypatch):
    core_client = pytest.importorskip("vllm.v1.engine.core_client")
    cls = core_client.DPLBAsyncMPClient
    monkeypatch.setattr(cls, "get_core_engine_for_request", cls.get_core_engine_for_request)
    monkeypatch.setenv("VLLM_TPU_DP_AFFINITY", "1")
    monkeypatch.delenv("VLLM_TPU_DP_AFFINITY_SLOT_GUARD", raising=False)
    from vllm_torchtpu.core import dp_affinity

    dp_affinity.patch_dp_affinity()
    cl = FakeClient()
    cl.vllm_config = types.SimpleNamespace(scheduler_config=types.SimpleNamespace(max_num_seqs=16))
    cls.get_core_engine_for_request(cl, Req("d1", list(range(2048))))
    router = cl.__dict__["_tpu_dp_router"]
    assert router.cap is None and router.chunk == 1024 and router.slack == 0.25
