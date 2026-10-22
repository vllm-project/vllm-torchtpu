# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
from pathlib import Path


def _load_toy_proxy_server():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "disagg"
        / "toy_proxy_server.py"
    )
    spec = importlib.util.spec_from_file_location("toy_proxy_server_adm", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_m = _load_toy_proxy_server()
_AdmissionBudget = _m._AdmissionBudget
_PrefixAffinityRouter = _m._PrefixAffinityRouter


def _run(coro):
    return asyncio.run(coro)


def test_budget_count_cap_blocks_and_releases() -> None:
    async def body():
        b = _AdmissionBudget("d", max_count=1, max_tokens=0)
        await b.acquire(10)
        second = asyncio.create_task(b.acquire(10))
        await asyncio.sleep(0.01)
        assert not second.done()
        assert b.waiting == 1
        await b.release(10)
        await asyncio.wait_for(second, 1.0)
        assert b.count == 1 and b.waiting == 0

    _run(body())


def test_budget_token_cap_counts_tokens_not_requests() -> None:
    async def body():
        b = _AdmissionBudget("d", max_count=0, max_tokens=100)
        await b.acquire(60)
        await b.acquire(30)
        third = asyncio.create_task(b.acquire(30))
        await asyncio.sleep(0.01)
        assert not third.done()
        await b.release(60)
        await asyncio.wait_for(third, 1.0)
        assert b.tokens == 60 and b.count == 2

    _run(body())


def test_budget_admits_oversize_request_only_when_empty() -> None:
    async def body():
        b = _AdmissionBudget("d", max_count=0, max_tokens=100)
        await b.acquire(500)
        assert b.count == 1
        nxt = asyncio.create_task(b.acquire(1))
        await asyncio.sleep(0.01)
        assert not nxt.done()
        await b.release(500)
        await asyncio.wait_for(nxt, 1.0)

    _run(body())


def test_budget_adjust_frees_room_when_estimate_was_high() -> None:
    async def body():
        b = _AdmissionBudget("d", max_count=0, max_tokens=100)
        await b.acquire(90)
        nxt = asyncio.create_task(b.acquire(50))
        await asyncio.sleep(0.01)
        assert not nxt.done()
        await b.adjust(-60)
        await asyncio.wait_for(nxt, 1.0)
        assert b.tokens == 80

    _run(body())


def _turns(router, text_prefix, n_turns, turn_len=4096):
    text = text_prefix
    ranks = []
    for _ in range(n_turns):
        text += "x" * turn_len
        rank, _ = router.route(router.hashes(text))
        router.done(rank)
        ranks.append(rank)
    return ranks


def test_router_residency_spreads_cold_sessions_and_keeps_turns_sticky() -> None:
    r = _PrefixAffinityRouter(n=2, chunk=1024, max_sessions=4)
    firsts = []
    for i in range(4):
        ranks = _turns(r, f"session-{i}-" * 100, 3)
        assert len(set(ranks)) == 1
        firsts.append(ranks[0])
    assert sorted(firsts) == [0, 0, 1, 1]
    assert r.stats()["resident"] == [2, 2]
    assert r.evicted == 0


def test_router_evicted_session_is_rerouted_as_cold() -> None:
    r = _PrefixAffinityRouter(n=2, chunk=1024, max_sessions=1)
    a = "aaaa-" * 1000
    b = "bbbb-" * 1000
    c = "cccc-" * 1000
    ra = r.route(r.hashes(a))[0]
    r.done(ra)
    rb = r.route(r.hashes(b))[0]
    r.done(rb)
    assert ra != rb
    # A third conversation evicts the LRU session of the rank it lands on.
    rc = r.route(r.hashes(c))[0]
    r.done(rc)
    assert r.evicted == 1
    evicted_text = a if rc == ra else b
    rank, sticky = r.route(r.hashes(evicted_text + "x" * 4096))
    assert not sticky
    assert r.rerouted == 1
    r.done(rank)


def test_router_saturation_relaxes_slack() -> None:
    r = _PrefixAffinityRouter(n=2, chunk=1024, slack=0, saturated_inflight=2)
    a = "aaaa-" * 1000
    ra, _ = r.route(r.hashes(a))
    r.done(ra)
    other = 1 - ra
    # Owner leg busier than the other beyond the slack: not saturated, so cold.
    r.inflight = [0, 0]
    r.inflight[ra] = 1
    rank, sticky = r.route(r.hashes(a + "x" * 4096))
    assert not sticky and rank == other
    r.done(rank)
    # Both legs at or above saturated_inflight: sticky wins despite imbalance.
    r.inflight = [2, 2]
    r.inflight[rank] = 5
    rank2, sticky2 = r.route(r.hashes(a + "x" * 8192))
    assert sticky2 and rank2 == rank
    assert r.sticky_saturated == 1


def test_prompt_token_helpers() -> None:
    assert _m._estimate_prompt_tokens("a" * 400, 4.0) == 100
    assert _m._expected_output_tokens({"max_tokens": 50}, 1024) == 50
    assert _m._expected_output_tokens({"max_tokens": 5000}, 1024) == 1024
    assert _m._expected_output_tokens({}, 1024) == 1024
    assert _m._prompt_tokens_from_prefill({"usage": {"prompt_tokens": 7}}, {}) == 7
    assert _m._prompt_tokens_from_prefill({}, {"num_tokens": 9}) == 10
    assert _m._prompt_tokens_from_prefill({}, {}) is None


def test_decode_router_balances_on_running_requests() -> None:
    r = _PrefixAffinityRouter(n=3, chunk=1024, slack=2, load_metric="running")
    # A cold conversation goes to the leg decoding the fewest requests, not to
    # the one with the fewest in flight (leg 1): requests in the prefill hop
    # hold no decode slot yet.
    r.inflight = [10, 4, 12]
    r.running = [0, 4, 1]
    rank, sticky = r.route(r.hashes("cold-" * 1000))
    assert not sticky and rank == 0
    r.done(rank)
    r.running = [5, 4, 1]
    rank, sticky = r.route(r.hashes("cold2-" * 1000))
    assert not sticky and rank == 2
    r.done(rank)
    # The same conversation stays on its leg while it is within the slack of
    # the least-running leg, and moves to the least-running leg beyond it.
    text = "conv-" * 1000
    owner, _ = r.route(r.hashes(text))
    r.done(owner)
    r.running = [3, 3, 3]
    r.running[owner] = 5
    rank, sticky = r.route(r.hashes(text + "x" * 4096))
    assert sticky and rank == owner
    r.done(rank)
    r.running[owner] = 6
    rank, sticky = r.route(r.hashes(text + "x" * 8192))
    assert not sticky and rank != owner


def test_decode_router_running_count_follows_start_and_done() -> None:
    r = _PrefixAffinityRouter(n=2, chunk=1024, load_metric="running")
    rank, _ = r.route(r.hashes("a" * 4096))
    assert r.inflight[rank] == 1 and r.running[rank] == 0
    r.start_running(rank)
    assert r.running[rank] == 1
    r.done(rank, was_running=True)
    assert r.inflight[rank] == 0 and r.running[rank] == 0
    # A request that never reached the decode hop leaves running untouched.
    rank, _ = r.route(r.hashes("b" * 4096))
    r.start_running(1 - rank)
    r.done(rank)
    assert r.running[1 - rank] == 1 and r.inflight == [0, 0]
    assert r.stats()["running"] == r.running
    assert r.stats()["load_metric"] == "running"


def test_inflight_metric_keeps_prefill_router_behaviour() -> None:
    r = _PrefixAffinityRouter(n=2, chunk=1024)
    r.inflight = [3, 0]
    r.running = [0, 5]
    rank, _ = r.route(r.hashes("p" * 4096))
    assert rank == 1
    r.done(rank)


def test_quad_attn_cost_model_balances_cold_and_preserves_sticky() -> None:
    r = _PrefixAffinityRouter(
        n=2, chunk=1024, slack=1, quad_attn_coeff=1.23e-5, token_slack=32768.0
    )
    # Rank 0 gets one massive 120K-token cold conversation; rank 1 gets two
    # short 2K-token cold requests. Even though rank 0 has fewer requests in
    # flight (1 vs 2), its quadratic effective token backlog is much larger,
    # so the next cold request goes to rank 1.
    long_text = "L" * (120 * 1024)
    r0, _ = r.route(r.hashes(long_text), prompt_tokens=120000, req_id="long0")
    assert r0 == 0
    r1a, _ = r.route(r.hashes("s1-" * 1024), prompt_tokens=2000, req_id="short1a")
    assert r1a == 1
    r1b, _ = r.route(r.hashes("s2-" * 1024), prompt_tokens=2000, req_id="short1b")
    assert r1b == 1
    assert r.inflight == [1, 2]
    assert r.inflight_eff_tokens[0] > r.inflight_eff_tokens[1]
    next_cold, sticky = r.route(
        r.hashes("s3-" * 1024), prompt_tokens=4000, req_id="short1c"
    )
    assert not sticky and next_cold == 1
    # Complete all in-flight requests and verify effective token accounting resets.
    r.done(0, req_id="long0")
    r.done(1, req_id="short1a")
    r.done(1, req_id="short1b")
    r.done(1, req_id="short1c")
    assert r.inflight_eff_tokens == [0.0, 0.0]

    # Now rank 0 has 3 small requests in flight (beyond request slack=1 vs
    # rank 1's 0 in flight), and a 124K-token continuation of long_text (120K
    # cached on rank 0, 4K new) arrives. Re-prefilling 124K cold on rank 1
    # costs ~218K effective tokens, whereas queuing the 4K warm delta on rank 0
    # costs ~10K effective tokens on top of rank 0's small backlog -> stays sticky!
    for i in range(3):
        rk, _ = r.route(
            r.hashes(f"small-{i}-" * 512), prompt_tokens=2000, req_id=f"sm{i}"
        )
        if rk != 0:
            r.done(rk, req_id=f"sm{i}")
            r.inflight[0] += 1
            r.inflight_eff_tokens[0] += r.effective_tokens(2000, 0)
    cont_rank, cont_sticky = r.route(
        r.hashes(long_text + "N" * 4096), prompt_tokens=124000, req_id="cont"
    )
    assert cont_sticky and cont_rank == 0
    assert r.sticky_quad_saved == 1


def test_decode_step_cost_model_dualpath_and_overflow_routing() -> None:
    dscm = _m._DecodeStepCostModel(mk_ceiling=20.0)
    r = _PrefixAffinityRouter(
        n=2,
        chunk=1024,
        slack=2,
        load_metric="running",
        decode_cost_model=dscm,
    )
    # Rank 0 runs 5 huge 180K-token sequences (900K active KV tokens);
    # Rank 1 runs 6 short 10K-token sequences (60K active KV tokens).
    # Even though Rank 0 has fewer running requests (5 vs 6), its active KV
    # bandwidth load is 15x higher, so a new cold request is routed to Rank 1.
    for i in range(5):
        r.running[0] += 1
        r.running_tokens[0] += 180000
    for i in range(6):
        r.running[1] += 1
        r.running_tokens[1] += 10000
    cold_rank, sticky = r.route(
        r.hashes("cold-req-" * 512), prompt_tokens=32000, req_id="c1"
    )
    assert not sticky and cold_rank == 1
    r.start_running(cold_rank, req_id="c1", tokens=32000)
    assert r.running_tokens[1] == 92000
    r.done(cold_rank, was_running=True, req_id="c1")
    assert r.running_tokens[1] == 60000

    # Warm prefix on Rank 1 stays sticky even when Rank 1 has 3 more running
    # requests than Rank 0 (beyond raw slack=2), because both are below
    # mk_ceiling=20 and the Raiden pull savings + lower KV token load win.
    r.running = [4, 7]
    r.running_tokens = [400000, 400000]
    warm_text = "warm-conv-" * 4000
    w_rank0, _ = r.route(r.hashes(warm_text), prompt_tokens=100000, req_id="w0")
    r.start_running(w_rank0, req_id="w0", tokens=100000)
    r.done(w_rank0, was_running=True, req_id="w0")
    r.running[w_rank0] = 7
    r.running[1 - w_rank0] = 4
    r.running_tokens = [400000, 400000]
    w_rank1, w_sticky = r.route(
        r.hashes(warm_text + "delta" * 512), prompt_tokens=104000, req_id="w1"
    )
    assert w_sticky and w_rank1 == w_rank0
    r.done(w_rank1, was_running=False, req_id="w1")

    # Once w_rank0 overflows mk_ceiling=20 while the other rank has free slots,
    # queue_wait_ms + tight overflow slack spill to the uncongested rank.
    r.running[w_rank0] = 24
    r.running[1 - w_rank0] = 12
    r.running_tokens[w_rank0] = 2400000
    r.running_tokens[1 - w_rank0] = 1200000
    spill_rank, spill_sticky = r.route(
        r.hashes(warm_text + "delta2" * 512), prompt_tokens=108000, req_id="w2"
    )
    assert not spill_sticky and spill_rank == (1 - w_rank0)
    r.done(spill_rank, was_running=False, req_id="w2")
