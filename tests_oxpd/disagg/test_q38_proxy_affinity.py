# SPDX-License-Identifier: Apache-2.0
"""PROXY_DECODE_ROUTING=session_affinity: a returning conversation turn goes to the decode that holds its earlier KV."""

import asyncio
import importlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "examples/disagg"))
pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
proxy = importlib.import_module("toy_proxy_server")

SYSTEM = {"role": "system", "content": "You are a coding agent. " * 40}  # ~1 KB shared head


def conv(name: str, turns: int) -> list[dict]:
    """Messages of a conversation's request at turn `turns` (1-based): each turn adds the
    assistant reply, a tool result and the next user message to the previous request."""
    msgs = [SYSTEM, {"role": "user", "content": f"[{name}] task: " + f"{name} details " * 30}]
    for t in range(1, turns):
        msgs += [
            {"role": "assistant", "content": f"[{name}] step {t} " * 10,
             "tool_calls": [{"id": f"{name}{t}", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"{name}{t}", "content": f"[{name}] output {t} " * 20},
            {"role": "user", "content": f"[{name}] continue {t}"},
        ]
    return msgs


def req(name: str, turn: int) -> dict:
    return {"model": "m", "messages": conv(name, turn), "tools": [{"type": "function", "function": {"name": "run"}}]}


def route(r, body: dict, rid: str):
    return r.choose(r.hashes(r.key(body)), rid)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("PROXY_DECODE_ROUTING", raising=False)
    assert proxy._make_decode_affinity(3) is None
    monkeypatch.setenv("PROXY_DECODE_ROUTING", "round_robin")
    assert proxy._make_decode_affinity(3) is None
    monkeypatch.setenv("PROXY_DECODE_ROUTING", "session_affinity")
    monkeypatch.setenv("PROXY_DECODE_AFFINITY_CHUNK", "512")
    monkeypatch.setenv("PROXY_DECODE_AFFINITY_MAX_SKEW", "4")
    r = proxy._make_decode_affinity(3)
    assert (r.n, r.chunk, r.max_skew) == (3, 512, 4)


def test_key_is_prefix_stable_across_turns():
    K = proxy._DecodeSessionAffinity.key
    for t in (1, 2, 3):
        a, b = K(req("A", t)), K(req("A", t + 1))
        assert b.startswith(a) and len(b) > len(a)
    assert not K(req("B", 2)).startswith(K(req("A", 1)))
    # tools are part of the key, ahead of the messages
    assert K({"messages": [SYSTEM], "tools": [1]}) != K({"messages": [SYSTEM]})
    assert K({"prompt": "abcdef"}).startswith(K({"prompt": "abc"}))
    assert K({"prompt": [1, 2, 3, 4]}).startswith(K({"prompt": [1, 2, 3]}))
    assert not K({"prompt": [1, 23]}).startswith(K({"prompt": [1, 2]}))  # token boundaries kept
    assert K({"prompt": ["a", "b"]}) == b"" and K({"prompt": [[1], [2]]}) == b""


def test_hashes_are_cumulative():
    r = proxy._DecodeSessionAffinity(2, chunk=64)
    a = r.hashes(b"x" * 64 + b"y" * 64)
    b = r.hashes(b"z" * 64 + b"y" * 64)
    assert len(a) == 2 and a[1] != b[1]  # same second chunk, different history
    assert r.hashes(b"x" * 127) == a[:1]  # a partial tail chunk is not hashed


def test_returning_turns_stick_to_their_decode():
    r = proxy._DecodeSessionAffinity(3, chunk=128, max_skew=0)
    first = {n: route(r, req(n, 1), f"{n}1") for n in "ABC"}
    assert sorted(v[0] for v in first.values()) == [0, 1, 2]
    assert all(v[1] in ("cold", "branch") for v in first.values())
    for n in "ABC":
        r.done(f"{n}1")
    for turn in (2, 3, 4):
        for n in "CAB":  # an order a round robin would scramble
            d, why, depth = route(r, req(n, turn), f"{n}{turn}")
            assert (d, why) == (first[n][0], "sticky") and depth > 0
            r.done(f"{n}{turn}")
    assert r.counts["sticky"] == 9 and r.inflight == [0, 0, 0]


def test_shared_head_alone_is_not_a_returning_turn():
    r = proxy._DecodeSessionAffinity(3, chunk=128, max_skew=0)
    d_a, _, _ = route(r, req("A", 1), "A1")  # stays in flight
    d_b, why_b, depth_b = route(r, req("B", 1), "B1")
    # B shares only the system prompt with A: it branches off A's chain, so it is balanced away.
    assert why_b == "branch" and depth_b > 0 and d_b != d_a
    d_b2, why_b2, _ = route(r, req("B", 2), "B2")
    assert (d_b2, why_b2) == (d_b, "sticky")
    d_a2, why_a2, _ = route(r, req("A", 2), "A2")
    assert (d_a2, why_a2) == (d_a, "sticky")


def test_next_is_the_first_successor_and_marks_chain_ends():
    r = proxy._DecodeSessionAffinity(2, chunk=64, max_skew=0)
    head, a, b = b"h" * 64, b"a" * 64, b"b" * 64
    route(r, {"prompt": head.decode()}, "0")
    h0 = r.hashes(head)[0]
    assert r.entries[h0][1] is None  # the end of a request
    route(r, {"prompt": (head + a).decode()}, "1")
    ha = r.hashes(head + a)[1]
    assert r.entries[h0][1] == ha  # continued by the next turn
    route(r, {"prompt": (head + b).decode()}, "2")
    assert r.entries[h0][1] == ha  # a later branch does not overwrite it


def test_exact_repeat_follows_the_decode_that_last_served_it():
    r = proxy._DecodeSessionAffinity(3, chunk=128, max_skew=2)
    d0, _, _ = route(r, req("A", 1), "a")
    r.done("a")
    r.inflight[d0] = 5
    d1, why, _ = route(r, req("A", 1), "b")  # guarded away
    assert why == "guard" and d1 != d0
    r.done("b")
    r.inflight[d0] = 0
    d2, why, _ = route(r, req("A", 1), "c")
    assert (d2, why) == (d1, "sticky")


def test_idle_ties_rotate():
    r = proxy._DecodeSessionAffinity(3, chunk=64, max_skew=0)
    picks = []
    for i in range(4):
        picks.append(route(r, {"prompt": f"p{i}"}, str(i))[0])
        r.done(str(i))
    assert picks == [0, 1, 2, 0]


def test_guard_overrides_an_overloaded_target_and_zero_disables_it():
    for skew, want in ((2, "guard"), (0, "sticky")):
        r = proxy._DecodeSessionAffinity(3, chunk=128, max_skew=skew)
        d, _, _ = route(r, req("A", 1), "A1")
        r.done("A1")
        r.inflight[d] = 5  # the target is busy, the others idle
        d2, why, _ = route(r, req("A", 2), "A2")
        assert why == want
        assert (d2 != d) if want == "guard" else (d2 == d)


def test_least_loaded_with_round_robin_ties_and_release():
    r = proxy._DecodeSessionAffinity(3, chunk=64, max_skew=0)
    picks = [route(r, {"prompt": f"p{i}"}, str(i))[0] for i in range(3)]  # short prompts: no digests
    assert picks == [0, 1, 2] and r.inflight == [1, 1, 1]
    r.done("1")
    assert route(r, {"prompt": "q"}, "q")[0] == 1
    r.done("unknown")  # harmless
    assert r.inflight == [1, 1, 1]


def test_entry_table_is_bounded():
    r = proxy._DecodeSessionAffinity(2, chunk=64, max_skew=0, max_entries=5)
    route(r, {"prompt": "a" * 64 * 8}, "1")
    assert len(r.entries) == 5


def _app(n_decodes: int, monkeypatch, mode: str | None):
    """The proxy app with mock backends: one prefill, n decodes recording what they receive."""
    if mode is None:
        monkeypatch.delenv("PROXY_DECODE_ROUTING", raising=False)
    else:
        monkeypatch.setenv("PROXY_DECODE_ROUTING", mode)
    monkeypatch.setenv("PROXY_DECODE_AFFINITY_CHUNK", "128")
    seen = [[] for _ in range(n_decodes)]
    uuid_box = [0]

    def prefill(request):
        uuid_box[0] += 1
        return httpx.Response(200, json={"kv_transfer_params": {"uuid": uuid_box[0]},
                                         "usage": {"prompt_tokens": 5}})

    def decode(i):
        def h(request):
            seen[i].append(json.loads(request.content))
            return httpx.Response(200, content=b'data: {"x": 1}\n\ndata: [DONE]\n\n')
        return h

    app = proxy.app
    app.state.prefill_clients = [{"client": httpx.AsyncClient(transport=httpx.MockTransport(prefill), base_url="http://p"),
                                  "host": "p", "port": 1, "id": 0}]
    app.state.decode_clients = [{"client": httpx.AsyncClient(transport=httpx.MockTransport(decode(i)), base_url=f"http://d{i}"),
                                 "host": f"d{i}", "port": 1, "id": i} for i in range(n_decodes)]
    import itertools
    app.state.prefill_iterator = itertools.cycle(range(1))
    app.state.decode_iterator = itertools.cycle(range(n_decodes))
    app.state.decode_affinity = proxy._make_decode_affinity(n_decodes)
    return app, seen


def _serve(app, bodies):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
            for b in bodies:
                r = await c.post("/v1/chat/completions", json={**b, "stream": True})
                assert r.status_code == 200 and b"[DONE]" in r.content
    asyncio.run(go())


def _where(seen, name):
    return [i for i, reqs in enumerate(seen) for b in reqs if f"[{name}] task" in b["messages"][1]["content"]]


ORDER = [("A", 1), ("B", 1), ("C", 1), ("C", 2), ("A", 2), ("B", 2), ("B", 3), ("C", 3), ("A", 3)]


def test_proxy_end_to_end_affinity_keeps_each_conversation_on_one_decode(monkeypatch):
    app, seen = _app(3, monkeypatch, "session_affinity")
    _serve(app, [req(n, t) for n, t in ORDER])
    for n in "ABC":
        places = _where(seen, n)
        assert len(places) == 3 and len(set(places)) == 1, (n, places)
    assert app.state.decode_affinity.inflight == [0, 0, 0]
    assert app.state.decode_affinity.counts["sticky"] == 6


def test_proxy_end_to_end_round_robin_unchanged_when_off(monkeypatch):
    app, seen = _app(3, monkeypatch, None)
    assert app.state.decode_affinity is None
    _serve(app, [req(n, t) for n, t in ORDER])
    # Round robin: request k goes to decode k mod 3, whatever the conversation.
    assert [len(s) for s in seen] == [3, 3, 3]
    assert sorted(_where(seen, "A")) == sorted(_where(seen, "B")) == sorted(_where(seen, "C")) == [0, 1, 2]
