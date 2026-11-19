# SPDX-License-Identifier: Apache-2.0
"""PROXY_DECODE_ROUTING=prefix_affinity (q38 CACHE-3): longest known prefix, shared heads included,
bounded load (1 + slack) x mean + 1 on proxy in-flight streams, max_skew off; decodes are (host, port) endpoints."""

import importlib.util
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "examples/disagg"))
pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
proxy = importlib.import_module("toy_proxy_server")
_spec = importlib.util.spec_from_file_location("q38_proxy_affinity_base_tests", HERE / "test_q38_proxy_affinity.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
req, route, ORDER = base.req, base.route, base.ORDER


def prefix_router(n=2, chunk=128, slack=0.25):
    return proxy._DecodeSessionAffinity(n, chunk=chunk, max_skew=0, rule="longest", slack=slack)


def test_flag_values(monkeypatch):
    for k in ("PROXY_DECODE_AFFINITY_CHUNK", "PROXY_DECODE_AFFINITY_MAX_SKEW", "PROXY_DECODE_AFFINITY_SLACK"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PROXY_DECODE_ROUTING", "prefix_affinity")
    r = proxy._make_decode_affinity(4)
    assert (r.n, r.chunk, r.max_skew, r.rule, r.slack) == (4, 2048, 0, "longest", 0.25)
    assert "prefix_affinity" in r.describe() and "slack=0.25" in r.describe()
    monkeypatch.setenv("PROXY_DECODE_AFFINITY_SLACK", "0.1")
    assert proxy._make_decode_affinity(2).slack == 0.1
    # DECPC's mode is unchanged
    monkeypatch.setenv("PROXY_DECODE_ROUTING", "session_affinity")
    r = proxy._make_decode_affinity(3)
    assert (r.max_skew, r.rule, r.slack) == (16, "sticky", None)
    monkeypatch.delenv("PROXY_DECODE_ROUTING")
    assert proxy._make_decode_affinity(3) is None
    with pytest.raises(ValueError):
        proxy._DecodeSessionAffinity(2, rule="nearest")


def test_shared_head_goes_to_the_heads_decode_under_prefix_but_not_session():
    for rule, want_dec, want_reason in (("longest", 0, "prefix"), ("sticky", 1, "branch")):
        r = proxy._DecodeSessionAffinity(2, chunk=128, max_skew=0, rule=rule, slack=0.25 if rule == "longest" else None)
        assert route(r, req("A", 1), "a1")[0] == 0          # A stays in flight on decode 0
        d, reason, depth = route(r, req("B", 1), "b1")     # B shares only the system head with A
        assert (d, reason) == (want_dec, want_reason) and depth > 0, (rule, d, reason)


def test_returning_turns_stay_sticky_under_prefix():
    r = prefix_router(3)
    first = {n: route(r, req(n, 1), n + "1")[0] for n in "ABC"}
    for n in "ABC":
        r.done(n + "1")
    for t in (2, 3):
        for n in "CAB":
            d, reason, _ = route(r, req(n, t), f"{n}{t}")
            assert (d, reason) == (first[n], "sticky")
            r.done(f"{n}{t}")


def _target_with_inflight(inflight, slack=0.25):
    r = prefix_router(len(inflight), slack=slack)
    route(r, req("A", 1), "a1")
    r.done("a1")
    assert r.entries  # A's chain is on decode 0
    r.inflight = list(inflight)
    return route(r, req("A", 2), "a2")


def test_bounded_load_edges():
    assert _target_with_inflight([5, 2])[:2] == (0, "sticky")    # 5 <= 1.25 x 3.5 + 1 = 5.375
    assert _target_with_inflight([6, 2])[:2] == (0, "sticky")    # 6 <= 1.25 x 4 + 1 = 6 (at the bound)
    assert _target_with_inflight([7, 2])[:2] == (1, "guard")     # 7 > 1.25 x 4.5 + 1 = 6.625
    assert _target_with_inflight([7, 2], slack=0.5)[:2] == (0, "sticky")   # 7 <= 1.5 x 4.5 + 1 = 7.75


def test_no_max_skew_under_prefix():
    # 17 above the least-loaded decode: DECPC's max_skew 16 would refuse; the bound 1.25 x 91.5 + 1 = 115.4 keeps it.
    assert _target_with_inflight([100, 83])[:2] == (0, "sticky")
    r = proxy._DecodeSessionAffinity(2, chunk=128, max_skew=16)
    route(r, req("A", 1), "a1"); r.done("a1"); r.inflight = [100, 83]
    assert route(r, req("A", 2), "a2")[:2] == (1, "guard")


def test_two_endpoints_on_one_host_are_two_decodes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["toy_proxy_server.py", "--prefiller-hosts", "p", "--prefiller-ports", "8400",
                                      "--decoder-hosts", "h", "h", "--decoder-ports", "7200", "7300"])
    args = proxy.parse_args()
    assert args.decoder_instances == [("h", 7200), ("h", 7300)]


def _app_endpoints(endpoints, monkeypatch, mode):
    """Proxy app with mock backends keyed by (host, port); records each forwarded decode body."""
    if mode is None:
        monkeypatch.delenv("PROXY_DECODE_ROUTING", raising=False)
    else:
        monkeypatch.setenv("PROXY_DECODE_ROUTING", mode)
    monkeypatch.setenv("PROXY_DECODE_AFFINITY_CHUNK", "128")
    monkeypatch.delenv("PROXY_DECODE_AFFINITY_SLACK", raising=False)
    monkeypatch.delenv("PROXY_DECODE_AFFINITY_MAX_SKEW", raising=False)
    seen = {ep: [] for ep in endpoints}
    box = [0]

    def prefill(request):
        box[0] += 1
        return httpx.Response(200, json={"kv_transfer_params": {"uuid": box[0]}, "usage": {"prompt_tokens": 5}})

    def decode(ep):
        def h(request):
            assert (request.url.host, request.url.port) == ep
            seen[ep].append(request.content)
            return httpx.Response(200, content=b'data: {"x": 1}\n\ndata: [DONE]\n\n')
        return h

    import itertools
    app = proxy.app
    app.state.prefill_clients = [{"client": httpx.AsyncClient(transport=httpx.MockTransport(prefill), base_url="http://p:1"),
                                  "host": "p", "port": 1, "id": 0}]
    app.state.decode_clients = [{"client": httpx.AsyncClient(transport=httpx.MockTransport(decode(ep)), base_url=f"http://{ep[0]}:{ep[1]}"),
                                 "host": ep[0], "port": ep[1], "id": i} for i, ep in enumerate(endpoints)]
    app.state.prefill_iterator = itertools.cycle(range(1))
    app.state.decode_iterator = itertools.cycle(range(len(endpoints)))
    app.state.decode_affinity = proxy._make_decode_affinity(len(endpoints))
    return app, seen


EPS = [("h", 7200), ("h", 7300)]


def _places(seen, name):
    return [ep for ep, bodies in seen.items() for b in bodies if f"[{name}] task" in json.loads(b)["messages"][1]["content"]]


def own_head(name: str, turn: int) -> dict:
    """req() with a per-conversation system prompt: no shared head between conversations."""
    b = req(name, turn)
    b["messages"] = [{"role": "system", "content": f"[{name}] own system prompt " * 40}] + b["messages"][1:]
    return b


def test_end_to_end_prefix_affinity_two_endpoints_one_host(monkeypatch):
    """No shared head: cold first turns rotate over the host's two endpoints; later turns stick to their endpoint."""
    app, seen = _app_endpoints(EPS, monkeypatch, "prefix_affinity")
    base._serve(app, [own_head(n, t) for n, t in ORDER])
    for n in "ABC":
        p = _places(seen, n)
        assert len(p) == 3 and len(set(p)) == 1, (n, p)
    assert all(seen[ep] for ep in EPS)               # both endpoints of the one host serve
    assert app.state.decode_affinity.inflight == [0, 0]
    c = app.state.decode_affinity.counts
    assert (c["cold"], c["sticky"], c["prefix"], c["branch"], c["guard"]) == (3, 6, 0, 0, 0)


def test_end_to_end_shared_head_follows_the_head_when_idle(monkeypatch):
    """Shared system head, serial requests (in-flight 0 at every choice, so the bound never binds): longest prefix
    sends every new conversation to the head's endpoint. Spreading happens only under load (bounded-load tests)."""
    app, seen = _app_endpoints(EPS, monkeypatch, "prefix_affinity")
    base._serve(app, [req(n, t) for n, t in ORDER])
    assert [len(v) for v in seen.values()] == [9, 0]
    c = app.state.decode_affinity.counts
    assert (c["cold"], c["prefix"], c["sticky"]) == (1, 2, 6)


def test_routing_identity_forwarded_bytes_equal_round_robin(monkeypatch):
    """Affinity only picks the endpoint: the multiset of forwarded decode bodies is byte-identical to round robin's."""
    bodies = [req(n, t) for n, t in ORDER]
    app, seen_aff = _app_endpoints(EPS, monkeypatch, "prefix_affinity")
    base._serve(app, bodies)
    app, seen_rr = _app_endpoints(EPS, monkeypatch, None)
    base._serve(app, bodies)
    flat = lambda s: sorted(b for v in s.values() for b in v)
    assert flat(seen_aff) == flat(seen_rr) and len(flat(seen_rr)) == len(bodies)
    assert [len(v) for v in seen_rr.values()] == [5, 4]  # round robin alternates endpoints
