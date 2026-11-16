# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


def _load_toy_proxy_server():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "disagg"
        / "toy_proxy_server.py"
    )
    spec = importlib.util.spec_from_file_location("toy_proxy_server_routing", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_proxy = _load_toy_proxy_server()
_LegRouter = _proxy._LegRouter


def _router(policy: str, n: int = 3, cap: int = 0):
    return _LegRouter("decode", n, policy, cap, log_interval_s=0)


def test_round_robin_is_the_default_order() -> None:
    r = _router("round_robin")
    assert [r.acquire("conv-a")[0] for _ in range(4)] == [0, 1, 2, 0]


def test_unknown_policy_is_refused() -> None:
    with pytest.raises(ValueError, match="prefix_hash"):
        _router("prefix_hash")


def test_sticky_keeps_each_conversation_on_its_instance() -> None:
    r = _router("sticky")
    first = {}
    for conv in ("a", "b", "c"):
        idx, route = r.acquire(conv)
        assert route == "new"
        first[conv] = idx
    # New conversations spread over the instances by in-flight count.
    assert sorted(first.values()) == [0, 1, 2]
    for _ in range(5):
        for conv in ("a", "b", "c"):
            idx, route = r.acquire(conv)
            assert (idx, route) == (first[conv], "sticky")
            r.release(idx)
    assert r.owned == [1, 1, 1]
    assert r.routes["sticky"] == 15


def test_new_conversation_goes_to_least_in_flight() -> None:
    r = _router("sticky")
    a, _ = r.acquire("a")  # instance 0, still in flight
    b, _ = r.acquire("b")  # instance 1, still in flight
    r.release(b)
    # 0 holds one request, 1 and 2 none; 1 owns a conversation, 2 does not.
    assert r.acquire("c") == (2, "new")
    r.release(2)
    r.release(a)
    # All idle, each owns one conversation and was sent one request: the
    # lowest index wins the tie.
    idx, route = r.acquire("d")
    assert route == "new"
    assert idx == 0


def test_cap_spills_to_least_in_flight_and_moves_the_conversation() -> None:
    r = _router("sticky", n=2, cap=2)
    assert r.acquire("a") == (0, "new")
    assert r.acquire("b") == (1, "new")
    assert r.acquire("a") == (0, "sticky")  # instance 0 now holds 2: at the cap
    # a's next turn finds instance 0 at the cap and instance 1 under it.
    assert r.acquire("a") == (1, "spill")
    assert r.owner["a"] == 1
    assert r.owned == [0, 2]
    r.release(0)
    r.release(0)
    r.release(1)
    # Once moved, the conversation stays on its new instance while it is under
    # the cap, even with instance 0 idle.
    assert r.acquire("a") == (1, "sticky")


def test_cap_keeps_the_conversation_when_every_instance_is_full() -> None:
    r = _router("sticky", n=2, cap=1)
    assert r.acquire("a") == (0, "new")
    assert r.acquire("b") == (1, "new")
    assert r.acquire("a") == (0, "full")
    assert r.owner["a"] == 0
    assert r.inflight == [2, 1]


def test_request_without_key_pins_nothing() -> None:
    r = _router("sticky")
    assert r.acquire(None) == (0, "nokey")
    assert r.acquire("") == (1, "nokey")
    assert r.owner == {}
    assert r.routes["nokey"] == 2


def test_least_loaded_ignores_the_key() -> None:
    r = _router("least_loaded", n=2)
    assert r.acquire("a") == (0, "least")
    assert r.acquire("a") == (1, "least")


def _run_proxy_requests(monkeypatch, convs: list[str | None]) -> tuple[list, list]:
    """Sends one streamed chat request per entry through the proxy app."""
    monkeypatch.setenv("PROXY_DECODE_ROUTING", "sticky")
    monkeypatch.setenv("PROXY_DECODE_STICKY_MAX_INFLIGHT", "8")
    monkeypatch.setenv("PROXY_LATENCY_LOG_INTERVAL", "0")
    _proxy.global_args = SimpleNamespace(
        prefiller_instances=[("prefill-0", 8000)],
        decoder_instances=[(f"decode-{i}", 7200) for i in range(3)],
    )
    seen_decode: list[str] = []

    def prefill_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["max_tokens"] == 1 and body["stream"] is False
        return httpx.Response(200, json={"kv_transfer_params": {"uuid": 7}})

    def decode_handler(host: str):
        def handler(request: httpx.Request) -> httpx.Response:
            seen_decode.append(host)
            assert json.loads(request.content)["kv_transfer_params"] == {"uuid": 7}
            return httpx.Response(200, content=b'data: {"x": 1}\n\ndata: [DONE]\n\n')

        return handler

    async def main():
        app = _proxy.app
        async with _proxy.lifespan(app):
            for info in app.state.prefill_clients:
                info["client"] = httpx.AsyncClient(
                    transport=httpx.MockTransport(prefill_handler),
                    base_url="http://prefill",
                )
            for info in app.state.decode_clients:
                info["client"] = httpx.AsyncClient(
                    transport=httpx.MockTransport(decode_handler(info["host"])),
                    base_url="http://decode",
                )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://proxy"
            ) as client:
                bodies = []
                for conv in convs:
                    headers = {} if conv is None else {"X-Correlation-ID": conv}
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={"messages": [], "stream": True, "max_tokens": 8},
                        headers=headers,
                    )
                    assert resp.status_code == 200
                    bodies.append(resp.content)
            router = app.state.decode_router
            # Every stream ended, so nothing is left in flight.
            assert router.inflight == [0, 0, 0]
            assert app.state.prefill_router.inflight == [0]
            return bodies

    bodies = asyncio.run(main())
    return bodies, seen_decode


def test_proxy_routes_turns_of_a_conversation_to_one_decode(monkeypatch) -> None:
    convs = ["a", "b", "c", "a", "b", "c", "a", None]
    bodies, seen = _run_proxy_requests(monkeypatch, convs)
    assert all(b.endswith(b"data: [DONE]\n\n") for b in bodies)
    by_conv: dict[str, set[str]] = {}
    for conv, host in zip(convs, seen):
        if conv is not None:
            by_conv.setdefault(conv, set()).add(host)
    assert all(len(hosts) == 1 for hosts in by_conv.values())
    # Requests run one at a time, so the three conversations spread over the
    # three decodes by owned count.
    assert len({next(iter(h)) for h in by_conv.values()}) == 3


@pytest.mark.parametrize("forward", [False, True])
def test_prefill_reply_without_kv_params_is_not_sent_to_decode(
    monkeypatch, capsys, forward
) -> None:
    """A prefill reply without kv_transfer_params fails the request with 502
    and one log line; decode, which would prefill the whole prompt itself, is
    never called. PROXY_FORWARD_WITHOUT_KV_PARAMS=1 forwards it as before."""
    monkeypatch.setenv("PROXY_DECODE_ROUTING", "sticky")
    monkeypatch.setenv("PROXY_LATENCY_LOG_INTERVAL", "0")
    monkeypatch.setattr(_proxy, "_FORWARD_WITHOUT_KV_PARAMS", forward)
    _proxy.global_args = SimpleNamespace(
        prefiller_instances=[("prefill-0", 8000)],
        decoder_instances=[("decode-0", 7200)],
    )
    decode_bodies: list[dict] = []

    def prefill_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "cmpl-1", "choices": []})

    def decode_handler(request: httpx.Request) -> httpx.Response:
        decode_bodies.append(json.loads(request.content))
        return httpx.Response(200, content=b'data: {"x": 1}\n\ndata: [DONE]\n\n')

    async def main():
        app = _proxy.app
        async with _proxy.lifespan(app):
            app.state.prefill_clients[0]["client"] = httpx.AsyncClient(
                transport=httpx.MockTransport(prefill_handler),
                base_url="http://prefill",
            )
            app.state.decode_clients[0]["client"] = httpx.AsyncClient(
                transport=httpx.MockTransport(decode_handler),
                base_url="http://decode",
            )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://proxy"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"messages": [], "stream": True, "max_tokens": 8},
                    headers={"X-Correlation-ID": "a"},
                )
            assert app.state.prefill_router.inflight == [0]
            assert app.state.decode_router.inflight == [0]
            return resp

    resp = asyncio.run(main())
    logged = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "missing_kv_transfer_params" in line
    ]
    if forward:
        assert resp.status_code == 200
        assert resp.content.endswith(b"data: [DONE]\n\n")
        assert len(decode_bodies) == 1
        assert "kv_transfer_params" not in decode_bodies[0]
        assert logged == []
    else:
        assert resp.status_code == 502
        assert "kv_transfer_params" in resp.json()["error"]["message"]
        assert decode_bodies == []
        assert len(logged) == 1 and 'conv="a"' in logged[0]
