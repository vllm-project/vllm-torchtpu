# SPDX-License-Identifier: Apache-2.0
"""PROXY_PREFILL_ROUTING=prefix_hash_sticky: prefix_hash with PROXY_PREFIX_HASH_SATURATED_INFLIGHT
defaulting to 1. Opt-in; every other routing value keeps the default 0."""

import argparse
import importlib.util
from pathlib import Path

import pytest


def _load_toy_proxy_server():
    path = (
        Path(__file__).resolve().parents[2] / "examples" / "disagg" / "toy_proxy_server.py"
    )
    spec = importlib.util.spec_from_file_location("toy_proxy_server_sticky", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_m = _load_toy_proxy_server()


@pytest.mark.parametrize(
    "routing, explicit, want",
    [
        (None, None, 0),
        ("prefix_hash", None, 0),
        ("round_robin", None, 0),
        ("dp_affinity", None, 0),
        ("prefix_hash_sticky", None, 1),
        (" Prefix_Hash_Sticky ", None, 1),
        ("prefix_hash_sticky", "0", 0),
        ("prefix_hash_sticky", "3", 3),
        ("prefix_hash", "2", 2),
    ],
)
def test_saturated_inflight_default_is_off_unless_sticky(monkeypatch, routing, explicit, want) -> None:
    for k, v in (("PROXY_PREFILL_ROUTING", routing), ("PROXY_PREFIX_HASH_SATURATED_INFLIGHT", explicit)):
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    assert _m._prefix_hash_saturated_inflight() == want


def _startup(monkeypatch, capsys, routing, n_prefill=2):
    from fastapi.testclient import TestClient

    if routing is None:
        monkeypatch.delenv("PROXY_PREFILL_ROUTING", raising=False)
    else:
        monkeypatch.setenv("PROXY_PREFILL_ROUTING", routing)
    monkeypatch.delenv("PROXY_PREFIX_HASH_SATURATED_INFLIGHT", raising=False)
    monkeypatch.delenv("PROXY_DECODE_ROUTING", raising=False)
    monkeypatch.setenv("PROXY_LATENCY_LOG_INTERVAL", "0")
    _m.global_args = argparse.Namespace(
        prefiller_instances=[(f"p{i}", 8000) for i in range(n_prefill)],
        decoder_instances=[("d0", 7200)],
    )
    with TestClient(_m.app):
        router = _m.app.state.prefill_router
    return router, capsys.readouterr().out


def test_default_prefix_hash_router_has_saturation_off(monkeypatch, capsys) -> None:
    router, out = _startup(monkeypatch, capsys, None)
    assert isinstance(router, _m._PrefixAffinityRouter)
    assert router.saturated_inflight == 0
    assert "routing=prefix_hash " in out and "saturated_inflight=off" in out


def test_sticky_alias_builds_the_prefix_hash_router_with_saturation_1(monkeypatch, capsys) -> None:
    router, out = _startup(monkeypatch, capsys, "prefix_hash_sticky")
    assert isinstance(router, _m._PrefixAffinityRouter)
    assert router.saturated_inflight == 1
    assert "routing=prefix_hash_sticky " in out and "saturated_inflight=1" in out


def test_sticky_alias_with_one_prefill_leg_is_round_robin(monkeypatch, capsys) -> None:
    router, out = _startup(monkeypatch, capsys, "prefix_hash_sticky", n_prefill=1)
    assert router is None
    assert "routing=round_robin" in out


def test_sticky_alias_does_not_select_the_llmd_proxy(monkeypatch) -> None:
    monkeypatch.setenv("PROXY_PREFILL_ROUTING", "prefix_hash_sticky")
    monkeypatch.delenv("PROXY_DECODE_ROUTING", raising=False)
    assert not _m._wants_llmd_proxy()
