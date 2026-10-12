# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only tests for vllm_torchtpu.core.prefix_aware_dp_router (no vLLM needed)."""

import array
import collections
import types
import zlib

import pytest

from vllm_torchtpu.core import prefix_aware_dp_router as router


def _make_client_cls():
    class FakeDPLBClient:
        FALLBACK = "fallback"

        def __init__(self, num_engines=4, client_count=1):
            self.core_engines = [f"engine{i}".encode() for i in range(num_engines)]
            self.lb_engines = [[0, 0, 0.0] for _ in range(num_engines)]
            self.engine_inflight = collections.Counter()
            self.reqs_in_flight = {}
            self.client_count = client_count
            self.fallback_calls = 0

        def get_core_engine_for_request(self, request):
            self.fallback_calls += 1
            return self.FALLBACK

    return FakeDPLBClient


def _req(tokens, rank=None, rid="r0"):
    return types.SimpleNamespace(
        prompt_token_ids=tokens, data_parallel_rank=rank, request_id=rid
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in (
        "ENABLE_PREFIX_ROUTING",
        "PREFIX_ROUTING_TOKENS",
        "PREFIX_ROUTING_MAX_QUEUE",
        "PREFIX_ROUTING_MAX_KV",
    ):
        monkeypatch.delenv(k, raising=False)


def test_prefix_route_index_matches_crc32_of_prefix():
    tokens = list(range(1000, 1700))
    expected = zlib.crc32(array.array("I", tokens[:512]).tobytes()) % 4
    assert router.prefix_route_index(tokens, 4, 512) == expected
    # Only the prefix matters.
    assert router.prefix_route_index(tokens + [7, 8, 9], 4, 512) == expected


def test_prefix_route_index_rejects_bad_args():
    with pytest.raises(ValueError):
        router.prefix_route_index([1, 2], 0, 512)
    with pytest.raises(ValueError):
        router.prefix_route_index([1, 2], 4, 0)


def test_routes_shared_prefix_to_same_engine_and_books_it():
    cls = _make_client_cls()
    assert router.install_prefix_aware_dp_routing(cls)
    client = cls()
    prefix = list(range(600))
    idx = router.prefix_route_index(prefix, 4, 512)
    chosen = client.get_core_engine_for_request(_req(prefix + [1], rid="a"))
    assert chosen == client.core_engines[idx]
    assert client.reqs_in_flight["a"] == chosen
    assert client.engine_inflight[chosen] == 1
    assert client.lb_engines[idx][0] == 1
    assert client.fallback_calls == 0


def test_falls_back_when_affine_engine_overloaded():
    cls = _make_client_cls()
    router.install_prefix_aware_dp_routing(cls)
    client = cls()
    tokens = list(range(50))
    idx = router.prefix_route_index(tokens, 4, 512)
    client.engine_inflight[client.core_engines[idx]] = 2  # == MAX_QUEUE default
    assert client.get_core_engine_for_request(_req(tokens)) == cls.FALLBACK
    client.engine_inflight.clear()
    client.lb_engines[idx] = [0, 0, 0.95]  # kv usage at the limit
    assert client.get_core_engine_for_request(_req(tokens)) == cls.FALLBACK
    client.lb_engines[idx] = [2, 0, 0.0]  # waiting at the limit
    assert client.get_core_engine_for_request(_req(tokens)) == cls.FALLBACK
    assert client.fallback_calls == 3


def test_explicit_rank_and_empty_prompt_use_stock_policy():
    cls = _make_client_cls()
    router.install_prefix_aware_dp_routing(cls)
    client = cls()
    assert client.get_core_engine_for_request(_req([1, 2, 3], rank=1)) == cls.FALLBACK
    assert client.get_core_engine_for_request(_req([])) == cls.FALLBACK
    assert client.get_core_engine_for_request(_req(None)) == cls.FALLBACK


def test_disabled_by_env_leaves_class_untouched(monkeypatch):
    monkeypatch.setenv("ENABLE_PREFIX_ROUTING", "0")
    cls = _make_client_cls()
    orig = cls.get_core_engine_for_request
    assert not router.install_prefix_aware_dp_routing(cls)
    assert cls.get_core_engine_for_request is orig


def test_install_is_idempotent():
    cls = _make_client_cls()
    assert router.install_prefix_aware_dp_routing(cls)
    patched = cls.get_core_engine_for_request
    assert router.install_prefix_aware_dp_routing(cls)
    assert cls.get_core_engine_for_request is patched
    client = cls()
    idx = router.prefix_route_index([5], 4, 512)
    client.engine_inflight[client.core_engines[idx]] = 9
    # A double wrap would recurse into the router again instead of the stock policy.
    assert client.get_core_engine_for_request(_req([5])) == cls.FALLBACK
    assert client.fallback_calls == 1
