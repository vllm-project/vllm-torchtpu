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
"""Prefix-affinity DP routing policy and client contracts; CPU only."""

from collections import Counter
from types import SimpleNamespace

import pytest
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import DPLBAsyncMPClient, EngineCoreClient

from vllm_torchtpu.core.dp_prefix_affinity import (
    PrefixAffinityDPLBAsyncMPClient,
    PrefixAffinityRouter,
    patch_make_async_mp_client,
    prefix_affinity_enabled,
)
from vllm_torchtpu.runner.utils import DP_PREFIX_AFFINITY_KEY

CHUNK = 4
SYSTEM = list(range(1000, 1000 + 8 * CHUNK))


def _conversation(conv: int, turns: int) -> list[list[int]]:
    """Prompts of one conversation: shared system prompt, then a growing tail."""
    tail: list[int] = []
    prompts = []
    for turn in range(turns):
        new_tokens = (10 if turn == 0 else 3) * CHUNK
        tail += [(conv + 1) * 100_000 + turn * 100 + i for i in range(new_tokens)]
        prompts.append(SYSTEM + tail)
    return prompts


def _route(router: PrefixAffinityRouter, prompt: list[int], loads: list[int]) -> int:
    """Router decision with the least-loaded fallback of the upstream client."""
    hashes = router.chunk_hashes(prompt)
    engine = router.choose(hashes, loads)
    if engine is None:
        engine = loads.index(min(loads))
    router.record(engine, hashes)
    return engine


def test_conversation_turns_stick_to_their_engine():
    router = PrefixAffinityRouter(2, CHUNK, capacity_chunks=10_000)
    convs = [_conversation(c, turns=5) for c in range(6)]
    loads = [0, 0]
    home = {}
    for c, prompts in enumerate(convs):
        home[c] = _route(router, prompts[0], loads)
        loads[home[c]] += 1
    assert sorted(Counter(home.values()).values()) == [3, 3]
    for turn in range(1, 5):
        for c, prompts in enumerate(convs):
            assert _route(router, prompts[turn], loads) == home[c]


def test_shared_system_prompt_alone_does_not_pin_new_conversations():
    router = PrefixAffinityRouter(2, CHUNK, capacity_chunks=10_000)
    # Cold start: only engine 0 holds the system prompt.
    router.record(0, router.chunk_hashes(_conversation(0, turns=1)[0]))
    new_prompt = SYSTEM + list(range(10 * CHUNK))
    assert router.choose(router.chunk_hashes(new_prompt), [5, 0]) is None
    # Warm: both engines hold it, so it never breaks the tie.
    router.record(1, router.chunk_hashes(_conversation(1, turns=1)[0]))
    short_tail = SYSTEM + list(range(CHUNK))
    assert router.choose(router.chunk_hashes(short_tail), [0, 9]) is None
    homes = Counter(
        _route(router, SYSTEM + [c] * (2 * CHUNK), loads=[c % 2, (c + 1) % 2])
        for c in range(10)
    )
    assert homes == Counter({0: 5, 1: 5})


@pytest.mark.parametrize(
    ("loads", "expected"),
    [
        ([0, 0], 0),
        ([8, 0], 0),  # absolute imbalance within bound
        ([30, 21], 0),  # relative imbalance within bound
        ([9, 0], None),
        ([40, 26], None),
    ],
)
def test_load_bound(loads, expected):
    router = PrefixAffinityRouter(2, CHUNK, capacity_chunks=10_000)
    prompts = _conversation(0, turns=2)
    router.record(0, router.chunk_hashes(prompts[0]))
    assert router.choose(router.chunk_hashes(prompts[1]), loads) == expected


def test_lru_evicts_leaves_first_and_forgets_cold_conversations():
    router = PrefixAffinityRouter(2, CHUNK, capacity_chunks=11)
    old, new = _conversation(0, turns=2), _conversation(1, turns=2)
    router.record(0, router.chunk_hashes(old[0]))
    router.record(0, router.chunk_hashes(new[0]))
    assert len(router.engines[0]) == 11
    # The cold conversation loses its whole tail before the shared root.
    system_chunks = len(SYSTEM) // CHUNK
    assert router.match_lengths(router.chunk_hashes(old[1])) == [system_chunks, 0]
    assert router.match_lengths(router.chunk_hashes(new[1])) == [11, 0]


def _client(num_engines: int = 2) -> PrefixAffinityDPLBAsyncMPClient:
    client = object.__new__(PrefixAffinityDPLBAsyncMPClient)
    client.core_engines = [bytes([i, 0]) for i in range(num_engines)]
    client.lb_engines = [[0, 0, 0.0] for _ in range(num_engines)]
    client.reqs_in_flight = {}
    client.engine_inflight = Counter()
    client.client_count = 1
    client.eng_start_index = 0
    client._cache_config = SimpleNamespace(num_gpu_blocks=2_000, block_size=CHUNK)
    client._router = None
    return client


def _request(request_id: str, prompt: list[int], dp_rank: int | None = None):
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt,
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=dp_rank,
    )


def test_client_routes_turns_through_upstream_bookkeeping():
    client = _client()
    a, b = _conversation(0, turns=3), _conversation(1, turns=3)
    first_a = client.get_core_engine_for_request(_request("a0", a[0]))
    first_b = client.get_core_engine_for_request(_request("b0", b[0]))
    assert first_a != first_b
    for turn in (1, 2):
        req = _request(f"a{turn}", a[turn])
        assert client.get_core_engine_for_request(req) == first_a
        assert req.data_parallel_rank == client.core_engines.index(first_a)
    assert client.engine_inflight == Counter({first_a: 3, first_b: 1})
    assert client._router.capacity_chunks == 4 * 2_000 // 2
    # An explicit X-data-parallel-rank wins and is learned for later turns.
    req = _request("b1", b[1], dp_rank=client.core_engines.index(first_a))
    assert client.get_core_engine_for_request(req) == first_a
    assert client.get_core_engine_for_request(_request("b2", b[2])) == first_a


def _vllm_config(additional_config, dp: int = 2, external_lb: bool = False):
    return SimpleNamespace(
        additional_config=additional_config,
        parallel_config=SimpleNamespace(
            data_parallel_size=dp, data_parallel_external_lb=external_lb
        ),
    )


def test_enable_flag_validation():
    assert not prefix_affinity_enabled(_vllm_config({}))
    assert not prefix_affinity_enabled(_vllm_config({DP_PREFIX_AFFINITY_KEY: False}))
    assert prefix_affinity_enabled(_vllm_config({DP_PREFIX_AFFINITY_KEY: True}))
    with pytest.raises(ValueError, match="must be a bool"):
        prefix_affinity_enabled(_vllm_config({DP_PREFIX_AFFINITY_KEY: 1}))
    with pytest.raises(ValueError, match="data-parallel-size"):
        prefix_affinity_enabled(_vllm_config({DP_PREFIX_AFFINITY_KEY: True}, dp=1))
    with pytest.raises(ValueError, match="external"):
        prefix_affinity_enabled(
            _vllm_config({DP_PREFIX_AFFINITY_KEY: True}, external_lb=True)
        )


def test_factory_selects_client_and_requires_one_api_server(monkeypatch):
    built = []

    def fake_init(self, vllm_config, *args, **kwargs):
        built.append(type(self))
        self.client_count = args[3]
        self.core_engines = [b"\x00\x00", b"\x01\x00"]

    monkeypatch.setattr(DPLBAsyncMPClient, "__init__", fake_init)
    monkeypatch.setattr(
        EngineCoreClient,
        "make_async_mp_client",
        EngineCoreClient.__dict__["make_async_mp_client"],
    )
    monkeypatch.delattr(EngineCoreClient, "_tpu_dp_prefix_affinity_patch", False)
    patch_make_async_mp_client()
    config = _vllm_config({DP_PREFIX_AFFINITY_KEY: True})
    config.cache_config = SimpleNamespace(num_gpu_blocks=None, block_size=CHUNK)
    client = EngineCoreClient.make_async_mp_client(config, None, False, None, 1, 0)
    assert isinstance(client, PrefixAffinityDPLBAsyncMPClient)
    with pytest.raises(ValueError, match="--api-server-count=1"):
        EngineCoreClient.make_async_mp_client(config, None, False, None, 2, 0)
    assert built == [PrefixAffinityDPLBAsyncMPClient] * 2
