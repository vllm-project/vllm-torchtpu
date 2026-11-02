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
"""DP routing by exact in-flight counts, on vLLM's real routing function."""

from collections import Counter
from types import SimpleNamespace

import pytest
from vllm.v1.engine.core_client import DPLBAsyncMPClient

from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.core import dp_lb_exact_inflight

ENGINES = [i.to_bytes(2, "little") for i in range(4)]


@pytest.fixture
def routing(monkeypatch):
    """Restore vLLM's routing method after each test."""
    monkeypatch.setattr(
        DPLBAsyncMPClient,
        "get_core_engine_for_request",
        DPLBAsyncMPClient.get_core_engine_for_request,
    )
    return DPLBAsyncMPClient.get_core_engine_for_request


def _wave_boundary_client():
    """Engine 0 just finished its request; engines 1-3 still decode theirs.
    The coordinator's snapshot predates the finish, so all four show one
    running request, and the scan starts at engine 1."""
    return SimpleNamespace(
        client_count=1,
        reqs_in_flight={},
        engine_inflight=Counter({e: 1 for e in ENGINES[1:]}),
        core_engines=ENGINES,
        lb_engines=[[0, 1, 0.0] for _ in ENGINES],
        eng_start_index=1,
    )


def _request(request_id):
    return SimpleNamespace(
        request_id=request_id, data_parallel_rank=None, pooling_params=None
    )


def _patch(monkeypatch, enabled):
    monkeypatch.setattr(
        dp_lb_exact_inflight.envs, "VLLM_TPU_DP_LB_EXACT_INFLIGHT", enabled
    )
    dp_lb_exact_inflight.patch_dp_lb_client()


def test_upstream_score_ties_the_freed_engine_with_busy_ones(routing):
    client = _wave_boundary_client()
    # The stale snapshot hides that engine 0 is free: the scan picks engine 1.
    assert routing(client, _request("a")) == ENGINES[1]


def test_routes_to_the_engine_with_fewest_in_flight(monkeypatch, routing):
    _patch(monkeypatch, True)
    client = _wave_boundary_client()
    snapshot = client.lb_engines
    route = DPLBAsyncMPClient.get_core_engine_for_request
    assert route(client, _request("a")) == ENGINES[0]
    assert client.lb_engines is snapshot
    assert client.engine_inflight[ENGINES[0]] == 1
    # A burst then spreads over the equally loaded engines.
    picks = [route(client, _request(r)) for r in "bcd"]
    assert len(set(picks)) == 3
    counts = [client.engine_inflight[e] for e in ENGINES]
    assert max(counts) - min(counts) == 1


def test_flag_off_leaves_vllm_routing(monkeypatch, routing):
    _patch(monkeypatch, False)
    assert DPLBAsyncMPClient.get_core_engine_for_request is routing


def test_refuses_several_api_servers(monkeypatch):
    monkeypatch.setattr(
        dp_lb_exact_inflight.envs, "VLLM_TPU_DP_LB_EXACT_INFLIGHT", True
    )
    with pytest.raises(ValueError, match="--api-server-count=1"):
        dp_lb_exact_inflight.check_api_server_count(
            SimpleNamespace(_api_process_count=4)
        )
    dp_lb_exact_inflight.check_api_server_count(SimpleNamespace(_api_process_count=1))


def test_flag_stays_out_of_the_compile_key():
    name = "VLLM_TPU_DP_LB_EXACT_INFLIGHT"
    assert name in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert name not in tpu_compiler._tpu_compile_env_factors()
