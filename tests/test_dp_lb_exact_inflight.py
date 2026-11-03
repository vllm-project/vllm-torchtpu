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


# ---------------------------------------------------------------------------
# Pipelined prefill fan-out (TPU_DP_PIPELINE_PREFILL)
# ---------------------------------------------------------------------------


@pytest.fixture
def fanout(monkeypatch):
    """The patched client methods on a client built without __init__; vLLM's
    methods are restored afterwards."""
    from vllm.v1.engine.core_client import DPAsyncMPClient

    for name in (
        "get_core_engine_for_request",
        "add_request_async",
        "abort_requests_async",
    ):
        monkeypatch.setattr(DPLBAsyncMPClient, name, getattr(DPLBAsyncMPClient, name))
    monkeypatch.setattr(
        DPLBAsyncMPClient,
        "process_engine_outputs",
        DPLBAsyncMPClient.__dict__["process_engine_outputs"],
    )
    monkeypatch.setattr(
        dp_lb_exact_inflight.envs, "VLLM_TPU_DP_LB_EXACT_INFLIGHT", True
    )
    monkeypatch.setattr(dp_lb_exact_inflight.envs, "TPU_DP_PIPELINE_PREFILL", True)
    monkeypatch.setattr(dp_lb_exact_inflight.envs, "TPU_DP_PIPELINE_STAGES", 4)
    monkeypatch.setattr(dp_lb_exact_inflight.envs, "TPU_DP_PIPELINE_MIN_TOKENS", 100)
    monkeypatch.setattr(
        dp_lb_exact_inflight.envs, "TPU_DP_PIPELINE_SPLITS", "2:30;3:20,41;4:14,29,44"
    )
    monkeypatch.setattr(dp_lb_exact_inflight.envs, "TPU_DP_PIPELINE_STAGE_TOKENS", "")
    dp_lb_exact_inflight.patch_dp_lb_client()

    client = DPLBAsyncMPClient.__new__(DPLBAsyncMPClient)
    sent, aborted = [], []

    async def send_input(request_type, request, engine=None):
        sent.append((engine, request))

    async def abort(request_ids, engine):
        aborted.append((engine, list(request_ids)))

    vars(client).update(
        client_count=1,
        client_index=0,
        current_wave=0,
        engines_running=True,
        reqs_in_flight={},
        engine_inflight=Counter(),
        core_engines=ENGINES,
        lb_engines=[[0, 0, 0.0] for _ in ENGINES],
        eng_start_index=0,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(num_hidden_layers=60)
            ),
            # Chunks of 128 tokens; the buckets of their rest.
            scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
            compilation_config=SimpleNamespace(compile_sizes=[16, 32, 64, 128]),
        ),
        _send_input=send_input,
        _abort_requests=abort,
        _ensure_stats_update_task=lambda: None,
        _ensure_output_queue_task=lambda: None,
        resources=SimpleNamespace(engine_dead=False),
    )
    assert DPAsyncMPClient.add_request_async is not DPLBAsyncMPClient.add_request_async
    return client, sent, aborted


def _core_request(request_id, num_tokens, max_tokens=1):
    from vllm import SamplingParams
    from vllm.v1.engine import EngineCoreRequest

    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=max_tokens, temperature=0.0, logprobs=1
        ),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def test_long_prompt_fans_out_to_idle_engines(fanout):
    import asyncio

    from vllm_torchtpu.distributed.dp_pipeline.plan import plan_of

    client, sent, _ = fanout
    asyncio.run(client.add_request_async(_core_request("a", 500)))
    *guests, (owner, request) = sent
    assert len(guests) == 3 and request.request_id == "a"
    plan = plan_of(request.sampling_params)
    assert plan.bounds == (0, 14, 29, 44, 60)
    assert plan.ranks[-1] == ENGINES.index(owner)
    # A request that ends at its first token needs no K/V hand-off.
    assert not plan.kv
    assert request.sampling_params.logprobs == 1
    # 500 tokens: three full chunks pipelined, the 116-token rest pads to the
    # full chunk's bucket and is pipelined too.
    assert plan.tokens == 500
    for stage, (engine, guest) in enumerate(guests):
        assert guest.request_id == f"a-dp-pipeline-{stage}"
        assert plan.ranks[stage] == ENGINES.index(engine) != plan.ranks[-1]
        assert plan_of(guest.sampling_params) == plan
        assert guest.sampling_params.max_tokens == 1
        assert guest.sampling_params.logprobs is None
        assert guest.sampling_params.structured_outputs is None
        assert len(guest.prompt_token_ids) == 500
    assert all(client.engine_inflight[e] == 1 for e in ENGINES)

    # Every engine is busy now: the next long prompt is prefilled alone.
    asyncio.run(client.add_request_async(_core_request("b", 500)))
    assert plan_of(sent[-1][1].sampling_params) is None and len(sent) == 5


def test_short_prompt_and_busy_engines_run_alone(fanout):
    import asyncio

    from vllm_torchtpu.distributed.dp_pipeline.plan import plan_of

    client, sent, _ = fanout
    asyncio.run(client.add_request_async(_core_request("a", 50)))
    assert len(sent) == 1 and plan_of(sent[0][1].sampling_params) is None
    client.engine_inflight.clear()
    client.reqs_in_flight.clear()
    client.engine_inflight.update({ENGINES[0]: 1})
    # Two engines own requests: the pipeline takes the two that are free.
    client.engine_inflight.update({ENGINES[1]: 1})
    asyncio.run(client.add_request_async(_core_request("b", 500)))
    plan = plan_of(sent[-1][1].sampling_params)
    assert plan.num_stages == 2 and plan.bounds == (0, 30, 60)
    assert set(plan.ranks) == {2, 3}


def test_helpers_count_in_flight_before_the_first_send(fanout):
    import asyncio

    client, _, _ = fanout
    seen = []

    async def send_input(request_type, request, engine=None):
        seen.append(dict(client.engine_inflight))

    client._send_input = send_input
    asyncio.run(client.add_request_async(_core_request("a", 500)))
    # A request routed while the first copy is being sent finds every engine
    # of the pipeline busy.
    assert seen[0] == {e: 1 for e in ENGINES}


def test_a_request_routed_during_a_fan_out_goes_out_after_it(fanout):
    import asyncio

    client, sent, _ = fanout

    async def send_input(request_type, request, engine=None):
        await asyncio.sleep(0)  # a send that waits on the socket
        sent.append((engine, request))

    client._send_input = send_input

    async def both():
        await asyncio.gather(
            client.add_request_async(_core_request("a", 500)),
            client.add_request_async(_core_request("b", 50)),
        )

    asyncio.run(both())
    order = [request.request_id for _, request in sent]
    assert order.index("b") > order.index("a")


def _outputs(produced=(), finished=()):
    return SimpleNamespace(
        outputs=[SimpleNamespace(request_id=r) for r in produced],
        finished_requests=set(finished),
    )


def test_the_owner_computes_a_small_rest_alone(fanout):
    import asyncio

    from vllm_torchtpu.distributed.dp_pipeline.plan import plan_of

    client, sent, _ = fanout
    asyncio.run(client.add_request_async(_core_request("a", 420)))
    *guests, (_, request) = sent
    plan = plan_of(request.sampling_params)
    # The 36-token rest pads to a smaller bucket than the chunks'.
    assert plan.tokens == 384 and plan.kv
    assert all(len(g.prompt_token_ids) == 384 for _, g in guests)
    assert len(request.prompt_token_ids) == 420


def test_too_few_full_chunks_prefill_alone(fanout, monkeypatch):
    import asyncio

    from vllm_torchtpu.distributed.dp_pipeline.plan import plan_of

    client, sent, _ = fanout
    # 60 tokens: no full 128-token chunk, and a rest below the chunk's bucket.
    monkeypatch.setattr(dp_lb_exact_inflight.envs, "TPU_DP_PIPELINE_MIN_TOKENS", 50)
    asyncio.run(client.add_request_async(_core_request("a", 60)))
    assert len(sent) == 1 and plan_of(sent[0][1].sampling_params) is None
    # 300 tokens: two full chunks are fewer than the 300 the pipeline needs.
    monkeypatch.setattr(dp_lb_exact_inflight.envs, "TPU_DP_PIPELINE_MIN_TOKENS", 300)
    asyncio.run(client.add_request_async(_core_request("b", 300)))
    assert len(sent) == 2 and plan_of(sent[1][1].sampling_params) is None


def test_a_decoding_pipelined_request_carries_kv(fanout):
    import asyncio

    from vllm_torchtpu.distributed.dp_pipeline.plan import plan_of

    client, sent, _ = fanout
    asyncio.run(client.add_request_async(_core_request("d", 500, max_tokens=8)))
    *guests, (_, request) = sent
    assert plan_of(request.sampling_params).kv
    assert all(g.sampling_params.max_tokens == 1 for _, g in guests)


def test_an_abort_reaches_a_pipelined_request_with_its_first_token(fanout):
    import asyncio

    client, sent, aborted = fanout
    asyncio.run(client.add_request_async(_core_request("a", 500, max_tokens=8)))
    asyncio.run(client.add_request_async(_core_request("short", 50)))
    asyncio.run(client.abort_requests_async(["a", "short"]))
    # Stopping an engine midway would stall the stages after it.
    assert [i for _, ids in aborted for i in ids] == ["short"]
    process = DPLBAsyncMPClient.process_engine_outputs
    asyncio.run(process(client, _outputs(produced=["a-dp-pipeline-0"])))
    assert [i for _, ids in aborted for i in ids] == ["short"]
    asyncio.run(process(client, _outputs(produced=["a"])))
    assert [i for _, ids in aborted for i in ids] == ["short", "a"]
    assert "a" not in client._tpu_dp_pipeline_guests
    # After its prompt, an abort reaches it at once.
    asyncio.run(client.add_request_async(_core_request("b", 500, max_tokens=8)))
    asyncio.run(process(client, _outputs(produced=["b"])))
    asyncio.run(client.abort_requests_async(["b"]))
    assert [i for _, ids in aborted for i in ids] == ["short", "a", "b"]


def test_shutdown_unlinks_the_servers_rings(monkeypatch):
    from vllm_torchtpu.distributed.dp_pipeline import transport

    calls = []

    class Client:
        core_engines = ENGINES
        vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=2)
        )

        async def add_request_async(self, request):
            pass

        async def abort_requests_async(self, request_ids):
            pass

        @staticmethod
        async def process_engine_outputs(self, outputs):
            pass

        def shutdown(self, timeout=None):
            calls.append(("engines", timeout))

    monkeypatch.setattr(
        transport, "unlink_rings", lambda *args: calls.append(("rings", *args))
    )
    dp_lb_exact_inflight._patch_pipeline_fanout(Client)
    Client().shutdown(5.0)
    server = dp_lb_exact_inflight._server_id()
    assert calls == [("engines", 5.0), ("rings", server, len(ENGINES), 2)]


def test_a_freshly_freed_engine_does_not_help(fanout, monkeypatch):
    import asyncio

    from vllm_torchtpu.distributed.dp_pipeline.plan import plan_of

    client, sent, _ = fanout
    monkeypatch.setattr(
        dp_lb_exact_inflight.envs, "TPU_DP_PIPELINE_HELPER_IDLE_MS", 60_000.0
    )
    asyncio.run(client.add_request_async(_core_request("a", 50)))
    freed = sent[-1][0]
    process = DPLBAsyncMPClient.process_engine_outputs
    asyncio.run(process(client, _outputs(finished=["a"])))
    asyncio.run(client.add_request_async(_core_request("b", 500)))
    plan = plan_of(sent[-1][1].sampling_params)
    # A closed-loop client refills a freed engine within milliseconds.
    helpers = [ENGINES[r] for r in plan.ranks[:-1]]
    assert freed not in helpers and len(helpers) == 2


def test_shorter_prompts_take_fewer_stages(fanout, monkeypatch):
    import asyncio

    from vllm_torchtpu.distributed.dp_pipeline.plan import plan_of

    client, sent, _ = fanout
    monkeypatch.setattr(
        dp_lb_exact_inflight.envs, "TPU_DP_PIPELINE_STAGE_TOKENS", "3:400;4:800"
    )
    asyncio.run(client.add_request_async(_core_request("a", 500)))
    assert plan_of(sent[-1][1].sampling_params).num_stages == 3


def test_pipeline_needs_exact_inflight_routing(monkeypatch):
    monkeypatch.setattr(dp_lb_exact_inflight.envs, "TPU_DP_PIPELINE_PREFILL", True)
    monkeypatch.setattr(
        dp_lb_exact_inflight.envs, "VLLM_TPU_DP_LB_EXACT_INFLIGHT", False
    )
    with pytest.raises(ValueError, match="VLLM_TPU_DP_LB_EXACT_INFLIGHT=1"):
        dp_lb_exact_inflight.check_api_server_count(
            SimpleNamespace(_api_process_count=1)
        )
