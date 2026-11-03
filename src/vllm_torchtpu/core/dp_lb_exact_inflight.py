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
"""Data-parallel routing by the API server's exact in-flight counts.

vLLM's ``DPLBAsyncMPClient`` scores each engine by ``max(client_count *
in_flight, waiting + running)``, the latter from the DP coordinator's latest
snapshot. The coordinator republishes at most every 100 ms (later under MoE
DP), each snapshot replaces the client's local counts, and a snapshot still
counts requests that finished after it was taken. When a closed-loop wave of
requests finishes within a few tens of milliseconds, an engine that just freed
up then ties with busy ones and the rotating tie-break can stack the next
requests on one engine. With one API server the client's own in-flight counts
are exact, so ``VLLM_TPU_DP_LB_EXACT_INFLIGHT`` has the upstream scan rank
engines by those alone; upstream's penalty on an engine's waiting queue under
KV-cache pressure goes with the snapshot. vLLM exposes no hook for this score.

With ``TPU_DP_PIPELINE_PREFILL`` the same client also decides which prompts are
prefilled as a pipeline over idle engines (``distributed/dp_pipeline``): it
routes the request as above, picks up to ``TPU_DP_PIPELINE_STAGES - 1`` engines
that have had nothing in flight for ``TPU_DP_PIPELINE_HELPER_IDLE_MS`` as
helpers, and sends each a copy of the request (``max_tokens=1``, the plan in
``extra_args``) before the request itself. The copies count as in flight on
their engines until they finish, so no request is routed to a helper while
another engine is free. vLLM has no hook for sending one request to several
engines either.
"""

import asyncio
import copy
import itertools
import math
import os
import time

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_MARKER = "_tpu_exact_inflight_patch"
_PIPELINE_MARKER = "_tpu_dp_pipeline_patch"


def check_api_server_count(parallel_config) -> None:
    """Refuse the flag with several API servers: each counts only the requests
    it routed itself."""
    num_api_servers = parallel_config._api_process_count
    if envs.VLLM_TPU_DP_LB_EXACT_INFLIGHT and num_api_servers > 1:
        raise ValueError(
            "VLLM_TPU_DP_LB_EXACT_INFLIGHT needs --api-server-count=1: each of "
            f"the {num_api_servers} API servers would rank engines by the "
            "requests it routed alone."
        )
    if envs.TPU_DP_PIPELINE_PREFILL and not envs.VLLM_TPU_DP_LB_EXACT_INFLIGHT:
        raise ValueError(
            "TPU_DP_PIPELINE_PREFILL needs VLLM_TPU_DP_LB_EXACT_INFLIGHT=1: helpers "
            "are the engines this API server knows to have nothing in flight."
        )


def patch_dp_lb_client() -> None:
    """With the flag set, rank engines by in-flight counts in
    ``DPLBAsyncMPClient.get_core_engine_for_request``."""
    if not envs.VLLM_TPU_DP_LB_EXACT_INFLIGHT:
        return
    from vllm.v1.engine.core_client import DPLBAsyncMPClient

    original = DPLBAsyncMPClient.get_core_engine_for_request
    if getattr(original, _MARKER, False):
        return

    def get_core_engine_for_request(self, request):
        # check_api_server_count refuses this at startup.
        assert self.client_count == 1, "VLLM_TPU_DP_LB_EXACT_INFLIGHT: one API server"
        snapshot = self.lb_engines
        # Upstream's score max(in_flight, waiting + running) becomes the exact
        # in-flight count; ties keep upstream's rotating scan.
        self.lb_engines = [
            [0, self.engine_inflight[engine], *counts[2:]]
            for engine, counts in zip(self.core_engines, snapshot)
        ]
        try:
            return original(self, request)
        finally:
            self.lb_engines = snapshot

    setattr(get_core_engine_for_request, _MARKER, True)
    DPLBAsyncMPClient.get_core_engine_for_request = get_core_engine_for_request
    logger.info("Applied TPU patch: DP load balancing by exact in-flight counts.")
    if envs.TPU_DP_PIPELINE_PREFILL:
        _patch_pipeline_fanout(DPLBAsyncMPClient)


_pipeline_ids = itertools.count(1)


def _server_id() -> int:
    from vllm_torchtpu.distributed.dp_pipeline.plan import server_id_of

    # The process id names this server's plans and hand-off rings (transport.py).
    return server_id_of(os.getpid())


def plan_pipeline(client, request):
    """Route ``request``; if it should be a pipelined prefill, return the plan
    and the helper engines in stage order, else None."""
    from vllm_torchtpu.distributed.dp_pipeline.plan import (
        PipelinePlan,
        make_pipeline_id,
        max_stages,
        pipelined_tokens,
        stage_bounds,
    )

    prompt = request.prompt_token_ids
    if (
        prompt is None
        or len(prompt) < envs.TPU_DP_PIPELINE_MIN_TOKENS
        or request.pooling_params is not None
        or request.data_parallel_rank is not None
        or request.request_id in client.reqs_in_flight
        or request.sampling_params is None
    ):
        return None
    owner = client.get_core_engine_for_request(request)
    idle_since = client.__dict__.get("_tpu_dp_pipeline_idle_since", {})
    rested = time.monotonic() - envs.TPU_DP_PIPELINE_HELPER_IDLE_MS / 1e3
    idle = [
        e
        for e in client.core_engines
        if e != owner
        and client.engine_inflight[e] == 0
        and idle_since.get(e, -math.inf) <= rested
    ]
    helpers = idle[: max_stages(len(prompt)) - 1]
    if not helpers:
        return None
    engines = [*helpers, owner]
    config = client.vllm_config
    num_layers = config.model_config.hf_text_config.num_hidden_layers
    tokens = pipelined_tokens(
        len(prompt),
        config.scheduler_config.max_num_batched_tokens,
        sorted(config.compilation_config.compile_sizes),
    )
    if tokens < envs.TPU_DP_PIPELINE_MIN_TOKENS:
        return None  # too few full chunks to pipeline
    plan = PipelinePlan(
        make_pipeline_id(_server_id(), next(_pipeline_ids)),
        tuple(client.core_engines.index(e) for e in engines),
        stage_bounds(len(engines), num_layers),
        tokens,
        # The owner goes on alone, past the pipelined tokens or its first
        # token, from every layer's K/V.
        kv=tokens < len(prompt) or request.sampling_params.max_tokens != 1,
    )
    return plan, helpers


def _with_plan(sampling_params, plan, *, helper: bool):
    from vllm_torchtpu.distributed.dp_pipeline.plan import PLAN_KEY

    params = copy.copy(sampling_params)
    params.extra_args = {**(sampling_params.extra_args or {}), PLAN_KEY: plan.to_json()}
    if helper:
        # A helper's copy ends with the token its last chunk samples, from a
        # mid-model hidden state, and nobody reads that token: no grammar
        # need follow it.
        params.max_tokens = 1
        params.min_tokens = 0
        params.logprobs = None
        params.prompt_logprobs = None
        params.structured_outputs = None
    return params


def _patch_pipeline_fanout(client_cls) -> None:
    """Send a pipelined request's helper copies, then the request itself."""
    from vllm.v1.engine import EngineCoreRequestType

    from vllm_torchtpu.distributed.dp_pipeline.transport import unlink_rings

    original_add = client_cls.add_request_async
    if getattr(original_add, _PIPELINE_MARKER, False):
        return
    original_abort = client_cls.abort_requests_async
    original_outputs = client_cls.process_engine_outputs
    original_shutdown = client_cls.shutdown

    async def add_request_async(self, request):
        # One add at a time: a request routed while a fan-out is sending would
        # reach a helper ahead of the helper's copy.
        lock = self.__dict__.setdefault("_tpu_dp_pipeline_lock", asyncio.Lock())
        async with lock:
            await add_request_locked(self, request)

    async def add_request_locked(self, request):
        planned = plan_pipeline(self, request)
        if planned is not None:
            plan, helpers = planned
            guests = []
            # Every copy is counted in flight before the first send yields, so
            # no request routed meanwhile takes a helper.
            for stage, engine in enumerate(helpers):
                guest = copy.copy(request)
                guest.request_id = f"{request.request_id}-dp-pipeline-{stage}"
                # A helper computes only the pipelined tokens.
                guest.prompt_token_ids = request.prompt_token_ids[: plan.tokens]
                guest.sampling_params = _with_plan(
                    request.sampling_params, plan, helper=True
                )
                guest.current_wave = self.current_wave
                guest.client_index = self.client_index
                self.reqs_in_flight[guest.request_id] = engine
                self.engine_inflight[engine] += 1
                guests.append((engine, guest))
            self.__dict__.setdefault("_tpu_dp_pipeline_guests", {})[
                request.request_id
            ] = [guest.request_id for _, guest in guests]
            request.sampling_params = _with_plan(
                request.sampling_params, plan, helper=False
            )
            for engine, guest in guests:
                await self._send_input(EngineCoreRequestType.ADD, guest, engine)
            logger.debug(
                "dp-pipeline %d: request %s on engines %s",
                plan.pipeline_id,
                request.request_id,
                plan.ranks,
            )
        # The request is already routed (plan_pipeline): vLLM's add keeps it
        # on the engine recorded for it.
        await original_add(self, request)

    async def abort_requests_async(self, request_ids):
        guests = self.__dict__.get("_tpu_dp_pipeline_guests", {})
        deferred = self.__dict__.setdefault("_tpu_dp_pipeline_aborts", set())
        # An engine stopped midway through a pipelined prompt would leave the
        # stages after it waiting for its hand-offs: the abort reaches the
        # owner with its first token, once every stage has run the prompt.
        deferred.update(r for r in request_ids if r in guests)
        await original_abort(self, [r for r in request_ids if r not in guests])

    async def process_engine_outputs(self, outputs):
        freed = {self.reqs_in_flight.get(r) for r in outputs.finished_requests or ()}
        await original_outputs(self, outputs)
        idle_since = self.__dict__.setdefault("_tpu_dp_pipeline_idle_since", {})
        now = time.monotonic()
        for engine in freed:
            if engine is not None and self.engine_inflight[engine] == 0:
                idle_since[engine] = now
        guests = self.__dict__.get("_tpu_dp_pipeline_guests")
        if not guests:
            return
        deferred = self.__dict__.get("_tpu_dp_pipeline_aborts", set())
        # A pipelined request's first output ends its prompt.
        ended = [o.request_id for o in outputs.outputs if o.request_id in guests]
        ended += [r for r in outputs.finished_requests or () if r in guests]
        for req_id in ended:
            guests.pop(req_id, None)
        due = [r for r in ended if r in deferred]
        deferred.difference_update(ended)
        if due:
            await original_abort(self, due)

    def shutdown(self, timeout=None):
        original_shutdown(self, timeout)
        # The engines, which map the rings, have stopped.
        unlink_rings(
            _server_id(),
            len(self.core_engines),
            self.vllm_config.parallel_config.tensor_parallel_size,
        )

    setattr(add_request_async, _PIPELINE_MARKER, True)
    client_cls.add_request_async = add_request_async
    client_cls.abort_requests_async = abort_requests_async
    client_cls.process_engine_outputs = staticmethod(process_engine_outputs)
    client_cls.shutdown = shutdown
    logger.info("Applied TPU patch: pipelined prefill fan-out over idle DP engines.")
