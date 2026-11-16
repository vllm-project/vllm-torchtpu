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
"""OpenTelemetry ``kv_fetch`` spans emitted from EngineCore.

vLLM's ``llm_request`` span is written by the API server when a request
finishes, from the engine's QUEUED / SCHEDULED / token timestamps. On a decode
engine of a P/D deployment, vLLM records SCHEDULED only after the remote KV
pull finishes, so ``time_in_queue`` contains the pull. EngineCore emits a
``kv_fetch`` span from the start of the async load until the request is
scheduled or freed, as a sibling of ``llm_request`` under the same trace
parent. It reports the seconds spent in each load stage as
``vllm.kv_fetch.<stage>_s``: ``in_progress`` is the pull, and
``completed_waiting`` the wait for a scheduler pass after it.

The span is emitted only when ``--otlp-traces-endpoint`` is set and the
scheduler has a KV connector.
"""

import functools
import time
from typing import TYPE_CHECKING, Any

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

KV_FETCH_SPAN_NAME = "kv_fetch"


def _ensure_tracer_provider(endpoint: str) -> bool:
    """Makes sure this process exports spans, initializing a tracer if needed.

    vLLM initializes EngineCore's tracer only from the endpoint the API server
    exported into the environment. Engines launched before any API server ran
    (data parallel with several API servers, headless engines) have none.

    Returns:
      Whether spans created in this process will be exported.
    """
    from vllm.tracing import init_tracer, is_tracing_available

    if not is_tracing_available():
        return False
    from opentelemetry import trace  # pyrefly: ignore[missing-import]
    from opentelemetry.sdk.trace import (  # pyrefly: ignore[missing-import]
        TracerProvider,
    )

    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        init_tracer(
            "vllm_torchtpu.engine_core",
            endpoint,
            {"vllm.process_kind": "engine_core"},
        )
    return isinstance(trace.get_tracer_provider(), TracerProvider)


def _emit_span(
    start_ns: int, end_ns: int, attributes: dict[str, float], context: Any
) -> None:
    from vllm.tracing import SpanKind, instrument_manual

    instrument_manual(
        span_name=KV_FETCH_SPAN_NAME,
        start_time=start_ns,
        end_time=end_ns,
        attributes=attributes,
        context=context,
        kind=SpanKind.INTERNAL,
    )


class _KvFetch:
    """Stage timeline of one request's async KV load."""

    __slots__ = ("start_ns", "stage", "stage_start_ns", "stage_s")

    def __init__(self, stage: str, now_ns: int) -> None:
        self.start_ns = now_ns
        self.stage = stage
        self.stage_start_ns = now_ns
        self.stage_s: dict[str, float] = {}


class KvFetchSpans:
    """Turns a scheduler's KV fetch stage transitions into `kv_fetch` spans.

    Wraps `Scheduler._set_kv_fetch_stage`, which vLLM calls on every stage
    change of an async KV load (`vllm.v1.metrics.stats.KV_FETCH_STAGES`), and
    with `None` once the request is scheduled or freed. A load is tracked from
    the first `in_progress` mark; time before it is queueing, which
    `time_in_queue` already reports. A request whose load failed may start
    another before it is scheduled; its stages accumulate into one span.
    """

    def __init__(self, original_set_stage: Any) -> None:
        self.original_set_stage = original_set_stage
        self.fetches: dict[str, _KvFetch] = {}

    def set_kv_fetch_stage(self, request: "Request", stage: str | None) -> None:
        from vllm.v1.metrics.stats import KV_FETCH_IN_PROGRESS

        self.original_set_stage(request, stage)
        req_id = request.request_id
        fetch = self.fetches.get(req_id)
        if fetch is None:
            if stage == KV_FETCH_IN_PROGRESS:
                self.fetches[req_id] = _KvFetch(KV_FETCH_IN_PROGRESS, time.time_ns())
            return
        now_ns = time.time_ns()
        elapsed_s = max(0, now_ns - fetch.stage_start_ns) / 1e9
        fetch.stage_s[fetch.stage] = fetch.stage_s.get(fetch.stage, 0.0) + elapsed_s
        if stage is not None:
            fetch.stage = stage
            fetch.stage_start_ns = now_ns
            return
        del self.fetches[req_id]
        headers = request.trace_headers
        # Without a propagated parent the span would land in an unrelated
        # trace (vLLM falls back to the process environment's context).
        if not headers or "traceparent" not in headers:
            return
        from vllm.tracing import extract_trace_context

        _emit_span(
            fetch.start_ns,
            now_ns,
            {f"vllm.kv_fetch.{s}_s": secs for s, secs in fetch.stage_s.items()},
            extract_trace_context(headers),
        )


def install_kv_fetch_spans(
    scheduler: Any, vllm_config: "VllmConfig"
) -> KvFetchSpans | None:
    """Wraps `scheduler._set_kv_fetch_stage` when there is something to trace.

    Returns:
      The installed tracker, or None when tracing is off, the scheduler has no
      KV connector (so no async loads), or no tracer could be set up.
    """
    endpoint = vllm_config.observability_config.otlp_traces_endpoint
    if not endpoint or scheduler.connector is None:
        return None
    if not _ensure_tracer_provider(endpoint):
        return None
    spans = KvFetchSpans(scheduler._set_kv_fetch_stage)
    scheduler._set_kv_fetch_stage = spans.set_kv_fetch_stage
    logger.info("Emitting kv_fetch spans to %s", endpoint)
    return spans


def _patch_vllm_kv_fetch_spans() -> None:
    """Installs `KvFetchSpans` on schedulers built while tracing is enabled."""
    from vllm.v1.core.sched.scheduler import Scheduler

    if Scheduler.__dict__.get("_tpu_kv_fetch_spans_patch", False):
        return
    original_init = Scheduler.__init__

    @functools.wraps(original_init)
    def init(self: Scheduler, vllm_config: "VllmConfig", *args, **kwargs) -> None:
        original_init(self, vllm_config, *args, **kwargs)
        install_kv_fetch_spans(self, vllm_config)

    Scheduler.__init__ = init  # type: ignore[method-assign]
    Scheduler._tpu_kv_fetch_spans_patch = True  # type: ignore[attr-defined]
