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
``kv_fetch`` span when the async load ends, as a sibling of ``llm_request``
under the same trace parent, with the time spent in each stage of the load.
Consumers subtract ``in_progress_s`` from ``time_in_queue`` to separate
queueing from the transfer.

The span is emitted only when ``--otlp-traces-endpoint`` is set and the
scheduler has a KV connector, and costs nothing otherwise.
"""

import functools
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

logger = logging.getLogger("vllm." + __name__)

KV_FETCH_SPAN_NAME = "kv_fetch"

# Seconds `request` spent in each stage of an async KV load. The suffix is the
# vLLM stage name (`vllm.v1.metrics.stats.KV_FETCH_STAGES`).
KV_FETCH_STAGE_ATTR_PREFIX = "vllm.kv_fetch."
KV_FETCH_STAGE_ATTR_SUFFIX = "_s"
KV_FETCH_OUTCOME_ATTR = "vllm.kv_fetch.outcome"
KV_FETCH_LOAD_FAILED_ATTR = "vllm.kv_fetch.load_failed"
KV_FETCH_NUM_TOKENS_ATTR = "vllm.kv_fetch.num_tokens"
REQUEST_ID_ATTR = "vllm.request_id"


def otlp_traces_endpoint(vllm_config: "VllmConfig") -> str | None:
    """Returns the configured OTLP traces endpoint, or None when tracing is off."""
    observability = getattr(vllm_config, "observability_config", None)
    return getattr(observability, "otlp_traces_endpoint", None) or None


def ensure_tracer_provider(endpoint: str, process_kind: str) -> bool:
    """Makes sure this process exports spans, initializing a tracer if needed.

    EngineCore already initializes one from the endpoint the API server
    exported into the environment. An engine started before the API server
    set the environment variable gets one here.

    Args:
      endpoint: OTLP traces endpoint from the vLLM config.
      process_kind: Label recorded as the `vllm.process_kind` resource
        attribute when this call creates the tracer.

    Returns:
      Whether spans created in this process will be exported.
    """
    try:
        # Optional dependency: vLLM installs it with its tracing extra.
        from opentelemetry import trace  # pyrefly: ignore[missing-import]
        from opentelemetry.sdk.trace import (  # pyrefly: ignore[missing-import]
            TracerProvider,
        )
    except ImportError:
        return False
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        return True
    from vllm.tracing import init_tracer

    init_tracer(
        f"vllm_torchtpu.{process_kind}",
        endpoint,
        {"vllm.process_kind": process_kind},
    )
    return isinstance(trace.get_tracer_provider(), TracerProvider)


def _emit_span(
    name: str,
    start_ns: int,
    end_ns: int,
    attributes: dict[str, Any],
    context: Any,
) -> None:
    from vllm.tracing import SpanKind, instrument_manual

    instrument_manual(
        span_name=name,
        start_time=start_ns,
        end_time=end_ns,
        attributes=attributes,
        context=context,
        kind=SpanKind.INTERNAL,
    )


class _KvFetch:
    """Stage timeline of one request's async KV load."""

    __slots__ = (
        "start_ns",
        "stage",
        "stage_start_ns",
        "stage_s",
        "num_tokens",
        "load_failed",
    )

    def __init__(self, stage: str, now_ns: int) -> None:
        self.start_ns = now_ns
        self.stage = stage
        self.stage_start_ns = now_ns
        self.stage_s: dict[str, float] = {}
        self.num_tokens: int | None = None
        self.load_failed = False

    def close_stage(self, now_ns: int) -> None:
        elapsed_s = max(0, now_ns - self.stage_start_ns) / 1e9
        self.stage_s[self.stage] = self.stage_s.get(self.stage, 0.0) + elapsed_s


class KvFetchSpans:
    """Turns a scheduler's KV fetch stage transitions into `kv_fetch` spans.

    Installed per scheduler instance over `Scheduler._set_kv_fetch_stage`,
    which vLLM calls on every stage change of an async KV load: waiting to
    start, in progress, completed and waiting to be scheduled, and done
    (`None`, either scheduled or freed). The wrapper runs whether or not vLLM
    tracks stage counts for its own metrics.
    """

    def __init__(self, scheduler: Any) -> None:
        self.scheduler = scheduler
        self.original_set_stage = scheduler._set_kv_fetch_stage
        self.fetches: dict[str, _KvFetch] = {}
        self._warned = False

    def set_kv_fetch_stage(self, request: "Request", stage: str | None) -> None:
        self.original_set_stage(request, stage)
        try:
            self._on_stage(request, stage)
        except Exception:
            if not self._warned:
                self._warned = True
                logger.warning("Failed to record a kv_fetch span", exc_info=True)

    def _on_stage(self, request: "Request", stage: str | None) -> None:
        from vllm.v1.metrics.stats import (
            KV_FETCH_COMPLETED_WAITING,
            KV_FETCH_IN_PROGRESS,
        )

        req_id = request.request_id
        fetch = self.fetches.get(req_id)
        if fetch is not None and fetch.stage == stage:
            # The scheduler re-marks a request still waiting to start on each
            # pass that skips it; the stage started at the first mark.
            return
        now_ns = time.time_ns()
        if fetch is not None:
            fetch.close_stage(now_ns)
        if stage is None:
            if fetch is not None:
                del self.fetches[req_id]
                self._emit(request, fetch, now_ns)
            return
        if fetch is None:
            fetch = _KvFetch(stage, now_ns)
            self.fetches[req_id] = fetch
        else:
            fetch.stage = stage
            fetch.stage_start_ns = now_ns
        if stage == KV_FETCH_IN_PROGRESS:
            fetch.num_tokens = request.num_computed_tokens
        elif stage == KV_FETCH_COMPLETED_WAITING:
            failed = getattr(self.scheduler, "failed_recving_kv_req_ids", ())
            fetch.load_failed = req_id in failed

    def _emit(self, request: "Request", fetch: _KvFetch, end_ns: int) -> None:
        from vllm.v1.metrics.stats import KV_FETCH_IN_PROGRESS

        # A request that never started a load (only waited to start) spent the
        # time in the queue, which `time_in_queue` already reports.
        if KV_FETCH_IN_PROGRESS not in fetch.stage_s:
            return
        headers = request.trace_headers
        # Without a propagated parent the span would land in an unrelated
        # trace (vLLM falls back to the process environment's context).
        if not headers or "traceparent" not in headers:
            return
        from vllm.tracing import extract_trace_context

        attributes: dict[str, Any] = {
            f"{KV_FETCH_STAGE_ATTR_PREFIX}{stage}{KV_FETCH_STAGE_ATTR_SUFFIX}": secs
            for stage, secs in fetch.stage_s.items()
        }
        attributes[KV_FETCH_OUTCOME_ATTR] = (
            "finished" if request.is_finished() else "scheduled"
        )
        attributes[KV_FETCH_LOAD_FAILED_ATTR] = fetch.load_failed
        attributes[REQUEST_ID_ATTR] = request.request_id
        if fetch.num_tokens is not None:
            attributes[KV_FETCH_NUM_TOKENS_ATTR] = fetch.num_tokens
        _emit_span(
            KV_FETCH_SPAN_NAME,
            fetch.start_ns,
            end_ns,
            attributes,
            extract_trace_context(headers),
        )


def install_kv_fetch_spans(
    scheduler: Any, vllm_config: "VllmConfig"
) -> KvFetchSpans | None:
    """Wraps `scheduler._set_kv_fetch_stage` when there is something to trace.

    Args:
      scheduler: A constructed vLLM scheduler.
      vllm_config: The config the scheduler was built from.

    Returns:
      The installed tracker, or None when tracing is off, the scheduler has no
      KV connector (so no async loads), or no tracer could be set up.
    """
    endpoint = otlp_traces_endpoint(vllm_config)
    if endpoint is None or getattr(scheduler, "connector", None) is None:
        return None
    if not ensure_tracer_provider(endpoint, "engine_core"):
        return None
    spans = KvFetchSpans(scheduler)
    scheduler.__dict__["_tpu_kv_fetch_spans"] = spans
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
