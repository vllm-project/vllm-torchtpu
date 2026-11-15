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
"""OpenTelemetry spans emitted from EngineCore and TPU workers.

vLLM's ``llm_request`` span is written by the API server when a request
finishes, from the engine's QUEUED / SCHEDULED / token timestamps. Two delays
are invisible in it:

* ``kv_fetch``: on a decode engine of a P/D deployment, vLLM records SCHEDULED
  only after the remote KV pull finishes, so ``time_in_queue`` contains the
  pull. EngineCore emits this span when the async load ends, as a sibling of
  ``llm_request`` under the same trace parent, with the time spent in each
  stage of the load. Consumers subtract ``in_progress_s`` from
  ``time_in_queue`` to separate queueing from the transfer.
* ``xla_compile``: a wall-clock window in which a worker compiled an XLA
  program while serving. Compilation blocks the engine loop, so every request
  on the same pod whose lifetime overlaps the window was delayed by it,
  whether it was queued, prefilling or decoding. The span is a trace root;
  consumers join it to requests on pod and time. It records the worker and
  data-parallel rank, but vLLM's ``llm_request`` span records neither, so on a
  pod running several data-parallel engines a join on pod and time also
  charges one engine's compile to requests on the others. That is right when
  the engines step together (MoE data parallelism, where an idle rank runs a
  dummy batch to match the busy ranks' collectives) and an overcount when they
  run independently.

Both are emitted only when ``--otlp-traces-endpoint`` is set, and cost nothing
otherwise.
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
XLA_COMPILE_SPAN_NAME = "xla_compile"

# Seconds `request` spent in each stage of an async KV load. The suffix is the
# vLLM stage name (`vllm.v1.metrics.stats.KV_FETCH_STAGES`).
KV_FETCH_STAGE_ATTR_PREFIX = "vllm.kv_fetch."
KV_FETCH_STAGE_ATTR_SUFFIX = "_s"
KV_FETCH_OUTCOME_ATTR = "vllm.kv_fetch.outcome"
KV_FETCH_LOAD_FAILED_ATTR = "vllm.kv_fetch.load_failed"
KV_FETCH_NUM_TOKENS_ATTR = "vllm.kv_fetch.num_tokens"
REQUEST_ID_ATTR = "vllm.request_id"

XLA_COMPILE_SECONDS_ATTR = "vllm.xla_compile.compile_s"
XLA_COMPILE_NUM_GRAPHS_ATTR = "vllm.xla_compile.num_graphs"
WORKER_RANK_ATTR = "vllm.worker.rank"
DP_RANK_ATTR = "vllm.dp_rank"


def otlp_traces_endpoint(vllm_config: "VllmConfig") -> str | None:
    """Returns the configured OTLP traces endpoint, or None when tracing is off."""
    observability = getattr(vllm_config, "observability_config", None)
    return getattr(observability, "otlp_traces_endpoint", None) or None


def ensure_tracer_provider(endpoint: str, process_kind: str) -> bool:
    """Makes sure this process exports spans, initializing a tracer if needed.

    EngineCore and multiprocessing workers already initialize one from the
    endpoint the API server exported into the environment. Processes that
    vLLM does not initialize (Ray workers, or engines started before the API
    server set the environment variable) get one here.

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


def _cache_misses(device_module: Any) -> int | None:
    get_misses = getattr(device_module, "_get_cache_misses", None)
    if get_misses is None:
        return None
    try:
        return int(get_misses())
    except Exception:
        return None


def _epoch_seconds(value: Any) -> float | None:
    timestamp = getattr(value, "timestamp", None)
    if timestamp is None:
        return None
    try:
        return float(timestamp())
    except (OverflowError, OSError, ValueError):
        return None


def _compile_seconds(entry: Any) -> float:
    duration = getattr(entry, "compilation_duration", None)
    total_seconds = getattr(duration, "total_seconds", None)
    if total_seconds is None:
        return 0.0
    return max(0.0, float(total_seconds()))


class XlaCompileSpans:
    """Emits an `xla_compile` span for each poll interval that compiled.

    TorchTPU keeps, per compiled program, how long it took to compile (zero
    until the compile finishes, and zero for good when the program was loaded
    from a persistent cache) and when it was last read; a fresh program's
    first read is the end of its compile. A poll after each worker step
    compares the summed compile time with the previous poll. The number of
    cache misses, which is cheap to read, gates the full scan.

    TorchTPU counts a miss and inserts its entry when the graph is looked up
    on its materialization thread, but compiles on a pool and only records the
    duration when the compile finishes; the worker thread does not wait for
    either unless it reads a result. A poll can therefore see a miss whose
    compile is still running. Such misses are carried: later polls rescan,
    backing off up to `_MAX_RESCAN_INTERVAL_S` apart, until as many new
    compiled programs have appeared or `_PENDING_TIMEOUT_S` passes with no new
    miss (the misses were persistent-cache loads, which never report a
    duration).

    The span ends at the newest fresh program's first read (or the poll time
    when every fresh program was already read again) and starts the compiled
    seconds earlier, clamped to the last poll before the misses appeared. It
    is an estimate when several programs compiled in one interval with gaps
    between them.
    """

    _MIN_RESCAN_INTERVAL_S = 0.05
    _MAX_RESCAN_INTERVAL_S = 1.0
    _PENDING_TIMEOUT_S = 600.0

    def __init__(self, device_module: Any, attributes: dict[str, Any]) -> None:
        self._device = device_module
        self._attributes = attributes
        self._misses: int | None = None
        self._compile_s = 0.0
        self._num_compiled = 0
        self._polled_at = time.time()
        # Misses not yet matched by a newly compiled program.
        self._unresolved = 0
        # Earliest time an unresolved miss's compile can have started.
        self._window_start = self._polled_at
        self._last_miss_at = self._polled_at
        # End of the latest span emitted.
        self._reported_until = 0.0
        self._next_scan_at = self._polled_at
        self._rescan_interval_s = self._MIN_RESCAN_INTERVAL_S
        self._warned = False
        self.reset_baseline()

    def reset_baseline(self) -> None:
        """Treats everything compiled so far as already reported."""
        self._polled_at = time.time()
        self._misses = _cache_misses(self._device)
        self._unresolved = 0
        entries = self._entries()
        if entries is None:
            return
        self._compile_s, self._num_compiled, _ = self._summarize(entries, None)

    def _entries(self) -> list[Any] | None:
        get_stats = getattr(self._device, "_get_cache_stats", None)
        if get_stats is None:
            return None
        return list(get_stats().per_entry_stats)

    @staticmethod
    def _summarize(
        entries: list[Any], since: float | None
    ) -> tuple[float, int, float | None]:
        total_s, num_compiled, newest_end = 0.0, 0, None
        for entry in entries:
            secs = _compile_seconds(entry)
            if secs <= 0.0:
                continue
            total_s += secs
            num_compiled += 1
            if since is None or getattr(entry, "read_count", 0) != 1:
                continue
            end = _epoch_seconds(getattr(entry, "last_read", None))
            if end is None or end <= since:
                continue
            if newest_end is None or end > newest_end:
                newest_end = end
        return total_s, num_compiled, newest_end

    def _schedule_rescan(self, now: float) -> None:
        """Sets when to look again for compiles that were still running."""
        if self._unresolved <= 0:
            return
        if now - self._last_miss_at >= self._PENDING_TIMEOUT_S:
            # Persistent-cache loads never report a compile duration.
            self._unresolved = 0
            return
        self._next_scan_at = now + self._rescan_interval_s
        self._rescan_interval_s = min(
            2 * self._rescan_interval_s, self._MAX_RESCAN_INTERVAL_S
        )

    def poll(self) -> None:
        try:
            self._poll()
        except Exception:
            if not self._warned:
                self._warned = True
                logger.warning("Failed to record an xla_compile span", exc_info=True)

    def _poll(self) -> None:
        now = time.time()
        prev_polled_at = self._polled_at
        self._polled_at = now
        misses = _cache_misses(self._device)
        if misses is not None and misses == self._misses:
            if self._unresolved <= 0 or now < self._next_scan_at:
                return
        else:
            if self._unresolved <= 0:
                self._window_start = prev_polled_at
            if misses is not None and self._misses is not None:
                self._unresolved += max(0, misses - self._misses)
                self._last_miss_at = now
                self._rescan_interval_s = self._MIN_RESCAN_INTERVAL_S
            self._misses = misses
        entries = self._entries()
        if entries is None:
            return
        window_start = self._window_start
        # A program reported by an earlier span can still be unread; its first
        # read must not end a span for a compile carried past it.
        total_s, num_compiled, newest_end = self._summarize(
            entries, max(window_start, self._reported_until)
        )
        compiled_s = total_s - self._compile_s
        new_graphs = num_compiled - self._num_compiled
        self._compile_s, self._num_compiled = total_s, num_compiled
        if new_graphs > 0:
            self._unresolved = max(0, self._unresolved - new_graphs)
        self._schedule_rescan(now)
        # Zero while the compiles are still running or when the misses were
        # served from a persistent compile cache; negative when entries were
        # evicted, which resynchronizes above.
        if compiled_s <= 0.0 or new_graphs <= 0:
            return
        end_s = min(newest_end if newest_end is not None else now, now)
        start_s = max(window_start, end_s - compiled_s)
        self._reported_until = max(self._reported_until, end_s)
        attributes = dict(self._attributes)
        attributes[XLA_COMPILE_SECONDS_ATTR] = compiled_s
        attributes[XLA_COMPILE_NUM_GRAPHS_ATTR] = new_graphs
        from opentelemetry.context import Context  # pyrefly: ignore[missing-import]

        # An empty context makes this a trace root; with None, vLLM would
        # parent it to whatever trace the process environment carries.
        _emit_span(
            XLA_COMPILE_SPAN_NAME,
            int(start_s * 1e9),
            int(end_s * 1e9),
            attributes,
            Context(),
        )


def maybe_create_xla_compile_spans(
    vllm_config: "VllmConfig", rank: int
) -> XlaCompileSpans | None:
    """Returns a compile-span emitter for this worker, or None when not needed.

    Only tensor-parallel rank 0 emits: the other ranks of a TP group compile
    the same programs at the same time.

    Args:
      vllm_config: The worker's vLLM config.
      rank: The worker's rank, recorded on the span.
    """
    endpoint = otlp_traces_endpoint(vllm_config)
    if endpoint is None:
        return None
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() != 0:
            return None
    except Exception:
        # Model parallel groups are not set up (e.g. a single-rank worker);
        # the worker is then the only rank and emits.
        logger.debug("No TP rank available; emitting compile spans", exc_info=True)
    import torch

    device_module = getattr(torch, "tpu", None)
    if device_module is None or not hasattr(device_module, "_get_cache_stats"):
        return None
    if not ensure_tracer_provider(endpoint, "worker"):
        return None
    attributes: dict[str, Any] = {WORKER_RANK_ATTR: rank}
    dp_rank = getattr(vllm_config.parallel_config, "data_parallel_rank", None)
    if dp_rank is not None:
        attributes[DP_RANK_ATTR] = dp_rank
    logger.info("Emitting xla_compile spans to %s", endpoint)
    return XlaCompileSpans(device_module, attributes)
