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
"""Helper utilities for request ID and KV cache profiler tracing."""

import contextlib
import functools
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

SPAN_ATTR_TIME_IN_KV_TRANSFER = "gen_ai.latency.time_in_kv_transfer"
SPAN_ATTR_COMPILE_TIME = "vllm.compile.time_s"
SPAN_ATTR_BLOCKED_BY_PREV_COMPILE = "vllm.compile.blocked_by_prev_compile"

_HEADER_KV_XFER_S = "_x_vllm_kv_xfer_s"
_HEADER_PREV_STEP_COMPILED = "_x_vllm_prev_step_compiled"
_HEADER_COMPILE_S = "_x_vllm_compile_s"

_xla_cache_checked_at: Any = None


def trim_request_id_suffix(request_id: str) -> str:
    parts = request_id.split("-")
    if len(parts) >= 6 and parts[0] == "cmpl":
        request_id = "-".join(parts[:6])
    return request_id


def extract_request_ids_for_tracing(
    req_ids: list[Any], start_index: int = 0, num_reqs: int = -1
) -> dict[str, str]:
    req_id_kwargs = {}
    try:
        active_ids = (
            req_ids[start_index : start_index + num_reqs] if num_reqs != -1 else req_ids
        )
        trimmed_req_ids = [
            trim_request_id_suffix(str(rid)) for rid in active_ids if rid is not None
        ]
        for i, rid in enumerate(trimmed_req_ids):
            req_id_kwargs[f"request_id{i + 1}"] = rid
    except Exception as e:
        logger.warning("Failed to extract request IDs for tracing: %s", e)

    return req_id_kwargs


def extract_kv_lens_for_tracing(
    num_computed_tokens: Any, start_index: int = 0, num_reqs: int = -1
) -> dict[str, int]:
    end_index = start_index + num_reqs if num_reqs != -1 else None
    active_lens = [int(kv_len) for kv_len in num_computed_tokens[start_index:end_index]]
    if not active_lens:
        return {}

    return {
        **{f"kv_len{i + 1}": kv_len for i, kv_len in enumerate(active_lens)},
        "min_kv_len": min(active_lens),
        "max_kv_len": max(active_lens),
        "avg_kv_len": sum(active_lens) // len(active_lens),
    }


def _poll_xla_cache_compile_secs() -> float:
    global _xla_cache_checked_at
    try:
        import torch

        get_stats = getattr(getattr(torch, "tpu", None), "_get_cache_stats", None)
        if get_stats is None:
            return 0.0
        entries = get_stats().per_entry_stats
    except Exception:
        return 0.0

    total_secs = 0.0
    newest = _xla_cache_checked_at
    for entry in entries:
        last_read = getattr(entry, "last_read", None)
        if last_read is None:
            continue
        if newest is None or last_read > newest:
            newest = last_read
        if getattr(entry, "read_count", 0) != 1:
            continue
        if _xla_cache_checked_at is not None and last_read <= _xla_cache_checked_at:
            continue
        duration = getattr(entry, "compilation_duration", None)
        total_seconds = getattr(duration, "total_seconds", None)
        if total_seconds is not None:
            total_secs += max(0.0, float(total_seconds()))
    _xla_cache_checked_at = newest
    return total_secs


def _patch_vllm_engine_core_tracing_attribution() -> None:
    """Propagate Scheduler KV-transfer and XLA compile attribution to ``llm_request`` spans.

    Adds three attributes to the ``llm_request`` span in ``OutputProcessor.do_tracing``:
      * ``gen_ai.latency.time_in_kv_transfer``: time spent in
        ``RequestStatus.WAITING_FOR_REMOTE_KVS`` waiting for async KV pull.
      * ``vllm.compile.time_s``: XLA compilation time incurred during steps in
        which this request was scheduled.
      * ``vllm.compile.blocked_by_prev_compile``: whether the immediately
        preceding active engine step triggered XLA compilation while this
        request waited in the ZMQ ``input_queue``.
    """
    import vllm.v1.engine.output_processor as output_processor_mod
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.output_processor import OutputProcessor
    from vllm.v1.request import RequestStatus

    if not getattr(Scheduler, "_tpu_engine_attribution_patched", False):
        orig_add_request = Scheduler.add_request
        orig_schedule = Scheduler.schedule
        orig_update_waiting_for_remote_kv = Scheduler._update_waiting_for_remote_kv
        orig_update_from_output = Scheduler.update_from_output

        @functools.wraps(orig_add_request)
        def _patched_add_request(
            self: Scheduler, request: Any, *args: Any, **kwargs: Any
        ) -> Any:
            if getattr(self, "_tpu_prev_step_compiled", False):
                request._tpu_blocked_by_prev_compile = True
            return orig_add_request(self, request, *args, **kwargs)

        @functools.wraps(orig_update_waiting_for_remote_kv)
        def _patched_update_waiting_for_remote_kv(
            self: Scheduler, request: Any
        ) -> None:
            start_mono = getattr(request, "_tpu_kv_xfer_start_mono", None)
            if start_mono is not None:
                elapsed = max(0.0, time.monotonic() - start_mono)
                request._tpu_kv_xfer_s = (
                    getattr(request, "_tpu_kv_xfer_s", 0.0) + elapsed
                )
                request._tpu_kv_xfer_start_mono = None
            orig_update_waiting_for_remote_kv(self, request)

        @functools.wraps(orig_schedule)
        def _patched_schedule(
            self: Scheduler, *args: Any, **kwargs: Any
        ) -> Any:
            self._tpu_prev_step_compiled = False  # type: ignore[attr-defined]
            sched_start_mono = time.monotonic()
            _poll_xla_cache_compile_secs()
            scheduler_output = orig_schedule(self, *args, **kwargs)
            for req in self.skipped_waiting:
                if (
                    req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
                    and getattr(req, "_tpu_kv_xfer_start_mono", None) is None
                ):
                    req._tpu_kv_xfer_start_mono = sched_start_mono
            return scheduler_output

        @functools.wraps(orig_update_from_output)
        def _patched_update_from_output(
            self: Scheduler,
            scheduler_output: Any,
            model_runner_output: Any,
        ) -> Any:
            step_compile_s = _poll_xla_cache_compile_secs()
            scheduled_reqs: dict[str, Any] = {}
            for req_id in scheduler_output.num_scheduled_tokens:
                req = self.requests.get(req_id)
                if req is None:
                    continue
                scheduled_reqs[req_id] = req
                if step_compile_s > 0.0:
                    req._tpu_compile_s = (
                        getattr(req, "_tpu_compile_s", 0.0) + step_compile_s
                    )
            engine_core_outputs = orig_update_from_output(
                self, scheduler_output, model_runner_output
            )
            self._tpu_prev_step_compiled = (  # type: ignore[attr-defined]
                step_compile_s > 0.0 and self.has_requests()
            )
            if isinstance(engine_core_outputs, dict):
                for client_outputs in engine_core_outputs.values():
                    for out in getattr(client_outputs, "outputs", ()):
                        if out.finish_reason is None or out.trace_headers is None:
                            continue
                        req = scheduled_reqs.get(out.request_id)
                        if req is None:
                            continue
                        headers = dict(out.trace_headers)
                        headers[_HEADER_KV_XFER_S] = (
                            f"{getattr(req, '_tpu_kv_xfer_s', 0.0):.6f}"
                        )
                        headers[_HEADER_PREV_STEP_COMPILED] = (
                            "1"
                            if getattr(req, "_tpu_blocked_by_prev_compile", False)
                            else "0"
                        )
                        headers[_HEADER_COMPILE_S] = (
                            f"{getattr(req, '_tpu_compile_s', 0.0):.6f}"
                        )
                        out.trace_headers = headers
            return engine_core_outputs

        Scheduler.add_request = _patched_add_request  # type: ignore[method-assign]
        Scheduler._update_waiting_for_remote_kv = (  # type: ignore[method-assign]
            _patched_update_waiting_for_remote_kv
        )
        Scheduler.schedule = _patched_schedule  # type: ignore[method-assign]
        Scheduler.update_from_output = (  # type: ignore[method-assign]
            _patched_update_from_output
        )
        Scheduler._tpu_engine_attribution_patched = True  # type: ignore[attr-defined]

    if not getattr(OutputProcessor, "_tpu_engine_attribution_patched", False):
        orig_do_tracing = OutputProcessor.do_tracing

        @functools.wraps(orig_do_tracing)
        def _patched_do_tracing(
            self: OutputProcessor,
            engine_core_output: Any,
            req_state: Any,
            iteration_stats: Any,
        ) -> None:
            extra_attrs: dict[str, Any] = {}
            headers = getattr(engine_core_output, "trace_headers", None)
            if isinstance(headers, dict):
                if _HEADER_KV_XFER_S in headers:
                    with contextlib.suppress(ValueError, TypeError):
                        extra_attrs[SPAN_ATTR_TIME_IN_KV_TRANSFER] = float(
                            headers[_HEADER_KV_XFER_S]
                        )
                if _HEADER_COMPILE_S in headers:
                    with contextlib.suppress(ValueError, TypeError):
                        extra_attrs[SPAN_ATTR_COMPILE_TIME] = float(
                            headers[_HEADER_COMPILE_S]
                        )
                if _HEADER_PREV_STEP_COMPILED in headers:
                    extra_attrs[SPAN_ATTR_BLOCKED_BY_PREV_COMPILE] = (
                        headers[_HEADER_PREV_STEP_COMPILED] == "1"
                    )

            orig_instrument_manual = output_processor_mod.instrument_manual

            def _wrapped_instrument_manual(*args: Any, **kwargs: Any) -> Any:
                attrs = kwargs.get("attributes")
                if isinstance(attrs, dict) and extra_attrs:
                    attrs.update(extra_attrs)
                return orig_instrument_manual(*args, **kwargs)

            output_processor_mod.instrument_manual = _wrapped_instrument_manual
            try:
                orig_do_tracing(self, engine_core_output, req_state, iteration_stats)
            finally:
                output_processor_mod.instrument_manual = orig_instrument_manual

        OutputProcessor.do_tracing = _patched_do_tracing  # type: ignore[method-assign]
        OutputProcessor._tpu_engine_attribution_patched = True  # type: ignore[attr-defined]
