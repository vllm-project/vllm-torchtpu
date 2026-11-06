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

import functools
import logging
from typing import Any

logger = logging.getLogger(__name__)

SPAN_ATTR_TIME_BEFORE_QUEUE = "gen_ai.latency.time_before_queue"
SPAN_ATTR_CACHED_TOKENS = "gen_ai.usage.cached_tokens"


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


def _patch_vllm_request_tracing_phase_gaps() -> None:
    """Enrich vLLM's ``llm_request`` OpenTelemetry span with pre-queue and cache attributes.

    vLLM v1 starts the ``llm_request`` span at ``metrics.arrival_time``, but only
    records ``metrics.queued_ts`` when ``EngineCore`` drains its ZMQ input queue
    between engine steps. Any time a request waits in the ZMQ input queue while
    ``EngineCore`` is busy falls outside ``gen_ai.latency.time_in_queue`` and
    ``gen_ai.latency.time_in_model_inference``.

    Adds two attributes to ``llm_request`` spans in ``OutputProcessor.do_tracing``:
      * ``gen_ai.latency.time_before_queue``: ``first_token_latency - (first_token_ts - queued_ts)``,
        using pure wall-clock and pure ``EngineCore`` monotonic intervals without
        comparing ``time.monotonic()`` across processes.
      * ``gen_ai.usage.cached_tokens``: prefix-cached prompt tokens (local or
        remote P-worker hit).
    """
    import vllm.v1.engine.output_processor as output_processor_mod
    from vllm.v1.engine.output_processor import OutputProcessor

    if getattr(OutputProcessor, "_tpu_phase_gaps_patched", False):
        return

    orig_do_tracing = OutputProcessor.do_tracing

    @functools.wraps(orig_do_tracing)
    def _patched_do_tracing(
        self: OutputProcessor,
        engine_core_output: Any,
        req_state: Any,
        iteration_stats: Any,
    ) -> None:
        metrics = req_state.stats
        extra_attrs: dict[str, Any] = {}
        if (
            metrics is not None
            and metrics.queued_ts > 0
            and metrics.first_token_ts > 0
            and metrics.first_token_latency > 0
        ):
            extra_attrs[SPAN_ATTR_TIME_BEFORE_QUEUE] = max(
                0.0,
                metrics.first_token_latency - (metrics.first_token_ts - metrics.queued_ts),
            )
        extra_attrs[SPAN_ATTR_CACHED_TOKENS] = int(
            getattr(req_state, "num_cached_tokens", 0) or 0
        )

        orig_instrument_manual = output_processor_mod.instrument_manual

        def _wrapped_instrument_manual(*args: Any, **kwargs: Any) -> Any:
            attrs = kwargs.get("attributes")
            if isinstance(attrs, dict):
                attrs.update(extra_attrs)
            return orig_instrument_manual(*args, **kwargs)

        output_processor_mod.instrument_manual = _wrapped_instrument_manual
        try:
            orig_do_tracing(self, engine_core_output, req_state, iteration_stats)
        finally:
            output_processor_mod.instrument_manual = orig_instrument_manual

    OutputProcessor.do_tracing = _patched_do_tracing  # type: ignore[method-assign]
    OutputProcessor._tpu_phase_gaps_patched = True  # type: ignore[attr-defined]
