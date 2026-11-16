# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
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
"""Flag-gated vLLM lifecycle hooks for multi-turn agentic serving telemetry."""

import contextlib
import contextvars
import functools
import json
import socket
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.telemetry.eviction_tracker import (
    BlockEvictionTracker,
    extract_hardware_span_attributes,
)
from vllm_torchtpu.telemetry.token_breakdown import (
    CATEGORIES,
    BoundedLRU,
    SessionTurnTracker,
    TokenBreakdown,
    compute_rolling_block_hashes,
)

logger = init_logger(__name__)

AGENT_TRACE_HEADERS: tuple[str, ...] = (
    "x-agent-session-id",
    "x-agent-parent-session-id",
    "x-agent-turn-id",
    "x-agent-turn-index",
)

SPAN_ATTRIBUTE_KEYS: tuple[str, ...] = (
    "gen_ai.session.id",
    "gen_ai.session.turn_id",
    "gen_ai.session.parent_id",
    "gen_ai.server.replica_id",
    *(f"gen_ai.usage.input_tokens.{cat}" for cat in CATEGORIES),
    "gen_ai.usage.mutation_offset_token_idx",
    "vllm.kv_cache.time_since_eviction_ms",
    "vllm.kv_cache.lru_horizon_ms",
    "vllm.kv_cache.session_evicted_at_ms",
    "vllm.kv_cache.dram_tokens",
    "vllm.kv_cache.dram_bytes",
    "vllm.kv_cache.dram_ms",
    "vllm.kv_cache.ssd_tokens",
    "vllm.kv_cache.ssd_bytes",
    "vllm.kv_cache.ssd_ms",
    "vllm.kv_cache.external_computed_tokens",
    "vllm.kv_cache.remote_tokens",
    "vllm.kv_cache.transfer_ms",
    "vllm.tpu.shape_padding_tokens",
)

_TRACING_CONSUMER_MODULES: tuple[str, ...] = (
    "vllm.tracing",
    "vllm.entrypoints.serve.engine.serving",
    "vllm.entrypoints.openai.engine.serving",
    "vllm.entrypoints.pooling.base.serving",
    "vllm.entrypoints.generate.base.serving",
    "vllm.entrypoints.generate.generative_scoring.serving",
)

_ACTIVE_TRACE_HEADERS: contextvars.ContextVar[dict[str, str] | None] = (
    contextvars.ContextVar("_ACTIVE_TRACE_HEADERS", default=None)
)

_INFLIGHT_REQUESTS_MAXSIZE = 16384
_DETECTED_VLLM_BLOCK_SIZE: int | None = None
_PATCHES_LOGGED: bool = False


@dataclass(slots=True)
class _InFlightRequestState:
    session_id: str | None = None
    breakdown: TokenBreakdown | None = None
    external_tokens: int = 0
    prefill_ts: float | None = None
    eviction_metrics: tuple[float | None, int | None] | None = None
    eviction_ts: float | None = None


def _extract_request_id(req_state: Any, engine_core_output: Any) -> str:
    return str(
        getattr(engine_core_output, "request_id", None)
        or getattr(req_state, "request_id", "")
        or ""
    )


def _extract_output_token_ids(req_state: Any, engine_core_output: Any) -> Sequence[int]:
    detok = getattr(req_state, "detokenizer", None)
    detok_ids = getattr(detok, "output_token_ids", None) if detok is not None else None
    if detok_ids:
        return detok_ids
    return getattr(engine_core_output, "new_token_ids", None) or ()


def _resolve_trace_headers(
    req_state: Any, engine_core_output: Any
) -> Mapping[str, str] | None:
    """Return trace headers from `engine_core_output` or `req_state` fallback."""
    return getattr(engine_core_output, "trace_headers", None) or getattr(
        req_state, "trace_headers", None
    )


def _resolve_num_cached_tokens(req_state: Any, engine_core_output: Any) -> int:
    """Extract `num_cached_tokens` from vLLM V1 `prefill_stats` or `RequestState`."""
    prefill_stats = getattr(engine_core_output, "prefill_stats", None)
    if prefill_stats is not None:
        return int(prefill_stats.num_cached_tokens)
    remote_cached = getattr(req_state, "remote_prefill_cached_tokens", None)
    if remote_cached is not None:
        return int(remote_cached)
    return int(getattr(req_state, "num_cached_tokens", 0) or 0)


def _resolve_num_external_cached_tokens(req_state: Any, engine_core_output: Any) -> int:
    """Extract external/DRAM cached tokens from `PrefillStats` or stashed state."""
    prefill_stats = getattr(engine_core_output, "prefill_stats", None)
    if prefill_stats is not None:
        return int(prefill_stats.num_external_cached_tokens)
    stashed = getattr(req_state, "_agentic_external_cached_tokens", None)
    return int(stashed or 0)


def extract_agent_headers(
    trace_headers: Mapping[str, str] | None,
) -> tuple[str | None, str | None, int | None]:
    """Extract `(session_id, parent_session_id, turn_id)` from trace headers."""
    if not trace_headers:
        return None, None, None
    lowered = {k.lower(): v for k, v in trace_headers.items() if v is not None}
    session_id = lowered.get("x-agent-session-id") or None
    parent_session_id = lowered.get("x-agent-parent-session-id") or None
    raw_turn = lowered.get("x-agent-turn-id")
    if raw_turn is None:
        raw_turn = lowered.get("x-agent-turn-index")
    turn_id: int | None = None
    if raw_turn is not None:
        try:
            turn_id = int(raw_turn)
        except (TypeError, ValueError):
            turn_id = None
    return session_id, parent_session_id, turn_id


def _extract_request_session_id(
    request: Any, fallback_headers: Mapping[str, str] | None = None
) -> str | None:
    hdrs = getattr(request, "trace_headers", None) or fallback_headers
    sid, _, _ = extract_agent_headers(hdrs)
    return sid or getattr(request, "session_id", None) or None


class AgenticTelemetryService:
    """Coordinates session tracking, eviction age, and OTel span attributes."""

    def __init__(
        self,
        *,
        block_size: int | None = None,
        replica_id: str | None = None,
        span_log_path: str | None = None,
        session_tracker: SessionTurnTracker | None = None,
        eviction_tracker: BlockEvictionTracker | None = None,
        inflight_maxsize: int = _INFLIGHT_REQUESTS_MAXSIZE,
    ) -> None:
        self.replica_id = replica_id or envs.VLLM_TPU_REPLICA_ID or socket.gethostname()
        self.span_log_path = (
            span_log_path
            if span_log_path is not None
            else envs.VLLM_TPU_AGENTIC_SPAN_LOG
        )
        env_bs = int(envs.VLLM_TPU_AGENTIC_BLOCK_SIZE or 0)
        detected_bs = _DETECTED_VLLM_BLOCK_SIZE or 0
        auto_bs = env_bs if env_bs > 0 else detected_bs
        self._explicit_block_size = (
            block_size is not None or session_tracker is not None or auto_bs > 0
        )
        self._block_size_checked = self._explicit_block_size
        resolved_block_size = (
            block_size if block_size is not None else (auto_bs if auto_bs > 0 else 16)
        )
        self.session_tracker = session_tracker or SessionTurnTracker(
            block_size=resolved_block_size
        )
        self.eviction_tracker = eviction_tracker or BlockEvictionTracker()
        self.category_totals: dict[str, int] = {cat: 0 for cat in CATEGORIES}
        self._inflight: BoundedLRU[str, _InFlightRequestState] = BoundedLRU(
            inflight_maxsize
        )

    def discard_request(self, request_id: str | None) -> None:
        """Drop in-flight bookkeeping for an aborted or rejected request."""
        if request_id:
            inflight = self._inflight.pop(request_id)
            if inflight is not None and inflight.session_id:
                self.session_tracker.discard_pending(inflight.session_id)

    def _maybe_sync_vllm_block_size(self) -> None:
        """Auto-detect `cache_config.block_size` once if not explicitly set."""
        if self._explicit_block_size or self._block_size_checked:
            return
        self._block_size_checked = True
        env_bs = int(envs.VLLM_TPU_AGENTIC_BLOCK_SIZE or 0)
        resolved = env_bs if env_bs > 0 else (_DETECTED_VLLM_BLOCK_SIZE or 0)
        if resolved > 0:
            self.session_tracker.block_size = resolved
            self._explicit_block_size = True
            return
        try:
            from vllm.config.vllm import get_current_vllm_config

            cfg = get_current_vllm_config()
            bs = getattr(getattr(cfg, "cache_config", None), "block_size", None)
            if isinstance(bs, int) and bs > 0:
                self.session_tracker.block_size = bs
                self._explicit_block_size = True
        except Exception as exc:
            logger.debug("Skipping vllm_config block_size lookup: %s", exc)

    def on_block_evicted(
        self,
        block_hash: Any,
        now: float | None = None,
        *,
        session_id: str | None = None,
    ) -> None:
        """Record an HBM block eviction event."""
        self.eviction_tracker.record_eviction(
            block_hash, now=now, session_id=session_id
        )

    def on_prefill_output(
        self, req_state: Any, engine_core_output: Any
    ) -> TokenBreakdown:
        """Classify prefill tokens on the first output step of a request."""
        req_id = _extract_request_id(req_state, engine_core_output)
        ext_cached = _resolve_num_external_cached_tokens(req_state, engine_core_output)
        inflight = self._inflight.get(req_id) if req_id else None
        if ext_cached > 0:
            if req_id:
                if inflight is None:
                    inflight = _InFlightRequestState()
                    self._inflight.put(req_id, inflight)
                inflight.external_tokens = ext_cached
            with contextlib.suppress(AttributeError):
                req_state._agentic_external_cached_tokens = ext_cached
        if inflight is not None and inflight.breakdown is not None:
            return inflight.breakdown

        self._maybe_sync_vllm_block_size()
        prefill_now = time.monotonic()

        session_id, parent_session_id, turn_id = extract_agent_headers(
            _resolve_trace_headers(req_state, engine_core_output)
        )
        prompt_token_ids = getattr(req_state, "prompt_token_ids", None) or ()
        num_cached = _resolve_num_cached_tokens(req_state, engine_core_output)

        breakdown = self.session_tracker.classify_prefill(
            prompt_token_ids,
            num_cached,
            session_id=session_id,
            turn_id=turn_id,
            parent_session_id=parent_session_id,
        )
        for cat, count in breakdown.as_category_dict().items():
            self.category_totals[cat] += count
        if req_id:
            if inflight is None:
                inflight = _InFlightRequestState()
            inflight.session_id = session_id
            inflight.breakdown = breakdown
            inflight.prefill_ts = prefill_now
            inflight.eviction_ts = self.eviction_tracker.session_evicted_ts(session_id)
            inflight.eviction_metrics = self._compute_eviction_snapshot(
                req_state,
                breakdown,
                session_id=session_id,
                eval_now=prefill_now,
                ext_cached=ext_cached,
            )
            self._inflight.put(req_id, inflight)
        return breakdown

    def _compute_eviction_snapshot(
        self,
        req_state: Any,
        breakdown: TokenBreakdown,
        *,
        session_id: str | None,
        eval_now: float | None,
        ext_cached: int = 0,
    ) -> tuple[float | None, int | None]:
        eviction_age_ms: float | None = None
        session_evicted_wall: int | None = None
        if breakdown.affinity_honored_eviction > 0 or ext_cached > 0:
            prompt_token_ids = getattr(req_state, "prompt_token_ids", None) or ()
            block_size = self.session_tracker.block_size
            t1_state = (
                self.session_tracker.tier1.peek(session_id) if session_id else None
            )
            verified_len = (
                self.session_tracker._verified_match_len.peek(session_id)
                if session_id
                else None
            )
            reusable_blocks = (
                (verified_len // block_size)
                if (t1_state is not None and verified_len is not None)
                else 0
            )
            all_hashes = compute_rolling_block_hashes(
                prompt_token_ids,
                block_size,
                prefix_hashes=t1_state.block_hashes if t1_state is not None else (),
                start_block=reusable_blocks,
            )
            hbm_hit = max(0, breakdown.cache_hit - max(0, int(ext_cached)))
            start_blk = hbm_hit // block_size
            missed_hashes = all_hashes[start_blk:]
            eviction_age_ms = self.eviction_tracker.time_since_eviction_ms(
                missed_hashes, now=eval_now, session_id=session_id
            )
            session_evicted_wall = self.eviction_tracker.session_evicted_at_ms(
                session_id
            )

        return eviction_age_ms, session_evicted_wall

    def on_request_finished(
        self, req_state: Any, engine_core_output: Any
    ) -> TokenBreakdown:
        """Finalize request state and update two-tier session history."""
        req_id = _extract_request_id(req_state, engine_core_output)
        inflight = self._inflight.pop(req_id) if req_id else None
        breakdown = inflight.breakdown if inflight is not None else None
        if breakdown is None:
            breakdown = self.on_prefill_output(req_state, engine_core_output)
            if req_id:
                self._inflight.pop(req_id)

        session_id, _, turn_id = extract_agent_headers(
            _resolve_trace_headers(req_state, engine_core_output)
        )
        prompt_token_ids = getattr(req_state, "prompt_token_ids", None) or ()
        output_token_ids = _extract_output_token_ids(req_state, engine_core_output)
        full_hashes, reused_blocks = self.session_tracker.record_turn_completion(
            session_id,
            prompt_token_ids,
            output_token_ids,
            turn_id=turn_id if turn_id is not None else 1,
        )
        if session_id and full_hashes:
            self.eviction_tracker.register_session_blocks(
                session_id,
                full_hashes[reused_blocks:],
            )
        return breakdown

    def build_span_attributes(
        self,
        req_state: Any,
        engine_core_output: Any,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Build agentic OTel span attributes for a completed request."""
        req_id = _extract_request_id(req_state, engine_core_output)
        prior_inflight = self._inflight.peek(req_id) if req_id else None
        ext_computed = prior_inflight.external_tokens if prior_inflight else 0
        if not ext_computed:
            ext_computed = _resolve_num_external_cached_tokens(
                req_state, engine_core_output
            )

        snap_evict = prior_inflight.eviction_metrics if prior_inflight else None
        prefill_evict_ts = prior_inflight.eviction_ts if prior_inflight else None
        prefill_ts = prior_inflight.prefill_ts if prior_inflight else None

        if prior_inflight is None and req_id:
            self.on_prefill_output(req_state, engine_core_output)
            populated = self._inflight.peek(req_id)
            if populated is not None:
                snap_evict = populated.eviction_metrics
                prefill_evict_ts = populated.eviction_ts
                prefill_ts = populated.prefill_ts

        breakdown = self.on_request_finished(req_state, engine_core_output)
        eval_now = now if now is not None else prefill_ts

        attrs = breakdown.to_span_attributes()
        attrs["gen_ai.server.replica_id"] = self.replica_id

        session_id, parent_session_id, turn_id = extract_agent_headers(
            _resolve_trace_headers(req_state, engine_core_output)
        )
        if session_id:
            attrs["gen_ai.session.id"] = session_id
        if turn_id is not None:
            attrs["gen_ai.session.turn_id"] = turn_id
        if parent_session_id and "gen_ai.session.parent_id" not in attrs:
            attrs["gen_ai.session.parent_id"] = parent_session_id

        if snap_evict is not None and now is None:
            eviction_age_ms, session_evicted_wall = snap_evict
        else:
            if prefill_evict_ts is None:
                prefill_evict_ts = self.eviction_tracker.session_evicted_ts(session_id)
            eviction_age_ms, session_evicted_wall = self._compute_eviction_snapshot(
                req_state,
                breakdown,
                session_id=session_id,
                eval_now=eval_now,
                ext_cached=ext_computed,
            )

        had_eviction = breakdown.affinity_honored_eviction > 0 or ext_computed > 0
        if (
            session_id
            and prefill_evict_ts is not None
            and (
                had_eviction
                or breakdown.cache_hit > 0
                or (turn_id is not None and turn_id > 1)
            )
        ):
            self.eviction_tracker.clear_session_eviction(
                session_id, up_to_ts=prefill_evict_ts
            )

        hw_attrs = extract_hardware_span_attributes(
            kv_transfer_params=getattr(engine_core_output, "kv_transfer_params", None),
            num_external_computed_tokens=ext_computed,
            time_since_eviction_ms=eviction_age_ms,
            lru_horizon_ms=None,
            session_evicted_at_ms=session_evicted_wall,
            tpu_shape_padding_tokens=None,
        )
        attrs.update(hw_attrs)
        if self.span_log_path:
            self._append_span_log(attrs, req_state, engine_core_output, breakdown)
        return attrs

    def _append_span_log(
        self,
        attrs: Mapping[str, Any],
        req_state: Any,
        engine_core_output: Any,
        breakdown: TokenBreakdown,
    ) -> None:
        log_record = dict(attrs)
        trace_hdrs = _resolve_trace_headers(req_state, engine_core_output) or {}
        lowered_hdrs = {k.lower(): v for k, v in trace_hdrs.items() if v is not None}
        if "traceparent" in lowered_hdrs:
            log_record.setdefault("traceparent", lowered_hdrs["traceparent"])
        prompt_token_ids = getattr(req_state, "prompt_token_ids", None) or ()
        output_token_ids = _extract_output_token_ids(req_state, engine_core_output)
        log_record.setdefault(
            "gen_ai.usage.prompt_tokens",
            len(prompt_token_ids) or breakdown.total_tokens,
        )
        log_record.setdefault("gen_ai.usage.completion_tokens", len(output_token_ids))
        ext_req_id = getattr(req_state, "external_req_id", None) or getattr(
            req_state, "request_id", None
        )
        if ext_req_id:
            log_record.setdefault("gen_ai.request.id", ext_req_id)
        stats = getattr(req_state, "stats", None)
        if stats is not None:
            ttft = getattr(stats, "first_token_latency", None)
            if isinstance(ttft, (int, float)):
                log_record.setdefault(
                    "gen_ai.latency.time_to_first_token", round(float(ttft), 6)
                )
            scheduled_ts = getattr(stats, "scheduled_ts", 0.0) or 0.0
            queued_ts = getattr(stats, "queued_ts", 0.0) or 0.0
            first_token_ts = getattr(stats, "first_token_ts", 0.0) or 0.0
            last_token_ts = getattr(stats, "last_token_ts", 0.0) or 0.0
            if scheduled_ts > 0 and queued_ts > 0:
                log_record.setdefault(
                    "gen_ai.latency.time_in_queue",
                    round(max(0.0, float(scheduled_ts - queued_ts)), 6),
                )
            if first_token_ts > 0 and scheduled_ts > 0:
                log_record.setdefault(
                    "gen_ai.latency.time_in_model_prefill",
                    round(max(0.0, float(first_token_ts - scheduled_ts)), 6),
                )
            if last_token_ts > 0 and first_token_ts > 0:
                decode_sec = max(0.0, float(last_token_ts - first_token_ts))
                log_record.setdefault(
                    "gen_ai.latency.time_in_model_decode",
                    round(decode_sec, 6),
                )
                if isinstance(ttft, (int, float)) and ttft > 0:
                    log_record.setdefault(
                        "gen_ai.latency.e2e",
                        round(float(ttft) + decode_sec, 6),
                    )
        try:
            with open(self.span_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, sort_keys=True) + "\n")
        except OSError as exc:
            logger.warning_once(
                "Failed to write agentic span log to %s: %s", self.span_log_path, exc
            )


_SERVICE: AgenticTelemetryService | None = None


def get_telemetry_service() -> AgenticTelemetryService:
    """Return the process-local `AgenticTelemetryService` singleton."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = AgenticTelemetryService()
    return _SERVICE


def reset_telemetry_service(
    service: AgenticTelemetryService | None = None,
) -> AgenticTelemetryService:
    """Reset or replace the process-local telemetry service (for testing)."""
    global _SERVICE
    _SERVICE = service if service is not None else AgenticTelemetryService()
    return _SERVICE


def patch_trace_headers(
    tracing_utils_mod: Any = None,
    *,
    refresh_consumers: bool | None = None,
) -> None:
    """Extend `vllm.tracing.utils.TRACE_HEADERS` and refresh consumer bindings."""
    should_refresh = (
        (tracing_utils_mod is None) if refresh_consumers is None else refresh_consumers
    )
    if tracing_utils_mod is None:
        from vllm.tracing import utils as tracing_utils_mod

    headers: list[str] = tracing_utils_mod.TRACE_HEADERS
    for hdr in AGENT_TRACE_HEADERS:
        if hdr not in headers:
            headers.append(hdr)

    if not getattr(tracing_utils_mod, "_tpu_agentic_trace_patch", False):
        orig_contains = getattr(tracing_utils_mod, "contains_trace_headers", None)
        orig_extract = getattr(tracing_utils_mod, "extract_trace_headers", None)

        def _wrapped_contains_trace_headers(raw_headers: Mapping[str, str]) -> bool:
            lowered = {k.lower(): v for k, v in raw_headers.items()}
            matched = (
                bool(orig_contains(lowered))
                if callable(orig_contains)
                else any(h.lower() in lowered for h in tracing_utils_mod.TRACE_HEADERS)
            )
            if not matched:
                _ACTIVE_TRACE_HEADERS.set(None)
            return matched

        def _wrapped_extract_trace_headers(
            raw_headers: Mapping[str, str],
        ) -> Mapping[str, str]:
            lowered = {k.lower(): v for k, v in raw_headers.items()}
            extracted = (
                dict(orig_extract(lowered))
                if callable(orig_extract)
                else {
                    h: lowered[h.lower()]
                    for h in tracing_utils_mod.TRACE_HEADERS
                    if h.lower() in lowered
                }
            )
            _ACTIVE_TRACE_HEADERS.set(dict(extracted) if extracted else None)
            return extracted

        if callable(orig_contains):
            _wrapped_contains_trace_headers = functools.wraps(orig_contains)(
                _wrapped_contains_trace_headers
            )
            tracing_utils_mod.contains_trace_headers = _wrapped_contains_trace_headers
        if callable(orig_extract):
            _wrapped_extract_trace_headers = functools.wraps(orig_extract)(
                _wrapped_extract_trace_headers
            )
            tracing_utils_mod.extract_trace_headers = _wrapped_extract_trace_headers
        tracing_utils_mod._agentic_contains_fn = _wrapped_contains_trace_headers
        tracing_utils_mod._agentic_extract_fn = _wrapped_extract_trace_headers
        tracing_utils_mod._tpu_agentic_trace_patch = True

    if not should_refresh:
        return

    contains_fn = getattr(tracing_utils_mod, "_agentic_contains_fn", None)
    extract_fn = getattr(tracing_utils_mod, "_agentic_extract_fn", None)
    for mod_name in _TRACING_CONSUMER_MODULES:
        mod: Any = sys.modules.get(mod_name)
        if mod is not None:
            if contains_fn is not None and hasattr(mod, "contains_trace_headers"):
                mod.contains_trace_headers = contains_fn
            if extract_fn is not None and hasattr(mod, "extract_trace_headers"):
                mod.extract_trace_headers = extract_fn


def patch_output_processor(
    output_processor_cls: Any = None,
    output_processor_mod: Any = None,
    service: AgenticTelemetryService | None = None,
) -> None:
    """Wrap `OutputProcessor` to classify prefill turns and enrich OTel spans."""
    if output_processor_mod is None or output_processor_cls is None:
        import vllm.v1.engine.output_processor as output_processor_mod

        output_processor_cls = output_processor_mod.OutputProcessor

    if output_processor_cls.__dict__.get("_tpu_agentic_output_patch", False):
        return

    orig_add_request = getattr(output_processor_cls, "add_request", None)
    orig_abort_requests = getattr(output_processor_cls, "abort_requests", None)
    orig_process_outputs = output_processor_cls.process_outputs
    orig_do_tracing = output_processor_cls.do_tracing

    if orig_add_request is not None:

        @functools.wraps(orig_add_request)
        def _wrapped_add_request(
            self: Any,
            request: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            try:
                active_hdrs = _ACTIVE_TRACE_HEADERS.get()
                parent_req = args[1] if len(args) > 1 else kwargs.get("parent_req")
                req_idx = (
                    args[2] if len(args) > 2 else int(kwargs.get("request_index", 0))
                )
                if active_hdrs is not None and (
                    parent_req is None
                    or req_idx + 1 >= int(getattr(parent_req, "n", 1) or 1)
                ):
                    _ACTIVE_TRACE_HEADERS.set(None)
                if getattr(request, "trace_headers", None) is None and active_hdrs:
                    with contextlib.suppress(AttributeError):
                        request.trace_headers = dict(active_hdrs)
                if not getattr(request, "session_id", None):
                    sid = _extract_request_session_id(request, active_hdrs)
                    if sid:
                        with contextlib.suppress(AttributeError):
                            request.session_id = sid
            except Exception as exc:
                logger.warning_once(
                    "Agentic telemetry pre-add_request hook failed: %s", exc
                )

            res = orig_add_request(self, request, *args, **kwargs)

            try:
                req_states = getattr(self, "request_states", {})
                req_state = req_states.get(getattr(request, "request_id", None))
                if req_state is not None and getattr(request, "trace_headers", None):
                    with contextlib.suppress(AttributeError):
                        req_state.trace_headers = dict(request.trace_headers)
            except Exception as exc:
                logger.warning_once(
                    "Agentic telemetry post-add_request hook failed: %s", exc
                )
            return res

        output_processor_cls.add_request = _wrapped_add_request

    if orig_abort_requests is not None:

        @functools.wraps(orig_abort_requests)
        def _wrapped_abort_requests(
            self: Any,
            request_ids: Iterable[str],
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            aborted = orig_abort_requests(self, request_ids, *args, **kwargs)
            try:
                svc = service or get_telemetry_service()
                for req_id in aborted or ():
                    svc.discard_request(req_id)
            except Exception as exc:
                logger.warning_once(
                    "Agentic telemetry abort_requests cleanup failed: %s", exc
                )
            return aborted

        output_processor_cls.abort_requests = _wrapped_abort_requests

    @functools.wraps(orig_process_outputs)
    def _wrapped_process_outputs(
        self: Any,
        engine_core_outputs: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        svc: AgenticTelemetryService | None = None
        candidates_without_tracing: list[tuple[str | None, Any, Any]] = []
        errored_req_ids: list[str] = []
        req_states: Mapping[str, Any] = {}
        try:
            svc = service or get_telemetry_service()
            svc.session_tracker.configure_from_tokenizer(
                getattr(self, "tokenizer", None)
            )
            req_states = getattr(self, "request_states", {})
            tracing_enabled = getattr(self, "tracing_enabled", False)

            for out in engine_core_outputs:
                req_id = getattr(out, "request_id", None)
                if getattr(out, "error", None) is not None:
                    if req_id:
                        errored_req_ids.append(req_id)
                    continue
                req_state = req_states.get(req_id)
                if req_state is None:
                    continue
                if getattr(req_state, "is_prefilling", False):
                    svc.on_prefill_output(req_state, out)
                if not tracing_enabled:
                    candidates_without_tracing.append((req_id, req_state, out))
        except Exception as exc:
            logger.warning_once(
                "Agentic telemetry pre-process_outputs hook failed: %s", exc
            )

        result = orig_process_outputs(
            self,
            engine_core_outputs,
            *args,
            **kwargs,
        )

        if svc is not None:
            try:
                for req_id in errored_req_ids:
                    svc.discard_request(req_id)
                post_states = getattr(self, "request_states", {})
                for req_id, req_state, out in candidates_without_tracing:
                    finished = getattr(out, "finish_reason", None) is not None or (
                        req_id is not None and req_id not in post_states
                    )
                    if not finished:
                        continue
                    if svc.span_log_path:
                        svc.build_span_attributes(req_state, out)
                    else:
                        svc.on_request_finished(req_state, out)
            except Exception as exc:
                logger.warning_once(
                    "Agentic telemetry post-process_outputs hook failed: %s", exc
                )

        return result

    @functools.wraps(orig_do_tracing)
    def _wrapped_do_tracing(
        self: Any,
        engine_core_output: Any,
        req_state: Any,
        iteration_stats: Any,
    ) -> None:
        extra_attrs: dict[str, Any] | None = None
        orig_instrument: Any = None
        try:
            svc = service or get_telemetry_service()
            extra_attrs = svc.build_span_attributes(req_state, engine_core_output)
            orig_instrument = getattr(output_processor_mod, "instrument_manual", None)
        except Exception as exc:
            logger.warning_once("Agentic telemetry do_tracing hook failed: %s", exc)

        if not extra_attrs or not callable(orig_instrument):
            orig_do_tracing(self, engine_core_output, req_state, iteration_stats)
            return

        @functools.wraps(orig_instrument)
        def _enriched_instrument_manual(
            span_name: str,
            start_time: int,
            attributes: dict[str, Any] | None = None,
            context: Any = None,
            kind: Any = None,
        ) -> None:
            merged = dict(attributes or {})
            merged.update(extra_attrs)
            return orig_instrument(
                span_name=span_name,
                start_time=start_time,
                attributes=merged,
                context=context,
                kind=kind,
            )

        output_processor_mod.instrument_manual = _enriched_instrument_manual
        try:
            orig_do_tracing(self, engine_core_output, req_state, iteration_stats)
        finally:
            output_processor_mod.instrument_manual = orig_instrument

    output_processor_cls.process_outputs = _wrapped_process_outputs
    output_processor_cls.do_tracing = _wrapped_do_tracing
    output_processor_cls._tpu_agentic_output_patch = True


def sync_block_size_from_config(vllm_config: Any) -> None:
    """Sync `cache_config.block_size` into process state and active service."""
    if not envs.VLLM_TPU_AGENTIC_TELEMETRY:
        return
    global _DETECTED_VLLM_BLOCK_SIZE
    try:
        bs = getattr(getattr(vllm_config, "cache_config", None), "block_size", None)
        if isinstance(bs, int) and bs > 0:
            _DETECTED_VLLM_BLOCK_SIZE = bs
            if _SERVICE is not None and int(envs.VLLM_TPU_AGENTIC_BLOCK_SIZE or 0) <= 0:
                _SERVICE.session_tracker.block_size = bs
                _SERVICE._explicit_block_size = True
                _SERVICE._block_size_checked = True
    except Exception as exc:
        logger.warning_once("Agentic telemetry block_size sync hook failed: %s", exc)


def install_agentic_telemetry_patches() -> None:
    """Install or refresh telemetry hooks when `VLLM_TPU_AGENTIC_TELEMETRY=1`."""
    global _PATCHES_LOGGED
    if not envs.VLLM_TPU_AGENTIC_TELEMETRY:
        return

    patch_trace_headers()
    try:
        patch_output_processor()
    except ImportError as exc:
        logger.warning_once(
            "Skipping OutputProcessor telemetry patch; OutputProcessor unavailable: %s",
            exc,
        )
    if not _PATCHES_LOGGED:
        _PATCHES_LOGGED = True
        logger.info("Installed vllm-torchtpu agentic telemetry patches")
