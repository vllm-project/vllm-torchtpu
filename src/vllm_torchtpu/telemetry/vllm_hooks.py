# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Flag-gated vLLM lifecycle hooks for CUJ-2 agentic serving telemetry."""

import contextvars
import json
import logging
import os
import socket
import sys
from collections.abc import Mapping
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.telemetry.eviction_tracker import (
    BlockEvictionTracker,
    extract_hardware_span_attributes,
)
from vllm_torchtpu.telemetry.token_breakdown import (
    CATEGORIES,
    SessionTurnTracker,
    TokenBreakdown,
    compute_rolling_block_hashes,
)

logger = logging.getLogger("vllm." + __name__)

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
    "gen_ai.usage.input_tokens.cache_hit",
    "gen_ai.usage.input_tokens.cross_session_opportunity",
    "gen_ai.usage.input_tokens.affinity_honored_eviction",
    "gen_ai.usage.input_tokens.routing_loss",
    "gen_ai.usage.input_tokens.prefix_mutation_loss",
    "gen_ai.usage.input_tokens.unreused_reply_loss",
    "gen_ai.usage.input_tokens.new_input",
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
    "vllm.tpu.shape_padding_tokens",
)

_ACTIVE_TRACE_HEADERS: contextvars.ContextVar[dict[str, str] | None] = (
    contextvars.ContextVar("_ACTIVE_TRACE_HEADERS", default=None)
)


def _resolve_trace_headers(
    req_state: Any, engine_core_output: Any
) -> Mapping[str, str] | None:
    """Return trace headers from `engine_core_output` or `req_state` fallback."""
    return getattr(engine_core_output, "trace_headers", None) or getattr(
        req_state, "trace_headers", None
    )


def _resolve_num_cached_tokens(req_state: Any, engine_core_output: Any) -> int:
    """Extract `num_cached_tokens` across direct fields, `prefill_stats`, or `req_state`."""
    num_cached = getattr(engine_core_output, "num_cached_tokens", None)
    if num_cached is None:
        prefill_stats = getattr(engine_core_output, "prefill_stats", None)
        if prefill_stats is not None:
            num_cached = getattr(prefill_stats, "num_cached_tokens", None)
    if num_cached is None:
        num_cached = getattr(req_state, "num_cached_tokens", 0) or 0
    return int(num_cached)


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


class AgenticTelemetryService:
    """Coordinates session turn tracking, block eviction age, and span emission."""

    def __init__(
        self,
        *,
        block_size: int | None = None,
        replica_id: str | None = None,
        span_log_path: str | None = None,
        session_tracker: SessionTurnTracker | None = None,
        eviction_tracker: BlockEvictionTracker | None = None,
    ) -> None:
        self.replica_id = (
            replica_id
            or os.environ.get("VLLM_REPLICA_ID")
            or os.environ.get("HOSTNAME")
            or socket.gethostname()
        )
        self.span_log_path = (
            span_log_path
            if span_log_path is not None
            else getattr(envs, "VLLM_TPU_AGENTIC_SPAN_LOG", "")
        )
        self._explicit_block_size = block_size is not None or (
            session_tracker is not None
        )
        resolved_block_size = (
            block_size
            if block_size is not None
            else int(os.environ.get("VLLM_TPU_AGENTIC_BLOCK_SIZE", "16") or 16)
        )
        self.session_tracker = session_tracker or SessionTurnTracker(
            block_size=resolved_block_size
        )
        self.eviction_tracker = eviction_tracker or BlockEvictionTracker()
        self.category_totals: dict[str, int] = {cat: 0 for cat in CATEGORIES}
        self._req_breakdowns: dict[str, TokenBreakdown] = {}

    def _maybe_sync_vllm_block_size(self) -> None:
        """Auto-detect `cache_config.block_size` (e.g. 64/128 on TPU) once if not explicitly set."""
        if self._explicit_block_size:
            return
        try:
            from vllm.config.vllm import get_current_vllm_config

            cfg = get_current_vllm_config()
            bs = getattr(getattr(cfg, "cache_config", None), "block_size", None)
            if isinstance(bs, int) and bs > 0:
                self.session_tracker.block_size = bs
                self._explicit_block_size = True
        except Exception:
            pass

    def on_block_evicted(self, block_hash: Any, now: float | None = None) -> None:
        """Record an HBM block eviction event."""
        self.eviction_tracker.record_eviction(block_hash, now=now)

    def on_prefill_output(
        self, req_state: Any, engine_core_output: Any
    ) -> TokenBreakdown:
        """Classify prefill tokens on the first output step of a request."""
        req_id = getattr(engine_core_output, "request_id", None) or getattr(
            req_state, "request_id", ""
        )
        if req_id and req_id in self._req_breakdowns:
            return self._req_breakdowns[req_id]

        self._maybe_sync_vllm_block_size()
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
            self._req_breakdowns[req_id] = breakdown
        return breakdown

    def on_request_finished(
        self, req_state: Any, engine_core_output: Any
    ) -> TokenBreakdown:
        """Finalize request state and update two-tier session history."""
        req_id = getattr(engine_core_output, "request_id", None) or getattr(
            req_state, "request_id", ""
        )
        breakdown = self._req_breakdowns.pop(req_id, None) if req_id else None
        if breakdown is None:
            breakdown = self.on_prefill_output(req_state, engine_core_output)
            if req_id:
                self._req_breakdowns.pop(req_id, None)

        session_id, _, turn_id = extract_agent_headers(
            _resolve_trace_headers(req_state, engine_core_output)
        )
        prompt_token_ids = getattr(req_state, "prompt_token_ids", None) or ()
        detok = getattr(req_state, "detokenizer", None)
        output_token_ids = (
            getattr(detok, "output_token_ids", None)
            if detok is not None
            else getattr(engine_core_output, "new_token_ids", None)
        ) or ()
        self.session_tracker.record_turn_completion(
            session_id,
            prompt_token_ids,
            output_token_ids,
            turn_id=turn_id if turn_id is not None else 1,
        )
        if session_id:
            full_tokens = (*prompt_token_ids, *output_token_ids)
            self.eviction_tracker.register_session_blocks(
                session_id,
                compute_rolling_block_hashes(
                    full_tokens, self.session_tracker.block_size
                ),
            )
        return breakdown

    def build_span_attributes(
        self,
        req_state: Any,
        engine_core_output: Any,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Build all CUJ-2 OTel span attributes for a completed request."""
        breakdown = self.on_request_finished(req_state, engine_core_output)
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

        eviction_age_ms: float | None = None
        if breakdown.affinity_honored_eviction > 0:
            prompt_token_ids = getattr(req_state, "prompt_token_ids", None) or ()
            block_size = self.session_tracker.block_size
            all_hashes = compute_rolling_block_hashes(prompt_token_ids, block_size)
            start_blk = breakdown.cache_hit // block_size
            missed_hashes = all_hashes[start_blk:]
            eviction_age_ms = self.eviction_tracker.time_since_eviction_ms(
                missed_hashes, now=now
            )
            if eviction_age_ms is None:
                eviction_age_ms = self.eviction_tracker.last_eviction_age_ms(now=now)

        hw_attrs = extract_hardware_span_attributes(
            kv_transfer_params=getattr(engine_core_output, "kv_transfer_params", None),
            num_external_computed_tokens=getattr(
                engine_core_output, "num_external_computed_tokens", 0
            )
            or 0,
            time_since_eviction_ms=eviction_age_ms,
            lru_horizon_ms=self.eviction_tracker.lru_horizon_ms(now=now),
            session_evicted_at_ms=self.eviction_tracker.session_evicted_at_ms(
                session_id
            ),
        )
        attrs.update(hw_attrs)
        if self.span_log_path:
            log_record = dict(attrs)
            trace_hdrs = _resolve_trace_headers(req_state, engine_core_output) or {}
            lowered_hdrs = {
                k.lower(): v for k, v in trace_hdrs.items() if v is not None
            }
            if "traceparent" in lowered_hdrs:
                log_record.setdefault("traceparent", lowered_hdrs["traceparent"])
            prompt_token_ids = getattr(req_state, "prompt_token_ids", None) or ()
            detok = getattr(req_state, "detokenizer", None)
            output_token_ids = (
                getattr(detok, "output_token_ids", None)
                if detok is not None
                else getattr(engine_core_output, "new_token_ids", None)
            ) or ()
            log_record.setdefault(
                "gen_ai.usage.prompt_tokens",
                len(prompt_token_ids) or breakdown.total_tokens,
            )
            log_record.setdefault(
                "gen_ai.usage.completion_tokens", len(output_token_ids)
            )
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
            except OSError:
                logger.debug("Failed to write CUJ-2 span log to %s", self.span_log_path)
        return attrs


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


def patch_trace_headers(tracing_utils_mod: Any = None) -> None:
    """Extend `vllm.tracing.utils.TRACE_HEADERS` with agent session headers."""
    if tracing_utils_mod is None:
        from vllm.tracing import utils as tracing_utils_mod

    headers: list[str] = tracing_utils_mod.TRACE_HEADERS
    for hdr in AGENT_TRACE_HEADERS:
        if hdr not in headers:
            headers.append(hdr)

    if getattr(tracing_utils_mod, "_tpu_agentic_trace_patch", False):
        return

    def _wrapped_contains_trace_headers(raw_headers: Mapping[str, str]) -> bool:
        lowered = {k.lower() for k in raw_headers}
        return any(h.lower() in lowered for h in tracing_utils_mod.TRACE_HEADERS)

    def _wrapped_extract_trace_headers(
        raw_headers: Mapping[str, str],
    ) -> Mapping[str, str]:
        lowered = {k.lower(): v for k, v in raw_headers.items()}
        extracted = {
            h: lowered[h.lower()]
            for h in tracing_utils_mod.TRACE_HEADERS
            if h.lower() in lowered
        }
        _ACTIVE_TRACE_HEADERS.set(dict(extracted) if extracted else None)
        return extracted

    if hasattr(tracing_utils_mod, "contains_trace_headers"):
        tracing_utils_mod.contains_trace_headers = _wrapped_contains_trace_headers
    if hasattr(tracing_utils_mod, "extract_trace_headers"):
        tracing_utils_mod.extract_trace_headers = _wrapped_extract_trace_headers
    tracing_utils_mod._tpu_agentic_trace_patch = True

    for mod_name in ("vllm.tracing", "vllm.entrypoints.serve.engine.serving"):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            if hasattr(mod, "contains_trace_headers"):
                mod.contains_trace_headers = _wrapped_contains_trace_headers
            if hasattr(mod, "extract_trace_headers"):
                mod.extract_trace_headers = _wrapped_extract_trace_headers


def patch_block_pool(
    block_pool_cls: Any = None,
    service: AgenticTelemetryService | None = None,
) -> None:
    """Wrap `BlockPool._maybe_evict_cached_block` to record eviction timestamps."""
    if block_pool_cls is None:
        from vllm.v1.core.block_pool import BlockPool as block_pool_cls

    if block_pool_cls.__dict__.get("_tpu_agentic_eviction_patch", False):
        return

    orig_evict = block_pool_cls._maybe_evict_cached_block

    def _wrapped_maybe_evict_cached_block(self: Any, block: Any) -> bool:
        raw_hash = getattr(block, "block_hash", None)
        evicted = orig_evict(self, block)
        if evicted and raw_hash is not None:
            svc = service or get_telemetry_service()
            svc.on_block_evicted(raw_hash)
        return evicted

    block_pool_cls._maybe_evict_cached_block = _wrapped_maybe_evict_cached_block
    block_pool_cls._tpu_agentic_eviction_patch = True


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
    orig_process_outputs = output_processor_cls.process_outputs
    orig_do_tracing = output_processor_cls.do_tracing

    if orig_add_request is not None:

        def _wrapped_add_request(
            self: Any,
            request: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            active_hdrs = _ACTIVE_TRACE_HEADERS.get()
            if getattr(request, "trace_headers", None) is None and active_hdrs:
                try:
                    request.trace_headers = dict(active_hdrs)
                except AttributeError:
                    pass
            res = orig_add_request(self, request, *args, **kwargs)
            req_states = getattr(self, "request_states", {})
            req_state = req_states.get(getattr(request, "request_id", None))
            if req_state is not None and getattr(request, "trace_headers", None):
                try:
                    req_state.trace_headers = dict(request.trace_headers)
                except AttributeError:
                    pass
            return res

        output_processor_cls.add_request = _wrapped_add_request

    def _wrapped_process_outputs(
        self: Any,
        engine_core_outputs: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        svc = service or get_telemetry_service()
        prefilling_outputs: list[tuple[Any, Any]] = []
        finished_without_tracing: list[tuple[Any, Any]] = []
        req_states = getattr(self, "request_states", {})
        tracing_enabled = getattr(self, "tracing_enabled", False)

        for out in engine_core_outputs:
            req_state = req_states.get(getattr(out, "request_id", None))
            if req_state is None:
                continue
            if getattr(req_state, "is_prefilling", False):
                prefilling_outputs.append((req_state, out))
            if getattr(out, "finish_reason", None) is not None and not tracing_enabled:
                finished_without_tracing.append((req_state, out))

        for req_state, out in prefilling_outputs:
            svc.on_prefill_output(req_state, out)

        result = orig_process_outputs(
            self,
            engine_core_outputs,
            *args,
            **kwargs,
        )

        for req_state, out in finished_without_tracing:
            if svc.span_log_path:
                svc.build_span_attributes(req_state, out)
            else:
                svc.on_request_finished(req_state, out)

        return result

    def _wrapped_do_tracing(
        self: Any,
        engine_core_output: Any,
        req_state: Any,
        iteration_stats: Any,
    ) -> None:
        svc = service or get_telemetry_service()
        extra_attrs = svc.build_span_attributes(req_state, engine_core_output)
        orig_instrument = output_processor_mod.instrument_manual

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


def install_agentic_telemetry_patches() -> None:
    """Install CUJ-2 agentic telemetry hooks when `VLLM_TPU_AGENTIC_TELEMETRY=1`."""
    if not envs.VLLM_TPU_AGENTIC_TELEMETRY:
        return

    patch_trace_headers()
    try:
        patch_block_pool()
    except ImportError:
        logger.debug("Skipping BlockPool telemetry patch; vllm BlockPool unavailable")
    try:
        patch_output_processor()
    except ImportError:
        logger.debug(
            "Skipping OutputProcessor telemetry patch; OutputProcessor unavailable"
        )
    logger.info("Installed vllm-torchtpu CUJ-2 agentic telemetry patches")
