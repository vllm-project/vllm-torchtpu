# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""CUJ-2 agentic serving telemetry schema contract and trace header propagation."""

from collections.abc import Mapping
import logging
import sys
from typing import Any

from vllm_torchtpu import envs

logger = logging.getLogger("vllm." + __name__)

CATEGORIES: tuple[str, ...] = (
    "cache_hit",
    "cross_session_opportunity",
    "affinity_honored_eviction",
    "routing_loss",
    "prefix_mutation_loss",
    "new_input",
)

AGENT_TRACE_HEADERS: tuple[str, ...] = (
    "x-agent-session-id",
    "x-agent-parent-session-id",
    "x-agent-turn-id",
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
    turn_id: int | None = None
    if raw_turn is not None:
        try:
            turn_id = int(raw_turn)
        except (TypeError, ValueError):
            turn_id = None
    return session_id, parent_session_id, turn_id


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
        return {
            h: lowered[h.lower()]
            for h in tracing_utils_mod.TRACE_HEADERS
            if h.lower() in lowered
        }

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


def install_agentic_telemetry_patches() -> None:
    """Install CUJ-2 agentic trace header hooks when `VLLM_TPU_AGENTIC_TELEMETRY=1`."""
    if not envs.VLLM_TPU_AGENTIC_TELEMETRY:
        return
    patch_trace_headers()
    logger.info("Installed vllm-torchtpu CUJ-2 agentic telemetry contract")
