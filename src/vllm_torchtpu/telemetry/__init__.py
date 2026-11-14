# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Host-side agentic serving telemetry for vllm-torchtpu."""

from vllm_torchtpu.telemetry.eviction_tracker import (
    BlockEvictionTracker,
    TpuPaddingCounterTracker,
    compute_request_padding_tokens,
    extract_hardware_span_attributes,
)
from vllm_torchtpu.telemetry.token_breakdown import (
    CATEGORIES,
    BoundedLRU,
    SessionTurnTracker,
    TokenBreakdown,
    compute_rolling_block_hashes,
)
from vllm_torchtpu.telemetry.vllm_hooks import (
    AGENT_TRACE_HEADERS,
    SPAN_ATTRIBUTE_KEYS,
    AgenticTelemetryService,
    extract_agent_headers,
    get_telemetry_service,
    install_agentic_telemetry_patches,
    patch_block_pool,
    patch_output_processor,
    patch_tpu_padding_metrics,
    patch_trace_headers,
    reset_telemetry_service,
)

__all__ = [
    "AGENT_TRACE_HEADERS",
    "AgenticTelemetryService",
    "BlockEvictionTracker",
    "BoundedLRU",
    "CATEGORIES",
    "SPAN_ATTRIBUTE_KEYS",
    "SessionTurnTracker",
    "TokenBreakdown",
    "TpuPaddingCounterTracker",
    "compute_request_padding_tokens",
    "compute_rolling_block_hashes",
    "extract_agent_headers",
    "extract_hardware_span_attributes",
    "get_telemetry_service",
    "install_agentic_telemetry_patches",
    "patch_block_pool",
    "patch_output_processor",
    "patch_tpu_padding_metrics",
    "patch_trace_headers",
    "reset_telemetry_service",
]
