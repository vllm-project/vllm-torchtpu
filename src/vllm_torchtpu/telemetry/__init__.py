# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""CUJ-2 agentic serving telemetry schema contract and trace header hooks."""

from vllm_torchtpu.telemetry.vllm_hooks import (
    AGENT_TRACE_HEADERS,
    CATEGORIES,
    SPAN_ATTRIBUTE_KEYS,
    extract_agent_headers,
    install_agentic_telemetry_patches,
    patch_trace_headers,
)

__all__ = [
    "AGENT_TRACE_HEADERS",
    "CATEGORIES",
    "SPAN_ATTRIBUTE_KEYS",
    "extract_agent_headers",
    "install_agentic_telemetry_patches",
    "patch_trace_headers",
]
