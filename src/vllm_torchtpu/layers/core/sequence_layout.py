# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Sequence layout plan and planning protocols for TPU.

Re-exports core primitives from `llm_tpu_core.layers.sequence_layout`.
"""

from typing import Any

from vllm_torchtpu.llm_tpu_core.layers.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
    DEFAULT_SEQUENCE_LAYOUT_PROTOCOL,
    PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
    AllSequenceLayoutPlanner,
    SequenceLayoutDescriptor,
    SequenceLayoutKind,
    SequenceLayoutPlan,
    SequenceLayoutPlanner,
    _first_ge,
    _localize_token_tensor_and_gather_indices,
    is_pcp_streaming_attention_metadata,
    is_pcp_streaming_sequence_layout,
    is_pcp_streaming_sequence_layout_descriptor,
)


def create_sequence_layout_planner(vllm_config: Any) -> SequenceLayoutPlanner:
    from vllm_torchtpu.layers.core.pcp_sequence_layout import PcpSequenceLayoutPlanner

    pcp_planner = PcpSequenceLayoutPlanner.from_vllm_config(vllm_config)
    if pcp_planner.enabled:
        return pcp_planner
    return AllSequenceLayoutPlanner()


__all__ = [
    "DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR",
    "DEFAULT_SEQUENCE_LAYOUT_PROTOCOL",
    "PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL",
    "AllSequenceLayoutPlanner",
    "SequenceLayoutDescriptor",
    "SequenceLayoutKind",
    "SequenceLayoutPlan",
    "SequenceLayoutPlanner",
    "_first_ge",
    "_localize_token_tensor_and_gather_indices",
    "create_sequence_layout_planner",
    "is_pcp_streaming_attention_metadata",
    "is_pcp_streaming_sequence_layout",
    "is_pcp_streaming_sequence_layout_descriptor",
]
