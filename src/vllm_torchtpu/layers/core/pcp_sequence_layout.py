# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper for PCP sequence layout planning.

Re-exports core primitives from `vllm_torchtpu.llm_tpu_core.layers.pcp_sequence_layout`.
Synchronizes monkeypatched attributes with llm_tpu_core.
"""

import sys
from typing import Any

from vllm_torchtpu.llm_tpu_core.layers import (
    pcp_sequence_layout as _core_pcp_sl,
)
from vllm_torchtpu.llm_tpu_core.layers.pcp_sequence_layout import (
    PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
    PcpPreparedBatch,
    PcpRequestSpan,
    PcpSequenceLayoutDecision,
    PcpSequenceLayoutEligibility,
    PcpSequenceLayoutMode,
    PcpSequenceLayoutPlanner,
    _debug_pcp_layout_window,
    _ensure_host_token_buffer_capacity,
    _get_native_pcp_rank,
    _get_native_pcp_world_size,
    _get_pcp_streaming_q_block_size,
    _pcp_all_gather_equal_tokens,
    _pcp_all_reduce_sum,
    prepare_pcp_sequence_layout,
)


class _PcpSequenceLayoutModuleWrapper:
    def __init__(self, wrapped_module):
        self.__dict__["_wrapped_module"] = wrapped_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value
        if hasattr(_core_pcp_sl, name):
            setattr(_core_pcp_sl, name, value)


sys.modules[__name__] = _PcpSequenceLayoutModuleWrapper(sys.modules[__name__])

__all__ = [
    "PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR",
    "PcpPreparedBatch",
    "PcpRequestSpan",
    "PcpSequenceLayoutDecision",
    "PcpSequenceLayoutEligibility",
    "PcpSequenceLayoutMode",
    "PcpSequenceLayoutPlanner",
    "_debug_pcp_layout_window",
    "_ensure_host_token_buffer_capacity",
    "_get_native_pcp_rank",
    "_get_native_pcp_world_size",
    "_get_pcp_streaming_q_block_size",
    "_pcp_all_gather_equal_tokens",
    "_pcp_all_reduce_sum",
    "prepare_pcp_sequence_layout",
]
